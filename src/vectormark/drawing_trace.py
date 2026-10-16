"""Stable, editable trace artifacts for the interactive drawing workflow."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from .color import extract_palette, quantize, white_canvas_mask
from .background import BackgroundColor, background_matches
from .contour import region_contours
from .emit import render_svg_doc
from .fit import fit_path_stages
from .path_fit_evidence import AutomaticFitEvidence
from .json_value import to_json_value
from .material_pretrace import MaterialPretraceResult
from .optimizer.corners import path_corner_diagnostics
from .pipeline import attach_coverage_field
from .segment import fill_tiny_isolated_holes, segment


@dataclass(frozen=True)
class TraceOptions:
    max_sampling_colors: int = 16
    requested_max_sampling_colors: int | Literal["auto"] | None = None
    reserve_white_background: bool = False
    min_region_size: int = 16
    max_hole_area: int = 128
    trace_level: Literal["pixel", "subpixel"] = "pixel"
    simplify_tolerance: float = 1.5
    curve_tolerance: float = 1.0
    fit_strategy: Literal["quadratic", "progressive", "progressive_allow_lines"] = "quadratic"
    support_weighted_corners: bool = False
    corner_rounding: bool = True
    remove_background: Literal["auto", "off", "on"] = "auto"
    # ``None`` means this lower-level engine was given RGB only and must retain
    # legacy behaviour.  MCP supplies an explicit value from the source image.
    source_has_alpha: bool | None = None
    background_tolerance: float = 32.0
    background_color: str | None = None


@dataclass(frozen=True)
class TraceCommand:
    id: str
    command: Literal["M", "L", "Q", "C", "A", "Z"]
    values: tuple[float, ...]
    anchor_kind: Literal["corner"] | None = None
    corner_id: str | None = None


@dataclass(frozen=True)
class TracePath:
    d: str
    fill_rule: Literal["nonzero", "evenodd"]
    commands: tuple[TraceCommand, ...]
    fit_stages: dict[str, str] = field(default_factory=dict)
    corner_diagnostics: dict[str, object] = field(default_factory=dict)
    path_evidence: tuple[AutomaticFitEvidence, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class RootPathEvidence:
    """One normalized leaf/subpath at root construction, before stitching.

    ``path_id`` is a target reference only. Correspondence addresses belong to
    ``fit.revision`` and must never be treated as current display command IDs.
    """

    region_id: str
    path_id: str
    fit: AutomaticFitEvidence
    stage: Literal["root-pre-stitch"] = "root-pre-stitch"

    def matches_output(self, d: str, *, stage: str, revision: str) -> bool:
        return stage == self.stage and revision == self.fit.revision and self.fit.matches_output(d)


@dataclass(frozen=True)
class TraceRegion:
    id: str
    source_label: int
    color: str
    mask: np.ndarray = field(compare=False, repr=False)
    contours: tuple[np.ndarray, ...] = field(compare=False, repr=False)
    trace_path: TracePath
    effective_trace_level: Literal["pixel", "subpixel"]


@dataclass(frozen=True)
class GeometryTraceRegion:
    """An authoritative foreground component, independent of its materials."""

    id: str
    mask: np.ndarray = field(compare=False, repr=False)
    contours: tuple[np.ndarray, ...] = field(compare=False, repr=False)
    trace_path: TracePath


_FOUR_CONNECTED = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=np.uint8)


def _split_pinched_alpha_component(mask: np.ndarray, *, min_region_size: int) -> list[np.ndarray]:
    """Split a native-alpha component joined only by a one-pixel pinch.

    Alpha artwork often has separately designed diamonds or petals whose
    antialiased tips meet at one pixel.  They should remain independently
    traceable surfaces; otherwise their union erases the intended transparent
    negative space.  Erode once only to discover disconnected cores, then
    assign every original pixel to its nearest core so the returned masks keep
    the source silhouette rather than its eroded version.
    """
    cores = binary_erosion(mask, structure=_FOUR_CONNECTED.astype(bool), border_value=0)
    core_labels, count = label(cores, structure=_FOUR_CONNECTED)
    if count < 2:
        return [mask]
    core_ids = [
        component_id
        for component_id in range(1, count + 1)
        if int(np.count_nonzero(core_labels == component_id)) >= min_region_size
    ]
    if len(core_ids) < 2:
        return [mask]

    # ``distance_transform_edt`` returns the nearest False location in
    # ``core_labels == 0``.  Those False locations are the retained core
    # pixels, whose labels identify the owning split surface.
    _distance, nearest = distance_transform_edt(core_labels == 0, return_indices=True)
    owner = core_labels[tuple(nearest)]
    pieces = [mask & (owner == component_id) for component_id in core_ids]
    if any(int(piece.sum()) < min_region_size for piece in pieces):
        return [mask]
    return pieces


@dataclass(frozen=True)
class TraceResult:
    width: int
    height: int
    options: TraceOptions
    regions: tuple[TraceRegion, ...]
    region_map_svg: str
    geometry_regions: tuple[GeometryTraceRegion, ...] = ()
    background: Mapping[str, object] = field(default_factory=dict)
    material_pretrace: MaterialPretraceResult | None = field(default=None, compare=False, repr=False)
    root_path_evidence: tuple[RootPathEvidence, ...] = field(default=(), repr=False)

    def to_public_dict(self, *, include_region_map: bool = False) -> dict[str, object]:
        """Return the stable, JSON-ready portion of the trace artifact."""
        result: dict[str, object] = {
            "width": self.width,
            "height": self.height,
            "options": asdict(self.options),
            "regions": [
                {
                    "id": region.id,
                    "source_label": region.source_label,
                    "color": region.color,
                    "effective_trace_level": region.effective_trace_level,
                    "trace_path": {
                        "d": region.trace_path.d,
                        "fill_rule": region.trace_path.fill_rule,
                        "commands": [
                            {
                                "id": command.id,
                                "command": command.command,
                                "values": list(command.values),
                                "anchor_kind": command.anchor_kind,
                                "corner_id": command.corner_id,
                            }
                            for command in region.trace_path.commands
                        ],
                    },
                }
                for region in self.regions
            ],
            "geometry": {
                "background": to_json_value(self.background),
                "regions": [
                    {
                        "id": region.id,
                        "trace_path": {
                            "d": region.trace_path.d,
                            "fill_rule": region.trace_path.fill_rule,
                            "commands": [
                                {
                                    "id": command.id,
                                    "command": command.command,
                                    "values": list(command.values),
                                    "anchor_kind": command.anchor_kind,
                                    "corner_id": command.corner_id,
                                }
                                for command in region.trace_path.commands
                            ],
                        },
                    }
                    for region in self.geometry_regions
                ],
            },
        }
        if include_region_map:
            result["region_map_svg"] = self.region_map_svg
        if self.material_pretrace is not None:
            result["material_pretrace"] = {
                "shape_region_count": self.material_pretrace.shape_region_count,
                "color_region_count": self.material_pretrace.color_region_count,
                "gradient_support_count": len(self.material_pretrace.gradient_supports),
                "resolved_max_sampling_colors": self.material_pretrace.resolved_max_sampling_colors,
            }
        return result


class TraceEngine(Protocol):
    def trace(self, rgb: np.ndarray, options: TraceOptions) -> TraceResult: ...


_PATH_TOKEN = re.compile(r"[MLQCAZ]|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_COORDINATE_COUNTS = {"M": 2, "L": 2, "Q": 4, "C": 6, "A": 7, "Z": 0}


def svg_path_commands(
    d: str,
    region_id: str,
    subpath_index: int = 0,
    *,
    declared_corners: tuple[tuple[float, float], ...] = (),
    declared_corner_spans: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (),
    corner_diagnostics: Mapping[str, object] | None = None,
    path_ids: Sequence[str] | None = None,
) -> tuple[TraceCommand, ...]:
    """Parse SVG path data into stable, target-scoped command IDs."""
    tokens = _PATH_TOKEN.findall(d)
    commands: list[TraceCommand] = []
    index = 0
    command_index = 0
    current_subpath = subpath_index
    corner_annotations = (
        corner_diagnostics.get("anchors", ())
        if corner_diagnostics is not None
        else path_corner_diagnostics(
            d,
            declared_corners=declared_corners,
            declared_corner_spans=declared_corner_spans,
        )["anchors"]
    )
    anchors_by_command = {
        (int(anchor["subpath"]), int(anchor["command_index"])): anchor
        for anchor in corner_annotations
        if isinstance(anchor, dict)
    }
    while index < len(tokens):
        command = tokens[index]
        if command not in _COORDINATE_COUNTS:
            raise ValueError(f"unsupported path data in trace: {d!r}")
        if command == "M" and commands:
            current_subpath += 1
            command_index = 0
        index += 1
        count = _COORDINATE_COUNTS[command]
        values = tuple(float(token) for token in tokens[index:index + count])
        if len(values) != count:
            raise ValueError(f"incomplete {command} command in trace: {d!r}")
        index += count
        annotation = anchors_by_command.get((current_subpath - subpath_index, command_index))
        commands.append(
            TraceCommand(
                id=f"{path_ids[current_subpath - subpath_index] if path_ids is not None else f'{region_id}.p{current_subpath}'}.c{command_index}",
                command=command,  # type: ignore[arg-type]
                values=values,
                anchor_kind="corner" if annotation is not None else None,
                corner_id=str(annotation["corner_id"]) if annotation is not None else None,
            )
        )
        command_index += 1
    return tuple(commands)


def _bbox_origin(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(xs.min())


def _region_map(width: int, height: int, regions: tuple[TraceRegion, ...]) -> str:
    body: list[str] = []
    for region in regions:
        ys, xs = np.nonzero(region.mask)
        cx = (float(xs.min()) + float(xs.max()) + 1.0) / 2.0
        cy = (float(ys.min()) + float(ys.max()) + 1.0) / 2.0
        body.append(
            f'<path id="map-{region.id}" fill="{region.color}" fill-opacity="0.35" '
            f'fill-rule="{region.trace_path.fill_rule}" d="{region.trace_path.d}"/>'
        )
        body.append(
            f'<text x="{cx:g}" y="{cy:g}" text-anchor="middle" '
            f'dominant-baseline="middle">{region.id}</text>'
        )
    return render_svg_doc(width, height, body)


def _fit_trace_path(
    mask: np.ndarray,
    *,
    region_id: str,
    options: TraceOptions,
) -> tuple[tuple[np.ndarray, ...], TracePath] | None:
    contours = tuple(contour for contour in region_contours(mask) if len(contour) >= 3)
    if not contours:
        return None
    path_stages = tuple(
        fit_path_stages(
            contour,
            epsilon=options.simplify_tolerance,
            max_error=options.curve_tolerance,
            cubic=False,
            progressive=options.fit_strategy != "quadratic",
            progressive_allow_lines=options.fit_strategy == "progressive_allow_lines",
            support_weighted_corners=options.support_weighted_corners,
            corner_rounding=options.corner_rounding,
            canvas_extent=float(max(mask.shape)),
            # Command reduction is an explicit downstream optimization, not a
            # trace-time choice.  Preserve the baseline fit here so the
            # editable drawing begins faithful to the extracted contour.
            prefer_simple_curves=False,
        )
        for contour in contours
    )
    paths = tuple(stages.atomic_d for stages in path_stages)
    corner_diagnostics = tuple(
        path_corner_diagnostics(
            d,
            declared_corners=stages.corner_points,
            declared_corner_spans=stages.corner_spans,
        )
        for d, stages in zip(paths, path_stages, strict=True)
    )
    commands = tuple(
        command
        for subpath_index, d in enumerate(paths)
        for command in svg_path_commands(
            d,
            region_id,
            subpath_index,
            declared_corners=path_stages[subpath_index].corner_points,
            declared_corner_spans=path_stages[subpath_index].corner_spans,
        )
    )
    return contours, TracePath(
        d=" ".join(paths),
        fill_rule="evenodd" if len(paths) > 1 else "nonzero",
        commands=commands,
        path_evidence=tuple(stages.baseline_evidence for stages in path_stages),
        fit_stages={
            "baseline": " ".join(stages.baseline_d for stages in path_stages),
            "simple": " ".join(stages.simple_d for stages in path_stages),
            "atomic": " ".join(paths),
        },
        corner_diagnostics=(
            corner_diagnostics[0]
            if len(corner_diagnostics) == 1
            else {"subpaths": corner_diagnostics}
        ),
    )


def _polyline_trace_path(
    contours: tuple[np.ndarray, ...],
    *,
    region_id: str,
) -> TracePath:
    """Expose raw material contours without eagerly curve-fitting them.

    Material traces are provenance for the decomposition.  The canonical
    drawing paths are fitted only after compatible material polyline merging,
    so fitting every palette fragment here would be both misleading and wasted
    work.  The polyline remains addressable through the normal trace-command
    IDs when raw provenance is explicitly inspected.
    """
    paths = [
        "M" + " L".join(f"{float(x):g} {float(y):g}" for x, y in contour) + " Z"
        for contour in contours
    ]
    d = " ".join(paths)
    commands = tuple(
        command
        for subpath_index, subpath_d in enumerate(paths)
        for command in svg_path_commands(subpath_d, region_id, subpath_index)
    )
    return TracePath(
        d=d,
        fill_rule="evenodd" if len(paths) > 1 else "nonzero",
        commands=commands,
        fit_stages={"polyline": d},
    )


def _geometry_trace(
    quantized: np.ndarray,
    *,
    options: TraceOptions,
    alpha: np.ndarray | None = None,
) -> tuple[tuple[GeometryTraceRegion, ...], dict[str, object]]:
    """Derive geometry from native alpha or a border-connected canvas plate.

    Palette regions remain raw provenance and fill evidence.  This trace instead
    owns the exterior geometry, so soft matte colours cannot turn into notches
    merely because they differ from the nearest material palette entry.
    """
    if options.remove_background == "off":
        return (), {"mode": "native_alpha" if options.source_has_alpha else "opaque", "applied": False}

    if alpha is not None and options.background_color is None:
        if alpha.shape != quantized.shape[:2]:
            raise ValueError("alpha coverage dimensions must match the trace image")
        return _geometry_components(
            alpha > 8,
            options=options,
            background={"mode": "native_alpha", "applied": True, "threshold": 8},
        )

    if options.remove_background == "auto" and options.source_has_alpha is not False and options.background_color is None:
        return (), {"mode": "native_alpha" if options.source_has_alpha else "opaque", "applied": False}

    border = np.concatenate((quantized[0], quantized[-1], quantized[:, 0], quantized[:, -1]))
    colors, counts = np.unique(border, axis=0, return_counts=True)
    background = BackgroundColor(
        color=options.background_color or "#{:02X}{:02X}{:02X}".format(*(int(value) for value in colors[int(counts.argmax())])),
        tolerance=options.background_tolerance,
    )
    candidates = background_matches(quantized, background)
    labels, _ = label(candidates, structure=np.ones((3, 3), dtype=np.uint8))
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    outer_background = candidates & np.isin(labels, border_labels)
    border_coverage = float(outer_background[0].sum() + outer_background[-1].sum()
                            + outer_background[:, 0].sum() + outer_background[:, -1].sum()) / float(len(border))
    if options.remove_background == "auto" and options.background_color is None and border_coverage < 0.75:
        return (), {
            "mode": "opaque",
            "applied": False,
            "confidence": border_coverage,
        }

    return _geometry_components(
        ~outer_background if alpha is None else (~outer_background & (alpha > 8)),
        options=options,
        background={
            "mode": "inferred_alpha",
            "applied": True,
            "color": background.color,
            "tolerance": options.background_tolerance,
            "confidence": border_coverage,
        },
    )


def _geometry_components(
    foreground: np.ndarray,
    *,
    options: TraceOptions,
    background: dict[str, object],
) -> tuple[tuple[GeometryTraceRegion, ...], dict[str, object]]:
    """Fit stable geometry roots from an already-established foreground mask."""
    # Geometry roots deliberately use 4-connectivity.  A one-pixel diagonal
    # touch is typically JPEG/AA noise, not a semantic join; keeping it with
    # 8-connectivity turns a discarded speck into an extra contour on an
    # otherwise clean root.  Native alpha can additionally preserve separate
    # intended surfaces when their antialiased tips create a one-pixel bridge.
    component_labels, component_count = label(
        foreground,
        structure=_FOUR_CONNECTED,
    )
    masks: list[np.ndarray] = []
    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        if background.get("mode") == "native_alpha":
            masks.extend(_split_pinched_alpha_component(component, min_region_size=options.min_region_size))
        else:
            masks.append(component)
    # The inferred canvas plate can leave isolated near-white pixels inside a
    # material after resampling.  Remove those topology-noise pinholes before
    # contour extraction; otherwise each becomes a visible degenerate SVG
    # subpath.  Native alpha remains literal source geometry.
    if background.get("mode") == "inferred_alpha":
        masks = [
            fill_tiny_isolated_holes(mask, max_area=min(options.max_hole_area, 4))[0]
            for mask in masks
        ]
    masks = [mask for mask in masks if int(mask.sum()) >= options.min_region_size]
    masks.sort(key=lambda mask: (-int(mask.sum()), *_bbox_origin(mask)))
    traced: list[GeometryTraceRegion] = []
    for index, mask in enumerate(masks, start=1):
        region_id = f"g{index}"
        fitted = _fit_trace_path(mask, region_id=region_id, options=options)
        if fitted is None:
            continue
        contours, trace_path = fitted
        frozen_mask = mask.copy()
        frozen_mask.setflags(write=False)
        frozen_contours = tuple(contour.copy() for contour in contours)
        for contour in frozen_contours:
            contour.setflags(write=False)
        traced.append(GeometryTraceRegion(region_id, frozen_mask, frozen_contours, trace_path))
    return tuple(traced), background


class PythonTraceEngine:
    """Pure-Python adapter from an RGB image to a deterministic trace artifact."""

    def trace(
        self,
        rgb: np.ndarray,
        options: TraceOptions,
        *,
        alpha: np.ndarray | None = None,
        geometry_rgb: np.ndarray | None = None,
    ) -> TraceResult:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("trace expects an RGB image with shape (height, width, 3)")
        height, width, _ = image.shape
        # Interactive tracing is governed by the explicit absolute region floor;
        # a relative palette floor must not erase a region before segmentation.
        white_canvas = white_canvas_mask(image) if options.reserve_white_background else None
        if white_canvas is None:
            palette = extract_palette(image, max_sampling_colors=options.max_sampling_colors, min_fraction=0.0)
        else:
            background_pixels = image[white_canvas]
            colors, counts = np.unique(background_pixels, axis=0, return_counts=True)
            background = colors[int(counts.argmax())][None, :]
            foreground = extract_palette(
                image[~white_canvas],
                max_sampling_colors=options.max_sampling_colors,
                min_fraction=0.0,
            )
            palette = np.vstack((background, foreground))
        quantized = quantize(image, palette)
        regions = segment(quantized, min_area=options.min_region_size)
        if options.trace_level == "subpixel":
            attach_coverage_field(regions, image, options.max_sampling_colors, palette=palette)

        ordered = sorted(
            regions,
            key=lambda region: (-region.area, *_bbox_origin(region.mask), region.color_hex, region.label),
        )
        traced: list[TraceRegion] = []
        for position, region in enumerate(ordered, start=1):
            region_id = f"r{position}"
            effective_level: Literal["pixel", "subpixel"] = (
                "subpixel" if options.trace_level == "subpixel" and region.coverage is not None else "pixel"
            )
            contours = tuple(
                contour
                for contour in region_contours(
                    region.mask,
                    coverage=region.coverage if effective_level == "subpixel" else None,
                )
                if len(contour) >= 3
            )
            mask = region.mask.copy()
            mask.setflags(write=False)
            immutable_contours = tuple(contour.copy() for contour in contours)
            for contour in immutable_contours:
                contour.setflags(write=False)
            traced.append(
                TraceRegion(
                    id=region_id,
                    source_label=region.label,
                    color=region.color_hex,
                    mask=mask,
                    contours=immutable_contours,
                    trace_path=_polyline_trace_path(immutable_contours, region_id=region_id),
                    effective_trace_level=effective_level,
                )
            )
        result_regions = tuple(traced)
        geometry_source = quantized if geometry_rgb is None else np.asarray(geometry_rgb, dtype=np.uint8)
        if geometry_source.shape != image.shape:
            raise ValueError("geometry RGB dimensions must match the trace image")
        geometry_regions, background = _geometry_trace(geometry_source, options=options, alpha=alpha)
        return TraceResult(
            width=width,
            height=height,
            options=options,
            regions=result_regions,
            region_map_svg=_region_map(width, height, result_regions),
            geometry_regions=geometry_regions,
            background=background,
        )

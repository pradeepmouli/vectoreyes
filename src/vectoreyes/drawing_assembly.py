"""Initial drawing-root assembly from retained material polylines and source paint.

This is the shared v0 assembly operation, not a refinement or export repair.
Actual fitting inputs and pre-fit corner evidence are retained with each path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import re
import numpy as np
from .candidate import Fill, FlatFill
from .components import decompose_components
from .contour import region_contours
from .drawing import Canvas, Drawing, DrawingRegion
from .drawing_topology import normalize_drawing_topology
from .corner_provenance import ingress_prefit_corners
from .immutable_value import freeze_value
from .path_geometry import CompoundPathGeometry, parse_svg_paths
from .path_topology import component_paths
from .paint_fitting import canonical_fill
from .drawing_trace import RootPathEvidence, TraceResult, _polyline_trace_path
from .fit import Shape, fit_path_stages
from .fill_fit import fit_fill
from .corner_diagnostics import path_corner_diagnostics
from .path_fit_evidence import AutomaticFitEvidence
from .pipeline import Options, _segment_image
from .segment import fill_small_compatible_holes
from .skia_geometry import SkPath, unary_union
from .surface_merge import material_groups
from .types import Region

DecompositionObserver = Callable[
    [str, tuple[Region, ...], tuple[Fill, ...] | None], None
]


@dataclass(frozen=True)
class RootDrawingResult:
    """Canonical roots, detached source observations, and actual fit evidence."""

    drawing: Drawing
    trace: TraceResult
    shape_metadata: Mapping[str, Mapping[str, object]]
    region_evidence: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'shape_metadata', freeze_value(self.shape_metadata))
        object.__setattr__(self, 'region_evidence', freeze_value(self.region_evidence))


@dataclass(frozen=True)
class _AssembledSurface:
    id: int
    shape: Shape
    fill: Fill
    z: float
    raster: np.ndarray
    source_label: int
    color_hex: str
    drawing_id: str
    source_regions: tuple[str, ...]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class _RootAssembly:
    """Transient fit records keyed internally until public addresses are final."""

    regions: tuple[_AssembledSurface, ...]
    evidence: Mapping[int, tuple[AutomaticFitEvidence, ...]]


@dataclass(frozen=True)
class _SurfaceFit:
    shape: Shape
    corners: dict[str, object]
    evidence: tuple[AutomaticFitEvidence, ...]


def _bind_root_evidence(
    trace: TraceResult, assembled: _RootAssembly
) -> RootDrawingResult:
    from .geometry_native import region_path
    from .refinement.gate import rasterize

    regions = []
    evidence = []
    observations = {}
    metadata = {}
    for root in assembled.regions:
        d = str(root.shape.params['d'])
        paths = tuple(part.strip() for part in re.findall(r'M[^M]*', d))
        fits = assembled.evidence[root.id]
        if len(paths) != len(fits) or any(not fit.matches_output(path)
                for path, fit in zip(paths, fits, strict=True)):
            raise ValueError('root fitting evidence does not match the retained leaf subpaths')
        evidence.extend(RootPathEvidence(root.drawing_id, f'{root.drawing_id}.p{index}', fit)
            for index, fit in enumerate(fits))
        parsed = parse_svg_paths(d, region_id=root.drawing_id)
        geometry = (parsed[0] if len(parsed) == 1 else CompoundPathGeometry(parsed,
            root.shape.params.get('fill_rule', 'nonzero'), 'pending component publication'))
        geometry, corners = ingress_prefit_corners(geometry, d, root.drawing_id,
            root.diagnostics.get('corners'))
        regions.append(DrawingRegion(root.drawing_id, geometry, canonical_fill(root.fill), root.z,
            source_region_ids=root.source_regions, diagnostics={**root.diagnostics, 'corners': corners}))
    normalized = normalize_drawing_topology(Drawing('geometry-evidence',
        Canvas(trace.width, trace.height), tuple(regions)), defer_overlaps=True)
    for original, output in zip(assembled.regions, normalized.regions, strict=True):
        def observe(region: DrawingRegion, ancestry: tuple[int, ...], mask: np.ndarray) -> None:
            observations[region.id] = {
                'id': ancestry[-1], 'ancestry': ancestry, 'raster_shape': mask.shape,
                'raster': mask.tobytes(), 'raster_source': None,
                'source_label': original.source_label, 'color_hex': original.color_hex,
                'original_kind': original.shape.kind, 'original_params': original.shape.params,
                'coverage': None, 'coverage_source': None, 'coverage_shape': None,
                'coverage_dtype': None, 'coverage_origin': None, 'coverage_canvas': None,
            }
            if region.is_leaf:
                metadata[region.id] = {key: value for key, value in original.shape.params.items() if key != 'd'}
                metadata[region.id]['path_ids'] = tuple(path.path.id for path in component_paths(region.geometry))
            for child in region.children:
                observe(child, (*ancestry, int(child.id.rsplit('-', 1)[1])),
                    rasterize(region_path(child, {}), original.raster.shape))
        observe(output, (original.id,), original.raster)
    owners = {
        path.path.id: leaf.id
        for root in normalized.regions
        for leaf in root.leaves()
        for path in component_paths(leaf.geometry)
    }
    return RootDrawingResult(
        normalized,
        replace(
            trace,
            root_path_evidence=tuple(
                replace(record, region_id=owners.get(record.path_id, record.region_id))
                for record in evidence
            ),
        ),
        metadata,
        observations,
    )


def assemble_drawing_roots(
    trace: TraceResult,
    rgb: np.ndarray | None = None,
    *,
    material_rgb: np.ndarray | None = None,
    min_region_fraction: float = 0.02,
    allow_gradients: bool = True,
    preserve_material_labels: bool | None = None,
    on_decomposition_stage: DecompositionObserver | None = None,
) -> RootDrawingResult:
    """Create retained roots, optionally reducing one-shot surfaces into roots.

    The raw trace deliberately preserves every palette fragment so path-command
    provenance is available on demand.  Its regions are therefore not the
    agent-facing roots.  When pixels are available, create those roots from the
    same raw regions that the trace exposed.  Material masks are formed from
    source-colour boundaries before their geometry and fills are fitted.
    ``preserve_material_labels`` keeps a geometry guide's explicit palette
    labels separate instead of treating similar colors as mergeable material.
    """
    if rgb is None:
        return _bind_root_evidence(trace, _raw_root_regions(trace))
    if not 0.0 <= min_region_fraction < 1.0:
        raise ValueError("min_region_fraction must be at least 0 and less than 1")
    if rgb.shape[:2] != (trace.height, trace.width):
        raise ValueError("material-merge RGB dimensions must match the trace canvas")
    if material_rgb is None and trace.material_pretrace is not None:
        material_rgb = trace.material_pretrace.material_proxy
    if preserve_material_labels is None:
        preserve_material_labels = trace.material_pretrace is not None
    segmentation_rgb = (
        rgb if material_rgb is None else np.asarray(material_rgb, dtype=np.uint8)
    )
    if segmentation_rgb.shape != rgb.shape:
        raise ValueError("material-trace RGB dimensions must match the source RGB")
    # Raw trace paths remain the plan's stable provenance.  Geometry roots use
    # the regular, AA-aware segmentation contract so a low-frequency AA shade
    # cannot punch a false notch out of a retained surface.  Each root records
    # the raw trace regions it covers.
    geometry_options = Options(
        epsilon=trace.options.simplify_tolerance,
        max_error=trace.options.curve_tolerance,
        cubic_paths=False,
        aa_contours=trace.options.trace_level == "subpixel",
        max_sampling_colors=trace.options.max_sampling_colors,
        # The relative material threshold belongs after boundary slicing and
        # compatible-material merging.  Keeping it out of seed segmentation
        # prevents small source fragments from disappearing before they can
        # become part of their intended material surface.
        min_region_fraction=0.0,
        min_region_size=trace.options.min_region_size,
    )
    _, _, seed_regions = _segment_image(segmentation_rgb, geometry_options)
    if trace.geometry_regions:
        # Background classification can reject a wholly transparent material
        # (for example the canvas color inside a native-alpha cutout). It does
        # not clip retained materials or set their subpixel contour positions.
        foreground = np.logical_or.reduce(
            [region.mask for region in trace.geometry_regions]
        )
        seed_regions = [
            region for region in seed_regions if np.any(region.mask & foreground)
        ]
    min_material_area = min_region_fraction * max(
        (region.area for region in seed_regions),
        default=0,
    )
    # Geometry is derived from the material polylines themselves. The separate
    # foreground trace is diagnostic only: it must neither replace a material's
    # subpixel exterior nor clip it back to a binary foreground contour.
    surfaces: list[tuple[Region, tuple[Region, ...]]] = []
    for component in decompose_components(seed_regions, (trace.height, trace.width)):
        labelled = list(component)
        groups = (
            [(index,) for index in range(len(labelled))]
            if preserve_material_labels
            else material_groups(labelled, rgb)
        )
        for group in groups:
            members = tuple(labelled[index] for index in group)
            representative = max(
                members, key=lambda member: (member.area, -member.label)
            )
            mask = np.logical_or.reduce([member.mask for member in members])
            surface = Region(representative.label, mask, representative.color_hex)
            if surface.area >= min_material_area:
                surfaces.append((surface, members))
    if on_decomposition_stage is not None:
        on_decomposition_stage(
            "material_union", tuple(surface for surface, _members in surfaces), None
        )
    filled_surfaces = tuple(
        (
            surface,
            members,
            fit_fill(
                surface.mask,
                rgb,
                flat_hex=surface.color_hex,
                allow_gradients=allow_gradients,
            ),
        )
        for surface, members in surfaces
    )
    if on_decomposition_stage is not None:
        on_decomposition_stage(
            "apply_fills",
            tuple(surface for surface, _members, _fill in filled_surfaces),
            tuple(selected.fill for _surface, _members, selected in filled_surfaces),
        )

    assembled: list[
        tuple[Region, np.ndarray, _SurfaceFit, Fill, tuple[str, ...], dict[str, object]]
    ] = []
    for surface, members, source_fill in filled_surfaces:
        cleaned_mask, filled_hole_area = fill_small_compatible_holes(
            surface.mask,
            rgb,
            max_area=trace.options.max_hole_area,
        )
        cleaned_surface = Region(
            surface.label,
            cleaned_mask,
            surface.color_hex,
            coverage=surface.coverage,
        )
        contours = _material_union_contours(
            members,
            added_pixels=cleaned_mask & ~surface.mask if filled_hole_area else None,
        )
        fitted_surface = _fit_contours_with_corners(contours, trace)
        if fitted_surface is None:
            continue
        corners = fitted_surface.corners
        source_regions = tuple(
            region.id for region in trace.regions if np.any(cleaned_mask & region.mask)
        )
        if source_regions:
            diagnostics: dict[str, object] = {
                "surface_merge": {
                    "sources": source_regions,
                    "merged": len(source_regions) > 1,
                    "geometry_source": "material_polyline_union",
                },
                "corners": corners,
            }
            if filled_hole_area:
                diagnostics["hole_cleanup"] = {
                    "filled_area": filled_hole_area,
                    "max_hole_area": trace.options.max_hole_area,
                }
            selected_fill = (
                fit_fill(
                    cleaned_mask,
                    rgb,
                    flat_hex=surface.color_hex,
                    allow_gradients=allow_gradients,
                )
                if filled_hole_area
                else source_fill
            )
            fill = selected_fill.fill
            diagnostics['fill'] = {'status': 'refit', 'mask': 'material_union',
                'sampled_pixels': selected_fill.sampled_pixels, 'selection': selected_fill.diagnostics()}
            assembled.append(
                (
                    cleaned_surface,
                    cleaned_mask,
                    fitted_surface,
                    fill,
                    source_regions,
                    diagnostics,
                )
            )
    trace_order = {region.id: index for index, region in enumerate(trace.regions)}
    # Geometry seeds can gain or lose a one-pixel AA fringe.  Ordering by that
    # area would make otherwise stable drawing IDs swap between equivalent
    # traces, so anchor each root to its earliest raw-trace provenance instead.
    ordered = sorted(
        assembled,
        key=lambda item: (
            min(trace_order[source] for source in item[4]),
            -item[1].sum(),
            item[0].label,
            item[0].color_hex,
        ),
    )
    return _bind_root_evidence(
        trace,
        _RootAssembly(
            tuple(
                  _AssembledSurface(
                    id=index,
                    shape=fitted.shape,
                    fill=fill,
                    z=index - 1,
                    raster=mask,
                    source_label=surface.label,
                    color_hex=surface.color_hex,
                    drawing_id=f"r{index}",
                    source_regions=source_regions,
                    diagnostics=diagnostics,
                )
                for index, (
                    surface,
                    mask,
                    fitted,
                    fill,
                    source_regions,
                    diagnostics,
                ) in enumerate(ordered, start=1)
            ),
            {index: item[2].evidence for index, item in enumerate(ordered, start=1)},
        ),
    )


def _raw_root_regions(trace: TraceResult) -> _RootAssembly:
    roots = []
    evidence = {}
    for index, region in enumerate(trace.regions):
        fitted = _fit_contours_with_corners(region.contours, trace)
        if fitted is None:
            raise ValueError("raw root fitting requires retained source contours")
        # Preserve the no-RGB entry's explicit fill-rule serialization.
        fitted.shape.params["fill_rule"] = (
            "evenodd" if len(fitted.evidence) > 1 else "nonzero"
        )
        evidence[index + 1] = fitted.evidence
        roots.append(
              _AssembledSurface(
                id=index + 1,
                shape=fitted.shape,
                fill=FlatFill(region.color),
                z=index,
                raster=region.mask,
                source_label=region.source_label,
                color_hex=region.color,
                drawing_id=region.id,
                source_regions=(region.id,),
                diagnostics={"corners": fitted.corners},
            )
        )
    return _RootAssembly(tuple(roots), evidence)


def _material_union_contours(
    members: Sequence[Region],
    *,
    added_pixels: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Union coverage-aware material polylines before any curve fitting.

    The material fields are sampled once. PathOps unions their polylines without
    a raster round trip, retaining every exterior and hole. Existing compatible
    hole cleanup may add interior pixels, but never replaces the source contour.
    """
    paths: list[SkPath] = []
    contour_groups = [
        tuple(region_contours(member.mask, coverage=member.coverage))
        for member in members
    ]
    if added_pixels is not None and added_pixels.any():
        contour_groups.append(tuple(region_contours(added_pixels)))
    for material_contours in contour_groups:
        if not material_contours:
            continue
        polyline = _polyline_trace_path(material_contours, region_id="material")
        paths.append(SkPath.from_svg_d(polyline.d, fill_rule="evenodd"))
    merged = unary_union(paths)
    contours: list[np.ndarray] = []
    for component in merged.geoms:
        for ring in (component.exterior, *component.interiors):
            points = np.asarray(ring.coords, dtype=float)
            # PathOps may retain an explicit closing vertex as well as the
            # closure marker. Remove exact duplicates, not geometric detail:
            # every source LineCommand must have a nonzero-length edge.
            if len(points):
                points = points[
                    np.r_[True, np.any(np.diff(points, axis=0) != 0.0, axis=1)]
                ]
            if len(points) >= 4:
                contours.append(points)
    return tuple(contours)


def _fit_contours_with_corners(
    contours: Sequence[np.ndarray], trace: TraceResult
) -> _SurfaceFit | None:
    contours = [contour for contour in contours if len(contour) >= 3]
    if not contours:
        return None
    stages = [
        fit_path_stages(
            contour,
            epsilon=trace.options.simplify_tolerance,
            max_error=trace.options.curve_tolerance,
            cubic=False,
            progressive=trace.options.fit_strategy != "quadratic",
            progressive_allow_lines=trace.options.fit_strategy
            == "progressive_allow_lines",
            support_weighted_corners=trace.options.support_weighted_corners,
            corner_rounding=trace.options.corner_rounding,
            canvas_extent=float(max(trace.width, trace.height)),
            # Root fitting follows the trace curve family, but it must not
            # run a second command-reduction search over a mask that has just
            # been material-merged.  This establishes the merged boundary.
            prefer_simple_curves=False,
        )
        for contour in contours
    ]
    paths = [stage.atomic_d for stage in stages]
    params: dict[str, object] = {"d": " ".join(paths)}
    if len(paths) > 1:
        params["fill_rule"] = "evenodd"
    diagnostics = tuple(
        path_corner_diagnostics(
            path,
            declared_corners=stage.corner_points,
            declared_corner_spans=stage.corner_spans,
        )
        for path, stage in zip(paths, stages, strict=True)
    )
    corners: dict[str, object] = (
        dict(diagnostics[0]) if len(diagnostics) == 1 else {"subpaths": diagnostics}
    )
    corners["path"] = str(params["d"])
    return _SurfaceFit(
        Shape("path", params),
        corners,
        tuple(stage.baseline_evidence for stage in stages),
    )

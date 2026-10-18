"""Pure serialization of retained canonical :class:`Drawing` values."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from collections.abc import Mapping
import colorsys
import hashlib
import re

from .drawing import Drawing, DrawingRegion
import numpy as np
from .skia_geometry import SkPath, svg_d_to_skia
from .geometry_native import region_path as _export_path, reflection as _reflection
from .background import BackgroundColor, background_matches
from .drawing_fill import (
    FlatFill,
    LinearGradientFill,
    RadialGradientFill,
    RasterFill,
)
from .json_value import to_json_value
from .path_geometry import (
    AffineTransform,
    CirclePrimitive,
    CloneGeometry,
    CornerAnchor,
    EllipsePrimitive,
    PathGeometry,
    PathGeometryWithCutouts,
    CompoundPathGeometry,
    PathGeometryWithSymmetry,
    PolygonPrimitive,
    PrimitiveGeometry,
    RectanglePrimitive,
    RoundedRectanglePrimitive,
)


@dataclass(frozen=True)
class CanonicalRenderedDrawing:
    svg: str
    report: Mapping[str, object]


@dataclass(frozen=True)
class _MapEntry:
    identity: str
    footprint: SkPath
    kind: str
    detail: str = ""
    path_d: str | None = None
    anchor_kind: str = ""
    source_anchor_id: str = ""
    marker_points: tuple[tuple[str, float, float], ...] = ()
    command_ids: tuple[str, ...] = ()


def _background_fill(leaf: DrawingRegion, background: BackgroundColor | None) -> bool:
    if background is None:
        return False
    fill = leaf.fill
    if isinstance(fill, FlatFill):
        colors = [fill.color]
    else:
        return False
    rgb = np.asarray([[int(color[i:i + 2], 16) for i in (1, 3, 5)] for color in colors])
    return bool(len(colors) and np.all(background_matches(rgb, background)))


def _fmt(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"


def _matrix(transform: AffineTransform) -> str:
    return "matrix({})".format(
        " ".join(
            _fmt(value)
            for value in (
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.tx,
                transform.ty,
            )
        )
    )


def _leaves(drawing: Drawing) -> tuple[DrawingRegion, ...]:
    leaves = tuple(leaf for root in drawing.regions for leaf in root.leaves())
    return tuple(sorted(leaves, key=lambda item: (item.z, item.id)))


def _geometry_kind(region: DrawingRegion) -> str:
    geometry = region.geometry
    if isinstance(geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        return "path"
    if isinstance(geometry, PathGeometryWithSymmetry):
        return "symmetric_path"
    if isinstance(geometry, CloneGeometry):
        return "clone"
    if isinstance(geometry, PrimitiveGeometry):
        return type(geometry.primitive).__name__.removesuffix("Primitive").lower()
    raise TypeError(f"unsupported canonical geometry: {type(geometry).__name__}")


def _fill_kind(region: DrawingRegion) -> str:
    if region.fill is None:
        return "none"
    return type(region.fill).__name__.removesuffix("Fill").lower()


def _path_summary(path: PathGeometry, *, correspondence: object | None) -> dict[str, object]:
    source_candidates: dict[str, set[str]] = {}
    fit_correspondence = getattr(correspondence, "correspondence", None)
    corner_features: list[dict[str, object]] = []
    if fit_correspondence is not None:
        for endpoint in fit_correspondence.endpoint_bindings:
            source_candidates.setdefault(endpoint.output_command_id, set()).add(endpoint.source_command_id)
    exact_source_commands = {
        command_id: next(iter(sources))
        for command_id, sources in source_candidates.items() if len(sources) == 1
    }
    start_id = path.path.commands[0].id
    closing_id = path.path.boundary_command_id(start_id)
    if closing_id in exact_source_commands:
        exact_source_commands[start_id] = exact_source_commands[closing_id]
    for anchor in path.anchors:
        if isinstance(anchor, CornerAnchor):
            roles = (("point", anchor.entry_command_id),) if anchor.entry_command_id == anchor.exit_command_id else (
                ("entry", anchor.entry_command_id), ("exit", anchor.exit_command_id),
            )
            corner_features.extend({"feature_id": anchor.id, "role": role, "command_id": command_id}
                                   for role, command_id in roles)
    corner_edit_supported = bool(exact_source_commands)
    return {
        "id": path.path.id,
        "d": path.to_svg_d(),
        "commands": [
            {
                "id": command.id,
                "type": type(command).__name__.removesuffix("Command").lower(),
                **{
                    key: to_json_value(value)
                    for key, value in asdict(command).items()
                    if key != "id"
                },
                "corner_boundary": {
                    "editable": command.id in exact_source_commands,
                },
            }
            for command in path.path.commands
        ],
        "anchors": [
            {
                "type": type(anchor).__name__.removesuffix("Anchor").lower(),
                **to_json_value(asdict(anchor)),
            }
            for anchor in path.anchors
        ],
        "corner_features": corner_features,
        "corner_edit": {
            "supported": corner_edit_supported,
            "reason": (
                None
                if corner_edit_supported
                else "no_exact_retained_source_output_correspondence"
            ),
        },
    }


def _geometry_summary(
    region: DrawingRegion, path_correspondence: dict[str, object]
) -> dict[str, object]:
    geometry = region.geometry
    kind = _geometry_kind(region)
    if isinstance(geometry, PathGeometry):
        return {
            "type": kind,
            **_path_summary(
                geometry, correspondence=path_correspondence.get(geometry.path.id)
            ),
        }
    if isinstance(geometry, CompoundPathGeometry):
        return {
            "type": "compound_path", "d": geometry.to_svg_d(),
            "fill_rule": geometry.fill_rule, "topology_pending": True,
            "reason": geometry.reason,
            "paths": [_path_summary(path, correspondence=path_correspondence.get(path.path.id)) for path in geometry.paths],
        }
    if isinstance(geometry, PathGeometryWithCutouts):
        return {
            "type": "path_with_cutouts",
            "d": geometry.to_svg_d(),
            "fill_rule": "evenodd",
            "paths": [
                _path_summary(
                    path, correspondence=path_correspondence.get(path.path.id)
                )
                for path in (geometry.outer, *geometry.cutouts)
            ],
        }
    if isinstance(geometry, PathGeometryWithSymmetry):
        return {
            "type": kind,
            "half": _path_summary(geometry.half, correspondence=None),
            "axis": to_json_value(asdict(geometry.axis)),
        }
    if isinstance(geometry, CloneGeometry):
        return {
            "type": kind,
            "source_region_id": geometry.source_region_id,
            "transforms": [to_json_value(asdict(item)) for item in geometry.transforms],
        }
    if isinstance(geometry, PrimitiveGeometry):
        return {
            "type": kind,
            "parameters": to_json_value(asdict(geometry.primitive)),
        }
    raise TypeError(f"unsupported canonical geometry: {type(geometry).__name__}")


def drawing_summary(
    drawing: Drawing, path_correspondence: tuple[object, ...] = ()
) -> dict[str, object]:
    """Project public target/group metadata directly from the canonical tree."""
    editable_paths: dict[str, dict[str, object]] = {}
    for record in path_correspondence:
        region_id = getattr(record, "output_region_id", None)
        path_id = getattr(record, "output_path_id", None)
        if isinstance(region_id, str) and isinstance(path_id, str):
            editable_paths.setdefault(region_id, {})[path_id] = record
    targets: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []

    def visit(region: DrawingRegion, root_id: str) -> None:
        if region.is_leaf:
            targets.append(
                {
                    "id": region.id,
                    "root_id": root_id,
                    "source_regions": list(region.source_region_ids),
                    "geometry": _geometry_summary(
                        region, editable_paths.get(region.id, {})
                    ),
                      "fill": _fill_kind(region),
                        "cutout": drawing.output_settings.background_cutouts != "off" and _background_fill(region, drawing.background),
                    "z": float(region.z),
                    "diagnostics": to_json_value(region.diagnostics),
                }
            )
            return
        groups.append(
            {
                "id": region.id,
                "child_ids": [child.id for child in region.children],
                "diagnostics": to_json_value(region.diagnostics),
            }
        )
        for child in region.children:
            visit(child, root_id)

    for root in drawing.regions:
        visit(root, root.id)
    return {"targets": targets, "groups": groups, "background": drawing.background.model_dump() if drawing.background is not None else None, "output_settings": drawing.output_settings.model_dump()}


def _geometry_element(region: DrawingRegion, geometry_id: str) -> str:
    geometry = region.geometry
    assert geometry is not None
    attrs = f'id="{escape(geometry_id)}"'
    if isinstance(geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        fill_rule = (
            ' fill-rule="evenodd"'
            if isinstance(geometry, PathGeometryWithCutouts) or (isinstance(geometry, CompoundPathGeometry) and geometry.fill_rule == "evenodd")
            else ""
        )
        return f'<path {attrs} d="{escape(geometry.to_svg_d())}"{fill_rule}/>'
    if isinstance(geometry, PathGeometryWithSymmetry):
        reflected = _reflection(geometry.axis.point, geometry.axis.angle_degrees)
        path_id = f"{geometry_id}-half"
        return (
            f'<g {attrs}><path id="{escape(path_id)}" '
            f'd="{escape(geometry.half.to_svg_d())}"/>'
            f'<use href="#{escape(path_id)}" transform="{_matrix(reflected)}"/></g>'
        )
    if isinstance(geometry, PrimitiveGeometry):
        primitive = geometry.primitive
        if isinstance(primitive, CirclePrimitive):
            return (
                f'<circle {attrs} cx="{_fmt(primitive.center[0])}" '
                f'cy="{_fmt(primitive.center[1])}" r="{_fmt(primitive.radius)}"/>'
            )
        if isinstance(primitive, EllipsePrimitive):
            return (
                f'<ellipse {attrs} cx="{_fmt(primitive.center[0])}" '
                f'cy="{_fmt(primitive.center[1])}" rx="{_fmt(primitive.radius_x)}" '
                f'ry="{_fmt(primitive.radius_y)}"/>'
            )
        if isinstance(primitive, RoundedRectanglePrimitive):
            return (
                f'<rect {attrs} x="{_fmt(primitive.origin[0])}" '
                f'y="{_fmt(primitive.origin[1])}" width="{_fmt(primitive.width)}" '
                f'height="{_fmt(primitive.height)}" rx="{_fmt(primitive.radius_x)}" '
                f'ry="{_fmt(primitive.radius_y)}"/>'
            )
        if isinstance(primitive, RectanglePrimitive):
            return (
                f'<rect {attrs} x="{_fmt(primitive.origin[0])}" '
                f'y="{_fmt(primitive.origin[1])}" width="{_fmt(primitive.width)}" '
                f'height="{_fmt(primitive.height)}"/>'
            )
        if isinstance(primitive, PolygonPrimitive):
            points = " ".join(
                f"{_fmt(x)},{_fmt(y)}" for x, y in primitive.vertices
            )
            return f'<polygon {attrs} points="{points}"/>'
    raise TypeError(f"unsupported canonical geometry: {type(geometry).__name__}")


def _fill_value(fill, fill_id: str) -> tuple[str, str | None]:
    if fill is None:
        return "none", None
    if isinstance(fill, FlatFill):
        return fill.color, None
    stops = "".join(
        f'<stop offset="{_fmt(stop.offset)}" stop-color="{escape(stop.color)}"/>'
        for stop in getattr(fill, "stops", ())
    )
    if isinstance(fill, LinearGradientFill):
        definition = (
            f'<linearGradient id="{fill_id}" gradientUnits="userSpaceOnUse" '
            f'x1="{_fmt(fill.start[0])}" y1="{_fmt(fill.start[1])}" '
            f'x2="{_fmt(fill.end[0])}" y2="{_fmt(fill.end[1])}" '
            f'gradientTransform="{_matrix(fill.transform)}">{stops}</linearGradient>'
        )
    elif isinstance(fill, RadialGradientFill):
        focal = fill.focal_point or fill.center
        definition = (
            f'<radialGradient id="{fill_id}" gradientUnits="userSpaceOnUse" '
            f'cx="{_fmt(fill.center[0])}" cy="{_fmt(fill.center[1])}" '
            f'r="{_fmt(fill.radius)}" fx="{_fmt(focal[0])}" fy="{_fmt(focal[1])}" '
            f'gradientTransform="{_matrix(fill.transform)}">{stops}</radialGradient>'
        )
    elif isinstance(fill, RasterFill):
        payload = escape(fill.png_base64)
        definition = (
            f'<pattern id="{fill_id}" patternUnits="userSpaceOnUse" '
            f'x="{_fmt(fill.origin[0])}" y="{_fmt(fill.origin[1])}" '
            f'width="{_fmt(fill.width)}" height="{_fmt(fill.height)}" '
            f'patternTransform="{_matrix(fill.transform)}">'
            f'<image x="{_fmt(fill.origin[0])}" y="{_fmt(fill.origin[1])}" '
            f'width="{_fmt(fill.width)}" height="{_fmt(fill.height)}" '
            f'href="data:image/png;base64,{payload}"/></pattern>'
        )
    else:
        raise TypeError(f"unsupported canonical fill: {type(fill).__name__}")
    return f"url(#{fill_id})", definition


def _paths_overlap(first: SkPath, second: SkPath) -> bool:
    if first.is_empty or second.is_empty:
        return False
    ax, ay, bx, by = first.bounds
    cx, cy, dx, dy = second.bounds
    if bx <= cx or dx <= ax or by <= cy or dy <= ay:
        return False
    return not first.intersection(second).is_empty


def _render_background_cutouts(
    drawing: Drawing, leaves: tuple[DrawingRegion, ...], path_correspondence: tuple[object, ...],
) -> CanonicalRenderedDrawing:
    """Final export projection only; later foreground never becomes a knockout."""
    by_id = {leaf.id: leaf for leaf in leaves}
    later_cutouts = SkPath()
    painted: list[str] = []
    definitions: list[str] = []
    omitted = 0
    affected = 0
    width, height = drawing.canvas.width, drawing.canvas.height
    for index, leaf in reversed(tuple(enumerate(leaves))):
        path = _export_path(leaf, by_id)
        if _background_fill(leaf, drawing.background):
            omitted += 1
            later_cutouts = later_cutouts.union(path)
            continue
        overlap = _paths_overlap(path, later_cutouts)
        mask_attr = ""
        if overlap:
            affected += 1
            if drawing.output_settings.background_cutouts == "subtract":
                path = path.difference(later_cutouts)
            else:
                mask_id = f"background-cutouts-{index}"
                definitions.append(
                    f'<mask id="{mask_id}" maskUnits="userSpaceOnUse" x="0" y="0" width="{_fmt(width)}" height="{_fmt(height)}" style="mask-type:luminance">'
                    f'<rect width="{_fmt(width)}" height="{_fmt(height)}" fill="white"/>'
                    f'<path d="{escape(later_cutouts.to_svg_d())}" fill="black" fill-rule="evenodd"/></mask>'
                )
                mask_attr = f' mask="url(#{mask_id})"'
        if path.is_empty:
            continue
        fill, definition = _fill_value(leaf.fill, f"paint-{index}")
        if definition is not None:
            definitions.append(definition)
        if not overlap and not isinstance(leaf.geometry, (CloneGeometry, PathGeometryWithSymmetry)):
            element = _geometry_element(leaf, leaf.id)
            painted.append(element.replace(f'id="{escape(leaf.id)}"', f'id="{escape(leaf.id)}" fill="{escape(fill)}"', 1))
        else:
            painted.append(f'<path id="{escape(leaf.id)}" d="{escape(path.to_svg_d())}" fill="{escape(fill)}" fill-rule="evenodd"{mask_attr}/>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(width)}" height="{_fmt(height)}" viewBox="0 0 {_fmt(width)} {_fmt(height)}">'
        f'<defs>{"".join(definitions)}</defs>{"".join(reversed(painted))}</svg>'
    )
    report = drawing_summary(drawing, path_correspondence)
    report["export"] = {
        "background_cutouts": drawing.output_settings.background_cutouts,
        "background_regions_omitted": omitted,
        "painted_regions_affected": affected,
    }
    return CanonicalRenderedDrawing(svg, report)


def render_drawing(
    drawing: Drawing, path_correspondence: tuple[object, ...] = ()
) -> CanonicalRenderedDrawing:
    """Serialize one stored Drawing without fitting, optimizing, or repair."""
    leaves = _leaves(drawing)
    if drawing.output_settings.background_cutouts != "off" and any(_background_fill(leaf, drawing.background) for leaf in leaves):
        return _render_background_cutouts(drawing, leaves, path_correspondence)
    by_id = {leaf.id: leaf for leaf in leaves}
    geometry_defs: list[str] = []
    clone_clip_defs: list[str] = []
    symmetry_clip_defs: list[str] = []
    fill_defs: list[str] = []
    body: list[str] = []
    for leaf in leaves:
        if not isinstance(leaf.geometry, CloneGeometry):
            geometry_defs.append(_geometry_element(leaf, f"geometry-{leaf.id}"))
            if isinstance(leaf.geometry, PathGeometryWithSymmetry):
                half_id = f"clip-{leaf.id}-half"
                reflected = _reflection(
                    leaf.geometry.axis.point, leaf.geometry.axis.angle_degrees
                )
                symmetry_clip_defs.append(
                    f'<clipPath id="clip-{escape(leaf.id)}" clipPathUnits="userSpaceOnUse">'
                    f'<path id="{escape(half_id)}" d="{escape(leaf.geometry.half.to_svg_d())}"/>'
                    f'<use href="#{escape(half_id)}" transform="{_matrix(reflected)}"/>'
                    f'</clipPath>'
                )
    for leaf in leaves:
        geometry = leaf.geometry
        if not isinstance(geometry, CloneGeometry):
            continue
        source = by_id.get(geometry.source_region_id)
        if source is None or isinstance(source.geometry, CloneGeometry):
            raise ValueError("clone source must identify a concrete retained region")
        uses = "".join(
            f'<use href="#geometry-{escape(source.id)}" transform="{_matrix(transform)}"/>'
            for transform in geometry.transforms
        )
        clone_clip_defs.append(
            f'<clipPath id="clip-{escape(leaf.id)}" clipPathUnits="userSpaceOnUse">'
            f'{uses}</clipPath>'
        )
    for index, leaf in enumerate(leaves):
        fill_value, definition = _fill_value(leaf.fill, f"paint-{index}")
        if definition is not None:
            fill_defs.append(definition)
        geometry = leaf.geometry
        assert geometry is not None
        if isinstance(geometry, (CloneGeometry, PathGeometryWithSymmetry)):
            clone_source = (
                f' data-clone-source="{escape(geometry.source_region_id)}"'
                if isinstance(geometry, CloneGeometry)
                else ""
            )
            body.append(
                f'<rect id="{escape(leaf.id)}" x="0" y="0" '
                f'width="{_fmt(drawing.canvas.width)}" height="{_fmt(drawing.canvas.height)}" '
                f'fill="{escape(fill_value)}" clip-path="url(#clip-{escape(leaf.id)})" '
                f'{clone_source}/>'
            )
        else:
            element = _geometry_element(leaf, leaf.id)
            body.append(
                element.replace(
                    f'id="{escape(leaf.id)}"',
                    f'id="{escape(leaf.id)}" fill="{escape(fill_value)}"',
                    1,
                )
            )
    width, height = drawing.canvas.width, drawing.canvas.height
    painted = "".join(body)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(width)}" '
        f'height="{_fmt(height)}" viewBox="0 0 {_fmt(width)} {_fmt(height)}">'
        f'<defs>{"".join(geometry_defs)}{"".join(clone_clip_defs)}'
        f'{"".join(symmetry_clip_defs)}{"".join(fill_defs)}</defs>'
        f'{painted}</svg>'
    )
    return CanonicalRenderedDrawing(
        svg, drawing_summary(drawing, path_correspondence)
    )


def render_labeled_drawing(drawing: Drawing) -> str:
    """Show region IDs spatially, with a matching color-coded side legend."""
    leaves = tuple(_leaves(drawing))
    by_id = {leaf.id: leaf for leaf in leaves}
    entries = tuple(
        _MapEntry(leaf.id, _export_path(leaf, by_id), "region")
        for leaf in leaves
    )
    return _render_selection_map(drawing, entries, title="Region map")


def render_path_map(drawing: Drawing, *, region_ids: tuple[str, ...] = (), command_types: tuple[str, ...] = ()) -> str:
    """Color retained segments and annotate their stored semantic anchors."""
    from .path_topology import component_paths

    leaves = tuple(_leaves(drawing))
    if set(region_ids) - {leaf.id for leaf in leaves}:
        raise ValueError("path map region_ids must identify leaf regions in this version")
    entries: list[_MapEntry] = []
    for leaf in leaves:
        if region_ids and leaf.id not in region_ids:
            continue
        geometry = leaf.geometry
        if isinstance(geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
            paths = component_paths(geometry)
        elif isinstance(geometry, PathGeometryWithSymmetry):
            paths = (geometry.half,)
        else:
            continue
        for path in paths:
            fragments = tuple(piece.strip() for piece in re.findall(r"[LQCA][^MLQCAZ]*", path.to_svg_d()))
            start = path.path.start
            pieces: dict[str, str] = {}
            selected: set[str] = set()
            for command, fragment in zip(path.path.drawable_commands, fragments, strict=True):
                family = {"L": "line", "Q": "quadratic", "C": "cubic", "A": "arc"}[fragment[0]]
                d = f"M{_fmt(start[0])} {_fmt(start[1])} {fragment}"
                pieces[command.id] = d
                if not command_types or family in command_types:
                    entries.append(_MapEntry(command.id, SkPath.from_svg_d(d), "segment", family, d))
                    selected.add(command.id)
                start = command.end
            for anchor in path.anchors:
                if command_types and not selected.intersection(anchor.command_ids):
                    continue
                corner = isinstance(anchor, CornerAnchor)
                rounded = corner and anchor.entry_command_id != anchor.exit_command_id
                boundaries = (
                    (("entry", anchor.entry_command_id), ("exit", anchor.exit_command_id))
                    if rounded else (("point", anchor.entry_command_id),) if corner
                    else tuple(("member", command_id) for command_id in anchor.command_ids)
                )
                markers = tuple((role, *path.endpoint(command_id)) for role, command_id in boundaries)
                owned = anchor.owned_command_ids if corner else ()
                span_d = " ".join(pieces[command_id] for command_id in owned)
                # A tiny diagnostic line gives isolated point anchors bounds;
                # the visible marker is drawn separately, never added to geometry.
                footprint_d = span_d or " ".join(
                    f"M{_fmt(x - 0.01)} {_fmt(y)} L{_fmt(x + 0.01)} {_fmt(y)}"
                    for _role, x, y in markers
                )
                identity = f"{path.path.id}/{anchor.id}"
                detail = anchor.kind + (" (rounded span)" if rounded else " (point)" if corner else "")
                entries.append(_MapEntry(identity, SkPath.from_svg_d(footprint_d), "anchor", detail,
                                         footprint_d, anchor.kind, anchor.id, markers, anchor.command_ids))
    entries.sort(key=lambda entry: entry.kind == "anchor")
    return _render_selection_map(drawing, tuple(entries), title="Path segment map")


def _render_selection_map(drawing: Drawing, entries: tuple[_MapEntry, ...], *, title: str) -> str:
    """One non-mutating layout for map colors, labels, callouts and legends."""
    visible = tuple(entry for entry in entries if not entry.footprint.is_empty)
    if visible:
        bounds = tuple(entry.footprint.bounds for entry in visible)
        min_x, min_y = min(b[0] for b in bounds), min(b[1] for b in bounds)
        max_x, max_y = max(b[2] for b in bounds), max(b[3] for b in bounds)
    else:
        min_x, min_y, max_x, max_y = 0.0, 0.0, drawing.canvas.width, drawing.canvas.height
    segments = title == "Path segment map"
    font = max(16.0, max(max_x - min_x, max_y - min_y) / (45.0 if segments else 28.0))
    pad = font * 1.2
    row = font * 1.8
    rows = max(8, min(24, int(max(max_y - min_y, row * 8) / row)))
    segment_count = sum(entry.kind != "anchor" for entry in visible)
    anchor_count = len(visible) - segment_count
    segment_columns = (segment_count + rows - 1) // rows
    columns = max(1, segment_columns + (anchor_count + rows - 1) // rows)
    legend_width = max(font * 9, font * 0.6 * (max((len(entry.identity) for entry in visible if entry.kind != "anchor"), default=12) + 6))
    anchor_legend_width = max(font * 9, font * 0.6 * (max((len(entry.identity) for entry in visible if entry.kind == "anchor"), default=12) + 6))
    legend_x = max_x + pad * 2
    top = min_y + (row if anchor_count else 0)
    bottom = max(max_y + pad, top + min(rows, max(1, len(visible))) * row + pad * 2)
    width = legend_x + segment_columns * legend_width + (columns - segment_columns) * anchor_legend_width + pad - (min_x - pad)
    height = bottom - (min_y - pad * 2.5)
    overlays = []
    labels = []
    legend = []
    for index, entry in enumerate(visible):
        identity, footprint, kind, detail = entry.identity, entry.footprint, entry.kind, entry.detail
        hue = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big") / 2**32
        rgb = colorsys.hls_to_rgb(hue, 0.40, 0.72)
        color = "#" + "".join(f"{round(channel * 255):02X}" for channel in rgb)
        d = entry.path_d or footprint.to_svg_d()
        key = escape(identity)
        identity_attribute = "command" if kind == "segment" else kind
        anchor_attributes = (
            f' data-anchor-kind="{escape(entry.anchor_kind)}" data-source-anchor-id="{escape(entry.source_anchor_id)}"'
            f' data-command-ids="{escape(" ".join(entry.command_ids))}"'
            if kind == "anchor" else ""
        )
        overlays.append(f'<path id="map-{key}" data-{identity_attribute}-id="{key}"{anchor_attributes} d="{escape(d)}" '
                        f'fill="{color if kind == "region" else "none"}" fill-opacity="0.20" '
                        f'fill-rule="evenodd" stroke="{color}" stroke-width="{_fmt(font / (3 if kind == "anchor" else 7))}">'
                        f'<title>{key} {escape(detail)}</title></path>')
        for role, mx, my in entry.marker_points:
            radius = font * 0.3
            attrs = f'data-anchor-role="{role}" data-anchor-ref="{key}" fill="{color}" stroke="white" stroke-width="{_fmt(font / 10)}"'
            if entry.anchor_kind == "corner" and role in {"entry", "exit"}:
                marker = f'<circle cx="{_fmt(mx)}" cy="{_fmt(my)}" r="{_fmt(radius)}" {attrs}/>'
            else:
                marker = f'<path d="M{_fmt(mx)} {_fmt(my-radius)} L{_fmt(mx+radius)} {_fmt(my)} L{_fmt(mx)} {_fmt(my+radius)} L{_fmt(mx-radius)} {_fmt(my)} Z" {attrs}/>'
            labels.append(f'<g><title>{key} {escape(detail)} {role}</title>{marker}</g>')
        x0, y0, x1, y1 = footprint.bounds
        native = svg_d_to_skia(d, fill_rule="evenodd")
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        candidates = [(x0 + (x1 - x0) * u / 10, y0 + (y1 - y0) * v / 10)
                      for u in range(1, 10) for v in range(1, 10)]
        inside = [point for point in candidates if native.contains(*point)]
        x, y = min(inside, key=lambda point: (point[0] - cx)**2 + (point[1] - cy)**2) if inside else (cx, cy)
        legend_index = index if kind != "anchor" else segment_columns * rows + index - segment_count
        lx = (legend_x + (index // rows) * legend_width if kind != "anchor" else
              legend_x + segment_columns * legend_width + ((index - segment_count) // rows) * anchor_legend_width)
        ly = top + (legend_index % rows) * row
        fits = (kind == "region" and x1 - x0 >= font * 0.58 * len(identity) and y1 - y0 >= font * 1.5 and bool(inside))
        if fits:
            labels.append(f'<text x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="middle" dominant-baseline="middle" '
                          f'font-size="{_fmt(font)}" fill="#111827" stroke="white" stroke-width="{_fmt(font / 8)}" '
                          f'paint-order="stroke">{key}</text>')
        legend.append(f'<g data-legend-id="{key}" data-legend-kind="{kind}"><rect x="{_fmt(lx)}" y="{_fmt(ly - font * 0.6)}" '
                      f'width="{_fmt(font * 0.65)}" height="{_fmt(font * 0.65)}" rx="{_fmt(font / 8)}" fill="{color}"/>'
                      f'<text x="{_fmt(lx + font)}" y="{_fmt(ly)}" font-size="{_fmt(font)}" fill="#111827">{key}</text>'
                      f'<text x="{_fmt(lx + font)}" y="{_fmt(ly + font * 0.7)}" font-size="{_fmt(font * 0.55)}" fill="#475569">{escape(detail)}</text></g>')
    if anchor_count:
        for heading, offset in (("Segments", 0), ("Anchors", segment_columns)):
            legend.append(f'<text x="{_fmt(legend_x + offset * legend_width)}" y="{_fmt(min_y)}" font-size="{_fmt(font)}" font-weight="bold">{heading}</text>')
    base = render_drawing(drawing).svg
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_fmt(width)}" height="{_fmt(height)}" '
        f'viewBox="{_fmt(min_x - pad)} {_fmt(min_y - pad * 2.5)} {_fmt(width)} {_fmt(height)}" font-family="sans-serif">'
        f'<title>{title}</title><rect x="{_fmt(min_x - pad)}" y="{_fmt(min_y - pad * 2.5)}" width="{_fmt(width)}" height="{_fmt(height)}" fill="white"/>'
        f'<g opacity="0.15">{base}</g><g data-vectormark-labels="true">{"".join(overlays)}{"".join(labels)}</g>'
        f'<text x="{_fmt(min_x)}" y="{_fmt(min_y - pad)}" font-size="{_fmt(font * 1.2)}" font-weight="bold">{title}</text>'
        f'<g data-vectormark-legend="true">{"".join(legend)}</g></svg>'
    )


def render_command_map(
    drawing: Drawing,
    path_correspondence: tuple[object, ...] = (),
    *,
    region_ids: tuple[str, ...] = (),
    command_types: tuple[str, ...] = (),
) -> str:
    """Overlay addressable command endpoints on the retained drawing, without refitting."""
    summary = drawing_summary(drawing, path_correspondence)
    targets = summary["targets"]
    known = {target["id"] for target in targets}
    if set(region_ids) - known:
        raise ValueError("command map region_ids must identify leaf regions in this version")
    labels: list[str] = []
    for target in targets:
        if region_ids and target["id"] not in region_ids:
            continue
        geometry = target["geometry"]
        paths = geometry.get("paths", [geometry.get("half", geometry)])
        for path in paths:
            for command in path.get("commands", []):
                if command_types and command["type"] not in command_types:
                    continue
                end = command.get("end")
                if end is None:
                    continue
                x, y = end
                label = escape(command["id"])
                labels.append(
                    f'<g data-command-id="{label}"><circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="3" '
                    f'fill="#e11d48" stroke="white" stroke-width="1"/>'
                    f'<text x="{_fmt(x + 6)}" y="{_fmt(y - 6)}" font-size="12" '
                    f'fill="#111827" stroke="white" stroke-width="3" paint-order="stroke">{label}</text></g>'
                )
    svg = render_drawing(drawing, path_correspondence).svg
    return svg.replace("</svg>", '<g data-vectormark-command-map="true">' + "".join(labels) + '</g></svg>')

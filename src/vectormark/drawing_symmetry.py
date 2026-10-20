"""Retained self-symmetry execution and shared numeric reconstruction.

Discovery and winner selection belong to optimizer passes. These operations
apply the selected fitter/axis through the existing topology/clone transaction.
Working-region evidence remains a transitional adapter, not a second scene.
"""

from __future__ import annotations
import math
from collections import namedtuple
from dataclasses import replace
from collections.abc import Mapping, Sequence
import numpy as np
from .drawing import DrawingRegion
from .drawing_actions import SymmetryAxisInput
from .fit import Shape, _fmt, fit_path
from .refine import fit_path_half_fit, half_ellipse_cap_half_fit, symmetric_half_fit
from .skia_geometry import SkPath, unary_union
from .path_geometry import parse_svg_paths
from .path_sampling import _parse_subpaths

Axis2D = namedtuple("Axis2D", "theta cx cy")
_SELF_AXIS_OVERLAP = 0.75
_MIN_SKIA_CONTOUR_AREA = 1.0
_MIN_SKIA_CONTOUR_AREA_FRACTION = 1e-4


def _polygonal_flat(flat: object) -> SkPath | None:
    if isinstance(flat, SkPath) and not flat.is_empty:
        return flat
    return None


def _svg_reflection_matrix(
    axis: Axis2D,
) -> tuple[float, float, float, float, float, float]:
    theta = float(axis.theta)
    ux = math.cos(theta)
    uy = math.sin(theta)
    a = 2.0 * ux * ux - 1.0
    b = 2.0 * ux * uy
    c = 2.0 * ux * uy
    d = 2.0 * uy * uy - 1.0
    e = float(axis.cx) - (a * float(axis.cx) + c * float(axis.cy))
    f = float(axis.cy) - (b * float(axis.cx) + d * float(axis.cy))
    vals = (a, b, c, d, e, f)
    return tuple(0.0 if abs(v) < 1e-12 else float(v) for v in vals)


def _reflect_flat(flat: SkPath, axis: Axis2D) -> SkPath:
    return flat.affine_transform(_svg_reflection_matrix(axis))


def _project_to_axis(point: tuple[float, float], axis: Axis2D) -> tuple[float, float]:
    """Orthogonally project a point onto a symmetry axis."""
    ux, uy = math.cos(float(axis.theta)), math.sin(float(axis.theta))
    dx, dy = point[0] - float(axis.cx), point[1] - float(axis.cy)
    distance = dx * ux + dy * uy
    return (float(axis.cx) + distance * ux, float(axis.cy) + distance * uy)


def _endpoint_indices(command: str) -> tuple[int, int] | None:
    return {
        "M": (0, 1),
        "L": (0, 1),
        "Q": (2, 3),
        "C": (4, 5),
        "A": (5, 6),
    }.get(command)


def _pin_half_shape_to_axis(
    shape: Shape, axis: Axis2D, *, tolerance: float = 2.0
) -> Shape | None:
    """Make a fitted half close exactly along its symmetry axis.

    The fitters intentionally focus on the exterior boundary.  Their closure
    points can still be sub-pixel-adjacent to the cut, which becomes a visible
    gap after mirroring.  Project only the start and final endpoint; SVG's Z
    then supplies a perfectly straight, coincident axis edge for Skia to union.
    """
    if shape.kind != "path":
        return None
    subpaths = _parse_subpaths(str(shape.params.get("d", "")))
    if len(subpaths) != 1 or len(subpaths[0]) < 3 or subpaths[0][0][0] != "M":
        return None

    commands = [(command, list(values)) for command, values in subpaths[0]]
    final_index = next(
        (
            index
            for index in range(len(commands) - 1, -1, -1)
            if _endpoint_indices(commands[index][0]) is not None
        ),
        None,
    )
    if final_index is None or final_index == 0:
        return None

    def pin(index: int) -> bool:
        command, values = commands[index]
        endpoint = _endpoint_indices(command)
        assert endpoint is not None
        current = (float(values[endpoint[0]]), float(values[endpoint[1]]))
        projected = _project_to_axis(current, axis)
        if math.dist(current, projected) > tolerance:
            return False
        values[endpoint[0]], values[endpoint[1]] = projected
        return True

    if not pin(0) or not pin(final_index):
        return None

    def command_d(command: str, values: list[float]) -> str:
        return (
            command
            if not values
            else f"{command}{' '.join(_fmt(value) for value in values)}"
        )

    params = dict(shape.params)
    params["d"] = " ".join(command_d(command, values) for command, values in commands)
    return Shape("path", params)


def _half_plane(flat: SkPath, axis: Axis2D) -> SkPath:
    minx, miny, maxx, maxy = flat.bounds
    span = max(maxx - minx, maxy - miny, 1.0) * 8.0
    ux = math.cos(axis.theta)
    uy = math.sin(axis.theta)
    nx = -uy
    ny = ux
    cx = float(axis.cx)
    cy = float(axis.cy)
    seam_x = nx * _SELF_AXIS_OVERLAP
    seam_y = ny * _SELF_AXIS_OVERLAP
    return SkPath(
        shell=[
            (cx - ux * span - seam_x, cy - uy * span - seam_y),
            (cx + ux * span - seam_x, cy + uy * span - seam_y),
            (cx + ux * span + nx * span, cy + uy * span + ny * span),
            (cx - ux * span + nx * span, cy - uy * span + ny * span),
        ]
    )


def _canonical_half(flat: SkPath, axis: Axis2D) -> SkPath | None:
    half = flat.intersection(_half_plane(flat, axis))
    if half.is_empty or not isinstance(half, SkPath):
        return None
    return half.buffer(0)


def _ring_d(coords) -> str:
    pts = list(coords)
    if len(pts) < 4:
        return ""
    body = " ".join(f"L{_fmt(float(x))} {_fmt(float(y))}" for x, y in pts[1:])
    return f"M{_fmt(float(pts[0][0]))} {_fmt(float(pts[0][1]))} {body} Z"


def _coords_area(coords) -> float:
    pts = list(coords)
    if len(pts) < 4:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:], strict=False):
        area += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(area) / 2.0


def _fit_ring_path(coords, *, epsilon: float, max_error: float, cubic: bool) -> str:
    pts = np.asarray(coords, dtype=float)
    if len(pts) < 4:
        return _ring_d(coords)
    try:
        return str(
            fit_path(
                pts,
                epsilon=epsilon,
                max_error=max_error,
                cubic=cubic,
                prefer_simple_curves=True,
            ).params["d"]
        )
    except Exception:
        return _ring_d(coords)


def _geometry_to_path_shape(
    flat: SkPath,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape | None:
    polygons = flat.geoms
    polygons = sorted(
        [poly for poly in polygons if not poly.is_empty],
        key=lambda poly: (
            round(float(poly.bounds[0]), 6),
            round(float(poly.bounds[1]), 6),
            -round(float(poly.area), 6),
        ),
    )
    max_area = max((float(poly.area) for poly in polygons), default=0.0)
    contour_area_floor = max(
        _MIN_SKIA_CONTOUR_AREA, max_area * _MIN_SKIA_CONTOUR_AREA_FRACTION
    )
    parts: list[str] = []
    has_holes = False
    for poly in polygons:
        if float(poly.area) < contour_area_floor:
            continue
        parts.append(
            _fit_ring_path(
                poly.exterior.coords, epsilon=epsilon, max_error=max_error, cubic=cubic
            )
        )
        kept_holes = [
            ring
            for ring in poly.interiors
            if _coords_area(ring.coords) >= contour_area_floor
        ]
        parts.extend(
            _fit_ring_path(
                ring.coords, epsilon=epsilon, max_error=max_error, cubic=cubic
            )
            for ring in kept_holes
        )
        has_holes = has_holes or bool(kept_holes)
    parts = [part for part in parts if part]
    if not parts:
        return None
    params: dict[str, object] = {"d": " ".join(parts)}
    if has_holes or len(polygons) > 1:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


def materialize_symmetric_half(
    region: DrawingRegion,
    *,
    axis: Axis2D,
    half: SkPath,
    reflected_half: SkPath,
    half_shape: Shape,
    residual: float,
) -> DrawingRegion:
    matrix = _svg_reflection_matrix(axis)
    half_shape = _pin_half_shape_to_axis(half_shape, axis) or half_shape
    fitted_half = SkPath.from_svg_d(half_shape.params['d'])
    if fitted_half.is_empty:
        fitted_half = half
    fitted_reflected_half = fitted_half.affine_transform(matrix)
    if fitted_reflected_half.is_empty:
        fitted_reflected_half = reflected_half
    return union_symmetric_regions(
        region,
        (fitted_half, fitted_reflected_half),
        diagnostics=_self_symmetry_diagnostics(axis, residual)["symmetry"],
    )


def union_symmetric_regions(
    region: DrawingRegion, pieces: Sequence[SkPath], *, diagnostics: Mapping[str, object],
) -> DrawingRegion:
    """Publish the selected vector union as canonical filled components."""
    from .drawing_topology import publish_region_paths

    footprint = unary_union(pieces)
    if footprint.is_empty:
        raise ValueError('self-symmetry materialization produced empty geometry')
    leaves = region.leaves()
    if any(leaf.fill != leaves[0].fill for leaf in leaves):
        raise ValueError('self-symmetry materialization requires one shared fill')
    consumed = tuple(leaf.id for leaf in leaves)
    merged = {**region.diagnostics,
        'symmetry': {**diagnostics, 'materialized': True},
        'materialization': {'operation': 'self_symmetry_union',
            'consumed_target_ids': consumed,
            'retired_target_ids': tuple(identity for identity in consumed if identity != region.id),
            'source_target_remap': tuple((identity, region.id) for identity in consumed)}}
    paths = parse_svg_paths(footprint.to_svg_d(), region_id=region.id)
    selected = replace(region, geometry=paths[0], children=(), fill=leaves[0].fill,
        source_region_ids=tuple(sorted(set(region.source_region_ids).union(
            *(leaf.source_region_ids for leaf in leaves)))), diagnostics=merged)
    return publish_region_paths(selected, paths, fill_rule='evenodd')


def _axis_diagnostics(axis: Axis2D) -> dict[str, object]:
    """Serialize every detected reflection line in the explicit-action format."""
    return SymmetryAxisInput(
        point=(float(axis.cx), float(axis.cy)),
        angle_degrees=math.degrees(float(axis.theta)),
    ).model_dump(mode="json")


def _self_symmetry_diagnostics(
    axis: Axis2D, residual: float
) -> dict[str, dict[str, object]]:
    return {
        "symmetry": {
            "accepted": True,
            "mode": "self",
            "axis": _axis_diagnostics(axis),
            "residual": float(residual),
        }
    }


def _intrinsic_symmetry_diagnostics(
    axis: Axis2D, residual: float
) -> dict[str, dict[str, object]]:
    """Report verified symmetry without replacing an explicit full geometry."""
    diagnostics = _self_symmetry_diagnostics(axis, residual)
    diagnostics["symmetry"]["mode"] = "intrinsic"
    return diagnostics


def _fit_polygon_path(
    poly: SkPath, *, epsilon: float, max_error: float, cubic: bool
) -> str | None:
    coords = np.asarray(poly.exterior.coords, dtype=float)
    if len(coords) < 4:
        return None
    try:
        fitted = fit_path(
            coords,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
            prefer_simple_curves=True,
        )
    except Exception:
        return None
    return str(fitted.params["d"])


def _fit_geometry_to_path_shape(
    flat: SkPath,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape | None:
    if not flat.is_empty and len(flat.geoms) > 1:
        return _geometry_to_path_shape(
            flat, epsilon=epsilon, max_error=max_error, cubic=cubic
        )
    if flat.is_empty:
        return None
    exterior = _fit_polygon_path(
        flat, epsilon=epsilon, max_error=max_error, cubic=cubic
    )
    if exterior is None:
        return _geometry_to_path_shape(
            flat, epsilon=epsilon, max_error=max_error, cubic=cubic
        )
    parts = [exterior]
    contour_area_floor = max(
        _MIN_SKIA_CONTOUR_AREA, float(flat.area) * _MIN_SKIA_CONTOUR_AREA_FRACTION
    )
    kept_holes = [
        ring
        for ring in flat.interiors
        if _coords_area(ring.coords) >= contour_area_floor
    ]
    parts.extend(
        _fit_ring_path(ring.coords, epsilon=epsilon, max_error=max_error, cubic=cubic)
        for ring in kept_holes
    )
    params: dict[str, object] = {"d": " ".join(part for part in parts if part)}
    if kept_holes:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


def _vertical_half_side(half: SkPath, axis: Axis2D) -> str:
    return "left" if float(half.centroid.x) < float(axis.cx) else "right"


def _symmetric_refine_half_shapes(
    flat: SkPath,
    half: SkPath,
    axis: Axis2D,
    *,
    epsilon: float,
    max_error: float,
    corner_radius: float,
    cubic: bool,
) -> tuple[tuple[str, Shape], ...]:
    if flat.interiors or len(flat.geoms) != 1:
        return ()
    # A half-ellipse cap is bilaterally symmetric about a vertical line.
    if abs(math.cos(float(axis.theta))) > 1e-6:
        return ()
    side = _vertical_half_side(half, axis)
    contour = np.asarray(flat.exterior.coords, dtype=float)
    shapes: list[tuple[str, Shape]] = []
    for fit_kind, build in (
        (
            "ellipse_cap",
            lambda: half_ellipse_cap_half_fit(
                contour, float(axis.cx), side=side, max_error=max_error
            ),
        ),
        (
            "symmetric_half",
            lambda: symmetric_half_fit(
                contour,
                float(axis.cx),
                side=side,
                corner_radius=corner_radius,
                epsilon=epsilon,
                max_error=max_error,
            ),
        ),
    ):
        try:
            shape = build()
        except Exception:
            continue
        if shape is not None:
            shapes.append((fit_kind, shape))
    return tuple(shapes)


def apply_intrinsic_symmetry(
    target: DrawingRegion, axis: Axis2D, residual: float
) -> DrawingRegion:
    if target.geometry is None:
        raise ValueError("intrinsic symmetry target has no geometry")
    return replace(target, diagnostics={**target.diagnostics,
        **_intrinsic_symmetry_diagnostics(axis, residual)})


def _selected_half_shape(
    flat: SkPath,
    half: SkPath,
    axis: Axis2D,
    *,
    fit_kind: str,
    side: str,
    corner_radius: float,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape:
    contour = np.asarray(flat.exterior.coords, dtype=float)
    if fit_kind == "ellipse_cap":
        shape = half_ellipse_cap_half_fit(
            contour, float(axis.cx), side=side, max_error=max_error
        )
    elif fit_kind == "symmetric_half":
        shape = symmetric_half_fit(
            contour,
            float(axis.cx),
            side=side,
            corner_radius=corner_radius,
            epsilon=epsilon,
            max_error=max_error,
        )
    elif fit_kind == "path_half":
        shape = fit_path_half_fit(
            contour,
            float(axis.cx),
            side=side,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
        )
    elif fit_kind == "geometry_path":
        shape = _fit_geometry_to_path_shape(
            half, epsilon=epsilon, max_error=max_error, cubic=cubic
        )
    else:
        raise ValueError(f"unsupported recorded symmetry fit kind: {fit_kind!r}")
    if shape is None:
        raise ValueError(
            "recorded symmetry fitter no longer applies to its bound input"
        )
    pinned = _pin_half_shape_to_axis(shape, axis)
    if pinned is None:
        raise ValueError("recorded symmetry result cannot be pinned to its axis")
    return pinned


def reconstruct_self_symmetry(
    target: DrawingRegion,
    reconstruction_flat: SkPath,
    *,
    axis: Axis2D,
    residual: float,
    fit_kind: str,
    side: str,
    corner_radius: float,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> DrawingRegion:
    if _polygonal_flat(reconstruction_flat) is None:
        raise ValueError("recorded self-symmetry target has no polygonal footprint")
    half = _canonical_half(reconstruction_flat, axis)
    if half is None:
        raise ValueError("recorded self-symmetry axis does not split its bound input")
    reflected_half = _reflect_flat(half, axis)
    shape = _selected_half_shape(
        reconstruction_flat,
        half,
        axis,
        fit_kind=fit_kind,
        side=side,
        corner_radius=corner_radius,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
    )
    return materialize_symmetric_half(
        target,
        axis=axis,
        half=half,
        reflected_half=reflected_half,
        half_shape=shape,
        residual=residual,
    )

"""Primitive fitting kernels, independent of the temporary region scene."""

from __future__ import annotations

import numpy as np

from .drawing_actions import PrimitiveKind
from .fit import Shape, recognize_polygon, recognize_primitive
from .skia_geometry import SkPath
from .path_geometry import (
    CirclePrimitive,
    EllipsePrimitive,
    PolygonPrimitive,
    PrimitiveGeometry,
    RectanglePrimitive,
    RoundedRectanglePrimitive,
)


def primitive_geometry_from_shape(shape: Shape) -> PrimitiveGeometry:
    """Decode numeric primitive output without fitting or region-scene access."""
    params = shape.params
    if shape.kind == "circle":
        return PrimitiveGeometry(CirclePrimitive((params["cx"], params["cy"]), params["r"]))
    if shape.kind == "ellipse":
        return PrimitiveGeometry(EllipsePrimitive(
            (params["cx"], params["cy"]), params["rx"], params["ry"]
        ))
    if shape.kind == "rect":
        origin = (params["x"], params["y"])
        if "rx" in params or "ry" in params:
            return PrimitiveGeometry(RoundedRectanglePrimitive(
                origin, params["w"], params["h"],
                params.get("rx", params.get("ry", 0.0)),
                params.get("ry", params.get("rx", 0.0)),
            ))
        return PrimitiveGeometry(RectanglePrimitive(origin, params["w"], params["h"]))
    if shape.kind == "polygon":
        return PrimitiveGeometry(PolygonPrimitive(tuple(tuple(point) for point in params["points"])))
    raise TypeError(f"automatic primitive intent requires a typed primitive, got {shape.kind!r}")


def _polygon_exterior_points(flat: object) -> tuple[np.ndarray, SkPath] | None:
    if not isinstance(flat, SkPath) or flat.is_empty:
        return None
    geoms = flat.geoms
    if len(geoms) != 1 or geoms[0].interiors:
        return None
    coords = np.asarray(geoms[0].exterior.coords, dtype=float)
    return (coords, flat) if len(coords) >= 4 else None


def recognize_primitive_geometry(footprint: SkPath, *, epsilon: float) -> PrimitiveGeometry | None:
    """Recognize one hole-free component, independently of paint or working IDs."""
    polygonal = _polygon_exterior_points(footprint)
    if polygonal is None:
        return None
    points, flat = polygonal
    primitive = recognize_primitive(points, epsilon=epsilon)
    if primitive is not None:
        return primitive_geometry_from_shape(primitive)
    from .drawing_simplification import _rounded_rect_radius
    fit = _rounded_rect_radius(points, flat, epsilon)
    if fit is None:
        return None
    radius, (x0, y0, x1, y1) = fit
    return PrimitiveGeometry(RoundedRectanglePrimitive(
        (x0, y0), x1 - x0, y1 - y0, radius, radius))


def fit_primitive_geometry(
    footprint: SkPath,
    kind: PrimitiveKind,
    *,
    epsilon: float,
    target_id: str,
) -> PrimitiveGeometry:
    """Estimate the requested primitive from the current retained vector boundary."""
    if kind not in {"rect", "rounded_rect", "circle", "ellipse", "polygon"}:
        raise ValueError(f"unsupported primitive {kind!r}")
    polygonal = _polygon_exterior_points(footprint)
    if polygonal is None:
        raise ValueError("primitive fitting requires one non-empty, hole-free vector component")
    points, flat = polygonal
    if kind in {"rect", "circle", "ellipse"}:
        fitted = recognize_primitive(points, epsilon=epsilon, kind=kind, require_match=False)
        if fitted is None:
            raise ValueError(f"cannot estimate {kind!r} for target {target_id!r}")
        return primitive_geometry_from_shape(fitted)
    if kind == "rounded_rect":
        from .drawing_simplification import _rounded_rect_radius
        fit = _rounded_rect_radius(points, flat, epsilon)
        if fit is None:
            # A rectangle is the zero-radius member of the rounded-rect family.
            rectangle = recognize_primitive(points, epsilon=epsilon, kind="rect")
            if rectangle is None:
                raise ValueError(f"cannot estimate 'rounded_rect' for target {target_id!r}")
            radius, bounds = 0.0, flat.bounds
        else:
            radius, bounds = fit
        x0, y0, x1, y1 = bounds
        return PrimitiveGeometry(RoundedRectanglePrimitive(
            (x0, y0), x1 - x0, y1 - y0, radius, radius))
    fitted = recognize_polygon(points, epsilon=epsilon)
    if fitted is None:
        raise ValueError(f"target {target_id!r} is not a polygon within epsilon={epsilon}")
    return PrimitiveGeometry(PolygonPrimitive(tuple(fitted.params["points"])))

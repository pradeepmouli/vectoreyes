"""Primitive fitting kernels, independent of the temporary region scene."""

from __future__ import annotations

import numpy as np

from .contour import outer_contour, region_corner_radius
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
    raster: np.ndarray,
    kind: PrimitiveKind,
    *,
    epsilon: float,
    target_id: str,
) -> PrimitiveGeometry:
    """Fit a public primitive to retained raster evidence, without mutating it.

    The raster deliberately remains authoritative after earlier geometry edits;
    fitting a rasterized current outline would compound those edits instead.
    """
    if kind not in {"rect", "rounded_rect", "circle", "ellipse", "polygon"}:
        raise ValueError(f"unsupported primitive {kind!r}")
    mask = np.asarray(raster)
    if mask.ndim != 2:
        raise ValueError("primitive fitting requires a two-dimensional raster")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError(f"cannot fit {kind!r} to an empty drawing region")
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    if kind == "rect":
        return PrimitiveGeometry(RectanglePrimitive((x, y), w, h))
    if kind == "rounded_rect":
        radius = region_corner_radius(mask)
        return PrimitiveGeometry(RoundedRectanglePrimitive((x, y), w, h, radius, radius))
    center = (x + w / 2, y + h / 2)
    if kind == "circle":
        return PrimitiveGeometry(CirclePrimitive(center, min(w, h) / 2))
    if kind == "ellipse":
        return PrimitiveGeometry(EllipsePrimitive(center, w / 2, h / 2))
    contour = outer_contour(mask)
    if len(contour) < 3:
        raise ValueError(f"cannot fit {kind!r} without an outer contour")
    fitted = recognize_polygon(contour, epsilon=epsilon)
    if fitted is None:
        raise ValueError(f"target {target_id!r} is not a polygon within epsilon={epsilon}")
    return PrimitiveGeometry(PolygonPrimitive(tuple(fitted.params["points"])))

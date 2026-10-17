"""Primitive fitting kernels, independent of the temporary region scene."""

from __future__ import annotations

import numpy as np

from .contour import outer_contour, region_corner_radius
from .drawing_actions import PrimitiveKind
from .fit import recognize_polygon
from .path_geometry import (
    CirclePrimitive,
    EllipsePrimitive,
    PolygonPrimitive,
    PrimitiveGeometry,
    RectanglePrimitive,
    RoundedRectanglePrimitive,
)


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

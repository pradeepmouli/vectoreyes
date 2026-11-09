from __future__ import annotations

import numpy as np
from skimage.draw import polygon as draw_polygon

from ..skia_geometry import SkPath, _ring_area, _ring_centroid, _point_in_ring

BUDGET = 0.02


def _fill_ring(coords: list[tuple[float, float]], shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=bool)
    if len(coords) < 3:
        return mask
    arr = np.array(coords, dtype=float)
    rr, cc = draw_polygon(arr[:, 1], arr[:, 0], shape=shape_hw)
    mask[rr, cc] = True
    return mask


def rasterize(geom: SkPath, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=bool)
    if geom.is_empty:
        return mask
    subpaths = geom.linearized_subpaths
    if not subpaths:
        return mask

    sorted_sp = sorted(subpaths, key=_ring_area, reverse=True)
    # Fill exterior subpaths; subtract holes
    exteriors: list[int] = []
    for i, sp in enumerate(sorted_sp):
        c = _ring_centroid(sp)
        is_hole = any(_point_in_ring(c, sorted_sp[j]) for j in exteriors)
        if is_hole:
            hole_mask = _fill_ring(sp, shape_hw)
            mask &= ~hole_mask
        else:
            exteriors.append(i)
            mask |= _fill_ring(sp, shape_hw)
    return mask


def coverage_residual(geom: SkPath, true_mask: np.ndarray) -> float:
    truth = np.asarray(true_mask, dtype=bool)
    pred = rasterize(geom, truth.shape)
    diff = np.logical_xor(pred, truth)
    denom = int(truth.sum())
    if denom == 0:
        return 0.0 if not diff.any() else 1.0
    return float(diff.sum() / denom)


def gate_ok(geom: SkPath, true_mask: np.ndarray, *, budget: float = BUDGET) -> bool:
    return coverage_residual(geom, true_mask) <= budget

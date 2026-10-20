import numpy as np

from vectormark.skia_geometry import SkPath
from vectormark.refinement.gate import BUDGET, coverage_residual, gate_ok, rasterize


def _disk_mask(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _square_mask(shape, top, left, bottom, right):
    mask = np.zeros(shape, dtype=bool)
    mask[top : bottom + 1, left : right + 1] = True
    return mask


def test_gate_accepts_matching_circle():
    h = w = 120
    true = _disk_mask(h, w, 60, 60, 40)
    circ = SkPath.make_circle(60, 60, 40)

    assert BUDGET == 0.02
    assert coverage_residual(circ, true) < 0.05
    assert gate_ok(circ, true)


def test_gate_rejects_wrong_shape():
    h = w = 120
    true = _disk_mask(h, w, 60, 60, 40)
    square = SkPath(shell=[(20, 20), (100, 20), (100, 100), (20, 100)])
    residual = coverage_residual(square, true)

    assert residual > BUDGET
    assert not gate_ok(square, true)
    assert gate_ok(square, true, budget=residual + 1e-9)


def test_rasterize_subtracts_hole_and_gate_accepts_matching_ring():
    shape = (12, 12)
    ring = SkPath(
        shell=[(1, 1), (10, 1), (10, 10), (1, 10)],
        holes=[[(4, 4), (7, 4), (7, 7), (4, 7)]],
    )
    true = _square_mask(shape, 1, 1, 10, 10)
    true[4:8, 4:8] = False

    raster = rasterize(ring, shape)

    assert np.array_equal(raster, true)
    assert not raster[5, 5]
    assert coverage_residual(ring, true) == 0.0
    assert gate_ok(ring, true)


def test_rasterize_unions_multipolygon_parts_and_gate_accepts_match():
    shape = (12, 12)
    p1 = SkPath(shell=[(1, 1), (3, 1), (3, 3), (1, 3)])
    p2 = SkPath(shell=[(7, 7), (9, 7), (9, 9), (7, 9)])
    geom = p1.union(p2)
    true = _square_mask(shape, 1, 1, 3, 3) | _square_mask(shape, 7, 7, 9, 9)

    raster = rasterize(geom, shape)

    assert np.array_equal(raster, true)
    assert raster[2, 2]
    assert raster[8, 8]
    assert coverage_residual(geom, true) == 0.0
    assert gate_ok(geom, true)

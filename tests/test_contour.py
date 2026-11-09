import numpy as np
from vectoreyes.contour import outer_contour, rdp, corner_indices, region_contours, _polygon_area, region_corner_radius


def test_region_contours_finds_hole_outer_first():
    mask = np.zeros((40, 40), bool)
    mask[8:32, 8:32] = True       # square
    mask[16:24, 16:24] = False    # hole / counter
    cs = region_contours(mask)
    assert len(cs) == 2                                  # outer + hole
    assert _polygon_area(cs[0]) > _polygon_area(cs[1])   # outer first


def test_outer_contour_of_square_is_closed_xy():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    c = outer_contour(mask)
    assert c.ndim == 2 and c.shape[1] == 2          # (N, 2) as (x, y)
    assert np.allclose(c[0], c[-1])                  # closed loop
    assert c[:, 0].min() < 6 and c[:, 0].max() > 13  # spans the square in x


def test_rdp_reduces_straight_run_to_endpoints():
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, 3]], float)
    simp = rdp(pts, epsilon=0.5)
    assert len(simp) == 3                            # (0,0),(3,0),(3,3)


def test_corner_indices_finds_square_corners():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    c = rdp(outer_contour(mask), epsilon=1.0)
    corners = corner_indices(c, angle_threshold_deg=45)
    assert len(corners) >= 4


def test_edge_touching_rect_contour_is_closed_and_full():
    mask = np.zeros((20, 20), bool); mask[0:10, 0:8] = True   # touches top + left edges
    c = outer_contour(mask)
    assert np.allclose(c[0], c[-1])                 # closed
    assert c[:, 0].min() <= 0.5 and c[:, 1].min() <= 0.5   # reaches the touched edges


def _sharp_square(side=60, pad=10):
    m = np.zeros((side + 2 * pad, side + 2 * pad), bool)
    m[pad:pad + side, pad:pad + side] = True
    return m


def _rounded_square(side=60, pad=10, r=10):
    # standard rounded rect: a point is inside iff its distance to the inner box
    # [x0+r, x1-r] x [y0+r, y1-r] (the clamped point) is <= r. Flat edges + quarter-circle corners.
    n = side + 2 * pad
    yy, xx = np.ogrid[:n, :n]
    x0, x1, y0, y1 = pad, pad + side - 1, pad, pad + side - 1
    cx = np.clip(xx, x0 + r, x1 - r)
    cy = np.clip(yy, y0 + r, y1 - r)
    return np.hypot(xx - cx, yy - cy) <= r


def test_corner_radius_sharp_square_is_zero():
    assert region_corner_radius(_sharp_square()) == 0.0


def test_corner_radius_rounded_square_recovers_radius():
    r = region_corner_radius(_rounded_square(r=12))
    assert 8.0 <= r <= 18.0          # ~12 plus the de-antialias pad, generous band


def test_corner_radius_monotonic_in_rounding():
    small = region_corner_radius(_rounded_square(r=6))
    large = region_corner_radius(_rounded_square(r=16))
    assert small > 0.0 and large > small


def test_corner_radius_tiny_or_empty_mask_is_zero():
    assert region_corner_radius(np.zeros((4, 4), bool)) == 0.0
    tiny = np.zeros((20, 20), bool); tiny[9:11, 9:11] = True
    assert region_corner_radius(tiny) == 0.0

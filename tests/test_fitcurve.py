import numpy as np
from vectoreyes._fitcurve import cubic_inflects, fit_cubic_beziers, fit_quadratic_beziers, qbezier


def test_fits_quarter_circle_within_tolerance():
    t = np.linspace(0, np.pi / 2, 40)
    pts = np.column_stack([np.cos(t), np.sin(t)]) * 50
    beziers = fit_quadratic_beziers(pts, max_error=0.5)
    assert len(beziers) >= 1
    # each quadratic is 3 control points of dim 2
    assert all(b.shape == (3, 2) for b in beziers)
    # endpoints match the data endpoints
    assert np.allclose(beziers[0][0], pts[0], atol=1e-6)
    assert np.allclose(beziers[-1][2], pts[-1], atol=1e-6)


def test_quadratic_fit_is_inflection_free():
    """Even on noisy data a quadratic segment cannot wobble: its signed curvature
    has one sign for all t (constant second derivative), so no S-curve appears."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, np.pi, 80)
    pts = np.column_stack([np.cos(t), np.sin(t)]) * 40 + rng.normal(0, 0.4, (80, 2))
    for b in fit_quadratic_beziers(pts, max_error=1.0):
        # second derivative of a quadratic is the constant 2*(P2 - 2P1 + P0);
        # the cross product of B'(t) and B'' keeps one sign across the segment.
        d2 = 2 * (b[2] - 2 * b[1] + b[0])
        ts = np.linspace(0, 1, 25)
        d1 = np.array([2 * (1 - tt) * (b[1] - b[0]) + 2 * tt * (b[2] - b[1]) for tt in ts])
        cross = d1[:, 0] * d2[1] - d1[:, 1] * d2[0]
        nz = cross[np.abs(cross) > 1e-9]
        assert nz.size == 0 or np.all(nz > 0) or np.all(nz < 0)  # no sign flip -> no inflection


def test_cubic_fit_splits_internal_inflections():
    pts = np.array([
        [0.0, 0.0],
        [8.0, 16.0],
        [16.0, -16.0],
        [24.0, 16.0],
        [32.0, -16.0],
        [40.0, 0.0],
    ])

    beziers = fit_cubic_beziers(pts, max_error=100.0)

    assert len(beziers) > 1
    assert all(cubic_inflects(ctrl) == [] for ctrl in beziers)


def test_cubic_inflects_detects_near_collinear_s_shape():
    ctrl = np.array(
        [
            [10.0, 10.0],
            [19.11, 10.4],
            [28.23, 10.8],
            [37.34, 11.2],
        ]
    )

    assert cubic_inflects(ctrl) != []

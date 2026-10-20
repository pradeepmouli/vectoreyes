import numpy as np
from vectormark.fill_fit import fit_fill
from vectormark.candidate import FlatFill, LinearGradientFill, RadialGradientFill


def _solid(h=40, w=40, color=(30, 120, 200)):
    rgb = np.zeros((h, w, 3), np.uint8); rgb[:] = color
    mask = np.ones((h, w), bool)
    return mask, rgb


def _hramp(h=40, w=80, c0=(20, 40, 200), c1=(220, 40, 20)):
    rgb = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        t = x / (w - 1)
        rgb[:, x] = [round(c0[i] + t * (c1[i] - c0[i])) for i in range(3)]
    return np.ones((h, w), bool), rgb


def test_uniform_region_is_flat():
    mask, rgb = _solid()
    fill = fit_fill(mask, rgb, flat_hex="#1E78C8").fill
    assert isinstance(fill, FlatFill) and fill.hex == "#1E78C8"


def test_flat_region_ignores_a_sparse_background_antialias_fringe():
    mask, rgb = _solid(h=50, w=50)
    rgb[:2, :] = (255, 255, 255)

    fill = fit_fill(mask, rgb, flat_hex="#1E78C8").fill

    assert isinstance(fill, FlatFill) and fill.hex == "#1E78C8"


def test_linear_ramp_is_gradient():
    # _best_parametric may choose linear or radial (whichever has lower mean ΔE);
    # for the searched parametric approach either kind represents the ramp correctly.
    mask, rgb = _hramp()
    fill = fit_fill(mask, rgb, flat_hex="#000000").fill
    assert isinstance(fill, (LinearGradientFill, RadialGradientFill))
    assert len(fill.stops) >= 2
    if isinstance(fill, LinearGradientFill):
        assert set(fill.geometry.keys()) >= {"x1", "y1", "x2", "y2"}
    else:
        assert set(fill.geometry.keys()) >= {"cx", "cy", "r"}


def test_flat_hex_used_only_for_flat_decision():
    # a ramp must NOT collapse to flat_hex even though one is provided
    mask, rgb = _hramp()
    fill = fit_fill(mask, rgb, flat_hex="#1E78C8").fill
    assert not isinstance(fill, FlatFill)

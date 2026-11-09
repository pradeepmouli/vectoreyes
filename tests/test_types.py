import numpy as np
from vectoreyes.types import Region, Axis


def test_region_holds_mask_and_color():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    r = Region(label=1, mask=mask, color_hex="#062336")
    assert r.area == 4
    assert r.color_hex == "#062336"


def test_axis_reflect_x():
    ax = Axis(x=10.0)
    assert ax.reflect_x(7.0) == 13.0

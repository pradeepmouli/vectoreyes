import numpy as np

from vectoreyes.candidate import FlatFill
from vectoreyes.fit import Shape, _smooth_quadratic_path_d
from vectoreyes.refinement.gate import rasterize
from vectoreyes.refinement.passes.smooth import smooth_pass
from vectoreyes.refinement.region_sample import RegionSample


def test_smooth_pass_adjusts_quadratic_tangents_within_the_error_budget():
    shape = Shape("path", {"d": "M0 0 Q2 8 10 10 Q18 8 20 0 L0 0 Z"})
    region = RegionSample(1, shape, FlatFill("#123456"), 0)

    proposals = smooth_pass(
        [region],
        {region.id: rasterize(region.footprint, (32, 32))},
        # Exact tangent projection moves this control by just over two pixels,
        # whose quadratic boundary deviation is just over one pixel.  Do not
        # accept a partial move merely to satisfy a tighter budget: partial
        # projections are not idempotent.
        max_error=1.1,
    )

    assert len(proposals) == 1
    smoothed = proposals[0].new_objects[0].current
    assert smoothed is not None
    assert smoothed.params["d"] != shape.params["d"]


def test_smooth_pass_skips_gradient_fills():
    from vectoreyes.candidate import LinearGradientFill

    shape = Shape("path", {"d": "M0 0 Q2 8 10 10 Q18 8 20 0 L0 0 Z"})
    region = RegionSample(
        1,
        shape,
        LinearGradientFill({"x1": 0, "y1": 0, "x2": 20, "y2": 0}, [(0, "#000000"), (1, "#ffffff")]),
        0,
    )

    assert smooth_pass([region], {region.id: rasterize(region.footprint, (32, 32))}, max_error=1.0) == []


def test_quadratic_to_line_corner_sharper_than_135_degrees_is_not_smoothed():
    path = "M0 0 Q2 8 10 10 L10 20 Z"

    assert _smooth_quadratic_path_d(path) == path


def test_quadratic_to_line_near_straight_join_can_be_smoothed():
    path = "M0 0 Q2 4 10 0 L20 0 Z"

    smoothed = _smooth_quadratic_path_d(path)

    # The line has no control point.  Keep its endpoint fixed and put the
    # preceding quadratic control exactly on the line's outgoing tangent.
    assert smoothed == "M0 0 Q1.06 0 10 0 L20 0 Z"
    assert _smooth_quadratic_path_d(smoothed) == smoothed


def test_smooth_pass_is_idempotent():
    shape = Shape("path", {"d": "M0 0 Q2 8 10 10 Q18 8 20 0 L0 0 Z"})
    region = RegionSample(1, shape, FlatFill("#123456"), 0)
    masks = {region.id: rasterize(region.footprint, (32, 32))}

    first = smooth_pass([region], masks, max_error=1.1)[0].new_objects[0]
    assert smooth_pass([first], {first.id: first.raster}, max_error=1.1) == []

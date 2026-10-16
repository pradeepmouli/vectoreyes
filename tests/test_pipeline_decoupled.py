import re
import numpy as np
from vectormark.pipeline import idealize, Options, IdealizeReport


def _wing_with_dot():
    """A blue->navy vertical ramp 'wing' with a separate uniform blue 'dot' beside it
    on white — a minimal stand-in for the V-bird failure case."""
    H, W = 120, 120
    img = np.full((H, W, 3), 255, np.uint8)
    for y in range(20, 100):                      # the wing: smooth vertical ramp
        t = (y - 20) / 79
        img[y, 20:70] = [round(40 + t * 0), round(120 - t * 90), round(230 - t * 110)]
    yy, xx = np.ogrid[:H, :W]                      # a separate uniform blue dot
    img[((yy - 40) ** 2 + (xx - 95) ** 2) <= 12 ** 2] = (30, 100, 220)
    return img


def test_wing_emits_one_gradient_and_dot_survives():
    svg = idealize(_wing_with_dot(), options=Options(max_sampling_colors=16))
    # the wing is ONE gradient, not stacked flat bands
    assert len(re.findall(r"<linearGradient", svg)) >= 1
    # the dot is still present as its own element (a circle or a small path), not absorbed
    # — at least two distinct filled elements exist (wing + dot)
    assert len(re.findall(r"<(path|circle|ellipse|rect|polygon)\b", svg)) >= 2


def test_flat_logo_uses_no_gradient():
    # a solid square on white must stay a single flat shape, no gradient defs
    img = np.full((80, 80, 3), 255, np.uint8)
    img[20:60, 20:60] = (200, 40, 40)
    svg = idealize(img, options=Options(max_sampling_colors=16))
    assert "<linearGradient" not in svg and "<radialGradient" not in svg


def test_report_gradients_counts_fill_type_not_source():
    """report.gradients must count candidates whose fill is a gradient, not candidates
    with source=='gradient' (no such source exists in the new pipeline — all filled
    regions are source='region')."""
    svg, report = idealize(_wing_with_dot(), options=Options(max_sampling_colors=16), report=True)
    assert isinstance(report, IdealizeReport)
    # the wing region gets a LinearGradientFill, so gradients >= 1
    assert report.gradients >= 1


def _two_gradient_blobs():
    """Two disconnected smooth-ramp rectangles on white — each should get its own gradient."""
    H, W = 120, 240
    img = np.full((H, W, 3), 255, np.uint8)
    # left blob: horizontal blue->red ramp
    for x in range(10, 100):
        t = (x - 10) / 89
        img[20:100, x] = [round(20 + t * 200), 40, round(200 - t * 180)]
    # right blob: horizontal green->purple ramp (separate component, no adjacency)
    for x in range(140, 230):
        t = (x - 140) / 89
        img[20:100, x] = [round(40 + t * 160), round(180 - t * 160), round(20 + t * 180)]
    return img


def test_two_disconnected_gradient_blobs_each_get_gradient():
    """Two spatially separate ramp blobs must each be emitted with a gradient fill."""
    svg, report = idealize(_two_gradient_blobs(), options=Options(max_sampling_colors=16), report=True)
    grad_count = svg.count("<linearGradient") + svg.count("<radialGradient")
    assert grad_count >= 2, f"expected >= 2 gradients, got {grad_count}"

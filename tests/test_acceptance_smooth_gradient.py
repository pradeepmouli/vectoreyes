"""Smooth-gradient reconstruction end-to-end through idealize. A smooth (non-posterized)
gradient mark becomes one shape + one <linearGradient>/<radialGradient>; flats and multi-blob
marks stay flat. Each positive test asserts that the end-to-end SVG contains exactly one
<linearGradient>/<radialGradient> and that the render ΔE is within tolerance, via the
colour-step merge + per-component fill."""

import numpy as np

from vectoreyes import Options, idealize
from vectoreyes.color import mean_delta_e
from tests._render import render_svg


def _smooth_linear_rect(h, w, x0, x1, c0, c1, bg=(255, 255, 255)):
    yy, xx = np.mgrid[:h, :w]
    t = ((xx - x0) / (x1 - x0)).clip(0, 1)
    img = np.full((h, w, 3), bg, np.uint8).astype(float)
    rect = np.zeros((h, w), bool); rect[int(h * 0.2):int(h * 0.8), x0:x1] = True
    for ch in range(3):
        img[:, :, ch][rect] = (c0[ch] + t * (c1[ch] - c0[ch]))[rect]
    return img.round().astype(np.uint8)


def _smooth_radial_disc(h, w, c, r, c0, c1, bg=(255, 255, 255)):
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - c[0], yy - c[1])
    t = (dist / r).clip(0, 1)
    img = np.full((h, w, 3), bg, np.uint8).astype(float)
    disc = dist <= r
    for ch in range(3):
        img[:, :, ch][disc] = (c0[ch] + t * (c1[ch] - c0[ch]))[disc]
    return img.round().astype(np.uint8)


def test_smooth_linear_rect_via_smooth_path():
    h, w = 160, 240
    # Low-contrast so the gradient merges into one component and fits a single linear gradient.
    img = _smooth_linear_rect(h, w, 40, 200, (85, 145, 225), (70, 125, 210))
    svg = idealize(img, options=Options())
    assert svg.count("<linearGradient") == 1            # ...so the smooth path produced it
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.06


def test_smooth_radial_disc_via_smooth_path():
    h, w = 200, 200
    # Low-contrast so the gradient merges into one component and fits a single radial gradient.
    img = _smooth_radial_disc(h, w, (100, 100), 85, (115, 185, 240), (85, 140, 215))
    svg = idealize(img, options=Options())
    assert svg.count("<radialGradient") == 1
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.07



def _rotate_img(img, deg):
    from PIL import Image
    return np.asarray(Image.fromarray(img).rotate(
        deg, resample=Image.BILINEAR, expand=True, fillcolor=(255, 255, 255)), np.uint8)


def test_bake_gradient_geometry_linear_and_radial():
    from vectoreyes.pipeline import _bake_gradient_geometry
    # 90° rotation about origin as an SVG affine (a,b,c,d,e,f): (x,y) -> (-y, x)
    bake = (0.0, 1.0, -1.0, 0.0, 0.0, 0.0)
    lin = _bake_gradient_geometry({"x1": 10.0, "y1": 0.0, "x2": 20.0, "y2": 0.0}, "linear", bake)
    assert abs(lin["x1"] - 0.0) < 1e-9 and abs(lin["y1"] - 10.0) < 1e-9
    assert abs(lin["x2"] - 0.0) < 1e-9 and abs(lin["y2"] - 20.0) < 1e-9
    rad = _bake_gradient_geometry({"cx": 10.0, "cy": 0.0, "r": 7.0}, "radial", bake)
    assert abs(rad["cx"] - 0.0) < 1e-9 and abs(rad["cy"] - 10.0) < 1e-9
    assert rad["r"] == 7.0                                # rigid affine preserves radius


def test_rectified_path_emits_gradient_nonflatten():
    base = _smooth_linear_rect(160, 240, 40, 200, (85, 145, 225), (70, 125, 210))
    img = _rotate_img(base, 30)                           # tilted rect -> rectified path
    h, w = img.shape[:2]
    svg = idealize(img, options=Options())
    assert "<g transform=" in svg                         # rectified path was taken
    assert svg.count("<linearGradient") == 1              # ...and a gradient was emitted
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.08


def test_rectified_path_emits_gradient_flatten():
    base = _smooth_linear_rect(160, 240, 40, 200, (85, 145, 225), (70, 125, 210))
    img = _rotate_img(base, 30)
    h, w = img.shape[:2]
    svg = idealize(img, options=Options(flatten=True))
    assert "<g transform=" not in svg                     # flatten bakes geometry: no wrapping <g>
    assert svg.count("<linearGradient") == 1              # gradient still emitted (baked geometry)
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.08

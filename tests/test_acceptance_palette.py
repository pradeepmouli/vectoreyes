import re

import numpy as np

from vectoreyes import Options, idealize


def _settir_synth():
    """Synthetic settir-spirit logo (brand-free): three solid navy blocks (a
    wordmark stand-in) plus a thin antialiased blue diagonal mark on white. The
    blue is AA-dispersed like a real thin mark, so it only survives palette
    extraction via perceptual clustering."""
    h = w = 200
    img = np.full((h, w, 3), 255, np.uint8)
    navy = (10, 30, 70)
    img[150:170, 20:60] = navy
    img[150:170, 80:120] = navy
    img[150:170, 140:180] = navy
    blue = np.array([1.0, 131.0, 253.0])
    white = np.array([255.0, 255.0, 255.0])
    covs = np.linspace(0.86, 0.99, 100)
    for i in range(100):
        y = 20 + i
        for dx, k in ((-1, 0.90), (0, 1.0), (1, 0.90), (2, 0.90)):
            x = 40 + i + dx
            if 0 <= x < w and 0 <= y < h:
                cov = covs[i] * k
                img[y, x] = np.round(blue * cov + white * (1 - cov)).astype(np.uint8)
    return img


def _fills(svg):
    def rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    # include both flat fills and gradient stop colors so gradient-filled elements are visible
    hexes = re.findall(r'fill="(#[0-9A-Fa-f]{6})"', svg)
    hexes += re.findall(r'stop-color="(#[0-9A-Fa-f]{6})"', svg)
    return [rgb(f) for f in hexes]


def test_settir_style_mark_keeps_blue_through_idealize():
    svg = idealize(_settir_synth(), options=Options(no_symmetry=True))
    fills = _fills(svg)
    assert any(b > 180 and r < 120 and g > 90 for r, g, b in fills), \
        f"blue mark lost; fills={fills}"

import re
from pathlib import Path

import numpy as np
from PIL import Image

from vectoreyes import Options, idealize
from tests._render import render_svg, ssim, mean_delta_e

FIX = Path(__file__).parent / "fixtures" / "daikonic" / "source.png"
ICON_BOTTOM = 392  # the wordmark sits below this row; idealize only the mark


def _icon_array():
    """The Daikonic icon WITHOUT the wordmark. The wordmark is asymmetric, so
    feeding the full image would defeat vertical-symmetry detection (no <use>
    mirror). `idealize` accepts a numpy array directly, so we crop and pass it."""
    src = np.asarray(Image.open(FIX).convert("RGB"), dtype=np.uint8)
    return src[:ICON_BOTTOM]


def test_daikonic_structure_and_symmetry():
    svg = idealize(_icon_array(), options=Options(epsilon=1.8, max_error=1.2))
    # the mark is built from a handful of shapes (bands are rounded trapezoids =
    # paths, not sharp rects), and the leaves are a <use> mirror
    assert svg.count("<path") >= 4
    # bilateral symmetry: the leaves emit a <use> mirror
    assert "<use" in svg
    # unified palette: at most one navy hex in flat fill attributes (AA navies collapsed;
    # gradient stop-colors are internal representation, not counted as palette entries)
    navies = set(re.findall(r'fill="(#0[0-9A-Fa-f]2[0-9A-Fa-f]3[0-9A-Fa-f])"', svg, re.IGNORECASE))
    assert len(navies) <= 1
    # the four brand colours survive, nothing extra
    fills = {f.upper() for f in re.findall(r'fill="(#[0-9A-Fa-f]{6})"', svg)}
    assert len(fills) == 4
    # a teal is present (green & blue high, red low) — exact hex varies by the
    # palette mode, so match the hue family rather than a literal hex
    def _rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    assert any(g > 110 and b > 110 and r < 120 for r, g, b in map(_rgb, fills))


def test_daikonic_output_is_exactly_symmetric():
    # the real target: an exactly bilaterally-symmetric vector (the source raster
    # is only ~0.98 symmetric due to anti-aliasing; the idealized output should
    # be ~1.0). Search a few columns near centre for the best fold axis.
    icon = _icon_array()
    h, w, _ = icon.shape
    out = render_svg(idealize(icon, options=Options(epsilon=1.8, max_error=1.2)), w, h)
    best = max(
        ssim(out[:, x - min(x, w - x):x][:, ::-1], out[:, x:x + min(x, w - x)])
        for x in range(w // 2 - 3, w // 2 + 4)
    )
    assert best >= 0.999, f"output not exactly symmetric: {best:.4f}"


def test_daikonic_renders_close_to_source():
    icon = _icon_array()
    h, w, _ = icon.shape
    svg = idealize(icon, options=Options(epsilon=1.8, max_error=1.2))
    out = render_svg(svg, w, h)
    score = ssim(icon, out)
    de = mean_delta_e(icon, out)
    assert score >= 0.90, f"SSIM too low: {score:.3f}"
    assert de <= 0.05, f"mean ΔE too high: {de:.3f}"


def test_daikonic_corner_radius_is_measured_tight_and_padded():
    """Per-region radius is MEASURED from the band corners + the de-antialiasing pad —
    tight (matches the reference) and far below the old fraction-of-height value."""
    from vectoreyes.contour import region_corner_radius, _CORNER_DEANTIALIAS_PAD
    from vectoreyes.color import extract_palette, quantize
    from vectoreyes.segment import segment

    icon = _icon_array()
    h, w, _ = icon.shape
    pal = extract_palette(icon, max_sampling_colors=16)
    regions = segment(quantize(icon, pal), min_area=max(16, round(0.001 * h * w)))
    radii = [region_corner_radius(r.mask) for r in regions]
    rounded = [r for r in radii if r > 0]
    assert rounded, "no rounded regions found in daikonic icon"
    median_r = sorted(rounded)[len(rounded) // 2]
    assert _CORNER_DEANTIALIAS_PAD <= median_r <= 12.0  # tight: per-region ~10.8 (not the old heuristic ~12.8)

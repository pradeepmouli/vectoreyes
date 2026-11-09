import numpy as np
from vectoreyes.color import srgb_to_oklab, oklab_to_srgb, delta_e
from vectoreyes.color import extract_palette, quantize


def test_oklab_roundtrip():
    rgb = np.array([[0.024, 0.137, 0.212], [0.99, 0.55, 0.15], [1, 1, 1]])
    back = oklab_to_srgb(srgb_to_oklab(rgb))
    assert np.allclose(rgb, back, atol=1e-4)


def test_white_maps_to_L_one():
    lab = srgb_to_oklab(np.array([[1.0, 1.0, 1.0]]))
    assert abs(lab[0, 0] - 1.0) < 1e-3
    assert abs(lab[0, 1]) < 1e-3 and abs(lab[0, 2]) < 1e-3


def test_delta_e_is_symmetric_and_zero_on_equal():
    a = np.array([0.5, 0.1, -0.05])
    b = np.array([0.4, 0.0, 0.02])
    assert delta_e(a, a) == 0.0
    assert abs(delta_e(a, b) - delta_e(b, a)) < 1e-12


def _band_image():
    """64x64: navy top, teal bottom, with a realistic 1px anti-alias gradient
    row between — each column is a distinct navy->teal blend, so no single blend
    colour is frequent enough to survive palette extraction (mirrors real AA,
    where edge blends spread across many low-count colours)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:31] = (6, 35, 54)       # navy
    navy = np.array([6.0, 35.0, 54.0])
    teal = np.array([61.0, 168.0, 157.0])
    for x in range(64):
        t = x / 63.0
        img[31, x] = np.round(navy * (1 - t) + teal * t).astype(np.uint8)
    img[32:] = (61, 168, 157)    # teal
    return img


def test_extract_palette_finds_two_true_colors_not_the_blend():
    pal = extract_palette(_band_image(), max_sampling_colors=8)
    assert len(pal) == 2
    hexes = {tuple(c) for c in pal}
    assert (6, 35, 54) in hexes and (61, 168, 157) in hexes


def test_quantize_collapses_blend_row():
    img = _band_image()
    pal = extract_palette(img, max_sampling_colors=8)
    q = quantize(img, pal)
    assert set(map(tuple, np.unique(q.reshape(-1, 3), axis=0))) <= {(6, 35, 54), (61, 168, 157)}


def test_extract_palette_never_empty_on_gradient():
    grad = np.zeros((40, 40, 3), np.uint8)
    for x in range(40):
        grad[:, x] = (x * 6, 100, 255 - x * 6)
    pal = extract_palette(grad)
    assert len(pal) >= 1


def _thin_aa_mark():
    """256x256 white canvas with a 3px-wide antialiased blue diagonal mark.
    Coverage varies smoothly per row, so the blue is dispersed across many AA
    shades — no single shade is frequent enough to survive the min_fraction
    floor, mirroring a real thin coloured mark (the settir blue)."""
    h = w = 256
    img = np.full((h, w, 3), 255, np.uint8)
    blue = np.array([1.0, 131.0, 253.0])
    white = np.array([255.0, 255.0, 255.0])
    covs = np.linspace(0.86, 0.99, h)
    for y in range(h):
        for dx, k in ((-1, 0.90), (0, 1.0), (1, 0.90)):
            x = y + dx
            if 0 <= x < w:
                cov = covs[y] * k
                img[y, x] = np.round(blue * cov + white * (1 - cov)).astype(np.uint8)
    return img


_THIN_AA = _thin_aa_mark()


def test_thin_aa_color_no_single_shade_survives_floor():
    """Documents the bug: no single blue shade reaches the frequency floor, so
    only clustering (aggregate weight) can recover the colour."""
    img = _THIN_AA
    flat = img.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    blueish = (colors[:, 2] > 200) & (colors[:, 0] < 120)
    assert blueish.any()
    assert counts[blueish].max() < 0.002 * len(flat)


def test_extract_palette_recovers_dispersed_thin_color():
    img = _THIN_AA
    pal = extract_palette(img)
    pal_lab = srgb_to_oklab(pal / 255.0)
    true_blue = srgb_to_oklab(np.array([1.0, 131.0, 253.0]) / 255.0)
    nearest = min(delta_e(true_blue, c) for c in pal_lab)
    assert nearest <= 0.10, f"blue not recovered: nearest ΔE {nearest:.3f}; palette={pal.tolist()}"


def test_palette_representatives_are_real_colors():
    img = _THIN_AA
    pal = extract_palette(img)
    present = set(map(tuple, np.unique(img.reshape(-1, 3), axis=0)))
    for c in pal:
        assert tuple(int(v) for v in c) in present


def test_palette_is_deterministic():
    img = _THIN_AA
    assert np.array_equal(extract_palette(img), extract_palette(img))


def test_palette_honours_max_sampling_colors():
    img = np.zeros((60, 60, 3), np.uint8)
    cols = [(200, 0, 0), (0, 200, 0), (0, 0, 200),
            (200, 200, 0), (0, 200, 200), (200, 0, 200)]
    for i, c in enumerate(cols):
        img[(i // 3) * 30:(i // 3) * 30 + 30, (i % 3) * 20:(i % 3) * 20 + 20] = c
    pal = extract_palette(img, max_sampling_colors=4)
    assert len(pal) == 4


def test_palette_excludes_below_floor_block():
    img = np.full((100, 100, 3), 255, np.uint8)
    img[:5, :1] = (200, 0, 0)        # 5 px / 10000 = 0.0005 < 0.002 floor
    pal = extract_palette(img)
    assert not any(tuple(int(v) for v in c) == (200, 0, 0) for c in pal)

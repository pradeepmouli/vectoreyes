import numpy as np
from vectoreyes.color import extract_palette, quantize
from vectoreyes.segment import fill_small_compatible_holes, fill_tiny_isolated_holes, segment, hexstr


def _logo_on_white():
    """48x48 white bg, a navy square and a separate teal square."""
    img = np.full((48, 48, 3), 255, dtype=np.uint8)
    img[6:20, 6:42] = (6, 35, 54)     # navy bar
    img[28:42, 6:42] = (61, 168, 157)  # teal bar
    return img


def test_segment_drops_white_background_and_returns_two_regions():
    img = _logo_on_white()
    q = quantize(img, extract_palette(img))
    regions = segment(q)
    assert len(regions) == 2
    assert {r.color_hex for r in regions} == {"#062336", "#3DA89D"}


def test_segment_retains_an_enclosed_component_with_the_canvas_color():
    img = np.full((48, 48, 3), 255, dtype=np.uint8)
    img[4:44, 4:44] = (240, 64, 88)
    img[14:34, 20:28] = (255, 255, 255)  # white glyph on a white canvas

    regions = segment(quantize(img, extract_palette(img)))

    assert {region.color_hex for region in regions} == {"#F04058", "#FFFFFF"}
    white = next(region for region in regions if region.color_hex == "#FFFFFF")
    assert int(white.mask.sum()) == 20 * 8


def test_hexstr_formats_uppercase():
    assert hexstr((6, 35, 54)) == "#062336"


def test_fill_small_compatible_holes_fills_quantization_island_but_preserves_counter():
    rgb = np.full((20, 30, 3), (20, 100, 220), dtype=np.uint8)
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:18, 2:14] = True
    mask[8:10, 7:9] = False  # same-colour palette island omitted from the surface
    mask[2:18, 16:28] = True
    mask[8:10, 21:23] = False
    rgb[8:10, 21:23] = (255, 255, 255)  # intentional contrasting counter

    cleaned, filled = fill_small_compatible_holes(mask, rgb, max_area=8)

    assert filled == 4
    assert cleaned[8:10, 7:9].all()
    assert not cleaned[8:10, 21:23].any()


def test_fill_tiny_isolated_holes_removes_only_pinholes():
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:18, 2:28] = True
    mask[7, 8] = False
    mask[9:11, 14:16] = False
    mask[12:17, 20:25] = False

    cleaned, filled = fill_tiny_isolated_holes(mask, max_area=4)

    assert filled == 5
    assert cleaned[7, 8]
    assert cleaned[9:11, 14:16].all()
    assert not cleaned[12:17, 20:25].any()

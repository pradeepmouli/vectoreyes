import numpy as np

from vectoreyes.candidate import LinearGradientFill, RadialGradientFill
from vectoreyes.refinement.trace import trace_regions
from vectoreyes.pipeline import Options, _segment_image


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _touches(mask_a, mask_b):
    return bool(
        (mask_a[1:, :] & mask_b[:-1, :]).any()
        or (mask_a[:-1, :] & mask_b[1:, :]).any()
        or (mask_a[:, 1:] & mask_b[:, :-1]).any()
        or (mask_a[:, :-1] & mask_b[:, 1:]).any()
    )


def test_trace_single_disk_one_object_path():
    img = np.full((120, 120, 3), 255, np.uint8)
    img[_disk(120, 120, 60, 60, 40)] = (200, 30, 30)

    objs, masks = trace_regions(img, Options())

    assert len(objs) == 1
    o = objs[0]
    assert o.current.kind == "path"
    assert o.id in masks and masks[o.id].sum() > 0
    assert abs(o.footprint.area - np.pi * 40**2) / (np.pi * 40**2) < 0.1


def test_trace_regions_carry_source_raster_and_diagnostics():
    img = np.full((80, 80, 3), 255, np.uint8)
    img[_disk(80, 80, 40, 40, 20)] = (20, 40, 80)

    regions, masks = trace_regions(img, Options())

    assert len(regions) == 1
    region = regions[0]
    assert region.id in masks
    assert np.array_equal(region.raster, masks[region.id])
    assert region.original == region.current
    assert region.source_label is not None
    assert region.color_hex == "#142850"
    assert region.diagnostics["trace"]["area"] == int(region.raster.sum())
    assert region.diagnostics["trace"]["contours"] == 1


def test_trace_small_hole_preserves_evenodd_path_and_area():
    h = w = 160
    img = np.full((h, w, 3), 255, np.uint8)
    outer_radius = 45
    hole_radius = 5
    outer = _disk(h, w, 80, 80, outer_radius)
    hole = _disk(h, w, 80, 80, hole_radius)
    img[outer] = (200, 30, 30)
    img[hole] = (255, 255, 255)

    objs, masks = trace_regions(img, Options())

    assert len(objs) == 1
    obj = objs[0]
    mask_area = int(masks[obj.id].sum())
    outer_area = int(outer.sum())
    expected_hole_area = int(hole.sum())

    assert obj.current.kind == "path"
    assert obj.current.params.get("fill_rule") == "evenodd"
    assert mask_area == outer_area - expected_hole_area
    assert abs(obj.footprint.area - mask_area) < expected_hole_area
    assert outer_area - obj.footprint.area > expected_hole_area * 0.5


def test_trace_gradient_strip_merges_adjacent_regions_to_one_gradient_object():
    h, w = 80, 120
    img = np.full((h, w, 3), 255, np.uint8)
    strip_y0, strip_y1 = 20, 60
    strip_x0, strip_x1 = 20, 100
    xs = np.linspace(0.0, 1.0, 80)
    left = np.array([220.0, 40.0, 40.0])
    right = np.array([40.0, 40.0, 220.0])
    strip = np.round(left[None, :] * (1.0 - xs[:, None]) + right[None, :] * xs[:, None]).astype(np.uint8)
    img[strip_y0:strip_y1, strip_x0:strip_x1] = strip[None, :, :]

    opt = Options(max_sampling_colors=3)
    _, _, regions = _segment_image(img, opt)
    strip_mask = np.zeros((h, w), dtype=bool)
    strip_mask[strip_y0:strip_y1, strip_x0:strip_x1] = True
    strip_regions = [
        region
        for region in regions
        if region.color_hex != "#FFFFFF" and (region.mask & strip_mask).any()
    ]

    assert len(strip_regions) >= 2
    assert any(
        _touches(left.mask, right.mask)
        for i, left in enumerate(strip_regions)
        for right in strip_regions[i + 1:]
    )

    objs, masks = trace_regions(img, opt)

    assert len(objs) == 1
    assert isinstance(objs[0].fill, (LinearGradientFill, RadialGradientFill))
    assert objs[0].id in masks
    assert masks[objs[0].id].sum() == 80 * 40

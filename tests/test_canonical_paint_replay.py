from dataclasses import replace

import numpy as np

from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
from vectoreyes.drawing_fill import FlatFill
from vectoreyes.path_geometry import geometry_from_svg_d


def leaf(identity, d, color='#00ff00', **kwargs):
    return DrawingRegion(identity, geometry_from_svg_d(d, region_id=identity),
        FlatFill(color), **kwargs)


def test_source_paint_uses_visible_surface_not_occluded_underlying_pixels():
    from vectoreyes.drawing_paint import refill_drawing_from_source

    rgb = np.full((24, 24, 3), [255, 0, 0], np.uint8)
    rgb[:, 12:] = [0, 0, 255]
    back = leaf('r1', 'M0 0 L24 0 L24 24 L0 24 Z')
    front = leaf('r2', 'M12 0 L24 0 L24 24 L12 24 Z', z=1)
    drawing = Drawing('paint', Canvas(24, 24), (back, front))
    result = refill_drawing_from_source(drawing, rgb, source_masks={},
        invalidate_all=True, allow_gradients=False)
    assert result.region('r1').fill == FlatFill('#ff0000')
    assert result.region('r2').fill == FlatFill('#0000ff')
    assert result.region('r1').geometry is back.geometry
    assert back.fill == FlatFill('#00ff00')


def test_changed_paint_preserves_unmodified_fill_and_refits_new_coverage():
    from vectoreyes.drawing_paint import refill_drawing_from_source

    rgb = np.full((24, 24, 3), [255, 0, 0], np.uint8)
    rgb[:, 12:] = [0, 0, 255]
    left = leaf('r1', 'M0 0 L10 0 L10 24 L0 24 Z')
    right = leaf('r2', 'M12 0 L24 0 L24 24 L12 24 Z')
    before = Drawing('paint', Canvas(24, 24), (left, right))
    moved = replace(right, geometry=geometry_from_svg_d(
        'M14 0 L24 0 L24 24 L14 24 Z', region_id='r2'))
    current = replace(before, regions=(left, moved))
    result = refill_drawing_from_source(current, rgb, source_masks={},
        before=before, allow_gradients=False)
    assert result.region('r1') is left
    assert result.region('r2').fill == FlatFill('#0000ff')
    assert result.region('r2').geometry is moved.geometry


def test_retained_material_coverage_excludes_different_paint_inside_the_footprint():
    from vectoreyes.drawing_paint import refill_drawing_from_source

    rgb = np.full((24, 24, 3), [0, 0, 255], np.uint8)
    rgb[7:17, 7:17] = [255, 0, 0]
    region = leaf('r1', 'M0 0 L24 0 L24 24 L0 24 Z')
    support = np.zeros((24, 24), bool)
    support[7:17, 7:17] = True
    result = refill_drawing_from_source(Drawing('paint', Canvas(24, 24), (region,)),
        rgb, source_masks={'r1': support}, invalidate_all=True, allow_gradients=False)
    assert result.region('r1').fill == FlatFill('#ff0000')
    assert result.region('r1').diagnostics['fill']['mask'] == 'visible_source_material_core'

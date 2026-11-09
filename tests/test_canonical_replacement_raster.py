import numpy as np
import pytest

from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
from vectoreyes.path_geometry import AffineTransform, CloneGeometry, PrimitiveGeometry, RectanglePrimitive, geometry_from_svg_d
from vectoreyes.replay_state import ReplayState


@pytest.mark.parametrize('kind', ['rectangle', 'cutout', 'clone'])
def test_replacement_raster_uses_retained_geometry_without_sampling_observations(kind, monkeypatch):
    from vectoreyes.drawing_automatic import _replacement_raster
    source = DrawingRegion('source', PrimitiveGeometry(RectanglePrimitive((1, 2), 4, 4)))
    if kind == 'rectangle':
        target = source
    elif kind == 'cutout':
        target = DrawingRegion('target', geometry_from_svg_d(
            'M1 2 L5 2 L5 6 L1 6 Z M2 3 L2 5 L4 5 L4 3 Z', region_id='target'))
    else:
        target = DrawingRegion('target', CloneGeometry('source', (AffineTransform(1, 0, 0, 1, 2, 0),)))
    roots = (source,) if target is source else (source, target)
    state = ReplayState(Drawing('raster', Canvas(9, 8), roots), {}, {})
    def forbidden(*args, **kwargs):
        raise AssertionError('canonical rasterization projected a sampling shape')
    monkeypatch.setattr('vectoreyes.drawing_adapter.current_shape_for_region', forbidden)
    monkeypatch.setattr('vectoreyes.refinement.region_sample.to_polygon', forbidden)
    actual = _replacement_raster(state, target, (8, 9))
    expected = np.zeros((8, 9), dtype=bool)
    # The existing rasterizer includes samples on integer boundary coordinates.
    if kind == 'clone':
        expected[2:7, 3:8] = True
    else:
        expected[2:7, 1:6] = True
    if kind == 'cutout':
        expected[3:6, 2:5] = False
    np.testing.assert_array_equal(actual, expected)

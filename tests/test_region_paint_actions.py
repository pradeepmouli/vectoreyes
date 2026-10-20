import numpy as np
import pytest

from vectormark.drawing import DrawingRegion
from vectormark.drawing_fill import FlatFill
from vectormark.path_geometry import PrimitiveGeometry, RectanglePrimitive


def region():
    return DrawingRegion('r1', PrimitiveGeometry(RectanglePrimitive((0, 0), 10, 10)),
        FlatFill('#000000'), source_region_ids=('source',))


def test_region_paint_fits_source_without_changing_geometry():
    from vectormark.paint_fitting import fit_region_paint
    original = region()
    rgb = np.full((10, 10, 3), (20, 100, 240), dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    updated = fit_region_paint(original, rgb, source_mask=mask, visible_mask=mask, allow_gradients=False)
    assert updated.geometry is original.geometry
    assert updated.fill == FlatFill('#1464f0')
    assert updated.source_region_ids == original.source_region_ids
    assert updated.diagnostics['fill'] == {'status': 'refit', 'mask': 'visible_source_material_core', 'sampled_pixels': 36}
    assert original.fill == FlatFill('#000000')


def test_region_paint_falls_back_to_visible_pixels_when_source_no_longer_overlaps():
    from vectormark.paint_fitting import fit_region_paint
    rgb = np.full((10, 10, 3), (240, 30, 20), dtype=np.uint8)
    updated = fit_region_paint(region(), rgb, source_mask=np.zeros((10, 10), dtype=bool),
        visible_mask=np.ones((10, 10), dtype=bool), allow_gradients=False)
    assert updated.fill == FlatFill('#f01e14')
    assert updated.diagnostics['fill']['mask'] == 'visible_footprint'


def test_region_paint_retains_thin_source_details_without_erosion():
    from vectormark.paint_fitting import fit_region_paint
    source = np.zeros((10, 10), dtype=bool)
    source[4] = True
    updated = fit_region_paint(region(), np.zeros((10, 10, 3), dtype=np.uint8),
        source_mask=source, visible_mask=source, allow_gradients=False)
    assert updated.diagnostics['fill']['mask'] == 'visible_source_material'
    assert updated.diagnostics['fill']['sampled_pixels'] == 10


def test_region_paint_rejects_wrong_visible_canvas():
    from vectormark.paint_fitting import fit_region_paint
    with pytest.raises(ValueError, match='visible mask.*RGB canvas'):
        fit_region_paint(region(), np.zeros((10, 10, 3), dtype=np.uint8),
            source_mask=np.ones((10, 10), dtype=bool), visible_mask=np.ones((8, 8), dtype=bool))


def test_working_refill_routes_only_invalidated_leaf_to_canonical_paint(monkeypatch):
    from vectormark import paint_fitting
    from vectormark.candidate import FlatFill as WorkingFill
    from vectormark.fit import Shape
    from vectormark.optimizer.vector_region import VectorRegion
    from vectormark.optimizer.paint_binding import refill_regions
    source = np.ones((10, 10), dtype=bool)
    first = VectorRegion.from_shape(id=1, drawing_id='r1', shape=Shape('rect', dict(x=0, y=0, w=4, h=4)),
        fill=None, z=0, raster=source)
    second = VectorRegion.from_shape(id=2, drawing_id='r2', shape=Shape('rect', dict(x=5, y=5, w=4, h=4)),
        fill=WorkingFill('#112233'), z=1, raster=source)
    calls = []
    execute = paint_fitting.fit_source_paint
    def record(*args, **kwargs):
        calls.append(kwargs['visible_mask'].copy())
        return execute(*args, **kwargs)
    monkeypatch.setattr(paint_fitting, 'fit_source_paint', record)
    result = refill_regions((first, second), np.full((10, 10, 3), 100, dtype=np.uint8), allow_gradients=False)
    assert len(calls) == 1 and calls[0].shape == source.shape
    assert result[1] is second
    assert result[0].current is first.current


def test_source_paint_result_is_geometry_independent_and_immutable():
    from vectormark.paint_fitting import fit_source_paint
    mask = np.ones((4, 4), dtype=bool)
    result = fit_source_paint(np.full((4, 4, 3), 123, dtype=np.uint8),
        source_mask=mask, visible_mask=mask, allow_gradients=False)
    assert result.fill == FlatFill('#7b7b7b')
    with pytest.raises(TypeError):
        result.diagnostics['status'] = 'changed'

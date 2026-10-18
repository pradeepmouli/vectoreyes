import numpy as np
import pytest

from vectormark.drawing_fill import FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from vectormark.drawing import DrawingRegion
from vectormark.drawing_actions import ActionSettings, FitPrimitiveAction
from vectormark.path_geometry import PrimitiveGeometry, RectanglePrimitive


@pytest.mark.parametrize('split', [False, True])
def test_region_path_action_publishes_canonical_components_and_cutouts(split):
    import vectormark.drawing_region_actions as actions
    from vectormark.drawing_actions import FitPathDrawingAction
    from vectormark.path_geometry import PathGeometryWithCutouts
    region = DrawingRegion('r2-1', PrimitiveGeometry(RectanglePrimitive((0, 0), 2, 2)),
        fill=FlatFill('#123456'), z=7, source_region_ids=('original',))
    raster = np.zeros((20, 20), dtype=bool)
    raster[1:10, 1:10] = True
    raster[3:6, 3:6] = False
    if split:
        raster[14:18, 14:18] = True
    result = actions.fit_region_path(region,
        FitPathDrawingAction(op='fit_path', target=region.id), raster=raster,
        settings=ActionSettings(epsilon=0.1, max_error=0.1, fit_strategy='quadratic'))
    assert result.id == region.id and result.is_branch == split
    assert len(result.leaves()) == (2 if split else 1)
    assert any(isinstance(leaf.geometry, PathGeometryWithCutouts) for leaf in result.leaves())
    for leaf in result.leaves():
        assert leaf.fill is region.fill and leaf.z == region.z
        assert leaf.source_region_ids == region.source_region_ids
        assert leaf.diagnostics['geometry'] == {'explicit': 'path'}
    assert isinstance(region.geometry, PrimitiveGeometry)


@pytest.mark.parametrize('branch', [False, True])
def test_region_path_action_requires_matching_leaf(branch):
    import vectormark.drawing_region_actions as actions
    from vectormark.drawing_actions import FitPathDrawingAction
    leaf = DrawingRegion('r1-1', PrimitiveGeometry(RectanglePrimitive((0, 0), 4, 4)))
    region = DrawingRegion('r1', children=(leaf,)) if branch else leaf
    with pytest.raises(ValueError, match='leaf' if branch else 'target'):
        actions.fit_region_path(region, FitPathDrawingAction(op='fit_path', target='r1'),
            raster=np.ones((8, 8), dtype=bool),
            settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))


def test_region_primitive_action_preserves_identity_paint_and_provenance():
    import vectormark.drawing_region_actions as actions
    original = DrawingRegion('r2-1', PrimitiveGeometry(RectanglePrimitive((1, 1), 2, 2)),
        fill=FlatFill('#123456'), z=7, source_region_ids=('r2', 'r3'),
        diagnostics={'source': {'label': 3}, 'geometry': {'explicit': 'path'}})
    raster = np.zeros((10, 12), dtype=bool)
    raster[2:8, 3:11] = True
    result = actions.fit_region_primitive(original,
        FitPrimitiveAction(op='fit_primitive', target='r2-1', type='rect'),
        raster=raster, settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    assert result.geometry == PrimitiveGeometry(RectanglePrimitive((3, 2), 8, 6))
    assert result.id == original.id
    assert result.fill == original.fill
    assert result.z == original.z
    assert result.source_region_ids == original.source_region_ids
    assert result.diagnostics['source'] == original.diagnostics['source']
    assert result.diagnostics['geometry'] == {'explicit': 'rect'}
    assert original.geometry == PrimitiveGeometry(RectanglePrimitive((1, 1), 2, 2))
    assert original.diagnostics['geometry'] == {'explicit': 'path'}


@pytest.mark.parametrize('branch', [False, True])
def test_region_primitive_action_rejects_wrong_target_or_branch(branch):
    import vectormark.drawing_region_actions as actions
    leaf = DrawingRegion('r1-1', PrimitiveGeometry(RectanglePrimitive((0, 0), 4, 4)))
    region = DrawingRegion('r1', children=(leaf,)) if branch else leaf
    action = FitPrimitiveAction(op='fit_primitive', target='r1', type='circle')
    with pytest.raises(ValueError, match='leaf' if branch else 'target'):
        actions.fit_region_primitive(region, action, raster=np.ones((8, 8), dtype=bool),
            settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))


@pytest.mark.parametrize(('spec', 'fill_type'), [
    ({'type': 'flat', 'color': '#abcdef'}, FlatFill),
    ({'type': 'linear_gradient', 'start': (1, 2), 'end': (7, 8),
      'stops': [{'offset': 0, 'color': '#000000'}, {'offset': 1, 'color': '#FFFFFF'}]}, LinearGradientFill),
    ({'type': 'radial_gradient', 'center': (3, 4), 'radius': 5,
      'stops': [{'offset': 0, 'color': '#000000'}, {'offset': 1, 'color': '#FFFFFF'}]}, RadialGradientFill),
    ({'type': 'raster', 'origin': (1, 2), 'width': 8, 'height': 9,
      'png_base64': 'iVBORw0KGgo='}, RasterFill),
])
def test_region_fill_action_changes_only_paint(spec, fill_type):
    import vectormark.drawing_region_actions as actions
    from vectormark.drawing_actions import SetFillAction
    region = DrawingRegion('r1', PrimitiveGeometry(RectanglePrimitive((0, 0), 8, 8)),
        fill=FlatFill('#123456'), z=3, source_region_ids=('original',), diagnostics={'label': 4})
    result = actions.set_region_fill(region,
        SetFillAction.model_validate({'op': 'set_fill', 'target': 'r1', 'fill': spec}))
    assert isinstance(result.fill, fill_type)
    assert result.geometry is region.geometry
    assert result.source_region_ids == region.source_region_ids
    assert result.diagnostics == region.diagnostics
    assert result.id == region.id and result.z == region.z
    assert region.fill == FlatFill('#123456')


def test_region_fill_action_rejects_wrong_target():
    import vectormark.drawing_region_actions as actions
    from vectormark.drawing_actions import SetFillAction
    region = DrawingRegion('r1', PrimitiveGeometry(RectanglePrimitive((0, 0), 8, 8)))
    with pytest.raises(ValueError, match='target'):
        actions.set_region_fill(region, SetFillAction.model_validate(
            {'op': 'set_fill', 'target': 'r2', 'fill': {'type': 'flat', 'color': '#ABCDEF'}}))

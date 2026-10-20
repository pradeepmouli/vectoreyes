import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.drawing_actions import AssignCloneAction
from vectormark.drawing_relationships import assign_clone
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion


def leaf(identity, x, color):
    return VectorRegion(identity, Shape('rect', {'x': x, 'y': 0, 'w': 4, 'h': 4}),
        FlatFill(color), raster=np.ones((12, 20), dtype=bool), source_regions=(f'material-{x}',))


def test_bound_clone_preserves_scoped_identity_and_independent_observations():
    from vectormark.drawing_adapter import bind_region_inputs
    source, target = leaf(1, 0, '#112233'), leaf(1, 8, '#ABCDEF')
    bound = bind_region_inputs({'source': source, 'target': target},
        paths={'source': (10, 1), 'target': (20, 1)})
    selected = assign_clone(bound.drawing, AssignCloneAction(op='assign_clone',
        source='source', target='target', transforms=({'tx': 8},)))
    result = bound.publish(selected)
    assert result.current.kind == 'use'
    assert result.current.params['href_obj_id'] == 1
    assert result.current.params['href_region_path'] == ('..', 10, 1)
    assert result.footprint.bounds == (8, 0, 12, 4)
    assert result.fill is target.fill and result.original is target.original
    assert result.id == 1 and result.drawing_id is None
    assert result.source_regions == ('material-8',)
    np.testing.assert_array_equal(result.raster, target.raster)


def test_binding_rejects_aliasing_one_scoped_region_as_source_and_target():
    from vectormark.drawing_adapter import bind_region_inputs
    source = leaf(1, 0, '#112233')
    with pytest.raises(ValueError, match='same scoped region'):
        bind_region_inputs({'source': source, 'target': source})


def test_binding_retains_selected_symbolic_dependency_for_cycle_validation():
    from vectormark.drawing_adapter import bind_region_inputs
    source, target = leaf(1, 0, '#112233'), leaf(2, 8, '#ABCDEF')
    source = source.with_current(Shape('use', {'href_obj_id': 2,
        'transform': (1, 0, 0, 1, -8, 0)}), footprint=source.footprint)
    bound = bind_region_inputs({'source': source, 'target': target})
    with pytest.raises(ValueError, match='cyclic'):
        assign_clone(bound.drawing, AssignCloneAction(op='assign_clone',
            source='source', target='target', transforms=({'tx': 8},)))


def test_resolver_binds_named_output_descendant_without_shadowing_input():
    from vectormark.optimizer.automatic_replay import (
        NamedRegionOutput, OperationOutputRegionRef, PassInputRegionRef, ReplayBatch, RegionResolver,
    )
    source, target = leaf(1, 0, '#112233'), leaf(1, 8, '#ABCDEF')
    root = VectorRegion.branch(id=10, children=(source,))
    output = VectorRegion.branch(id=20, children=(target,))
    resolver = RegionResolver([root], ReplayBatch('batch', 'revision', (), ()))
    resolver.register('split', NamedRegionOutput('result', (20,)), output)
    bound = resolver.bind_inputs({
        'source': PassInputRegionRef('batch', 'revision', (10, 1)),
        'target': OperationOutputRegionRef('split', 'result', (1,)),
    })
    assert bound.paths == {'source': (10, 1), 'target': (20, 1)}
    assert bound.drawing.region('source').geometry.primitive.origin == (0, 0)
    assert bound.drawing.region('target').geometry.primitive.origin == (8, 0)
    with pytest.raises(ValueError, match='batch revision'):
        resolver.bind_inputs({'source': PassInputRegionRef('batch', 'wrong', (10, 1))})
    with pytest.raises(ValueError, match='forward-referenced'):
        resolver.bind_inputs({'source': OperationOutputRegionRef('future', 'result')})


def test_bound_baked_copy_uses_selected_transform_without_replacing_target_paint():
    from vectormark.drawing_adapter import bind_region_inputs
    source, target = leaf(1, 0, '#112233'), leaf(2, 8, '#ABCDEF')
    bound = bind_region_inputs({'source': source, 'target': target})
    selected = assign_clone(bound.drawing, AssignCloneAction(op='assign_clone',
        source='source', target='target', transforms=({'a': -1, 'tx': 12},)))
    result = bound.publish(selected, symbolic=False)
    assert result.current.kind == 'path'
    assert result.footprint.bounds == (8, 0, 12, 4)
    assert result.fill is target.fill
    assert source.current.kind == 'rect' and target.current.kind == 'rect'


@pytest.mark.parametrize('refresh', [False, True])
def test_bound_clone_refresh_preserves_metadata_and_exact_scoped_reference(refresh):
    from vectormark.drawing_adapter import bind_region_inputs
    from vectormark.optimizer.automatic_replay import apply_clone_reference_update
    source, target = leaf(1, 2, '#112233'), leaf(2, 8, '#ABCDEF')
    clone = target.with_current(Shape('use', {'href_obj_id': 5,
        'href_region_path': (5,), 'transform': (1, 0, 0, 1, 8, 0), 'note': 'retained'}),
        footprint=target.footprint)
    bound = bind_region_inputs({'source': source, 'target': clone},
        paths={'source': (10, 1), 'target': (20, 2)})
    result = apply_clone_reference_update(bound, previous_source_id=5,
        previous_source_path=(5,), refresh_footprint=refresh)
    assert result.current.params['href_obj_id'] == 1
    assert result.current.params['href_region_path'] == ('..', 10, 1)
    assert result.current.params['note'] == 'retained'
    assert result.footprint.bounds == ((10, 0, 14, 4) if refresh else (8, 0, 12, 4))
    assert result.fill is clone.fill and result.drawing_id is None
    assert result.diagnostics['clone_reference_remap']['source_region_path'] == (10, 1)
    assert clone.current.params['href_obj_id'] == 5

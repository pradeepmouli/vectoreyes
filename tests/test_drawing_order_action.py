from dataclasses import replace

import pytest

from vectormark.drawing import Canvas, Drawing, DrawingRegion
from vectormark.drawing_actions import SetZOrderAction
from vectormark.drawing_fill import FlatFill
from vectormark.path_geometry import PrimitiveGeometry, RectanglePrimitive


def scene():
    geometry = PrimitiveGeometry(RectanglePrimitive((0, 0), 4, 4))
    def leaf(name, z):
        return DrawingRegion(name, geometry, fill=FlatFill('#123456'), z=z,
            source_region_ids=(name,), diagnostics={'original': name})
    return Drawing('drw_order', Canvas(20, 20), (
        DrawingRegion('r1', children=(leaf('r1-1', 5), leaf('r1-2', 2)), z=8),
        leaf('r2', 3), leaf('r3', 3),
    ))


def test_canonical_order_expands_group_in_existing_paint_order():
    import vectormark.drawing_region_actions as actions
    original = scene()
    result = actions.set_drawing_z_order(original,
        SetZOrderAction(op='set_z_order', targets=('r3', 'r1', 'r2')))
    assert [result.region(name).z for name in ('r3', 'r1-2', 'r1-1', 'r2')] == [0, 1, 2, 3]
    assert result.drawing_id == original.drawing_id
    assert result.canvas == original.canvas
    assert result.region('r1').z == original.region('r1').z
    assert [child.id for child in result.region('r1').children] == ['r1-1', 'r1-2']
    for root in original.regions:
        for leaf in root.leaves():
            updated = result.region(leaf.id)
            assert updated.geometry is leaf.geometry
            assert updated.fill is leaf.fill
            assert updated.source_region_ids == leaf.source_region_ids
            assert updated.diagnostics == leaf.diagnostics
    assert original.region('r1-1').z == 5


def test_canonical_order_preserves_ties_by_tree_order():
    import vectormark.drawing_region_actions as actions
    original = scene()
    group = original.region('r1')
    original = replace(original, regions=(replace(group,
        children=tuple(replace(child, z=2) for child in group.children)), *original.regions[1:]))
    result = actions.set_drawing_z_order(original,
        SetZOrderAction(op='set_z_order', targets=('r1', 'r2', 'r3')))
    assert result.region('r1-1').z == 0
    assert result.region('r1-2').z == 1
    assert result.region('r2').z < result.region('r3').z


@pytest.mark.parametrize(('targets', 'message'), [
    (('r1', 'r2'), 'every retained leaf'),
    (('r1', 'r1-1', 'r2', 'r3'), 'repeated'),
    (('r1', 'r2', 'r4'), 'unknown region'),
])
def test_canonical_order_rejects_incomplete_overlapping_or_unknown_targets(targets, message):
    import vectormark.drawing_region_actions as actions
    with pytest.raises(ValueError, match=message):
        actions.set_drawing_z_order(scene(), SetZOrderAction(op='set_z_order', targets=targets))

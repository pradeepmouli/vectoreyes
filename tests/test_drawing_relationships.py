from dataclasses import replace

import pytest

from vectormark.drawing import Canvas, Drawing, DrawingRegion
from vectormark.drawing_actions import AlignAction, AssignCloneAction, SetSymmetryAction
from vectormark.drawing_fill import FlatFill
from vectormark.path_geometry import (
    AffineTransform, CloneGeometry, PrimitiveGeometry, RectanglePrimitive,
    geometry_from_svg_d,
)


def scene():
    first = DrawingRegion('r1', PrimitiveGeometry(RectanglePrimitive((0, 0), 4, 4)), FlatFill('#112233'))
    second = DrawingRegion('r2', PrimitiveGeometry(RectanglePrimitive((8, 8), 2, 2)), FlatFill('#abcdef'))
    return Drawing('drw_test', Canvas(20, 20), (first, second))


def test_clone_action_retains_all_transforms_and_independent_paint():
    from vectormark.drawing_relationships import assign_clone
    drawing = scene()
    action = AssignCloneAction(op='assign_clone', source='r1', target='r2', transforms=(
        {'tx': 8}, {'ty': 8},
    ))
    updated = assign_clone(drawing, action)
    assert updated.geometry == CloneGeometry('r1', (AffineTransform(tx=8), AffineTransform(ty=8)))
    assert updated.fill == drawing.region('r2').fill
    assert drawing.region('r2').geometry != updated.geometry


def test_clone_assignment_resolves_existing_clone_to_concrete_source():
    from vectormark.drawing_relationships import assign_clone
    drawing = scene()
    existing = replace(drawing.region('r2'), geometry=CloneGeometry('r1', (AffineTransform(tx=3),)))
    third = replace(drawing.region('r1'), id='r3')
    drawing = replace(drawing, regions=(drawing.region('r1'), existing, third))
    result = assign_clone(drawing, AssignCloneAction(op='assign_clone', source='r2', target='r3',
        transforms=({'tx': 4},)))
    assert result.geometry == CloneGeometry('r1', (AffineTransform(tx=7),))


def test_clone_assignment_rejects_source_cycle():
    from vectormark.drawing_relationships import assign_clone
    drawing = scene()
    drawing = replace(drawing, regions=(replace(drawing.region('r1'),
        geometry=CloneGeometry('r2', (AffineTransform(),))), drawing.region('r2')))
    with pytest.raises(ValueError, match='cycle'):
        assign_clone(drawing, AssignCloneAction(op='assign_clone', source='r1', target='r2',
            transforms=({},)))


def test_alignment_translates_only_requested_axis_and_preserves_command_ids():
    from vectormark.drawing_relationships import align_region
    drawing = scene()
    geometry = geometry_from_svg_d('M0 0 Q2 0 4 0 L4 4 L0 4 Z', region_id='r1')
    drawing = replace(drawing, regions=(replace(drawing.region('r1'), geometry=geometry), drawing.region('r2')))
    result = align_region(drawing, AlignAction(op='align', target='r1', reference='r2', axes=('x',)))
    assert [c.id for c in result.geometry.path.commands] == [c.id for c in geometry.path.commands]
    assert result.geometry.path.commands[0].end == (7, 0)
    assert result.geometry.path.commands[1].control == (9, 0)
    assert result.fill == drawing.region('r1').fill


def test_explicit_symmetry_keeps_cutout_and_returns_one_joined_region():
    from vectormark.drawing_relationships import set_region_symmetry
    from vectormark.geometry_native import region_path
    drawing = scene()
    geometry = geometry_from_svg_d('M0 0 L4 0 L4 4 L0 4 Z M1 1 L3 1 L3 3 L1 3 Z', region_id='r1', fill_rule='evenodd')
    drawing = replace(drawing, regions=(replace(drawing.region('r1'), geometry=geometry), drawing.region('r2')))
    result = set_region_symmetry(drawing, SetSymmetryAction(op='set_symmetry', target='r1',
        axis={'point': (2, 2), 'angle_degrees': 90}))
    assert result.id == 'r1' and result.is_leaf
    assert region_path(result, {'r1': result}).area == pytest.approx(12)
    assert result.diagnostics['symmetry']['mode'] == 'explicit_self'


def test_symmetry_rejects_axis_outside_region_without_mutation():
    from vectormark.drawing_relationships import set_region_symmetry
    drawing = scene()
    with pytest.raises(ValueError, match='two non-empty halves'):
        set_region_symmetry(drawing, SetSymmetryAction(op='set_symmetry', target='r1',
            axis={'point': (20, 0), 'angle_degrees': 90}))
    assert drawing == scene()

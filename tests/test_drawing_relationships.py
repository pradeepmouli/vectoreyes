from dataclasses import replace

import pytest

from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
from vectoreyes.drawing_actions import AlignAction, AssignCloneAction, SetSymmetryAction
from vectoreyes.drawing_fill import FlatFill
from vectoreyes.path_geometry import (
    AffineTransform, CloneGeometry, PrimitiveGeometry, RectanglePrimitive,
    geometry_from_svg_d,
)


def scene():
    first = DrawingRegion('r1', PrimitiveGeometry(RectanglePrimitive((0, 0), 4, 4)), FlatFill('#112233'))
    second = DrawingRegion('r2', PrimitiveGeometry(RectanglePrimitive((8, 8), 2, 2)), FlatFill('#abcdef'))
    return Drawing('drw_test', Canvas(20, 20), (first, second))


def test_explicit_symmetry_uses_the_pass_reconstruction_with_identical_parameters():
    import math
    from vectoreyes.drawing_relationships import set_region_symmetry
    from vectoreyes.drawing_symmetry import Axis2D, set_symmetry
    from vectoreyes.geometry_native import region_path

    drawing = scene()
    action = SetSymmetryAction(op='set_symmetry', target='r1',
        axis={'point': [2, 2], 'angle_degrees': 90},
        residual=0.025, side='negative')
    source = drawing.region('r1')
    expected = set_symmetry(source, region_path(source, {}),
        axis=Axis2D(math.pi / 2, 2, 2), residual=0.025, side='negative')

    actual = set_region_symmetry(drawing, action)

    assert actual == expected


def test_canonical_replacement_expands_clones_using_retained_component_slots():
    from vectoreyes.drawing_relationships import replace_region
    drawing = scene()
    source = drawing.region('r1')
    clone = replace(drawing.region('r2'), geometry=CloneGeometry('r1', (AffineTransform(tx=8),)), z=4)
    drawing = replace(drawing, regions=(source, clone))
    children = (replace(source, id='r1-1'), replace(source, id='r1-3'))
    selected = replace(source, geometry=None, children=children,
        diagnostics={'topology': {'operation': 'retained_topology'}})
    result = replace_region(drawing, selected)
    expanded = result.region('r2')
    assert expanded.is_branch
    assert tuple(child.id for child in expanded.children) == ('r2-1', 'r2-3')
    assert tuple(child.geometry.source_region_id for child in expanded.children) == ('r1-1', 'r1-3')
    assert all(child.fill == clone.fill and child.z == 4 for child in expanded.children)
    assert all(child.geometry.transforms == (AffineTransform(tx=8),) for child in expanded.children)
    assert drawing.region('r2').is_leaf


def test_clone_action_retains_all_transforms_and_independent_paint():
    from vectoreyes.drawing_relationships import assign_clone
    drawing = scene()
    action = AssignCloneAction(op='assign_clone', source='r1', target='r2', transforms=(
        {'tx': 8}, {'ty': 8},
    ))
    updated = assign_clone(drawing, action)
    assert updated.geometry == CloneGeometry('r1', (AffineTransform(tx=8), AffineTransform(ty=8)))
    assert updated.fill == drawing.region('r2').fill
    assert drawing.region('r2').geometry != updated.geometry


def test_clone_assignment_resolves_existing_clone_to_concrete_source():
    from vectoreyes.drawing_relationships import assign_clone
    drawing = scene()
    existing = replace(drawing.region('r2'), geometry=CloneGeometry('r1', (AffineTransform(tx=3),)))
    third = replace(drawing.region('r1'), id='r3')
    drawing = replace(drawing, regions=(drawing.region('r1'), existing, third))
    result = assign_clone(drawing, AssignCloneAction(op='assign_clone', source='r2', target='r3',
        transforms=({'tx': 4},)))
    assert result.geometry == CloneGeometry('r1', (AffineTransform(tx=7),))


def test_clone_assignment_rejects_source_cycle():
    from vectoreyes.drawing_relationships import assign_clone
    drawing = scene()
    drawing = replace(drawing, regions=(replace(drawing.region('r1'),
        geometry=CloneGeometry('r2', (AffineTransform(),))), drawing.region('r2')))
    with pytest.raises(ValueError, match='cycle'):
        assign_clone(drawing, AssignCloneAction(op='assign_clone', source='r1', target='r2',
            transforms=({},)))


def test_alignment_translates_only_requested_axis_and_preserves_command_ids():
    from vectoreyes.drawing_relationships import align_region
    drawing = scene()
    geometry = geometry_from_svg_d('M0 0 Q2 0 4 0 L4 4 L0 4 Z', region_id='r1')
    drawing = replace(drawing, regions=(replace(drawing.region('r1'), geometry=geometry), drawing.region('r2')))
    result = align_region(drawing, AlignAction(op='align', target='r1', reference='r2', axes=('x',)))
    assert [c.id for c in result.geometry.path.commands] == [c.id for c in geometry.path.commands]
    assert result.geometry.path.commands[0].end == (7, 0)
    assert result.geometry.path.commands[1].control == (9, 0)
    assert result.fill == drawing.region('r1').fill


def test_explicit_symmetry_keeps_cutout_and_returns_one_joined_region():
    from vectoreyes.drawing_relationships import set_region_symmetry
    from vectoreyes.geometry_native import region_path
    drawing = scene()
    geometry = geometry_from_svg_d('M0 0 L4 0 L4 4 L0 4 Z M1 1 L3 1 L3 3 L1 3 Z', region_id='r1', fill_rule='evenodd')
    drawing = replace(drawing, regions=(replace(drawing.region('r1'), geometry=geometry), drawing.region('r2')))
    result = set_region_symmetry(drawing, SetSymmetryAction(op='set_symmetry', target='r1',
        axis={'point': (2, 2), 'angle_degrees': 90}))
    assert result.id == 'r1' and result.is_leaf
    assert region_path(result, {'r1': result}).area == pytest.approx(12)
    assert result.diagnostics['symmetry']['mode'] == 'self'


def test_symmetry_rejects_axis_outside_region_without_mutation():
    from vectoreyes.drawing_relationships import set_region_symmetry
    drawing = scene()
    with pytest.raises(ValueError, match='two non-empty halves'):
        set_region_symmetry(drawing, SetSymmetryAction(op='set_symmetry', target='r1',
            axis={'point': (20, 0), 'angle_degrees': 90}))
    assert drawing == scene()

import numpy as np
import pytest
import math

from vectoreyes.candidate import FlatFill
from vectoreyes.drawing_actions import SymmetryAxisInput
from vectoreyes.fit import Shape
from vectoreyes.refinement.region_sample import RegionSample


def choose(source, target, *, matrix, symbolic=True, family_size=None, axis=None):
    from vectoreyes.drawing_adapter import bind_region_inputs
    from vectoreyes.refinement.automatic_replay import (
        AssignCloneOperation, MirrorRegionOperation, apply_relationship_choice, draft_input_ref,
    )
    refs = draft_input_ref(source.id), draft_input_ref(target.id)
    action = (AssignCloneOperation(*refs, matrix, symbolic, family_size) if axis is None else
        MirrorRegionOperation(*refs, (math.radians(axis.angle_degrees), *axis.point), matrix))
    return apply_relationship_choice(action, bind_region_inputs({'source': source, 'target': target}))


def pair():
    source = RegionSample(1, Shape('path', {'d': 'M1 2 L5 2 L5 6 L1 6 Z'}),
        FlatFill('#112233'), 1, raster=np.ones((12, 20), dtype=bool), drawing_id='r1')
    target = RegionSample(2, Shape('rect', {'x': 9, 'y': 2, 'w': 4, 'h': 4}),
        FlatFill('#abcdef'), 4, raster=np.ones((12, 20), dtype=bool), drawing_id='r2')
    return source, target


@pytest.mark.parametrize('symbolic', [False, True])
def test_selected_clone_publication_preserves_target_paint_and_source_evidence(symbolic):
    source, target = pair()
    result = choose(source, target, matrix=(1, 0, 0, 1, 8, 0),
        symbolic=symbolic, family_size=4)[0]
    if symbolic:
        assert result.current == Shape('use', {'href_obj_id': 1, 'transform': (1, 0, 0, 1, 8, 0)})
    else:
        geometry = result.current.params['canonical_path_geometry']
        assert [command.end for command in geometry.path.commands] == [(9, 2), (13, 2), (13, 6), (9, 6)]
        assert geometry.path.id == 'r2.p0'
        assert geometry.to_svg_d() == result.current.params['d']
    assert result.fill is target.fill and result.original is target.original
    assert result.z == target.z and result.drawing_id == target.drawing_id
    assert np.array_equal(result.raster, target.raster)
    assert result.footprint.bounds == (9, 2, 13, 6)
    assert result.diagnostics['clones']['family_size'] == 4
    assert result.diagnostics['clones']['matched_source'] == 1
    assert target.current.kind == 'rect' and source.current.params['d'].startswith('M1 2')


def test_selected_mirror_pair_keeps_source_and_independent_target_paint():
    source, target = pair()
    result = choose(source, target,
        axis=SymmetryAxisInput(point=(7, 0), angle_degrees=90), matrix=(-1, 0, 0, 1, 14, 0))
    assert result[0] is source
    geometry = result[1].current.params['canonical_path_geometry']
    assert [command.end for command in geometry.path.commands] == [(13, 2), (9, 2), (9, 6), (13, 6)]
    assert geometry.to_svg_d() == result[1].current.params['d']
    assert result[1].fill is target.fill and result[1].original is target.original
    assert result[1].footprint.bounds == (9, 2, 13, 6)
    assert result[1].diagnostics['symmetry'] == {'accepted': True, 'mode': 'pair',
        'matched_source': 1, 'axis': {'point': (7.0, 0.0), 'angle_degrees': 90.0}}


@pytest.mark.parametrize('mirror', [False, True])
def test_transformed_copy_updates_canonical_coordinates_and_retains_corners(mirror):
    from dataclasses import replace
    from vectoreyes.path_geometry import CornerAnchor, geometry_from_svg_d
    source, target = pair()
    path = geometry_from_svg_d(source.current.params['d'], region_id='r1')
    end = path.path.commands[1].id
    path = replace(path, anchors=(CornerAnchor('r1.p0.corner0', end, end),))
    source = source.with_current(Shape('path', {'d': path.to_svg_d(), 'canonical_path_geometry': path}))
    if mirror:
        result = choose(source, target,
            axis=SymmetryAxisInput(point=(7, 0), angle_degrees=90),
            matrix=(-1, 0, 0, 1, 14, 0))[1]
        expected = (9, 2)
    else:
        result = choose(source, target, matrix=(1, 0, 0, 1, 8, 0), symbolic=False)[0]
        expected = (13, 2)
    updated = result.current.params['canonical_path_geometry']
    assert updated.path.commands[1].end == expected
    assert updated.to_svg_d() == result.current.params['d']
    endpoint = updated.path.commands[1].id
    assert updated.anchors == (CornerAnchor('r2.p0.corner0', endpoint, endpoint),)
    assert all(command.id.startswith('r2.p0.') for command in updated.path.commands)
    assert source.current.params['canonical_path_geometry'] is path


def test_transformed_cutout_keeps_sparse_command_bindings_and_rounded_numeric_output():
    from dataclasses import replace
    from vectoreyes.drawing_adapter import transform_working_shape
    from vectoreyes.path_geometry import CornerAnchor, geometry_from_svg_d
    geometry = geometry_from_svg_d('M0 0 L10 0 L10 10 L0 10 Z M2 2 L4 2 L4 4 L2 4 Z',
        region_id='r1', fill_rule='evenodd')
    hole = geometry.cutouts[0]
    commands = tuple(replace(command, id=f'r1.p1.c{index * 7}')
        for index, command in enumerate(hole.path.commands))
    hole = replace(hole, path=replace(hole.path, commands=commands),
        anchors=(CornerAnchor('hole.corner', commands[1].id, commands[1].id),))
    geometry = replace(geometry, cutouts=(hole,))
    result = transform_working_shape(Shape('path', {'d': geometry.to_svg_d(),
        'canonical_path_geometry': geometry, 'fill_rule': 'evenodd'}), (1, 0, 0, 1, .123456789, 2))
    updated = result.params['canonical_path_geometry']
    assert updated.outer.path.start == (.12, 2)
    assert updated.cutouts[0].path.start == (2.12, 4)
    assert updated.cutouts[0].anchors == hole.anchors
    assert tuple(command.id for command in updated.cutouts[0].path.commands) == tuple(command.id for command in commands)
    assert updated.to_svg_d() == result.params['d']
    assert result.params['fill_rule'] == 'evenodd'

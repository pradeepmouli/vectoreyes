from dataclasses import replace

import numpy as np
import pytest

from tests.test_canonical_recipe_replay import inputs
from vectormark.drawing import DrawingRegion
from vectormark.drawing_actions import DrawingPlan
from vectormark.drawing_fill import FlatFill
from vectormark.drawing_replay import _propose_public_operation, bind_recipe_operation
from vectormark.geometry_native import region_path
from vectormark.path_geometry import PathGeometry, PrimitiveGeometry, geometry_from_svg_d
from vectormark.replay_state import ReplayState


def proposal_case(op, *, second=False):
    evidence, paint, recipe = inputs()
    raster = np.zeros((8, 8), dtype=bool)
    raster[1:7, 1:7] = True
    observations = {'r1': {'id': 1, 'ancestry': (1,), 'raster': raster.tobytes(),
        'raster_shape': raster.shape, 'source_label': 7, 'color_hex': '#123456'}}
    drawing = evidence.drawing
    if second:
        region = DrawingRegion('r2', geometry_from_svg_d('M3 1 L7 1 L7 6 L3 6 Z', region_id='r2'),
            FlatFill('#654321'))
        drawing = replace(drawing, regions=(*drawing.regions, region))
        observations['r2'] = {**observations['r1'], 'id': 2, 'ancestry': (2,)}
    state = ReplayState(drawing, {'r1': {'obsolete_fit_receipt': 'old'}}, observations)
    action = DrawingPlan.model_validate({'version': 'vectormark.plan.v1',
        'drawing_id': 'test', 'base_version': 'v0', 'ops': [op]}).ops[0]
    operation = bind_recipe_operation(operation_id='geometry', action=action,
        settings=recipe.settings, evidence_revision=evidence.revision, state=state)
    return state, _propose_public_operation(evidence, paint, operation, state,
        recipe=recipe, corner_replays=[])


@pytest.mark.parametrize(('op', 'geometry_type'), [
    ({'op': 'fit_primitive', 'target': 'r1', 'type': 'rect'}, PrimitiveGeometry),
    ({'op': 'fit_path', 'target': 'r1'}, PathGeometry),
])
def test_region_fit_proposals_need_raster_but_no_original_shape_codec(op, geometry_type):
    before, after = proposal_case(op)
    assert isinstance(after.drawing.region('r1').geometry, geometry_type)
    assert after.region_evidence == before.region_evidence
    assert after.drawing.region('r1').fill == FlatFill('#123456')
    assert after.shape_metadata['r1'] == {}


def test_alignment_proposal_uses_canonical_geometry_and_clears_old_fit_metadata():
    before, after = proposal_case({'op': 'align', 'target': 'r1', 'reference': 'r2', 'axes': ['x']}, second=True)
    indexed = {region.id: region for region in after.drawing.regions}
    assert region_path(after.drawing.region('r1'), indexed).bounds == (2.5, 1., 7.5, 6.)
    assert after.drawing.region('r2') is before.drawing.region('r2')
    assert after.region_evidence == before.region_evidence
    assert after.shape_metadata['r1'] == {}


def test_split_proposal_inherits_source_observations_and_rasterizes_new_components():
    before, after = proposal_case({'op': 'split', 'target': 'r1',
        'divider': {'type': 'line', 'points': [[3, 0], [3, 8]]}})
    assert tuple(child.id for child in after.drawing.region('r1').children) == ('r1-1', 'r1-2')
    masks = []
    for number in (1, 2):
        info = after.region_evidence[f'r1-{number}']
        assert info['ancestry'] == (1, number)
        assert info['source_label'] == 7
        masks.append(np.frombuffer(info['raster'], dtype=bool).reshape(info['raster_shape']))
    assert all(mask.any() for mask in masks)
    assert not np.array_equal(*masks)
    assert after.region_evidence['r1'] == before.region_evidence['r1']
    assert set(after.shape_metadata) == {'r1-1', 'r1-2'}


def test_merge_proposal_unions_retained_source_rasters_and_retires_old_observations():
    _, after = proposal_case({'op': 'merge', 'regions': ['r1', 'r2'], 'id': 'merged'}, second=True)
    assert tuple(region.id for region in after.drawing.regions) == ('merged',)
    assert set(after.region_evidence) == {'merged'}
    info = after.region_evidence['merged']
    assert info['id'] == 3 and info['ancestry'] == (3,)
    assert info['original_kind'] == 'path'
    assert info['source_label'] is None
    assert np.frombuffer(info['raster'], dtype=bool).sum() == 36


def test_geometry_proposal_does_not_treat_an_empty_symmetry_observation_as_a_relationship():
    evidence, paint, recipe = inputs()
    leaf = replace(evidence.drawing.region('r1'), id='r1-1')
    branch = DrawingRegion('r1', None, children=(leaf,), diagnostics={'symmetry': None})
    state = ReplayState(replace(evidence.drawing, regions=(branch,)), {},
        {'r1': {'ancestry': (1,)}, 'r1-1': {'ancestry': (1, 1)}})
    action = DrawingPlan.model_validate({'version': 'vectormark.plan.v1',
        'drawing_id': 'test', 'base_version': 'v0', 'ops': [
            {'op': 'align', 'target': 'r1-1', 'reference': 'r1', 'axes': ['x']}]}).ops[0]
    operation = bind_recipe_operation(operation_id='align', action=action,
        settings=recipe.settings, evidence_revision=evidence.revision, state=state)
    result = _propose_public_operation(evidence, paint, operation, state, recipe=recipe, corner_replays=[])
    assert result.drawing.region('r1-1').geometry == leaf.geometry

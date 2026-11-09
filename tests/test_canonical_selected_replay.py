from dataclasses import replace

import pytest

from tests.test_canonical_recipe_replay import inputs
from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
from vectoreyes.drawing_actions import SetFillAction
from vectoreyes.drawing_fill import FlatFill
from vectoreyes.path_geometry import geometry_from_svg_d
from vectoreyes.replay_state import ReplayState


def selected_recipe():
    from vectoreyes.drawing_replay import bind_recipe_operation
    from vectoreyes.selected_outcomes import retain_selected_outcome

    evidence, paint, recipe = inputs()
    # Address evidence is sufficient; no original-shape or raster codec exists.
    evidence = replace(evidence, region_evidence={'r1': {'ancestry': (1,)}})
    before = ReplayState(evidence.drawing, evidence.shape_metadata, evidence.region_evidence)
    action = SetFillAction.model_validate({'op': 'set_fill', 'target': 'r1',
        'fill': {'type': 'flat', 'color': '#abcdef'}})
    operation = bind_recipe_operation(operation_id='paint', action=action,
        settings=recipe.settings, evidence_revision=evidence.revision, state=before)
    after = replace(before, drawing=replace(before.drawing,
        regions=(replace(before.drawing.region('r1'), fill=FlatFill('#abcdef')),)))
    operation, _ = retain_selected_outcome(operation, before, after,
        evidence=evidence, paint=paint, upstream=recipe)
    return evidence, paint, replace(recipe, steps=(*recipe.steps, operation))


def test_selected_recipe_replays_canonical_result_without_working_shape_observations():
    from vectoreyes.drawing_replay import replay_drawing

    evidence, paint, recipe = selected_recipe()
    result = replay_drawing(evidence, paint, recipe, finalize_output=False)
    assert result.drawing.region('r1').fill == FlatFill('#abcdef')
    assert result.drawing.region('r1').geometry == evidence.drawing.region('r1').geometry
    assert result.region_evidence == {'r1': {'ancestry': (1,)}}


@pytest.mark.parametrize('change', ['paint', 'geometry', 'observations'])
def test_selected_canonical_dependency_rejects_changed_inputs(change):
    from vectoreyes.drawing_replay import replay_drawing

    evidence, paint, recipe = selected_recipe()
    if change == 'observations':
        evidence = replace(evidence, region_evidence={'r1': {'ancestry': (1,), 'source_label': 7}})
    else:
        original = evidence.drawing.region('r1')
        updated = (replace(original, fill=FlatFill('#654321')) if change == 'paint'
            else replace(original, geometry=geometry_from_svg_d(
                'M1 1 L7 1 L6 6 L1 6 Z', region_id='r1')))
        evidence = replace(evidence, drawing=replace(evidence.drawing, regions=(updated,)))
    with pytest.raises(ValueError, match='selected operation dependencies changed'):
        replay_drawing(evidence, paint, recipe, finalize_output=False)


def test_selected_root_delta_prunes_retired_evidence_and_keeps_unmodified_roots():
    from vectoreyes.drawing_recipe import SelectedOperationOutcome
    from vectoreyes.selected_outcomes import apply_selected_outcome

    geometry = geometry_from_svg_d('M0 0 L4 0 L4 4 L0 4 Z', region_id='r1')
    child = DrawingRegion('r1-1', geometry, FlatFill('#123456'))
    root = DrawingRegion('r1', children=(child,))
    other = DrawingRegion('r2', geometry, FlatFill('#654321'))
    before = ReplayState(Drawing('test', Canvas(8, 8), (root, other)),
        {'r1': {'old': 1}, 'r1-1': {'old': 2}, 'r2': {'keep': 3}},
        {'r1': {'raster': b'\x01', 'raster_shape': (1, 1)}, 'r1-1': {'old': 2}, 'r2': {'keep': 3}})
    selected = replace(root, geometry=geometry, children=(), fill=FlatFill('#abcdef'))
    outcome = SelectedOperationOutcome('dependency', (selected,), ('r2', 'r1'),
        {'r1': {'new': 4}}, {'r1': {'raster': None, 'raster_source': 'r1', 'raster_shape': (1, 1)}})
    result = apply_selected_outcome(outcome, before)
    assert tuple(region.id for region in result.drawing.regions) == ('r2', 'r1')
    assert result.drawing.region('r2') is other
    assert result.drawing.region('r1').fill == FlatFill('#abcdef')
    assert result.shape_metadata == {'r1': {'new': 4}, 'r2': {'keep': 3}}
    assert set(result.region_evidence) == {'r1', 'r2'}
    assert result.region_evidence['r1']['raster'] == b'\x01'
    assert result.region_evidence['r1']['raster_source'] is None
    assert before.drawing.region('r1').children == (child,)


def test_selected_delta_rejects_unavailable_borrowed_source_coverage():
    from vectoreyes.drawing_recipe import SelectedOperationOutcome
    from vectoreyes.selected_outcomes import apply_selected_outcome

    evidence, _, _ = inputs()
    state = ReplayState(evidence.drawing, {}, {})
    outcome = SelectedOperationOutcome('dependency', evidence.drawing.regions, ('r1',),
        region_evidence={'r1': {'raster': None, 'raster_source': 'missing'}})
    with pytest.raises(ValueError, match='borrowed.*coverage.*missing'):
        apply_selected_outcome(outcome, state)


def test_paint_and_order_proposals_need_no_working_shape_or_raster_codec():
    from vectoreyes.drawing_actions import DrawingPlan
    from vectoreyes.drawing_replay import extend_recipe_with_plan, replay_drawing

    evidence, paint, base = inputs()
    second = DrawingRegion('r2', geometry_from_svg_d(
        'M1 1 L3 1 L3 3 L1 3 Z', region_id='r2'), FlatFill('#654321'))
    evidence = replace(evidence, drawing=replace(evidence.drawing,
        regions=(*evidence.drawing.regions, second)),
        region_evidence={'r1': {'ancestry': (1,)}, 'r2': {'ancestry': (2,)}})
    plan = DrawingPlan.model_validate({'version': 'vectoreyes.plan.v1',
        'drawing_id': 'test', 'base_version': 'v0', 'ops': [
            {'op': 'set_fill', 'target': 'r1', 'fill': {'type': 'flat', 'color': '#abcdef'}},
            {'op': 'set_z_order', 'targets': ['r2', 'r1']},
            {'op': 'set_fill', 'target': 'r2', 'fill': {'type': 'flat', 'color': '#fedcba'}},
        ]})
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    result = replay_drawing(evidence, paint, recipe, finalize_output=False)
    assert result.drawing.region('r1').fill == FlatFill('#abcdef')
    assert result.drawing.region('r2').fill == FlatFill('#fedcba')
    assert result.drawing.region('r1').z == 1 and result.drawing.region('r2').z == 0
    assert result.drawing.region('r1').geometry == evidence.drawing.region('r1').geometry
    assert result.drawing.region('r2').geometry == second.geometry
    assert result.region_evidence == evidence.region_evidence


def test_empty_plan_keeps_canonical_input_without_working_projection():
    from vectoreyes.drawing_actions import DrawingPlan
    from vectoreyes.drawing_replay import extend_recipe_with_plan

    evidence, paint, base = inputs()
    plan = DrawingPlan.model_validate({'version': 'vectoreyes.plan.v1',
        'drawing_id': 'test', 'base_version': 'v0', 'ops': []})
    result = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan)
    assert result.recipe == base

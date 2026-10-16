from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from vectormark.drawing_actions import ActionSettings, DrawingPlan
from vectormark.drawing_engine import RasterImage
from vectormark.drawing_recipe import EffectiveDrawingRecipe, build_geometry_evidence, build_paint_evidence
from vectormark.drawing_refine import resolved_action_settings
from vectormark.drawing_render import render_drawing
from vectormark.drawing_trace import TraceOptions, TraceResult
from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion
import vectormark.drawing_replay as replay_module
from vectormark.drawing_state import DrawingStore, VersionCandidate


def fixture():
    source = RasterImage(8, 8, 'rgb8', bytes([255]) * 8 * 8 * 3)
    trace = TraceResult(8, 8, TraceOptions(), (), '<svg/>')
    region = VectorRegion.from_shape(id=1, shape=Shape('rect', {'x': 1., 'y': 1., 'w': 4., 'h': 4.}),
        fill=FlatFill('#112233'), z=0, raster=np.ones((8, 8), dtype=bool), drawing_id='r1', source_regions=('r1',))
    evidence = build_geometry_evidence(kind='source', trace=trace, geometry_source=source, trace_roots=(region,))
    paint = build_paint_evidence(source)
    settings = ActionSettings(epsilon=1, max_error=1.5, fit_strategy='quadratic')
    return evidence, paint, EffectiveDrawingRecipe(evidence.revision, evidence.engine_revision, settings)


def test_purpose_and_execution_are_independent_and_resolved():
    evidence, _, _ = fixture()
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1', drawing_id='drw_test', base_version='v0',
        settings=dict(purpose='design_fidelity', execution='exploratory'),
        ops=[dict(op='fit_primitive', target='r1', type='circle', settings=dict(purpose='both'))]))
    settings = resolved_action_settings(evidence.trace, plan, plan.ops[0])
    assert settings.purpose == 'both'
    assert settings.execution == 'exploratory'
    assert settings.max_error == evidence.trace.options.curve_tolerance
    deterministic_design = settings.model_copy(update={'execution': 'deterministic'})
    assert deterministic_design.purpose == 'both'
    with pytest.raises(ValidationError):
        ActionSettings(epsilon=1, max_error=1, fit_strategy='quadratic', purpose='prettier')


def test_proposals_can_vary_but_reproduction_does_not_generate(monkeypatch):
    evidence, paint, recipe = fixture()
    recipe = replace(recipe, settings=recipe.settings.model_copy(update={'purpose': 'design_fidelity', 'execution': 'exploratory'}))
    assert hasattr(replay_module, 'propose_effective_recipe'), 'proposal execution must be explicit'
    assert hasattr(replay_module, 'reproduce_effective_recipe'), 'reproduction must not execute the recipe'
    execute = replay_module._replay_regions_with_producers
    calls = []
    def varying(*args, **kwargs):
        regions, producers, corners = execute(*args, **kwargs)
        calls.append(len(calls) + 1)
        region = VectorRegion.from_shape(id=1, shape=Shape('rect', {'x': 1., 'y': 1., 'w': float(calls[-1] + 2), 'h': 4.}),
            fill=FlatFill('#112233'), z=0, raster=np.ones((8, 8), dtype=bool), drawing_id='r1', source_regions=('r1',))
        return (region,), producers, corners
    monkeypatch.setattr(replay_module, '_replay_regions_with_producers', varying)
    first = replay_module.propose_effective_recipe(drawing_id='drw_test', evidence=evidence, paint=paint, recipe=recipe)
    first_svg = render_drawing(first.drawing).svg
    second = replay_module.propose_effective_recipe(drawing_id='drw_test', evidence=evidence, paint=paint, recipe=recipe)
    assert render_drawing(second.drawing).svg != first_svg
    for _ in range(2):
        reproduced = replay_module.reproduce_effective_recipe(drawing=first.drawing, evidence=evidence, paint=paint,
            recipe=recipe, replay=first.replay)
        assert render_drawing(reproduced).svg == first_svg
    assert calls == [1, 2]
    store = DrawingStore()
    def candidate(proposal):
        return VersionCandidate(drawing=proposal.drawing, evidence=evidence,
            submitted_plan=None, submitted_edits=(), effective_recipe=recipe,
            replay=proposal.replay, baseline_candidate_index=0)
    store.create('drw_test', paint, (candidate(first),))
    sibling_a = store.append('drw_test', 'v0', (candidate(first),))[0]
    sibling_b = store.append('drw_test', 'v0', (candidate(second),))[0]
    assert sibling_a.id != sibling_b.id
    assert render_drawing(store.get_version('drw_test', sibling_a.id).drawing).svg == first_svg
    assert render_drawing(store.get_version('drw_test', sibling_b.id).drawing).svg != first_svg
    assert calls == [1, 2]
    changed = replace(recipe, settings=recipe.settings.model_copy(update={'max_error': 2.}))
    with pytest.raises(ValueError, match='recipe revision mismatch'):
        replay_module.reproduce_effective_recipe(drawing=first.drawing, evidence=evidence, paint=paint, recipe=changed, replay=first.replay)
    changed_paint = build_paint_evidence(RasterImage(8, 8, 'rgb8', bytes([0]) * 8 * 8 * 3))
    with pytest.raises(ValueError, match='paint revision mismatch'):
        replay_module.reproduce_effective_recipe(drawing=first.drawing, evidence=evidence, paint=changed_paint, recipe=recipe, replay=first.replay)
    with pytest.raises(ValueError, match='output revision mismatch'):
        replay_module.reproduce_effective_recipe(drawing=second.drawing, evidence=evidence, paint=paint, recipe=recipe, replay=first.replay)
    changed_evidence = build_geometry_evidence(kind='source',
        trace=replace(evidence.trace, options=replace(evidence.trace.options, curve_tolerance=2.)),
        geometry_source=evidence.geometry_source, trace_roots=evidence.trace_roots)
    with pytest.raises(ValueError, match='evidence revision mismatch'):
        replay_module.reproduce_effective_recipe(drawing=first.drawing, evidence=changed_evidence,
            paint=paint, recipe=recipe, replay=first.replay)


def test_store_rejects_stale_dependencies_without_consuming_sibling_ids():
    evidence, paint, recipe = fixture()
    result = replay_module.propose_effective_recipe(drawing_id='drw_test', evidence=evidence, paint=paint, recipe=recipe)
    def candidate(outcome=result, intent=recipe):
        return VersionCandidate(drawing=outcome.drawing, evidence=evidence,
            submitted_plan=None, submitted_edits=(), effective_recipe=intent,
            replay=outcome.replay, baseline_candidate_index=0)
    store = DrawingStore()
    store.create('drw_test', paint, (candidate(),))
    changed = replace(recipe, settings=recipe.settings.model_copy(update={'max_error': 2.}))
    with pytest.raises(ValueError, match='recipe revision mismatch'):
        store.append('drw_test', 'v0', (candidate(intent=changed),))
    invalid_paint = replace(result, replay=replace(result.replay, paint_revision='stale'))
    with pytest.raises(ValueError, match='paint revision mismatch'):
        store.append('drw_test', 'v0', (candidate(invalid_paint),))
    def fail_preparation(_versions):
        raise RuntimeError('publication preparation failed')
    with pytest.raises(RuntimeError, match='publication preparation failed'):
        store.append_prepared('drw_test', 'v0', (candidate(),), prepare=fail_preparation)
    first = store.append('drw_test', 'v0', (candidate(),))[0]
    second_result = replay_module.propose_effective_recipe(drawing_id='drw_test', evidence=evidence, paint=paint, recipe=changed)
    second = store.append('drw_test', 'v0', (candidate(second_result, changed),))[0]
    assert (first.id, second.id) == ('v0.0', 'v0.1')
    assert store.get_version('drw_test', 'v0.0').effective_recipe == recipe
    assert store.get_version('drw_test', 'v0.1').effective_recipe == changed
    assert store.get_version('drw_test', 'v0').replay == result.replay


def test_public_refinement_retains_resolved_semantics(monkeypatch):
    import sys
    sys.path.insert(0, 'tests')
    import test_drawing_recipe_actions as helpers
    from vectormark import mcp_server
    monkeypatch.setattr(mcp_server, '_DRAWINGS', DrawingStore())
    ctx, traced = helpers._trace(helpers._single_region_uri())
    refined = helpers._refine(ctx, traced, [dict(op='fit_primitive', target='r1', type='rect',
        settings=dict(purpose='design_fidelity', execution='exploratory'))])
    version = mcp_server._DRAWINGS.get_version(traced['drawing_id'], refined['version'])
    operation = version.effective_recipe.operations[-1]
    result = version.replay.action_results[-1]
    assert result['operation_id'] == operation.id
    assert result['origin'] == operation.origin
    assert result['purpose'] == 'design_fidelity'
    assert result['execution'] == 'exploratory'
    assert result['settings']['max_error'] == operation.settings.max_error


@pytest.mark.parametrize('action', [
    dict(op='detect_primitives', target='r1'),
    dict(op='detect_symmetry', target='r1'),
    dict(op='update_geometry', target='r1', ops=[dict(op='simplify')]),
])
def test_agent_batch_semantics_are_retained(action, monkeypatch):
    import sys
    sys.path.insert(0, 'tests')
    import test_drawing_recipe_actions as helpers
    from vectormark import mcp_server
    monkeypatch.setattr(mcp_server, '_DRAWINGS', DrawingStore())
    ctx, traced = helpers._trace(helpers._single_region_uri())
    refined = helpers._refine(ctx, traced, [{**action,
        'settings': dict(purpose='both', execution='exploratory', max_error=2.)}])
    version = mcp_server._DRAWINGS.get_version(traced['drawing_id'], refined['version'])
    batch = version.effective_recipe.automatic_batches[-1]
    assert batch.origin == 'agent'
    assert batch.settings.purpose == 'both'
    assert batch.settings.execution == 'exploratory'
    assert batch.settings.max_error == 2.
    result = next(item for item in version.replay.action_results
        if item['operation_id'] == batch.batch_id and item['origin'] == 'agent')
    assert result['purpose'] == 'both'
    assert result['origin'] == 'agent'
    changed_batch = replace(batch, settings=batch.settings.model_copy(update={'purpose': 'accuracy'}))
    changed_recipe = replace(version.effective_recipe,
        steps=tuple(changed_batch if step is batch else step for step in version.effective_recipe.steps))
    assert changed_recipe.revision != version.effective_recipe.revision


def test_corner_edits_retain_resolved_settings_and_receipt():
    import sys
    sys.path.insert(0, 'tests')
    import test_drawing_corner_store_replay as helpers
    from vectormark.corner_recipe import AddCorner
    from vectormark.path_geometry import CornerAnchor
    from vectormark.drawing_recipe import ApplyCornerRecipeOperation
    store, base, paint, evidence = helpers._source_store()
    source_command = evidence.root_paths[0].fit.source.path.commands[10].id
    settings = base.resolved_settings.model_copy(update={
        'purpose': 'design_fidelity', 'execution': 'exploratory', 'max_error': 2.})
    recipe, _ = replay_module.extend_recipe_with_corner_edits(
        evidence=evidence, paint=paint, base_recipe=base.effective_recipe,
        base_version_id=base.id, selection=helpers._selection(base, evidence),
        edits=(AddCorner(CornerAnchor('agent.corner', source_command, source_command)),),
        settings=settings)
    application = next(step for step in recipe.steps if isinstance(step, ApplyCornerRecipeOperation))
    assert application.settings == settings
    assert application.recipe.options.max_error == 2.
    proposal = replay_module.propose_effective_recipe(drawing_id=base.drawing.drawing_id,
        evidence=evidence, paint=paint, recipe=recipe)
    result = next(item for item in proposal.replay.action_results if item['kind'] == 'corner_recipe')
    assert result['purpose'] == 'design_fidelity'
    assert result['execution'] == 'exploratory'


def test_public_corner_action_resolves_plan_and_action_settings():
    import sys
    sys.path.insert(0, 'tests')
    import test_drawing_corner_store_replay as helpers
    from vectormark.drawing_recipe import ApplyCornerRecipeOperation
    store, base, paint, evidence = helpers._source_store()
    geometry = base.drawing.regions[0].geometry
    command_id = geometry.path.commands[0].id
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id=base.drawing.drawing_id, base_version=base.id,
        settings=dict(purpose='design_fidelity', execution='exploratory', max_error=2.),
        ops=[dict(op='add_corner', target='r1', path_id=geometry.path.id,
            feature_id='agent.corner', entry_command_id=command_id,
            exit_command_id=command_id, settings=dict(purpose='both'))]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base.effective_recipe, plan=plan).recipe
    application = next(step for step in recipe.steps if isinstance(step, ApplyCornerRecipeOperation))
    assert application.settings.purpose == 'both'
    assert application.settings.execution == 'exploratory'
    assert application.recipe.options.max_error == 2.

from dataclasses import replace
from pathlib import Path
import sys

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

sys.path.insert(0, str(Path(__file__).resolve().parent))


def fixture():
    source = RasterImage(8, 8, 'rgb8', bytes([255]) * 8 * 8 * 3)
    trace = TraceResult(8, 8, TraceOptions(), (), '<svg/>')
    region = VectorRegion.from_shape(id=1, shape=Shape('rect', {'x': 1., 'y': 1., 'w': 4., 'h': 4.}),
        fill=FlatFill('#112233'), z=0, raster=np.ones((8, 8), dtype=bool), drawing_id='r1', source_regions=('r1',))
    evidence = build_geometry_evidence(kind='source', trace=trace, geometry_source=source, trace_roots=(region,))
    paint = build_paint_evidence(source)
    settings = ActionSettings(epsilon=1, max_error=1.5, fit_strategy='quadratic')
    return evidence, paint, EffectiveDrawingRecipe(evidence.revision, evidence.engine_revision, settings)


def test_selected_components_preserve_retired_path_slots_and_clone_ownership():
    from vectormark.drawing_adapter import drawing_from_vector_regions
    from vectormark.drawing_recipe import SelectedGeometryOutcome
    from vectormark.optimizer.topology import normalize_retained_topology
    d = ('M0 0 L4 0 L4 4 L0 4 Z '
         'M1 1 L2 1 L2 2 L1 2 Z '
         'M5 5 L7 5 L7 7 L5 7 Z')
    root = VectorRegion.from_shape(id=1, shape=Shape('path', {'d': d}),
        fill=FlatFill('#112233'), z=0, raster=np.ones((8, 8), dtype=bool),
        drawing_id='r1', source_regions=('r1',))
    clone = replace(root, id=2, drawing_id='r2', source_regions=('r2',),
        current=Shape('use', {'href_obj_id': 1, 'transform': (1, 0, 0, 1, 0, 0)}))
    # Reuse begins from the concrete pre-fit region, not an already split tree.
    before = (replace(root, current=Shape('rect', dict(x=0, y=0, w=7, h=7))), clone)
    after = normalize_retained_topology((root, clone))
    selected = drawing_from_vector_regions('drw_test', width=8, height=8, regions=after).region('r1')
    assert [leaf.id for leaf in selected.leaves()] == ['r1-1', 'r1-3']
    result = replay_module._apply_selected_geometry(SelectedGeometryOutcome('dep', selected), before)
    drawing = drawing_from_vector_regions('drw_test', width=8, height=8, regions=result)
    assert [leaf.id for leaf in drawing.region('r1').leaves()] == ['r1-1', 'r1-3']
    assert [leaf.id for leaf in drawing.region('r2').leaves()] == ['r2-1', 'r2-3']
    assert drawing.region('r1').diagnostics['topology'] == selected.diagnostics['topology']


@pytest.mark.parametrize('with_cutout', [False, True])
@pytest.mark.parametrize('with_clone', [False, True])
def test_selected_path_fit_retains_split_components_without_refitting(monkeypatch, with_cutout, with_clone):
    import vectormark.drawing_region_actions as actions
    evidence, paint, _ = fixture()
    raster = np.zeros((8, 8), dtype=bool)
    raster[0:4, 0:4] = True
    raster[5:7, 5:7] = True
    if with_cutout:
        raster[1:3, 1:3] = False
    root = replace(evidence.trace_roots[0], raster=raster)
    roots = (root,)
    if with_clone:
        clone = replace(root, id=2, drawing_id='r2', source_regions=('r2',),
            current=Shape('use', {'href_obj_id': 1, 'transform': (1, 0, 0, 1, 0, 0)}))
        roots = (root, clone)
    evidence = build_geometry_evidence(kind='source', trace=evidence.trace,
        geometry_source=evidence.geometry_source, trace_roots=roots)
    base = EffectiveDrawingRecipe(evidence.revision, evidence.engine_revision,
        ActionSettings(epsilon=0.1, max_error=0.1, fit_strategy='quadratic'))
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1', drawing_id='drw_test',
        base_version='v0', ops=[dict(op='fit_path', target='r1')]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    monkeypatch.setattr(actions, 'fit_region_path',
        lambda *args, **kwargs: pytest.fail('selected component fit was regenerated'))
    first = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    assert first.drawing.region('r1').is_branch
    assert len(first.drawing.region('r1').leaves()) == 2
    if with_cutout:
        from vectormark.path_geometry import PathGeometryWithCutouts
        assert any(isinstance(leaf.geometry, PathGeometryWithCutouts)
                   for leaf in first.drawing.region('r1').leaves())
    if with_clone:
        from vectormark.path_geometry import CloneGeometry
        assert len(first.drawing.region('r2').leaves()) == 2
        assert all(isinstance(leaf.geometry, CloneGeometry)
                   for leaf in first.drawing.region('r2').leaves())
    second = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    assert render_drawing(first.drawing).svg == render_drawing(second.drawing).svg
    target = first.drawing.region('r1').leaves()[0].id
    child_plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1', drawing_id='drw_test',
        base_version='v1', ops=[dict(op='set_fill', target=target,
            fill=dict(type='flat', color='#ff0000'))]))
    child_recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=recipe, plan=child_plan).recipe
    child = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=child_recipe)
    assert child.drawing.region(target).fill.color == '#ff0000'


def test_order_plan_delegates_to_canonical_drawing_action(monkeypatch):
    import vectormark.drawing_region_actions as actions
    from vectormark.drawing import Drawing
    evidence, paint, base = fixture()
    calls = []
    execute = actions.set_drawing_z_order
    def record(drawing, action):
        calls.append(drawing)
        return execute(drawing, action)
    monkeypatch.setattr(actions, 'set_drawing_z_order', record)
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0', ops=[dict(op='set_z_order', targets=['r1'])]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    replay_module.propose_effective_recipe(drawing_id='drw_test', evidence=evidence, paint=paint, recipe=recipe)
    assert calls and all(isinstance(drawing, Drawing) for drawing in calls)


def test_radial_fill_plan_preserves_noncentral_focal_point_through_replay():
    evidence, paint, base = fixture()
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0', ops=[dict(op='set_fill', target='r1',
            fill=dict(type='radial_gradient', center=[4, 4], radius=4, focal_point=[2, 3],
                stops=[dict(offset=0, color='#FFFFFF'), dict(offset=1, color='#000000')]))]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    result = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    assert result.drawing.regions[0].fill.focal_point == (2., 3.)
    svg = render_drawing(result.drawing).svg
    assert 'fx="2"' in svg and 'fy="3"' in svg


def test_working_radial_fill_renderer_preserves_focal_point():
    from vectormark.candidate import RadialGradientFill
    from vectormark.emit import resolve_fill
    definitions = []
    resolve_fill(RadialGradientFill({'cx': 4, 'cy': 4, 'r': 4, 'fx': 2, 'fy': 3},
        [(0., '#FFFFFF'), (1., '#000000')]), definitions)
    assert 'fx="2"' in definitions[0] and 'fy="3"' in definitions[0]


def test_fill_plan_executes_canonical_region_action_without_geometry_solver(monkeypatch):
    import vectormark.drawing_region_actions as actions
    from vectormark.drawing import DrawingRegion
    evidence, paint, base = fixture()
    calls = []
    execute = actions.set_region_fill
    def record(region, action):
        calls.append(region)
        return execute(region, action)
    monkeypatch.setattr(actions, 'set_region_fill', record)
    monkeypatch.setattr(actions, 'fit_primitive_geometry',
        lambda *args, **kwargs: pytest.fail('paint assignment invoked geometry fitter'))
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0',
        ops=[dict(op='set_fill', target='r1', fill=dict(type='flat', color='#AA2200'))]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    result = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    assert calls and all(isinstance(region, DrawingRegion) for region in calls)
    assert result.drawing.regions[0].fill.color == '#aa2200'


def test_path_plan_delegates_to_typed_raster_kernel_once(monkeypatch):
    import vectormark.path_fitting as fitting
    evidence, paint, base = fixture()
    calls = []
    execute = fitting.fit_raster_paths
    def record(raster, **kwargs):
        calls.append((raster.copy(), kwargs))
        return execute(raster, **kwargs)
    monkeypatch.setattr(fitting, 'fit_raster_paths', record)
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0',
        ops=[dict(op='fit_path', target='r1', settings=dict(max_error=2.0,
            fit_strategy='progressive'))]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], evidence.trace_roots[0].raster)
    assert calls[0][1]['region_id'] == 'r1'
    assert calls[0][1]['settings'].max_error == 2.0
    assert calls[0][1]['settings'].fit_strategy == 'progressive'


def test_whole_region_fit_retains_matching_prefit_corner_ingress():
    from vectormark.path_fitting import fit_raster_paths
    from vectormark.optimizer.corners import path_corner_diagnostics
    evidence, paint, base = fixture()
    fitted = fit_raster_paths(evidence.trace_roots[0].raster, region_id='r1', settings=base.settings)[0]
    d = fitted.to_svg_d()
    point = fitted.path.drawable_commands[0].end
    corners = path_corner_diagnostics(d, declared_corners=(point,))
    assert len(corners['spans']) == 1
    source = evidence.trace_roots[0].with_current(Shape('path', {'d': d}),
        diagnostics={'corners': corners})
    evidence = build_geometry_evidence(kind='source', trace=evidence.trace,
        geometry_source=evidence.geometry_source, trace_roots=(source,))
    base = EffectiveDrawingRecipe(evidence.revision, evidence.engine_revision, base.settings)
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0', ops=[dict(op='fit_path', target='r1')]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    result = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    assert len(result.drawing.region('r1').geometry.anchors) == 1


def test_path_plan_executes_canonical_region_action(monkeypatch):
    import vectormark.drawing_region_actions as actions
    from vectormark.drawing import DrawingRegion
    evidence, paint, base = fixture()
    calls = []
    execute = actions.fit_region_path
    def record(region, action, **kwargs):
        calls.append((region, kwargs['settings']))
        return execute(region, action, **kwargs)
    monkeypatch.setattr(actions, 'fit_region_path', record)
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0',
        ops=[dict(op='fit_path', target='r1', settings=dict(epsilon=0.25))]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    assert len(calls) == 1
    assert isinstance(calls[0][0], DrawingRegion)
    assert calls[0][1].epsilon == 0.25


def test_primitive_plan_executes_the_canonical_region_action(monkeypatch):
    import vectormark.drawing_region_actions as actions
    from vectormark.drawing import DrawingRegion
    evidence, paint, base = fixture()
    calls = []
    execute = actions.fit_region_primitive
    def record(region, action, **kwargs):
        calls.append((region, kwargs['settings']))
        return execute(region, action, **kwargs)
    monkeypatch.setattr(actions, 'fit_region_primitive', record)
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0',
        ops=[dict(op='fit_primitive', target='r1', type='rect',
            settings=dict(epsilon=0.25))]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    assert len(calls) == 1
    assert isinstance(calls[0][0], DrawingRegion)
    assert calls[0][1].epsilon == 0.25


@pytest.mark.parametrize('kind', ['rect', 'rounded_rect', 'ellipse', 'circle', 'polygon'])
def test_primitive_action_fits_retained_raster_not_current_outline(kind):
    evidence, paint, base = fixture()
    # The retained raster is 8x8, while the displayed rectangle is only 4x4.
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0',
        ops=[dict(op='fit_primitive', target='r1', type=kind)]))
    selected = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    result = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=selected)
    from vectormark.path_geometry import (
        CirclePrimitive, EllipsePrimitive, PolygonPrimitive,
        RectanglePrimitive, RoundedRectanglePrimitive,
    )
    primitive = result.drawing.regions[0].geometry.primitive
    if kind == 'circle':
        assert primitive == CirclePrimitive((4., 4.), 4.)
    elif kind == 'ellipse':
        assert primitive == EllipsePrimitive((4., 4.), 4., 4.)
    elif kind == 'rounded_rect':
        assert primitive == RoundedRectanglePrimitive((0., 0.), 8., 8., 0., 0.)
    elif kind == 'rect':
        assert primitive == RectanglePrimitive((0., 0.), 8., 8.)
    else:
        assert isinstance(primitive, PolygonPrimitive)
        assert len(primitive.vertices) == 4
    assert result.drawing.regions[0].diagnostics['geometry']['explicit'] == kind


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


def test_selected_fit_is_reused_when_extending_recipe_without_calling_fitter(monkeypatch):
    import vectormark.drawing_region_actions as refine_module
    evidence, paint, base = fixture()
    calls = []
    def varying_fit(*args, **kwargs):
        calls.append(len(calls) + 1)
        from vectormark.path_geometry import PrimitiveGeometry, RectanglePrimitive
        return PrimitiveGeometry(RectanglePrimitive((1., 1.), float(2 + calls[-1]), 4.))
    monkeypatch.setattr(refine_module, 'fit_primitive_geometry', varying_fit)
    def plan(ops):
        return DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
            drawing_id='drw_test', base_version='v0', ops=ops))
    fit = dict(op='fit_primitive', target='r1', type='rect',
        settings=dict(purpose='design_fidelity', execution='exploratory'))
    selected = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan([fit])).recipe
    first = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=selected)
    assert calls == [1]
    child = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=selected, plan=plan([dict(op='set_fill', target='r1',
            fill=dict(type='flat', color='#FF0000'))])).recipe
    child_result = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=child)
    assert calls == [1]
    assert child_result.drawing.regions[0].geometry == first.drawing.regions[0].geometry
    reproposed = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan([fit])).recipe
    second = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=reproposed)
    assert calls == [1, 2]
    assert second.drawing.regions[0].geometry != first.drawing.regions[0].geometry


def test_selected_fit_rejects_changed_settings_or_paint_without_refitting(monkeypatch):
    import vectormark.drawing_region_actions as refine_module
    evidence, paint, base = fixture()
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0',
        ops=[dict(op='fit_primitive', target='r1', type='rect')]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    operation = recipe.operations[0]
    monkeypatch.setattr(refine_module, 'fit_primitive_geometry',
        lambda *args, **kwargs: pytest.fail('selected fit was regenerated'))
    changed = replace(operation, settings=operation.settings.model_copy(update={'max_error': 2.}))
    changed_recipe = replace(recipe, steps=tuple(changed if step is operation else step for step in recipe.steps))
    with pytest.raises(ValueError, match='selected geometry dependencies changed'):
        replay_module.propose_effective_recipe(drawing_id='drw_test',
            evidence=evidence, paint=paint, recipe=changed_recipe)
    changed_paint = build_paint_evidence(RasterImage(8, 8, 'rgb8', bytes([0]) * 8 * 8 * 3))
    with pytest.raises(ValueError, match='selected geometry dependencies changed'):
        replay_module.propose_effective_recipe(drawing_id='drw_test',
            evidence=evidence, paint=changed_paint, recipe=recipe)
    changed_upstream = replace(recipe, settings=recipe.settings.model_copy(update={'purpose': 'both'}))
    with pytest.raises(ValueError, match='selected geometry dependencies changed'):
        replay_module.propose_effective_recipe(drawing_id='drw_test',
            evidence=evidence, paint=paint, recipe=changed_upstream)


@pytest.mark.parametrize('method', ['fit_path', 'update_geometry'])
def test_selected_path_geometry_replays_without_running_path_fitter(method, monkeypatch):
    import test_drawing_corner_store_replay as helpers
    import vectormark.drawing_refine as refine_module
    store, base, paint, evidence = helpers._source_store()
    commands = base.drawing.regions[0].geometry.path.commands
    action = dict(op=method, target='r1')
    if method == 'update_geometry':
        action['ops'] = [dict(op='fit', between=[commands[0].id, commands[1].id], type='quadratic')]
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id=base.drawing.drawing_id, base_version=base.id, ops=[action]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base.effective_recipe, plan=plan).recipe
    assert recipe.operations[0].selected_geometry is not None
    expected = replay_module.propose_effective_recipe(drawing_id=base.drawing.drawing_id,
        evidence=evidence, paint=paint, recipe=recipe)
    def fail(*args, **kwargs):
        pytest.fail('selected path was refitted')
    import vectormark.drawing_region_actions as actions
    monkeypatch.setattr(actions, 'fit_region_path', fail)
    monkeypatch.setattr(refine_module, '_path_shape', fail)
    actual = replay_module.propose_effective_recipe(drawing_id=base.drawing.drawing_id,
        evidence=evidence, paint=paint, recipe=recipe)
    assert actual.drawing == expected.drawing
    assert actual.replay.path_correspondence == expected.replay.path_correspondence


def test_selected_fit_preserves_canonical_corner_diagnostics():
    evidence, paint, base = fixture()
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0', ops=[dict(op='fit_path', target='r1')]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    operation = recipe.operations[0]
    uncached = replace(recipe, steps=tuple(
        replace(step, selected_geometry=None) if step is operation else step
        for step in recipe.steps))
    selected = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    generated = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=uncached)
    assert selected.drawing.regions[0].geometry == generated.drawing.regions[0].geometry
    assert selected.drawing.regions[0].diagnostics == generated.drawing.regions[0].diagnostics


def test_selected_fit_preserves_unrelated_symmetry_materialization():
    import test_retained_symmetry_export as symmetry_helpers
    import test_drawing_recipe_actions as helpers
    forest = symmetry_helpers._forest()
    target = VectorRegion.from_shape(id=3,
        shape=Shape('rect', dict(x=42., y=2., w=4., h=4.)), fill=FlatFill('#112233'),
        z=2, raster=np.ones((24, 48), dtype=bool), drawing_id='r3', source_regions=('r3',))
    evidence, paint, base = helpers._direct_recipe_fixture((*forest, target), width=48, height=24)
    plan = DrawingPlan.model_validate(dict(version='vectormark.plan.v1',
        drawing_id='drw_test', base_version='v0', ops=[dict(op='fit_primitive', target='r3', type='rect')]))
    recipe = replay_module.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=base, plan=plan).recipe
    selected = replay_module.propose_effective_recipe(drawing_id='drw_test',
        evidence=evidence, paint=paint, recipe=recipe)
    assert not selected.drawing.regions[0].children

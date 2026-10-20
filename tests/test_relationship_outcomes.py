from dataclasses import replace

import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.drawing_actions import ActionSettings
from vectormark.drawing_plan import DrawingPlan
from tests._working_evidence import capture_working_geometry_evidence
from vectormark.optimizer.assembly_binding import working_regions_from_evidence
from vectormark.drawing_recipe import EffectiveDrawingRecipe, build_paint_evidence
from vectormark.drawing_replay import extend_recipe_with_plan, propose_effective_recipe
from vectormark.drawing_render import render_drawing
from vectormark.drawing_trace import TraceOptions, TraceResult
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion
from vectormark.drawing_engine import RasterImage


def evidence_fixture():
    source = RasterImage(16, 16, 'rgb8', bytes([255]) * 16 * 16 * 3)
    trace = TraceResult(16, 16, TraceOptions(), (), '<svg/>')
    roots = tuple(VectorRegion.from_shape(id=index, shape=Shape('rect', dict(x=x, y=1., w=4., h=4.)),
        fill=FlatFill('#112233'), z=index, raster=np.ones((16, 16), dtype=bool),
        drawing_id=f'r{index}', source_regions=(f'r{index}',)) for index, x in ((1, 1.), (2, 9.)))
    evidence = capture_working_geometry_evidence(kind='source', trace=trace, geometry_source=source, trace_roots=roots)
    return evidence, build_paint_evidence(source), EffectiveDrawingRecipe(evidence.revision,
        evidence.engine_revision, ActionSettings(epsilon=1, max_error=1.5, fit_strategy='quadratic'))


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('transforms,left', [([{'tx': 8}], 9), ([{'tx': 8}, {'tx': 4}], 5)])
def test_source_edit_refreshes_clone_footprint_before_later_partition_and_replay(nested, transforms, left):
    source = RasterImage(16, 16, 'rgb8', bytes([255]) * 16 * 16 * 3)
    trace = TraceResult(16, 16, TraceOptions(), (), '<svg/>')
    roots = tuple(VectorRegion.from_shape(id=index,
        shape=Shape('rect', dict(x=x, y=y, w=4., h=4.)), fill=FlatFill('#112233'),
        z=index, raster=np.ones((16, 16), dtype=bool), drawing_id=f'r{index}',
        source_regions=(f'r{index}',)) for index, x, y in ((1, 1., 1.), (2, 9., 1.), (3, 1., 9.)))
    source_id = 'r1-1' if nested else 'r1'
    if nested:
        roots = (VectorRegion.branch(id=10, children=(replace(roots[0], drawing_id=source_id),), drawing_id='r1'), *roots[1:])
    evidence = capture_working_geometry_evidence(kind='source', trace=trace, geometry_source=source, trace_roots=roots)
    paint = build_paint_evidence(source)
    base = EffectiveDrawingRecipe(evidence.revision, evidence.engine_revision,
        ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    plan = DrawingPlan(version='vectormark.plan.v1', drawing_id='drw_test', base_version='v0', ops=[
        {'op': 'assign_clone', 'source': source_id, 'target': 'r2', 'transforms': transforms},
        {'op': 'align', 'target': source_id, 'reference': 'r3', 'axes': ['y']},
    ])
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    replay = propose_effective_recipe(evidence=evidence, paint=paint, recipe=recipe, drawing_id='drw_test')
    from vectormark.geometry_native import region_path
    indexed = {region.id: region for region in replay.drawing.regions}
    for root in replay.drawing.regions:
        indexed.update({leaf.id: leaf for leaf in root.leaves()})
    assert region_path(replay.drawing.region('r2'), indexed).bounds == (left, 9, 13, 13)
    assert region_path(replay.drawing.region('r1'), indexed).bounds == (1, 9, 5, 13)
    split = DrawingPlan(version='vectormark.plan.v1', drawing_id='drw_test', base_version='v1', ops=[
        {'op': 'split', 'target': 'r2', 'divider': {'type': 'line', 'points': [(0, 11), (16, 11)]}},
    ])
    revised = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=recipe, plan=split).recipe
    result = propose_effective_recipe(evidence=evidence, paint=paint, recipe=revised, drawing_id='drw_test')
    assert result.drawing.region('r2').is_branch


def test_clone_geometry_is_refreshed_before_paint_invalidation():
    from vectormark.drawing_refine import refine

    roots = tuple(VectorRegion.from_shape(id=index,
        shape=Shape('rect', dict(x=x, y=y, w=4., h=4.)), fill=FlatFill('#112233'),
        z=index, raster=np.ones((16, 16), dtype=bool), drawing_id=f'r{index}')
        for index, x, y in ((1, 1., 1.), (2, 9., 1.), (3, 1., 9.)))
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:8, :, 0] = 255
    image[8:, :, 1] = 255
    plan = DrawingPlan(version='vectormark.plan.v1', drawing_id='drw_test', base_version='v0', ops=[
        {'op': 'assign_clone', 'source': 'r1', 'target': 'r2', 'transforms': [{'tx': 8}]},
        {'op': 'align', 'target': 'r1', 'reference': 'r3', 'axes': ['y']},
    ])
    actual = refine(TraceResult(16, 16, TraceOptions(), (), '<svg/>'), roots, plan, rgb=image)
    assert next(region for region in actual if region.drawing_id == 'r2').fill.hex.lower() == '#00ff00'


@pytest.mark.parametrize('operation', [
    {'op': 'assign_clone', 'source': 'r1', 'target': 'r2', 'transforms': [{'tx': 8}]},
    {'op': 'align', 'target': 'r2', 'reference': 'r1', 'axes': ['y']},
    {'op': 'set_symmetry', 'target': 'r1', 'axis': {'point': [3, 3], 'angle_degrees': 90}},
    {'op': 'set_fill', 'target': 'r1', 'fill': {'type': 'flat', 'color': '#123456'}},
    {'op': 'set_z_order', 'targets': ['r2', 'r1']},
    {'op': 'merge', 'id': 'joined', 'regions': ['r1', 'r2']},
    {'op': 'split', 'target': 'r1', 'divider': {'type': 'line', 'points': [[3, 0], [3, 8]]}},
])
def test_nonfit_selected_outcomes_replay_without_executing_actions(operation, monkeypatch):
    from vectormark import drawing_refine
    evidence, paint, base = evidence_fixture()
    plan = DrawingPlan(version='vectormark.plan.v1', drawing_id='drw_test', base_version='v0', ops=[operation])
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    assert recipe.operations[0].selected_outcome is not None
    original = propose_effective_recipe(evidence=evidence, paint=paint, recipe=recipe, drawing_id='drw_test')
    def forbidden(*args, **kwargs):
        raise AssertionError('selected action executed again')
    monkeypatch.setattr(drawing_refine, 'refine', forbidden)
    import vectormark.drawing_replay as replay
    monkeypatch.setattr(replay, '_propose_public_operation', forbidden)
    repeated = propose_effective_recipe(evidence=evidence, paint=paint, recipe=recipe, drawing_id='drw_test')
    assert render_drawing(repeated.drawing).svg == render_drawing(original.drawing).svg
    assert repeated.drawing == original.drawing


@pytest.mark.parametrize('name,operation', [
    ('assign_clone', {'op': 'assign_clone', 'source': 'r1', 'target': 'r2', 'transforms': [{'tx': 8}]}),
    ('align_region', {'op': 'align', 'target': 'r2', 'reference': 'r1', 'axes': ['y']}),
    ('set_region_symmetry', {'op': 'set_symmetry', 'target': 'r1', 'axis': {'point': [3, 3], 'angle_degrees': 90}}),
])
def test_relationship_plans_delegate_to_canonical_actions_once(name, operation, monkeypatch):
    from vectormark import drawing_relationships
    evidence, paint, base = evidence_fixture()
    original = getattr(drawing_relationships, name)
    calls = []
    def record(drawing, action):
        calls.append(action.op)
        return original(drawing, action)
    monkeypatch.setattr(drawing_relationships, name, record)
    plan = DrawingPlan(version='vectormark.plan.v1', drawing_id='drw_test', base_version='v0', ops=[operation])
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    propose_effective_recipe(evidence=evidence, paint=paint, recipe=recipe, drawing_id='drw_test')
    assert calls == [operation['op']]


def test_selected_outcome_preserves_new_component_coverage():
    from vectormark.drawing_recipe import RecipeOperation
    from vectormark.drawing_actions import FitPathDrawingAction
    from tests._working_evidence import retain_working_operation_outcome
    from vectormark.types import CoverageField
    evidence, paint, upstream = evidence_fixture()
    roots = working_regions_from_evidence(evidence)
    root = roots[0]
    coverage = CoverageField(np.full((16, 16), .5), (0, 0), (16, 16))
    child = VectorRegion.from_shape(id=1, drawing_id='r1-1', shape=root.current,
        fill=root.fill, z=root.z, raster=root.raster, coverage=coverage, source_regions=('r1',))
    branch = VectorRegion.branch(id=1, drawing_id='r1', children=(child,), source_regions=('r1',))
    operation = RecipeOperation('test.fit', 'agent', FitPathDrawingAction(op='fit_path', target='r1'), upstream.settings, ())
    _, projected = retain_working_operation_outcome(operation, roots,
        (branch, roots[1]), evidence=evidence, paint=paint, upstream=upstream)
    actual = projected[0].children[0].coverage
    assert isinstance(actual, CoverageField)
    assert np.array_equal(actual.values, coverage.values)
    assert actual.origin == coverage.origin and actual.canvas_shape == coverage.canvas_shape


def test_selected_outcome_borrows_unchanged_rasters_from_bound_input():
    from vectormark.drawing_adapter import working_region_evidence
    evidence, _, _ = evidence_fixture()
    roots = working_regions_from_evidence(evidence)
    records = working_region_evidence(roots, baseline=roots)
    assert records['r1']['raster_source'] == 'r1'
    assert records['r1']['raster'] is None


def test_nonfit_outcome_rejects_changed_settings_with_operation_error():
    from vectormark.drawing_recipe import RecipeOperation
    evidence, paint, base = evidence_fixture()
    plan = DrawingPlan(version='vectormark.plan.v1', drawing_id='drw_test', base_version='v0',
        ops=[{'op': 'align', 'target': 'r2', 'reference': 'r1', 'axes': ['y']}])
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    stale = replace(recipe, steps=tuple(replace(step, settings=step.settings.model_copy(update={'max_error': 3}))
        if isinstance(step, RecipeOperation) else step for step in recipe.steps))
    with pytest.raises(ValueError, match='selected operation dependencies changed'):
        propose_effective_recipe(evidence=evidence, paint=paint, recipe=stale, drawing_id='drw_test')


@pytest.mark.parametrize('duplicate', [True, False])
def test_outcome_rejects_ambiguous_or_empty_root_ownership(duplicate):
    from vectormark.drawing_recipe import SelectedOperationOutcome
    evidence, _, _ = evidence_fixture()
    root = evidence.drawing.regions[0]
    with pytest.raises(ValueError, match='unique retained output addresses'):
        SelectedOperationOutcome('dep', (root, root) if duplicate else (), ('r1',) if duplicate else ())


def test_deliberate_relationship_proposals_can_vary_but_selected_replay_does_not(monkeypatch):
    from vectormark import drawing_relationships
    from vectormark.path_geometry import CloneGeometry, AffineTransform
    evidence, paint, base = evidence_fixture()
    execute = drawing_relationships.assign_clone
    calls = []
    def varying(drawing, action):
        calls.append(action.op)
        target = execute(drawing, action)
        return replace(target, geometry=CloneGeometry('r1', (AffineTransform(tx=len(calls) * 4),)))
    monkeypatch.setattr(drawing_relationships, 'assign_clone', varying)
    plan = DrawingPlan(version='vectormark.plan.v1', drawing_id='drw_test', base_version='v0',
        ops=[{'op': 'assign_clone', 'source': 'r1', 'target': 'r2', 'transforms': [{'tx': 8}]}])
    recipes = [extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe for _ in range(2)]
    children = [propose_effective_recipe(evidence=evidence, paint=paint, recipe=recipe, drawing_id='drw_test') for recipe in recipes]
    assert len(calls) == 2
    assert children[0].drawing.region('r2').geometry.transforms[0].tx == 4
    assert children[1].drawing.region('r2').geometry.transforms[0].tx == 8

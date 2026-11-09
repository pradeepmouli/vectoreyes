from dataclasses import replace

import pytest
from pydantic import ValidationError

from tests.test_canonical_structural_replay import leaf
from tests.test_drawing_recipe_actions import _direct_recipe_fixture
from vectoreyes.drawing_actions import DrawingPlan
from vectoreyes.drawing_proposals import detect_path_regions, detect_seam_regions
from vectoreyes.drawing_replay import (
    _replay_drawing_with_producers, extend_recipe_with_plan, propose_effective_recipe,
)
from vectoreyes.fit import Shape


def cleanup_case(d, ops, *, settings=None):
    evidence, paint, base = _direct_recipe_fixture(
        (leaf(1, Shape('path', {'d': d}), '#123456'),), width=300, height=440)
    plan = DrawingPlan.model_validate({'version': 'vectoreyes.plan.v1',
        'drawing_id': 'cleanup', 'base_version': 'v0',
        'ops': [{'op': 'update_geometry', 'target': 'r1',
                 'settings': {'epsilon': 1, 'max_error': 1, **(settings or {})}, 'ops': ops}]})
    state, _, _ = _replay_drawing_with_producers(evidence, paint, base, finalize_output=False)
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    return evidence, paint, state, recipe


def test_public_smoothing_matches_shared_engine_and_replays_without_rediscovery(monkeypatch):
    evidence, paint, state, recipe = cleanup_case(
        'M0 0 Q2 9.8 10 10 Q18 9.8 20 0 L0 0 Z',
        [{'op': 'smooth', 'max_boundary_error': 1.1}], settings={'max_error': 2})
    expected = detect_path_regions(state, operation='smooth', max_error=1.1, target_id='r1')
    assert recipe.automatic_batches[0].operations
    actual = propose_effective_recipe(drawing_id='cleanup', evidence=evidence, paint=paint, recipe=recipe)
    assert actual.drawing.region('r1').geometry == expected.state.drawing.region('r1').geometry
    assert actual.drawing.region('r1').geometry != state.drawing.region('r1').geometry
    def forbidden(*args, **kwargs):
        raise AssertionError('accepted smoothing must not be rediscovered during replay')
    monkeypatch.setattr('vectoreyes.drawing_proposals.discover_path_proposals', forbidden)
    assert propose_effective_recipe(drawing_id='cleanup', evidence=evidence, paint=paint,
        recipe=recipe).drawing == actual.drawing


def test_cleanup_zero_boundary_budget_is_not_replaced_by_default():
    from vectoreyes.drawing_actions import ActionSettings, SmoothPathAction
    from vectoreyes.drawing_refine import resolved_path_cleanup_controls
    settings = ActionSettings(epsilon=1, max_error=2, fit_strategy='quadratic')
    assert resolved_path_cleanup_controls(SmoothPathAction(op='smooth', max_boundary_error=0), settings)[3] == 0
    assert resolved_path_cleanup_controls(SmoothPathAction(op='smooth'), settings)[3] == 2


def test_smoothing_preserves_order_among_direct_path_edits():
    from tests.test_drawing_recipe_actions import _sequential_update_drawing
    from vectoreyes.drawing_recipe import RecipeOperation
    from vectoreyes.refinement.automatic_replay import ReplayBatch
    d = 'M0 0 Q2 9.8 10 10 Q18 9.8 20 0 L0 0 Z'
    ops = [{'op': 'smooth'}, {'op': 'fit', 'target': 'r1.p0.c3', 'type': 'keep'}, {'op': 'smooth'}]
    evidence, paint, _, recipe = cleanup_case(d, ops, settings={'max_error': 1.1})
    stages = tuple(step for step in recipe.steps if isinstance(step, (ReplayBatch, RecipeOperation)))
    assert tuple(type(step) for step in stages) == (ReplayBatch, RecipeOperation, ReplayBatch)
    plan = DrawingPlan.model_validate({'version': 'vectoreyes.plan.v1', 'drawing_id': 'cleanup',
        'base_version': 'v0', 'ops': [{'op': 'update_geometry', 'target': 'r1',
            'settings': {'epsilon': 1, 'max_error': 1.1}, 'ops': ops}]})
    actual = propose_effective_recipe(drawing_id='cleanup', evidence=evidence, paint=paint, recipe=recipe)
    assert actual.drawing == _sequential_update_drawing('cleanup', evidence, paint, plan)


def test_public_simplification_retains_controls_and_matches_shared_engine():
    d = ('M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 '
         'Q256.31 399.74 253.5 413 L247 417.5 Z')
    evidence, paint, state, recipe = cleanup_case(d, [{'op': 'simplify',
        'linelet_only': True, 'samples': 24, 'cubic': True, 'max_boundary_error': 1.25}])
    expected = detect_path_regions(state, operation='simplify', epsilon=1, max_error=1,
        linelet_only=True, samples=24, cubic=True, max_boundary_error=1.25, target_id='r1')
    assert recipe.automatic_batches[0].operations
    action = recipe.automatic_batches[0].operations[0].intent.action
    assert (action.linelet_only, action.samples, action.cubic, action.max_boundary_error) == (True, 24, True, 1.25)
    actual = propose_effective_recipe(drawing_id='cleanup', evidence=evidence, paint=paint, recipe=recipe)
    assert actual.drawing.region('r1').geometry == expected.state.drawing.region('r1').geometry


@pytest.mark.parametrize('post_symmetry', [False, True])
def test_public_stitch_forwards_cubic_and_seam_policy(post_symmetry):
    regions = (leaf(1, Shape('path', {'d': 'M10 10 L49.2 10 L49.2 70 L10 70 Z'}), '#112233'),
        replace(leaf(2, Shape('path', {'d': 'M50.4 10 L90 10 L90 70 L50.4 70 Z'}), '#445566'), z=1))
    evidence, paint, base = _direct_recipe_fixture(regions, width=120, height=80)
    state, _, _ = _replay_drawing_with_producers(evidence, paint, base, finalize_output=False)
    plan = DrawingPlan.model_validate({'version': 'vectoreyes.plan.v1',
        'drawing_id': 'cleanup', 'base_version': 'v0', 'ops': [{'op': 'stitch',
        'settings': {'epsilon': 1, 'max_error': 1},
        'targets': ['r1', 'r2'], 'cubic': True, 'post_symmetry': post_symmetry, 'allow_underlap': False}]})
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    expected = detect_seam_regions(state, epsilon=1, max_error=1, target_ids=('r1', 'r2'),
        cubic=True, post_symmetry=post_symmetry, allow_underlap=False)
    if not post_symmetry:
        assert expected.batches[0].operations
        assert recipe.automatic_batches[0].operations[0].intent.action.cubic is True
    assert recipe.automatic_batches[0].operations == expected.batches[0].operations


@pytest.mark.parametrize('operation', [
    {'op': 'simplify', 'samples': 1}, {'op': 'simplify', 'max_boundary_error': -1},
    {'op': 'smooth', 'max_boundary_error': -1},
])
def test_cleanup_controls_reject_invalid_values(operation):
    with pytest.raises(ValidationError):
        DrawingPlan.model_validate({'version': 'vectoreyes.plan.v1', 'drawing_id': 'cleanup',
            'base_version': 'v0', 'ops': [{'op': 'update_geometry', 'target': 'r1', 'ops': [operation]}]})

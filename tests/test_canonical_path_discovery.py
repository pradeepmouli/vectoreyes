from dataclasses import replace

import pytest

from tests.test_canonical_structural_replay import leaf
from vectormark.corner_diagnostics import path_corner_diagnostics
from vectormark.drawing import Canvas
from vectormark.drawing_automatic import replay_automatic_batches
from vectormark.fit import Shape
from vectormark.refinement.automatic_replay import CornerDiagnosticsObservation
from vectormark.refinement.replay_binding import capture_replay_state
from vectormark.replay_identity import canonical_input_revision


def state_for(d, *, rounded=False):
    region = leaf(1, Shape('path', {'d': d}), '#123456')
    if rounded:
        region = replace(region, diagnostics={'corners': path_corner_diagnostics(
            d, declared_corner_spans=(((10, 0), (14, 4)),))})
    return capture_replay_state((region,), Canvas(300, 440))


def forbid_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('path discovery requires a sampling-scene projection')
    monkeypatch.setattr('vectormark.refinement.region_sample.RegionSample.__init__', forbidden)
    monkeypatch.setattr('vectormark.drawing_adapter.bind_path_input', forbidden)


def test_smooth_discovery_selects_and_replays_canonical_paths(monkeypatch):
    from vectormark.drawing_proposals import detect_path_regions
    state = state_for('M0 0 Q2 9.8 10 10 Q18 9.8 20 0 L0 0 Z')
    forbid_sampling(monkeypatch)
    result = detect_path_regions(state, operation='smooth', max_error=1.1)
    assert result.state.drawing.region('r1').geometry != state.drawing.region('r1').geometry
    assert [item.status for item in result.outcomes] == ['accepted']
    replayed = replay_automatic_batches(state, result.batches)
    assert canonical_input_revision(replayed) == canonical_input_revision(result.state)


def test_declared_corner_discovery_normalizes_without_sampling_or_redetection(monkeypatch):
    from vectormark.drawing_proposals import detect_path_regions
    state = state_for('M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z', rounded=True)
    forbid_sampling(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('normalization must not redetect corners')
    monkeypatch.setattr('vectormark.corner_diagnostics.path_corner_diagnostics', forbidden)
    result = detect_path_regions(state, operation='normalize_corners', max_error=1.)
    assert result.state.drawing.region('r1').geometry.to_svg_d().count('Q') == 1
    assert len(result.batches[0].operations) == 1
    replayed = replay_automatic_batches(state, result.batches)
    assert canonical_input_revision(replayed) == canonical_input_revision(result.state)


@pytest.mark.parametrize('enabled', [False, True])
def test_missing_corner_declaration_is_observation_only(monkeypatch, enabled):
    from vectormark.drawing_proposals import detect_path_regions
    state = state_for('M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z')
    state = replace(state, drawing=replace(state.drawing, regions=(replace(
        state.drawing.regions[0], diagnostics={}),)))
    forbid_sampling(monkeypatch)
    result = detect_path_regions(state, operation='normalize_corners',
        max_error=1., corner_normalize=enabled)
    assert result.state.drawing.region('r1').geometry == state.drawing.region('r1').geometry
    assert isinstance(result.outcomes[0].intent, CornerDiagnosticsObservation)
    report = result.state.drawing.region('r1').diagnostics['corners']
    assert report['skipped'][0]['reason'] == 'missing_or_stale_provenance'


def test_public_corner_normalization_does_not_project_sampling_scene(monkeypatch):
    from tests.test_drawing_recipe_actions import _direct_recipe_fixture
    from vectormark.drawing_plan import DrawingPlan
    from vectormark.drawing_replay import extend_recipe_with_plan, propose_effective_recipe
    d = 'M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z'
    region = replace(leaf(1, Shape('path', {'d': d}), '#123456'), diagnostics={
        'corners': path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (14, 4)),))})
    evidence, paint, base = _direct_recipe_fixture((region,), width=300, height=440)
    plan = DrawingPlan.model_validate({'version': 'vectormark.plan.v1',
        'drawing_id': 'drw_normalize', 'base_version': 'v0',
        'ops': [{'op': 'update_geometry', 'target': 'r1',
                 'ops': [{'op': 'normalize_corners'}]}]})
    def forbidden(*args, **kwargs):
        raise AssertionError('public corner cleanup projected a sampling scene')
    monkeypatch.setattr('vectormark.drawing_replay.working_regions_from_state', forbidden)
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    result = propose_effective_recipe(drawing_id='drw_normalize', evidence=evidence, paint=paint, recipe=recipe)
    assert result.drawing.region('r1').geometry.to_svg_d().count('Q') == 1


def test_simplification_discovery_prepares_once_and_selection_reuses_the_choice(monkeypatch):
    from vectormark.drawing_proposals import discover_simplification_proposals, select_canonical_pass
    state = state_for('M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 Q256.31 399.74 253.5 413 L247 417.5 Z')
    forbid_sampling(monkeypatch)
    proposals = discover_simplification_proposals(state, epsilon=1., max_error=1.,
        linelet_only=True, max_boundary_error=1.)
    assert proposals
    def forbidden(*args, **kwargs):
        raise AssertionError('simplification must not run twice during selection')
    monkeypatch.setattr('vectormark.drawing_automatic.select_automatic_paths', forbidden)
    result = select_canonical_pass(state, proposals, pass_name='simplify')
    assert result.outcomes[0].status == 'accepted'
    assert result.state.drawing.region('r1').geometry != state.drawing.region('r1').geometry


@pytest.mark.parametrize('covered', [False, True])
def test_simplification_keeps_uncovered_cutout_and_can_drop_a_covered_one(monkeypatch, covered):
    from tests.refinement.test_pass_simplify import _dense_rounded_rect_d
    from vectormark.drawing_proposals import detect_path_regions
    outer = _dense_rounded_rect_d(8., 8., 88., 88., 18., samples=12)
    hole = 'M35 35 L50 35 L42 50 Z'
    background = leaf(1, Shape('path', {'d': f'{outer} {hole}', 'fill_rule': 'evenodd'}), '#123456')
    regions = (background, replace(leaf(2, Shape('path', {'d': hole}), '#ffffff'), z=1)) if covered else (background,)
    state = capture_replay_state(regions, Canvas(300, 440))
    forbid_sampling(monkeypatch)
    result = detect_path_regions(state, operation='simplify', epsilon=1., max_error=1.)
    selected = result.state.drawing.region('r1').geometry.to_svg_d()
    assert selected.count('M') == (1 if covered else 2)
    if covered:
        assert selected.count('Q') == 8
        assert result.state.drawing.region('r2').geometry == state.drawing.region('r2').geometry
    replayed = replay_automatic_batches(state, result.batches)
    assert canonical_input_revision(replayed) == canonical_input_revision(result.state)


def test_prepared_discovery_cannot_be_reused_with_different_settings():
    from vectormark.drawing_proposals import discover_simplification_proposals, select_canonical_pass
    state = state_for('M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 Q256.31 399.74 253.5 413 L247 417.5 Z')
    proposal, = discover_simplification_proposals(state, epsilon=1., max_error=1., linelet_only=True)
    changed = replace(proposal, intent=replace(proposal.intent,
        action=replace(proposal.intent.action, max_error=2.)))
    with pytest.raises(ValueError, match='prepared proposal binding'):
        select_canonical_pass(state, (changed,), pass_name='simplify')


def test_public_simplification_does_not_project_sampling_scene(monkeypatch):
    from tests.test_drawing_recipe_actions import _direct_recipe_fixture
    from vectormark.drawing_plan import DrawingPlan
    from vectormark.drawing_replay import extend_recipe_with_plan, propose_effective_recipe
    d = 'M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 Q256.31 399.74 253.5 413 L247 417.5 Z'
    region = leaf(1, Shape('path', {'d': d}), '#123456')
    evidence, paint, base = _direct_recipe_fixture((region,), width=300, height=440)
    plan = DrawingPlan.model_validate({'version': 'vectormark.plan.v1',
        'drawing_id': 'drw_simplify', 'base_version': 'v0',
        'ops': [{'op': 'update_geometry', 'target': 'r1',
                 'settings': {'epsilon': 1., 'max_error': 1.},
                 'ops': [{'op': 'simplify'}]}]})
    def forbidden(*args, **kwargs):
        raise AssertionError('public simplification projected a sampling scene')
    monkeypatch.setattr('vectormark.drawing_replay.working_regions_from_state', forbidden)
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    assert recipe.automatic_batches[0].operations
    result = propose_effective_recipe(drawing_id='drw_simplify', evidence=evidence, paint=paint, recipe=recipe)
    assert result.drawing.region('r1').geometry.to_svg_d() != d


def test_simplification_does_not_rewrite_a_retained_clone_source():
    from vectormark.drawing_proposals import discover_simplification_proposals
    from vectormark.path_geometry import AffineTransform, CloneGeometry
    d = 'M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 Q256.31 399.74 253.5 413 L247 417.5 Z'
    state = state_for(d)
    assert discover_simplification_proposals(state, epsilon=1., max_error=1., linelet_only=True)
    instance = leaf(2, Shape('path', {'d': d}), '#abcdef')
    state = capture_replay_state((leaf(1, Shape('path', {'d': d}), '#123456'), instance), Canvas(300, 440))
    source, target = state.drawing.regions
    state = replace(state, drawing=replace(state.drawing, regions=(source, replace(target,
        geometry=CloneGeometry(source.id, (AffineTransform(1, 0, 0, 1, 20, 0),))))))
    assert not discover_simplification_proposals(state, epsilon=1., max_error=1., linelet_only=True)


@pytest.mark.parametrize('change', ['revision', 'scope'])
def test_discovery_prepared_choice_is_pinned_to_revision_and_scope(change):
    from vectormark.drawing_proposals import discover_simplification_proposals, select_canonical_pass
    d = 'M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 Q256.31 399.74 253.5 413 L247 417.5 Z'
    state = state_for(d)
    proposal, = discover_simplification_proposals(state, epsilon=1., max_error=1., linelet_only=True)
    prepared = replace(proposal.prepared, **({'input_revision': 'stale'} if change == 'revision' else {'scope': (10,)}))
    with pytest.raises(ValueError, match='prepared proposal binding'):
        select_canonical_pass(state, (replace(proposal, prepared=prepared),), pass_name='simplify')


def test_linelet_cleanup_does_not_treat_multiple_subpaths_as_a_single_run():
    from tests.refinement.test_pass_simplify import _dense_rounded_rect_d
    from vectormark.drawing_proposals import discover_simplification_proposals
    outer = _dense_rounded_rect_d(8., 8., 88., 88., 18., samples=12)
    region = leaf(1, Shape('path', {'d': f'{outer} M35 35 L50 35 L42 50 Z', 'fill_rule': 'evenodd'}), '#123456')
    state = capture_replay_state((region,), Canvas(300, 440))
    assert not discover_simplification_proposals(state, epsilon=1., max_error=1., linelet_only=True)

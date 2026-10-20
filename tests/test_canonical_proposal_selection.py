from dataclasses import replace

from tests.test_canonical_multi_replay import forbid_working_execution
from tests.test_canonical_structural_replay import leaf
from vectormark.drawing import Canvas
from vectormark.drawing_automatic import replay_automatic_batches
from vectormark.fit import Shape
from vectormark.path_geometry import PathGeometry, PrimitiveGeometry, RectanglePrimitive
from vectormark.quality import GeometryPolicy
from vectormark.refinement.automatic_replay import primitive_intent
from vectormark.refinement.replay_binding import capture_replay_state
from vectormark.replay_identity import canonical_input_revision


def fixture():
    return capture_replay_state((leaf(1, Shape('path', {'d': 'M0 0 L10 0 L10 10 L0 10 Z'}), '#123456'),), Canvas(300, 440))


def test_primitive_discovery_selection_and_replay_need_no_sampling_scene(monkeypatch):
    from vectormark.drawing_proposals import discover_primitive_proposals, select_canonical_pass
    state = fixture()
    forbid_working_execution(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('canonical proposal discovery must not construct a RegionSample')
    monkeypatch.setattr('vectormark.refinement.region_sample.RegionSample.__init__', forbidden)
    proposals = discover_primitive_proposals(state, epsilon=.1)
    result = select_canonical_pass(state, proposals, pass_name='primitives_pass')
    assert result.state.drawing.region('r1').geometry.primitive.width == 10
    assert result.outcomes[0].status == 'accepted'
    replayed = replay_automatic_batches(state, (result.batch,), finalize_topology=False)
    assert canonical_input_revision(replayed) == canonical_input_revision(result.state)
    assert isinstance(state.drawing.region('r1').geometry, PathGeometry)


def test_rejected_candidate_does_not_consume_the_input_or_change_the_next_choice():
    from vectormark.drawing_proposals import CanonicalProposal, select_canonical_pass
    state = fixture()
    bad = primitive_intent(1, PrimitiveGeometry(RectanglePrimitive((0, 0), 100, 10)), epsilon=.1)
    good = primitive_intent(1, PrimitiveGeometry(RectanglePrimitive((0, 0), 10, 10)), epsilon=.1)
    policy = GeometryPolicy(max_boundary_error_px=.1)
    result = select_canonical_pass(state, (CanonicalProposal((1,), bad, policy),
        CanonicalProposal((1,), good, policy)), pass_name='candidate_test')
    assert [item.status for item in result.outcomes] == ['rejected', 'accepted']
    assert len(result.batch.entries) == 1
    assert result.state.drawing.region('r1').geometry.primitive.width == 10
    assert result.outcomes[0].operation.assessment.max_boundary_error_px > 80


def test_proposal_overlap_and_stale_binding_are_explicit_rejections():
    from vectormark.drawing_proposals import discover_primitive_proposals, select_canonical_pass
    state = fixture()
    proposal, = discover_primitive_proposals(state, epsilon=.1)
    result = select_canonical_pass(state, (proposal, proposal), pass_name='candidate_test')
    assert [item.status for item in result.outcomes] == ['accepted', 'rejected']
    assert result.outcomes[1].diagnostics['reason'] == 'overlapping_accepted_proposal'
    # Changing the source after selection cannot replay the retained batch.
    changed = replace(state, drawing=replace(state.drawing,
        regions=(replace(state.drawing.regions[0], geometry=PrimitiveGeometry(
            RectanglePrimitive((0, 0), 12, 10))),)))
    import pytest
    with pytest.raises(ValueError, match='input revision mismatch'):
        replay_automatic_batches(changed, (result.batch,))


def test_public_primitive_detection_uses_canonical_discovery(monkeypatch):
    from tests.test_drawing_recipe_actions import _disk_region_uri, _trace, _refine
    ctx, traced = _trace(_disk_region_uri())
    def forbidden(*args, **kwargs):
        raise AssertionError('primitive plan discovery projected a sampling scene')
    monkeypatch.setattr('vectormark.refinement.replay_binding.working_regions_from_state', forbidden)
    monkeypatch.setattr('vectormark.drawing_refine.run_detection_with_intent', forbidden)
    result = _refine(ctx, traced, [{'op': 'detect_primitives', 'target': 'r1'}])
    assert result['version'] != traced['version']


def test_optional_fit_without_candidate_is_rejected_without_consuming_input(monkeypatch):
    from vectormark.drawing_proposals import CanonicalProposal, discover_primitive_proposals, select_canonical_pass
    from vectormark.refinement.automatic_replay import PassInputRegionRef, SmoothPathOperation, operation_intent
    state = fixture()
    monkeypatch.setattr('vectormark.drawing_automatic.select_automatic_paths', lambda *args, **kwargs: None)
    action = SmoothPathOperation(PassInputRegionRef('', '', (1,)), 1.)
    skipped = CanonicalProposal((1,), operation_intent(action, (action.target,), (1,)))
    good, = discover_primitive_proposals(state, epsilon=.1)
    result = select_canonical_pass(state, (skipped, good), pass_name='candidate_test')
    assert [item.status for item in result.outcomes] == ['rejected', 'accepted']
    assert result.outcomes[0].diagnostics['reason'] == 'no_applicable_candidate'
    assert len(result.batch.entries) == 1


def test_scoped_detection_keeps_repeated_child_ids_separate():
    from vectormark.drawing_proposals import detect_primitive_regions
    from vectormark.refinement.region_sample import RegionSample
    shape = Shape('path', {'d': 'M0 0 L10 0 L10 10 L0 10 Z'})
    first = RegionSample.branch(id=10, drawing_id='parentA', children=(leaf(1, shape, '#123456'),))
    child = replace(leaf(1, shape, '#abcdef'), drawing_id='other')
    second = RegionSample.branch(id=20, drawing_id='parentB', children=(child,))
    state = capture_replay_state((first, second), Canvas(300, 440))
    first_id = state.drawing.regions[0].children[0].id
    second_id = state.drawing.regions[1].children[0].id
    result = detect_primitive_regions(state, epsilon=.1, target_id=second_id)
    assert isinstance(result.state.drawing.region(first_id).geometry, PathGeometry)
    assert isinstance(result.state.drawing.region(second_id).geometry, PrimitiveGeometry)
    assert [batch.scope for batch in result.batches] == [(10,), (20,), ()]
    replayed = replay_automatic_batches(state, result.batches)
    assert canonical_input_revision(replayed) == canonical_input_revision(result.state)


def test_invalid_prepared_geometry_returns_a_failed_assessment(monkeypatch):
    from vectormark.drawing_automatic import AutomaticBatchApplication
    from vectormark.drawing_proposals import discover_primitive_proposals, select_canonical_pass
    from vectormark.path_geometry import AffineTransform, CloneGeometry
    state = fixture()
    proposals = discover_primitive_proposals(state, epsilon=.1)
    original_prepare = AutomaticBatchApplication.prepare
    def invalid_prepare(self, intent):
        selected = original_prepare(self, intent)
        bundle, = selected.regions
        broken = replace(bundle.region, geometry=CloneGeometry('missing', (AffineTransform(1, 0, 0, 1, 0, 0),)))
        return replace(selected, regions=(replace(bundle, region=broken),))
    monkeypatch.setattr(AutomaticBatchApplication, 'prepare', invalid_prepare)
    result = select_canonical_pass(state, proposals, pass_name='candidate_test')
    assert result.outcomes[0].status == 'rejected'
    assert 'invalid_geometry' in result.outcomes[0].assessment.reasons
    assert not result.batch.entries
    assert canonical_input_revision(result.state) == canonical_input_revision(state)

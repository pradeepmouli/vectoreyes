from dataclasses import replace

import numpy as np
import pytest

from tests.test_canonical_recipe_replay import inputs
from tests.refinement.test_intent_replay import _disk_image
from vectoreyes.drawing import Canvas
from vectoreyes.refinement.automatic_replay import replay_batches
from vectoreyes.refinement.framework import select_refinements_with_intent
from vectoreyes.refinement.passes.primitives import primitives_pass
from vectoreyes.refinement.replay_binding import capture_replay_state, working_regions_from_state
from vectoreyes.replay_identity import canonical_input_revision
from vectoreyes.refinement.trace import trace_regions
from vectoreyes.pipeline import Options
from vectoreyes.replay_state import ReplayState


def primitive_case(*, annotated=False):
    image = _disk_image()
    objects, _ = trace_regions(image, Options())
    objects = [replace(obj, drawing_id=f'r{obj.id}') for obj in objects]
    if annotated:
        from vectoreyes.fit import Shape
        objects = [replace(obj, current=Shape(obj.current.kind,
            {**obj.current.params, 'trace_annotation': 'retire when replaced'})) for obj in objects]
    state = capture_replay_state(objects, Canvas(image.shape[1], image.shape[0]))
    objects = list(working_regions_from_state(state))
    masks = {obj.id: obj.raster for obj in objects}
    recorded = select_refinements_with_intent(objects, masks, [primitives_pass])
    expected = capture_replay_state(replay_batches(objects, recorded.batches, masks=masks), state.drawing.canvas)
    return state, recorded.batches, expected


def test_empty_automatic_program_needs_no_working_observations():
    from vectoreyes.drawing_automatic import replay_automatic_batches
    evidence, _, _ = inputs()
    state = ReplayState(evidence.drawing, {}, {})
    assert replay_automatic_batches(state, ()) is state


@pytest.mark.parametrize('kind', ['smooth', 'simplify', 'corners'])
def test_path_choices_replay_without_working_action_publication(kind, monkeypatch):
    from vectoreyes.drawing_automatic import replay_automatic_batches
    from vectoreyes.candidate import FlatFill
    from vectoreyes.fit import Shape
    from vectoreyes.refinement.region_sample import RegionSample
    from vectoreyes.refinement.passes.smooth import smooth_pass
    from vectoreyes.refinement.passes.simplify import simplify_pass
    from vectoreyes.refinement.passes.corners import corner_normalize_pass

    if kind == 'simplify':
        d = 'M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 Q256.31 399.74 253.5 413 L247 417.5 Z'
        configure = lambda objects, masks: simplify_pass(objects, masks, epsilon=1.,
            max_error=1., linelet_only=True, max_boundary_error=1.)
        size = (460, 300)
    elif kind == 'smooth':
        d = 'M0 0 Q2 9.8 10 10 Q18 9.8 20 0 L0 0 Z'
        configure = lambda objects, masks: smooth_pass(objects, masks, max_error=1.1)
        size = (32, 32)
    else:
        d = 'M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z'
        configure = lambda objects, masks: corner_normalize_pass(objects, masks,
            max_error=1., enabled=True)
        size = (16, 24)
    from vectoreyes.refinement.gate import rasterize
    from vectoreyes.refinement.region_sample import to_polygon
    shape = Shape('path', {'d': d})
    source = RegionSample.from_shape(id=1, drawing_id='r1', shape=shape,
        fill=FlatFill('#112233'), z=0, raster=rasterize(to_polygon(shape), size))
    state = capture_replay_state((source,), Canvas(size[1], size[0]))
    objects = list(working_regions_from_state(state))
    masks = {1: objects[0].raster}
    recorded = select_refinements_with_intent(objects, masks, [configure])
    assert recorded.batches[0].operations
    expected = capture_replay_state(replay_batches(objects, recorded.batches, masks=masks), state.drawing.canvas)

    def forbidden(*args, **kwargs):
        raise AssertionError('path choice used working action publication')
    monkeypatch.setattr('vectoreyes.refinement.automatic_replay._execute_intent', forbidden)
    result = replay_automatic_batches(state, recorded.batches)
    assert result.drawing == expected.drawing
    assert result.region_evidence == expected.region_evidence
    assert canonical_input_revision(result) == canonical_input_revision(expected)
    assert state.drawing.region('r1').fill == result.drawing.region('r1').fill


@pytest.mark.parametrize('annotated', [False, True])
def test_retained_primitive_replays_canonically_without_numeric_action_execution(monkeypatch, annotated):
    from vectoreyes.drawing_automatic import replay_automatic_batches
    state, batches, expected = primitive_case(annotated=annotated)
    def forbidden(*args, **kwargs):
        raise AssertionError('primitive used the working action executor')
    monkeypatch.setattr('vectoreyes.refinement.automatic_replay._execute_intent', forbidden)
    result = replay_automatic_batches(state, batches)
    assert result.drawing.regions[0].geometry == batches[0].operations[0].intent.action.primitive
    assert result.drawing.regions[0].fill == state.drawing.regions[0].fill
    assert result.region_evidence == expected.region_evidence
    assert canonical_input_revision(result) == canonical_input_revision(expected)


def test_intrinsic_symmetry_replays_without_geometry_publication(monkeypatch):
    from vectoreyes.drawing_automatic import replay_automatic_batches
    from vectoreyes.candidate import FlatFill
    from vectoreyes.fit import Shape
    from vectoreyes.refinement.region_sample import RegionSample
    from vectoreyes.refinement.passes.symmetry import symmetry_pass
    from vectoreyes.refinement.gate import rasterize
    from vectoreyes.refinement.region_sample import to_polygon
    shape = Shape('rect', {'x': 20., 'y': 20., 'w': 40., 'h': 40.})
    source = RegionSample.from_shape(id=3, drawing_id='r3', shape=shape,
        fill=FlatFill('#445566'), z=0, raster=rasterize(to_polygon(shape), (96, 96)))
    state = capture_replay_state((source,), Canvas(96, 96))
    objects = list(working_regions_from_state(state))
    masks = {3: objects[0].raster}
    recorded = select_refinements_with_intent(objects, masks, [symmetry_pass])
    from vectoreyes.refinement.automatic_replay import DeclareIntrinsicSymmetryOperation
    assert isinstance(recorded.batches[0].operations[0].intent.action, DeclareIntrinsicSymmetryOperation)
    expected = capture_replay_state(replay_batches(objects, recorded.batches, masks=masks), state.drawing.canvas)
    def forbidden(*args, **kwargs):
        raise AssertionError('intrinsic declaration used working action publication')
    monkeypatch.setattr('vectoreyes.refinement.automatic_replay._execute_intent', forbidden)
    result = replay_automatic_batches(state, recorded.batches)
    assert result.drawing == expected.drawing
    assert result.drawing.region('r3').geometry == state.drawing.region('r3').geometry
    assert canonical_input_revision(result) == canonical_input_revision(expected)


def test_nested_primitive_replay_preserves_scoped_identity_and_receipt(monkeypatch):
    from vectoreyes.drawing_automatic import replay_automatic_batches
    from vectoreyes.refinement.region_sample import RegionSample
    state, _, _ = primitive_case()
    child, = working_regions_from_state(state)
    first = RegionSample.branch(id=10, drawing_id='r10', children=(replace(child, drawing_id='r10-1'),))
    second = RegionSample.branch(id=20, drawing_id='r20', children=(replace(child, drawing_id='r20-1'),))
    state = capture_replay_state((first, second), state.drawing.canvas)
    objects = list(working_regions_from_state(state))
    masks = {obj.id: obj.raster for obj in objects}
    recorded = select_refinements_with_intent(objects, masks, [primitives_pass])
    assert any(batch.scope for batch in recorded.batches if batch.operations)
    expected = capture_replay_state(replay_batches(objects, recorded.batches, masks=masks), state.drawing.canvas)
    def forbidden(*args, **kwargs):
        raise AssertionError('scoped primitive used working action publication')
    monkeypatch.setattr('vectoreyes.refinement.automatic_replay._execute_intent', forbidden)
    result = replay_automatic_batches(state, recorded.batches)
    assert result.drawing == expected.drawing
    assert result.region_evidence == expected.region_evidence
    assert canonical_input_revision(result) == canonical_input_revision(expected)


def test_canonical_automatic_replay_rejects_changed_input_before_publication():
    from vectoreyes.drawing_automatic import replay_automatic_batches
    from vectoreyes.drawing_fill import FlatFill
    state, batches, _ = primitive_case()
    changed = replace(state, drawing=replace(state.drawing, regions=(
        replace(state.drawing.regions[0], fill=FlatFill('#ffffff')),)))
    with pytest.raises(ValueError, match='input revision mismatch'):
        replay_automatic_batches(changed, batches)
    assert changed.drawing.regions[0].geometry == state.drawing.regions[0].geometry


def test_corner_observation_replay_keeps_geometry_and_exact_numeric_receipt(monkeypatch):
    from vectoreyes.drawing_automatic import replay_automatic_batches
    from vectoreyes.candidate import FlatFill
    from vectoreyes.fit import Shape
    from vectoreyes.refinement.region_sample import RegionSample
    from vectoreyes.refinement.passes.corners import corner_normalize_pass

    source = RegionSample.from_shape(id=1, drawing_id='r1',
        shape=Shape('path', {'d': 'M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z'}),
        fill=FlatFill('#112233'), z=0, raster=np.ones((16, 24), dtype=bool))
    state = capture_replay_state((source,), Canvas(24, 16))
    objects = list(working_regions_from_state(state))
    masks = {1: objects[0].raster}
    def configured(objects, masks):
        return corner_normalize_pass(objects, masks, max_error=1., enabled=True)
    recorded = select_refinements_with_intent(objects, masks, [configured])
    from vectoreyes.refinement.automatic_replay import AutomaticObservationRecord, CornerDiagnosticsObservation
    batch = recorded.batches[0]
    entry = batch.entries[0]
    observation = CornerDiagnosticsObservation(entry.intent.operation_id, entry.intent.action.target,
        {'skipped': [{'reason': 'missing_or_stale_provenance'}]})
    batches = (replace(batch, entries=(AutomaticObservationRecord(observation, entry.pass_name,
        entry.proposal_ids, entry.assessment),)),)
    expected = capture_replay_state(replay_batches(objects, batches, masks=masks), state.drawing.canvas)
    def forbidden(*args, **kwargs):
        raise AssertionError('observation used the working action executor')
    monkeypatch.setattr('vectoreyes.refinement.automatic_replay._execute_intent', forbidden)
    result = replay_automatic_batches(state, batches)
    assert result.drawing.region('r1').geometry == state.drawing.region('r1').geometry
    assert result.drawing.region('r1').diagnostics['corners']['skipped'][0]['reason'] == 'missing_or_stale_provenance'
    assert canonical_input_revision(result) == canonical_input_revision(expected)

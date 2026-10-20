from dataclasses import replace

import pytest

from tests.test_canonical_automatic_replay import primitive_case
from vectormark.candidate import FlatFill
from vectormark.drawing import Canvas
from vectormark.fit import Shape
from vectormark.refinement.automatic_replay import (
    AssignCloneOperation, AutomaticIntent, AutomaticOperationRecord,
    DeclareIntrinsicSymmetryOperation, NamedRegionOutput, OperationOutputRegionRef,
    replay_batches,
)
from vectormark.refinement.framework import select_refinements_with_intent
from vectormark.refinement.passes.clones import clones_pass
from vectormark.refinement.replay_binding import capture_replay_state, working_regions_from_state
from vectormark.refinement.region_sample import RegionSample
from vectormark.replay_identity import canonical_input_revision


def forbid_working_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('canonical transaction used working execution')
    monkeypatch.setattr('vectormark.refinement.automatic_replay._execute_intent', forbidden)
    monkeypatch.setattr('vectormark.refinement.replay_binding.working_regions_from_state', forbidden)


def test_symbolic_clone_publication_keeps_independent_paint_and_source_mask(monkeypatch):
    from vectormark.drawing_automatic import replay_automatic_batches
    from vectormark.refinement.gate import rasterize
    from vectormark.refinement.region_sample import to_polygon
    def square(identity, x, color):
        shape = Shape('path', {'d': f'M{x} 10 L{x + 12} 10 L{x + 12} 22 L{x} 22 Z'})
        return RegionSample.from_shape(id=identity, drawing_id=f'r{identity}', shape=shape,
            fill=FlatFill(color), z=0, raster=rasterize(to_polygon(shape), (48, 80)))
    state = capture_replay_state((square(1, 10, '#112233'), square(2, 50, '#abcdef')), Canvas(80, 48))
    objects = list(working_regions_from_state(state))
    masks = {obj.id: obj.raster for obj in objects}
    recorded = select_refinements_with_intent(objects, masks, [clones_pass])
    action = recorded.batches[0].operations[0].intent.action
    assert isinstance(action, AssignCloneOperation) and action.symbolic
    expected = capture_replay_state(replay_batches(objects, recorded.batches, masks=masks), state.drawing.canvas)
    forbid_working_execution(monkeypatch)
    actual = replay_automatic_batches(state, recorded.batches)
    assert actual.drawing == expected.drawing
    assert actual.drawing.region('r2').fill == state.drawing.region('r2').fill
    assert actual.region_evidence == expected.region_evidence
    assert canonical_input_revision(actual) == canonical_input_revision(expected)


def test_named_output_reads_the_producer_not_the_pass_snapshot(monkeypatch):
    from vectormark.drawing_automatic import replay_automatic_batches
    state, batches, _ = primitive_case()
    batch = batches[0]
    producer = batch.operations[0]
    reference = OperationOutputRegionRef(producer.intent.operation_id, producer.intent.outputs[0].name)
    action = DeclareIntrinsicSymmetryOperation(reference, (0., 60., 60.), 0.)
    consumer = AutomaticOperationRecord(AutomaticIntent('consumer', action, (reference,),
        (NamedRegionOutput('result', producer.intent.outputs[0].ancestry),)), 'intrinsic', producer.proposal_ids, producer.assessment)
    batches = (replace(batch, entries=(producer, consumer)),)
    objects = list(working_regions_from_state(state))
    expected = capture_replay_state(replay_batches(objects, batches, masks={obj.id: obj.raster for obj in objects}), state.drawing.canvas)
    forbid_working_execution(monkeypatch)
    actual = replay_automatic_batches(state, batches)
    assert actual.drawing == expected.drawing
    assert actual.drawing.regions[0].geometry == producer.intent.action.primitive
    assert canonical_input_revision(actual) == canonical_input_revision(expected)


def test_named_output_forward_reference_is_rejected_before_publication():
    from vectormark.drawing_automatic import replay_automatic_batches
    state, batches, _ = primitive_case()
    producer = batches[0].operations[0]
    action = DeclareIntrinsicSymmetryOperation(OperationOutputRegionRef('future', 'result'), (0., 60., 60.), 0.)
    bad = replace(producer, intent=AutomaticIntent('consumer', action, (action.target,), ()))
    with pytest.raises(ValueError, match='forward-referenced'):
        replay_automatic_batches(state, (replace(batches[0], entries=(bad, producer)),))

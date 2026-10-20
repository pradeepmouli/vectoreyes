import numpy as np
import pytest

from tests.test_canonical_multi_replay import forbid_working_execution
from vectormark.candidate import FlatFill
from vectormark.drawing import Canvas
from vectormark.drawing_automatic import replay_automatic_batches
from vectormark.fit import Shape
from vectormark.refinement.automatic_replay import (
    AnnulusPrimitive, AssignCloneOperation, AutomaticIntent, AutomaticOperationRecord,
    CompletedOcclusionPrimitive, MirrorRegionOperation, NamedRegionOutput,
    PassInputRegionRef, ReconstructOcclusionOperation, ReplayBatch,
    SimplifyLineletOperation, bind_intent, replay_batches,
)
from vectormark.refinement.gate import rasterize
from vectormark.refinement.replay_binding import capture_replay_state, working_regions_from_state
from vectormark.refinement.region_sample import RegionSample, to_polygon
from vectormark.path_geometry import PathGeometryWithCutouts
from vectormark.quality import GeometryAssessment, GeometryPolicy
from vectormark.replay_identity import canonical_input_revision


def leaf(identity, shape, color):
    return RegionSample(identity, shape, FlatFill(color), drawing_id=f'r{identity}',
        raster=rasterize(to_polygon(shape), (440, 300)))


def committed(state, action, consumed, outputs):
    refs = tuple(PassInputRegionRef('', '', (identity,)) for identity in consumed)
    revision = canonical_input_revision(state)
    intent = bind_intent(AutomaticIntent('', action, refs,
        tuple(NamedRegionOutput(f'region_{identity}', (identity,)) for identity in outputs)),
        batch_id='transaction', input_revision=revision, ordinal=0)
    assessment = GeometryAssessment(GeometryPolicy(), True, 0, 0, 0, 1, 1, 0, 0, ())
    return ReplayBatch('transaction', revision, (), (
        AutomaticOperationRecord(intent, 'selected', consumed, assessment),))


def numeric_expected(state, batch):
    objects = list(working_regions_from_state(state))
    selected = replay_batches(objects, (batch,), masks={obj.id: obj.raster for obj in objects})
    return capture_replay_state(selected, state.drawing.canvas)


@pytest.mark.parametrize('kind', ['copy', 'mirror'])
def test_relationship_transaction_publishes_all_outputs_without_working_scene(monkeypatch, kind):
    source = leaf(1, Shape('path', {'d': 'M10 10 L22 10 L22 22 L10 22 Z'}), '#123456')
    target = leaf(2, Shape('path', {'d': 'M50 10 L62 10 L62 22 L50 22 Z'}), '#abcdef')
    state = capture_replay_state((source, target), Canvas(300, 440))
    refs = tuple(PassInputRegionRef('', '', (i,)) for i in (1, 2))
    action = (AssignCloneOperation(*refs, (1, 0, 0, 1, 40, 0), False) if kind == 'copy'
        else MirrorRegionOperation(*refs, (0., 36., 16.), (-1, 0, 0, 1, 72, 0)))
    consumed = (2,) if kind == 'copy' else (1, 2)
    batch = committed(state, action, consumed, consumed)
    expected = numeric_expected(state, batch)
    forbid_working_execution(monkeypatch)
    result = replay_automatic_batches(state, (batch,))
    assert result.drawing == expected.drawing
    assert result.region_evidence == expected.region_evidence
    assert result.drawing.region('r2').fill.color == '#abcdef'
    assert canonical_input_revision(result) == canonical_input_revision(expected)


def test_covering_simplification_retains_covering_geometry_and_observations(monkeypatch):
    d = ('M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 '
        'Q256.31 399.74 253.5 413 L247 417.5 Z')
    state = capture_replay_state((leaf(1, Shape('path', {'d': d}), '#123456'),
        leaf(2, Shape('rect', {'x': 0, 'y': 0, 'w': 1, 'h': 1}), '#abcdef')), Canvas(300, 440))
    action = SimplifyLineletOperation(PassInputRegionRef('', '', (1,)),
        (PassInputRegionRef('', '', (2,)),), 1., 1., 16, False, True, 1., True, True)
    batch = committed(state, action, (1, 2), (1, 2))
    expected = numeric_expected(state, batch)
    forbid_working_execution(monkeypatch)
    result = replay_automatic_batches(state, (batch,))
    assert result.drawing == expected.drawing
    assert result.drawing.region('r1').geometry != state.drawing.region('r1').geometry
    assert result.drawing.region('r2').geometry == state.drawing.region('r2').geometry
    assert result.region_evidence == expected.region_evidence
    assert canonical_input_revision(result) == canonical_input_revision(expected)


def test_occlusion_transaction_retires_sources_and_publishes_editable_child_observations(monkeypatch):
    state = capture_replay_state((leaf(1, Shape('circle', {'cx': 16, 'cy': 16, 'r': 10}), '#123456'),
        leaf(2, Shape('rect', {'x': 80, 'y': 0, 'w': 10, 'h': 10}), '#abcdef')),
        Canvas(300, 440))
    ref = PassInputRegionRef('', '', (1,))
    action = ReconstructOcclusionOperation((ref,), 1, 0., (440, 300), (
        CompletedOcclusionPrimitive(ref, 'ring', 3, AnnulusPrimitive((16, 16), 10, 5), 0),))
    batch = committed(state, action, (1,), (1,))
    expected = numeric_expected(state, batch)
    # A second receipt checks generated source observations, not just final paint.
    following = ReplayBatch('next', canonical_input_revision(expected), (), ())
    forbid_working_execution(monkeypatch)
    result = replay_automatic_batches(state, (batch, following))
    assert result.drawing == expected.drawing
    assert isinstance(result.drawing.region('r1-3').geometry, PathGeometryWithCutouts)
    assert result.region_evidence == expected.region_evidence
    assert result.drawing.region('r2') == state.drawing.region('r2')
    assert np.array_equal(np.frombuffer(result.region_evidence['r1']['raster'], bool),
        np.frombuffer(state.region_evidence['r1']['raster'], bool))


def test_generated_child_observations_keep_full_scope_with_repeated_local_ids(monkeypatch):
    from tests.test_canonical_seam_rebinding import nested_state
    state = nested_state()
    ref = PassInputRegionRef('', '', (1,))
    revision = canonical_input_revision(state, (10,))
    action = ReconstructOcclusionOperation((ref,), 1, 0., (16, 16), (
        CompletedOcclusionPrimitive(ref, 'ring', 3, AnnulusPrimitive((6, 6), 5, 2), 0),))
    producer = bind_intent(AutomaticIntent('', action, (ref,), (
        NamedRegionOutput('branch', (1,)),)), batch_id='scoped', input_revision=revision, ordinal=0)
    assessment = GeometryAssessment(GeometryPolicy(), True, 0, 0, 0, 1, 1, 0, 0, ())
    # The ordinary scoped batch keeps generated children nested; a later batch
    # edits those children in their own scope rather than lifting them to roots.
    batch = ReplayBatch('scoped', revision, (10,), (
        AutomaticOperationRecord(producer, 'selected', (1,), assessment),))
    objects = list(working_regions_from_state(state))
    expected = capture_replay_state(replay_batches(objects, (batch,),
        masks={obj.id: obj.raster for obj in objects}), state.drawing.canvas)
    forbid_working_execution(monkeypatch)
    result = replay_automatic_batches(state, (batch,))
    assert result.drawing == expected.drawing
    assert result.region_evidence == expected.region_evidence
    assert result.region_evidence['r10-1-3']['ancestry'] == (10, 1, 3)
    assert result.drawing.region('r20') == state.drawing.region('r20')
from dataclasses import replace



def test_generated_named_child_can_be_the_source_of_an_inline_copy(monkeypatch):
    from vectormark.refinement.automatic_replay import OperationOutputRegionRef
    state = capture_replay_state((leaf(1, Shape('circle', {'cx': 16, 'cy': 16, 'r': 10}), '#123456'),
        leaf(2, Shape('rect', {'x': 80, 'y': 0, 'w': 10, 'h': 10}), '#abcdef')),
        Canvas(300, 440))
    ref = PassInputRegionRef('', '', (1,))
    action = ReconstructOcclusionOperation((ref,), 1, 0., (440, 300), (
        CompletedOcclusionPrimitive(ref, 'ring', 3, AnnulusPrimitive((16, 16), 10, 5), 0),))
    batch = committed(state, action, (1,), (1,))
    producer = batch.operations[0]
    producer = replace(producer, intent=replace(producer.intent,
        outputs=(*producer.intent.outputs, NamedRegionOutput('ring', (1, 3)))))
    source = OperationOutputRegionRef(producer.intent.operation_id, 'ring')
    target = PassInputRegionRef(batch.batch_id, batch.input_revision, (2,))
    clone = AssignCloneOperation(source, target, (1, 0, 0, 1, 70, 0), False)
    intent = bind_intent(AutomaticIntent('', clone, (source, target), (NamedRegionOutput('copy', (2,)),)),
        batch_id=batch.batch_id, input_revision=batch.input_revision, ordinal=1)
    batch = replace(batch, entries=(producer, AutomaticOperationRecord(intent, 'copy', (2,), producer.assessment)))
    expected = numeric_expected(state, batch)
    forbid_working_execution(monkeypatch)
    result = replay_automatic_batches(state, (batch,))
    assert result.drawing == expected.drawing
    assert result.region_evidence == expected.region_evidence
    assert canonical_input_revision(result) == canonical_input_revision(expected)


def test_inline_copy_of_a_clone_keeps_the_instance_projection_policy(monkeypatch):
    source = leaf(1, Shape('circle', {'cx': 16, 'cy': 16, 'r': 10}), '#123456')
    matrix = (1, 0, 0, 1, 40, 0)
    footprint = source.footprint.affine_transform(matrix)
    instance = RegionSample(2, Shape('use', {'href_obj_id': 1, 'transform': matrix}),
        FlatFill('#445566'), drawing_id='r2', footprint=footprint,
        raster=rasterize(footprint, (440, 300)))
    target = leaf(3, Shape('rect', {'x': 90, 'y': 0, 'w': 10, 'h': 10}), '#abcdef')
    state = capture_replay_state((source, instance, target), Canvas(300, 440))
    action = AssignCloneOperation(PassInputRegionRef('', '', (2,)),
        PassInputRegionRef('', '', (3,)), matrix, False)
    batch = committed(state, action, (3,), (3,))
    expected = numeric_expected(state, batch)
    forbid_working_execution(monkeypatch)
    result = replay_automatic_batches(state, (batch,))
    assert result.drawing == expected.drawing
    assert result.region_evidence == expected.region_evidence
    assert canonical_input_revision(result) == canonical_input_revision(expected)


def test_later_pass_input_replacement_restores_its_snapshot_observations(monkeypatch):
    from vectormark.refinement.automatic_replay import FitPrimitiveOperation
    from vectormark.path_geometry import CirclePrimitive, PrimitiveGeometry
    state = capture_replay_state((leaf(1, Shape('circle', {'cx': 16, 'cy': 16, 'r': 10}), '#123456'),), Canvas(300, 440))
    ref = PassInputRegionRef('', '', (1,))
    branch = ReconstructOcclusionOperation((ref,), 1, 0., (440, 300), (
        CompletedOcclusionPrimitive(ref, 'ring', 3, AnnulusPrimitive((16, 16), 10, 5), 0),))
    batch = committed(state, branch, (1,), (1,))
    selected = FitPrimitiveOperation(ref, PrimitiveGeometry(CirclePrimitive((16, 16), 10)), 1.)
    intent = bind_intent(AutomaticIntent('', selected, (ref,), (NamedRegionOutput('result', (1,)),)),
        batch_id=batch.batch_id, input_revision=batch.input_revision, ordinal=1)
    batch = replace(batch, entries=(*batch.entries,
        AutomaticOperationRecord(intent, 'primitive', (1,), batch.operations[0].assessment)))
    expected = numeric_expected(state, batch)
    forbid_working_execution(monkeypatch)
    result = replay_automatic_batches(state, (batch,))
    assert result.drawing == expected.drawing
    assert result.region_evidence == expected.region_evidence
    assert canonical_input_revision(result) == canonical_input_revision(expected)

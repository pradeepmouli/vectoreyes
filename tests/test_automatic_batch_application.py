from dataclasses import replace

import pytest

from tests.test_canonical_structural_replay import committed, leaf
from tests.test_canonical_multi_replay import forbid_working_execution
from vectormark.drawing import Canvas
from vectormark.fit import Shape
from vectormark.path_geometry import PrimitiveGeometry, RectanglePrimitive
from vectormark.refinement.automatic_replay import FitPrimitiveOperation, PassInputRegionRef
from vectormark.refinement.replay_binding import capture_replay_state
from vectormark.replay_identity import canonical_input_revision


def fixture():
    state = capture_replay_state((leaf(1, Shape('rect', {'x': 0, 'y': 0, 'w': 4, 'h': 6}), '#123456'),), Canvas(300, 440))
    action = FitPrimitiveOperation(PassInputRegionRef('', '', (1,)),
        PrimitiveGeometry(RectanglePrimitive((0, 0), 5, 6)), 1)
    return state, committed(state, action, (1,), (1,))


def test_prepared_canonical_choices_do_not_mutate_their_pass_snapshot(monkeypatch):
    from vectormark.drawing_automatic import AutomaticBatchApplication
    state, batch = fixture()
    revision = canonical_input_revision(state)
    forbid_working_execution(monkeypatch)
    context = AutomaticBatchApplication(state, batch)
    selected = context.prepare(batch.entries[0].intent)
    assert selected.regions[0].region.geometry.primitive.width == 5
    assert context.finish().drawing == state.drawing
    assert canonical_input_revision(context.finish()) == revision
    assert canonical_input_revision(state) == revision


def test_accept_installs_the_prepared_selection_without_executing_it_again(monkeypatch):
    from vectormark.drawing_automatic import AutomaticBatchApplication
    state, batch = fixture()
    context = AutomaticBatchApplication(state, batch)
    selected = context.prepare(batch.entries[0].intent)
    def unavailable(*args, **kwargs):
        raise AssertionError('selection execution must not repeat at acceptance')
    monkeypatch.setattr('vectormark.drawing_automatic._select_regions', unavailable)
    context.accept(batch.entries[0], selected)
    result = context.finish()
    assert result.drawing.region('r1').geometry.primitive.width == 5
    assert result.region_evidence['r1']['original_params']['w'] == 4
    assert state.drawing.region('r1').geometry.primitive.width == 4


def test_rejected_assessment_cannot_publish_a_prepared_selection():
    from vectormark.drawing_automatic import AutomaticBatchApplication
    state, batch = fixture()
    context = AutomaticBatchApplication(state, batch)
    record = batch.entries[0]
    selected = context.prepare(record.intent)
    invalid = replace(record, assessment=replace(record.assessment, accepted=False, reasons=('bad',)))
    with pytest.raises(ValueError, match='accepted assessment'):
        context.accept(invalid, selected)
    assert canonical_input_revision(context.finish()) == canonical_input_revision(state)


def test_prepared_outputs_cannot_be_attached_to_different_intent():
    from vectormark.drawing_automatic import AutomaticBatchApplication
    state, batch = fixture()
    context = AutomaticBatchApplication(state, batch)
    record = batch.entries[0]
    selected = context.prepare(record.intent)
    other = replace(record, intent=replace(record.intent, action=replace(record.intent.action,
        primitive=PrimitiveGeometry(RectanglePrimitive((0, 0), 6, 6)))))
    with pytest.raises(ValueError, match='prepared selection binding'):
        context.accept(other, selected)
    assert canonical_input_revision(context.finish()) == canonical_input_revision(state)

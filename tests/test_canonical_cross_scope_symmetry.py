from dataclasses import replace

import pytest

from vectormark.drawing import Canvas
from vectormark.drawing_automatic import replay_automatic_batches
from vectormark.drawing_proposals import detect_symmetry_regions, discover_symmetry_proposals, select_canonical_pass
from vectormark.fit import Shape
from vectormark.path_geometry import CloneGeometry
from vectormark.refinement.region_sample import RegionSample
from vectormark.refinement.replay_binding import capture_replay_state
from tests.test_canonical_structural_replay import leaf
from tests.test_canonical_symmetry_discovery import ellipse_path


def forest():
    source = leaf(11, Shape('path', {'d': ellipse_path().to_svg_d()}), '#112233')
    parent = RegionSample.branch(id=10, drawing_id='parent', children=(source,))
    from vectormark.drawing_fill import FlatFill
    from vectormark.refinement.gate import rasterize
    footprint = ellipse_path().affine_transform((1, 0, 0, 1, 100, 0))
    clone = RegionSample(2, Shape('use', {'href_obj_id': 11, 'href_region_path': (10, 11),
        'transform': (1, 0, 0, 1, 100, 0)}), FlatFill('#abcdef'), drawing_id='r2',
        footprint=footprint, raster=rasterize(footprint, (440, 300)))
    return capture_replay_state((parent, clone), Canvas(300, 440))


def test_nested_source_and_external_clone_publish_as_one_root_transaction(monkeypatch):
    state = forest()
    def forbidden(*args, **kwargs):
        raise AssertionError('cross-scope transaction projected a sampling scene')
    monkeypatch.setattr('vectormark.refinement.region_sample.RegionSample.__init__', forbidden)
    result = detect_symmetry_regions(state)
    records = [record for batch in result.batches for record in batch.operations
        if getattr(record.intent.action, 'owner_roots', ())]
    assert len(records) == 1
    record = records[0]
    assert record.proposal_ids == (2, 10)
    assert record.intent.action.target.ancestry == (10, 11)
    assert {ref.ancestry for ref in record.intent.action.owner_roots} == {(10,), (2,)}
    assert {identity for identity, _ in record.assessment.region_assessments} == {
        state.drawing.region('parent').children[0].id, 'r2'}
    assert result.state.drawing.region('parent').children[0].diagnostics['symmetry']['materialized']
    instance = result.state.drawing.region('r2')
    assert isinstance(instance.geometry, CloneGeometry)
    assert instance.geometry.transforms == state.drawing.region('r2').geometry.transforms
    assert instance.fill == state.drawing.region('r2').fill
    monkeypatch.setattr('vectormark.symmetry_detection._best_self_reconstruction', forbidden)
    replayed = replay_automatic_batches(state, result.batches)
    assert replayed.drawing == result.state.drawing
    assert replayed.region_evidence == result.state.region_evidence


def test_rejected_cross_scope_fit_changes_neither_owner_nor_clone():
    from vectormark.quality import GeometryPolicy
    state = forest()
    proposals = tuple(proposal for proposal in discover_symmetry_proposals(state)
        if getattr(proposal.intent.action, 'owner_roots', ()))
    assert proposals
    strict = GeometryPolicy(max_boundary_error_px=0, max_perimeter_residual_error_px=0,
        max_relative_area_error=0, preserve_topology=True)
    result = select_canonical_pass(state, tuple(replace(proposal, geometry_policy=strict)
        for proposal in proposals), pass_name='symmetry_pass')
    assert result.outcomes and all(outcome.status == 'rejected' for outcome in result.outcomes)
    assert not result.batch.operations
    assert result.state.drawing == state.drawing


@pytest.mark.parametrize('omitted', ['clones', 'owner'])
def test_cross_scope_receipt_cannot_omit_a_dependent_owner(omitted):
    state = forest()
    result = detect_symmetry_regions(state)
    batch = next(batch for batch in result.batches if any(
        getattr(record.intent.action, 'owner_roots', ()) for record in batch.operations))
    record = next(record for record in batch.operations if getattr(record.intent.action, 'owner_roots', ()))
    action = record.intent.action
    action = (replace(action, dependent_clones=()) if omitted == 'clones' else
        replace(action, owner_roots=tuple(ref for ref in action.owner_roots if ref.ancestry != (2,))))
    invalid = replace(record, intent=replace(record.intent, action=action))
    with pytest.raises(ValueError, match='dependent'):
        replay_automatic_batches(state, tuple(replace(item, entries=(invalid,))
            if item is batch else item for item in result.batches))


def test_cross_scope_root_id_cannot_shadow_the_nested_source():
    from vectormark.drawing import DrawingRegion
    from vectormark.path_geometry import geometry_from_svg_d
    state = forest()
    unrelated = DrawingRegion('r11', geometry_from_svg_d('M200 0 L210 0 L210 10 L200 10 Z', region_id='r11'))
    info = {**state.region_evidence['r2'], 'id': 11, 'ancestry': (11,)}
    state = replace(state, drawing=replace(state.drawing, regions=(*state.drawing.regions, unrelated)),
        region_evidence={**state.region_evidence, 'r11': info})
    proposals = tuple(proposal for proposal in discover_symmetry_proposals(state)
        if getattr(proposal.intent.action, 'owner_roots', ()))
    assert len(proposals) == 1 and set(proposals[0].proposal_ids) == {2, 10}
    result = select_canonical_pass(state, proposals, pass_name='symmetry_pass')
    assert result.batch.operations
    assert result.state.drawing.region('r11') == unrelated
    assert result.state.drawing.region('r2').geometry.source_region_id == state.drawing.region('parent').children[0].id
    assert replay_automatic_batches(state, (result.batch,)).drawing == result.state.drawing

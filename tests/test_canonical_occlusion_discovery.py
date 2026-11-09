from dataclasses import replace

import numpy as np
import pytest

from tests.refinement.test_pass_occlusion import _synthetic_mastercard
from vectoreyes.drawing import Canvas
from vectoreyes.drawing_automatic import replay_automatic_batches
from vectoreyes.pipeline import Options
from vectoreyes.refinement.replay_binding import capture_replay_state
from vectoreyes.refinement.trace import trace_regions
from vectoreyes.replay_identity import canonical_input_revision


def fixture():
    regions, masks = trace_regions(_synthetic_mastercard(), Options())
    regions = [replace(region, drawing_id=f'r{region.id}') for region in regions]
    return regions, masks, capture_replay_state(regions, Canvas(360, 240))


@pytest.mark.parametrize('no_symmetry', [False, True])
def test_occlusion_discovery_uses_retained_evidence_without_sampling_scene(monkeypatch, no_symmetry):
    from vectoreyes.drawing_proposals import detect_occlusion_regions
    regions, _masks, state = fixture()
    before = canonical_input_revision(state)
    def forbidden(*args, **kwargs):
        raise AssertionError('occlusion discovery projected a sampling scene')
    monkeypatch.setattr('vectoreyes.refinement.region_sample.RegionSample.__init__', forbidden)
    monkeypatch.setattr('vectoreyes.drawing_adapter.bind_region_inputs', forbidden)
    result = detect_occlusion_regions(state, no_symmetry=no_symmetry)
    assert len(result.state.drawing.regions) == 1
    branch = result.state.drawing.regions[0]
    assert len(branch.children) == 3
    record, = result.batches[0].operations
    assert record.proposal_ids == tuple(sorted(region.id for region in regions))
    assert len(record.intent.action.children) == 3
    assert canonical_input_revision(state) == before
    monkeypatch.setattr('vectoreyes.occlusion_detection.reconstruct_scene', forbidden)
    replayed = replay_automatic_batches(state, result.batches)
    assert canonical_input_revision(replayed) == canonical_input_revision(result.state)
    assert replayed.region_evidence == result.state.region_evidence


def test_occlusion_discovery_preserves_numeric_selection_and_layering():
    from vectoreyes.drawing_proposals import detect_occlusion_regions
    from vectoreyes.refinement.automatic_replay import select_refinements_with_intent
    from vectoreyes.refinement.passes.occlusion import occlusion_pass
    regions, masks, state = fixture()
    expected = select_refinements_with_intent(regions, masks, [occlusion_pass])
    expected_state = capture_replay_state(expected.objects, state.drawing.canvas)
    actual = detect_occlusion_regions(state).state
    assert actual.drawing == expected_state.drawing
    assert actual.region_evidence == expected_state.region_evidence


def test_occlusion_proposals_can_be_targeted_at_any_consumed_material():
    from vectoreyes.drawing_proposals import detect_occlusion_regions
    _regions, _masks, state = fixture()
    for region in state.drawing.regions:
        result = detect_occlusion_regions(state, target_id=region.id)
        assert len(result.state.drawing.regions) == 1
        assert result.batches[0].operations


def test_single_region_does_not_propose_occlusion():
    from vectoreyes.drawing_proposals import detect_occlusion_regions
    _regions, _masks, state = fixture()
    drawing = replace(state.drawing, regions=state.drawing.regions[:1])
    result = detect_occlusion_regions(replace(state, drawing=drawing))
    assert not result.batches[0].operations
    assert result.state.drawing.regions == drawing.regions


@pytest.mark.parametrize('tiled', [False, True])
def test_retained_coverage_decoding_keeps_coordinates_and_read_only_values(tiled):
    from vectoreyes.replay_state import decode_region_coverage
    from vectoreyes.types import CoverageField
    values = np.arange(6, dtype=float).reshape(2, 3) / 6
    info = {'coverage': values.tobytes(), 'coverage_dtype': values.dtype.str,
        'coverage_shape': values.shape, 'coverage_origin': (7, 8) if tiled else None,
        'coverage_canvas': (32, 32) if tiled else None}
    result = decode_region_coverage(info)
    if tiled:
        assert isinstance(result, CoverageField)
        assert result.origin == (7, 8) and result.canvas_shape == (32, 32)
        result = result.values
    assert np.array_equal(result, values)
    assert not result.flags.writeable
    assert decode_region_coverage({}) is None


def test_occlusion_numeric_inputs_are_detached_before_discovery(monkeypatch):
    from vectoreyes.occlusion_detection import OcclusionDetectionInput, discover_occlusion_intents
    raster = np.ones((24, 24), dtype=bool)
    coverage = np.ones((24, 24), dtype=float)
    def mutate(regions, axis, shape_hw):
        assert axis is None
        for region in regions:
            region.mask[:] = False
            region.coverage[:] = 0
        return [], regions
    monkeypatch.setattr('vectoreyes.occlusion_detection.reconstruct_scene', mutate)
    monkeypatch.setattr('vectoreyes.occlusion_detection.detect_axis',
        lambda *_: pytest.fail('no_symmetry must not run axis detection'))
    assert not discover_occlusion_intents(tuple(OcclusionDetectionInput(i, raster, '#112233', 0, coverage)
        for i in (1, 2)), shape_hw=raster.shape, next_id=3, no_symmetry=True)
    assert raster.all() and coverage.all()


def test_occlusion_discovery_preserves_scoped_identity_and_replay():
    from vectoreyes.drawing_proposals import detect_occlusion_regions
    from vectoreyes.refinement.region_sample import RegionSample
    regions, _masks, _state = fixture()
    state = capture_replay_state(tuple(RegionSample.branch(id=i, drawing_id=f'parent{i}', children=regions)
        for i in (10, 20)), Canvas(360, 240))
    result = detect_occlusion_regions(state)
    assert [batch.scope for batch in result.batches] == [(10,), (20,), ()]
    assert [len(batch.operations) for batch in result.batches] == [1, 1, 0]
    assert all(len(parent.children) == 1 and len(parent.children[0].children) == 3
        for parent in result.state.drawing.regions)
    assert canonical_input_revision(replay_automatic_batches(state, result.batches)) == canonical_input_revision(result.state)


def test_occlusion_selection_rejects_excess_error_without_publishing():
    from vectoreyes.drawing_proposals import discover_occlusion_proposals, select_canonical_pass
    from vectoreyes.quality import GeometryPolicy
    _regions, _masks, state = fixture()
    proposals = tuple(replace(item, geometry_policy=GeometryPolicy(max_boundary_error_px=0))
        for item in discover_occlusion_proposals(state))
    assert proposals
    result = select_canonical_pass(state, proposals, pass_name='occlusion')
    assert not result.batch.operations
    assert all(item.status == 'rejected' for item in result.outcomes)
    assert canonical_input_revision(result.state) == canonical_input_revision(state)


def test_occlusion_cannot_retire_a_live_clone_source():
    from vectoreyes.drawing_proposals import discover_occlusion_proposals
    from vectoreyes.path_geometry import CloneGeometry, AffineTransform
    _regions, _masks, state = fixture()
    source = state.drawing.regions[0]
    clone = replace(state.drawing.regions[1], geometry=CloneGeometry(source.id, (AffineTransform(),)))
    drawing = replace(state.drawing, regions=(source, clone, *state.drawing.regions[2:]))
    assert not discover_occlusion_proposals(replace(state, drawing=drawing))


def test_shared_discovery_keeps_annulus_and_cavity_source_dependencies():
    from vectoreyes.occlusion_detection import OcclusionDetectionInput, discover_occlusion_intents
    from vectoreyes.refinement.automatic_replay import AnnulusPrimitive, CompletedOcclusionPrimitive, RetainedOcclusionSource
    yy, xx = np.ogrid[:160, :200]
    d2 = (xx-70)**2 + (yy-70)**2
    occluder = (xx-135)**2 + (yy-70)**2 <= 38**2
    masks = (((d2 <= 45**2) & (d2 >= 25**2)) & ~occluder, occluder, d2 < 25**2)
    inputs = tuple(OcclusionDetectionInput(i, mask, color, 5+i)
        for i, mask, color in zip((1, 2, 3), masks, ('#3366cc', '#cc3333', '#22aa66'), strict=True))
    intent, = discover_occlusion_intents(inputs, shape_hw=(160, 200), next_id=4, no_symmetry=True)
    children = intent.action.children
    assert any(isinstance(child, CompletedOcclusionPrimitive) and isinstance(child.primitive, AnnulusPrimitive)
        for child in children)
    retained, = (child for child in children if isinstance(child, RetainedOcclusionSource))
    assert retained.source.ancestry == (3,)
    assert intent.action.base_z == 6

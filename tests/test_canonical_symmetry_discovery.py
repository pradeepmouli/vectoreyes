from dataclasses import replace

import numpy as np
import pytest

from tests.test_canonical_structural_replay import leaf
from vectormark.drawing import Canvas
from vectormark.drawing_automatic import replay_automatic_batches
from vectormark.fit import Shape
from vectormark.geometry_native import region_path
from vectormark.refinement.replay_binding import capture_replay_state
from vectormark.replay_identity import canonical_input_revision
from vectormark.skia_geometry import SkPath


def ellipse_path():
    return SkPath(shell=[(48 + 20*np.cos(t), 48 + 30*np.sin(t))
        for t in np.linspace(0, 2*np.pi, 32, endpoint=False)])


def state_for(*regions):
    return capture_replay_state(regions, Canvas(300, 440))


def forbid_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('symmetry discovery projected a sampling scene')
    monkeypatch.setattr('vectormark.refinement.region_sample.RegionSample.__init__', forbidden)
    monkeypatch.setattr('vectormark.drawing_adapter.bind_region_inputs', forbidden)


@pytest.mark.parametrize('target', [None, 'r1', 'r2'])
def test_mirror_discovery_preserves_independent_paint_and_replays(monkeypatch, target):
    from vectormark.drawing_proposals import detect_symmetry_regions
    source = SkPath(shell=[(10, 10), (20, 10), (15, 25)])
    mirrored = source.affine_transform((-1, 0, 0, 1, 80, 0))
    state = state_for(leaf(1, Shape('path', {'d': source.to_svg_d()}), '#112233'),
        leaf(2, Shape('path', {'d': mirrored.to_svg_d()}), '#abcdef'))
    forbid_sampling(monkeypatch)
    result = detect_symmetry_regions(state, target_id=target)
    record, = result.batches[0].operations
    from vectormark.refinement.automatic_replay import MirrorRegionOperation
    assert isinstance(record.intent.action, MirrorRegionOperation)
    selected = result.state.drawing.region('r2')
    assert selected.fill == state.drawing.region('r2').fill
    indexed = {r.id: r for r in result.state.drawing.regions}
    assert region_path(selected, indexed).symmetric_difference(mirrored).area < 1e-8
    replayed = replay_automatic_batches(state, result.batches)
    assert canonical_input_revision(replayed) == canonical_input_revision(result.state)


def test_primitive_symmetry_is_an_intrinsic_declaration(monkeypatch):
    from vectormark.drawing_proposals import detect_symmetry_regions
    state = state_for(leaf(1, Shape('rect', {'x': 20, 'y': 20, 'w': 40, 'h': 50}), '#112233'))
    forbid_sampling(monkeypatch)
    result = detect_symmetry_regions(state)
    assert result.state.drawing.region('r1').geometry == state.drawing.region('r1').geometry
    assert result.state.drawing.region('r1').diagnostics['symmetry']['mode'] == 'intrinsic'
    assert len(result.batches[0].operations) == 1


def test_self_symmetry_records_selected_fit_without_sampling(monkeypatch):
    from vectormark.drawing_proposals import detect_symmetry_regions
    state = state_for(leaf(1, Shape('path', {'d': ellipse_path().to_svg_d()}), '#112233'))
    forbid_sampling(monkeypatch)
    result = detect_symmetry_regions(state)
    selected = result.state.drawing.region('r1')
    assert selected.diagnostics['symmetry']['mode'] == 'self'
    assert selected.diagnostics['symmetry']['materialized'] is True
    def forbidden(*args, **kwargs):
        raise AssertionError('replay redetected the symmetry winner')
    monkeypatch.setattr('vectormark.symmetry_detection._best_self_reconstruction', forbidden)
    assert canonical_input_revision(replay_automatic_batches(state, result.batches)) == canonical_input_revision(result.state)


def test_material_partition_is_not_reconstructed_as_a_semantic_shape(monkeypatch):
    from vectormark.drawing_proposals import detect_symmetry_regions
    region = leaf(1, Shape('path', {'d': ellipse_path().to_svg_d()}), '#112233')
    region = replace(region, diagnostics={'material_surface': {'root': 9}})
    state = state_for(region)
    forbid_sampling(monkeypatch)
    result = detect_symmetry_regions(state)
    assert not result.batches[0].operations
    assert result.state.drawing.region('r1').geometry == state.drawing.region('r1').geometry


def test_public_symmetry_detection_does_not_project_sampling_scene(monkeypatch):
    from tests.test_drawing_recipe_actions import _direct_recipe_fixture
    from vectormark.drawing_plan import DrawingPlan
    from vectormark.drawing_replay import extend_recipe_with_plan, propose_effective_recipe
    regions = (leaf(1, Shape('path', {'d': ellipse_path().to_svg_d()}), '#112233'),)
    evidence, paint, base = _direct_recipe_fixture(regions, width=300, height=440)
    plan = DrawingPlan.model_validate({'version': 'vectormark.plan.v1', 'drawing_id': 'drw_symmetry',
        'base_version': 'v0', 'ops': [{'op': 'detect_symmetry', 'target': 'r1'}]})
    def forbidden(*args, **kwargs):
        raise AssertionError('public symmetry discovery projected a sampling scene')
    monkeypatch.setattr('vectormark.refinement.replay_binding.working_regions_from_state', forbidden)
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    assert sum(len(batch.operations) for batch in recipe.automatic_batches) == 1
    result = propose_effective_recipe(drawing_id='drw_symmetry', evidence=evidence, paint=paint, recipe=recipe)
    assert result.drawing.region('r1').diagnostics['symmetry']['materialized'] is True


def test_non_path_polygon_preserves_existing_discovery_eligibility():
    from vectormark.drawing_proposals import detect_symmetry_regions
    state = state_for(leaf(1, Shape('polygon', {'points': [(20, 20), (40, 20), (30, 40)]}), '#112233'))
    assert not detect_symmetry_regions(state).batches[0].operations


def test_local_clone_is_assessed_with_its_reconstructed_source(monkeypatch):
    from vectormark.drawing_proposals import detect_clone_regions, detect_symmetry_regions
    source = ellipse_path()
    state = state_for(*(leaf(i, Shape('path', {'d': source.affine_transform((1, 0, 0, 1, x, 0)).to_svg_d()}), color)
        for i, x, color in ((1, 0, '#112233'), (2, 100, '#abcdef'))))
    cloned = detect_clone_regions(state).state
    forbid_sampling(monkeypatch)
    result = detect_symmetry_regions(cloned)
    record, = result.batches[0].operations
    assert record.proposal_ids == (1, 2)
    assert len(record.intent.action.dependent_clones) == 1
    assert result.state.drawing.region('r2').fill == cloned.drawing.region('r2').fill
    assert canonical_input_revision(replay_automatic_batches(cloned, result.batches)) == canonical_input_revision(result.state)


def test_scoped_symmetry_keeps_repeated_local_ids_separate():
    from vectormark.drawing_proposals import detect_symmetry_regions
    from vectormark.refinement.region_sample import RegionSample
    source = leaf(1, Shape('path', {'d': ellipse_path().to_svg_d()}), '#112233')
    state = state_for(*(RegionSample.branch(id=i, drawing_id=f'parent{i}', children=(source,)) for i in (10, 20)))
    result = detect_symmetry_regions(state)
    assert [batch.scope for batch in result.batches] == [(10,), (20,), ()]
    assert [len(batch.operations) for batch in result.batches] == [1, 1, 0]
    assert canonical_input_revision(replay_automatic_batches(state, result.batches)) == canonical_input_revision(result.state)


def test_cross_scope_clone_source_is_not_locally_reconstructed(monkeypatch):
    from vectormark.drawing_proposals import detect_symmetry_regions, discover_symmetry_proposals
    from vectormark.candidate import FlatFill
    from vectormark.refinement.region_sample import RegionSample
    from vectormark.refinement.gate import rasterize
    source = leaf(11, Shape('path', {'d': ellipse_path().to_svg_d()}), '#112233')
    branch = RegionSample.branch(id=10, drawing_id='parent', children=(source,))
    transformed = ellipse_path().affine_transform((1, 0, 0, 1, 100, 0))
    clone = RegionSample(2, Shape('use', {'href_obj_id': 11, 'transform': (1, 0, 0, 1, 100, 0)}),
        FlatFill('#abcdef'), drawing_id='r2', footprint=transformed,
        raster=rasterize(transformed, (440, 300)), diagnostics={'clones': {'matched_source': 11}})
    state = state_for(branch, clone)
    forbid_sampling(monkeypatch)
    local = discover_symmetry_proposals(state, scope=(10,))
    assert all(not getattr(proposal.intent.action, 'dependent_clones', ()) for proposal in local)
    result = detect_symmetry_regions(state)
    after = result.state.drawing.region('parent').children[0]
    assert after.diagnostics['symmetry']['materialized']
    assert any(getattr(record.intent.action, 'owner_roots', ())
        for batch in result.batches if not batch.scope for record in batch.operations)
    assert canonical_input_revision(replay_automatic_batches(state, result.batches)) == canonical_input_revision(result.state)


@pytest.mark.parametrize('shape', [
    Shape('path', {'d': ellipse_path().to_svg_d()}),
    Shape('rect', {'x': 20, 'y': 20, 'w': 40, 'h': 50}),
    Shape('polygon', {'points': [(20, 20), (40, 20), (30, 40)]}),
])
def test_canonical_symmetry_matches_existing_numeric_selection(shape):
    from vectormark.drawing_proposals import detect_symmetry_regions
    from vectormark.refinement.passes.symmetry import symmetry_pass
    from vectormark.refinement.automatic_replay import select_refinements_with_intent
    source = leaf(1, shape, '#112233')
    state = state_for(source)
    expected = select_refinements_with_intent([source], {1: source.raster}, [symmetry_pass])
    expected_state = state_for(*expected.objects)
    actual = detect_symmetry_regions(state).state
    assert actual.drawing.region('r1').geometry == expected_state.drawing.region('r1').geometry
    assert actual.drawing.region('r1').diagnostics.get('symmetry') == expected_state.drawing.region('r1').diagnostics.get('symmetry')

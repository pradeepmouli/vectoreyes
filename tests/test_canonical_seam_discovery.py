from dataclasses import replace

import pytest

from tests.test_canonical_structural_replay import leaf
from vectoreyes.drawing import Canvas
from vectoreyes.drawing_automatic import replay_automatic_batches
from vectoreyes.fit import Shape
from vectoreyes.geometry_native import region_path
from vectoreyes.refinement.replay_binding import capture_replay_state
from vectoreyes.replay_identity import canonical_input_revision


def regions():
    return (leaf(1, Shape('path', {'d': 'M10 10 L49.2 10 L49.2 70 L10 70 Z'}), '#112233'),
        leaf(2, Shape('path', {'d': 'M50.4 10 L90 10 L90 70 L50.4 70 Z'}), '#abcdef'))


def fixture():
    return capture_replay_state(regions(), Canvas(300, 440))


@pytest.mark.parametrize('allow_underlap', [False, True])
def test_canonical_seam_discovery_needs_no_sampling_scene_and_replays(monkeypatch, allow_underlap):
    from vectoreyes.drawing_proposals import detect_seam_regions
    state = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError('seam discovery projected a sampling scene')
    monkeypatch.setattr('vectoreyes.refinement.region_sample.RegionSample.__init__', forbidden)
    monkeypatch.setattr('vectoreyes.drawing_adapter.bind_region_inputs', forbidden)
    result = detect_seam_regions(state, allow_underlap=allow_underlap)
    assert len(result.batches[0].operations) == 1
    indexed = {item.id: item for item in result.state.drawing.regions}
    assert region_path(indexed['r1'], indexed).bounds[2] == pytest.approx(49.8)
    assert region_path(indexed['r2'], indexed).bounds[0] == pytest.approx(49.8)
    assert indexed['r2'].fill == state.drawing.region('r2').fill
    monkeypatch.setattr('vectoreyes.seam_detection._candidate_for_pair', forbidden)
    assert canonical_input_revision(replay_automatic_batches(state, result.batches)) == canonical_input_revision(result.state)


def test_seam_discovery_matches_existing_numeric_selection():
    from vectoreyes.drawing_proposals import detect_seam_regions
    from vectoreyes.refinement.automatic_replay import select_refinements_with_intent
    from vectoreyes.refinement.passes.seams import seams_pass
    source = regions()
    state = capture_replay_state(source, Canvas(300, 440))
    expected = select_refinements_with_intent(list(source), {item.id: item.raster for item in source}, [seams_pass])
    expected_state = capture_replay_state(expected.objects, state.drawing.canvas)
    actual = detect_seam_regions(state).state
    assert actual.drawing == expected_state.drawing
    assert actual.region_evidence == expected_state.region_evidence


def test_distinct_material_groups_do_not_get_stitched():
    from vectoreyes.drawing_proposals import detect_seam_regions
    source = tuple(replace(item, diagnostics={'surface_fill': {'group': item.id}}) for item in regions())
    state = capture_replay_state(source, Canvas(300, 440))
    assert not detect_seam_regions(state).batches[0].operations


def test_public_stitch_plan_does_not_project_a_sampling_scene(monkeypatch):
    from tests.test_drawing_recipe_actions import _direct_recipe_fixture
    from vectoreyes.drawing_plan import DrawingPlan
    from vectoreyes.drawing_replay import extend_recipe_with_plan, propose_effective_recipe
    evidence, paint, base = _direct_recipe_fixture(regions(), width=300, height=440)
    plan = DrawingPlan.model_validate({'version': 'vectoreyes.plan.v1', 'drawing_id': 'drw_seams',
        'base_version': 'v0', 'ops': [{'op': 'stitch', 'targets': ['r1', 'r2']}]})
    def forbidden(*args, **kwargs):
        raise AssertionError('public stitch projected a sampling scene')
    monkeypatch.setattr('vectoreyes.refinement.replay_binding.working_regions_from_state', forbidden)
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    assert sum(len(batch.operations) for batch in recipe.automatic_batches) == 1
    result = propose_effective_recipe(drawing_id='drw_seams', evidence=evidence, paint=paint, recipe=recipe)
    assert result.drawing.region('r1').diagnostics['seams']['accepted'] is True


@pytest.mark.parametrize('selected', [('r1',), ('r2',), ('r1', 'r2')])
def test_stitch_can_target_either_side_or_both(selected):
    from vectoreyes.drawing_proposals import detect_seam_regions
    assert detect_seam_regions(fixture(), target_ids=selected).batches[0].operations


def test_post_symmetry_mode_does_not_revisit_unrelated_roots():
    from vectoreyes.drawing_proposals import detect_seam_regions
    state = fixture()
    result = detect_seam_regions(state, post_symmetry=True)
    assert not any(batch.operations for batch in result.batches)
    assert canonical_input_revision(result.state) == canonical_input_revision(state)


def test_nested_repeated_local_ids_keep_distinct_seam_updates():
    from vectoreyes.drawing_proposals import detect_seam_regions
    from vectoreyes.refinement.region_sample import RegionSample
    upper = regions()
    lower = (leaf(1, Shape('path', {'d': 'M10 100 L49.2 100 L49.2 160 L10 160 Z'}), '#112233'),
        leaf(2, Shape('path', {'d': 'M50.4 100 L90 100 L90 160 L50.4 160 Z'}), '#abcdef'))
    parent = RegionSample.branch(id=10, drawing_id='parent', children=(
        RegionSample.branch(id=20, children=upper), RegionSample.branch(id=30, children=lower)))
    state = capture_replay_state((parent,), Canvas(300, 440))
    result = detect_seam_regions(state)
    indexed = {item.id: item for item in result.state.drawing.regions[0].leaves()}
    for child in result.state.drawing.regions[0].children:
        left, right = child.children
        assert region_path(left, indexed).bounds[2] == pytest.approx(49.8)
        assert region_path(right, indexed).bounds[0] == pytest.approx(49.8)
    assert canonical_input_revision(replay_automatic_batches(state, result.batches)) == canonical_input_revision(result.state)


def test_shared_seam_candidates_keep_full_ancestry_for_duplicate_leaf_ids():
    from vectoreyes.seam_detection import SeamDetectionInput, discover_seam_choices
    from vectoreyes.refinement.automatic_replay import PassInputRegionRef
    from vectoreyes.refinement.region_sample import to_polygon
    from vectoreyes.skia_geometry import unary_union
    source = regions()
    footprint = unary_union([item.footprint for item in source])
    nodes = [SeamDetectionInput(PassInputRegionRef('', '', (10,)), None, footprint,
        (0, 0, 10, 10), footprint, {})]
    for branch, dy in ((20, 0), (30, 100)):
        clip = footprint.affine_transform((1, 0, 0, 1, 0, dy))
        nodes.append(SeamDetectionInput(PassInputRegionRef('', '', (10, branch)), None, clip,
            (0, 0, 10, branch), footprint, {}))
        for item in source:
            shape = Shape('path', {'d': item.footprint.affine_transform((1, 0, 0, 1, 0, dy)).to_svg_d()})
            nodes.append(SeamDetectionInput(PassInputRegionRef('', '', (10, branch, item.id)),
                shape, to_polygon(shape), (0, item.z, 10, item.id), footprint, {}))
    choice, = discover_seam_choices(tuple(nodes), allow_underlap=False)
    assert set(choice.updates) == {(10, 20, 1), (10, 20, 2), (10, 30, 1), (10, 30, 2)}
    assert {tuple(pair.left.ancestry) for pair in choice.intent.action.pairs} == {(10, 20, 1), (10, 30, 1)}


@pytest.mark.parametrize('allow_underlap', [False, True])
def test_underlap_toggle_is_retained_in_actual_constraint_kind(allow_underlap):
    from vectoreyes.seam_detection import SeamDetectionInput, discover_seam_choices
    from vectoreyes.refinement.automatic_replay import PathOpsUnderlapConstraint, PassInputRegionRef
    from vectoreyes.skia_geometry import SkPath
    source = (leaf(21, Shape('path', {'d': 'M0 0 L100 0 L0 100 Z'}), '#112233'),
        leaf(22, Shape('path', {'d': 'M100 1 L100 100 L1 100 Z'}), '#abcdef'))
    clip = SkPath(shell=[(0, 0), (100, 0), (100, 100), (0, 100)])
    inputs = (SeamDetectionInput(PassInputRegionRef('', '', (10,)), None,
        clip, (0., 0., 10, 10), clip, {}), *tuple(
        SeamDetectionInput(PassInputRegionRef('', '', (10, item.id)), item.current,
            item.footprint, (0., item.z, 10, item.id), clip, {}) for item in source))
    choices = discover_seam_choices(inputs, allow_underlap=allow_underlap)
    pairs = [pair for choice in choices for pair in choice.intent.action.pairs]
    assert pairs
    assert any(isinstance(pair, PathOpsUnderlapConstraint) for pair in pairs) is allow_underlap
    if allow_underlap:
        receipts = [item for choice in choices
            for item in choice.intent.action.path_correspondence if item.target.ancestry == (10, 21)]
        assert receipts
        mapping = {item.input_index: item.output_index for item in receipts[0].commands}
        assert mapping.get(2) != 2


def test_targeting_a_nested_leaf_can_repair_its_seam_with_an_outside_region():
    from vectoreyes.drawing_proposals import detect_seam_regions
    from vectoreyes.refinement.region_sample import RegionSample
    left, right = regions()
    parent = RegionSample.branch(id=10, drawing_id='parent', children=(left,))
    state = capture_replay_state((parent, right), Canvas(300, 440))
    target = state.drawing.regions[0].children[0].id
    result = detect_seam_regions(state, target_ids=(target,))
    assert sum(len(batch.operations) for batch in result.batches) == 1


def test_numeric_replay_does_not_alias_repeated_leaf_ids_within_one_owner():
    from vectoreyes.refinement.region_sample import RegionSample
    from vectoreyes.refinement.passes.seams import seams_pass
    from vectoreyes.refinement.automatic_replay import select_refinements_with_intent, replay_batches
    left, right = regions()
    right = replace(right, id=1, drawing_id='other')
    parent = RegionSample.branch(id=10, drawing_id='parent', children=(
        RegionSample.branch(id=20, children=(left,)), RegionSample.branch(id=30, children=(right,))))
    masks = {10: parent.raster}
    result = select_refinements_with_intent([parent], masks, [seams_pass])
    replayed = replay_batches([parent], result.batches, masks=masks)
    expected = capture_replay_state(result.objects, Canvas(300, 440))
    actual = capture_replay_state(replayed, Canvas(300, 440))
    assert actual.drawing == expected.drawing
    assert len({region.geometry.to_svg_d() for region in actual.drawing.regions[0].leaves()}) == 2


def test_post_symmetry_mode_repairs_only_the_marked_branch():
    from vectoreyes.drawing_proposals import detect_seam_regions
    from vectoreyes.refinement.region_sample import RegionSample
    parent = RegionSample.branch(id=10, drawing_id='parent', children=regions(),
        diagnostics={'symmetry': {'mode': 'self'}})
    state = capture_replay_state((parent,), Canvas(300, 440))
    result = detect_seam_regions(state, post_symmetry=True)
    assert sum(len(batch.operations) for batch in result.batches) == 1
    assert canonical_input_revision(replay_automatic_batches(state, result.batches)) == canonical_input_revision(result.state)

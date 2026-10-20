from dataclasses import replace

import pytest

from tests.test_canonical_structural_replay import leaf
from vectormark.drawing import Canvas
from vectormark.drawing_automatic import replay_automatic_batches
from vectormark.fit import Shape
from vectormark.geometry_native import region_path
from vectormark.path_geometry import CloneGeometry
from vectormark.refinement.replay_binding import capture_replay_state
from vectormark.replay_identity import canonical_input_revision
from vectormark.skia_geometry import SkPath


def polygon():
    return SkPath(shell=[(0, 0), (6, 0), (6, 2), (2, 2), (2, 7), (0, 7)])


def family():
    source = polygon()
    shapes = (source, source.affine_transform((1, 0, 0, 1, 20, 0)),
        source.affine_transform((0, 1, -1, 0, 50, 20)))
    regions = tuple(leaf(index, Shape('path', {'d': shape.to_svg_d()}), color)
        for index, (shape, color) in enumerate(zip(shapes, ('#112233', '#abcdef', '#f04060'), strict=True), 1))
    return capture_replay_state(regions, Canvas(300, 440))


def test_geometry_matcher_finds_translation_without_a_region_model():
    from vectormark.clone_detection import CloneDetectionInput, find_clone_matches
    source = polygon()
    matches = find_clone_matches((CloneDetectionInput(1, source),
        CloneDetectionInput(2, source.affine_transform((1, 0, 0, 1, 20, 0)))))
    match, = matches
    assert (match.source_id, match.target_id) == (1, 2)
    assert match.matrix == pytest.approx((1, 0, 0, 1, 20, 0))
    assert match.family_size is None


def test_geometry_matcher_prefers_one_source_for_a_rotated_family():
    from vectormark.clone_detection import CloneDetectionInput, find_clone_matches
    source = polygon()
    targets = {1: source, 2: source.affine_transform((1, 0, 0, 1, 20, 0)),
        3: source.affine_transform((0, 1, -1, 0, 50, 20))}
    matches = find_clone_matches(tuple(CloneDetectionInput(identity, path) for identity, path in targets.items()))
    assert [(item.source_id, item.target_id, item.family_size) for item in matches] == [(1, 2, 3), (1, 3, 3)]
    for match in matches:
        assert match.transformed.symmetric_difference(targets[match.target_id]).area < 1e-8


def test_geometry_matcher_retains_reflection_not_only_rotation():
    from vectormark.clone_detection import CloneDetectionInput, find_clone_matches
    source = polygon()
    target = source.affine_transform((-1, 0, 0, 1, 30, 0))
    match, = find_clone_matches((CloneDetectionInput(1, source), CloneDetectionInput(2, target)))
    a, b, c, d, _e, _f = match.matrix
    assert a*d - b*c < 0
    assert match.transformed.symmetric_difference(target).area < 1e-8


def test_strict_pair_cannot_turn_a_selected_family_source_into_another_clone():
    from vectormark.clone_detection import CloneDetectionInput, find_clone_matches
    inputs = tuple(CloneDetectionInput(identity, SkPath(shell=[
        (left, 0), (left+width, 0), (left+width, 10), (left, 10)]))
        for identity, left, width in ((1, 0, 10.), (2, 20, 10.08), (3, 40, 10.6), (4, 60, 10.6)))
    matches = find_clone_matches(inputs)
    # r2 is within the strict pair tolerance of r1, but only r2 supports the
    # complete three-member family with r3/r4. Family selection has priority.
    assert [(item.source_id, item.target_id) for item in matches] == [(2, 3), (2, 4)]


def test_canonical_clone_discovery_keeps_paint_and_needs_no_sampling_scene(monkeypatch):
    from vectormark.drawing_proposals import detect_clone_regions
    state = family()
    def forbidden(*args, **kwargs):
        raise AssertionError('canonical clone discovery projected a sampling scene')
    monkeypatch.setattr('vectormark.refinement.region_sample.RegionSample.__init__', forbidden)
    monkeypatch.setattr('vectormark.drawing_adapter.bind_region_inputs', forbidden)
    result = detect_clone_regions(state)
    assert len(result.batches[0].operations) == 2
    indexed = {region.id: region for region in result.state.drawing.regions}
    for identity in ('r2', 'r3'):
        selected = indexed[identity]
        assert isinstance(selected.geometry, CloneGeometry)
        assert selected.geometry.source_region_id == 'r1'
        assert selected.fill == state.drawing.region(identity).fill
        assert selected.diagnostics['clones']['family_size'] == 3
        assert region_path(selected, indexed).symmetric_difference(
            region_path(state.drawing.region(identity), {r.id: r for r in state.drawing.regions})).area < 1e-8
    replayed = replay_automatic_batches(state, result.batches)
    assert canonical_input_revision(replayed) == canonical_input_revision(result.state)


@pytest.mark.parametrize('target,count', [('r1', 2), ('r2', 1)])
def test_targeted_clone_discovery_can_address_source_or_instance(target, count):
    from vectormark.drawing_proposals import detect_clone_regions
    result = detect_clone_regions(family(), target_id=target)
    assert sum(len(batch.operations) for batch in result.batches) == count


def test_self_symmetric_regions_are_not_reinterpreted_as_clone_sources():
    from vectormark.drawing_proposals import detect_clone_regions
    state = family()
    state = replace(state, drawing=replace(state.drawing, regions=tuple(replace(region,
        diagnostics={'symmetry': {'mode': 'self'}}) for region in state.drawing.regions)))
    result = detect_clone_regions(state)
    assert not any(batch.operations for batch in result.batches)


def test_scoped_clone_discovery_does_not_alias_repeated_local_ids():
    from vectormark.drawing_proposals import detect_clone_regions
    from vectormark.refinement.region_sample import RegionSample
    source = polygon()
    children = tuple(leaf(index, Shape('path', {'d': source.affine_transform((1, 0, 0, 1, (index-1)*20, 0)).to_svg_d()}), '#112233')
        for index in (1, 2))
    state = capture_replay_state(tuple(RegionSample.branch(id=identity, drawing_id=f'parent{identity}',
        children=children) for identity in (10, 20)), Canvas(300, 440))
    result = detect_clone_regions(state)
    assert [batch.scope for batch in result.batches] == [(10,), (20,), ()]
    assert sum(len(batch.operations) for batch in result.batches) == 2
    for parent in result.state.drawing.regions:
        source, target = parent.children
        assert target.geometry.source_region_id == source.id
    replayed = replay_automatic_batches(state, result.batches)
    assert canonical_input_revision(replayed) == canonical_input_revision(result.state)


def test_public_clone_detection_does_not_project_sampling_scene(monkeypatch):
    from tests.test_drawing_recipe_actions import _direct_recipe_fixture
    from vectormark.drawing_plan import DrawingPlan
    from vectormark.drawing_replay import extend_recipe_with_plan, propose_effective_recipe
    source = polygon()
    regions = tuple(leaf(index, Shape('path', {'d': source.affine_transform((1, 0, 0, 1, (index-1)*20, 0)).to_svg_d()}), '#112233')
        for index in (1, 2))
    evidence, paint, base = _direct_recipe_fixture(regions, width=300, height=440)
    plan = DrawingPlan.model_validate({'version': 'vectormark.plan.v1', 'drawing_id': 'drw_clones',
        'base_version': 'v0', 'ops': [{'op': 'detect_clones', 'target': 'r1'}]})
    def forbidden(*args, **kwargs):
        raise AssertionError('public clone discovery projected a sampling scene')
    monkeypatch.setattr('vectormark.refinement.replay_binding.working_regions_from_state', forbidden)
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    assert sum(len(batch.operations) for batch in recipe.automatic_batches) == 1
    result = propose_effective_recipe(drawing_id='drw_clones', evidence=evidence, paint=paint, recipe=recipe)
    assert isinstance(result.drawing.region('r2').geometry, CloneGeometry)

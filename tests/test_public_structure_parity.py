from dataclasses import replace

import pytest

from tests.test_canonical_structural_replay import leaf
from tests.test_drawing_recipe_actions import _direct_recipe_fixture
from vectoreyes.drawing_actions import DrawingPlan
from vectoreyes.drawing_replay import extend_recipe_with_plan, propose_effective_recipe
from vectoreyes.fit import Shape
from vectoreyes.geometry_native import region_path
from vectoreyes.path_geometry import CloneGeometry, CompoundPathGeometry, parse_svg_paths
from pydantic import ValidationError


def submit(regions, ops, *, state=None):
    evidence, paint, base = _direct_recipe_fixture(regions, width=300, height=440)
    if state is not None:
        from vectoreyes.drawing_recipe import build_geometry_evidence
        evidence = build_geometry_evidence(kind=evidence.kind, trace=evidence.trace,
            geometry_source=evidence.geometry_source, drawing=state.drawing,
            shape_metadata=state.shape_metadata, region_evidence=state.region_evidence)
        base = replace(base, evidence_revision=evidence.revision)
    plan = DrawingPlan.model_validate({'version': 'vectoreyes.plan.v1',
        'drawing_id': 'structure', 'base_version': 'v0', 'ops': ops})
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    result = propose_effective_recipe(drawing_id='structure', evidence=evidence, paint=paint, recipe=recipe)
    return evidence, paint, recipe, result


@pytest.mark.parametrize('transforms', [
    [{'a': -1, 'd': 1, 'tx': 40}],
    [{'tx': 40}, {'tx': 60}],
    [{'a': 2, 'd': 2, 'tx': 40}],
])
def test_explicit_baked_clone_matches_symbolic_geometry(transforms):
    regions = (leaf(1, Shape('circle', {'cx': 10, 'cy': 10, 'r': 5}), '#112233'),
        leaf(2, Shape('rect', {'x': 40, 'y': 0, 'w': 10, 'h': 10}), '#445566'))
    op = {'op': 'assign_clone', 'source': 'r1', 'target': 'r2', 'transforms': transforms}
    _, _, _, symbolic = submit(regions, [{**op, 'symbolic': True}])
    _, _, _, baked = submit(regions, [{**op, 'symbolic': False}])
    assert isinstance(symbolic.drawing.region('r2').geometry, CloneGeometry)
    assert not isinstance(baked.drawing.region('r2').geometry, CloneGeometry)
    indexed = {item.id: item for root in symbolic.drawing.regions for item in (root, *root.leaves())}
    expected = region_path(symbolic.drawing.region('r2'), indexed)
    actual = region_path(baked.drawing.region('r2'), {})
    assert actual.symmetric_difference(expected).area < .001


def graph():
    return {'op': 'reconstruct_occlusion', 'target': 'r1', 'regions': ['r1', 'r2'],
        'children': [
            {'type': 'primitive', 'id': 'left', 'source': 'r1', 'z': 0,
             'primitive': {'type': 'circle', 'center': [30, 30], 'radius': 20}},
            {'type': 'primitive', 'id': 'right', 'source': 'r2', 'z': 1,
             'primitive': {'type': 'circle', 'center': [50, 30], 'radius': 20}},
            {'type': 'lens', 'id': 'lens', 'source': 'r1', 'z': 2,
             'left': 'left', 'right': 'right'},
        ]}


def test_explicit_occlusion_graph_executes_and_replays_without_discovery(monkeypatch):
    regions = (leaf(1, Shape('circle', {'cx': 30, 'cy': 30, 'r': 20}), '#112233'),
        leaf(2, Shape('circle', {'cx': 50, 'cy': 30, 'r': 20}), '#445566'),
        leaf(3, Shape('rect', {'x': 100, 'y': 100, 'w': 10, 'h': 10}), '#778899'))
    def forbidden(*args, **kwargs):
        raise AssertionError('a declared graph must not run occlusion detection')
    monkeypatch.setattr('vectoreyes.occlusion_detection.reconstruct_scene', forbidden)
    evidence, paint, recipe, result = submit(regions, [graph()])
    branch = result.drawing.region('r1')
    assert branch.is_branch and len(branch.children) == 3
    assert tuple(child.z for child in branch.children) == (0, 1, 2)
    assert {root.id for root in result.drawing.regions} == {'r1', 'r3'}
    left, right, lens = (region_path(child, {}) for child in branch.children)
    assert lens.symmetric_difference(left.intersection(right)).area < .1
    assert propose_effective_recipe(drawing_id='structure', evidence=evidence, paint=paint,
        recipe=recipe).drawing == result.drawing


def test_explicit_occlusion_rejects_bad_dependency_before_mutation():
    action = graph()
    action['children'][-1]['left'] = 'absent'
    regions = tuple(leaf(i, Shape('circle', {'cx': i * 20, 'cy': 30, 'r': 20}), '#112233')
        for i in (1, 2))
    before = tuple(region.current for region in regions)
    with pytest.raises(ValueError, match='absent dependency'):
        submit(regions, [action])
    assert tuple(region.current for region in regions) == before


def test_public_topology_normalization_preserves_paths_and_records_ownership():
    d = 'M0 0 L8 0 L8 8 L0 8 Z M12 0 L20 0 L20 8 L12 8 Z'
    paths = parse_svg_paths(d, region_id='r1')
    shape = Shape('path', {'d': d, 'fill_rule': 'evenodd',
        'path_ids': tuple(path.path.id for path in paths),
        'canonical_path_geometry': CompoundPathGeometry(paths, 'evenodd', 'pending component publication')})
    from vectoreyes.refinement.replay_binding import capture_replay_state
    from vectoreyes.drawing import Canvas
    regions = (leaf(1, shape, '#112233'),)
    raw = capture_replay_state(regions, Canvas(300, 440))
    evidence, paint, recipe, result = submit(regions,
        [{'op': 'normalize_topology'}], state=raw)
    assert result.drawing.region('r1').is_branch
    assert len(result.drawing.region('r1').children) == 2
    assert tuple(child.geometry for child in result.drawing.region('r1').children) == paths
    assert recipe.automatic_batches[0].operations[0].intent.action.path_correspondence
    assert propose_effective_recipe(drawing_id='structure', evidence=evidence, paint=paint,
        recipe=recipe).drawing == result.drawing


def test_topology_changing_declarations_require_a_new_version_for_followup_edits():
    with pytest.raises(ValidationError, match='must be final'):
        DrawingPlan.model_validate({'version': 'vectoreyes.plan.v1', 'drawing_id': 'structure',
            'base_version': 'v0', 'ops': [{'op': 'normalize_topology'},
                {'op': 'fit_primitive', 'target': 'r1', 'type': 'rect'}]})


@pytest.mark.parametrize('primitive', [
    {'type': 'annulus', 'center': [30, 30], 'outer_radius': 20, 'inner_radius': 10},
    {'type': 'ellipse', 'center': [30, 30], 'radius_x': 20, 'radius_y': 10},
    {'type': 'rect', 'origin': [10, 10], 'width': 40, 'height': 30},
    {'type': 'rounded_rect', 'origin': [10, 10], 'width': 40, 'height': 30, 'radius_x': 5, 'radius_y': 5},
    {'type': 'polygon', 'vertices': [[10, 10], [40, 10], [30, 40]]},
])
def test_declared_occlusion_accepts_all_supported_primitive_values(primitive):
    regions = (leaf(1, Shape('circle', {'cx': 30, 'cy': 30, 'r': 20}), '#112233'),)
    _, _, _, result = submit(regions, [{'op': 'reconstruct_occlusion', 'target': 'r1',
        'regions': ['r1'], 'children': [{'type': 'primitive', 'id': 'shape',
            'source': 'r1', 'primitive': primitive}]}])
    path = region_path(result.drawing.region('r1').children[0], {})
    assert path.area > 0
    if primitive['type'] == 'annulus':
        assert not path.to_skia().contains(30, 30)


def test_declared_occlusion_rejects_retiring_a_live_clone_source():
    regions = (leaf(1, Shape('circle', {'cx': 30, 'cy': 30, 'r': 20}), '#112233'),
        leaf(2, Shape('circle', {'cx': 50, 'cy': 30, 'r': 20}), '#445566'),
        leaf(3, Shape('rect', {'x': 100, 'y': 100, 'w': 10, 'h': 10}), '#778899'))
    clone = {'op': 'assign_clone', 'source': 'r1', 'target': 'r3', 'transforms': [{'tx': 80}]}
    with pytest.raises(ValueError, match='live clone source'):
        submit(regions, [clone, graph()])


def test_baked_path_has_target_command_addresses():
    regions = (leaf(1, Shape('path', {'d': 'M10 10 Q15 0 20 10 L10 10 Z'}), '#112233'),
        leaf(2, Shape('rect', {'x': 40, 'y': 0, 'w': 10, 'h': 10}), '#445566'))
    _, _, _, result = submit(regions, [{'op': 'assign_clone', 'source': 'r1', 'target': 'r2',
        'symbolic': False, 'transforms': [{'tx': 40}]}])
    from vectoreyes.path_topology import component_paths
    for path in component_paths(result.drawing.region('r2').geometry):
        assert path.path.id.startswith('r2.')
        assert all(command.id.startswith('r2.') for command in path.path.commands)


def test_baked_paths_preserve_features_with_instance_owned_ids():
    from vectoreyes.drawing import Drawing, DrawingRegion, Canvas
    from vectoreyes.drawing_actions import AssignCloneAction
    from vectoreyes.drawing_relationships import assign_clone
    from vectoreyes.path_geometry import CornerAnchor, InflectionAnchor
    path = parse_svg_paths('M10 10 Q15 0 20 10 L10 10 Z', region_id='r1')[0]
    path = replace(path, anchors=(CornerAnchor('r1.p0.corner0', 'r1.p0.c1', 'r1.p0.c1', radius=0),
        InflectionAnchor('r1.p0.inflection0', ('r1.p0.c1',))))
    source = DrawingRegion('r1', path)
    target = DrawingRegion('r2', path)
    drawing = Drawing('structure', Canvas(100, 100), (source, target))
    result = assign_clone(drawing, AssignCloneAction.model_validate({'op': 'assign_clone',
        'source': 'r1', 'target': 'r2', 'symbolic': False, 'transforms': [{'tx': 40}]}))
    assert len(result.geometry.anchors) == 2
    assert all(anchor.id.startswith('r2.') for anchor in result.geometry.anchors)
    assert all(item.startswith('r2.') for anchor in result.geometry.anchors for item in anchor.command_ids)
    assert result.geometry.path.drawable_commands[0].end == (60, 10)
    assert source.geometry == path

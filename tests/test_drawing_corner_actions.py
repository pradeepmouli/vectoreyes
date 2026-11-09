from dataclasses import replace
from pathlib import Path
import sys

import pytest

from vectoreyes.corner_recipe import AddCorner, CornerRecipeExecutor
from vectoreyes.drawing_engine import PathFitOptions
from vectoreyes.path_geometry import CornerAnchor, geometry_from_svg_d

sys.path.insert(0, str(Path(__file__).resolve().parent))


def fixture(cutout=False):
    from test_drawing_corner_store_replay import _source_store
    _, base, paint, evidence = _source_store(cutout=cutout)
    region = base.drawing.region('r1')
    path = region.geometry.cutouts[0] if cutout else region.geometry
    source = next(item.fit for item in evidence.root_paths if item.path_id == path.path.id)
    recipe = CornerRecipeExecutor().create(source, PathFitOptions(1, 1.5, 'quadratic'))
    return region, path, source, recipe, base, paint, evidence


@pytest.mark.parametrize('cutout', [False, True])
def test_corner_region_action_refits_one_path_and_preserves_region_values(cutout):
    from vectoreyes.drawing_corner_actions import apply_region_corner_recipe
    region, path, source, recipe, *_ = fixture(cutout)
    direct = CornerRecipeExecutor().replay(recipe, source)
    updated, replay = apply_region_corner_recipe(region, path_id=path.path.id, recipe=recipe, evidence=source)
    result = updated.geometry.cutouts[0] if cutout else updated.geometry
    assert updated.fill == region.fill and updated.z == region.z
    assert updated.source_region_ids == region.source_region_ids
    assert [command.end for command in result.path.commands] == [command.end for command in direct.fit.geometry.path.commands]
    assert all(command.id.startswith(path.path.id + '.c') for command in result.path.commands)
    assert replay == direct
    if cutout:
        assert updated.geometry.outer is region.geometry.outer
    assert updated is not region


def test_corner_action_rejects_stale_path_before_fitting():
    from vectoreyes.drawing_corner_actions import apply_region_corner_recipe
    region, _, source, recipe, *_ = fixture()
    class ForbiddenExecutor:
        def replay(self, *args):
            pytest.fail('stale selection reached the fitter')
    with pytest.raises(ValueError, match='target path is stale'):
        apply_region_corner_recipe(region, path_id='r1.p999', recipe=recipe, evidence=source, executor=ForbiddenExecutor())


def annotated_path():
    path = geometry_from_svg_d('M0 0 L10 0 L10 10 L0 10 Z', region_id='r1')
    command = path.path.commands[1].id
    return replace(path, anchors=(CornerAnchor('r1.p0.corner0', command, command),))


def test_annotation_rebinding_keeps_new_coordinates_and_exact_old_features():
    from vectoreyes.drawing_corner_actions import rebind_corner_annotations
    annotated = annotated_path()
    parsed = geometry_from_svg_d('M0 0 L11 0 L10 10 L0 10 Z', region_id='r1')
    result = rebind_corner_annotations(parsed, annotated)
    assert result.path is parsed.path
    assert result.anchors == annotated.anchors
    assert result.path.commands[1].end == (11, 0)


@pytest.mark.parametrize('d,region_id,reason', [
    ('M0 0 L10 0 L10 10 L0 10 Z', 'r2', 'path identity'),
    ('M0 0 Q5 0 10 0 L10 10 L0 10 Z', 'r1', 'command topology'),
    ('M0 0 L10 0 L10 10 L5 10 L0 10 Z', 'r1', 'command topology'),
])
def test_annotation_rebinding_rejects_changed_ownership(d, region_id, reason):
    from vectoreyes.drawing_corner_actions import rebind_corner_annotations
    with pytest.raises(ValueError, match=reason):
        rebind_corner_annotations(geometry_from_svg_d(d, region_id=region_id), annotated_path())


def test_source_bound_replay_routes_through_canonical_region_action(monkeypatch):
    from vectoreyes import drawing_corner_actions
    from vectoreyes.drawing import DrawingRegion
    from vectoreyes.drawing_recipe import VersionPathSelection
    from vectoreyes.drawing_replay import extend_recipe_with_corner_edits, propose_effective_recipe
    region, path, source, _, base, paint, evidence = fixture()
    source_command = source.source.path.commands[0].id
    displayed = path.path.commands[0].id
    selection = VersionPathSelection(base.id, evidence.revision, base.effective_recipe.revision,
        (1,), path.path.id, displayed, displayed)
    recipe, _ = extend_recipe_with_corner_edits(evidence=evidence, paint=paint,
        base_recipe=base.effective_recipe, base_version_id=base.id, selection=selection,
        edits=(AddCorner(CornerAnchor('agent.corner', source_command, source_command)),))
    original = drawing_corner_actions.apply_region_corner_recipe
    calls = []
    def record(region, **kwargs):
        assert isinstance(region, DrawingRegion)
        calls.append(region.id)
        return original(region, **kwargs)
    monkeypatch.setattr(drawing_corner_actions, 'apply_region_corner_recipe', record)
    propose_effective_recipe(drawing_id=base.drawing.drawing_id, evidence=evidence, paint=paint, recipe=recipe)
    assert calls == ['r1']

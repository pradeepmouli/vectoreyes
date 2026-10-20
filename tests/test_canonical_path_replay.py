from tests.test_canonical_geometry_proposals import proposal_case
from tests.test_drawing_corner_store_replay import _source_store
from vectormark.drawing_replay import _path_record_for_display, _source_command_for_display
from vectormark.replay_state import ReplayState


def test_path_remove_proposal_uses_commands_without_original_shape_observations():
    before, after = proposal_case({'op': 'update_geometry', 'target': 'r1',
        'ops': [{'op': 'remove', 'target': 'r1.p0.c2'}]})
    path = after.drawing.region('r1').geometry.path
    assert tuple(command.end for command in path.commands) == ((1., 1.), (6., 1.), (1., 6.))
    assert after.region_evidence == before.region_evidence
    assert after.shape_metadata['r1']['path_command_lineage']['commands']['r1.p0.c3'] == 'r1.p0.c2'


def test_correspondence_lookup_needs_only_canonical_commands_and_source_ancestry():
    _, base, _, evidence = _source_store()
    state = ReplayState(evidence.drawing, evidence.shape_metadata,
        {identity: {'ancestry': info['ancestry']} for identity, info in evidence.region_evidence.items()})
    region = state.drawing.regions[0]
    path = region.geometry.path
    record = _path_record_for_display(evidence, state, region.id, path.id, [], base.effective_recipe)
    assert record.output_region_id == region.id and record.output_path_id == path.id
    source_id = _source_command_for_display(evidence, state, region.id, path.id,
        path.commands[0].id, [], base.effective_recipe)
    source = evidence.root_paths[0].fit.source.path
    point = next(command.end for command in source.commands if command.id == source_id)
    assert abs(point[0] - path.commands[0].end[0]) <= 1.5
    assert abs(point[1] - path.commands[0].end[1]) <= 1.5
    assert state.region_evidence == {identity: {'ancestry': info['ancestry']}
        for identity, info in evidence.region_evidence.items()}


def test_corner_recipe_publishes_canonical_state_and_retains_source_observations():
    from vectormark.corner_recipe import CornerRecipeExecutor
    from vectormark.drawing_engine import PathFitOptions
    from vectormark.drawing_recipe import ApplyCornerRecipeOperation
    from vectormark.drawing_replay import _execute_corner_recipe
    from vectormark.replay_identity import canonical_input_revision

    _, base, _, evidence = _source_store()
    state = ReplayState(evidence.drawing, evidence.shape_metadata, evidence.region_evidence)
    source = evidence.root_paths[0]
    recipe = CornerRecipeExecutor().create(source.fit, PathFitOptions(1., 1.5, 'quadratic'))
    application = ApplyCornerRecipeOperation(recipe=recipe, evidence_revision=evidence.revision,
        region_ancestry=(1,), target_region_id='r1', target_path_id=source.path_id,
        input_revision=canonical_input_revision(state), settings=base.effective_recipe.settings)
    result, replay = _execute_corner_recipe(application, evidence, state)
    assert isinstance(result, ReplayState)
    assert result.region_evidence == state.region_evidence
    assert result.drawing.region('r1').geometry.to_svg_d() == replay.fit.geometry.to_svg_d()

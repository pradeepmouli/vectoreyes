from dataclasses import replace

import pytest

from vectormark.drawing import Canvas
from vectormark.drawing_automatic import replay_automatic_batches
from vectormark.refinement.replay_binding import capture_replay_state
from vectormark.refinement_settings import RefinementPasses
from tests.test_canonical_seam_discovery import regions


def disabled(**enabled):
    return RefinementPasses(**{name: enabled.get(name, False)
        for name in RefinementPasses.model_fields})


def test_engine_retains_replayable_pass_results_without_reexecuting_discovery(monkeypatch):
    from tests.test_python_drawing_engine import source
    from vectormark.python_drawing_engine import PythonDrawingEngine
    from vectormark.drawing_engine import TraceDrawingRequest
    from vectormark.drawing_trace import TraceOptions
    from vectormark.drawing_replay import propose_effective_recipe
    from vectormark.drawing_recipe import drawing_revision
    import vectormark.drawing_refinement as refinement
    _, raster = source()
    engine = PythonDrawingEngine()
    baseline = engine.trace_drawing(TraceDrawingRequest('drw_stages', raster,
        TraceOptions(max_sampling_colors=2, remove_background='off'),
        min_region_fraction=0, initial_stitch=False))
    original = refinement.refine_canonical_drawing
    calls = []
    def observe(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(refinement, 'refine_canonical_drawing', observe)
    result = engine.refine_automatically(baseline.drawing, baseline.evidence)
    assert len(calls) == 1
    assert result.passes
    assert result.passes[-1].name == 'resolve_original_paint'
    previous = baseline.drawing
    for stage in result.passes:
        assert stage.status in {'changed', 'unchanged', 'disabled'}
        if stage.result is None:
            assert stage.status == 'disabled'
            continue
        assert (stage.result.drawing != previous) == (stage.status == 'changed')
        replayed = propose_effective_recipe(drawing_id='drw_stages',
            evidence=stage.result.evidence.geometry, paint=stage.result.evidence.paint,
            recipe=stage.result.evidence.recipe)
        assert replayed.drawing == stage.result.drawing
        assert drawing_revision(replayed.drawing) == drawing_revision(stage.result.drawing)
        assert replayed.replay.path_correspondence == stage.result.evidence.replay.path_correspondence
        previous = stage.result.drawing
    assert previous == result.drawing


def test_disabled_refinement_passes_have_receipts_not_duplicate_snapshots():
    from tests.test_python_drawing_engine import source
    from vectormark.python_drawing_engine import PythonDrawingEngine
    from vectormark.drawing_engine import TraceDrawingRequest
    from vectormark.drawing_trace import TraceOptions
    _, raster = source()
    engine = PythonDrawingEngine()
    baseline = engine.trace_drawing(TraceDrawingRequest('drw_disabled', raster,
        TraceOptions(max_sampling_colors=2, remove_background='off'),
        min_region_fraction=0, initial_stitch=False))
    result = engine.refine_automatically(baseline.drawing, baseline.evidence,
        refinement_passes=disabled())
    assert len(result.passes) == 10
    assert all(stage.status == 'disabled' and stage.result is None for stage in result.passes[:-1])


def test_automatic_engine_refinement_never_projects_a_sampling_scene(monkeypatch):
    from tests.test_python_drawing_engine import source
    from vectormark.python_drawing_engine import PythonDrawingEngine
    from vectormark.drawing_engine import TraceDrawingRequest
    from vectormark.drawing_trace import TraceOptions
    from vectormark.drawing_replay import propose_effective_recipe
    _, raster = source()
    engine = PythonDrawingEngine()
    baseline = engine.trace_drawing(TraceDrawingRequest('drw_auto', raster,
        TraceOptions(max_sampling_colors=2, remove_background='off'),
        min_region_fraction=0, initial_stitch=False))
    def forbidden(*args, **kwargs):
        raise AssertionError('automatic refinement projected a sampling scene')
    monkeypatch.setattr('vectormark.refinement.region_sample.RegionSample.__init__', forbidden)
    monkeypatch.setattr('vectormark.refinement.replay_binding.working_regions_from_state', forbidden)
    refined = engine.refine_automatically(baseline.drawing, baseline.evidence)
    replayed = propose_effective_recipe(drawing_id='drw_auto', evidence=refined.evidence.geometry,
        paint=refined.evidence.paint, recipe=refined.evidence.recipe)
    assert replayed.drawing == refined.drawing
    assert refined.drawing.regions


@pytest.mark.parametrize('stitch', [False, True])
def test_canonical_multi_pass_respects_toggles_and_replays(stitch):
    from vectormark.drawing_refinement import refine_canonical_drawing
    state = capture_replay_state(regions(), Canvas(300, 440))
    observed = []
    result = refine_canonical_drawing(state, epsilon=1, max_error=1,
        refinement_passes=disabled(stitch=stitch),
        on_pass=lambda name, current: observed.append((name, current)))
    assert [name for name, _ in observed] == (['seams_pass'] if stitch else [])
    assert len({batch.batch_id for batch in result.batches}) == len(result.batches)
    replayed = replay_automatic_batches(state, result.batches)
    assert replayed.drawing == result.state.drawing
    left = result.state.drawing.region('r1').geometry.to_svg_d()
    assert ('49.8' in left) is stitch


def test_canonical_multi_pass_visits_nested_scopes_without_aliasing_ids():
    from vectormark.drawing_refinement import refine_canonical_drawing
    from vectormark.refinement.region_sample import RegionSample
    left, right = regions()
    parent = RegionSample.branch(id=10, drawing_id='parent', children=(
        RegionSample.branch(id=20, children=(left, right)),
        RegionSample.branch(id=30, children=(replace(left, drawing_id='other-left'),
            replace(right, drawing_id='other-right')))))
    state = capture_replay_state((parent,), Canvas(300, 440))
    result = refine_canonical_drawing(state, epsilon=1, max_error=1,
        refinement_passes=disabled(stitch=True))
    seams = [batch for batch in result.batches if any(
        record.pass_name == 'seams_pass' for record in batch.entries)]
    assert {batch.scope for batch in seams} >= {(10, 20), (10, 30)}
    assert replay_automatic_batches(state, result.batches).drawing == result.state.drawing


def test_shared_schedule_retains_clone_before_symmetry_and_terminal_stitch():
    from vectormark.refinement_settings import refinement_pass_names
    assert refinement_pass_names(RefinementPasses()) == (
        'occlusion_pass', 'split_compound_pass', 'primitives_pass', 'clones_pass',
        'symmetry_pass', 'linelet_simplify_pass', 'smooth_pass',
        'corner_normalize_pass', 'seams_pass')
    assert refinement_pass_names(disabled(), no_symmetry=True) == ('corner_normalize_pass',)
    assert 'symmetry_pass' not in refinement_pass_names(RefinementPasses(), no_symmetry=True)


def test_compound_stage_records_component_ownership_and_rejects_stale_correspondence():
    from vectormark.drawing_refinement import refine_canonical_drawing
    from vectormark.fit import Shape
    from vectormark.path_geometry import CompoundPathGeometry, parse_svg_paths
    from tests.test_canonical_structural_replay import leaf
    d = 'M0 0 L8 0 L8 8 L0 8 Z M12 0 L20 0 L20 8 L12 8 Z'
    shape = Shape('path', {'d': d, 'fill_rule': 'evenodd', 'path_ids': ('r1.p0', 'r1.p1'),
        'canonical_path_geometry': CompoundPathGeometry(parse_svg_paths(d, region_id='r1'),
            'evenodd', 'pending component publication')})
    state = capture_replay_state((leaf(1, shape, '#123456'),), Canvas(300, 440))
    result = refine_canonical_drawing(state, epsilon=1, max_error=1,
        refinement_passes=disabled(split_compound=True))
    assert len(result.state.drawing.region('r1').children) == 2
    batch = next(batch for batch in result.batches if batch.operations)
    assert batch.operations[0].intent.action.path_correspondence
    assert replay_automatic_batches(state, result.batches).drawing == result.state.drawing
    operation = batch.operations[0]
    invalid = replace(operation, intent=replace(operation.intent,
        action=replace(operation.intent.action, path_correspondence=())))
    with pytest.raises(ValueError, match='correspondence is stale'):
        replay_automatic_batches(state, (replace(batch, entries=(invalid,)),))


def test_primitive_stage_changes_geometry_without_changing_paint():
    from vectormark.drawing_refinement import refine_canonical_drawing
    from vectormark.path_geometry import PrimitiveGeometry
    state = capture_replay_state(regions(), Canvas(300, 440))
    result = refine_canonical_drawing(state, epsilon=1, max_error=1,
        refinement_passes=disabled(primitives=True))
    assert all(isinstance(region.geometry, PrimitiveGeometry) for region in result.state.drawing.regions)
    assert [region.fill for region in result.state.drawing.regions] == [region.fill for region in state.drawing.regions]
    assert replay_automatic_batches(state, result.batches).drawing == result.state.drawing


def test_clone_stage_preserves_rotation_translation_and_independent_colors():
    from tests.test_canonical_clone_discovery import family
    from vectormark.drawing_refinement import refine_canonical_drawing
    from vectormark.path_geometry import CloneGeometry
    state = family()
    result = refine_canonical_drawing(state, epsilon=1, max_error=1,
        refinement_passes=disabled(clones=True))
    assert sum(isinstance(region.geometry, CloneGeometry) for region in result.state.drawing.regions) == 2
    assert [region.fill for region in result.state.drawing.regions] == [region.fill for region in state.drawing.regions]
    assert replay_automatic_batches(state, result.batches).drawing == result.state.drawing


@pytest.mark.parametrize('enabled', [False, True])
def test_corner_toggle_changes_only_the_declared_corner_fit(enabled):
    from tests.test_canonical_path_discovery import state_for
    from vectormark.drawing_refinement import refine_canonical_drawing
    state = state_for('M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z', rounded=True)
    result = refine_canonical_drawing(state, epsilon=1, max_error=1, corner_normalize=enabled,
        refinement_passes=disabled())
    assert result.state.drawing.region('r1').geometry.to_svg_d().count('Q') == (1 if enabled else 2)
    assert replay_automatic_batches(state, result.batches).drawing == result.state.drawing

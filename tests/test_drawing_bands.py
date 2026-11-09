"""Bands share one source-bound operation across plans and automatic passes."""

import numpy as np
import pytest


def only_bands():
    from vectoreyes.refinement_settings import RefinementPasses
    return RefinementPasses(**{name: name == 'bands' for name in RefinementPasses.model_fields})


def traced_band():
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    from vectoreyes.drawing_engine import RasterImage, TraceDrawingRequest
    from vectoreyes.drawing_trace import TraceOptions
    image = np.full((64, 240, 3), 255, np.uint8)
    image[20:40, 12:228] = (20, 100, 240)
    engine = PythonDrawingEngine()
    result = engine.trace_drawing(TraceDrawingRequest('drw_band',
        RasterImage(240, 64, 'rgb8', image.tobytes()),
        TraceOptions(max_sampling_colors=2, remove_background='white'),
        min_region_fraction=0, initial_stitch=False))
    return engine, result


def test_band_toggle_is_off_default_but_in_full_registry_and_schema():
    from vectoreyes.refinement_settings import RefinementPasses, refinement_pass_names
    defaults = RefinementPasses()
    assert getattr(defaults, 'bands', None) is False
    assert 'bands_pass' not in refinement_pass_names(defaults)
    assert 'bands_pass' in refinement_pass_names(defaults, include_disabled=True)
    assert 'bands_pass' in refinement_pass_names(only_bands())
    assert RefinementPasses.model_json_schema()['properties']['bands']['default'] is False
    from vectoreyes.mcp_server import TracePassOutput
    assert 'diagnostics' in TracePassOutput.model_fields


def test_legacy_numerical_schedule_can_select_bands_without_guessing_sources():
    from vectoreyes.pipeline import Options, _refinement_passes
    stages = _refinement_passes(Options(refinement=True, refinement_passes=only_bands()))
    stage = next(item for item in stages if item.__name__ == 'bands_pass')
    with pytest.warns(RuntimeWarning, match='original source'):
        assert stage([], {}) == []


def test_enabled_band_pass_retains_an_ordinary_replayable_revision(monkeypatch):
    from vectoreyes.drawing_recipe import RecipeOperation
    from vectoreyes.drawing_replay import propose_effective_recipe
    engine, baseline = traced_band()
    refined = engine.refine_automatically(baseline.drawing, baseline.evidence,
        refinement_passes=only_bands())
    stage = next((item for item in refined.passes if item.name == 'bands_pass'), None)
    assert stage is not None, 'enabled off-default pass was lost by revision enumeration'
    assert stage.result is not None
    operations = [step for step in stage.result.evidence.recipe.steps
        if isinstance(step, RecipeOperation) and step.action.op == 'set_band']
    assert operations, 'generic discovery did not find the two long corner-bounded rectangle sides'
    assert operations[0].selected_outcome is not None
    def forbidden(*args, **kwargs):
        raise AssertionError('band numeric kernel ran during selected replay')
    monkeypatch.setattr(engine.__class__, 'reconstruct_band', forbidden)
    replayed = propose_effective_recipe(drawing_id=baseline.drawing.drawing_id,
        evidence=stage.result.evidence.geometry, paint=stage.result.evidence.paint,
        recipe=stage.result.evidence.recipe)
    assert replayed.drawing == stage.result.drawing


def test_explicit_and_auto_use_identical_action_and_keep_all_corner_endpoints():
    from vectoreyes.drawing_actions import DrawingPlan
    from vectoreyes.drawing_recipe import RecipeOperation
    from vectoreyes.drawing_replay import extend_recipe_with_plan, propose_effective_recipe
    engine, baseline = traced_band()
    refined = engine.refine_automatically(baseline.drawing, baseline.evidence,
        refinement_passes=only_bands())
    stage = next(item for item in refined.passes if item.name == 'bands_pass').result
    operations = [step for step in stage.evidence.recipe.steps
        if isinstance(step, RecipeOperation) and step.action.op == 'set_band']
    plan = DrawingPlan(version='vectoreyes.plan.v1', drawing_id=baseline.drawing.drawing_id,
        base_version='v0', ops=tuple(item.action for item in operations))
    evidence = baseline.evidence
    extension = extend_recipe_with_plan(evidence=evidence.geometry, paint=evidence.paint,
        base_recipe=evidence.recipe, plan=plan)
    explicit = propose_effective_recipe(drawing_id=baseline.drawing.drawing_id,
        evidence=evidence.geometry, paint=evidence.paint, recipe=extension.recipe)
    assert explicit.drawing.region('r1').geometry == stage.drawing.region('r1').geometry
    before = baseline.drawing.region('r1').geometry
    after = stage.drawing.region('r1').geometry
    assert {anchor.id for anchor in before.anchors} == {anchor.id for anchor in after.anchors}
    for left, right in zip(before.anchors, after.anchors, strict=True):
        assert before.endpoint(left.entry_command_id) == after.endpoint(right.entry_command_id)
        assert before.endpoint(left.exit_command_id) == after.endpoint(right.exit_command_id)
    assert stage.evidence.replay.path_correspondence


def test_source_less_canonical_band_pass_is_an_explicit_diagnostic_skip():
    from vectoreyes.drawing_refinement import refine_canonical_drawing
    from vectoreyes.replay_state import ReplayState
    engine, baseline = traced_band()
    state = ReplayState(baseline.drawing, baseline.evidence.geometry.shape_metadata,
        baseline.evidence.geometry.region_evidence)
    result = refine_canonical_drawing(state, epsilon=1, max_error=2,
        refinement_passes=only_bands())
    assert result.state.drawing.regions == state.drawing.regions
    stage = dict(result.passes)['bands_pass']
    assert stage.diagnostics['reason'] == 'original_source_context_unavailable'


def test_disjoint_bands_can_share_one_region_without_retiring_its_other_sides():
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    from vectoreyes.drawing_engine import RasterImage, TraceDrawingRequest
    from vectoreyes.drawing_trace import TraceOptions
    from vectoreyes.drawing_recipe import RecipeOperation
    image = np.full((136, 240, 3), 255, np.uint8)
    for start, end, color in ((8, 32, (200, 20, 20)), (40, 72, (20, 200, 20)),
            (80, 104, (20, 20, 200))):
        image[start:end, 12:228] = color
    engine = PythonDrawingEngine()
    baseline = engine.trace_drawing(TraceDrawingRequest('drw_two_bands',
        RasterImage(240, 136, 'rgb8', image.tobytes()),
        TraceOptions(max_sampling_colors=4, remove_background='white'),
        min_region_fraction=0, initial_stitch=False))
    refined = engine.refine_automatically(baseline.drawing, baseline.evidence,
        refinement_passes=only_bands())
    operations = [item for item in refined.evidence.recipe.steps
        if isinstance(item, RecipeOperation) and item.action.op == 'set_band']
    # The middle material is the largest and therefore r1. Its independent top
    # and bottom sides form two smaller gaps with the neighbouring materials.
    assert sum('r1' in {item.action.first.region, item.action.second.region}
        for item in operations) == 2


def curved_source():
    """A generated annular sector with exact source addresses, not raster guesses."""
    from dataclasses import replace
    from hashlib import sha256
    from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
    from vectoreyes.drawing_fill import FlatFill
    from vectoreyes.drawing_actions import ActionSettings
    from vectoreyes.drawing_engine import RasterImage
    from vectoreyes.drawing_recipe import build_geometry_evidence, build_paint_evidence, EffectiveDrawingRecipe
    from vectoreyes.drawing_trace import RootPathEvidence, TraceOptions, TraceResult
    from vectoreyes.path_geometry import ClosedPath, CornerAnchor, LineCommand, MoveCommand, PathGeometry
    from vectoreyes.path_fit_evidence import AutomaticFitEvidence, FitPathCorrespondence, FitSpanCorrespondence, FitBoundaryCorrespondence, FitEndpointCorrespondence
    from vectoreyes.replay_state import ReplayState
    from vectoreyes.drawing_bands import BandSourceContext
    theta = np.linspace(-0.6, 0.6, 41)
    first = np.c_[np.cos(theta), np.sin(theta)] * 100 + (120, 120)
    second = np.c_[np.cos(theta[::-1]), np.sin(theta[::-1])] * 112 + (120, 120)
    points = np.vstack([first, second, first[:1]])
    commands = (MoveCommand('r1.p0.c0', points[0]), *(LineCommand(f'r1.p0.c{i}', p) for i, p in enumerate(points[1:], 1)))
    corners = (0, 40, 41, 81)
    path = PathGeometry(ClosedPath('r1.p0', commands), tuple(CornerAnchor(f'r1.p0.corner{i}', f'r1.p0.c{j}', f'r1.p0.c{j}') for i, j in enumerate(corners)))
    source = PathGeometry(ClosedPath('source.p0', tuple(replace(cmd, id=f'source.p0.c{i}') for i, cmd in enumerate(commands))),
        tuple(CornerAnchor(f'source.corner{i}', f'source.p0.c{j}', f'source.p0.c{j}') for i, j in enumerate(corners)))
    correspondence = FitPathCorrespondence(tuple(FitBoundaryCorrespondence(f'source.corner{i}', 'point',
        f'source.p0.c{j or 82}', f'r1.p0.c{j}') for i, j in enumerate(corners)), tuple(FitSpanCorrespondence('side',
        tuple(f'source.p0.c{i}' for i in range(a+1, b+1)), tuple(f'r1.p0.c{i}' for i in range(a+1, b+1)))
        for a, b in ((0, 40), (40, 41), (41, 81), (81, 82))),
        tuple(FitEndpointCorrespondence(f'source.p0.c{i or 82}', f'r1.p0.c{i}') for i in range(len(commands))))
    fit = AutomaticFitEvidence('curved-fixture', source, path.path.id,
        sha256(path.to_svg_d().encode()).hexdigest(), correspondence)
    drawing = Drawing('drw_curve', Canvas(256, 256), (DrawingRegion('r1', path, FlatFill('#112233'), source_region_ids=('r1',)),))
    raster = RasterImage(256, 256, 'rgb8', bytes([255]) * (256*256*3))
    trace = TraceResult(256, 256, TraceOptions(), (), '<svg/>', root_path_evidence=(RootPathEvidence('r1', path.path.id, fit),))
    metadata = {'r1': {'path_ids': (path.path.id,), 'fill_rule': 'nonzero'}}
    observations = {'r1': {'id': 1, 'ancestry': (1,), 'original_kind': None,
        'original_params': {}, 'source_label': 1, 'color_hex': '#112233',
        'raster': bytes(256 * 256), 'raster_shape': (256, 256), 'raster_source': None,
        'coverage': None, 'coverage_shape': None, 'coverage_dtype': None,
        'coverage_origin': None, 'coverage_canvas': None, 'coverage_source': None}}
    evidence = build_geometry_evidence(kind='source', trace=trace, geometry_source=raster,
        drawing=drawing, shape_metadata=metadata, region_evidence=observations)
    paint = build_paint_evidence(raster)
    recipe = EffectiveDrawingRecipe(evidence.revision, evidence.engine_revision,
        ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    return ReplayState(drawing, metadata, observations), BandSourceContext(evidence, paint, recipe)


def test_generated_curved_band_publishes_and_preserves_exact_source_receipts():
    from vectoreyes.drawing_refinement import refine_canonical_drawing
    from vectoreyes.drawing_automatic import replay_automatic_batches
    from vectoreyes.drawing_replay import _path_record_for_display, _original_source_span
    from dataclasses import replace
    state, context = curved_source()
    result = refine_canonical_drawing(state, epsilon=1, max_error=2,
        refinement_passes=only_bands(), band_source_context=context)
    assert result.batches, dict(result.passes)['bands_pass'].diagnostics
    final = result.state.drawing.region('r1').geometry
    before = state.drawing.region('r1').geometry
    assert len(final.path.commands) < len(before.path.commands)
    assert replay_automatic_batches(state, result.batches).drawing == result.state.drawing
    for prior, current in zip(before.anchors, final.anchors, strict=True):
        assert before.endpoint(prior.entry_command_id) == final.endpoint(current.entry_command_id)
    from vectoreyes.drawing_recipe import ResolveOriginalPaintOperation
    recipe = replace(context.recipe, steps=(*result.batches,
        ResolveOriginalPaintOperation(scope='all', share_root_gradients=True)))
    record = _path_record_for_display(context.evidence, result.state, 'r1', final.path.id, [], recipe)
    # A subsequent side edit resolves the exact original samples from the new
    # displayed corner endpoints, not the reconstructed current cubics.
    points = _original_source_span(context.evidence, record, final,
        final.anchors[0].exit_command_id, final.anchors[1].entry_command_id)
    source_commands = context.evidence.root_paths[0].fit.source.path.commands
    expected = np.array([cmd.end for cmd in (source_commands[-1], *source_commands[:41])])
    np.testing.assert_array_equal(points, expected)
    from vectoreyes.drawing_actions import DrawingPlan
    from vectoreyes.drawing_replay import extend_recipe_with_plan, propose_effective_recipe
    plan = DrawingPlan(version='vectoreyes.plan.v1', drawing_id=state.drawing.drawing_id,
        base_version='v1', ops=({'op': 'update_geometry', 'target': 'r1', 'ops': ({'op': 'fit',
            'between': (final.anchors[0].exit_command_id, final.anchors[1].entry_command_id),
            'strategy': 'progressive'},)},))
    extension = extend_recipe_with_plan(evidence=context.evidence, paint=context.paint,
        base_recipe=recipe, plan=plan)
    edited = propose_effective_recipe(drawing_id=state.drawing.drawing_id,
        evidence=context.evidence, paint=context.paint, recipe=extension.recipe)
    assert edited.replay.path_correspondence


def test_missing_original_source_is_rejected_even_with_plausible_current_curves():
    from dataclasses import replace
    from vectoreyes.drawing_bands import discover_band_actions, set_band
    state, context = curved_source()
    context = replace(context, evidence=replace(context.evidence,
        trace=replace(context.evidence.trace, root_path_evidence=())))
    with pytest.raises(ValueError, match='source correspondence'):
        set_band(state, next(discover_band_actions(state)), context)


def test_band_scene_guard_rejects_new_overlap_without_relying_on_native_is_valid():
    from dataclasses import replace
    from vectoreyes.drawing import DrawingRegion
    from vectoreyes.path_geometry import geometry_from_svg_d
    from vectoreyes.drawing_bands import _guard_scene
    state, _ = curved_source()
    second = DrawingRegion('r2', geometry_from_svg_d('M0 0 L10 0 L10 10 L0 10 Z', region_id='r2'))
    before = replace(state.drawing, regions=(*state.drawing.regions, second))
    moved = replace(second, geometry=geometry_from_svg_d('M200 100 L230 100 L230 140 L200 140 Z', region_id='r2'))
    after = replace(before, regions=(before.regions[0], moved))
    with pytest.raises(ValueError, match='overlap'):
        _guard_scene(before, after, {'r2'})


def test_selected_band_replay_rejects_a_changed_canonical_input():
    from dataclasses import replace
    from vectoreyes.drawing_refinement import refine_canonical_drawing
    from vectoreyes.drawing_automatic import replay_automatic_batches
    state, context = curved_source()
    result = refine_canonical_drawing(state, epsilon=1, max_error=2,
        refinement_passes=only_bands(), band_source_context=context)
    altered = replace(state, drawing=replace(state.drawing,
        regions=(replace(state.drawing.regions[0], z=1),)))
    with pytest.raises(ValueError, match='input revision'):
        replay_automatic_batches(altered, result.batches)


def test_explicit_band_never_consumes_original_corner_owned_commands():
    from dataclasses import replace
    from vectoreyes.path_geometry import CornerAnchor
    from vectoreyes.drawing_bands import discover_band_actions, set_band
    state, context = curved_source()
    root = context.evidence.root_paths[0]
    source = root.fit.source
    rounded = CornerAnchor('source.rounded', 'source.p0.c10', 'source.p0.c20',
        tuple(f'source.p0.c{i}' for i in range(11, 21)))
    root = replace(root, fit=replace(root.fit, source=replace(source, anchors=(*source.anchors, rounded))))
    context = replace(context, evidence=replace(context.evidence,
        trace=replace(context.evidence.trace, root_path_evidence=(root,))))
    with pytest.raises(ValueError, match='original source corner-owned'):
        set_band(state, next(discover_band_actions(state)), context)


def test_band_scene_guard_rejects_a_new_hole():
    from dataclasses import replace
    from vectoreyes.path_geometry import PathGeometryWithCutouts, geometry_from_svg_d
    from vectoreyes.drawing_bands import _guard_scene
    state, _ = curved_source()
    outer = geometry_from_svg_d('M0 0 L100 0 L100 100 L0 100 Z', region_id='outer')
    cutout = geometry_from_svg_d('M20 20 L20 40 L40 40 L40 20 Z', region_id='cutout')
    changed = replace(state.drawing.regions[0], geometry=PathGeometryWithCutouts(outer, (cutout,)))
    after = replace(state.drawing, regions=(changed,))
    with pytest.raises(ValueError, match='component or hole topology'):
        _guard_scene(state.drawing, after, {'r1'})


def test_pipeline_reports_why_source_less_band_stage_was_skipped():
    from vectoreyes.pipeline import Options, _idealize_refinement
    image = np.full((48, 80, 3), 255, np.uint8)
    image[12:36, 8:72] = (20, 100, 240)
    _, report = _idealize_refinement(image,
        Options(refinement=True, refinement_passes=only_bands()), 80, 48)
    assert report.diagnostics.to_dict()['refinement_passes']['bands_pass']['reason'] == (
        'original_source_context_unavailable')


def guard_scene(d, *others):
    from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
    from vectoreyes.path_geometry import parse_svg_paths
    return Drawing('guard', Canvas(256, 256),
        (DrawingRegion('r1', parse_svg_paths(d, region_id='r1')[0]), *others))


def test_band_guard_rejects_square_to_bow_tie():
    from vectoreyes.drawing_bands import _guard_scene
    before = guard_scene('M0 0 L10 0 L10 10 L0 10 Z')
    after = guard_scene('M0 0 L10 10 L0 10 L10 0 Z')
    with pytest.raises(ValueError, match='nonintersection'):
        _guard_scene(before, after, {'r1'})


def test_band_guard_rejects_corner_fixed_crossing_within_source_error_budget():
    from vectoreyes.drawing_bands import _guard_scene
    from vectoreyes.band_offsets import offset_polyline
    from vectoreyes.quality import polyline_boundary_distance
    before = guard_scene('M9 0 L10 0 L10 10 L9 10 Z')
    after = guard_scene('M9 0 L10 0 C8.4 4 8.4 6 10 10 L9 10 Z')
    curve = (((10, 0), (8.4, 4), (8.4, 6), (10, 10)),)
    polyline, bound = offset_polyline(curve, tolerance=0.005)
    source_error = polyline_boundary_distance([((10, 0), (10, 10))], [polyline],
        closed=False, precision=0.005, direction='replacement_to_source') + bound
    assert source_error < 1.5
    assert {cmd.end for cmd in before.regions[0].geometry.path.commands} == {
        cmd.end for cmd in after.regions[0].geometry.path.commands}
    with pytest.raises(ValueError, match='nonintersection'):
        _guard_scene(before, after, {'r1'})


def test_band_guard_includes_translated_clone_against_unrelated_obstacle():
    from vectoreyes.drawing import DrawingRegion
    from vectoreyes.path_geometry import CloneGeometry, AffineTransform, parse_svg_paths
    from vectoreyes.drawing_bands import _guard_scene
    clone = DrawingRegion('r2', CloneGeometry('r1', (AffineTransform(tx=100),)))
    obstacle = DrawingRegion('r3', parse_svg_paths(
        'M110.5 4 L111.5 4 L111.5 6 L110.5 6 Z', region_id='r3')[0])
    before = guard_scene('M0 0 L10 0 L10 10 L0 10 Z', clone, obstacle)
    after = guard_scene('M0 0 L10 0 L11 5 L10 10 L0 10 Z', clone, obstacle)
    with pytest.raises(ValueError, match='overlap'):
        _guard_scene(before, after, {'r1'})


def test_band_guard_checks_dependent_clone_component_topology():
    from vectoreyes.drawing import DrawingRegion
    from vectoreyes.path_geometry import CloneGeometry, AffineTransform
    from vectoreyes.drawing_bands import _guard_scene
    clone = DrawingRegion('r2', CloneGeometry('r1',
        (AffineTransform(tx=100), AffineTransform(tx=111))))
    before = guard_scene('M0 0 L10 0 L10 10 L0 10 Z', clone)
    after = guard_scene('M0 0 L10 0 L12 5 L10 10 L0 10 Z', clone)
    with pytest.raises(ValueError, match='component or hole topology'):
        _guard_scene(before, after, {'r1'})


def test_source_ranked_band_admission_skips_crossed_best_candidate():
    engine, baseline = traced_band()
    refined = engine.refine_automatically(baseline.drawing, baseline.evidence,
        refinement_passes=only_bands())
    stage = next(item for item in refined.passes if item.name == 'bands_pass')
    assert stage.status == 'changed', stage.diagnostics
    report = stage.result.drawing.region('r1').diagnostics['bands']
    rejected = report['rejected_candidates']
    assert rejected and 'nonintersection' in rejected[0]['reason']
    assert max(report['source_errors']) >= max(rejected[0]['source_errors'])
    assert (report['transition_fractions'], report['primary_side']) != (
        rejected[0]['transition_fractions'], rejected[0]['primary_side'])


def test_all_scene_invalid_candidates_reject_without_mutating_input(monkeypatch):
    from vectoreyes.drawing_bands import BandSourceContext, discover_band_actions, set_band
    from vectoreyes.drawing_replay import _replay_drawing_with_producers
    import vectoreyes.drawing_bands as bands
    engine, baseline = traced_band()
    state, _, corners = _replay_drawing_with_producers(baseline.evidence.geometry,
        baseline.evidence.paint, baseline.evidence.recipe, finalize_output=False)
    original = bands._numeric_selection
    def only_best(*args, **kwargs):
        return original(*args, **kwargs)[:1]
    monkeypatch.setattr(bands, '_numeric_selection', only_best)
    before = state.drawing.regions
    context = BandSourceContext(baseline.evidence.geometry, baseline.evidence.paint,
        baseline.evidence.recipe, tuple(corners), engine)
    with pytest.raises(ValueError, match='no scene-safe band candidate.*nonintersection'):
        set_band(state, next(discover_band_actions(state)), context)
    assert state.drawing.regions == before

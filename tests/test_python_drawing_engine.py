from dataclasses import replace

import numpy as np
import pytest

from vectormark.drawing_actions import DrawingPlan
from vectormark.drawing_engine import TraceDrawingRequest, RasterImage
from vectormark.optimizer.assembly_binding import working_regions_from_evidence
from vectormark.drawing_recipe import (
    EffectiveDrawingRecipe, ResolveOriginalPaintOperation,
    build_paint_evidence, drawing_revision, )
from vectormark.drawing_refine import stitch_regions_with_intent
from vectormark.optimizer.assembly_binding import assemble_working_roots
from vectormark.drawing_replay import propose_effective_recipe
from vectormark.drawing_actions import ActionSettings
from vectormark.drawing_trace import PythonTraceEngine, TraceOptions
from vectormark.python_drawing_engine import PythonDrawingEngine


def source():
    rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
    rgb[6:26, 6:26] = (20, 80, 140)
    return rgb, RasterImage(32, 32, "rgb8", rgb.tobytes())


def test_automatic_refinement_records_before_final_output_palette():
    from vectormark.optimizer.replay_binding import replay_working_regions
    from vectormark.output_settings import DrawingOutputSettings
    from vectormark.optimizer.automatic_replay import input_revision

    _, raster = source()
    engine = PythonDrawingEngine()
    baseline = engine.trace_drawing(TraceDrawingRequest(
        "drw_palette", raster,
        TraceOptions(max_sampling_colors=2, remove_background="off"),
        min_region_fraction=0, initial_stitch=False,
        output_settings=DrawingOutputSettings(max_output_colors=1, allow_gradients=False),
    ))
    state = baseline.evidence
    before_output = replay_working_regions(state.geometry, state.paint, state.recipe, finalize_output=False)
    after_output = replay_working_regions(state.geometry, state.paint, state.recipe)
    assert input_revision(list(before_output)) != input_revision(list(after_output))

    refined = engine.refine_automatically(baseline.drawing, state)
    replayed = propose_effective_recipe(drawing_id="drw_palette", evidence=state.geometry,
        paint=state.paint, recipe=refined.evidence.recipe)
    assert drawing_revision(replayed.drawing) == drawing_revision(refined.drawing)
    assert refined.drawing.output_settings.max_output_colors == 1


@pytest.mark.parametrize("strategy", ["quadratic", "progressive", "progressive_allow_lines"])
@pytest.mark.parametrize("stitch", [False, True])
@pytest.mark.parametrize("trace_level", ["pixel", "subpixel"])
def test_engine_v0_matches_existing_operations(strategy, stitch, trace_level):
    from vectormark.drawing_assembly import assemble_drawing_roots
    from vectormark.drawing_recipe import build_geometry_evidence
    rgb, raster = source()
    options = TraceOptions(max_sampling_colors=2, remove_background="off", fit_strategy=strategy, trace_level=trace_level)
    trace = PythonTraceEngine().trace(rgb, options, geometry_rgb=rgb)
    rooted = assemble_drawing_roots(trace, rgb, min_region_fraction=0, allow_gradients=False)
    evidence = build_geometry_evidence(kind="source", trace=rooted.trace,
        geometry_source=raster, drawing=rooted.drawing,
        shape_metadata=rooted.shape_metadata, region_evidence=rooted.region_evidence)
    from vectormark.output_settings import DrawingOutputSettings
    output = DrawingOutputSettings(allow_gradients=False)
    paint = build_paint_evidence(raster, output_settings=output)
    settings = ActionSettings(epsilon=options.simplify_tolerance,
        max_error=options.curve_tolerance, fit_strategy=strategy)
    stitched = stitch_regions_with_intent(rooted.trace, working_regions_from_evidence(evidence), enabled=stitch)
    recipe = EffectiveDrawingRecipe(evidence.revision, evidence.engine_revision, settings,
        steps=(*stitched.batches, ResolveOriginalPaintOperation(scope="retain", share_root_gradients=True)))
    expected = propose_effective_recipe(drawing_id="drw_engine", evidence=evidence, paint=paint, recipe=recipe)
    result = PythonDrawingEngine().trace_drawing(TraceDrawingRequest("drw_engine", raster, options,
        min_region_fraction=0, initial_stitch=stitch, output_settings=output))
    assert drawing_revision(result.drawing) == drawing_revision(expected.drawing)
    assert result.evidence.recipe == recipe
    assert result.evidence.replay == expected.replay


def test_engine_plan_uses_existing_executor_and_rejects_stale_scene():
    _, raster = source()
    engine = PythonDrawingEngine()
    baseline = engine.trace_drawing(TraceDrawingRequest("drw_engine", raster,
        TraceOptions(max_sampling_colors=2, remove_background="off"), min_region_fraction=0))
    target = baseline.drawing.regions[0].leaves()[0].id
    plan = DrawingPlan.model_validate(dict(version="vectormark.plan.v1", drawing_id="drw_engine",
        base_version="v0", ops=[dict(op="set_fill", target=target,
            fill=dict(type="flat", color="#FF0000"))]))
    result = engine.apply_plan(baseline.drawing, baseline.evidence, plan)
    assert result.drawing.regions[0].leaves()[0].fill != baseline.drawing.regions[0].leaves()[0].fill
    with pytest.raises(ValueError, match="output revision mismatch"):
        engine.apply_plan(result.drawing, baseline.evidence, plan)
    with pytest.raises(ValueError, match="drawing ID"):
        engine.apply_plan(baseline.drawing, baseline.evidence,
            plan.model_copy(update={"drawing_id": "drw_other"}))


def test_corner_replay_uses_engine_entry_point_by_default():
    from vectormark.corner_recipe import CornerRecipeExecutor
    assert isinstance(CornerRecipeExecutor()._path_fitter, PythonDrawingEngine)


def test_standalone_rgba_trace_uses_alpha_but_retains_composited_paint():
    from PIL import Image
    from vectormark.pipeline import _flatten_on_white
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[6:26, 6:26] = (20, 80, 140, 255)
    engine = PythonDrawingEngine()
    options = TraceOptions(max_sampling_colors=2, source_has_alpha=True)
    result = engine.trace_drawing(TraceDrawingRequest("drw_rgba",
        RasterImage(32, 32, "rgba8", rgba.tobytes()), options, min_region_fraction=0))
    rgb = _flatten_on_white(Image.fromarray(rgba))
    expected_trace = engine.trace_polylines(rgb, options, alpha=rgba[:, :, 3], geometry_rgb=rgb)
    expected = engine.fit_trace("drw_rgba", expected_trace, RasterImage(32, 32, "rgb8", rgb.tobytes()),
        min_region_fraction=0)
    assert result.evidence.paint.source.pixels == rgb.tobytes()
    assert drawing_revision(result.drawing) == drawing_revision(expected.drawing)


def test_path_action_and_drawing_action_share_plan_executor():
    from vectormark.drawing_engine import DrawingActionRequest, PathActionRequest
    from vectormark.drawing_actions import FitPathAction, UpdateGeometryAction
    from vectormark.path_geometry import PathGeometry
    _, raster = source()
    engine = PythonDrawingEngine()
    baseline = engine.trace_drawing(TraceDrawingRequest("drw_engine", raster,
        TraceOptions(max_sampling_colors=2, remove_background="off"), min_region_fraction=0))
    target = next(leaf for root in baseline.drawing.regions for leaf in root.leaves()
        if isinstance(leaf.geometry, PathGeometry))
    action = FitPathAction(op="fit", target=target.geometry.path.drawable_commands[0].id, type="keep")
    settings = baseline.evidence.recipe.settings
    path_result = engine.apply_path_action(PathActionRequest(baseline.drawing, target.id,
        action, settings, baseline.evidence, "v0.3"))
    drawing_result = engine.apply_action(DrawingActionRequest(baseline.drawing, baseline.evidence,
        UpdateGeometryAction(op="update_geometry", target=target.id, ops=(action,)), settings, "v0.3"))
    actual = next(leaf for root in drawing_result.drawing.regions for leaf in root.leaves() if leaf.id == target.id)
    assert path_result.geometry == actual.geometry
    assert path_result.affected_command_ids == ()


def test_source_pretrace_labels_keep_existing_merge_policy():
    from vectormark.material_pretrace import MaterialPretraceResult
    yy, xx = np.mgrid[:100, :120]
    inside = (yy > 9) & (yy < 90) & (xx > 9) & (xx < 110)
    left = xx < 60 + 18 * np.sin(yy * np.pi / 10)
    rgb = np.full((100, 120, 3), 255, dtype=np.uint8)
    rgb[inside & left] = (6, 35, 54)
    rgb[inside & ~left] = (120, 140, 160)
    raster = RasterImage(120, 100, "rgb8", rgb.tobytes())
    options = TraceOptions(max_sampling_colors=3, remove_background="off", min_region_size=1)
    pretrace = MaterialPretraceResult(rgb, rgb, rgb, rgb, 3, 3, (), 3)
    trace = replace(PythonTraceEngine().trace(rgb, options), material_pretrace=pretrace)
    assert len(assemble_working_roots(trace, rgb, min_region_fraction=0).regions) == 2
    result = PythonDrawingEngine().fit_trace("drw_proxy", trace, raster,
        min_region_fraction=0, initial_stitch=False)
    assert len(result.drawing.regions) == 2


def test_geometry_guide_keeps_quantized_boundary_classification():
    rgb = np.full((40, 40, 3), 255, dtype=np.uint8)
    rgb[8:32, 8:20] = (10, 40, 80)
    rgb[8:32, 20:32] = (235, 235, 235)
    options = TraceOptions(max_sampling_colors=2, source_has_alpha=False,
        remove_background="auto", min_region_size=1)
    expected = PythonTraceEngine().trace(rgb, options, alpha=None)
    actual = PythonDrawingEngine().trace_polylines(rgb, options, alpha=None)
    assert [int(region.mask.sum()) for region in actual.geometry_regions] == [
        int(region.mask.sum()) for region in expected.geometry_regions]


def test_action_region_changes_include_branch_identity(monkeypatch):
    import vectormark.python_drawing_engine as engine_module
    from vectormark.drawing import DrawingRegion
    _, raster = source()
    engine = PythonDrawingEngine()
    baseline = engine.trace_drawing(TraceDrawingRequest("drw_engine", raster,
        TraceOptions(max_sampling_colors=2, remove_background="off"), min_region_fraction=0))
    original = engine_module.propose_effective_recipe

    def grouped_proposal(**kwargs):
        execution = original(**kwargs)
        drawing = replace(execution.drawing, regions=(DrawingRegion("group", children=execution.drawing.regions),))
        return replace(execution, drawing=drawing,
            replay=replace(execution.replay, output_revision=drawing_revision(drawing)))

    monkeypatch.setattr(engine_module, "propose_effective_recipe", grouped_proposal)
    plan = DrawingPlan.model_validate(dict(version="vectormark.plan.v1", drawing_id="drw_engine",
        base_version="v0", ops=[]))
    result = engine.apply_plan(baseline.drawing, baseline.evidence, plan)
    assert result.drawing.regions[0].is_branch
    assert result.created_region_ids == ("group",)
    assert result.retired_region_ids == ()

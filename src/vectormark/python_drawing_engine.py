"""Python operation entry point; transport and version publication live elsewhere.

The engine orchestrates the existing trace, fitting and recipe executors. It
does not own a second scene, store, fitter, or action vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypedDict

import numpy as np

from .background import BackgroundColor
from .color import infer_max_sampling_colors, white_canvas_mask
from .drawing import Drawing, _walk_regions
from .drawing_actions import (
    ActionSettings, ActionSettingsOverride, DrawingPlan, UpdateGeometryAction,
)
from .drawing_engine import (
    DrawingActionRequest, DrawingActionResult, FitPathRequest, FitPathResult,
    PathActionRequest, PathActionResult, RasterImage, TraceDrawingRequest, TraceDrawingResult,
)
from .drawing_recipe import (
    EffectiveDrawingRecipe, GeometryEvidence, PaintEvidence, ReplayRecord, ResolvedCornerEdit,
    ResolveOriginalPaintOperation, build_geometry_evidence, build_paint_evidence,
    validate_replay_record,
)
from .drawing_refine import auto_refine_with_intent, drawing_summary, root_regions, stitch_regions_with_intent
from .drawing_replay import (
    extend_recipe_with_plan, propose_effective_recipe, raster_pixels,
)
from .drawing_trace import PythonTraceEngine, TraceOptions, TraceResult
from .material_pretrace import preprocess_material_trace
from .output_settings import DrawingOutputSettings
from .path_geometry import PathGeometry
from .python_path_fitter import PythonPathFitter
from .refinement_settings import RefinementPasses


@dataclass(frozen=True)
class PythonEngineEvidence:
    """Existing source/paint/recipe receipts, without another authoritative scene."""

    geometry: GeometryEvidence
    paint: PaintEvidence
    recipe: EffectiveDrawingRecipe
    replay: ReplayRecord


@dataclass(frozen=True)
class PythonPlanResult(DrawingActionResult[PythonEngineEvidence]):
    submitted_edits: tuple[ResolvedCornerEdit, ...] = ()


class _RegionChanges(TypedDict):
    affected_region_ids: tuple[str, ...]
    created_region_ids: tuple[str, ...]
    retired_region_ids: tuple[str, ...]


def _region_changes(before: Drawing, after: Drawing) -> _RegionChanges:
    old = {region.id: region for root in before.regions for region in _walk_regions(root)}
    new = {region.id: region for root in after.regions for region in _walk_regions(root)}
    return {
        "affected_region_ids": tuple(key for key in new if key in old and new[key] != old[key]),
        "created_region_ids": tuple(key for key in new if key not in old),
        "retired_region_ids": tuple(key for key in old if key not in new),
    }


class PythonDrawingEngine:
    """Shared deterministic operations for v0, automatic and agent orchestration."""

    def fit_path(self, request: FitPathRequest) -> FitPathResult:
        return PythonPathFitter().fit_path(request)

    def trace_polylines(
        self, rgb: np.ndarray, options: TraceOptions, *,
        alpha: np.ndarray | None = None, gradient_pretrace: bool = False,
        geometry_rgb: np.ndarray | None = None,
    ) -> TraceResult:
        """Trace prepared pixels with the same material proxy and palette policy."""
        pretrace = (
            preprocess_material_trace(rgb, min_region_size=options.min_region_size)
            if gradient_pretrace else None
        )
        trace_rgb = pretrace.material_proxy if pretrace is not None else rgb
        white_canvas = white_canvas_mask(trace_rgb)
        automatic = options.requested_max_sampling_colors == "auto"
        colors = (
            pretrace.resolved_max_sampling_colors if automatic and pretrace is not None
            else infer_max_sampling_colors(trace_rgb, exclude_mask=white_canvas) if automatic
            else options.max_sampling_colors
        )
        trace = PythonTraceEngine().trace(
            trace_rgb,
            replace(options, max_sampling_colors=colors,
                reserve_white_background=automatic and white_canvas is not None
                if automatic else options.reserve_white_background),
            alpha=alpha, geometry_rgb=geometry_rgb,
        )
        return replace(trace, material_pretrace=pretrace) if pretrace is not None else trace

    def trace_drawing(self, request: TraceDrawingRequest) -> TraceDrawingResult[PythonEngineEvidence]:
        channels = 3 if request.source.pixel_format == "rgb8" else 4
        pixels = np.frombuffer(request.source.pixels, dtype=np.uint8).reshape(
            request.source.height, request.source.width, channels)
        alpha = None
        if channels == 4:
            # Match image preparation: geometry sees alpha; paint sees white-composited RGB.
            from PIL import Image
            from .pipeline import _flatten_on_white
            alpha = pixels[:, :, 3].copy()
            rgb = np.asarray(_flatten_on_white(Image.fromarray(pixels)), dtype=np.uint8)
        else:
            rgb = pixels
        source = RasterImage(request.source.width, request.source.height, "rgb8", rgb.tobytes())
        trace = self.trace_polylines(rgb, request.options, alpha=alpha,
            gradient_pretrace=request.gradient_pretrace, geometry_rgb=rgb)
        return self.fit_trace(request.drawing_id, trace, source,
            min_region_fraction=request.min_region_fraction,
            initial_stitch=request.initial_stitch, output_settings=request.output_settings)

    def fit_trace(
        self, drawing_id: str, trace: TraceResult, source: RasterImage, *,
        min_region_fraction: float = 0.02, initial_stitch: bool = True,
        output_settings: DrawingOutputSettings | None = None,
        kind: Literal["source", "geometry-guide"] = "source",
        paint: PaintEvidence | None = None,
    ) -> TraceDrawingResult[PythonEngineEvidence]:
        """Fit traced regions and stitch using the existing operation implementations."""
        output_settings = output_settings or DrawingOutputSettings()
        rgb = raster_pixels(source)
        rooted = root_regions(trace, rgb, min_region_fraction=min_region_fraction,
            preserve_material_labels=True if kind == "geometry-guide" else None,
            allow_gradients=output_settings.allow_gradients if kind == "source" else False)
        trace = rooted.trace
        background = (
            BackgroundColor(color=str(trace.background["color"]), tolerance=float(trace.background["tolerance"]))
            if trace.background.get("applied") and trace.background.get("color") else None
        )
        if kind == "geometry-guide" and paint is None:
            raise ValueError("geometry-guide tracing requires retained source paint")
        paint = paint or build_paint_evidence(source, background=background, output_settings=output_settings)
        geometry = build_geometry_evidence(kind=kind, trace=trace,
            geometry_source=source, trace_roots=rooted.regions)
        settings = ActionSettings(epsilon=trace.options.simplify_tolerance,
            max_error=trace.options.curve_tolerance, fit_strategy=trace.options.fit_strategy)
        stitched = stitch_regions_with_intent(trace, geometry.trace_roots, enabled=initial_stitch)
        recipe = EffectiveDrawingRecipe(geometry.revision, geometry.engine_revision, settings,
            steps=(*stitched.batches, ResolveOriginalPaintOperation(scope="retain" if kind == "source" else "all",
                share_root_gradients=trace.material_pretrace is None)))
        execution = propose_effective_recipe(drawing_id=drawing_id,
            evidence=geometry, paint=paint, recipe=recipe)
        return TraceDrawingResult(execution.drawing,
            PythonEngineEvidence(geometry, paint, recipe, execution.replay),
            diagnostics=drawing_summary(trace, stitched.regions))

    def refine_automatically(
        self, drawing: Drawing, evidence: PythonEngineEvidence, *,
        corner_normalize: bool = False, refinement_passes: RefinementPasses | None = None,
    ) -> DrawingActionResult[PythonEngineEvidence]:
        """Orchestrate existing optimizer passes and retain their accepted operations."""
        validate_replay_record(drawing=drawing, evidence=evidence.geometry,
            paint=evidence.paint, recipe=evidence.recipe, replay=evidence.replay)
        baseline = propose_effective_recipe(drawing_id=drawing.drawing_id,
            evidence=evidence.geometry, paint=evidence.paint, recipe=evidence.recipe)
        trace = evidence.geometry.trace
        automatic = auto_refine_with_intent(trace, baseline.regions,
            rgb=raster_pixels(evidence.paint.source) if evidence.geometry.kind == "source" else None,
            allow_gradients=evidence.paint.output_settings.allow_gradients,
            corner_normalize=corner_normalize, refinement_passes=refinement_passes)
        recipe = replace(evidence.recipe, steps=(*evidence.recipe.steps, *automatic.batches,
            ResolveOriginalPaintOperation(scope="all", share_root_gradients=trace.material_pretrace is None)))
        execution = propose_effective_recipe(drawing_id=drawing.drawing_id,
            evidence=evidence.geometry, paint=evidence.paint, recipe=recipe)
        return DrawingActionResult(execution.drawing,
            replace(evidence, recipe=recipe, replay=execution.replay),
            **_region_changes(drawing, execution.drawing))

    def apply_plan(
        self, drawing: Drawing, evidence: PythonEngineEvidence, plan: DrawingPlan,
    ) -> PythonPlanResult:
        """Execute the existing plan vocabulary against one receipt-validated scene."""
        if drawing.drawing_id != plan.drawing_id:
            raise ValueError("plan drawing ID does not match input drawing")
        validate_replay_record(drawing=drawing, evidence=evidence.geometry,
            paint=evidence.paint, recipe=evidence.recipe, replay=evidence.replay)
        extension = extend_recipe_with_plan(evidence=evidence.geometry, paint=evidence.paint,
            base_recipe=evidence.recipe, plan=plan)
        execution = propose_effective_recipe(drawing_id=drawing.drawing_id,
            evidence=evidence.geometry, paint=evidence.paint, recipe=extension.recipe)
        return PythonPlanResult(execution.drawing,
            replace(evidence, recipe=extension.recipe, replay=execution.replay),
            **_region_changes(drawing, execution.drawing),
            submitted_edits=extension.submitted_edits)

    def apply_action(
        self, request: DrawingActionRequest[PythonEngineEvidence],
    ) -> DrawingActionResult[PythonEngineEvidence]:
        plan = DrawingPlan(version="vectormark.plan.v1", drawing_id=request.drawing.drawing_id,
            base_version=request.base_version, settings=ActionSettingsOverride(**request.settings.model_dump()),
            ops=(request.action,))
        return self.apply_plan(request.drawing, request.evidence, plan)

    def apply_path_action(self, request: PathActionRequest[PythonEngineEvidence]) -> PathActionResult:
        target = next((leaf for root in request.drawing.regions for leaf in root.leaves()
            if leaf.id == request.region_id), None)
        if target is None or not isinstance(target.geometry, PathGeometry):
            raise ValueError("path action requires a single path region")
        result = self.apply_action(DrawingActionRequest(request.drawing, request.evidence,
            UpdateGeometryAction(op="update_geometry", target=request.region_id, ops=(request.action,)),
            request.settings, request.base_version))
        updated = next(leaf for root in result.drawing.regions for leaf in root.leaves()
            if leaf.id == request.region_id)
        if not isinstance(updated.geometry, PathGeometry):
            raise ValueError("path action did not produce a single path")
        before = {command.id: command for command in target.geometry.path.commands}
        return PathActionResult(updated.geometry,
            tuple(command.id for command in updated.geometry.path.commands
                if before.get(command.id) != command), result.diagnostics)

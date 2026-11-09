"""Python operation entry point; transport and version publication live elsewhere.

The engine orchestrates the existing trace, fitting and recipe executors. It
does not own a second scene, store, fitter, or action vocabulary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal, TypedDict

import numpy as np

from .background import BackgroundColor
from .bands import BandDetection, BandReconstruction, BandRequest, detect_band, reconstruct_band
from .band_offsets import BandOffsets
from .band_joins import BandEndpoints, BandJoinResult, join_band_offsets
from .boundary_detail import BoundaryDetailAssessment, BoundaryDetailRequest, assess_boundary_detail
from .color import infer_max_sampling_colors, white_canvas_mask
from .drawing import Drawing, _walk_regions
from .drawing_actions import (
    ActionSettings, ActionSettingsOverride, DrawingPlan, UpdateGeometryAction,
)
from .drawing_engine import (
    DetectPathFeaturesRequest, DetectPathFeaturesResult,
    AutomaticRefinementResult, DrawingPassResult,
    DrawingActionRequest, DrawingActionResult, FitPathRequest, FitPathResult,
    PathActionRequest, PathActionResult, RasterImage, TraceDrawingRequest, TraceDrawingResult,
)
from .refinement.assembly_binding import working_regions_from_evidence
from .drawing_recipe import (
    EffectiveDrawingRecipe, GeometryEvidence, PaintEvidence, ReplayRecord, ResolvedCornerEdit,
    ResolveOriginalPaintOperation, build_geometry_evidence, build_paint_evidence,
    validate_replay_record, )
from .drawing_refine import stitch_regions_with_intent
from .drawing_render import trace_summary
from .drawing_assembly import assemble_drawing_roots
from .drawing_replay import (
    extend_recipe_with_plan, propose_effective_recipe, raster_pixels,
    _finalize_replay_state, _record_execution, _replay_drawing_with_producers,
    _resolve_original_paint, )
from .drawing_trace import PythonTraceEngine, TraceOptions, TraceResult
from .material_pretrace import preprocess_material_trace
from .output_settings import DrawingOutputSettings
from .path_geometry import PathGeometry
from .python_path_fitter import PythonPathFitter, validate_explicit_fit_source
from .fit import _detect_path_features
from .path_fit_evidence import bind_source_path_features, source_node_command_id
from .refinement_settings import RefinementPasses, refinement_pass_names
from .quality import GeometryPolicy
from .revision_quality import QualityReport, RegionMatch, assess_revision


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

    def assess_boundary_detail(self, request: BoundaryDetailRequest) -> BoundaryDetailAssessment:
        return assess_boundary_detail(request)

    def detect_band(self, request: BandRequest) -> BandDetection:
        return detect_band(request)

    def reconstruct_band(self, request: BandRequest, *, width: float | None = None) -> BandReconstruction:
        return reconstruct_band(request, width=width)

    def join_band(self, request: BandRequest, offsets: BandOffsets, *,
                  endpoints: BandEndpoints | None = None) -> BandJoinResult:
        return join_band_offsets(offsets, request.first, request.second,
                                 max_error=request.max_error, endpoints=endpoints)

    def fit_path(self, request: FitPathRequest) -> FitPathResult:
        return PythonPathFitter().fit_path(request)

    def detect_path_features(self, request: DetectPathFeaturesRequest) -> DetectPathFeaturesResult:
        validate_explicit_fit_source(request.source)
        if request.source.anchors:
            raise ValueError("automatic detection cannot replace existing declarations; use explicit fitting")
        command_indices = {command.id: index for index, command in enumerate(request.source.path.commands[:-1])}
        command_indices[request.source.path.commands[-1].id] = 0
        if not set(request.forced_corner_command_ids).issubset(command_indices):
            raise ValueError("forced corner commands must belong to the source path")
        features = _detect_path_features(
            np.asarray([command.end for command in request.source.path.commands], dtype=float),
            epsilon=request.options.simplify_tolerance,
            progressive=request.options.fit_strategy != "quadratic",
            support_weighted_corners=request.options.support_weighted_corners,
            corner_rounding=request.options.corner_rounding,
            canvas_extent=request.canvas_extent,
            forced_corner_indices=tuple(command_indices[key] for key in request.forced_corner_command_ids),
        )
        binding = bind_source_path_features(request.source.path, features.point_indices, features.corner_ranges)
        semantic_boundaries = {
            command_id for anchor in binding.geometry.anchors
            for command_id in (anchor.entry_command_id, anchor.exit_command_id)
        }
        return DetectPathFeaturesResult(binding.geometry,
            tuple(command_id for index in features.cut_indices
                if (command_id := source_node_command_id(request.source.path, index)) not in semantic_boundaries),
            binding.diagnostics)

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
        rooted = assemble_drawing_roots(trace, rgb, min_region_fraction=min_region_fraction,
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
            geometry_source=source, drawing=rooted.drawing,
            shape_metadata=rooted.shape_metadata, region_evidence=rooted.region_evidence)
        settings = ActionSettings(epsilon=trace.options.simplify_tolerance,
            max_error=trace.options.curve_tolerance, fit_strategy=trace.options.fit_strategy)
        stitched = stitch_regions_with_intent(trace, working_regions_from_evidence(geometry), enabled=initial_stitch)
        recipe = EffectiveDrawingRecipe(geometry.revision, geometry.engine_revision, settings,
            steps=(*stitched.batches, ResolveOriginalPaintOperation(scope="retain" if kind == "source" else "all",
                share_root_gradients=trace.material_pretrace is None)))
        execution = propose_effective_recipe(drawing_id=drawing_id,
            evidence=geometry, paint=paint, recipe=recipe)
        return TraceDrawingResult(execution.drawing,
            PythonEngineEvidence(geometry, paint, recipe, execution.replay),
            diagnostics=trace_summary(execution.drawing, asdict(trace.options),
                execution.replay.path_correspondence))

    def assess_revision(
        self, reference: Drawing, before: Drawing, candidate: Drawing, *,
        policy: GeometryPolicy, reference_policy: GeometryPolicy | None = None,
        matches: Sequence[RegionMatch] | None = None,
    ) -> QualityReport:
        """Measure canonical revisions without fitting, mutation or paint scoring."""
        return assess_revision(reference, before, candidate, policy=policy,
            reference_policy=reference_policy, matches=matches)

    def refine_automatically(
        self, drawing: Drawing, evidence: PythonEngineEvidence, *,
        corner_normalize: bool = False, refinement_passes: RefinementPasses | None = None,
    ) -> AutomaticRefinementResult[PythonEngineEvidence]:
        """Orchestrate existing refinement passes and retain their accepted operations."""
        validate_replay_record(drawing=drawing, evidence=evidence.geometry,
            paint=evidence.paint, recipe=evidence.recipe, replay=evidence.replay)
        # Appended steps replay before the final output-palette projection.
        # Capture their inputs at that same boundary, not from the published fill.
        baseline, _producers, corner_replays = _replay_drawing_with_producers(
            evidence.geometry, evidence.paint, evidence.recipe, finalize_output=False)
        trace = evidence.geometry.trace
        from .drawing_refinement import refine_canonical_drawing
        from .drawing_bands import BandSourceContext
        automatic = refine_canonical_drawing(baseline,
            epsilon=trace.options.simplify_tolerance, max_error=trace.options.curve_tolerance,
            corner_normalize=corner_normalize, refinement_passes=refinement_passes,
            band_source_context=BandSourceContext(evidence.geometry, evidence.paint,
                evidence.recipe, tuple(corner_replays), self))
        observed = dict(automatic.passes)
        recipe = evidence.recipe
        passes: list[DrawingPassResult[PythonEngineEvidence]] = []
        previous = drawing
        for name in refinement_pass_names(RefinementPasses(), include_disabled=True):
            stage = observed.get(name)
            if stage is None:
                passes.append(DrawingPassResult(name, "disabled"))
                continue
            recipe = replace(recipe, steps=(*recipe.steps, *stage.batches))
            execution = _record_execution(drawing.drawing_id, evidence.geometry, evidence.paint,
                recipe, _finalize_replay_state(stage.state, evidence.paint), corner_replays)
            changed = execution.drawing != previous
            result = DrawingActionResult(execution.drawing,
                replace(evidence, recipe=recipe, replay=execution.replay),
                **_region_changes(previous, execution.drawing))
            passes.append(DrawingPassResult(name, "changed" if changed else "unchanged", result,
                diagnostics=stage.diagnostics))
            previous = execution.drawing
        paint_step = ResolveOriginalPaintOperation(scope="all",
            share_root_gradients=trace.material_pretrace is None)
        recipe = replace(recipe, steps=(*recipe.steps, paint_step))
        final_state = _finalize_replay_state(_resolve_original_paint(
            automatic.state, evidence.geometry, evidence.paint, paint_step), evidence.paint)
        execution = _record_execution(drawing.drawing_id, evidence.geometry, evidence.paint,
            recipe, final_state, corner_replays)
        final = DrawingActionResult(execution.drawing,
            replace(evidence, recipe=recipe, replay=execution.replay),
            **_region_changes(previous, execution.drawing))
        changed = execution.drawing != previous
        passes.append(DrawingPassResult("resolve_original_paint",
            "changed" if changed else "unchanged", final))
        return AutomaticRefinementResult(final.drawing, final.evidence,
            **_region_changes(drawing, final.drawing), passes=tuple(passes))

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
        plan = DrawingPlan(version="vectoreyes.plan.v1", drawing_id=request.drawing.drawing_id,
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

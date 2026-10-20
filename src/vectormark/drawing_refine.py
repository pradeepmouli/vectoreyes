"""Private working-evidence execution bridge for drawing plans.

Versions retain canonical Drawings. This module temporarily executes against
cached RegionSample evidence; canonical serialization lives in drawing_render.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from .drawing_actions import ActionSettings, SetZOrderAction
from .drawing_plan import DrawingPlan, PlanValidationError, validate_plan
from .drawing_adapter import publish_region_update, publish_drawing_update, drawing_from_region_samples, normalize_drawing_addresses, rebind_working_corner_provenance, restore_working_root_metadata, working_leaf_targets
from .drawing_trace import TraceResult
from .refinement.region_sample import RegionSample
from .refinement.automatic_replay import (
    AutomaticActionOutcome,
    ReplayBatch,
    select_refinements_with_intent,
)
from .refinement.passes import corner_normalize_pass, clones_pass, primitives_pass, seams_pass, simplify_pass, split_compound_pass, symmetry_pass
from .refinement.symmetry_materialize import materialize_self_symmetry
from .refinement.topology_binding import normalize_working_topology
from .refinement_settings import RefinementPasses
from .refinement.paint_binding import refill_changed_regions_from_source
from . import drawing_region_actions, drawing_relationships, path_operations, drawing_partition


DrawingRegions = tuple[RegionSample, ...]


@dataclass(frozen=True)
class RecordedRefinement:
    """Transient region result plus exact accepted-action replay records."""

    regions: DrawingRegions
    batches: tuple[ReplayBatch, ...]
    outcomes: tuple[AutomaticActionOutcome, ...]


def ordered_path_operation_stages(
    operations: Sequence[object],
) -> tuple[tuple[str, tuple[object, ...]], ...]:
    """Partition path edits without changing cleanup position or repetition."""
    stages: list[tuple[str, tuple[object, ...]]] = []
    direct: list[object] = []
    for operation in operations:
        name = (
            operation.get("op")
            if isinstance(operation, Mapping)
            else getattr(operation, "op", None)
        )
        if name in {"simplify", "normalize_corners"}:
            if direct:
                stages.append(("direct", tuple(direct)))
                direct = []
            stages.append((str(name), (operation,)))
        else:
            direct.append(operation)
    if direct:
        stages.append(("direct", tuple(direct)))
    return tuple(stages)










def stitch_regions(trace: TraceResult, regions: Sequence[RegionSample]) -> DrawingRegions:
    """Normalize shared root boundaries before retaining interactive ``v0``.

    Initial trace stitching may snap existing shared geometry, but it must not
    construct an underlap beneath a neighbouring region. More invasive seam
    reconstruction remains available to explicit and automatic refinement.
    """
    return stitch_regions_with_intent(trace, regions).regions


def stitch_regions_with_intent(
    trace: TraceResult, regions: Sequence[RegionSample], *, enabled: bool = True
) -> RecordedRefinement:
    """Normalize initial seams and retain the accepted concrete constraints."""
    if not enabled:
        return RecordedRefinement(normalize_drawing_addresses(tuple(regions)), (), ())
    recorded = run_detection_with_intent(
            regions,
            trace,
            "stitch",
            None,
            epsilon=trace.options.simplify_tolerance,
            max_error=trace.options.curve_tolerance,
            allow_underlap=False,
    )
    return replace(recorded, regions=normalize_drawing_addresses(recorded.regions))


def auto_refine(
    trace: TraceResult,
    regions: Sequence[RegionSample],
    *,
    rgb: np.ndarray | None = None,
    corner_normalize: bool = False,
    refinement_passes: RefinementPasses | None = None,
    on_pass: Callable[[str, DrawingRegions], None] | None = None,
) -> DrawingRegions:
    """Run the one-shot refinement over already-traced drawing roots.

    ``trace_drawing(refine="auto")`` must not enter the refinement's alternate
    raster-to-region path.  The trace artifact is the drawing's source of truth:
    it owns stable region and command IDs, and every auto-refined version must
    remain a descendant of those roots.
    """
    return auto_refine_with_intent(
        trace,
        regions,
        rgb=rgb,
        corner_normalize=corner_normalize,
        refinement_passes=refinement_passes,
        on_pass=on_pass,
    ).regions


def auto_refine_with_intent(
    trace: TraceResult,
    regions: Sequence[RegionSample],
    *,
    rgb: np.ndarray | None = None,
    allow_gradients: bool = True,
    corner_normalize: bool = False,
    refinement_passes: RefinementPasses | None = None,
    on_pass: Callable[[str, DrawingRegions], None] | None = None,
) -> RecordedRefinement:
    """Adapt diagnostic region inputs/outputs around canonical automatic execution."""
    from .drawing import Canvas
    from .refinement.replay_binding import capture_trace_replay_state, working_regions_from_state
    from .drawing_refinement import refine_canonical_drawing
    from .drawing_paint import refill_drawing_from_source, source_paint_masks
    if rgb is not None and rgb.shape[:2] != (trace.height, trace.width):
        raise ValueError("fill RGB dimensions must match the trace canvas")
    state = capture_trace_replay_state(regions, Canvas(trace.width, trace.height))
    def _capture(pass_name, current):
        if on_pass is not None:
            on_pass(pass_name, working_regions_from_state(current))
    result = refine_canonical_drawing(state, epsilon=trace.options.simplify_tolerance,
        max_error=trace.options.curve_tolerance, corner_normalize=corner_normalize,
        refinement_passes=refinement_passes, on_pass=_capture)
    state = result.state
    if rgb is not None:
        state = replace(state, drawing=refill_drawing_from_source(state.drawing, rgb,
            source_masks=source_paint_masks(state.region_evidence), invalidate_all=True,
            allow_gradients=allow_gradients, share_root_gradients=trace.material_pretrace is None))
    return RecordedRefinement(working_regions_from_state(state), result.batches, result.outcomes)


def refine(
    trace: TraceResult,
    base: Sequence[RegionSample],
    plan: DrawingPlan,
    *,
    rgb: np.ndarray | None = None,
    fit_sources: Mapping[str, np.ndarray] | None = None,
) -> DrawingRegions:
    """Execute *plan* as immutable transformations over native vector regions."""
    if rgb is not None and rgb.shape[:2] != (trace.height, trace.width):
        raise ValueError("fill RGB dimensions must match the trace canvas")
    regions = normalize_working_topology(materialize_self_symmetry(base))
    drawing = drawing_from_region_samples(
        plan.drawing_id, width=trace.width, height=trace.height, regions=regions,
    )
    validate_plan(plan, drawing)
    z_order: SetZOrderAction | None = None

    for index, action in enumerate(plan.ops):
        before = regions
        op = action.model_dump(exclude_none=True)
        name = action.op
        settings = _action_settings(trace, plan, op)
        epsilon, max_error = settings.epsilon, settings.max_error
        if index and name in {"fit_path", "fit_primitive", "set_fill", "merge", "split"}:
            # Temporary adapter: later actions must see earlier plan edits,
            # while the first action reuses the already validated projection.
            drawing = drawing_from_region_samples(
                plan.drawing_id, width=trace.width, height=trace.height, regions=regions,
            )
        if name == "merge":
            regions = publish_drawing_update(drawing_partition.merge_regions(drawing, action), regions)
        elif name == "split":
            try:
                regions = publish_drawing_update(drawing_partition.split_region(drawing, action), regions)
            except ValueError as error:
                raise PlanValidationError(f"/ops/{index}/divider", str(error)) from error
        elif name in {"detect_primitives", "detect_symmetry", "detect_clones"}:
            regions = _run_detection(regions, trace, name, op.get("target"), epsilon=epsilon, max_error=max_error)
        elif name == "stitch":
            regions = _run_detection(
                regions,
                trace,
                name,
                op.get("targets"),
                epsilon=epsilon,
                max_error=max_error,
            )
        elif name in {"set_symmetry", "assign_clone", "align"}:
            drawing = drawing_from_region_samples(plan.drawing_id, width=trace.width,
                height=trace.height, regions=regions)
            execute = {"set_symmetry": drawing_relationships.set_region_symmetry,
                "assign_clone": drawing_relationships.assign_clone,
                "align": drawing_relationships.align_region}[name]
            regions = publish_region_update(execute(drawing, action), regions)
        elif name == "fit_path":
            targets = working_leaf_targets(regions)
            target = targets[op["target"]]
            try:
                updated = drawing_region_actions.fit_region_path(
                    drawing.region(target.id), action, raster=target.region.raster, settings=settings,
                )
                regions = publish_region_update(updated, regions)
            except ValueError as error:
                raise PlanValidationError(f"/ops/{index}/target", str(error)) from error
        elif name == "fit_primitive":
            targets = working_leaf_targets(regions)
            target = targets[op["target"]]
            try:
                updated = drawing_region_actions.fit_region_primitive(
                    drawing.region(target.id), action,
                    raster=target.region.raster, settings=settings,
                )
            except ValueError as error:
                raise PlanValidationError(f"/ops/{index}/target", str(error)) from error
            regions = publish_region_update(updated, regions)
        elif name == "update_geometry":
            geometry_ops = action.ops
            target_id = op["target"]
            for stage_name, stage_ops in ordered_path_operation_stages(geometry_ops):
                if stage_name != "direct":
                    regions = _run_detection(
                        regions,
                        trace,
                        stage_name,
                        target_id,
                        epsilon=epsilon,
                        max_error=max_error,
                    )
                    continue
                targets = working_leaf_targets(regions)
                target = targets[target_id]
                drawing = drawing_from_region_samples(
                    plan.drawing_id, width=trace.width, height=trace.height, regions=regions,
                )
                try:
                    updated, receipt = path_operations.apply_region_path_operations(
                        drawing, action.model_copy(update={"ops": stage_ops}), settings=settings,
                        fit_sources=fit_sources,
                        source_text=str(target.region.current.params["d"]),
                        previous_lineage=target.region.current.params.get("path_command_lineage"),
                    )
                    metadata = dict(receipt)
                    if "fill_rule" in target.region.current.params:
                        metadata["fill_rule"] = target.region.current.params["fill_rule"]
                    regions = publish_region_update(updated, regions,
                        shape_metadata={updated.id: metadata})
                except ValueError as error:
                    raise PlanValidationError(f"/ops/{index}/ops", str(error)) from error
        elif name == "set_fill":
            target_id = op["target"]
            updated = drawing_region_actions.set_region_fill(drawing.region(target_id), action)
            regions = publish_region_update(updated, regions)
        elif name == "set_z_order":
            z_order = action

        # Refresh resolved clone and branch footprints before paint invalidation.
        regions = normalize_working_topology(regions, normalize_components=False)
        # Geometry is authoritative.  A changed leaf must not retain a fill
        # fitted to its earlier mask; clear it, then run the one shared fill
        # pass against its new vector footprint.  An explicit ``set_fill`` is
        # intentionally excluded because it is already the caller's final
        # fill decision for that operation.
        if rgb is not None and name not in {"set_fill", "set_z_order"}:
            regions = refill_changed_regions_from_source(before, regions, rgb)
        regions = normalize_working_topology(regions)
    if z_order is not None:
        drawing = drawing_from_region_samples(
            plan.drawing_id, width=trace.width, height=trace.height, regions=regions,
        )
        ordered = drawing_region_actions.set_drawing_z_order(drawing, z_order)
        for root in ordered.regions:
            regions = publish_region_update(root, regions)
    return normalize_working_topology(normalize_drawing_addresses(regions))


def _run_detection(
    regions: Sequence[RegionSample],
    trace: TraceResult,
    operation: str,
    target_id: object | None,
    *,
    epsilon: float,
    max_error: float,
    allow_underlap: bool = True,
) -> DrawingRegions:
    """Run one existing refinement pass over the cached region forest."""
    return run_detection_with_intent(
        regions,
        trace,
        operation,
        target_id,
        epsilon=epsilon,
        max_error=max_error,
        allow_underlap=allow_underlap,
    ).regions


def run_detection_with_intent(
    regions: Sequence[RegionSample],
    trace: TraceResult,
    operation: str,
    target_id: object | None,
    *,
    epsilon: float,
    max_error: float,
    allow_underlap: bool = True,
) -> RecordedRefinement:
    """Run one producer and return only its typed accepted intent for replay."""
    regions = normalize_working_topology(materialize_self_symmetry(regions))
    cubic = False
    if operation == "detect_primitives":
        def pass_fn(objects, masks):
            return primitives_pass(objects, masks, epsilon=epsilon)
    elif operation == "detect_symmetry":
        def pass_fn(objects, masks):
            return symmetry_pass(objects, masks, epsilon=epsilon, max_error=max_error, cubic=cubic)
    elif operation == "detect_clones":
        def pass_fn(objects, masks):
            return clones_pass(objects, masks, symbolic=True)
    elif operation == "split":
        pass_fn = split_compound_pass
    elif operation == "simplify":
        def pass_fn(objects, masks):
            return simplify_pass(
                objects,
                masks,
                epsilon=epsilon,
                max_error=max_error,
                cubic=cubic,
                max_boundary_error=max_error,
            )
    elif operation == "stitch":
        def pass_fn(objects, masks):
            return seams_pass(
                objects,
                masks,
                epsilon=epsilon,
                max_error=max_error,
                cubic=cubic,
                allow_underlap=allow_underlap,
            )
    elif operation == "normalize_corners":
        def pass_fn(objects, masks):
            return corner_normalize_pass(objects, masks, max_error=max_error, enabled=True)
    else:
        raise ValueError(f"unknown detection operation: {operation}")

    selected_ids = (
        (target_id,)
        if type(target_id) is str
        else tuple(target_id)
        if isinstance(target_id, Sequence) and not isinstance(target_id, (str, bytes))
        else ()
    )
    if selected_ids:
        selected_region_ids = {working_leaf_targets(regions)[selected_id].region.id for selected_id in selected_ids}
        original_pass = pass_fn

        def pass_fn(objects, masks):  # type: ignore[no-redef]
            proposals = original_pass(objects, masks)
            if operation != "detect_clones":
                return [proposal for proposal in proposals if selected_region_ids.intersection(proposal.obj_ids)]
            return [
                proposal
                for proposal in proposals
                if selected_region_ids.intersection(proposal.obj_ids)
                or any(
                    replacement.diagnostics.get("clones", {}).get("matched_source") in selected_region_ids
                    for replacement in proposal.new_objects
                )
            ]

    before = {region.id: region for region in regions}
    masks = {region.id: np.asarray(region.raster, dtype=bool) for region in regions}
    optimized = select_refinements_with_intent(list(regions), masks, [pass_fn])
    restored = tuple(
        restore_working_root_metadata(region, before.get(region.id))
        for region in optimized.objects
    )
    result = normalize_working_topology(
        rebind_working_corner_provenance(
            regions, restored, tolerance=max(1.0, epsilon * 2.0)
        )
    )
    return RecordedRefinement(result, optimized.batches, optimized.outcomes)


def _action_settings(trace: TraceResult, plan: DrawingPlan, op: Mapping[str, object]) -> ActionSettings:
    """Resolve retained trace settings, plan settings, then one action override."""
    plan_settings = plan.settings
    action_override = op.get("settings", {})
    assert isinstance(action_override, Mapping)
    return ActionSettings(
        epsilon=float(
            action_override.get(
                "epsilon",
                plan_settings.epsilon
                if plan_settings.epsilon is not None
                else trace.options.simplify_tolerance,
            )
        ),
        max_error=float(
            action_override.get(
                "max_error",
                plan_settings.max_error
                if plan_settings.max_error is not None
                else trace.options.curve_tolerance,
            )
        ),
        fit_strategy=action_override.get(
            "fit_strategy",
            plan_settings.fit_strategy
            if plan_settings.fit_strategy is not None
            else trace.options.fit_strategy,
        ),
        purpose=action_override.get("purpose", plan_settings.purpose or "accuracy"),
        execution=action_override.get("execution", plan_settings.execution or "deterministic"),
    )


def resolved_action_settings(
    trace: TraceResult, plan: DrawingPlan, action: object
) -> ActionSettings:
    """Expose the one retained/plan/action settings resolution to recipe capture."""
    if not hasattr(action, "model_dump"):
        raise TypeError("drawing action must be a typed action contract")
    return _action_settings(trace, plan, action.model_dump(exclude_none=True))

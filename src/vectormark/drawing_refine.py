"""Apply drawing plans directly to cached :class:`VectorRegion` roots.

The interactive drawing state is the optimizer's native region forest.  SVG and
reports are derived artifacts, never a second editable scene representation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np

from .candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .drawing_actions import ActionSettings, SetZOrderAction
from .drawing_plan import DrawingPlan, PlanValidationError, validate_plan
from .drawing_adapter import apply_region_geometry, publish_region_update, publish_primitive_geometry, drawing_from_vector_regions, working_fill_from_spec as _fill, normalize_drawing_addresses, root_region_address, rebind_working_corner_provenance, restore_working_root_metadata, WorkingLeafTarget, working_leaf_targets, replace_working_leaf
from .drawing_fill import serialize_fill
from .drawing_trace import TraceResult, svg_path_commands
from .emit import render_svg_doc, resolve_fill, resolve_use_shape, shape_to_svg
from .fit import Shape
from .optimizer.vector_region import VectorRegion
from .optimizer.automatic_replay import (
    AutomaticActionOutcome,
    ReplayBatch,
    optimize_with_intent,
)
from .optimizer.passes import corner_normalize_pass, clones_pass, primitives_pass, seams_pass, simplify_pass, split_compound_pass, symmetry_pass
from .optimizer.symmetry_materialize import index_region_paths, materialize_self_symmetry, resolve_source_path
from .drawing_topology import normalize_retained_topology
from .pipeline import Options
from .refinement_settings import RefinementPasses
from .drawing_paint import refill_all_regions_from_source, refill_changed_regions_from_source
from . import drawing_region_actions, drawing_relationships, path_operations, drawing_partition


DrawingRegions = tuple[VectorRegion, ...]


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


@dataclass(frozen=True)
class RenderedDrawing:
    """A transient SVG/report projection of one drawing-version's regions."""

    svg: str
    report: Mapping[str, object]


def render_drawing(trace: TraceResult, regions: Sequence[VectorRegion]) -> RenderedDrawing:
    """Serialize retained geometry and paint; execution owns all reconstruction."""
    def validate(region: VectorRegion) -> None:
        symmetry = region.diagnostics.get("symmetry")
        if isinstance(symmetry, Mapping) and (
            (region.is_branch and symmetry.get("mode") == "self")
            or symmetry.get("mode") == "self_mirror"
        ):
            raise ValueError("self-symmetry must be materialized before rendering")
        for child in region.children:
            validate(child)
    for region in regions:
        validate(region)
    targets = working_leaf_targets(regions)
    ordered = sorted(targets.values(), key=lambda target: (target.region.z, target.region.id))
    reference_index = index_region_paths(regions)
    targets_by_path = {target.region_path: target for target in ordered}
    resolved_sources: dict[str, str] = {}
    defs: list[str] = []
    # A clone shares geometry, not paint.  An SVG ``<use fill=...>`` cannot
    # override a ``fill`` attribute on the referenced source element, so the
    # reusable geometry lives in an unpainted definition.  The source and each
    # clone are then independently painted instances with stable target IDs.
    clone_definition_ids: dict[str, str] = {}
    clone_sources: dict[str, WorkingLeafTarget] = {}
    targets_by_id = {target.id: target for target in ordered}
    for target in ordered:
        current = target.region.current
        if current is None or current.kind != "use":
            continue
        source_path = resolve_source_path(reference_index, target.region_path, current.params)
        if source_path not in targets_by_path:
            raise ValueError("clone source is not a retained target")
        source_target_id = targets_by_path[source_path].id
        resolved_sources[target.id] = source_target_id
        clone_diagnostics = target.region.diagnostics.get("clones")
        is_clone = isinstance(clone_diagnostics, Mapping) or isinstance(target.region.diagnostics.get("clone"), Mapping)
        if not is_clone:
            continue
        source_target = targets_by_id[source_target_id]
        source_shape = source_target.region.current
        if source_shape is None or source_shape.kind == "use":
            raise ValueError("clone source must be retained concrete geometry")
        clone_sources[source_target_id] = source_target
        clone_definition_ids[source_target_id] = f"{source_target_id}-geometry"

    for source_target_id, source_target in clone_sources.items():
        source_shape = source_target.region.current
        assert source_shape is not None
        defs.append(shape_to_svg(source_shape, None, clone_definition_ids[source_target_id]))

    body: list[str] = []
    report_targets: list[dict[str, object]] = []
    for target in ordered:
        assert target.region.current is not None
        current = target.region.current
        id_map = {current.params["href_obj_id"]: resolved_sources[target.id]} if current.kind == "use" else {}
        shape = resolve_use_shape(current, id_map)
        fill = target.region.fill
        if fill is None:
            raise ValueError(f"drawing region {target.id!r} has no fill")
        resolved_fill = resolve_fill(fill, defs)
        if target.id in clone_definition_ids:
            shape = Shape(
                "use",
                {"href": clone_definition_ids[target.id], "transform": (1, 0, 0, 1, 0, 0)},
            )
        elif shape.kind == "use":
            clone_diagnostics = target.region.diagnostics.get("clones")
            is_clone = isinstance(clone_diagnostics, Mapping) or isinstance(target.region.diagnostics.get("clone"), Mapping)
            if is_clone:
                source_target_id = resolved_sources[target.id]
                definition_id = clone_definition_ids.get(source_target_id)
                if definition_id is not None:
                    params = dict(shape.params)
                    params["href"] = definition_id
                    shape = Shape("use", params)
        body.append(shape_to_svg(shape, resolved_fill, target.id))
        report_targets.append(
            {
                "id": target.id,
                "root_id": target.root_id,
                "source_regions": tuple(sorted(target.source_regions)),
                "geometry": shape.kind,
                "fill": type(fill).__name__,
                "z": target.region.z,
                "diagnostics": _public_diagnostics(target.region.diagnostics),
            }
        )
    return RenderedDrawing(
        render_svg_doc(trace.width, trace.height, body, defs),
        {"targets": tuple(report_targets), "groups": _drawing_groups(regions)},
    )


def drawing_summary(trace: TraceResult, regions: Sequence[VectorRegion]) -> dict[str, object]:
    """Return the compact, merged-root view agents receive after interactive trace."""
    targets = working_leaf_targets(regions)
    ordered = sorted(targets.values(), key=lambda target: (target.region.z, target.region.id))
    summaries: list[dict[str, object]] = []
    for target in ordered:
        assert target.region.current is not None
        shape = target.region.current
        geometry: dict[str, object] = {"type": shape.kind}
        if shape.kind == "path":
            geometry["d"] = shape.params["d"]
            if shape.params.get("fill_rule"):
                geometry["fill_rule"] = shape.params["fill_rule"]
            commands = [
                {
                    "id": command.id,
                    "command": command.command,
                    "values": list(command.values),
                    "anchor_kind": command.anchor_kind,
                    "corner_id": command.corner_id,
                }
                for command in svg_path_commands(
                    str(shape.params["d"]),
                    target.id,
                    path_ids=shape.params.get("path_ids"),
                    corner_diagnostics=(
                        target.region.diagnostics.get("corners")
                        if isinstance(target.region.diagnostics.get("corners"), Mapping)
                        and target.region.diagnostics["corners"].get("path") == str(shape.params["d"])
                        else None
                    ),
                )
            ]
            geometry["commands"] = commands
        summaries.append(
            {
                "id": target.id,
                "root_id": target.root_id,
                "source_regions": tuple(sorted(target.source_regions)),
                "geometry": geometry,
                "fill": _public_fill(target.region.fill),
            }
        )
    return {"width": trace.width, "height": trace.height, "options": asdict(trace.options), "regions": summaries,
            "groups": _drawing_groups(regions)}


def labeled_drawing_svg(trace: TraceResult, regions: Sequence[VectorRegion]) -> str:
    """Render the public, plan-addressable region map for a drawing version."""
    from .drawing_render import render_labeled_drawing

    return render_labeled_drawing(drawing_from_vector_regions(
        "region-map", width=trace.width, height=trace.height, regions=regions,
    ))


def stitch_regions(trace: TraceResult, regions: Sequence[VectorRegion]) -> DrawingRegions:
    """Normalize shared root boundaries before retaining interactive ``v0``.

    Initial trace stitching may snap existing shared geometry, but it must not
    construct an underlap beneath a neighbouring region. More invasive seam
    reconstruction remains available to explicit and automatic refinement.
    """
    return stitch_regions_with_intent(trace, regions).regions


def stitch_regions_with_intent(
    trace: TraceResult, regions: Sequence[VectorRegion], *, enabled: bool = True
) -> RecordedRefinement:
    """Normalize initial seams and retain the accepted concrete constraints."""
    if not enabled:
        return RecordedRefinement(normalize_drawing_addresses(tuple(regions)), (), ())
    recorded = _run_detection_recorded(
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
    regions: Sequence[VectorRegion],
    *,
    rgb: np.ndarray | None = None,
    corner_normalize: bool = False,
    refinement_passes: RefinementPasses | None = None,
    on_pass: Callable[[str, DrawingRegions], None] | None = None,
) -> DrawingRegions:
    """Run the one-shot optimizer over already-traced drawing roots.

    ``trace_drawing(refine="auto")`` must not enter the optimizer's alternate
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
    regions: Sequence[VectorRegion],
    *,
    rgb: np.ndarray | None = None,
    allow_gradients: bool = True,
    corner_normalize: bool = False,
    refinement_passes: RefinementPasses | None = None,
    on_pass: Callable[[str, DrawingRegions], None] | None = None,
) -> RecordedRefinement:
    """Run automatic refinement and retain its exact committed operations."""
    # Import locally: pipeline owns the pass ordering, while drawing_refine owns
    # the persistent VectorRegion forest used by MCP drawing versions.
    from .pipeline import _optimizer_passes

    options = Options(
        epsilon=trace.options.simplify_tolerance,
        max_error=trace.options.curve_tolerance,
        cubic_paths=False,
        aa_contours=trace.options.trace_level == "subpixel",
        max_sampling_colors=trace.options.max_sampling_colors,
        min_region_size=trace.options.min_region_size,
        optimizer=True,
        corner_normalize=corner_normalize,
        refinement_passes=refinement_passes or RefinementPasses(),
    )
    if rgb is not None and rgb.shape[:2] != (trace.height, trace.width):
        raise ValueError("fill RGB dimensions must match the trace canvas")
    regions = normalize_retained_topology(materialize_self_symmetry(regions))
    before = {region.id: region for region in regions}
    masks = {region.id: np.asarray(region.raster, dtype=bool) for region in regions}
    def _capture(pass_name: str, current: list[VectorRegion]) -> None:
        if on_pass is not None:
            on_pass(
                pass_name,
                normalize_retained_topology(normalize_drawing_addresses(tuple(
                    restore_working_root_metadata(region, before.get(region.id)) for region in current
                ))),
            )

    optimized = optimize_with_intent(
        list(regions), masks, _optimizer_passes(options), on_pass=_capture
    )
    result = normalize_retained_topology(normalize_drawing_addresses(
        tuple(
            restore_working_root_metadata(region, before.get(region.id))
            for region in optimized.objects
        )
    ))
    if rgb is None:
        return RecordedRefinement(result, optimized.batches, optimized.outcomes)
    # Automatic refinement is a geometry pipeline.  Re-fit every final leaf
    # from the source only after the full pass sequence has established its
    # footprint, rather than retaining palette-fragment fills merely because a
    # particular leaf's geometry key happened not to change.
    result = refill_all_regions_from_source(
        result,
        rgb,
        allow_gradients=allow_gradients,
        # Gradient pretrace has already converted quantization bands into
        # intentional material supports. Sibling leaves may therefore be
        # different gradients that merely share one exterior geometry root.
        share_root_gradients=trace.material_pretrace is None,
    )
    return RecordedRefinement(result, optimized.batches, optimized.outcomes)


def refine(
    trace: TraceResult,
    base: Sequence[VectorRegion],
    plan: DrawingPlan,
    *,
    rgb: np.ndarray | None = None,
    fit_sources: Mapping[str, np.ndarray] | None = None,
) -> DrawingRegions:
    """Execute *plan* as immutable transformations over native vector regions."""
    if rgb is not None and rgb.shape[:2] != (trace.height, trace.width):
        raise ValueError("fill RGB dimensions must match the trace canvas")
    regions = normalize_retained_topology(materialize_self_symmetry(base))
    drawing = drawing_from_vector_regions(
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
        if index and name in {"fit_path", "fit_primitive", "set_fill"}:
            # Temporary adapter: later actions must see earlier plan edits,
            # while the first action reuses the already validated projection.
            drawing = drawing_from_vector_regions(
                plan.drawing_id, width=trace.width, height=trace.height, regions=regions,
            )
        if name == "merge":
            regions = drawing_partition.merge_regions(regions, action)
        elif name == "split":
            try:
                regions = drawing_partition.split_region(regions, action)
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
            drawing = drawing_from_vector_regions(plan.drawing_id, width=trace.width,
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
                regions = apply_region_geometry(updated, regions, ingress_prefit_corners=True)
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
            regions = replace_working_leaf(
                regions,
                target.id,
                lambda region: publish_primitive_geometry(
                    region, updated.geometry,
                    diagnostics={"geometry": updated.diagnostics["geometry"]},
                ),
            )
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
                drawing = drawing_from_vector_regions(
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
                    regions = apply_region_geometry(updated, regions,
                        shape_metadata={updated.id: metadata})
                except ValueError as error:
                    raise PlanValidationError(f"/ops/{index}/ops", str(error)) from error
        elif name == "set_fill":
            target_id = op["target"]
            updated = drawing_region_actions.set_region_fill(drawing.region(target_id), action)
            # Public fill assignments have no affine overrides.
            # This conversion remains only until working-region paint migrates.
            fill = _fill(serialize_fill(updated.fill))
            regions = replace_working_leaf(regions, target_id, lambda region: _with_fill(region, fill))
        elif name == "set_z_order":
            z_order = action

        # Refresh resolved clone and branch footprints before paint invalidation.
        regions = normalize_retained_topology(regions, normalize_components=False)
        # Geometry is authoritative.  A changed leaf must not retain a fill
        # fitted to its earlier mask; clear it, then run the one shared fill
        # pass against its new vector footprint.  An explicit ``set_fill`` is
        # intentionally excluded because it is already the caller's final
        # fill decision for that operation.
        if rgb is not None and name not in {"set_fill", "set_z_order"}:
            regions = refill_changed_regions_from_source(before, regions, rgb)
        regions = normalize_retained_topology(regions)
    if z_order is not None:
        drawing = drawing_from_vector_regions(
            plan.drawing_id, width=trace.width, height=trace.height, regions=regions,
        )
        ordered = drawing_region_actions.set_drawing_z_order(drawing, z_order)
        for root in ordered.regions:
            for leaf in root.leaves():
                regions = replace_working_leaf(regions, leaf.id, lambda region, z=leaf.z: _with_z(region, z))
    return normalize_retained_topology(normalize_drawing_addresses(regions))


def _run_detection(
    regions: Sequence[VectorRegion],
    trace: TraceResult,
    operation: str,
    target_id: object | None,
    *,
    epsilon: float,
    max_error: float,
    allow_underlap: bool = True,
) -> DrawingRegions:
    """Run one existing optimizer pass over the cached region forest."""
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
    regions: Sequence[VectorRegion],
    trace: TraceResult,
    operation: str,
    target_id: object | None,
    *,
    epsilon: float,
    max_error: float,
    allow_underlap: bool = True,
) -> RecordedRefinement:
    """Run one producer and return only its typed accepted intent for replay."""
    regions = normalize_retained_topology(materialize_self_symmetry(regions))
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
    optimized = optimize_with_intent(list(regions), masks, [pass_fn])
    restored = tuple(
        restore_working_root_metadata(region, before.get(region.id))
        for region in optimized.objects
    )
    result = normalize_retained_topology(
        rebind_working_corner_provenance(
            regions, restored, tolerance=max(1.0, epsilon * 2.0)
        )
    )
    return RecordedRefinement(result, optimized.batches, optimized.outcomes)


# Transitional private alias for internal callers retained until Task 17C removes
# the VectorRegion adapter boundary.
_run_detection_recorded = run_detection_with_intent


def _drawing_groups(regions: Sequence[VectorRegion]) -> tuple[dict[str, object], ...]:
    """Derive unpainted containers independently of the painted target list."""
    groups = []
    def visit(region, address):
        if region.is_leaf:
            return
        children = tuple(child.drawing_id or f"{address}-{child.id}" for child in region.children)
        groups.append({"id": address, "child_ids": children, "diagnostics": _public_diagnostics(region.diagnostics)})
        for child, child_id in zip(region.children, children, strict=True):
            visit(child, child_id)
    for region in regions:
        visit(region, root_region_address(region))
    return tuple(groups)




def _with_fill(region: VectorRegion, fill: Fill) -> VectorRegion:
    assert region.current is not None
    return region.with_current(region.current, fill=fill, footprint=region.footprint)


def _with_z(region: VectorRegion, z: float) -> VectorRegion:
    assert region.current is not None
    return region.with_current(region.current, z=z, footprint=region.footprint)


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


def _public_fill(fill: Fill | None) -> dict[str, object]:
    if isinstance(fill, FlatFill):
        return {"type": "flat", "color": fill.hex}
    if isinstance(fill, LinearGradientFill):
        return {"type": "linear_gradient", "geometry": dict(fill.geometry), "stops": list(fill.stops)}
    if isinstance(fill, RadialGradientFill):
        return {"type": "radial_gradient", "geometry": dict(fill.geometry), "stops": list(fill.stops)}
    if isinstance(fill, RasterFill):
        return {"type": "raster", "geometry": dict(fill.geometry)}
    return {"type": "none"}


def _public_diagnostics(value: object) -> object:
    """Make optimizer diagnostics safe to return in an MCP structured result."""
    if isinstance(value, Mapping):
        return {str(key): _public_diagnostics(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_public_diagnostics(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in {bool, int, float, str}:
        return value
    return repr(value)

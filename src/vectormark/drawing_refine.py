"""Apply drawing plans directly to cached :class:`VectorRegion` roots.

The interactive drawing state is the optimizer's native region forest.  SVG and
reports are derived artifacts, never a second editable scene representation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np

from .candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .drawing_actions import ActionSettings, SetZOrderAction
from .drawing_plan import DrawingPlan, PlanValidationError, validate_plan
from .drawing_adapter import apply_region_geometry, publish_region_update, publish_primitive_geometry, drawing_from_vector_regions, working_fill_from_spec as _fill, normalize_drawing_addresses, root_region_address
from .drawing_fill import serialize_fill
from .drawing_trace import TraceResult, svg_path_commands
from .emit import render_svg_doc, resolve_fill, resolve_use_shape, shape_to_svg
from .fit import Shape, _fmt
from .optimizer.vector_region import VectorRegion, _parse_subpaths, _sample_subpath
from .optimizer.corners import rebind_prefit_corner_diagnostics
from .optimizer.gate import rasterize
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
from .skia_geometry import SkPath, unary_union
from . import drawing_region_actions, drawing_relationships, path_operations
from .geometry_native import split_path_by_line as _split_footprint_by_line


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


@dataclass(frozen=True)
class _Target:
    id: str
    root_id: str
    region: VectorRegion
    source_regions: frozenset[str]
    region_path: tuple[int, ...]


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
    targets = _targets(regions)
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
    clone_sources: dict[str, _Target] = {}
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
    targets = _targets(regions)
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
                    _restore_root_metadata(region, before.get(region.id)) for region in current
                ))),
            )

    optimized = optimize_with_intent(
        list(regions), masks, _optimizer_passes(options), on_pass=_capture
    )
    result = normalize_retained_topology(normalize_drawing_addresses(
        tuple(
            _restore_root_metadata(region, before.get(region.id))
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
            regions = _group_regions(regions, op)
        elif name == "split":
            try:
                regions = _split_region(regions, op)
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
            targets = _targets(regions)
            target = targets[op["target"]]
            try:
                updated = drawing_region_actions.fit_region_path(
                    drawing.region(target.id), action, raster=target.region.raster, settings=settings,
                )
                regions = apply_region_geometry(updated, regions, ingress_prefit_corners=True)
            except ValueError as error:
                raise PlanValidationError(f"/ops/{index}/target", str(error)) from error
        elif name == "fit_primitive":
            targets = _targets(regions)
            target = targets[op["target"]]
            try:
                updated = drawing_region_actions.fit_region_primitive(
                    drawing.region(target.id), action,
                    raster=target.region.raster, settings=settings,
                )
            except ValueError as error:
                raise PlanValidationError(f"/ops/{index}/target", str(error)) from error
            regions = _replace_leaf(
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
                targets = _targets(regions)
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
            regions = _replace_leaf(regions, target_id, lambda region: _with_fill(region, fill))
        elif name == "set_z_order":
            z_order = action

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
                regions = _replace_leaf(regions, leaf.id, lambda region, z=leaf.z: _with_z(region, z))
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
        selected_region_ids = {_targets(regions)[selected_id].region.id for selected_id in selected_ids}
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
        _restore_root_metadata(region, before.get(region.id))
        for region in optimized.objects
    )
    result = normalize_retained_topology(
        _rebind_corner_provenance(
            regions, restored, tolerance=max(1.0, epsilon * 2.0)
        )
    )
    return RecordedRefinement(result, optimized.batches, optimized.outcomes)


# Transitional private alias for internal callers retained until Task 17C removes
# the VectorRegion adapter boundary.
_run_detection_recorded = run_detection_with_intent


def _rebind_corner_provenance(
    before: Sequence[VectorRegion],
    after: Sequence[VectorRegion],
    *,
    tolerance: float,
) -> DrawingRegions:
    """Preserve trace-declared corners through a command-preserving rewrite."""
    originals = {leaf.id: leaf for root in before for leaf in root.leaves()}

    def visit(region: VectorRegion) -> VectorRegion:
        if region.is_branch:
            children = tuple(visit(child) for child in region.children)
            # VectorRegion equality deliberately ignores diagnostics.  Corner
            # provenance is a diagnostics-only update, so retain the rebuilt
            # branch whenever any child instance changed.
            return (
                region.with_children(children)
                if any(child is not original for child, original in zip(children, region.children, strict=True))
                else region
            )
        assert region.current is not None
        original = originals.get(region.id)
        if (
            original is None
            or original.current is None
            or original.current.kind != "path"
            or region.current.kind != "path"
        ):
            return region
        corners = original.diagnostics.get("corners")
        if (
            not isinstance(corners, Mapping)
            or not corners.get("spans")
            or corners.get("path") != str(original.current.params.get("d", ""))
        ):
            return region
        rebound = rebind_prefit_corner_diagnostics(
            str(region.current.params["d"]),
            corners,
            tolerance=tolerance,
        )
        return region.with_diagnostics({"corners": rebound})

    return tuple(visit(region) for region in after)


def _restore_root_metadata(region: VectorRegion, original: VectorRegion | None) -> VectorRegion:
    """Optimizer passes preserve geometry IDs, while drawing IDs stay root-owned."""
    if original is None or original.drawing_id is None:
        return region
    if region.is_leaf:
        assert region.current is not None
        return VectorRegion(
            id=region.id,
            current=region.current,
            original=region.original,
            fill=region.fill,
            z=region.z,
            footprint=region.footprint,
            raster=region.raster,
            source_label=region.source_label,
            color_hex=region.color_hex,
            drawing_id=original.drawing_id,
            source_regions=original.source_regions,
            coverage=region.coverage,
            diagnostics=region.diagnostics,
        )
    return VectorRegion.branch(
        id=region.id,
        children=region.children,
        z=region.z,
        raster=region.raster,
        footprint=region.footprint,
        fill=region.fill,
        source_label=region.source_label,
        color_hex=region.color_hex,
        drawing_id=original.drawing_id,
        source_regions=original.source_regions,
        diagnostics=region.diagnostics,
        original=region.original,
    )


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


def _targets(regions: Sequence[VectorRegion]) -> dict[str, _Target]:
    """Derive public target handles from the region tree; do not persist an index."""
    result: dict[str, _Target] = {}

    def visit(region: VectorRegion, root_label: str, legacy_label: str, path: tuple[int, ...]) -> None:
        if region.is_leaf:
            label = _leaf_address(root_label, region, legacy_label)
            if label in result:
                raise ValueError(f"duplicate drawing region id: {label}")
            # Target lineage belongs to the retained leaf, not to its root
            # container.  A root can be a semantic union while its children
            # remain independently traced, cloned, or split regions.
            sources = frozenset(region.source_regions or (root_label,))
            result[label] = _Target(label, root_label, region, sources, path)
            return
        for child in region.children:
            visit(child, root_label, f"{legacy_label}-{child.id}", (*path, child.id))

    for region in regions:
        drawing_id = region.drawing_id
        if type(drawing_id) is not str:
            raise ValueError("drawing root is missing a stable drawing_id")
        visit(region, drawing_id, drawing_id, (region.id,))
    return result


def _leaf_address(root_address: str, region: VectorRegion, legacy_address: str) -> str:
    """Return a compact retained leaf address, with legacy-tree fallback."""
    candidate = region.drawing_id
    if isinstance(candidate, str) and (
        candidate.startswith(f"{root_address}-")
        or (candidate == root_address and legacy_address == root_address)
    ):
        return candidate
    return legacy_address


def _replace_leaf(
    regions: Sequence[VectorRegion], target_id: str, transform: Callable[[VectorRegion], VectorRegion]
) -> DrawingRegions:
    """Replace one labeled leaf while retaining the rest of the forest by value."""
    changed = False

    def visit(region: VectorRegion, root_label: str, legacy_label: str) -> VectorRegion:
        nonlocal changed
        if region.is_leaf:
            label = _leaf_address(root_label, region, legacy_label)
            if label == target_id:
                changed = True
                return transform(region)
            return region
        children = tuple(
            visit(child, root_label, f"{legacy_label}-{child.id}")
            for child in region.children
        )
        return region.with_children(children) if children != region.children else region

    replaced = tuple(visit(region, root_region_address(region), root_region_address(region)) for region in regions)
    if not changed:
        raise KeyError(target_id)
    return replaced


def _group_regions(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    """Create a semantic target with a SkPath-unioned, seam-free outline."""
    targets = _targets(regions)
    member_ids = _expand_merge_member_ids(targets, tuple(op["regions"]))
    members = tuple(targets[region_id] for region_id in member_ids)
    member_regions = tuple(member.region for member in members)
    first = member_regions[0]
    assert first.current is not None and first.fill is not None
    raster = np.zeros_like(first.raster, dtype=bool)
    for member in member_regions:
        raster |= member.raster
    footprints = [member.footprint for member in member_regions]
    if not all(isinstance(footprint, SkPath) for footprint in footprints):
        raise ValueError("merge requires polygonal vector-region footprints")
    footprint = unary_union([footprint for footprint in footprints if isinstance(footprint, SkPath)])
    shape = Shape("path", {"d": footprint.to_svg_d()})
    merged = VectorRegion.from_shape(
        id=max(region.id for region in _all_regions(regions)) + 1,
        shape=shape,
        fill=first.fill,
        z=min(member.z for member in member_regions),
        raster=raster,
        footprint=footprint,
        color_hex=first.color_hex,
        drawing_id=op["id"],
        source_regions=tuple(sorted({source for member in members for source in member.source_regions})),
        diagnostics={"merge": {"unioned": True, "sources": member_ids}},
    )
    return _remove_leaves(regions, frozenset(member_ids)) + (merged,)


def _expand_merge_member_ids(targets: Mapping[str, _Target], requested_ids: Sequence[object]) -> tuple[str, ...]:
    """Resolve a branch/root reference to every leaf currently retained below it."""
    resolved: list[str] = []
    for requested in requested_ids:
        assert isinstance(requested, str)
        members = (requested,) if requested in targets else tuple(
            sorted(target_id for target_id in targets if target_id.startswith(f"{requested}-"))
        )
        if not members:
            raise KeyError(requested)
        for member in members:
            if member in resolved:
                raise ValueError(f"merge region {requested!r} resolves to an already selected child")
            resolved.append(member)
    return tuple(resolved)


def _split_region(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    """Replace one retained target with two exact vector clips across an explicit divider."""
    targets = _targets(regions)
    target_id = op["target"]
    assert isinstance(target_id, str)
    target = targets[target_id]
    source = target.region
    if not source.is_leaf or source.current is None or source.fill is None:
        raise ValueError("split requires a filled leaf target")
    if not isinstance(source.footprint, SkPath):
        raise ValueError("split requires a polygonal vector-region footprint")

    divider = op["divider"]
    assert isinstance(divider, Mapping)
    divider_type = divider["type"]
    if divider_type == "line":
        points = divider["points"]
        assert isinstance(points, Sequence)
        first = points[0]
        second = points[1]
        assert isinstance(first, Sequence) and isinstance(second, Sequence)
        parts = _split_footprint_by_line(
            source.footprint,
            (float(first[0]), float(first[1])),
            (float(second[0]), float(second[1])),
        )
    elif divider_type == "path":
        d = divider["d"]
        assert isinstance(d, str)
        parts = _split_footprint_by_path(source.footprint, d)
    else:  # guarded by plan validation; keeps this executor safe for direct callers.
        raise ValueError("divider type must be 'line' or 'path'")

    if len(parts) != 2 or any(part.is_empty or part.area <= 1e-6 for part in parts):
        raise ValueError("divider does not divide the target into two non-empty regions")

    next_id = max(region.id for region in _all_regions(regions)) + 1
    children: list[VectorRegion] = []
    for index, footprint in enumerate(parts):
        shape = Shape("path", {"d": footprint.to_svg_d()})
        children.append(
            VectorRegion.from_shape(
                id=source.id if index == 0 else next_id,
                shape=shape,
                fill=source.fill,
                z=source.z + index * 0.001,
                footprint=footprint,
                raster=rasterize(footprint, source.raster.shape),
                source_label=source.source_label,
                color_hex=source.color_hex,
                source_regions=source.source_regions,
                coverage=source.coverage,
                diagnostics={
                    "split": {
                        "source": target_id,
                        "divider": dict(divider),
                        "part": index + 1,
                    }
                },
            )
        )
    branch = VectorRegion.branch(
        id=source.id,
        children=tuple(children),
        z=source.z,
        raster=source.raster,
        footprint=source.footprint,
        fill=source.fill,
        source_label=source.source_label,
        color_hex=source.color_hex,
        drawing_id=source.drawing_id,
        source_regions=source.source_regions,
        diagnostics={"split": {"source": target_id, "divider": dict(divider), "children": 2}},
    )
    return _replace_leaf(regions, target_id, lambda _region: branch)


def _split_footprint_by_path(footprint: SkPath, d: str) -> tuple[SkPath, SkPath]:
    """Split one simple footprint using an open divider whose endpoints touch its boundary."""
    subpaths = _parse_subpaths(d)
    if len(subpaths) != 1 or not subpaths[0] or subpaths[0][0][0] != "M" or any(kind == "Z" for kind, _ in subpaths[0]):
        raise ValueError("path divider must contain exactly one open SVG subpath")
    sampled_divider = _sample_subpath(subpaths[0], samples=32)
    if len(sampled_divider) < 2:
        raise ValueError("path divider must contain a drawable segment")
    rings = footprint.linearized_subpaths
    if len(rings) != 1:
        raise ValueError("path divider currently requires a simple target without holes or disconnected islands")
    ring = rings[0]
    start, start_segment, start_t, start_distance = _project_to_ring(sampled_divider[0], ring)
    end, end_segment, end_t, end_distance = _project_to_ring(sampled_divider[-1], ring)
    min_x, min_y, max_x, max_y = footprint.bounds
    endpoint_tolerance = max(1.0, min(3.0, math.hypot(max_x - min_x, max_y - min_y) * 0.005))
    if max(start_distance, end_distance) > endpoint_tolerance:
        raise ValueError("path divider endpoints must lie on the target boundary")
    boundary = _forward_ring_arc(ring, end, end_segment, end_t, start, start_segment, start_t)
    divider_d = _snapped_open_divider_d(subpaths[0], start, end)
    boundary_d = " ".join(f"L{_fmt(x)} {_fmt(y)}" for x, y in boundary[1:])
    clip = SkPath.from_svg_d(f"{divider_d} {boundary_d} Z")
    first = footprint.intersection(clip)
    second = footprint.difference(first)
    return first, second


def _snapped_open_divider_d(
    tokens: Sequence[tuple[str, list[float]]], start: tuple[float, float], end: tuple[float, float]
) -> str:
    """Serialize one open divider while snapping only its boundary endpoints."""
    normalized = [(kind, list(values)) for kind, values in tokens]
    normalized[0][1][:2] = [start[0], start[1]]
    final_kind, final_values = normalized[-1]
    endpoint_offset = {"L": 0, "Q": 2, "C": 4, "A": 5}.get(final_kind)
    if endpoint_offset is None:
        raise ValueError("path divider must end with a drawable SVG command")
    final_values[endpoint_offset:endpoint_offset + 2] = [end[0], end[1]]
    parts: list[str] = []
    for kind, values in normalized:
        parts.append(kind + " ".join(_fmt(value) for value in values))
    return " ".join(parts)


def _project_to_ring(
    point: tuple[float, float], ring: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float], int, float, float]:
    px, py = point
    best: tuple[tuple[float, float], int, float, float] | None = None
    for index, start in enumerate(ring):
        end = ring[(index + 1) % len(ring)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq <= 1e-12 else max(0.0, min(1.0, ((px - start[0]) * dx + (py - start[1]) * dy) / length_sq))
        projected = (start[0] + t * dx, start[1] + t * dy)
        distance = math.hypot(px - projected[0], py - projected[1])
        if best is None or distance < best[3]:
            best = (projected, index, t, distance)
    assert best is not None
    return best


def _forward_ring_arc(
    ring: Sequence[tuple[float, float]],
    start: tuple[float, float],
    start_segment: int,
    start_t: float,
    end: tuple[float, float],
    end_segment: int,
    end_t: float,
) -> list[tuple[float, float]]:
    """Return the forward boundary arc, including projected start/end points."""
    if start_segment == end_segment and start_t <= end_t:
        return [start, end]
    points = [start]
    index = (start_segment + 1) % len(ring)
    while index != (end_segment + 1) % len(ring):
        points.append(ring[index])
        index = (index + 1) % len(ring)
    points.append(end)
    return points


def _remove_leaves(regions: Sequence[VectorRegion], removed_ids: frozenset[str]) -> DrawingRegions:
    """Remove selected labeled leaves, pruning only now-empty branch nodes."""

    def visit(region: VectorRegion, root_label: str, legacy_label: str) -> VectorRegion | None:
        if region.is_leaf:
            label = _leaf_address(root_label, region, legacy_label)
            return None if label in removed_ids else region
        children = tuple(
            child
            for child in (
                visit(child, root_label, f"{legacy_label}-{child.id}")
                for child in region.children
            )
            if child is not None
        )
        if not children:
            return None
        return region if children == region.children else region.with_children(children)

    return tuple(
        region
        for root in regions
        if (region := visit(root, root_region_address(root), root_region_address(root))) is not None
    )


def _all_regions(regions: Sequence[VectorRegion]) -> tuple[VectorRegion, ...]:
    all_regions: list[VectorRegion] = []
    for region in regions:
        all_regions.append(region)
        all_regions.extend(region.leaves()) if region.is_branch else None
    return tuple(all_regions)


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

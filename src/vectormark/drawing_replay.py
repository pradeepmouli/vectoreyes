"""Execute an effective recipe from retained source evidence only."""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256

import numpy as np

from .corner_recipe import (
    AddCorner,
    CornerRecipeEdit,
    CornerRecipeExecutor,
    CornerRecipeReplay,
    RemoveCorner,
    ReplaceCorner,
)
from .drawing import Drawing
from .path_topology import component_paths
from .drawing_adapter import (
    apply_region_geometry, drawing_from_vector_regions, vector_region_address_index,
    restore_selected_roots, working_region_evidence,
    publish_region_update, restore_corner_annotations, shape_from_region_geometry,
)
from .drawing_actions import (
    ActionSettings,
    ActionSettingsOverride,
    AddCornerAction,
    DrawingPlan,
    FitPathDrawingAction,
    FitPrimitiveAction,
    RemoveCornerAction,
    ReplaceCornerAction,
    SetFillAction,
    SetZOrderAction,
    StitchAction,
    UpdateGeometryAction,
)
from .drawing_recipe import (
    EvidenceTargetBinding,
    EffectiveDrawingRecipe,
    GeometryEvidence,
    ApplyCornerRecipeOperation,
    PaintEvidence,
    PathReplayCorrespondence,
    NamedOutputBinding,
    ReplayRecord,
    RecipeOperation,
    SelectedOperationOutcome,
    RecipeStep,
    ResolvedCornerEdit,
    ResolveOriginalPaintOperation,
    SourcePathBinding,
    VersionPathSelection,
    drawing_revision,
    operation_outcome_dependencies,
    snapshot_geometry_evidence,
    snapshot_paint_evidence,
    validate_replay_record,
)
from .drawing_refine import (
    apply_output_fill_settings,
    DrawingRegions,
    refine,
    refill_all_regions_from_source,
    refill_changed_regions_from_source,
    ordered_path_operation_stages,
    resolved_action_settings,
    run_detection_with_intent,
)
from .drawing_engine import PathFitOptions
from .path_fit_evidence import (
    FitBoundaryCorrespondence,
    FitEndpointCorrespondence,
    FitPathCorrespondence,
    FitSpanCorrespondence,
)
from .optimizer.automatic_replay import (
    ApplySeamConstraintsOperation,
    NormalizeRetainedTopologyOperation,
    PathEndpointRef,
    ReplayBatch,
    bind_intent,
    input_revision,
    replay_batches,
)
from .optimizer.topology import normalize_retained_topology
from .optimizer.symmetry_materialize import materialize_self_symmetry
from .optimizer.vector_region import VectorRegion, _parse_subpaths
from .optimizer.passes.seams import _command_id as seam_command_id
from .optimizer.passes.seams import _path_id as seam_path_id
from .path_geometry import AffineTransform
from .path_geometry import (
    CloneGeometry,
    CornerAnchor,
    PathGeometry,
    PathGeometryWithCutouts,
    CompoundPathGeometry,
    PathGeometryWithSymmetry,
    PrimitiveGeometry,
    parse_svg_paths,
)
from . import drawing_corner_actions
from .drawing_corner_actions import displayed_corner_id as _displayed_corner_id


@dataclass(frozen=True)
class ReplayExecution:
    """A canonical output and its executor-issued binding receipt."""

    drawing: Drawing
    replay: ReplayRecord
    regions: DrawingRegions


@dataclass(frozen=True)
class RecipeExtension:
    recipe: EffectiveDrawingRecipe
    submitted_edits: tuple[ResolvedCornerEdit, ...] = ()


@dataclass(frozen=True)
class _ProducedTarget:
    operation_id: str
    output_name: str
    descendant_ancestry: tuple[int, ...] = ()


def raster_pixels(source) -> np.ndarray:
    """Decode immutable packed paint pixels for the temporary Python adapter."""
    channels = 3 if source.pixel_format == "rgb8" else 4
    pixels = np.frombuffer(source.pixels, dtype=np.uint8).reshape(
        source.height, source.width, channels
    )
    return np.array(pixels[:, :, :3], copy=True)


def replay_regions(
    evidence: GeometryEvidence,
    paint: PaintEvidence,
    recipe: EffectiveDrawingRecipe,
    *, finalize_output: bool = True,
) -> DrawingRegions:
    """Replay exact actions on a fresh evidence snapshot, never a parent drawing."""
    regions, _producers, _corner_replays = _replay_regions_with_producers(
        evidence, paint, recipe, finalize_output=finalize_output
    )
    return regions


def _replay_regions_with_producers(
    evidence: GeometryEvidence,
    paint: PaintEvidence,
    recipe: EffectiveDrawingRecipe,
    *,
    rebound_steps: list[RecipeStep] | None = None,
    rebind_from_index: int = 0,
    finalize_output: bool = True,
) -> tuple[
    DrawingRegions,
    dict[str, _ProducedTarget],
    list[tuple[ApplyCornerRecipeOperation, CornerRecipeReplay]],
]:
    if recipe.evidence_revision != evidence.revision:
        raise ValueError("effective recipe evidence revision mismatch")
    if recipe.engine_revision != evidence.engine_revision:
        raise ValueError("effective recipe engine revision mismatch")
    regions = tuple(normalize_retained_topology(evidence.trace_roots))
    producers: dict[str, _ProducedTarget] = {}
    corner_replays: list[tuple[ApplyCornerRecipeOperation, CornerRecipeReplay]] = []
    paint_input: DrawingRegions | None = None
    index = 0
    while index < len(recipe.steps):
        step = recipe.steps[index]
        if isinstance(step, ReplayBatch):
            if paint_input is None:
                paint_input = regions
            batches = []
            while index < len(recipe.steps) and isinstance(
                recipe.steps[index], ReplayBatch
            ):
                batches.append(recipe.steps[index])
                if rebound_steps is not None:
                    rebound_steps.append(recipe.steps[index])
                index += 1
            group_masks = {
                int(root.id): np.asarray(root.raster, dtype=bool).copy()
                for root in regions
            }
            regions = tuple(normalize_retained_topology(replay_batches(
                list(regions), tuple(batches), masks=group_masks
            )))
            _register_automatic_outputs(tuple(batches), regions, producers)
            continue
        if isinstance(step, ResolveOriginalPaintOperation):
            if (
                rebound_steps is not None
                and index >= rebind_from_index
                and step.scope == "changed"
                and paint_input is not None
            ):
                step = replace(step, input_revision=input_revision(list(paint_input)))
            if evidence.geometry_to_paint != AffineTransform():
                raise ValueError(
                    "non-identity geometry-to-paint refill requires an exact sampler"
                )
            rgb = raster_pixels(paint.source)
            if step.scope == "retain":
                pass
            elif step.scope == "all":
                regions = refill_all_regions_from_source(
                    regions,
                    rgb,
                    share_root_gradients=step.share_root_gradients,
                    allow_gradients=paint.output_settings.allow_gradients,
                )
            else:
                if paint_input is None:
                    raise ValueError("changed paint resolution has no preceding geometry input")
                if input_revision(list(paint_input)) != step.input_revision:
                    raise ValueError("changed paint resolution input revision mismatch")
                regions = refill_changed_regions_from_source(paint_input, regions, rgb, allow_gradients=paint.output_settings.allow_gradients)
            paint_input = None
            if rebound_steps is not None:
                rebound_steps.append(step)
            index += 1
            continue
        if isinstance(step, ApplyCornerRecipeOperation):
            if rebound_steps is not None and index >= rebind_from_index:
                step = replace(step, input_revision=input_revision(list(regions)))
            if paint_input is None:
                paint_input = regions
            regions, corner_replay = _execute_corner_recipe(step, evidence, regions)
            corner_replays.append((step, corner_replay))
            if rebound_steps is not None:
                rebound_steps.append(step)
            index += 1
            continue
        if isinstance(step, RecipeOperation):
            self_contained_paint = isinstance(step.action, (SetFillAction, SetZOrderAction))
            if self_contained_paint and paint_input is not None:
                raise ValueError("public paint action follows unresolved geometry paint")
            if not self_contained_paint and paint_input is None:
                paint_input = regions
            _validate_operation_bindings(
                step,
                regions,
                drawing_id="recipe-binding",
                evidence_revision=evidence.revision,
                producers=producers,
            )
            before_operation = regions
            regions = _execute_public_operation(
                evidence, paint, step, regions, rgb=None,
                recipe=replace(recipe, steps=recipe.steps[:index]),
                corner_replays=corner_replays,
            )
            _register_public_outputs(step, before_operation, regions, producers)
            if rebound_steps is not None:
                rebound_steps.append(step)
            index += 1
            continue
        raise TypeError(f"unsupported effective recipe step: {type(step).__qualname__}")
    if finalize_output:
        regions = apply_output_fill_settings(regions, raster_pixels(paint.source), paint.output_settings)
    return regions, producers, corner_replays


def _path_geometry(
    geometry: object, path_id: str
) -> PathGeometry | None:
    if isinstance(geometry, PathGeometry):
        return geometry if geometry.path.id == path_id else None
    if isinstance(geometry, (PathGeometryWithCutouts, CompoundPathGeometry)):
        from .path_topology import component_paths
        return next(
            (
                path
                for path in component_paths(geometry)
                if path.path.id == path_id
            ),
            None,
        )
    return None


def _path_geometries(
    geometry: object,
) -> tuple[PathGeometry, ...]:
    if isinstance(geometry, PathGeometry):
        return (geometry,)
    if isinstance(geometry, CompoundPathGeometry):
        return geometry.paths
    if isinstance(geometry, PathGeometryWithCutouts):
        return (geometry.outer, *geometry.cutouts)
    return ()


def _execute_corner_recipe(
    application: ApplyCornerRecipeOperation,
    evidence: GeometryEvidence,
    regions: DrawingRegions,
) -> tuple[DrawingRegions, CornerRecipeReplay]:
    if application.evidence_revision != evidence.revision:
        raise ValueError("corner recipe application evidence binding is stale")
    if _region_ancestries(regions, "corner-binding").get(
        application.target_region_id
    ) != application.region_ancestry:
        raise ValueError("corner recipe application region binding is stale")
    if input_revision(list(regions)) != application.input_revision:
        raise ValueError("corner recipe application input revision is stale")
    matches = tuple(
        item
        for item in evidence.root_paths
        if item.region_id == application.target_region_id
        and item.path_id == application.target_path_id
        and item.fit.source.path.id == application.recipe.source_path_id
    )
    if len(matches) != 1:
        raise ValueError("corner recipe source binding is absent or ambiguous")
    source = matches[0]
    current = next(
        leaf
        for root in regions
        for leaf in root.leaves()
        if leaf.drawing_id == application.target_region_id
    )
    if current.current is None or current.current.kind != "path":
        raise ValueError("corner recipe target is not a path")
    drawing = drawing_from_vector_regions(
        "corner-execution", width=1.0, height=1.0, regions=regions
    )
    updated, replay = drawing_corner_actions.apply_region_corner_recipe(
        drawing.region(application.target_region_id), path_id=application.target_path_id,
        recipe=application.recipe, evidence=source.fit,
    )
    replaced = updated.geometry
    path_ids = tuple(path.path.id for path in _path_geometries(replaced))
    result = publish_region_update(updated, regions, shape_metadata={updated.id: {
        'path_ids': path_ids, 'fill_rule': replaced.fill_rule if isinstance(replaced, CompoundPathGeometry)
        else 'evenodd' if isinstance(replaced, PathGeometryWithCutouts) else 'nonzero',
    }})
    if application.downstream_batches:
        masks = {
            int(root.id): np.asarray(root.raster, dtype=bool).copy()
            for root in result
        }
        result = tuple(normalize_retained_topology(replay_batches(
            list(result), application.downstream_batches, masks=masks
        )))
        result = restore_corner_annotations(
            result,
            target_region_id=application.target_region_id,
            annotated=replaced,
        )
    return result, replay


def _region_ancestries(
    regions: DrawingRegions, drawing_id: str
) -> dict[str, tuple[int, ...]]:
    drawing = drawing_from_vector_regions(
        drawing_id,
        width=1.0,
        height=1.0,
        regions=regions,
    )
    result: dict[str, tuple[int, ...]] = {}

    def visit(source, canonical, ancestry: tuple[int, ...]) -> None:
        result[canonical.id] = ancestry
        for source_child, canonical_child in zip(
            source.children, canonical.children, strict=True
        ):
            visit(
                source_child,
                canonical_child,
                (*ancestry, int(source_child.id)),
            )

    for source, canonical in zip(regions, drawing.regions, strict=True):
        visit(source, canonical, (int(source.id),))
    return result


def _action_region_roles(action, regions: DrawingRegions) -> tuple[tuple[str, str], ...]:
    payload = action.model_dump(exclude_none=True)
    roles: list[tuple[str, str]] = []
    for field_name in ("source", "target", "reference"):
        value = payload.get(field_name)
        if isinstance(value, str):
            roles.append((field_name, value))
    for field_name in ("regions", "targets"):
        values = payload.get(field_name)
        if isinstance(values, (tuple, list)):
            roles.extend(
                (f"{field_name}.{index}", value)
                for index, value in enumerate(values)
                if isinstance(value, str)
            )
    if isinstance(action, StitchAction) and action.targets is None:
        roles.extend(
            (f"targets.{index}", region_id)
            for index, region_id in enumerate(_region_ancestries(regions, "binding"))
        )
    return tuple(dict.fromkeys(roles))


def _action_command_roles(action) -> tuple[tuple[str, str], ...]:
    if not isinstance(action, UpdateGeometryAction):
        return ()
    roles: list[tuple[str, str]] = []
    for index, path_action in enumerate(action.ops):
        payload = path_action.model_dump(exclude_none=True)
        for field_name in ("target", "reference"):
            value = payload.get(field_name)
            if isinstance(value, str):
                roles.append((f"ops.{index}.{field_name}", value))
        between = payload.get("between")
        if isinstance(between, (tuple, list)):
            roles.extend(
                (f"ops.{index}.between.{position}", value)
                for position, value in enumerate(between)
                if isinstance(value, str)
            )
    return tuple(roles)


def _command_owners(regions: DrawingRegions) -> dict[str, str]:
    drawing = drawing_from_vector_regions(
        "recipe-command-binding", width=1.0, height=1.0, regions=regions
    )
    owners: dict[str, str] = {}
    for root in drawing.regions:
        for leaf in root.leaves():
            geometry = leaf.geometry
            paths = ()
            if isinstance(geometry, PathGeometry):
                paths = (geometry,)
            elif isinstance(geometry, (PathGeometryWithCutouts, CompoundPathGeometry)):
                paths = _path_geometries(geometry)
            elif isinstance(geometry, PathGeometryWithSymmetry):
                paths = (geometry.half,)
            for path in paths:
                for command in path.path.commands:
                    previous = owners.setdefault(command.id, leaf.id)
                    if previous != leaf.id:
                        raise ValueError(
                            f"path command binding is ambiguous: {command.id!r}"
                        )
    return owners


def bind_recipe_operation(
    *,
    operation_id: str,
    action,
    settings,
    evidence_revision: str,
    regions: DrawingRegions,
    producers: dict[str, _ProducedTarget] | None = None,
) -> RecipeOperation:
    """Bind an accepted public action to exact current region ancestry."""
    ancestries = _region_ancestries(regions, "binding")
    producers = producers or {}
    bindings = []
    for role, region_id in _action_region_roles(action, regions):
        try:
            ancestry = ancestries[region_id]
        except KeyError as exc:
            raise ValueError(f"recipe target is absent: {region_id!r}") from exc
        producer = producers.get(region_id)
        bindings.append(
            NamedOutputBinding(
                role=role,
                operation_id=producer.operation_id,
                output_name=producer.output_name,
                target_region_id=region_id,
                descendant_ancestry=producer.descendant_ancestry,
            )
            if producer is not None
            else EvidenceTargetBinding(
                role=role,
                evidence_revision=evidence_revision,
                region_ancestry=ancestry,
                target_region_id=region_id,
            )
        )
    command_owners = _command_owners(regions)
    for role, command_id in _action_command_roles(action):
        try:
            region_id = command_owners[command_id]
            ancestry = ancestries[region_id]
        except KeyError as exc:
            raise ValueError(f"recipe path command is absent: {command_id!r}") from exc
        producer = producers.get(region_id)
        bindings.append(
            NamedOutputBinding(
                role=role,
                operation_id=producer.operation_id,
                output_name=producer.output_name,
                target_region_id=region_id,
                command_id=command_id,
                descendant_ancestry=producer.descendant_ancestry,
            )
            if producer is not None
            else EvidenceTargetBinding(
                role=role,
                evidence_revision=evidence_revision,
                region_ancestry=ancestry,
                target_region_id=region_id,
                source_path_id=command_id,
            )
        )
    return RecipeOperation(
        id=operation_id,
        origin="agent",
        action=action,
        settings=settings,
        bindings=tuple(bindings),
    )


def _validate_operation_bindings(
    operation: RecipeOperation,
    regions: DrawingRegions,
    *,
    drawing_id: str,
    evidence_revision: str,
    producers: dict[str, _ProducedTarget],
) -> None:
    ancestries = _region_ancestries(regions, drawing_id)
    command_owners = _command_owners(regions)
    expected = [
        *((role, value, None) for role, value in _action_region_roles(operation.action, regions)),
        *((role, command_owners.get(value), value) for role, value in _action_command_roles(operation.action)),
    ]
    if any(region_id is None for _role, region_id, _command in expected):
        missing = next(
            command for _role, region_id, command in expected if region_id is None
        )
        raise ValueError(f"public action path command is absent: {missing!r}")
    by_role = {binding.role: binding for binding in operation.bindings}
    if len(by_role) != len(operation.bindings) or set(by_role) != {
        role for role, _region_id, _command in expected
    }:
        raise ValueError("public action binding coverage does not match its references")
    for role, region_id, command_id in expected:
        assert region_id is not None
        binding = by_role[role]
        if binding.target_region_id != region_id:
            raise ValueError(f"public action target binding is stale: {role!r}")
        if isinstance(binding, EvidenceTargetBinding):
            if region_id in producers:
                raise ValueError("generated public action target requires a named-output binding")
            if binding.evidence_revision != evidence_revision:
                raise ValueError("public action evidence binding is stale")
            if binding.source_path_id != command_id:
                raise ValueError(f"public action path binding is stale: {role!r}")
            if ancestries.get(region_id) != binding.region_ancestry:
                raise ValueError(
                    f"public action region binding is stale: {region_id!r}"
                )
            continue
        producer = producers.get(region_id)
        if producer is None or (
            binding.operation_id,
            binding.output_name,
            binding.descendant_ancestry,
        ) != (
            producer.operation_id,
            producer.output_name,
            producer.descendant_ancestry,
        ):
            raise ValueError(f"public action named-output binding is stale: {region_id!r}")
        if binding.command_id != command_id:
            raise ValueError(f"public action named-output path binding is stale: {role!r}")


def _register_automatic_outputs(
    batches: tuple[ReplayBatch, ...],
    regions: DrawingRegions,
    producers: dict[str, _ProducedTarget],
) -> None:
    by_ancestry = {
        ancestry: region_id
        for region_id, ancestry in _region_ancestries(regions, "automatic-output").items()
    }
    for batch in batches:
        for record in batch.operations:
            for output in record.intent.outputs:
                region_id = by_ancestry.get(tuple(output.ancestry))
                if region_id is not None:
                    producers[region_id] = _ProducedTarget(
                        record.intent.operation_id, output.name
                    )


def _register_public_outputs(
    operation: RecipeOperation,
    before: DrawingRegions,
    current: DrawingRegions,
    producers: dict[str, _ProducedTarget],
) -> None:
    previous = set(_region_ancestries(before, "public-output"))
    for region_id in set(_region_ancestries(current, "public-output")) - previous:
        producers[region_id] = _ProducedTarget(operation.id, region_id)


def _execute_public_operation(
    evidence: GeometryEvidence,
    paint: PaintEvidence,
    operation: RecipeOperation,
    regions: DrawingRegions,
    *,
    rgb: np.ndarray | None,
    recipe: EffectiveDrawingRecipe,
    corner_replays: list[tuple[ApplyCornerRecipeOperation, CornerRecipeReplay]],
) -> DrawingRegions:
    if operation.selected_outcome is not None:
        expected = operation_outcome_dependencies(
            operation, evidence=evidence, paint=paint, upstream=recipe,
            input_revision=input_revision(list(regions)),
        )
        if expected != operation.selected_outcome.dependency_revision:
            raise ValueError("selected operation dependencies changed; submit a new proposal")
        return _apply_selected_outcome(operation.selected_outcome, regions)
    action = operation.action
    if isinstance(action, (AddCornerAction, ReplaceCornerAction, RemoveCornerAction)):
        raise ValueError("corner edits must use their source-bound recipe executor")
    if isinstance(action, StitchAction):
        raise ValueError("stitch must retain accepted concrete seam operations")
    if isinstance(action, UpdateGeometryAction) and any(
        item.op in {"simplify", "normalize_corners"} for item in action.ops
    ):
        raise ValueError(
            "semantic path cleanup must retain accepted concrete operations"
        )
    action = action.model_copy(update={"settings": ActionSettingsOverride()})
    plan = DrawingPlan(
        version="vectormark.plan.v1",
        drawing_id="recipe-execution",
        base_version="v0",
        settings=ActionSettingsOverride(**operation.settings.model_dump()),
        ops=(action,),
    )
    return refine(
        evidence.trace,
        regions,
        plan,
        rgb=rgb,
        fit_sources=(
            _fit_source_spans(evidence, regions, action, corner_replays, recipe)
            if isinstance(action, UpdateGeometryAction) else None
        ),
    )


def _apply_selected_outcome(
    outcome: SelectedOperationOutcome, regions: DrawingRegions,
) -> DrawingRegions:
    return restore_selected_roots(outcome.regions, outcome.root_order, regions,
        shape_metadata=outcome.shape_metadata, region_evidence=outcome.region_evidence)


def _retain_selected_outcome(
    operation: RecipeOperation, before: DrawingRegions, after: DrawingRegions,
    *, evidence: GeometryEvidence, paint: PaintEvidence, upstream: EffectiveDrawingRecipe,
) -> tuple[RecipeOperation, DrawingRegions]:
    drawing = drawing_from_vector_regions("selected-operation", width=1, height=1, regions=after)
    prior = {root.drawing_id: input_revision([root]) for root in before}
    selected = tuple(root for root, native_root in zip(drawing.regions, after)
        if prior.get(root.id) != input_revision([native_root]))
    native = {leaf.drawing_id: leaf for root in after for leaf in root.leaves()}
    metadata = {}
    for root in selected:
        for leaf in root.leaves():
            target = native[leaf.id]
            assert target.current is not None
            core_params = {} if isinstance(leaf.geometry, CloneGeometry) else shape_from_region_geometry(leaf, {}).params
            metadata[leaf.id] = {key: value for key, value in target.current.params.items() if key not in core_params}
    outcome = SelectedOperationOutcome(
        dependency_revision=operation_outcome_dependencies(
            operation, evidence=evidence, paint=paint, upstream=upstream,
            input_revision=input_revision(list(before)),
        ),
        regions=selected, root_order=tuple(root.id for root in drawing.regions), shape_metadata=metadata,
        region_evidence=working_region_evidence(tuple(root for root in after if root.drawing_id in {item.id for item in selected}), baseline=before),
    )
    # Capture and reuse traverse the same representation boundary, so following
    # steps bind to exactly the same geometry and metadata in either case.
    return replace(operation, selected_outcome=outcome), _apply_selected_outcome(outcome, before)


def _fit_source_spans(
    evidence: GeometryEvidence,
    regions: DrawingRegions,
    action: UpdateGeometryAction,
    corner_replays: list[tuple[ApplyCornerRecipeOperation, CornerRecipeReplay]],
    recipe: EffectiveDrawingRecipe,
) -> dict[str, np.ndarray]:
    """Resolve fitted public boundaries to exact original material spans."""
    drawing = drawing_from_vector_regions("fit-source", width=1.0, height=1.0, regions=regions)
    target = next((leaf for root in drawing.regions for leaf in root.leaves() if leaf.id == action.target), None)
    if target is None:
        raise ValueError("fit source region is absent")
    result: dict[str, np.ndarray] = {}
    for fit in action.ops:
        if fit.op != "fit" or fit.type == "keep":
            continue
        selected = fit.target if fit.target is not None else fit.between[0]
        path_id = selected.rsplit(".c", 1)[0]
        path = _path_geometry(target.geometry, path_id)
        if path is None:
            raise ValueError("fit source path is absent")
        ids = tuple(command.id for command in path.path.commands)
        if fit.between is not None:
            left, right = fit.between
            start_id = left
            end_id = right
        else:
            end_id = fit.target
            start_id = ids[ids.index(end_id) - 1]
        record = _path_record_for_display(evidence, regions, action.target, path_id, corner_replays, recipe)
        try:
            source_start = _source_command_from_correspondence(record.correspondence, path.path.boundary_command_id(start_id))
            source_end = _source_command_from_correspondence(record.correspondence, path.path.boundary_command_id(end_id))
        except ValueError as exc:
            raise ValueError(f"fit {start_id}..{end_id} has no exact original material-polyline span") from exc
        sources = tuple(item.fit.source for item in evidence.root_paths if item.fit.source.path.id == record.binding.source_path_id)
        if len(sources) != 1:
            raise ValueError("fit original material path is absent or ambiguous")
        source = sources[0]
        owned = _directed_source_owned_commands(source, source_start, source_end)
        if not owned:
            raise ValueError("fit original material span is empty or ambiguous")
        points = {command.id: command.end for command in source.path.commands}
        result[end_id] = np.asarray([points[source_start], *(points[item] for item in owned)], dtype=float)
    return result


def _path_record_for_display(
    evidence: GeometryEvidence,
    regions: DrawingRegions,
    target_region_id: str,
    target_path_id: str,
    corner_replays: list[tuple[ApplyCornerRecipeOperation, CornerRecipeReplay]],
    recipe: EffectiveDrawingRecipe,
) -> PathReplayCorrespondence:
    drawing = drawing_from_vector_regions(
        "path-correspondence", width=1.0, height=1.0, regions=regions
    )
    matching = tuple(
        record
        for record in _path_replay_records(
            evidence, recipe, regions, drawing, corner_replays
        )
        if record.output_region_id == target_region_id
        and record.output_path_id == target_path_id
    )
    if len(matching) != 1:
        raise ValueError("corner action source correspondence is absent or ambiguous")
    return matching[0]


def _source_path_for_target(
    evidence: GeometryEvidence,
    regions: DrawingRegions,
    target_region_id: str,
    target_path_id: str,
    corner_replays: list[tuple[ApplyCornerRecipeOperation, CornerRecipeReplay]],
    recipe: EffectiveDrawingRecipe,
) -> PathGeometry:
    record = _path_record_for_display(
        evidence,
        regions,
        target_region_id,
        target_path_id,
        corner_replays,
        recipe,
    )
    sources = tuple(
        item.fit.source
        for item in evidence.root_paths
        if item.path_id == record.binding.target_path_id
        and item.fit.source.path.id == record.binding.source_path_id
    )
    if len(sources) != 1:
        raise ValueError("corner action source path is absent or ambiguous")
    return sources[0]


def _source_command_for_display(
    evidence: GeometryEvidence,
    regions: DrawingRegions,
    target_region_id: str,
    target_path_id: str,
    command_id: str,
    corner_replays: list[tuple[ApplyCornerRecipeOperation, CornerRecipeReplay]],
    recipe: EffectiveDrawingRecipe,
) -> str:
    drawing = drawing_from_vector_regions(
        "corner-command", width=1.0, height=1.0, regions=regions
    )
    target = next((leaf for root in drawing.regions for leaf in root.leaves()
                   if leaf.id == target_region_id), None)
    geometry = None if target is None else _path_geometry(target.geometry, target_path_id)
    if geometry is None:
        raise ValueError("corner action displayed path is absent")
    command_id = geometry.path.boundary_command_id(command_id)
    record = _path_record_for_display(
        evidence,
        regions,
        target_region_id,
        target_path_id,
        corner_replays,
        recipe,
    )
    return _source_command_from_correspondence(
        record.correspondence, command_id
    )


def _source_feature_from_correspondence(
    geometry: PathGeometry,
    correspondence: FitPathCorrespondence,
    feature_id: str,
    *, source_path_id: str,
) -> str:
    """Resolve one reported corner by its exact retained boundary receipt."""
    anchors = tuple(anchor for anchor in geometry.anchors if anchor.id == feature_id)
    if len(anchors) != 1 or not isinstance(anchors[0], CornerAnchor):
        raise ValueError("corner action feature ID is absent or ambiguous")
    anchor = anchors[0]

    def boundary(command_id: str) -> str:
        # A closed path's move and explicit closing drawable name the same
        # topological node. Evidence deliberately binds the latter, not M.
        return geometry.path.boundary_command_id(command_id)

    expected = (
        {"point": boundary(anchor.entry_command_id)}
        if anchor.entry_command_id == anchor.exit_command_id
        else {
            "entry": boundary(anchor.entry_command_id),
            "exit": boundary(anchor.exit_command_id),
        }
    )
    candidates = {
        item.feature_id
        for item in correspondence.boundaries
        if {
            receipt.role: boundary(receipt.output_command_id)
            for receipt in correspondence.boundaries
            if receipt.feature_id == item.feature_id
        } == expected
    }
    named = {item for item in candidates
             if _displayed_corner_id(item, source_path_id, geometry.path.id) == feature_id}
    if named:
        candidates = named
    if len(candidates) != 1:
        raise ValueError("corner action feature has no unique source correspondence")
    return next(iter(candidates))


def _source_command_from_correspondence(
    correspondence: FitPathCorrespondence, command_id: str
) -> str:
    candidates = {
        item.source_command_id
        for item in correspondence.endpoint_bindings
        if item.output_command_id == command_id
    }
    if len(candidates) != 1:
        raise ValueError("corner action command has no unique exact source endpoint correspondence")
    return next(iter(candidates))


def _scoped_regions(
    regions: DrawingRegions, scope: tuple[int, ...]
) -> tuple[VectorRegion, ...]:
    current = tuple(regions)
    for region_id in scope:
        matches = tuple(item for item in current if int(item.id) == int(region_id))
        if len(matches) != 1:
            raise ValueError("corner downstream replay scope is absent or ambiguous")
        current = tuple(matches[0].children)
    return current


def _region_at_ancestry(
    regions: DrawingRegions, ancestry: tuple[int, ...]
) -> VectorRegion:
    current = tuple(regions)
    selected: VectorRegion | None = None
    for region_id in ancestry:
        matches = tuple(item for item in current if int(item.id) == int(region_id))
        if len(matches) != 1:
            raise ValueError("downstream seam region ancestry is absent or ambiguous")
        selected = matches[0]
        current = tuple(selected.children)
    if selected is None:
        raise ValueError("downstream seam region ancestry must not be empty")
    return selected


def _rebind_initial_endpoint_refs(
    value: object,
    *,
    batch_scope: tuple[int, ...],
    template_input: DrawingRegions,
    replay_input: DrawingRegions,
) -> object:
    if isinstance(value, PathEndpointRef):
        ancestry = (*batch_scope, *value.region.ancestry)
        old_region = _region_at_ancestry(template_input, ancestry)
        new_region = _region_at_ancestry(replay_input, ancestry)
        if old_region.current is None or new_region.current is None:
            raise ValueError("downstream seam endpoint target has no geometry")
        if value.path_id != seam_path_id(old_region.current):
            return value
        old_paths = _parse_subpaths(str(old_region.current.params.get("d", "")))
        matches = tuple(
            (subpath_index, command_index)
            for subpath_index, subpath in enumerate(old_paths)
            for command_index, (command, values) in enumerate(subpath)
            if seam_command_id(subpath_index, command_index, command, values)
            == value.command_id
        )
        if len(matches) != 1:
            raise ValueError("downstream seam endpoint command is absent or ambiguous")
        subpath_index, command_index = matches[0]
        new_paths = _parse_subpaths(str(new_region.current.params.get("d", "")))
        try:
            command, values = new_paths[subpath_index][command_index]
        except IndexError as exc:
            raise ValueError(
                "corner edit changes geometry with downstream compound seam stages; "
                "no exact endpoint rebinding is available"
            ) from exc
        if command != old_paths[subpath_index][command_index][0]:
            raise ValueError(
                "corner edit changes geometry with downstream compound seam stages; "
                "no exact endpoint rebinding is available"
            )
        return replace(
            value,
            path_id=seam_path_id(new_region.current),
            command_id=seam_command_id(
                subpath_index, command_index, command, values
            ),
        )
    if isinstance(value, tuple):
        return tuple(
            _rebind_initial_endpoint_refs(
                item,
                batch_scope=batch_scope,
                template_input=template_input,
                replay_input=replay_input,
            )
            for item in value
        )
    if is_dataclass(value):
        return replace(
            value,
            **{
                item.name: _rebind_initial_endpoint_refs(
                    getattr(value, item.name),
                    batch_scope=batch_scope,
                    template_input=template_input,
                    replay_input=replay_input,
                )
                for item in fields(value)
            },
        )
    return value


def _rebase_downstream_batches(
    templates: tuple[tuple[ReplayBatch, DrawingRegions], ...],
    regions: DrawingRegions,
    *,
    target_region_id: str,
    output_revision: str,
) -> tuple[ReplayBatch, ...]:
    current = tuple(regions)
    rebased: list[ReplayBatch] = []
    for batch_index, (template, template_input) in enumerate(templates):
        revision = input_revision(list(_scoped_regions(current, template.scope)))
        batch_id = "corner.downstream." + sha256(
            f"{template.batch_id}:{output_revision}:{batch_index}".encode("utf-8")
        ).hexdigest()[:20]
        entries = []
        input_leaf = next(
            leaf
            for root in template_input
            for leaf in root.leaves()
            if leaf.drawing_id == target_region_id
        )
        output_leaf = next(
            leaf
            for root in current
            for leaf in root.leaves()
            if leaf.drawing_id == target_region_id
        )
        if input_leaf.current is None or output_leaf.current is None:
            raise ValueError("corner downstream replay target has no geometry")
        if str(input_leaf.current.params.get("d", "")) != str(
            output_leaf.current.params.get("d", "")
        ):
            raise ValueError(
                "corner edit changes geometry with downstream compound seam stages; "
                "no exact endpoint rebinding is available"
            )
        for ordinal, record in enumerate(template.operations):
            mapped_action = _rebind_initial_endpoint_refs(
                record.intent.action,
                batch_scope=template.scope,
                template_input=template_input,
                replay_input=current,
            )
            draft = replace(
                record.intent, operation_id="", action=mapped_action
            )
            entries.append(replace(
                record,
                intent=bind_intent(
                    draft,
                    batch_id=batch_id,
                    input_revision=revision,
                    ordinal=ordinal,
                ),
            ))
        batch = replace(template, batch_id=batch_id, input_revision=revision,
                        entries=tuple(entries))
        masks = {
            int(root.id): np.asarray(root.raster, dtype=bool).copy()
            for root in current
        }
        current = tuple(replay_batches(list(current), (batch,), masks=masks))
        rebased.append(batch)
    return tuple(rebased)


def _batch_inputs_from_start(
    batches: tuple[ReplayBatch, ...], regions: DrawingRegions
) -> tuple[tuple[ReplayBatch, DrawingRegions], ...]:
    current = tuple(regions)
    result: list[tuple[ReplayBatch, DrawingRegions]] = []
    for batch in batches:
        result.append((batch, current))
        masks = {
            int(root.id): np.asarray(root.raster, dtype=bool).copy()
            for root in current
        }
        current = tuple(replay_batches(list(current), (batch,), masks=masks))
    return tuple(result)


def _directed_source_owned_commands(
    source: PathGeometry,
    entry_command_id: str,
    exit_command_id: str,
) -> tuple[str, ...]:
    if entry_command_id == exit_command_id:
        return ()
    command_ids = tuple(command.id for command in source.path.commands)
    try:
        start = command_ids.index(entry_command_id)
        end = command_ids.index(exit_command_id)
    except ValueError as exc:
        raise ValueError("corner action boundary is absent from its source path") from exc
    result: list[str] = []
    index = (start + 1) % len(command_ids)
    while True:
        result.append(command_ids[index])
        if index == end:
            return tuple(result)
        index = (index + 1) % len(command_ids)
        if index == (start + 1) % len(command_ids):
            raise ValueError("corner action boundaries do not define a directed source span")


def extend_recipe_with_plan(
    *,
    evidence: GeometryEvidence,
    paint: PaintEvidence,
    base_recipe: EffectiveDrawingRecipe,
    plan: DrawingPlan,
) -> RecipeExtension:
    """Resolve a submitted plan into an inherited, source-replayable recipe."""
    # Output palette constraints are terminal, not recipe steps. Capture new
    # operations from the same intermediate state full replay will encounter,
    # before its final output-color conversion changes the paint revision.
    regions, producers, corner_replays = _replay_regions_with_producers(
        evidence, paint, base_recipe, finalize_output=False
    )
    base_regions = regions
    base_corner_replays = corner_replays
    base_drawing = drawing_from_vector_regions(
        "corner-action", width=1.0, height=1.0, regions=base_regions
    )
    steps = list(base_recipe.steps)
    submitted_edits: list[ResolvedCornerEdit] = []
    operation_offset = len(base_recipe.operations)
    for index, action in enumerate(plan.ops):
        settings = resolved_action_settings(evidence.trace, plan, action)
        if isinstance(action, (AddCornerAction, ReplaceCornerAction, RemoveCornerAction)):
            current_recipe = EffectiveDrawingRecipe(
                evidence_revision=base_recipe.evidence_revision,
                engine_revision=base_recipe.engine_revision,
                settings=base_recipe.settings,
                steps=tuple(steps),
            )
            ancestry = _region_ancestries(regions, "corner-action").get(action.target)
            if ancestry is None:
                raise ValueError(f"corner action target is absent: {action.target!r}")
            drawing = drawing_from_vector_regions(
                "corner-action", width=1.0, height=1.0, regions=regions
            )
            target = next(
                (
                    leaf
                    for root in drawing.regions
                    for leaf in root.leaves()
                    if leaf.id == action.target
                ),
                None,
            )
            geometry = None if target is None else _path_geometry(
                target.geometry, action.path_id
            )
            if geometry is None:
                raise ValueError("corner action path ID is stale")
            base_target = next(
                (leaf for root in base_drawing.regions for leaf in root.leaves()
                 if leaf.id == action.target),
                None,
            )
            base_geometry = None if base_target is None else _path_geometry(
                base_target.geometry, action.path_id
            )
            retained_feature_id = action.feature_id
            if isinstance(action, (ReplaceCornerAction, RemoveCornerAction)):
                use_base = base_geometry is not None and any(
                    anchor.id == action.feature_id for anchor in base_geometry.anchors
                )
                record = _path_record_for_display(
                    evidence, base_regions if use_base else regions,
                    action.target, action.path_id,
                    base_corner_replays if use_base else corner_replays,
                    base_recipe if use_base else current_recipe,
                )
                retained_feature_id = _source_feature_from_correspondence(
                    base_geometry if use_base else geometry,
                    record.correspondence, action.feature_id,
                    source_path_id=record.binding.source_path_id,
                )
            if isinstance(action, RemoveCornerAction):
                entry_command_id = exit_command_id = geometry.path.commands[0].id
                edit: CornerRecipeEdit = RemoveCorner(retained_feature_id)
            else:
                # Submitted command IDs belong to the selected base version,
                # not to the transient refit produced by an earlier plan op.
                base_command_ids = (
                    {command.id for command in base_geometry.path.commands}
                    if base_geometry is not None else set()
                )
                use_base_commands = (
                    action.entry_command_id in base_command_ids
                    and action.exit_command_id in base_command_ids
                )
                command_regions = base_regions if use_base_commands else regions
                command_replays = base_corner_replays if use_base_commands else corner_replays
                command_recipe = base_recipe if use_base_commands else current_recipe
                source_entry = _source_command_for_display(
                    evidence,
                    command_regions,
                    action.target,
                    action.path_id,
                    action.entry_command_id,
                    command_replays,
                    command_recipe,
                )
                source_exit = _source_command_for_display(
                    evidence,
                    command_regions,
                    action.target,
                    action.path_id,
                    action.exit_command_id,
                    command_replays,
                    command_recipe,
                )
                source_path = _source_path_for_target(
                    evidence,
                    regions,
                    action.target,
                    action.path_id,
                    corner_replays,
                    current_recipe,
                )
                owned = _directed_source_owned_commands(
                    source_path, source_entry, source_exit
                )
                anchor = CornerAnchor(
                    retained_feature_id,
                    source_entry,
                    source_exit,
                    owned,
                    action.radius,
                )
                edit = (
                    AddCorner(anchor)
                    if isinstance(action, AddCornerAction)
                    else ReplaceCorner(retained_feature_id, anchor)
                )
                entry_command_id = action.entry_command_id
                exit_command_id = action.exit_command_id
                if use_base_commands:
                    current_record = _path_record_for_display(
                        evidence, regions, action.target, action.path_id,
                        corner_replays, current_recipe,
                    )

                    def current_boundary(source_command_id: str) -> str:
                        matches = {
                            receipt.output_command_id
                    for receipt in current_record.correspondence.endpoint_bindings
                            if receipt.source_command_id == source_command_id
                        }
                        if len(matches) != 1:
                            raise ValueError("corner plan boundary has no exact current binding")
                        return next(iter(matches))

                    entry_command_id = current_boundary(source_entry)
                    exit_command_id = current_boundary(source_exit)
            selection = VersionPathSelection(
                plan.base_version,
                evidence.revision,
                current_recipe.revision,
                ancestry,
                action.path_id,
                entry_command_id,
                exit_command_id,
            )
            current_recipe, resolved = extend_recipe_with_corner_edits(
                evidence=evidence,
                paint=paint,
                base_recipe=current_recipe,
                base_version_id=plan.base_version,
                selection=selection,
                edits=(edit,),
                settings=settings,
            )
            submitted_edits.extend(resolved)
            steps = list(current_recipe.steps)
            regions, producers, corner_replays = _replay_regions_with_producers(
                evidence, paint, current_recipe, finalize_output=False
            )
            continue
        before = regions
        detection = (
            action.op
            if action.op in {"detect_primitives", "detect_symmetry", "detect_clones", "stitch"}
            else None
        )
        target = (
            getattr(action, "targets", None)
            if action.op == "stitch"
            else getattr(action, "target", None)
        )
        if detection is not None:
            recorded = run_detection_with_intent(
                regions,
                evidence.trace,
                detection,
                target,
                epsilon=settings.epsilon,
                max_error=settings.max_error,
            )
            steps.extend(replace(batch, settings=settings, origin="agent")
                         for batch in recorded.batches)
            regions = recorded.regions
            _register_automatic_outputs(recorded.batches, regions, producers)
        else:
            stages = (
                ordered_path_operation_stages(action.ops)
                if isinstance(action, UpdateGeometryAction)
                else (("direct", (action,)),)
            )
            for stage_index, (stage_name, stage_actions) in enumerate(stages):
                if stage_name != "direct":
                    recorded = run_detection_with_intent(
                        regions,
                        evidence.trace,
                        stage_name,
                        action.target,
                        epsilon=settings.epsilon,
                        max_error=settings.max_error,
                    )
                    steps.extend(replace(batch, settings=settings, origin="agent")
                                 for batch in recorded.batches)
                    regions = recorded.regions
                    _register_automatic_outputs(
                        recorded.batches, regions, producers
                    )
                    continue
                direct_action = (
                    action.model_copy(update={"ops": tuple(stage_actions)})
                    if isinstance(action, UpdateGeometryAction)
                    else action
                )
                operation = bind_recipe_operation(
                    operation_id=(
                        f"agent.{operation_offset + index}.{action.op}.{stage_index}"
                    ),
                    action=direct_action,
                    settings=settings,
                    evidence_revision=evidence.revision,
                    regions=regions,
                    producers=producers,
                )
                before_operation = regions
                regions = _execute_public_operation(
                    evidence, paint, operation, regions, rgb=None,
                    recipe=replace(base_recipe, steps=tuple(steps)),
                    corner_replays=corner_replays,
                )
                operation, regions = _retain_selected_outcome(
                    operation, before_operation, regions, evidence=evidence,
                    paint=paint, upstream=replace(base_recipe, steps=tuple(steps)),
                )
                steps.append(operation)
                _register_public_outputs(
                    operation, before_operation, regions, producers
                )
        if action.op not in {"set_fill", "set_z_order"}:
            paint_step = ResolveOriginalPaintOperation(
                scope="changed",
                share_root_gradients=evidence.trace.material_pretrace is None,
                input_revision=input_revision(list(before)),
            )
            steps.append(paint_step)
            regions = refill_changed_regions_from_source(
                before, regions, raster_pixels(paint.source), allow_gradients=paint.output_settings.allow_gradients
            )
    return RecipeExtension(EffectiveDrawingRecipe(
        evidence_revision=base_recipe.evidence_revision,
        engine_revision=base_recipe.engine_revision,
        settings=base_recipe.settings,
        steps=tuple(steps),
    ), tuple(submitted_edits))


def extend_recipe_with_corner_edits(
    *,
    evidence: GeometryEvidence,
    paint: PaintEvidence,
    base_recipe: EffectiveDrawingRecipe,
    base_version_id: str,
    selection: VersionPathSelection,
    edits: tuple[CornerRecipeEdit, ...],
    settings: ActionSettings | None = None,
) -> tuple[EffectiveDrawingRecipe, tuple[ResolvedCornerEdit, ...]]:
    """Resolve typed corner edits against one exact displayed/source path pair."""
    if not edits:
        raise ValueError("corner recipe revision requires at least one edit")
    if selection.version_id != base_version_id:
        raise ValueError("corner selection version is stale")
    if selection.evidence_revision != evidence.revision:
        raise ValueError("corner selection evidence revision is stale")
    if selection.recipe_revision != base_recipe.revision:
        raise ValueError("corner selection recipe revision is stale")
    regions, _producers, corner_replays = _replay_regions_with_producers(
        evidence, paint, base_recipe, finalize_output=False
    )
    by_ancestry = {
        ancestry: region_id
        for region_id, ancestry in _region_ancestries(regions, "corner-selection").items()
    }
    try:
        target_region_id = by_ancestry[tuple(selection.region_ancestry)]
    except KeyError as exc:
        raise ValueError("corner selection region ancestry is stale") from exc
    drawing = drawing_from_vector_regions(
        "corner-selection", width=1.0, height=1.0, regions=regions
    )
    selected = tuple(
        leaf
        for root in drawing.regions
        for leaf in root.leaves()
        if leaf.id == target_region_id
    )
    if len(selected) != 1:
        raise ValueError("corner selection must identify one retained leaf")
    geometry = _path_geometry(selected[0].geometry, selection.path_id)
    if geometry is None:
        raise ValueError("corner selection path ID is stale")
    source_path = _source_path_for_target(
        evidence, regions, target_region_id, selection.path_id, corner_replays, base_recipe
    )
    command_ids = {
        command.id for path in (geometry, source_path) for command in path.path.commands
    }
    if selection.entry_command_id not in command_ids:
        raise ValueError("corner selection entry command is stale")
    if selection.exit_command_id not in command_ids:
        raise ValueError("corner selection exit command is stale")
    steps = list(base_recipe.steps)
    replay_by_application = {id(application): replay for application, replay in corner_replays}
    applications = [
        (index, step)
        for index, step in enumerate(steps)
        if isinstance(step, ApplyCornerRecipeOperation)
        and step.evidence_revision == evidence.revision
        and step.region_ancestry == selection.region_ancestry
        and replay_by_application.get(id(step)) is not None
        and step.target_path_id == selection.path_id
    ]
    if len(applications) > 1:
        raise ValueError("corner selection has ambiguous inherited recipes")
    path_record = _path_record_for_display(
        evidence,
        regions,
        target_region_id,
        selection.path_id,
        corner_replays,
        base_recipe,
    )
    sources = tuple(
        item
        for item in evidence.root_paths
        if item.path_id == path_record.binding.target_path_id
        and item.fit.source.path.id == path_record.binding.source_path_id
    )
    if len(sources) != 1:
        raise ValueError("corner selection source mapping is absent or ambiguous")
    source = sources[0]
    executor = CornerRecipeExecutor()
    if applications:
        application_index, application = applications[0]
        recipe = application.recipe
    else:
        application_index = -1
        recipe = executor.create(
            source.fit,
            PathFitOptions(
                base_recipe.settings.epsilon,
                base_recipe.settings.max_error,
                base_recipe.settings.fit_strategy,
            ),
        )

    settings = settings or (
        application.settings if applications else None
    ) or base_recipe.settings
    recipe = replace(recipe, options=PathFitOptions(
        settings.epsilon, settings.max_error, settings.fit_strategy,
    ))
    resolved: list[ResolvedCornerEdit] = []
    current_recipe = recipe
    for edit in edits:
        declarations = {
            declaration.anchor.id: declaration.anchor
            for declaration in current_recipe.declarations
        }
        if isinstance(edit, (AddCorner, ReplaceCorner)):
            anchor = edit.declaration
        elif isinstance(edit, RemoveCorner):
            try:
                anchor = declarations[edit.feature_id]
            except KeyError as exc:
                raise ValueError(
                    f"corner removal references unknown feature: {edit.feature_id!r}"
                ) from exc
        else:
            raise TypeError(f"unsupported corner edit: {type(edit).__qualname__}")
        binding = SourcePathBinding(
            evidence_revision=evidence.revision,
            region_ancestry=path_record.binding.region_ancestry,
            target_path_id=path_record.binding.target_path_id,
            source_path_id=source.fit.source.path.id,
            source_entry_command_id=anchor.entry_command_id,
            source_exit_command_id=anchor.exit_command_id,
            source_owned_command_ids=anchor.owned_command_ids,
        )
        resolved.append(ResolvedCornerEdit(selection, binding, edit))
        current_recipe = executor.revise(current_recipe, source.fit, (edit,))

    if applications:
        application_index, application = applications[0]
        pre_application_recipe = replace(
            base_recipe, steps=tuple(steps[:application_index])
        )
        pre_application_regions = replay_regions(
            evidence, paint, pre_application_recipe, finalize_output=False
        )
        prior_raw_application = replace(application, downstream_batches=())
        prior_raw_regions, _prior_raw_replay = _execute_corner_recipe(
            prior_raw_application, evidence, pre_application_regions
        )
        downstream_templates = _batch_inputs_from_start(
            application.downstream_batches, prior_raw_regions
        )
    else:
        downstream_templates_list: list[tuple[ReplayBatch, DrawingRegions]] = []
        downstream_indexes: set[int] = set()
        for step_index, step in enumerate(base_recipe.steps):
            if not (
                isinstance(step, ReplayBatch)
                and step.operations
                and all(
                    isinstance(record.intent.action, ApplySeamConstraintsOperation)
                    for record in step.operations
                )
            ):
                continue
            prefix = replace(base_recipe, steps=base_recipe.steps[:step_index])
            downstream_templates_list.append(
                (step, replay_regions(evidence, paint, prefix, finalize_output=False))
            )
            downstream_indexes.add(step_index)
        downstream_templates = tuple(downstream_templates_list)
        steps = [
            step
            for step_index, step in enumerate(base_recipe.steps)
            if step_index not in downstream_indexes
        ]
        pre_application_recipe = replace(base_recipe, steps=tuple(steps))
        pre_application_regions = replay_regions(
            evidence, paint, pre_application_recipe, finalize_output=False
        )

    raw_application = ApplyCornerRecipeOperation(
        recipe=current_recipe,
        evidence_revision=evidence.revision,
        region_ancestry=selection.region_ancestry,
        target_region_id=target_region_id,
        target_path_id=path_record.binding.target_path_id,
        input_revision=input_revision(list(pre_application_regions)),
        settings=settings,
    )
    raw_regions, raw_replay = _execute_corner_recipe(
        raw_application, evidence, pre_application_regions
    )
    downstream_batches = _rebase_downstream_batches(
        downstream_templates,
        raw_regions,
        target_region_id=target_region_id,
        output_revision=raw_replay.output_revision,
    )
    revised_application = replace(
        raw_application, downstream_batches=downstream_batches
    )
    if application_index >= 0:
        steps[application_index] = revised_application
    else:
        steps.extend((
            revised_application,
            ResolveOriginalPaintOperation(
                scope="changed",
                share_root_gradients=evidence.trace.material_pretrace is None,
                input_revision=input_revision(list(pre_application_regions)),
            ),
        ))
    revised = EffectiveDrawingRecipe(
        evidence_revision=base_recipe.evidence_revision,
        engine_revision=base_recipe.engine_revision,
        settings=base_recipe.settings,
        steps=tuple(steps),
    )
    # An earlier declaration edit changes the inputs of later corner and paint
    # operations, even when those operations address another region. Rebuild
    # their receipts while executing the revised recipe, then validate the
    # resulting immutable recipe through the ordinary strict replay path.
    rebound_steps: list[RecipeStep] = []
    _replay_regions_with_producers(
        evidence,
        paint,
        revised,
        rebound_steps=rebound_steps,
        rebind_from_index=max(application_index, 0),
    )
    revised = replace(revised, steps=tuple(rebound_steps))
    # Execute now so no malformed recipe can reach store publication.
    replay_regions(evidence, paint, revised)
    return revised, tuple(resolved)


def _operation_result_records(recipe: EffectiveDrawingRecipe) -> tuple[dict[str, object], ...]:
    """Report retained actions regardless of their execution representation."""
    results: list[dict[str, object]] = []
    for step in recipe.steps:
        if isinstance(step, RecipeOperation):
            operation_id, kind, status = step.id, "action", step.status
        elif isinstance(step, ReplayBatch):
            operation_id, kind = step.batch_id, "automatic_batch"
            status = "applied" if step.operations else "no_change"
        elif isinstance(step, ApplyCornerRecipeOperation):
            operation_id = f"corner.{step.target_region_id}.{step.target_path_id}"
            kind, status = "corner_recipe", "applied"
        else:
            continue
        settings = step.settings or recipe.settings
        results.append({
            "operation_id": operation_id,
            "kind": kind,
            "origin": step.origin,
            "purpose": settings.purpose,
            "execution": settings.execution,
            "status": status,
            "settings": settings.model_dump(),
        })
        if isinstance(step, ApplyCornerRecipeOperation):
            results.extend(_operation_result_records(replace(
                recipe, steps=step.downstream_batches,
            )))
    return tuple(results)


def propose_effective_recipe(
    *,
    drawing_id: str,
    evidence: GeometryEvidence,
    paint: PaintEvidence,
    recipe: EffectiveDrawingRecipe,
) -> ReplayExecution:
    """Execute intent to propose a fresh result, without publishing a version.

    Exploratory execution may return a different valid result on another call.
    Reproduction instead validates and uses a committed canonical outcome.
    """
    evidence = snapshot_geometry_evidence(evidence)
    paint = snapshot_paint_evidence(paint)
    if (paint.canvas.width, paint.canvas.height) != (
        float(evidence.trace.width),
        float(evidence.trace.height),
    ):
        raise ValueError("paint and geometry evidence canvases must match")
    regions, _producers, corner_replays = _replay_regions_with_producers(
        evidence, paint, recipe
    )
    drawing = drawing_from_vector_regions(
        drawing_id,
        width=paint.canvas.width,
        height=paint.canvas.height,
        regions=regions,
    )
    drawing = replace(drawing, background=paint.background, output_settings=paint.output_settings)
    replay = ReplayRecord(
        evidence_revision=evidence.revision,
        recipe_revision=recipe.revision,
        engine_revision=evidence.engine_revision,
        output_revision=drawing_revision(drawing),
        paint_revision=paint.revision,
        action_results=_operation_result_records(recipe),
        path_correspondence=_path_replay_records(
            evidence, recipe, regions, drawing, corner_replays
        ),
    )
    return ReplayExecution(drawing, replay, regions)


def reproduce_effective_recipe(
    *, drawing: Drawing, evidence: GeometryEvidence, paint: PaintEvidence,
    recipe: EffectiveDrawingRecipe, replay: ReplayRecord,
) -> Drawing:
    """Return an exact selected outcome; never rediscover or refit its geometry."""
    evidence = snapshot_geometry_evidence(evidence)
    paint = snapshot_paint_evidence(paint)
    validate_replay_record(
        drawing=drawing, evidence=evidence, paint=paint, recipe=recipe, replay=replay,
    )
    return drawing


def _remap_correspondence_to_geometry(
    correspondence: FitPathCorrespondence,
    *,
    recorded_output_path_id: str,
    geometry: PathGeometry,
    command_indexes: dict[int, int] | None = None,
) -> FitPathCorrespondence | None:
    prefix = f"{recorded_output_path_id}.c"

    def mapped(command_id: str) -> str | None:
        if not command_id.startswith(prefix):
            raise ValueError("path correspondence output command has an invalid identity")
        try:
            index = int(command_id[len(prefix):])
            mapped_index = (
                index if command_indexes is None else command_indexes.get(index)
            )
            if mapped_index is None:
                return None
            return geometry.path.commands[mapped_index].id
        except (ValueError, IndexError) as exc:
            raise ValueError("path correspondence output command is out of range") from exc

    boundaries: list[FitBoundaryCorrespondence] = []
    for item in correspondence.boundaries:
        output_command_id = mapped(item.output_command_id)
        if output_command_id is not None:
            boundaries.append(FitBoundaryCorrespondence(
                item.feature_id,
                item.role,
                item.source_command_id,
                output_command_id,
            ))
    spans: list[FitSpanCorrespondence] = []
    for item in correspondence.spans:
        output_command_ids = tuple(
            mapped(command_id) for command_id in item.output_command_ids
        )
        if all(command_id is not None for command_id in output_command_ids):
            spans.append(FitSpanCorrespondence(
                item.kind,
                item.source_command_ids,
                tuple(
                    command_id
                    for command_id in output_command_ids
                    if command_id is not None
                ),
                item.feature_id,
            ))
    # A downstream boolean/stitch may consume every complete fitted span.
    # Retain the geometry, but do not fabricate an editable source receipt.
    if not spans:
        return None
    return FitPathCorrespondence(
        tuple(boundaries),
        tuple(spans),
        tuple(
            FitEndpointCorrespondence(item.source_command_id, output_id)
            for item in correspondence.endpoints
            if (output_id := mapped(item.output_command_id)) is not None
        ),
    )


def _seam_command_mapping(
    batches: tuple[ReplayBatch, ...],
    *,
    ancestry: tuple[int, ...],
    subpath_index: int,
    command_count: int,
) -> tuple[dict[int, int], bool, tuple[int, ...], int] | None:
    """Compose only executor-issued seam lineage for one revision-bound path."""
    mapping = {index: index for index in range(command_count)}
    touched = False
    current_ancestry = ancestry
    current_subpath_index = subpath_index
    for batch in batches:
        for record in batch.operations:
            action = record.intent.action
            if isinstance(action, NormalizeRetainedTopologyOperation):
                matches = tuple(
                    item
                    for item in action.path_correspondence
                    if (*batch.scope, *item.target.ancestry) == current_ancestry
                    and item.input_subpath_index == current_subpath_index
                )
                if len(matches) > 1:
                    raise ValueError("retained topology path correspondence is ambiguous")
                if not matches:
                    affected_roots = tuple(
                        (*batch.scope, *root.ancestry) for root in action.roots
                    )
                    if any(
                        current_ancestry[: len(root)] == root
                        for root in affected_roots
                    ):
                        return None
                    continue
                topology = matches[0]
                if topology.command_count != len(mapping):
                    return None
                current_ancestry = (*batch.scope, *topology.output_ancestry)
                current_subpath_index = topology.output_subpath_index
                continue
            if not isinstance(action, ApplySeamConstraintsOperation):
                return None
            matches = tuple(
                item
                for item in action.path_correspondence
                if (*batch.scope, *item.target.ancestry) == current_ancestry
                and item.subpath_index == current_subpath_index
            )
            if len(matches) > 1:
                raise ValueError("seam path correspondence is ambiguous")
            if not matches:
                continue
            touched = True
            step = {
                item.input_index: item.output_index
                for item in matches[0].commands
            }
            mapping = {
                source_index: step[current_index]
                for source_index, current_index in mapping.items()
                if current_index in step
            }
    return mapping, touched, current_ancestry, current_subpath_index


def _recipe_command_mapping(
    recipe: EffectiveDrawingRecipe,
    *,
    region_id: str,
    path_id: str,
    ancestry: tuple[int, ...],
    subpath_index: int,
    command_count: int,
    output_ancestry: tuple[int, ...],
    output_subpath_index: int,
    output_matches_source: bool,
) -> dict[int, int] | None:
    batches: list[ReplayBatch] = []
    has_public_fit = False
    for step in recipe.steps:
        if isinstance(step, ResolveOriginalPaintOperation):
            continue
        if isinstance(step, ApplyCornerRecipeOperation):
            if (step.target_region_id, step.target_path_id) == (region_id, path_id):
                return None
            continue
        if isinstance(step, RecipeOperation):
            if isinstance(step.action, (SetFillAction, SetZOrderAction)):
                continue
            if isinstance(step.action, UpdateGeometryAction) and all(
                operation.op == "fit" for operation in step.action.ops
            ):
                # Fits issue endpoint lineage, composed by the caller below.
                has_public_fit = True
                continue
            return None
        if isinstance(step, ReplayBatch):
            if has_public_fit:
                # Lineage binds the preceding automatic output, not a later
                # boolean or optimizer rewrite.
                return None
            batches.append(step)
            continue
        return None
    result = _seam_command_mapping(
        tuple(batches),
        ancestry=ancestry,
        subpath_index=subpath_index,
        command_count=command_count,
    )
    if result is None:
        return None
    mapping, touched, actual_ancestry, actual_subpath_index = result
    if (
        actual_ancestry != output_ancestry
        or actual_subpath_index != output_subpath_index
    ):
        return None
    if not touched and not output_matches_source:
        return None
    return mapping


def _compose_public_path_lineage(
    command_indexes: dict[int, int],
    *,
    path_id: str,
    lineage: Mapping[str, object] | None,
    input_d: str,
    output_d: str,
) -> dict[int, int]:
    """Compose executor-proven endpoint bindings, never infer retired points."""
    if lineage is None:
        return command_indexes
    if lineage["input_d"] != input_d:
        raise ValueError("public path lineage does not bind the retained output")
    if lineage["output_d"] != output_d:
        raise ValueError("public path lineage output is stale")
    prefix = f"{path_id}.c"
    raw_commands = lineage["commands"]
    if not isinstance(raw_commands, Mapping):
        raise ValueError("public path lineage commands are malformed")
    commands: dict[str, str] = {}
    for old, new in raw_commands.items():
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError("public path lineage command identities are malformed")
        commands[old] = new
    if any(
        not commands[f"{prefix}{intermediate}"].startswith(prefix)
        for intermediate in command_indexes.values()
        if f"{prefix}{intermediate}" in commands
    ):
        raise ValueError("public path lineage crosses path ownership")
    return {
        source: int(commands[f"{prefix}{intermediate}"].removeprefix(prefix))
        for source, intermediate in command_indexes.items()
        if f"{prefix}{intermediate}" in commands
    }


def _source_binding(
    evidence_revision: str,
    ancestry: tuple[int, ...],
    target_path_id: str,
    source,
) -> SourcePathBinding:
    commands = source.path.commands
    return SourcePathBinding(
        evidence_revision=evidence_revision,
        region_ancestry=ancestry,
        target_path_id=target_path_id,
        source_path_id=source.path.id,
        source_entry_command_id=commands[0].id,
        source_exit_command_id=commands[-1].id,
        source_owned_command_ids=tuple(command.id for command in commands),
    )


def _correspondence_command_count(
    correspondence: FitPathCorrespondence, output_path_id: str
) -> int:
    prefix = f"{output_path_id}.c"
    command_ids = [
        item.output_command_id for item in correspondence.boundaries
    ]
    command_ids.extend(
        command_id
        for item in correspondence.spans
        for command_id in item.output_command_ids
    )
    indexes = [
        int(command_id[len(prefix):])
        for command_id in command_ids
        if command_id.startswith(prefix)
    ]
    if not indexes:
        raise ValueError("path fit correspondence has no output command identities")
    return max(indexes) + 1


def _path_replay_records(
    evidence: GeometryEvidence,
    recipe: EffectiveDrawingRecipe,
    regions: DrawingRegions,
    drawing: Drawing,
    corner_replays: list[tuple[ApplyCornerRecipeOperation, CornerRecipeReplay]],
) -> tuple[PathReplayCorrespondence, ...]:
    ancestries = _region_ancestries(regions, drawing.drawing_id)
    source_address_index = vector_region_address_index(evidence.trace_roots)
    source_ancestries = {
        region_id: ancestry
        for ancestry, region_id in source_address_index.items()
    }
    if len(source_ancestries) != len(source_address_index):
        raise ValueError("source region ownership is ambiguous")
    leaves = {
        leaf.id: leaf for root in drawing.regions for leaf in root.leaves()
    }
    working_leaves = {
        leaf.drawing_id: leaf for root in regions for leaf in root.leaves()
    }
    records: list[PathReplayCorrespondence] = []
    subpath_indices: dict[tuple[str, str], int] = {}
    next_subpath: dict[str, int] = {}
    for root in evidence.root_paths:
        index = next_subpath.get(root.region_id, 0)
        subpath_indices[(root.region_id, root.path_id)] = index
        next_subpath[root.region_id] = index + 1
    applied_paths = {
        (application.target_region_id, application.target_path_id)
        for application, _ in corner_replays
    }
    for application, corner_replay in corner_replays:
        sources = tuple(
            item.fit.source
            for item in evidence.root_paths
            if item.region_id == application.target_region_id
            and item.path_id == application.target_path_id
        )
        if len(sources) != 1:
            raise ValueError("corner replay receipt source binding is absent or ambiguous")
        final_leaf = leaves.get(application.target_region_id)
        final_path = None if final_leaf is None else _path_geometry(
            final_leaf.geometry, application.target_path_id
        )
        if final_path is None:
            continue
        mapping_result = _seam_command_mapping(
            application.downstream_batches,
            ancestry=application.region_ancestry,
            subpath_index=subpath_indices[
                (application.target_region_id, application.target_path_id)
            ],
            command_count=len(corner_replay.fit.geometry.path.commands),
        )
        if mapping_result is None:
            continue
        (
            command_indexes,
            _touched,
            output_ancestry,
            output_subpath_index,
        ) = mapping_result
        working = working_leaves[application.target_region_id]
        lineage = None if working.current is None else working.current.params.get("path_command_lineage")
        command_indexes = _compose_public_path_lineage(
            command_indexes,
            path_id=application.target_path_id,
            lineage=lineage if isinstance(lineage, Mapping) else None,
            input_d=corner_replay.fit.geometry.to_svg_d(),
            output_d="" if working.current is None else str(working.current.params["d"]),
        )
        if (
            output_ancestry != ancestries[application.target_region_id]
            or output_subpath_index
            != subpath_indices[
                (application.target_region_id, application.target_path_id)
            ]
        ):
            continue
        correspondence = _remap_correspondence_to_geometry(
            corner_replay.fit.correspondence,
            recorded_output_path_id=corner_replay.fit.geometry.path.id,
            geometry=final_path,
            command_indexes=command_indexes,
        )
        if correspondence is None:
            continue
        records.append(PathReplayCorrespondence(
            binding=_source_binding(
                evidence.revision,
                application.region_ancestry,
                application.target_path_id,
                sources[0],
            ),
            output_revision=corner_replay.output_revision,
            output_region_id=application.target_region_id,
            output_region_ancestry=application.region_ancestry,
            output_path_id=final_path.path.id,
            correspondence=correspondence,
        ))
    for root in evidence.root_paths:
        if (root.region_id, root.path_id) in applied_paths:
            continue
        outputs = tuple(
            (leaf, index, path)
            for leaf in leaves.values()
            for index, path in enumerate(_path_geometries(leaf.geometry))
            if path.path.id == root.path_id
        )
        if len(outputs) > 1:
            raise ValueError("replay output path ownership is ambiguous")
        if not outputs or root.region_id not in source_ancestries:
            continue
        leaf, output_subpath_index, geometry = outputs[0]
        ancestry = source_ancestries[root.region_id]
        working = working_leaves[leaf.id]
        lineage = None if working.current is None else working.current.params.get("path_command_lineage")
        retained_d = str(lineage["input_d"]) if isinstance(lineage, Mapping) else geometry.to_svg_d()
        retained_path_d = (
            parse_svg_paths(retained_d, region_id=leaf.id)[output_subpath_index].to_svg_d()
            if isinstance(lineage, Mapping) else retained_d
        )
        command_indexes = _recipe_command_mapping(
            recipe,
            region_id=root.region_id,
            path_id=root.path_id,
            ancestry=ancestry,
            subpath_index=subpath_indices[(root.region_id, root.path_id)],
            command_count=_correspondence_command_count(
                root.fit.correspondence, root.fit.output_path_id
            ),
            output_ancestry=ancestries[leaf.id],
            output_subpath_index=output_subpath_index,
            output_matches_source=root.fit.matches_output(retained_path_d),
        )
        if command_indexes is None:
            continue
        command_indexes = _compose_public_path_lineage(
            command_indexes,
            path_id=root.path_id,
            lineage=lineage if isinstance(lineage, Mapping) else None,
            input_d=retained_d,
            output_d="" if working.current is None else str(working.current.params["d"]),
        )
        correspondence = _remap_correspondence_to_geometry(
            root.fit.correspondence,
            recorded_output_path_id=root.fit.output_path_id,
            geometry=geometry,
            command_indexes=command_indexes,
        )
        if correspondence is None:
            continue
        records.append(PathReplayCorrespondence(
            binding=_source_binding(
                evidence.revision, ancestry, root.path_id, root.fit.source
            ),
            output_revision=root.fit.revision,
            output_region_id=leaf.id,
            output_region_ancestry=ancestries[leaf.id],
            output_path_id=geometry.path.id,
            correspondence=correspondence,
        ))
    return tuple(records)

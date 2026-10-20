from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Iterable
from typing import Protocol

import numpy as np

from .gate import BUDGET, rasterize
from ..skia_geometry import SkPath, unary_union
from ..quality import GeometryAssessment, GeometryPolicy, assess_geometry
from .vector_region import VectorRegion
from .symmetry_materialize import (
    index_region_paths,
    remap_symmetry_clones,
    resolve_source_path,
)
from .automatic_replay import (
    AutomaticActionOutcome,
    AutomaticOperationRecord,
    AutomaticObservationRecord,
    AutomaticRecord,
    CornerDiagnosticsObservation,
    OptimizationResult,
    ReplayBatch,
    DependentCloneRefresh,
    PassInputRegionRef,
    ProposalIntent,
    RefreshDependentClonesOperation,
    batch_identity,
    bind_intent,
    input_revision,
    operation_intent,
)

@dataclass(frozen=True)
class Proposal:
    obj_ids: tuple[int, ...]
    new_objects: list[VectorRegion]
    geometry_policy: GeometryPolicy | None = None
    intent: ProposalIntent = field(kw_only=True)

# The primitive fitter already admits a candidate with up to 4% geometric
# residual.  The framework guard must not reject a proposal that its producing
# pass has already accepted, while still stopping destructive scene rewrites.
_TRACE_FIDELITY_BUDGET = BUDGET * 2
_BUDGET_OMITTED = object()


@dataclass
class _ReplayCapture:
    batches: list[ReplayBatch]
    outcomes: list[AutomaticActionOutcome]


class Pass(Protocol):
    def __call__(
        self,
        objects: list[VectorRegion],
        masks: dict[int, np.ndarray],
    ) -> list[Proposal]: ...


def _proposal_key(proposal: Proposal) -> tuple[int, ...]:
    return tuple(sorted(int(obj_id) for obj_id in proposal.obj_ids))


def _object_key(obj: VectorRegion) -> tuple[float, int]:
    return (float(obj.z), int(obj.id))


def _shape_key(obj: VectorRegion) -> tuple[object, ...]:
    if obj.current is None:
        return (
            *_object_key(obj),
            "branch",
            tuple(_shape_key(child) for child in obj.leaves()),
            repr(obj.fill),
        )
    return (
        *_object_key(obj),
        obj.current.kind,
        tuple(sorted((str(k), repr(v)) for k, v in obj.current.params.items())),
        repr(obj.fill),
    )


def _proposal_sort_key(proposal: Proposal) -> tuple[tuple[int, ...], tuple[tuple[object, ...], ...]]:
    return (_proposal_key(proposal), tuple(sorted(_shape_key(obj) for obj in proposal.new_objects)))


from ..replay_inputs import union_masks


def _replacement_preserves_scene(
    replacements: list[VectorRegion],
    source_objects: list[VectorRegion],
    *,
    policy: GeometryPolicy,
) -> GeometryAssessment:
    """Require a replacement to preserve the vector scene it consumes.

    The first optimizer pass receives raw traced paths.  Later passes receive
    their vector successors, including intentional splits and occluders.  This
    keeps every proposal geometrically faithful without ever comparing it to
    the input raster or trying to reconstruct a root trace through new nodes.
    """
    try:
        source_geometry = unary_union([obj.footprint for obj in source_objects])
        predicted_geometry = unary_union([replacement.footprint for replacement in replacements])
        return assess_geometry(source_geometry, predicted_geometry, policy)
    except Exception:
        return assess_geometry(SkPath(), SkPath(), policy)


def _effective_policy_value(
    policy: GeometryPolicy | None, budget: float | object
) -> GeometryPolicy:
    if budget is _BUDGET_OMITTED:
        return policy if policy is not None else GeometryPolicy(
            max_relative_area_error=_TRACE_FIDELITY_BUDGET, preserve_topology=False
        )
    caller_limit = float(budget)
    if policy is None:
        return GeometryPolicy(max_relative_area_error=caller_limit, preserve_topology=False)
    relative_limit = policy.max_relative_area_error
    if not np.isfinite(caller_limit) or relative_limit is None or caller_limit < relative_limit:
        relative_limit = caller_limit
    return GeometryPolicy(
        max_relative_area_error=relative_limit,
        max_boundary_error_px=policy.max_boundary_error_px,
        max_perimeter_residual_error_px=policy.max_perimeter_residual_error_px,
        preserve_topology=policy.preserve_topology,
    )


def _effective_policy(proposal: Proposal, budget: float | object) -> GeometryPolicy:
    return _effective_policy_value(proposal.geometry_policy, budget)


def _pass_name(pass_fn: Pass) -> str:
    name = getattr(pass_fn, "__name__", None)
    if name:
        return str(name)
    return pass_fn.__class__.__name__


def accepted_proposal_diagnostics(
    pass_name: str, proposal_ids: tuple[int, ...], assessment: GeometryAssessment,
) -> dict[str, dict[str, object]]:
    """One diagnostic payload for accepted canonical and numeric replay results."""
    return {pass_name: {'accepted': True,
        'proposal_ids': [int(obj_id) for obj_id in proposal_ids],
        'geometry_assessment': assessment.diagnostics()}}


def _annotate_replacement(
    replacement: VectorRegion,
    *,
    pass_name: str,
    proposal_ids: tuple[int, ...],
    raster: np.ndarray,
    assessment: GeometryAssessment,
) -> VectorRegion:
    return replacement.with_diagnostics(
        raster=raster,
        diagnostics=accepted_proposal_diagnostics(pass_name, proposal_ids, assessment),
    )


def _masks_from_objects(objects, fallback: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for obj in objects:
        if obj.raster.size:
            out[obj.id] = np.asarray(obj.raster, dtype=bool).copy()
        elif obj.id in fallback:
            out[obj.id] = np.asarray(fallback[obj.id], dtype=bool).copy()
    return out


def _assign_replacement_masks(
    replacements: list[VectorRegion],
    proposal_ids: tuple[int, ...],
    *,
    pass_original_by_id: dict[int, VectorRegion],
    pass_masks: dict[int, np.ndarray],
    current_masks: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Apply the optimizer's exact accepted-proposal raster ownership rules."""
    from ..replay_inputs import assign_replacement_masks
    by_id = {region.id: region for region in replacements}
    return assign_replacement_masks(tuple(region.id for region in replacements), proposal_ids,
        original_ids=pass_original_by_id, pass_masks=pass_masks, current_masks=current_masks,
        raster_for=lambda identity, shape: rasterize(by_id[identity].footprint, shape))


def _optimize_branch_children(
    objects: list[VectorRegion],
    pass_fn: Pass,
    pass_name: str,
    *,
    budget: float | object,
    capture: _ReplayCapture | None,
    scope: tuple[int, ...],
) -> list[VectorRegion]:
    updated: list[VectorRegion] = []
    pending = _ReplayCapture([], [])
    for obj in objects:
        if obj.is_leaf:
            updated.append(obj)
            continue
        child_objects = list(obj.children)
        child_masks = _masks_from_objects(child_objects, {})
        child_capture = _ReplayCapture([], []) if capture is not None else None
        if budget is _BUDGET_OMITTED:
            optimized_children = optimize(
                child_objects,
                child_masks,
                [pass_fn],
                _capture=child_capture,
                _scope=(*scope, int(obj.id)),
            )
        else:
            optimized_children = optimize(
                child_objects,
                child_masks,
                [pass_fn],
                budget=float(budget),
                _capture=child_capture,
                _scope=(*scope, int(obj.id)),
            )
        if child_capture is not None:
            pending.batches.extend(child_capture.batches)
            pending.outcomes.extend(child_capture.outcomes)
        if len(optimized_children) == len(obj.children) and all(
            optimized is original
            for optimized, original in zip(optimized_children, obj.children, strict=True)
        ):
            updated.append(obj)
            continue
        updated.append(
            obj.with_children(
                optimized_children,
                diagnostics={
                    pass_name: {
                        "children_optimized": True,
                    }
                },
            )
        )
    # Recursion can reconstruct a source whose clones live outside its subtree.
    # Treat that source and every dependent clone as one outer transaction;
    # an accepted child proposal alone cannot authorize publishing those clones.
    before_leaves = {path: leaf for path, leaf in index_region_paths(objects).items() if leaf.is_leaf}
    sources = {
        path: path for path, leaf in index_region_paths(updated).items()
        if leaf.is_leaf and path in before_leaves
        and leaf.diagnostics.get("symmetry", {}).get("materialized")
        and leaf.current != before_leaves[path].current
    }
    if not sources:
        if capture is not None:
            capture.batches.extend(pending.batches)
            capture.outcomes.extend(pending.outcomes)
        return updated
    candidate = list(remap_symmetry_clones(updated, sources, before=objects))
    after_leaves = {path: leaf for path, leaf in index_region_paths(candidate).items() if leaf.is_leaf}
    affected = set(sources)
    affected.update(
        path for path, leaf in before_leaves.items()
        if leaf.current is not None and leaf.current.kind == "use"
        and path in after_leaves and after_leaves[path] is not leaf
        and "clone_reference_remap" in after_leaves[path].diagnostics
    )
    if affected == set(sources):
        if capture is not None:
            capture.batches.extend(pending.batches)
            capture.outcomes.extend(pending.outcomes)
        return updated
    replacements = [after_leaves[i] for i in sorted(affected)]
    assessment = _replacement_preserves_scene(
        replacements, [before_leaves[i] for i in sorted(affected)],
        policy=_effective_policy_value(None, budget),
    )
    root_ids = tuple(sorted({int(path[0]) for path in affected}))
    root_refs = tuple(PassInputRegionRef("", "", (obj_id,)) for obj_id in root_ids)
    prior = index_region_paths(objects)
    refreshes: list[DependentCloneRefresh] = []
    for path in sorted(affected - set(sources)):
        original = prior[path]
        assert original.current is not None
        old_path = resolve_source_path(
            prior, path, original.current.params, allow_external=True
        )
        if old_path is None:
            raise ValueError("dependent clone transaction lost its original source binding")
        new_path = sources.get(old_path, old_path)
        refreshes.append(DependentCloneRefresh(
            clone=PassInputRegionRef("", "", path),
            source=PassInputRegionRef("", "", new_path),
            previous_source_id=int(original.current.params["href_obj_id"]),
            previous_source_path=tuple(old_path),
        ))
    action = RefreshDependentClonesOperation(root_refs, tuple(refreshes))
    revision_masks = _masks_from_objects(updated, {})
    revision = input_revision(updated, revision_masks)
    refresh_pass_name = f"{pass_name}:dependent_clone_refresh"
    refresh_batch_id = batch_identity(refresh_pass_name, scope, revision, 0)
    refresh_intent = bind_intent(
        operation_intent(action, root_refs, root_ids),
        batch_id=refresh_batch_id,
        input_revision=revision,
        ordinal=0,
    )
    if assessment.accepted:
        if capture is not None:
            capture.batches.extend(pending.batches)
            capture.outcomes.extend(pending.outcomes)
            capture.batches.append(ReplayBatch(
                refresh_batch_id,
                revision,
                scope,
                (AutomaticOperationRecord(
                    refresh_intent,
                    refresh_pass_name,
                    root_ids,
                    assessment,
                ),),
            ))
            operation = capture.batches[-1].operations[0]
            capture.outcomes.append(AutomaticActionOutcome(
                operation,
                "accepted",
                {"pass_name": refresh_pass_name},
            ))
        return candidate
    if capture is not None:
        capture.outcomes.extend(
            AutomaticActionOutcome(
                outcome.operation,
                "rolled_back" if outcome.status == "accepted" else outcome.status,
                {"pass_name": pass_name, "reason": "self_symmetry_clone_transaction"},
            )
            for outcome in pending.outcomes
        )
        capture.outcomes.append(AutomaticActionOutcome(
            AutomaticOperationRecord(
                refresh_intent, refresh_pass_name, root_ids, assessment
            ),
            "rejected",
            {"pass_name": refresh_pass_name, "reason": "self_symmetry_clone_transaction"},
        ))
    return [obj.with_diagnostics({pass_name: {
        "accepted": False,
        "reason": "self_symmetry_clone_transaction",
        "proposal_ids": [path[-1] for path in sorted(affected)],
        "proposal_region_paths": sorted(affected),
        "geometry_assessment": assessment.diagnostics(),
    }}) if any(path[0] == obj.id for path in affected) else obj for obj in objects]


def optimize(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    passes: Iterable[Pass],
    *,
    budget: float | object = _BUDGET_OMITTED,
    on_pass: Callable[[str, list[VectorRegion]], None] | None = None,
    _capture: _ReplayCapture | None = None,
    _scope: tuple[int, ...] = (),
) -> list[VectorRegion]:
    current_objects = sorted(objects, key=_object_key)
    current_masks = {obj_id: np.asarray(mask, dtype=bool).copy() for obj_id, mask in masks.items()}

    for pass_ordinal, pass_fn in enumerate(passes):
        pass_name = _pass_name(pass_fn)
        current_objects = _optimize_branch_children(
            current_objects,
            pass_fn,
            pass_name,
            budget=budget,
            capture=_capture,
            scope=_scope,
        )
        current_masks = _masks_from_objects(current_objects, current_masks)
        current_objects = sorted(current_objects, key=_object_key)
        pass_objects = list(current_objects)
        pass_masks = {obj_id: mask.copy() for obj_id, mask in current_masks.items()}
        revision = input_revision(pass_objects, pass_masks)
        batch_id = batch_identity(pass_name, _scope, revision, pass_ordinal)
        pass_original_by_id = {obj.id: obj for obj in pass_objects}
        proposals = sorted(pass_fn(pass_objects, pass_masks), key=_proposal_sort_key)
        if _capture is not None and any(proposal.intent is None for proposal in proposals):
            raise ValueError(
                f"{pass_name} produced a proposal without concrete automatic intent"
            )
        consumed_in_pass: set[int] = set()
        committed_operations: list[AutomaticRecord] = []

        for proposal_ordinal, proposal in enumerate(proposals):
            bound_intent = (
                bind_intent(
                    proposal.intent,
                    batch_id=batch_id,
                    input_revision=revision,
                    ordinal=proposal_ordinal,
                )
                if _capture is not None and proposal.intent is not None else None
            )
            def reject_without_assessment(reason: str) -> None:
                if _capture is not None and bound_intent is not None:
                    operation = (
                        AutomaticObservationRecord(
                            bound_intent, pass_name, _proposal_key(proposal), None
                        )
                        if isinstance(bound_intent, CornerDiagnosticsObservation)
                        else AutomaticOperationRecord(
                            bound_intent, pass_name, _proposal_key(proposal), None
                        )
                    )
                    _capture.outcomes.append(AutomaticActionOutcome(
                        operation=operation,
                        status="rejected",
                        diagnostics={"pass_name": pass_name, "reason": reason},
                    ))

            raw_proposal_ids = [int(obj_id) for obj_id in proposal.obj_ids]
            if len(set(raw_proposal_ids)) != len(raw_proposal_ids):
                reject_without_assessment("duplicate_proposal_id")
                continue
            proposal_ids = _proposal_key(proposal)
            if not proposal_ids:
                reject_without_assessment("empty_proposal")
                continue
            if not proposal.new_objects:
                reject_without_assessment("empty_replacement")
                continue
            replacement_ids = [int(obj.id) for obj in proposal.new_objects]
            if len(set(replacement_ids)) != len(replacement_ids):
                reject_without_assessment("duplicate_replacement_id")
                continue
            if any(obj_id in consumed_in_pass for obj_id in proposal_ids):
                reject_without_assessment("overlapping_accepted_proposal")
                continue

            if any(obj_id not in pass_original_by_id or obj_id not in pass_masks for obj_id in proposal_ids):
                reject_without_assessment("missing_input_or_mask")
                continue
            if any(
                replacement.id in pass_original_by_id and replacement.id not in proposal_ids
                for replacement in proposal.new_objects
            ):
                reject_without_assessment("replacement_collides_with_input")
                continue

            if isinstance(bound_intent, CornerDiagnosticsObservation):
                if len(proposal_ids) != 1 or len(proposal.new_objects) != 1:
                    raise ValueError("corner diagnostic observation must retain one target")
                target = pass_original_by_id[proposal_ids[0]]
                expected = target.with_diagnostics({"corners": bound_intent.diagnostics})
                if input_revision(proposal.new_objects) != input_revision([expected]):
                    raise ValueError(
                        "corner diagnostic observation attempted a non-diagnostic mutation"
                    )
            current_object_ids = {obj.id for obj in current_objects}
            if any(
                replacement.id in current_object_ids and replacement.id not in proposal_ids
                for replacement in proposal.new_objects
            ):
                reject_without_assessment("replacement_collides_with_current")
                continue

            union_mask = union_masks(pass_masks, proposal_ids)
            source_objects = [pass_original_by_id[obj_id] for obj_id in proposal_ids]
            assessment_objects = proposal.new_objects
            replacement_leaves = [leaf for obj in proposal.new_objects for leaf in obj.leaves()]
            if (
                any(leaf.diagnostics.get("symmetry", {}).get("materialized") for leaf in replacement_leaves)
                and any("clone_reference_remap" in leaf.diagnostics for leaf in replacement_leaves)
            ):
                # A clone under a semantic branch paints its leaf footprint,
                # not the branch's retained exterior. Assess the actual joint
                # source/instance transaction, including cross-subtree clones.
                assessment_objects = replacement_leaves
                source_objects = [leaf for obj in source_objects for leaf in obj.leaves()]
            assessment = _replacement_preserves_scene(
                assessment_objects,
                source_objects,
                policy=_effective_policy(proposal, budget),
            )
            if not assessment.accepted:
                rejected = set(proposal_ids)
                current_objects = [
                    obj.with_diagnostics({pass_name: {
                        "accepted": False,
                        "proposal_ids": [int(obj_id) for obj_id in proposal_ids],
                        "geometry_assessment": assessment.diagnostics(),
                    }}) if obj.id in rejected else obj
                    for obj in current_objects
                ]
                if _capture is not None and bound_intent is not None:
                    operation = (
                        AutomaticObservationRecord(
                            bound_intent, pass_name, proposal_ids, assessment
                        )
                        if isinstance(bound_intent, CornerDiagnosticsObservation)
                        else AutomaticOperationRecord(
                            bound_intent, pass_name, proposal_ids, assessment
                        )
                    )
                    _capture.outcomes.append(AutomaticActionOutcome(
                        operation=operation,
                        status="rejected",
                        diagnostics={"pass_name": pass_name},
                    ))
                continue
            consumed_in_pass.update(proposal_ids)

            replacements = sorted(proposal.new_objects, key=_object_key)
            remaining = [obj for obj in current_objects if obj.id not in proposal_ids]
            replacement_masks = _assign_replacement_masks(
                replacements,
                proposal_ids,
                pass_original_by_id=pass_original_by_id,
                pass_masks=pass_masks,
                current_masks=current_masks,
            )

            annotated_replacements = [
                _annotate_replacement(
                    replacement,
                    pass_name=pass_name,
                    proposal_ids=proposal_ids,
                    raster=replacement_masks[replacement.id],
                    assessment=assessment,
                )
                for replacement in replacements
            ]
            current_objects = sorted(remaining + annotated_replacements, key=_object_key)
            if _capture is not None and bound_intent is not None:
                committed = (
                    AutomaticObservationRecord(
                        intent=bound_intent,
                        pass_name=pass_name,
                        proposal_ids=proposal_ids,
                        assessment=assessment,
                    )
                    if isinstance(bound_intent, CornerDiagnosticsObservation)
                    else AutomaticOperationRecord(
                        intent=bound_intent,
                        pass_name=pass_name,
                        proposal_ids=proposal_ids,
                        assessment=assessment,
                    )
                )
                committed_operations.append(committed)
                _capture.outcomes.append(AutomaticActionOutcome(
                    operation=committed,
                    status="accepted",
                    diagnostics={"pass_name": pass_name},
                ))

        if _capture is not None:
            _capture.batches.append(ReplayBatch(
                batch_id=batch_id,
                input_revision=revision,
                scope=_scope,
                entries=tuple(committed_operations),
            ))
        if on_pass is not None:
            on_pass(pass_name, list(current_objects))

    return current_objects


def optimize_with_intent(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    passes: Iterable[Pass],
    *,
    budget: float | object = _BUDGET_OMITTED,
    on_pass: Callable[[str, list[VectorRegion]], None] | None = None,
) -> OptimizationResult:
    """Run the optimizer and retain only committed concrete automatic intent."""
    capture = _ReplayCapture([], [])
    optimized = optimize(
        objects,
        masks,
        passes,
        budget=budget,
        on_pass=on_pass,
        _capture=capture,
    )
    return OptimizationResult(optimized, tuple(capture.batches), tuple(capture.outcomes))

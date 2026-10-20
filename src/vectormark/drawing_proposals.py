"""Automatic choices assessed and published directly against retained drawings."""

from dataclasses import dataclass, replace

from .drawing import DrawingRegion, _walk_regions
from .drawing_automatic import AutomaticBatchApplication, AutomaticChoiceNotApplicable
from .geometry_native import region_path
from .path_geometry import CloneGeometry
from .primitive_fitting import recognize_primitive_geometry
from .proposal_validation import proposal_rejection_reason
from .quality import GeometryPolicy, assess_geometry, assess_geometry_regions
from .skia_geometry import SkPath
from .refinement.automatic_replay import (
    AutomaticActionOutcome, AutomaticObservationRecord, AutomaticOperationRecord,
    CornerDiagnosticsObservation, ProposalIntent, ReplayBatch, batch_identity,
    bind_intent, primitive_intent,
)
from .refinement.framework import _BUDGET_OMITTED, _effective_policy_value
from .replay_identity import canonical_input_revision, scoped_input_regions
from .replay_state import ReplayState


@dataclass(frozen=True)
class CanonicalProposal:
    proposal_ids: tuple[int, ...]
    intent: ProposalIntent
    geometry_policy: GeometryPolicy | None = None
    region_geometry_policy: GeometryPolicy | None = None


@dataclass(frozen=True)
class CanonicalPassResult:
    state: ReplayState
    batch: ReplayBatch
    outcomes: tuple[AutomaticActionOutcome, ...]


@dataclass(frozen=True)
class CanonicalDetectionResult:
    state: ReplayState
    batches: tuple[ReplayBatch, ...]
    outcomes: tuple[AutomaticActionOutcome, ...]


def detect_primitive_regions(
    state: ReplayState, *, epsilon: float, target_id: str | None = None,
) -> CanonicalDetectionResult:
    """Select primitives in child scopes first, retaining full region addressing."""
    selected_ids = None if target_id is None else {
        region.id for region in _walk_regions(state.drawing.region(target_id))}
    scopes: list[tuple[int, ...]] = []

    def visit(roots, scope):
        for region in roots:
            if region.children:
                visit(region.children, (*scope, int(state.region_evidence[region.id]['id'])))
        scopes.append(scope)

    visit(state.drawing.regions, ())
    batches, outcomes = [], []
    for scope in scopes:
        proposals = discover_primitive_proposals(state, epsilon=epsilon, scope=scope)
        if selected_ids is not None:
            local_ids = {int(state.region_evidence[item.id]['id'])
                for item in scoped_input_regions(state, scope) if item.id in selected_ids}
            proposals = tuple(item for item in proposals if local_ids.intersection(item.proposal_ids))
        result = select_canonical_pass(state, proposals, pass_name='pass_fn', scope=scope)
        state = result.state
        batches.append(result.batch)
        outcomes.extend(result.outcomes)
    from .drawing_topology import normalize_drawing_topology
    from .replay_state import publish_drawing_state
    state = publish_drawing_state(state, normalize_drawing_topology(state.drawing))
    return CanonicalDetectionResult(state, tuple(batches), tuple(outcomes))


def discover_primitive_proposals(
    state: ReplayState, *, epsilon: float, scope: tuple[int, ...] = (),
) -> tuple[CanonicalProposal, ...]:
    """Recognize primitives without projecting a disposable sampling scene."""
    by_id = {item.id: item for root in state.drawing.regions for item in _walk_regions(root)}
    proposals = []
    for region in sorted(scoped_input_regions(state, scope),
                         key=lambda item: int(state.region_evidence[item.id]['id'])):
        if not region.is_leaf or isinstance(region.geometry, CloneGeometry):
            continue
        primitive = recognize_primitive_geometry(region_path(region, by_id), epsilon=epsilon)
        if primitive is None or primitive == region.geometry:
            continue
        identity = int(state.region_evidence[region.id]['id'])
        policy = GeometryPolicy(max_boundary_error_px=epsilon,
            max_perimeter_residual_error_px=epsilon, preserve_topology=True)
        proposals.append(CanonicalProposal((identity,),
            primitive_intent(identity, primitive, epsilon=epsilon), policy))
    return tuple(proposals)


def _leaf_footprints(roots: tuple[DrawingRegion, ...], context: dict[str, DrawingRegion]):
    indexed = {**context, **{item.id: item for root in roots for item in _walk_regions(root)}}
    return {item.id: region_path(item, indexed)
        for root in roots for item in _walk_regions(root) if item.is_leaf}


def select_canonical_pass(
    state: ReplayState, proposals: tuple[CanonicalProposal, ...], *, pass_name: str,
    scope: tuple[int, ...] = (), pass_ordinal: int = 0,
    budget: float | object = _BUDGET_OMITTED,
) -> CanonicalPassResult:
    """Select immutable proposals; publish the exact prepared choice once.

    The pass snapshot remains authoritative for addressing and error checks.
    Only accepted candidates consume inputs or enter the reproducible batch.
    """
    revision = canonical_input_revision(state, scope)
    batch = ReplayBatch(batch_identity(pass_name, scope, revision, pass_ordinal), revision, scope, ())
    application = AutomaticBatchApplication(state, batch)
    original = {int(state.region_evidence[item.id]['id']): item
        for item in scoped_input_regions(state, scope)}
    context = {item.id: item for root in state.drawing.regions for item in _walk_regions(root)}
    consumed: set[int] = set()
    entries, outcomes = [], []
    for ordinal, proposal in enumerate(proposals):
        intent = bind_intent(proposal.intent, batch_id=batch.batch_id,
            input_revision=revision, ordinal=ordinal)
        identities = tuple(sorted(proposal.proposal_ids))
        record_type = AutomaticObservationRecord if isinstance(intent, CornerDiagnosticsObservation) else AutomaticOperationRecord
        current_ids = {int(application.observations[item.id]['id']) for item in application.current}
        reason = proposal_rejection_reason(proposal.proposal_ids, None,
            input_ids=original, mask_ids=application.masks, consumed_ids=consumed, current_ids=current_ids)
        selected = None
        assessment = None
        if reason is None:
            try:
                selected = application.prepare(intent)
            except AutomaticChoiceNotApplicable:
                reason = 'no_applicable_candidate'
        if reason is None:
            replacements = tuple(item.region for item in selected.regions)
            replacement_ids = tuple(int(item.observation['id']) for item in selected.regions)
            reason = proposal_rejection_reason(proposal.proposal_ids, replacement_ids,
                input_ids=original, mask_ids=application.masks, consumed_ids=consumed, current_ids=current_ids)
            if reason is None:
                policy = _effective_policy_value(proposal.geometry_policy, budget)
                region_policy = (_effective_policy_value(proposal.region_geometry_policy, budget)
                    if proposal.region_geometry_policy is not None else None)
                try:
                    assessment = assess_geometry_regions(
                        _leaf_footprints(tuple(original[identity] for identity in identities), context),
                        _leaf_footprints(replacements, context), policy, region_policy=region_policy)
                except Exception:
                    assessment = assess_geometry(SkPath(), SkPath(), policy)
        record = record_type(intent, pass_name, identities, assessment)
        accepted = reason is None and assessment is not None and assessment.accepted
        diagnostics = {'pass_name': pass_name}
        if reason is not None:
            diagnostics['reason'] = reason
        outcomes.append(AutomaticActionOutcome(record,
            'accepted' if accepted else 'rejected', diagnostics))
        if accepted:
            application.accept(record, selected)
            entries.append(record)
            consumed.update(identities)
    return CanonicalPassResult(application.finish(), replace(batch, entries=tuple(entries)), tuple(outcomes))

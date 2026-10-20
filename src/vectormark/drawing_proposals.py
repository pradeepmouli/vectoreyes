"""Automatic choices assessed and published directly against retained drawings."""

from dataclasses import dataclass, field, replace
from collections.abc import Callable
from typing import Mapping

from .drawing import DrawingRegion, _walk_regions
from .drawing_automatic import AutomaticBatchApplication, AutomaticChoiceNotApplicable, PreparedAutomaticSelection
from .geometry_native import region_path
from .path_geometry import CloneGeometry, PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry
from .primitive_fitting import recognize_primitive_geometry
from .proposal_validation import proposal_rejection_reason
from .quality import GeometryPolicy, assess_geometry, assess_geometry_regions
from .skia_geometry import SkPath
from .refinement.automatic_replay import (
    AutomaticActionOutcome, AutomaticObservationRecord, AutomaticOperationRecord,
    CornerDiagnosticsObservation, ProposalIntent, ReplayBatch, batch_identity,
    bind_intent, primitive_intent, draft_input_ref, operation_intent,
    SmoothPathOperation, NormalizeCornersOperation, SimplifyLineletOperation, AssignCloneOperation,
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
    prepared: PreparedAutomaticSelection | None = field(default=None, kw_only=True)
    addressed_ids: tuple[int, ...] = field(default=(), kw_only=True)


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
    return _select_region_scopes(state,
        lambda snapshot, scope: discover_primitive_proposals(snapshot, epsilon=epsilon, scope=scope),
        target_id=target_id)


def _select_region_scopes(
    state: ReplayState,
    discover: Callable[[ReplayState, tuple[int, ...]], tuple[CanonicalProposal, ...]],
    *, target_id: str | None, pass_name: str = 'pass_fn',
) -> CanonicalDetectionResult:
    """Share child-first discovery and full-scope publication across detectors."""
    selected_ids = None if target_id is None else {
        region.id for region in _walk_regions(state.drawing.region(target_id))}
    scopes: list[tuple[int, ...]] = []

    def visit(roots: tuple[DrawingRegion, ...], scope: tuple[int, ...]) -> None:
        for region in roots:
            if region.children:
                visit(region.children, (*scope, int(state.region_evidence[region.id]['id'])))
        scopes.append(scope)

    visit(state.drawing.regions, ())
    batches, outcomes = [], []
    for scope in scopes:
        proposals = discover(state, scope)
        if selected_ids is not None:
            local_ids = {int(state.region_evidence[item.id]['id'])
                for item in scoped_input_regions(state, scope) if item.id in selected_ids}
            proposals = tuple(item for item in proposals if local_ids.intersection(item.addressed_ids or item.proposal_ids))
        result = select_canonical_pass(state, proposals, pass_name=pass_name, scope=scope)
        state = result.state
        batches.append(result.batch)
        outcomes.extend(result.outcomes)
    from .drawing_topology import normalize_drawing_topology
    from .replay_state import publish_drawing_state
    state = publish_drawing_state(state, normalize_drawing_topology(state.drawing))
    return CanonicalDetectionResult(state, tuple(batches), tuple(outcomes))


def discover_clone_proposals(
    state: ReplayState, *, symbolic: bool = True, scope: tuple[int, ...] = (),
) -> tuple[CanonicalProposal, ...]:
    """Match canonical leaf footprints without projecting a sampling scene."""
    from .clone_detection import CloneDetectionInput, find_clone_matches
    indexed = {item.id: item for root in state.drawing.regions for item in _walk_regions(root)}
    inputs = []
    for region in scoped_input_regions(state, scope):
        if not region.is_leaf or region.fill is None or isinstance(region.geometry, CloneGeometry):
            continue
        symmetry = region.diagnostics.get('symmetry')
        if isinstance(symmetry, Mapping) and symmetry.get('mode') == 'self':
            continue
        footprint = region_path(region, indexed)
        if not footprint.is_empty:
            inputs.append(CloneDetectionInput(int(state.region_evidence[region.id]['id']), footprint))
    proposals = []
    for match in find_clone_matches(tuple(inputs)):
        source, target = draft_input_ref(match.source_id), draft_input_ref(match.target_id)
        action = AssignCloneOperation(source, target, match.matrix, symbolic, match.family_size)
        proposals.append(CanonicalProposal((match.target_id,),
            operation_intent(action, (source, target), (match.target_id,)),
            addressed_ids=(match.source_id, match.target_id)))
    return tuple(proposals)


def discover_occlusion_proposals(
    state: ReplayState, *, no_symmetry: bool = False, scope: tuple[int, ...] = (),
) -> tuple[CanonicalProposal, ...]:
    """Discover completed primitive graphs directly from retained raster evidence."""
    from .drawing_fill import FlatFill
    from .occlusion_detection import OcclusionDetectionInput, discover_occlusion_intents
    from .replay_state import retained_region_raster, decode_region_coverage

    roots = scoped_input_regions(state, scope)
    inputs = []
    for region in roots:
        if not region.is_leaf:
            continue
        info = state.region_evidence[region.id]
        color = region.fill.color if isinstance(region.fill, FlatFill) else info['color_hex']
        raster = retained_region_raster(state, region.id)
        if color is not None and raster.size:
            inputs.append(OcclusionDetectionInput(int(info['id']), raster, color,
                region.z, decode_region_coverage(info)))
    if len(inputs) < 2:
        return ()
    # Retiring a source requires an explicit clone remapping transaction, not a
    # local occlusion replacement that would leave the old address unresolved.
    referenced = {item.geometry.source_region_id for root in state.drawing.regions
        for item in _walk_regions(root) if isinstance(item.geometry, CloneGeometry)}
    protected = {int(state.region_evidence[item.id]['id']) for item in roots
        if item.id in referenced or isinstance(item.geometry, CloneGeometry)}
    next_id = max((int(state.region_evidence[item.id]['id']) for item in roots), default=-1)+1
    shape_hw = retained_region_raster(state, roots[0].id).shape
    return tuple(CanonicalProposal(tuple(int(ref.ancestry[-1]) for ref in intent.action.component), intent)
        for intent in discover_occlusion_intents(tuple(inputs), shape_hw=shape_hw,
            next_id=next_id, no_symmetry=no_symmetry)
        if not protected.intersection(int(ref.ancestry[-1]) for ref in intent.action.component))


def detect_occlusion_regions(
    state: ReplayState, *, target_id: str | None = None, no_symmetry: bool = False,
) -> CanonicalDetectionResult:
    """Select occlusion reconstructions in fully addressed child-first scopes."""
    return _select_region_scopes(state,
        lambda snapshot, scope: discover_occlusion_proposals(snapshot,
            no_symmetry=no_symmetry, scope=scope), target_id=target_id)


def discover_symmetry_proposals(
    state: ReplayState, *, epsilon: float = 1.5, max_error: float = 1.0,
    cubic: bool = False, scope: tuple[int, ...] = (),
) -> tuple[CanonicalProposal, ...]:
    """Discover retained mirror/intrinsic/self choices using shared numeric kernels."""
    import numpy as np
    from . import symmetry_detection as detection
    from .drawing_fill import FlatFill
    from .path_geometry import PrimitiveGeometry, CirclePrimitive, EllipsePrimitive, RectanglePrimitive, PolygonPrimitive
    from .fit import Shape
    from .refinement.region_sample import to_polygon
    from .refinement.automatic_replay import (
        MirrorRegionOperation, DeclareIntrinsicSymmetryOperation, ReconstructSelfSymmetryOperation,
    )

    indexed = {item.id: item for root in state.drawing.regions for item in _walk_regions(root)}
    roots = scoped_input_regions(state, scope)
    local_ids = {item.id for root in roots for item in _walk_regions(root)}
    usable = []
    for region in sorted(roots, key=lambda item: int(state.region_evidence[item.id]['id'])):
        if not region.is_leaf or isinstance(region.geometry, CloneGeometry):
            continue
        if isinstance(region.diagnostics.get('compound'), Mapping):
            continue
        flat = region_path(region, indexed)
        if flat.is_empty or not detection._automatic_symmetry_is_tractable(flat):
            continue
        usable.append((region, flat, int(state.region_evidence[region.id]['id'])))

    proposals, paired = [], set()
    for index, (_source, source_flat, source_id) in enumerate(usable):
        if source_id in paired:
            continue
        for _target, target_flat, target_id in usable[index+1:]:
            if target_id in paired:
                continue
            match = detection.mirror_pair(source_flat, target_flat)
            if match is None:
                continue
            axis, matrix = match
            refs = (draft_input_ref(source_id), draft_input_ref(target_id))
            action = MirrorRegionOperation(*refs, detection._axis_tuple(axis), matrix)
            proposals.append(CanonicalProposal((source_id, target_id),
                operation_intent(action, refs, (source_id, target_id))))
            paired.update((source_id, target_id))
            break

    for region, flat, identity in usable:
        if identity in paired:
            continue
        seed = isinstance(region.diagnostics.get('geometry_seed'), Mapping)
        if isinstance(region.diagnostics.get('material_surface'), Mapping) and not seed:
            continue
        explicit = region.diagnostics.get('geometry', {})
        explicit_kind = explicit.get('explicit') if isinstance(explicit, Mapping) else None
        primitive = region.geometry.primitive if isinstance(region.geometry, PrimitiveGeometry) else None
        intrinsic_only = ((not isinstance(region.fill, FlatFill) and not seed)
            or isinstance(primitive, (CirclePrimitive, EllipsePrimitive)) or type(primitive) is RectanglePrimitive
            or explicit_kind in {'circle', 'ellipse', 'rect', 'rounded_rect', 'cap', 'trapezoid', 'rounded_trapezoid'})
        if isinstance(primitive, PolygonPrimitive) and not intrinsic_only:
            continue
        dependents = [root for root in roots if root.id != region.id and any(
            isinstance(item.geometry, CloneGeometry) and item.geometry.source_region_id == region.id
            for item in _walk_regions(root))]
        external_dependents = any(isinstance(item.geometry, CloneGeometry)
            and item.geometry.source_region_id == region.id and item.id not in local_ids
            for item in indexed.values())
        best = None
        if not intrinsic_only and not external_dependents:
            info = state.region_evidence[region.id]
            reconstruction = flat
            if seed and info['original_kind'] is not None:
                original = to_polygon(Shape(info['original_kind'], dict(info['original_params'])))
                if not original.is_empty:
                    reconstruction = original
            mask = np.frombuffer(info['raster'], dtype=bool).reshape(info['raster_shape'])
            best = detection._best_self_reconstruction(reconstruction, mask,
                epsilon=epsilon, max_error=max_error, cubic=cubic,
                preserve_explicit_geometry=explicit_kind is not None)
        ref = draft_input_ref(identity)
        if best is not None:
            dependent_ids = tuple(int(state.region_evidence[item.id]['id']) for item in dependents)
            refs = (ref, *(draft_input_ref(item) for item in dependent_ids))
            ids = (identity, *dependent_ids)
            action = ReconstructSelfSymmetryOperation(ref, refs[1:], detection._axis_tuple(best.axis),
                best.residual, best.fit_kind, best.side, best.corner_radius, epsilon, max_error, cubic)
            proposals.append(CanonicalProposal(ids, operation_intent(action, refs, ids)))
        else:
            intrinsic = detection._best_self_axis(flat)
            if intrinsic is not None and (intrinsic_only or intrinsic[1] <= 1e-9):
                axis, residual = intrinsic
                action = DeclareIntrinsicSymmetryOperation(ref, detection._axis_tuple(axis), residual)
                proposals.append(CanonicalProposal((identity,), operation_intent(action, (ref,), (identity,))))
    return tuple(proposals)


def detect_symmetry_regions(
    state: ReplayState, *, target_id: str | None = None, epsilon: float = 1.5,
    max_error: float = 1.0, cubic: bool = False,
) -> CanonicalDetectionResult:
    """Select canonical symmetry choices in child-first, fully addressed scopes."""
    return _select_region_scopes(state,
        lambda snapshot, scope: discover_symmetry_proposals(snapshot, epsilon=epsilon,
            max_error=max_error, cubic=cubic, scope=scope), target_id=target_id)


def detect_clone_regions(
    state: ReplayState, *, target_id: str | None = None, symbolic: bool = True,
) -> CanonicalDetectionResult:
    """Select clone relationships while retaining independent target paint."""
    return _select_region_scopes(state,
        lambda snapshot, scope: discover_clone_proposals(snapshot, symbolic=symbolic, scope=scope),
        target_id=target_id)


def discover_path_proposals(
    state: ReplayState, *, operation: str, max_error: float,
    corner_normalize: bool = True, scope: tuple[int, ...] = (),
) -> tuple[CanonicalProposal, ...]:
    """Declare path cleanup choices; selection prepares each numeric fit once."""
    from .corner_diagnostics import retained_corner_evidence
    if operation not in {'smooth', 'normalize_corners'}:
        raise ValueError(f'unknown canonical path discovery operation: {operation}')
    proposals = []
    for region in sorted(scoped_input_regions(state, scope),
                         key=lambda item: int(state.region_evidence[item.id]['id'])):
        if not region.is_leaf or not isinstance(region.geometry,
                (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
            continue
        identity = int(state.region_evidence[region.id]['id'])
        target = draft_input_ref(identity)
        if operation == 'smooth':
            intent = operation_intent(SmoothPathOperation(target, max_error), (target,), (identity,))
        else:
            d = region.geometry.to_svg_d()
            authoritative, declared = retained_corner_evidence(d, region.diagnostics.get('corners'))
            if corner_normalize and authoritative:
                intent = operation_intent(NormalizeCornersOperation(target, max_error, declared),
                    (target,), (identity,))
            else:
                intent = CornerDiagnosticsObservation('', target, declared)
        proposals.append(CanonicalProposal((identity,), intent))
    return tuple(proposals)


def detect_path_regions(
    state: ReplayState, *, operation: str, max_error: float,
    corner_normalize: bool = True, target_id: str | None = None,
    epsilon: float = 1.5, samples: int = 16, cubic: bool = False,
    linelet_only: bool = False, max_boundary_error: float | None = None,
) -> CanonicalDetectionResult:
    """Run one path-only discovery operation without a sampling-scene projection."""
    if operation == 'simplify':
        return _select_region_scopes(state,
            lambda snapshot, scope: discover_simplification_proposals(snapshot, epsilon=epsilon,
                max_error=max_error, samples=samples, cubic=cubic, linelet_only=linelet_only,
                max_boundary_error=max_boundary_error, scope=scope), target_id=target_id)
    return _select_region_scopes(state,
        lambda snapshot, scope: discover_path_proposals(snapshot, operation=operation,
            max_error=max_error, corner_normalize=corner_normalize, scope=scope), target_id=target_id)


def discover_simplification_proposals(
    state: ReplayState, *, epsilon: float, max_error: float, samples: int = 16,
    cubic: bool = False, linelet_only: bool = False,
    max_boundary_error: float | None = None, scope: tuple[int, ...] = (),
) -> tuple[CanonicalProposal, ...]:
    """Choose the existing covered-cutout or topology-preserving fit exactly once."""
    from .drawing_simplification import covering_later_paths
    from .path_sampling import _parse_subpaths
    revision = canonical_input_revision(state, scope)
    batch = ReplayBatch(batch_identity('simplification_discovery', scope, revision, 0), revision, scope, ())
    application = AutomaticBatchApplication(state, batch)
    roots = scoped_input_regions(state, scope)
    by_id = {item.id: item for root in state.drawing.regions for item in _walk_regions(root)}
    referenced = {item.geometry.source_region_id for item in by_id.values()
        if isinstance(item.geometry, CloneGeometry)}
    footprints = {root.id: region_path(root, by_id) for root in roots}
    proposals = []
    for region in sorted(roots, key=lambda item: int(state.region_evidence[item.id]['id'])):
        if (not region.is_leaf or region.id in referenced or not isinstance(region.geometry,
                (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)) or footprints[region.id].is_empty):
            continue
        identity = int(state.region_evidence[region.id]['id'])
        d = region.geometry.to_svg_d()
        if linelet_only and len(_parse_subpaths(d)) > 1:
            continue
        target = draft_input_ref(identity)

        def prepared_proposal(action: SimplifyLineletOperation, covers: tuple[int, ...]) -> CanonicalProposal | None:
            refs = (target, *(draft_input_ref(item) for item in covers))
            identities = (identity, *covers)
            intent = operation_intent(action, refs, identities)
            bound = bind_intent(intent, batch_id=batch.batch_id, input_revision=revision, ordinal=len(proposals))
            try:
                selected = application.prepare(bound)
            except AutomaticChoiceNotApplicable:
                return None
            return CanonicalProposal(identities, intent, prepared=selected)

        if state.shape_metadata.get(region.id, {}).get('fill_rule'):
            covers = covering_later_paths(d, identity=identity, z=region.z,
                candidates=tuple((int(state.region_evidence[item.id]['id']), item.z, footprints[item.id]) for item in roots))
            if covers is not None:
                cover_refs = tuple(draft_input_ref(item) for item in covers)
                preferred = prepared_proposal(SimplifyLineletOperation(target, cover_refs,
                    epsilon, max_error, samples, cubic, linelet_only, max_boundary_error, False, False), tuple(covers))
                if preferred is not None:
                    proposals.append(preferred)
                    continue
        chosen = prepared_proposal(SimplifyLineletOperation(target, (), epsilon, max_error,
            samples, cubic, linelet_only, max_boundary_error, True, True), ())
        if chosen is not None:
            proposals.append(chosen)
    return tuple(proposals)


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
                if proposal.prepared is None:
                    selected = application.prepare(intent)
                else:
                    selected = proposal.prepared
                    rebound = bind_intent(selected.intent, batch_id=batch.batch_id,
                        input_revision=revision, ordinal=ordinal)
                    if selected.input_revision != revision or selected.scope != scope or rebound != intent:
                        raise ValueError('prepared proposal binding does not match its selected input and settings')
                    selected = replace(selected, intent=intent)
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

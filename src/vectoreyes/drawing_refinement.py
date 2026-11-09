"""Default refinement orchestrates independently replayable canonical actions."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

from .drawing_proposals import (
    CanonicalDetectionResult, CanonicalProposal, _select_region_scopes,
    discover_clone_proposals, discover_occlusion_proposals, discover_path_proposals,
    discover_primitive_proposals, discover_seam_proposals,
    discover_simplification_proposals, discover_symmetry_proposals,
)
from .refinement_settings import RefinementPasses, refinement_pass_names
from .replay_state import ReplayState


def discover_topology_proposals(state: ReplayState, *, scope: tuple[int, ...] = ()) -> tuple[CanonicalProposal, ...]:
    """Record component ownership using the same preparation as strict replay."""
    from .drawing_automatic import (
        AutomaticBatchApplication, PreparedAutomaticSelection, _capture_inputs, _prepare_topology_regions,
    )
    from .replay_identity import canonical_input_revision, scoped_input_regions
    from .refinement.automatic_replay import (
        NormalizeRetainedTopologyOperation, ReplayBatch, batch_identity, bind_intent,
        draft_input_ref, operation_intent,
    )
    roots = scoped_input_regions(state, scope)
    refs = tuple(draft_input_ref(int(state.region_evidence[root.id]['id'])) for root in roots)
    action = NormalizeRetainedTopologyOperation(refs, (), ())
    revision = canonical_input_revision(state, scope)
    batch = ReplayBatch(batch_identity('topology_discovery', scope, revision, 0), revision, scope, ())
    application = AutomaticBatchApplication(state, batch)
    bound_input = bind_intent(operation_intent(action, refs, ()), batch_id=batch.batch_id,
        input_revision=revision, ordinal=0)
    selected, correspondence = _prepare_topology_regions(state, bound_input.action, application.resolver,
        application.observations, application.metadata)
    changed = tuple(after for before, after in zip(roots, selected, strict=True) if before != after)
    if not changed:
        return ()
    identities = tuple(int(application.observations[root.id]['id']) for root in changed)
    action = replace(action, affected_ids=identities, path_correspondence=correspondence)
    intent = operation_intent(action, refs, identities)
    bound = bind_intent(intent, batch_id=batch.batch_id, input_revision=revision, ordinal=0)
    prepared = PreparedAutomaticSelection(revision, bound,
        _capture_inputs(changed, application.observations, application.metadata), scope)
    return (CanonicalProposal(identities, intent, prepared=prepared),)


@dataclass(frozen=True)
class CanonicalRefinementResult(CanonicalDetectionResult):
    """Computed per-pass states and their exact batches, without rediscovery."""

    passes: tuple[tuple[str, CanonicalDetectionResult], ...] = ()


def refine_canonical_drawing(
    state: ReplayState, *, epsilon: float, max_error: float,
    corner_normalize: bool = False, refinement_passes: RefinementPasses | None = None,
    no_symmetry: bool = False, cubic: bool = False,
    on_pass: Callable[[str, ReplayState], None] | None = None,
    band_source_context=None,
) -> CanonicalRefinementResult:
    """Run the shared schedule on canonical values; retain every accepted choice."""
    discoverers = {
        'occlusion_pass': partial(discover_occlusion_proposals, no_symmetry=no_symmetry),
        'split_compound_pass': discover_topology_proposals,
        'primitives_pass': partial(discover_primitive_proposals, epsilon=epsilon),
        'clones_pass': discover_clone_proposals,
        'symmetry_pass': partial(discover_symmetry_proposals, epsilon=epsilon, max_error=max_error, cubic=cubic),
        'linelet_simplify_pass': partial(discover_simplification_proposals, epsilon=epsilon,
            max_error=max_error, cubic=cubic, linelet_only=True, max_boundary_error=max_error),
        'smooth_pass': partial(discover_path_proposals, operation='smooth', max_error=max_error),
        'corner_normalize_pass': partial(discover_path_proposals, operation='normalize_corners',
            max_error=max_error, corner_normalize=corner_normalize),
        'seams_pass': partial(discover_seam_proposals, epsilon=epsilon, max_error=max_error, cubic=cubic),
    }
    batches, outcomes, passes = [], [], []
    for ordinal, name in enumerate(refinement_pass_names(refinement_passes or RefinementPasses(),
            no_symmetry=no_symmetry)):
        if name == 'corner_normalize_pass' and not corner_normalize:
            continue
        if name == 'bands_pass':
            from .drawing_bands import refine_band_regions
            result = refine_band_regions(state, context=band_source_context, preceding_steps=tuple(batches))
        else:
            discover = discoverers[name]
            result = _select_region_scopes(state, lambda current, scope: discover(current, scope=scope),
                target_id=None, pass_name=name, pass_ordinal=ordinal)
        state = result.state
        batches.extend(result.batches)
        outcomes.extend(result.outcomes)
        passes.append((name, result))
        if on_pass is not None:
            on_pass(name, state)
    return CanonicalRefinementResult(state, tuple(batches), tuple(outcomes), tuple(passes))

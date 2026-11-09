"""Retain and apply selected canonical root deltas without working scenes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from .drawing import DrawingRegion, _walk_regions
from .drawing_recipe import (
    EffectiveDrawingRecipe, GeometryEvidence, PaintEvidence, RecipeOperation,
    SelectedOperationOutcome, _digest, _snapshot_canonical, operation_outcome_dependencies,
)
from .replay_state import ReplayState


def selected_input_revision(state: ReplayState) -> str:
    """Bind the complete canonical input, independent of temporary drawing IDs."""
    return _digest(('selected-input.v1', state.drawing.canvas, state.drawing.regions,
        state.shape_metadata, state.region_evidence))


def region_ancestries(state: ReplayState) -> dict[str, tuple[int, ...]]:
    result = {}
    for root in state.drawing.regions:
        for region in _walk_regions(root):
            observation = state.region_evidence.get(region.id, {})
            if 'ancestry' not in observation:
                raise ValueError(f'region source ancestry is absent: {region.id!r}')
            result[region.id] = tuple(observation['ancestry'])
    return result


def _root_inputs(state: ReplayState, root: DrawingRegion) -> tuple[object, ...]:
    identities = {region.id for region in _walk_regions(root)}
    return (root, {key: value for key, value in state.shape_metadata.items() if key in identities},
        {key: value for key, value in state.region_evidence.items() if key in identities})


def _borrow_observations(
    selected: Mapping[str, Mapping[str, object]], before: ReplayState,
) -> dict[str, dict[str, object]]:
    result = {}
    for identity, observation in selected.items():
        info = dict(observation)
        prior = before.region_evidence.get(identity)
        if prior is not None:
            for field, shape_fields in (
                ('raster', ('raster_shape',)),
                ('coverage', ('coverage_shape', 'coverage_dtype', 'coverage_origin', 'coverage_canvas')),
            ):
                if (field in info and info.get(field) == prior.get(field)
                        and all(info.get(key) == prior.get(key) for key in shape_fields)):
                    info[field] = None
                    info[f'{field}_source'] = identity
        result[identity] = info
    return result


def apply_selected_outcome(outcome: SelectedOperationOutcome, before: ReplayState) -> ReplayState:
    prior = {root.id: root for root in before.drawing.regions}
    changed = {root.id: _snapshot_canonical(root) for root in outcome.regions}
    if any(identity not in changed and identity not in prior for identity in outcome.root_order):
        raise ValueError('selected outcome references an unavailable root')
    roots = tuple(changed[identity] if identity in changed else prior[identity] for identity in outcome.root_order)
    live = {region.id for root in roots for region in _walk_regions(root)}
    selected = {region.id for root in changed.values() for region in _walk_regions(root)}
    metadata = {key: value for key, value in before.shape_metadata.items() if key in live - selected}
    metadata.update({key: value for key, value in outcome.shape_metadata.items() if key in selected})
    observations = {key: value for key, value in before.region_evidence.items() if key in live - selected}
    for identity, observation in outcome.region_evidence.items():
        if identity not in selected:
            continue
        info = dict(observation)
        for field in ('raster', 'coverage'):
            source = info.get(f'{field}_source')
            if source is not None:
                prior_info = before.region_evidence.get(source)
                if prior_info is None or field not in prior_info:
                    raise ValueError(f'borrowed source coverage is missing: {source!r}')
                info[field] = prior_info[field]
                info[f'{field}_source'] = None
        observations[identity] = info
    return ReplayState(replace(before.drawing, regions=roots),
        _snapshot_canonical(metadata), _snapshot_canonical(observations))


def retain_selected_outcome(
    operation: RecipeOperation, before: ReplayState, after: ReplayState, *,
    evidence: GeometryEvidence, paint: PaintEvidence, upstream: EffectiveDrawingRecipe,
) -> tuple[RecipeOperation, ReplayState]:
    prior = {root.id: _digest(_root_inputs(before, root)) for root in before.drawing.regions}
    roots = tuple(root for root in after.drawing.regions
        if prior.get(root.id) != _digest(_root_inputs(after, root)))
    selected_ids = {region.id for root in roots for region in _walk_regions(root)}
    outcome = SelectedOperationOutcome(
        dependency_revision=operation_outcome_dependencies(operation, evidence=evidence,
            paint=paint, upstream=upstream, input_revision=selected_input_revision(before)),
        regions=_snapshot_canonical(roots), root_order=tuple(root.id for root in after.drawing.regions),
        shape_metadata=_snapshot_canonical({key: value for key, value in after.shape_metadata.items() if key in selected_ids}),
        region_evidence=_snapshot_canonical(_borrow_observations({key: value for key, value in after.region_evidence.items()
            if key in selected_ids}, before)),
    )
    return replace(operation, selected_outcome=outcome), apply_selected_outcome(outcome, before)

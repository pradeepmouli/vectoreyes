"""Canonical replay of retained automatic choices, with explicit numeric boundaries."""

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .drawing import DrawingRegion, _walk_regions
from .drawing_relationships import replace_region
from .drawing_topology import normalize_drawing_topology
from .replay_state import ReplayState, publish_drawing_state
from .selected_outcomes import region_ancestries
from .optimizer.automatic_replay import (
    AutomaticIntent, CornerDiagnosticsObservation, FitPrimitiveOperation,
    ReplayBatch, SmoothPathOperation, SimplifyLineletOperation,
    NormalizeCornersOperation, DeclareIntrinsicSymmetryOperation, AssignCloneOperation,
    MirrorRegionOperation, ReconstructOcclusionOperation,
)
from .replay_inputs import ReplayInputResolver, assign_replacement_masks, clone_report
from .path_geometry import PathGeometry
from .path_topology import component_paths, replace_component_paths
from .immutable_value import freeze_value


@dataclass(frozen=True)
class SelectedAutomaticPaths:
    paths: tuple[PathGeometry, ...]
    preserve_subpaths: bool = True
    path_text: str | None = None
    diagnostics: Mapping[str, object] | None = None


def select_automatic_paths(
    action: SmoothPathOperation | SimplifyLineletOperation | NormalizeCornersOperation,
    paths: tuple[PathGeometry, ...], *, source_text: str,
    original: str | None = None, fill_rule: str | None = None,
) -> SelectedAutomaticPaths | None:
    """Apply recorded settings once, independent of working-record publication."""
    if isinstance(action, SmoothPathOperation):
        from .drawing_smoothing import smooth_paths
        selected = smooth_paths(paths, max_boundary_error=action.max_boundary_error)
        return None if selected is None else SelectedAutomaticPaths(tuple(selected))
    if isinstance(action, SimplifyLineletOperation):
        from .drawing_simplification import simplify_paths
        selected = simplify_paths(paths, epsilon=action.epsilon, max_error=action.max_error,
            samples=action.samples, preserve_subpaths=action.preserve_subpaths, cubic=action.cubic,
            original=original, fill_rule=fill_rule, check_geometry=action.check_geometry,
            linelet_only=action.linelet_only, max_boundary_error=action.max_boundary_error)
        return None if selected is None else SelectedAutomaticPaths(tuple(selected), action.preserve_subpaths)
    from .corner_normalization import normalize_declared_paths
    selected, report = normalize_declared_paths(paths, declaration=action.declaration,
        max_error=action.max_error, source_text=source_text)
    return SelectedAutomaticPaths(tuple(selected), path_text=str(report['path']), diagnostics={'corners': report})


@dataclass(frozen=True)
class AutomaticRegionInput:
    region: DrawingRegion
    observation: Mapping[str, object]
    metadata: Mapping[str, object]
    children: tuple['AutomaticRegionInput', ...]


def _capture_inputs(regions, observations, metadata) -> tuple[AutomaticRegionInput, ...]:
    return tuple(AutomaticRegionInput(region, freeze_value(observations[region.id]),
        freeze_value(metadata.get(region.id, {})), _capture_inputs(region.children, observations, metadata))
        for region in regions)


def _index_inputs(regions: tuple[AutomaticRegionInput, ...], parent: tuple[int, ...] = ()):
    result = {}
    for region in regions:
        path = (*parent, int(region.observation['id']))
        result[path] = region
        result.update(_index_inputs(region.children, path))
    return result


def _restore_input(selected: AutomaticRegionInput, observations, metadata) -> DrawingRegion:
    observations[selected.region.id] = dict(selected.observation)
    if selected.region.is_leaf:
        metadata[selected.region.id] = dict(selected.metadata)
    for child in selected.children:
        _restore_input(child, observations, metadata)
    return selected.region


def _direct_batch_group(state: ReplayState, batches: tuple[ReplayBatch, ...]) -> bool:
    for batch in batches:
        for record in batch.entries:
            intent = record.intent
            if not isinstance(intent, CornerDiagnosticsObservation) and not (
                    isinstance(intent, AutomaticIntent) and isinstance(intent.action,
                        (FitPrimitiveOperation, SmoothPathOperation, SimplifyLineletOperation,
                         NormalizeCornersOperation, DeclareIntrinsicSymmetryOperation,
                         AssignCloneOperation, MirrorRegionOperation, ReconstructOcclusionOperation))):
                return False
    return True


def _selected_shape(state, region, metadata):
    from .drawing_adapter import current_shape_for_region
    indexed = {item.id: item for root in state.drawing.regions for item in _walk_regions(root)}
    return current_shape_for_region(region, indexed, region_ancestries(state), metadata.get(region.id, {}))


def _select_relationship(state, action, resolver, observations, metadata):
    from .drawing_actions import AssignCloneAction
    from .drawing_relationships import assign_clone
    from .drawing_adapter import _geometry, resolved_instance_shape, transform_working_shape, working_fill_from_spec
    from .drawing_fill import serialize_fill
    from .geometry_native import region_path
    from .path_geometry import CloneGeometry
    from .fit import Shape
    from .replay_inputs import mirror_report

    source_input, target_input = resolver.resolve(action.source), resolver.resolve(action.target)
    source, target = source_input.region, _restore_input(target_input, observations, metadata)
    snapshot = state.drawing
    available = {region.id for root in snapshot.regions for region in _walk_regions(root)}
    for region in (source, target):
        snapshot = (replace_region(snapshot, region) if region.id in available
            else replace(snapshot, regions=(*snapshot.regions, region)))
        available.add(region.id)
    assigned = assign_clone(snapshot, AssignCloneAction(op='assign_clone', source=source.id,
        target=target.id, transforms=(dict(zip(('a', 'b', 'c', 'd', 'tx', 'ty'), action.transform)),)))
    symbolic = isinstance(action, AssignCloneOperation) and action.symbolic
    projected_diagnostics = {}
    metadata[target.id] = {}
    if symbolic:
        geometry = assigned.geometry
    else:
        if isinstance(source.geometry, CloneGeometry):
            by_id = {region.id: region for root in snapshot.regions for region in _walk_regions(root)}
            source_shape = resolved_instance_shape(Shape('use', {}), region_path(source, by_id))
        else:
            source_shape = _selected_shape(state, source, {source.id: source_input.metadata})
        shape = transform_working_shape(source_shape, action.transform)
        geometry = _geometry(shape.kind, shape.params, target.id, {})
        if shape.kind == 'path' and 'canonical_path_geometry' not in shape.params:
            from .corner_provenance import ingress_prefit_corners
            geometry, projected_diagnostics['corners'] = ingress_prefit_corners(
                geometry, str(shape.params['d']), target.id, target.diagnostics.get('corners'),
                path_ids=shape.params.get('path_ids'))
        core = {'d', 'canonical_path_geometry'} if shape.kind == 'path' else set(shape.params)
        metadata[target.id] = {key: value for key, value in shape.params.items() if key not in core}
    if isinstance(action, AssignCloneOperation):
        fill_name = 'NoneType' if target.fill is None else type(working_fill_from_spec(serialize_fill(target.fill))).__name__
        diagnostics = {'clones': clone_report(int(source_input.observation['id']), fill_name,
            action.transform, action.family_size)}
    else:
        diagnostics = {'symmetry': mirror_report(int(source_input.observation['id']), action.axis)}
    selected = replace(target, geometry=geometry,
        diagnostics={**target.diagnostics, **diagnostics, **projected_diagnostics})
    return ((selected,) if isinstance(action, AssignCloneOperation)
        else (_restore_input(source_input, observations, metadata), selected))


def _select_occlusion(action, resolver, observations, metadata):
    from .drawing_occlusion import OcclusionSource, reconstruct_occlusion, validate_occlusion_children
    from .drawing_adapter import generated_tree_observations
    from .drawing_fill import FlatFill

    validate_occlusion_children(action)
    references = tuple(dict.fromkeys((*action.component, *(child.source for child in action.children))))
    sources = {}
    for ref in references:
        bound = resolver.resolve(ref)
        region, info = bound.region, bound.observation
        color = region.fill.color if isinstance(region.fill, FlatFill) else info['color_hex']
        if color is None:
            raise ValueError('occlusion replay source has no flat paint')
        raster = np.frombuffer(info['raster'], dtype=bool).reshape(info['raster_shape'])
        sources[ref] = OcclusionSource(int(info['id']), region.id, color, raster)
    branch_ref = next(ref for ref in action.component if sources[ref].id == action.branch_id)
    selected = reconstruct_occlusion(action, sources, region_id=sources[branch_ref].region_id)
    identities = {selected.id: action.branch_id}
    for region in _walk_regions(selected):
        if region.id != selected.id:
            identities[region.id] = int(region.id.rsplit('-', 1)[1])
    rasters = [sources[ref].raster for ref in action.component if sources[ref].raster.size]
    raster = np.logical_or.reduce(rasters) if rasters else np.zeros(action.shape_hw, bool)
    generated, generated_metadata = generated_tree_observations(selected, identities=identities,
        raster_shape=action.shape_hw, retained_rasters={selected.id: raster},
        parent_ancestry=tuple(resolver.resolve(branch_ref).observation['ancestry'])[:-1])
    observations.update(generated)
    metadata.update(generated_metadata)
    return (selected,)


def _select_regions(state, intent, resolver, observations, metadata):
    if isinstance(intent, AutomaticIntent):
        action = intent.action
        if isinstance(action, (AssignCloneOperation, MirrorRegionOperation)):
            return _select_relationship(state, action, resolver, observations, metadata)
        if isinstance(action, ReconstructOcclusionOperation):
            return _select_occlusion(action, resolver, observations, metadata)
    selected = _restore_input(resolver.resolve(
        intent.target if isinstance(intent, CornerDiagnosticsObservation) else intent.action.target), observations, metadata)
    if isinstance(intent, CornerDiagnosticsObservation):
        return (replace(selected, diagnostics={**selected.diagnostics, 'corners': intent.diagnostics}),)
    if isinstance(action, FitPrimitiveOperation):
        metadata[selected.id] = {}
        return (replace(selected, geometry=action.primitive),)
    if isinstance(action, DeclareIntrinsicSymmetryOperation):
        from .drawing_symmetry import Axis2D, apply_intrinsic_symmetry
        return (apply_intrinsic_symmetry(selected, Axis2D(*action.axis), action.residual),)
    paths = {path.path.id: path for path in component_paths(selected.geometry)}
    path_ids = metadata.get(selected.id, {}).get('path_ids', tuple(paths))
    ordered = tuple(paths[identity] for identity in path_ids)
    info = observations[selected.id]
    choice = select_automatic_paths(action, ordered, source_text=' '.join(path.to_svg_d() for path in ordered),
        original=str(info['original_params']['d']) if info['original_kind'] == 'path' else None,
        fill_rule=metadata.get(selected.id, {}).get('fill_rule'))
    if choice is None:
        name = 'smooth' if isinstance(action, SmoothPathOperation) else 'simplify'
        raise ValueError(f'recorded {name} operation no longer applies to its bound input')
    by_id = {path.path.id: path for path in choice.paths}
    geometry = replace_component_paths(selected.geometry,
        tuple(by_id[path.path.id] for path in component_paths(selected.geometry))) \
        if choice.preserve_subpaths else choice.paths[0]
    selected = replace(selected, geometry=geometry, diagnostics={**selected.diagnostics, **(choice.diagnostics or {})})
    metadata[selected.id]['path_ids'] = tuple(path.path.id for path in choice.paths)
    if not choice.preserve_subpaths:
        metadata[selected.id].pop('fill_rule', None)
    covering = tuple(_restore_input(resolver.resolve(ref), observations, metadata)
        for ref in action.covering) if isinstance(action, SimplifyLineletOperation) else ()
    return (selected, *covering)


def replay_automatic_batches(state: ReplayState, batches: tuple[ReplayBatch, ...], *,
                            finalize_topology: bool = True) -> ReplayState:
    """Replay eligible choices canonically while retaining exact numeric receipts."""
    if not batches:
        return state
    from .optimizer.replay_binding import replay_numeric_batches
    from .replay_identity import bind_automatic_inputs, scoped_input_regions
    if not _direct_batch_group(state, batches):
        return replay_numeric_batches(state, batches, finalize_topology=finalize_topology)
    from .optimizer.framework import accepted_proposal_diagnostics

    for batch in batches:
        masks = bind_automatic_inputs(state, batch)
        indexed = {path: state.drawing.region(identity) for identity, path in region_ancestries(state).items()}
        observations = {identity: dict(info) for identity, info in state.region_evidence.items()}
        metadata = {identity: dict(info) for identity, info in state.shape_metadata.items()}
        scoped = scoped_input_regions(state, batch.scope)
        current = list(scoped)
        inputs = _index_inputs(_capture_inputs(scoped, observations, metadata))
        original_ids = {int(observations[region.id]['id']) for region in scoped}
        current_masks = {identity: mask.copy() for identity, mask in masks.items()}
        resolver = ReplayInputResolver(inputs, batch, lambda region: _index_inputs((region,)))
        for record in batch.entries:
            intent = record.intent
            replacements = _select_regions(state, intent, resolver, observations, metadata)
            by_numeric = {int(observations[region.id]['id']): region for region in replacements}
            replacement_masks = assign_replacement_masks(tuple(by_numeric), record.proposal_ids,
                original_ids=original_ids, pass_masks=masks, current_masks=current_masks,
                raster_for=lambda identity, shape: _replacement_raster(state, by_numeric[identity], metadata, shape))
            if isinstance(intent, AutomaticIntent):
                outputs = _index_inputs(_capture_inputs(replacements, observations, metadata))
                for output in intent.outputs:
                    if output.ancestry not in outputs:
                        raise ValueError(f'automatic replay named output is absent: {output.name!r}')
                    resolver.register(intent.operation_id, output, outputs[output.ancestry])
            if record.assessment is None:
                raise ValueError('committed automatic operation requires an assessment')
            current = [region for region in current if int(observations[region.id]['id']) not in record.proposal_ids]
            for region in replacements:
                current.append(replace(region, diagnostics={**region.diagnostics,
                    **accepted_proposal_diagnostics(record.pass_name, record.proposal_ids, record.assessment)}))
                mask = replacement_masks[int(observations[region.id]['id'])]
                observations[region.id].update(raster=mask.tobytes(), raster_shape=mask.shape, raster_source=None)
            current.sort(key=lambda region: (float(region.z), int(observations[region.id]['id'])))
        drawing = state.drawing
        if batch.scope:
            parent = indexed[batch.scope]
            drawing = replace_region(drawing, replace(parent, children=tuple(current)))
        else:
            drawing = replace(drawing, regions=tuple(current))
        state = publish_drawing_state(replace(state, region_evidence=observations), drawing, shape_metadata=metadata)
        # The transaction selects metadata too: an explicit empty projection is
        # distinct from the defaults used when publishing a new public path.
        state = replace(state, shape_metadata={identity: metadata.get(identity, info)
            for identity, info in state.shape_metadata.items()})
    return publish_drawing_state(state, normalize_drawing_topology(state.drawing)) if finalize_topology else state


def _replacement_raster(state, region, metadata, shape):
    from .optimizer.gate import rasterize
    from .optimizer.vector_region import to_polygon
    return rasterize(to_polygon(_selected_shape(state, region, metadata)), shape)

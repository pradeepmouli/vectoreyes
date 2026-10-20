"""Canonical replay of retained automatic choices, with explicit numeric boundaries."""

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .drawing import DrawingRegion, _walk_regions
from .drawing_relationships import replace_region
from .drawing_topology import normalize_drawing_topology
from .replay_state import ReplayState, publish_drawing_state
from .selected_outcomes import region_ancestries
from .refinement.automatic_replay import (
    AutomaticIntent, CornerDiagnosticsObservation, FitPrimitiveOperation,
    ReplayBatch, SmoothPathOperation, SimplifyLineletOperation,
    NormalizeCornersOperation, DeclareIntrinsicSymmetryOperation, AssignCloneOperation,
    MirrorRegionOperation, ReconstructOcclusionOperation,
    ApplySeamConstraintsOperation, RefreshDependentClonesOperation,
    ReconstructSelfSymmetryOperation, NormalizeRetainedTopologyOperation, PassInputRegionRef,
    AutomaticRecord, ProposalIntent,
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


def _selected_shape(state, region, metadata):
    from .drawing_adapter import current_shape_for_region
    indexed = {item.id: item for root in state.drawing.regions for item in _walk_regions(root)}
    return current_shape_for_region(region, indexed, region_ancestries(state), metadata.get(region.id, {}))


def _publish_shape(region, shape, metadata):
    """Publish a numeric kernel result without creating a disposable region."""
    from .drawing_adapter import _geometry
    geometry = _geometry(shape.kind, shape.params, region.id, {})
    diagnostics = {}
    if shape.kind == 'path' and 'canonical_path_geometry' not in shape.params:
        from .corner_provenance import ingress_prefit_corners
        geometry, diagnostics['corners'] = ingress_prefit_corners(
            geometry, str(shape.params['d']), region.id, region.diagnostics.get('corners'),
            path_ids=shape.params.get('path_ids'))
    core = {'d', 'canonical_path_geometry'} if shape.kind == 'path' else set(shape.params)
    metadata[region.id] = {key: value for key, value in shape.params.items() if key not in core}
    return replace(region, geometry=geometry, diagnostics={**region.diagnostics, **diagnostics})


def _replace_bound_children(region, replacements):
    if region.id in replacements:
        return replacements[region.id]
    return replace(region, children=tuple(_replace_bound_children(child, replacements)
        for child in region.children)) if region.children else region


def _select_seams(state, action, resolver, observations, metadata):
    from .drawing_seams import SeamInput, execute_seam_constraints
    from .geometry_native import region_path
    from .path_geometry import CloneGeometry
    from .drawing_adapter import transform_working_shape

    inputs = {}
    for pair in action.pairs:
        for ref in (pair.left, pair.right):
            key = tuple(ref.ancestry)
            if key in inputs:
                continue
            bound = resolver.resolve(ref)
            owner = resolver.resolve(PassInputRegionRef(ref.batch_id, ref.input_revision, (key[0],)))
            shape = _selected_shape(state, bound.region, {bound.region.id: bound.metadata})
            if isinstance(bound.region.geometry, CloneGeometry):
                source_id = bound.region.geometry.source_region_id
                sources = [item for item in _walk_regions(owner.region) if item.id == source_id and item.is_leaf]
                if len(sources) != 1:
                    raise ValueError('seam replay clone source is absent or ambiguous')
                source_bound = next(item for item in _index_inputs((owner,)).values() if item.region.id == source_id)
                shape = transform_working_shape(_selected_shape(state, sources[0],
                    {source_id: source_bound.metadata}), shape.params['transform'])
            by_id = {item.id: item for item in _walk_regions(owner.region)}
            inputs[key] = SeamInput(shape, int(bound.observation['id']), int(owner.observation['id']),
                (float(owner.region.z), float(bound.region.z), int(owner.observation['id']),
                 int(bound.observation['id'])), region_path(owner.region, by_id))
    roots = tuple(_restore_input(resolver.resolve(ref), observations, metadata) for ref in action.roots)
    replacements = {}
    for key, update in execute_seam_constraints(action, inputs).items():
        ref = PassInputRegionRef(action.roots[0].batch_id, action.roots[0].input_revision, key)
        region = resolver.resolve(ref).region
        selected = _publish_shape(region, update.shape, metadata)
        candidate = update.candidate
        replacements[region.id] = replace(selected, diagnostics={**selected.diagnostics, 'seams': {
            'accepted': True, 'paired_with': int(update.paired_with), 'selected': update.selected,
            'axis': candidate.axis, 'gap_before': abs(candidate.a_value - candidate.b_value),
            'seam': candidate.seam}})
    return tuple(_replace_bound_children(root, replacements) for root in roots)


def _select_refresh(state, action, resolver, observations, metadata):
    from .path_geometry import CloneGeometry
    from .replay_inputs import clone_reference_diagnostics
    roots = tuple(_restore_input(resolver.resolve(ref), observations, metadata) for ref in action.roots)
    replacements = {}
    for refresh in action.clones:
        target = resolver.resolve(refresh.clone).region
        source = resolver.resolve(refresh.source).region
        if not isinstance(target.geometry, CloneGeometry):
            raise ValueError('dependent clone refresh requires clone geometry')
        diagnostics = clone_reference_diagnostics(target.diagnostics,
            source_id=int(resolver.resolve(refresh.source).observation['id']),
            source_path=tuple(refresh.source.ancestry), previous_source_id=refresh.previous_source_id,
            previous_source_path=refresh.previous_source_path)
        replacements[target.id] = replace(target,
            geometry=replace(target.geometry, source_region_id=source.id),
            diagnostics={**target.diagnostics, **diagnostics})
    return tuple(_replace_bound_children(root, replacements) for root in roots)


def _select_self_symmetry(state, action, resolver, observations, metadata):
    from .drawing_symmetry import Axis2D, reconstruct_self_symmetry
    from .geometry_native import region_path
    from .drawing_adapter import generated_tree_observations
    from .refinement.region_sample import to_polygon
    from .fit import Shape

    bound = resolver.resolve(action.target)
    target = _restore_input(bound, observations, metadata)
    by_id = {item.id: item for root in state.drawing.regions for item in _walk_regions(root)}
    footprint = region_path(target, by_id)
    info = bound.observation
    if target.is_leaf and isinstance(target.diagnostics.get('geometry_seed'), Mapping) and info['original_kind'] is not None:
        seeded = to_polygon(Shape(info['original_kind'], dict(info['original_params'])))
        if not seeded.is_empty:
            footprint = seeded
    selected = reconstruct_self_symmetry(target, footprint, axis=Axis2D(*action.axis),
        residual=action.residual, fit_kind=action.fit_kind, side=action.side,
        corner_radius=action.corner_radius, epsilon=action.epsilon,
        max_error=action.max_error, cubic=action.cubic)
    if selected.is_branch:
        identities = {selected.id: int(info['id'])}
        def identify(region):
            for index, child in enumerate(region.children, 1):
                identities[child.id] = index
                identify(child)
        identify(selected)
        raster = np.frombuffer(info['raster'], dtype=bool).reshape(info['raster_shape'])
        generated, generated_metadata = generated_tree_observations(selected, identities=identities,
            raster_shape=raster.shape, retained_rasters={selected.id: raster},
            parent_ancestry=tuple(info['ancestry'])[:-1])
        for key in ('source_label', 'color_hex', 'coverage', 'coverage_source', 'coverage_shape',
                    'coverage_dtype', 'coverage_origin', 'coverage_canvas'):
            generated[selected.id][key] = info[key]
        observations.update(generated)
        metadata.update(generated_metadata)
    else:
        leaves = tuple(item for item in _index_inputs((bound,)).values() if item.region.is_leaf)
        consumed = tuple(dict.fromkeys(int(item.observation['id']) for item in leaves))
        selected = replace(selected, fill=leaves[0].region.fill, diagnostics={**selected.diagnostics,
            'materialization': {**selected.diagnostics.get('materialization', {}),
                'consumed_region_ids': consumed,
                'retired_region_ids': tuple(identity for identity in consumed if identity != int(info['id'])),
                'source_region_remap': tuple((identity, int(info['id'])) for identity in consumed)}},
            source_region_ids=tuple(sorted(set(target.source_region_ids).union(
                *(leaf.region.source_region_ids for leaf in leaves)))))
        observations[selected.id].update(original_kind=leaves[0].observation['original_kind'],
            original_params=dict(leaves[0].observation['original_params']))
        metadata[selected.id] = {'fill_rule': 'evenodd'}
    if action.dependent_clones:
        from .replay_inputs import clone_reference_diagnostics
        from .path_geometry import CloneGeometry
        dependent = []
        def refresh(clone):
            if clone.children:
                return replace(clone, children=tuple(refresh(child) for child in clone.children))
            if not isinstance(clone.geometry, CloneGeometry) or clone.geometry.source_region_id != target.id:
                return clone
            diagnostics = clone_reference_diagnostics(clone.diagnostics,
                source_id=int(info['id']), source_path=tuple(action.target.ancestry),
                previous_source_id=int(info['id']), previous_source_path=tuple(action.target.ancestry))
            return replace(clone, diagnostics={**clone.diagnostics, **diagnostics})
        for ref in action.dependent_clones:
            clone = _restore_input(resolver.resolve(ref), observations, metadata)
            dependent.append(refresh(clone))
        return (selected, *dependent)
    return (selected,)


def _topology_shape(region, metadata):
    from .drawing_adapter import shape_from_region_geometry
    from .fit import Shape
    shape = shape_from_region_geometry(region, metadata)
    if shape.kind != 'path':
        return shape
    paths = {path.path.id: path for path in component_paths(region.geometry)}
    order = metadata.get('path_ids', tuple(paths))
    return Shape('path', {**shape.params, 'd': ' '.join(paths[identity].to_svg_d() for identity in order)})


def _topology_records(regions, observations, metadata):
    from .drawing_seams import seam_path_revision
    from .path_geometry import CloneGeometry
    result = {}
    for root in regions:
        for region in root.leaves():
            if isinstance(region.geometry, CloneGeometry):
                continue
            shape = _topology_shape(region, metadata.get(region.id, {}))
            if shape.kind != 'path':
                continue
            paths = {path.path.id: path for path in component_paths(region.geometry)}
            order = metadata.get(region.id, {}).get('path_ids', tuple(paths))
            for index, identity in enumerate(order):
                result.setdefault(identity, []).append((tuple(observations[region.id]['ancestry']),
                    index, seam_path_revision(str(shape.params['d']), index), len(paths[identity].path.commands)))
    return result


def _select_topology(state, action, resolver, observations, metadata):
    from .geometry_native import region_path
    from .drawing_topology import match_topology_paths, _component_corners, retired_corner_features
    from .refinement.gate import rasterize
    from .refinement.region_sample import to_polygon
    from .fit import Shape
    from .path_geometry import CloneGeometry

    inputs = tuple(resolver.resolve(ref) for ref in action.roots)
    roots = tuple(_restore_input(bound, observations, metadata) for bound in inputs)
    before = _topology_records(roots, observations, metadata)
    selected = normalize_drawing_topology(replace(state.drawing, regions=roots),
        allow_external=action.allow_external)
    bound_by_id = {bound.region.id: bound for bound in _index_inputs(inputs).values()}
    selected_by_id = {item.id: item for root in selected.regions for item in _walk_regions(root)}

    def publish(region, prior=None):
        bound = bound_by_id.get(region.id, prior)
        if bound is None:
            raise ValueError('retained topology output has no source observation')
        source = bound.region
        if source.is_branch:
            return replace(region, children=tuple(publish(child) for child in region.children))
        if isinstance(source.geometry, CloneGeometry):
            if region.is_branch:
                ancestry = tuple(bound.observation['ancestry'])
                observations[region.id] = dict(bound.observation)
                children = []
                for child in region.children:
                    info = dict(bound.observation)
                    raster = rasterize(region_path(child, selected_by_id), info['raster_shape'])
                    info.update(id=int(child.id.rsplit('-', 1)[1]), ancestry=(*ancestry, int(child.id.rsplit('-', 1)[1])),
                        raster=raster.tobytes(), raster_source=None)
                    observations[child.id] = info
                    metadata[child.id] = dict(bound.metadata)
                    children.append(child)
                return replace(region, children=tuple(children), diagnostics={**region.diagnostics,
                    'topology': {**region.diagnostics['topology'], 'source_region_path':
                        tuple(bound_by_id[source.geometry.source_region_id].observation['ancestry'])}})
            return region
        paths = component_paths(source.geometry)
        identities = tuple(bound.metadata.get('path_ids', tuple(path.path.id for path in paths)))
        retained = {path.path.id for leaf in region.leaves() for path in component_paths(leaf.geometry)}
        if not region.is_branch and set(identities) == retained:
            return region
        source_d = _topology_shape(source, bound.metadata).params['d']
        children = []
        for leaf in region.leaves():
            members = {path.path.id for path in component_paths(leaf.geometry)}
            indices = tuple(i for i, identity in enumerate(identities) if identity in members)
            output_ids = tuple(identities[i] for i in indices)
            local = int(leaf.id.rsplit('-', 1)[1]) if region.is_branch else int(bound.observation['id'])
            output_metadata = {**bound.metadata, 'path_ids': output_ids,
                'fill_rule': 'evenodd' if len(indices) > 1 else 'nonzero'}
            output_shape = _topology_shape(leaf, output_metadata)
            info = dict(bound.observation)
            ancestry = (*tuple(info['ancestry']), local) if region.is_branch else tuple(info['ancestry'])
            raster = rasterize(to_polygon(Shape('path', dict(output_shape.params))), info['raster_shape'])
            info.update(id=local, ancestry=ancestry, raster=raster.tobytes(), raster_source=None)
            observations[leaf.id] = info
            metadata[leaf.id] = output_metadata
            children.append(replace(leaf, diagnostics={**source.diagnostics,
                'corners': _component_corners(source.diagnostics.get('corners'), source_d,
                    output_shape.params['d'], indices), 'topology': leaf.diagnostics['topology']}))
        receipt = {**region.diagnostics['topology'], 'retired_feature_ids': retired_corner_features(
            source_d, source.diagnostics.get('corners', {}), identities,
            region.diagnostics['topology']['retired_path_ids'])}
        if region.is_branch:
            return replace(region, children=tuple(children), diagnostics={**region.diagnostics, 'topology': receipt})
        return replace(children[0], diagnostics={**children[0].diagnostics, 'topology': receipt})

    roots = tuple(publish(region) for region in selected.regions)
    recorded = tuple(replace(item, target=replace(item.target, batch_id='', input_revision=''))
        for item in action.path_correspondence)
    if match_topology_paths(before, _topology_records(roots, observations, metadata)) != recorded:
        raise ValueError('retained topology path correspondence is stale')
    return tuple(region for region in roots if int(observations[region.id]['id']) in action.affected_ids)


def _select_relationship(state, action, resolver, observations, metadata):
    from .drawing_actions import AssignCloneAction
    from .drawing_relationships import assign_clone
    from .drawing_adapter import resolved_instance_shape, transform_working_shape, working_fill_from_spec
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
        projected = _publish_shape(target, shape, metadata)
        geometry = projected.geometry
        projected_diagnostics = dict(projected.diagnostics)
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


class AutomaticChoiceNotApplicable(ValueError):
    """A well-bound optional action produced no candidate for this input."""


def _select_regions(state, intent, resolver, observations, metadata):
    if isinstance(intent, AutomaticIntent):
        action = intent.action
        if isinstance(action, (AssignCloneOperation, MirrorRegionOperation)):
            return _select_relationship(state, action, resolver, observations, metadata)
        if isinstance(action, ReconstructOcclusionOperation):
            return _select_occlusion(action, resolver, observations, metadata)
        if isinstance(action, ApplySeamConstraintsOperation):
            return _select_seams(state, action, resolver, observations, metadata)
        if isinstance(action, RefreshDependentClonesOperation):
            return _select_refresh(state, action, resolver, observations, metadata)
        if isinstance(action, ReconstructSelfSymmetryOperation):
            return _select_self_symmetry(state, action, resolver, observations, metadata)
        if isinstance(action, NormalizeRetainedTopologyOperation):
            return _select_topology(state, action, resolver, observations, metadata)
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
        raise AutomaticChoiceNotApplicable(f'recorded {name} operation no longer applies to its bound input')
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


@dataclass(frozen=True)
class PreparedAutomaticSelection:
    input_revision: str
    intent: ProposalIntent
    regions: tuple[AutomaticRegionInput, ...]


class AutomaticBatchApplication:
    """One canonical publication boundary shared by discovery and replay.

    Preparation reads pinned pass inputs and returns detached selected roots.
    Rejected choices do not leak annotations or source observations into the
    current drawing. Acceptance installs the already prepared result exactly.
    """

    def __init__(self, state: ReplayState, batch: ReplayBatch) -> None:
        from .replay_identity import bind_automatic_inputs, scoped_input_regions
        self.state = state
        self.batch = batch
        self.masks = bind_automatic_inputs(state, batch)
        self.indexed = {path: state.drawing.region(identity)
            for identity, path in region_ancestries(state).items()}
        self.observations = {identity: dict(info) for identity, info in state.region_evidence.items()}
        self.metadata = {identity: dict(info) for identity, info in state.shape_metadata.items()}
        scoped = scoped_input_regions(state, batch.scope)
        self.current = list(scoped)
        inputs = _index_inputs(_capture_inputs(scoped, self.observations, self.metadata))
        self.original_ids = {int(self.observations[region.id]['id']) for region in scoped}
        self.current_masks = {identity: mask.copy() for identity, mask in self.masks.items()}
        self.resolver = ReplayInputResolver(inputs, batch, lambda region: _index_inputs((region,)))

    def prepare(self, intent: ProposalIntent) -> PreparedAutomaticSelection:
        observations = {identity: dict(info) for identity, info in self.observations.items()}
        metadata = {identity: dict(info) for identity, info in self.metadata.items()}
        regions = _select_regions(self.state, intent, self.resolver, observations, metadata)
        return PreparedAutomaticSelection(self.batch.input_revision, intent,
            _capture_inputs(regions, observations, metadata))

    def accept(self, record: AutomaticRecord, selected: PreparedAutomaticSelection) -> None:
        from .refinement.framework import accepted_proposal_diagnostics
        if selected.input_revision != self.batch.input_revision or selected.intent != record.intent:
            raise ValueError('prepared selection binding does not match the committed intent')
        if record.assessment is None or not record.assessment.accepted:
            raise ValueError('committed automatic operation requires an accepted assessment')
        observations = {identity: dict(info) for identity, info in self.observations.items()}
        metadata = {identity: dict(info) for identity, info in self.metadata.items()}
        masks = {identity: mask.copy() for identity, mask in self.current_masks.items()}
        replacements = tuple(_restore_input(item, observations, metadata) for item in selected.regions)
        by_numeric = {int(observations[region.id]['id']): region for region in replacements}
        replacement_masks = assign_replacement_masks(tuple(by_numeric), record.proposal_ids,
            original_ids=self.original_ids, pass_masks=self.masks, current_masks=masks,
            raster_for=lambda identity, shape: _replacement_raster(self.state, by_numeric[identity], metadata, shape))
        intent = record.intent
        outputs = _index_inputs(selected.regions)
        if isinstance(intent, AutomaticIntent):
            if len({output.name for output in intent.outputs}) != len(intent.outputs):
                raise ValueError('duplicate automatic replay named output')
            for output in intent.outputs:
                if output.ancestry not in outputs:
                    raise ValueError(f'automatic replay named output is absent: {output.name!r}')
        current = [region for region in self.current
            if int(observations[region.id]['id']) not in record.proposal_ids]
        for region in replacements:
            current.append(replace(region, diagnostics={**region.diagnostics,
                **accepted_proposal_diagnostics(record.pass_name, record.proposal_ids, record.assessment)}))
            mask = replacement_masks[int(observations[region.id]['id'])]
            observations[region.id].update(raster=mask.tobytes(), raster_shape=mask.shape, raster_source=None)
        current.sort(key=lambda region: (float(region.z), int(observations[region.id]['id'])))
        if isinstance(intent, AutomaticIntent):
            self.resolver.register_many(intent.operation_id,
                tuple((output, outputs[output.ancestry]) for output in intent.outputs))
        self.current, self.observations, self.metadata, self.current_masks = current, observations, metadata, masks

    def finish(self) -> ReplayState:
        drawing = self.state.drawing
        if self.batch.scope:
            parent = self.indexed[self.batch.scope]
            drawing = replace_region(drawing, replace(parent, children=tuple(self.current)))
        else:
            drawing = replace(drawing, regions=tuple(self.current))
        state = publish_drawing_state(replace(self.state, region_evidence=self.observations),
            drawing, shape_metadata=self.metadata)
        # Explicit empty metadata is a selected projection, not missing defaults.
        return replace(state, shape_metadata={identity: self.metadata.get(identity, info)
            for identity, info in state.shape_metadata.items()})


def replay_automatic_batches(state: ReplayState, batches: tuple[ReplayBatch, ...], *,
                            finalize_topology: bool = True) -> ReplayState:
    """Replay recorded choices through the same canonical publication boundary."""
    if not batches:
        return state
    for batch in batches:
        application = AutomaticBatchApplication(state, batch)
        for record in batch.entries:
            application.accept(record, application.prepare(record.intent))
        state = application.finish()
    return publish_drawing_state(state, normalize_drawing_topology(state.drawing)) if finalize_topology else state


def _replacement_raster(state, region, metadata, shape):
    from .refinement.gate import rasterize
    from .refinement.region_sample import to_polygon
    return rasterize(to_polygon(_selected_shape(state, region, metadata)), shape)

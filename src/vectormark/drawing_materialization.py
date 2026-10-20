"""Canonical self-symmetry publication and exact clone-source remapping."""

from collections.abc import Mapping
from dataclasses import replace

from .drawing import DrawingRegion, _walk_regions
from .path_geometry import CloneGeometry
from .replay_state import ReplayState, publish_drawing_state


def publish_symmetry_selection(
    target: DrawingRegion, selected: DrawingRegion,
    observations: dict[str, dict[str, object]], metadata: dict[str, dict[str, object]], *,
    retain_paths: bool = False,
) -> DrawingRegion:
    """Publish a selected union with source observations and lineage, not a scene projection."""
    from .drawing_adapter import generated_tree_observations
    import numpy as np
    info = dict(observations[target.id])
    leaves = target.leaves()
    source_info = {leaf.id: dict(observations[leaf.id]) for leaf in leaves}
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
        first = source_info[leaves[0].id]
        observations[selected.id].update(original_kind=first['original_kind'],
            original_params=dict(first['original_params']))
        metadata[selected.id] = {'fill_rule': 'evenodd'}
    consumed = tuple(dict.fromkeys(int(source_info[leaf.id]['id']) for leaf in leaves))
    receipt = {**selected.diagnostics.get('materialization', {}),
            'consumed_region_ids': consumed,
            'retired_region_ids': tuple(identity for identity in consumed if identity != int(info['id'])),
            'source_region_remap': tuple((identity, int(info['id'])) for identity in consumed)}
    if retain_paths:
        receipt['source_region_path_remap'] = tuple((tuple(source_info[leaf.id]['ancestry']),
            tuple(info['ancestry'])) for leaf in leaves)
    return replace(selected, diagnostics={**selected.diagnostics, 'materialization': receipt})


def materialize_self_symmetry_state(state: ReplayState) -> ReplayState:
    """Consume declared vector halves and remap all exact dependent clone identities."""
    from .drawing_symmetry import union_symmetric_regions
    from .geometry_native import region_path
    from .replay_inputs import clone_reference_diagnostics
    from .drawing_topology import normalize_drawing_topology
    indexed = {region.id: region for root in state.drawing.regions for region in _walk_regions(root)}
    observations = {identity: dict(info) for identity, info in state.region_evidence.items()}
    metadata = {identity: dict(info) for identity, info in state.shape_metadata.items()}
    remaps = {}

    def visit(region: DrawingRegion) -> DrawingRegion:
        symmetry = region.diagnostics.get('symmetry')
        if region.children and isinstance(symmetry, Mapping) and symmetry.get('mode') == 'self' \
                and not symmetry.get('materialized'):
            selected = union_symmetric_regions(region,
                tuple(region_path(leaf, indexed) for leaf in region.leaves()), diagnostics=symmetry)
            for consumed in (region, *region.leaves()):
                remaps[consumed.id] = region.id
            return publish_symmetry_selection(region, selected, observations, metadata, retain_paths=True)
        if region.children:
            children = tuple(visit(child) for child in region.children)
            return replace(region, children=children) if children != region.children else region
        return region

    roots = tuple(visit(root) for root in state.drawing.regions)
    if not remaps:
        return state
    selected_by_id = {region.id: region for root in roots for region in _walk_regions(root)}
    def refresh(region: DrawingRegion) -> DrawingRegion:
        if region.children:
            return replace(region, children=tuple(refresh(child) for child in region.children))
        geometry = region.geometry
        if not isinstance(geometry, CloneGeometry) or geometry.source_region_id not in remaps:
            return region
        source_id = remaps[geometry.source_region_id]
        if not selected_by_id[source_id].is_leaf:
            raise ValueError('dependent clone source requires concrete materialized geometry')
        source = observations[source_id]
        previous = state.region_evidence[geometry.source_region_id]
        diagnostics = clone_reference_diagnostics(region.diagnostics,
            source_id=int(source['id']), source_path=tuple(source['ancestry']),
            previous_source_id=int(previous['id']), previous_source_path=tuple(previous['ancestry']))
        metadata[region.id] = {}
        return replace(region, geometry=replace(geometry, source_region_id=source_id),
            diagnostics={**region.diagnostics, **diagnostics})
    drawing = replace(state.drawing, regions=tuple(refresh(root) for root in roots))
    return publish_drawing_state(replace(state, region_evidence=observations),
        normalize_drawing_topology(drawing), shape_metadata=metadata)

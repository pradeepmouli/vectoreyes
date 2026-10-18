"""Canonical component publication shared by region actions and scene adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

from .drawing import DrawingRegion
from .path_geometry import CompoundPathGeometry, PathGeometry
from .path_topology import classify_filled_paths, component_paths


def publish_region_paths(
    region: DrawingRegion, paths: Sequence[PathGeometry], *,
    fill_rule: Literal['nonzero', 'evenodd'], defer_overlaps: bool = False,
    source_path_order: Sequence[str] | None = None,
) -> DrawingRegion:
    """Publish filled components without changing curves, paint or lineage.

    Source path slots determine child addresses even when earlier paths have
    been retired. Scene-wide clone rewrites remain the caller's transaction.
    """
    if not region.is_leaf:
        raise ValueError('path publication requires a drawing-region leaf')
    paths = tuple(paths)
    identities = tuple(path.path.id for path in paths)
    order = tuple(source_path_order) if source_path_order is not None else identities
    if (len(set(identities)) != len(identities) or len(set(order)) != len(order)
            or not set(identities).issubset(order)):
        raise ValueError('component path order must uniquely include every retained path')
    topology = classify_filled_paths(paths, fill_rule=fill_rule, defer_overlaps=defer_overlaps)
    diagnostics = dict(region.diagnostics)
    if isinstance(topology.components[0], CompoundPathGeometry):
        compound = topology.components[0]
        return replace(region, geometry=compound, diagnostics={**diagnostics, 'topology': {
            'operation': 'deferred_topology', 'pending': True, 'reason': compound.reason,
            'source_path_ids': identities, 'resolution': 'stitch_or_plan_geometry_edit',
        }})
    if diagnostics.get('topology', {}).get('pending'):
        diagnostics['topology'] = {'operation': 'resolved_topology', 'pending': False}
    if len(topology.components) == 1 and not topology.redundant_paths:
        return replace(region, geometry=topology.components[0], diagnostics=diagnostics)
    children = []
    ownership = []
    for component in topology.components:
        members = {path.path.id for path in component_paths(component)}
        indices = tuple(i for i, identity in enumerate(identities) if identity in members)
        output_ids = tuple(identities[i] for i in indices)
        outer = component_paths(component)[0].path.id
        address = f'{region.id}-{order.index(outer) + 1}' if len(topology.components) > 1 else region.id
        children.append(replace(region, id=address, geometry=component, diagnostics={
            **diagnostics, 'topology': {'source_target_id': region.id,
                'source_path_ids': output_ids, 'source_subpath_indices': indices},
        }))
        ownership.extend((identity, address) for identity in output_ids)
    retired = tuple(path.path.id for path in topology.redundant_paths)
    receipt = {
        'operation': 'retained_topology', 'source_target_id': region.id,
        'source_path_ids': identities, 'path_owner_remap': tuple(ownership),
        'created_target_ids': tuple(child.id for child in children) if len(children) > 1 else (),
        'retired_target_ids': (), 'retired_path_ids': retired,
        'retired_feature_ids': tuple(anchor.id for path in topology.redundant_paths for anchor in path.anchors),
        'preserved_command_ids': tuple(command.id for path in paths if path.path.id not in retired
                                       for command in path.path.commands),
    }
    if len(children) == 1:
        return replace(children[0], diagnostics={**children[0].diagnostics, 'topology': receipt})
    return replace(region, geometry=None, children=tuple(children),
        diagnostics={**diagnostics, 'topology': receipt})

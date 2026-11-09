"""Canonical component publication shared by region actions and scene adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal

from .drawing import Drawing, DrawingRegion
from .path_geometry import CloneGeometry, CompoundPathGeometry, PathGeometry, PathGeometryWithCutouts
from .path_topology import classify_filled_paths, component_paths


def match_topology_paths(inputs, outputs):
    """Validate retained ownership from representation-independent path records."""
    from .refinement.automatic_replay import TopologyPathCorrespondence, PassInputRegionRef
    records = []
    for path_id, input_values in inputs.items():
        output_values = outputs.get(path_id, [])
        if not output_values:
            continue
        if len(input_values) != 1 or len(output_values) != 1:
            raise ValueError('retained topology path ownership is ambiguous')
        input_ancestry, input_index, input_revision, input_count = input_values[0]
        output_ancestry, output_index, output_revision, output_count = output_values[0]
        if input_revision != output_revision or input_count != output_count:
            raise ValueError('retained topology rewrote a surviving path')
        records.append(TopologyPathCorrespondence(
            target=PassInputRegionRef('', '', input_ancestry),
            input_subpath_index=input_index, input_path_id=path_id,
            input_path_revision=input_revision, output_ancestry=output_ancestry,
            output_subpath_index=output_index, output_path_id=path_id,
            output_path_revision=output_revision, command_count=input_count))
    return tuple(records)


def retired_corner_features(source_text, evidence, identities, retired):
    """Retire observed features with their source paths, including diagnostic-only declarations."""
    from .corner_diagnostics import source_ordered_corner_diagnostics
    if not isinstance(evidence, Mapping):
        return ()
    evidence = source_ordered_corner_diagnostics(source_text, evidence)
    return tuple(
        f"{identities[int(span['subpath'])]}.{str(span['corner_id']).split('.', 1)[-1]}"
        for span in (evidence.get('spans') or ()) if isinstance(span, Mapping)
        and type(span.get('subpath')) is int and 0 <= span['subpath'] < len(identities)
        and identities[span['subpath']] in retired and 'corner_id' in span)


def expand_clone_components(region: DrawingRegion, source: DrawingRegion) -> DrawingRegion:
    """Expand a clone over already selected component slots without fitting."""
    if not isinstance(region.geometry, CloneGeometry) or not source.is_branch:
        raise ValueError('clone component expansion requires a clone and selected component group')
    children = tuple(replace(region,
        id=f'{region.id}-{child.id[len(source.id) + 1:]}',
        geometry=replace(region.geometry, source_region_id=child.id),
        diagnostics={**region.diagnostics, 'clone': {'source': child.id},
            'topology': {'source_target_id': region.id, 'clone_source_target_id': child.id}},
    ) for child in source.children)
    return replace(region, geometry=None, children=children, diagnostics={**region.diagnostics,
        'topology': {'operation': 'retained_topology_clone', 'source_target_id': region.id,
            'created_target_ids': tuple(child.id for child in children), 'retired_target_ids': ()}})


def normalize_drawing_topology(
    drawing: Drawing, *, allow_external: bool = False, defer_overlaps: bool = False,
    component_path_orders: Mapping[str, Sequence[str]] | None = None,
    normalize_components: bool = True,
) -> Drawing:
    """Normalize canonical component ownership and clone dependencies atomically."""
    prior: dict[str, DrawingRegion] = {}
    def index(region: DrawingRegion) -> None:
        prior[region.id] = region
        for child in region.children:
            index(child)
    for root in drawing.regions:
        index(root)
    built: dict[str, DrawingRegion] = {}
    active: set[str] = set()

    def visit(identity: str) -> DrawingRegion:
        if identity in built:
            return built[identity]
        if identity in active:
            raise ValueError('cyclic retained clone reference')
        active.add(identity)
        region = prior[identity]
        selected = region
        if region.is_branch:
            children = tuple(visit(child.id) for child in region.children)
            if any(first is not second for first, second in zip(children, region.children, strict=True)):
                selected = replace(region, children=children)
        elif isinstance(region.geometry, CloneGeometry):
            source_id = region.geometry.source_region_id
            if source_id not in prior:
                if not allow_external:
                    raise ValueError('clone source is not retained')
            else:
                source = visit(source_id)
                if normalize_components and source.is_branch:
                    if prior[source_id].is_branch or source.diagnostics.get('topology', {}).get('operation') != 'retained_topology':
                        raise ValueError('clone source requires retained concrete geometry')
                    selected = expand_clone_components(region, source)
        elif normalize_components and isinstance(region.geometry,
                (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
            paths = component_paths(region.geometry)
            order = tuple((component_path_orders or {}).get(identity, (path.path.id for path in paths)))
            positions = {path_id: number for number, path_id in enumerate(order)}
            if not all(path.path.id in positions for path in paths):
                raise ValueError('component path order must uniquely include every retained path')
            paths = tuple(sorted(paths, key=lambda path: positions[path.path.id]))
            rule = region.geometry.fill_rule if isinstance(region.geometry, CompoundPathGeometry) else (
                'evenodd' if isinstance(region.geometry, PathGeometryWithCutouts) else 'nonzero')
            selected = publish_region_paths(region, paths, fill_rule=rule,
                source_path_order=order, defer_overlaps=defer_overlaps
                    or bool(region.diagnostics.get('topology', {}).get('pending')))
            if selected == region and selected.diagnostics == region.diagnostics:
                selected = region
        active.remove(identity)
        built[identity] = selected
        return selected

    roots = tuple(visit(region.id) for region in drawing.regions)
    return drawing if all(first is second for first, second in zip(roots, drawing.regions, strict=True)) else replace(drawing, regions=roots)


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
          **diagnostics,
          **({'corners': _component_corners(diagnostics['corners'],
              ' '.join(path.to_svg_d() for path in paths), component.to_svg_d(), indices)}
             if 'corners' in diagnostics else {}),
          'topology': {'source_target_id': region.id,
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



def _component_corners(evidence: object, d: str, output_d: str, indices: tuple[int, ...]) -> dict[str, object]:
    """Move existing declarations by exact source-subpath index, never coordinates."""
    from .corner_diagnostics import source_ordered_corner_diagnostics, recount_corner_diagnostics
    if not isinstance(evidence, Mapping):
        return {"path": output_d, "spans": [], "skipped": [{"reason": "missing_or_stale_provenance"}]}
    if evidence.get("path") != d:
        return {**evidence, "skipped": [*evidence.get("skipped", ()), {"reason": "missing_or_stale_provenance"}]}
    evidence = source_ordered_corner_diagnostics(d, evidence)
    output = {**evidence, "path": output_d}
    skipped = list(evidence.get("skipped", ()))
    for field in ("spans", "anchors"):
        items = []
        declarations = evidence.get(field, ())
        if not isinstance(declarations, Sequence) or isinstance(declarations, (str, bytes)):
            skipped.append({"reason": "invalid_corner_provenance", "field": field})
            declarations = ()
        for item in declarations:
            if not isinstance(item, Mapping) or type(item.get("subpath")) is not int or not 0 <= item["subpath"] < d.count("M"):
                skipped.append({"reason": "invalid_corner_provenance", "field": field})
                continue
            if item["subpath"] not in indices:
                continue
            old = int(item["subpath"])
            new = indices.index(old)
            identity = str(item.get("corner_id", ""))
            prefix = f"p{old}."
            if identity.startswith(prefix):
                identity = f"p{new}." + identity[len(prefix):]
            items.append({**item, "subpath": new, "corner_id": identity})
        output[field] = items
    output.pop("subpaths", None)
    output["skipped"] = skipped
    return recount_corner_diagnostics(output_d, output)

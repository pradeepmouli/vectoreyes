"""Canonical component publication shared by region actions and scene adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer.vector_region import VectorRegion
    from .optimizer.automatic_replay import NormalizeRetainedTopologyOperation, TopologyPathCorrespondence

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



def _component_corners(evidence: object, d: str, output_d: str, indices: tuple[int, ...]) -> dict[str, object]:
    """Move existing declarations by exact source-subpath index, never coordinates."""
    from .optimizer.corners import source_ordered_corner_diagnostics, recount_corner_diagnostics
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


def normalize_retained_topology(
    regions: Sequence[VectorRegion], *, allow_external: bool = False,
    defer_overlaps: bool = False,
    component_path_orders: Mapping[str, Sequence[str]] | None = None,
) -> tuple[VectorRegion, ...]:
    """Publish real component children, exact references and identity correspondence.

    All decisions use the original scoped forest before any replacement, so
    failure is atomic and new child IDs cannot capture an unrelated clone.
    Retained replay may supply original source-path slots after redundant paths
    have been retired, preserving component IDs without recreating that geometry.
    """
    from .drawing_adapter import transform_working_shape
    from .fit import Shape
    from .path_geometry import parse_svg_paths
    from .optimizer.gate import rasterize
    from .optimizer.corners import source_ordered_corner_diagnostics
    from .optimizer.symmetry_materialize import index_region_paths, relative_source_path, resolve_source_path
    from .optimizer.vector_region import VectorRegion, to_polygon
    import re
    prior = index_region_paths(regions)
    built: dict[tuple[int, ...], VectorRegion] = {}
    active: set[tuple[int, ...]] = set()

    def branch(region, children, diagnostics):
        return VectorRegion(
            region.id, None, region.fill, region.z, region.footprint,
            raster=region.raster, original=region.original, children=children,
            source_label=region.source_label, color_hex=region.color_hex,
            drawing_id=region.drawing_id, source_regions=region.source_regions,
            coverage=region.coverage, diagnostics={**region.diagnostics, "topology": diagnostics},
        )

    def visit(path):
        if path in built:
            return built[path]
        if path in active:
            raise ValueError("cyclic retained clone reference")
        active.add(path)
        region = prior[path]
        result = region
        if region.is_branch:
            children = tuple(visit((*path, child.id)) for child in region.children)
            if any(a is not b for a, b in zip(children, region.children, strict=True)):
                result = region.with_children(children)
        elif region.current.kind == "path":
            shape = region.current
            d = str(shape.params["d"])
            address = region.drawing_id or "-".join(str(part) for part in path)
            parsed = parse_svg_paths(d, region_id=address, path_ids=shape.params.get("path_ids"))
            identities = tuple(item.path.id for item in parsed)
            source_order = tuple((component_path_orders or {}).get(address, identities))
            from .path_geometry import CompoundPathGeometry

            published = publish_region_paths(
                DrawingRegion(address, parsed[0], diagnostics=region.diagnostics), parsed,
                fill_rule=shape.params.get("fill_rule", "nonzero"), source_path_order=source_order,
                defer_overlaps=defer_overlaps or bool(region.diagnostics.get("topology", {}).get("pending")))
            if isinstance(published.geometry, CompoundPathGeometry):
                result = region.with_current(Shape("path", {**shape.params, "path_ids": tuple(item.path.id for item in parsed)}),
                    footprint=region.footprint, diagnostics={"topology": published.diagnostics['topology']})
                active.remove(path)
                built[path] = result
                return result
            if region.diagnostics.get("topology", {}).get("pending"):
                params = dict(shape.params)
                params.pop("canonical_path_geometry", None)
                shape = Shape("path", params)
                region = region.with_current(shape, footprint=region.footprint,
                    diagnostics={"topology": {"operation": "resolved_topology", "pending": False}})
                result = region
            retained_ids = {item.path.id for leaf in published.leaves() for item in component_paths(leaf.geometry)}
            if published.is_branch or set(identities) != retained_ids:
                # Source text is preserved verbatim per subpath, keeping fit
                # digest bindings valid even when serialization style differs.
                raw = tuple(part.strip() for part in re.findall(r"M[^M]*", d))
                children = []
                for selected in published.leaves():
                    component = selected.geometry
                    members = {item.path.id for item in component_paths(component)}
                    indices = tuple(i for i, identity in enumerate(identities) if identity in members)
                    output_d = " ".join(raw[i] for i in indices)
                    output_ids = tuple(identities[i] for i in indices)
                    local_id = int(selected.id.rsplit('-', 1)[1]) if published.is_branch else region.id
                    child_address = selected.id
                    output_shape = Shape("path", {**shape.params, "d": output_d, "path_ids": output_ids,
                                                   "fill_rule": "evenodd" if len(indices) > 1 else "nonzero"})
                    footprint = to_polygon(output_shape)
                    diagnostics = {**region.diagnostics, "corners": _component_corners(
                        region.diagnostics.get("corners"), d, output_d, indices,
                    ), "topology": selected.diagnostics['topology']}
                    child = VectorRegion(
                        local_id,
                        output_shape, region.fill, region.z, footprint,
                        raster=rasterize(footprint, region.raster.shape) if region.raster.size else region.raster,
                        original=region.original, drawing_id=child_address,
                        source_label=region.source_label, color_hex=region.color_hex,
                        source_regions=region.source_regions, coverage=region.coverage, diagnostics=diagnostics,
                    )
                    children.append(child)
                correspondence = dict(published.diagnostics['topology'])
                retired = correspondence['retired_path_ids']
                evidence = region.diagnostics.get("corners", {})
                if isinstance(evidence, Mapping):
                    evidence = source_ordered_corner_diagnostics(d, evidence)
                retired_features = tuple(
                    f"{identities[int(span['subpath'])]}.{str(span['corner_id']).split('.', 1)[-1]}"
                    for span in (evidence.get("spans") or ()) if isinstance(span, Mapping)
                    and type(span.get("subpath")) is int and 0 <= span["subpath"] < len(identities)
                    and identities[span["subpath"]] in retired and "corner_id" in span
                ) if isinstance(evidence, Mapping) else ()
                correspondence['retired_feature_ids'] = retired_features
                result = branch(region, tuple(children), correspondence) if len(children) > 1 else children[0].with_diagnostics({"topology": correspondence})
        elif region.current.kind == "use":
            source_path = resolve_source_path(prior, path, region.current.params, allow_external=allow_external)
            if source_path is None:
                active.remove(path)
                built[path] = result
                return result
            source = visit(source_path)
            if source.is_branch:
                if not prior[source_path].is_leaf or source.diagnostics.get("topology", {}).get("operation") != "retained_topology":
                    raise ValueError("clone source requires retained concrete geometry")
                children = []
                address = region.drawing_id or "-".join(str(part) for part in path)
                for child in source.children:
                    child_path = (*path, child.id)
                    source_child_path = (*source_path, child.id)
                    params = {**region.current.params, "href_obj_id": child.id,
                              "href_region_path": relative_source_path(child_path, source_child_path)}
                    footprint = to_polygon(transform_working_shape(child.current, tuple(params["transform"])))
                    children.append(VectorRegion(
                        child.id, Shape("use", params), region.fill, region.z, footprint,
                        raster=rasterize(footprint, region.raster.shape) if region.raster.size else region.raster,
                        original=region.original, drawing_id=f"{address}-{child.id}",
                        source_label=region.source_label, color_hex=region.color_hex,
                        source_regions=region.source_regions, coverage=region.coverage,
                        diagnostics={**region.diagnostics, "clone": {"source": child.drawing_id},
                                     "topology": {"source_target_id": address, "clone_source_target_id": child.drawing_id}},
                    ))
                result = branch(region, tuple(children), {
                    "operation": "retained_topology_clone", "source_target_id": address,
                    "source_region_path": source_path,
                    "created_target_ids": tuple(child.drawing_id for child in children), "retired_target_ids": (),
                })
            else:
                # Freeze the pre-normalization scoped binding before new local
                # component IDs can shadow an otherwise unique external ID.
                params = {**region.current.params, "href_region_path": relative_source_path(path, source_path)}
                if params != region.current.params:
                    result = region.with_current(Shape("use", params), footprint=region.footprint)
        active.remove(path)
        built[path] = result
        return result

    return tuple(visit((region.id,)) for region in regions)


def topology_path_correspondence(
    before: list[VectorRegion], after: tuple[VectorRegion, ...]
) -> tuple[TopologyPathCorrespondence, ...]:
    """Bind every surviving path to its exact normalized owner and ordinal."""
    from .optimizer.automatic_replay import TopologyPathCorrespondence, PassInputRegionRef
    from .path_geometry import parse_svg_paths
    from .drawing_seams import seam_path_revision
    from .optimizer.symmetry_materialize import index_region_paths
    def paths(regions):
        result: dict[str, list[tuple[tuple[int, ...], int, str, int]]] = {}
        for ancestry, region in index_region_paths(regions).items():
            if not region.is_leaf or region.current.kind != "path":
                continue
            d = str(region.current.params.get("d", ""))
            address = region.drawing_id or "-".join(str(part) for part in ancestry)
            parsed = parse_svg_paths(
                d,
                region_id=address,
                path_ids=region.current.params.get("path_ids"),
            )
            for index, path in enumerate(parsed):
                result.setdefault(path.path.id, []).append((
                    tuple(ancestry),
                    index,
                    seam_path_revision(d, index),
                    len(path.path.commands),
                ))
        return result

    inputs = paths(before)
    outputs = paths(after)
    records: list[TopologyPathCorrespondence] = []
    for path_id, input_values in inputs.items():
        output_values = outputs.get(path_id, [])
        if not output_values:
            continue
        if len(input_values) != 1 or len(output_values) != 1:
            raise ValueError("retained topology path ownership is ambiguous")
        input_ancestry, input_index, input_revision, input_count = input_values[0]
        output_ancestry, output_index, output_revision, output_count = output_values[0]
        if input_revision != output_revision or input_count != output_count:
            raise ValueError("retained topology rewrote a surviving path")
        records.append(TopologyPathCorrespondence(
            target=PassInputRegionRef("", "", input_ancestry),
            input_subpath_index=input_index,
            input_path_id=path_id,
            input_path_revision=input_revision,
            output_ancestry=output_ancestry,
            output_subpath_index=output_index,
            output_path_id=path_id,
            output_path_revision=output_revision,
            command_count=input_count,
        ))
    return tuple(records)


def apply_retained_topology(
    action: NormalizeRetainedTopologyOperation, snapshot: list[VectorRegion],
) -> list[VectorRegion]:
    """Normalize the bound forest and verify exact surviving-path ownership."""
    normalized = normalize_retained_topology(snapshot, allow_external=action.allow_external)
    recorded = tuple(replace(item, target=replace(item.target, batch_id="", input_revision=""))
                     for item in action.path_correspondence)
    if topology_path_correspondence(snapshot, normalized) != recorded:
        raise ValueError("retained topology path correspondence is stale")
    affected = set(action.affected_ids)
    return [obj for obj in normalized if int(obj.id) in affected]

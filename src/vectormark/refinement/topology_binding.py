"""Disposable working-record bindings for canonical topology operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from ..drawing import Canvas, Drawing, DrawingRegion
from ..drawing_topology import _component_corners, expand_clone_components, normalize_drawing_topology
from ..path_topology import component_paths

if TYPE_CHECKING:
    from .region_sample import RegionSample
    from .automatic_replay import NormalizeRetainedTopologyOperation, TopologyPathCorrespondence


def normalize_working_topology(
    regions: Sequence[RegionSample], *, allow_external: bool = False,
    defer_overlaps: bool = False,
    component_path_orders: Mapping[str, Sequence[str]] | None = None,
    normalize_components: bool = True,
) -> tuple[RegionSample, ...]:
    """Publish real component children, exact references and identity correspondence.

    All decisions use the original scoped forest before any replacement, so
    failure is atomic and new child IDs cannot capture an unrelated clone.
    Retained replay may supply original source-path slots after redundant paths
    have been retired, preserving component IDs without recreating that geometry.
    ``normalize_components=False`` refreshes resolved clone/branch geometry only,
    allowing paint invalidation before the existing component-publication step.
    """
    from ..drawing_adapter import transform_working_shape, region_sample_address_index, _project_cached_paths, drawing_from_region_samples
    from ..geometry_native import affine_tuple
    from ..path_geometry import CloneGeometry
    from ..skia_geometry import unary_union
    from ..fit import Shape
    from ..path_geometry import parse_svg_paths
    from ..refinement.gate import rasterize
    from ..drawing_topology import retired_corner_features
    from ..refinement.symmetry_materialize import index_region_paths, relative_source_path, resolve_source_path
    from ..refinement.region_sample import RegionSample, to_polygon
    import re
    prior = index_region_paths(regions)
    built: dict[tuple[int, ...], RegionSample] = {}
    active: set[tuple[int, ...]] = set()

    def branch(region, children, diagnostics):
        return RegionSample(
            region.id, None, region.fill, region.z, region.footprint,
            raster=region.raster, original=region.original, children=children,
            source_label=region.source_label, color_hex=region.color_hex,
            drawing_id=region.drawing_id, source_regions=region.source_regions,
            coverage=region.coverage, diagnostics={**region.diagnostics, "topology": diagnostics},
        )

    def clone_components(region, source, path, source_path, geometry):
        if not prior[source_path].is_leaf or source.diagnostics.get('topology', {}).get('operation') != 'retained_topology':
            raise ValueError('clone source requires retained concrete geometry')
        address = region.drawing_id or '-'.join(str(part) for part in path)
        source_address = source.drawing_id or '-'.join(str(part) for part in source_path)
        canonical_source = drawing_from_region_samples('topology-source', width=1, height=1,
            regions=(replace(source, drawing_id=source_address),)).regions[0]
        selected = expand_clone_components(DrawingRegion(address, geometry,
            diagnostics=region.diagnostics), canonical_source)
        children = []
        for child, output in zip(source.children, selected.children, strict=True):
            if region.current.kind == 'use':
                params = {**region.current.params, 'href_obj_id': child.id,
                    'href_region_path': relative_source_path((*path, child.id), (*source_path, child.id))}
                footprint = to_polygon(transform_working_shape(child.current, tuple(params['transform'])))
                shape = Shape('use', params)
            else:
                footprint = unary_union([child.footprint.affine_transform(affine_tuple(transform))
                    for transform in geometry.transforms])
                shape = Shape('path', {**region.current.params, 'd': footprint.to_svg_d(),
                    'canonical_clone_geometry': output.geometry})
            children.append(RegionSample(child.id, shape, region.fill, region.z, footprint,
                raster=rasterize(footprint, region.raster.shape) if region.raster.size else region.raster,
                original=region.original, drawing_id=output.id, source_label=region.source_label,
                color_hex=region.color_hex, source_regions=region.source_regions,
                coverage=region.coverage, diagnostics=output.diagnostics))
        return branch(region, tuple(children), {**selected.diagnostics['topology'],
            'source_region_path': source_path})

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
                result = replace(result, footprint=unary_union([child.footprint for child in children]))
        elif isinstance(region.current.params.get("canonical_clone_geometry"), CloneGeometry):
            geometry = region.current.params["canonical_clone_geometry"]
            sources = {address: ancestry for ancestry, address in region_sample_address_index(regions).items()}
            source_path = sources.get(geometry.source_region_id)
            if source_path is None:
                if not allow_external:
                    raise ValueError("clone source is not retained")
            else:
                source = visit(source_path)
                if source.is_branch and normalize_components:
                    result = clone_components(region, source, path, source_path, geometry)
                else:
                    footprint = unary_union([source.footprint.affine_transform(affine_tuple(transform))
                        for transform in geometry.transforms])
                    if footprint.to_svg_d() != region.footprint.to_svg_d():
                        result = region.with_current(Shape("path", {**region.current.params,
                            "d": footprint.to_svg_d()}), footprint=footprint)
        elif region.current.kind == "path" and normalize_components:
            shape = region.current
            d = str(shape.params["d"])
            address = region.drawing_id or "-".join(str(part) for part in path)
            cached = shape.params.get('canonical_path_geometry')
            parsed = (_project_cached_paths(cached, d, region_id=address,
                    path_ids=shape.params.get('path_ids')) if cached is not None
                else parse_svg_paths(d, region_id=address, path_ids=shape.params.get('path_ids')))
            identities = tuple(item.path.id for item in parsed)
            source_order = tuple((component_path_orders or {}).get(address, identities))
            from ..path_geometry import CompoundPathGeometry

            staged = CompoundPathGeometry(parsed, shape.params.get('fill_rule', 'nonzero'),
                'pending component publication') if len(parsed) > 1 else parsed[0]
            published = normalize_drawing_topology(
                Drawing('topology-input', Canvas(1, 1), (DrawingRegion(address, staged,
                    diagnostics=region.diagnostics),)),
                component_path_orders={address: source_order}, defer_overlaps=defer_overlaps,
            ).region(address)
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
                    if (cached := shape.params.get("canonical_path_geometry")) is not None:
                        from ..drawing_corner_actions import rebind_corner_annotations
                        from ..path_topology import replace_component_paths
                        cached_paths = {item.path.id: item for item in component_paths(cached)}
                        local_paths = tuple(rebind_corner_annotations(item, cached_paths[item.path.id])
                            for item in component_paths(component))
                        local_geometry = replace_component_paths(component, local_paths)
                        output_shape = Shape("path", {**output_shape.params,
                            "canonical_path_geometry": local_geometry})
                    footprint = to_polygon(output_shape)
                    diagnostics = {**region.diagnostics, "corners": _component_corners(
                        region.diagnostics.get("corners"), d, output_d, indices,
                    ), "topology": selected.diagnostics['topology']}
                    child = RegionSample(
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
                correspondence['retired_feature_ids'] = retired_corner_features(
                    d, region.diagnostics.get('corners', {}), identities, retired)
                result = branch(region, tuple(children), correspondence) if len(children) > 1 else children[0].with_diagnostics({"topology": correspondence})
        elif region.current.kind == "use":
            source_path = resolve_source_path(prior, path, region.current.params, allow_external=allow_external)
            if source_path is None:
                active.remove(path)
                built[path] = result
                return result
            source = visit(source_path)
            if source.is_branch and normalize_components:
                from ..path_geometry import AffineTransform
                source_address = source.drawing_id or '-'.join(str(part) for part in source_path)
                geometry = CloneGeometry(source_address,
                    (AffineTransform(*region.current.params['transform']),))
                result = clone_components(region, source, path, source_path, geometry)
            else:
                # Freeze the pre-normalization scoped binding before new local
                # component IDs can shadow an otherwise unique external ID.
                params = {**region.current.params, "href_region_path": relative_source_path(path, source_path)}
                footprint = source.footprint.affine_transform(tuple(params["transform"]))
                if params != region.current.params or footprint.to_svg_d() != region.footprint.to_svg_d():
                    result = region.with_current(Shape("use", params), footprint=footprint)
        active.remove(path)
        built[path] = result
        return result

    return tuple(visit((region.id,)) for region in regions)


def topology_path_correspondence(
    before: list[RegionSample], after: tuple[RegionSample, ...]
) -> tuple[TopologyPathCorrespondence, ...]:
    """Bind every surviving path to its exact normalized owner and ordinal."""
    from ..drawing_topology import match_topology_paths
    from ..path_geometry import parse_svg_paths
    from ..drawing_seams import seam_path_revision
    from ..refinement.symmetry_materialize import index_region_paths
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

    return match_topology_paths(paths(before), paths(after))


def apply_retained_topology(
    action: NormalizeRetainedTopologyOperation, snapshot: list[RegionSample],
) -> list[RegionSample]:
    """Normalize the bound forest and verify exact surviving-path ownership."""
    normalized = normalize_working_topology(snapshot, allow_external=action.allow_external)
    recorded = tuple(replace(item, target=replace(item.target, batch_id="", input_revision=""))
                     for item in action.path_correspondence)
    if topology_path_correspondence(snapshot, normalized) != recorded:
        raise ValueError("retained topology path correspondence is stale")
    affected = set(action.affected_ids)
    return [obj for obj in normalized if int(obj.id) in affected]

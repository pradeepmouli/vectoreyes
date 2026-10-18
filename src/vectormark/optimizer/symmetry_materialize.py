"""Explicit retained self-symmetry union, shared by automatic and manual actions."""

from collections.abc import Mapping, Sequence

from ..fit import Shape
from ..skia_geometry import SkPath, unary_union
from .vector_region import VectorRegion, to_polygon


RegionPath = tuple[int, ...]


def index_region_paths(regions: Sequence[VectorRegion]) -> dict[RegionPath, VectorRegion]:
    """Index retained locations, preserving IDs that repeat in different scopes."""
    result: dict[RegionPath, VectorRegion] = {}
    def visit(region: VectorRegion, path: RegionPath) -> None:
        if path in result:
            raise ValueError("duplicate region identity within one scope")
        result[path] = region
        for child in region.children:
            visit(child, (*path, child.id))
    for region in regions:
        visit(region, (region.id,))
    return result


def resolve_source_path(
    index: Mapping[RegionPath, VectorRegion], owner: RegionPath,
    params: Mapping[str, object], *, allow_external: bool = False,
) -> RegionPath | None:
    """Resolve explicit relative paths, then local IDs, then unique external IDs.

    A relative path starts at the clone's containing scope; leading ``..``
    steps make it stable when a recursive optimizer operates on a subtree.
    Only absent external context may be deferred by an execution subpass.
    """
    source_id = params.get("href_obj_id")
    if type(source_id) is not int:
        raise ValueError("clone reference requires an integer source ID")
    explicit = params.get("href_region_path")
    if explicit is not None:
        if not isinstance(explicit, (tuple, list)) or not explicit:
            raise ValueError("invalid explicit clone reference path")
        up = 0
        while up < len(explicit) and explicit[up] == "..":
            up += 1
        tail = explicit[up:]
        if not tail or any(type(part) is not int for part in tail) or tail[-1] != source_id:
            raise ValueError("invalid explicit clone reference path")
        scope = owner[:-1]
        if up > len(scope):
            if allow_external:
                return None
            raise ValueError("clone reference path escapes the retained root")
        path = (*scope[:len(scope) - up], *tail)
        if path not in index:
            raise ValueError("explicit clone source is missing")
        return path
    local = (*owner[:-1], source_id)
    if local in index:
        return local
    candidates = [path for path, region in index.items() if region.is_leaf and region.id == source_id]
    if len(candidates) > 1:
        raise ValueError("ambiguous cross-scope clone source")
    if candidates:
        return candidates[0]
    if allow_external:
        return None
    raise ValueError("clone source is missing")


def relative_source_path(owner: RegionPath, source: RegionPath) -> tuple[int | str, ...]:
    """Encode an exact reference relative to its owner's containing scope."""
    scope = owner[:-1]
    common = 0
    while common < min(len(scope), len(source)) and scope[common] == source[common]:
        common += 1
    return (*(".." for _ in scope[common:]), *source[common:])


def union_self_symmetry(
    region: VectorRegion,
    pieces: Sequence[SkPath],
    *,
    diagnostics: Mapping[str, object],
) -> VectorRegion:
    """Join existing vector pieces into one fresh, painted retained region."""
    footprint = unary_union(pieces)
    if footprint.is_empty:
        raise ValueError("self-symmetry materialization produced empty geometry")
    leaves = region.leaves()
    if any(leaf.fill != leaves[0].fill for leaf in leaves):
        raise ValueError("self-symmetry materialization requires one shared fill")
    consumed = tuple(dict.fromkeys(leaf.id for leaf in leaves))
    merged = dict(region.diagnostics)
    merged["symmetry"] = {**diagnostics, "materialized": True}
    merged["materialization"] = {
        "operation": "self_symmetry_union",
        "consumed_region_ids": consumed,
        "retired_region_ids": tuple(i for i in consumed if i != region.id),
        "source_region_remap": tuple((i, region.id) for i in consumed),
        "consumed_target_ids": tuple(leaf.drawing_id for leaf in leaves if leaf.drawing_id),
    }
    return VectorRegion(
        region.id, Shape("path", {"d": footprint.to_svg_d(), "fill_rule": "evenodd"}),
        leaves[0].fill, region.z, footprint=footprint, raster=region.raster,
        original=leaves[0].original, source_label=region.source_label,
        color_hex=region.color_hex, drawing_id=region.drawing_id,
        source_regions=tuple(sorted(set(region.source_regions).union(
            *(leaf.source_regions for leaf in leaves)))),
        coverage=region.coverage, diagnostics=merged,
    )


def remap_symmetry_clones(
    regions: Sequence[VectorRegion], sources: Mapping[RegionPath, RegionPath],
    *, before: Sequence[VectorRegion],
) -> tuple[VectorRegion, ...]:
    """Remap exact before-forest bindings, never an unrelated same-numbered ID."""
    from ..drawing_adapter import transform_working_shape
    prior = index_region_paths(before)
    current = index_region_paths(regions)
    def visit(region: VectorRegion, path: RegionPath) -> VectorRegion:
        if region.is_branch:
            children = tuple(visit(child, (*path, child.id)) for child in region.children)
            return region.with_children(children) if any(a is not b for a, b in zip(children, region.children)) else region
        shape = region.current
        assert shape is not None
        original = prior.get(path)
        if shape.kind != "use" or original is None or original.current is None or original.current.kind != "use":
            return region
        old_path = resolve_source_path(prior, path, original.current.params, allow_external=True)
        if old_path is None:
            return region
        new_path = sources.get(old_path, old_path)
        source = current.get(new_path)
        if source is None or source.current is None:
            raise ValueError("remapped clone source requires retained concrete geometry")
        params = {**shape.params, "href_obj_id": source.id}
        # Pin cross-scope bindings explicitly, including when a newly retained
        # local ID would otherwise shadow the source from the original forest.
        if new_path != (*path[:-1], source.id) or "href_region_path" in params:
            params["href_region_path"] = relative_source_path(path, new_path)
        if old_path not in sources and params == shape.params:
            return region
        matrix = tuple(shape.params["transform"])
        footprint = (to_polygon(transform_working_shape(source.current, matrix))
                     if old_path in sources else region.footprint)
        diagnostics = {"clone_reference_remap": {
            "previous_source_region_id": original.current.params["href_obj_id"], "source_region_id": source.id,
            "previous_source_region_path": old_path, "source_region_path": new_path,
            "reason": "self_symmetry_materialization",
        }}
        for key, field in (("clone", "source_id"), ("clones", "matched_source")):
            relation = region.diagnostics.get(key)
            if isinstance(relation, Mapping):
                diagnostics[key] = {**relation, field: source.id}
        return region.with_current(
            Shape("use", params), footprint=footprint,
            diagnostics=diagnostics,
        )
    return tuple(visit(region, (region.id,)) for region in regions)


def materialize_self_symmetry(regions: Sequence[VectorRegion]) -> tuple[VectorRegion, ...]:
    """Consume self-symmetry branches; repeated materialization is a no-op."""
    sources: dict[RegionPath, RegionPath] = {}
    prior = index_region_paths(regions)
    def visit(region: VectorRegion, path: RegionPath) -> VectorRegion:
        if region.is_leaf:
            return region
        symmetry = region.diagnostics.get("symmetry")
        if isinstance(symmetry, Mapping) and symmetry.get("mode") == "self":
            leaves = region.leaves()
            if any(not isinstance(leaf.footprint, SkPath) for leaf in leaves):
                raise ValueError("self-symmetry materialization requires vector footprints")
            joined = union_self_symmetry(region, [leaf.footprint for leaf in leaves], diagnostics=symmetry)
            consumed = [location for location, leaf in prior.items()
                        if location[:len(path)] == path and leaf.is_leaf]
            for location in (path, *consumed):
                sources[location] = path
            joined = joined.with_diagnostics({"materialization": {
                **joined.diagnostics["materialization"],
                "source_region_path_remap": tuple((location, path) for location in consumed),
            }})
            return joined
        children = tuple(visit(child, (*path, child.id)) for child in region.children)
        return region.with_children(children) if any(a is not b for a, b in zip(children, region.children)) else region
    materialized = tuple(visit(region, (region.id,)) for region in regions)
    return remap_symmetry_clones(materialized, sources, before=regions) if sources else materialized

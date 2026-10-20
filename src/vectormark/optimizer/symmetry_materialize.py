"""Explicit retained self-symmetry union, shared by automatic and manual actions."""

from collections.abc import Mapping, Sequence

from ..skia_geometry import SkPath, unary_union
from .vector_region import VectorRegion


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


class UnboundCloneSourceError(ValueError):
    """A valid reference whose source is outside the available scope."""


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
            raise UnboundCloneSourceError("explicit clone source is missing")
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
    from ..drawing_symmetry import union_symmetric_regions
    from ..drawing_adapter import publish_collapsed_region

    selected = union_symmetric_regions(_canonical_symmetry_input(region), pieces,
        diagnostics=diagnostics)
    return publish_collapsed_region(selected, region, footprint=unary_union(pieces))


def _canonical_symmetry_input(region: VectorRegion):
    from dataclasses import replace
    from ..drawing_adapter import drawing_from_vector_regions, normalize_drawing_addresses, resolved_input_shape

    def resolved(item: VectorRegion) -> VectorRegion:
        if item.is_leaf:
            return replace(item, current=resolved_input_shape(item))
        return replace(item, children=tuple(resolved(child) for child in item.children))

    return drawing_from_vector_regions('symmetry-input', width=1, height=1,
        regions=normalize_drawing_addresses((replace(resolved(region),
            drawing_id=region.drawing_id or f'input-{region.id}'),))).regions[0]


def _is_geometry_root_leaf(obj: VectorRegion) -> bool:
    return (obj.is_leaf and obj.drawing_id is not None
        and isinstance(obj.diagnostics.get('geometry_seed'), Mapping))


def _geometry_root_footprint(obj: VectorRegion, fallback: SkPath) -> SkPath:
    from .vector_region import to_polygon

    if not _is_geometry_root_leaf(obj) or obj.original is None:
        return fallback
    seeded = to_polygon(obj.original)
    return seeded if not seeded.is_empty else fallback


def _materialized_self_symmetry_region(obj, **settings):
    from ..drawing_symmetry import materialize_symmetric_half
    from ..drawing_adapter import publish_collapsed_region
    from ..geometry_native import region_path

    selected = materialize_symmetric_half(_canonical_symmetry_input(obj), **settings)
    return publish_collapsed_region(selected, obj, footprint=region_path(selected, {}))


def remap_symmetry_clones(
    regions: Sequence[VectorRegion], sources: Mapping[RegionPath, RegionPath],
    *, before: Sequence[VectorRegion],
) -> tuple[VectorRegion, ...]:
    """Remap exact before-forest bindings, never an unrelated same-numbered ID."""
    from ..drawing_adapter import bind_region_inputs
    from .automatic_replay import apply_clone_reference_update
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
        bindings = bind_region_inputs({'source': source, 'target': region},
            paths={'source': new_path, 'target': path})
        return apply_clone_reference_update(bindings,
            previous_source_id=original.current.params["href_obj_id"],
            previous_source_path=old_path, refresh_footprint=old_path in sources,
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
        if (isinstance(symmetry, Mapping) and symmetry.get("mode") == "self"
                and not symmetry.get('materialized')):
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

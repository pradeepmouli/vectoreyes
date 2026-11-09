"""Disposable flat numeric inputs for shared seam discovery (never a retained scene)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from .fit import Shape
from .skia_geometry import SkPath
from .drawing_adapter import transform_working_shape
from .path_sampling import _parse_subpaths
from .refinement.automatic_replay import (
    ApplySeamConstraintsOperation, AxisSeamConstraint, EndpointAssignment,
    EndpointConstraintStage, PassInputRegionRef, PathOpsUnderlapConstraint,
    VertexClusterConstraint, ProposalIntent, operation_intent,
)
from .drawing_seams import (
    _SEAM_TOL, _JUNCTION_TOL, _SeamCandidate, SeamInput, _LeafUpdate,
    _CoordinateRef, _EndpointMatch, _path_endpoints_within,
    _pathops_shared_boundary_pair, _rewrite_shape_side, _true_up_path_vertices,
    _endpoint_stage, _path_endpoint_refs,
    _shape_from_tokens, _endpoint_ref, _path_correspondence,
)


@dataclass(frozen=True)
class SeamDetectionInput:
    reference: PassInputRegionRef
    shape: Shape | None
    footprint: SkPath
    order: tuple[float, float, int, int]
    owner_clip: SkPath
    diagnostics: Mapping[str, object]
    clone_source: tuple[int, ...] | None = None

    @property
    def id(self) -> int:
        return self.reference.ancestry[-1]


@dataclass(frozen=True)
class SeamDiscoveryChoice:
    owner_ids: tuple[int, ...]
    intent: ProposalIntent
    updates: Mapping[tuple[int, ...], _LeafUpdate]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'updates', MappingProxyType(dict(self.updates)))


def _leaf_key(ref: SeamDetectionInput) -> tuple[int, ...]:
    return tuple(ref.reference.ancestry)


def _paint_order(ref: SeamDetectionInput) -> tuple[float, float, int, int]:
    return ref.order


def _seam_input(ref: SeamDetectionInput, shape: Shape) -> SeamInput:
    return SeamInput(shape, ref.id, ref.reference.ancestry[0], ref.order, ref.owner_clip)


def _shape_bounds(obj: SeamDetectionInput) -> tuple[float, float, float, float] | None:
    if obj.footprint is None or getattr(obj.footprint, "is_empty", False):
        return None
    minx, miny, maxx, maxy = obj.footprint.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _same_surface_fill_group(a: SeamDetectionInput, b: SeamDetectionInput) -> bool:
    """Whether automatic stitching may alter this pair's shared boundary.

    Trace/refine assigns a ``surface_fill`` group after geometry fitting.  Its
    members are palette fragments of one material; neighbouring members of a
    different group can instead be a deliberate counter or overlaid detail.
    Legacy/manual regions carry no group and retain the previous permissive
    behavior.
    """
    # A compound child represents a nested fill/cutout layer from one source
    # path.  Its boundary is containment, not an independently traced material
    # seam.  It must never be moved toward a sibling by generic seam repair.
    if isinstance(a.diagnostics.get("compound"), Mapping) or isinstance(b.diagnostics.get("compound"), Mapping):
        return False
    a_fill = a.diagnostics.get("surface_fill")
    b_fill = b.diagnostics.get("surface_fill")
    if not isinstance(a_fill, Mapping) or not isinstance(b_fill, Mapping):
        return True
    return a_fill.get("group") == b_fill.get("group")


def _regions_close(a: SeamDetectionInput, b: SeamDetectionInput, *, tol: float) -> bool:
    """Cheaply reject separated bounds before asking Skia for exact distance.

    ``SkPath.distance`` is intentionally exact, but it is disproportionately
    expensive for the large, detailed paths retained by an interactive trace.
    The Euclidean distance between two axis-aligned bounding boxes is a lower
    bound on the path distance, so a bound that exceeds ``tol`` can never
    contain a seam candidate.  Nearby bounds still use the exact geometry
    check below.
    """
    a_bounds = _shape_bounds(a)
    b_bounds = _shape_bounds(b)
    if a_bounds is None or b_bounds is None:
        return False
    ax0, ay0, ax1, ay1 = a_bounds
    bx0, by0, bx1, by1 = b_bounds
    dx = max(ax0 - bx1, bx0 - ax1, 0.0)
    dy = max(ay0 - by1, by0 - ay1, 0.0)
    if dx * dx + dy * dy > tol * tol:
        return False
    if (
        a.shape is not None
        and b.shape is not None
        and a.shape.kind == "path"
        and b.shape.kind == "path"
    ):
        # Endpoint proximity is the actionable criterion for a path seam:
        # subsequent rewriting can move only those coordinates.  Avoid an
        # exact path-to-path distance here; it is costly for detailed,
        # overlapping traced paths and contributes no additional signal.
        return _path_endpoints_within(a.shape, b.shape, tol=tol)
    try:
        return float(a.footprint.distance(b.footprint)) <= tol
    except Exception:
        return False


def _overlap(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float] | None:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return None
    return float(lo), float(hi)


def _candidate_for_pair(a: SeamDetectionInput, b: SeamDetectionInput, *, tol: float) -> _SeamCandidate | None:
    if a.shape is None or b.shape is None:
        return None
    if a.shape.kind == "use" or b.shape.kind == "use":
        return None
    if not _regions_close(a, b, tol=tol):
        return None

    a_bounds = _shape_bounds(a)
    b_bounds = _shape_bounds(b)
    if a_bounds is None or b_bounds is None:
        return None
    ax0, ay0, ax1, ay1 = a_bounds
    bx0, by0, bx1, by1 = b_bounds

    y_span = _overlap(ay0, ay1, by0, by1)
    if y_span is not None:
        if 0.0 < bx0 - ax1 <= tol:
            seam = (ax1 + bx0) / 2.0
            return _SeamCandidate("x", "right", "left", ax1, bx0, seam, *y_span)
        if 0.0 < ax0 - bx1 <= tol:
            seam = (bx1 + ax0) / 2.0
            return _SeamCandidate("x", "left", "right", ax0, bx1, seam, *y_span)

    x_span = _overlap(ax0, ax1, bx0, bx1)
    if x_span is not None:
        if 0.0 < by0 - ay1 <= tol:
            seam = (ay1 + by0) / 2.0
            return _SeamCandidate("y", "bottom", "top", ay1, by0, seam, *x_span)
        if 0.0 < ay0 - by1 <= tol:
            seam = (by1 + ay0) / 2.0
            return _SeamCandidate("y", "top", "bottom", ay0, by1, seam, *x_span)
    if a.shape.kind != "path" or b.shape.kind != "path":
        return None
    return _SeamCandidate("vertices", "", "", 0.0, 0.0, 0.0, 0.0, 0.0)


def _apply_leaf_pair(
    a_ref: SeamDetectionInput,
    b_ref: SeamDetectionInput,
    updates_by_leaf: dict[tuple[int, ...], _LeafUpdate],
    nodes_by_path: dict[tuple[int, ...], SeamDetectionInput],
    intent_ref_by_key: dict[tuple[int, ...], PassInputRegionRef],
    constraints: list[AxisSeamConstraint | PathOpsUnderlapConstraint],
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
    allow_underlap: bool,
) -> tuple[int, int] | None:
    a = a_ref
    if a.shape is None:
        return None
    b = b_ref
    if b.shape is None:
        return None

    def current_shape(ref: SeamDetectionInput) -> Shape | None:
        key = _leaf_key(ref)
        if key in updates_by_leaf:
            return updates_by_leaf[key].shape
        leaf = ref
        if leaf.shape is None:
            return None
        if leaf.shape.kind != "use":
            return leaf.shape
        source = nodes_by_path.get(ref.clone_source)
        if source is None or source.shape is None:
            return None
        source_key = _leaf_key(source)
        source_shape = updates_by_leaf[source_key].shape if source_key in updates_by_leaf else source.shape
        return transform_working_shape(source_shape, leaf.shape.params["transform"])

    a_current = current_shape(a_ref)
    b_current = current_shape(b_ref)
    if a_current is None or b_current is None:
        return None
    a_for_candidate = replace(a, shape=a_current)
    b_for_candidate = replace(b, shape=b_current)
    candidate = _candidate_for_pair(a_for_candidate, b_for_candidate, tol=_SEAM_TOL)
    if candidate is None:
        return None
    a_key = _leaf_key(a_ref)
    b_key = _leaf_key(b_ref)
    if allow_underlap and candidate.axis == "vertices":
        pathops = _pathops_shared_boundary_pair(
            _seam_input(a_ref, a_current),
            _seam_input(b_ref, b_current),
            a_current,
            b_current,
            tol=_SEAM_TOL,
        )
        if pathops is not None:
            a_shape, b_shape, candidate = pathops
            constraints.append(PathOpsUnderlapConstraint(
                left=intent_ref_by_key[a_key],
                right=intent_ref_by_key[b_key],
                bottom="left" if _paint_order(a_ref) <= _paint_order(b_ref) else "right",
                tolerance=_SEAM_TOL,
            ))
            updates_by_leaf[a_key] = _LeafUpdate(
                a_shape,
                int(b.id),
                candidate,
                "pathops_shared_boundary",
            )
            updates_by_leaf[b_key] = _LeafUpdate(
                b_shape,
                int(a.id),
                candidate,
                "pathops_authoritative",
            )
            return a_ref.reference.ancestry[0], b_ref.reference.ancestry[0]
    a_shape = _rewrite_shape_side(
        a_current,
        side=candidate.a_side,
        candidate=candidate,
        side_value=candidate.a_value,
        tol=_SEAM_TOL,
    )
    b_shape = _rewrite_shape_side(
        b_current,
        side=candidate.b_side,
        candidate=candidate,
        side_value=candidate.b_value,
        tol=_SEAM_TOL,
    )
    if a_shape is None or b_shape is None:
        return None
    before_cleanup_a = a_shape
    before_cleanup_b = b_shape
    before_matches: list[_EndpointMatch] = []
    a_shape, b_shape, vertices_changed = _true_up_path_vertices(
        a_shape, b_shape, tol=_SEAM_TOL, _matches=before_matches
    )
    before_stage = _endpoint_stage(
        intent_ref_by_key[a_key],
        intent_ref_by_key[b_key],
        before_cleanup_a,
        before_cleanup_b,
        before_matches,
    )
    # Inflections may describe intentional S-curves anywhere in the region.
    # Stitching projects shared coordinates only; curve refitting is a separate
    # operation and must not silently rewrite unrelated geometry.
    after_stage = None
    selected = "vertex_midpoint" if vertices_changed or candidate.axis == "vertices" else "midpoint"
    constraints.append(AxisSeamConstraint(
        left=intent_ref_by_key[a_key],
        right=intent_ref_by_key[b_key],
        axis=candidate.axis,
        left_side=candidate.a_side,
        right_side=candidate.b_side,
        left_value=candidate.a_value,
        right_value=candidate.b_value,
        seam=candidate.seam,
        span_min=candidate.span_min,
        span_max=candidate.span_max,
        before_cleanup=before_stage,
        after_cleanup=after_stage,
        selected=selected,
    ))
    updates_by_leaf[a_key] = _LeafUpdate(a_shape, int(b.id), candidate, selected)
    updates_by_leaf[b_key] = _LeafUpdate(b_shape, int(a.id), candidate, selected)
    return a_ref.reference.ancestry[0], b_ref.reference.ancestry[0]


def _owner_components(edges: list[tuple[int, int]]) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: dict[int, set[int]] = {}
    for value in parent:
        groups.setdefault(find(value), set()).add(value)
    return [groups[key] for key in sorted(groups)]


def _cluster_leaf_vertices(
    updates_by_leaf: dict[tuple[int, ...], _LeafUpdate],
    owner_ids: set[int],
    intent_ref_by_key: dict[tuple[int, ...], PassInputRegionRef],
    constraints: list[VertexClusterConstraint],
    clustered_refs: set[PassInputRegionRef],
    *,
    tol: float,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> None:
    keys = [
        key for key, update in updates_by_leaf.items()
        if (
            key[0] in owner_ids
            and update.shape.kind == "path"
            and update.candidate.axis != "pathops"
        )
    ]
    if len(keys) < 2:
        return

    tokens_by_key = {
        key: _parse_subpaths(str(updates_by_leaf[key].shape.params.get("d", "")))
        for key in keys
    }
    changed_keys: set[tuple[int, int]] = set()

    def apply_clusters(*, link_tol: float, min_keys: int, allow_same_key_links: bool) -> None:
        refs: list[_CoordinateRef] = []
        for key, tokens in tokens_by_key.items():
            refs.extend(_path_endpoint_refs(key, tokens))
        if len(refs) < 2:
            return

        parent = list(range(len(refs)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        # Only endpoints in neighboring ``link_tol`` cells can be joined.
        # The former all-pairs loop made a detailed traced surface quadratic in
        # its command count (and could exhaust memory before v0 was retained).
        # Looking only backward through the spatial buckets considers each pair
        # once while preserving the original strict distance predicate.
        buckets: dict[tuple[int, int], list[int]] = {}
        for i, a_ref in enumerate(refs):
            cell_x = math.floor(a_ref.x / link_tol)
            cell_y = math.floor(a_ref.y / link_tol)
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    for j in buckets.get((cell_x + dx_cell, cell_y + dy_cell), []):
                        b_ref = refs[j]
                        if not allow_same_key_links and a_ref.key == b_ref.key:
                            continue
                        dx = a_ref.x - b_ref.x
                        dy = a_ref.y - b_ref.y
                        if 0.0 < math.hypot(dx, dy) <= link_tol:
                            union(i, j)
            buckets.setdefault((cell_x, cell_y), []).append(i)

        groups: dict[int, list[_CoordinateRef]] = {}
        for index, ref in enumerate(refs):
            groups.setdefault(find(index), []).append(ref)

        shape_by_key = {
            key: _shape_from_tokens(updates_by_leaf[key].shape, tokens)
            for key, tokens in tokens_by_key.items()
        }
        assignments: list[EndpointAssignment] = []
        for group in groups.values():
            if len({ref.key for ref in group}) < min_keys:
                continue
            x = sum(ref.x for ref in group) / len(group)
            y = sum(ref.y for ref in group) / len(group)
            assignments.append(EndpointAssignment(
                members=tuple(
                    _endpoint_ref(intent_ref_by_key[ref.key], shape_by_key[ref.key], ref)
                    for ref in group
                ),
                point=(x, y),
            ))
            for ref in group:
                tokens = tokens_by_key[ref.key]
                values = tokens[ref.subpath_index][ref.token_index][1]
                values[ref.x_index] = x
                values[ref.y_index] = y
                changed_keys.add(ref.key)
        if assignments:
            constraints.append(VertexClusterConstraint(
                EndpointConstraintStage(tuple(assignments))
            ))

    apply_clusters(link_tol=tol, min_keys=2, allow_same_key_links=False)
    apply_clusters(link_tol=_JUNCTION_TOL, min_keys=3, allow_same_key_links=True)

    for key in changed_keys:
        update = updates_by_leaf[key]
        shape = _shape_from_tokens(update.shape, tokens_by_key[key])
        # Preserve the curve family while projecting shared coordinates.
        updates_by_leaf[key] = _LeafUpdate(
            shape,
            update.paired_with,
            update.candidate,
            "vertex_cluster",
        )
        clustered_refs.add(intent_ref_by_key[key])


def discover_seam_choices(
    nodes: tuple[SeamDetectionInput, ...], *, epsilon: float = 1.0,
    max_error: float = 1.0, cubic: bool = False, post_symmetry: bool = False,
    allow_underlap: bool = True,
) -> tuple[SeamDiscoveryChoice, ...]:
    """Discover ordered repairs from flat numeric records and ancestry only."""
    indexed = {_leaf_key(node): node for node in nodes}
    children: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for path in indexed:
        if len(path) > 1:
            children.setdefault(path[:-1], []).append(path)
    owners = sorted((node for path, node in indexed.items() if len(path) == 1),
        key=lambda node: (node.order[0], node.id))
    leaves = {path: node for path, node in indexed.items() if path not in children}
    intent_ref_by_key = {path: node.reference for path, node in leaves.items()}
    updates_by_leaf, owner_edges, pair_constraints = {}, [], []

    def between(a, b):
        if not _regions_close(a, b, tol=_SEAM_TOL):
            return []
        a_children, b_children = children.get(_leaf_key(a), []), children.get(_leaf_key(b), [])
        if not a_children and not b_children:
            return [(a, b)]
        pairs = []
        if a_children and b_children:
            for left in a_children:
                for right in b_children:
                    pairs.extend(between(indexed[left], indexed[right]))
        elif a_children:
            for left in a_children:
                pairs.extend(between(indexed[left], b))
        else:
            for right in b_children:
                pairs.extend(between(a, indexed[right]))
        return pairs

    def within(node):
        descendants = children.get(_leaf_key(node), [])
        pairs = []
        for i, left in enumerate(descendants):
            for right in descendants[i+1:]:
                pairs.extend(between(indexed[left], indexed[right]))
        for path in descendants:
            pairs.extend(within(indexed[path]))
        return pairs

    def record_pair(a, b):
        if not _same_surface_fill_group(a, b):
            return
        edge = _apply_leaf_pair(a, b, updates_by_leaf, indexed, intent_ref_by_key,
            pair_constraints, epsilon=epsilon, max_error=max_error, cubic=cubic,
            allow_underlap=allow_underlap)
        if edge is not None:
            owner_edges.append(edge)

    for owner in owners:
        symmetry = owner.diagnostics.get('symmetry')
        if not post_symmetry or isinstance(symmetry, Mapping) and symmetry.get('mode') == 'self':
            for a, b in within(owner):
                record_pair(a, b)
    if not post_symmetry:
        for i, owner in enumerate(owners):
            for other in owners[i+1:]:
                for a, b in between(owner, other):
                    record_pair(a, b)

    choices = []
    for owner_ids in _owner_components(owner_edges):
        cluster_constraints, clustered_refs = [], set()
        _cluster_leaf_vertices(updates_by_leaf, owner_ids, intent_ref_by_key,
            cluster_constraints, clustered_refs, tol=_SEAM_TOL,
            epsilon=epsilon, max_error=max_error, cubic=cubic)
        root_refs = tuple(PassInputRegionRef('', '', (identity,)) for identity in sorted(owner_ids))
        component_pairs = tuple(item for item in pair_constraints if item.left.ancestry[0] in owner_ids)
        pathops_rebuilt = {tuple(item.left.ancestry if item.bottom == 'left' else item.right.ancestry)
            for item in component_pairs if isinstance(item, PathOpsUnderlapConstraint)}
        updates = {path: update for path, update in updates_by_leaf.items() if path[0] in owner_ids}
        correspondence = tuple(item for path, update in sorted(updates.items())
            if leaves[path].shape is not None and leaves[path].shape.kind == 'path' and update.shape.kind == 'path'
            for item in _path_correspondence(intent_ref_by_key[path], leaves[path].shape, update.shape,
                positional_mutation=path not in pathops_rebuilt))
        action = ApplySeamConstraintsOperation(root_refs, component_pairs, tuple(cluster_constraints),
            tuple(sorted(clustered_refs, key=lambda ref: ref.ancestry)), float(epsilon),
            float(max_error), bool(cubic), correspondence)
        choices.append(SeamDiscoveryChoice(tuple(sorted(owner_ids)),
            operation_intent(action, root_refs, tuple(sorted(owner_ids))), updates))
    return tuple(choices)

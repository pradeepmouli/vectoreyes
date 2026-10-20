from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...fit import Shape
from ..automatic_replay import (
    ApplySeamConstraintsOperation,
    AxisSeamConstraint,
    EndpointAssignment,
    EndpointConstraintStage,
    PassInputRegionRef,
    PathOpsUnderlapConstraint,
    VertexClusterConstraint,
    operation_intent,
)
from ..framework import Proposal
from ...drawing_adapter import transform_working_shape, publish_seam_shape, replace_private_leaves
from ..symmetry_materialize import index_region_paths
from ..vector_region import VectorRegion
from ...path_sampling import _parse_subpaths
from ...drawing_simplification import simplify_path_data
from ...drawing_seams import (
    _SEAM_TOL, _JUNCTION_TOL, _SeamCandidate, SeamInput, _LeafUpdate,
    _CoordinateRef, _EndpointMatch, _path_endpoints_within,
    _pathops_shared_boundary_pair, _rewrite_shape_side,
    _true_up_path_vertices, _endpoint_stage, _cleanup_inflecting_cubics,
    _path_endpoint_refs, _shape_from_tokens, _endpoint_ref,
    _path_correspondence,
)


@dataclass(frozen=True)
class _LeafRef:
    owner: VectorRegion
    leaf: VectorRegion


def _leaf_key(ref: _LeafRef) -> tuple[int, int]:
    return int(ref.owner.id), int(ref.leaf.id)


def _paint_order(ref: _LeafRef) -> tuple[float, float, int, int]:
    return float(ref.owner.z), float(ref.leaf.z), int(ref.owner.id), int(ref.leaf.id)


def _seam_input(ref: _LeafRef, shape: Shape) -> SeamInput:
    return SeamInput(shape, int(ref.leaf.id), int(ref.owner.id), _paint_order(ref), ref.owner.footprint)


def _shape_bounds(obj: VectorRegion) -> tuple[float, float, float, float] | None:
    if obj.footprint is None or getattr(obj.footprint, "is_empty", False):
        return None
    minx, miny, maxx, maxy = obj.footprint.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _same_surface_fill_group(a: VectorRegion, b: VectorRegion) -> bool:
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
    if isinstance(a.diagnostics.get("compound"), dict) or isinstance(b.diagnostics.get("compound"), dict):
        return False
    a_fill = a.diagnostics.get("surface_fill")
    b_fill = b.diagnostics.get("surface_fill")
    if not isinstance(a_fill, dict) or not isinstance(b_fill, dict):
        return True
    return a_fill.get("group") == b_fill.get("group")


def _regions_close(a: VectorRegion, b: VectorRegion, *, tol: float) -> bool:
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
        a.current is not None
        and b.current is not None
        and a.current.kind == "path"
        and b.current.kind == "path"
    ):
        # Endpoint proximity is the actionable criterion for a path seam:
        # subsequent rewriting can move only those coordinates.  Avoid an
        # exact path-to-path distance here; it is costly for detailed,
        # overlapping traced paths and contributes no additional signal.
        return _path_endpoints_within(a.current, b.current, tol=tol)
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


def _candidate_for_pair(a: VectorRegion, b: VectorRegion, *, tol: float) -> _SeamCandidate | None:
    if a.current is None or b.current is None:
        return None
    if a.current.kind == "use" or b.current.kind == "use":
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
    if a.current.kind != "path" or b.current.kind != "path":
        return None
    return _SeamCandidate("vertices", "", "", 0.0, 0.0, 0.0, 0.0, 0.0)


def _cleanup_linelets(
    shape: Shape,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape:
    simplified = simplify_path_data(
        str(shape.params['d']),
        epsilon=epsilon,
        max_error=max_error,
        samples=16,
        cubic=cubic,
        linelet_only=True,
        fill_rule=shape.params.get('fill_rule'),
    )
    if simplified is None:
        return shape
    return Shape('path', {'d': simplified, **({'fill_rule': shape.params['fill_rule']}
        if 'fill_rule' in shape.params and len(_parse_subpaths(simplified)) > 1 else {})})


def _cleanup_mutated_path(
    shape: Shape,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape:
    cleaned = _cleanup_linelets(shape, epsilon=epsilon, max_error=max_error, cubic=cubic)
    return _cleanup_inflecting_cubics(cleaned, max_error=max_error, line_epsilon=epsilon)


def _leaf_pairs_between(
    a_owner: VectorRegion,
    a_region: VectorRegion,
    b_owner: VectorRegion,
    b_region: VectorRegion,
    *,
    tol: float,
) -> list[tuple[_LeafRef, _LeafRef]]:
    if not _regions_close(a_region, b_region, tol=tol):
        return []
    if a_region.is_leaf and b_region.is_leaf:
        return [(_LeafRef(owner=a_owner, leaf=a_region), _LeafRef(owner=b_owner, leaf=b_region))]

    pairs: list[tuple[_LeafRef, _LeafRef]] = []
    if a_region.is_branch and b_region.is_branch:
        for a_child in a_region.children:
            for b_child in b_region.children:
                pairs.extend(_leaf_pairs_between(a_owner, a_child, b_owner, b_child, tol=tol))
        return pairs

    if a_region.is_branch:
        for a_child in a_region.children:
            pairs.extend(_leaf_pairs_between(a_owner, a_child, b_owner, b_region, tol=tol))
        return pairs

    for b_child in b_region.children:
        pairs.extend(_leaf_pairs_between(a_owner, a_region, b_owner, b_child, tol=tol))
    return pairs


def _leaf_pairs_within(
    owner: VectorRegion,
    region: VectorRegion,
    *,
    tol: float,
) -> list[tuple[_LeafRef, _LeafRef]]:
    if region.is_leaf:
        return []

    pairs: list[tuple[_LeafRef, _LeafRef]] = []
    children = list(region.children)
    for index, a_child in enumerate(children):
        for b_child in children[index + 1:]:
            pairs.extend(_leaf_pairs_between(owner, a_child, owner, b_child, tol=tol))
    for child in children:
        pairs.extend(_leaf_pairs_within(owner, child, tol=tol))
    return pairs


def _apply_leaf_pair(
    a_ref: _LeafRef,
    b_ref: _LeafRef,
    updates_by_leaf: dict[tuple[int, int], _LeafUpdate],
    owner_leaf_maps: dict[int, dict[int, VectorRegion]],
    intent_ref_by_key: dict[tuple[int, int], PassInputRegionRef],
    constraints: list[AxisSeamConstraint | PathOpsUnderlapConstraint],
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
    allow_underlap: bool,
) -> tuple[int, int] | None:
    a = a_ref.leaf
    if a.current is None:
        return None
    b = b_ref.leaf
    if b.current is None:
        return None

    def current_shape(ref: _LeafRef) -> Shape | None:
        key = _leaf_key(ref)
        if key in updates_by_leaf:
            return updates_by_leaf[key].shape
        leaf = ref.leaf
        if leaf.current is None:
            return None
        if leaf.current.kind != "use":
            return leaf.current
        source_id = int(leaf.current.params.get("href_obj_id", -1))
        source = owner_leaf_maps.get(int(ref.owner.id), {}).get(source_id)
        if source is None or source.current is None:
            return None
        source_key = (int(ref.owner.id), source_id)
        source_shape = updates_by_leaf[source_key].shape if source_key in updates_by_leaf else source.current
        return transform_working_shape(source_shape, leaf.current.params["transform"])

    a_current = current_shape(a_ref)
    b_current = current_shape(b_ref)
    if a_current is None or b_current is None:
        return None
    a_for_candidate = a.with_current(a_current, footprint=a.footprint)
    b_for_candidate = b.with_current(b_current, footprint=b.footprint)
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
            return int(a_ref.owner.id), int(b_ref.owner.id)
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
    # Simplification has its own pass before stitch in the automatic workflow.
    # Do retain the local inflecting-cubic repair caused by moving an endpoint,
    # but do not rerun general linelet/path simplification for every pair.
    cleaned_a = _cleanup_inflecting_cubics(a_shape, max_error=max_error, line_epsilon=epsilon)
    cleaned_b = _cleanup_inflecting_cubics(b_shape, max_error=max_error, line_epsilon=epsilon)
    if cleaned_a != a_shape or cleaned_b != b_shape:
        a_shape, b_shape = cleaned_a, cleaned_b
        after_cleanup_a = a_shape
        after_cleanup_b = b_shape
        after_matches: list[_EndpointMatch] = []
        a_shape, b_shape, cleanup_vertices_changed = _true_up_path_vertices(
            a_shape, b_shape, tol=_SEAM_TOL, _matches=after_matches
        )
        vertices_changed = vertices_changed or cleanup_vertices_changed
        after_stage = _endpoint_stage(
            intent_ref_by_key[a_key],
            intent_ref_by_key[b_key],
            after_cleanup_a,
            after_cleanup_b,
            after_matches,
        )
    else:
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
    return int(a_ref.owner.id), int(b_ref.owner.id)


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
    updates_by_leaf: dict[tuple[int, int], _LeafUpdate],
    owner_ids: set[int],
    intent_ref_by_key: dict[tuple[int, int], PassInputRegionRef],
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
        # Stitching owns only shared boundary coordinates.  Running the
        # linelet simplifier here can rewrite unrelated outer geometry, which
        # makes the framework reject the entire seam proposal before its exact
        # endpoint projection is applied.  Preserve the local inflecting-cubic
        # guard, but leave all general simplification to its dedicated pass.
        shape = _cleanup_inflecting_cubics(shape, max_error=max_error, line_epsilon=epsilon)
        updates_by_leaf[key] = _LeafUpdate(
            shape,
            update.paired_with,
            update.candidate,
            "vertex_cluster",
        )
        clustered_refs.add(intent_ref_by_key[key])


def _replacement_for_update(ref: _LeafRef, update: _LeafUpdate) -> VectorRegion:
    return publish_seam_shape(
        ref.leaf,
        update.shape,
        update.paired_with,
        update.candidate,
        selected=update.selected,
    )


def seams_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.0,
    max_error: float = 1.0,
    cubic: bool = False,
    post_symmetry: bool = False,
    allow_underlap: bool = True,
) -> list[Proposal]:
    owners = sorted(objects, key=lambda current: (float(current.z), int(current.id)))
    owner_by_id = {int(owner.id): owner for owner in owners}
    owner_leaf_maps = {
        int(owner.id): {int(leaf.id): leaf for leaf in owner.leaves()}
        for owner in owners
    }
    leaf_ref_by_key: dict[tuple[int, int], _LeafRef] = {}
    intent_ref_by_key = {
        (int(path[0]), int(region.id)): PassInputRegionRef("", "", tuple(path))
        for path, region in index_region_paths(objects).items()
        if region.is_leaf
    }
    updates_by_leaf: dict[tuple[int, int], _LeafUpdate] = {}
    owner_edges: list[tuple[int, int]] = []
    pair_constraints: list[AxisSeamConstraint | PathOpsUnderlapConstraint] = []

    def record_pair(a_ref: _LeafRef, b_ref: _LeafRef) -> None:
        if not _same_surface_fill_group(a_ref.leaf, b_ref.leaf):
            return
        leaf_ref_by_key.setdefault(_leaf_key(a_ref), a_ref)
        leaf_ref_by_key.setdefault(_leaf_key(b_ref), b_ref)
        edge = _apply_leaf_pair(
            a_ref,
            b_ref,
            updates_by_leaf,
            owner_leaf_maps,
            intent_ref_by_key,
            pair_constraints,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
            allow_underlap=allow_underlap,
        )
        if edge is not None:
            owner_edges.append(edge)

    if post_symmetry:
        # Before refinement, stitch can repair every retained root.  After
        # symmetry, only a self-symmetry branch gained a new coincident axis;
        # revisiting unrelated compound roots is redundant and expensive.
        for owner in owners:
            symmetry = owner.diagnostics.get("symmetry")
            if isinstance(symmetry, dict) and symmetry.get("mode") == "self":
                for a_ref, b_ref in _leaf_pairs_within(owner, owner, tol=_SEAM_TOL):
                    record_pair(a_ref, b_ref)
    else:
        for owner in owners:
            for a_ref, b_ref in _leaf_pairs_within(owner, owner, tol=_SEAM_TOL):
                record_pair(a_ref, b_ref)

        for owner_index, a_owner in enumerate(owners):
            for b_owner in owners[owner_index + 1:]:
                leaf_pairs = _leaf_pairs_between(a_owner, a_owner, b_owner, b_owner, tol=_SEAM_TOL)
                for a_ref, b_ref in leaf_pairs:
                    record_pair(a_ref, b_ref)

    proposals: list[Proposal] = []
    for owner_ids in _owner_components(owner_edges):
        cluster_constraints: list[VertexClusterConstraint] = []
        clustered_refs: set[PassInputRegionRef] = set()
        _cluster_leaf_vertices(
            updates_by_leaf,
            owner_ids,
            intent_ref_by_key,
            cluster_constraints,
            clustered_refs,
            tol=_SEAM_TOL,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
        )
        replacements: list[VectorRegion] = []
        for owner_id in sorted(owner_ids):
            owner = owner_by_id[owner_id]
            leaf_replacements: dict[int, VectorRegion] = {}
            for key, update in updates_by_leaf.items():
                if key[0] != owner_id:
                    continue
                leaf_replacements[key[1]] = _replacement_for_update(leaf_ref_by_key[key], update)
            if not leaf_replacements:
                continue
            replacements.append(replace_private_leaves(owner, leaf_replacements))
        if replacements:
            # Endpoint projection can fold a tiny closing curve back across
            # its own contour or move a cutout across its exterior. Validate
            # the complete proposal with the same retained-topology contract
            # used by replay before recording any seam constraints. Keep the
            # original geometry when the proposed adjustment is unsafe; the
            # normalized result is a predicate, not replacement geometry.
            from ..topology_binding import normalize_working_topology

            try:
                normalize_working_topology(replacements, allow_external=True)
            except ValueError:
                continue
            root_refs = tuple(
                PassInputRegionRef("", "", (owner_id,))
                for owner_id in sorted(owner_ids)
            )
            component_pairs = tuple(
                constraint for constraint in pair_constraints
                if int(constraint.left.ancestry[0]) in owner_ids
            )
            pathops_rebuilt = {
                tuple(
                    constraint.left.ancestry
                    if constraint.bottom == "left"
                    else constraint.right.ancestry
                )
                for constraint in component_pairs
                if isinstance(constraint, PathOpsUnderlapConstraint)
            }
            path_correspondence = tuple(
                correspondence
                for key, update in sorted(updates_by_leaf.items())
                if key[0] in owner_ids
                and leaf_ref_by_key[key].leaf.current is not None
                and leaf_ref_by_key[key].leaf.current.kind == "path"
                and update.shape.kind == "path"
                for correspondence in _path_correspondence(
                    intent_ref_by_key[key],
                    leaf_ref_by_key[key].leaf.current,
                    update.shape,
                    positional_mutation=key not in pathops_rebuilt,
                )
            )
            action = ApplySeamConstraintsOperation(
                roots=root_refs,
                pairs=component_pairs,
                clusters=tuple(cluster_constraints),
                clustered=tuple(sorted(clustered_refs, key=lambda ref: ref.ancestry)),
                epsilon=float(epsilon),
                max_error=float(max_error),
                cubic=bool(cubic),
                path_correspondence=path_correspondence,
            )
            proposals.append(Proposal(
                tuple(sorted(owner_ids)),
                replacements,
                intent=operation_intent(action, root_refs, tuple(sorted(owner_ids))),
            ))
    return proposals

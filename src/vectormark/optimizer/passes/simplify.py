from __future__ import annotations


import numpy as np

from ...skia_geometry import SkPath, unary_union
from ...fit import Shape
from ...drawing_simplification import _path_subpaths, _simplified_path_shape
from ..automatic_replay import SimplifyLineletOperation, draft_input_ref, operation_intent
from ..framework import Proposal
from ..vector_region import VectorRegion, _parse_subpaths


def _simplifiable_polygon(flat: object) -> bool:
    return isinstance(flat, SkPath) and not flat.is_empty


def _covering_later_ids(
    obj: VectorRegion,
    candidate: Shape,
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
) -> list[int] | None:
    del candidate, masks
    if obj.current is None or obj.current.kind != "path":
        return None
    subpaths = _path_subpaths(obj.current, samples=24)
    if len(subpaths) <= 1:
        return []
    outer_index = max(range(len(subpaths)), key=lambda index: subpaths[index][2])
    dropped_polygons: list[SkPath] = []
    for index, (_tokens, points, _area) in enumerate(subpaths):
        if index == outer_index:
            continue
        poly = SkPath(shell=list(points))
        poly = poly.buffer(0)
        if not poly.is_empty:
            dropped_polygons.append(poly)
    if not dropped_polygons:
        return []
    dropped = unary_union(dropped_polygons)

    cover_ids: list[int] = []
    cover_geoms: list[SkPath] = []
    for other in sorted(objects, key=lambda item: (float(item.z), int(item.id))):
        if other.id == obj.id or other.z <= obj.z:
            continue
        try:
            if isinstance(other.footprint, SkPath):
                overlap_area = float(other.footprint.intersection(dropped).area)
            else:
                overlap_area = 0.0
        except Exception:
            overlap_area = 0.0
        if overlap_area > 1e-9:
            cover_ids.append(int(other.id))
            if isinstance(other.footprint, SkPath):
                cover_geoms.append(other.footprint)

    if not cover_geoms:
        return None
    covered = unary_union(cover_geoms)
    uncovered = dropped.difference(covered)
    if float(getattr(uncovered, "area", 0.0)) > 1e-6:
        return None
    return cover_ids


def _referenced_source_ids(objects: list[VectorRegion]) -> set[int]:
    referenced: set[int] = set()
    for obj in objects:
        if not obj.is_leaf or obj.current is None or obj.current.kind != "use":
            continue
        href_obj_id = obj.current.params.get("href_obj_id")
        if href_obj_id is None:
            continue
        referenced.add(int(href_obj_id))
    return referenced


def simplify_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    samples: int = 16,
    cubic: bool = False,
    linelet_only: bool = False,
    max_boundary_error: float | None = None,
) -> list[Proposal]:
    proposals: list[Proposal] = []
    referenced_source_ids = _referenced_source_ids(objects)
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        if int(obj.id) in referenced_source_ids:
            continue
        if not _simplifiable_polygon(obj.footprint):
            continue
        if linelet_only and obj.current.kind == "path":
            # A compound path's subpaths are topology (counters, islands, or
            # disconnected material fragments), not one run of linelets.
            # The boundary-residual check compares samples against every edge;
            # applying it across all subpaths is both semantically unsafe and
            # disproportionately expensive on a detailed trace.
            if len(_parse_subpaths(str(obj.current.params.get("d", "")))) > 1:
                continue
        if obj.current.kind == "path" and obj.current.params.get("fill_rule"):
            solid = _simplified_path_shape(
                obj.current,
                epsilon=epsilon,
                max_error=max_error,
                samples=samples,
                preserve_subpaths=False,
                cubic=cubic,
                original=obj.original,
                check_geometry=False,
                linelet_only=linelet_only,
                max_boundary_error=max_boundary_error,
            )
            if solid is not None:
                cover_ids = _covering_later_ids(obj, solid, objects, masks)
                if cover_ids is not None:
                    by_id = {int(current.id): current for current in objects}
                    proposal_ids = tuple([int(obj.id), *cover_ids])
                    new_objects = [obj.with_current(solid), *(by_id[obj_id] for obj_id in cover_ids)]
                    target_ref = draft_input_ref(obj.id)
                    cover_refs = tuple(draft_input_ref(obj_id) for obj_id in cover_ids)
                    action = SimplifyLineletOperation(
                        target_ref,
                        cover_refs,
                        epsilon,
                        max_error,
                        samples,
                        cubic,
                        linelet_only,
                        max_boundary_error,
                        False,
                        False,
                    )
                    proposals.append(Proposal(
                        proposal_ids,
                        new_objects,
                        intent=operation_intent(action, (target_ref, *cover_refs), proposal_ids),
                    ))
                    continue
        simplified = _simplified_path_shape(
            obj.current,
            epsilon=epsilon,
            max_error=max_error,
            samples=samples,
            cubic=cubic,
            original=obj.original,
            linelet_only=linelet_only,
            max_boundary_error=max_boundary_error,
        )
        if simplified is None:
            continue
        target_ref = draft_input_ref(obj.id)
        action = SimplifyLineletOperation(
            target_ref,
            (),
            epsilon,
            max_error,
            samples,
            cubic,
            linelet_only,
            max_boundary_error,
            True,
            True,
        )
        proposals.append(Proposal(
            (obj.id,),
            [obj.with_current(simplified)],
            intent=operation_intent(action, (target_ref,), (obj.id,)),
        ))

    return proposals

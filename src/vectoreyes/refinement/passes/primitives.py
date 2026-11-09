from __future__ import annotations

import numpy as np

from ...skia_geometry import SkPath
from ...drawing import DrawingRegion
from ...drawing_adapter import publish_selected_leaf
from ...primitive_fitting import recognize_primitive_geometry
from ...quality import GeometryPolicy, assess_geometry
from ..framework import Proposal
from ..automatic_replay import primitive_intent
from ..region_sample import RegionSample


def _residual_area(original: SkPath, candidate: SkPath) -> float:
    """Area occupied by one geometry but not the other."""
    try:
        return float(original.symmetric_difference(candidate).area)
    except Exception:
        return float("inf")


def _within_boundary_residual_budget(
    original: SkPath,
    candidate: SkPath,
    *,
    epsilon: float,
) -> bool:
    """Accept only residual area explainable by an ``epsilon``-wide boundary.

    Dividing residual area by the filled area makes a large icon overly
    permissive: a visibly different rounded rectangle can still be only a tiny
    percentage of its interior.  The area of an ``epsilon``-wide perimeter
    corridor instead grows with the geometry that can actually move.
    """
    return assess_geometry(
        original,
        candidate,
        GeometryPolicy(max_perimeter_residual_error_px=epsilon, preserve_topology=False),
    ).accepted


def primitives_pass(
    objects: list[RegionSample],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
) -> list[Proposal]:
    del masks

    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        primitive = recognize_primitive_geometry(obj.footprint, epsilon=epsilon)
        if primitive is None:
            continue
        selected = DrawingRegion(id=obj.drawing_id or f'automatic-{obj.id}',
            geometry=primitive, diagnostics=obj.diagnostics)
        replacement = publish_selected_leaf(selected, obj)
        if replacement.current == obj.current:
            continue
        policy = GeometryPolicy(
            max_boundary_error_px=epsilon,
            max_perimeter_residual_error_px=epsilon,
            preserve_topology=True,
        )
        proposals.append(Proposal(
            (obj.id,),
            [replacement],
            policy,
            intent=primitive_intent(obj.id, primitive, epsilon=epsilon),
        ))

    return proposals

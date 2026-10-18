from __future__ import annotations

import numpy as np

from ...skia_geometry import SkPath
from ...fit import Shape, recognize_primitive
from ...quality import GeometryPolicy, assess_geometry
from ..framework import Proposal
from ..automatic_replay import primitive_intent
from ..vector_region import VectorRegion, to_polygon

def _polygon_exterior_points(flat: object) -> tuple[np.ndarray, SkPath] | None:
    if not isinstance(flat, SkPath) or flat.is_empty:
        return None

    geoms = flat.geoms
    if len(geoms) != 1:
        return None
    poly = geoms[0]
    if poly.interiors:
        return None

    coords = np.asarray(poly.exterior.coords, dtype=float)
    if len(coords) < 4:
        return None
    return coords, flat


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


def _recognize_rounded_rect(points: np.ndarray, flat: SkPath, *, epsilon: float):
    """Recognize an axis-aligned rounded rectangle independently of its fill."""
    # Keep the radius estimator shared with path simplification so native
    # primitive fitting and path fallback make the same geometric judgment.
    from ...drawing_simplification import _rounded_rect_radius

    fit = _rounded_rect_radius(points, flat, epsilon)
    if fit is None:
        return None
    radius, (x0, y0, x1, y1) = fit
    return Shape("rect", {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "rx": radius, "ry": radius})


def primitives_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.5,
) -> list[Proposal]:
    del masks

    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        polygonal = _polygon_exterior_points(obj.footprint)
        if polygonal is None:
            continue
        points, fit_geometry = polygonal

        primitive = recognize_primitive(points, epsilon=epsilon)
        if primitive is None:
            primitive = _recognize_rounded_rect(points, fit_geometry, epsilon=epsilon)
        if primitive is None or primitive == obj.current:
            continue
        policy = GeometryPolicy(
            max_boundary_error_px=epsilon,
            max_perimeter_residual_error_px=epsilon,
            preserve_topology=True,
        )
        assessment = assess_geometry(fit_geometry, to_polygon(primitive), policy)
        proposals.append(Proposal(
            (obj.id,),
            [obj.with_current(primitive)],
            policy,
            intent=primitive_intent(obj.id, primitive, epsilon=epsilon),
        ))

    return proposals

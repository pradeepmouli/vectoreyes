from __future__ import annotations

import math

import numpy as np

from ...skia_geometry import SkPath
from ...drawing_adapter import bind_region_inputs
from ..automatic_replay import AssignCloneOperation, draft_input_ref, operation_intent
from ..framework import Proposal
from ..region_sample import RegionSample

_ANGLE_EPS = 1e-9
_AREA_RATIO_TOL = 0.01
_PERIMETER_RATIO_TOL = 0.01
_GEOM_RESIDUAL_TOL = 0.01
_REFLECTION_RESIDUAL_TOL = 0.02
_FAMILY_RATIO_TOL = 0.05
_FAMILY_RESIDUAL_TOL = 0.05
_MIN_FAMILY_SIZE = 3


def _polygonal_flat(flat: object) -> SkPath | None:
    if isinstance(flat, SkPath) and not flat.is_empty:
        return flat
    return None


def _has_self_symmetry(region: RegionSample) -> bool:
    symmetry = region.diagnostics.get("symmetry")
    return isinstance(symmetry, dict) and symmetry.get("mode") == "self"


def _shape_descriptor(flat: SkPath) -> tuple[float, float, int]:
    components = tuple(getattr(flat, "geoms", ()) or (flat,))
    # Ignore detached trace specks when comparing topology. Their area remains
    # part of the footprint residual, but a sub-pixel artifact must not make an
    # otherwise single-component family fail the descriptor prefilter.
    minimum_component_area = max(1.0, float(flat.area) * 0.005)
    meaningful_components = sum(
        float(component.area) >= minimum_component_area
        for component in components
    )
    return (float(flat.area), float(flat.length), max(meaningful_components, 1))


def _within_ratio(a: float, b: float, tol: float) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale <= tol


def _normalize_angle(theta: float) -> float:
    out = math.fmod(theta + math.pi, 2.0 * math.pi)
    if out < 0.0:
        out += 2.0 * math.pi
    out -= math.pi
    if abs(out) < _ANGLE_EPS:
        return 0.0
    return out


def _orientation_candidates(flat: SkPath) -> list[float]:
    rect = flat.minimum_rotated_rectangle
    if rect.is_empty:
        return [0.0]

    coords = list(rect.exterior.coords)
    angles: list[float] = []
    for (x1, y1), (x2, y2) in zip(coords, coords[1:], strict=False):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < _ANGLE_EPS and abs(dy) < _ANGLE_EPS:
            continue
        angle = _normalize_angle(math.atan2(dy, dx))
        if all(abs(_normalize_angle(angle - existing)) > 1e-6 for existing in angles):
            angles.append(angle)
    return angles or [0.0]


def _matrix_for_rotation_and_centroids(
    canonical_centroid: tuple[float, float],
    target_centroid: tuple[float, float],
    theta: float,
) -> tuple[float, float, float, float, float, float]:
    a = math.cos(theta)
    b = math.sin(theta)
    c = -math.sin(theta)
    d = math.cos(theta)
    cx, cy = canonical_centroid
    tx, ty = target_centroid
    e = tx - (a * cx + c * cy)
    f = ty - (b * cx + d * cy)
    vals = (a, b, c, d, e, f)
    return tuple(0.0 if abs(v) < 1e-12 else float(v) for v in vals)


def _best_transform(
    canonical: SkPath,
    target: SkPath,
    *,
    geom_tolerance: float = _GEOM_RESIDUAL_TOL,
    reflection_tolerance: float = _REFLECTION_RESIDUAL_TOL,
) -> tuple[tuple[float, float, float, float, float, float], SkPath] | None:
    canonical_centroid = (float(canonical.centroid.x), float(canonical.centroid.y))
    target_centroid = (float(target.centroid.x), float(target.centroid.y))
    canonical_angles = _orientation_candidates(canonical)
    target_angles = _orientation_candidates(target)

    candidates: list[tuple[float, int, tuple[float, float, float, float, float, float], SkPath]] = []
    for canon_angle in canonical_angles:
        for target_angle in target_angles:
            theta = _normalize_angle(target_angle - canon_angle)
            matrix = _matrix_for_rotation_and_centroids(canonical_centroid, target_centroid, theta)
            transformed = canonical.affine_transform(matrix)
            scale = max(float(target.area), float(transformed.area), 1.0)
            residual = float(transformed.symmetric_difference(target).area / scale)
            candidates.append((residual, 0, matrix, transformed))

    dx = target_centroid[0] - canonical_centroid[0]
    dy = target_centroid[1] - canonical_centroid[1]
    if abs(dx) > _ANGLE_EPS or abs(dy) > _ANGLE_EPS:
        # Reflection across the line through the centroid midpoint whose tangent
        # is perpendicular to the source->target direction maps one centroid to
        # the other. This covers mirrored siblings such as the Daikonic leaves.
        theta = math.atan2(dy, dx) + math.pi / 2.0
        ux, uy = math.cos(theta), math.sin(theta)
        a, b, c, d = 2 * ux * ux - 1, 2 * ux * uy, 2 * ux * uy, 2 * uy * uy - 1
        mx = (canonical_centroid[0] + target_centroid[0]) / 2.0
        my = (canonical_centroid[1] + target_centroid[1]) / 2.0
        e = mx - (a * mx + c * my)
        f = my - (b * mx + d * my)
        matrix = tuple(0.0 if abs(value) < 1e-12 else float(value) for value in (a, b, c, d, e, f))
        transformed = canonical.affine_transform(matrix)
        scale = max(float(target.area), float(transformed.area), 1.0)
        candidates.append((float(transformed.symmetric_difference(target).area / scale), 1, matrix, transformed))

    if not candidates:
        return None

    residual, _reflection_rank, matrix, transformed = min(
        candidates,
        key=lambda item: (
            round(item[0], 12),
            item[1],
            abs(_normalize_angle(math.atan2(item[2][1], item[2][0]))),
            tuple(round(v, 12) for v in item[2]),
        ),
    )
    determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2]
    tolerance = reflection_tolerance if determinant < 0.0 else geom_tolerance
    if residual > tolerance:
        return None
    return matrix, transformed


def _clone_proposal(
    canonical: RegionSample,
    target: RegionSample,
    *,
    matrix: tuple[float, float, float, float, float, float],
    transformed: SkPath,
    symbolic: bool,
    family_size: int | None = None,
) -> Proposal:
    from ..automatic_replay import apply_relationship_choice
    source_ref = draft_input_ref(canonical.id)
    target_ref = draft_input_ref(target.id)
    action = AssignCloneOperation(source_ref, target_ref, matrix, symbolic, family_size)
    replacement = apply_relationship_choice(action,
        bind_region_inputs({'source': canonical, 'target': target}), footprint=transformed)[0]
    return Proposal(
        (target.id,),
        [replacement],
        intent=operation_intent(action, (source_ref, target_ref), (target.id,)),
    )


def clones_pass(
    objects: list[RegionSample],
    masks: dict[int, np.ndarray],
    *,
    symbolic: bool = True,
) -> list[Proposal]:
    """Find congruent regions; interactive callers may retain symbolic clones."""
    del masks
    usable: list[tuple[RegionSample, SkPath, tuple[float, float, int]]] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        flat = _polygonal_flat(obj.footprint)
        if flat is None or obj.fill is None or obj.current.kind == "use" or _has_self_symmetry(obj):
            continue
        usable.append((obj, flat, _shape_descriptor(flat)))

    proposals: list[Proposal] = []
    queued_targets: set[int] = set()

    # A repeated family carries stronger semantic evidence than one approximate
    # pair. Raster overlap and seam slicing can perturb otherwise congruent
    # members by several percent, so admit the looser tolerance only when one
    # canonical agrees with at least two targets.
    for index, (canonical_obj, canonical_flat, canonical_desc) in enumerate(usable):
        if canonical_obj.id in queued_targets:
            continue
        matches: list[
            tuple[
                RegionSample,
                tuple[float, float, float, float, float, float],
                SkPath,
            ]
        ] = []
        for target_obj, target_flat, target_desc in usable[index + 1:]:
            if target_obj.id in queued_targets:
                continue
            if canonical_desc[2] != target_desc[2]:
                continue
            if not _within_ratio(canonical_desc[0], target_desc[0], _FAMILY_RATIO_TOL):
                continue
            if not _within_ratio(canonical_desc[1], target_desc[1], _FAMILY_RATIO_TOL):
                continue
            best = _best_transform(
                canonical_flat,
                target_flat,
                geom_tolerance=_FAMILY_RESIDUAL_TOL,
                reflection_tolerance=_FAMILY_RESIDUAL_TOL,
            )
            if best is None:
                continue
            matrix, transformed = best
            matches.append((target_obj, matrix, transformed))

        family_size = len(matches) + 1
        if family_size < _MIN_FAMILY_SIZE:
            continue
        for target_obj, matrix, transformed in matches:
            proposals.append(
                _clone_proposal(
                    canonical_obj,
                    target_obj,
                    matrix=matrix,
                    transformed=transformed,
                    symbolic=symbolic,
                    family_size=family_size,
                )
            )
            queued_targets.add(int(target_obj.id))

    for index, (target_obj, target_flat, target_desc) in enumerate(usable):
        if target_obj.id in queued_targets:
            continue
        matched = False
        for canonical_obj, canonical_flat, canonical_desc in usable[:index]:
            if canonical_obj.id in queued_targets:
                continue
            if canonical_desc[2] != target_desc[2]:
                continue
            if not _within_ratio(canonical_desc[0], target_desc[0], _AREA_RATIO_TOL):
                continue
            if not _within_ratio(canonical_desc[1], target_desc[1], _PERIMETER_RATIO_TOL):
                continue

            best = _best_transform(canonical_flat, target_flat)
            if best is None:
                continue
            matrix, transformed = best

            proposals.append(
                _clone_proposal(
                    canonical_obj,
                    target_obj,
                    matrix=matrix,
                    transformed=transformed,
                    symbolic=symbolic,
                )
            )
            queued_targets.add(int(target_obj.id))
            matched = True
            break
        if matched:
            continue

    return proposals

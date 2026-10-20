"""Geometry-only congruence matching shared by automatic discovery routes."""

from dataclasses import dataclass
import math

from .skia_geometry import SkPath

_ANGLE_EPS = 1e-9
_AREA_RATIO_TOL = 0.01
_PERIMETER_RATIO_TOL = 0.01
_GEOM_RESIDUAL_TOL = 0.01
_REFLECTION_RESIDUAL_TOL = 0.02
_FAMILY_RATIO_TOL = 0.05
_FAMILY_RESIDUAL_TOL = 0.05
_MIN_FAMILY_SIZE = 3


@dataclass(frozen=True)
class CloneDetectionInput:
    id: int
    geometry: SkPath


@dataclass(frozen=True)
class CloneMatch:
    source_id: int
    target_id: int
    matrix: tuple[float, float, float, float, float, float]
    transformed: SkPath
    family_size: int | None = None


def _shape_descriptor(flat: SkPath) -> tuple[float, float, int]:
    components = tuple(getattr(flat, 'geoms', ()) or (flat,))
    minimum_component_area = max(1.0, float(flat.area) * 0.005)
    meaningful_components = sum(float(component.area) >= minimum_component_area for component in components)
    return float(flat.area), float(flat.length), max(meaningful_components, 1)


def _within_ratio(a: float, b: float, tol: float) -> bool:
    return abs(a - b) / max(abs(a), abs(b), 1.0) <= tol


def _normalize_angle(theta: float) -> float:
    out = math.fmod(theta + math.pi, 2.0 * math.pi)
    if out < 0.0:
        out += 2.0 * math.pi
    out -= math.pi
    return 0.0 if abs(out) < _ANGLE_EPS else out


def _orientation_candidates(flat: SkPath) -> list[float]:
    rect = flat.minimum_rotated_rectangle
    if rect.is_empty:
        return [0.0]
    coords = list(rect.exterior.coords)
    angles: list[float] = []
    for (x1, y1), (x2, y2) in zip(coords, coords[1:], strict=False):
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dx) < _ANGLE_EPS and abs(dy) < _ANGLE_EPS:
            continue
        angle = _normalize_angle(math.atan2(dy, dx))
        if all(abs(_normalize_angle(angle - existing)) > 1e-6 for existing in angles):
            angles.append(angle)
    return angles or [0.0]


def _matrix_for_rotation_and_centroids(
    canonical_centroid: tuple[float, float], target_centroid: tuple[float, float], theta: float,
) -> tuple[float, float, float, float, float, float]:
    a, b = math.cos(theta), math.sin(theta)
    c, d = -math.sin(theta), math.cos(theta)
    cx, cy = canonical_centroid
    tx, ty = target_centroid
    vals = (a, b, c, d, tx - (a * cx + c * cy), ty - (b * cx + d * cy))
    return tuple(0.0 if abs(v) < 1e-12 else float(v) for v in vals)


def _best_transform(
    canonical: SkPath, target: SkPath, *, geom_tolerance: float = _GEOM_RESIDUAL_TOL,
    reflection_tolerance: float = _REFLECTION_RESIDUAL_TOL,
) -> tuple[tuple[float, float, float, float, float, float], SkPath] | None:
    canonical_centroid = float(canonical.centroid.x), float(canonical.centroid.y)
    target_centroid = float(target.centroid.x), float(target.centroid.y)
    canonical_angles, target_angles = _orientation_candidates(canonical), _orientation_candidates(target)
    candidates = []
    for canon_angle in canonical_angles:
        for target_angle in target_angles:
            theta = _normalize_angle(target_angle - canon_angle)
            matrix = _matrix_for_rotation_and_centroids(canonical_centroid, target_centroid, theta)
            transformed = canonical.affine_transform(matrix)
            scale = max(float(target.area), float(transformed.area), 1.0)
            candidates.append((float(transformed.symmetric_difference(target).area / scale), 0, matrix, transformed))
    dx, dy = target_centroid[0] - canonical_centroid[0], target_centroid[1] - canonical_centroid[1]
    if abs(dx) > _ANGLE_EPS or abs(dy) > _ANGLE_EPS:
        theta = math.atan2(dy, dx) + math.pi / 2.0
        ux, uy = math.cos(theta), math.sin(theta)
        a, b, c, d = 2 * ux * ux - 1, 2 * ux * uy, 2 * ux * uy, 2 * uy * uy - 1
        mx, my = (canonical_centroid[0] + target_centroid[0]) / 2.0, (canonical_centroid[1] + target_centroid[1]) / 2.0
        e, f = mx - (a * mx + c * my), my - (b * mx + d * my)
        matrix = tuple(0.0 if abs(value) < 1e-12 else float(value) for value in (a, b, c, d, e, f))
        transformed = canonical.affine_transform(matrix)
        scale = max(float(target.area), float(transformed.area), 1.0)
        candidates.append((float(transformed.symmetric_difference(target).area / scale), 1, matrix, transformed))
    if not candidates:
        return None
    residual, _rank, matrix, transformed = min(candidates, key=lambda item: (
        round(item[0], 12), item[1], abs(_normalize_angle(math.atan2(item[2][1], item[2][0]))),
        tuple(round(v, 12) for v in item[2])))
    determinant = matrix[0] * matrix[3] - matrix[1] * matrix[2]
    if residual > (reflection_tolerance if determinant < 0.0 else geom_tolerance):
        return None
    return matrix, transformed


def find_clone_matches(inputs: tuple[CloneDetectionInput, ...]) -> tuple[CloneMatch, ...]:
    """Select non-chained families, then strict pairs, using existing tolerances."""
    usable = [(item.id, item.geometry, _shape_descriptor(item.geometry))
        for item in sorted(inputs, key=lambda item: item.id)
        if isinstance(item.geometry, SkPath) and not item.geometry.is_empty]
    proposals: list[CloneMatch] = []
    queued_targets: set[int] = set()
    family_sources: set[int] = set()
    for index, (source_id, source_flat, source_desc) in enumerate(usable):
        if source_id in queued_targets:
            continue
        matches = []
        for target_id, target_flat, target_desc in usable[index + 1:]:
            if target_id in queued_targets or source_desc[2] != target_desc[2]:
                continue
            if not _within_ratio(source_desc[0], target_desc[0], _FAMILY_RATIO_TOL):
                continue
            if not _within_ratio(source_desc[1], target_desc[1], _FAMILY_RATIO_TOL):
                continue
            best = _best_transform(source_flat, target_flat,
                geom_tolerance=_FAMILY_RESIDUAL_TOL, reflection_tolerance=_FAMILY_RESIDUAL_TOL)
            if best is not None:
                matches.append((target_id, *best))
        family_size = len(matches) + 1
        if family_size < _MIN_FAMILY_SIZE:
            continue
        family_sources.add(source_id)
        for target_id, matrix, transformed in matches:
            proposals.append(CloneMatch(source_id, target_id, matrix, transformed, family_size))
            queued_targets.add(target_id)
    for index, (target_id, target_flat, target_desc) in enumerate(usable):
        if target_id in queued_targets or target_id in family_sources:
            continue
        for source_id, source_flat, source_desc in usable[:index]:
            if source_id in queued_targets or source_desc[2] != target_desc[2]:
                continue
            if not _within_ratio(source_desc[0], target_desc[0], _AREA_RATIO_TOL):
                continue
            if not _within_ratio(source_desc[1], target_desc[1], _PERIMETER_RATIO_TOL):
                continue
            best = _best_transform(source_flat, target_flat)
            if best is None:
                continue
            proposals.append(CloneMatch(source_id, target_id, *best))
            queued_targets.add(target_id)
            break
    return tuple(proposals)

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .skia_geometry import SkPath


@dataclass(frozen=True)
class GeometryPolicy:
    """Explicit, conjunctive budgets for comparing two filled geometries."""

    max_relative_area_error: float | None = None
    max_boundary_error_px: float | None = None
    max_perimeter_residual_error_px: float | None = None
    preserve_topology: bool = True


@dataclass(frozen=True)
class GeometryAssessment:
    policy: GeometryPolicy
    accepted: bool
    symmetric_difference_area: float
    relative_area_error: float
    max_boundary_error_px: float
    source_components: int
    replacement_components: int
    source_holes: int
    replacement_holes: int
    reasons: tuple[str, ...]
    boundary_measurement: str = (
        "SkPath-linearized polylines; all vertices plus adaptive segment samples; "
        "conservative upper bound within 0.05 px of polyline Hausdorff distance"
    )

    def diagnostics(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "policy": {
                "max_relative_area_error": self.policy.max_relative_area_error,
                "max_boundary_error_px": self.policy.max_boundary_error_px,
                "max_perimeter_residual_error_px": self.policy.max_perimeter_residual_error_px,
                "preserve_topology": self.policy.preserve_topology,
            },
            "symmetric_difference_area": self.symmetric_difference_area,
            "relative_area_error": self.relative_area_error,
            "max_boundary_error_px": self.max_boundary_error_px,
            "topology": {
                "source_components": self.source_components,
                "replacement_components": self.replacement_components,
                "source_holes": self.source_holes,
                "replacement_holes": self.replacement_holes,
            },
            "reasons": list(self.reasons),
            "boundary_measurement": self.boundary_measurement,
        }


def _valid_limit(value: float | None) -> bool:
    return value is None or (math.isfinite(float(value)) and float(value) >= 0.0)


def _topology(path: SkPath) -> tuple[int, int]:
    return len(path.geoms), sum(len(part.interiors) for part in path.geoms)


def symmetric_boundary_distance(
    source: SkPath,
    replacement: SkPath,
    *,
    precision: float = 0.05,
    max_samples_per_ring: int | None = None,
) -> float:
    """Measure closed SkPath boundaries with the shared polyline metric."""
    try:
        if not math.isfinite(precision) or precision <= 0.0:
            return float("inf")
        if source.symmetric_difference(replacement).area == 0.0:
            return 0.0
        return polyline_boundary_distance(
            source.linearized_subpaths, replacement.linearized_subpaths,
            precision=precision, max_samples_per_ring=max_samples_per_ring,
        )
    except Exception:
        return float("inf")


def polyline_boundary_distance(
    source_rings: Sequence[Sequence[tuple[float, float]]],
    replacement_rings: Sequence[Sequence[tuple[float, float]]],
    *,
    closed: bool = True,
    precision: float = 0.05,
    max_samples_per_ring: int | None = None,
    direction: Literal["both", "source_to_replacement", "replacement_to_source"] = "both",
) -> float:
    """Bound directed or symmetric polyline distance in pixel units.

    The default samples every linearized vertex and subdivides every segment
    until the 1-Lipschitz distance-to-boundary function can vary by at most
    ``precision`` between a sample and any intervening point.  The returned
    value adds that bound, so acceptance cannot hide an unsampled polyline
    defect.  ``max_samples_per_ring`` retains the refinement simplifier's older
    bounded vertex-only approximation when explicitly requested.
    """
    try:
        if not math.isfinite(precision) or precision <= 0.0:
            return float("inf")
        if direction not in {"both", "source_to_replacement", "replacement_to_source"}:
            return float("inf")
        def edges(rings):
            return [
                (point, ring[(index + 1) % len(ring)])
                for ring in rings if len(ring) >= 2
                for index, point in enumerate(ring if closed else ring[:-1])
            ]

        def point_distance(point, candidates):
            p = np.asarray(point, dtype=float)
            if not np.all(np.isfinite(p)) or not candidates:
                return float("inf")
            starts = np.asarray([start for start, _ in candidates], dtype=float)
            deltas = np.asarray([np.asarray(end) - np.asarray(start) for start, end in candidates])
            denominators = np.einsum("ij,ij->i", deltas, deltas)
            numerators = np.einsum("ij,ij->i", p - starts, deltas)
            t = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators != 0)
            projections = starts + np.clip(t, 0.0, 1.0)[:, None] * deltas
            return float(np.min(np.linalg.norm(p - projections, axis=1)))

        def legacy_samples(rings):
            if max_samples_per_ring is None or max_samples_per_ring <= 0:
                return []
            return [
                point
                for ring in rings if ring
                for point in ring[::max(1, math.ceil(len(ring) / max_samples_per_ring))]
            ]

        def directed_bound(rings, candidates):
            vertices = [point for ring in rings for point in ring]
            if not vertices:
                return float("inf")
            distances = {tuple(point): point_distance(point, candidates) for point in vertices}
            lower = max(distances.values())
            intervals = []
            serial = 0
            for ring in rings:
                for index, start in enumerate(ring if closed else ring[:-1]):
                    end = ring[(index + 1) % len(ring)]
                    length = math.dist(start, end)
                    upper = (distances[tuple(start)] + distances[tuple(end)] + length) / 2.0
                    heapq.heappush(intervals, (-upper, serial, start, end,
                                               distances[tuple(start)], distances[tuple(end)]))
                    serial += 1
            while intervals and -intervals[0][0] > lower + precision:
                _neg_upper, _serial, start, end, start_distance, end_distance = heapq.heappop(intervals)
                midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
                midpoint_distance = point_distance(midpoint, candidates)
                lower = max(lower, midpoint_distance)
                for left, right, left_distance, right_distance in (
                    (start, midpoint, start_distance, midpoint_distance),
                    (midpoint, end, midpoint_distance, end_distance),
                ):
                    upper = (left_distance + right_distance + math.dist(left, right)) / 2.0
                    heapq.heappush(intervals, (-upper, serial, left, right, left_distance, right_distance))
                    serial += 1
            return max(lower, -intervals[0][0] if intervals else lower)

        source_edges, replacement_edges = edges(source_rings), edges(replacement_rings)
        if not source_edges or not replacement_edges:
            return float("inf")
        if max_samples_per_ring is not None:
            source_samples = legacy_samples(source_rings)
            replacement_samples = legacy_samples(replacement_rings)
            if not source_samples or not replacement_samples:
                return float("inf")
            forward = max(point_distance(point, replacement_edges) for point in source_samples)
            reverse = max(point_distance(point, source_edges) for point in replacement_samples)
        else:
            forward = directed_bound(source_rings, replacement_edges) if direction != "replacement_to_source" else 0.0
            reverse = directed_bound(replacement_rings, source_edges) if direction != "source_to_replacement" else 0.0
        if direction == "source_to_replacement":
            return forward
        if direction == "replacement_to_source":
            return reverse
        return max(forward, reverse)
    except Exception:
        return float("inf")


def assess_geometry(source: SkPath, replacement: SkPath, policy: GeometryPolicy) -> GeometryAssessment:
    """Measure *source* and *replacement* once and apply every declared limit."""
    reasons: list[str] = []
    limits_valid = all(_valid_limit(value) for value in (
        policy.max_relative_area_error,
        policy.max_boundary_error_px,
        policy.max_perimeter_residual_error_px,
    ))
    try:
        source_area = float(source.area)
        replacement_area = float(replacement.area)
        source_length = float(source.length)
        replacement_length = float(replacement.length)
        residual_area = float(source.symmetric_difference(replacement).area)
        source_components, source_holes = _topology(source)
        replacement_components, replacement_holes = _topology(replacement)
    except Exception:
        source_area = replacement_area = source_length = replacement_length = residual_area = float("inf")
        source_components = source_holes = replacement_components = replacement_holes = 0
    finite = all(math.isfinite(value) for value in (
        source_area, replacement_area, source_length, replacement_length, residual_area
    )) and source_area > 0.0 and replacement_area > 0.0
    relative = residual_area / max(source_area, 1.0) if finite else float("inf")
    boundary = symmetric_boundary_distance(source, replacement) if finite else float("inf")
    if not limits_valid:
        reasons.append("invalid_tolerance")
    if not finite or not math.isfinite(boundary):
        reasons.append("invalid_geometry")
    if policy.preserve_topology and (source_components, source_holes) != (replacement_components, replacement_holes):
        reasons.append("topology_changed")
    if _valid_limit(policy.max_relative_area_error) and policy.max_relative_area_error is not None and relative > policy.max_relative_area_error:
        reasons.append("relative_area_error_exceeded")
    if _valid_limit(policy.max_boundary_error_px) and policy.max_boundary_error_px is not None and boundary > policy.max_boundary_error_px:
        reasons.append("boundary_error_exceeded")
    if _valid_limit(policy.max_perimeter_residual_error_px) and policy.max_perimeter_residual_error_px is not None:
        corridor = max(source_length, replacement_length, 1.0) * policy.max_perimeter_residual_error_px
        if residual_area > corridor:
            reasons.append("perimeter_residual_error_exceeded")
    return GeometryAssessment(
        policy=policy, accepted=not reasons, symmetric_difference_area=residual_area,
        relative_area_error=relative, max_boundary_error_px=boundary,
        source_components=source_components, replacement_components=replacement_components,
        source_holes=source_holes, replacement_holes=replacement_holes, reasons=tuple(reasons),
    )

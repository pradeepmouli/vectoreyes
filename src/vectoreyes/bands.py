"""Non-mutating constant-width proposals for two explicitly scoped source spans.

Callers supply original, corner-excluded material polylines in one coordinate
system. These kernels neither guess path ownership nor publish path edits.
Offsets are evaluated from one cubic centerline; they are not independently
fitted cubics (a cubic's exact offset is generally not a cubic).
"""

from dataclasses import dataclass
import math

import numpy as np

from .quality import polyline_boundary_distance
from .band_offsets import BandOffsets, approximate_band_offsets, offset_polyline, exact_polyline_vertices

Point = tuple[float, float]


@dataclass(frozen=True)
class BandRequest:
    first: tuple[Point, ...]
    second: tuple[Point, ...]
    width_tolerance: float = 1.5
    tangent_tolerance_degrees: float = 10.0
    tangent_support_fraction: float = 0.02
    min_support: float = 32.0
    max_error: float = 1.5
    samples: int = 129

    def __post_init__(self) -> None:
        for name in ("first", "second"):
            points = np.asarray(getattr(self, name), dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError("band spans require at least three finite 2D points")
            if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-9):
                raise ValueError("band spans must not contain zero-length edges")
            object.__setattr__(self, name, tuple(map(tuple, points)))
        for value in (self.width_tolerance, self.tangent_tolerance_degrees, self.min_support, self.max_error):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("band tolerances and support must be finite and positive")
        if self.tangent_tolerance_degrees >= 90 or not 33 <= self.samples <= 1025:
            raise ValueError("band sampling or tangent tolerance is out of range")
        if not math.isfinite(self.tangent_support_fraction) or not 0 < self.tangent_support_fraction <= 0.25:
            raise ValueError("tangent support must be a fraction in (0, 0.25]")


@dataclass(frozen=True)
class BandDetection:
    accepted: bool
    reason: str
    width: float | None = None
    width_variation: float | None = None
    tangent_error_degrees: float | None = None
    center_samples: tuple[Point, ...] = ()
    signed_width: float | None = None


@dataclass(frozen=True)
class BandReconstruction:
    accepted: bool
    reason: str
    detection: BandDetection
    centerline_controls: tuple[Point, ...] = ()
    width: float | None = None
    source_errors: tuple[float, float] = (math.inf, math.inf)
    preflight_only: bool = True
    metric: str = "certified_svg_and_analytic_offset_to_source_upper_bound"
    offsets: BandOffsets | None = None


def _sample(points: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    positions = np.linspace(0, distance[-1], count)
    return np.column_stack([np.interp(positions, distance, points[:, i]) for i in (0, 1)]), float(distance[-1])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def detect_band(request: BandRequest) -> BandDetection:
    """Test constant normal width and monotonic, tangent-compatible support."""
    first, length = _sample(np.asarray(request.first), request.samples)
    second = np.asarray(request.second)
    if length < request.min_support:
        return BandDetection(False, "insufficient_support")
    if np.linalg.norm(first[0] - second[-1]) + np.linalg.norm(first[-1] - second[0]) < np.linalg.norm(first[0] - second[0]) + np.linalg.norm(first[-1] - second[-1]):
        second = second[::-1]
    support_samples = max(1, round((request.samples - 1) * request.tangent_support_fraction))
    indices = np.arange(len(first))
    tangent = first[np.minimum(indices + support_samples, len(first) - 1)] - first[np.maximum(indices - support_samples, 0)]
    speed = np.linalg.norm(tangent, axis=1)
    if np.min(speed) <= 1e-9:
        return BandDetection(False, "degenerate_tangent_support")
    tangent /= speed[:, None]
    normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    starts, vectors = second[:-1], np.diff(second, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    arc = np.r_[0.0, np.cumsum(lengths)]
    widths, errors, positions, centers = [], [], [], []
    for point, normal, direction in zip(first, normals, tangent, strict=True):
        denominator = _cross(normal, vectors)
        valid = np.abs(denominator) > 1e-9
        ray = np.divide(_cross(starts - point, vectors), denominator, out=np.full(len(vectors), np.inf), where=valid)
        fraction = np.divide(_cross(starts - point, normal), denominator, out=np.full(len(vectors), np.inf), where=valid)
        indices = np.flatnonzero(valid & (fraction >= -1e-8) & (fraction <= 1 + 1e-8))
        if not len(indices):
            return BandDetection(False, "incomplete_normal_correspondence")
        index = indices[np.argmin(np.abs(ray[indices]))]
        width = float(ray[index])
        widths.append(width)
        position = float(arc[index] + np.clip(fraction[index], 0, 1) * lengths[index])
        support = length * request.tangent_support_fraction
        neighborhood = np.column_stack([np.interp([position - support, position + support], arc, second[:, coordinate]) for coordinate in (0, 1)])
        second_tangent = neighborhood[1] - neighborhood[0]
        second_speed = float(np.linalg.norm(second_tangent))
        if second_speed <= 1e-9:
            return BandDetection(False, "degenerate_tangent_support")
        errors.append(math.degrees(math.acos(float(np.clip(np.dot(direction, second_tangent / second_speed), -1, 1)))))
        positions.append(position)
        centers.append(tuple(point + normal * width / 2))
    values = np.asarray(widths)
    signed = float(np.median(values))
    variation = float(np.max(np.abs(values - signed)))
    angle = float(max(errors))
    reason = "constant_width"
    if np.min(values) <= 0 <= np.max(values) or abs(signed) <= 1e-6:
        reason = "crossing_or_zero_width"
    elif np.min(np.diff(positions)) < -1e-6:
        reason = "non_monotonic_correspondence"
    elif variation > request.width_tolerance:
        reason = "width_variation"
    elif angle > request.tangent_tolerance_degrees:
        reason = "tangent_mismatch"
    return BandDetection(reason == "constant_width", reason, abs(signed), variation, angle, tuple(centers), signed)


def reconstruct_band(request: BandRequest, *, width: float | None = None) -> BandReconstruction:
    """Propose a shared cubic centerline and certified SVG offset curves.

    Explicit width may bypass width/tangent detection admission, never geometric
    correspondence or source checks. Source errors bound both returned cubics
    and their analytic offsets, including approximation and linearization error.
    This remains preflight until corner bindings and topology are admitted.
    """
    if width is not None and (not math.isfinite(width) or width <= 0):
        raise ValueError("band width must be finite and positive")
    detection = detect_band(request)
    explicit_admission = width is not None and detection.reason in {"width_variation", "tangent_mismatch"}
    if not detection.accepted and not explicit_admission:
        return BandReconstruction(False, detection.reason, detection)
    selected_width = detection.width if width is None else float(width)
    if selected_width is None or not math.isfinite(selected_width) or selected_width <= 0:
        raise ValueError("band width must be finite and positive")
    points = np.asarray(detection.center_samples)
    _, length = _sample(points, request.samples)
    if length <= 1e-9:
        return BandReconstruction(False, "degenerate_centerline", detection)
    parameters = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))] / length
    t = parameters[:, None]
    basis = np.column_stack([3 * (1 - parameters) ** 2 * parameters, 3 * (1 - parameters) * parameters ** 2])
    remainder = points - (1 - t) ** 3 * points[0] - t ** 3 * points[-1]
    controls = np.vstack([points[0], np.linalg.lstsq(basis, remainder, rcond=None)[0], points[-1]])
    # Verification must not consume a material share of the requested fit
    # budget: three independent upper bounds are added below, not averaged.
    precision = min(0.005, request.max_error / 100)
    try:
        offsets = approximate_band_offsets(tuple(map(tuple, controls)), selected_width, tolerance=precision)
        if detection.signed_width < 0:
            offsets = BandOffsets(offsets.second, offsets.first, offsets.parameter_ranges, offsets.error_bound)
        candidates = tuple(offset_polyline(curves, tolerance=precision) for curves in (offsets.first, offsets.second))
    except ValueError:
        return BandReconstruction(False, "uncertifiable_offset", detection)
    errors = tuple(polyline_boundary_distance([exact_polyline_vertices(source)], [candidate], closed=False,
                                             precision=precision, direction="replacement_to_source")
                   + linearization_bound + offsets.error_bound
                   for source, (candidate, linearization_bound)
                   in zip((request.first, request.second), candidates, strict=True))
    accepted = max(errors) <= request.max_error
    return BandReconstruction(accepted, "source_supported" if accepted else "source_fidelity", detection,
                              tuple(map(tuple, controls)), selected_width, errors, offsets=offsets)

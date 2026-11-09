"""Diagnostic-only, multiscale detail comparison for explicitly paired boundaries.

This is not a complexity score or a corner detector. Local turning differences
are evidence to review alongside boundary distance and topology, not proof that
a feature is noise. No command-count term or acceptance threshold is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil, isfinite
from typing import Literal

import numpy as np
import skia
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar

from .path_geometry import LineCommand, PathGeometry
from .skia_geometry import svg_d_to_skia


@dataclass(frozen=True)
class BoundaryDetailPolicy:
    """Scales/spacing are source pixels; registration is caller-owned and isotropic."""

    scales_px: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    sample_spacing_px: float = 0.25
    units_per_pixel: float = 1.0
    max_samples: int = 16384

    def __post_init__(self) -> None:
        object.__setattr__(self, "scales_px", tuple(self.scales_px))
        values = (*self.scales_px, self.sample_spacing_px, self.units_per_pixel)
        if not self.scales_px or not all(isfinite(v) and v > 0 for v in values):
            raise ValueError("detail scales, spacing and pixel registration must be finite and positive")
        if tuple(sorted(set(self.scales_px))) != self.scales_px:
            raise ValueError("detail scales must be unique and increasing")
        if self.sample_spacing_px > min(self.scales_px) / 4:
            raise ValueError("sample spacing must resolve the smallest scale with at least four samples")
        if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, int) or self.max_samples < 64:
            raise ValueError("sample budget must be an integer of at least 64")


@dataclass(frozen=True)
class BoundaryDetailRequest:
    """One region-local boundary pair, not a guessed component/hole correspondence.

    Supply retained material polyline evidence for raster detail assessment.
    Cleaned or declared references are allowed but explicitly labeled; a fitted
    path must never be passed off as the original source. Both geometries must
    already be registered to the same canvas. Anchors do not affect measurement.
    """

    region_id: str
    source: PathGeometry
    candidate: PathGeometry
    source_revision: str
    candidate_revision: str
    reference_kind: Literal["retained_polyline", "cleaned_polyline", "declared_geometry"]
    policy: BoundaryDetailPolicy = field(default_factory=BoundaryDetailPolicy)

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.region_id, self.source_revision, self.candidate_revision)):
            raise ValueError("region and both revision identities are required")
        if self.reference_kind not in {"retained_polyline", "cleaned_polyline", "declared_geometry"}:
            raise ValueError("unknown detail reference kind")
        if not isinstance(self.source, PathGeometry) or not isinstance(self.candidate, PathGeometry):
            raise ValueError("detail assessment requires one explicitly paired closed path on each side")
        if self.reference_kind != "declared_geometry" and any(
            not isinstance(command, LineCommand) for command in self.source.path.drawable_commands
        ):
            raise ValueError("polyline detail references must contain only line commands")


@dataclass(frozen=True)
class DetailSample:
    source_fraction: float
    source_point: tuple[float, float]
    candidate_point: tuple[float, float]
    source_curvature_per_px: float
    candidate_curvature_per_px: float
    unsupported_curvature_per_px: float
    missing_curvature_per_px: float
    signed_curvature_difference_per_px: float


@dataclass(frozen=True)
class DetailScale:
    scale_px: float
    unsupported_turn_radians: float
    missing_turn_radians: float
    signed_turn_difference_radians: float
    samples: tuple[DetailSample, ...]


@dataclass(frozen=True)
class BoundaryDetailAssessment:
    region_id: str
    source_path_id: str
    candidate_path_id: str
    source_revision: str
    candidate_revision: str
    reference_kind: str
    policy: BoundaryDetailPolicy
    status: Literal["assessed", "not_assessed"]
    reason: str | None = None
    source_perimeter_px: float | None = None
    candidate_perimeter_px: float | None = None
    alignment_rms_px: float | None = None
    sample_count: int = 0
    scales: tuple[DetailScale, ...] = ()
    alignment_method: str = "winding-normalized cyclic spatial least-squares; normalized arc length"

    def diagnostics(self) -> dict[str, object]:
        """JSON-safe localized measurements; intentionally no accepted/quality score."""
        return asdict(self)


def _measure(geometry: PathGeometry, units_per_pixel: float) -> skia.PathMeasure:
    native = svg_d_to_skia(geometry.to_svg_d())
    native.transform(skia.Matrix.Scale(1 / units_per_pixel, 1 / units_per_pixel))
    if not native.isFinite():
        raise ValueError("non_finite_boundary")
    # Skia integrates the geometry rather than taking N samples per command.
    measure = skia.PathMeasure(native, False, 64)
    if not measure.isClosed() or not isfinite(measure.getLength()) or measure.getLength() <= 0:
        raise ValueError("degenerate_boundary")
    return measure


def _sample(measure: skia.PathMeasure, count: int, offset: float = 0.0) -> np.ndarray:
    points = np.asarray([
        tuple(measure.getPosTan(float(distance))[0])
        for distance in np.arange(count) * (measure.getLength() / count)
    ], dtype=float)
    area = np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1))
    if not np.isfinite(points).all() or abs(area) <= 1e-12 * measure.getLength() ** 2:
        raise ValueError("degenerate_boundary")
    if offset == 0:
        return points if area > 0 else points[::-1].copy()
    indices = np.arange(count) + offset
    if area < 0:
        indices = count - 1 - indices
    return np.asarray([
        tuple(measure.getPosTan(float(distance))[0])
        for distance in (indices % count) * (measure.getLength() / count)
    ], dtype=float)


def _shift(points: np.ndarray, offset: float) -> np.ndarray:
    index = np.arange(len(points), dtype=float) + offset
    return np.column_stack([
        np.interp(index % len(points), np.arange(len(points) + 1), np.r_[points[:, axis], points[0, axis]])
        for axis in range(points.shape[1])
    ])


def _align(source: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, float, float]:
    correlation = np.fft.ifft(
        np.sum(np.conj(np.fft.fft(source, axis=0)) * np.fft.fft(candidate, axis=0), axis=1)
    ).real
    best = int(np.argmax(correlation))

    def error(offset: float) -> float:
        return float(np.mean(np.sum((source - _shift(candidate, offset)) ** 2, axis=1)))

    result = minimize_scalar(error, bounds=(best - 1, best + 1), method="bounded",
        options={"xatol": 1e-8})
    # Retain an exact integer match instead of introducing interpolation error.
    offset = min((float(best), float(result.x)), key=error)
    return _shift(candidate, offset), offset, float(np.sqrt(error(offset)))


def _curvature(points: np.ndarray, perimeter: float, scale: float) -> np.ndarray:
    step = perimeter / len(points)
    tangent = gaussian_filter1d(points, scale / step, order=1, axis=0, mode="wrap")
    if np.any(np.linalg.norm(tangent, axis=1) < 1e-10):
        raise ValueError("ambiguous_tangent_at_scale")
    before = _shift(tangent, -scale / step)
    after = _shift(tangent, scale / step)
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = np.sum(before * after, axis=1)
    return np.arctan2(cross, dot) / (2 * scale)


def assess_boundary_detail(request: BoundaryDetailRequest) -> BoundaryDetailAssessment:
    """Compare local turning at explicit scales without fitting or modifying paths.

    Positions are returned in source pixels. Integrals use source arc length,
    including candidate excess, so both directions share a documented measure.
    Spatial phase alignment is not semantic feature correspondence: review the
    RMS and localized samples alongside independent boundary-distance metrics.
    """
    common = dict(region_id=request.region_id, source_path_id=request.source.path.id,
        candidate_path_id=request.candidate.path.id, source_revision=request.source_revision,
        candidate_revision=request.candidate_revision, reference_kind=request.reference_kind,
        policy=request.policy)
    try:
        source_measure = _measure(request.source, request.policy.units_per_pixel)
        candidate_measure = _measure(request.candidate, request.policy.units_per_pixel)
        source_length, candidate_length = source_measure.getLength(), candidate_measure.getLength()
        count = max(64, ceil(max(source_length, candidate_length) / request.policy.sample_spacing_px))
        if count > request.policy.max_samples:
            raise ValueError("sample_budget_exceeded")
        if 4 * max(request.policy.scales_px) >= min(source_length, candidate_length):
            raise ValueError("scale_exceeds_boundary_support")
        source, candidate = _sample(source_measure, count), _sample(candidate_measure, count)
        _aligned, offset, _rms = _align(source, candidate)
        # Re-evaluate the actual curve at the phase, not an interpolated
        # curvature signature: interpolation itself can look like lost detail.
        aligned = _sample(candidate_measure, count, offset)
        rms = float(np.sqrt(np.mean(np.sum((source - aligned) ** 2, axis=1))))
        scales: list[DetailScale] = []
        for scale in request.policy.scales_px:
            original_turn = _curvature(source, source_length, scale)
            fitted_turn = _curvature(aligned, candidate_length, scale)
            difference = np.abs(fitted_turn) - np.abs(original_turn)
            unsupported, missing = np.maximum(difference, 0), np.maximum(-difference, 0)
            signed = np.abs(fitted_turn - original_turn)
            samples = tuple(DetailSample(i / count, tuple(source[i]), tuple(aligned[i]),
                float(original_turn[i]), float(fitted_turn[i]), float(unsupported[i]),
                float(missing[i]), float(signed[i])) for i in range(count))
            step = source_length / count
            scales.append(DetailScale(scale, float(unsupported.sum() * step),
                float(missing.sum() * step), float(signed.sum() * step), samples))
        return BoundaryDetailAssessment(**common, status="assessed", source_perimeter_px=source_length,
            candidate_perimeter_px=candidate_length, alignment_rms_px=rms,
            sample_count=count, scales=tuple(scales))
    except ValueError as error:
        return BoundaryDetailAssessment(**common, status="not_assessed", reason=str(error))

"""Source-guarded joins between a certified band interior and fixed endpoints.

These are numeric proposals, not drawing publication. The constant-width
certificate covers the interior only; corner transitions are ordinary cubics.
"""

from dataclasses import dataclass
import math

import numpy as np

from ._fitcurve import _generate_cubic
from .band_offsets import BandOffsets, Cubic, Point, offset_polyline, exact_polyline_vertices
from .quality import polyline_boundary_distance

BandEndpoints = tuple[tuple[Point, Point], tuple[Point, Point]]


@dataclass(frozen=True)
class BandJoinResult:
    accepted: bool
    reason: str
    first: tuple[Cubic, ...]
    second: tuple[Cubic, ...]
    source_errors: tuple[float, float]
    preflight_only: bool = True


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("degenerate band join tangent")
    return vector / length


def _source_tangent(points: np.ndarray, *, end: bool) -> np.ndarray:
    """Use a quarter of the transition's arc support, not one raster edge."""
    supported = points[::-1] if end else points
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(supported, axis=0), axis=1))]
    point = np.array([np.interp(arc[-1] / 4, arc, supported[:, axis]) for axis in (0, 1)])
    direction = point - supported[0]
    return _unit(-direction if end else direction)


def _bridge(points: np.ndarray, start: np.ndarray, end: np.ndarray,
            start_tangent: np.ndarray, end_tangent: np.ndarray) -> Cubic:
    samples = points.copy()
    samples[0], samples[-1] = start, end
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
    if arc[-1] <= 1e-12:
        raise ValueError("degenerate band join")
    controls = _generate_cubic(samples, arc / arc[-1], _unit(start_tangent), -_unit(end_tangent))
    return tuple(tuple(float(value) for value in point) for point in controls)


def _join_one(curves: tuple[Cubic, ...], source: np.ndarray,
              endpoints: tuple[Point, Point]) -> tuple[Cubic, ...]:
    start, end = np.asarray(curves[0][0]), np.asarray(curves[-1][-1])
    left = int(np.argmin(np.linalg.norm(source - start, axis=1)))
    right = int(np.argmin(np.linalg.norm(source - end, axis=1)))
    if left >= right:
        raise ValueError("band interior has no ordered source support")
    source = source.copy()
    source[0], source[-1] = endpoints
    result = []
    if tuple(start) != tuple(source[0]):
        points = np.vstack([source[:left+1], start])
        result.append(_bridge(points, source[0], start,
                              _source_tangent(points, end=False), np.asarray(curves[0][1]) - start))
    result.extend(curves)
    if tuple(end) != tuple(source[-1]):
        points = np.vstack([end, source[right:]])
        result.append(_bridge(points, end, source[-1],
                              end - np.asarray(curves[-1][-2]), _source_tangent(points, end=True)))
    return tuple(result)


def join_band_offsets(offsets: BandOffsets, first: tuple[Point, ...], second: tuple[Point, ...],
                      *, max_error: float,
                      endpoints: BandEndpoints | None = None) -> BandJoinResult:
    """Keep all corner endpoints fixed and independently guard complete edges.

    ``offsets`` must come from the certified approximation kernel. The second
    source may be reversed; output traversal follows the certified interior.
    Explicit retained endpoints override source samples as join targets, never
    as the error reference. No corner kind, command identity or topology is
    inferred here.
    """
    if not math.isfinite(max_error) or max_error <= 0:
        raise ValueError("band join error must be finite and positive")
    sources = []
    for points, curves in zip((first, second), (offsets.first, offsets.second), strict=True):
        source = np.asarray(points, dtype=float)
        if source.ndim != 2 or source.shape[1] != 2 or len(source) < 3 or not np.isfinite(source).all():
            raise ValueError("band joins require finite source polylines")
        if not curves:
            raise ValueError("band joins require nonempty certified interior curves")
        direct = np.linalg.norm(source[0] - curves[0][0]) + np.linalg.norm(source[-1] - curves[-1][-1])
        reverse = np.linalg.norm(source[-1] - curves[0][0]) + np.linalg.norm(source[0] - curves[-1][-1])
        sources.append(source[::-1] if reverse < direct else source)
    if endpoints is None:
        endpoints = tuple((tuple(source[0]), tuple(source[-1])) for source in sources)
    declared = np.asarray(endpoints, dtype=float)
    if declared.shape != (2, 2, 2) or not np.isfinite(declared).all():
        raise ValueError("band joins require two finite endpoint pairs")
    joined = tuple(_join_one(curves, source, ends) for curves, source, ends in
                   zip((offsets.first, offsets.second), sources, endpoints, strict=True))
    precision = min(0.005, max_error / 100)
    errors = []
    for curves, source in zip(joined, sources, strict=True):
        polyline, linearization_bound = offset_polyline(curves, tolerance=precision)
        errors.append(polyline_boundary_distance([exact_polyline_vertices(source)], [polyline], closed=False,
                                                precision=precision, direction="replacement_to_source")
                      + linearization_bound + offsets.error_bound)
    accepted = max(errors) <= max_error
    return BandJoinResult(accepted, "source_supported" if accepted else "source_fidelity",
                          joined[0], joined[1], tuple(errors))

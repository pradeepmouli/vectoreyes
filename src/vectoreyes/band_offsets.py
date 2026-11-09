"""Bounded cubic approximation of analytic constant-normal-width offsets.

Certificates compare the offset and its Hermite cubic over whole parameter
intervals. They use convex-hull derivative bounds, not a fixed sample grid.
"""

from dataclasses import dataclass
import math

import numpy as np

Point = tuple[float, float]
Cubic = tuple[Point, Point, Point, Point]


def exact_polyline_vertices(points) -> tuple[Point, ...]:
    """Remove only exact collinear forward nodes, never approximate a source."""
    compact = []
    for value in points:
        point = np.asarray(value, dtype=float)
        while len(compact) >= 2:
            a, b = compact[-1] - compact[-2], point - compact[-1]
            if a[0] * b[1] - a[1] * b[0] != 0 or float(a @ b) < 0:
                break
            compact.pop()
        compact.append(point)
    return tuple(map(tuple, compact))


@dataclass
class _WorkBudget:
    remaining: int

    def __post_init__(self) -> None:
        if isinstance(self.remaining, bool) or not isinstance(self.remaining, int) or self.remaining <= 0:
            raise ValueError("certification work limit must be a positive integer")

    def consume(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError("band certification work limit exceeded")


@dataclass(frozen=True)
class BandOffsets:
    first: tuple[Cubic, ...]
    second: tuple[Cubic, ...]
    parameter_ranges: tuple[tuple[float, float], ...]
    error_bound: float


def offset_polyline(curves: tuple[Cubic, ...], *, tolerance: float) -> tuple[tuple[Point, ...], float]:
    """Linearize cubic pieces with a same-parameter convex-hull error bound."""
    if not curves or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("nonempty curves and a finite positive tolerance are required")
    points = [curves[0][0]]
    bounds = []
    budget = _WorkBudget(65536)

    def flatten(control: np.ndarray, depth: int = 0) -> None:
        budget.consume()
        chord_controls = np.linspace(control[0], control[-1], 4)
        bound = float(np.max(np.linalg.norm(control-chord_controls, axis=1)))
        if bound <= tolerance:
            points.append(tuple(float(x) for x in control[-1]))
            bounds.append(bound)
            return
        if depth >= 24:
            raise ValueError("could not certify band linearization")
        left, right = _split(control)
        flatten(left, depth+1)
        flatten(right, depth+1)

    for curve in curves:
        control = np.asarray(curve, dtype=float)
        if control.shape != (4, 2) or not np.isfinite(control).all() or tuple(control[0]) != points[-1]:
            raise ValueError("band curves must be finite and connected")
        flatten(control)
    return tuple(points), max(bounds)


def _split(control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels = [control]
    while len(levels[-1]) > 1:
        levels.append((levels[-1][:-1] + levels[-1][1:]) / 2)
    return np.asarray([level[0] for level in levels]), np.asarray([level[-1] for level in reversed(levels)])


def _offset_jet(control: np.ndarray, t: float, distance: float) -> tuple[np.ndarray, np.ndarray]:
    p = control
    point = (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t*t*p[2] + t**3*p[3]
    velocity = 3*((1-t)**2*(p[1]-p[0]) + 2*(1-t)*t*(p[2]-p[1]) + t*t*(p[3]-p[2]))
    acceleration = 6*((1-t)*(p[2]-2*p[1]+p[0]) + t*(p[3]-2*p[2]+p[1]))
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-12:
        raise ValueError("singular band centerline")
    tangent = velocity / speed
    normal = np.array([-tangent[1], tangent[0]])
    tangent_derivative = acceleration/speed - velocity*float(velocity@acceleration)/speed**3
    offset_velocity = velocity + distance*np.array([-tangent_derivative[1], tangent_derivative[0]])
    if float(offset_velocity @ velocity) <= 1e-12:
        raise ValueError("singular or folded band offset")
    return point + distance*normal, offset_velocity


def _second_derivative_bound(control: np.ndarray, distance: float) -> float:
    """Conservative bound in this restricted curve's local [0,1] parameter."""
    chord = control[-1]-control[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-12:
        return math.inf
    velocity = 3*np.diff(control, axis=0)
    speed_lower = float(np.min(velocity @ (chord/chord_length)))
    if speed_lower <= 1e-12:
        return math.inf
    speed_upper = float(np.max(np.linalg.norm(velocity, axis=1)))
    acceleration = 6*np.diff(control, n=2, axis=0)
    acceleration_upper = float(np.max(np.linalg.norm(acceleration, axis=1)))
    jerk = float(np.linalg.norm(6*np.diff(control, n=3, axis=0)[0]))
    # |N''| <= 2|v''|/|v| + 6|v'|²/|v|², where v is center velocity.
    normal_second = 2*jerk/speed_lower + 6*acceleration_upper**2/speed_lower**2
    curvature_upper = speed_upper*acceleration_upper/speed_lower**3
    if abs(distance)*curvature_upper >= 1:
        return math.inf  # subdivide until regularity is established
    return acceleration_upper + abs(distance)*normal_second


def _certificate(source: np.ndarray, fitted: np.ndarray, distance: float, tolerance: float, budget: _WorkBudget, depth: int = 0) -> float | None:
    budget.consume()
    endpoints = [_offset_jet(source, t, distance)[0] for t in (0.0, 1.0)]
    endpoint_error = max(float(np.linalg.norm(endpoints[0]-fitted[0])), float(np.linalg.norm(endpoints[1]-fitted[-1])))
    if endpoint_error > tolerance:
        return None
    offset_second = _second_derivative_bound(source, distance)
    cubic_second = float(np.max(np.linalg.norm(6*np.diff(fitted, n=2, axis=0), axis=1)))
    # A twice differentiable vector function differs from its linear endpoint
    # interpolant by at most sup|f''|/8 on [0,1]. Apply to offset-minus-fit.
    bound = endpoint_error + (offset_second+cubic_second)/8
    if bound <= tolerance:
        return bound
    source_left, source_right = _split(source)
    fitted_left, fitted_right = _split(fitted)
    midpoint = _offset_jet(source, 0.5, distance)[0]
    if float(np.linalg.norm(midpoint-fitted_left[-1])) > tolerance:
        return None
    if depth >= 18:
        raise ValueError("could not certify band offset approximation")
    left = _certificate(source_left, fitted_left, distance, tolerance, budget, depth+1)
    if left is None:
        return None
    right = _certificate(source_right, fitted_right, distance, tolerance, budget, depth+1)
    return None if right is None else max(left, right)


def approximate_band_offsets(control: Cubic, width: float, *, tolerance: float = 0.02, max_nodes: int = 65536) -> BandOffsets:
    """Approximate both analytic offsets, preserving common knots and tangents.

    Reject singular/folded offsets and exhausted certificates. The bound is on
    unrounded controls; consumers must preserve full precision or budget their
    serialization error separately before publishing.
    """
    budget = _WorkBudget(max_nodes)
    root = np.asarray(control, dtype=float)
    if root.shape != (4, 2) or not np.isfinite(root).all():
        raise ValueError("band centerline requires four finite 2D controls")
    if not math.isfinite(width) or width <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("band width and approximation tolerance must be positive and finite")
    first, second, ranges, bounds = [], [], [], []

    def construct(source: np.ndarray, lo: float, hi: float, depth: int = 0) -> None:
        budget.consume()
        candidates = []
        certificates = []
        for distance in (-width/2, width/2):
            start, start_velocity = _offset_jet(root, lo, distance)
            end, end_velocity = _offset_jet(root, hi, distance)
            fitted = np.array([start, start+start_velocity*(hi-lo)/3, end-end_velocity*(hi-lo)/3, end])
            candidates.append(tuple(tuple(float(x) for x in point) for point in fitted))
            certificates.append(_certificate(source, fitted, distance, tolerance, budget))
        if all(value is not None for value in certificates):
            first.append(candidates[0]); second.append(candidates[1])
            ranges.append((lo, hi)); bounds.extend(certificates)
            return
        if depth >= 16:
            raise ValueError("could not certify band offset approximation")
        left, right = _split(source)
        middle = (lo+hi)/2
        construct(left, lo, middle, depth+1)
        construct(right, middle, hi, depth+1)

    construct(root, 0.0, 1.0)
    return BandOffsets(tuple(first), tuple(second), tuple(ranges), float(max(bounds)))

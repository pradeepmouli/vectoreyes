from __future__ import annotations

import math
import re

import numpy as np

from ...skia_geometry import SkPath, unary_union
from ...fit import Shape, SIMPLE_CURVE_RESIDUAL_TOL, _fmt, fit_path, minimum_line_length
from ..automatic_replay import SimplifyLineletOperation, draft_input_ref, operation_intent
from ..framework import Proposal
from ..vector_region import VectorRegion, _parse_subpaths, _ring_area, _sample_subpath

_PATH_COMMAND = re.compile(r"[MLQCAZ]")
_MAX_GEOMETRY_RESIDUAL = SIMPLE_CURVE_RESIDUAL_TOL + 0.001
_MAX_SMOOTH_CURVE_GEOMETRY_RESIDUAL = SIMPLE_CURVE_RESIDUAL_TOL + 0.004
_MAX_SHORT_LINELET_GEOMETRY_RESIDUAL = 0.06
_COMMAND_COST = {"M": 0, "Z": 0, "L": 1, "Q": 2, "C": 3, "A": 3}
_FORCED_CURVE_JOIN_DEG = 35.0


def _command_count(d: str) -> int:
    return len(_PATH_COMMAND.findall(d))


def _closed_points(points: list[tuple[float, float]]) -> np.ndarray | None:
    if len(points) < 4:
        return None
    arr = np.asarray(points, dtype=float)
    if not np.allclose(arr[0], arr[-1]):
        arr = np.vstack([arr, arr[0]])
    return arr


def _subpath_d(tokens: list[tuple[str, list[float]]]) -> str:
    parts: list[str] = []
    for command, values in tokens:
        if values:
            parts.append(f"{command}{' '.join(_fmt(v) for v in values)}")
        else:
            parts.append(command)
    return " ".join(parts)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    cos_angle = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _command_end(command: str, values: list[float], start: np.ndarray) -> np.ndarray:
    if command in {"M", "L"}:
        return np.array(values[:2], dtype=float)
    if command == "Q":
        return np.array(values[2:4], dtype=float)
    if command == "C":
        return np.array(values[4:6], dtype=float)
    if command == "A":
        return np.array(values[5:7], dtype=float)
    return start.copy()


def _command_start_tangent(command: str, values: list[float], start: np.ndarray, end: np.ndarray) -> np.ndarray:
    if command in {"L", "Z", "A"}:
        return end - start
    if command == "Q":
        return np.array(values[:2], dtype=float) - start
    if command == "C":
        return np.array(values[:2], dtype=float) - start
    return np.zeros(2, dtype=float)


def _command_end_tangent(command: str, values: list[float], start: np.ndarray, end: np.ndarray) -> np.ndarray:
    if command in {"L", "Z", "A"}:
        return end - start
    if command == "Q":
        return end - np.array(values[:2], dtype=float)
    if command == "C":
        return end - np.array(values[2:4], dtype=float)
    return np.zeros(2, dtype=float)


def _path_segments(tokens: list[tuple[str, list[float]]]) -> list[dict[str, object]]:
    current: np.ndarray | None = None
    start: np.ndarray | None = None
    segments: list[dict[str, object]] = []
    for command, values in tokens:
        if command == "M":
            current = np.array(values[:2], dtype=float)
            start = current.copy()
            continue
        if current is None:
            continue
        end = start.copy() if command == "Z" and start is not None else _command_end(command, values, current)
        segments.append(
            {
                "command": command,
                "start": current,
                "end": end,
                "start_tangent": _command_start_tangent(command, values, current, end),
                "end_tangent": _command_end_tangent(command, values, current, end),
            }
        )
        current = end
    return segments


def _curve_join_angle_penalty_d(d: str) -> float:
    penalty = 0.0
    for tokens in _parse_subpaths(d):
        segments = _path_segments(tokens)
        for left, right in zip(segments, segments[1:], strict=False):
            if not (
                (left["command"] in {"Q", "C", "A"} and right["command"] in {"Q", "C", "A"})
                or (left["command"] == "Q" and right["command"] == "L")
            ):
                continue
            if not np.allclose(left["end"], right["start"]):
                continue
            penalty = max(penalty, _angle_between(left["end_tangent"], right["start_tangent"]))
    return penalty


def _sharp_curve_join_points(tokens: list[tuple[str, list[float]]]) -> list[tuple[float, float]]:
    segments = _path_segments(tokens)
    points: list[tuple[float, float]] = []
    for left, right in zip(segments, [*segments[1:], segments[0]] if segments else [], strict=False):
        if left["command"] not in {"Q", "C", "A"} and right["command"] not in {"Q", "C", "A"}:
            continue
        if not np.allclose(left["end"], right["start"]):
            continue
        angle = _angle_between(left["end_tangent"], right["start_tangent"])
        if angle >= _FORCED_CURVE_JOIN_DEG:
            point = left["end"]
            points.append((float(point[0]), float(point[1])))
    return points


def _forced_corners(tokens: list[tuple[str, list[float]]]) -> np.ndarray | None:
    if not any(command in {"Q", "C", "A"} for command, _values in tokens):
        return None
    corners: list[tuple[float, float]] = []
    for command, values in tokens:
        if command == "M" and len(values) >= 2:
            corners.append((float(values[0]), float(values[1])))
        elif command == "L" and len(values) >= 2:
            corners.append((float(values[0]), float(values[1])))
    corners.extend(_sharp_curve_join_points(tokens))
    if not corners:
        return None
    unique = sorted(set(corners), key=corners.index)
    return np.asarray(unique, dtype=float)


def _point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    span = end - start
    denom = float(np.dot(span, span))
    if denom == 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, span) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * span)))


def _has_short_linelet(tokens: list[tuple[str, list[float]]], *, epsilon: float) -> bool:
    return _short_linelet_count_tokens(tokens, epsilon=epsilon) > 0


def _short_linelet_count_tokens(tokens: list[tuple[str, list[float]]], *, epsilon: float) -> int:
    if not any(command in {"Q", "C", "A"} for command, _values in tokens):
        return 0
    current: np.ndarray | None = None
    min_length = minimum_line_length(epsilon)
    count = 0
    for command, values in tokens:
        if command == "M" and len(values) >= 2:
            current = np.array(values[:2], dtype=float)
        elif command == "L" and current is not None and len(values) >= 2:
            end = np.array(values[:2], dtype=float)
            if float(np.linalg.norm(end - current)) < min_length:
                count += 1
            current = end
        elif command == "Q" and len(values) >= 4:
            current = np.array(values[2:4], dtype=float)
        elif command == "C" and len(values) >= 6:
            current = np.array(values[4:6], dtype=float)
        elif command == "A" and len(values) >= 7:
            current = np.array(values[5:7], dtype=float)
    return count


def _short_linelet_count_d(d: str, *, epsilon: float) -> int:
    return sum(_short_linelet_count_tokens(tokens, epsilon=epsilon) for tokens in _parse_subpaths(d))


def _curve_is_straight(points: list[np.ndarray], start: np.ndarray, end: np.ndarray, epsilon: float) -> bool:
    if float(np.linalg.norm(end - start)) < minimum_line_length(epsilon):
        return False
    return all(_point_line_distance(point, start, end) <= epsilon for point in points)


def _normalized_subpath_d(tokens: list[tuple[str, list[float]]], *, epsilon: float) -> str:
    parts: list[str] = []
    current: np.ndarray | None = None
    start: np.ndarray | None = None
    for command, values in tokens:
        if command == "M":
            current = np.array(values[:2], dtype=float)
            start = current.copy()
            parts.append(f"M{_fmt(current[0])} {_fmt(current[1])}")
        elif command == "L":
            current = np.array(values[:2], dtype=float)
            parts.append(f"L{_fmt(current[0])} {_fmt(current[1])}")
        elif command == "Q" and current is not None:
            control = np.array(values[:2], dtype=float)
            end = np.array(values[2:4], dtype=float)
            if _curve_is_straight([control], current, end, epsilon):
                parts.append(f"L{_fmt(end[0])} {_fmt(end[1])}")
            else:
                parts.append(_subpath_d([(command, values)]))
            current = end
        elif command == "C" and current is not None:
            control1 = np.array(values[:2], dtype=float)
            control2 = np.array(values[2:4], dtype=float)
            end = np.array(values[4:6], dtype=float)
            if _curve_is_straight([control1, control2], current, end, epsilon):
                parts.append(f"L{_fmt(end[0])} {_fmt(end[1])}")
            else:
                parts.append(_subpath_d([(command, values)]))
            current = end
        elif command == "Z":
            parts.append("Z")
            current = start
        else:
            parts.append(_subpath_d([(command, values)]))
    return " ".join(parts)


def _path_subpaths(
    shape: Shape,
    *,
    samples: int = 24,
) -> list[tuple[list[tuple[str, list[float]]], list[tuple[float, float]], float]]:
    subpaths = []
    for tokens in _parse_subpaths(str(shape.params.get("d", ""))):
        points = _sample_subpath(tokens, samples=samples)
        if len(points) < 3:
            continue
        subpaths.append((tokens, points, _ring_area(points)))
    return subpaths


def _quadratic_arc_segment(
    cx: float,
    cy: float,
    radius: float,
    a0: float,
    a1: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    p0 = np.array([cx + radius * math.cos(a0), cy + radius * math.sin(a0)], dtype=float)
    p1 = np.array([cx + radius * math.cos(a1), cy + radius * math.sin(a1)], dtype=float)
    d0 = np.array([-math.sin(a0), math.cos(a0)], dtype=float)
    d1 = np.array([-math.sin(a1), math.cos(a1)], dtype=float)
    matrix = np.column_stack([d0, -d1])
    if abs(float(np.linalg.det(matrix))) < 1e-9:
        control = (p0 + p1) / 2.0
    else:
        t, _ = np.linalg.solve(matrix, p1 - p0)
        control = p0 + float(t) * d0
    return (float(control[0]), float(control[1])), (float(p1[0]), float(p1[1]))


def _append_quadratic_arc(parts: list[str], cx: float, cy: float, radius: float, a0: float, a1: float) -> None:
    mid = (a0 + a1) / 2.0
    for start, end in ((a0, mid), (mid, a1)):
        control, point = _quadratic_arc_segment(cx, cy, radius, start, end)
        parts.append(f"Q{_fmt(control[0])} {_fmt(control[1])} {_fmt(point[0])} {_fmt(point[1])}")


def _side_radius_estimates(points: np.ndarray, x0: float, y0: float, x1: float, y1: float, tol: float) -> list[float]:
    estimates: list[float] = []
    width = x1 - x0
    height = y1 - y0
    sides = (
        (points[points[:, 1] <= y0 + tol][:, 0], width, lambda lo, hi: (lo - x0, x1 - hi)),
        (points[points[:, 1] >= y1 - tol][:, 0], width, lambda lo, hi: (lo - x0, x1 - hi)),
        (points[points[:, 0] <= x0 + tol][:, 1], height, lambda lo, hi: (lo - y0, y1 - hi)),
        (points[points[:, 0] >= x1 - tol][:, 1], height, lambda lo, hi: (lo - y0, y1 - hi)),
    )
    for values, span, fn in sides:
        if len(values) < 2:
            continue
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo < span * 0.20:
            continue
        estimates.extend(r for r in fn(lo, hi) if r > tol)
    return estimates


def _rounded_rect_radius(points: np.ndarray, poly: SkPath, epsilon: float) -> tuple[float, tuple[float, float, float, float]] | None:
    x0, y0, x1, y1 = (float(v) for v in poly.bounds)
    width = x1 - x0
    height = y1 - y0
    min_dim = min(width, height)
    if width < 4.0 or height < 4.0:
        return None

    bbox_area = width * height
    if bbox_area <= 0.0:
        return None
    fill_ratio = float(poly.area) / bbox_area
    if not 0.65 <= fill_ratio < 0.995:
        return None

    tol = max(1.0, epsilon * 1.5, min_dim * 0.01)
    estimates = _side_radius_estimates(points, x0, y0, x1, y1, tol)
    deficit = max(0.0, bbox_area - float(poly.area))
    area_radius = math.sqrt(deficit / (4.0 - math.pi)) if deficit > 0.0 else 0.0
    if area_radius > tol:
        estimates.append(area_radius)
    if not estimates:
        return None

    radius = max(float(np.median(estimates)), area_radius)
    if radius < max(2.0, min_dim * 0.025) or radius > min_dim * 0.48:
        return None
    return radius, (x0, y0, x1, y1)


def _rounded_rect_path(points: list[tuple[float, float]], *, epsilon: float) -> str | None:
    contour = _closed_points(points)
    if contour is None:
        return None
    poly = SkPath(shell=list(contour[:-1]))
    poly = poly.buffer(0)
    if poly.is_empty:
        return None

    fit = _rounded_rect_radius(contour[:-1], poly, epsilon)
    if fit is None:
        return None
    radius, (x0, y0, x1, y1) = fit

    parts = [f"M{_fmt(x0 + radius)} {_fmt(y0)}", f"L{_fmt(x1 - radius)} {_fmt(y0)}"]
    _append_quadratic_arc(parts, x1 - radius, y0 + radius, radius, -math.pi / 2.0, 0.0)
    parts.append(f"L{_fmt(x1)} {_fmt(y1 - radius)}")
    _append_quadratic_arc(parts, x1 - radius, y1 - radius, radius, 0.0, math.pi / 2.0)
    parts.append(f"L{_fmt(x0 + radius)} {_fmt(y1)}")
    _append_quadratic_arc(parts, x0 + radius, y1 - radius, radius, math.pi / 2.0, math.pi)
    parts.append(f"L{_fmt(x0)} {_fmt(y0 + radius)}")
    _append_quadratic_arc(parts, x0 + radius, y0 + radius, radius, math.pi, math.pi * 1.5)
    parts.append("Z")
    return " ".join(parts)


def _candidate_shape(
    shape: Shape,
    parts: list[str],
    *,
    preserve_fill_rule: bool = True,
) -> Shape:
    params = {"d": " ".join(parts)}
    if preserve_fill_rule and len(parts) > 1 and shape.params.get("fill_rule"):
        params["fill_rule"] = shape.params["fill_rule"]
    return Shape("path", params)


def _ordered_subpath_indices(subpaths) -> list[int]:
    if not subpaths:
        return []
    outer_index = max(range(len(subpaths)), key=lambda i: subpaths[i][2])
    return [outer_index, *(i for i in range(len(subpaths)) if i != outer_index)]


def _simplified_subpath_d(
    tokens: list[tuple[str, list[float]]],
    points: list[tuple[float, float]],
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> str:
    original = _subpath_d(tokens)
    candidates = [_normalized_subpath_d(tokens, epsilon=epsilon)]
    rounded = _rounded_rect_path(points, epsilon=epsilon)
    if rounded is not None and _path_command_cost_d(rounded) < _path_command_cost_d(original):
        return rounded

    contour = _closed_points(points)
    if contour is not None:
        forced_corners = _forced_corners(tokens)
        if _has_short_linelet(tokens, epsilon=epsilon):
            linelet_epsilon = max(epsilon, minimum_line_length(epsilon) * 0.375)
            candidates.append(
                str(
                    fit_path(
                        contour,
                        epsilon=linelet_epsilon,
                        max_error=max_error,
                        cubic=False,
                        prefer_simple_curves=True,
                    ).params["d"]
                )
            )
            if cubic:
                candidates.append(
                    str(
                        fit_path(
                            contour,
                            epsilon=linelet_epsilon,
                            max_error=max_error,
                            cubic=True,
                            prefer_simple_curves=True,
                        ).params["d"]
                    )
                )
        candidates.append(
            str(
                fit_path(
                    contour,
                    epsilon=epsilon,
                    max_error=max_error,
                    cubic=False,
                    forced_corners=forced_corners,
                    prefer_simple_curves=True,
                ).params["d"]
            )
        )
        if forced_corners is not None and _line_command_count_d(original) == 0:
            candidates.append(
                str(
                    fit_path(
                        contour,
                        epsilon=epsilon,
                        max_error=max_error,
                        cubic=False,
                        prefer_simple_curves=True,
                    ).params["d"]
                )
            )
        if forced_corners is not None and _line_command_count_d(original) == 0 and cubic:
            candidates.append(
                str(
                    fit_path(
                        contour,
                        epsilon=epsilon,
                        max_error=max_error,
                        cubic=True,
                        prefer_simple_curves=True,
                    ).params["d"]
                )
            )
        if cubic:
            candidates.append(
                str(
                    fit_path(
                        contour,
                        epsilon=epsilon,
                        max_error=max_error,
                        cubic=True,
                        forced_corners=forced_corners,
                        prefer_simple_curves=True,
                    ).params["d"]
                )
            )

    def candidate_key(d: str) -> tuple[int, int, int]:
        return (_short_linelet_count_d(d, epsilon=epsilon), _path_command_cost_d(d), _path_segment_count_d(d))

    if _short_linelet_count_d(original, epsilon=epsilon) == 0 and _line_command_count_d(original) == 0:
        all_curve_candidates = [candidate for candidate in candidates if _line_command_count_d(candidate) == 0]
        if all_curve_candidates:
            improved_candidates = [
                candidate
                for candidate in all_curve_candidates
                if candidate_key(candidate) < candidate_key(original)
                or _is_smoother_curve_join_candidate(original, candidate)
            ]
            candidates = improved_candidates or all_curve_candidates
            best = min(
                candidates,
                key=lambda d: (
                    _short_linelet_count_d(d, epsilon=epsilon),
                    _curve_join_angle_penalty_d(d),
                    _path_command_cost_d(d),
                    _path_segment_count_d(d),
                ),
            )
            if candidate_key(best) < candidate_key(original) or _is_smoother_curve_join_candidate(original, best):
                return best
            return original

    best = min(
        candidates,
        key=lambda d: (
            _short_linelet_count_d(d, epsilon=epsilon),
            _path_command_cost_d(d),
            _curve_join_angle_penalty_d(d),
            _path_segment_count_d(d),
        ),
    )
    if candidate_key(best) < candidate_key(original) or _is_smoother_curve_join_candidate(original, best):
        return best
    return original


def _candidate_parts(
    subpaths,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
    preserve_subpaths: bool,
    max_boundary_error: float | None = None,
) -> list[str]:
    # A small loop can be a deliberate counter. Preserve both correspondence
    # and every boundary; size alone is not permission to change topology.
    indices = list(range(len(subpaths))) if preserve_subpaths else _ordered_subpath_indices(subpaths)[:1]
    boundary_limit = max_error if max_boundary_error is None else max_boundary_error
    parts: list[str] = []
    for index in indices:
        tokens, points, _area = subpaths[index]
        fitted = _simplified_subpath_d(
            tokens,
            points,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
        )
        if len(subpaths) == 1:
            # Single-boundary paths retain the existing whole-path acceptance
            # policy, including its explicit max_boundary_error override.
            parts.append(fitted)
            continue
        # Do not discard a safe exterior improvement because a different
        # boundary's refit failed. Try local command normalization, then retain
        # the exact input coordinates for that boundary.
        original_d = " ".join(
            command + " ".join(str(value) for value in values)
            for command, values in tokens
        )
        original = Shape("path", {"d": original_d})
        for d in dict.fromkeys((fitted, _normalized_subpath_d(tokens, epsilon=epsilon))):
            candidate = Shape("path", {"d": d})
            if (
                _same_subpath_topology(original, candidate)
                and _geometry_residual(original, candidate) <= _MAX_GEOMETRY_RESIDUAL
                and _boundary_max_deviation(original, candidate) <= boundary_limit
            ):
                parts.append(d)
                break
        else:
            parts.append(original_d)
    return parts


def _same_subpath_topology(current: Shape, candidate: Shape) -> bool:
    """Preserve loop structure and pairwise containment, not just total area."""
    from ..vector_region import to_polygon

    try:
        before_tokens = _parse_subpaths(str(current.params.get("d", "")))
        after_tokens = _parse_subpaths(str(candidate.params.get("d", "")))
        if len(before_tokens) != len(after_tokens):
            return False
        before = [SkPath.from_svg_d(_subpath_d(tokens)).buffer(0) for tokens in before_tokens]
        after = [SkPath.from_svg_d(_subpath_d(tokens)).buffer(0) for tokens in after_tokens]
        for old, new in zip(before, after, strict=True):
            if old.is_empty or new.is_empty or old.area <= 0 or new.area <= 0:
                return False
            if len(old.geoms) != len(new.geoms):
                return False
            if sum(len(part.interiors) for part in old.geoms) != sum(len(part.interiors) for part in new.geoms):
                return False

        # Equal contour containment is not sufficient under nonzero winding:
        # reversing a cutout can fill it without moving any boundary points.
        old_filled = to_polygon(current).buffer(0)
        new_filled = to_polygon(candidate).buffer(0)
        if sorted(len(part.interiors) for part in old_filled.geoms) != sorted(len(part.interiors) for part in new_filled.geoms):
            return False

        def relation(a: SkPath, b: SkPath) -> tuple[bool, bool, bool]:
            return (a.difference(b).area <= 1e-9, b.difference(a).area <= 1e-9,
                    a.intersection(b).area > 1e-9)

        return all(
            relation(before[i], before[j]) == relation(after[i], after[j])
            for i in range(len(before)) for j in range(i + 1, len(before))
        )
    except Exception:
        return False


def _path_segment_count(shape: Shape) -> int | None:
    if shape.kind != "path":
        return None
    d = str(shape.params.get("d", ""))
    return _path_segment_count_d(d)


def _path_segment_count_d(d: str) -> int:
    return _command_count(d)


def _path_command_cost(shape: Shape) -> int | None:
    if shape.kind != "path":
        return None
    return _path_command_cost_d(str(shape.params.get("d", "")))


def _path_command_cost_d(d: str) -> int:
    return sum(_COMMAND_COST.get(command, 0) for command in _PATH_COMMAND.findall(d))


def _line_command_count_d(d: str) -> int:
    return sum(1 for command in _PATH_COMMAND.findall(d) if command == "L")


def _is_smoother_curve_join_candidate(current_d: str, candidate_d: str) -> bool:
    if _line_command_count_d(candidate_d) != _line_command_count_d(current_d):
        return False
    if _path_command_cost_d(candidate_d) > _path_command_cost_d(current_d):
        return False
    return _curve_join_angle_penalty_d(candidate_d) + 1e-6 < _curve_join_angle_penalty_d(current_d)


def _geometry_residual(current: Shape, candidate: Shape) -> float:
    from ..vector_region import to_polygon

    try:
        current_geometry = to_polygon(current)
        candidate_geometry = to_polygon(candidate)
        scale = max(float(current_geometry.area), 1.0)
        return float(current_geometry.symmetric_difference(candidate_geometry).area / scale)
    except Exception:
        return float("inf")


def _boundary_max_deviation(current: Shape, candidate: Shape) -> float:
    """Approximate the symmetric boundary Hausdorff distance in pixels.

    Area-only residuals are blind to a long, visibly bowed edge: its displaced
    strip can occupy little area relative to the whole region.  SkPath exposes
    linearised curve samples, so measure the largest sampled boundary movement
    in either direction and keep simplification within its declared fit error.
    """
    from ...quality import symmetric_boundary_distance
    from ..vector_region import to_polygon

    try:
        # Preserve this pass's established bounded 64-sample approximation;
        # acceptance policies use the shared helper's conservative mode.
        return symmetric_boundary_distance(
            to_polygon(current), to_polygon(candidate), max_samples_per_ring=64
        )
    except Exception:
        return float("inf")


def _is_improvement(
    current: Shape,
    candidate: Shape,
    *,
    original: Shape | None = None,
    check_geometry: bool = True,
    epsilon: float = 1.0,
    max_boundary_error: float | None = None,
    linelet_only: bool = False,
    allow_new_lines: bool = False,
) -> bool:
    candidate_segments = _path_segment_count(candidate)
    current_segments = _path_segment_count(current)
    candidate_cost = _path_command_cost(candidate)
    current_cost = _path_command_cost(current)
    if candidate_segments is None or current_segments is None or candidate_cost is None or current_cost is None:
        return False
    if candidate == current:
        return False
    current_short_linelets = _short_linelet_count_d(str(current.params.get("d", "")), epsilon=epsilon)
    candidate_short_linelets = _short_linelet_count_d(str(candidate.params.get("d", "")), epsilon=epsilon)
    if linelet_only and candidate_short_linelets >= current_short_linelets:
        return False
    current_d = str(current.params.get("d", ""))
    candidate_d = str(candidate.params.get("d", ""))
    smoother_curve_join = _is_smoother_curve_join_candidate(current_d, candidate_d)
    if (
        not allow_new_lines
        and
        current_short_linelets == 0
        and _line_command_count_d(current_d) == 0
        and _line_command_count_d(candidate_d) > 0
    ):
        return False
    if not (
        candidate_short_linelets < current_short_linelets
        or candidate_cost < current_cost
        or smoother_curve_join
    ):
        return False
    if check_geometry:
        residual_limit = _MAX_GEOMETRY_RESIDUAL
        if candidate_short_linelets < current_short_linelets:
            residual_limit = _MAX_SHORT_LINELET_GEOMETRY_RESIDUAL
        if (
            current_short_linelets == 0
            and candidate_short_linelets == 0
            and _line_command_count_d(candidate_d) == _line_command_count_d(current_d)
        ):
            residual_limit = max(residual_limit, _MAX_SMOOTH_CURVE_GEOMETRY_RESIDUAL)
        if _geometry_residual(current, candidate) > residual_limit:
            return False
        if max_boundary_error is not None and _boundary_max_deviation(current, candidate) > max_boundary_error:
            return False
    original_segments = _path_segment_count(original) if original is not None else None
    original_cost = _path_command_cost(original) if original is not None else None
    if original_segments is None:
        return True
    if candidate_short_linelets < current_short_linelets:
        return True
    if smoother_curve_join:
        return True
    if original_cost is None:
        return True
    return candidate_cost <= original_cost


def _simplified_path_shape(
    shape: Shape,
    *,
    epsilon: float,
    max_error: float,
    samples: int,
    preserve_subpaths: bool = True,
    cubic: bool = False,
    original: Shape | None = None,
    check_geometry: bool = True,
    linelet_only: bool = False,
    allow_new_lines: bool = False,
    max_boundary_error: float | None = None,
) -> Shape | None:
    if shape.kind != "path":
        return None

    d = str(shape.params.get("d", ""))
    if not d:
        return None

    subpaths = _path_subpaths(shape, samples=samples)
    if not subpaths:
        return None
    parts = _candidate_parts(
        subpaths,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
        preserve_subpaths=preserve_subpaths,
        max_boundary_error=max_boundary_error,
    )
    if not parts:
        return None
    candidate = _candidate_shape(shape, parts, preserve_fill_rule=preserve_subpaths)
    if preserve_subpaths and not _same_subpath_topology(shape, candidate):
        return None
    if not _is_improvement(
        shape,
        candidate,
        original=original,
        check_geometry=check_geometry,
        epsilon=epsilon,
        max_boundary_error=max_boundary_error,
        linelet_only=linelet_only,
        allow_new_lines=allow_new_lines,
    ):
        return None
    return candidate


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

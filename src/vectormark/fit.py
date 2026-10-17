"""C5/C6: primitive recognition (this task) + segment path fitting (next task)."""

from __future__ import annotations

import math
import logging
import re
from dataclasses import dataclass

import numpy as np
from skimage.measure import CircleModel, EllipseModel

from ._fitcurve import (
    _generate_free_cubic,
    cbezier,
    cubic_inflects,
    fit_cubic_beziers,
    fit_cubic_once,
    fit_quadratic_beziers,
    fit_quadratic_once,
)
from .contour import corner_curve_radius_at, corner_curve_span_at, corner_indices, corner_turn_index_at, rdp
from .skia_geometry import SkPath
from .path_fit_evidence import AutomaticFitEvidence, FittedRun, automatic_fit_evidence

_LOGGER = logging.getLogger(__name__)

# Curved runs are fit with quadratic Béziers by default: a parabola cannot
# inflect, so it averages the quantization staircase into smooth convex arcs.
# Cubics (opt-in, see fit_path's `cubic` flag) carry an extra DOF that better
# represents genuinely complex contours, but on a staircased raster that DOF
# just chases the noise (rippled/fragmented edges) — they are a complexity
# tool, not a denoiser. When cubics ARE requested, RDP-denoise each run first
# at sub-pixel tolerance to collapse the staircase before fitting.
PATH_DENOISE_EPS = 0.5
MIN_LINE_LENGTH_FACTOR = 8.0
MIN_SUPPORTED_CORNER_TURN_DEGREES = 10.0
MIN_PROTECTED_POLYLINE_LINE_LENGTH = 64.0
FIT_REFERENCE_CANVAS_EXTENT = 1254.0
SUPPORTED_CORNER_MAX_MEMBER_GAP = 64.0
SUPPORTED_CORNER_MAX_ARC = 128.0
CORNER_LOCAL_TURN_FRACTION = 0.25
CORNER_MAX_SUPPORT_FRACTION = 0.5
CORNERLET_MIN_TURN_DEGREES = 40.0
SHARP_CORNER_TURN_DEGREES = 90.0
SIMPLE_CURVE_RESIDUAL_TOL = 0.032
SIDE_NORMALIZATION_FACTOR = 3.0
_PATH_TOKEN = re.compile(r"[MLQCZ]|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass
class Shape:
    kind: str                 # "circle" | "ellipse" | "rect" | "polygon" | "path"
    params: dict


@dataclass(frozen=True)
class PathFitStages:
    """The independently reviewable stages of a fitted contour.

    ``baseline_d`` is the direct corner-split fit.  ``simple_d`` is the
    residual-gated looser candidate selected by ``prefer_simple_curves``.
    ``atomic_d`` is the final local line/curve cleanup. Keeping these values
    separate makes tracing observable without changing the public ``fit_path``
    result or silently folding an optimizer decision into the raw trace.
    """

    baseline_d: str
    simple_d: str
    atomic_d: str
    baseline_evidence: AutomaticFitEvidence
    # Pre-fit contour corners are the authoritative anchor provenance.  Keep
    # them separate from the emitted SVG so downstream passes never have to
    # guess a corner again from fitted tangents.
    corner_points: tuple[tuple[float, float], ...] = ()
    # Each entry is the pre-fit entry/exit boundary of one rounded corner.
    # The fitted Q/C commands between those points are its members.
    corner_spans: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = ()


@dataclass(frozen=True)
class _AutomaticPathFit:
    d: str
    corner_points: tuple[tuple[float, float], ...]
    corner_spans: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    evidence: AutomaticFitEvidence


@dataclass(frozen=True)
class _DetectedPathFeatures:
    """Pre-fit source boundaries, detached from mutable tracing buffers.

    Fitting cuts include protected lines and non-semantic closure boundaries;
    only point indices and rounded ranges declare corners.
    """

    ring: tuple[tuple[float, float], ...]
    cut_indices: tuple[int, ...]
    point_indices: tuple[int, ...]
    corner_ranges: tuple[tuple[int, int], ...]


def _max_residual(model, pts: np.ndarray) -> float:
    return float(np.abs(model.residuals(pts)).max())


def _circle_residual_within_epsilon(model: CircleModel, pts: np.ndarray, *, epsilon: float) -> bool:
    """Accept a circle despite one quantization outlier, never a broad mismatch."""
    residuals = np.abs(model.residuals(pts))
    # Circle edges often contain one isolated AA stair-step.  Requiring the
    # 99th percentile to meet epsilon accepts that negligible defect while the
    # bounded maximum still rejects a clipped circle, flat dome, or connector.
    return bool(
        np.quantile(residuals, 0.99) <= epsilon
        and residuals.max() <= epsilon + max(0.25, epsilon * 0.25)
    )


def _estimate_ellipse(pts: np.ndarray) -> EllipseModel | None:
    """Reject a numerically unusable optional fit, not unrelated code errors."""
    reason = None
    try:
        model = EllipseModel.from_estimate(pts)
    except TypeError as error:
        # skimage 0.26 may retain a complex eigenvector dtype until phi %= pi.
        # Do not catch arbitrary TypeErrors from the estimator or residual code.
        message = str(error)
        if "unsupported operand type(s) for %=" not in message or "'numpy.complex" not in message:
            raise
        model = None
        reason = "complex_parameters"
    if model is not None:
        parameters = np.asarray([*model.center, *model.axis_lengths, model.theta])
        if (
            not np.isrealobj(parameters)
            or not np.isfinite(parameters).all()
            or np.any(parameters[2:4] <= 0)
        ):
            reason = "invalid_parameters"
    if reason is not None:
        _LOGGER.warning(
            "Rejected ellipse candidate: %s", reason,
            extra={"primitive": "ellipse", "reason": reason},
        )
        return None
    return model


def recognize_primitive(contour: np.ndarray, *, epsilon: float) -> Shape | None:
    """Return a native-primitive Shape if `contour` matches one within ε, else None."""
    pts = np.asarray(contour, dtype=float)
    # A closed four-edge rectangle has five samples including its repeated
    # start point.  It is still a perfectly valid primitive candidate even
    # though circle/ellipse estimation needs more observations.
    if len(pts) < 4:
        return None
    poly = SkPath(shell=list(map(tuple, pts)))
    normalized = poly.buffer(0)
    if normalized.is_empty or poly.area < 1:
        return None
    if abs(normalized.area - poly.area) / max(poly.area, 1.0) > 0.02:
        return None

    if len(pts) >= 8:
        # circle
        cm = CircleModel.from_estimate(pts)
        if cm and _circle_residual_within_epsilon(cm, pts, epsilon=epsilon):
            xc, yc = cm.center
            return Shape("circle", {"cx": xc, "cy": yc, "r": cm.radius})

        # ellipse (axis-aligned check: snap small thetas to 0 for symmetric output)
        em = _estimate_ellipse(pts)
        if em:
            xc, yc = em.center
            a, b = em.axis_lengths
            theta = em.theta
            if _max_residual(em, pts) <= epsilon and (abs(theta) < 0.08 or abs(abs(theta) - np.pi) < 0.08):
                return Shape("ellipse", {"cx": xc, "cy": yc, "rx": a, "ry": b})

    # axis-aligned rectangle: bbox fill ratio near 1, rotated-rect ~ axis-aligned,
    # AND genuinely four sharp corners. The corner check is what stops a *rounded
    # trapezoid* (tapered sides + filleted corners) from being flattened to a
    # rect — those simplify to >4 vertices and fall through to the symmetry-locked
    # path fit, which keeps their true shape.
    minx, miny, maxx, maxy = poly.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    rot = poly.minimum_rotated_rectangle
    rx, ry = rot.exterior.xy
    edge_angles = np.arctan2(np.diff(ry), np.diff(rx))
    axis_aligned = np.all(np.minimum(np.abs(edge_angles % (np.pi / 2)),
                                     np.pi / 2 - np.abs(edge_angles % (np.pi / 2))) < 0.06)
    simp = rdp(pts, epsilon)
    if np.allclose(simp[0], simp[-1]):
        simp = simp[:-1]
    four_corners = len(simp) == 4
    if bbox_area > 0 and poly.area / bbox_area > 0.985 and axis_aligned and four_corners:
        return Shape("rect", {"x": minx, "y": miny, "w": maxx - minx, "h": maxy - miny})

    return None


def recognize_polygon(contour: np.ndarray, *, epsilon: float, max_vertices: int = 8) -> Shape | None:
    """Emit a <polygon> when the contour simplifies to few straight edges."""
    pts = np.asarray(contour, dtype=float)
    if len(pts) < 3:
        return None
    simp = rdp(pts, epsilon)
    if np.allclose(simp[0], simp[-1]):
        simp = simp[:-1]
    if not (3 <= len(simp) <= max_vertices):
        return None
    # every original point must lie within ε of the simplified polygon edges
    if _max_point_to_polyline(pts, np.vstack([simp, simp[0]])) > epsilon:
        return None
    return Shape("polygon", {"points": [(float(x), float(y)) for x, y in simp]})


def _max_point_to_polyline(pts: np.ndarray, poly: np.ndarray) -> float:
    worst = 0.0
    segs = np.stack([poly[:-1], poly[1:]], axis=1)
    for p in pts:
        d = min(_point_seg_dist(p, s[0], s[1]) for s in segs)
        worst = max(worst, d)
    return worst


def _point_seg_dist(p, a, b) -> float:
    ab = b - a
    t = 0.0 if np.dot(ab, ab) == 0 else np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.hypot(*(p - (a + t * ab))))


def _segment_is_straight(seg: np.ndarray, epsilon: float) -> bool:
    if len(seg) <= 2:
        return True
    return _max_point_to_polyline(seg, np.vstack([seg[0], seg[-1]])) <= epsilon


def _segment_length(seg: np.ndarray) -> float:
    return float(np.linalg.norm(seg[-1] - seg[0]))


def minimum_line_length(epsilon: float) -> float:
    return max(4.0, epsilon * MIN_LINE_LENGTH_FACTOR)


def _protected_line_spans(ring: np.ndarray, *, canvas_scale: float = 1.0) -> list[tuple[int, int]]:
    """Return only literal long straight runs already present in the polyline.

    This intentionally does *not* fit an RDP chord as a line.  A protected
    line is an existing contiguous collinear run in the traced polyline, long
    enough to be a deliberate graphic edge rather than a staircase fragment
    or an approximation of an organic contour.
    """
    if len(ring) < 2:
        return []
    points = np.vstack([ring, ring[0]])
    edges = np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    directions = [
        _unit_vector(edge) if length > 1e-9 else None
        for edge, length in zip(edges, lengths, strict=True)
    ]
    if not any(direction is not None for direction in directions):
        return []

    def same_direction(left: np.ndarray | None, right: np.ndarray | None) -> bool:
        return left is not None and right is not None and bool(np.allclose(left, right, atol=1e-9))

    # Start immediately after a direction break so a literal run crossing the
    # closed-ring seam is considered as one run rather than two fragments.
    count = len(edges)
    start = next(
        (index for index in range(count) if not same_direction(directions[index - 1], directions[index])),
        0,
    )
    spans: list[tuple[int, int]] = []
    offset = 0
    while offset < count:
        run_start = (start + offset) % count
        direction = directions[run_start]
        run_length = 0.0
        run_edges = 0
        while offset + run_edges < count:
            current = (start + offset + run_edges) % count
            if not same_direction(direction, directions[current]):
                break
            run_length += float(lengths[current])
            run_edges += 1
        # Subpixel coverage can trim one pixel from each end of an otherwise
        # literal straight side. Do not require a 64px design edge to retain
        # all 64px after that coverage transition.
        if run_length >= max(4.0, MIN_PROTECTED_POLYLINE_LINE_LENGTH * canvas_scale - 2.0):
            spans.append((run_start, (run_start + run_edges) % count))
        offset += max(run_edges, 1)
    return spans


def _inside_closed_span(index: int, start: int, end: int) -> bool:
    """Whether ``index`` is strictly inside the forward closed-ring span."""
    if start < end:
        return start < index < end
    return index > start or index < end


def _suppress_cornerlet_vertices(
    points: np.ndarray,
    *,
    min_adjacent_length: float,
    max_deviation: float | None = None,
    closed: bool = True,
) -> np.ndarray:
    """Remove short, high-turn analysis vertices without changing the contour.

    RDP can retain a sub-pixel AA step beside a long design edge. Merely
    rejecting that vertex as a corner leaves it in place to poison the support
    measurement of its neighbours. Collapse only vertices that combine a
    short side with a corner-like turn. When a deviation bound is supplied,
    check the complete removed interval so repeated suppression cannot replace
    a smooth arc with a sharp polygon.
    """
    pts = np.asarray(points, dtype=float)
    return pts[_cornerlet_retained_indices(
        pts, min_adjacent_length=min_adjacent_length, closed=closed,
        max_deviation=max_deviation,
    )]


def _cornerlet_retained_indices(
    points: np.ndarray, *, min_adjacent_length: float, closed: bool,
    max_deviation: float | None = None,
) -> np.ndarray:
    """Share cornerlet classification while protecting open-span endpoints."""
    retained = list(range(len(points)))
    changed = True
    while changed and len(retained) > (3 if closed else 2):
        changed = False
        for index in range(len(retained)) if closed else range(1, len(retained) - 1):
            previous = points[retained[(index - 1) % len(retained)]]
            current = points[retained[index]]
            following = points[retained[(index + 1) % len(retained)]]
            incoming = current - previous
            outgoing = following - current
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if min(incoming_length, outgoing_length) >= min_adjacent_length:
                continue
            if incoming_length <= 1e-9 or outgoing_length <= 1e-9:
                retained.pop(index)
                changed = True
                break
            turn = abs(float(np.degrees(np.arctan2(
                incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
                np.dot(incoming, outgoing),
            ))))
            if turn < CORNERLET_MIN_TURN_DEGREES:
                continue
            if max_deviation is not None:
                start = retained[(index - 1) % len(retained)]
                end = retained[(index + 1) % len(retained)]
                # Include every original vertex already removed between these
                # neighbours, not only the current vertex. Closed intervals
                # may wrap through the contour's starting point.
                interval = (
                    points[start:end + 1] if end > start
                    else np.vstack((points[start:], points[:end + 1]))
                )
                if not _segment_is_straight(interval, max_deviation):
                    continue
            retained.pop(index)
            changed = True
            break
    return np.asarray(retained, dtype=np.intp)


def _cornerlet_fit_indices(points: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Omit noisy source intervals, not ordinary samples or declared cuts.

    Analysis simplification only locates short high-turn deviations. Replace
    an interval only if every original sample stays within the existing trace
    tolerance of its surviving chord. Returned indices refer to original data.
    """
    keep = np.ones(len(points), dtype=bool)
    if len(points) < 3 or epsilon <= 0:
        return np.flatnonzero(keep)
    analysis = rdp(points, min(PATH_DENOISE_EPS, epsilon))
    retained = _cornerlet_retained_indices(
        analysis, min_adjacent_length=minimum_line_length(epsilon), closed=False,
    )
    if len(retained) == len(analysis):
        return np.flatnonzero(keep)
    source_indices: list[int] = []
    cursor = 0
    for point in analysis:
        matches = np.flatnonzero(np.all(points[cursor:] == point, axis=1))
        if not len(matches):
            raise RuntimeError("cornerlet analysis lost source correspondence")
        index = cursor + int(matches[0])
        source_indices.append(index)
        cursor = index + 1
    for left, right in zip(retained[:-1], retained[1:], strict=True):
        if right == left + 1:
            continue
        start, end = source_indices[left], source_indices[right]
        if _segment_is_straight(points[start:end + 1], epsilon):
            keep[start + 1:end] = False
    return np.flatnonzero(keep)


def _advance_ring_index(ring: np.ndarray, index: int, step: int, distance: float) -> int:
    """Walk a closed contour by arc length without inventing an anchor point."""
    remaining = float(distance)
    current = index
    count = len(ring)
    while remaining > 1e-9:
        following = (current + step) % count
        edge = ring[following] - ring[current]
        length = float(np.linalg.norm(edge))
        if length >= remaining:
            return following
        remaining -= length
        current = following
    return current


def _arc_distance(ring: np.ndarray, start: int, end: int) -> float:
    """Forward closed-ring arc distance from ``start`` to ``end``."""
    total = 0.0
    current = start
    while current != end:
        following = (current + 1) % len(ring)
        total += float(np.linalg.norm(ring[following] - ring[current]))
        current = following
    return total


def _protected_corner_ranges(
    ring: np.ndarray,
    corners: set[int],
    *,
    epsilon: float,
    canvas_scale: float = 1.0,
) -> list[tuple[int, int]]:
    """Expand clustered pre-fit cornerlets into protected contour intervals.

    A rounded corner is often represented by two or more nearby high-turn
    samples.  Treating each as a separate cut leaks its transition samples into
    the neighboring side.  This groups only such adjacent detections, expands
    their two outer boundaries along existing raw samples, and leaves isolated
    sharp corners untouched. Coarse recovery samples are only clustering
    evidence: one isolated sample is not enough to declare another corner.
    """
    if len(corners) < 2:
        return []
    ordered = sorted(corners)
    group_gap = max(12.0, epsilon * 8.0) * canvas_scale
    groups: list[list[int]] = [[ordered[0]]]
    for index in ordered[1:]:
        if _arc_distance(ring, groups[-1][-1], index) <= group_gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    if len(groups) > 1 and _arc_distance(ring, groups[-1][-1], groups[0][0]) <= group_gap:
        groups[0] = groups[-1] + groups[0]
        groups.pop()

    guard = max(4.0, epsilon * 4.0)
    ranges: list[tuple[int, int]] = []
    for group in groups:
        if len(group) < 2:
            continue
        start = _advance_ring_index(ring, group[0], -1, guard)
        end = _advance_ring_index(ring, group[-1], 1, guard)
        ranges.append((start, end))
    return ranges


def _merge_corner_ranges(
    ring: np.ndarray,
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Union overlapping forward corner spans without inventing boundaries."""
    if not ranges:
        return []
    count = len(ring)
    covered: set[int] = set()
    for start, end in ranges:
        current = start
        covered.add(current)
        while current != end:
            current = (current + 1) % count
            covered.add(current)
    if len(covered) == count:
        return [(0, count - 1)]

    ordered = sorted(covered)
    groups: list[list[int]] = [[ordered[0]]]
    for index in ordered[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == count - 1:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    return [(group[0], group[-1]) for group in groups]


def _supported_corner_ranges(
    ring: np.ndarray,
    simplified: np.ndarray,
    *,
    epsilon: float,
    min_turn_degrees: float,
    min_support_length: float,
    canvas_scale: float = 1.0,
) -> list[tuple[int, int]]:
    """Find localized accumulated turns beside a credible supporting edge.

    Scan the normal-resolution polyline directly, collect adjacent turns with
    the same orientation, and retain the interval only when it exceeds the
    caller's turn threshold and has credible exterior supporting edges. That
    supported neighborhood is evidence, not the corner itself: low-turn valleys
    separate local corners, and a corner must be compact relative to its support.
    A broad, gently curving side remains ordinary path geometry.
    """
    points = np.asarray(simplified, dtype=float)
    if len(points) < 3:
        return []

    count = len(points)
    turns: list[float] = []
    for index in range(count):
        previous = points[(index - 1) % count]
        current = points[index]
        following = points[(index + 1) % count]
        incoming = current - previous
        outgoing = following - current
        if np.linalg.norm(incoming) <= 1e-9 or np.linalg.norm(outgoing) <= 1e-9:
            turns.append(0.0)
            continue
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        turns.append(float(np.degrees(np.arctan2(cross, np.dot(incoming, outgoing)))))

    ranges: list[tuple[int, int]] = []
    claimed_members: set[int] = set()
    guard = max(4.0, epsilon * 4.0)
    for anchor in sorted(range(count), key=lambda index: -abs(turns[index])):
        if anchor in claimed_members:
            continue
        anchor_turn = turns[anchor]
        if abs(anchor_turn) < 1e-9:
            continue
        orientation = 1.0 if anchor_turn > 0.0 else -1.0
        directional_members: dict[int, list[int]] = {-1: [], 1: []}
        visited = {anchor}

        for step in (-1, 1):
            current = anchor
            travelled = 0.0
            while True:
                candidate = (current + step) % count
                # A uniformly curved closed outline can keep the same turn
                # orientation all the way around the ring. Never let one
                # directional search wrap through the anchor and invert the
                # start/end interval assembled below.
                if candidate in visited:
                    break
                edge_length = float(np.linalg.norm(points[candidate] - points[current]))
                if (
                    edge_length > SUPPORTED_CORNER_MAX_MEMBER_GAP * canvas_scale
                    or travelled + edge_length > SUPPORTED_CORNER_MAX_ARC * canvas_scale
                    or turns[candidate] * orientation <= 0.0
                ):
                    break
                travelled += edge_length
                current = candidate
                directional_members[step].append(candidate)
                visited.add(candidate)

        members = [*reversed(directional_members[-1]), anchor, *directional_members[1]]
        start, end = members[0], members[-1]
        if any(index in claimed_members for index in members):
            continue
        member_turns = [abs(turns[index]) for index in members]
        accumulated_turn = sum(member_turns)
        incoming_support = float(np.linalg.norm(points[start] - points[(start - 1) % count]))
        outgoing_support = float(np.linalg.norm(points[(end + 1) % count] - points[end]))
        if (
            accumulated_turn < min_turn_degrees
            or min(incoming_support, outgoing_support) < min_support_length
        ):
            continue

        claimed_members.update(members)

        # Explaining a percentage of the *whole* neighborhood's turn can join
        # two real corners across an ordinary curved side. Split at weak-turn
        # valleys instead. Claim the neighborhood above so a discarded valley
        # cannot be reconsidered as a new, weaker corner on the next iteration.
        local_groups: list[list[int]] = []
        group: list[int] = []
        # A stronger neighboring corner must not suppress a shallower turn
        # that independently meets the caller's detection threshold.
        turn_floor = min(min_turn_degrees, max(member_turns) * CORNER_LOCAL_TURN_FRACTION)
        for index in members:
            if abs(turns[index]) >= turn_floor:
                group.append(index)
            elif group:
                local_groups.append(group)
                group = []
        if group:
            local_groups.append(group)

        max_extent = min(incoming_support, outgoing_support) * CORNER_MAX_SUPPORT_FRACTION
        for group in local_groups:
            if sum(abs(turns[index]) for index in group) < min_turn_degrees:
                continue
            raw_start = int(np.argmin(np.hypot(*(ring - points[group[0]]).T)))
            raw_end = int(np.argmin(np.hypot(*(ring - points[group[-1]]).T)))
            # Uniform curvature has no valley. Require spatial concentration
            # too, so a long arc between straight supports is not one corner.
            extent = _arc_distance(ring, raw_start, raw_end) if raw_start != raw_end else 0.0
            if extent + 2.0 * guard > max_extent:
                continue
            ranges.append((
                _advance_ring_index(ring, raw_start, -1, guard),
                _advance_ring_index(ring, raw_end, 1, guard),
            ))
    return list(dict.fromkeys(ranges))


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _path_command_count(d: str) -> int:
    return sum(1 for char in d if char in "MLQCAZ")


def _parse_path_commands(d: str) -> list[tuple[str, list[float]]] | None:
    raw = _PATH_TOKEN.findall(d)
    if not raw:
        return None
    coord_count = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}
    commands: list[tuple[str, list[float]]] = []
    i = 0
    while i < len(raw):
        command = raw[i]
        if command not in coord_count:
            return None
        i += 1
        count = coord_count[command]
        values = [float(token) for token in raw[i:i + count]]
        if len(values) != count:
            return None
        i += count
        commands.append((command, values))
    return commands


def _closed_ring_points(pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    if np.allclose(pts[0], pts[-1]):
        return pts[:-1]
    return pts


def _path_area_residual(contour: np.ndarray, d: str) -> float:
    ring = _closed_ring_points(np.asarray(contour, dtype=float))
    if len(ring) < 3:
        return float("inf")
    try:
        source = SkPath(shell=list(map(tuple, ring)))
        fitted = SkPath.from_svg_d(d)
        if source.is_empty or fitted.is_empty:
            return float("inf")
        scale = max(float(source.area), float(fitted.area), 1.0)
        return float(source.symmetric_difference(fitted).area / scale)
    except Exception:
        return float("inf")


def _curve_controls_are_straight(start: np.ndarray, controls: list[np.ndarray], end: np.ndarray, epsilon: float) -> bool:
    if float(np.linalg.norm(end - start)) < minimum_line_length(epsilon):
        return False
    return _max_point_to_polyline(np.asarray(controls, dtype=float), np.vstack([start, end])) <= epsilon


def _unit_vector(v: np.ndarray) -> np.ndarray | None:
    n = float(np.hypot(v[0], v[1]))
    if n <= 1e-9:
        return None
    return np.asarray(v, dtype=float) / n


# Joins below this are intentional corners more often than trace noise.  A
# right angle remains protected, while shallow near-corners can be flattened.
_SMOOTH_MIN_INTERIOR_ANGLE_DEGREES = 120.0
_SMOOTH_MAX_NODE_SHIFT = 1.0


def _curve_endpoint(command: str, values: list[float]) -> np.ndarray | None:
    if command == "Q":
        return np.array(values[2:4], dtype=float)
    if command == "C":
        return np.array(values[4:6], dtype=float)
    return None


def _curve_endpoint_slot(command: str, values: list[float]) -> tuple[np.ndarray, int] | None:
    if command == "Q":
        return np.array(values[2:4], dtype=float), 2
    if command == "C":
        return np.array(values[4:6], dtype=float), 4
    return None


def _curve_end_control(command: str, values: list[float]) -> tuple[np.ndarray, int] | None:
    if command == "Q":
        return np.array(values[0:2], dtype=float), 0
    if command == "C":
        return np.array(values[2:4], dtype=float), 2
    return None


def _curve_start_control(command: str, values: list[float]) -> tuple[np.ndarray, int] | None:
    if command in {"Q", "C"}:
        return np.array(values[0:2], dtype=float), 0
    return None


def _project_onto_line(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray | None:
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 1e-9:
        return None
    return start + direction * float(np.dot(point - start, direction) / length_squared)


def _smooth_curve_path_d(d: str) -> str:
    commands = _parse_path_commands(d)
    if commands is None:
        return d

    # Keep the unmodified start point of every command so a quadratic can be
    # degree-elevated exactly if its single control is claimed by both joins.
    starts: list[np.ndarray | None] = []
    cursor: np.ndarray | None = None
    subpath_start: np.ndarray | None = None
    for command, values in commands:
        starts.append(None if cursor is None else cursor.copy())
        if command == "M":
            cursor = np.asarray(values[:2], dtype=float)
            subpath_start = cursor.copy()
        elif command == "L":
            cursor = np.asarray(values[:2], dtype=float)
        elif command == "Q":
            cursor = np.asarray(values[2:4], dtype=float)
        elif command == "C":
            cursor = np.asarray(values[4:6], dtype=float)
        elif command == "Z":
            cursor = None if subpath_start is None else subpath_start.copy()

    # Compute every target from the unmodified command list.  A Q has a single
    # control shared by its two joins, while a C has distinct start/end control
    # slots; keying targets by slot lets C joins smooth independently.
    proposals: list[
        tuple[
            dict[tuple[int, int], np.ndarray],
            tuple[tuple[int, int], ...],
            tuple[int, int],
            np.ndarray | None,
            int,
        ]
    ] = []
    for i in range(len(commands) - 1):
        command, values = commands[i]
        next_command, next_values = commands[i + 1]
        if command not in {"Q", "C"} or next_command not in {"Q", "C", "L"}:
            continue
        endpoint = _curve_endpoint_slot(command, values)
        previous = _curve_end_control(command, values)
        following = _curve_start_control(next_command, next_values)
        if endpoint is None or previous is None:
            continue
        join, endpoint_offset = endpoint
        previous_control, previous_offset = previous
        next_control = np.array(next_values[0:2], dtype=float) if following is None else following[0]
        incoming = join - previous_control
        outgoing = next_control - join
        in_dir = _unit_vector(incoming)
        out_dir = _unit_vector(outgoing)
        if in_dir is None or out_dir is None:
            continue
        # ``in_dir`` and ``out_dir`` agree for a straight continuation.  The
        # corresponding path interior angle is 180° minus their turn angle;
        # avoid smoothing any join sharper than 135° so fitted corners retain
        # their intentional geometry.
        min_dot = math.cos(math.radians(180.0 - _SMOOTH_MIN_INTERIOR_ANGLE_DEGREES))
        # Include the exact 135° boundary despite float round-off.
        if float(np.dot(in_dir, out_dir)) + 1e-9 < min_dot:
            continue
        tangent = out_dir if next_command == "L" else _unit_vector(in_dir + out_dir)
        if tangent is None:
            continue
        prev_control = join - tangent * float(np.hypot(incoming[0], incoming[1]))
        targets = {(i, previous_offset): prev_control}
        if following is not None:
            _control, next_offset = following
            targets[(i + 1, next_offset)] = join + tangent * float(np.hypot(outgoing[0], outgoing[1]))
        node_target = _project_onto_line(join, previous_control, next_control)
        proposals.append((targets, tuple(targets), (i, endpoint_offset), node_target, i))

    control_use_count: dict[tuple[int, int], int] = {}
    for _targets, controls, _node_slot, _node_target, _join_index in proposals:
        for control in controls:
            control_use_count[control] = control_use_count.get(control, 0) + 1

    # A quadratic has one control for both of its joins.  If both joins ask
    # for a tangent projection, elevate the Q to its exact cubic equivalent
    # and give each side its own requested handle.  This replaces the old node
    # projection, which could flatten or move the path rather than preserving
    # the fitted chain.
    requests: dict[tuple[int, str], np.ndarray] = {}
    for targets, _controls, _node_slot, _node_target, join_index in proposals:
        if commands[join_index][0] == "Q" and (join_index, 0) in targets:
            requests[(join_index, "end")] = targets[(join_index, 0)]
        next_index = join_index + 1
        if next_index < len(commands) and commands[next_index][0] == "Q" and (next_index, 0) in targets:
            requests[(next_index, "start")] = targets[(next_index, 0)]

    promoted: set[int] = set()
    for index in sorted({command_index for command_index, _side in requests}):
        start_target = requests.get((index, "start"))
        end_target = requests.get((index, "end"))
        if start_target is None or end_target is None:
            continue
        command, values = commands[index]
        start = starts[index]
        if command != "Q" or start is None:
            continue
        control = np.asarray(values[:2], dtype=float)
        end = np.asarray(values[2:4], dtype=float)
        # Exact quadratic-to-cubic degree elevation.  Apply the two requested
        # join handles after that conversion, never by moving either endpoint.
        first = start + (control - start) * (2.0 / 3.0)
        second = end + (control - end) * (2.0 / 3.0)
        elevated = [
            float(first[0]),
            float(first[1]),
            float(second[0]),
            float(second[1]),
            float(end[0]),
            float(end[1]),
        ]
        elevated[:2] = [float(start_target[0]), float(start_target[1])]
        elevated[2:4] = [float(end_target[0]), float(end_target[1])]
        commands[index] = (
            "C",
            elevated,
        )
        promoted.add(index)
    for targets, controls, node_slot, node_target, _join_index in proposals:
        if any(command_index in promoted for command_index, _offset in controls):
            continue
        if all(control_use_count[control] == 1 for control in controls):
            for control, target in targets.items():
                command_index, offset = control
                _command, values = commands[command_index]
                values[offset:offset + 2] = [
                    float(target[0]),
                    float(target[1]),
                ]
            continue
        # Never resolve a shared-control conflict by moving the join node: it
        # changes the outline.  A Q with two join requests was elevated above;
        # any other conflict is left untouched.
        del node_slot, node_target

    parts: list[str] = []
    for command, values in commands:
        if command == "Z":
            parts.append("Z")
        else:
            parts.append(f"{command}{' '.join(_fmt(value) for value in values)}")
    return " ".join(parts)


def _smooth_quadratic_path_d(d: str) -> str:
    """Backward-compatible name for the now curve-generic join smoother."""
    return _smooth_curve_path_d(d)


def _curve_join_min_dot_d(d: str) -> float:
    commands = _parse_path_commands(d)
    if commands is None:
        return 1.0
    dots: list[float] = []
    for i in range(len(commands) - 1):
        command, values = commands[i]
        next_command, next_values = commands[i + 1]
        if command not in {"Q", "C"} or next_command not in {"Q", "C", "L"}:
            continue
        join = _curve_endpoint(command, values)
        previous = _curve_end_control(command, values)
        following = _curve_start_control(next_command, next_values)
        if join is None or previous is None:
            continue
        outgoing_point = np.array(next_values[0:2], dtype=float) if following is None else following[0]
        in_dir = _unit_vector(join - previous[0])
        out_dir = _unit_vector(outgoing_point - join)
        if in_dir is not None and out_dir is not None:
            dots.append(float(np.dot(in_dir, out_dir)))
    return min(dots) if dots else 1.0


def _append_quadratic_run(
    d: str,
    seg: np.ndarray,
    max_error: float,
    *,
    line_epsilon: float,
    collapse_straight: bool = True,
    endpoints: list[np.ndarray | None] | None = None,
) -> str:
    curves = fit_quadratic_beziers(seg, max_error)
    for b in curves:
        if endpoints is not None:
            endpoints.append(b[2].copy())
        if collapse_straight and _curve_controls_are_straight(b[0], [b[1]], b[2], line_epsilon):
            d += f"L{_fmt(b[2][0])} {_fmt(b[2][1])} "
        else:
            d += f"Q{_fmt(b[1][0])} {_fmt(b[1][1])} {_fmt(b[2][0])} {_fmt(b[2][1])} "
    return d


def _atomic_simplify_line_pairs_d(d: str) -> str:
    """Collapse a shallow consecutive line pair into one line or quadratic.

    The point shared by two line commands is a removable node above 160°.
    Between 135° and 160° it remains useful evidence of a bend, but a single
    quadratic represents that bend without the visible corner.
    """
    commands = _parse_path_commands(d)
    if commands is None:
        return d

    def endpoint(command: str, values: list[float]) -> np.ndarray | None:
        if command in {"M", "L"}:
            return np.asarray(values[:2], dtype=float)
        if command == "Q":
            return np.asarray(values[2:4], dtype=float)
        if command == "C":
            return np.asarray(values[4:6], dtype=float)
        return None

    def format_commands(items: list[tuple[str, list[float]]]) -> str:
        return " ".join("Z" if command == "Z" else f"{command}{' '.join(_fmt(value) for value in values)}" for command, values in items)

    changed = True
    current = commands
    while changed:
        changed = False
        output: list[tuple[str, list[float]]] = []
        cursor: np.ndarray | None = None
        i = 0
        while i < len(current):
            command, values = current[i]
            if command == "M":
                output.append((command, values))
                cursor = endpoint(command, values)
                i += 1
                continue
            if (
                command == "L"
                and cursor is not None
                and i + 1 < len(current)
                and current[i + 1][0] == "L"
            ):
                middle = np.asarray(values[:2], dtype=float)
                end = np.asarray(current[i + 1][1][:2], dtype=float)
                incoming = _unit_vector(middle - cursor)
                outgoing = _unit_vector(end - middle)
                if incoming is not None and outgoing is not None:
                    turn = math.degrees(math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))))
                    interior_angle = 180.0 - turn
                    if interior_angle > 160.0:
                        output.append(("L", [float(end[0]), float(end[1])]))
                        cursor = end
                        i += 2
                        changed = True
                        continue
                    if interior_angle > 135.0:
                        output.append(("Q", [float(middle[0]), float(middle[1]), float(end[0]), float(end[1])]))
                        cursor = end
                        i += 2
                        changed = True
                        continue
            output.append((command, values))
            point = endpoint(command, values)
            if point is not None:
                cursor = point
            i += 1
        current = output
    return format_commands(current)


def atomic_flatten_path_d(d: str, *, epsilon: float) -> str:
    """Apply endpoint-preserving terminal path cleanup.

    A quadratic whose maximum deviation from its endpoint chord is within
    ``epsilon`` is just a line with a redundant control point.  Once those
    redundant quadratics have become lines, the existing shallow ``L + L``
    transform can reduce the newly adjacent pair to a line, a single quadratic,
    or leave the corner intact according to its 160°/135° bands.
    """
    commands = _parse_path_commands(d)
    if commands is None:
        return d

    starts: list[np.ndarray | None] = []
    cursor: np.ndarray | None = None
    subpath_start: np.ndarray | None = None
    for command, values in commands:
        starts.append(None if cursor is None else cursor.copy())
        if command == "M":
            cursor = np.asarray(values[:2], dtype=float)
            subpath_start = cursor.copy()
        elif command == "L":
            cursor = np.asarray(values[:2], dtype=float)
        elif command == "Q":
            cursor = np.asarray(values[2:4], dtype=float)
        elif command == "C":
            cursor = np.asarray(values[4:6], dtype=float)
        elif command == "A":
            cursor = np.asarray(values[5:7], dtype=float)
        elif command == "Z":
            cursor = None if subpath_start is None else subpath_start.copy()

    def tangent(index: int, *, entering: bool) -> np.ndarray | None:
        command, values = commands[index]
        start = starts[index]
        if start is None:
            return None
        if command == "L":
            return np.asarray(values[:2], dtype=float) - start
        if command == "Q":
            control = np.asarray(values[:2], dtype=float)
            end = np.asarray(values[2:4], dtype=float)
            preferred = end - control if entering else control - start
            fallback = end - start
            return preferred if _unit_vector(preferred) is not None else fallback
        if command == "C":
            first = np.asarray(values[:2], dtype=float)
            second = np.asarray(values[2:4], dtype=float)
            end = np.asarray(values[4:6], dtype=float)
            preferred = end - second if entering else first - start
            fallback = end - start
            return preferred if _unit_vector(preferred) is not None else fallback
        return None

    def sharp_join(before: np.ndarray | None, after: np.ndarray | None) -> bool:
        incoming = _unit_vector(before) if before is not None else None
        outgoing = _unit_vector(after) if after is not None else None
        if incoming is None or outgoing is None:
            return False
        turn = math.degrees(math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))))
        return 180.0 - turn < 135.0

    def neighboring_segment(index: int, step: int) -> int | None:
        candidate = index + step
        while 0 <= candidate < len(commands):
            command = commands[candidate][0]
            if command in {"M", "Z"}:
                return None
            if command in {"L", "Q", "C"}:
                return candidate
            candidate += step
        return None

    flattened: list[tuple[str, list[float]]] = []
    for index, (command, values) in enumerate(commands):
        if command == "M":
            flattened.append((command, values))
            continue
        current = starts[index]
        if command == "Q" and current is not None:
            control = np.asarray(values[:2], dtype=float)
            end = np.asarray(values[2:4], dtype=float)
            chord = end - current
            chord_length = float(np.linalg.norm(chord))
            if chord_length > 1e-9:
                # B(1/2) is displaced from the chord by exactly half the
                # control point's perpendicular displacement.
                control_distance = abs(float(np.cross(chord, control - current))) / chord_length
                previous = neighboring_segment(index, -1)
                following = neighboring_segment(index, 1)
                between_sharp_corners = (
                    previous is not None
                    and following is not None
                    and sharp_join(tangent(previous, entering=True), tangent(index, entering=False))
                    and sharp_join(tangent(index, entering=True), tangent(following, entering=False))
                )
                if control_distance * 0.5 <= epsilon and between_sharp_corners:
                    flattened.append(("L", [float(end[0]), float(end[1])]))
                    continue
            flattened.append((command, values))
            continue
        flattened.append((command, values))

    def format_commands(items: list[tuple[str, list[float]]]) -> str:
        return " ".join(
            "Z" if command == "Z" else f"{command}{' '.join(_fmt(value) for value in values)}"
            for command, values in items
        )

    return _atomic_simplify_line_pairs_d(format_commands(flattened))


def _residual_gated_atomic_simplify_d(contour: np.ndarray, d: str) -> str:
    simplified = _atomic_simplify_line_pairs_d(d)
    if _path_area_residual(contour, simplified) <= max(SIMPLE_CURVE_RESIDUAL_TOL, _path_area_residual(contour, d) + 0.005):
        return simplified
    return d


def _append_cubic_run(d: str, seg: np.ndarray, max_error: float, *, line_epsilon: float, endpoints: list[np.ndarray | None] | None = None) -> str:
    run = rdp(seg, PATH_DENOISE_EPS) if len(seg) > 2 else seg
    current = np.array([float(_fmt(seg[0][0])), float(_fmt(seg[0][1]))], dtype=float)
    for b in fit_cubic_beziers(run, max_error):
        rounded = np.array(
            [
                current,
                [float(_fmt(b[1][0])), float(_fmt(b[1][1]))],
                [float(_fmt(b[2][0])), float(_fmt(b[2][1]))],
                [float(_fmt(b[3][0])), float(_fmt(b[3][1]))],
            ],
            dtype=float,
        )
        if cubic_inflects(rounded):
            samples = np.asarray([cbezier(b, t) for t in np.linspace(0.0, 1.0, 9)], dtype=float)
            replacement = _append_quadratic_run(
                "",
                samples,
                max_error,
                line_epsilon=line_epsilon,
                collapse_straight=False,
            )
            d += replacement
            if endpoints is not None:
                # Internal endpoints sampled on a fitted C are synthetic, not
                # authoritative source nodes. Only its original end is bound.
                count = len(re.findall(r"[LQ]", replacement))
                endpoints.extend([None] * (count - 1))
                endpoints.append(b[3].copy())
            current = np.array([float(_fmt(samples[-1][0])), float(_fmt(samples[-1][1]))], dtype=float)
            continue
        d += (
            f"C{_fmt(rounded[1][0])} {_fmt(rounded[1][1])} {_fmt(rounded[2][0])} {_fmt(rounded[2][1])} "
            f"{_fmt(rounded[3][0])} {_fmt(rounded[3][1])} "
        )
        if endpoints is not None:
            endpoints.append(b[3].copy())
        current = rounded[3]
    return d


def _chord_parameters(points: np.ndarray) -> np.ndarray:
    lengths = np.zeros(len(points), dtype=float)
    lengths[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    return lengths / lengths[-1] if lengths[-1] > 1e-9 else lengths


def _fit_free_cubic_once(points: np.ndarray) -> np.ndarray | None:
    """Fit a cubic with fixed endpoints and unconstrained interior controls."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return None
    return _generate_free_cubic(pts, _chord_parameters(pts))


def _quadratic_max_residual(points: np.ndarray, control: np.ndarray) -> float:
    """Maximum raw-sample distance from a quadratic fitted to normalized data."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    t = _chord_parameters(pts)[:, None]
    if not np.any(t):
        return 0.0
    inverse = 1.0 - t
    curve = inverse * inverse * control[0] + 2.0 * inverse * t * control[1] + t * t * control[2]
    return float(np.max(np.linalg.norm(curve - pts, axis=1)))


def _cubic_max_residual(points: np.ndarray, control: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    t = _chord_parameters(pts)[:, None]
    if not np.any(t):
        return 0.0
    inverse = 1.0 - t
    curve = (
        inverse**3 * control[0]
        + 3.0 * inverse * inverse * t * control[1]
        + 3.0 * inverse * t * t * control[2]
        + t**3 * control[3]
    )
    return float(np.max(np.linalg.norm(curve - pts, axis=1)))


def _cubic_inflection_split_indices(points: np.ndarray, control: np.ndarray) -> list[int]:
    """Map a cubic's interior inflections back to safe raw-side boundaries."""
    parameters = _chord_parameters(np.asarray(points, dtype=float))
    cuts = {
        int(np.argmin(np.abs(parameters - root)))
        for root in cubic_inflects(control)
    }
    return sorted(index for index in cuts if 1 < index < len(points) - 2)


def _dual_quadratic_inflection_fit(
    points: np.ndarray,
    *,
    split: int,
    epsilon: float,
) -> tuple[list[tuple[str, np.ndarray]], float] | None:
    """Fit two normalized Q spans around one known cubic inflection."""
    if not 1 < split < len(points) - 2:
        return None
    segments: list[tuple[str, np.ndarray]] = []
    worst_residual = 0.0
    for span in (points[: split + 1], points[split:]):
        normalized = rdp(span, max(PATH_DENOISE_EPS, epsilon * SIDE_NORMALIZATION_FACTOR))
        quadratic, _error, _fit_split = fit_quadratic_once(normalized, epsilon)
        segments.append(("Q", quadratic))
        worst_residual = max(worst_residual, _quadratic_max_residual(span, quadratic))
    return segments, worst_residual


def _progressive_side_segments(
    seg: np.ndarray,
    _max_error: float,
    *,
    line_epsilon: float,
    allow_lines: bool,
    _depth: int = 0,
) -> list[tuple[str, np.ndarray]]:
    """Refit one corner-bounded side as Q/C spans split at inflections."""
    if allow_lines and (len(seg) < 3 or (
        _segment_is_straight(seg, line_epsilon)
        and _segment_length(seg) >= minimum_line_length(line_epsilon)
    )):
        return [("L", seg[-1])]
    cubic = _fit_free_cubic_once(seg)
    if cubic is not None:
        cubic_error = _cubic_max_residual(seg, cubic)
        inflection_cuts = _cubic_inflection_split_indices(seg, cubic)
        if len(inflection_cuts) == 1:
            dual_quadratic = _dual_quadratic_inflection_fit(
                seg,
                split=inflection_cuts[0],
                epsilon=line_epsilon,
            )
            if (
                dual_quadratic is not None
                and dual_quadratic[1] <= _max_error
                and dual_quadratic[1] < cubic_error
            ):
                return dual_quadratic[0]
            if cubic_error <= _max_error:
                return [("C", cubic)]
        if len(inflection_cuts) >= 2 and _depth < 4:
            boundaries = [0, *inflection_cuts, len(seg) - 1]
            result: list[tuple[str, np.ndarray]] = []
            for start, end in zip(boundaries, boundaries[1:]):
                result.extend(
                    _progressive_side_segments(
                        seg[start : end + 1],
                        _max_error,
                        line_epsilon=line_epsilon,
                        allow_lines=allow_lines,
                        _depth=_depth + 1,
                    )
                )
            return result

    # All candidate families use the requested source-sample error bound.
    # Do not accept one normalized quadratic for an entire side merely because
    # its pointwise error is small relative to a long chord.  That shortcut
    # visibly pulls a broad band inward even when the samples are locally close
    # to the curve.  Let the residual-preserving recursive fitter subdivide it.
    return _recursive_progressive_segments(
        seg,
        _max_error,
        line_epsilon=line_epsilon,
        allow_lines=allow_lines,
    )


def _recursive_progressive_segments(
    seg: np.ndarray,
    max_error: float,
    *,
    line_epsilon: float,
    allow_lines: bool,
) -> list[tuple[str, np.ndarray]]:
    """The residual-preserving L → Q → C → split fallback for one side."""
    if allow_lines and (len(seg) < 3 or (
        _segment_is_straight(seg, line_epsilon)
        and _segment_length(seg) >= minimum_line_length(line_epsilon)
    )):
        return [("L", seg[-1])]
    quadratic, quadratic_error, quadratic_split = fit_quadratic_once(seg, max_error)
    max_squared_error = max_error * max_error
    if quadratic_error < max_squared_error:
        return [("Q", quadratic)]
    cubic, cubic_error, cubic_split = fit_cubic_once(seg, max_error, guard_inflections=False)
    if cubic_error < max_squared_error:
        return [("C", cubic)]
    split = quadratic_split if quadratic_error <= cubic_error else cubic_split
    split = max(1, min(split, len(seg) - 2))
    return _recursive_progressive_segments(
        seg[:split + 1], max_error, line_epsilon=line_epsilon, allow_lines=allow_lines
    ) + _recursive_progressive_segments(
        seg[split:], max_error, line_epsilon=line_epsilon, allow_lines=allow_lines
    )


def _append_progressive_run(
    d: str,
    seg: np.ndarray,
    max_error: float,
    *,
    line_epsilon: float,
    allow_lines: bool,
    endpoints: list[np.ndarray | None] | None = None,
) -> str:
    for kind, primitive in _progressive_side_segments(
        seg, max_error, line_epsilon=line_epsilon, allow_lines=allow_lines
    ):
        if endpoints is not None:
            endpoints.append((primitive if kind == "L" else primitive[-1]).copy())
        if kind == "L":
            d += f"L{_fmt(primitive[0])} {_fmt(primitive[1])} "
        elif kind == "Q":
            d += f"Q{_fmt(primitive[1][0])} {_fmt(primitive[1][1])} {_fmt(primitive[2][0])} {_fmt(primitive[2][1])} "
        else:
            d += (
                f"C{_fmt(primitive[1][0])} {_fmt(primitive[1][1])} {_fmt(primitive[2][0])} {_fmt(primitive[2][1])} "
                f"{_fmt(primitive[3][0])} {_fmt(primitive[3][1])} "
            )
    return d


def _curved_run_d(
    seg: np.ndarray,
    max_error: float,
    *,
    cubic: bool,
    line_epsilon: float,
    progressive: bool = False,
    progressive_allow_lines: bool = True,
    endpoints: list[np.ndarray | None] | None = None,
) -> str:
    if progressive:
        return _append_progressive_run(
            "",
            seg,
            max_error,
            line_epsilon=line_epsilon,
            allow_lines=progressive_allow_lines,
            endpoints=endpoints,
        )
    if not cubic:
        return _append_quadratic_run(
            "", seg, max_error, line_epsilon=line_epsilon,
            collapse_straight=False, endpoints=endpoints,
        )
    return _append_cubic_run("", seg, max_error, line_epsilon=line_epsilon, endpoints=endpoints)


def _detect_path_features(
    contour: np.ndarray,
    *,
    epsilon: float,
    progressive: bool = False,
    support_weighted_corners: bool = False,
    corner_rounding: bool = True,
    canvas_extent: float = FIT_REFERENCE_CANVAS_EXTENT,
    forced_corners: np.ndarray | None = None,
    forced_corner_indices: tuple[int, ...] = (),
) -> _DetectedPathFeatures:
    """Discover source corners and fitting boundaries without fitting curves."""
    pts = np.asarray(contour, dtype=float)
    closed = np.allclose(pts[0], pts[-1])
    ring = pts[:-1] if closed else pts
    if not math.isfinite(canvas_extent) or canvas_extent <= 0:
        raise ValueError("canvas_extent must be finite and positive")
    canvas_scale = canvas_extent / FIT_REFERENCE_CANVAS_EXTENT

    def localize_corners(seed: np.ndarray, indices: list[int]) -> np.ndarray:
        # RDP supplies candidates, not authoritative apex locations. Locate
        # the source turn before measuring its point/rounded classification.
        seed = seed.copy()
        raw_indices = [int(np.argmin(np.linalg.norm(ring - point, axis=1))) for point in seed]
        for index in indices:
            raw_index = raw_indices[index]
            # The turn filter samples tangents at +/-2 scale units. Sparse
            # fitted polygons and symmetry cut endpoints have no raw samples
            # on both sides at that resolution; retain their exact vertices.
            if max(np.linalg.norm(ring[(raw_index + step) % len(ring)] - ring[raw_index])
                   for step in (-1, 1)) > 2.0 * canvas_scale:
                continue
            localized = corner_turn_index_at(
                ring, raw_index, scale=canvas_scale, aa_margin=max(2.0, epsilon * 2.0),
            )
            # Localization must not cross an adjacent candidate vertex. This
            # also protects synthetic cut endpoints used by symmetry fitting.
            previous, following = raw_indices[index - 1], raw_indices[(index + 1) % len(seed)]
            if 0 < (localized - previous) % len(ring) < (following - previous) % len(ring):
                seed[index] = ring[localized]
        return seed

    # Detection establishes corner candidates at the normal trace tolerance.
    # Localize point candidates on the original ring before classification;
    # rounded turns receive independently measured entry/exit boundaries.
    simp = rdp(ring, epsilon)
    corner_simp = _suppress_cornerlet_vertices(
        simp,
        min_adjacent_length=minimum_line_length(epsilon),
    )
    corners = corner_indices(
        np.vstack([corner_simp, corner_simp[0]]),
        angle_threshold_deg=40,
        min_adjacent_length=minimum_line_length(epsilon),
        # Binary/AA boundaries often soften a true corner over ~2px.  Sample
        # tangents across a 4px neighbourhood so the splitter sees the two
        # exterior edges rather than the short transition between them.
        support_radius=max(4.0, epsilon * 2.0),
    )
    corner_simp = localize_corners(corner_simp, corners)
    corner_pts = corner_simp[corners] if corners else np.empty((0, 2), dtype=float)
    aa_corner_margin = max(2.0, epsilon * 2.0)
    local_corner_radii: dict[int, float | None] = {}

    def source_corner_radius(seed: np.ndarray, index: int) -> float | None:
        """Measure one detected corner on the unsimplified sub-pixel ring."""
        raw_indices = [
            int(np.argmin(np.hypot(*(ring - point).T)))
            for point in seed
        ]
        raw_index = raw_indices[index]
        # Zero is a measured point corner, not a failed measurement. Once a
        # seed supplies a conclusive classification, later seeds may locate
        # that corner but must not silently reclassify it.
        if local_corner_radii.get(raw_index) is not None:
            return local_corner_radii[raw_index]
        radius = corner_curve_radius_at(
            ring,
            previous_index=raw_indices[(index - 1) % len(seed)],
            corner_index=raw_indices[index],
            following_index=raw_indices[(index + 1) % len(seed)],
            aa_margin=aa_corner_margin,
        )
        local_corner_radii[raw_index] = radius
        return radius

    for index in corners:
        source_corner_radius(corner_simp, index)
    source_rounded_corner_cut_idx = {
        index for index, radius in local_corner_radii.items()
        if radius is not None and radius > 0.0
    }
    sharp_corner_pts: list[np.ndarray] = []
    for index in corners:
        previous = corner_simp[(index - 1) % len(corner_simp)]
        current = corner_simp[index]
        following = corner_simp[(index + 1) % len(corner_simp)]
        incoming = current - previous
        outgoing = following - current
        turn = abs(float(np.degrees(np.arctan2(
            incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
            np.dot(incoming, outgoing),
        ))))
        if turn >= SHARP_CORNER_TURN_DEGREES:
            sharp_corner_pts.append(current)
    recovery_cut_idx: set[int] = set()
    recovered_span_cut_idx: set[int] = set()
    recovered_rounded_corner_cut_idx: set[int] = set()
    recovered_points = np.empty((0, 2), dtype=float)
    suppressed_seed_points = np.empty((0, 2), dtype=float)
    protected_lines = _protected_line_spans(ring, canvas_scale=canvas_scale) if progressive else []
    if progressive:
        # Some long organic corners are softened by anti-aliasing over more
        # than the normal trace tolerance.  Add (rather than replace with)
        # coarse corner anchors, so topology is stable without moving the
        # normal-resolution corners already found above.
        corner_seed = _suppress_cornerlet_vertices(
            rdp(ring, epsilon * 4.0),
            min_adjacent_length=max(minimum_line_length(epsilon), minimum_line_length(epsilon * 4.0) * canvas_scale),
            # Coarse RDP edges are chords, not source tangents. Repeatedly
            # deleting their short neighbours can manufacture a sharp turn on
            # a smooth arc. Bound that change by the normal trace tolerance.
            max_deviation=epsilon,
        )
        seed_corners = corner_indices(
            np.vstack([corner_seed, corner_seed[0]]),
            angle_threshold_deg=40,
            min_adjacent_length=max(minimum_line_length(epsilon), minimum_line_length(epsilon * 4.0) * canvas_scale),
            support_radius=max(4.0, epsilon * 8.0 * canvas_scale),
        )
        corner_seed = localize_corners(corner_seed, seed_corners)
        source_rounded_corner_cut_idx.update(
            int(np.argmin(np.hypot(*(ring - corner_seed[index]).T)))
            for index in seed_corners
            if local_corner_radii.get(int(np.argmin(np.hypot(*(ring - corner_seed[index]).T)))) is None
            if (radius := source_corner_radius(corner_seed, index)) is not None
            and radius > 0.0
        )
        if seed_corners:
            seed_points = corner_seed[seed_corners]
            if sharp_corner_pts:
                sharp_points = np.asarray(sharp_corner_pts, dtype=float)
                duplicate_radius = max(16.0, epsilon * 16.0) * canvas_scale
                duplicate_seed_points = [
                    point
                    for point in seed_points
                    if float(np.min(np.linalg.norm(sharp_points - point, axis=1))) <= duplicate_radius
                ]
                if duplicate_seed_points:
                    suppressed_seed_points = np.asarray(duplicate_seed_points, dtype=float)
                seed_points = np.asarray([
                    point
                    for point in seed_points
                    if float(np.min(np.linalg.norm(sharp_points - point, axis=1))) > duplicate_radius
                ])
            if len(seed_points):
                corner_pts = np.vstack([corner_pts, seed_points])
        # Baseline and coarse-seed detections are independently authoritative.
        # Recovery below may help them form a rounded span, but must never
        # erase one merely because it rediscovered the same raw location.
        established_corner_pts = corner_pts.copy()
        # A design corner may be softened over several raster samples, making
        # its immediate RDP neighbours too short for the normal cornerlet
        # guard.  A long support window recovers it only when the exterior
        # turn persists across a substantial run on both sides.
        recovery_seed = _suppress_cornerlet_vertices(
            rdp(ring, epsilon * 2.0),
            min_adjacent_length=max(minimum_line_length(epsilon), minimum_line_length(epsilon * 2.0) * canvas_scale),
        )
        recovery_corners = corner_indices(
            np.vstack([recovery_seed, recovery_seed[0]]),
            angle_threshold_deg=25.0,
            min_adjacent_length=0.0,
            support_radius=max(24.0, epsilon * 32.0) * canvas_scale,
        )
        recovery_seed = localize_corners(recovery_seed, recovery_corners)
        if recovery_corners:
            recovered_points = recovery_seed[recovery_corners]
            corner_pts = np.vstack([corner_pts, recovered_points])
            recovery_cut_idx = {
                int(np.argmin(np.hypot(*(ring - point).T)))
                for point in recovered_points
            }
            # A shallow recovery corner is only a semantic point boundary when
            # its tangent change persists over a meaningful exterior run.  The
            # normal recovery window finds the softened transition; this wider
            # window verifies that it is a design corner rather than a small
            # local kink.  In particular, this retains radish's lower orange
            # apex without promoting the nearby rounded right-end transition.
            long_support_indices = set(corner_indices(
                np.vstack([recovery_seed, recovery_seed[0]]),
                angle_threshold_deg=25.0,
                min_adjacent_length=0.0,
                support_radius=max(64.0, epsilon * 64.0) * canvas_scale,
            ))
            duplicate_radius = max(4.0, epsilon * 4.0)
            span_duplicate_radius = max(16.0, epsilon * 16.0) * canvas_scale
            for index in recovery_corners:
                current = recovery_seed[index]
                raw_index = int(np.argmin(np.hypot(*(ring - current).T)))
                radius = source_corner_radius(recovery_seed, index)
                source_rounded = radius is not None and radius > 0.0
                if len(established_corner_pts):
                    nearest_established = float(np.min(
                        np.linalg.norm(established_corner_pts - current, axis=1)
                    ))
                    if nearest_established <= duplicate_radius:
                        continue
                    # A second recovery point next to a pre-existing point is
                    # part of that rounded corner, not another corner.  Keep
                    # it only to define the protected rounded-corner span.
                    if nearest_established <= span_duplicate_radius:
                        recovered_span_cut_idx.add(raw_index)
                        continue
                if len(suppressed_seed_points) and float(np.min(
                    np.linalg.norm(suppressed_seed_points - current, axis=1)
                )) <= max(4.0, epsilon * 4.0):
                    continue
                if not source_rounded and index not in long_support_indices:
                    continue
                # Recovery detects a softened design turn. Its support proves
                # that the turn matters, but does not turn the rounded source
                # transition into a sharp point corner. Protect a small span
                # around it so fitting retains the local curvature.
                recovered_rounded_corner_cut_idx.add(raw_index)
    else:
        established_corner_pts = corner_pts.copy()
    # Preserve the actual detector output separately from the fallback split
    # point needed only to make a closed contour fit-able. Fitting boundaries
    # are not automatically semantic corner anchors.
    declared_corner_pts = established_corner_pts.copy()
    if not len(corner_pts):
        corner_pts = simp[[0]]
    detected_cut_idx = {int(np.argmin(np.hypot(*(ring - cp).T))) for cp in corner_pts}
    established_cut_idx = {
        int(np.argmin(np.hypot(*(ring - point).T)))
        for point in established_corner_pts
    }
    point_corner_idx = {
        int(np.argmin(np.hypot(*(ring - point).T)))
        for point in declared_corner_pts
    }
    cut_idx = set(detected_cut_idx)
    if progressive:
        # Protect pre-fit straight sides before progressive recursion.  Their
        # endpoints are ordinary cut boundaries, but no automatically detected
        # corner may subdivide the line between those endpoints.
        for start, end in protected_lines:
            cut_idx = {
                index for index in cut_idx
                if not _inside_closed_span(index, start, end)
            }
            cut_idx.update((start, end))
    forced_cut_idx: set[int] = set(forced_corner_indices)
    if any(index < 0 or index >= len(ring) for index in forced_cut_idx):
        raise ValueError("forced corner indices must belong to the source ring")
    if forced_corners is not None and len(forced_corners):
        forced_cut_idx.update({
            int(np.argmin(np.hypot(*(ring - point).T)))
            for point in np.asarray(forced_corners, dtype=float)
        })
    cut_idx.update(forced_cut_idx)
    point_corner_idx.update(forced_cut_idx)
    corner_ranges = (
        _protected_corner_ranges(
            ring,
            established_cut_idx | recovered_span_cut_idx,
            epsilon=epsilon,
            canvas_scale=canvas_scale,
        )
        if progressive
        else []
    )
    if progressive and support_weighted_corners:
        corner_ranges.extend(
            _supported_corner_ranges(
                ring,
                corner_simp,
                epsilon=epsilon,
                min_turn_degrees=MIN_SUPPORTED_CORNER_TURN_DEGREES,
                min_support_length=MIN_PROTECTED_POLYLINE_LINE_LENGTH * canvas_scale,
                canvas_scale=canvas_scale,
            )
        )
        guard = max(4.0, epsilon * 4.0)
        corner_ranges.extend(
            (
                _advance_ring_index(ring, index, -1, guard),
                _advance_ring_index(ring, index, 1, guard),
            )
            for index in recovered_rounded_corner_cut_idx
        )
    # Only a source inset exceeding AA and edge-fit uncertainty is rounded,
    # even when the fillet simplified to one RDP vertex.
    guard = max(4.0, epsilon * 4.0)
    corner_ranges.extend(
        (
            _advance_ring_index(ring, index, -1, guard),
            _advance_ring_index(ring, index, 1, guard),
        )
        for index in source_rounded_corner_cut_idx
    )
    corner_ranges = _merge_corner_ranges(ring, list(dict.fromkeys(corner_ranges)))
    if corner_rounding:
        # Locate the turn itself, not its broad detection-support window. Side
        # runs and the protected corner share these exact source boundaries.
        localized_ranges: list[tuple[int, int]] = []
        localized_centers: set[int] = set()
        measured_point_indices = {
            index for index in established_cut_idx
            if local_corner_radii.get(index) == 0.0
        } | forced_cut_idx
        # AA can place two measured point candidates on the same transition.
        # Keep its strongest supported turn once, but never combine opposing
        # turns (a short serif/step) or discard an explicit forced corner.
        scored_points: list[tuple[int, float]] = []
        for index in measured_point_indices - forced_cut_idx:
            before = _advance_ring_index(ring, index, -1, aa_corner_margin * 2.0)
            after = _advance_ring_index(ring, index, 1, aa_corner_margin * 2.0)
            incoming, outgoing = ring[index] - ring[before], ring[after] - ring[index]
            turn = float(np.arctan2(
                incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
                np.dot(incoming, outgoing),
            ))
            scored_points.append((index, turn))
        kept_points: list[tuple[int, float]] = []
        duplicate_points: set[int] = set()
        for index, turn in sorted(scored_points, key=lambda item: (-abs(item[1]), item[0])):
            if any(turn * other_turn > 0.0 and min(
                _arc_distance(ring, index, other), _arc_distance(ring, other, index),
            ) <= aa_corner_margin * 2.0 for other, other_turn in kept_points):
                duplicate_points.add(index)
            else:
                kept_points.append((index, turn))
        cut_idx.difference_update(duplicate_points)
        point_corner_idx.difference_update(duplicate_points)
        for index in sorted((established_cut_idx | source_rounded_corner_cut_idx) - forced_cut_idx):
            span = corner_curve_span_at(
                ring, index, scale=canvas_scale, aa_margin=aa_corner_margin,
                rounding_known=False,
            )
            if span is not None:
                localized_ranges.append(span)
                localized_centers.add(index)
        # Radius measured at a coarse seed may say zero even when the actual
        # source turn has a resolvable rounded extent. Localization owns the
        # classification; explicit forced point declarations remain fixed.
        measured_point_indices.difference_update(localized_centers)
        # Every rounded proposal, including a positive radius estimate, must
        # pass the same AA-aware source-span test. Candidate protection and
        # radius estimates may not create a fallback rounded classification.
        corner_ranges = _merge_corner_ranges(ring, localized_ranges)
        corner_ranges = [
            (start, end) for start, end in corner_ranges
            if not any(index in {start, end} or _inside_closed_span(index, start, end)
                       for index in measured_point_indices)
        ]
        cut_idx.difference_update(localized_centers)
        point_corner_idx.difference_update(localized_centers)
    if not corner_rounding:
        # Keep detected point boundaries, but do not replace them with an
        # automatically protected rounded interval. Source samples remain
        # authoritative; this is not the downstream normalization operation.
        corner_ranges = []
    # Broad-window recovery points may help nearby detections form a rounded
    # corner range, but must never survive as independent point anchors.  The
    # support-weighted detector below owns the distinct shallow-corner case.
    recovery_only_cut_idx = (
        recovery_cut_idx
        - established_cut_idx
        - recovered_span_cut_idx
        - recovered_rounded_corner_cut_idx
        - forced_cut_idx
    )
    cut_idx.difference_update(recovery_only_cut_idx)
    point_corner_idx.difference_update(recovery_only_cut_idx)
    for start, end in corner_ranges:
        cut_idx = {
            index for index in cut_idx
            if not _inside_closed_span(index, start, end)
        }
        point_corner_idx = {
            index for index in point_corner_idx
            if index not in {start, end} and not _inside_closed_span(index, start, end)
        }
        cut_idx.update((start, end))
    return _DetectedPathFeatures(
        tuple((float(point[0]), float(point[1])) for point in ring),
        tuple(sorted(cut_idx)),
        tuple(sorted(point_corner_idx)),
        tuple(corner_ranges),
    )


def _fit_path_d_with_evidence(
    contour: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool = False,
    progressive: bool = False,
    progressive_allow_lines: bool = True,
    support_weighted_corners: bool = False,
    corner_rounding: bool = True,
    canvas_extent: float = FIT_REFERENCE_CANVAS_EXTENT,
    forced_corners: np.ndarray | None = None,
) -> _AutomaticPathFit:
    """Orchestrate feature discovery, shared numerical fitting and evidence binding."""
    features = _detect_path_features(
        contour, epsilon=epsilon, progressive=progressive,
        support_weighted_corners=support_weighted_corners,
        corner_rounding=corner_rounding, canvas_extent=canvas_extent,
        forced_corners=forced_corners,
    )
    ring = np.asarray(features.ring, dtype=float)
    command_end_indices: list[int | None] = []
    d, runs = _fit_path_d_from_cuts(
        ring,
        list(features.cut_indices),
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
        progressive=progressive,
        progressive_allow_lines=progressive_allow_lines,
        command_end_indices=command_end_indices,
    )

    corner_spans: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for start, end in features.corner_ranges:
        corner_spans.append((
            (float(ring[start][0]), float(ring[start][1])),
            (float(ring[end][0]), float(ring[end][1])),
        ))
    # Protected lines introduce fitting boundaries only. The intervening span
    # may be a whole note head, side, or beam, not a detected rounded corner.
    return _AutomaticPathFit(
        d,
        tuple(features.ring[index] for index in features.point_indices),
        tuple(corner_spans),
            automatic_fit_evidence(
            ring, features.point_indices, features.corner_ranges, runs, d,
            settings=(epsilon, max_error, cubic, progressive, progressive_allow_lines,
                        support_weighted_corners, corner_rounding, canvas_extent,
                        None if forced_corners is None else np.asarray(forced_corners).tolist()),
                command_end_indices=tuple(command_end_indices),
        ),
    )


def _fit_path_d_with_corners(contour: np.ndarray, **options) -> tuple[
    str, tuple[tuple[float, float], ...],
    tuple[tuple[tuple[float, float], tuple[float, float]], ...],
]:
    """Project the numerical fit for callers that need only SVG diagnostics."""
    fitted = _fit_path_d_with_evidence(contour, **options)
    return fitted.d, fitted.corner_points, fitted.corner_spans


def _fit_path_d_from_cuts(
    ring: np.ndarray,
    cut_idx: list[int],
    *,
    epsilon: float,
    max_error: float,
    cubic: bool = False,
    progressive: bool = False,
    progressive_allow_lines: bool = True,
    excluded_indices: frozenset[int] = frozenset(),
    command_end_indices: list[int | None] | None = None,
) -> tuple[str, tuple[FittedRun, ...]]:
    """Fit directed runs between established cuts without detecting features.

    The returned run records are ``(source_start, source_end, command_count)``.
    They describe numerical fitting spans only; callers decide which, if any,
    carry semantic feature ownership.
    """
    if len(cut_idx) == 1:
        # A closed contour needs two fitting runs, but the existing cut is a
        # declared boundary and must survive. Add only a non-semantic opposite
        # split; sorting preserves directed ring traversal from the lower cut.
        auxiliary = (cut_idx[0] + len(ring) // 2) % len(ring)
        if auxiliary == cut_idx[0]:
            raise ValueError("closed contour requires two distinct fitting points")
        cut_idx = sorted((cut_idx[0], auxiliary))
    elif not cut_idx:
        cut_idx = [0, len(ring) // 2]

    d = f"M{_fmt(ring[cut_idx[0]][0])} {_fmt(ring[cut_idx[0]][1])} "
    runs: list[FittedRun] = []
    for k in range(len(cut_idx)):
        i0 = cut_idx[k]
        i1 = cut_idx[(k + 1) % len(cut_idx)]
        indices = list(range(i0, i1 + 1)) if i1 > i0 else [*range(i0, len(ring)), *range(i1 + 1)]
        indices = [index for index in indices if index in {i0, i1} or index not in excluded_indices]
        seg = ring[indices]
        if progressive:
            retained = _cornerlet_fit_indices(seg, epsilon=epsilon)
            seg = seg[retained]
            indices = [indices[index] for index in retained]
        endpoints: list[np.ndarray | None] | None = [] if command_end_indices is not None else None
        if len(seg) < 2:
            continue
        if _segment_is_straight(seg, epsilon) and _segment_length(seg) >= minimum_line_length(epsilon):
            run_d = f"L{_fmt(seg[-1][0])} {_fmt(seg[-1][1])} "
            if endpoints is not None:
                endpoints.append(seg[-1].copy())
        else:
            run_d = _curved_run_d(
                seg,
                max_error,
                cubic=cubic,
                line_epsilon=epsilon,
                progressive=progressive,
                progressive_allow_lines=progressive_allow_lines,
                endpoints=endpoints,
            )
        d += run_d
        runs.append(FittedRun(i0, i1, len(re.findall(r"[LQC]", run_d))))
        if endpoints is not None:
            if len(endpoints) != runs[-1].command_count:
                raise RuntimeError("fitted endpoint accounting is inconsistent")
            for position, point in enumerate(endpoints):
                if position == len(endpoints) - 1:
                    command_end_indices.append(i1)
                    continue
                # Compare the fitter's unrounded endpoint only with this
                # directed input span. Never search rendered/fitted geometry.
                matches = [] if point is None else [
                    index for index, sample in zip(indices, seg, strict=True)
                    if np.array_equal(sample, point)
                ]
                command_end_indices.append(matches[0] if len(matches) == 1 else None)
    d += "Z"
    return d, tuple(runs)


def _fit_path_d(
    contour: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool = False,
    progressive: bool = False,
    progressive_allow_lines: bool = True,
    support_weighted_corners: bool = False,
    corner_rounding: bool = True,
    canvas_extent: float = FIT_REFERENCE_CANVAS_EXTENT,
    forced_corners: np.ndarray | None = None,
) -> str:
    """Fit a contour, discarding provenance for internal candidate searches."""
    return _fit_path_d_with_corners(
        contour,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
        progressive=progressive,
        progressive_allow_lines=progressive_allow_lines,
        support_weighted_corners=support_weighted_corners,
        corner_rounding=corner_rounding,
        canvas_extent=canvas_extent,
        forced_corners=forced_corners,
    )[0]


def _simpler_curve_candidate(
    contour: np.ndarray,
    *,
    baseline_d: str,
    epsilon: float,
    max_error: float,
    cubic: bool,
    progressive: bool,
    progressive_allow_lines: bool,
    support_weighted_corners: bool,
    corner_rounding: bool,
    canvas_extent: float,
    forced_corners: np.ndarray | None,
) -> str:
    # Trace fitting is deliberately faithful: tangent cleanup is a downstream
    # optimizer pass, not a candidate-ranking transform.  Applying it here
    # can pass an area-residual gate while visibly distorting a circle or a
    # shallow rounded corner before the drawing has stable regions to inspect.
    candidates = [baseline_d]
    for eps_factor, err_factor in (
        (1.5, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (1.0, 20.0),
        (10.0, 30.0),
    ):
        candidate = _fit_path_d(
            contour,
            epsilon=max(epsilon * eps_factor, epsilon + 0.5),
            max_error=max(max_error * err_factor, max_error + 0.75),
            cubic=cubic,
            progressive=progressive,
            progressive_allow_lines=progressive_allow_lines,
            support_weighted_corners=support_weighted_corners,
            corner_rounding=corner_rounding,
            canvas_extent=canvas_extent,
            forced_corners=forced_corners,
        )
        if _path_area_residual(contour, candidate) > SIMPLE_CURVE_RESIDUAL_TOL:
            continue
        if _path_command_count(candidate) > _path_command_count(baseline_d):
            continue
        candidates.append(candidate)
    return min(candidates, key=lambda d: (1.0 - _curve_join_min_dot_d(d), _path_command_count(d)))


def fit_path_stages(
    contour: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool = False,
    progressive: bool = False,
    progressive_allow_lines: bool = True,
    support_weighted_corners: bool = False,
    corner_rounding: bool = True,
    canvas_extent: float = FIT_REFERENCE_CANVAS_EXTENT,
    forced_corners: np.ndarray | None = None,
    prefer_simple_curves: bool = False,
) -> PathFitStages:
    """Fit a contour with geometric supports scaled to the whole canvas.

    ``canvas_extent`` is the canvas's longer side, never a region's bounding
    box. Pixel fitting tolerances and AA margins remain unscaled.
    """
    fitted = _fit_path_d_with_evidence(
        contour,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
        progressive=progressive,
        progressive_allow_lines=progressive_allow_lines,
        support_weighted_corners=support_weighted_corners,
        corner_rounding=corner_rounding,
        canvas_extent=canvas_extent,
        forced_corners=forced_corners,
    )
    baseline_d, corner_points, corner_spans = fitted.d, fitted.corner_points, fitted.corner_spans
    if not prefer_simple_curves:
        return PathFitStages(
            baseline_d=baseline_d,
            simple_d=baseline_d,
            atomic_d=baseline_d,
            corner_points=corner_points,
            corner_spans=corner_spans,
            baseline_evidence=fitted.evidence,
        )

    simple_d = _simpler_curve_candidate(
        contour,
        baseline_d=baseline_d,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
        progressive=progressive,
        progressive_allow_lines=progressive_allow_lines,
        support_weighted_corners=support_weighted_corners,
        corner_rounding=corner_rounding,
        canvas_extent=canvas_extent,
        forced_corners=forced_corners,
    )
    return PathFitStages(
        baseline_d=baseline_d,
        simple_d=simple_d,
        atomic_d=_residual_gated_atomic_simplify_d(contour, simple_d),
        corner_points=corner_points,
        corner_spans=corner_spans,
        baseline_evidence=fitted.evidence,
    )


def fit_path(
    contour: np.ndarray,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool = False,
    progressive: bool = False,
    progressive_allow_lines: bool = True,
    support_weighted_corners: bool = False,
    corner_rounding: bool = True,
    canvas_extent: float = FIT_REFERENCE_CANVAS_EXTENT,
    forced_corners: np.ndarray | None = None,
    prefer_simple_curves: bool = False,
) -> Shape:
    """Corner-split the contour; emit lines for straight runs, Béziers otherwise.

    ``cubic=False`` (default) fits each curved run with inflection-free
    quadratics — robust against the quantization staircase. ``cubic=True`` fits
    curved runs with denoised, inflection-guarded cubics; see PATH_DENOISE_EPS.

    ``prefer_simple_curves`` keeps the normal fit as a baseline, then accepts a
    looser smooth-curve fit only when it reduces path commands and stays close to
    the same filled contour.
    """
    stages = fit_path_stages(
        contour,
        epsilon=epsilon,
        max_error=max_error,
        cubic=cubic,
        progressive=progressive,
        progressive_allow_lines=progressive_allow_lines,
        support_weighted_corners=support_weighted_corners,
        corner_rounding=corner_rounding,
        canvas_extent=canvas_extent,
        forced_corners=forced_corners,
        prefer_simple_curves=prefer_simple_curves,
    )
    return Shape("path", {"d": stages.atomic_d})

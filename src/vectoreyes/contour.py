"""C3/C4: sub-pixel contour extraction, RDP simplification, corner detection."""

from __future__ import annotations

from bisect import bisect_left, insort

import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.measure import find_contours
from .types import CoverageField

_CORNER_RDP_EPS = 2.0        # contour->polygon simplification tolerance (px): 2.0 avoids arc fragmentation
_CORNER_STRAIGHT_TOL = 2.5   # max TLS perpendicular residual (px) for an edge to count as a straight side
_CORNER_MAX_FILLET_FRAC = 0.30  # max inset as a fraction of the shorter adjacent edge (a fillet, not a cap)
_CORNER_MIN_FILLET = 2.5     # min angle-corrected radius (px) to treat a corner as rounded; below -> sharp (0)
_CORNER_DEANTIALIAS_PAD = 2.0  # added to a DETECTED fillet radius only (never to a sharp 0)


def _polygon_area(c: np.ndarray) -> float:
    """Absolute shoelace area of an (N, 2) closed polygon."""
    x, y = c[:, 0], c[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def _coverage_contours(mask: np.ndarray, coverage: CoverageField | np.ndarray | None) -> list[np.ndarray]:
    if isinstance(coverage, CoverageField):
        if coverage.canvas_shape != mask.shape:
            raise ValueError("coverage canvas does not match the region mask")
        field = coverage.values
        x, y = coverage.origin
    else:
        field = mask.astype(float) if coverage is None else coverage
        x = y = 0
    contours = find_contours(np.pad(field, 1), 0.5)
    return [np.column_stack([c[:, 1] - 1 + x, c[:, 0] - 1 + y]) for c in contours]


def region_contours(mask: np.ndarray, *, coverage: CoverageField | np.ndarray | None = None) -> list[np.ndarray]:
    """All sub-pixel contours (outer + holes), (N,2) (x,y), area-sorted (outer first).
    With `coverage` (a float α field, boundary at 0.5) the contour is extracted from the
    smooth field instead of the bilevel mask; `coverage=None` is the original behavior."""
    out = _coverage_contours(mask, coverage)
    out.sort(key=_polygon_area, reverse=True)
    return out


def outer_contour(mask: np.ndarray, *, coverage: CoverageField | np.ndarray | None = None) -> np.ndarray:
    """Longest sub-pixel contour of `mask`, as an (N, 2) array of (x, y) points.
    With `coverage` (a float α field, boundary at 0.5) the contour is extracted from the
    smooth field instead of the bilevel mask; `coverage=None` is the original behavior."""
    contours = _coverage_contours(mask, coverage)
    if not contours:
        return np.empty((0, 2))
    longest = max(contours, key=len)                 # rows of (row, col) == (y, x)
    return longest


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker polyline simplification."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    line = end - start
    norm = np.hypot(*line)
    if norm == 0:
        d = np.hypot(*(pts - start).T)
    else:
        diff = pts - start
        # 2-D cross product as an explicit scalar (np.cross on 2-D vectors is
        # deprecated in NumPy 2.0)
        d = np.abs(line[0] * diff[:, 1] - line[1] * diff[:, 0]) / norm
    idx = int(d.argmax())
    if d[idx] > epsilon:
        left = rdp(pts[: idx + 1], epsilon)
        right = rdp(pts[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def corner_indices(
    poly: np.ndarray,
    *,
    angle_threshold_deg: float = 40.0,
    min_adjacent_length: float = 0.0,
    support_radius: float = 0.0,
) -> list[int]:
    """Indices of vertices with a sufficiently sharp, well-supported turn.

    `poly` is assumed closed (first point repeated at the end).

    ``support_radius`` is a maximum, not a symmetric window. Each side grows
    independently while its tangent remains stable, stopping before a distinct
    neighboring bend. Nearby same-direction detections from one AA transition
    are reduced to their strongest turn; opposing turns remain independent.
    """
    pts = np.asarray(poly, dtype=float)
    n = len(pts) - 1 if np.allclose(pts[0], pts[-1]) else len(pts)
    if n < 3:
        return []
    thresh = np.radians(angle_threshold_deg)
    radius = max(float(support_radius), 0.0)
    edge_lengths = np.hypot(
        *(np.roll(pts[:n], -1, axis=0) - pts[:n]).T
    )
    arc_at_vertex = np.concatenate(([0.0], np.cumsum(edge_lengths[:-1])))
    perimeter = float(edge_lengths.sum())

    def point_at_arc(index: int, step: int, distance: float) -> np.ndarray:
        """Grow one side's support without crossing another resolved bend."""
        origin = pts[index % n]
        point = pts[index % n].astype(float, copy=True)
        remaining = distance
        current = index % n
        # A tangent estimate should span raster noise, not turn onto the next
        # design side. This stability bound is separate from corner acceptance.
        stability_dot = float(np.cos(np.radians(10.0)))
        for _ in range(n):
            if remaining <= 1e-9:
                break
            following = (current + step) % n
            end = pts[following].astype(float, copy=False)
            direction = end - point
            length = float(np.hypot(*direction))
            if length <= 1e-12:
                current = following
                continue
            support = point - origin
            support_length = float(np.hypot(*support))
            if support_length > 1e-9:
                alignment = float(np.dot(support, direction) / (support_length * length))
                if alignment < stability_dot:
                    break
            if length >= remaining:
                return point + direction * (remaining / length)
            point = end.copy()
            remaining -= length
            current = following
        return point

    def cyclic_distance(left: float, right: float) -> float:
        distance = abs(right - left)
        return min(distance, perimeter - distance)

    scored: list[tuple[int, float, int]] = []
    for i in range(n):
        prev, cur, nxt = pts[(i - 1) % n], pts[i % n], pts[(i + 1) % n]
        # The support window below makes the *angle* AA-tolerant.  It does not
        # make a tiny adjacent RDP run into meaningful corner evidence: a
        # short/short or short/long kink is a cornerlet and must not split the
        # fitted contour.  This is deliberately an OR condition.
        immediate_incoming = float(np.hypot(*(cur - prev)))
        immediate_outgoing = float(np.hypot(*(nxt - cur)))
        if min(immediate_incoming, immediate_outgoing) < min_adjacent_length:
            continue
        if radius:
            v1 = cur - point_at_arc(i, -1, radius)
            v2 = point_at_arc(i, 1, radius) - cur
        else:
            v1, v2 = cur - prev, nxt - cur
        incoming_length = float(np.hypot(*v1))
        outgoing_length = float(np.hypot(*v2))
        if incoming_length == 0 or outgoing_length == 0:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]   # 2-D scalar cross (NumPy 2.0)
        ang = np.arctan2(cross, np.dot(v1, v2))
        turn = abs(float(ang))
        if turn >= thresh:
            scored.append((i, turn, 1 if ang > 0.0 else -1))
    if not radius:
        return [index for index, _score, _orientation in scored]

    # An AA corner often produces two adjacent candidate vertices.  Retain the
    # stronger turn once, rather than splitting the curve fit at both sides of
    # the same soft transition.
    selected: list[int] = []
    selected_arcs_by_orientation: dict[int, list[float]] = {-1: [], 1: []}
    for index, _score, orientation in sorted(scored, key=lambda item: (-item[1], item[0])):
        arc = float(arc_at_vertex[index])
        # Opposing bends bound a step/serif, not duplicate AA observations of
        # one corner. Deduplicate only turns with matching orientation.
        selected_arcs = selected_arcs_by_orientation[orientation]
        insertion = bisect_left(selected_arcs, arc)
        neighbours: list[float] = []
        if selected_arcs:
            neighbours.append(selected_arcs[insertion % len(selected_arcs)])
            neighbours.append(selected_arcs[(insertion - 1) % len(selected_arcs)])
        if any(cyclic_distance(arc, existing) <= radius for existing in neighbours):
            continue
        selected.append(index)
        insort(selected_arcs, arc)
    return sorted(selected)


def _slice_loop(contour: np.ndarray, i: int, j: int) -> np.ndarray:
    """Points of a closed contour from index i to j inclusive, handling wraparound."""
    return contour[i:j + 1] if i <= j else np.vstack([contour[i:], contour[:j + 1]])


def _edge_line(edge: np.ndarray):
    """Total-least-squares line over the middle 60% of an edge (excludes the corner
    transitions at both ends). Returns (centroid, unit_direction, max_perp_residual) or
    None if the edge is too short."""
    k = len(edge)
    if k < 5:
        return None
    mid = edge[k // 5: k - k // 5]
    if len(mid) < 3:
        return None
    c = mid.mean(axis=0)
    _, _, vt = np.linalg.svd(mid - c, full_matrices=False)
    d = vt[0]
    perp = np.abs((mid - c) @ np.array([-d[1], d[0]]))
    return c, d, float(perp.max())


def _line_intersection(c1, d1, c2, d2):
    """Intersection of lines (c1 + t*d1) and (c2 + s*d2), or None if near-parallel."""
    a = np.column_stack([d1, -d2])
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t = np.linalg.solve(a, c2 - c1)
    return c1 + t[0] * d1


def _corner_radius_at(
    before: np.ndarray, after: np.ndarray, vertex: np.ndarray,
    *, aa_margin: float | None = None,
):
    """Angle-corrected fillet radius at one polygon corner, or None if it is not a clean
    corner. `before`/`after` are the contour points on the two edges meeting at `vertex`.

    Fit a straight line to each edge; intersect them at the sharp-corner point P; measure
    the inset (nearest contour point near the vertex to P); convert that inset to a true
    radius using the corner angle (for a fillet radius r in a corner of interior angle
    theta, the contour's nearest approach to P is r*(1/sin(theta/2) - 1))."""
    l1, l2 = _edge_line(before), _edge_line(after)
    if l1 is None or l2 is None:
        return None
    (c1, d1, r1), (c2, d2, r2) = l1, l2
    if r1 > _CORNER_STRAIGHT_TOL or r2 > _CORNER_STRAIGHT_TOL:
        return None                                  # an edge isn't a straight side
    p = _line_intersection(c1, d1, c2, d2)
    if p is None:
        return None
    near = np.vstack([before[-3:], after[:3], vertex[None]])
    inset = float(np.min(np.hypot(near[:, 0] - p[0], near[:, 1] - p[1])))
    edge_len = min(float(np.hypot(*(before[0] - before[-1]))),
                   float(np.hypot(*(after[0] - after[-1]))))
    if edge_len <= 0 or inset > _CORNER_MAX_FILLET_FRAC * edge_len:
        return None                                  # too deep -> a cap, not a corner fillet
    if aa_margin is not None:
        if not np.isfinite(aa_margin) or aa_margin < 0:
            raise ValueError("aa_margin must be finite and nonnegative")
        # The AA allowance measures visible contour displacement, not the
        # angle-amplified fillet radius. Include line-fit uncertainty: the
        # intersection of shallow edges is less well-conditioned.
        sine = abs(float(d1[0] * d2[1] - d1[1] * d2[0]))
        if sine <= 1e-9:
            return None
        intersection_uncertainty = (r1 + r2) / sine
        if inset <= aa_margin + intersection_uncertainty:
            return 0.0
    # interior angle between the two edges (directions point from P into each edge body)
    m1 = before[:max(1, len(before) // 2)].mean(axis=0) - p
    m2 = after[len(after) // 2:].mean(axis=0) - p
    n1, n2 = np.hypot(*m1) or 1.0, np.hypot(*m2) or 1.0
    theta = float(np.arccos(np.clip((m1 / n1) @ (m2 / n2), -1.0, 1.0)))
    if theta <= 1e-2:
        return None
    denom = 1.0 / np.sin(theta / 2.0) - 1.0
    if denom < 0.05:                                 # near-straight join: not a real corner
        return None
    return inset / denom


def corner_curve_radius_at(
    contour: np.ndarray,
    *,
    previous_index: int,
    corner_index: int,
    following_index: int,
    aa_margin: float | None = None,
) -> float | None:
    """Measure one source-contour corner curve from three established vertices.

    The vertices must be indexes into the same closed, sub-pixel contour. A
    returned value is the geometric fillet radius; ``None`` means the adjacent
    sides do not supply enough clean line evidence to classify the turn.
    When ``aa_margin`` is supplied, a source inset within that pixel allowance
    plus edge-fit uncertainty returns zero (a point boundary). Radius describes
    rounding only after this source-displacement gate succeeds.
    """
    points = np.asarray(contour, dtype=float)
    if len(points) < 3:
        return None
    count = len(points)
    before = _slice_loop(points, previous_index % count, corner_index % count)
    after = _slice_loop(points, corner_index % count, following_index % count)
    return _corner_radius_at(before, after, points[corner_index % count], aa_margin=aa_margin)


def _corner_turn_at(
    contour: np.ndarray, index: int, *, scale: float, aa_margin: float,
    rounding_known: bool = False, point_only: bool = False,
) -> tuple[int, int] | None:
    """Locate an established corner's local turn on the source polyline.

    Tangents are sampled by arc length, independently of raster sample density.
    A point corner sampled through the same filter supplies the blur baseline;
    only excess turn extent beyond the AA allowance establishes new rounding.
    ``rounding_known`` locates the extent of an already classified rounded turn.
    It does not discover corners or change source points.
    """
    ring = np.asarray(contour, dtype=float)
    if len(ring) < 3:
        return None
    points = np.vstack([ring, ring[0]])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    window = min(24.0 * scale, arc[-1] / 8.0)
    if window < 12.0 * scale:
        return None
    offsets = np.linspace(-window, window, 97)
    spacing = float(offsets[1] - offsets[0])
    center = arc[index % len(ring)]

    def sample(distances: np.ndarray) -> np.ndarray:
        positions = (center + distances) % arc[-1]
        return np.column_stack([
            np.interp(positions, arc, points[:, axis]) for axis in (0, 1)
        ])

    def angles(tangents: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(
            np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0])),
            scale / spacing, mode="nearest",
        )

    angle = angles(sample(offsets + 2.0 * scale) - sample(offsets - 2.0 * scale))
    outer = np.abs(offsets) >= window * 0.6
    turn = float(np.median(angle[outer & (offsets > 0)]) - np.median(angle[outer & (offsets < 0)]))
    if abs(turn) < np.radians(25.0) or abs(turn) > np.radians(170.0):
        return None
    local = np.flatnonzero(np.abs(offsets) <= window * 0.5)
    peak = int(local[np.argmax(np.gradient(angle, offsets)[local] * np.sign(turn))])
    peak_offset = offsets[peak]
    before = float(np.median(angle[(offsets >= peak_offset - 10.0 * scale) & (offsets <= peak_offset - 6.0 * scale)]))
    after = float(np.median(angle[(offsets >= peak_offset + 6.0 * scale) & (offsets <= peak_offset + 10.0 * scale)]))
    turn = after - before
    if abs(turn) < np.radians(25.0) or abs(turn) > np.radians(170.0):
        return None

    def extent(profile: np.ndarray, peak_index: int) -> tuple[int, int] | None:
        progress = (profile - before) / turn
        left = np.flatnonzero((np.arange(len(offsets)) < peak_index) & (progress <= 0.1))
        right = np.flatnonzero((np.arange(len(offsets)) > peak_index) & (progress >= 0.9))
        return (int(left[-1]), int(right[0])) if len(left) and len(right) else None

    measured = extent(angle, peak)
    incoming = np.array([np.cos(before), np.sin(before)])
    outgoing = np.array([np.cos(after), np.sin(after)])

    def point_corner(distances: np.ndarray) -> np.ndarray:
        return np.where((distances < 0)[:, None], distances[:, None] * incoming, distances[:, None] * outgoing)

    ideal = angles(point_corner(offsets + 2.0 * scale) - point_corner(offsets - 2.0 * scale))
    ideal += round((before - ideal[0]) / (2.0 * np.pi)) * 2.0 * np.pi
    sharp = extent(ideal, len(offsets) // 2)
    if measured is None or sharp is None:
        return None
    width = offsets[measured[1]] - offsets[measured[0]]
    blur_width = offsets[sharp[1]] - offsets[sharp[0]]
    # A turn spread through most of the support is side curvature, not a
    # localized corner transition. Never promote that support to ownership.
    if width > window * 0.75:
        return None
    if point_only:
        # A rounded turn already has entry/exit localization below; moving its
        # seed to the peak can incorrectly erase that rounded classification.
        if width - blur_width > max(scale, aa_margin):
            return None
        position = (center + peak_offset) % arc[-1]
        distances = np.abs(arc[:-1] - position)
        peak_index = int(np.argmin(np.minimum(distances, arc[-1] - distances)))
        return peak_index, peak_index
    if not rounding_known and width - blur_width <= max(scale, aa_margin):
        return None
    indices = []
    for boundary in measured:
        position = (center + offsets[boundary]) % arc[-1]
        distances = np.abs(arc[:-1] - position)
        indices.append(int(np.argmin(np.minimum(distances, arc[-1] - distances))))
    return (indices[0], indices[1]) if indices[0] != indices[1] else None


def corner_turn_index_at(
    contour: np.ndarray, index: int, *, scale: float, aa_margin: float,
) -> int:
    """Localize an existing detection on the raw ring before classification.

    This does not detect additional corners or change source coordinates.
    Rounded and unsupported turns retain their seed: rounded turns use their
    independently localized entry/exit span, not a point apex.
    """
    location = _corner_turn_at(contour, index, scale=scale, aa_margin=aa_margin, point_only=True)
    return index if location is None else location[0]


def corner_curve_span_at(
    contour: np.ndarray, index: int, *, scale: float, aa_margin: float,
    rounding_known: bool = False,
) -> tuple[int, int] | None:
    """Locate the AA-aware rounded extent of an established source turn."""
    return _corner_turn_at(
        contour, index, scale=scale, aa_margin=aa_margin,
        rounding_known=rounding_known,
    )


def region_corner_radius(mask: np.ndarray, *, coverage: CoverageField | np.ndarray | None = None) -> float:
    """One representative corner-fillet radius (px) for the shape in `mask`, measured from
    its outer contour; 0.0 for a sharp-cornered shape. rdp-approximates the contour to
    find its corners, measures the angle-corrected fillet radius at each (see
    _corner_radius_at), and returns the median — padded by _CORNER_DEANTIALIAS_PAD only
    when a real fillet is detected, so a sharp corner reads exactly 0.0. Pass `coverage`
    to extract from the smooth soft field instead of the bilevel mask."""
    cs = region_contours(mask, coverage=coverage)
    if not cs or len(cs[0]) < 12:
        return 0.0
    contour = cs[0]
    verts = rdp(contour, _CORNER_RDP_EPS)
    if len(verts) >= 2 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]                           # drop the duplicated closing point
    n = len(verts)
    if n < 3:
        return 0.0
    idx = [int(np.argmin(np.sum((contour - v) ** 2, axis=1))) for v in verts]
    radii: list[float] = []
    for i in range(n):
        before = _slice_loop(contour, idx[(i - 1) % n], idx[i])
        after = _slice_loop(contour, idx[i], idx[(i + 1) % n])
        r = _corner_radius_at(before, after, contour[idx[i]])
        if r is not None:
            radii.append(r)
    if not radii:
        return 0.0
    radii.sort()
    median_r = radii[len(radii) // 2]
    if median_r < _CORNER_MIN_FILLET:
        return 0.0                                   # sharp -> exactly 0 (no pad)
    return round(median_r + _CORNER_DEANTIALIAS_PAD, 1)

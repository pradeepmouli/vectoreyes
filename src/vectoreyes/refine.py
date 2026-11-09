"""Symmetry-locked fitting of axis-straddling regions.

A straddling region (the dome, the tip) is fit by its RIGHT-half outline only;
the closed path is then `half + mirror(half)`, so the result is *exactly*
symmetric about the axis — unlike `fit_path` on the raw contour, which inherits
the raster's left/right wobble. Straight runs become `L`, curved runs become
cubic Béziers (Schneider), corners preserved.
"""

from __future__ import annotations

import numpy as np

from scipy.optimize import least_squares

from ._fitcurve import fit_quadratic_beziers
from .contour import rdp
from .fit import Shape, _fmt, _max_point_to_polyline, _segment_is_straight, fit_path

_KAPPA = 0.5522847498
_FILLET_K = 0.5522847498  # cubic handle for a quarter-circle-ish fillet


def _fillet_seg(p_in: np.ndarray, corner: np.ndarray, p_out: np.ndarray) -> tuple:
    """A cubic fillet from p_in to p_out that bulges toward the sharp `corner`.
    Shared by every segment so all rounded corners look identical."""
    c1 = p_in + _FILLET_K * (corner - p_in)
    c2 = p_out + _FILLET_K * (corner - p_out)
    return ("C", c1, c2, p_out)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.hypot(*v)
    return v / n if n else v


def _tangent_in(prev_node: np.ndarray, seg: tuple) -> np.ndarray:
    """Unit tangent pointing INTO the segment's end node."""
    if seg[0] == "L":
        return _unit(seg[1] - prev_node)
    if seg[0] == "Q":
        return _unit(seg[2] - seg[1])      # end - control
    return _unit(seg[3] - seg[2])          # cubic: end - 2nd control


def _tangent_out(node: np.ndarray, seg: tuple) -> np.ndarray:
    """Unit tangent pointing OUT of the segment's start node."""
    return _unit(seg[1] - node)            # first control / endpoint - start


def _round_corners(start: np.ndarray, segs: list[tuple], seg_corner: list[bool], r: float):
    """Replace each flagged sharp corner (the start node of segs[j]) with a fillet
    of radius `r`: trim the incoming and outgoing edges back by r and bridge them
    with `_fillet_seg`. Quadratic out-segments stay tangent because the trimmed
    start slides along their own control handle."""
    nodes = [start] + [(s[1] if s[0] == "L" else s[-1]) for s in segs]
    new_segs: list[tuple] = []
    for j, s in enumerate(segs):
        if seg_corner[j] and new_segs:
            corner = nodes[j]
            in_dir = _tangent_in(nodes[j - 1], segs[j - 1])
            out_dir = _tangent_out(corner, s)
            # clamp the trim so it never eats more than ~40% of either neighbour
            len_in = np.hypot(*(corner - nodes[j - 1]))
            len_out = np.hypot(*(nodes[j + 1] - corner))
            rr = min(r, 0.4 * len_in, 0.4 * len_out)
            if rr > 0.5:
                t_in = corner - rr * in_dir
                t_out = corner + rr * out_dir
                prev = new_segs[-1]
                new_segs[-1] = (*prev[:-1], t_in)   # retarget the incoming edge's end
                new_segs.append(_fillet_seg(t_in, corner, t_out))
        new_segs.append(s)
    return start, new_segs


def _longest_true_run_circular(mask: np.ndarray) -> tuple[int, int]:
    """(start_index, length) of the longest run of True in a circular bool array."""
    n = len(mask)
    if mask.all():
        return 0, n
    if not mask.any():
        return 0, 0
    off = int(np.argmin(mask))           # rotate to start at a False
    r = np.roll(mask, -off)
    best_start, best_len, i = 0, 0, 0
    while i < n:
        if r[i]:
            j = i
            while j < n and r[j]:
                j += 1
            if j - i > best_len:
                best_start, best_len = i, j - i
            i = j
        else:
            i += 1
    return (best_start + off) % n, best_len


def _side_half(contour: np.ndarray, axis_x: float, *, side: str) -> np.ndarray | None:
    """The contour arc on one side of the axis, ordered top (min y) -> bottom."""
    n = len(contour)
    axis_tol = 1e-6
    if side == "right":
        mask = contour[:, 0] >= axis_x - axis_tol
    elif side == "left":
        mask = contour[:, 0] <= axis_x + axis_tol
    else:
        raise ValueError(f"unsupported side: {side!r}")
    start, length = _longest_true_run_circular(mask)
    if length < 3:
        return None
    half = contour[[(start + k) % n for k in range(length)]].astype(float)
    if half[0, 1] > half[-1, 1]:          # ensure top -> bottom
        half = half[::-1].copy()
    return half


def _dedupe_consecutive_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    out = [points[0]]
    for point in points[1:]:
        if not np.allclose(point, out[-1]):
            out.append(point)
    return np.asarray(out, dtype=float)


def _axis_crossings(contour: np.ndarray, axis_x: float) -> list[np.ndarray]:
    crossings: list[np.ndarray] = []
    for a, b in zip(contour, np.vstack([contour[1:], contour[:1]]), strict=False):
        ax = float(a[0] - axis_x)
        bx = float(b[0] - axis_x)
        if abs(ax) <= 1e-6:
            crossings.append(np.array([axis_x, float(a[1])], dtype=float))
            continue
        if ax * bx >= 0.0 or abs(float(b[0] - a[0])) <= 1e-9:
            continue
        t = float((axis_x - a[0]) / (b[0] - a[0]))
        if 0.0 <= t <= 1.0:
            crossings.append(np.array([axis_x, float(a[1] + t * (b[1] - a[1]))], dtype=float))
    return crossings


def _axis_bounded_side_half(contour: np.ndarray, axis_x: float, *, side: str, epsilon: float) -> np.ndarray | None:
    half = _side_half(contour, axis_x, side=side)
    if half is None or len(half) < 3:
        return None
    half = _dedupe_consecutive_points(half)
    if len(half) < 3:
        return None

    crossings = _axis_crossings(np.asarray(contour, dtype=float), axis_x)
    y_tol = max(1.5, epsilon * 1.5)

    def axis_point_near(point: np.ndarray) -> np.ndarray | None:
        if not crossings:
            return None
        crossing = min(crossings, key=lambda candidate: abs(float(candidate[1]) - float(point[1])))
        if abs(float(crossing[1]) - float(point[1])) <= y_tol:
            return crossing
        return None

    if abs(float(half[0][0]) - axis_x) > 1.0:
        crossing = axis_point_near(half[0])
        if crossing is not None:
            half = np.vstack([crossing, half])
    elif abs(float(half[0][0]) - axis_x) <= 1.0:
        half[0, 0] = axis_x

    if abs(float(half[-1][0]) - axis_x) > 1.0:
        crossing = axis_point_near(half[-1])
        if crossing is not None:
            half = np.vstack([half, crossing])
    elif abs(float(half[-1][0]) - axis_x) <= 1.0:
        half[-1, 0] = axis_x

    half = _dedupe_consecutive_points(half)
    if len(half) < 3:
        return None
    if abs(float(half[0][0]) - axis_x) > 1.0 or abs(float(half[-1][0]) - axis_x) > 1.0:
        return None
    if not np.allclose(half[0], half[-1]):
        half = np.vstack([half, half[0]])
    return half


def _right_half(contour: np.ndarray, axis_x: float) -> np.ndarray | None:
    """The contour arc on the x >= axis side, ordered top (min y) -> bottom."""
    return _side_half(contour, axis_x, side="right")


def _open_corners(poly: np.ndarray, *, angle_threshold_deg: float = 40.0) -> list[int]:
    """Indices of interior vertices of an OPEN polyline whose turn exceeds the threshold."""
    thresh = np.radians(angle_threshold_deg)
    out: list[int] = []
    for i in range(1, len(poly) - 1):
        v1, v2 = poly[i] - poly[i - 1], poly[i + 1] - poly[i]
        if np.hypot(*v1) == 0 or np.hypot(*v2) == 0:
            continue
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(np.arctan2(cross, np.dot(v1, v2))) >= thresh:
            out.append(i)
    return out


def _fit_open_segments(pts: np.ndarray, epsilon: float, max_error: float, corner_radius: float = 0.0):
    """Fit an open polyline to a start point + list of ('L', p) | ('Q', c, p2).

    Curved runs become *quadratic* Béziers — inflection-free by construction — so
    the mirrored result is convex-only and cannot wobble into an S-curve. When
    `corner_radius` > 0 the sharp corners between runs are rounded with that shared
    radius so this shape's corners match every other segment's.
    """
    simp = rdp(pts, epsilon)
    corner_pts = simp[_open_corners(simp)] if len(simp) > 2 else np.empty((0, 2))
    corner_idx = {int(np.argmin(np.hypot(*(pts - cp).T))) for cp in corner_pts}
    cuts = sorted({0, len(pts) - 1} | corner_idx)
    segs: list[tuple] = []
    seg_corner: list[bool] = []      # True where a seg's START node is a sharp corner
    for k in range(len(cuts) - 1):
        sub = pts[cuts[k]:cuts[k + 1] + 1]
        if len(sub) < 2:
            continue
        at_corner = cuts[k] in corner_idx
        if _segment_is_straight(sub, epsilon):
            segs.append(("L", sub[-1].copy()))
            seg_corner.append(at_corner)
        else:
            for i, b in enumerate(fit_quadratic_beziers(sub, max_error)):
                segs.append(("Q", b[1].copy(), b[2].copy()))
                seg_corner.append(at_corner and i == 0)
    start = pts[0].copy()
    if corner_radius > 0:
        start, segs = _round_corners(start, segs, seg_corner, corner_radius)
    return start, segs


def _emit_symmetric(start: np.ndarray, segs: list[tuple], axis_x: float) -> str:
    """Closed, exactly-symmetric path: forward half, then mirror-reverse the half."""
    f = _fmt

    def mir(p):
        return (2 * axis_x - p[0], p[1])

    starts = [start]
    for s in segs:
        starts.append(s[1] if s[0] == "L" else s[-1])

    d = f"M{f(start[0])} {f(start[1])} "
    for s in segs:
        if s[0] == "L":
            d += f"L{f(s[1][0])} {f(s[1][1])} "
        elif s[0] == "Q":
            d += f"Q{f(s[1][0])} {f(s[1][1])} {f(s[2][0])} {f(s[2][1])} "
        else:  # cubic
            d += (f"C{f(s[1][0])} {f(s[1][1])} {f(s[2][0])} {f(s[2][1])} "
                  f"{f(s[3][0])} {f(s[3][1])} ")
    for i in range(len(segs) - 1, -1, -1):
        s, p_prev = segs[i], starts[i]
        if s[0] == "L":
            m = mir(p_prev)
            d += f"L{f(m[0])} {f(m[1])} "
        elif s[0] == "Q":
            # reversing a quadratic swaps endpoints; the single control point stays
            c, p0 = mir(s[1]), mir(p_prev)
            d += f"Q{f(c[0])} {f(c[1])} {f(p0[0])} {f(p0[1])} "
        else:  # cubic
            c2, c1, p0 = mir(s[2]), mir(s[1]), mir(p_prev)
            d += f"C{f(c2[0])} {f(c2[1])} {f(c1[0])} {f(c1[1])} {f(p0[0])} {f(p0[1])} "
    return d + "Z"


def _emit_half(start: np.ndarray, segs: list[tuple]) -> str:
    f = _fmt
    d = f"M{f(start[0])} {f(start[1])} "
    for s in segs:
        if s[0] == "L":
            d += f"L{f(s[1][0])} {f(s[1][1])} "
        elif s[0] == "Q":
            d += f"Q{f(s[1][0])} {f(s[1][1])} {f(s[2][0])} {f(s[2][1])} "
        else:
            d += (
                f"C{f(s[1][0])} {f(s[1][1])} {f(s[2][0])} {f(s[2][1])} "
                f"{f(s[3][0])} {f(s[3][1])} "
            )
    return d + "Z"


def _mirror_segs(segs: list[tuple], axis_x: float) -> list[tuple]:
    def mir(p: np.ndarray) -> np.ndarray:
        return np.array([2 * axis_x - p[0], p[1]], dtype=float)

    out: list[tuple] = []
    for s in segs:
        if s[0] == "L":
            out.append(("L", mir(s[1])))
        elif s[0] == "Q":
            out.append(("Q", mir(s[1]), mir(s[2])))
        else:
            out.append(("C", mir(s[1]), mir(s[2]), mir(s[3])))
    return out


def _rounded_trapezoid_segments(
    contour: np.ndarray, axis_x: float, *, radius: float, max_error: float
) -> tuple[np.ndarray, list[tuple]] | None:
    pts = np.asarray(contour, dtype=float)
    y_top, y_bot = float(pts[:, 1].min()), float(pts[:, 1].max())
    height = y_bot - y_top
    if height < 10:
        return None
    r = min(radius, 0.45 * height)

    # right-edge sample points, away from the corner fillets
    sel = (pts[:, 1] > y_top + r) & (pts[:, 1] < y_bot - r) & (pts[:, 0] > axis_x)
    edge = pts[sel]
    if len(edge) < 5:
        return None
    # least-squares right edge as x = m*y + b
    A = np.column_stack([edge[:, 1], np.ones(len(edge))])
    (m, b), *_ = np.linalg.lstsq(A, edge[:, 0], rcond=None)
    if np.abs(edge[:, 0] - (m * edge[:, 1] + b)).max() > max_error * 1.6:
        return None                                  # side isn't a clean straight line

    hw_top = (m * y_top + b) - axis_x
    hw_bot = (m * y_bot + b) - axis_x
    if min(hw_top, hw_bot) <= r:
        return None

    # require BOTH a flat top and a flat bottom edge (a trapezoid), not a rounded
    # apex or a point (dome, tip) — those collapse one extreme row to ~zero width
    # while the other stays full. Compare the two edge widths to each other.
    near_top = pts[pts[:, 1] < y_top + 2.0]
    near_bot = pts[pts[:, 1] > y_bot - 2.0]
    if len(near_top) == 0 or len(near_bot) == 0:
        return None
    top_w = near_top[:, 0].max() - near_top[:, 0].min()
    bot_w = near_bot[:, 0].max() - near_bot[:, 0].min()
    if min(top_w, bot_w) < 0.5 * max(top_w, bot_w):   # one edge is a point -> not a trapezoid
        return None

    down = np.array([m, 1.0]) / np.hypot(m, 1.0)      # unit vector along the edge (top->bottom)

    c0_top = np.array([axis_x + hw_top, y_top])
    c0_bot = np.array([axis_x + hw_bot, y_bot])
    p_top_h = np.array([axis_x + hw_top - r, y_top])   # tangent on the top edge
    p_top_s = c0_top + r * down                        # tangent on the side (below top corner)
    p_bot_s = c0_bot - r * down                        # tangent on the side (above bottom corner)
    p_bot_h = np.array([axis_x + hw_bot - r, y_bot])   # tangent on the bottom edge

    start = np.array([axis_x, y_top])
    segs = [
        ("L", p_top_h),
        _fillet_seg(p_top_h, c0_top, p_top_s),
        ("L", p_bot_s),
        _fillet_seg(p_bot_s, c0_bot, p_bot_h),
        ("L", np.array([axis_x, y_bot])),
    ]
    return start, segs


def rounded_trapezoid_fit(
    contour: np.ndarray, axis_x: float, *, radius: float, max_error: float
) -> Shape | None:
    """Fit a clean, symmetric rounded trapezoid (flat top/bottom, straight
    tapering sides, filleted corners) when the region is band-like. The corner
    `radius` is the shared mark-level value (not a per-segment fraction of height),
    so every band rounds the same. Returns None if the side isn't a clean straight
    taper, so genuinely curved regions fall through to `symmetric_fit`.
    """
    fitted = _rounded_trapezoid_segments(contour, axis_x, radius=radius, max_error=max_error)
    if fitted is None:
        return None
    start, segs = fitted
    return Shape("path", {"d": _emit_symmetric(start, segs, axis_x)})


def rounded_trapezoid_half_fit(
    contour: np.ndarray,
    axis_x: float,
    *,
    side: str,
    radius: float,
    max_error: float,
) -> Shape | None:
    fitted = _rounded_trapezoid_segments(contour, axis_x, radius=radius, max_error=max_error)
    if fitted is None:
        return None
    start, segs = fitted
    if side == "left":
        start = np.array([2 * axis_x - start[0], start[1]], dtype=float)
        segs = _mirror_segs(segs, axis_x)
    elif side != "right":
        raise ValueError(f"unsupported side: {side!r}")
    return Shape("path", {"d": _emit_half(start, segs)})


def _half_ellipse_cap_d(axis_x: float, y_bot: float, rx: float, ry: float, r: float = 0.0) -> str:
    """Flat-bottomed half-ellipse (dome) as two kappa quarter-arcs + flat base.

    With `r` > 0 the two base corners are rounded by that shared radius: the dome
    becomes the upper half of an ellipse centred `r` above the base (radii rx,
    ry-r) so its sides still meet the fillets with a vertical tangent, then a
    quarter-circle fillet drops to the flat base — tangent-continuous, exactly
    symmetric, and rounded to match every other segment.
    """
    k = _KAPPA
    f = _fmt
    left, right = axis_x - rx, axis_x + rx
    if r <= 0.5:
        y_top = y_bot - ry
        return (
            f"M{f(left)} {f(y_bot)} "
            f"C{f(left)} {f(y_bot - k * ry)} {f(axis_x - k * rx)} {f(y_top)} {f(axis_x)} {f(y_top)} "
            f"C{f(axis_x + k * rx)} {f(y_top)} {f(right)} {f(y_bot - k * ry)} {f(right)} {f(y_bot)} "
            f"Z"
        )
    r = min(r, 0.45 * rx, 0.45 * ry)
    cy = y_bot - r          # centre of the dome ellipse / fillet centres' y
    ry2 = ry - r
    y_top = y_bot - ry      # = cy - ry2
    return (
        f"M{f(left)} {f(cy)} "
        # dome: upper half-ellipse (radii rx, ry2) about (axis, cy)
        f"C{f(left)} {f(cy - k * ry2)} {f(axis_x - k * rx)} {f(y_top)} {f(axis_x)} {f(y_top)} "
        f"C{f(axis_x + k * rx)} {f(y_top)} {f(right)} {f(cy - k * ry2)} {f(right)} {f(cy)} "
        # right base fillet -> flat base -> left base fillet
        f"C{f(right)} {f(cy + k * r)} {f(right - r + k * r)} {f(y_bot)} {f(right - r)} {f(y_bot)} "
        f"L{f(left + r)} {f(y_bot)} "
        f"C{f(left + r - k * r)} {f(y_bot)} {f(left)} {f(cy + k * r)} {f(left)} {f(cy)} "
        f"Z"
    )


def _half_ellipse_cap_half_d(axis_x: float, y_bot: float, rx: float, ry: float, *, side: str) -> str:
    """One side of a flat-bottomed half-ellipse cap, bounded by the symmetry axis."""
    k = _KAPPA
    f = _fmt
    y_top = y_bot - ry
    if side == "left":
        left = axis_x - rx
        return (
            f"M{f(axis_x)} {f(y_top)} "
            f"C{f(axis_x - k * rx)} {f(y_top)} {f(left)} {f(y_bot - k * ry)} {f(left)} {f(y_bot)} "
            f"L{f(axis_x)} {f(y_bot)} "
            f"Z"
        )
    if side == "right":
        right = axis_x + rx
        return (
            f"M{f(axis_x)} {f(y_top)} "
            f"C{f(axis_x + k * rx)} {f(y_top)} {f(right)} {f(y_bot - k * ry)} {f(right)} {f(y_bot)} "
            f"L{f(axis_x)} {f(y_bot)} "
            f"Z"
        )
    raise ValueError(f"unsupported cap side: {side!r}")


def _half_ellipse_cap_params(
    contour: np.ndarray,
    axis_x: float,
    *,
    max_error: float,
) -> tuple[float, float, float] | None:
    pts = np.asarray(contour, dtype=float)
    y_bot = float(pts[:, 1].max())
    arc = pts[pts[:, 1] < y_bot - 1.0]               # exclude the flat base
    if len(arc) < 10:
        return None
    # the base must be genuinely flat (wide), else it's a point/teardrop, not a cap
    base = pts[pts[:, 1] > y_bot - 2.0]
    base_w = base[:, 0].max() - base[:, 0].min()
    # the base must be the wide, flat edge of a cap — not a point (teardrop/tip)
    if base_w < 0.7 * (pts[:, 0].max() - pts[:, 0].min()):
        return None
    rx0 = max(base_w / 2.0, 1.0)
    ry0 = max(y_bot - float(arc[:, 1].min()), 1.0)
    x = arc[:, 0] - axis_x
    y = arc[:, 1] - y_bot

    def resid(p):
        rx, ry = p
        return np.hypot(x / rx, y / ry) - 1.0        # algebraic ellipse residual

    sol = least_squares(resid, [rx0, ry0], bounds=([1.0, 1.0], [np.inf, np.inf]))
    rx, ry = float(sol.x[0]), float(sol.x[1])
    # convert the algebraic residual to an approximate pixel distance
    if np.abs(resid(sol.x)).max() * min(rx, ry) > max_error * 2.0:
        return None
    return y_bot, rx, ry


def half_ellipse_cap_fit(
    contour: np.ndarray, axis_x: float, *, corner_radius: float = 0.0, max_error: float
) -> Shape | None:
    """Primitive-deform: fit a flat-bottomed half-ellipse (dome) to the contour
    via scipy least-squares over (rx, ry), then emit it exactly symmetric. Cleaner
    than the free half-outline fit when the region really is a half-ellipse cap.
    Returns None if the arc doesn't match an ellipse within tolerance.
    """
    params = _half_ellipse_cap_params(contour, axis_x, max_error=max_error)
    if params is None:
        return None
    y_bot, rx, ry = params
    return Shape("path", {"d": _half_ellipse_cap_d(axis_x, y_bot, rx, ry, corner_radius)})


def half_ellipse_cap_half_fit(
    contour: np.ndarray,
    axis_x: float,
    *,
    side: str,
    max_error: float,
) -> Shape | None:
    """Fit one half of a flat-bottomed cap for internal mirror reconstruction."""
    params = _half_ellipse_cap_params(contour, axis_x, max_error=max_error)
    if params is None:
        return None
    y_bot, rx, ry = params
    return Shape("path", {"d": _half_ellipse_cap_half_d(axis_x, y_bot, rx, ry, side=side)})


def fit_path_half_fit(
    contour: np.ndarray,
    axis_x: float,
    *,
    side: str,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape | None:
    """Fit a closed half-region bounded by the symmetry axis with the generic path fitter."""
    half = _axis_bounded_side_half(contour, axis_x, side=side, epsilon=epsilon)
    if half is None:
        return None
    return fit_path(half, epsilon=epsilon, max_error=max_error, cubic=cubic, prefer_simple_curves=True)


def symmetric_polygon_fit(contour: np.ndarray, axis_x: float, *, epsilon: float, max_vertices: int = 10) -> Shape | None:
    """Exactly-symmetric straight-edged polygon: simplify the right-half outline to
    line segments and mirror it. Keeps SHARP corners (a diamond stays a diamond),
    unlike `symmetric_fit` which curve-fits every run. Returns None when the half-
    outline isn't cleanly polygonal, so genuinely curved regions fall through.
    """
    half = _right_half(contour, axis_x)
    if half is None or len(half) < 2:
        return None
    simp = rdp(half, epsilon).astype(float)
    if not (3 <= len(simp) <= max_vertices):
        return None
    # A real polygon edge deviates only by anti-aliasing noise (well under epsilon);
    # a curve approximated by a polyline sits at the rdp epsilon ceiling. Gate at a
    # fraction of epsilon so a high-curvature arc isn't faceted into a polygon.
    # (Measured: polygon edges ~0.8px, curves ~1.2-1.5px at epsilon=1.5.)
    if _max_point_to_polyline(half, simp) > 0.65 * epsilon:
        return None
    # ...but a SMALL gentle arc can also sit under that bar, so additionally require
    # every interior vertex to be a genuine SHARP corner. A diamond has one sharp
    # turn; a faceted curve spreads many ~25° turns — which fail this gate.
    if len(_open_corners(simp, angle_threshold_deg=30.0)) != len(simp) - 2:
        return None
    simp[0, 0] = axis_x          # pin the top axis-crossing
    simp[-1, 0] = axis_x         # pin the bottom axis-crossing
    start = simp[0].copy()
    segs = [("L", p.copy()) for p in simp[1:]]
    return Shape("path", {"d": _emit_symmetric(start, segs, axis_x)})


def symmetric_fit(
    contour: np.ndarray, axis_x: float, *, corner_radius: float = 0.0, epsilon: float, max_error: float
) -> Shape | None:
    """Fit a straddling region's half-outline and mirror it → exactly-symmetric path.
    Sharp corners are rounded with the shared `corner_radius` so the cone's corners
    match the bands and the cap."""
    half = _right_half(contour, axis_x)
    if half is None or len(half) < 3:
        return None
    start, segs = _fit_open_segments(half, epsilon, max_error, corner_radius)
    if not segs:
        return None
    start[0] = axis_x                      # pin apex to the axis
    last = segs[-1]
    (last[1] if last[0] == "L" else last[-1])[0] = axis_x   # pin bottom to the axis
    return Shape("path", {"d": _emit_symmetric(start, segs, axis_x)})


def symmetric_half_fit(
    contour: np.ndarray,
    axis_x: float,
    *,
    side: str,
    corner_radius: float = 0.0,
    epsilon: float,
    max_error: float,
) -> Shape | None:
    """Fit one side of a straddling region for internal mirror reconstruction."""
    half = _side_half(contour, axis_x, side=side)
    if half is None or len(half) < 3:
        return None
    half = _dedupe_consecutive_points(half)
    if len(half) < 3:
        return None
    if abs(float(half[0][0]) - axis_x) > 1.0:
        crossings = _axis_crossings(np.asarray(contour, dtype=float), axis_x)
        if crossings:
            top_crossing = min(crossings, key=lambda point: abs(float(point[1]) - float(half[0][1])))
            if abs(float(top_crossing[1]) - float(half[0][1])) <= max(1.5, epsilon * 1.5):
                half = np.vstack([top_crossing, half])
    prefix: list[tuple] = []
    start = half[0].copy()
    if (
        abs(float(start[0]) - axis_x) <= 1.0
        and abs(float(half[1][1]) - float(start[1])) <= max(1.5, epsilon * 1.5)
        and abs(float(half[1][0]) - axis_x) >= 4.0
    ):
        prefix.append(("L", half[1].copy()))
        fit_half = half[1:]
    else:
        fit_half = half
    fit_start, fit_segs = _fit_open_segments(fit_half, epsilon, max_error, corner_radius)
    if not fit_segs:
        return None
    if prefix:
        start = half[0].copy()
        segs = [*prefix, *fit_segs]
    else:
        start = fit_start
        segs = fit_segs
    start[0] = axis_x
    last = segs[-1]
    (last[1] if last[0] == "L" else last[-1])[0] = axis_x
    return Shape("path", {"d": _emit_half(start, segs)})

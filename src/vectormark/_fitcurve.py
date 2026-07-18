"""Convex-only (quadratic-Bézier) curve fitting.

A quadratic Bézier is an affine image of the parabola ``t -> (t, t**2)``: its
second derivative is constant, so its curvature never changes sign. It is
therefore *inflection-free* — it cannot wobble into an S-curve the way a fitted
cubic can when it chases anti-aliasing noise. We fit each curved run with one or
more quadratics (subdividing on error), so every emitted free arc is convex.

Structure follows Schneider's cubic fitter (Graphics Gems, 1990) — chord-length
parameterize, solve for the single interior control point by least squares,
Newton-reparameterize, recurse on the worst point — but with the quadratic basis,
which has one free control point instead of two. Returns ``(3, 2)`` arrays.
"""

from __future__ import annotations

import numpy as np

qbezier = lambda ctrl, t: (  # noqa: E731 - quadratic Bézier point at t
    (1 - t) ** 2 * ctrl[0]
    + 2 * (1 - t) * t * ctrl[1]
    + t ** 2 * ctrl[2]
)


def fit_quadratic_beziers(points: np.ndarray, max_error: float) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return []
    return _fit_quadratic(pts, max_error)


def _fit_quadratic(pts, error):
    if len(pts) == 2:
        return [np.array([pts[0], (pts[0] + pts[1]) / 2.0, pts[1]])]
    u = _chord_length_parameterize(pts)
    ctrl = _generate_quadratic(pts, u)
    max_err, split = _compute_max_error(pts, ctrl, u)
    if max_err < error * error:
        return [ctrl]
    for _ in range(20):
        u = _reparameterize(pts, u, ctrl)
        ctrl = _generate_quadratic(pts, u)
        max_err, split = _compute_max_error(pts, ctrl, u)
        if max_err < error * error:
            return [ctrl]
    left = _fit_quadratic(pts[: split + 1], error)
    right = _fit_quadratic(pts[split:], error)
    return left + right


def _generate_quadratic(pts, u):
    """Closed-form least-squares interior control point P1 (endpoints pinned).

    For B(u) = (1-u)^2 P0 + 2(1-u)u P1 + u^2 P2 the only unknown is P1; the
    residual is linear in P1 with weight w = 2(1-u)u, so P1 minimizing
    sum |B(u_i) - p_i|^2 is sum(w_i r_i) / sum(w_i^2), per axis.
    """
    first, last = pts[0], pts[-1]
    w = 2 * (1 - u) * u
    rhs = pts - ((1 - u) ** 2)[:, None] * first - (u ** 2)[:, None] * last
    denom = float((w * w).sum())
    if denom < 1e-12:
        p1 = (first + last) / 2.0
    else:
        p1 = (w[:, None] * rhs).sum(axis=0) / denom
    return np.array([first, p1, last])


def _reparameterize(pts, u, ctrl):
    u = np.asarray(u, dtype=float)
    t = u[:, None]
    v = 1.0 - t
    point = v * v * ctrl[0] + 2.0 * v * t * ctrl[1] + t * t * ctrl[2]
    delta = point - pts
    derivative = 2.0 * (v * (ctrl[1] - ctrl[0]) + t * (ctrl[2] - ctrl[1]))
    second_derivative = 2.0 * (ctrl[2] - 2.0 * ctrl[1] + ctrl[0])
    denominator = (
        np.einsum("ij,ij->i", derivative, derivative)
        + np.einsum("ij,j->i", delta, second_derivative)
    )
    numerator = np.einsum("ij,ij->i", delta, derivative)
    result = u.copy()
    valid = denominator != 0.0
    result[valid] -= numerator[valid] / denominator[valid]
    return result


def _newton(p, u, ctrl):
    d = qbezier(ctrl, u) - p
    q1 = 2 * (1 - u) * (ctrl[1] - ctrl[0]) + 2 * u * (ctrl[2] - ctrl[1])
    q2 = 2 * (ctrl[2] - 2 * ctrl[1] + ctrl[0])
    denom = q1 @ q1 + d @ q2
    return u if denom == 0 else u - (d @ q1) / denom


def _chord_length_parameterize(pts):
    u = np.zeros(len(pts))
    u[1:] = np.cumsum(np.hypot(*np.diff(pts, axis=0).T))
    return u / u[-1] if u[-1] else u


def _compute_max_error(pts, ctrl, u):
    u = np.asarray(u, dtype=float)[:, None]
    v = 1.0 - u
    fitted = v * v * ctrl[0] + 2.0 * v * u * ctrl[1] + u * u * ctrl[2]
    delta = fitted - pts
    errs = np.einsum("ij,ij->i", delta, delta)
    split = int(errs.argmax())
    return float(errs[split]), max(1, min(split, len(pts) - 2))


def fit_quadratic_once(points: np.ndarray, max_error: float) -> tuple[np.ndarray, float, int]:
    """Fit one quadratic, returning its squared residual and useful split."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 2:
        control = (pts[0] + pts[1]) / 2.0
        return np.array([pts[0], control, pts[1]]), 0.0, 1
    u = _chord_length_parameterize(pts)
    ctrl = _generate_quadratic(pts, u)
    max_err, split = _compute_max_error(pts, ctrl, u)
    if max_err < max_error * max_error:
        for _ in range(20):
            u = _reparameterize(pts, u, ctrl)
            ctrl = _generate_quadratic(pts, u)
            max_err, split = _compute_max_error(pts, ctrl, u)
            if max_err < max_error * max_error:
                break
    return ctrl, max_err, split


# --- Inflection-guarded cubic fitting ----------------------------------------
# A cubic Bézier has up to two interior inflection points (curvature sign
# changes). We fit each curved run with Schneider's full cubic least-squares,
# then guard: if the fitted cubic inflects inside (0,1) we split the run at the
# inflection and refit each half, so every emitted arc stays inflection-free
# (no S-curve wobble) while still carrying the extra degree of freedom a
# quadratic lacks on a convex run.

cbezier = lambda ctrl, t: (  # noqa: E731 - cubic Bézier point at t
    (1 - t) ** 3 * ctrl[0]
    + 3 * (1 - t) ** 2 * t * ctrl[1]
    + 3 * (1 - t) * t ** 2 * ctrl[2]
    + t ** 3 * ctrl[3]
)


def _unit(v):
    n = float(np.hypot(v[0], v[1]))
    return v / n if n else np.asarray(v, dtype=float)


def _cross(u, v):
    return float(u[0] * v[1] - u[1] * v[0])


def cubic_inflects(ctrl) -> list[float]:
    """Parameter values t in (0,1) where a planar cubic's curvature changes sign.

    cross(B'(t), B''(t)) is a quadratic in t (the cubic leading term cancels);
    its real roots in the open interval are the inflection points. At most two.
    """
    p0, p1, p2, p3 = (np.asarray(c, dtype=float) for c in ctrl)
    a, b, c = p1 - p0, p2 - p1, p3 - p2          # first differences of control net
    # B'(t) ∝ A0 + A1 t + A2 t^2 ; B''(t) ∝ B0 + B1 t  (B1 == A2)
    a0, a1, a2 = a, 2 * (b - a), a - 2 * b + c
    b0 = b - a
    c2 = _cross(a1, a2) + _cross(a2, b0)
    c1 = _cross(a0, a2) + _cross(a1, b0)
    c0 = _cross(a0, b0)
    def curvature_poly(t: float) -> float:
        return (c2 * t + c1) * t + c0

    def real_sign_change(t: float) -> bool:
        delta = 1e-4
        left = curvature_poly(max(0.0, t - delta))
        right = curvature_poly(min(1.0, t + delta))
        scale = max(abs(c0), abs(c1), abs(c2), 1.0)
        tol = 1e-9 * scale
        if abs(left) <= tol and abs(right) <= tol:
            return False
        return left * right < 0.0

    roots: list[float] = []
    if abs(c2) < 1e-12:
        if abs(c1) > 1e-12:
            t = -c0 / c1
            if 1e-6 < t < 1.0 - 1e-6 and real_sign_change(t):
                roots.append(t)
    else:
        disc = c1 * c1 - 4 * c2 * c0
        if disc >= 0:
            s = disc ** 0.5
            for t in ((-c1 - s) / (2 * c2), (-c1 + s) / (2 * c2)):
                if 1e-6 < t < 1.0 - 1e-6 and real_sign_change(float(t)):
                    roots.append(float(t))
    return sorted(roots)


def fit_cubic_beziers(
    points: np.ndarray,
    max_error: float,
    *,
    guard_inflections: bool = True,
) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return []
    return _fit_cubic(pts, max_error, guard_inflections=guard_inflections)


def _endpoint_tangents(pts):
    """Robust inward unit tangents at the two endpoints.

    A single-point baseline (``pts[1]-pts[0]``) is fragile on quantization
    staircases: the first step is often an axis-aligned tread perpendicular to
    the true edge direction, throwing the control point sideways. Estimate each
    tangent over a short multi-point baseline instead, which averages the
    staircase tread out toward the run's true direction.
    """
    k = min(3, len(pts) - 1)
    return _unit(pts[k] - pts[0]), _unit(pts[-1 - k] - pts[-1])


def _fit_cubic(pts, error, *, guard_inflections: bool):
    if len(pts) == 2:
        d = (pts[1] - pts[0]) / 3.0
        return [np.array([pts[0], pts[0] + d, pts[1] - d, pts[1]])]
    t_hat1, t_hat2 = _endpoint_tangents(pts)
    u = _chord_length_parameterize(pts)
    ctrl = _generate_cubic(pts, u, t_hat1, t_hat2)
    max_err, split = _compute_max_error_cubic(pts, ctrl, u)
    inflects = cubic_inflects(ctrl) if guard_inflections else []
    squared_error = error * error
    if max_err < squared_error and not inflects:
        return [ctrl]
    if not inflects:
        for _ in range(20):
            u = _reparameterize_cubic(pts, u, ctrl)
            ctrl = _generate_cubic(pts, u, t_hat1, t_hat2)
            max_err, split = _compute_max_error_cubic(pts, ctrl, u)
            inflects = cubic_inflects(ctrl) if guard_inflections else []
            if max_err < squared_error and not inflects:
                return [ctrl]
            if inflects:
                break
    if inflects:
        split = int(np.argmin(np.abs(u - inflects[0])))  # split at the inflection
    split = max(1, min(split, len(pts) - 2))
    return (
        _fit_cubic(pts[: split + 1], error, guard_inflections=guard_inflections)
        + _fit_cubic(pts[split:], error, guard_inflections=guard_inflections)
    )


def _generate_cubic(pts, u, t_hat1, t_hat2):
    """Schneider least-squares: solve the two interior control points along the
    endpoint tangents (Graphics Gems, 1990). Falls back to the Wu/Barsky
    thirds heuristic when the normal equations are singular or yield a
    non-positive tangent magnitude."""
    first, last = pts[0], pts[-1]
    a1 = (3 * (1 - u) ** 2 * u)[:, None] * t_hat1
    a2 = (3 * (1 - u) * u ** 2)[:, None] * t_hat2
    c00 = float((a1 * a1).sum())
    c01 = float((a1 * a2).sum())
    c11 = float((a2 * a2).sum())
    b0 = ((1 - u) ** 3)[:, None] * first
    b1 = (3 * (1 - u) ** 2 * u)[:, None] * first
    b2 = (3 * (1 - u) * u ** 2)[:, None] * last
    b3 = (u ** 3)[:, None] * last
    tmp = pts - (b0 + b1 + b2 + b3)
    x0 = float((a1 * tmp).sum())
    x1 = float((a2 * tmp).sum())
    det = c00 * c11 - c01 * c01
    seg_len = float(np.hypot(*(last - first)))
    eps = 1e-6 * seg_len
    if abs(det) < 1e-12:
        alpha_l = alpha_r = seg_len / 3.0
    else:
        alpha_l = (x0 * c11 - x1 * c01) / det
        alpha_r = (c00 * x1 - c01 * x0) / det
    if alpha_l < eps or alpha_r < eps:
        alpha_l = alpha_r = seg_len / 3.0
    return np.array([first, first + t_hat1 * alpha_l, last + t_hat2 * alpha_r, last])


def _reparameterize_cubic(pts, u, ctrl):
    u = np.asarray(u, dtype=float)
    t = u[:, None]
    v = 1.0 - t
    point = (
        v * v * v * ctrl[0]
        + 3.0 * v * v * t * ctrl[1]
        + 3.0 * v * t * t * ctrl[2]
        + t * t * t * ctrl[3]
    )
    delta = point - pts
    derivative = 3.0 * (
        v * v * (ctrl[1] - ctrl[0])
        + 2.0 * v * t * (ctrl[2] - ctrl[1])
        + t * t * (ctrl[3] - ctrl[2])
    )
    second_derivative = 6.0 * (
        v * (ctrl[2] - 2.0 * ctrl[1] + ctrl[0])
        + t * (ctrl[3] - 2.0 * ctrl[2] + ctrl[1])
    )
    denominator = (
        np.einsum("ij,ij->i", derivative, derivative)
        + np.einsum("ij,ij->i", delta, second_derivative)
    )
    numerator = np.einsum("ij,ij->i", delta, derivative)
    result = u.copy()
    valid = denominator != 0.0
    result[valid] -= numerator[valid] / denominator[valid]
    return result


def _newton_cubic(p, u, ctrl):
    d = cbezier(ctrl, u) - p
    q1 = 3 * (
        (1 - u) ** 2 * (ctrl[1] - ctrl[0])
        + 2 * (1 - u) * u * (ctrl[2] - ctrl[1])
        + u ** 2 * (ctrl[3] - ctrl[2])
    )
    q2 = 6 * ((1 - u) * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + u * (ctrl[3] - 2 * ctrl[2] + ctrl[1]))
    denom = q1 @ q1 + d @ q2
    return u if denom == 0 else u - (d @ q1) / denom


def _compute_max_error_cubic(pts, ctrl, u):
    u = np.asarray(u, dtype=float)[:, None]
    v = 1.0 - u
    fitted = (
        v * v * v * ctrl[0]
        + 3.0 * v * v * u * ctrl[1]
        + 3.0 * v * u * u * ctrl[2]
        + u * u * u * ctrl[3]
    )
    delta = fitted - pts
    errs = np.einsum("ij,ij->i", delta, delta)
    split = int(errs.argmax())
    return float(errs[split]), max(1, min(split, len(pts) - 2))


def fit_cubic_once(
    points: np.ndarray,
    max_error: float,
    *,
    guard_inflections: bool = True,
) -> tuple[np.ndarray, float, int]:
    """Fit one cubic, returning its squared residual and useful split."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 2:
        delta = (pts[1] - pts[0]) / 3.0
        return np.array([pts[0], pts[0] + delta, pts[1] - delta, pts[1]]), 0.0, 1
    tangent_start, tangent_end = _endpoint_tangents(pts)
    u = _chord_length_parameterize(pts)
    ctrl = _generate_cubic(pts, u, tangent_start, tangent_end)
    max_err, split = _compute_max_error_cubic(pts, ctrl, u)
    if max_err < max_error * max_error and (not guard_inflections or not cubic_inflects(ctrl)):
        for _ in range(20):
            u = _reparameterize_cubic(pts, u, ctrl)
            ctrl = _generate_cubic(pts, u, tangent_start, tangent_end)
            max_err, split = _compute_max_error_cubic(pts, ctrl, u)
            if max_err < max_error * max_error and (not guard_inflections or not cubic_inflects(ctrl)):
                break
    return ctrl, max_err, split

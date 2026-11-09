"""C1/C2: vertical bilateral symmetry detection + region classification."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy import ndimage as ndi

from .types import Axis, Region

Axis2D = namedtuple("Axis2D", "theta cx cy")


def _perimeter(mask: np.ndarray) -> int:
    """Boundary-pixel count: the one-pixel ring of `mask` minus its erosion."""
    if not mask.any():
        return 0
    return int((mask & ~ndi.binary_erosion(mask)).sum())


def reflection_off_count(fg_xy, axis: Axis2D, dist: np.ndarray, *, tol_px: float = 1.5) -> int:
    """Count foreground points whose reflection across `axis` lands farther than
    `tol_px` outside the shape (background distance transform `dist`). Resampling
    free: reflects coordinates, looks up the distance transform — no raster rotate."""
    xs, ys = fg_xy
    dx, dy = np.cos(axis.theta), np.sin(axis.theta)
    vx, vy = xs - axis.cx, ys - axis.cy
    t = vx * dx + vy * dy
    rx = axis.cx + (2.0 * t * dx - vx)
    ry = axis.cy + (2.0 * t * dy - vy)
    h, w = dist.shape
    ri = np.rint(ry).astype(int)
    ci = np.rint(rx).astype(int)
    in_bounds = (ri >= 0) & (ri < h) & (ci >= 0) & (ci < w)
    off = int((~in_bounds).sum())
    if in_bounds.any():
        off += int((dist[ri[in_bounds], ci[in_bounds]] > tol_px).sum())
    return off


K_BAND = 0.3   # off-count / perimeter floor: off <= 30 % of the boundary length.
               # Genuine symmetric regions: 0.00 (flat bands) – 0.27 (asana petals).
               # False over-fires rejected here: icloud cloud ~0.78, pokeball outer ~0.58.
               # Set tight enough to block those while safely above the genuine cluster.


def _fg_xy(mask):
    ys, xs = np.nonzero(mask)
    return xs, ys


def sym_off_ratio(mask: np.ndarray, axis: Axis2D) -> float:
    """Self-reflection off-count / perimeter of `mask` across `axis`.

    Returns 0.0 for an empty mask.  Used in `region_is_self_symmetric` and surfaced
    in pipeline sym_diags so over-fires are visible without re-running detection.
    """
    if not mask.any():
        return 0.0
    dist = ndi.distance_transform_edt(~mask)
    off = reflection_off_count(_fg_xy(mask), axis, dist)
    peri = _perimeter(mask)
    return off / peri if peri > 0 else (0.0 if off == 0 else float("inf"))


def region_is_self_symmetric(region, axis: Axis2D, *, max_off_ratio: float = K_BAND) -> bool:
    return sym_off_ratio(region.mask, axis) <= max_off_ratio


def regions_mirror_pair(a, b, axis: Axis2D, *, max_off_ratio: float = K_BAND) -> bool:
    if getattr(a, "color_hex", None) != getattr(b, "color_hex", None):
        return False
    aa, ab = int(a.mask.sum()), int(b.mask.sum())
    if aa == 0 or ab == 0 or min(aa, ab) / max(aa, ab) < 0.9:
        return False
    dist_b = ndi.distance_transform_edt(~b.mask)
    off = reflection_off_count(_fg_xy(a.mask), axis, dist_b)
    peri_a = _perimeter(a.mask)
    ratio = off / peri_a if peri_a > 0 else (0.0 if off == 0 else float("inf"))
    return ratio <= max_off_ratio


AxisProposal = namedtuple("AxisProposal", "theta cx cy weight")


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def propose_axes(
    regions,
    *,
    theta_steps: int = 12,
    straddle_off_ratio: float = K_BAND,
    pair_off_ratio: float = K_BAND,
):
    props: list[AxisProposal] = []
    ordered = sorted(regions, key=lambda r: r.label)
    # self-axes
    for r in ordered:
        if not r.mask.any():
            continue
        cx, cy = _centroid(r.mask)
        area = int(r.mask.sum())
        for theta in np.linspace(0.0, np.pi, theta_steps, endpoint=False):
            if region_is_self_symmetric(r, Axis2D(float(theta), cx, cy), max_off_ratio=straddle_off_ratio):
                props.append(AxisProposal(float(theta), cx, cy, area))
    # pair-bisectors
    h, w = ordered[0].mask.shape if ordered else (1, 1)
    diag = float(np.hypot(h, w))
    cents = {r.label: _centroid(r.mask) for r in ordered}
    areas = {r.label: int(r.mask.sum()) for r in ordered}
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if a.color_hex != b.color_hex:
                continue
            aa, ab = areas[a.label], areas[b.label]
            if aa == 0 or ab == 0 or min(aa, ab) / max(aa, ab) < 0.9:
                continue
            (ax, ay), (bx, by) = cents[a.label], cents[b.label]
            if np.hypot(bx - ax, by - ay) > 0.5 * diag:
                continue
            theta = (np.arctan2(by - ay, bx - ax) + np.pi / 2) % np.pi
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            if regions_mirror_pair(a, b, Axis2D(float(theta), mx, my), max_off_ratio=pair_off_ratio):
                props.append(AxisProposal(float(theta), mx, my, aa + ab))
    return props


def _offset(theta, cx, cy):
    return cx * np.sin(theta) - cy * np.cos(theta)


def cluster_axes(proposals, *, d_theta: float = 0.09, d_offset: float = 2.0, min_weight: int = 1):
    items = sorted(proposals, key=lambda p: (p.theta, _offset(p.theta, p.cx, p.cy)))
    clusters: list[list[AxisProposal]] = []
    for p in items:
        po = _offset(p.theta, p.cx, p.cy)
        for c in clusters:
            q = c[0]
            if abs(p.theta - q.theta) <= d_theta and abs(po - _offset(q.theta, q.cx, q.cy)) <= d_offset:
                c.append(p)
                break
        else:
            clusters.append([p])
    out = []
    for c in clusters:
        w = float(sum(p.weight for p in c))
        if w < min_weight:
            continue
        tw = sum(p.weight for p in c)
        theta = float(sum(p.theta * p.weight for p in c) / tw)
        cx = float(sum(p.cx * p.weight for p in c) / tw)
        cy = float(sum(p.cy * p.weight for p in c) / tw)
        out.append((Axis2D(theta, cx, cy), w))
    # equal support -> prefer the axis closest to vertical (canonical primary), then deterministic
    out.sort(key=lambda aw: (-aw[1], abs(aw[0].theta - np.pi / 2), aw[0].theta, _offset(*aw[0])))
    return out


SymmetricGroup = namedtuple("SymmetricGroup", "axes straddlers pairs loners")


def _threshold_to_off_ratio(value: float, default_value: float) -> float:
    if default_value >= 1.0:
        return 0.0
    return K_BAND * max(0.0, 1.0 - float(value)) / max(1e-9, 1.0 - default_value)


def detect_symmetry_groups(
    regions,
    *,
    straddle_iou: float = 0.96,
    pair_iou: float = 0.90,
):
    ordered = sorted(regions, key=lambda r: r.label)
    straddle_off_ratio = _threshold_to_off_ratio(straddle_iou, STRADDLE_MIN_IOU)
    pair_off_ratio = _threshold_to_off_ratio(pair_iou, 1.0 - SYM_TOL)
    ranked = cluster_axes(propose_axes(
        ordered,
        straddle_off_ratio=straddle_off_ratio,
        pair_off_ratio=pair_off_ratio,
    ))
    claimed: set[int] = set()
    groups: list[dict] = []   # {"axes":[Axis2D], "straddlers":[], "pairs":[], "members":set}
    for axis, _w in ranked:
        avail = [r for r in ordered if r.label not in claimed]
        straddlers = [
            r for r in avail
            if region_is_self_symmetric(r, axis, max_off_ratio=straddle_off_ratio)
        ]
        pairs = []
        used = {r.label for r in straddlers}
        rest = [r for r in avail if r.label not in used]
        for i, a in enumerate(rest):
            if a.label in used:
                continue
            for b in rest[i + 1:]:
                if b.label in used:
                    continue
                if regions_mirror_pair(a, b, axis, max_off_ratio=pair_off_ratio):
                    pairs.append((a, b)); used.add(a.label); used.add(b.label); break
        members = {r.label for r in straddlers} | used
        if straddlers or pairs:
            claimed |= members
            groups.append({"axes": [axis], "straddlers": straddlers, "pairs": pairs, "members": members})
        else:
            # axis claims no NEW regions: if some existing group's straddlers ALL also
            # satisfy this axis, it is a secondary axis of that figure -> append.
            for g in groups:
                if g["straddlers"] and all(
                    region_is_self_symmetric(r, axis, max_off_ratio=straddle_off_ratio)
                    for r in g["straddlers"]
                ):
                    g["axes"].append(axis)
                    break
    result = [SymmetricGroup(tuple(g["axes"]), g["straddlers"], g["pairs"], []) for g in groups]
    loners = [r for r in ordered if r.label not in claimed]
    result.append(SymmetricGroup((), [], [], loners))
    return result


# One tolerance for "is this bilaterally symmetric?", shared by every acceptance site
# so they cannot disagree. It is a reflection *mismatch* fraction: a shape counts as
# symmetric about an axis when at most SYM_TOL of it fails to overlap its reflection
# (equivalently, reflection IoU >= 1 - SYM_TOL). Using a looser bar for the straddle
# gate than for axis detection let near-symmetric *pointed* regions be force-mirrored
# about an axis their apex misses, splitting the apex into a fork.
SYM_TOL = 0.10

# False straddlers (icloud ~0.904, telegram ~0.917) sit well below this threshold;
# genuine self-symmetric marks (gdrive ~0.995, appstore ~0.966) sit at or above it.
# Half-mirroring a SINGLE region erases its real asymmetry, so the straddle gate needs
# a stricter bar than the pair gate (which matches two genuine mirror twins).
STRADDLE_MIN_IOU = 0.96


def _reflect_cols(mask: np.ndarray, axis_x: float) -> np.ndarray:
    """Reflect a boolean mask across the vertical line x = axis_x (nearest col)."""
    h, w = mask.shape
    cols = np.arange(w)
    src = np.rint(2.0 * axis_x - cols).astype(int)
    valid = (src >= 0) & (src < w)
    out = np.zeros_like(mask)
    out[:, valid] = mask[:, src[valid]]
    return out


def _mismatch(mask: np.ndarray, axis_x: float) -> float:
    refl = _reflect_cols(mask, axis_x)
    inter = np.logical_and(mask, refl).sum()
    union = np.logical_or(mask, refl).sum()
    return 1.0 - (inter / union if union else 1.0)


def detect_axis(silhouette: np.ndarray, *, tol: float = SYM_TOL) -> Axis | None:
    """Find the best vertical symmetry axis; None if mismatch exceeds `tol`.

    Searches candidate columns near the foreground centroid at 0.5px resolution.
    """
    ys, xs = np.nonzero(silhouette)
    if xs.size == 0:
        return None
    cx = round(float(xs.mean()))
    candidates = np.arange(cx - 6, cx + 6 + 0.5, 0.5)
    scored = [(float(_mismatch(silhouette, a)), float(a)) for a in candidates]
    best_mismatch, best_x = min(scored)
    return Axis(x=best_x) if best_mismatch <= tol else None


def _axis_mismatch(
    fg_xy: tuple[np.ndarray, np.ndarray], cx: float, cy: float, theta: float,
    dist: np.ndarray, *, tol_px: float = 1.5,
) -> float:
    """Fraction of foreground points whose reflection across the line through
    (cx, cy) at angle `theta` lands off the shape (farther than `tol_px`).

    Resampling-free: it reflects point *coordinates* exactly and looks each up in
    the precomputed distance transform of the background, so it does not inherit the
    staircasing a whole-raster rotation would. Reflection preserves area, so
    "reflected ⊆ shape (within tol)" already implies bilateral symmetry about the
    line — no need to also test the reverse direction.
    """
    xs, ys = fg_xy
    dx, dy = np.cos(theta), np.sin(theta)
    vx, vy = xs - cx, ys - cy
    t = vx * dx + vy * dy                       # projection onto the axis direction
    rx = cx + (2.0 * t * dx - vx)               # reflect: keep along-axis, flip ⟂
    ry = cy + (2.0 * t * dy - vy)
    h, w = dist.shape
    ri = np.clip(np.rint(ry).astype(int), 0, h - 1)
    ci = np.clip(np.rint(rx).astype(int), 0, w - 1)
    return float((dist[ri, ci] > tol_px).mean())


def detect_symmetry_rotation(silhouette: np.ndarray, *, tol: float = SYM_TOL) -> float | None:
    """Degrees of rotation that bring a *tilted* mirror axis to vertical, or None.

    For an off-axis bilaterally-symmetric mark the mirror line is one of the two PCA
    principal axes — but which one is not known a priori (the long axis is the
    mirror line for a beet, the short axis for a fat lens). Score *each* principal
    axis with a resampling-free reflection test (refining the angle locally), keep
    the better, and accept only if its off-shape fraction is within `tol`.

    The returned angle `rho` is applied as `ndi.rotate(arr, -rho)` to rectify and
    inverted in SVG as `rotate(-rho)` to wrap the emitted body back into place.
    """
    ys, xs = np.nonzero(silhouette)
    if xs.size < 3:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    pts = np.column_stack([xs - cx, ys - cy]).astype(float)
    _evals, evecs = np.linalg.eigh(pts.T @ pts / len(pts))
    dist = ndi.distance_transform_edt(~silhouette)
    fg_xy = (xs.astype(float), ys.astype(float))
    best: tuple[float, float] | None = None
    for i in range(2):
        base = float(np.arctan2(evecs[1, i], evecs[0, i]))
        # local refinement: PCA is coarse on noisy boundaries
        for dth in np.radians(np.arange(-5.0, 5.0 + 0.5, 0.5)):
            theta = base + dth
            mm = _axis_mismatch(fg_xy, cx, cy, theta, dist)
            if best is None or mm < best[0]:
                best = (mm, theta)
    if best is None or best[0] > tol:
        return None
    # `best[1]` is the mirror-axis angle φ; rotating the image by ndi.rotate(-rho)
    # with rho = -(φ + 90) brings that axis vertical (the 90° accounts for ndi's
    # CCW-in-array sign convention vs the atan2 line angle). The ±180° freedom in φ
    # only flips the rectified mark top-for-bottom — still vertically symmetric.
    return float(-(np.degrees(best[1]) + 90.0))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def classify_regions(
    regions: list[Region], axis: Axis, *, pair_iou: float = 1.0 - SYM_TOL,
    straddle_iou: float = STRADDLE_MIN_IOU,
    diagnostics: list | None = None,
) -> tuple[list[Region], list[tuple[Region, Region]], list[Region]]:
    """Split regions into self-symmetric straddlers, mirror pairs, and lone
    asymmetric leftovers.

    The three categories must be fit differently, so they are kept distinct: a
    straddler is fit half-and-mirrored about the axis, a pair is fit once and
    `<use>`-mirrored, and a loner is fit as-is with no symmetry (forcing the
    half-outline mirror onto a genuinely asymmetric region would distort it).

    The straddle gate uses a STRICTER bar (STRADDLE_MIN_IOU = 0.96) than the pair
    gate (1 - SYM_TOL = 0.90). Half-mirroring a single region erases its real
    asymmetry; false straddlers (icloud ~0.90, telegram ~0.917) fall below 0.96.
    The pair gate is looser because misclassifying a pair just shows the wrong
    region mirrored, which is less destructive than erasing asymmetry. Below either
    bar, regions fall through to loners (fit as-is). (Corpus check: genuine
    mirror-twin pairs score IoU >= 0.96; false pairs sit at <= 0.74.)"""
    straddlers: list[Region] = []
    pairs: list[tuple[Region, Region]] = []
    loners: list[Region] = []
    used: set[int] = set()
    for r in regions:
        if r.label in used:
            continue
        self_refl = _reflect_cols(r.mask, axis.x)
        self_iou = _iou(r.mask, self_refl)
        if self_iou >= straddle_iou:
            straddlers.append(r)
            used.add(r.label)
            if diagnostics is not None:
                diagnostics.append((r.label, round(float(self_iou), 4), "straddler"))
            continue
        # find a partner whose mask matches r's reflection
        partner = None
        for other in regions:
            if other.label in used or other.label == r.label:
                continue
            if other.color_hex == r.color_hex and _iou(self_refl, other.mask) >= pair_iou:
                partner = other
                break
        if partner is not None:
            # canonical = the one whose centroid is on the +x (right) side
            canon, mirror = (r, partner)
            if np.nonzero(r.mask)[1].mean() < axis.x:
                canon, mirror = partner, r
            pairs.append((canon, mirror))
            used.update({r.label, partner.label})
            if diagnostics is not None:
                diagnostics.append((r.label, round(float(self_iou), 4), "pair"))
                partner_refl = _reflect_cols(partner.mask, axis.x)
                diagnostics.append((partner.label, round(float(_iou(partner.mask, partner_refl)), 4), "pair"))
        else:
            loners.append(r)  # asymmetric and unpaired: fit as-is, no forced mirror
            used.add(r.label)
            if diagnostics is not None:
                diagnostics.append((r.label, round(float(self_iou), 4), "loner"))
    return straddlers, pairs, loners

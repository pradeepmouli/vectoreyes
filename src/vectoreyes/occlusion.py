# SPDX-License-Identifier: MIT
"""Occlusion reconstruction: explain overlapping regions as a z-ordered stack of
completed primitives (see docs/superpowers/specs/2026-06-04-occlusion-reconstruction-design.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
from skimage.draw import polygon2mask
from skimage.measure import CircleModel, EllipseModel
from skimage.morphology import convex_hull_image

from .contour import rdp, region_contours
from .fit import Shape, _fmt
from .types import Axis, Region


@dataclass
class ScenePrimitive:
    """A completed shape that may be partially occluded by higher-z primitives."""
    kind: str                 # "circle" | "ellipse" | "annulus" | "polygon"
    params: dict
    color_hex: str
    z: int
    source_label: int | None = None


def has_bite(mask: np.ndarray, *, max_solidity: float = 0.92) -> bool:
    """True when the region is non-convex enough to be a plausible occluded fragment
    (a crescent), i.e. its solidity (area / convex-hull area) is below the bar."""
    area = int(mask.sum())
    if area == 0:
        return False
    hull = int(convex_hull_image(mask).sum())
    return hull > 0 and (area / hull) < max_solidity


def label_boundary(
    region: Region, others: list[Region], *, reach: int = 2, contour_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (contour Nx2 as (x,y), seam_bool N) for the region's `contour_index`-th
    contour (0 = outer boundary, 1 = largest hole, ...). A contour point is a seam if
    any OTHER region's mask sits within `reach` px of it; else it is own boundary."""
    contours = region_contours(region.mask)
    if contour_index >= len(contours):
        return np.empty((0, 2)), np.empty((0,), bool)
    contour = contours[contour_index]
    if not others:
        return contour, np.zeros(len(contour), bool)
    # Union all other masks first, then dilate once: dilation distributes over union
    # (Minkowski sum identity), so (dil A | dil B) == dil(A | B) — identical bits,
    # one morphology pass instead of len(others).
    others_union = np.zeros_like(region.mask)
    for o in others:
        others_union |= o.mask
    near = binary_dilation(others_union, iterations=reach)
    h, w = region.mask.shape
    xs = np.clip(np.rint(contour[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(contour[:, 1]).astype(int), 0, h - 1)
    seam = near[ys, xs]
    return contour, seam


def region_adjacency(regions: list[Region]) -> dict[int, set[int]]:
    """label -> set of labels whose masks touch it (8-connectivity, 1px dilation)."""
    adj: dict[int, set[int]] = {r.label: set() for r in regions}
    dilated = {r.label: binary_dilation(r.mask) for r in regions}
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            if (dilated[a.label] & b.mask).any():
                adj[a.label].add(b.label)
                adj[b.label].add(a.label)
    return adj


def _own_arc_span_deg(own_pts: np.ndarray, cx: float, cy: float) -> float:
    """Angular span (deg) of the own points about (cx, cy). Full circle -> ~360."""
    ang = np.sort(np.arctan2(own_pts[:, 1] - cy, own_pts[:, 0] - cx))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return float(np.degrees(2 * np.pi - gaps.max()))     # span covered = full minus largest gap


def _fit_line(pts: np.ndarray) -> tuple[float, float, float, float] | None:
    """Total-least-squares line through `pts`. Returns (a, b, c, max_residual) for
    the line a*x + b*y = c with a**2 + b**2 == 1 (so a*x + b*y - c is signed
    perpendicular distance), or None for fewer than 2 points or all-identical points
    (degenerate, no defined direction)."""
    pts = np.asarray(pts, float)
    if len(pts) < 2 or np.allclose(pts, pts[0]):
        return None
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean)
    direction = vt[0]                                  # principal axis of the points
    a, b = float(-direction[1]), float(direction[0])   # unit normal
    c = a * mean[0] + b * mean[1]
    resid = float(np.abs(pts @ np.array([a, b]) - c).max())
    return a, b, c, resid


def _line_intersect(
    l1: tuple[float, float, float, float],
    l2: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Intersection of lines (a1,b1,c1) and (a2,b2,c2); None when |det| < _PARALLEL_TOL
    (1e-9), i.e. the unit normals are essentially exactly parallel."""
    a1, b1, c1 = l1[0], l1[1], l1[2]
    a2, b2, c2 = l2[0], l2[1], l2[2]
    det = a1 * b2 - a2 * b1
    if abs(det) < _PARALLEL_TOL:
        return None
    return ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)


def _is_convex(pts: list[tuple[float, float]]) -> bool:
    """True if the closed vertex ring turns consistently one way (all cross products
    share a sign). Rejects concave or self-intersecting rings."""
    n = len(pts)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cx, cy = pts[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return sign != 0


def _own_runs(contour: np.ndarray, seam: np.ndarray) -> list[np.ndarray]:
    """Maximal contiguous runs of own (non-seam) contour points, each as an open
    (M, 2) polyline. The contour is cyclic, so a run may wrap across index 0.
    Precondition: len(seam) == len(contour)."""
    n = len(seam)
    own = ~np.asarray(seam, bool)
    if n == 0 or not own.any():
        return []
    if own.all():
        return [np.asarray(contour, float)]
    # a run starts at an own point whose predecessor (cyclically) is a seam point
    starts = [i for i in range(n) if own[i] and not own[(i - 1) % n]]
    runs: list[np.ndarray] = []
    c = np.asarray(contour, float)
    for s in starts:
        idx = []
        i = s
        while own[i % n] and len(idx) < n:
            idx.append(i % n)
            i += 1
        runs.append(c[idx])
    return runs


def _fit_candidate_pts(own: np.ndarray) -> np.ndarray:
    """Return the convex-hull vertices of `own` when feasible, else `own` itself.
    Using the convex hull isolates the outer perimeter, discarding inner concave arcs
    that arise when seam detection cannot mark the full interior boundary."""
    from scipy.spatial import ConvexHull, QhullError
    if len(own) < 4:
        return own
    try:
        hull = ConvexHull(own)
        return own[hull.vertices]
    except QhullError:
        return own


def _fit_circle(
    contour: np.ndarray, seam: np.ndarray, *, max_residual: float, min_arc_deg: float
) -> dict | None:
    """Fit a circle to the own-boundary points (convex hull, to drop concave inner
    arcs). Returns {"cx","cy","r"} or None if too few points, residual too large, or
    the own arc spans less than `min_arc_deg`."""
    own = np.asarray(contour, float)[~seam]
    if len(own) < 8:
        return None
    fit_pts = _fit_candidate_pts(own)
    if len(fit_pts) < 8:
        return None
    cm = CircleModel.from_estimate(fit_pts)
    if not cm or np.abs(cm.residuals(fit_pts)).max() > max_residual:
        return None
    cx, cy = float(cm.center[0]), float(cm.center[1])
    if _own_arc_span_deg(fit_pts, cx, cy) < min_arc_deg:
        return None
    return {"cx": cx, "cy": cy, "r": float(cm.radius)}


def complete_primitive(
    contour: np.ndarray, seam: np.ndarray, *, max_residual: float, min_arc_deg: float
) -> dict | None:
    """Fit a circle (then ellipse) to the OWN-boundary points, completing across the
    seam. Returns {"kind","params"} or None when the own arc can't constrain a fit.

    Fitting is performed on the convex hull of the own points so that inner concave
    arcs (which occur when the occluder's overlap zone is not passed as a neighbor
    region and thus not captured by seam detection) do not bias the fit."""
    own = np.asarray(contour, float)[~seam]
    if len(own) < 8:
        return None
    fit_pts = _fit_candidate_pts(own)
    if len(fit_pts) < 8:
        return None
    circ = _fit_circle(contour, seam, max_residual=max_residual, min_arc_deg=min_arc_deg)
    if circ is not None:
        cx, cy = circ["cx"], circ["cy"]
        seam_pts = np.asarray(contour, float)[seam]
        inside = len(seam_pts) == 0 or np.all(
            (seam_pts[:, 0] - cx) ** 2 + (seam_pts[:, 1] - cy) ** 2 <= (circ["r"] + max_residual) ** 2
        )
        if inside:
            return {"kind": "circle", "params": {"cx": cx, "cy": cy, "r": circ["r"]}}
    em = EllipseModel.from_estimate(fit_pts)
    if em and np.abs(em.residuals(fit_pts)).max() <= max_residual:
        xc, yc = float(em.center[0]), float(em.center[1])
        a, b = (float(v) for v in em.axis_lengths)
        if abs(em.theta) < 0.08 or abs(abs(em.theta) - np.pi) < 0.08:
            if _own_arc_span_deg(fit_pts, xc, yc) >= min_arc_deg:
                return {"kind": "ellipse", "params": {"cx": xc, "cy": yc, "rx": a, "ry": b}}
    return None


def complete_annulus(
    region: Region, others: list[Region], *, max_residual: float, min_arc_deg: float,
    concentric_tol: float,
    outer_boundary: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict | None:
    """Fit an annulus (two concentric circles) from a ring fragment's outer and inner
    own arcs. Returns {"kind":"annulus","params":{cx,cy,r_outer,r_inner}} or None when
    the region has no hole, either circle can't be fit, the radii don't nest, or the
    centres aren't concentric within `concentric_tol`.

    `outer_boundary` optionally supplies an already-computed (contour, seam) for
    contour_index=0 so it isn't recomputed when the caller already holds it."""
    if len(region_contours(region.mask)) < 2:
        return None                                       # no hole -> not a ring
    outer_c, outer_seam = (outer_boundary if outer_boundary is not None
                           else label_boundary(region, others, contour_index=0))
    inner_c, inner_seam = label_boundary(region, others, contour_index=1)
    outer = _fit_circle(outer_c, outer_seam, max_residual=max_residual, min_arc_deg=min_arc_deg)
    inner = _fit_circle(inner_c, inner_seam, max_residual=max_residual, min_arc_deg=min_arc_deg)
    if outer is None or inner is None or inner["r"] >= outer["r"]:
        return None
    if np.hypot(outer["cx"] - inner["cx"], outer["cy"] - inner["cy"]) > concentric_tol:
        return None
    cx = (outer["cx"] + inner["cx"]) / 2
    cy = (outer["cy"] + inner["cy"]) / 2
    return {"kind": "annulus",
            "params": {"cx": cx, "cy": cy, "r_outer": outer["r"], "r_inner": inner["r"]}}


def _cavity_contained_labels(region: Region, others: list[Region]) -> set[int]:
    """Labels of neighbors lying wholly inside an enclosed cavity of ``region``.

    Such neighbors are retained artwork, not occluders: they must not turn the
    cavity boundary into a seam, and reconstruction must leave their regions intact.
    A neighbor crossing a cavity boundary has pixels outside ``cavities`` and is
    therefore still treated as a real occluder.
    """
    cavities = binary_fill_holes(region.mask) & ~region.mask
    if not cavities.any():
        return set()
    return {
        other.label
        for other in others
        if other.mask.any() and not np.any(other.mask & ~cavities)
    }


def _retained_region_shape(region: Region) -> Shape:
    """Carry retained raster artwork through reconstruction as its source contour."""
    return Shape("path", {
        "d": retained_region_path_data(region.mask),
        "color_hex": region.color_hex,
        "z": -1,
        "source_label": region.label,
    })


def retained_region_path_data(mask: np.ndarray) -> str:
    """Return the existing source-contour encoding without a scene wrapper."""
    parts: list[str] = []
    for contour in region_contours(mask):
        if len(contour) < 3:
            continue
        parts.append(f"M{_fmt(contour[0, 0])} {_fmt(contour[0, 1])}")
        parts.extend(f"L{_fmt(x)} {_fmt(y)}" for x, y in contour[1:])
        parts.append("Z")
    return " ".join(parts)


def _subseq_indices(pts: np.ndarray, sub: np.ndarray) -> list[int]:
    """Indices in `pts` of the subsequence `sub`. `rdp` preserves its input points
    exactly and in order, so a monotone forward cursor with exact equality maps each
    `sub` point back to `pts` — robust to repeated/near-duplicate points where a
    nearest-point (argmin) search could pick a wrong or out-of-order index."""
    idxs: list[int] = []
    j = 0
    for v in sub:
        while j < len(pts) and not (pts[j, 0] == v[0] and pts[j, 1] == v[1]):
            j += 1
        if j >= len(pts):
            break
        idxs.append(j)
        j += 1
    return idxs


def _collinear(l1: tuple, l2: tuple) -> bool:
    """True if two normalized-normal lines (a,b,c) describe the same line — parallel
    normals and matching offset (the SVD normal sign is arbitrary, so align it)."""
    a1, b1, c1 = l1[0], l1[1], l1[2]
    a2, b2, c2 = l2[0], l2[1], l2[2]
    if abs(a1 * b2 - a2 * b1) > _COLLINEAR_SIN_TOL:        # normals not parallel
        return False
    s = 1.0 if (a1 * a2 + b1 * b2) >= 0 else -1.0          # align normal directions
    return abs(c1 - s * c2) <= _COLLINEAR_OFFSET_TOL


def _coalesce_collinear(lines: list[tuple]) -> list[tuple]:
    """Drop consecutive (and wraparound) collinear duplicates from a cyclic line list,
    so one polygon edge split by a mid-edge bite into two collinear fits becomes a
    single supporting line (else their intersection is parallel -> None)."""
    if len(lines) < 2:
        return lines
    merged = [lines[0]]
    for ln in lines[1:]:
        if not _collinear(merged[-1], ln):
            merged.append(ln)
    while len(merged) > 2 and _collinear(merged[-1], merged[0]):
        merged.pop()
    return merged


def complete_polygon(
    region: Region, others: list[Region], *, max_residual: float, max_vertices: int,
    boundary: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict | None:
    """Recover a convex polygon from a partially-occluded fragment. Fit a line to every
    visible (own-boundary) edge across ALL own runs, kept in boundary (cyclic) order,
    then intersect consecutive lines cyclically — each intersection spanning a seam gap
    recovers a corner hidden behind an occluder (so a polygon bitten by several separate
    occluders is still recovered, as long as every edge stays partly visible). Returns
    {"kind":"polygon","params":{"points":[(x,y),...]}} or None when the fragment is
    curved, under-constrained, or non-convex.

    `boundary` optionally supplies an already-computed (contour, seam) from
    `label_boundary` so it isn't recomputed."""
    contour, seam = boundary if boundary is not None else label_boundary(region, others)
    if len(contour) == 0 or not seam.any():
        return None                                    # no occlusion -> not this completer
    runs = _own_runs(contour, seam)
    if not runs:
        return None
    lines: list[tuple[float, float, float, float]] = []
    for run in runs:                                   # runs are already in cyclic order
        if len(run) < 2:
            continue
        idxs = _subseq_indices(run, rdp(run, max_residual))   # edge breakpoints in this run
        for i in range(len(idxs) - 1):
            ln = _fit_line(run[idxs[i]: idxs[i + 1] + 1])
            if ln is None or ln[3] > max_residual:
                return None
            lines.append(ln)
    lines = _coalesce_collinear(lines)                 # merge one edge split by a mid-edge bite
    if len(lines) < 3 or len(lines) > max_vertices:
        return None
    verts: list[tuple[float, float]] = []
    for i in range(len(lines)):
        p = _line_intersect(lines[i], lines[(i + 1) % len(lines)])
        if p is None:
            return None
        verts.append(p)
    if not _is_convex(verts):
        return None
    return {"kind": "polygon", "params": {"points": verts}}


def _complete_member(region: Region, others: list[Region]) -> dict | None:
    """Complete a group member: annulus if it has a hole, else circle/ellipse, else a
    convex polygon. The curved fitters run first (they reject straight edges), so only
    genuinely polygonal fragments reach complete_polygon.

    The outer boundary (contour_index=0) is computed once and forwarded to both
    complete_annulus and complete_polygon so label_boundary isn't called twice."""
    retained_labels = _cavity_contained_labels(region, others)
    occluders = [other for other in others if other.label not in retained_labels]
    contour, seam = label_boundary(region, occluders)        # outer boundary (contour_index=0)
    ann = complete_annulus(region, occluders, max_residual=_MAX_RESIDUAL,
                           min_arc_deg=_MIN_ARC_DEG, concentric_tol=_CONCENTRIC_TOL,
                           outer_boundary=(contour, seam))
    if ann is not None:
        return ann
    prim = complete_primitive(contour, seam, max_residual=_MAX_RESIDUAL, min_arc_deg=_MIN_ARC_DEG)
    if prim is not None:
        return prim
    return complete_polygon(region, others, max_residual=_MAX_RESIDUAL,
                            max_vertices=_MAX_VERTICES, boundary=(contour, seam))


def intersection_lens_d(a: dict, b: dict) -> str | None:
    """SVG path (two A arcs) for the lens = intersection of circles a and b.
    Returns None if the circles don't properly overlap (disjoint or one contains
    the other). Params dicts use cx, cy, r."""
    ax, ay, ar = a["cx"], a["cy"], a["r"]
    bx, by, br = b["cx"], b["cy"], b["r"]
    dx, dy = bx - ax, by - ay
    dist = float(np.hypot(dx, dy))
    if dist <= abs(ar - br) or dist >= ar + br or dist == 0:
        return None                                       # nested or disjoint
    t = (dist * dist + ar * ar - br * br) / (2 * dist)
    h2 = ar * ar - t * t
    if h2 <= 0:
        return None
    hh = float(np.sqrt(h2))
    ux, uy = dx / dist, dy / dist                          # unit a->b
    mx, my = ax + t * ux, ay + t * uy                      # chord midpoint
    p1 = (mx - hh * (-uy), my - hh * ux)                   # one crossing
    p2 = (mx + hh * (-uy), my + hh * ux)                   # the other
    f = _fmt
    return (
        f"M{f(p1[0])} {f(p1[1])} "
        f"A{f(ar)} {f(ar)} 0 0 1 {f(p2[0])} {f(p2[1])} "
        f"A{f(br)} {f(br)} 0 0 1 {f(p1[0])} {f(p1[1])} Z"
    )


def primitive_mask(prim: dict, h: int, w: int) -> np.ndarray:
    """Boolean mask of a completed primitive (circle, annulus, polygon, or ellipse) on an (h, w) grid."""
    yy, xx = np.ogrid[:h, :w]
    p = prim["params"]
    if prim["kind"] == "circle":
        return (xx - p["cx"]) ** 2 + (yy - p["cy"]) ** 2 <= p["r"] ** 2
    if prim["kind"] == "annulus":
        d2 = (xx - p["cx"]) ** 2 + (yy - p["cy"]) ** 2
        return (d2 <= p["r_outer"] ** 2) & (d2 >= p["r_inner"] ** 2)
    if prim["kind"] == "polygon":
        pts = prim["params"]["points"]
        rc = np.array([(y, x) for x, y in pts], dtype=float)   # skimage wants (row, col)
        return polygon2mask((h, w), rc)
    return ((xx - p["cx"]) / p["rx"]) ** 2 + ((yy - p["cy"]) / p["ry"]) ** 2 <= 1.0


def stack_agreement(
    prims, lens, regions: list[Region], h: int, w: int,
    *, retained_regions: list[Region] | None = None,
) -> float:
    """Paint prims (then the lens) in z-order into a colour-label image and compare,
    over the union of the regions and any pixels painted by the stack, to each
    region's own colour. Any candidate paint over exact retained-region pixels is a
    hard failure before the general comparison's boundary tolerance. Returns [0, 1]."""
    BG = "\x00"
    retained_regions = retained_regions or []
    retained_tokens = {r.label: f"\x01retained:{r.label}" for r in retained_regions}
    painted = np.full((h, w), BG, dtype=object)
    painted_any = np.zeros((h, w), bool)
    candidate_painted = np.zeros((h, w), bool)
    # Retained artwork remains outside the reconstructed branch. Paint it beneath
    # the candidate for a conservative gate: a candidate that intrudes into its
    # exact source pixels must disagree, even when the two regions share a color.
    for r in retained_regions:
        painted[r.mask] = retained_tokens[r.label]
        painted_any |= r.mask
    for prim in sorted(prims, key=lambda p: p["z"]):
        m = primitive_mask(prim, h, w)
        painted[m] = prim["color"]
        painted_any |= m
        candidate_painted |= m
    if lens is not None:
        a, b = (prims[lens["lens_of"][0]], prims[lens["lens_of"][1]])
        lm = primitive_mask(a, h, w) & primitive_mask(b, h, w)
        painted[lm] = lens["mask_color"]
        painted_any |= lm
        candidate_painted |= lm
    if any(np.any(candidate_painted & r.mask) for r in retained_regions):
        return 0.0
    truth = np.full((h, w), BG, dtype=object)
    region_union = np.zeros((h, w), bool)
    for r in regions:
        truth[r.mask] = retained_tokens.get(r.label, r.color_hex)
        region_union |= r.mask
    compare_mask = region_union | painted_any
    if not compare_mask.any():
        return 0.0
    # Tolerate a 1px boundary shell: sub-pixel fit error shows up as a thin ring of
    # disagreement (worst on thin annulus bands, where it can reach several percent).
    # Erode the disagreement so only thick — genuinely wrong — mismatches count; a
    # wrong reconstruction (filled hole, wrong radius) still scores far below the bar.
    disagree = compare_mask & (painted != truth)
    core = binary_erosion(disagree, iterations=1)
    return 1.0 - float(core.sum()) / float(compare_mask.sum())


_MAX_RESIDUAL = 1.6
_MAX_VERTICES = 8
_MIN_ARC_DEG = 110.0
# With the 1px-boundary-tolerant gate, correct reconstructions score ~0.97–1.0 and a
# wrong one (filled hole, wrong radius) ~0.7, so 0.96 separates them with wide margin.
_GATE_AGREEMENT = 0.96
_CONCENTRIC_TOL = 2.0
_MIN_OVERLAP_PX = 12
_OVERLAP_OWNERSHIP = 0.5
_PARALLEL_TOL = 1e-9  # |det| of two unit normals == sin(angle); ~1e-7 deg, i.e. exact-parallel only
# Two fits of the SAME polygon edge (split by a mid-edge bite) agree to within these:
# normals near-parallel (sin ~< 4.6 deg, well under any real convex exterior angle) and
# offsets within a couple px. Genuinely distinct edges differ by far more on one or both.
_COLLINEAR_SIN_TOL = 0.08
_COLLINEAR_OFFSET_TOL = 2.0


def _pair_constraint(
    prim_i: dict, region_i: Region, prim_j: dict, region_j: Region, h: int, w: int
) -> str | None:
    """Over/under for an overlapping pair, from which region owns the overlap zone in
    the raster: the colour that survives where the two completed shapes overlap is the
    one on top. Returns "i_over_j", "j_over_i", or None (no real overlap, or a
    distinct-coloured intersection owned by neither — e.g. a Mastercard lens)."""
    overlap = primitive_mask(prim_i, h, w) & primitive_mask(prim_j, h, w)
    n = int(overlap.sum())
    if n < _MIN_OVERLAP_PX:
        return None
    in_i = int((overlap & region_i.mask).sum())
    in_j = int((overlap & region_j.mask).sum())
    if max(in_i, in_j) < _OVERLAP_OWNERSHIP * n or in_i == in_j:
        return None
    return "i_over_j" if in_i > in_j else "j_over_i"


def _topo_order(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    """Kahn topological sort. `edges` are (under, over): under is painted first.
    Returns a paint order (z = position), or None if the constraints are cyclic."""
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    queue = [i for i in range(n) if indeg[i] == 0]
    order: list[int] = []
    while queue:
        node = queue.pop()
        order.append(node)
        for m in adj[node]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return order if len(order) == n else None


def _snap_pair(p_left: dict, p_right: dict, axis_x: float) -> tuple[dict, dict]:
    """Force two completed circles to be exact mirror images about x = axis_x."""
    r = (p_left["params"]["r"] + p_right["params"]["r"]) / 2
    cy = (p_left["params"]["cy"] + p_right["params"]["cy"]) / 2
    off = (abs(p_left["params"]["cx"] - axis_x) + abs(p_right["params"]["cx"] - axis_x)) / 2
    left = {"kind": "circle", "params": {"cx": axis_x - off, "cy": cy, "r": r}}
    right = {"kind": "circle", "params": {"cx": axis_x + off, "cy": cy, "r": r}}
    return left, right


def reconstruct_scene(
    regions: list[Region], axis: Axis | None, shape_hw: tuple[int, int]
) -> tuple[list, list[Region]]:
    """Return (reconstructed, remaining). `reconstructed` mixes ScenePrimitive and
    lens Shape objects in paint order; `remaining` are regions to fit the old way.
    Only adjacent groups that complete, admit a global paint order, and pass the
    consistency gate are reconstructed."""
    h, w = shape_hw
    by_label = {r.label: r for r in regions}
    adj = region_adjacency(regions)
    reconstructed: list = []
    consumed: set[int] = set()

    for r in regions:
        if r.label in consumed or not has_bite(r.mask):
            continue
        group: set[int] = set()
        stack = [r.label]
        while stack:
            lab = stack.pop()
            if lab in group:
                continue
            group.add(lab)
            stack.extend(adj[lab] - group)
        group_regions = [by_label[l] for l in sorted(group) if l not in consumed]
        if len(group_regions) < 2:
            continue

        retained_labels: set[int] = set()
        for gr in group_regions:
            retained_labels.update(_cavity_contained_labels(
                gr, [other for other in group_regions if other.label != gr.label]
            ))

        completed: list[tuple[Region, dict]] = []
        for gr in group_regions:
            if gr.label in retained_labels:
                continue
            others = [o for o in group_regions if o.label != gr.label]
            prim = _complete_member(gr, others)
            if prim is not None:
                completed.append((gr, prim))
        if len(completed) < 2:
            continue

        reg_list = [cr for cr, _ in completed]
        prim_list = [p for _, p in completed]
        completed_labels = {cr.label for cr in reg_list}
        retained_regions = [gr for gr in group_regions if gr.label in retained_labels]
        leftover = [
            gr for gr in group_regions
            if gr.label not in completed_labels and gr.label not in retained_labels
        ]

        # global paint order from pairwise overlap ownership
        edges: list[tuple[int, int]] = []
        for i in range(len(completed)):
            for j in range(i + 1, len(completed)):
                cons = _pair_constraint(prim_list[i], reg_list[i], prim_list[j], reg_list[j], h, w)
                if cons == "i_over_j":
                    edges.append((j, i))          # under=j, over=i
                elif cons == "j_over_i":
                    edges.append((i, j))
        order = _topo_order(len(completed), edges)
        if order is None:
            continue                              # cyclic -> weave -> decline
        z_of = {idx: k for k, idx in enumerate(order)}

        prims = [
            {"kind": prim_list[i]["kind"], "params": prim_list[i]["params"],
             "color": reg_list[i].color_hex, "z": z_of[i], "source_label": reg_list[i].label}
            for i in range(len(completed))
        ]

        # Mastercard: exactly two circles + one distinct-coloured lens -> snap + lens.
        # Snap first (it moves the circles), then derive the lens path; only treat the
        # leftover as a lens if that path is real, so we never consume a region we can't
        # emit (degenerate / non-overlapping circles fall through and the gate decides).
        lens = None
        lens_region = None
        lens_d = None
        if (len(completed) == 2 and prim_list[0]["kind"] == prim_list[1]["kind"] == "circle"
                and len(leftover) == 1):
            if axis is not None:
                lo, hi = (0, 1) if prim_list[0]["params"]["cx"] <= prim_list[1]["params"]["cx"] else (1, 0)
                sl, sr = _snap_pair(prim_list[lo], prim_list[hi], axis.x)
                prims[lo]["params"] = sl["params"]
                prims[hi]["params"] = sr["params"]
            lens_d = intersection_lens_d(prims[0]["params"], prims[1]["params"])
            if lens_d is not None:
                lens_region = leftover[0]
                lens = {"mask_color": lens_region.color_hex, "lens_of": (0, 1)}

        if stack_agreement(
            prims, lens, group_regions, h, w, retained_regions=retained_regions
        ) < _GATE_AGREEMENT:
            continue

        reconstructed.extend(_retained_region_shape(region) for region in retained_regions)
        for p in sorted(prims, key=lambda p: p["z"]):
            reconstructed.append(ScenePrimitive(
                p["kind"],
                p["params"],
                p["color"],
                p["z"],
                p["source_label"],
            ))
        if lens is not None:
            reconstructed.append(Shape("path", {
                "d": lens_d,
                "color_hex": lens["mask_color"],
                "z": len(prims),
                "source_label": lens_region.label if lens_region is not None else None,
                "lens_of_labels": (
                    reg_list[lens["lens_of"][0]].label,
                    reg_list[lens["lens_of"][1]].label,
                ),
            }))
        consumed.update(completed_labels)
        consumed.update(retained_labels)
        if lens_region is not None:
            consumed.add(lens_region.label)

    remaining = [r for r in regions if r.label not in consumed]
    return reconstructed, remaining

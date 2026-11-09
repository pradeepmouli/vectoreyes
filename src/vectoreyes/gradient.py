# SPDX-License-Identifier: MIT
"""Gradient detection: recover a gradient's footprint from quantized bands, fit a
linear/radial colour model against the original image, and emit one gradient-filled
shape when it re-renders faithfully (see
docs/superpowers/specs/2026-06-06-gradient-handling-design.md)."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .color import srgb_to_oklab

_MIN_BANDS = 3
_GATE_DELTA_E = 0.05      # mean OKLab ΔE bar: a fitted model must re-render within this to be accepted
_MIN_STOP_SPAN = 0.02    # OKLab end-to-end travel a fit must show to count as a gradient;
                         # rejects flat regions whose antialiasing noise fits a near-constant
                         # gradient (genuine gradients span >=0.039, flat-with-AA spans ~0).
_BLOB_DOMINANCE = 0.85   # smooth-gradient path: min fraction of the foreground that must lie
                         # in a single connected component for the mark to be treated as one
                         # gradient blob (rejects multi-glyph wordmarks before any fit).
_STRETCH_GRID_STEPS = (8, 16, 24, 32, 48)   # NxN downsample sizes for the stretch-fill;
                                            # the renderer bilinearly stretches the grid
                                            # back over the footprint. Last entry is the cap.
_STRETCH_TARGET = 0.05                       # grow the grid until mean per-pixel ΔE <= this.


def _hex_to_oklab(hex_colors: list[str]) -> np.ndarray:
    rgb = np.array(
        [[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in hex_colors],
        float,
    )
    return srgb_to_oklab(rgb / 255.0)


def _principal_axis(vectors: np.ndarray, *, eps: float) -> np.ndarray | None:
    """Mean-centre `vectors` and return the unit principal direction (top SVD right-
    singular vector), or None if the spread is below `eps` (degenerate/flat)."""
    centred = vectors - vectors.mean(axis=0)
    if np.abs(centred).max() < eps:
        return None
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0]


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(round(v)) for v in np.clip(rgb, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def _fit_stops(t: np.ndarray, rgb: np.ndarray, k: int = 7) -> list[tuple[float, str]]:
    """Sample k evenly-spaced stops; each stop colour = median original RGB of pixels
    in its t-neighbourhood. `t` in [0,1], `rgb` is (N,3) uint8-ish."""
    edges = np.linspace(0.0, 1.0, k)
    half = 0.5 / (k - 1)
    stops: list[tuple[float, str]] = []
    for e in edges:
        sel = np.abs(t - e) <= half
        if not sel.any():
            sel = np.argsort(np.abs(t - e))[:32]       # nearest fallback
        stops.append((float(e), _rgb_to_hex(np.median(rgb[sel], axis=0))))
    return stops


def _stop_colors_oklab(stops: list[tuple[float, str]]) -> np.ndarray:
    return _hex_to_oklab([c for _, c in stops])


def _stop_span(stops: list[tuple[float, str]]) -> float:
    """Max OKLab distance of any stop from the first stop — the gradient's largest colour
    travel. Uses max-from-first rather than first-vs-last so a cyclic field (e.g. a halo
    that returns to its starting colour) still registers travel; ~0 for a flat region fit
    as a near-constant gradient."""
    cols = _stop_colors_oklab(stops)
    return float(np.linalg.norm(cols - cols[0], axis=1).max())


def _reduce_stops(stops: list[tuple[float, str]], *, max_delta_e: float) -> list[tuple[float, str]]:
    """Greedily drop interior stops whose removal keeps the piecewise-linear (in OKLab)
    reconstruction within max_delta_e of the full stop set."""
    kept = list(stops)
    changed = True
    while changed and len(kept) > 2:
        changed = False
        ref_off = np.array([o for o, _ in kept])
        ref_cols = _stop_colors_oklab(kept)
        for i in range(1, len(kept) - 1):
            trial = kept[:i] + kept[i + 1:]
            offs = np.array([o for o, _ in trial])
            cols = _stop_colors_oklab(trial)
            approx = np.column_stack([np.interp(ref_off, offs, cols[:, ch]) for ch in range(3)])
            if np.linalg.norm(approx - ref_cols, axis=1).max() <= max_delta_e:
                kept = trial
                changed = True
                break
    return kept


def _radial_spread(pts: np.ndarray, oklab: np.ndarray, c: np.ndarray, nbins: int = 16) -> float:
    """Mean within-bin colour variance when binning pixels by distance from c. Lower =
    more concentric (a better radial centre)."""
    r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    rmax = r.max()
    if rmax < 1e-6:
        return np.inf
    bins = np.clip((r / rmax * nbins).astype(int), 0, nbins - 1)
    total, count = 0.0, 0
    for b in range(nbins):
        sel = bins == b
        if sel.sum() >= 2:
            total += float(np.var(oklab[sel], axis=0).sum())
            count += 1
    return total / count if count else np.inf


def _radial_model_from_center(c: np.ndarray, pts: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Build a radial gradient model centred at `c` (xy) over `pts`/`rgb`."""
    r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    rmax = float(r.max())
    if rmax < 1e-6:
        return None
    stops = _reduce_stops(_fit_stops(r / rmax, rgb), max_delta_e=_GATE_DELTA_E)
    return {"kind": "radial",
            "geometry": {"cx": float(c[0]), "cy": float(c[1]), "r": rmax},
            "stops": stops}


def _linear_model_from_axis(u: np.ndarray, pts: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Build a linear gradient model along unit axis `u` over `pts`/`rgb`."""
    proj = pts @ u
    t0, t1 = float(proj.min()), float(proj.max())
    if t1 - t0 < 1e-6:
        return None
    tn = (proj - t0) / (t1 - t0)
    mean = pts.mean(axis=0)
    mt = float(mean @ u)
    p1 = mean + (t0 - mt) * u
    p2 = mean + (t1 - mt) * u
    stops = _reduce_stops(_fit_stops(tn, rgb), max_delta_e=_GATE_DELTA_E)
    return {"kind": "linear",
            "geometry": {"x1": float(p1[0]), "y1": float(p1[1]),
                         "x2": float(p2[0]), "y2": float(p2[1])},
            "stops": stops}


def _fit_radial(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Fit a radial gradient: estimate the centre as the centroid of the extreme along
    the principal colour axis (try both ends; keep the more concentric), then fit stops
    vs normalized radius. The centroid-of-extreme heuristic can yield a slightly-off
    centre for clipped or asymmetric footprints; the Task 6 consistency gate is the
    backstop that rejects fits that don't re-render faithfully."""
    axis = _principal_axis(oklab, eps=1e-8)
    if axis is None:
        return None
    s = (oklab - oklab.mean(axis=0)) @ axis             # 1-D colour coordinate
    order = np.argsort(s)
    k = max(3, len(s) // 10)
    best_c, best_spread = None, np.inf
    for idx in (order[-k:], order[:k]):                 # inner-extreme and outer-extreme clusters
        c = pts[idx].mean(axis=0)
        spread = _radial_spread(pts, oklab, c)
        if spread < best_spread:
            best_spread, best_c = spread, c
    if best_c is None:
        return None
    c = best_c
    return _radial_model_from_center(c, pts, rgb)


def _fit_linear(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Fit a linear gradient. pts:(N,2) xy, oklab:(N,3), rgb:(N,3). Returns a model or
    None if the points don't span an axis."""
    A = np.column_stack([pts, np.ones(len(pts))])
    coef, *_ = np.linalg.lstsq(A, oklab, rcond=None)   # rows: [p, q, r] per channel column
    G = coef[:2, :].T                                   # (3 channels, 2) per-channel gradient
    if np.abs(G).max() < 1e-8:
        return None
    _, _, vt = np.linalg.svd(G, full_matrices=False)
    u = vt[0]                                           # unit axis direction
    return _linear_model_from_axis(u, pts, rgb)


def _model_mean_de_on_points(model: dict, pts: np.ndarray, rgb: np.ndarray, *,
                             truth_oklab: np.ndarray | None = None) -> float:
    """Mean OKLab ΔE of a model's rendered colour vs `rgb` at `pts` (footprint pixels).

    `truth_oklab` optionally supplies the pre-computed `srgb_to_oklab(rgb / 255.0)` so
    the caller can hoist the invariant transform out of a search loop (bit-identical: same
    input array, same function). When absent the transform is computed here as before."""
    rendered = _interp_stops_rgb(_model_t(model, pts), model["stops"])
    truth = truth_oklab if truth_oklab is not None else srgb_to_oklab(rgb / 255.0)
    return float(np.linalg.norm(srgb_to_oklab(rendered / 255.0) - truth, axis=1).mean())


def _search_centers(centers, pts: np.ndarray, rgb: np.ndarray, *,
                    truth_oklab: np.ndarray | None = None):
    """Lowest-mean-ΔE radial model over candidate centres. Returns (de, model, cx, cy) or None."""
    best = None
    for cx, cy in centers:
        model = _radial_model_from_center(np.array([cx, cy]), pts, rgb)
        if model is None:
            continue
        de = _model_mean_de_on_points(model, pts, rgb, truth_oklab=truth_oklab)
        if best is None or de < best[0]:
            best = (de, model, cx, cy)
    return best


def _fit_radial_searched(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Radial fit via a deterministic centre search (coarse grid over the bbox extended
    +/-50%, then a finer local grid around the best). Beats the principal-axis-extreme
    heuristic for corner-anchored / clipped fields. `oklab` is the pre-computed truth OKLab
    threaded in from `_best_parametric` to avoid recomputing it on every model eval."""
    xs, ys = pts[:, 0], pts[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    gx = np.linspace(x0 - (x1 - x0) * 0.5, x1 + (x1 - x0) * 0.5, 13)
    gy = np.linspace(y0 - (y1 - y0) * 0.5, y1 + (y1 - y0) * 0.5, 13)
    coarse = _search_centers([(cx, cy) for cx in gx for cy in gy], pts, rgb, truth_oklab=oklab)
    if coarse is None:
        return None
    _, _, bcx, bcy = coarse
    sx, sy = gx[1] - gx[0], gy[1] - gy[0]               # one coarse step
    rx = np.linspace(bcx - sx, bcx + sx, 7)
    ry = np.linspace(bcy - sy, bcy + sy, 7)
    fine = _search_centers([(cx, cy) for cx in rx for cy in ry], pts, rgb, truth_oklab=oklab)
    return (fine or coarse)[1]


def _fit_linear_searched(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Linear fit via an axis-angle search (5° coarse steps, then a finer local sweep).
    `oklab` is the pre-computed truth OKLab from `_best_parametric`, threaded through to
    avoid recomputing it on every model eval inside the angle search."""
    def at(ang):
        m = _linear_model_from_axis(np.array([np.cos(ang), np.sin(ang)]), pts, rgb)
        return None if m is None else (_model_mean_de_on_points(m, pts, rgb, truth_oklab=oklab), m, ang)
    best = None
    for ang in np.linspace(0.0, np.pi, 36, endpoint=False):
        r = at(ang)
        if r is not None and (best is None or r[0] < best[0]):
            best = r
    if best is None:
        return None
    step = np.pi / 36
    for ang in np.linspace(best[2] - step, best[2] + step, 7):
        r = at(ang)
        if r is not None and r[0] < best[0]:
            best = r
    return best[1]


def _best_parametric(mask: np.ndarray, rgb_image: np.ndarray) -> tuple[dict, float, float] | None:
    """Best searched parametric (linear or radial) model for the footprint, with its
    mean and median per-pixel ΔE. No acceptance cut (the ladder applies the gates).
    None if neither model travels at least _MIN_STOP_SPAN."""
    ys, xs = np.where(mask)
    if len(xs) < 3 * _MIN_BANDS:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    rgb = rgb_image[ys, xs].astype(float)
    oklab = srgb_to_oklab(rgb / 255.0)
    best = None
    for fit in (_fit_linear_searched, _fit_radial_searched):
        model = fit(pts, oklab, rgb)
        if model is None or _stop_span(model["stops"]) < _MIN_STOP_SPAN:
            continue
        pde = _per_pixel_delta_e(model, ys, xs, rgb_image, truth_oklab=oklab)
        mean_de, median_de = float(pde.mean()), float(np.median(pde))
        if best is None or mean_de < best[1]:
            best = (model, mean_de, median_de)
    return best


def _interp_stops_rgb(t: np.ndarray, stops: list[tuple[float, str]]) -> np.ndarray:
    offs = np.array([o for o, _ in stops])
    cols = np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for _, c in stops], float)
    return np.column_stack([np.interp(t, offs, cols[:, ch]) for ch in range(3)])


def _model_t(model: dict, pts: np.ndarray) -> np.ndarray:
    g = model["geometry"]
    if model["kind"] == "linear":
        d = np.array([g["x2"] - g["x1"], g["y2"] - g["y1"]], float)
        L = float(d @ d) or 1.0
        return (((pts[:, 0] - g["x1"]) * d[0] + (pts[:, 1] - g["y1"]) * d[1]) / L).clip(0, 1)
    r = np.hypot(pts[:, 0] - g["cx"], pts[:, 1] - g["cy"]) / (g["r"] or 1.0)
    return r.clip(0, 1)


def _per_pixel_delta_e(model: dict, ys: np.ndarray, xs: np.ndarray, rgb_image: np.ndarray, *,
                       truth_oklab: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel OKLab ΔE between the model's rendered colour and the actual pixel at (ys, xs).

    `truth_oklab` optionally supplies the pre-computed `srgb_to_oklab(pixels / 255.0)` so
    the caller can avoid recomputing the invariant transform (bit-identical)."""
    pts = np.column_stack([xs, ys]).astype(float)
    rendered = _interp_stops_rgb(_model_t(model, pts), model["stops"])
    truth = truth_oklab if truth_oklab is not None else srgb_to_oklab(rgb_image[ys, xs].astype(float) / 255.0)
    return np.linalg.norm(srgb_to_oklab(rendered / 255.0) - truth, axis=1)


def _agreement_delta_e(model: dict, mask: np.ndarray, rgb_image: np.ndarray) -> float:
    """Mean OKLab ΔE between the rendered model and the original over the footprint."""
    ys, xs = np.where(mask)
    return float(_per_pixel_delta_e(model, ys, xs, rgb_image).mean())


def fit_gradient(mask: np.ndarray, rgb_image: np.ndarray) -> dict | None:
    """Fit a linear (then radial) gradient to the original pixels under `mask`. Returns
    the model only if it re-renders within _GATE_DELTA_E, else None."""
    ys, xs = np.where(mask)
    if len(xs) < 3 * _MIN_BANDS:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    rgb = rgb_image[ys, xs].astype(float)
    oklab = srgb_to_oklab(rgb / 255.0)
    best = None
    for fit in (_fit_linear, _fit_radial):
        model = fit(pts, oklab, rgb)
        if model is None:
            continue
        if _stop_span(model["stops"]) < _MIN_STOP_SPAN:
            continue                       # near-constant: a flat region, not a gradient
        de = _agreement_delta_e(model, mask, rgb_image)
        if de <= _GATE_DELTA_E and (best is None or de < best[0]):
            best = (de, model)
    return best[1] if best is not None else None


def _dominant_blob_fraction(mask: np.ndarray) -> float:
    """Fraction of the foreground occupied by its largest connected component
    (4-connectivity). 1.0 = one solid blob; ~0 = many disconnected pieces; 0.0 if empty."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    labels, n = ndi.label(mask)
    if n == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]       # drop background label 0
    return float(sizes.max()) / total


def _png_b64(small: np.ndarray) -> str:
    """Bare base64 PNG (no data-URI prefix) of an HxWx3 uint8 array."""
    buf = io.BytesIO()
    Image.fromarray(small.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fit_stretch(mask: np.ndarray, rgb_image: np.ndarray) -> dict | None:
    """Downsample the footprint bbox to NxN and let the renderer stretch it back
    (bilinear) to reproduce a smooth 2-D field one gradient can't. Grows N over
    _STRETCH_GRID_STEPS until the upsampled reconstruction's mean per-pixel ΔE over
    the footprint is <= _STRETCH_TARGET, else returns the lowest-ΔE (most faithful)
    grid found across _STRETCH_GRID_STEPS. None if the bbox is degenerate (<2px a
    side)."""
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    if bw < 2 or bh < 2:
        return None
    sub = mask[y0:y1, x0:x1]
    if sub.all():
        crop = rgb_image[y0:y1, x0:x1]                 # no hole: unchanged fast path
    else:
        crop = rgb_image[y0:y1, x0:x1].astype(float)
        crop[~sub] = rgb_image[ys, xs].astype(float).mean(axis=0)   # footprint mean; no outside-mask bleed
        crop = crop.round().astype(np.uint8)
    ry, rx = ys - y0, xs - x0
    truth = srgb_to_oklab(rgb_image[ys, xs].astype(float) / 255.0)
    geometry = {"x": float(x0), "y": float(y0), "w": float(bw), "h": float(bh)}
    best = None
    for n in _STRETCH_GRID_STEPS:
        small = np.asarray(Image.fromarray(crop).resize((n, n), Image.BILINEAR))
        up = np.asarray(Image.fromarray(small).resize((bw, bh), Image.BILINEAR)).astype(float)
        recon = up[ry, rx]
        de = float(np.linalg.norm(srgb_to_oklab(recon / 255.0) - truth, axis=1).mean())
        if best is None or de < best[0]:
            best = (de, small)
        if de <= _STRETCH_TARGET:
            break
    return {"kind": "raster", "geometry": geometry, "png_b64": _png_b64(best[1])}



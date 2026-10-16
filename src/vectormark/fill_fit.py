# SPDX-License-Identifier: MIT
"""Per-shape fill decision: given a shape's silhouette mask and the source pixels,
return the best Fill (flat or parametric gradient). Geometry is never touched here."""

from __future__ import annotations

import numpy as np

from .candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill
from .color import srgb_to_oklab
from .gradient import _GATE_DELTA_E, _best_parametric

# Conservative pre-check threshold: mean OKLab distance from centroid below this →
# region is effectively flat, skip the expensive parametric search entirely.
# Real gradients produce mean centroid-spread ≈ 0.010+ (genuine spans ≥ 0.039
# end-to-end per _MIN_STOP_SPAN; flat-with-AA spans ≈ 0). 0.005 is well below.
# TEMPORARY perf guard (avoids a flat-logo regression vs the pre-decoupling pipeline).
# This is a heuristic shortcut around the exact algorithm; RETIRE it once the parametric
# search / occlusion hot paths are made fast natively (algorithmic + numba/Rust follow-up),
# so correctness never depends on a threshold. See the perf follow-up.
_FLAT_OKLAB_SPREAD_THRESHOLD = 0.005
_DOMINANT_RGB_FRACTION = 0.90


def _dominant_rgb_fraction(pixels: np.ndarray) -> float:
    """Return the share represented by the most common sampled RGB color."""
    if len(pixels) == 0:
        return 0.0
    _colors, counts = np.unique(pixels, axis=0, return_counts=True)
    return float(counts.max() / len(pixels))


def fit_fill(mask: np.ndarray, rgb: np.ndarray, *, flat_hex: str,
             max_gradient_de: float = _GATE_DELTA_E,
             allow_gradients: bool = True) -> Fill:
    """Decide a shape's fill from the source pixels under `mask`.

    Returns FlatFill(flat_hex) unless a searched parametric gradient (linear or radial)
    both exists and re-renders within `max_gradient_de` mean OKLab ΔE; in that case the
    corresponding gradient fill is returned. The silhouette is the caller's; this only
    chooses how to paint inside it.

    ``allow_gradients=False`` returns the supplied flat paint without evaluating
    gradient candidates; upstream material-field evidence is unchanged.
    """
    if not allow_gradients:
        return FlatFill(flat_hex)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return FlatFill(flat_hex)
    sampled = rgb[ys, xs]
    # A vector silhouette can cover a thin antialias fringe from the source
    # background.  Those sparse white/blended samples have a large OKLab
    # distance from an otherwise flat interior and can spuriously satisfy the
    # parametric-gradient gate.  A genuine gradient does not have one exact
    # RGB color covering almost all of its area.
    if _dominant_rgb_fraction(sampled) >= _DOMINANT_RGB_FRACTION:
        return FlatFill(flat_hex)

    pixels = sampled.astype(float)
    oklab = srgb_to_oklab(pixels / 255.0)
    centroid = oklab.mean(axis=0)
    spread = float(np.linalg.norm(oklab - centroid, axis=1).mean())
    if spread < _FLAT_OKLAB_SPREAD_THRESHOLD:
        return FlatFill(flat_hex)

    best = _best_parametric(mask, rgb)
    if best is None:
        return FlatFill(flat_hex)
    model, mean_de, _median_de = best
    if mean_de > max_gradient_de:
        return FlatFill(flat_hex)
    if model["kind"] == "linear":
        return LinearGradientFill(model["geometry"], model["stops"])
    return RadialGradientFill(model["geometry"], model["stops"])

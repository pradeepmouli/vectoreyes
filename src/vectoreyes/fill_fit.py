# SPDX-License-Identifier: MIT
"""Per-shape fill decision: given a shape's silhouette mask and the source pixels,
return the best Fill (flat or parametric gradient). Geometry is never touched here."""

from __future__ import annotations

import numpy as np
import math
from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True)
class FillFitResult:
    """A selected numeric fill and the evidence or fallback that selected it."""

    fill: Fill
    status: Literal['flat_evidence', 'gradient_accepted', 'forced_flat', 'fallback_flat', 'empty_support']
    reason: str
    sampled_pixels: int
    gradient_searched: bool = False
    max_gradient_delta_e: float = _GATE_DELTA_E
    mean_delta_e: float | None = None
    median_delta_e: float | None = None
    dominant_rgb_fraction: float | None = None
    oklab_spread: float | None = None

    def diagnostics(self) -> dict[str, object]:
        def finite(value: float | None) -> float | None:
            return value if value is None or math.isfinite(value) else None
        return {'status': self.status, 'reason': self.reason,
            'sampled_pixels': self.sampled_pixels, 'gradient_searched': self.gradient_searched,
            'max_gradient_delta_e': finite(self.max_gradient_delta_e),
            'mean_delta_e': finite(self.mean_delta_e), 'median_delta_e': finite(self.median_delta_e),
            'dominant_rgb_fraction': finite(self.dominant_rgb_fraction),
            'oklab_spread': finite(self.oklab_spread)}


def _dominant_rgb_fraction(pixels: np.ndarray) -> float:
    """Return the share represented by the most common sampled RGB color."""
    if len(pixels) == 0:
        return 0.0
    _colors, counts = np.unique(pixels, axis=0, return_counts=True)
    return float(counts.max() / len(pixels))


def fit_fill(mask: np.ndarray, rgb: np.ndarray, *, flat_hex: str,
             max_gradient_de: float = _GATE_DELTA_E,
             allow_gradients: bool = True) -> FillFitResult:
    """Decide a shape's fill from the source pixels under `mask`.

    Selects FlatFill(flat_hex) unless a searched parametric gradient (linear or radial)
    both exists and re-renders within `max_gradient_de` mean OKLab ΔE; in that case the
    corresponding gradient fill is returned. The silhouette is the caller's; this only
    chooses how to paint inside it.

    ``allow_gradients=False`` returns the supplied flat paint without evaluating
    gradient candidates; upstream material-field evidence is unchanged.
    """
    ys, xs = np.where(mask)
    count = len(xs)
    evidence: dict[str, object] = {'sampled_pixels': count, 'max_gradient_delta_e': max_gradient_de}
    def flat(status, reason) -> FillFitResult:
        return FillFitResult(FlatFill(flat_hex), status, reason, **evidence)
    if len(xs) == 0:
        return flat('empty_support', 'empty_support')
    if not allow_gradients:
        return flat('forced_flat', 'gradients_disabled')
    sampled = rgb[ys, xs]
    # A vector silhouette can cover a thin antialias fringe from the source
    # background.  Those sparse white/blended samples have a large OKLab
    # distance from an otherwise flat interior and can spuriously satisfy the
    # parametric-gradient gate.  A genuine gradient does not have one exact
    # RGB color covering almost all of its area.
    dominant = _dominant_rgb_fraction(sampled)
    evidence['dominant_rgb_fraction'] = dominant
    if dominant >= _DOMINANT_RGB_FRACTION:
        return flat('flat_evidence', 'dominant_rgb')

    pixels = sampled.astype(float)
    oklab = srgb_to_oklab(pixels / 255.0)
    centroid = oklab.mean(axis=0)
    spread = float(np.linalg.norm(oklab - centroid, axis=1).mean())
    evidence['oklab_spread'] = spread
    if spread < _FLAT_OKLAB_SPREAD_THRESHOLD:
        return flat('flat_evidence', 'low_oklab_spread')

    if not math.isfinite(max_gradient_de) or max_gradient_de < 0.:
        return flat('fallback_flat', 'invalid_gradient_budget')
    best = _best_parametric(mask, rgb)
    evidence['gradient_searched'] = True
    if best is None:
        return flat('fallback_flat', 'no_gradient_candidate')
    model, mean_de, median_de = best
    evidence.update(mean_delta_e=float(mean_de), median_delta_e=float(median_de))
    if not math.isfinite(mean_de) or not math.isfinite(median_de):
        return flat('fallback_flat', 'invalid_gradient_error')
    if mean_de > max_gradient_de:
        return flat('fallback_flat', 'gradient_error_exceeded')
    if model["kind"] == "linear":
        fill = LinearGradientFill(model["geometry"], model["stops"])
    else:
        fill = RadialGradientFill(model["geometry"], model["stops"])
    return FillFitResult(fill, 'gradient_accepted', 'gradient_within_budget', **evidence)

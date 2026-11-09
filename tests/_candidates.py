"""Eval-only candidate-variant generator: produce the competing renderings for one
region by forcing each strategy. NOT production code — production multi-candidate
generation is roadmap slice 4. Lives in tests so it never reaches `idealize`."""

from __future__ import annotations

import numpy as np

from vectoreyes.candidate import (
    Candidate, FlatFill, LinearGradientFill, RadialGradientFill,
)
from vectoreyes.contour import region_contours
from vectoreyes.fit import fit_path, recognize_primitive
from vectoreyes.gradient import fit_gradient
from vectoreyes.types import Region


def generate_candidates(
    region: Region, source_rgb: np.ndarray, *, epsilon: float = 1.5, max_error: float = 1.0,
) -> list[Candidate]:
    """Competing candidates for `region`: smooth-path+flat (always), primitive+flat
    (if recognised), and gradient (if `fit_gradient` returns a model)."""
    contours = region_contours(region.mask)
    if not contours:
        return []
    contour = contours[0]
    cands: list[Candidate] = []

    path = fit_path(contour, epsilon=epsilon, max_error=max_error)
    cands.append(Candidate(path, FlatFill(region.color_hex), "region"))

    prim = recognize_primitive(contour, epsilon=epsilon)
    if prim is not None:
        cands.append(Candidate(prim, FlatFill(region.color_hex), "region"))

    model = fit_gradient(region.mask, source_rgb)
    if model is not None:
        g = model["geometry"]
        grad_fill = (LinearGradientFill(g, model["stops"]) if model["kind"] == "linear"
                     else RadialGradientFill(g, model["stops"]))
        cands.append(Candidate(path, grad_fill, "gradient"))

    return cands

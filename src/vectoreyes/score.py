"""Candidate scorer (prototype): structural-prior gates -> render-ΔE fidelity
gate -> structured description-length parsimony -> ΔE tiebreak. Ranks how an
element could be rendered. Standalone; not wired into the pipeline (slice 4)."""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image

_renderer_warned = False   # warn at most once per process when resvg is unavailable

from .candidate import Candidate, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .color import mean_delta_e
from .emit import (
    fill_rule_for, path_svg, render_svg_doc, resolve_fill, shape_to_path_d,
)
from .gradient import _BLOB_DOMINANCE, _MIN_STOP_SPAN, _dominant_blob_fraction, _stop_span
from .types import Region

# --- parsimony weights (tunable) -------------------------------------------------
_GEOM_PARAMS = {"circle": 3, "rect": 4, "ellipse": 5, "annulus": 6}
_CMD_COST = {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}   # coords per path command
_FILL_FLAT = 1.0


def _path_cost(d: str) -> float:
    """Description length of a path `d`: sum of per-command coordinate counts."""
    # Assumes M/L/C/Q/Z commands only — the sole set this package's emitters produce.
    return float(sum(_CMD_COST.get(cmd, 0) for cmd in re.findall(r"[MLCQZ]", d.upper())))


def parsimony_cost(cand: Candidate) -> float:
    """Structured description-length of a candidate: geometry params + fill params.
    Lower is simpler. Decoupled from SVG text formatting."""
    g = cand.geometry
    if g.kind == "polygon":
        geom = 2.0 * len(g.params["points"])
    elif g.kind == "path":
        geom = _path_cost(g.params["d"])
    else:
        geom = float(_GEOM_PARAMS[g.kind])

    f = cand.fill
    if isinstance(f, FlatFill):
        fill = _FILL_FLAT
    elif isinstance(f, LinearGradientFill):
        fill = 4.0 + 2.0 * len(f.stops)
    elif isinstance(f, RadialGradientFill):
        fill = 3.0 + 2.0 * len(f.stops)
    else:  # RasterFill: one embedded image, fixed cost
        fill = 8.0
    return geom + fill


# --- fidelity (render-ΔE via optional resvg) -------------------------------------
class SvgRendererUnavailable(RuntimeError):
    """Raised when SVG rasterization is unavailable in the current runtime."""


def _warn_renderer_unavailable() -> None:
    """Surface the silent scoring downgrade once per process (not per region/element)."""
    global _renderer_warned
    if not _renderer_warned:
        _renderer_warned = True
        warnings.warn(
            "resvg-py is not installed: render-ΔE scoring is disabled, so candidate "
            "selection falls back to the cascade-priority pick. Install the scoring "
            "extra (pip install 'vectoreyes[scoring]') to enable scored selection.",
            RuntimeWarning, stacklevel=2,
        )


def _candidate_svg(cand: Candidate, w: int, h: int) -> str:
    defs: list[str] = []
    fill = resolve_fill(cand.fill, defs)
    rule = fill_rule_for(cand.geometry)
    body = [path_svg(shape_to_path_d(cand.geometry), fill, rule)]
    return render_svg_doc(w, h, body, defs)


def _rasterize(svg: str, w: int, h: int) -> np.ndarray:
    """SVG -> (h, w, 3) uint8 composited on white. (Mirrors tests/_render.render_svg;
    kept local so src/ does not import test helpers — unify in a later DRY pass.)"""
    try:
        import resvg_py
    except ModuleNotFoundError as exc:
        raise SvgRendererUnavailable("install resvg-py to enable render-based scoring") from exc

    png = resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h)
    img = Image.open(io.BytesIO(bytes(png)))
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
    return np.asarray(bg, dtype=np.uint8)


def render_delta_e(
    cand: Candidate, source_rgb: np.ndarray, region: Region, *,
    bbox: tuple[int, int, int, int] | None = None,
) -> float:
    """Render the candidate over the source canvas and compare (mean OKLab ΔE)
    against the source within the region's footprint. 0 = identical. When `bbox`
    (x0, y0, x1, y1) is given, the rasterization is still full-canvas (resvg needs
    the canvas dims), but the ΔE comparison is restricted to that crop (mask
    intersected with the bbox) — a speed optimization on the comparison; identical
    result to full-canvas for the compared pixels."""
    h, w = source_rgb.shape[:2]
    mask = region.mask
    if not mask.any():
        return float("inf")
    raster = _rasterize(_candidate_svg(cand, w, h), w, h)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
        if x1 > x0 and y1 > y0:
            sub = mask[y0:y1, x0:x1]
            if sub.any():
                return mean_delta_e(source_rgb[y0:y1, x0:x1][sub], raster[y0:y1, x0:x1][sub])
    return mean_delta_e(source_rgb[mask], raster[mask])


# --- structural priors (hard gates) ---------------------------------------------
def structural_priors(cand: Candidate, region: Region) -> tuple[bool, str | None]:
    """Hard gates generalising the proven gradient guards. A failure disqualifies
    the candidate (with a reason); flat/primitive/path candidates have no prior."""
    f = cand.fill
    if isinstance(f, (LinearGradientFill, RadialGradientFill)):
        if _stop_span(f.stops) < _MIN_STOP_SPAN:
            return False, "gradient stop-span below minimum"
        if _dominant_blob_fraction(region.mask) < _BLOB_DOMINANCE:
            return False, "gradient footprint not a single dominant blob"
    return True, None


# --- the scorer ------------------------------------------------------------------
@dataclass
class ScoreBreakdown:
    delta_e: float             # render-ΔE fidelity (lower = better); inf if priors failed
    parsimony: float           # structured description-length (lower = better)
    priors_ok: bool            # passed all structural-prior hard gates
    reject_reason: str | None  # which prior failed (transparency for override)
    qualified: bool            # priors_ok AND delta_e <= fidelity_tol


def rank_candidates(
    cands: list[Candidate], source_rgb: np.ndarray, region: Region, *,
    fidelity_tol: float = 0.06, bbox: tuple[int, int, int, int] | None = None,
) -> list[tuple[Candidate, ScoreBreakdown]]:
    """Rank candidates best-first by the lexicographic rule: qualifiers (priors ok
    AND ΔE <= fidelity_tol) first, ordered by parsimony then ΔE; disqualified
    candidates follow, ordered by ΔE. Returns the full list with breakdowns so a
    caller can inspect/override. When `bbox` (x0, y0, x1, y1) is given it is
    forwarded to render_delta_e for a bbox-cropped fidelity comparison."""
    scored: list[tuple[Candidate, ScoreBreakdown]] = []
    renderer_available = True
    for c in cands:
        ok, reason = structural_priors(c, region)
        de = float("inf")
        if ok and renderer_available:
            try:
                de = render_delta_e(c, source_rgb, region, bbox=bbox)
            except SvgRendererUnavailable:
                renderer_available = False
                _warn_renderer_unavailable()
        par = parsimony_cost(c)
        scored.append((c, ScoreBreakdown(de, par, ok, reason, ok and de <= fidelity_tol)))

    if not renderer_available:
        # No fidelity signal: keep the candidates' incoming (cascade-priority) order,
        # priors-failures last. scored[0] is then the cascade pick — the pre-scoring
        # behaviour, made explicit (and surfaced via the one-time warning above).
        scored.sort(key=lambda cb: not cb[1].priors_ok)
        return scored

    scored.sort(key=lambda cb: (
        not cb[1].qualified,
        cb[1].parsimony if cb[1].qualified else 0.0,
        cb[1].delta_e,
    ))
    return scored

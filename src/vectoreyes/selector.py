"""Geometry candidate generation + scored selection (slice 4a). Replaces
_fit_region's first-non-None cascade with collect-all-then-score: the generator
emits exactly the cascade's fitters (so the selector can only re-prioritise among
known-good geometries, never invent a new output type), in cascade-priority order
(so candidates[0] == the old cascade pick)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .candidate import Candidate, FlatFill
from .contour import region_contours
from .fit import Shape, fit_path, recognize_polygon, recognize_primitive
from .refine import (
    half_ellipse_cap_fit, rounded_trapezoid_fit, symmetric_fit, symmetric_polygon_fit,
)
from .score import rank_candidates
from .selection import (
    PRIMITIVE, TRAPEZOID, SYM_POLYGON, CAP, SYMMETRIC, POLYGON, PATH,
    HOLED_SYM, HOLED_PATH,
    ElementSelection, validate_strategies,
)
from .types import Axis, Region


@dataclass(frozen=True)
class GeomCandidate:
    """A generated geometry paired with the fitter (strategy) that produced it."""
    strategy: str
    shape: Shape


def _snap_to_axis(shape: Shape, axis: Axis) -> Shape:
    """Force x-centre of a straddling primitive onto the axis for exact symmetry."""
    if shape.kind in ("circle", "ellipse"):
        shape.params["cx"] = axis.x
    elif shape.kind == "rect":
        shape.params["x"] = axis.x - shape.params["w"] / 2
    return shape


def generate_geometry_candidates(
    region: Region, opt, axis: Axis | None, corner_radius: float,
) -> list[GeomCandidate]:
    """All geometry fits the cascade could produce for this region, in cascade
    priority order (candidates[0].shape == the old _fit_region pick), non-None only,
    each tagged with its producing strategy.

    For a straddler (axis set) the non-symmetric fallbacks (recognize_polygon,
    fit_path) are added ONLY when nothing symmetry-preserving was produced — i.e.
    neither a refine fit (`sym`) NOR an axis-snapped primitive — so the scorer can
    never pick a cheaper non-symmetric geometry over a valid symmetric one."""
    contours = [c for c in region_contours(region.mask, coverage=region.coverage) if len(c) >= 3]
    if not contours:
        return []

    if len(contours) > 1:                       # holed / counter
        # When every contour straddles cleanly, the symmetric half-mirror is the ONLY
        # candidate (matching the cascade) — never also offer the non-symmetric
        # per-contour fit, which the scorer could otherwise pick and break symmetry.
        if axis is not None:
            halves = [
                symmetric_fit(c, axis.x, corner_radius=corner_radius,
                              epsilon=opt.epsilon, max_error=opt.max_error)
                for c in contours
            ]
            if all(s is not None for s in halves):
                d = " ".join(s.params["d"] for s in halves)
                return [GeomCandidate(HOLED_SYM, Shape("path", {"d": d, "fill_rule": "evenodd"}))]
        # No clean symmetric construction: faithful per-contour fit (even-odd).
        d = " ".join(
            fit_path(c, epsilon=opt.epsilon, max_error=opt.max_error, cubic=opt.cubic_paths).params["d"]
            for c in contours
        )
        return [GeomCandidate(HOLED_PATH, Shape("path", {"d": d, "fill_rule": "evenodd"}))]

    contour = contours[0]
    cands: list[GeomCandidate] = []

    prim = recognize_primitive(contour, epsilon=opt.epsilon)
    if prim is not None:
        cands.append(GeomCandidate(PRIMITIVE, _snap_to_axis(prim, axis) if axis is not None else prim))

    sym: list[GeomCandidate] = []
    if axis is not None:
        trap = rounded_trapezoid_fit(contour, axis.x, radius=corner_radius, max_error=opt.max_error)
        if trap is not None:
            sym.append(GeomCandidate(TRAPEZOID, trap))
        poly = symmetric_polygon_fit(contour, axis.x, epsilon=opt.epsilon)
        if poly is not None:
            sym.append(GeomCandidate(SYM_POLYGON, poly))
        cap = half_ellipse_cap_fit(contour, axis.x, corner_radius=corner_radius, max_error=opt.max_error)
        if cap is not None:
            sym.append(GeomCandidate(CAP, cap))
        s = symmetric_fit(contour, axis.x, corner_radius=corner_radius,
                          epsilon=opt.epsilon, max_error=opt.max_error)
        if s is not None:
            sym.append(GeomCandidate(SYMMETRIC, s))
    cands.extend(sym)

    # An axis-snapped primitive is symmetry-preserving (its centre is forced onto the
    # axis), so a symmetry-preserving candidate exists when EITHER a refine fit (`sym`)
    # OR a recognized primitive is present.
    has_symmetry_preserving = bool(sym) or (axis is not None and prim is not None)

    # Non-symmetric fallbacks: only when there is no symmetry to preserve (axis None) OR
    # no symmetry-preserving candidate was produced — guarantees a non-empty set without
    # ever letting a non-symmetric candidate compete with a symmetric one.
    if axis is None or not has_symmetry_preserving:
        gpoly = recognize_polygon(contour, epsilon=opt.epsilon)
        if gpoly is not None:
            cands.append(GeomCandidate(POLYGON, gpoly))
        cands.append(GeomCandidate(PATH, fit_path(contour, epsilon=opt.epsilon, max_error=opt.max_error, cubic=opt.cubic_paths)))

    return cands


def _region_bbox(mask: np.ndarray, margin: int = 2) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(w, int(xs.max()) + 1 + margin), min(h, int(ys.max()) + 1 + margin))


def select_geometry(
    region: Region, opt, axis: Axis | None, corner_radius: float,
    source_rgb: np.ndarray | None, *,
    element: ElementSelection | None = None, eid: str = "?",
) -> tuple[Shape | None, str | None]:
    """Generate geometry candidates, optionally apply a manual `element` policy, score,
    and return `(winning_shape, strategy_label)`. With `element` None this is the 4a
    auto-selector (pure pass-through). `eid` only labels warning messages.

    Stages (skipped when element is None): validate the policy's strategy labels;
    pre-restriction keeps only `allow`ed strategies (warn + restore if that empties the
    set); post-override returns the highest-ranked candidate whose strategy == `force`
    (warn + auto winner if absent)."""
    if element is not None:
        validate_strategies(element)

    cands = generate_geometry_candidates(region, opt, axis, corner_radius)
    if not cands:
        return None, None

    if element is not None and element.allow is not None:
        kept = [gc for gc in cands if gc.strategy in element.allow]
        if not kept:
            warnings.warn(
                f"selection {eid}: allow={sorted(element.allow)} removed all candidates "
                f"(have {[gc.strategy for gc in cands]}); ignoring restriction",
                UserWarning, stacklevel=2,
            )
        else:
            cands = kept

    force = element.force if element is not None else None

    if source_rgb is None:
        if force is not None:
            hit = next((gc for gc in cands if gc.strategy == force), None)
            if hit is None:
                warnings.warn(
                    f"selection {eid}: force='{force}' not among "
                    f"{[gc.strategy for gc in cands]}; using '{cands[0].strategy}'",
                    UserWarning, stacklevel=2,
                )
                hit = cands[0]
            return hit.shape, hit.strategy
        return cands[0].shape, cands[0].strategy

    wrapped = [Candidate(gc.shape, FlatFill(region.color_hex), "region", strategy=gc.strategy)
               for gc in cands]
    bbox = _region_bbox(region.mask)
    tol = getattr(opt, "fidelity_tol", 0.06)
    ranked = rank_candidates(wrapped, source_rgb, region, fidelity_tol=tol, bbox=bbox)

    if force is not None:
        hit = next((c for c, _ in ranked if c.strategy == force), None)
        if hit is None:
            warnings.warn(
                f"selection {eid}: force='{force}' not among "
                f"{[c.strategy for c, _ in ranked]}; using auto winner "
                f"'{ranked[0][0].strategy}'",
                UserWarning, stacklevel=2,
            )
            hit = ranked[0][0]
        return hit.geometry, hit.strategy
    winner = ranked[0][0]
    return winner.geometry, winner.strategy

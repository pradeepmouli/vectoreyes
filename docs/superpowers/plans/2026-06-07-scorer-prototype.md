# Fidelity + Parsimony Scorer (Prototype) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone candidate scorer that ranks how an element could be rendered — structural-prior hard gates → render-ΔE fidelity gate → structured description-length parsimony → ΔE tiebreak — evaluated against a labelled synthetic corpus (false-accept target 0), not wired into `idealize`.

**Architecture:** New `src/vectoreyes/score.py` (pure scoring + a resvg-backed fidelity measure). A test-side candidate-variant generator (`tests/_candidates.py`) and a labelled synthetic eval (`tests/test_score_eval.py`) exercise it; a dev script runs it over the untracked real-logo corpus. Reuses `Candidate`/`Fill` (slice 2), the gradient guards, and the existing fitters.

**Tech Stack:** Python, numpy, resvg-py (promoted to runtime dep), OKLab `mean_delta_e`, pytest.

**Spec:** `docs/superpowers/specs/2026-06-07-scorer-prototype-design.md`
**Branch:** `feat/scorer` (off `master`; the slice-2 `Candidate` type is already on master via PR #16).

---

## Background the implementer needs

Reused symbols (do NOT modify them):
- `from .candidate import Candidate, FlatFill, LinearGradientFill, RadialGradientFill` — `Candidate(geometry: Shape, fill: Fill, source: str, mirror=None)`. `LinearGradientFill(geometry: dict {x1,y1,x2,y2}, stops: list)`, `RadialGradientFill(geometry: dict {cx,cy,r}, stops)`, `FlatFill(hex)`.
- `from .fit import Shape, recognize_primitive, fit_path` — `Shape(kind, params)`; `recognize_primitive(contour, *, epsilon) -> Shape | None`; `fit_path(contour, *, epsilon, max_error) -> Shape`. Shape kinds: `circle{cx,cy,r}`, `ellipse{cx,cy,rx,ry}`, `rect{x,y,w,h}`, `polygon{points}`, `annulus{cx,cy,r_outer,r_inner}`, `path{d, [fill_rule]}`.
- `from .emit import shape_to_path_d, path_svg, render_svg_doc, linear_gradient_def, radial_gradient_def` — `shape_to_path_d(shape) -> str` (handles every kind incl. `path` → returns `d`); `path_svg(d, fill, fill_rule=None) -> str`; `render_svg_doc(w, h, body: list[str], defs=None) -> str`; `linear_gradient_def(id, x1,y1,x2,y2, stops) -> str`; `radial_gradient_def(id, cx,cy,r, stops) -> str`.
- `from .gradient import _stop_span, _dominant_blob_fraction, fit_gradient, _MIN_STOP_SPAN, _BLOB_DOMINANCE` — `_stop_span(stops) -> float` (OKLab end-to-end travel); `_dominant_blob_fraction(mask) -> float`; `fit_gradient(mask, rgb_image) -> dict | None` with keys `kind` ("linear"/"radial"), `geometry`, `stops`; `_MIN_STOP_SPAN=0.02`; `_BLOB_DOMINANCE=0.85`.
- `from .color import mean_delta_e` — `mean_delta_e(a, b) -> float` mean per-pixel OKLab ΔE of two uint8 (...,3) arrays.
- `from .contour import region_contours` — `region_contours(mask) -> list[np.ndarray]` (outer first).
- `from .types import Region` — `Region(label: int, mask: np.ndarray bool, color_hex: str)`, `.area` property.
- Rasterize pattern (from `tests/_render.py`): `resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h)` → PNG bytes → PIL → composite on white → uint8 (H,W,3).

Run tests with `.venv/bin/pytest`. `score.py` must NOT be imported by `pipeline.py`/`idealize` (slice 4 wires it in); the existing suite stays green.

---

## Task 1: `parsimony_cost` (structured description-length)

**Files:**
- Create: `src/vectoreyes/score.py`
- Test: `tests/test_score.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_score.py`:

```python
from vectoreyes.candidate import (
    Candidate, FlatFill, LinearGradientFill, RadialGradientFill,
)
from vectoreyes.fit import Shape
from vectoreyes.score import parsimony_cost


def _flat(shape):
    return Candidate(shape, FlatFill("#123456"), "region")


def test_parsimony_primitive_cheaper_than_path():
    circle = _flat(Shape("circle", {"cx": 5, "cy": 5, "r": 4}))
    path = _flat(Shape("path", {"d": "M0 0 C1 1 2 2 3 3 C4 4 5 5 6 6 C7 7 8 8 9 9 Z"}))
    assert parsimony_cost(circle) < parsimony_cost(path)


def test_parsimony_flat_cheaper_than_gradient_same_geometry():
    geom = Shape("rect", {"x": 0, "y": 0, "w": 10, "h": 10})
    flat = Candidate(geom, FlatFill("#000000"), "region")
    grad = Candidate(geom, LinearGradientFill({"x1": 0, "y1": 0, "x2": 10, "y2": 0},
                                              [(0.0, "#000"), (1.0, "#fff")]), "gradient")
    assert parsimony_cost(flat) < parsimony_cost(grad)


def test_parsimony_polygon_scales_with_vertices():
    tri = _flat(Shape("polygon", {"points": [(0, 0), (1, 0), (0, 1)]}))
    hexa = _flat(Shape("polygon", {"points": [(0, 0), (1, 0), (2, 1), (1, 2), (0, 2), (-1, 1)]}))
    assert parsimony_cost(hexa) > parsimony_cost(tri)
```

- [ ] **Step 2: Run it, verify FAIL**

Run: `.venv/bin/pytest tests/test_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vectoreyes.score'`.

- [ ] **Step 3: Create `src/vectoreyes/score.py` with the parsimony layer**

```python
"""Candidate scorer (prototype): structural-prior gates -> render-ΔE fidelity
gate -> structured description-length parsimony -> ΔE tiebreak. Ranks how an
element could be rendered. Standalone; not wired into the pipeline (slice 4)."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
import resvg_py
from PIL import Image

from .candidate import Candidate, FlatFill, LinearGradientFill, RadialGradientFill
from .color import mean_delta_e
from .emit import (
    linear_gradient_def, path_svg, radial_gradient_def, render_svg_doc, shape_to_path_d,
)
from .gradient import _BLOB_DOMINANCE, _MIN_STOP_SPAN, _dominant_blob_fraction, _stop_span
from .types import Region

# --- parsimony weights (tunable) -------------------------------------------------
_GEOM_PARAMS = {"circle": 3, "rect": 4, "ellipse": 5, "annulus": 6}
_CMD_COST = {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}   # coords per path command
_FILL_FLAT = 1.0


def _path_cost(d: str) -> float:
    """Description length of a path `d`: sum of per-command coordinate counts."""
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
    else:  # RadialGradientFill
        fill = 3.0 + 2.0 * len(f.stops)
    return geom + fill
```

- [ ] **Step 4: Run it, verify PASS**

Run: `.venv/bin/pytest tests/test_score.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/score.py tests/test_score.py
git commit -m "feat(score): structured description-length parsimony_cost

Geometry params + fill params per candidate; lower = simpler. Primitive <
many-Bezier path; flat < gradient. Tunable weights. First piece of the
standalone candidate scorer.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: `render_delta_e` (resvg fidelity) + runtime dep

**Files:**
- Modify: `src/vectoreyes/score.py` (add render layer)
- Modify: `pyproject.toml` (move `resvg-py` dev → runtime)
- Test: `tests/test_score.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_score.py`:

```python
import numpy as np
from vectoreyes.types import Region
from vectoreyes.score import render_delta_e


def _disk_region(cx, cy, r, h, w, color_hex):
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    return Region(1, mask, color_hex)


def test_render_delta_e_near_zero_on_matching_flat():
    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(30, 30, 18, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    cand = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}),
                     FlatFill("#1e64eb"), "region")
    assert render_delta_e(cand, src, region) < 0.03


def test_render_delta_e_large_on_wrong_color():
    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(30, 30, 18, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    wrong = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}),
                      FlatFill("#ff0000"), "region")
    assert render_delta_e(wrong, src, region) > 0.2
```

- [ ] **Step 2: Run it, verify FAIL**

Run: `.venv/bin/pytest tests/test_score.py::test_render_delta_e_near_zero_on_matching_flat -v`
Expected: FAIL — `render_delta_e` not defined.

- [ ] **Step 3: Add the render layer to `score.py`**

Append to `src/vectoreyes/score.py`:

```python
# --- fidelity (render-ΔE via resvg) ---------------------------------------------
def _resolve_fill_str(fill, defs: list[str]) -> str:
    """Flat -> hex; gradient -> register a def (g{N}) and return url(#...)."""
    if isinstance(fill, FlatFill):
        return fill.hex
    g = fill.geometry
    gid = f"g{len(defs)}"
    if isinstance(fill, LinearGradientFill):
        defs.append(linear_gradient_def(gid, g["x1"], g["y1"], g["x2"], g["y2"], fill.stops))
    else:
        defs.append(radial_gradient_def(gid, g["cx"], g["cy"], g["r"], fill.stops))
    return f"url(#{gid})"


def _candidate_svg(cand: Candidate, w: int, h: int) -> str:
    defs: list[str] = []
    fill = _resolve_fill_str(cand.fill, defs)
    rule = cand.geometry.params.get("fill_rule",
                                    "evenodd" if cand.geometry.kind == "annulus" else None)
    body = [path_svg(shape_to_path_d(cand.geometry), fill, rule)]
    return render_svg_doc(w, h, body, defs)


def _rasterize(svg: str, w: int, h: int) -> np.ndarray:
    """SVG -> (h, w, 3) uint8 composited on white. (Mirrors tests/_render.render_svg;
    kept local so src/ does not import test helpers — unify in a later DRY pass.)"""
    png = resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h)
    img = Image.open(io.BytesIO(bytes(png)))
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
    return np.asarray(bg, dtype=np.uint8)


def render_delta_e(cand: Candidate, source_rgb: np.ndarray, region: Region) -> float:
    """Render the candidate over the source canvas and compare (mean OKLab ΔE)
    against the source within the region's footprint. 0 = identical."""
    h, w = source_rgb.shape[:2]
    mask = region.mask
    if not mask.any():
        return float("inf")
    raster = _rasterize(_candidate_svg(cand, w, h), w, h)
    return mean_delta_e(source_rgb[mask], raster[mask])
```

- [ ] **Step 4: Move `resvg-py` to a runtime dependency**

In `pyproject.toml`, the `dependencies` array currently lists `numpy`, `scipy`, `scikit-image`, `pillow`. Add `"resvg-py>=0.1.7"` to it, and REMOVE `resvg-py>=0.1.7` from the `dev` extra (the `dev` line `dev = ["pytest>=8.0", "resvg-py>=0.1.7"]` becomes `dev = ["pytest>=8.0"]`). `score.py` (production code) imports `resvg_py`.

- [ ] **Step 5: Run the tests, verify PASS**

Run: `.venv/bin/pytest tests/test_score.py -v`
Expected: all pass (5 total now).

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/score.py tests/test_score.py pyproject.toml
git commit -m "feat(score): render-ΔE fidelity via resvg

render_delta_e rasterizes a candidate over the source canvas and compares
mean OKLab ΔE within the region footprint. Promote resvg-py to a runtime
dependency (score.py renders candidates). Local _rasterize mirrors the test
helper (DRY unify deferred).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: `structural_priors` (hard gates from the proven guards)

**Files:**
- Modify: `src/vectoreyes/score.py`
- Test: `tests/test_score.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_score.py`:

```python
from vectoreyes.score import structural_priors


def _square_region(h, w, color_hex):
    mask = np.zeros((h, w), bool)
    mask[10:50, 10:50] = True
    return Region(1, mask, color_hex)


def test_priors_reject_degenerate_gradient_stop_span():
    region = _square_region(60, 60, "#202020")
    # near-identical stops => stop-span below _MIN_STOP_SPAN
    grad = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     LinearGradientFill({"x1": 10, "y1": 10, "x2": 50, "y2": 10},
                                        [(0.0, "#202020"), (1.0, "#212121")]), "gradient")
    ok, reason = structural_priors(grad, region)
    assert ok is False and "stop-span" in reason


def test_priors_pass_flat_and_primitive():
    region = _square_region(60, 60, "#202020")
    flat = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     FlatFill("#202020"), "region")
    ok, reason = structural_priors(flat, region)
    assert ok is True and reason is None
```

- [ ] **Step 2: Run it, verify FAIL**

Run: `.venv/bin/pytest tests/test_score.py::test_priors_reject_degenerate_gradient_stop_span -v`
Expected: FAIL — `structural_priors` not defined.

- [ ] **Step 3: Add `structural_priors` to `score.py`**

Append to `src/vectoreyes/score.py`:

```python
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
```

- [ ] **Step 4: Run the tests, verify PASS**

Run: `.venv/bin/pytest tests/test_score.py -v`
Expected: all pass (7 total).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/score.py tests/test_score.py
git commit -m "feat(score): structural-prior hard gates

Generalise the proven gradient guards (min stop-span, single-dominant-blob)
into disqualifying predicates with reasons. Kills near-degenerate 'gradients'
on flat regions (the Pinterest/Vimeo over-accept) before any render.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: `ScoreBreakdown` + `rank_candidates` (lexicographic)

**Files:**
- Modify: `src/vectoreyes/score.py`
- Test: `tests/test_score.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_score.py`:

```python
from vectoreyes.score import ScoreBreakdown, rank_candidates


def test_rank_prefers_parsimony_among_fidelity_qualifiers():
    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(30, 30, 18, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    circle = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}),
                       FlatFill("#1e64eb"), "region")
    # a faithful but heavy path approximation of the same disk
    path_d = "M48 30 C48 40 40 48 30 48 C20 48 12 40 12 30 C12 20 20 12 30 12 C40 12 48 20 48 30 Z"
    path = Candidate(Shape("path", {"d": path_d}), FlatFill("#1e64eb"), "region")
    ranked = rank_candidates([path, circle], src, region, fidelity_tol=0.06)
    assert ranked[0][0] is circle                 # both qualify on ΔE; circle wins on parsimony
    assert all(isinstance(b, ScoreBreakdown) for _, b in ranked)
    assert ranked[0][1].qualified is True


def test_rank_disqualifies_degenerate_gradient_keeps_flat():
    region = _square_region(60, 60, "#202020")
    src = np.full((60, 60, 3), 255, np.uint8)
    src[region.mask] = (32, 32, 32)
    flat = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     FlatFill("#202020"), "region")
    grad = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     LinearGradientFill({"x1": 10, "y1": 10, "x2": 50, "y2": 10},
                                        [(0.0, "#202020"), (1.0, "#212121")]), "gradient")
    ranked = rank_candidates([grad, flat], src, region)
    assert ranked[0][0] is flat                   # gradient disqualified by prior; flat wins
    grad_bd = next(b for c, b in ranked if c is grad)
    assert grad_bd.priors_ok is False and grad_bd.qualified is False
```

- [ ] **Step 2: Run it, verify FAIL**

Run: `.venv/bin/pytest tests/test_score.py::test_rank_prefers_parsimony_among_fidelity_qualifiers -v`
Expected: FAIL — `rank_candidates`/`ScoreBreakdown` not defined.

- [ ] **Step 3: Add `ScoreBreakdown` + `rank_candidates` to `score.py`**

Append to `src/vectoreyes/score.py`:

```python
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
    fidelity_tol: float = 0.06,
) -> list[tuple[Candidate, ScoreBreakdown]]:
    """Rank candidates best-first by the lexicographic rule: qualifiers (priors ok
    AND ΔE <= fidelity_tol) first, ordered by parsimony then ΔE; disqualified
    candidates follow, ordered by ΔE. Returns the full list with breakdowns so a
    caller can inspect/override."""
    scored: list[tuple[Candidate, ScoreBreakdown]] = []
    for c in cands:
        ok, reason = structural_priors(c, region)
        de = render_delta_e(c, source_rgb, region) if ok else float("inf")
        par = parsimony_cost(c)
        scored.append((c, ScoreBreakdown(de, par, ok, reason, ok and de <= fidelity_tol)))

    scored.sort(key=lambda cb: (
        not cb[1].qualified,                          # qualified first
        cb[1].parsimony if cb[1].qualified else 0.0,  # then most parsimonious
        cb[1].delta_e,                                # then best fidelity (tiebreak)
    ))
    return scored
```

- [ ] **Step 4: Run the tests, verify PASS**

Run: `.venv/bin/pytest tests/test_score.py -v`
Expected: all pass (9 total).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/score.py tests/test_score.py
git commit -m "feat(score): rank_candidates (lexicographic fidelity-gate)

ScoreBreakdown + rank_candidates: structural priors -> ΔE fidelity gate ->
parsimony -> ΔE tiebreak. Returns the full ranked list with breakdowns so an
agent/user can inspect and override. Winner is element 0.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Candidate-variant generator (eval support)

**Files:**
- Create: `tests/_candidates.py`
- Test: `tests/test_candidates_gen.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_candidates_gen.py`:

```python
import numpy as np

from vectoreyes.candidate import FlatFill, LinearGradientFill, RadialGradientFill
from tests._candidates import generate_candidates


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def test_generate_includes_primitive_and_path_for_a_disk():
    from vectoreyes.types import Region
    h = w = 80
    src = np.full((h, w, 3), 255, np.uint8)
    mask = _disk(40, 40, 25, h, w)
    src[mask] = (30, 100, 235)
    region = Region(1, mask, "#1e64eb")
    cands = generate_candidates(region, src)
    kinds = {c.geometry.kind for c in cands}
    assert "circle" in kinds          # primitive-snap
    assert "path" in kinds            # smooth-path
    assert all(isinstance(c.fill, FlatFill) for c in cands)   # flat disk => no gradient candidate
```

- [ ] **Step 2: Run it, verify FAIL**

Run: `.venv/bin/pytest tests/test_candidates_gen.py -v`
Expected: FAIL — `tests._candidates` not found.

- [ ] **Step 3: Create `tests/_candidates.py`**

```python
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
```

- [ ] **Step 4: Run the test, verify PASS**

Run: `.venv/bin/pytest tests/test_candidates_gen.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/_candidates.py tests/test_candidates_gen.py
git commit -m "test(score): eval-only candidate-variant generator

generate_candidates produces competing renderings (smooth-path+flat,
primitive+flat, gradient) for one region by forcing each strategy. Test-side
only; production multi-candidate generation is slice 4.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Labelled synthetic eval (headline; false-accept = 0)

**Files:**
- Test: `tests/test_score_eval.py`

- [ ] **Step 1: Write the eval**

Create `tests/test_score_eval.py`:

```python
"""Labelled synthetic corpus for the scorer: each case's correct winner is known
by construction. The headline gate is false-accept == 0 — especially the
flat-square-stays-flat over-accept regression."""
import numpy as np
import pytest

from vectoreyes.candidate import FlatFill, LinearGradientFill, RadialGradientFill
from vectoreyes.types import Region
from vectoreyes.score import rank_candidates
from tests._candidates import generate_candidates


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _flat_circle():
    h = w = 100
    src = np.full((h, w, 3), 255, np.uint8)
    mask = _disk(50, 50, 32, h, w)
    src[mask] = (30, 100, 235)
    return src, Region(1, mask, "#1e64eb"), "primitive_flat"


def _flat_square():
    h = w = 100
    src = np.full((h, w, 3), 255, np.uint8)
    mask = np.zeros((h, w), bool); mask[25:75, 25:75] = True
    src[mask] = (200, 30, 30)
    return src, Region(1, mask, "#c81e1e"), "primitive_flat"


def _linear_gradient_rect():
    h = w = 120
    src = np.full((h, w, 3), 255, np.uint8)
    mask = np.zeros((h, w), bool); mask[30:90, 20:100] = True
    c0 = np.array([37, 99, 235]); c1 = np.array([219, 39, 119])
    xs = np.arange(20, 100)
    for i, x in enumerate(xs):
        t = i / (len(xs) - 1)
        src[30:90, x] = np.round(c0 * (1 - t) + c1 * t).astype(np.uint8)
    return src, Region(1, mask, "#7b51ab"), "gradient"


def _radial_gradient_disc():
    h = w = 120
    src = np.full((h, w, 3), 255, np.uint8)
    cx, cy, r = 60, 60, 45
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= r
    c0 = np.array([125, 211, 252]); c1 = np.array([29, 78, 216])
    t = np.clip(dist / r, 0, 1)[..., None]
    grad = np.round(c0 * (1 - t) + c1 * t).astype(np.uint8)
    src[mask] = grad[mask]
    return src, Region(1, mask, "#52a5e4"), "gradient"


CASES = [_flat_circle(), _flat_square(), _linear_gradient_rect(), _radial_gradient_disc()]


def _winner_label(cand):
    if isinstance(cand.fill, (LinearGradientFill, RadialGradientFill)):
        return "gradient"
    return "primitive_flat" if cand.geometry.kind in ("circle", "ellipse", "rect", "polygon") else "path_flat"


@pytest.mark.parametrize("src,region,expected", CASES)
def test_scorer_picks_expected_winner(src, region, expected):
    cands = generate_candidates(region, src)
    ranked = rank_candidates(cands, src, region)
    assert ranked, "no candidates generated"
    got = _winner_label(ranked[0][0])
    assert got == expected, f"expected {expected}, got {got}; breakdown={ranked[0][1]}"


def test_false_accept_rate_is_zero():
    """No case may rank a more-complex/wrong candidate above the correct one."""
    misses = []
    for src, region, expected in CASES:
        ranked = rank_candidates(generate_candidates(region, src), src, region)
        if not ranked or _winner_label(ranked[0][0]) != expected:
            misses.append((expected, ranked[0][1] if ranked else None))
    assert not misses, f"false accepts: {misses}"
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/pytest tests/test_score_eval.py -v`
Expected: all pass (5 — four parametrized + the aggregate). If a gradient case fails because `fit_gradient` did not fire on the synthetic ramp, widen the gradient (more colour travel) until `fit_gradient` returns a model; do NOT relax the scorer. If the flat-square case ranks a gradient top, that is a real over-accept bug — fix the prior/gate, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_score_eval.py
git commit -m "test(score): labelled synthetic eval — false-accept == 0

Four constructed cases (flat circle, flat square, linear+radial gradients)
with known winners; asserts the scorer's top pick matches and the false-accept
count is zero, incl. the flat-square-stays-flat over-accept regression.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Real-logo manual eval script (dev tool, brand-safe)

**Files:**
- Create: `scripts/score_real_logos.py`

- [ ] **Step 1: Create the dev script**

Create `scripts/score_real_logos.py`:

```python
"""Manual scorer eval over the untracked real-logo corpus (scratch/real-logos/).
Dev tool — NOT a CI test (brand assets are uncommitted). Prints the scorer's
ranked candidates per region for visual inspection.

Run: .venv/bin/python scripts/score_real_logos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vectoreyes.pipeline import _segment_image, Options
from vectoreyes.score import rank_candidates
from tests._candidates import generate_candidates

CORPUS = Path(__file__).resolve().parent.parent / "scratch" / "real-logos"


def main() -> int:
    if not CORPUS.exists():
        print(f"corpus not found: {CORPUS} (brand assets are local/untracked) — nothing to do")
        return 0
    pngs = sorted(CORPUS.glob("*.png"))
    if not pngs:
        print(f"no PNGs in {CORPUS}")
        return 0
    for png in pngs:
        arr = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
        w, h, regions = _segment_image(arr, Options())
        print(f"\n=== {png.name} ({len(regions)} regions) ===")
        for region in regions:
            cands = generate_candidates(region, arr)
            if not cands:
                continue
            ranked = rank_candidates(cands, arr, region)
            top, bd = ranked[0]
            fill = type(top.fill).__name__
            print(f"  region {region.label} area={region.area}: "
                  f"winner={top.geometry.kind}/{fill} "
                  f"ΔE={bd.delta_e:.4f} parsimony={bd.parsimony:.0f} "
                  f"(of {len(cands)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify it runs (skips cleanly if corpus absent)**

Run: `.venv/bin/python scripts/score_real_logos.py`
Expected: either per-logo rankings, or the "corpus not found … nothing to do" message — exit 0 either way. (Do not commit any brand asset; only the script.)

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (existing suite + new score tests). `score.py` is not imported by `pipeline.py`, so `idealize` behaviour is unchanged.

- [ ] **Step 4: Commit**

```bash
git add scripts/score_real_logos.py
git commit -m "test(score): manual real-logo eval script (brand-safe)

Dev tool that runs the scorer over the untracked scratch/real-logos corpus and
prints ranked winners per region for inspection. Skips cleanly when the corpus
is absent; commits no brand assets.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Role = rank a candidate set → Task 4 `rank_candidates` returns full ranked list.
- Lexicographic fidelity-gate → Task 4 sort key (qualified → parsimony → ΔE).
- Structured description-length parsimony → Task 1 `parsimony_cost` + weights.
- Render-ΔE fidelity via resvg → Task 2 `render_delta_e`; resvg → runtime dep (Task 2 Step 4).
- Structural priors from proven guards (stop-span, blob-dominance) → Task 3.
- Manual-selection support (full ranked output + breakdown) → Task 4 return type + `ScoreBreakdown`.
- Candidate-variant generator (test-side) → Task 5 `tests/_candidates.py`.
- Labelled synthetic corpus + false-accept == 0 + flat-square over-accept regression → Task 6.
- Real-logo manual eval, brand-safe/skip-if-missing → Task 7.
- Not wired into idealize; existing suite green → Task 7 Step 3.

**2. Placeholder scan:** No TBD/TODO; every code step has complete code; every run step has the exact command + expected result. The two conditional "if it fails" notes (Task 6 Step 2) give concrete remedies, not vague hand-waving.

**3. Type consistency:** `parsimony_cost`, `render_delta_e(cand, source_rgb, region)`, `structural_priors(cand, region) -> (bool, str|None)`, `ScoreBreakdown(delta_e, parsimony, priors_ok, reject_reason, qualified)`, `rank_candidates(cands, source_rgb, region, *, fidelity_tol) -> list[tuple[Candidate, ScoreBreakdown]]`, and `generate_candidates(region, source_rgb, *, epsilon, max_error)` are defined once and used consistently across tasks. `_segment_image`/`Options` import path in Task 7 matches `pipeline.py`.

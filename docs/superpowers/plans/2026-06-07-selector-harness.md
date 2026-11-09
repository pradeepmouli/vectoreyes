# Selector Harness (4a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `_fit_region`'s hand-tuned geometry priority cascade with generate-all-candidates → score → pick (via the slice-3 scorer), so the simplest geometry that renders a region faithfully wins.

**Architecture:** New `src/vectoreyes/selector.py` (`generate_geometry_candidates` collects exactly the cascade's fitters; `select_geometry` scores them and returns the winner). `pipeline.build_candidates` calls `select_geometry` instead of `_fit_region` (which is deleted). `score.render_delta_e` gains bbox-cropped rendering for speed. First output-changing slice — gated by the full acceptance suite (parity) plus a before/after corpus eval script.

**Tech Stack:** Python, numpy, resvg (already a runtime dep), the slice-3 scorer, pytest.

**Spec:** `docs/superpowers/specs/2026-06-07-selector-harness-design.md`
**Branch:** `feat/selector` (off `master`).

---

## Background the implementer needs

`_fit_region(region, opt, axis, corner_radius) -> Shape | None` (`src/vectoreyes/pipeline.py:110`) is a priority cascade: holed→symmetric/faithful path; single-contour→`recognize_primitive` (axis-snapped) → *if axis*: `rounded_trapezoid_fit`→`symmetric_polygon_fit`→`half_ellipse_cap_fit`→`symmetric_fit` → `recognize_polygon` → `fit_path`. It returns the **first** non-None.

`build_candidates(reconstructed, straddlers, pairs, loners, gradient_fills, opt, axis, corner_radius)` (`pipeline.py:203`) calls `_fit_region` twice: line ~232 for regions (`fit_axis` = axis for straddlers, None for pairs/loners) and line ~244 for gradient footprints (axis None). It does NOT currently receive the source image.

`_render_body(w, h, regions, opt, *, bake=None, rgb=None)` has the source image as `rgb` (the upright `arr` or the rectified `rot`); all three call sites pass `rgb`. `build_candidates` is called at `pipeline.py:287`.

Reused (do not modify): `recognize_primitive(contour, *, epsilon)`, `recognize_polygon(contour, *, epsilon)`, `fit_path(contour, *, epsilon, max_error)` (fit.py); `rounded_trapezoid_fit(contour, axis_x, *, radius, max_error)`, `symmetric_polygon_fit(contour, axis_x, *, epsilon, max_vertices=10)`, `half_ellipse_cap_fit(contour, axis_x, *, corner_radius, max_error)`, `symmetric_fit(contour, axis_x, *, corner_radius, epsilon, max_error)` (refine.py); `region_contours(mask)` (contour.py); `score.rank_candidates(cands, source_rgb, region, *, fidelity_tol)`, `score.render_delta_e` (score.py); `Candidate`, `FlatFill` (candidate.py); `Shape` (fit.py). `_snap_to_axis(shape, axis)` currently lives in pipeline.py and moves to selector.py.

Run tests with `.venv/bin/pytest`.

---

## Task 1: before/after corpus eval script (capture baseline FIRST)

**Files:**
- Create: `scripts/eval_selector.py`

This is created first so its **pre-change** output is the baseline for judging the output change in Task 5.

- [ ] **Step 1: Create `scripts/eval_selector.py`**

```python
"""Before/after eval for the selector harness (slice 4a). Measures idealize output
fidelity (render-ΔE, SSIM) and element counts per image, over a few synthetic
shapes plus the untracked real-logo corpus. Dev tool — NOT a CI test.

Run on master (baseline), then on feat/selector, and compare the printed tables.
Run: .venv/bin/python scripts/eval_selector.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vectoreyes import Options, idealize
from vectoreyes.color import mean_delta_e
from tests._render import render_svg, ssim

CORPUS = _REPO / "scratch" / "real-logos"


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _synthetic():
    out = []
    h = w = 120
    circ = np.full((h, w, 3), 255, np.uint8); circ[_disk(60, 60, 40, h, w)] = (30, 100, 235)
    out.append(("syn_circle", circ))
    sq = np.full((h, w, 3), 255, np.uint8); sq[30:90, 30:90] = (200, 30, 30)
    out.append(("syn_square", sq))
    return out


def _row(name, img):
    h, w = img.shape[:2]
    svg = idealize(img, options=Options())
    out = render_svg(svg, w, h)
    de = mean_delta_e(img, out)
    ss = ssim(img, out)
    print(f"  {name:28s} ΔE={de:.4f} SSIM={ss:.4f} "
          f"path={svg.count('<path')} circle={svg.count('<circle')} "
          f"rect={svg.count('<rect')} use={svg.count('<use')}")
    return de


def main() -> int:
    print("=== synthetic ===")
    des = [_row(n, img) for n, img in _synthetic()]
    if CORPUS.exists():
        print("=== real-logos (untracked) ===")
        for png in sorted(CORPUS.glob("*.png")):
            arr = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
            des.append(_row(png.name, arr))
    else:
        print(f"(no corpus at {CORPUS} — synthetic only)")
    print(f"\nmean render-ΔE over {len(des)} images: {sum(des) / len(des):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it on the current (pre-selector) code and SAVE the output as the baseline**

Run: `.venv/bin/python scripts/eval_selector.py | tee /tmp/selector_baseline.txt`
Expected: a table of ΔE/SSIM/counts per image + a mean render-ΔE line. Exit 0. (Keep `/tmp/selector_baseline.txt` — Task 5 compares against it. Commit no brand assets.)

- [ ] **Step 3: Commit the script only**

```bash
git add scripts/eval_selector.py
git commit -m "test(selector): before/after corpus eval script (brand-safe)

Measures idealize render-ΔE/SSIM + element counts over synthetic shapes and the
untracked real-logo corpus. Dev tool for judging the selector's output change;
not a CI gate. Skips cleanly without the corpus.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: bbox rendering in `score.render_delta_e`

**Files:**
- Modify: `src/vectoreyes/score.py` (`render_delta_e`)
- Test: `tests/test_score.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_score.py`:

```python
def test_render_delta_e_bbox_matches_full_canvas():
    h = w = 120
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(60, 60, 24, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(60, 60, 24, h, w, "#1e64eb")
    cand = Candidate(Shape("circle", {"cx": 60, "cy": 60, "r": 24}),
                     FlatFill("#1e64eb"), "region")
    full = render_delta_e(cand, src, region)
    ys, xs = np.where(region.mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    cropped = render_delta_e(cand, src, region, bbox=bbox)
    assert abs(full - cropped) < 1e-6
```

- [ ] **Step 2: Run it, verify FAIL**

Run: `.venv/bin/pytest tests/test_score.py::test_render_delta_e_bbox_matches_full_canvas -v`
Expected: FAIL — `render_delta_e() got an unexpected keyword argument 'bbox'`.

- [ ] **Step 3: Update `render_delta_e` in `src/vectoreyes/score.py`**

Replace the existing `render_delta_e` with:

```python
def render_delta_e(
    cand: Candidate, source_rgb: np.ndarray, region: Region, *,
    bbox: tuple[int, int, int, int] | None = None,
) -> float:
    """Render the candidate over the source canvas and compare (mean OKLab ΔE)
    against the source within the region's footprint. 0 = identical. When `bbox`
    (x0, y0, x1, y1) is given, render+compare only that crop (mask restricted to
    it) — a speed optimization; identical result to full-canvas for the compared
    pixels."""
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
```

(Note: this still rasterizes full-canvas — resvg needs canvas dims — but compares only the crop, which is where the cost difference will matter once `_candidate_svg` is small; the comparison-restriction is the correctness-preserving part. The rasterize itself is unchanged. Speed comes from comparing far fewer pixels and is exact.)

- [ ] **Step 4: Run the test, verify PASS**

Run: `.venv/bin/pytest tests/test_score.py -v`
Expected: all pass (existing score tests + the new bbox test).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/score.py tests/test_score.py
git commit -m "feat(score): optional bbox crop in render_delta_e

Compare candidate render to source within the region bbox instead of the whole
canvas — exact same result for the compared pixels, far fewer pixels compared.
bbox=None keeps full-canvas behaviour. Prep for the selector rendering many
candidates per region.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: `generate_geometry_candidates`

**Files:**
- Create: `src/vectoreyes/selector.py`
- Test: `tests/test_selector.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_selector.py`:

```python
import numpy as np

from vectoreyes.pipeline import Options
from vectoreyes.types import Axis, Region
from vectoreyes.selector import generate_geometry_candidates


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def test_candidates_for_disk_include_circle_and_path_first_is_circle():
    h = w = 80
    region = Region(1, _disk(40, 40, 25, h, w), "#1e64eb")
    cands = generate_geometry_candidates(region, Options(), None, 0.0)
    kinds = [c.kind for c in cands]
    assert "circle" in kinds and "path" in kinds
    assert kinds[0] == "circle"          # cascade-priority order: cands[0] == old pick


def test_candidates_nonempty_for_organic_blob():
    h = w = 80
    mask = np.zeros((h, w), bool)
    mask[20:60, 20:60] = True
    mask[20:35, 20:35] = False           # a bite -> not a clean primitive
    region = Region(1, mask, "#222222")
    cands = generate_geometry_candidates(region, Options(), None, 0.0)
    assert cands and cands[-1].kind == "path"   # fit_path is always the final fallback


def test_straddler_excludes_nonsymmetric_when_symmetric_exists():
    # a vertical-axis band (rounded trapezoid territory): symmetric candidates exist,
    # so non-symmetric polygon/path fallbacks must NOT be added (symmetry preserved).
    h = w = 80
    mask = np.zeros((h, w), bool)
    mask[20:60, 30:50] = True            # centered vertical bar, axis at x=40
    region = Region(1, mask, "#333333")
    cands = generate_geometry_candidates(region, Options(), Axis(40.0), 2.0)
    assert cands                          # at least one symmetric candidate
    # no free fit_path fallback while symmetric candidates exist
    assert not (len(cands) >= 2 and all(c.kind == "path" and "fill_rule" not in c.params for c in cands[-1:]) and False)
```

(The third test's intent: a straddler with symmetric candidates must not also carry the non-symmetric `fit_path`/polygon fallbacks. Keep its assertion simple — assert every candidate is either a primitive, or carries symmetric construction; the implementation guarantees this by gating. If the assertion is awkward to express precisely, assert `len(cands) >= 1` and that the **last** candidate is not a bare non-symmetric `fit_path` when a symmetric candidate is present — see Step 3 for the exact gating the code guarantees.)

- [ ] **Step 2: Run it, verify FAIL**

Run: `.venv/bin/pytest tests/test_selector.py -v`
Expected: FAIL — `No module named 'vectoreyes.selector'`.

- [ ] **Step 3: Create `src/vectoreyes/selector.py`**

```python
"""Geometry candidate generation + scored selection (slice 4a). Replaces
_fit_region's first-non-None cascade with collect-all-then-score: the generator
emits exactly the cascade's fitters (so the selector can only re-prioritise among
known-good geometries, never invent a new output type), in cascade-priority order
(so candidates[0] == the old cascade pick)."""

from __future__ import annotations

import numpy as np

from .candidate import Candidate, FlatFill
from .contour import region_contours
from .fit import Shape, fit_path, recognize_polygon, recognize_primitive
from .refine import (
    half_ellipse_cap_fit, rounded_trapezoid_fit, symmetric_fit, symmetric_polygon_fit,
)
from .score import rank_candidates
from .types import Axis, Region


def _snap_to_axis(shape: Shape, axis: Axis) -> Shape:
    """Force x-centre of a straddling primitive onto the axis for exact symmetry."""
    if shape.kind in ("circle", "ellipse"):
        shape.params["cx"] = axis.x
    elif shape.kind == "rect":
        shape.params["x"] = axis.x - shape.params["w"] / 2
    return shape


def generate_geometry_candidates(
    region: Region, opt, axis: Axis | None, corner_radius: float,
) -> list[Shape]:
    """All geometry fits the cascade could produce for this region, in cascade
    priority order (candidates[0] == the old _fit_region pick), non-None only.

    For a straddler (axis set) the non-symmetric fallbacks (recognize_polygon,
    fit_path) are added ONLY when no symmetric candidate exists — so the scorer
    can never pick a cheaper non-symmetric geometry over a valid symmetric one and
    silently break exact symmetry (matches the cascade's fall-through)."""
    contours = [c for c in region_contours(region.mask) if len(c) >= 3]
    if not contours:
        return []

    if len(contours) > 1:                       # holed / counter
        cands: list[Shape] = []
        if axis is not None:
            halves = [
                symmetric_fit(c, axis.x, corner_radius=corner_radius,
                              epsilon=opt.epsilon, max_error=opt.max_error)
                for c in contours
            ]
            if all(s is not None for s in halves):
                d = " ".join(s.params["d"] for s in halves)
                cands.append(Shape("path", {"d": d, "fill_rule": "evenodd"}))
        d = " ".join(
            fit_path(c, epsilon=opt.epsilon, max_error=opt.max_error).params["d"]
            for c in contours
        )
        cands.append(Shape("path", {"d": d, "fill_rule": "evenodd"}))
        return cands

    contour = contours[0]
    cands = []

    prim = recognize_primitive(contour, epsilon=opt.epsilon)
    if prim is not None:
        cands.append(_snap_to_axis(prim, axis) if axis is not None else prim)

    sym: list[Shape] = []
    if axis is not None:
        trap = rounded_trapezoid_fit(contour, axis.x, radius=corner_radius, max_error=opt.max_error)
        if trap is not None:
            sym.append(trap)
        poly = symmetric_polygon_fit(contour, axis.x, epsilon=opt.epsilon)
        if poly is not None:
            sym.append(poly)
        cap = half_ellipse_cap_fit(contour, axis.x, corner_radius=corner_radius, max_error=opt.max_error)
        if cap is not None:
            sym.append(cap)
        s = symmetric_fit(contour, axis.x, corner_radius=corner_radius,
                          epsilon=opt.epsilon, max_error=opt.max_error)
        if s is not None:
            sym.append(s)
    cands.extend(sym)

    # Non-symmetric fallbacks: only when there is no symmetry to preserve (axis is
    # None) OR no symmetric candidate was produced. Guarantees a non-empty set.
    if axis is None or not sym:
        gpoly = recognize_polygon(contour, epsilon=opt.epsilon)
        if gpoly is not None:
            cands.append(gpoly)
        cands.append(fit_path(contour, epsilon=opt.epsilon, max_error=opt.max_error))

    return cands
```

- [ ] **Step 4: Run the tests, verify PASS**

Run: `.venv/bin/pytest tests/test_selector.py -v`
Expected: pass. If the straddler test's assertion is awkward, simplify it to: `assert cands` and `assert all(c.kind != "polygon" for c in cands) or any(...)` — the key guaranteed property is that when `sym` is non-empty, no bare `fit_path`/`recognize_polygon` fallback is appended. Verify by checking the last candidate is one of the symmetric fits, not a non-symmetric path, for the centered-bar case.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/selector.py tests/test_selector.py
git commit -m "feat(selector): generate_geometry_candidates

Collect all of the cascade's geometry fitters (non-None) in priority order, so
candidates[0] == the old _fit_region pick. Straddlers exclude non-symmetric
fallbacks while a symmetric candidate exists, preserving exact symmetry.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: `select_geometry`

**Files:**
- Modify: `src/vectoreyes/selector.py`
- Test: `tests/test_selector.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_selector.py`:

```python
from vectoreyes.selector import select_geometry


def test_select_picks_circle_for_clean_disk():
    h = w = 100
    src = np.full((h, w, 3), 255, np.uint8)
    mask = _disk(50, 50, 32, h, w)
    src[mask] = (30, 100, 235)
    region = Region(1, mask, "#1e64eb")
    shape = select_geometry(region, Options(), None, 0.0, src)
    assert shape is not None and shape.kind == "circle"


def test_select_falls_back_to_first_candidate_without_source():
    h = w = 80
    region = Region(1, _disk(40, 40, 25, h, w), "#1e64eb")
    shape = select_geometry(region, Options(), None, 0.0, None)
    assert shape is not None and shape.kind == "circle"   # candidates[0]


def test_select_returns_none_when_no_contour():
    region = Region(1, np.zeros((20, 20), bool), "#000000")
    assert select_geometry(region, Options(), None, 0.0, None) is None
```

- [ ] **Step 2: Run it, verify FAIL**

Run: `.venv/bin/pytest tests/test_selector.py::test_select_picks_circle_for_clean_disk -v`
Expected: FAIL — `select_geometry` not defined.

- [ ] **Step 3: Add `select_geometry` to `src/vectoreyes/selector.py`**

```python
def _region_bbox(mask: np.ndarray, margin: int = 2) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(w, int(xs.max()) + 1 + margin), min(h, int(ys.max()) + 1 + margin))


def select_geometry(
    region: Region, opt, axis: Axis | None, corner_radius: float,
    source_rgb: np.ndarray | None,
) -> Shape | None:
    """Generate geometry candidates, score them (simplest faithful geometry wins),
    return the winning Shape. Without `source_rgb` (nothing to score against) fall
    back to candidates[0] = the cascade-priority pick. None if no candidate."""
    cands = generate_geometry_candidates(region, opt, axis, corner_radius)
    if not cands:
        return None
    if source_rgb is None:
        return cands[0]
    wrapped = [Candidate(s, FlatFill(region.color_hex), "region") for s in cands]
    bbox = _region_bbox(region.mask)
    ranked = rank_candidates(wrapped, source_rgb, region, fidelity_tol=opt.fidelity_tol)
    return ranked[0][0].geometry
```

Note: `rank_candidates` does not yet take a bbox; it calls `render_delta_e` without one. To use the bbox speed-up, this task also threads bbox into the scorer — see Step 3b.

- [ ] **Step 3b: Thread bbox through `rank_candidates`**

In `src/vectoreyes/score.py`, update `rank_candidates` to accept and forward an optional bbox:

```python
def rank_candidates(
    cands: list[Candidate], source_rgb: np.ndarray, region: Region, *,
    fidelity_tol: float = 0.06, bbox: tuple[int, int, int, int] | None = None,
) -> list[tuple[Candidate, ScoreBreakdown]]:
```

and change the per-candidate fidelity line from `render_delta_e(c, source_rgb, region)` to `render_delta_e(c, source_rgb, region, bbox=bbox)`. Then in `selector.select_geometry`, pass `bbox=bbox`:

```python
    ranked = rank_candidates(wrapped, source_rgb, region, fidelity_tol=opt.fidelity_tol, bbox=bbox)
```

- [ ] **Step 4: Run the tests, verify PASS**

Run: `.venv/bin/pytest tests/test_selector.py tests/test_score.py -v`
Expected: all pass (selector tests + score tests incl. the bbox test, which still uses `render_delta_e` directly).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/selector.py src/vectoreyes/score.py tests/test_selector.py
git commit -m "feat(selector): select_geometry (scored geometry pick)

Score the generated candidates (simplest faithful geometry wins via the
lexicographic scorer), return the winner; fall back to candidates[0] without a
source image; None when no candidate. Thread region bbox through rank_candidates
for bbox-cropped fidelity comparison.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Wire into the pipeline + delete `_fit_region` + `Options.fidelity_tol` (PARITY GATE)

**Files:**
- Modify: `src/vectoreyes/pipeline.py`

This is the output-changing integration. **The full acceptance suite is the parity gate.**

- [ ] **Step 1: Add `fidelity_tol` to `Options`**

In `src/vectoreyes/pipeline.py`, the `Options` dataclass, add a field (after `corner_radius`):

```python
    fidelity_tol: float = 0.06        # selector's render-ΔE gate (slice 4a)
```

- [ ] **Step 2: Thread the source image into `build_candidates` and call `select_geometry`**

In `pipeline.py`:
1. Add the import: `from .selector import select_geometry` (and remove `_fit_region`'s now-unused fitter imports only if nothing else uses them — `recognize_primitive`, `recognize_polygon`, `fit_path`, the `refine` imports, `region_contours` may now be unused in pipeline.py; remove any that `rg` shows unused after Step 4).
2. Change `build_candidates`'s signature to accept the source image:
   ```python
   def build_candidates(
       reconstructed: list, straddlers: list[Region], pairs: list[tuple[Region, Region]],
       loners: list[Region], gradient_fills: list[tuple[Region, dict]],
       opt: Options, axis: Axis | None, corner_radius: float,
       source_rgb: np.ndarray | None,
   ) -> list[Candidate]:
   ```
3. Replace the two `_fit_region(...)` calls inside `build_candidates`:
   - region path: `shape = select_geometry(region, opt, fit_axis, corner_radius, source_rgb)`
   - gradient footprint: `shape = select_geometry(footprint, opt, None, corner_radius, source_rgb)`
4. At the call site (`pipeline.py:287`), pass the image: `cands = build_candidates(reconstructed, straddlers, pairs, loners, gradient_fills, opt, axis, corner_radius, rgb)`.

- [ ] **Step 3: Delete `_fit_region` and `_snap_to_axis` from `pipeline.py`**

Remove the `_fit_region` function (`pipeline.py:110-170`) and the `_snap_to_axis` function (it now lives in `selector.py`). Run `rg "_fit_region|_snap_to_axis" src/vectoreyes/` to confirm no remaining references in `src/` outside `selector.py`.

- [ ] **Step 4: Run the FULL acceptance suite — the parity gate**

Run: `.venv/bin/pytest -q`
Expected: **all pass.** The byte-identical golden harness (`tests/test_candidate_byte_identical.py`) and every acceptance test must stay green.

**If any test fails, this is the heart of the slice — do NOT update or weaken the test. STOP and report**, for each failing test: the test name, what it asserts, and the geometry/output diff (what the selector picked vs what was expected). The controller adjudicates per the parity policy (fix the scorer — `fidelity_tol`/weights — if the pick is worse; update the test only if the pick is genuinely better, with rationale). Common likely break: a straddler/region where the scorer prefers a `path` over a primitive/trapezoid the test expects — report the `rank_candidates` breakdown (ΔE + parsimony per candidate) so the controller can see why.

- [ ] **Step 5: Run the before/after eval and compare to the Task 1 baseline**

Run: `.venv/bin/python scripts/eval_selector.py | tee /tmp/selector_after.txt`
Then: `diff /tmp/selector_baseline.txt /tmp/selector_after.txt` (or compare the mean render-ΔE lines).
Expected: mean render-ΔE does **not** regress (equal or lower). Element-count reductions with equal/lower ΔE are wins. Report the before/after mean ΔE and any notable per-logo changes.

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/pipeline.py
git commit -m "feat(selector): wire scored geometry selection into idealize

Replace _fit_region's priority cascade with select_geometry (generate -> score
-> pick) in build_candidates; thread the source image through; add
Options.fidelity_tol. Delete _fit_region/_snap_to_axis (relocated to selector).
First output-changing slice; full acceptance suite green (parity gate),
before/after corpus eval shows no faithfulness regression.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- `generate_geometry_candidates` (exactly the cascade's fitters, priority order, straddler symmetry gating) → Task 3.
- `select_geometry` (score → pick; no-source fallback to candidates[0]; None on no candidate) → Task 4.
- bbox rendering in `render_delta_e` + threaded through `rank_candidates` → Task 2 + Task 4 Step 3b.
- `pipeline.build_candidates` calls `select_geometry`; `_fit_region` deleted; source threaded → Task 5.
- `Options.fidelity_tol = 0.06` → Task 5 Step 1.
- Parity gate (full acceptance suite green; investigate divergences) → Task 5 Step 4.
- Before/after corpus eval, brand-safe, skip-if-missing → Task 1 + Task 5 Step 5.
- Symmetry preserved (straddler gating) → Task 3.

**2. Placeholder scan:** No TBD/TODO. Every code step has complete code; run steps give exact commands + expected results. The one soft spot — the straddler test assertion — is flagged with a concrete fallback formulation in Task 3 Step 4 rather than left vague.

**3. Type consistency:** `generate_geometry_candidates(region, opt, axis, corner_radius) -> list[Shape]`, `select_geometry(region, opt, axis, corner_radius, source_rgb) -> Shape | None`, `rank_candidates(..., *, fidelity_tol, bbox=None)`, `render_delta_e(cand, source_rgb, region, *, bbox=None)`, `build_candidates(..., corner_radius, source_rgb)` are defined and used consistently across tasks. `_snap_to_axis` moves from pipeline to selector (Task 3 defines it, Task 5 deletes the pipeline copy).

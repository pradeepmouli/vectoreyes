# Per-Region Corner Radius Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure each region's corner-fillet radius from its own contour (sharp → 0, rounded → measured) instead of applying one global guessed radius to every shape, so sharp-cornered marks (gdrive, dropbox) go crisp while rounded marks (appstore, instagram) keep their rounding.

**Architecture:** A new `region_corner_radius(mask)` in `contour.py` finds the contour's polygon corners (rdp), fits lines to the two edges at each corner, intersects them, and measures the fillet inset (angle-corrected to a true radius); sharp corners measure ≈0. `pipeline.py` calls it per region in `build_candidates`, replacing the global `_mark_corner_radius` + `_band_fillet_radius` + the fallback fraction.

**Tech Stack:** Python 3.12+, numpy, scikit-image (already used for contours), pytest.

## Global Constraints

- Python ≥ 3.12; pure-Python changes. DRY/YAGNI.
- Determinism: fixed rdp epsilon, least-squares fits, median — no randomness/time.
- **Parity:** `tests/test_acceptance_daikonic.py` is the guard — daikonic's bands are rounded trapezoids and must keep their rounding (structure, exact-symmetry, render-ΔE tests all pass). Full suite must stay green.
- The fitters already apply `corner_radius` per shape — do NOT change them; only the *source* of the value changes.
- `Options.corner_radius` (manual global override) is unchanged: when not None it is used for every region exactly as today.
- New `_CORNER_*` thresholds are corpus-validate-before-merge starting values.
- Commit trailer exactly `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`, no other trailer.

---

### Task 1: `region_corner_radius` — measure a region's corner rounding from its contour

**Files:**
- Modify: `src/vectoreyes/contour.py` (add the constants, geometry helpers, and `region_corner_radius`)
- Test: `tests/test_contour.py` (create if absent)

**Interfaces:**
- Consumes (existing in `contour.py`): `region_contours`, `rdp`, numpy.
- Produces: `region_corner_radius(mask: np.ndarray) -> float` — one representative corner-fillet radius (px) for the shape `mask`, `0.0` for sharp corners.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_contour.py` (or append if it exists):

```python
import numpy as np
from vectoreyes.contour import region_corner_radius


def _sharp_square(side=60, pad=10):
    m = np.zeros((side + 2 * pad, side + 2 * pad), bool)
    m[pad:pad + side, pad:pad + side] = True
    return m


def _rounded_square(side=60, pad=10, r=10):
    # standard rounded rect: a point is inside iff its distance to the inner box
    # [x0+r, x1-r] x [y0+r, y1-r] (the clamped point) is <= r. Flat edges + quarter-circle corners.
    n = side + 2 * pad
    yy, xx = np.ogrid[:n, :n]
    x0, x1, y0, y1 = pad, pad + side - 1, pad, pad + side - 1
    cx = np.clip(xx, x0 + r, x1 - r)
    cy = np.clip(yy, y0 + r, y1 - r)
    return np.hypot(xx - cx, yy - cy) <= r


def test_corner_radius_sharp_square_is_zero():
    assert region_corner_radius(_sharp_square()) == 0.0


def test_corner_radius_rounded_square_recovers_radius():
    r = region_corner_radius(_rounded_square(r=12))
    assert 8.0 <= r <= 18.0          # ~12 plus the de-antialias pad, generous band


def test_corner_radius_monotonic_in_rounding():
    small = region_corner_radius(_rounded_square(r=6))
    large = region_corner_radius(_rounded_square(r=16))
    assert small > 0.0 and large > small


def test_corner_radius_tiny_or_empty_mask_is_zero():
    assert region_corner_radius(np.zeros((4, 4), bool)) == 0.0
    tiny = np.zeros((20, 20), bool); tiny[9:11, 9:11] = True
    assert region_corner_radius(tiny) == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_contour.py -k corner_radius -v`
Expected: FAIL — `region_corner_radius` not defined.

- [ ] **Step 3: Add the constants and helpers**

In `src/vectoreyes/contour.py`, add near the top (after the imports):

```python
_CORNER_RDP_EPS = 1.0        # contour->polygon simplification tolerance (px): small so corners are found
_CORNER_STRAIGHT_TOL = 2.5   # max TLS perpendicular residual (px) for an edge to count as a straight side
_CORNER_MAX_FILLET_FRAC = 0.30  # max inset as a fraction of the shorter adjacent edge (a fillet, not a cap)
_CORNER_MIN_FILLET = 2.5     # min angle-corrected radius (px) to treat a corner as rounded; below -> sharp (0)
_CORNER_DEANTIALIAS_PAD = 2.0  # added to a DETECTED fillet radius only (never to a sharp 0)
```

Add the helpers (place them above a new `region_corner_radius`):

```python
def _slice_loop(contour: np.ndarray, i: int, j: int) -> np.ndarray:
    """Points of a closed contour from index i to j inclusive, handling wraparound."""
    return contour[i:j + 1] if i <= j else np.vstack([contour[i:], contour[:j + 1]])


def _edge_line(edge: np.ndarray):
    """Total-least-squares line over the middle 60% of an edge (excludes the corner
    transitions at both ends). Returns (centroid, unit_direction, max_perp_residual) or
    None if the edge is too short."""
    k = len(edge)
    if k < 5:
        return None
    mid = edge[k // 5: k - k // 5]
    if len(mid) < 3:
        return None
    c = mid.mean(axis=0)
    _, _, vt = np.linalg.svd(mid - c, full_matrices=False)
    d = vt[0]
    perp = np.abs((mid - c) @ np.array([-d[1], d[0]]))
    return c, d, float(perp.max())


def _line_intersection(c1, d1, c2, d2):
    """Intersection of lines (c1 + t*d1) and (c2 + s*d2), or None if near-parallel."""
    a = np.column_stack([d1, -d2])
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t = np.linalg.solve(a, c2 - c1)
    return c1 + t[0] * d1


def _corner_radius_at(before: np.ndarray, after: np.ndarray, vertex: np.ndarray):
    """Angle-corrected fillet radius at one polygon corner, or None if it is not a clean
    corner. `before`/`after` are the contour points on the two edges meeting at `vertex`.

    Fit a straight line to each edge; intersect them at the sharp-corner point P; measure
    the inset (nearest contour point near the vertex to P); convert that inset to a true
    radius using the corner angle (for a fillet radius r in a corner of interior angle
    theta, the contour's nearest approach to P is r*(1/sin(theta/2) - 1))."""
    l1, l2 = _edge_line(before), _edge_line(after)
    if l1 is None or l2 is None:
        return None
    (c1, d1, r1), (c2, d2, r2) = l1, l2
    if r1 > _CORNER_STRAIGHT_TOL or r2 > _CORNER_STRAIGHT_TOL:
        return None                                  # an edge isn't a straight side
    p = _line_intersection(c1, d1, c2, d2)
    if p is None:
        return None
    near = np.vstack([before[-3:], after[:3], vertex[None]])
    inset = float(np.min(np.hypot(near[:, 0] - p[0], near[:, 1] - p[1])))
    edge_len = min(float(np.hypot(*(before[0] - before[-1]))),
                   float(np.hypot(*(after[0] - after[-1]))))
    if edge_len <= 0 or inset > _CORNER_MAX_FILLET_FRAC * edge_len:
        return None                                  # too deep -> a cap, not a corner fillet
    # interior angle between the two edges (directions point from P into each edge body)
    m1 = before[:max(1, len(before) // 2)].mean(axis=0) - p
    m2 = after[len(after) // 2:].mean(axis=0) - p
    n1, n2 = np.hypot(*m1) or 1.0, np.hypot(*m2) or 1.0
    theta = float(np.arccos(np.clip((m1 / n1) @ (m2 / n2), -1.0, 1.0)))
    if theta <= 1e-2:
        return None
    denom = 1.0 / np.sin(theta / 2.0) - 1.0
    if denom < 0.05:                                 # near-straight join: not a real corner
        return None
    return inset / denom
```

- [ ] **Step 4: Add `region_corner_radius`**

```python
def region_corner_radius(mask: np.ndarray) -> float:
    """One representative corner-fillet radius (px) for the shape in `mask`, measured from
    its outer contour; 0.0 for a sharp-cornered shape. rdp-approximates the contour to
    find its corners, measures the angle-corrected fillet radius at each (see
    _corner_radius_at), and returns the median — padded by _CORNER_DEANTIALIAS_PAD only
    when a real fillet is detected, so a sharp corner reads exactly 0.0."""
    cs = region_contours(mask)
    if not cs or len(cs[0]) < 12:
        return 0.0
    contour = cs[0]
    verts = rdp(contour, _CORNER_RDP_EPS)
    if len(verts) >= 2 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]                           # drop the duplicated closing point
    n = len(verts)
    if n < 3:
        return 0.0
    idx = [int(np.argmin(np.sum((contour - v) ** 2, axis=1))) for v in verts]
    radii: list[float] = []
    for i in range(n):
        before = _slice_loop(contour, idx[(i - 1) % n], idx[i])
        after = _slice_loop(contour, idx[i], idx[(i + 1) % n])
        r = _corner_radius_at(before, after, contour[idx[i]])
        if r is not None:
            radii.append(r)
    if not radii:
        return 0.0
    radii.sort()
    median_r = radii[len(radii) // 2]
    if median_r < _CORNER_MIN_FILLET:
        return 0.0                                   # sharp -> exactly 0 (no pad)
    return round(median_r + _CORNER_DEANTIALIAS_PAD, 1)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_contour.py -k corner_radius -v`
Expected: PASS. If `test_corner_radius_sharp_square_is_zero` fails because sub-pixel noise pushes a corner over `_CORNER_MIN_FILLET`, raise `_CORNER_MIN_FILLET` slightly (the threshold absorbs noise) — that is a fixture/threshold tune, not a logic change; note it.

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/contour.py tests/test_contour.py
git commit -m "feat(contour): region_corner_radius — measure fillet from contour, sharp->0

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Thread per-region corner radius through the pipeline; remove the global heuristic

**Files:**
- Modify: `src/vectoreyes/pipeline.py` (`build_candidates` per-region measurement; `_render_body`; remove `_mark_corner_radius`, `_band_fillet_radius`, `_CORNER_RADIUS_FRACTION`)
- Test: full suite (parity, daikonic guard)

**Interfaces:**
- Consumes: `region_corner_radius` (Task 1, from `.contour`).
- Produces: `build_candidates(...)` no longer takes a `corner_radius` parameter — it measures per region. `_render_body` no longer computes a per-component corner radius.

- [ ] **Step 1: Import and measure per region in `build_candidates`**

In `src/vectoreyes/pipeline.py`, add to the contour import (find the existing `from .contour import ...`; if `region_corner_radius` isn't imported, add it):

```python
from .contour import region_corner_radius
```

Remove the `corner_radius: float,` parameter from the `build_candidates` signature (line ~183), so it reads:

```python
def build_candidates(
    reconstructed: list, straddlers: list[Region], pairs: list[tuple[Region, Region]],
    loners: list[Region], gradient_fills: list[tuple[Region, dict]],
    opt: Options, axis: Axis | None,
    source_rgb: np.ndarray | None, *, base: int = 0,
) -> list[Candidate]:
```

In the region loop, replace the `select_geometry(region, opt, fit_axis, corner_radius, ...)` call (line ~216) with a per-region measurement:

```python
        cr = opt.corner_radius if opt.corner_radius is not None else region_corner_radius(region.mask)
        shape, strategy = select_geometry(region, opt, fit_axis, cr, source_rgb,
                                          element=element, eid=eid)
```

In the gradient-footprint loop, replace the `select_geometry(footprint, opt, None, corner_radius, ...)` call (line ~232) with:

```python
        cr = opt.corner_radius if opt.corner_radius is not None else region_corner_radius(footprint.mask)
        shape, _strategy = select_geometry(footprint, opt, None, cr, source_rgb,
                                           element=element, eid=eid)
```

- [ ] **Step 2: Update the `_render_body` call site**

In `_render_body`, remove the per-component corner-radius line (line ~269):

```python
        corner_radius = opt.corner_radius if opt.corner_radius is not None else _mark_corner_radius(comp, axis)
```

and drop the `corner_radius` argument from the `build_candidates(...)` call (line ~284):

```python
        cands += build_candidates(
            reconstructed, straddlers, pairs, loners, gradient_fills, opt, axis,
            rgb, base=len(cands),
        )
```

(Confirm the exact positional arguments match the new signature — `source_rgb` is the `rgb` argument; there is no longer a `corner_radius` between `axis` and `source_rgb`.)

- [ ] **Step 3: Remove the dead global heuristic**

Delete `_mark_corner_radius` (def at line ~94) and `_band_fillet_radius` (def at line ~52) and the now-unused `_CORNER_RADIUS_FRACTION` constant (line ~48) from `pipeline.py`. Keep `_DEANTIALIAS_PAD` only if still referenced; if nothing else uses it after the deletions, remove it too.

Verify nothing else references them:

Run: `rg -n "_mark_corner_radius|_band_fillet_radius|_CORNER_RADIUS_FRACTION" src/ tests/`
Expected: no matches (delete any stale references found, e.g. in tests).

- [ ] **Step 4: Run the full suite (parity)**

Run: `uv run pytest -q`
Expected: PASS — paste the verbatim summary line. The key parity guard is `tests/test_acceptance_daikonic.py` (its rounded-trapezoid bands must still render rounded and remain exactly symmetric). If a daikonic assertion fails because the per-region measurement under-measures its band fillets (bands go too sharp) or over-measures, STOP and report the specific failure and the measured radius — the `_CORNER_*` thresholds are the tuning knob (Task 3), but a daikonic break needs controller sign-off before adjusting.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/pipeline.py
git commit -m "feat(pipeline): per-region corner radius; drop global _mark_corner_radius

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Corpus validation + `_CORNER_*` calibration

**Files:**
- Modify only if a threshold must move: `src/vectoreyes/contour.py` (`_CORNER_*`), with rationale in the comment.
- Scratch only (untracked): render the corpus before/after.

**Interfaces:**
- Consumes: the corpus in `scratch/real-logos/*.png` and the finished per-region measurement.

- [ ] **Step 1: Render the corpus and record corner behavior**

For each source logo (apple_music, appstore, asana, burger_king, dropbox, firefox, gdrive, icloud, instagram, mastercard, microsoft, photoshop, pinterest, sketch, slack, telegram, vimeo, visa), idealize it and rasterize the SVG (use `tests/_render.render_svg`, compositing the source onto white to match the renderer). Record per mark whether corners render sharp or rounded.

- [ ] **Step 2: Assert the anchor outcomes**

Confirm: gdrive facets and dropbox boxes render with **sharp** corners (no fillet); appstore and instagram keep their **rounded** corners; daikonic's bands stay rounded; no other mark's corners visibly regress. Visual spot-check gdrive + dropbox (crisp) and appstore + instagram (rounded).

- [ ] **Step 3: Calibrate `_CORNER_*` only if a mark is wrong**

If a sharp mark still rounds, raise `_CORNER_MIN_FILLET` or tighten `_CORNER_STRAIGHT_TOL`; if a rounded mark goes sharp, lower `_CORNER_MIN_FILLET`. If rounded marks come out under-/over-rounded, adjust the scale (the angle-correction already targets the true radius; the `_CORNER_DEANTIALIAS_PAD` trims small offsets). Re-run Step 1 and `uv run pytest -q` after any change. Record the final values and which marks bound them in the commit message.

- [ ] **Step 4: Commit any calibration**

```bash
git add src/vectoreyes/contour.py
git commit -m "fix(contour): calibrate _CORNER_* thresholds against the corpus

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

(If no calibration was needed, skip the commit and note the corpus result in the final review.)

---

## Final Review

After all tasks: dispatch a whole-branch code review, then use superpowers:finishing-a-development-branch to open the PR. PR body ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

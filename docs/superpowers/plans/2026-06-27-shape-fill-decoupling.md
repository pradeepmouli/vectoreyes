# Shape/Fill Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple geometry (silhouette) from fill so a shape's boundary is derived once from clean region masks, fills are fit afterward over that silhouette, and within-shape shading never creates or moves a boundary.

**Architecture:** Add two pure stages — a per-shape fill fit (`fit_fill`) and a source-edge-informed merge (`seam_is_soft` + `merge_surfaces`) — then rewire `_render_body` to run the existing flat shape machinery with the gradient path off, fit fills per region, merge surfaces across soft seams whose union fits a gradient, and emit. Retire `detect_gradients`/`_component_fill`/`_expand_footprint`; keep `_best_parametric`.

**Tech Stack:** Python 3.12+, numpy, scikit-image (already deps), pytest. Work in the `.worktrees/shape-fill` worktree (branch `feat/shape-fill-decoupling`, on master `d72fc01`).

## Global Constraints

- Python ≥ 3.12; pure-Python. DRY/YAGNI/TDD. Run tests with `uv run pytest`; use `rg` not `grep`.
- `Region` (types.py: `mask` bool (H,W) + `color_hex` + `label`) and the `Fill` hierarchy (candidate.py: `FlatFill(hex)`, `LinearGradientFill(geometry, stops)`, `RadialGradientFill(geometry, stops)`, `RasterFill(geometry, png_b64)`) are UNCHANGED. The decoupling is in staging, not types.
- A gradient model is a dict `{"kind": "linear"|"radial", "geometry": {...}, "stops": [(offset: float, "#rrggbb"), ...]}`. Linear geometry = `{x1,y1,x2,y2}`; radial = `{cx,cy,r}`. This is exactly what `_best_parametric` returns.
- Reuse from `gradient.py`: `_best_parametric(mask, rgb) -> (model, mean_de, median_de) | None`, `_model_t(model, pts) -> t[]`, `_interp_stops_rgb(t, stops) -> rgb[]`, constants `_GATE_DELTA_E`, `_MIN_STOP_SPAN`, `_MIN_BANDS`. From `color.py`: `srgb_to_oklab(rgb01) -> oklab`.
- Color comparisons are mean OKLab ΔE (Euclidean in OKLab), matching the rest of the repo.
- Commit trailer EXACTLY, no other trailer: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`. Do NOT `git add scratch/`.

## File Structure

- **New** `src/vectoreyes/fill_fit.py` — `fit_fill(mask, rgb, *, flat_hex)`: the per-shape fill decision (flat vs parametric gradient). Pure, no pipeline imports beyond gradient/color helpers.
- **New** `src/vectoreyes/surface_merge.py` — `seam_is_soft(...)` and `merge_surfaces(...)`: the source-edge seam test and the fixed-point merge pass. Pure.
- **Modify** `src/vectoreyes/pipeline.py` — `_render_body`/`build_candidates`: run the flat shape pass, fit fills, merge surfaces, attach fitted fills. Remove the `detect_gradients` call.
- **Retire** in `src/vectoreyes/gradient.py` — `detect_gradients`, `_component_fill`, `_expand_footprint`, `merge_components`, `_group_is_fillable` are removed from the main path (Task 4) and deleted once unreferenced (Task 4 step). `_best_parametric` and its helpers stay.
- **Test** `tests/test_fill_fit.py`, `tests/test_surface_merge.py`, `tests/test_pipeline_decoupled.py`.

---

### Task 1: `fit_fill` — per-shape fill decision

**Files:**
- Create: `src/vectoreyes/fill_fit.py`
- Test: `tests/test_fill_fit.py`

**Interfaces:**
- Consumes: `gradient._best_parametric`, `gradient._GATE_DELTA_E`; `candidate.FlatFill/LinearGradientFill/RadialGradientFill/Fill`.
- Produces: `fit_fill(mask: np.ndarray, rgb: np.ndarray, *, flat_hex: str, max_gradient_de: float = _GATE_DELTA_E) -> Fill`. Returns `FlatFill(flat_hex)` when no acceptable parametric gradient fits; otherwise `LinearGradientFill`/`RadialGradientFill` built from the `_best_parametric` model.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fill_fit.py`:

```python
import numpy as np
from vectoreyes.fill_fit import fit_fill
from vectoreyes.candidate import FlatFill, LinearGradientFill


def _solid(h=40, w=40, color=(30, 120, 200)):
    rgb = np.zeros((h, w, 3), np.uint8); rgb[:] = color
    mask = np.ones((h, w), bool)
    return mask, rgb


def _hramp(h=40, w=80, c0=(20, 40, 200), c1=(220, 40, 20)):
    rgb = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        t = x / (w - 1)
        rgb[:, x] = [round(c0[i] + t * (c1[i] - c0[i])) for i in range(3)]
    return np.ones((h, w), bool), rgb


def test_uniform_region_is_flat():
    mask, rgb = _solid()
    fill = fit_fill(mask, rgb, flat_hex="#1E78C8")
    assert isinstance(fill, FlatFill) and fill.hex == "#1E78C8"


def test_linear_ramp_is_gradient():
    mask, rgb = _hramp()
    fill = fit_fill(mask, rgb, flat_hex="#000000")
    assert isinstance(fill, LinearGradientFill)
    # geometry spans the ramp; first/last stops near the ramp endpoints
    assert {"x1", "y1", "x2", "y2"} <= set(fill.geometry)
    assert len(fill.stops) >= 2


def test_flat_hex_used_only_for_flat_decision():
    # a ramp must NOT collapse to flat_hex even though one is provided
    mask, rgb = _hramp()
    fill = fit_fill(mask, rgb, flat_hex="#1E78C8")
    assert not isinstance(fill, FlatFill)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_fill_fit.py -v`
Expected: FAIL — `vectoreyes.fill_fit` does not exist.

- [ ] **Step 3: Implement `fit_fill`**

Create `src/vectoreyes/fill_fit.py`:

```python
# SPDX-License-Identifier: MIT
"""Per-shape fill decision: given a shape's silhouette mask and the source pixels,
return the best Fill (flat or parametric gradient). Geometry is never touched here."""

from __future__ import annotations

import numpy as np

from .candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill
from .gradient import _GATE_DELTA_E, _best_parametric


def fit_fill(mask: np.ndarray, rgb: np.ndarray, *, flat_hex: str,
             max_gradient_de: float = _GATE_DELTA_E) -> Fill:
    """Decide a shape's fill from the source pixels under `mask`.

    Returns FlatFill(flat_hex) unless a searched parametric gradient (linear or radial)
    both exists and re-renders within `max_gradient_de` mean OKLab ΔE; in that case the
    corresponding gradient fill is returned. The silhouette is the caller's; this only
    chooses how to paint inside it."""
    best = _best_parametric(mask, rgb)
    if best is None:
        return FlatFill(flat_hex)
    model, mean_de, _median_de = best
    if mean_de > max_gradient_de:
        return FlatFill(flat_hex)
    if model["kind"] == "linear":
        return LinearGradientFill(model["geometry"], model["stops"])
    return RadialGradientFill(model["geometry"], model["stops"])
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_fill_fit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/fill_fit.py tests/test_fill_fit.py
git commit -m "feat(fill): fit_fill — per-shape flat/gradient fill decision

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: seam tests — gradient continuity (B) + source-edge softness (A)

**Files:**
- Create: `src/vectoreyes/surface_merge.py`
- Test: `tests/test_surface_merge.py`

The merge (Task 3) is a hybrid: **B is primary** — when both regions fit gradients, merge if one's color at the seam matches the other's within a ΔE (lead-meets-tail). **A is the fallback** — when a narrow region devolved to a flat fill (too little span to fit a gradient), merge on source-seam softness instead. This task builds both predicates.

**Interfaces:**
- Consumes: `color.srgb_to_oklab`; `gradient._model_t`, `gradient._interp_stops_rgb`; `candidate.Fill/LinearGradientFill/RadialGradientFill`; `scipy.ndimage`.
- Produces:
  - `seam_pairs(mask_a, mask_b, rgb) -> tuple[np.ndarray, np.ndarray]` — for every 4-adjacent pixel pair straddling the A|B boundary, the two `(N,3)` uint8 color arrays `(colors_a, colors_b)`. Empty `(0,3)` arrays if not 4-adjacent.
  - `seam_band(mask_a, mask_b, *, width=2) -> tuple[np.ndarray, np.ndarray]` — `(ys, xs)` of `mask_a` pixels within `width` px of `mask_b` (the A-side of the seam).
  - `seam_is_soft(mask_a, mask_b, rgb, *, edge_de=0.06) -> bool` **(A)** — True iff adjacent AND the median straddling-pair OKLab ΔE is below `edge_de`. A within-surface band seam is soft; a real object edge — even a same-color feature's border — spikes above `edge_de`.
  - `gradients_continuous(fill_a, mask_a, fill_b, mask_b, *, seam_de=0.045) -> bool` **(B)** — True iff BOTH fills are gradients, the masks are adjacent, and each fill's model rendered at the shared seam agrees with the other's (mean OKLab ΔE < `seam_de`). Returns False if either fill is flat (that pair takes the A path).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_surface_merge.py`:

```python
import numpy as np
from vectoreyes.surface_merge import seam_is_soft, gradients_continuous
from vectoreyes.fill_fit import fit_fill


def _ramp(h, w, c0, c1):
    rgb = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        t = x / (w - 1)
        rgb[:, x] = [round(c0[i] + t * (c1[i] - c0[i])) for i in range(3)]
    return rgb


def _full_ramp_split():
    # one blue->red ramp, split into left and right halves as two masks over one canvas
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    return rgb, a, b


def test_within_ramp_seam_is_soft():
    rgb, a, b = _full_ramp_split()
    assert seam_is_soft(a, b, rgb)                       # one continuous ramp sliced in two


def test_hard_color_step_is_not_soft():
    # left half ramps to red; right half is a uniform blue patch (a "dot") -> sharp step
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    rgb[:, W // 2:] = (20, 60, 210)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    assert not seam_is_soft(a, b, rgb)


# NOTE: a same-color-at-seam feature is NOT distinguishable by seam_is_soft — if a feature's
# color matches the ramp at the contact point there is no step, so the seam genuinely IS soft.
# That protection lives at the merge level (the union-fits-gradient guard), exercised by the
# Task 3 merge tests (test_two_distinct_flats_do_not_merge / test_flat_dot_on_ramp_stays_separate).


def test_non_adjacent_masks_are_not_soft():
    rgb, a, _b = _full_ramp_split()
    far = np.zeros_like(a); far[:5, 70:75] = True         # disjoint from a (a is left half)
    assert not seam_is_soft(a, far, rgb)


# ── B: gradient-continuity path (both regions wide enough to fit a gradient) ──

def test_two_gradient_halves_are_continuous():
    # each half of a wide ramp spans enough colour to fit a gradient -> compare at the seam
    rgb, a, b = _full_ramp_split()
    fa, fb = fit_fill(a, rgb, flat_hex="#000000"), fit_fill(b, rgb, flat_hex="#000000")
    assert gradients_continuous(fa, a, fb, b)


def test_gradient_meets_flat_is_not_continuous():
    # one side ramps, the other is a uniform patch (flat) -> B does not apply (False)
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    rgb[:, W // 2:] = (20, 60, 210)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    fa, fb = fit_fill(a, rgb, flat_hex="#000000"), fit_fill(b, rgb, flat_hex="#143CD2")
    assert not gradients_continuous(fa, a, fb, b)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_surface_merge.py -k "soft or continuous" -v`
Expected: FAIL — `vectoreyes.surface_merge` does not exist.

- [ ] **Step 3: Implement the seam test**

Create `src/vectoreyes/surface_merge.py`:

```python
# SPDX-License-Identifier: MIT
"""Fill-informed surface merge: two adjacent shapes are one surface only when the source
has NO hard edge across their shared border (a within-gradient band seam) and the union
fits a parametric gradient. Boundaries come from clean masks; this decides which masks
are one surface, never how to draw an edge."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .candidate import Fill, LinearGradientFill, RadialGradientFill
from .color import srgb_to_oklab
from .gradient import _interp_stops_rgb, _model_t

_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _kind(fill: Fill) -> str | None:
    if isinstance(fill, LinearGradientFill):
        return "linear"
    if isinstance(fill, RadialGradientFill):
        return "radial"
    return None


def _model(fill: Fill) -> dict:
    return {"kind": _kind(fill), "geometry": fill.geometry, "stops": fill.stops}


def seam_pairs(mask_a: np.ndarray, mask_b: np.ndarray, rgb: np.ndarray):
    """Colors of every 4-adjacent pixel pair straddling the A|B boundary.

    Returns (colors_a, colors_b), each (N,3) uint8 — colors_a[k] in mask_a is 4-adjacent
    to colors_b[k] in mask_b."""
    cols_a, cols_b = [], []
    H, W = mask_a.shape
    for dy, dx in _NEIGHBORS:
        # b_shift[y,x] = mask_b[y+dy, x+dx]; valid region trims the wrapped edge.
        b_shift = np.zeros_like(mask_b)
        ys = slice(max(0, -dy), H - max(0, dy))
        xs = slice(max(0, -dx), W - max(0, dx))
        ys2 = slice(max(0, dy), H - max(0, -dy))
        xs2 = slice(max(0, dx), W - max(0, -dx))
        b_shift[ys, xs] = mask_b[ys2, xs2]
        border = mask_a & b_shift
        if not border.any():
            continue
        ay, ax = np.where(border)
        cols_a.append(rgb[ay, ax])
        cols_b.append(rgb[ay + dy, ax + dx])
    if not cols_a:
        return np.empty((0, 3), np.uint8), np.empty((0, 3), np.uint8)
    return np.concatenate(cols_a), np.concatenate(cols_b)


def seam_is_soft(mask_a: np.ndarray, mask_b: np.ndarray, rgb: np.ndarray,
                 *, edge_de: float = 0.06) -> bool:
    """(A) True iff the masks are 4-adjacent and the source color steps smoothly across
    their shared border (median straddling-pair OKLab ΔE < edge_de). A real object edge —
    even one whose two sides are color-similar — spikes above edge_de."""
    ca, cb = seam_pairs(mask_a, mask_b, rgb)
    if len(ca) == 0:
        return False
    de = np.linalg.norm(srgb_to_oklab(ca / 255.0) - srgb_to_oklab(cb / 255.0), axis=1)
    return float(np.median(de)) < edge_de


def seam_band(mask_a: np.ndarray, mask_b: np.ndarray, *, width: int = 2):
    """(ys, xs) of mask_a pixels within `width` px of mask_b — the A-side of the seam."""
    band = mask_a & ndimage.binary_dilation(mask_b, iterations=width)
    return np.where(band)


def _rendered_oklab(model: dict, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    pts = np.column_stack([xs, ys]).astype(float)
    return srgb_to_oklab(_interp_stops_rgb(_model_t(model, pts), model["stops"]) / 255.0)


def gradients_continuous(fill_a: Fill, mask_a: np.ndarray, fill_b: Fill, mask_b: np.ndarray,
                         *, seam_de: float = 0.045) -> bool:
    """(B) True iff both fills are gradients over adjacent masks whose models render to
    agreeing colours across the shared seam (mean OKLab ΔE < seam_de) — one gradient's
    colour at the seam matches the other's. False if either fill is flat (A handles that)."""
    if _kind(fill_a) is None or _kind(fill_b) is None:
        return False
    ys_a, xs_a = seam_band(mask_a, mask_b)
    ys_b, xs_b = seam_band(mask_b, mask_a)
    if len(xs_a) == 0 or len(xs_b) == 0:
        return False                                     # not adjacent
    ma, mb = _model(fill_a), _model(fill_b)
    de1 = np.linalg.norm(_rendered_oklab(ma, ys_b, xs_b) - _rendered_oklab(mb, ys_b, xs_b), axis=1).mean()
    de2 = np.linalg.norm(_rendered_oklab(mb, ys_a, xs_a) - _rendered_oklab(ma, ys_a, xs_a), axis=1).mean()
    return max(float(de1), float(de2)) < seam_de
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_surface_merge.py -k "soft or continuous" -v`
Expected: PASS. (If `test_within_ramp_seam_is_soft` or `test_two_gradient_halves_are_continuous` is borderline, `edge_de`/`seam_de` are calibrated in Task 5 — do not loosen `edge_de` so far that `test_hard_color_step_is_not_soft` also passes.)

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/surface_merge.py tests/test_surface_merge.py
git commit -m "feat(merge): seam_is_soft (A) + gradients_continuous (B) seam predicates

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: `merge_surfaces` — hybrid gradient-continuity (B) / soft-seam (A) merge

**Files:**
- Modify: `src/vectoreyes/surface_merge.py`
- Test: `tests/test_surface_merge.py`

**Interfaces:**
- Consumes: `gradients_continuous` + `seam_is_soft` (Task 2), `fill_fit.fit_fill` (Task 1), `types.Region`, `candidate.Fill/LinearGradientFill/RadialGradientFill`.
- Produces: `merge_surfaces(filled: list[tuple[Region, Fill]], rgb: np.ndarray, *, seam_de=0.045, edge_de=0.06) -> list[tuple[Region, Fill]]` — repeatedly merges any two adjacent surfaces, using the hybrid criterion: **(B)** when both fills are gradients, merge if `gradients_continuous` (one gradient's color matches the other's at the seam within `seam_de`); **(A)** otherwise (at least one devolved to a flat fill), merge if `seam_is_soft` (no source edge across the border, `edge_de`). Either path additionally requires the union to fit a parametric gradient (`fit_fill(union)` returns a gradient) — the merged fill IS that union gradient. Keeps the larger-area member's `label`/`color_hex`. Runs to a fixed point. Deterministic: surfaces are scanned in descending area so the partition is order-independent.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_surface_merge.py`:

```python
from vectoreyes.types import Region
from vectoreyes.surface_merge import merge_surfaces
from vectoreyes.fill_fit import fit_fill as _ff
from vectoreyes.candidate import LinearGradientFill


def _region(label, mask, hex_):
    return Region(label=label, mask=mask, color_hex=hex_)


def test_ramp_halves_merge_into_one_gradient():
    # wide halves each fit a gradient -> B path
    rgb, a, b = _full_ramp_split()
    filled = [(_region(0, a, "#1428C8"), _ff(a, rgb, flat_hex="#1428C8")),
              (_region(1, b, "#DC2814"), _ff(b, rgb, flat_hex="#DC2814"))]
    out = merge_surfaces(filled, rgb)
    assert len(out) == 1
    region, fill = out[0]
    assert isinstance(fill, LinearGradientFill)
    assert region.mask.sum() == (a | b).sum()           # union silhouette, clean


def test_narrow_bands_collapse_to_one_gradient():
    # narrow strips of one smooth ramp collapse to a single gradient: a strip too narrow
    # to fit a gradient devolves to flat and merges via the A path; a wide-enough one via
    # B. Either way the outcome is one gradient.
    H, W = 40, 96
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    filled = []
    for k in range(0, W, 8):
        m = np.zeros((H, W), bool); m[:, k:k + 8] = True
        filled.append((_region(k, m, "#808080"), _ff(m, rgb, flat_hex="#808080")))
    out = merge_surfaces(filled, rgb)
    assert len(out) == 1 and isinstance(out[0][1], LinearGradientFill)


def test_flat_dot_on_ramp_stays_separate():
    # the "dot" has a HARD border against the ramp -> hard seam -> never merged,
    # even though its blue is in the wing's color family.
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    rgb[:, W // 2:] = (20, 60, 210)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    filled = [(_region(0, a, "#1428C8"), _ff(a, rgb, flat_hex="#1428C8")),
              (_region(1, b, "#143CD2"), _ff(b, rgb, flat_hex="#143CD2"))]
    out = merge_surfaces(filled, rgb)
    assert len(out) == 2                                 # NOT merged

def test_two_distinct_flats_do_not_merge():
    # two solid colors meeting at a soft-ish AA edge: union is NOT a gradient -> no merge.
    H, W = 40, 80
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, : W // 2] = (200, 40, 40); rgb[:, W // 2:] = (40, 160, 60)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    filled = [(_region(0, a, "#C82828"), _ff(a, rgb, flat_hex="#C82828")),
              (_region(1, b, "#28A03C"), _ff(b, rgb, flat_hex="#28A03C"))]
    assert len(merge_surfaces(filled, rgb)) == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_surface_merge.py -k "merge or stays_separate or distinct_flats or collapse" -v`
Expected: FAIL — `merge_surfaces` not defined.

- [ ] **Step 3: Implement `merge_surfaces`**

Append to `src/vectoreyes/surface_merge.py` (add `from .types import Region` and `from .fill_fit import fit_fill` at the top — `LinearGradientFill`/`RadialGradientFill` are already imported in Task 2. `fit_fill` imports from `gradient`, and `surface_merge` does not import `pipeline`, so there is no cycle):

```python
_GRADIENT = (LinearGradientFill, RadialGradientFill)


def merge_surfaces(filled: list[tuple["Region", Fill]], rgb: np.ndarray, *,
                   seam_de: float = 0.045, edge_de: float = 0.06) -> list[tuple["Region", Fill]]:
    """Fixed-point hybrid merge of adjacent surfaces. (B) both gradients -> merge when one's
    colour matches the other's at the seam (gradients_continuous, seam_de). (A) at least one
    flat (a narrow region that devolved) -> merge when the source has no edge across the seam
    (seam_is_soft, edge_de). Either path also requires the union to fit a parametric gradient,
    which becomes the merged fill. Keeps the larger member's label/color_hex. Deterministic:
    descending-area scan -> order-independent partition. A hard-bordered feature (the dot)
    never merges; two distinct flats whose union is not a gradient never merge."""
    surfaces = list(filled)
    merged = True
    while merged:
        merged = False
        surfaces.sort(key=lambda rf: rf[0].mask.sum(), reverse=True)
        for i in range(len(surfaces)):
            ri, fi = surfaces[i]
            for j in range(i + 1, len(surfaces)):
                rj, fj = surfaces[j]
                if isinstance(fi, _GRADIENT) and isinstance(fj, _GRADIENT):
                    ok = gradients_continuous(fi, ri.mask, fj, rj.mask, seam_de=seam_de)  # B
                else:
                    ok = seam_is_soft(ri.mask, rj.mask, rgb, edge_de=edge_de)             # A
                if not ok:
                    continue
                union = ri.mask | rj.mask
                rep = ri if ri.mask.sum() >= rj.mask.sum() else rj
                new_fill = fit_fill(union, rgb, flat_hex=rep.color_hex)
                if not isinstance(new_fill, _GRADIENT):
                    continue                              # union isn't a gradient: not a merge
                new_region = Region(label=rep.label, mask=union, color_hex=rep.color_hex)
                surfaces = ([s for k, s in enumerate(surfaces) if k not in (i, j)]
                            + [(new_region, new_fill)])
                merged = True
                break
            if merged:
                break
    return surfaces
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_surface_merge.py -v`
Expected: PASS (all of Task 2 + Task 3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/surface_merge.py tests/test_surface_merge.py
git commit -m "feat(merge): merge_surfaces — hybrid gradient-continuity (B) / soft-seam (A)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Rewire the pipeline; retire the footprint path

**Files:**
- Modify: `src/vectoreyes/pipeline.py` (`_render_body`, `build_candidates`)
- Modify: `src/vectoreyes/gradient.py` (delete the retired functions once unreferenced)
- Test: `tests/test_pipeline_decoupled.py`

**Interfaces:**
- Consumes: `fill_fit.fit_fill`, `surface_merge.merge_surfaces`, existing `select_geometry`, `classify_regions`, `reconstruct_scene`, `decompose_components`.
- Produces: the same `_render_body(...) -> (body, defs, cands, axes)` contract, but candidates now carry per-shape fitted fills and no `detect_gradients` footprints. `build_candidates` gains a `fills: dict[int, Fill]` parameter keyed by `Region.label` (the fitted fill for each surface region); a region absent from `fills` falls back to `FlatFill(region.color_hex)`.

**Context for the implementer:** Today `_render_body` (pipeline.py ~205-227) does, per component:
`gradient_fills, comp = detect_gradients(comp, rgb)` → `classify_regions` → `build_candidates(..., gradient_fills, ...)`. `build_candidates` (pipeline.py ~116-185) emits occlusion/lens, then flat regions (`select_geometry` + `FlatFill(region.color_hex)`), then gradient footprints. The new flow replaces the footprint mechanism with fit-then-merge BEFORE classification, so geometry (`select_geometry`) runs once, on the merged surfaces, and emits gradient fills from the merge.

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_pipeline_decoupled.py`:

```python
import re
import numpy as np
from vectoreyes.pipeline import idealize, Options


def _wing_with_dot():
    """A blue->navy vertical ramp 'wing' with a separate uniform blue 'dot' beside it
    on white — a minimal stand-in for the V-bird failure case."""
    H, W = 120, 120
    img = np.full((H, W, 3), 255, np.uint8)
    for y in range(20, 100):                      # the wing: smooth vertical ramp
        t = (y - 20) / 79
        img[y, 20:70] = [round(40 + t * 0), round(120 - t * 90), round(230 - t * 110)]
    yy, xx = np.ogrid[:H, :W]                      # a separate uniform blue dot
    img[((yy - 40) ** 2 + (xx - 95) ** 2) <= 12 ** 2] = (30, 100, 220)
    return img


def test_wing_emits_one_gradient_and_dot_survives():
    svg = idealize(_wing_with_dot(), options=Options(max_colors=16))
    # the wing is ONE gradient, not stacked flat bands
    assert len(re.findall(r"<linearGradient", svg)) >= 1
    # the dot is still present as its own element (a circle or a small path), not absorbed
    # — at least two distinct filled elements exist (wing + dot)
    assert len(re.findall(r"<(path|circle|ellipse|rect|polygon)\b", svg)) >= 2


def test_flat_logo_uses_no_gradient():
    # a solid square on white must stay a single flat shape, no gradient defs
    img = np.full((80, 80, 3), 255, np.uint8)
    img[20:60, 20:60] = (200, 40, 40)
    svg = idealize(img, options=Options(max_colors=16))
    assert "<linearGradient" not in svg and "<radialGradient" not in svg
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_pipeline_decoupled.py -v`
Expected: FAIL — the wing currently emits stacked flat bands (no `<linearGradient`) and/or the dot is absorbed, per the documented failure.

- [ ] **Step 3: Rewire `build_candidates` to accept fitted fills**

In `src/vectoreyes/pipeline.py`, change `build_candidates`'s signature to drop `gradient_fills` and add `fills` (ensure `Fill` is imported: `from .candidate import ... Fill`):

```python
def build_candidates(
    reconstructed: list, straddlers: list[Region], pairs: list[tuple[Region, Region]],
    loners: list[Region], fills: dict[int, Fill],
    opt: Options, axis: Axis | None,
    source_rgb: np.ndarray | None, *, base: int = 0,
) -> list[Candidate]:
```

Replace the flat-region append (the line `cands.append(Candidate(shape, FlatFill(region.color_hex), "region", ...))`) so the fill comes from `fills` (fitted), defaulting to flat:

```python
        fill = fills.get(region.label, FlatFill(region.color_hex))
        cands.append(Candidate(shape, fill, "region",
                               mirror=axis if is_pair else None, strategy=strategy))
```

Delete the entire trailing `for footprint, model in gradient_fills:` loop (pipeline.py ~165-183) — gradient footprints no longer exist; gradients arrive as a surface region's `fill`.

- [ ] **Step 4: Rewire `_render_body` to fit + merge before classify**

In `src/vectoreyes/pipeline.py` `_render_body`, replace the per-component block

```python
        gradient_fills: list[tuple[Region, dict]] = []
        if rgb is not None:
            gradient_fills, comp = detect_gradients(comp, rgb)

        if axis is not None:
            straddlers, pairs, loners = classify_regions(comp, axis)
        else:
            straddlers, pairs, loners = list(comp), [], []

        cands += build_candidates(
            reconstructed, straddlers, pairs, loners, gradient_fills, opt, axis, rgb,
            base=len(cands),
        )
```

with

```python
        # Fill-fit each region, then merge surfaces whose gradients are continuous.
        # Shapes come from these (possibly unioned) clean masks; geometry is fit once
        # per surface in build_candidates. No footprint reconstruction.
        if rgb is not None:
            filled = [(r, fit_fill(r.mask, rgb, flat_hex=r.color_hex)) for r in comp]
            filled = merge_surfaces(filled, rgb)
            comp = [r for r, _ in filled]
            fills = {r.label: f for r, f in filled}
        else:
            fills = {}

        if axis is not None:
            straddlers, pairs, loners = classify_regions(comp, axis)
        else:
            straddlers, pairs, loners = list(comp), [], []

        cands += build_candidates(
            reconstructed, straddlers, pairs, loners, fills, opt, axis, rgb,
            base=len(cands),
        )
```

Add imports at the top of `pipeline.py`: `from .fill_fit import fit_fill` and `from .surface_merge import merge_surfaces`; remove `from .gradient import detect_gradients`.

> Note for the implementer: `reconstruct_scene` still runs before this block (occlusion is geometry, unaffected). The merged `comp` regions keep clean masks, so `classify_regions` (symmetry) and `select_geometry` operate on whole surfaces — a merged wing is one region, so corner-rounding and primitive recognition see the full silhouette (the "path could be a primitive" promotion happens for free via the existing `select_geometry` on the union mask). Do not special-case it.

- [ ] **Step 5: Run the integration test + full suite**

Run: `uv run pytest tests/test_pipeline_decoupled.py -v`
Expected: PASS — wing emits one `<linearGradient>`, dot survives as a separate element; flat square stays flat.

Run: `uv run pytest -q`
Expected: gradient-related tests in `tests/test_gradient.py` that asserted the OLD footprint behavior will FAIL — that is expected (the footprint path is retired). Update or remove those assertions to reflect the decoupled behavior (a gradient is now a surface region's fill, not a `detect_gradients` footprint). Do NOT weaken a test to pass; rewrite it against the new contract or delete it if it tested only the retired internal. Record each changed/deleted test in the commit message.

- [ ] **Step 6: Delete the retired functions**

Once `rg -n "detect_gradients|_component_fill|_expand_footprint|merge_components|_group_is_fillable" src/ tests/` shows no remaining non-definition references, delete those functions from `src/vectoreyes/gradient.py`. Keep `_best_parametric`, `_model_t`, `_interp_stops_rgb`, `_per_pixel_delta_e`, `_stop_span`, the `_fit_*` searched fitters, and the constants. Re-run `uv run pytest -q`.

Expected: PASS (full suite green after test updates).

- [ ] **Step 7: Commit**

```bash
git add src/vectoreyes/pipeline.py src/vectoreyes/gradient.py tests/
git commit -m "feat(pipeline): decouple fill from shape — fit+merge surfaces, retire footprint path

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Corpus validation + V-bird acceptance

**Files:**
- Create: `tests/test_decoupled_corpus.py` (markers/opt-in if the corpus is large)
- Modify: calibration constants in `surface_merge.py` / `fill_fit.py` only if the corpus shows it

**Interfaces:**
- Consumes: the full decoupled pipeline (Tasks 1-4); the repo's existing corpus fixtures under `scratch/real-logos/` (UNTRACKED — never `git add scratch/`).

**Context:** The acceptance target is the V-bird at `/Users/pmouli/.claude/image-cache/765f9098-2b6a-45f9-b368-3394ea44b5c0/3.png` (if present) and the existing gradient corpus (firefox, instagram, pokeball, gdrive). Flats must be visibly unchanged; gradient logos must have crisp silhouettes and one gradient per surface.

- [ ] **Step 1: Write a corpus comparison harness (opt-in)**

Create `tests/test_decoupled_corpus.py`:

```python
import os
import re
import glob
import numpy as np
import pytest
from PIL import Image
from vectoreyes.pipeline import idealize, Options, _flatten_on_white

CORPUS = sorted(glob.glob(os.path.expanduser("scratch/real-logos/*.png")))
pytestmark = pytest.mark.skipif(not CORPUS, reason="corpus not present")


@pytest.mark.parametrize("path", CORPUS)
def test_corpus_emits_well_formed_svg(path):
    arr = _flatten_on_white(Image.open(path))
    svg = idealize(arr, options=Options(max_colors=16))
    assert svg.startswith("<svg ") and svg.rstrip().endswith("</svg>")
    # at least one painted element exists
    assert re.search(r"<(path|circle|ellipse|rect|polygon)\b", svg)
```

- [ ] **Step 2: Run the harness; record before/after element counts**

Run: `uv run pytest tests/test_decoupled_corpus.py -v`
Then, for the named gradient logos, capture element + gradient counts to a scratch report (NOT committed):

```bash
uv run python - <<'PY'
import glob, re, numpy as np
from PIL import Image
from vectoreyes.pipeline import idealize, Options, _flatten_on_white
for p in ["firefox","instagram","pokeball","gdrive"]:
    import os
    f=f"scratch/real-logos/{p}.png"
    if not os.path.exists(f): continue
    svg=idealize(_flatten_on_white(Image.open(f)), options=Options(max_colors=16))
    print(p, "paths", len(re.findall(r"<path", svg)), "lin", len(re.findall(r"<linearGradient", svg)),
          "rad", len(re.findall(r"<radialGradient", svg)))
PY
```

Expected: gradient logos show ≥1 gradient and FEWER total paths than the banded baseline; flats show 0 gradients.

- [ ] **Step 3: Calibrate `seam_de` / `edge_de` only if needed**

If a gradient logo under-merges (residual bands) or a distinct surface over-merges, adjust the merge thresholds in `surface_merge.py`: `seam_de` (the B-path gradient-continuity tolerance, start 0.045) and `edge_de` (the A-path soft-seam threshold, start 0.06); and the `max_gradient_de` acceptance in `fit_fill` (start `_GATE_DELTA_E`) only if needed. Re-run Step 2 and the Task 2/3 unit tests after each change. Document the final values and the logo that drove them in the commit message. Do NOT change `_GATE_DELTA_E` or `_MIN_STOP_SPAN` (shared with other code) without a separate justification.

- [ ] **Step 4: V-bird visual check (manual)**

Render the V-bird through the pipeline and confirm by eye (a scratch PNG, not committed): both wings are single gradients, the middle blue dot is one clean circle, silhouette edges are crisp (no frayed boundary). If the dot still fragments, it is a *segmentation* issue upstream of this work — record it as a follow-up, do not loosen the merge to mask it.

- [ ] **Step 5: Commit**

```bash
git add tests/test_decoupled_corpus.py src/vectoreyes/surface_merge.py src/vectoreyes/fill_fit.py
git commit -m "test(decoupled): corpus validation + calibrated seam tolerances

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final Review

After all tasks: `uv run pytest -q` green; dispatch a whole-branch review (most-capable model) over `master..HEAD`; then superpowers:finishing-a-development-branch to open the PR. PR body ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

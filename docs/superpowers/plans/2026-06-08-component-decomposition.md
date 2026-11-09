# Component Decomposition (Slice 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split a mark into spatially-separated components via a gutter-based recursive X-Y cut, then run the existing per-mark analysis (symmetry, corner-radius, occlusion, gradients, candidate build) independently per component — giving per-component vertical symmetry and unblocking multi-blob gradients — while staying byte-identical for marks with no qualifying gutter.

**Architecture:** A new `decompose_components` (recursive X-Y cut on the union silhouette) is called at the top of `_render_body`; the existing per-mark analysis runs in a loop, one iteration per component, concatenating per-component candidate lists into one list that the **unchanged** single emit loop renders (so `sN` ids stay global/continuous). Components keep full-canvas coordinates.

**Tech Stack:** Python 3, numpy, pytest. Tests run with `.venv/bin/python -m pytest`.

**Branch:** `feat/components` (already created off master).

---

## Pre-flight

- [ ] **Confirm branch + baseline**

```bash
cd /Users/pmouli/GitHub.nosync/active/py/vectoreyes
git branch --show-current      # expect: feat/components
.venv/bin/python -m pytest -q   # baseline: expect "215 passed"
```

## File Structure

- **Create** `src/vectoreyes/components.py` — `decompose_components(regions, shape)` + private gutter helpers. One responsibility: partition regions into spatially-separated components. Imports only numpy + `.types.Region`.
- **Create** `tests/test_components.py` — unit tests for the decomposition.
- **Modify** `src/vectoreyes/pipeline.py` — `_render_body` loops over components; import `decompose_components`.
- **Modify** `tests/test_pipeline.py` — per-component symmetry, multi-blob gradient, continuous-`sN`, parity tests.

---

### Task 1: `components.py` — recursive X-Y cut decomposition

**Files:**
- Create: `src/vectoreyes/components.py`
- Test: `tests/test_components.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_components.py`:

```python
import numpy as np

from vectoreyes.types import Region
from vectoreyes.components import decompose_components


def _rect_region(rid, r0, r1, c0, c1, h, w, hexc="#222222"):
    m = np.zeros((h, w), bool)
    m[r0:r1, c0:c1] = True
    return Region(rid, m, hexc)


def _disk_region(rid, cy, cx, rad, h, w, hexc="#222222"):
    yy, xx = np.ogrid[:h, :w]
    return Region(rid, (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2, hexc)


def test_single_region_is_one_component():
    h = w = 80
    regs = [_disk_region(1, 40, 40, 20, h, w)]
    comps = decompose_components(regs, (h, w))
    assert len(comps) == 1 and comps[0] == regs


def test_two_widely_separated_blobs_split_left_to_right():
    h, w = 60, 140
    left = _disk_region(1, 30, 25, 18, h, w)        # cols ~7..43
    right = _rect_region(2, 12, 48, 95, 130, h, w)  # cols 95..130
    comps = decompose_components([left, right], (h, w))   # gutter cols ~43..95 (~52 wide)
    assert len(comps) == 2
    assert comps[0] == [left] and comps[1] == [right]     # reading order L->R


def test_vertical_stack_splits_top_to_bottom():
    h, w = 140, 60
    top = _disk_region(1, 25, 30, 18, h, w)         # rows ~7..43
    bot = _rect_region(2, 95, 130, 12, 48, h, w)    # rows 95..130
    comps = decompose_components([top, bot], (h, w))      # gutter rows ~43..95 (~52 tall)
    assert len(comps) == 2
    assert comps[0] == [top] and comps[1] == [bot]        # reading order top->bottom


def test_borderline_narrow_gap_stays_one_component():
    # two-band-logo geometry: bands at rows 8-26 and 34-52 -> ~8px gap on a 44px block
    # (~18% < 30% threshold) must NOT split.
    h, w = 60, 80
    top = _rect_region(1, 8, 26, 12, 68, h, w)
    bot = _rect_region(2, 34, 52, 20, 60, h, w)
    comps = decompose_components([top, bot], (h, w))
    assert len(comps) == 1
    assert set(id(r) for r in comps[0]) == {id(top), id(bot)}


def test_nested_icon_over_two_word_row():
    # icon on top (rows 5..35), then a row with two blobs split by a vertical gutter
    h, w = 100, 120
    icon = _rect_region(1, 5, 35, 50, 70, h, w)        # top block
    word_l = _rect_region(2, 60, 90, 10, 45, h, w)     # bottom-left
    word_r = _rect_region(3, 60, 90, 75, 110, h, w)    # bottom-right
    comps = decompose_components([icon, word_l, word_r], (h, w))
    assert len(comps) == 3
    assert comps[0] == [icon]                          # horizontal cut first: icon on top
    assert comps[1] == [word_l] and comps[2] == [word_r]  # then vertical: L, R


def test_partition_is_clean_no_loss_no_duplication():
    h, w = 60, 140
    a = _disk_region(1, 30, 25, 18, h, w)
    b = _rect_region(2, 12, 48, 95, 130, h, w)
    comps = decompose_components([a, b], (h, w))
    flat = [r for c in comps for r in c]
    assert len(flat) == 2
    assert {id(r) for r in flat} == {id(a), id(b)}     # every input exactly once
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_components.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vectoreyes.components'`

- [ ] **Step 3: Write the implementation**

Create `src/vectoreyes/components.py`:

```python
"""Gutter-based component decomposition (slice 5). Recursive X-Y cut: split a mark on
the widest full-span whitespace gutter (horizontal or vertical) wider than a
conservative scale-relative threshold, recurse on each block, and return the components
in reading order. A mark with no qualifying gutter is a single component — the parity
path that keeps single-component output byte-identical to the pre-slice-5 pipeline."""

from __future__ import annotations

import numpy as np

from .types import Region

# Conservative, scale-relative: a gutter must be at least this fraction of the block's
# extent along the cut axis (or the absolute floor, whichever is larger) to split.
# Tuned so obvious multi-element marks split but borderline intra-mark gaps do not.
_GUTTER_FRACTION = 0.3
_GUTTER_ABS_FLOOR = 6


def _interior_gaps(occ: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of empty (False) cells strictly between the first and last occupied
    cell of `occ` (a 1D bool occupancy profile). Each run is returned as [start, end)."""
    idx = np.flatnonzero(occ)
    if len(idx) == 0:
        return []
    first, last = int(idx[0]), int(idx[-1])
    gaps: list[tuple[int, int]] = []
    run_start: int | None = None
    for i in range(first, last + 1):
        if not occ[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            gaps.append((run_start, i))
            run_start = None
    return gaps


def _best_gutter(sil: np.ndarray) -> tuple[str, float] | None:
    """The widest qualifying full-span gutter in the silhouette. Returns ("h", y_cut) or
    ("v", x_cut) (cut position in pixels), or None if no gutter qualifies. On a width
    tie, prefers the cut closest to the block centre (most even split)."""
    ys, xs = np.where(sil)
    if len(ys) == 0:
        return None
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    block = sil[r0:r1, c0:c1]

    cands: list[tuple[int, float, str, float]] = []  # (-width sorts widest first via key)

    row_occ = block.any(axis=1)                       # occupancy per row
    h_min = max(_GUTTER_ABS_FLOOR, _GUTTER_FRACTION * (r1 - r0))
    h_centre = (r0 + r1) / 2
    for s, e in _interior_gaps(row_occ):
        width = e - s
        if width >= h_min:
            cut = r0 + (s + e) / 2
            cands.append((width, abs(cut - h_centre), "h", cut))

    col_occ = block.any(axis=0)                       # occupancy per column
    v_min = max(_GUTTER_ABS_FLOOR, _GUTTER_FRACTION * (c1 - c0))
    v_centre = (c0 + c1) / 2
    for s, e in _interior_gaps(col_occ):
        width = e - s
        if width >= v_min:
            cut = c0 + (s + e) / 2
            cands.append((width, abs(cut - v_centre), "v", cut))

    if not cands:
        return None
    cands.sort(key=lambda t: (-t[0], t[1]))           # widest, then most even
    _, _, axis, cut = cands[0]
    return (axis, cut)


def _partition(regions: list[Region], axis: str, cut: float) -> tuple[list[Region], list[Region]]:
    """Split regions by the side of `cut` their pixel-centroid lies on. The gutter is
    empty, so no region's pixels lie in the cut band — every region falls cleanly to one
    side. `axis`=="h" splits above/below (row centroid); "v" splits left/right (col)."""
    a: list[Region] = []
    b: list[Region] = []
    for r in regions:
        rr, cc = np.where(r.mask)
        centroid = rr.mean() if axis == "h" else cc.mean()
        (a if centroid < cut else b).append(r)
    return a, b


def decompose_components(regions: list[Region], shape: tuple[int, int]) -> list[list[Region]]:
    """Partition `regions` into spatially-separated components by recursive X-Y cut on
    the union silhouette, in reading order (top->bottom, left->right). Returns
    [regions] (one component) when there is <=1 region or no qualifying gutter — the
    parity path."""
    if len(regions) <= 1:
        return [regions]
    sil = np.zeros(shape, bool)
    for r in regions:
        sil |= r.mask
    gutter = _best_gutter(sil)
    if gutter is None:
        return [regions]
    axis, cut = gutter
    a, b = _partition(regions, axis, cut)
    if not a or not b:                                # degenerate guard (no real split)
        return [regions]
    return decompose_components(a, shape) + decompose_components(b, shape)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_components.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/components.py tests/test_components.py
git commit -m "feat(components): gutter-based recursive X-Y cut decomposition (slice 5)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Wire decomposition into `_render_body` + integration tests

**Files:**
- Modify: `src/vectoreyes/pipeline.py` (`_render_body` lines ~210-231; add import)
- Test: `tests/test_pipeline.py`

The current `_render_body` (lines ~210-231) computes one mark-wide `axis`/`corner_radius` and runs `reconstruct_scene`/`detect_gradients`/`classify_regions`/`build_candidates` over all `regions`, then a single emit loop over `cands`. Wrap the analysis in a per-component loop that accumulates into `cands`; leave the emit loop unchanged.

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_pipeline.py`:

```python
def _gradient_bar(img, r0, r1, c0, c1, c_lo, c_hi):
    # horizontal left->right gradient from c_lo to c_hi inside the box
    width = c1 - c0
    for i, x in enumerate(range(c0, c1)):
        t = i / max(1, width - 1)
        img[r0:r1, x] = tuple(int(round(c_lo[k] + t * (c_hi[k] - c_lo[k]))) for k in range(3))


def test_two_separated_gradient_blobs_emit_two_gradients():
    # two gradient bars separated by a wide vertical gutter: each is a single dominant
    # blob within its own component, so both pass the single-blob gate (this is the
    # multi-blob unblock — together they would fail _dominant_blob_fraction).
    h, w = 60, 200
    img = np.full((h, w, 3), 255, np.uint8)
    _gradient_bar(img, 15, 45, 10, 70, (10, 20, 200), (200, 40, 20))      # left bar  cols 10..70
    _gradient_bar(img, 15, 45, 130, 190, (20, 200, 30), (10, 40, 220))    # right bar cols 130..190
    # gutter cols 70..130 = 60 wide; block extent 10..190 = 180; threshold 0.3*180=54 -> splits
    svg = idealize(img)
    assert svg.count("<linearGradient") + svg.count("<radialGradient") >= 2


def test_two_component_ids_are_continuous_no_collision():
    # two separated solid blobs -> two components -> ids s0, s1 with no gap/collision
    import re
    h, w = 60, 160
    img = np.full((h, w, 3), 255, np.uint8)
    img[15:45, 10:50] = (10, 30, 90)       # left square  cols 10..50
    img[15:45, 110:150] = (90, 30, 10)     # right square cols 110..150 (gutter 50..110 = 60)
    svg = idealize(img)
    ids = re.findall(r'id="(s\d+)"', svg)
    assert ids == ["s0", "s1"]             # global, continuous across the component boundary


def test_per_component_vertical_symmetry_emits_mirror_use():
    # LEFT component: a mirror PAIR of squares about col 30 (cols 10-25 and 35-50).
    # RIGHT component: a single offset square (cols 120-150). The WHOLE silhouette has
    # no vertical mirror, so the pre-slice-5 mark-wide axis is None and the left pair is
    # NOT deduped (no <use>). Per component, the left component's local axis (col 30)
    # makes the two squares a mirror pair -> one element + a <use> mirror.
    from vectoreyes.symmetry import detect_axis
    h, w = 70, 170
    img = np.full((h, w, 3), 255, np.uint8)
    img[20:50, 10:25] = (20, 40, 80)       # left-of-pair
    img[20:50, 35:50] = (20, 40, 80)       # right-of-pair (mirror about col 30)
    img[20:50, 120:150] = (20, 40, 80)     # lone square right (gutter 50..120 = 70 wide)
    silhouette = np.any(img != 255, axis=2)
    assert detect_axis(silhouette) is None              # premise: no global vertical axis
    svg = idealize(img)
    assert "<use" in svg                                # the pair dedups only per-component
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k "gradient_blobs or continuous or mirror_use" -q`
Expected: FAIL — without per-component analysis the second gradient is gated out by the single-blob test, and the left pair isn't deduped so no `<use>` is emitted. (The continuous-id test may already pass depending on pre-change emit; the gradient and mirror_use tests must fail pre-change.)

- [ ] **Step 3: Implement the per-component loop**

Add the import near the other `from .` imports in `pipeline.py`:

```python
from .components import decompose_components
```

Replace the block in `_render_body` from `silhouette = np.any(...)` through the `cands = build_candidates(...)` call (the per-mark analysis, ~lines 210-231) with a per-component loop. The result list MUST be named `cands` so the existing emit loop below is untouched:

```python
    components = decompose_components(regions, (h, w))
    defs: list[str] = []
    cands: list[Candidate] = []
    for comp in components:
        silhouette = np.any([r.mask for r in comp], axis=0)
        axis = None if opt.no_symmetry else detect_axis(silhouette)
        corner_radius = opt.corner_radius if opt.corner_radius is not None else _mark_corner_radius(comp, axis)

        reconstructed, comp = reconstruct_scene(comp, axis, (h, w))

        # Per component: one local axis + fillet radius, its own occlusion/gradient pass.
        # (Single-component marks take this loop exactly once -> identical to pre-slice-5.)
        gradient_fills: list[tuple[Region, dict]] = []
        if rgb is not None:
            gradient_fills, comp = detect_gradients(comp, rgb)

        if axis is not None:
            straddlers, pairs, loners = classify_regions(comp, axis)
        else:
            straddlers, pairs, loners = list(comp), [], []

        cands += build_candidates(
            reconstructed, straddlers, pairs, loners, gradient_fills, opt, axis, corner_radius, rgb
        )
```

Notes for the implementer:
- The `defs: list[str] = []` line moves up here (it was previously declared mid-function); ensure there is exactly ONE `defs = []` and the emit loop's `_fill_attr` still appends to it.
- Do NOT change the `emit`/`_fill_attr` closures or the `for cand in cands:` emit loop — they stay exactly as they are, iterating the accumulated `cands` and assigning `s{eid}` globally.
- The old standalone NOTE comment about "full pre-strip mark" no longer applies per-mark; replace it with the per-component comment shown above.

- [ ] **Step 4: Run the new tests, then the full suite (parity gate)**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k "gradient_blobs or continuous or mirror_use" -q`
Expected: PASS (3 passed)

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS, **including `tests/test_candidate_byte_identical.py`** with no re-capture. If any golden diverges, STOP — do not re-capture. Report the diverging case(s) with the before/after so the controller can adjudicate (a mark with a qualifying gutter changing is expected to be surfaced, not silently accepted).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): per-component analysis via gutter decomposition (slice 5)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Full suite green incl. byte-identical goldens**

Run: `.venv/bin/python -m pytest -q`
Expected: all passed; `tests/test_candidate_byte_identical.py` unchanged (no re-capture). Any divergence surfaced to the controller, not silently re-baselined.

- [ ] Dispatch the final whole-branch code review, then use `superpowers:finishing-a-development-branch`.

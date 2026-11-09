# Polygon Occlusion Completer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct a convex polygon (diamond, triangle, hexagon, …) that is partially occluded, by fitting supporting lines to its visible edges and intersecting consecutive lines to recover every corner — including the one hidden behind the occluder.

**Architecture:** A new `complete_polygon` completer in `occlusion.py` plus a `"polygon"` arm in `primitive_mask`. The existing N-shape orchestration (`reconstruct_scene`, `_pair_constraint`, `_topo_order`, `stack_agreement`, the global z-order) and the emit layer (`shape_to_svg`/`shape_to_path_d` already render `"polygon"`) are reused unchanged. The completer fits one total-least-squares line per RDP-simplified edge of the own (non-seam) boundary, then intersects each consecutive pair of lines cyclically; the wrap-around intersection of the last and first line recovers the hidden corner.

**Tech Stack:** Python, NumPy, scikit-image (`skimage.draw.polygon2mask`, already a dependency), the existing `rdp` simplifier and `label_boundary` own/seam splitter. Tests: pytest, the repo's `tests/_render.py` (`render_svg`, `ssim`).

---

## Background the implementer needs

Read `docs/superpowers/specs/2026-06-06-polygon-occlusion-design.md` for the design. Key facts about the existing code you will build on:

- **`label_boundary(region, others, *, reach=2, contour_index=0) -> (contour, seam)`** (`src/vectoreyes/occlusion.py`). `contour` is an `(N, 2)` array of `(x, y)` points tracing the region's outer boundary as a **closed cyclic loop**; `seam` is a length-`N` bool array — `True` where another region's mask is within `reach` px (the occluded part), `False` on the region's own (truly visible) boundary.
- **`rdp(points, epsilon) -> np.ndarray`** (`src/vectoreyes/contour.py`): Ramer–Douglas–Peucker polyline simplification. Returns a subsequence of the input points (originals preserved, in order); guarantees every dropped point lies within `epsilon` of the simplified polyline.
- **`Region`** (`src/vectoreyes/types.py`): dataclass with `.label: int`, `.mask: np.ndarray` (bool `(H, W)`), `.color_hex: str`.
- **`primitive_mask(prim, h, w) -> np.ndarray`** (`occlusion.py`): boolean mask of a completed primitive. Already handles `"circle"`, `"annulus"`, `"ellipse"`. This is the single chokepoint `_pair_constraint`, `_topo_order`, and `stack_agreement` reason through — once it handles `"polygon"`, the whole orchestration does.
- **`_complete_member(region, others) -> dict | None`** (`occlusion.py`): dispatches a group member to `complete_annulus` (if it has a hole) then `complete_primitive` (circle/ellipse). We append a polygon arm.
- **Emit is already done:** `shape_to_svg` emits `<polygon points="…"/>` and `shape_to_path_d` emits `M…L…L…Z` for `kind == "polygon"` with `params = {"points": [(x, y), …]}`. The pipeline's `ScenePrimitive` render path (`src/vectoreyes/pipeline.py:226-235`) is generic: a polygon primitive renders via `shape_to_svg` (non-flatten) or `shape_to_path_d` (flatten, `rule=None`) with **no change required**.
- **Module-level constants** already in `occlusion.py`: `_MAX_RESIDUAL = 1.6`, `_GATE_AGREEMENT = 0.96`, etc. We add `_MAX_VERTICES = 8`.

**Representation contract:** a completed polygon is `{"kind": "polygon", "params": {"points": [(x, y), …]}}` where `points` is a list of `(float, float)` tuples ordered around the ring — exactly what `recognize_polygon` and the emit layer already use.

**Decline contract:** if completion fails (too few edges, a line's residual exceeds tolerance, parallel adjacent lines, a non-convex recovered ring, or too many vertices) `complete_polygon` returns `None` and the member falls back to today's per-region fit. A polygon with a *whole edge* hidden is not detected by the completer itself — it is caught by the existing consistency gate (`stack_agreement < _GATE_AGREEMENT`), which declines the whole group. This is the same safety contract as the annulus pass.

---

## File Structure

- **Modify** `src/vectoreyes/occlusion.py` — add `_fit_line`, `_line_intersect`, `_is_convex`, `_own_runs`, `complete_polygon`; extend `primitive_mask` with a `"polygon"` case; add the polygon arm to `_complete_member`; add `_MAX_VERTICES`; add `from skimage.draw import polygon2mask`.
- **Modify** `tests/test_occlusion.py` — append unit tests for the new helpers and `complete_polygon`, the `primitive_mask` polygon case, and a reconstruct-scene decline test.
- **Create** `tests/test_acceptance_polygon.py` — end-to-end acceptance fixtures (two overlapping diamonds; diamond × disk) through `idealize`.

No changes to `pipeline.py`, `emit.py`, or `fit.py`.

---

### Task 1: Geometry helpers — line fit, intersection, convexity

**Files:**
- Modify: `src/vectoreyes/occlusion.py` (add three module-level helper functions, after `_own_arc_span_deg` near line 79)
- Test: `tests/test_occlusion.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_occlusion.py`:

```python
def test_fit_line_recovers_horizontal_line():
    from vectoreyes.occlusion import _fit_line
    pts = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0], [6.0, 5.0]])
    a, b, c, resid = _fit_line(pts)
    # line is y == 5  ->  normal is (0, 1), offset c == 5
    assert resid < 1e-6
    assert abs(a * 3.0 + b * 5.0 - c) < 1e-6      # a point on the line satisfies ax+by=c
    assert abs(a * 3.0 + b * 9.0 - c) > 3.0       # a point 4px off has large signed distance


def test_fit_line_reports_residual_for_noisy_points():
    from vectoreyes.occlusion import _fit_line
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 3.0], [6.0, 0.0]])  # one 3px outlier
    _, _, _, resid = _fit_line(pts)
    assert resid > 1.0


def test_line_intersect_crossing_lines():
    from vectoreyes.occlusion import _fit_line, _line_intersect
    horiz = _fit_line(np.array([[0.0, 4.0], [8.0, 4.0]]))
    vert = _fit_line(np.array([[5.0, 0.0], [5.0, 9.0]]))
    p = _line_intersect(horiz, vert)
    assert p is not None
    assert abs(p[0] - 5.0) < 1e-6 and abs(p[1] - 4.0) < 1e-6


def test_line_intersect_parallel_returns_none():
    from vectoreyes.occlusion import _fit_line, _line_intersect
    l1 = _fit_line(np.array([[0.0, 0.0], [8.0, 0.0]]))
    l2 = _fit_line(np.array([[0.0, 3.0], [8.0, 3.0]]))
    assert _line_intersect(l1, l2) is None


def test_is_convex_accepts_square_rejects_arrow():
    from vectoreyes.occlusion import _is_convex
    square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    assert _is_convex(square)
    concave = [(0.0, 0.0), (4.0, 0.0), (2.0, 2.0), (4.0, 4.0), (0.0, 4.0)]  # arrowhead notch
    assert not _is_convex(concave)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_occlusion.py -k "fit_line or line_intersect or is_convex" -q`
Expected: FAIL with `ImportError: cannot import name '_fit_line'` (etc.).

- [ ] **Step 3: Implement the helpers**

In `src/vectoreyes/occlusion.py`, add after `_own_arc_span_deg` (≈ line 79):

```python
def _fit_line(pts: np.ndarray) -> tuple[float, float, float, float] | None:
    """Total-least-squares line through `pts`. Returns (a, b, c, max_residual) for
    the line a*x + b*y = c with a**2 + b**2 == 1 (so a*x + b*y - c is signed
    perpendicular distance), or None for fewer than 2 points."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return None
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean)
    direction = vt[0]                                  # principal axis of the points
    a, b = float(-direction[1]), float(direction[0])   # unit normal
    c = a * mean[0] + b * mean[1]
    resid = float(np.abs(pts @ np.array([a, b]) - c).max())
    return a, b, c, resid


def _line_intersect(l1: tuple, l2: tuple) -> tuple[float, float] | None:
    """Intersection of lines (a1,b1,c1) and (a2,b2,c2); None if (near-)parallel."""
    a1, b1, c1 = l1[0], l1[1], l1[2]
    a2, b2, c2 = l2[0], l2[1], l2[2]
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return ((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det)


def _is_convex(pts: list[tuple[float, float]]) -> bool:
    """True if the closed vertex ring turns consistently one way (all cross products
    share a sign). Rejects concave or self-intersecting rings."""
    n = len(pts)
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cx, cy = pts[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return sign != 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -k "fit_line or line_intersect or is_convex" -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): line-fit, intersection, convexity helpers for polygon completion"
```

---

### Task 2: Own-boundary run extraction

**Files:**
- Modify: `src/vectoreyes/occlusion.py` (add `_own_runs` after the Task 1 helpers)
- Test: `tests/test_occlusion.py` (append)

The own (non-seam) boundary of an occluded region forms one or more contiguous arcs along the cyclic contour. `_own_runs` returns each maximal contiguous run as an open polyline, handling the wrap across index 0.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_occlusion.py`:

```python
def test_own_runs_single_contiguous_arc():
    from vectoreyes.occlusion import _own_runs
    contour = np.array([[float(i), 0.0] for i in range(10)])
    seam = np.array([False, False, False, True, True, True, False, False, False, False])
    runs = _own_runs(contour, seam)
    assert len(runs) == 1                       # the seam splits the cyclic loop into one own arc
    assert len(runs[0]) == 7                     # indices 6,7,8,9,0,1,2 wrap across 0


def test_own_runs_no_seam_returns_whole_contour():
    from vectoreyes.occlusion import _own_runs
    contour = np.array([[float(i), 0.0] for i in range(5)])
    seam = np.zeros(5, bool)
    runs = _own_runs(contour, seam)
    assert len(runs) == 1 and len(runs[0]) == 5


def test_own_runs_all_seam_returns_empty():
    from vectoreyes.occlusion import _own_runs
    contour = np.array([[float(i), 0.0] for i in range(5)])
    runs = _own_runs(contour, np.ones(5, bool))
    assert runs == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_occlusion.py -k own_runs -q`
Expected: FAIL with `ImportError: cannot import name '_own_runs'`.

- [ ] **Step 3: Implement `_own_runs`**

Add after the Task 1 helpers in `src/vectoreyes/occlusion.py`:

```python
def _own_runs(contour: np.ndarray, seam: np.ndarray) -> list[np.ndarray]:
    """Maximal contiguous runs of own (non-seam) contour points, each as an open
    (M, 2) polyline. The contour is cyclic, so a run may wrap across index 0."""
    n = len(seam)
    own = ~np.asarray(seam, bool)
    if n == 0 or not own.any():
        return []
    if own.all():
        return [np.asarray(contour, float)]
    # a run starts at an own point whose predecessor (cyclically) is a seam point
    starts = [i for i in range(n) if own[i] and not own[(i - 1) % n]]
    runs: list[np.ndarray] = []
    for s in starts:
        idx = []
        i = s
        while own[i % n] and len(idx) < n:
            idx.append(i % n)
            i += 1
        runs.append(np.asarray(contour, float)[idx])
    return runs
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -k own_runs -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): extract contiguous own-boundary runs from a cyclic contour"
```

---

### Task 3: `complete_polygon`

**Files:**
- Modify: `src/vectoreyes/occlusion.py` (add `complete_polygon` after `complete_annulus`; add `_MAX_VERTICES` constant near `_MAX_RESIDUAL`)
- Test: `tests/test_occlusion.py` (append; uses a new diamond fixture helper)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_occlusion.py` (the file already imports `numpy as np` and constructs `Region` objects — reuse that style; if a `Region` import is not present at the top, add `from vectoreyes.types import Region`):

```python
def _diamond_mask(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (np.abs(xx - cx) + np.abs(yy - cy)) <= r          # L1 ball == axis-aligned diamond


def _disk_mask(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def test_complete_polygon_recovers_occluded_diamond():
    from vectoreyes.occlusion import complete_polygon, _MAX_RESIDUAL, _MAX_VERTICES
    h, w = 160, 220
    cx, cy, r = 80, 80, 46
    diamond = _diamond_mask(cx, cy, r, h, w) & ~_disk_mask(126, 80, 30, h, w)  # right corner bitten
    occluder = _disk_mask(126, 80, 30, h, w)
    region = Region(label=1, mask=diamond, color_hex="#3366cc")
    other = Region(label=2, mask=occluder, color_hex="#cc3333")
    prim = complete_polygon(region, [other], max_residual=_MAX_RESIDUAL, max_vertices=_MAX_VERTICES)
    assert prim is not None and prim["kind"] == "polygon"
    pts = prim["params"]["points"]
    assert len(pts) == 4
    # the four recovered corners are each within tolerance of a true diamond corner
    truth = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    for tx, ty in truth:
        assert min(np.hypot(px - tx, py - ty) for px, py in pts) <= 2.5


def test_complete_polygon_rejects_curved_disk_fragment():
    from vectoreyes.occlusion import complete_polygon, _MAX_RESIDUAL, _MAX_VERTICES
    h, w = 160, 200
    disk = _disk_mask(90, 80, 50, h, w) & ~_disk_mask(150, 80, 28, h, w)   # big disk, small bite
    occluder = _disk_mask(150, 80, 28, h, w)
    region = Region(label=1, mask=disk, color_hex="#3366cc")
    other = Region(label=2, mask=occluder, color_hex="#cc3333")
    # a curved boundary RDP-splits into more than _MAX_VERTICES straight edges -> rejected
    assert complete_polygon(region, [other], max_residual=_MAX_RESIDUAL, max_vertices=_MAX_VERTICES) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_occlusion.py -k complete_polygon -q`
Expected: FAIL with `ImportError: cannot import name 'complete_polygon'`.

- [ ] **Step 3: Implement `complete_polygon` and add the constant**

In `src/vectoreyes/occlusion.py`, add `_MAX_VERTICES = 8` immediately after the `_MAX_RESIDUAL = 1.6` line (in the constants block). Add the function after `complete_annulus`:

```python
def complete_polygon(
    region: Region, others: list[Region], *, max_residual: float, max_vertices: int
) -> dict | None:
    """Recover a convex polygon from a partially-occluded fragment. Fit a line to each
    visible (own-boundary) edge, then intersect consecutive lines cyclically — the
    wrap-around intersection of the last and first line recovers the corner hidden
    behind the occluder. Returns {"kind":"polygon","params":{"points":[(x,y),...]}}
    or None when the fragment is curved, under-constrained, or non-convex."""
    contour, seam = label_boundary(region, others)
    if len(contour) == 0 or not seam.any():
        return None                                    # no occlusion -> not this completer
    runs = _own_runs(contour, seam)
    if not runs:
        return None
    poly = max(runs, key=len)                          # the visible boundary arc
    simp = rdp(poly, max_residual)
    if len(simp) < 4:                                  # need >= 3 edges (incl. the 2 seam stubs)
        return None
    # map interior RDP vertices back to indices in `poly`; the two ends are the stubs
    idxs = [0]
    for v in simp[1:-1]:
        idxs.append(int(np.argmin(np.hypot(poly[:, 0] - v[0], poly[:, 1] - v[1]))))
    idxs.append(len(poly) - 1)
    lines = []
    for i in range(len(idxs) - 1):
        seg = poly[idxs[i]: idxs[i + 1] + 1]
        ln = _fit_line(seg)
        if ln is None or ln[3] > max_residual:
            return None
        lines.append(ln)
    if len(lines) < 3 or len(lines) > max_vertices:
        return None
    verts: list[tuple[float, float]] = []
    for i in range(len(lines)):
        p = _line_intersect(lines[i], lines[(i + 1) % len(lines)])
        if p is None:
            return None
        verts.append(p)
    if not _is_convex(verts):
        return None
    return {"kind": "polygon", "params": {"points": verts}}
```

Add `rdp` to the import from `.contour` at the top of the file. The current import is:

```python
from .contour import region_contours
```

Change it to:

```python
from .contour import rdp, region_contours
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -k complete_polygon -q`
Expected: PASS (2 passed).

If `test_complete_polygon_rejects_curved_disk_fragment` does NOT yield `None` (the curved arc happened to RDP-split into ≤ 8 edges), increase the disk radius in that test to `60` so the visible arc spans more edges — a curved boundary must exceed `_MAX_VERTICES` straight segments. Do not loosen `_MAX_VERTICES`.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): complete_polygon recovers a convex polygon from an occluded fragment"
```

---

### Task 4: `primitive_mask` polygon case

**Files:**
- Modify: `src/vectoreyes/occlusion.py` (`primitive_mask` ≈ lines 211-220; add the `skimage.draw` import)
- Test: `tests/test_occlusion.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_occlusion.py`:

```python
def test_primitive_mask_polygon_membership():
    from vectoreyes.occlusion import primitive_mask
    # a 10x10 axis-aligned square from (2,2) to (7,7)
    prim = {"kind": "polygon", "params": {"points": [(2.0, 2.0), (7.0, 2.0), (7.0, 7.0), (2.0, 7.0)]}}
    mask = primitive_mask(prim, 12, 12)
    assert mask[4, 4]            # (row=4, col=4) is inside
    assert not mask[0, 0]        # corner of the grid is outside
    assert mask.dtype == bool
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py -k primitive_mask_polygon -q`
Expected: FAIL — `primitive_mask` falls through to the ellipse branch and raises `KeyError: 'rx'` (a polygon prim has no `rx`).

- [ ] **Step 3: Implement the polygon case**

Add the import near the other skimage imports at the top of `src/vectoreyes/occlusion.py`:

```python
from skimage.draw import polygon2mask
```

In `primitive_mask`, add the polygon branch before the final ellipse `return` (after the `annulus` branch):

```python
    if prim["kind"] == "polygon":
        pts = prim["params"]["points"]
        rc = np.array([(y, x) for x, y in pts], dtype=float)   # skimage wants (row, col)
        return polygon2mask((h, w), rc)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_occlusion.py -k primitive_mask_polygon -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): primitive_mask rasterizes a polygon primitive"
```

---

### Task 5: Wire polygon into `_complete_member`

**Files:**
- Modify: `src/vectoreyes/occlusion.py` (`_complete_member` ≈ lines 174-181)
- Test: `tests/test_occlusion.py` (append)

Polygon becomes the **last** dispatch arm — after the curved (circle/ellipse) fitters, which reject straight edges. A disk/ring is still caught by the earlier arms; only genuinely straight-edged fragments reach `complete_polygon`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_occlusion.py` (reuses `_diamond_mask`/`_disk_mask` from Task 3):

```python
def test_complete_member_dispatches_diamond_to_polygon():
    from vectoreyes.occlusion import _complete_member
    h, w = 160, 220
    diamond = _diamond_mask(80, 80, 46, h, w) & ~_disk_mask(126, 80, 30, h, w)
    occluder = _disk_mask(126, 80, 30, h, w)
    region = Region(label=1, mask=diamond, color_hex="#3366cc")
    other = Region(label=2, mask=occluder, color_hex="#cc3333")
    prim = _complete_member(region, [other])
    assert prim is not None and prim["kind"] == "polygon"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py -k complete_member_dispatches_diamond -q`
Expected: FAIL — `_complete_member` returns `None` (the diamond matches neither annulus nor circle/ellipse, and the polygon arm does not exist yet).

- [ ] **Step 3: Add the polygon arm**

Replace the body of `_complete_member` in `src/vectoreyes/occlusion.py` with:

```python
def _complete_member(region: Region, others: list[Region]) -> dict | None:
    """Complete a group member: annulus if it has a hole, else circle/ellipse, else a
    convex polygon. The curved fitters run first (they reject straight edges), so only
    genuinely polygonal fragments reach complete_polygon."""
    ann = complete_annulus(region, others, max_residual=_MAX_RESIDUAL,
                           min_arc_deg=_MIN_ARC_DEG, concentric_tol=_CONCENTRIC_TOL)
    if ann is not None:
        return ann
    contour, seam = label_boundary(region, others)
    prim = complete_primitive(contour, seam, max_residual=_MAX_RESIDUAL, min_arc_deg=_MIN_ARC_DEG)
    if prim is not None:
        return prim
    return complete_polygon(region, others, max_residual=_MAX_RESIDUAL, max_vertices=_MAX_VERTICES)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_occlusion.py -k complete_member_dispatches_diamond -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): dispatch polygonal fragments to complete_polygon"
```

---

### Task 6: Acceptance — two overlapping diamonds & diamond × disk

**Files:**
- Create: `tests/test_acceptance_polygon.py`

Both fixtures go end-to-end through `idealize`. The real proof is a **recovered-vertex check** (the reconstructed polygon's corners lie within tolerance of the known geometry — a fallback fit would emit a path or a bite-distorted polygon); render SSIM is a faithful-render floor.

- [ ] **Step 1: Write the failing acceptance tests**

Create `tests/test_acceptance_polygon.py`:

```python
"""Polygon occlusion reconstruction, end-to-end through idealize.

A reconstructed convex polygon emits as a <polygon> whose vertices match the true
(un-occluded) corners. The per-region fallback instead emits a path or a polygon
distorted by the occluder's bite, so a recovered-vertex check distinguishes a real
reconstruction from a fallback. SSIM is a faithful-render sanity floor."""

import re

import numpy as np

from vectoreyes import Options, idealize
from tests._render import render_svg, ssim


def _paint(layers, h, w):
    img = np.full((h, w, 3), 255, np.uint8)
    for mask, color in layers:
        img[mask] = color
    return img


def _diamond(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (np.abs(xx - cx) + np.abs(yy - cy)) <= r


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _polygon_points(svg):
    """Every <polygon> element's vertices, as a list of (x, y) lists."""
    out = []
    for pstr in re.findall(r'<polygon[^>]*points="([^"]*)"', svg):
        pts = [tuple(float(v) for v in pair.split(",")) for pair in pstr.split()]
        out.append(pts)
    return out


def _corners_match(poly_pts, truth, tol=3.0):
    """True if every truth corner is within `tol` of some recovered vertex."""
    return all(min(np.hypot(px - tx, py - ty) for px, py in poly_pts) <= tol
               for tx, ty in truth)


_BLUE, _RED = (51, 102, 204), (204, 51, 51)


def test_two_overlapping_diamonds_reconstruct():
    h, w = 160, 220
    a = _diamond(80, 80, 46, h, w)
    b = _diamond(140, 80, 46, h, w)
    img = _paint([(a, _BLUE), (b, _RED)], h, w)        # B painted over A
    svg = idealize(img, options=Options())
    polys = _polygon_points(svg)
    assert len(polys) == 2                              # two clean convex polygons
    a_truth = [(80, 34), (126, 80), (80, 126), (34, 80)]
    b_truth = [(140, 34), (186, 80), (140, 126), (94, 80)]
    assert any(_corners_match(p, a_truth) for p in polys)
    assert any(_corners_match(p, b_truth) for p in polys)
    assert ssim(render_svg(svg, w, h), img) >= 0.95


def test_diamond_occluded_by_disk_reconstructs():
    h, w = 160, 200
    diamond = _diamond(80, 80, 46, h, w)
    disk = _disk(135, 80, 38, h, w)
    img = _paint([(diamond, _BLUE), (disk, _RED)], h, w)   # disk painted over diamond
    svg = idealize(img, options=Options())
    polys = _polygon_points(svg)
    assert len(polys) == 1                                 # the diamond, recovered
    assert svg.count("<circle") == 1                       # ...and the disk as a circle
    assert _corners_match(polys[0], [(80, 34), (126, 80), (80, 126), (34, 80)])
    assert ssim(render_svg(svg, w, h), img) >= 0.95
```

- [ ] **Step 2: Run the acceptance tests to verify they fail**

Run: `uv run pytest tests/test_acceptance_polygon.py -q`
Expected: FAIL — before the completer is wired end-to-end the diamonds fall back to per-region fits, so either `len(polys) != 2` / `!= 1`, the vertex check fails, or there are extra `<polygon>`s from a fallback. (If Tasks 1-5 are already merged on the branch, these may instead pass directly — that is fine; the point is they must pass after Step 3.)

- [ ] **Step 3: Make them pass**

No new production code should be required — Tasks 1-5 wired the full path. If a test fails:
- **Wrong polygon count / extra polygons:** the group is not forming or the gate is declining. Print `reconstruct_scene(...)`'s `remaining` in a scratch check; confirm the occluded diamond `has_bite` (it must be non-convex enough — the bite from the overlap provides this) and that `stack_agreement` ≥ `_GATE_AGREEMENT`. Nudge the fixture overlap (move centers 4-6px closer) so the bite is clear, rather than weakening any gate.
- **Vertex check fails by a hair:** widen `tol` to `3.5` (sub-pixel contour offset on 45° edges), not further.

- [ ] **Step 4: Run the acceptance tests to verify they pass**

Run: `uv run pytest tests/test_acceptance_polygon.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_acceptance_polygon.py
git commit -m "test(acceptance): polygon occlusion — overlapping diamonds and diamond x disk"
```

---

### Task 7: Decline a whole-edge-hidden polygon (safety contract) + full regression

**Files:**
- Modify: `tests/test_occlusion.py` (append the decline test)

When an entire edge of the polygon is hidden, the wrap-around intersection recovers a wrong corner; the completer cannot detect this, but the consistency gate must — `reconstruct_scene` declines the group and leaves the regions in `remaining` for faithful fallback. This test asserts the safety contract directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_occlusion.py` (reuses `_diamond_mask`/`_disk_mask`):

```python
def test_reconstruct_scene_declines_polygon_with_whole_edge_hidden():
    from vectoreyes.occlusion import reconstruct_scene
    h, w = 170, 200
    # a rectangle occluder covers the diamond's entire lower-right edge (corner (126,80)
    # to corner (80,126)) plus both its endpoints -> a whole edge is unrecoverable.
    yy, xx = np.ogrid[:h, :w]
    rect = (xx >= 86) & (xx <= 170) & (yy >= 78) & (yy <= 170)
    diamond = _diamond_mask(80, 80, 46, h, w) & ~rect
    region = Region(label=1, mask=diamond, color_hex="#3366cc")
    occ = Region(label=2, mask=rect, color_hex="#cc3333")
    reconstructed, remaining = reconstruct_scene([region, occ], None, (h, w))
    # the bitten diamond cannot be faithfully reconstructed -> it is declined (left to fallback)
    assert 1 in {r.label for r in remaining}
```

- [ ] **Step 2: Run the test to verify it fails or passes correctly**

Run: `uv run pytest tests/test_occlusion.py -k declines_polygon_with_whole_edge -q`
Expected: PASS if the gate correctly declines. If it FAILS (the diamond was reconstructed), the gate accepted a wrong polygon — inspect by printing the recovered vertices and `stack_agreement`; the fixture must hide a *whole* edge (verify the rect fully covers the segment between the two corners). Adjust the rect bounds so the entire lower-right edge is under it; do not change `_GATE_AGREEMENT`.

- [ ] **Step 3: (only if needed) tighten the fixture**

If the test does not yet demonstrate a decline, extend the rectangle so it unambiguously covers the whole lower-right edge and both adjacent corners (e.g. `(xx >= 80)`), keeping enough of the top-left edges visible that `complete_polygon` still returns *some* polygon (otherwise the test is vacuous). The assertion stays: `assert 1 in {r.label for r in remaining}`.

- [ ] **Step 4: Run the full suite (regression gate)**

Run as its own statement (never pipe pytest through `tail`/`head` before a commit — it masks the exit code):

```bash
uv run pytest -q
```
Expected: PASS — all prior tests green, including Mastercard, Daikonic, the annulus acceptance fixtures, and the Olympic-weave decline (all untouched), plus the new polygon tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_occlusion.py
git commit -m "test(occlusion): decline a polygon with a whole edge hidden (gate safety contract)"
```

---

## Self-Review

**1. Spec coverage:**
- Annulus/N-shape reuse, unchanged orchestration → Tasks 3-5 add only the completer + dispatch + mask; no orchestration edits. ✓
- `complete_polygon` (own/seam split, RDP edges, line fit, consecutive intersection, wrap = hidden corner, convexity + residual + vertex-count accept gate) → Task 3. ✓
- `_complete_member` polygon as last arm → Task 5. ✓
- `primitive_mask` polygon case via `polygon2mask` → Task 4. ✓
- Emit & pipeline unchanged → confirmed in File Structure (no tasks needed). ✓
- `_MAX_VERTICES = 8` constant → Task 3. ✓
- Unit tests (recover diamond, reject curved disk, convexity guard, primitive_mask membership) → Tasks 1, 3, 4. ✓
- Acceptance (two diamonds, diamond × disk; vertex check + SSIM ≥ 0.95) → Task 6. ✓
- Decline a whole-edge-hidden fragment via the gate → Task 7. ✓
- Regression (Mastercard, Daikonic, annulus, weave) → Task 7 full-suite run. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Every code step shows complete code; every command shows the expected result. ✓

**3. Type consistency:** `complete_polygon(region, others, *, max_residual, max_vertices)` is identical across Tasks 3, 5, and the tests. `_fit_line` returns a 4-tuple `(a, b, c, resid)`; `_line_intersect` reads indices `[0],[1],[2]` of that tuple — consistent. Polygon representation `{"kind":"polygon","params":{"points":[(x,y),…]}}` is identical in `complete_polygon`, `primitive_mask`, and the emit layer it feeds. `_own_runs(contour, seam)` signature matches its call in `complete_polygon`. ✓

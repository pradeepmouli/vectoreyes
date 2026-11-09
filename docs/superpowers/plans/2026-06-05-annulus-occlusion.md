# Annulus Occlusion Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the occlusion pass to reconstruct overlapping **annuli (rings)** and **N-shape adjacency groups** with an inferred global paint order, while keeping Mastercard (two circles + lens) working.

**Architecture:** Add an `"annulus"` primitive (two concentric circles, even-odd path). Add `complete_annulus` (fit outer + inner own arcs, concentricity-gated). Replace the `len(completed) != 2` gate with: complete every member → infer pairwise over/under from overlap ownership → topological sort to a global z-order (decline on a cycle) → existing consistency gate over the N-stack. The two-circle-with-lens case is retained as a guarded branch.

**Tech Stack:** Python, NumPy, `skimage` `CircleModel`, `scipy.ndimage`, the existing `occlusion.py` / `emit.py`.

**Spec:** `docs/superpowers/specs/2026-06-05-annulus-occlusion-design.md`

---

## File structure

- `src/vectoreyes/emit.py` — add `_annulus_path_d`; handle `"annulus"` in `shape_to_svg` and `shape_to_path_d`. (Pipeline needs no change — `_render_body` already routes `ScenePrimitive` through these.)
- `src/vectoreyes/occlusion.py` — `label_boundary(contour_index=…)`; extract `_fit_circle`; add `complete_annulus`; `primitive_mask` annulus case; `_pair_constraint` + `_topo_order`; rewrite `reconstruct_scene`. New constants `_CONCENTRIC_TOL`, `_MIN_OVERLAP_PX`, `_OVERLAP_OWNERSHIP`.
- `tests/test_emit.py` — annulus emit tests (create if absent).
- `tests/test_occlusion.py` — unit tests for the new occlusion functions.
- `tests/test_acceptance_annulus.py` — end-to-end acceptance + negative/regression (create).

---

### Task 1: Annulus SVG emit

**Files:**
- Modify: `src/vectoreyes/emit.py`
- Test: `tests/test_emit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_emit.py
import numpy as np

from vectoreyes.emit import shape_to_svg, shape_to_path_d
from vectoreyes.fit import Shape
from tests._render import render_svg


def _annulus_shape():
    return Shape("annulus", {"cx": 60.0, "cy": 60.0, "r_outer": 40.0, "r_inner": 22.0})


def test_annulus_svg_is_evenodd_path():
    svg = shape_to_svg(_annulus_shape(), "#3366cc", "s0")
    assert "<path" in svg and 'fill-rule="evenodd"' in svg
    # two subpaths (outer + inner): two move commands
    assert svg.count("M") == 2


def test_annulus_renders_hollow_center():
    d = shape_to_path_d(_annulus_shape())
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120">'
           f'<path fill="#000" fill-rule="evenodd" d="{d}"/></svg>')
    img = render_svg(svg, 120, 120)
    assert tuple(img[60, 60]) == (255, 255, 255)   # center is the hole -> white
    assert tuple(img[60, 26]) == (0, 0, 0)         # on the ring band -> black
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_emit.py -q`
Expected: FAIL — `shape_to_svg` raises `ValueError: unknown shape kind: annulus`.

- [ ] **Step 3: Implement**

In `src/vectoreyes/emit.py`, add the helper after `_ellipse_path_d`:

```python
def _annulus_path_d(cx: float, cy: float, r_outer: float, r_inner: float) -> str:
    """Two concentric circle subpaths (outer + inner); under even-odd fill the
    inner subpath cuts the hole."""
    return _ellipse_path_d(cx, cy, r_outer, r_outer) + " " + _ellipse_path_d(cx, cy, r_inner, r_inner)
```

In `shape_to_svg`, add before the `path` branch:

```python
    if shape.kind == "annulus":
        d = _annulus_path_d(p["cx"], p["cy"], p["r_outer"], p["r_inner"])
        return f'<path {common} fill-rule="evenodd" d="{d}"/>'
```

In `shape_to_path_d`, add before the final `raise`:

```python
    if shape.kind == "annulus":
        return _annulus_path_d(p["cx"], p["cy"], p["r_outer"], p["r_inner"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_emit.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/emit.py tests/test_emit.py
git commit -m "feat(emit): annulus as an even-odd two-circle path"
```

---

### Task 2: `label_boundary` per-contour

**Files:**
- Modify: `src/vectoreyes/occlusion.py:38-56`
- Test: `tests/test_occlusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_occlusion.py  (add)
import numpy as np
from vectoreyes.types import Region
from vectoreyes.occlusion import label_boundary


def _ring_region(label, cx, cy, r_out, r_in, h=120, w=120, color="#3366cc"):
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (d2 <= r_out ** 2) & (d2 >= r_in ** 2)
    return Region(label, mask, color)


def test_label_boundary_reads_inner_contour():
    ring = _ring_region(1, 60, 60, 40, 22)
    # a disk overlapping the ring's lower-right, touching both its outer and inner edges
    yy, xx = np.ogrid[:120, :120]
    occ = Region(2, ((xx - 85) ** 2 + (yy - 75) ** 2) <= 30 ** 2, "#cc3333")
    outer, outer_seam = label_boundary(ring, [occ], contour_index=0)
    inner, inner_seam = label_boundary(ring, [occ], contour_index=1)
    assert len(outer) > 0 and len(inner) > 0
    assert outer_seam.any()        # the occluder reaches the outer rim
    assert inner_seam.any()        # ...and the inner rim, only readable via contour_index=1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py::test_label_boundary_reads_inner_contour -q`
Expected: FAIL — `label_boundary() got an unexpected keyword argument 'contour_index'`.

- [ ] **Step 3: Implement**

Replace `label_boundary` in `src/vectoreyes/occlusion.py`:

```python
def label_boundary(
    region: Region, others: list[Region], *, reach: int = 2, contour_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (contour Nx2 as (x,y), seam_bool N) for the region's `contour_index`-th
    contour (0 = outer boundary, 1 = largest hole, ...). A contour point is a seam if
    any OTHER region's mask sits within `reach` px of it; else it is own boundary."""
    contours = region_contours(region.mask)
    if contour_index >= len(contours):
        return np.empty((0, 2)), np.empty((0,), bool)
    contour = contours[contour_index]
    if not others:
        return contour, np.zeros(len(contour), bool)
    near = np.zeros_like(region.mask)
    for o in others:
        near |= binary_dilation(o.mask, iterations=reach)
    h, w = region.mask.shape
    xs = np.clip(np.rint(contour[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.rint(contour[:, 1]).astype(int), 0, h - 1)
    seam = near[ys, xs]
    return contour, seam
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: PASS (new test + all existing occlusion tests, since the default `contour_index=0` preserves behaviour).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): label_boundary reads any contour (outer or hole)"
```

---

### Task 3: Extract `_fit_circle`

**Files:**
- Modify: `src/vectoreyes/occlusion.py` (`complete_primitive` circle branch)
- Test: `tests/test_occlusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_occlusion.py  (add)
from vectoreyes.occlusion import _fit_circle


def test_fit_circle_recovers_full_circle():
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    contour = np.column_stack([50 + 30 * np.cos(th), 60 + 30 * np.sin(th)])
    seam = np.zeros(len(contour), bool)
    fit = _fit_circle(contour, seam, max_residual=1.6, min_arc_deg=110.0)
    assert fit is not None
    assert abs(fit["cx"] - 50) < 1 and abs(fit["cy"] - 60) < 1 and abs(fit["r"] - 30) < 1


def test_fit_circle_rejects_short_arc():
    th = np.linspace(0, 0.3, 40)            # ~17 deg, far below min_arc_deg
    contour = np.column_stack([50 + 30 * np.cos(th), 60 + 30 * np.sin(th)])
    seam = np.zeros(len(contour), bool)
    assert _fit_circle(contour, seam, max_residual=1.6, min_arc_deg=110.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py::test_fit_circle_recovers_full_circle -q`
Expected: FAIL — `cannot import name '_fit_circle'`.

- [ ] **Step 3: Implement**

In `src/vectoreyes/occlusion.py`, add `_fit_circle` above `complete_primitive`:

```python
def _fit_circle(
    contour: np.ndarray, seam: np.ndarray, *, max_residual: float, min_arc_deg: float
) -> dict | None:
    """Fit a circle to the own-boundary points (convex hull, to drop concave inner
    arcs). Returns {"cx","cy","r"} or None if too few points, residual too large, or
    the own arc spans less than `min_arc_deg`."""
    own = np.asarray(contour, float)[~seam]
    if len(own) < 8:
        return None
    fit_pts = _fit_candidate_pts(own)
    if len(fit_pts) < 8:
        return None
    cm = CircleModel.from_estimate(fit_pts)
    if not cm or np.abs(cm.residuals(fit_pts)).max() > max_residual:
        return None
    cx, cy = float(cm.center[0]), float(cm.center[1])
    if _own_arc_span_deg(fit_pts, cx, cy) < min_arc_deg:
        return None
    return {"cx": cx, "cy": cy, "r": float(cm.radius)}
```

Then refactor `complete_primitive`'s circle branch to use it (DRY). Replace lines from `cm = CircleModel.from_estimate(fit_pts)` through the circle `return` with:

```python
    circ = _fit_circle(contour, seam, max_residual=max_residual, min_arc_deg=min_arc_deg)
    if circ is not None:
        cx, cy = circ["cx"], circ["cy"]
        seam_pts = np.asarray(contour, float)[seam]
        inside = len(seam_pts) == 0 or np.all(
            (seam_pts[:, 0] - cx) ** 2 + (seam_pts[:, 1] - cy) ** 2 <= (circ["r"] + max_residual) ** 2
        )
        if inside:
            return {"kind": "circle", "params": {"cx": cx, "cy": cy, "r": circ["r"]}}
```

(Leave the ellipse branch unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: PASS — new `_fit_circle` tests plus the existing `complete_primitive` tests (behaviour preserved).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "refactor(occlusion): extract _fit_circle (own-arc circle fit)"
```

---

### Task 4: `primitive_mask` annulus case

**Files:**
- Modify: `src/vectoreyes/occlusion.py:156-162`
- Test: `tests/test_occlusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_occlusion.py  (add)
from vectoreyes.occlusion import primitive_mask


def test_primitive_mask_annulus():
    prim = {"kind": "annulus", "params": {"cx": 60, "cy": 60, "r_outer": 40, "r_inner": 22}}
    m = primitive_mask(prim, 120, 120)
    assert not m[60, 60]            # hole
    assert m[60, 60 - 31]          # on the band (31 px out, between 22 and 40)
    assert not m[60, 60 - 50]      # outside the outer radius
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py::test_primitive_mask_annulus -q`
Expected: FAIL — annulus falls into the ellipse branch and raises `KeyError: 'rx'`.

- [ ] **Step 3: Implement**

Replace `primitive_mask` in `src/vectoreyes/occlusion.py`:

```python
def primitive_mask(prim: dict, h: int, w: int) -> np.ndarray:
    """Boolean mask of a completed circle / ellipse / annulus on an (h, w) grid."""
    yy, xx = np.ogrid[:h, :w]
    p = prim["params"]
    if prim["kind"] == "circle":
        return (xx - p["cx"]) ** 2 + (yy - p["cy"]) ** 2 <= p["r"] ** 2
    if prim["kind"] == "annulus":
        d2 = (xx - p["cx"]) ** 2 + (yy - p["cy"]) ** 2
        return (d2 <= p["r_outer"] ** 2) & (d2 >= p["r_inner"] ** 2)
    return ((xx - p["cx"]) / p["rx"]) ** 2 + ((yy - p["cy"]) / p["ry"]) ** 2 <= 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): primitive_mask handles annulus"
```

---

### Task 5: `complete_annulus`

**Files:**
- Modify: `src/vectoreyes/occlusion.py` (add `complete_annulus`, `_complete_member`, `_CONCENTRIC_TOL`)
- Test: `tests/test_occlusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_occlusion.py  (add)
from vectoreyes.occlusion import complete_annulus


def _occluded_ring(h=140, w=140):
    # a ring whose right rim is clipped by a disk that sits mostly outside it (so the
    # disk's own boundary — bordering background — spans well past min_arc_deg too)
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - 70) ** 2 + (yy - 70) ** 2
    ring = (d2 <= 45 ** 2) & (d2 >= 25 ** 2)
    occ = ((xx - 112) ** 2 + (yy - 70) ** 2) <= 24 ** 2
    return Region(1, ring & ~occ, "#3366cc"), Region(2, occ, "#cc3333")


def test_complete_annulus_recovers_ring():
    ring, occ = _occluded_ring()
    prim = complete_annulus(ring, [occ], max_residual=1.6, min_arc_deg=110.0, concentric_tol=2.0)
    assert prim is not None and prim["kind"] == "annulus"
    p = prim["params"]
    assert abs(p["cx"] - 70) < 2 and abs(p["cy"] - 70) < 2
    assert abs(p["r_outer"] - 45) < 2 and abs(p["r_inner"] - 25) < 2


def test_complete_annulus_rejects_solid_disk():
    yy, xx = np.ogrid[:120, :120]
    disk = Region(1, ((xx - 60) ** 2 + (yy - 60) ** 2) <= 40 ** 2, "#3366cc")
    assert complete_annulus(disk, [], max_residual=1.6, min_arc_deg=110.0, concentric_tol=2.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py::test_complete_annulus_recovers_ring -q`
Expected: FAIL — `cannot import name 'complete_annulus'`.

- [ ] **Step 3: Implement**

In `src/vectoreyes/occlusion.py`, add the constant near the others (`_MAX_RESIDUAL = 1.6` block):

```python
_CONCENTRIC_TOL = 2.0
```

Add `complete_annulus` after `complete_primitive`:

```python
def complete_annulus(
    region: Region, others: list[Region], *, max_residual: float, min_arc_deg: float,
    concentric_tol: float,
) -> dict | None:
    """Fit an annulus (two concentric circles) from a ring fragment's outer and inner
    own arcs. Returns {"kind":"annulus","params":{cx,cy,r_outer,r_inner}} or None when
    the region has no hole, either circle can't be fit, the radii don't nest, or the
    centres aren't concentric within `concentric_tol`."""
    if len(region_contours(region.mask)) < 2:
        return None                                       # no hole -> not a ring
    outer_c, outer_seam = label_boundary(region, others, contour_index=0)
    inner_c, inner_seam = label_boundary(region, others, contour_index=1)
    outer = _fit_circle(outer_c, outer_seam, max_residual=max_residual, min_arc_deg=min_arc_deg)
    inner = _fit_circle(inner_c, inner_seam, max_residual=max_residual, min_arc_deg=min_arc_deg)
    if outer is None or inner is None or inner["r"] >= outer["r"]:
        return None
    if np.hypot(outer["cx"] - inner["cx"], outer["cy"] - inner["cy"]) > concentric_tol:
        return None
    cx = (outer["cx"] + inner["cx"]) / 2
    cy = (outer["cy"] + inner["cy"]) / 2
    return {"kind": "annulus",
            "params": {"cx": cx, "cy": cy, "r_outer": outer["r"], "r_inner": inner["r"]}}


def _complete_member(region: Region, others: list[Region]) -> dict | None:
    """Complete a group member as an annulus if it has a hole, else as a circle/ellipse."""
    ann = complete_annulus(region, others, max_residual=_MAX_RESIDUAL,
                           min_arc_deg=_MIN_ARC_DEG, concentric_tol=_CONCENTRIC_TOL)
    if ann is not None:
        return ann
    contour, seam = label_boundary(region, others)
    return complete_primitive(contour, seam, max_residual=_MAX_RESIDUAL, min_arc_deg=_MIN_ARC_DEG)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): complete_annulus from outer+inner own arcs"
```

---

### Task 6: Pairwise over/under + topological order

**Files:**
- Modify: `src/vectoreyes/occlusion.py` (add `_pair_constraint`, `_topo_order`, `_MIN_OVERLAP_PX`, `_OVERLAP_OWNERSHIP`)
- Test: `tests/test_occlusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_occlusion.py  (add)
from vectoreyes.occlusion import _pair_constraint, _topo_order


def test_pair_constraint_from_overlap_ownership():
    # explicit, self-contained: a disk painted on top of a ring owns the overlap zone
    h = w = 140
    yy, xx = np.ogrid[:h, :w]
    ring_full = ((xx - 70) ** 2 + (yy - 70) ** 2 <= 45 ** 2) & ((xx - 70) ** 2 + (yy - 70) ** 2 >= 25 ** 2)
    disk_full = (xx - 105) ** 2 + (yy - 70) ** 2 <= 28 ** 2
    ring_vis = Region(1, ring_full & ~disk_full, "#3366cc")   # ring loses the overlap
    disk_vis = Region(2, disk_full, "#cc3333")                # disk keeps it (on top)
    ring_prim = {"kind": "annulus", "params": {"cx": 70, "cy": 70, "r_outer": 45, "r_inner": 25}}
    disk_prim = {"kind": "circle", "params": {"cx": 105, "cy": 70, "r": 28}}
    # i=ring, j=disk; the disk owns the overlap -> j_over_i
    assert _pair_constraint(ring_prim, ring_vis, disk_prim, disk_vis, h, w) == "j_over_i"


def test_topo_order_linear_and_cycle():
    assert _topo_order(3, [(0, 1), (1, 2)]) == [0, 1, 2]      # 0 under 1 under 2
    assert _topo_order(3, [(0, 1), (1, 2), (2, 0)]) is None   # cycle -> decline
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py::test_topo_order_linear_and_cycle -q`
Expected: FAIL — `cannot import name '_pair_constraint'`.

- [ ] **Step 3: Implement**

In `src/vectoreyes/occlusion.py`, add constants near `_MAX_RESIDUAL`:

```python
_MIN_OVERLAP_PX = 12
_OVERLAP_OWNERSHIP = 0.5
```

Add the functions (before `reconstruct_scene`):

```python
def _pair_constraint(
    prim_i: dict, region_i: Region, prim_j: dict, region_j: Region, h: int, w: int
) -> str | None:
    """Over/under for an overlapping pair, from which region owns the overlap zone in
    the raster: the colour that survives where the two completed shapes overlap is the
    one on top. Returns "i_over_j", "j_over_i", or None (no real overlap, or a
    distinct-coloured intersection owned by neither — e.g. a Mastercard lens)."""
    overlap = primitive_mask(prim_i, h, w) & primitive_mask(prim_j, h, w)
    n = int(overlap.sum())
    if n < _MIN_OVERLAP_PX:
        return None
    in_i = int((overlap & region_i.mask).sum())
    in_j = int((overlap & region_j.mask).sum())
    if max(in_i, in_j) < _OVERLAP_OWNERSHIP * n or in_i == in_j:
        return None
    return "i_over_j" if in_i > in_j else "j_over_i"


def _topo_order(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    """Kahn topological sort. `edges` are (under, over): under is painted first.
    Returns a paint order (z = position), or None if the constraints are cyclic."""
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    queue = [i for i in range(n) if indeg[i] == 0]
    order: list[int] = []
    while queue:
        node = queue.pop()
        order.append(node)
        for m in adj[node]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return order if len(order) == n else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): pairwise over/under + topological paint order"
```

---

### Task 7: Generalize `reconstruct_scene` to N shapes

**Files:**
- Modify: `src/vectoreyes/occlusion.py:207-289` (rewrite `reconstruct_scene`)
- Test: `tests/test_occlusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_occlusion.py  (add)
from vectoreyes.occlusion import reconstruct_scene


def test_reconstruct_ring_plus_disk():
    ring, occ = _occluded_ring()
    reconstructed, remaining = reconstruct_scene([ring, occ], None, (140, 140))
    kinds = sorted(p.kind for p in reconstructed if hasattr(p, "kind"))
    assert kinds == ["annulus", "circle"]
    assert remaining == []                       # both consumed


def test_reconstruct_declines_weave():
    # three mutually-interlocked rings whose overlap ownership is cyclic. Construct
    # it exactly: each ring keeps everything EXCEPT where the NEXT ring in the cycle
    # overlaps it (A loses to C, B loses to A, C loses to B) -> A-over-B, B-over-C,
    # C-over-A. No global paint order exists -> decline.
    h = w = 180
    yy, xx = np.ogrid[:h, :w]
    def ring(cx, cy):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return (d2 <= 40 ** 2) & (d2 >= 24 ** 2)
    a, b, c = ring(75, 75), ring(110, 75), ring(92, 110)
    regions = [Region(1, a & ~c, "#1111ee"),     # A over B, under C
               Region(2, b & ~a, "#eeee11"),     # B over C, under A
               Region(3, c & ~b, "#11aa11")]     # C over A, under B
    reconstructed, remaining = reconstruct_scene(regions, None, (h, w))
    assert reconstructed == []                    # cyclic -> declined
    assert len(remaining) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py::test_reconstruct_ring_plus_disk -q`
Expected: FAIL — current `reconstruct_scene` only completes circle/ellipse crescents and hard-gates at exactly two; the ring is not completed as an annulus.

- [ ] **Step 3: Implement**

Replace `reconstruct_scene` in `src/vectoreyes/occlusion.py` with:

```python
def reconstruct_scene(
    regions: list[Region], axis: Axis | None, shape_hw: tuple[int, int]
) -> tuple[list, list[Region]]:
    """Return (reconstructed, remaining). `reconstructed` mixes ScenePrimitive and
    lens Shape objects in paint order; `remaining` are regions to fit the old way.
    Only adjacent groups that complete, admit a global paint order, and pass the
    consistency gate are reconstructed."""
    h, w = shape_hw
    by_label = {r.label: r for r in regions}
    adj = region_adjacency(regions)
    reconstructed: list = []
    consumed: set[int] = set()

    for r in regions:
        if r.label in consumed or not has_bite(r.mask):
            continue
        group: set[int] = set()
        stack = [r.label]
        while stack:
            lab = stack.pop()
            if lab in group:
                continue
            group.add(lab)
            stack.extend(adj[lab] - group)
        group_regions = [by_label[l] for l in sorted(group) if l not in consumed]
        if len(group_regions) < 2:
            continue

        completed: list[tuple[Region, dict]] = []
        for gr in group_regions:
            others = [o for o in group_regions if o.label != gr.label]
            prim = _complete_member(gr, others)
            if prim is not None:
                completed.append((gr, prim))
        if len(completed) < 2:
            continue

        reg_list = [cr for cr, _ in completed]
        prim_list = [p for _, p in completed]
        completed_labels = {cr.label for cr in reg_list}
        leftover = [gr for gr in group_regions if gr.label not in completed_labels]

        # global paint order from pairwise overlap ownership
        edges: list[tuple[int, int]] = []
        for i in range(len(completed)):
            for j in range(i + 1, len(completed)):
                cons = _pair_constraint(prim_list[i], reg_list[i], prim_list[j], reg_list[j], h, w)
                if cons == "i_over_j":
                    edges.append((j, i))          # under=j, over=i
                elif cons == "j_over_i":
                    edges.append((i, j))
        order = _topo_order(len(completed), edges)
        if order is None:
            continue                              # cyclic -> weave -> decline
        z_of = {idx: k for k, idx in enumerate(order)}

        prims = [
            {"kind": prim_list[i]["kind"], "params": prim_list[i]["params"],
             "color": reg_list[i].color_hex, "z": z_of[i]}
            for i in range(len(completed))
        ]

        # Mastercard: exactly two circles + one distinct-coloured lens -> snap + lens
        lens = None
        lens_region = None
        if (len(completed) == 2 and prim_list[0]["kind"] == prim_list[1]["kind"] == "circle"
                and len(leftover) == 1):
            lens_region = leftover[0]
            if axis is not None:
                lo, hi = (0, 1) if prim_list[0]["params"]["cx"] <= prim_list[1]["params"]["cx"] else (1, 0)
                sl, sr = _snap_pair(prim_list[lo], prim_list[hi], axis.x)
                prims[lo]["params"] = sl["params"]
                prims[hi]["params"] = sr["params"]
            lens = {"mask_color": lens_region.color_hex, "lens_of": (0, 1)}

        if stack_agreement(prims, lens, group_regions, h, w) < _GATE_AGREEMENT:
            continue

        for p in sorted(prims, key=lambda p: p["z"]):
            reconstructed.append(ScenePrimitive(p["kind"], p["params"], p["color"], p["z"]))
        if lens is not None:
            d = intersection_lens_d(prims[0]["params"], prims[1]["params"])
            if d is not None:
                reconstructed.append(Shape("path", {"d": d, "color_hex": lens["mask_color"],
                                                    "z": len(prims)}))
        consumed.update(completed_labels)
        if lens_region is not None:
            consumed.add(lens_region.label)

    remaining = [r for r in regions if r.label not in consumed]
    return reconstructed, remaining
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: PASS — new ring+disk and weave-decline tests, plus all existing occlusion unit tests.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
git commit -m "feat(occlusion): N-shape reconstruction with inferred global z-order"
```

---

### Task 8: End-to-end acceptance + regression

**Files:**
- Test: `tests/test_acceptance_annulus.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_acceptance_annulus.py
import numpy as np

from vectoreyes import Options, idealize
from tests._render import render_svg, ssim


def _paint(layers, h=160, w=200):
    """layers: list of (mask, (r,g,b)) painted in order (later on top)."""
    img = np.full((h, w, 3), 255, np.uint8)
    for mask, color in layers:
        img[mask] = color
    return img


def _ring(cx, cy, r_out, r_in, h=160, w=200):
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 <= r_out ** 2) & (d2 >= r_in ** 2)


def _disk(cx, cy, r, h=160, w=200):
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2


def test_ring_plus_disk_reconstructs_and_renders():
    img = _paint([(_ring(80, 80, 50, 28), (51, 102, 204)),
                  (_disk(120, 80, 30), (204, 51, 51))])
    svg = idealize(img, options=Options())
    assert 'fill-rule="evenodd"' in svg                  # an annulus was emitted
    assert ssim(render_svg(svg, 200, 160), img) >= 0.95


def test_two_overlapping_rings_reconstruct():
    img = _paint([(_ring(75, 80, 45, 26), (51, 102, 204)),
                  (_ring(120, 80, 45, 26), (240, 200, 20))])
    svg = idealize(img, options=Options())
    assert svg.count('fill-rule="evenodd"') >= 2          # two annuli
    assert ssim(render_svg(svg, 200, 160), img) >= 0.95


def test_three_ring_consistent_stack_reconstructs():
    # painter's order left-to-right gives a clean acyclic z-order -> three annuli
    # (exercises the N=3 topological sort end-to-end)
    img = _paint([(_ring(60, 80, 40, 23), (51, 102, 204)),
                  (_ring(100, 80, 40, 23), (240, 200, 20)),
                  (_ring(140, 80, 40, 23), (20, 160, 60))])
    svg = idealize(img, options=Options())
    assert svg.count('fill-rule="evenodd"') >= 3          # three annuli
    assert ssim(render_svg(svg, 200, 160), img) >= 0.95


def test_mastercard_still_reconstructs():
    # two overlapping disks + a distinct-coloured lens (orange) painted on top
    yy, xx = np.ogrid[:160, :200]
    red = _disk(85, 80, 45)
    yellow = _disk(120, 80, 45)
    lens = red & yellow
    img = np.full((160, 200, 3), 255, np.uint8)
    img[red] = (224, 52, 75)
    img[yellow] = (240, 160, 20)
    img[lens] = (247, 158, 27)
    svg = idealize(img, options=Options())
    assert svg.count("<circle") >= 1 and "<use" in svg    # circle + mirror, unchanged
    assert ssim(render_svg(svg, 200, 160), img) >= 0.95
```

- [ ] **Step 2: Run test to verify it fails**

First run only the ring tests (Mastercard should already pass from prior tasks):
Run: `uv run pytest tests/test_acceptance_annulus.py -q`
Expected: the two ring tests are the meaningful check; they exercise emit + reconstruct end-to-end. If a ring test fails, it is a real integration gap — debug before proceeding. (They are expected to PASS given Tasks 1–7; this task is the end-to-end safety net.)

- [ ] **Step 3: Implement**

No new production code — this task verifies the integration of Tasks 1–7. If `test_ring_plus_disk_reconstructs_and_renders` fails because `_render_body` does not route the annulus correctly, confirm `shape_to_svg`/`shape_to_path_d` handle `"annulus"` (Task 1) — no pipeline edit should be required.

**Fixture-tuning note (not a placeholder):** these synthetic positions are chosen so each shape's own boundary spans well past `_MIN_ARC_DEG` (110°). If a completion fails to trigger, the cause is almost always a too-short own arc — move the occluder *outward* so more of the occluded shape's rim borders background, or shrink the occluder. Adjust positions, not thresholds; the gate and arc-span bars stay as the spec defines them.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — all prior tests, the new acceptance tests, and the existing Daikonic/Mastercard acceptance tests (regression).

- [ ] **Step 5: Commit**

```bash
git add tests/test_acceptance_annulus.py
git commit -m "test(acceptance): annulus occlusion end-to-end + Mastercard regression"
```

---

## Final verification

- [ ] Run the full suite: `uv run pytest -q` — all green.
- [ ] Manually confirm a ring+disk and a two-ring image idealize to annulus paths and render faithfully (the acceptance tests cover this).
- [ ] Confirm Daikonic and Mastercard acceptance tests are unchanged.

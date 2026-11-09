# Region-Level Symmetry Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect mirror symmetry from regions (axes emerge from region self-axes + pair bisectors, any orientation, multiple per figure) under an absolute area-aware boundary test, replacing the component-silhouette + tunable-IoU detector; reconstruct exactly about the primary axis using the existing machinery.

**Architecture:** New region-level entry point `detect_symmetry_groups(regions)` in `symmetry.py` built from four pure helpers (reflection test, axis proposals, clustering, classification). `_render_body` calls it once on all regions and threads the resulting per-region axis+role into the existing per-component reconstruct/merge/build path. Occlusion and merge keep their component pass; symmetry stops reading components.

**Tech Stack:** numpy, scipy.ndimage (distance transform, already used by `_axis_mismatch`), shapely only in tests (symmetry IoU assertions). Python ≥ 3.12, pure-Python, TDD.

## Global Constraints

- Python ≥ 3.12, pure-Python. `rg` not `grep`. Determinism preserved: all ordering value-based (θ, then offset, then label) — no dict/set iteration order or `Math.random`/time.
- **Absolute decision, not a threshold:** a region is symmetric about a line iff its reflected foreground lands off-shape by no more than a ~1px boundary band — `off_count ≤ K_BAND · perimeter`, `K_BAND = 1.5`. Reuse the resampling-free `_axis_mismatch` (reflect coordinates, distance-transform lookup); never whole-raster rotate a mask.
- **Detection-only:** reuse existing reconstruction (`reconstruct_scene`, `build_candidates`, `select_geometry`) unchanged. Reconstruct exactly about the **primary** axis (`axes[0]`); detect+report all axes; defer full dihedral folding.
- **Corpus non-regression is mandatory and is verified by a byte + diagnostic baseline/diff** over the corpus including `daikonic`. Byte-identical AND diagnostic-identical logos pass silently; changed logos are surfaced for human (not Haiku) case-by-case review. Known false positives `icloud` and `telegram` MUST stay rejected; `daikonic` MUST gain exact body symmetry (sym-IoU 1.0).
- Keep `detect_axis`/`classify_regions`/`detect_symmetry_rotation` in place (still imported elsewhere/by tests); this change adds the new path and rewires the one call site in `_render_body`.
- Commit trailer EXACTLY:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## File Structure

- `src/vectoreyes/symmetry.py` — add `Axis2D`, `_perimeter`, `reflection_off_count`, `region_is_self_symmetric`, `regions_mirror_pair`, `propose_axes`, `cluster_axes`, `detect_symmetry_groups`. Reuse existing `_axis_mismatch`.
- `src/vectoreyes/pipeline.py` — rewire the symmetry block of `_render_body`; extend diagnostics to report all axes.
- `tests/test_symmetry_regions.py` (new) — unit tests for every helper + the entry point.
- `tests/test_symmetry_corpus_regression.py` (new) — the byte + diagnostic baseline/diff gate.
- `scratch/sym_baseline.py` (new, NOT committed) — one-off baseline capture script.

---

### Task 1: `Axis2D` + resampling-free reflection off-count

**Files:**
- Modify: `src/vectoreyes/symmetry.py`
- Test: `tests/test_symmetry_regions.py` (create)

**Interfaces:**
- Consumes: existing `_axis_mismatch(fg_xy, cx, cy, theta, dist, *, tol_px=1.5)`.
- Produces:
  - `Axis2D = namedtuple("Axis2D", "theta cx cy")` — a mirror line at angle `theta` (radians, in `[0, pi)`) through point `(cx, cy)`.
  - `_perimeter(mask: np.ndarray) -> int` — boundary-pixel count `int((mask & ~binary_erosion(mask)).sum())`.
  - `reflection_off_count(fg_xy, axis: Axis2D, dist: np.ndarray, *, tol_px: float = 1.5) -> int` — number of foreground points whose reflection across `axis` lands farther than `tol_px` outside the shape whose background distance-transform is `dist`. (Like `_axis_mismatch` but returns the COUNT, not the mean.)

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from vectoreyes.symmetry import Axis2D, _perimeter, reflection_off_count
from scipy import ndimage as ndi

def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r

def test_perimeter_of_disk_is_boundary_band():
    m = _disk(60, 60, 30, 30, 20)
    p = _perimeter(m)
    assert 100 < p < 180   # ~2*pi*r, a one-pixel ring

def test_reflection_off_count_zero_for_symmetric_axis():
    m = _disk(60, 60, 30, 30, 20)           # disk: symmetric about every axis through center
    fg = np.nonzero(m)                       # (rows, cols) == (ys, xs)
    fg_xy = (fg[1], fg[0])                   # _axis_mismatch wants (xs, ys)
    dist = ndi.distance_transform_edt(~m)
    axis = Axis2D(theta=0.0, cx=30.0, cy=30.0)        # horizontal line through center
    assert reflection_off_count(fg_xy, axis, dist) == 0

def test_reflection_off_count_large_for_wrong_axis():
    # an L-shaped (asymmetric) region: reflection lands well off-shape
    m = np.zeros((60, 60), bool); m[10:50, 10:20] = True; m[40:50, 10:45] = True
    fg = np.nonzero(m); fg_xy = (fg[1], fg[0])
    dist = ndi.distance_transform_edt(~m)
    cy, cx = [c.mean() for c in fg[::-1]][::-1]   # centroid (cy, cx)
    axis = Axis2D(theta=np.pi / 2, cx=float(cx), cy=float(cy))  # vertical through centroid
    assert reflection_off_count(fg_xy, axis, dist) > _perimeter(m)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/test_symmetry_regions.py -x -q`
Expected: FAIL — `ImportError: cannot import name 'Axis2D'`.

- [ ] **Step 3: Implement**

Add to `src/vectoreyes/symmetry.py` (near the existing `_axis_mismatch`):

```python
from collections import namedtuple
from scipy import ndimage as _ndi

Axis2D = namedtuple("Axis2D", "theta cx cy")


def _perimeter(mask: np.ndarray) -> int:
    """Boundary-pixel count: the one-pixel ring of `mask` minus its erosion."""
    if not mask.any():
        return 0
    return int((mask & ~_ndi.binary_erosion(mask)).sum())


def reflection_off_count(fg_xy, axis: Axis2D, dist: np.ndarray, *, tol_px: float = 1.5) -> int:
    """Count foreground points whose reflection across `axis` lands farther than
    `tol_px` outside the shape (background distance transform `dist`). Resampling
    free: reflects coordinates, looks up the distance transform — no raster rotate."""
    xs, ys = fg_xy
    dx, dy = np.cos(axis.theta), np.sin(axis.theta)
    vx, vy = xs - axis.cx, ys - axis.cy
    t = vx * dx + vy * dy
    rx = axis.cx + (2.0 * t * dx - vx)
    ry = axis.cy + (2.0 * t * dy - vy)
    h, w = dist.shape
    ri = np.clip(np.rint(ry).astype(int), 0, h - 1)
    ci = np.clip(np.rint(rx).astype(int), 0, w - 1)
    return int((dist[ri, ci] > tol_px).sum())
```

- [ ] **Step 4: Run to verify it passes**

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/test_symmetry_regions.py -x -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/symmetry.py tests/test_symmetry_regions.py
git commit -m "feat(symmetry): Axis2D + resampling-free reflection off-count

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Absolute self-symmetry + mirror-pair tests

**Files:**
- Modify: `src/vectoreyes/symmetry.py`
- Test: `tests/test_symmetry_regions.py` (extend)

**Interfaces:**
- Consumes: `Axis2D`, `_perimeter`, `reflection_off_count` (Task 1); `Region` from `.types` (has `.mask` bool (H,W), `.label int`).
- Produces:
  - `K_BAND = 1.5` (module constant — the absolute boundary-band factor).
  - `region_is_self_symmetric(region, axis: Axis2D) -> bool` — `reflection_off_count(region_fg, axis, dist_of_region) ≤ K_BAND * _perimeter(region.mask)`.
  - `regions_mirror_pair(a, b, axis: Axis2D) -> bool` — `a` reflected across `axis` lands on `b`'s shape within the band: `reflection_off_count(a_fg, axis, dist_of_b) ≤ K_BAND * _perimeter(a.mask)` AND areas are within 10% (`min/max(area) ≥ 0.9`), so two different-size shapes are not called a pair.

- [ ] **Step 1: Write the failing test**

```python
from vectoreyes.symmetry import region_is_self_symmetric, regions_mirror_pair, Axis2D, K_BAND
from vectoreyes.types import Region

def _region(mask, label=1):
    return Region(label=label, mask=mask, color_hex="#000000")

def test_disk_is_self_symmetric_about_any_central_axis():
    m = _disk(80, 80, 40, 40, 25)
    for theta in (0.0, np.pi / 4, np.pi / 2):
        assert region_is_self_symmetric(_region(m), Axis2D(theta, 40.0, 40.0))

def test_asymmetric_region_is_not_self_symmetric():
    m = np.zeros((80, 80), bool); m[20:60, 20:30] = True; m[50:60, 20:55] = True  # L
    fg = np.nonzero(m); cy, cx = float(fg[0].mean()), float(fg[1].mean())
    assert not region_is_self_symmetric(_region(m), Axis2D(np.pi / 2, cx, cy))

def test_mirror_pair_detected_and_size_guarded():
    left = np.zeros((80, 80), bool); left[30:50, 15:25] = True
    right = np.zeros((80, 80), bool); right[30:50, 55:65] = True   # mirror of left about x=40
    axis = Axis2D(np.pi / 2, 40.0, 40.0)                            # vertical through x=40
    assert regions_mirror_pair(_region(left, 1), _region(right, 2), axis)
    big = np.zeros((80, 80), bool); big[20:60, 50:70] = True        # bigger -> not a pair
    assert not regions_mirror_pair(_region(left, 1), _region(big, 3), axis)
```

- [ ] **Step 2: Run to verify it fails** — `ImportError: region_is_self_symmetric`.

- [ ] **Step 3: Implement**

```python
K_BAND = 1.5   # absolute boundary-band factor: off-area <= K_BAND * perimeter (~1.5px band)


def _fg_xy(mask):
    ys, xs = np.nonzero(mask)
    return xs, ys


def region_is_self_symmetric(region, axis: Axis2D) -> bool:
    mask = region.mask
    if not mask.any():
        return False
    dist = _ndi.distance_transform_edt(~mask)
    off = reflection_off_count(_fg_xy(mask), axis, dist)
    return off <= K_BAND * _perimeter(mask)


def regions_mirror_pair(a, b, axis: Axis2D) -> bool:
    aa, ab = int(a.mask.sum()), int(b.mask.sum())
    if aa == 0 or ab == 0 or min(aa, ab) / max(aa, ab) < 0.9:
        return False
    dist_b = _ndi.distance_transform_edt(~b.mask)
    off = reflection_off_count(_fg_xy(a.mask), axis, dist_b)
    return off <= K_BAND * _perimeter(a.mask)
```

- [ ] **Step 4: Run to verify it passes** — PASS.

- [ ] **Step 5: Commit** `feat(symmetry): absolute area-aware self-symmetry + mirror-pair tests`.

---

### Task 3: Axis proposals from regions (self-axes + pair bisectors)

**Files:**
- Modify: `src/vectoreyes/symmetry.py`
- Test: `tests/test_symmetry_regions.py` (extend)

**Interfaces:**
- Consumes: Task 1–2 helpers; `Region`.
- Produces:
  - `AxisProposal = namedtuple("AxisProposal", "theta cx cy weight")` — a proposed line + supporting area weight.
  - `propose_axes(regions: list[Region], *, theta_steps: int = 12) -> list[AxisProposal]` —
    - **self-axes:** for each region, sweep `theta` over `np.linspace(0, np.pi, theta_steps, endpoint=False)` about the region centroid; for each θ where `region_is_self_symmetric`, append `AxisProposal(theta, cx, cy, area)`. (Coarse sweep; clustering in Task 4 merges near-duplicates — refinement is unnecessary because the absolute test already accepts a band of near-axes and the cluster centroid recovers the true angle.)
    - **pair-bisectors:** for each unordered region pair pruned to plausible partners (`0.9 ≤ min/max area ≤ 1.0` and centroid separation `≤ 0.5 * image diagonal`), the mirror line is the perpendicular bisector of the centroid segment: direction `d = (cb - ca)`, `theta = (atan2(d.y, d.x) + pi/2) mod pi`, through the midpoint; if `regions_mirror_pair(a, b, line)`, append `AxisProposal(theta, mid_x, mid_y, area_a + area_b)`.
    - Deterministic: regions iterated in `label` order; pairs in `(label_a, label_b)` order.

- [ ] **Step 1: Write the failing test**

```python
from vectoreyes.symmetry import propose_axes

def test_propose_axes_finds_vertical_for_centered_symmetric_region():
    m = _disk(100, 120, 50, 60, 30)                    # centered disk
    props = propose_axes([_region(m)])
    # a disk proposes many self-axes through its center; at least one ~vertical
    assert any(abs(p.theta - np.pi / 2) < 0.3 and abs(p.cx - 60) < 2 for p in props)

def test_propose_axes_finds_pair_bisector():
    left = np.zeros((80, 120), bool); left[30:50, 20:35] = True
    right = np.zeros((80, 120), bool); right[30:50, 85:100] = True   # mirror about x=60
    props = propose_axes([_region(left, 1), _region(right, 2)])
    assert any(abs(p.theta - np.pi / 2) < 0.2 and abs(p.cx - 60) < 3 for p in props)
```

- [ ] **Step 2: Run to verify it fails** — `ImportError: propose_axes`.

- [ ] **Step 3: Implement**

```python
AxisProposal = namedtuple("AxisProposal", "theta cx cy weight")


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def propose_axes(regions, *, theta_steps: int = 12):
    props: list[AxisProposal] = []
    ordered = sorted(regions, key=lambda r: r.label)
    # self-axes
    for r in ordered:
        if not r.mask.any():
            continue
        cx, cy = _centroid(r.mask)
        area = int(r.mask.sum())
        for theta in np.linspace(0.0, np.pi, theta_steps, endpoint=False):
            if region_is_self_symmetric(r, Axis2D(float(theta), cx, cy)):
                props.append(AxisProposal(float(theta), cx, cy, area))
    # pair-bisectors
    h, w = ordered[0].mask.shape if ordered else (1, 1)
    diag = float(np.hypot(h, w))
    cents = {r.label: _centroid(r.mask) for r in ordered}
    areas = {r.label: int(r.mask.sum()) for r in ordered}
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            aa, ab = areas[a.label], areas[b.label]
            if aa == 0 or ab == 0 or min(aa, ab) / max(aa, ab) < 0.9:
                continue
            (ax, ay), (bx, by) = cents[a.label], cents[b.label]
            if np.hypot(bx - ax, by - ay) > 0.5 * diag:
                continue
            theta = (np.arctan2(by - ay, bx - ax) + np.pi / 2) % np.pi
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            if regions_mirror_pair(a, b, Axis2D(float(theta), mx, my)):
                props.append(AxisProposal(float(theta), mx, my, aa + ab))
    return props
```

- [ ] **Step 4: Run to verify it passes** — PASS.

- [ ] **Step 5: Commit** `feat(symmetry): propose_axes — region self-axes + pair bisectors`.

---

### Task 4: Cluster proposals into axes

**Files:**
- Modify: `src/vectoreyes/symmetry.py`
- Test: `tests/test_symmetry_regions.py` (extend)

**Interfaces:**
- Consumes: `AxisProposal`, `Axis2D` (Task 3).
- Produces:
  - `cluster_axes(proposals, *, d_theta=0.09, d_offset=2.0, min_weight=1) -> list[tuple[Axis2D, float]]` — group proposals whose `(theta, signed_offset)` are within `d_theta` rad (~5°) and `d_offset` px; `signed_offset = cx*sin(theta) - cy*cos(theta)` (distance from origin to the line). Each cluster returns `(Axis2D at the area-weighted-mean theta & a point on the line, total_weight)`; drop clusters with `total_weight < min_weight`. Sorted by `-total_weight`, ties by `(theta, offset)`. Deterministic.

- [ ] **Step 1: Write the failing test**

```python
from vectoreyes.symmetry import cluster_axes, AxisProposal, Axis2D

def test_cluster_merges_near_duplicates_and_ranks_by_weight():
    props = [
        AxisProposal(np.pi/2, 60, 50, 100), AxisProposal(np.pi/2 + 0.02, 61, 50, 100),  # vertical, heavy
        AxisProposal(0.0, 60, 50, 30),                                                   # horizontal, light
    ]
    axes = cluster_axes(props)
    assert len(axes) == 2
    (a0, w0), (a1, w1) = axes
    assert w0 > w1 and abs(a0.theta - np.pi/2) < 0.05    # heaviest first == vertical
```

- [ ] **Step 2: Run to verify it fails** — `ImportError: cluster_axes`.

- [ ] **Step 3: Implement**

```python
def _offset(theta, cx, cy):
    return cx * np.sin(theta) - cy * np.cos(theta)


def cluster_axes(proposals, *, d_theta: float = 0.09, d_offset: float = 2.0, min_weight: int = 1):
    items = sorted(proposals, key=lambda p: (p.theta, _offset(p.theta, p.cx, p.cy)))
    clusters: list[list[AxisProposal]] = []
    for p in items:
        po = _offset(p.theta, p.cx, p.cy)
        for c in clusters:
            q = c[0]
            if abs(p.theta - q.theta) <= d_theta and abs(po - _offset(q.theta, q.cx, q.cy)) <= d_offset:
                c.append(p)
                break
        else:
            clusters.append([p])
    out = []
    for c in clusters:
        w = float(sum(p.weight for p in c))
        if w < min_weight:
            continue
        tw = sum(p.weight for p in c)
        theta = float(sum(p.theta * p.weight for p in c) / tw)
        cx = float(sum(p.cx * p.weight for p in c) / tw)
        cy = float(sum(p.cy * p.weight for p in c) / tw)
        out.append((Axis2D(theta, cx, cy), w))
    out.sort(key=lambda aw: (-aw[1], aw[0].theta, _offset(*aw[0])))
    return out
```

- [ ] **Step 4: Run to verify it passes** — PASS.

- [ ] **Step 5: Commit** `feat(symmetry): cluster_axes — merge proposals in (theta, offset) space`.

---

### Task 5: `detect_symmetry_groups` entry point

**Files:**
- Modify: `src/vectoreyes/symmetry.py`
- Test: `tests/test_symmetry_regions.py` (extend)

**Interfaces:**
- Consumes: `propose_axes`, `cluster_axes`, `region_is_self_symmetric`, `regions_mirror_pair`, `Axis2D`, `Region`.
- Produces:
  - `SymmetricGroup = namedtuple("SymmetricGroup", "axes straddlers pairs loners")` — `axes: list[Axis2D]` (primary first), `straddlers: list[Region]`, `pairs: list[tuple[Region, Region]]`, `loners: list[Region]`.
  - `detect_symmetry_groups(regions: list[Region]) -> list[SymmetricGroup]` —
    1. `axes = cluster_axes(propose_axes(regions))` → ranked `[(Axis2D, weight), ...]`.
    2. Greedy claim in weight order: for each axis, gather unclaimed regions that are `region_is_self_symmetric` (straddlers) or form a `regions_mirror_pair` with another unclaimed region (pairs); claim them. A region joins the FIRST (heaviest) axis that accepts it.
    3. Each axis that claimed ≥1 region becomes a group with `axes=[that axis]` for now (secondary axes that claim no NEW regions are appended to the group whose region-set they also satisfy — i.e. if a heavier axis already claimed all of a square's regions, a lighter orthogonal axis those same regions also satisfy is appended to that group's `axes`).
    4. All unclaimed regions become one trailing group `SymmetricGroup(axes=[], straddlers=[], pairs=[], loners=unclaimed)`.
    - Deterministic throughout (regions in label order; axes in weight order).

- [ ] **Step 1: Write the failing test**

```python
from vectoreyes.symmetry import detect_symmetry_groups

def test_radish_plus_text_isolates_symmetric_subset():
    # 3 centered symmetric bands (radish) + 1 off-center asymmetric blob (text)
    bands = []
    for i, y in enumerate((20, 45, 70)):
        m = np.zeros((140, 120), bool); m[y:y+18, 30:90] = True; bands.append(_region(m, i + 1))
    text = np.zeros((140, 120), bool); text[110:130, 10:40] = True  # off-axis, asymmetric placement
    groups = detect_symmetry_groups(bands + [_region(text, 9)])
    sym = [g for g in groups if g.axes]
    assert sym, "a symmetric group must be found"
    g = sym[0]
    assert abs(g.axes[0].theta - np.pi/2) < 0.1 and abs(g.axes[0].cx - 60) < 3
    assert len(g.straddlers) == 3
    claimed = {r.label for r in g.straddlers}
    assert 9 not in claimed   # the text region is NOT pulled into the symmetric group

def test_two_independent_symmetric_figures_two_groups():
    d1 = _disk(80, 200, 40, 40, 22); d2 = _disk(80, 200, 40, 160, 22)
    groups = [g for g in detect_symmetry_groups([_region(d1, 1), _region(d2, 2)]) if g.axes]
    # each disk is self-symmetric about its own centre -> at least two distinct axes/groups
    assert sum(len(g.straddlers) for g in groups) == 2

def test_determinism_repeated_runs_identical():
    m = _disk(80, 80, 40, 40, 25)
    a = detect_symmetry_groups([_region(m)])
    b = detect_symmetry_groups([_region(m)])
    assert [(g.axes, [r.label for r in g.straddlers]) for g in a] == \
           [(g.axes, [r.label for r in g.straddlers]) for g in b]
```

- [ ] **Step 2: Run to verify it fails** — `ImportError: detect_symmetry_groups`.

- [ ] **Step 3: Implement**

```python
SymmetricGroup = namedtuple("SymmetricGroup", "axes straddlers pairs loners")


def detect_symmetry_groups(regions):
    ordered = sorted(regions, key=lambda r: r.label)
    ranked = cluster_axes(propose_axes(ordered))
    claimed: set[int] = set()
    groups: list[dict] = []   # {"axes":[Axis2D], "straddlers":[], "pairs":[], "members":set}
    for axis, _w in ranked:
        avail = [r for r in ordered if r.label not in claimed]
        straddlers = [r for r in avail if region_is_self_symmetric(r, axis)]
        pairs = []
        used = {r.label for r in straddlers}
        rest = [r for r in avail if r.label not in used]
        for i, a in enumerate(rest):
            if a.label in used:
                continue
            for b in rest[i + 1:]:
                if b.label in used:
                    continue
                if regions_mirror_pair(a, b, axis):
                    pairs.append((a, b)); used.add(a.label); used.add(b.label); break
        members = {r.label for r in straddlers} | used
        if straddlers or pairs:
            claimed |= members
            groups.append({"axes": [axis], "straddlers": straddlers, "pairs": pairs, "members": members})
        else:
            # axis claims no NEW regions: if some existing group's straddlers ALL also
            # satisfy this axis, it is a secondary axis of that figure -> append.
            for g in groups:
                if g["straddlers"] and all(region_is_self_symmetric(r, axis) for r in g["straddlers"]):
                    g["axes"].append(axis)
                    break
    result = [SymmetricGroup(tuple(g["axes"]), g["straddlers"], g["pairs"], []) for g in groups]
    loners = [r for r in ordered if r.label not in claimed]
    result.append(SymmetricGroup((), [], [], loners))
    return result
```

- [ ] **Step 4: Run to verify it passes** — PASS (3 tests). Then run the WHOLE new test file: `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/test_symmetry_regions.py -q` → all green.

- [ ] **Step 5: Commit** `feat(symmetry): detect_symmetry_groups — region-level multi-axis detection`.

---

### Task 6: Wire `detect_symmetry_groups` into `_render_body`

**Files:**
- Modify: `src/vectoreyes/pipeline.py` (the symmetry block of `_render_body`, ~lines 246–284, and `IdealizeReport` axis diagnostics)
- Test: `tests/test_symmetry_regions.py` (extend with an end-to-end idealize assertion)

**Interfaces:**
- Consumes: `detect_symmetry_groups`, `SymmetricGroup`, `Axis2D` from `.symmetry`.
- Produces: unchanged `_render_body` return shape; `IdealizeReport.axes` now lists ALL detected axes (primary + secondary) mapped to output frame; `IdealizeReport.symmetry` per-region `(label, score, decision)` retained.

**Approach (read carefully — symmetry goes global, occlusion/merge stay per-component):**
1. Before the component loop, compute `groups = [] if opt.no_symmetry else detect_symmetry_groups(regions)`.
2. Build lookups keyed by region label:
   - `region_axis: dict[int, Axis2D]` = each straddler/pair member → its group's PRIMARY axis (`group.axes[0]`).
   - `region_role: dict[int, str]` = `"straddler"` / `"pair"` / `"loner"`.
   - `pair_partner: dict[int, int]` = both directions for each pair.
   - `all_axes: list[Axis2D]` = every `axis` in every group's `axes` (for diagnostics).
3. In the per-component loop, REPLACE `detect_axis(silhouette)` and `classify_regions(comp, axis)`:
   - The component's axis (for `reconstruct_scene`, which is vertical-only today) is the primary axis of the group its regions belong to **only if that axis is vertical within tolerance** (`abs(theta - pi/2) < 0.05`); otherwise pass `axis=None` to `reconstruct_scene` (a non-vertical primary still reconstructs via the existing rectify path inside `select_geometry`/`build_candidates`, which already receives the per-region `fit_axis`). Convert the chosen `Axis2D` to the existing vertical `Axis(x=cx)` for `reconstruct_scene`/`build_candidates` when vertical; else `Axis(x=None-equivalent)` per current `axis is None` path.
   - `straddlers = [r for r in comp if region_role.get(r.label) == "straddler"]`; `pairs` reassembled from `comp` members via `pair_partner` (both members in this comp); `loners = the rest`.
   - Append `(r.label, score, decision)` to `sym_diags` for each region from `region_role` (score = the off-count ratio, or reuse existing diag format).
4. `frame_axes`: for EACH axis in `all_axes`, append an `AxisLine` segment (use the silhouette y-extent for vertical axes; for non-vertical, a short segment centered at `(cx, cy)` along `theta`). Map through the affine as today.

> Note: keep `reconstruct_scene`'s existing signature. The minimal correct wiring is: where the old code had a single vertical `axis` per component, derive it from `region_axis` for that component's regions (they share a group); if the group's primary axis is non-vertical, treat the component as `axis=None` for the (vertical-only) scene reconstruction but still emit the per-region mirror in `build_candidates` via the existing `mirror=` path. This preserves all existing occlusion/merge behavior.

- [ ] **Step 1: Write the failing test (end-to-end daikonic gains exact symmetry)**

```python
import re
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely.affinity import scale as shp_scale
from vectoreyes.pipeline import idealize, Options

def _sym_iou_of_largest_body(svg, axis_x):
    ds = re.findall(r'<path[^>]*\bd="([^"]*)"', svg)
    # (test helper: flatten path, build polygon, reflect about axis_x, return IoU; see tests/_symhelp.py)
    from tests._symhelp import sym_iou_largest
    return sym_iou_largest(svg, axis_x)

def test_daikonic_body_is_exactly_symmetric_end_to_end():
    arr = np.asarray(Image.open("tests/fixtures/daikonic/source.png").convert("RGBA"))
    bg = Image.new("RGB", (arr.shape[1], arr.shape[0]), (255, 255, 255))
    im = Image.fromarray(arr); bg.paste(im, mask=im.split()[3])
    svg, rep = idealize(np.asarray(bg.convert("RGB"), np.uint8), options=Options(), report=True)
    assert rep.axes, "an axis must now be detected for the radish"
    assert _sym_iou_of_largest_body(svg, rep.axes[0].x1) >= 0.999
```

Add `tests/_symhelp.py` with `sym_iou_largest(svg, axis_x)` (flatten paths to polygons, take the largest on-axis shape, reflect about `axis_x`, return intersection/union) — the same geometry used during investigation.

- [ ] **Step 2: Run to verify it fails** — currently `rep.axes` is empty for daikonic → assertion fails.

- [ ] **Step 3: Implement** the wiring described above in `_render_body`.

- [ ] **Step 4: Run** the new test + the FULL suite: `PYTHONPATH=src ../../.venv/bin/python -m pytest -q --ignore=tests/test_mcp_server.py --ignore=tests/test_mcp_image.py`. The daikonic test PASSES. Other goldens MAY move (that is Task 7's gate) — if any non-symmetry test (occlusion/gradient) breaks, STOP and report.

- [ ] **Step 5: Commit** `feat(pipeline): region-level symmetry in _render_body (multi-axis, primary reconstruct)`.

---

### Task 7: Corpus byte + diagnostic regression gate (incl. daikonic)

**Files:**
- Create: `scratch/sym_baseline.py` (NOT committed), `tests/test_symmetry_corpus_regression.py`
- Test: itself.

**Interfaces:**
- Consumes: `idealize(..., report=True)`; the corpus images under `scratch/real-logos/*.png` PLUS `tests/fixtures/daikonic/source.png`.

**Approach:** Capture a baseline of (SVG bytes hash, symmetry diagnostics) **from the pre-change base commit** for all corpus logos, store as a JSON fixture, then a test re-runs current code and diffs. Logos that differ are reported (not auto-failed) so a human adjudicates case-by-case; but the test HARD-fails on the two non-negotiables.

- [ ] **Step 1:** Write `scratch/sym_baseline.py` that, run on the BASE commit (before Task 6), writes `tests/fixtures/sym_baseline.json`: `{name: {"svg_sha": sha256(svg), "axes": [...], "decisions": [(label, decision), ...]}}` for every corpus logo + daikonic. Run it on base, commit ONLY the JSON.

- [ ] **Step 2: Write the test**

```python
import hashlib, json, numpy as np
from pathlib import Path
from PIL import Image
from vectoreyes.pipeline import idealize, Options

CORPUS = Path("scratch/real-logos")
BASE = json.loads(Path("tests/fixtures/sym_baseline.json").read_text())

def _load(p):
    im = Image.open(p)
    if im.mode == "RGBA":
        b = Image.new("RGB", im.size, (255, 255, 255)); b.paste(im, mask=im.split()[3]); im = b
    return np.asarray(im.convert("RGB"), np.uint8)

def test_corpus_symmetry_changes_are_only_the_intended_ones():
    changed = []
    for name, base in BASE.items():
        src = CORPUS / f"{name}.png"
        path = src if src.exists() else Path("tests/fixtures/daikonic/source.png")
        svg = idealize(_load(path), options=Options())
        if hashlib.sha256(svg.encode()).hexdigest() != base["svg_sha"]:
            changed.append(name)
    # NON-NEGOTIABLES (hard gate):
    assert "icloud" not in changed or _still_rejected("icloud")   # no NEW false symmetry
    assert "telegram" not in changed or _still_rejected("telegram")
    assert "daikonic" in changed                                  # daikonic MUST improve
    # everything else: report for human case-by-case review
    print("CHANGED (review case-by-case):", sorted(changed))
```

Add `_still_rejected(name)` = re-run with report and assert no straddler/pair decision appears for that mark's main region (its symmetry stays rejected). Keep the helper small and explicit.

- [ ] **Step 3: Run** — `PYTHONPATH=src ../../.venv/bin/python -m pytest tests/test_symmetry_corpus_regression.py -q -s`. Read the CHANGED list aloud in the report.

- [ ] **Step 4:** For every name in CHANGED, render `input | base | new` and review (human, not Haiku). Adjudicate case-by-case per the spec; fix real regressions, accept intended gains. Record the verdicts in the task report.

- [ ] **Step 5: Commit** `test(symmetry): corpus byte+diagnostic regression gate (incl. daikonic)`.

---

## Notes
- `reconstruct_scene` and `build_candidates` are reused unchanged; if Task 6 reveals their signatures cannot express a non-vertical primary axis without edits, STOP and report before modifying them (that would expand scope past detection-only).
- `K_BAND`, `theta_steps`, `d_theta`, `d_offset` are grounded defaults, not tuning knobs — do not sweep them to make a corpus logo pass; if a logo needs a different value, that is a finding to report, not a silent change.

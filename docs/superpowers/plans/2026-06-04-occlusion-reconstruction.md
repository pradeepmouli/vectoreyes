# Occlusion Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct overlapping flat shapes (e.g. Mastercard's two disks) as a z-ordered stack of completed primitives + an explicit overlap shape, instead of leaving them as the crescents segmentation returns.

**Architecture:** A new `reconstruct_scene` pass between `segment()` and the per-region fit loop. For each region with a concave "bite", label its contour own-vs-seam, fit a circle/ellipse to the own boundary only (completing it across the seam), derive the overlap lens, and emit a z-ordered stack — but only if re-rendering the stack reproduces the segmentation (consistency gate). Everything else passes through to today's `_fit_region` unchanged.

**Tech Stack:** Python 3.13, numpy, scikit-image (`CircleModel`, `EllipseModel`, `label`, binary dilation via `scipy.ndimage`), shapely (already a dep), pytest, `uv`.

**Reference:** Design spec at `docs/superpowers/specs/2026-06-04-occlusion-reconstruction-design.md`.

**Conventions (read before starting):**
- Every new source file starts with `# SPDX-License-Identifier: MIT` (the `packages`/lib code is MIT).
- Run tests with `uv run pytest`. NEVER pipe pytest through `tail`/`head` before a `&&` commit — the pipe masks the failing exit code. Run `uv run pytest -q` as its own statement and read the result, or use `uv run pytest ... && git commit`.
- Commit with `SKIP_SIMPLE_GIT_HOOKS=1 git commit`. End commit messages with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- Existing types: `Region(label:int, mask:bool[H,W], color_hex:str)` with `.area`; `Axis(x:float)` with `.reflect_x(x)`. Both in `src/vectoreyes/types.py`.
- `region_contours(mask) -> list[np.ndarray]` (each (N,2) as (x,y), outer first) in `contour.py`.
- `Shape(kind:str, params:dict)` in `fit.py`; `shape_to_svg(shape, fill, elem_id)` and `path_svg(d, fill, fill_rule=None)` in `emit.py`.

---

## Decisions baked in (read once)

- **Acceptance fixture is synthetic**, not the real Mastercard PNG (trademark/license + determinism). A helper draws two overlapping disks of distinct colors with a third color painted in the overlap. Real Mastercard is a manual smoke check only.
- **The consistency gate renders analytically** (numpy disk/ellipse masks), not through resvg — no new runtime dependency.
- **Completed mirror pairs are emitted as two exactly-symmetric `<circle>`s**, not `<circle>+<use>`. The spec's `<use>` form is a later DRY optimization; the hard requirement is *exact symmetry* + *editable primitives*, both satisfied by two snapped circles. This is a conscious, documented narrowing of the spec.
- **The overlap lens is emitted with SVG `A` (elliptical-arc) commands** (two arcs). `A` never reaches `reflect_path_d` (only pairs are reflected, and the lens is not a pair), so no tokenizer change is needed. `shape_to_path_d` passes `kind=="path"` through unchanged, so `--flatten` already works for the lens.

---

## File Structure

- **Create `src/vectoreyes/occlusion.py`** — the whole subsystem: `ScenePrimitive`, adjacency, bite detection, boundary labelling, the circle/ellipse completer, the lens builder, `reconstruct_scene`, and the analytic consistency gate. One cohesive module (shares small helpers); split later if it grows past ~250 lines.
- **Modify `src/vectoreyes/pipeline.py`** — call `reconstruct_scene` in `idealize`; emit returned `ScenePrimitive`s and lens `Shape`s in the draw loop (they already carry geometry); pass un-reconstructed regions through the existing path.
- **Create `tests/test_occlusion.py`** — unit tests for every component.
- **Create `tests/test_acceptance_occlusion.py`** — the synthetic-Mastercard gating test + the Daikonic-unchanged negative test.

---

## Task 1: ScenePrimitive type + region adjacency

**Files:**
- Create: `src/vectoreyes/occlusion.py`
- Test: `tests/test_occlusion.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_occlusion.py
import numpy as np
from vectoreyes.types import Region
from vectoreyes.occlusion import ScenePrimitive, region_adjacency


def _region(label, mask, color="#000000"):
    return Region(label, mask.astype(bool), color)


def test_scene_primitive_holds_geometry():
    p = ScenePrimitive(kind="circle", params={"cx": 1.0, "cy": 2.0, "r": 3.0}, color_hex="#FF0000", z=0)
    assert p.kind == "circle" and p.params["r"] == 3.0 and p.color_hex == "#FF0000" and p.z == 0


def test_region_adjacency_touching_vs_separate():
    a = np.zeros((20, 30), bool); a[5:15, 4:14] = True
    b = np.zeros((20, 30), bool); b[5:15, 14:24] = True   # shares the x=14 seam with a
    c = np.zeros((20, 30), bool); c[5:15, 26:29] = True   # gap from b
    adj = region_adjacency([_region(1, a), _region(2, b), _region(3, c)])
    assert adj[1] == {2} and adj[2] == {1} and adj[3] == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vectoreyes.occlusion'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/vectoreyes/occlusion.py
# SPDX-License-Identifier: MIT
"""Occlusion reconstruction: explain overlapping regions as a z-ordered stack of
completed primitives (see docs/superpowers/specs/2026-06-04-occlusion-reconstruction-design.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from .types import Region


@dataclass
class ScenePrimitive:
    """A completed shape that may be partially occluded by higher-z primitives."""
    kind: str                 # "circle" | "ellipse"
    params: dict
    color_hex: str
    z: int


def region_adjacency(regions: list[Region]) -> dict[int, set[int]]:
    """label -> set of labels whose masks touch it (8-connectivity, 1px dilation)."""
    adj: dict[int, set[int]] = {r.label: set() for r in regions}
    dilated = {r.label: binary_dilation(r.mask) for r in regions}
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            if (dilated[a.label] & b.mask).any():
                adj[a.label].add(b.label)
                adj[b.label].add(a.label)
    return adj
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_occlusion.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
SKIP_SIMPLE_GIT_HOOKS=1 git commit -m "feat(occlusion): ScenePrimitive type + region adjacency

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Bite (concavity) detection

**Files:**
- Modify: `src/vectoreyes/occlusion.py`
- Test: `tests/test_occlusion.py`

A crescent (a disk with another disk subtracted) has low *solidity* (region area / convex-hull area); a disk, trapezoid, or dome is convex (solidity ≈ 1). Solidity is the cheap trigger; the completer + gate do the real validation.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_occlusion.py
from vectoreyes.occlusion import has_bite


def test_has_bite_crescent_vs_convex():
    H = W = 80
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - 35) ** 2 + (yy - 40) ** 2 <= 28 ** 2
    occ = (xx - 60) ** 2 + (yy - 40) ** 2 <= 28 ** 2
    crescent = disk & ~occ
    assert has_bite(crescent) is True
    assert has_bite(disk) is False                      # convex
    rect = np.zeros((H, W), bool); rect[10:70, 10:40] = True
    assert has_bite(rect) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py::test_has_bite_crescent_vs_convex -q`
Expected: FAIL — `ImportError: cannot import name 'has_bite'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add imports at top of occlusion.py
from skimage.morphology import convex_hull_image

# add function
def has_bite(mask: np.ndarray, *, max_solidity: float = 0.92) -> bool:
    """True when the region is non-convex enough to be a plausible occluded fragment
    (a crescent), i.e. its solidity (area / convex-hull area) is below the bar."""
    area = int(mask.sum())
    if area == 0:
        return False
    hull = int(convex_hull_image(mask).sum())
    return hull > 0 and (area / hull) < max_solidity
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_occlusion.py::test_has_bite_crescent_vs_convex -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
SKIP_SIMPLE_GIT_HOOKS=1 git commit -m "feat(occlusion): bite (concavity) detection via solidity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Own-vs-seam boundary labelling

**Files:**
- Modify: `src/vectoreyes/occlusion.py`
- Test: `tests/test_occlusion.py`

For a region's outer contour, a point is a **seam** if some *other* region's mask is within ~2px of it (the contour borders another region there), else it's **own** (borders background).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_occlusion.py
from vectoreyes.occlusion import label_boundary


def test_label_boundary_marks_the_bite_as_seam():
    H = W = 90
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - 38) ** 2 + (yy - 45) ** 2 <= 30 ** 2
    occ = (xx - 66) ** 2 + (yy - 45) ** 2 <= 30 ** 2
    crescent = disk & ~occ
    other = occ & ~disk                                  # the occluding region's visible part
    contour, seam = label_boundary(_region(1, crescent), [_region(2, other)])
    assert contour.shape[1] == 2 and seam.dtype == bool and len(seam) == len(contour)
    # seam points lie on the right (toward the occluder); own points on the left rim
    seam_x = contour[seam][:, 0].mean()
    own_x = contour[~seam][:, 0].mean()
    assert seam_x > own_x                                # the bite is on the +x side
    assert seam.any() and (~seam).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py::test_label_boundary_marks_the_bite_as_seam -q`
Expected: FAIL — `ImportError: cannot import name 'label_boundary'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add imports
from .contour import region_contours

# add function
def label_boundary(
    region: Region, others: list[Region], *, reach: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Return (outer_contour Nx2 as (x,y), seam_bool N). A contour point is a seam
    if any OTHER region's mask sits within `reach` px of it; else it is own boundary."""
    contours = region_contours(region.mask)
    if not contours:
        return np.empty((0, 2)), np.empty((0,), bool)
    contour = contours[0]
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

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_occlusion.py::test_label_boundary_marks_the_bite_as_seam -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
SKIP_SIMPLE_GIT_HOOKS=1 git commit -m "feat(occlusion): own-vs-seam boundary labelling

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Circle/ellipse completer

**Files:**
- Modify: `src/vectoreyes/occlusion.py`
- Test: `tests/test_occlusion.py`

Fit a circle (then an ellipse) to the **own-boundary points only**. Accept iff the residual is tight, the own arc spans enough angle to constrain the fit, and the seam points lie inside the completed shape (consistent with being occluded).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_occlusion.py
from vectoreyes.occlusion import complete_primitive


def test_complete_primitive_recovers_full_disk_from_crescent():
    H = W = 120
    yy, xx = np.ogrid[:H, :W]
    cx0, cy0, r0 = 45.0, 60.0, 34.0
    disk = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= r0 ** 2
    occ = (xx - 78) ** 2 + (yy - 60) ** 2 <= 34 ** 2
    crescent = disk & ~occ
    other = occ & ~disk
    contour, seam = label_boundary(_region(1, crescent), [_region(2, other)])
    prim = complete_primitive(contour, seam, max_residual=1.5, min_arc_deg=120.0)
    assert prim is not None and prim["kind"] == "circle"
    assert abs(prim["params"]["cx"] - cx0) < 2.0
    assert abs(prim["params"]["cy"] - cy0) < 2.0
    assert abs(prim["params"]["r"] - r0) < 2.0


def test_complete_primitive_rejects_when_own_arc_too_short():
    # almost everything is seam -> not enough own boundary to constrain a circle
    contour = np.array([[0, 0], [1, 0], [2, 1], [2, 2], [1, 3], [0, 3]], float)
    seam = np.ones(len(contour), bool); seam[0] = False
    assert complete_primitive(contour, seam, max_residual=1.5, min_arc_deg=120.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py -k complete_primitive -q`
Expected: FAIL — `ImportError: cannot import name 'complete_primitive'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add imports
from skimage.measure import CircleModel, EllipseModel

# add helper + function
def _own_arc_span_deg(own_pts: np.ndarray, cx: float, cy: float) -> float:
    """Angular span (deg) of the own points about (cx, cy). Full circle -> ~360."""
    ang = np.sort(np.arctan2(own_pts[:, 1] - cy, own_pts[:, 0] - cx))
    if len(ang) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    return float(np.degrees(2 * np.pi - gaps.max()))     # span covered = full minus largest gap


def complete_primitive(
    contour: np.ndarray, seam: np.ndarray, *, max_residual: float, min_arc_deg: float
) -> dict | None:
    """Fit a circle (then ellipse) to the OWN-boundary points, completing across the
    seam. Returns {"kind","params"} or None when the own arc can't constrain a fit."""
    own = np.asarray(contour, float)[~seam]
    if len(own) < 8:
        return None
    cm = CircleModel.from_estimate(own)
    if cm and np.abs(cm.residuals(own)).max() <= max_residual:
        cx, cy = float(cm.center[0]), float(cm.center[1])
        if _own_arc_span_deg(own, cx, cy) >= min_arc_deg:
            seam_pts = np.asarray(contour, float)[seam]
            inside = len(seam_pts) == 0 or np.all(
                (seam_pts[:, 0] - cx) ** 2 + (seam_pts[:, 1] - cy) ** 2 <= (cm.radius + max_residual) ** 2
            )
            if inside:
                return {"kind": "circle", "params": {"cx": cx, "cy": cy, "r": float(cm.radius)}}
    em = EllipseModel.from_estimate(own)
    if em and np.abs(em.residuals(own)).max() <= max_residual:
        xc, yc = float(em.center[0]), float(em.center[1])
        a, b = (float(v) for v in em.axis_lengths)
        if abs(em.theta) < 0.08 or abs(abs(em.theta) - np.pi) < 0.08:
            if _own_arc_span_deg(own, xc, yc) >= min_arc_deg:
                return {"kind": "ellipse", "params": {"cx": xc, "cy": yc, "rx": a, "ry": b}}
    return None
```

> Note: `CircleModel.from_estimate` returns `(model, inliers)` in newer skimage and is truthy on success; `recognize_primitive` in `fit.py` already uses this exact idiom — match it (`cm = CircleModel.from_estimate(own)` then `if cm and ...`). If the installed skimage returns a tuple, mirror whatever `fit.py` does.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_occlusion.py -k complete_primitive -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
SKIP_SIMPLE_GIT_HOOKS=1 git commit -m "feat(occlusion): circle/ellipse completer from own-boundary arc

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Two-circle intersection lens path

**Files:**
- Modify: `src/vectoreyes/occlusion.py`
- Test: `tests/test_occlusion.py`

The overlap of two completed circles is a lens bounded by two arcs. Emit it as an SVG path of two elliptical-arc (`A`) commands between the circle–circle intersection points.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_occlusion.py
import re
from vectoreyes.occlusion import intersection_lens_d


def test_intersection_lens_path_two_arcs_between_crossing_points():
    # two unit-ish circles overlapping along x; intersection points at x=5, y=±h
    a = {"cx": 0.0, "cy": 0.0, "r": 6.0}
    b = {"cx": 8.0, "cy": 0.0, "r": 6.0}
    d = intersection_lens_d(a, b)
    assert d is not None
    assert d.count("A") == 2 and d.startswith("M") and d.rstrip().endswith("Z")
    # the two crossing points are symmetric about y=0 at x=4 (midpoint of centers)
    nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", d)]
    assert any(abs(n - 4.0) < 0.5 for n in nums)


def test_intersection_lens_none_when_disjoint():
    a = {"cx": 0.0, "cy": 0.0, "r": 3.0}
    b = {"cx": 20.0, "cy": 0.0, "r": 3.0}
    assert intersection_lens_d(a, b) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py -k intersection_lens -q`
Expected: FAIL — `ImportError: cannot import name 'intersection_lens_d'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add import
from .fit import _fmt

def intersection_lens_d(a: dict, b: dict) -> str | None:
    """SVG path (two A arcs) for the lens = intersection of circles a and b.
    Returns None if the circles don't properly overlap (disjoint or one contains
    the other). Params dicts use cx, cy, r."""
    ax, ay, ar = a["cx"], a["cy"], a["r"]
    bx, by, br = b["cx"], b["cy"], b["r"]
    dx, dy = bx - ax, by - ay
    dist = float(np.hypot(dx, dy))
    if dist <= abs(ar - br) or dist >= ar + br or dist == 0:
        return None                                       # nested or disjoint
    # distance from a-centre to the chord, then the half-chord height
    t = (dist * dist + ar * ar - br * br) / (2 * dist)
    h2 = ar * ar - t * t
    if h2 <= 0:
        return None
    hh = float(np.sqrt(h2))
    ux, uy = dx / dist, dy / dist                          # unit a->b
    mx, my = ax + t * ux, ay + t * uy                      # chord midpoint
    p1 = (mx - hh * (-uy), my - hh * ux)                   # one crossing
    p2 = (mx + hh * (-uy), my + hh * ux)                   # the other
    f = _fmt
    # arc of A from p1 to p2 bulging toward b, then arc of B from p2 to p1 bulging
    # toward a. large-arc-flag=0 (minor arcs); sweep flags chosen for the lens.
    return (
        f"M{f(p1[0])} {f(p1[1])} "
        f"A{f(ar)} {f(ar)} 0 0 1 {f(p2[0])} {f(p2[1])} "
        f"A{f(br)} {f(br)} 0 0 1 {f(p1[0])} {f(p1[1])} Z"
    )
```

> If a rendered check later shows the lens bulging the wrong way, flip the two sweep flags (the `0 0 1` → `0 0 0`); the geometry (crossing points) is correct regardless. Verify visually in Task 8.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_occlusion.py -k intersection_lens -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
SKIP_SIMPLE_GIT_HOOKS=1 git commit -m "feat(occlusion): two-circle intersection lens path (A arcs)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Analytic consistency gate

**Files:**
- Modify: `src/vectoreyes/occlusion.py`
- Test: `tests/test_occlusion.py`

Render a candidate stack of primitives to a label image (analytic disk/ellipse masks, painted in z-order) and measure per-pixel agreement against the regions' own color assignment over their union.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_occlusion.py
from vectoreyes.occlusion import primitive_mask, stack_agreement


def test_primitive_mask_circle():
    m = primitive_mask({"kind": "circle", "params": {"cx": 10.0, "cy": 10.0, "r": 5.0}}, 20, 20)
    assert m[10, 10] and not m[10, 17] and m.dtype == bool


def test_stack_agreement_high_for_true_reconstruction():
    H = W = 100
    yy, xx = np.ogrid[:H, :W]
    da = (xx - 38) ** 2 + (yy - 50) ** 2 <= 30 ** 2
    db = (xx - 62) ** 2 + (yy - 50) ** 2 <= 30 ** 2
    red = da & ~db; yellow = db & ~da; lens = da & db
    regions = [_region(1, red, "#FF0000"), _region(2, yellow, "#FFFF00"), _region(3, lens, "#FFA500")]
    prims = [
        {"kind": "circle", "params": {"cx": 38.0, "cy": 50.0, "r": 30.0}, "color": "#FF0000", "z": 0},
        {"kind": "circle", "params": {"cx": 62.0, "cy": 50.0, "r": 30.0}, "color": "#FFFF00", "z": 1},
    ]
    lens_shape = {"mask_color": "#FFA500", "lens_of": (0, 1)}
    assert stack_agreement(prims, lens_shape, regions, H, W) > 0.97


def test_stack_agreement_low_for_wrong_reconstruction():
    H = W = 100
    yy, xx = np.ogrid[:H, :W]
    da = (xx - 38) ** 2 + (yy - 50) ** 2 <= 30 ** 2
    regions = [_region(1, da, "#FF0000")]
    prims = [{"kind": "circle", "params": {"cx": 38.0, "cy": 50.0, "r": 45.0}, "color": "#FF0000", "z": 0}]
    assert stack_agreement(prims, None, regions, H, W) < 0.9   # oversized disk disagrees
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py -k "primitive_mask or stack_agreement" -q`
Expected: FAIL — `ImportError: cannot import name 'primitive_mask'`.

- [ ] **Step 3: Write minimal implementation**

```python
def primitive_mask(prim: dict, h: int, w: int) -> np.ndarray:
    """Boolean mask of a completed circle/ellipse on an (h, w) grid."""
    yy, xx = np.ogrid[:h, :w]
    p = prim["params"]
    if prim["kind"] == "circle":
        return (xx - p["cx"]) ** 2 + (yy - p["cy"]) ** 2 <= p["r"] ** 2
    return ((xx - p["cx"]) / p["rx"]) ** 2 + ((yy - p["cy"]) / p["ry"]) ** 2 <= 1.0


def stack_agreement(prims, lens, regions: list[Region], h: int, w: int) -> float:
    """Paint prims (then the lens) in z-order into a colour-label image and compare,
    over the union of the regions, to each region's own colour. Returns [0, 1]."""
    BG = "\x00"
    painted = np.full((h, w), BG, dtype=object)
    for prim in sorted(prims, key=lambda p: p["z"]):
        painted[primitive_mask(prim, h, w)] = prim["color"]
    if lens is not None:
        a, b = (prims[lens["lens_of"][0]], prims[lens["lens_of"][1]])
        painted[primitive_mask(a, h, w) & primitive_mask(b, h, w)] = lens["mask_color"]
    truth = np.full((h, w), BG, dtype=object)
    union = np.zeros((h, w), bool)
    for r in regions:
        truth[r.mask] = r.color_hex
        union |= r.mask
    if not union.any():
        return 0.0
    return float((painted[union] == truth[union]).mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_occlusion.py -k "primitive_mask or stack_agreement" -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
SKIP_SIMPLE_GIT_HOOKS=1 git commit -m "feat(occlusion): analytic consistency gate (paint + compare)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: reconstruct_scene orchestration

**Files:**
- Modify: `src/vectoreyes/occlusion.py`
- Test: `tests/test_occlusion.py`

Tie it together: find bite-triggered adjacent groups, complete each region's primitive, snap a mirror pair to exact symmetry about the axis, derive the lens, gate the whole group, and return `(reconstructed, remaining)`. `reconstructed` is a list of `ScenePrimitive` plus optional lens `Shape`s; `remaining` is the regions to fit the old way.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_occlusion.py
from vectoreyes.types import Axis
from vectoreyes.fit import Shape
from vectoreyes.occlusion import reconstruct_scene


def _two_disk_mark(H=140, W=200, gap=24, r=44):
    yy, xx = np.ogrid[:H, :W]
    cx = W // 2
    da = (xx - (cx - gap)) ** 2 + (yy - H // 2) ** 2 <= r ** 2
    db = (xx - (cx + gap)) ** 2 + (yy - H // 2) ** 2 <= r ** 2
    return da, db, cx


def test_reconstruct_scene_two_disks_with_lens():
    H, W = 140, 200
    da, db, cx = _two_disk_mark(H, W)
    regions = [
        _region(1, da & ~db, "#FF0000"),
        _region(2, db & ~da, "#FFFF00"),
        _region(3, da & db, "#FFA500"),
    ]
    reconstructed, remaining = reconstruct_scene(regions, Axis(x=float(cx)), (H, W))
    prims = [e for e in reconstructed if isinstance(e, ScenePrimitive)]
    lenses = [e for e in reconstructed if isinstance(e, Shape)]
    assert len(prims) == 2 and len(lenses) == 1
    # mirror pair: equal radii, centres symmetric about the axis
    assert abs(prims[0].params["r"] - prims[1].params["r"]) < 1.0
    assert abs((prims[0].params["cx"] + prims[1].params["cx"]) / 2 - cx) < 1.0
    assert remaining == []                               # all three regions consumed


def test_reconstruct_scene_passes_through_non_occluded():
    H = W = 80
    yy, xx = np.ogrid[:H, :W]
    band = np.zeros((H, W), bool); band[20:40, 10:70] = True     # convex, no bite
    regions = [_region(1, band, "#062336")]
    reconstructed, remaining = reconstruct_scene(regions, Axis(x=40.0), (H, W))
    assert reconstructed == [] and [r.label for r in remaining] == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_occlusion.py -k reconstruct_scene -q`
Expected: FAIL — `ImportError: cannot import name 'reconstruct_scene'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add imports
from .fit import Shape
from .types import Axis

_MAX_RESIDUAL = 1.6
_MIN_ARC_DEG = 110.0
_GATE_AGREEMENT = 0.97


def _snap_pair(p_left: dict, p_right: dict, axis_x: float) -> tuple[dict, dict]:
    """Force two completed circles to be exact mirror images about x = axis_x."""
    r = (p_left["params"]["r"] + p_right["params"]["r"]) / 2
    cy = (p_left["params"]["cy"] + p_right["params"]["cy"]) / 2
    off = (abs(p_left["params"]["cx"] - axis_x) + abs(p_right["params"]["cx"] - axis_x)) / 2
    left = {"kind": "circle", "params": {"cx": axis_x - off, "cy": cy, "r": r}}
    right = {"kind": "circle", "params": {"cx": axis_x + off, "cy": cy, "r": r}}
    return left, right


def reconstruct_scene(
    regions: list[Region], axis: Axis | None, shape_hw: tuple[int, int]
) -> tuple[list, list[Region]]:
    """Return (reconstructed, remaining). `reconstructed` mixes ScenePrimitive and
    lens Shape objects in paint order; `remaining` are regions to fit the old way.
    Only adjacent groups that pass the consistency gate are reconstructed."""
    h, w = shape_hw
    by_label = {r.label: r for r in regions}
    adj = region_adjacency(regions)
    reconstructed: list = []
    consumed: set[int] = set()

    # only consider regions that actually have a concave bite
    for r in regions:
        if r.label in consumed or not has_bite(r.mask):
            continue
        # transitive group: the whole connected cluster of touching regions, because
        # the two crescents touch each other only THROUGH the lens (red-orange-yellow)
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
        # complete a primitive for every NON-convex member (the crescents); convex
        # members (the lens) are not completed, they become the overlap shape
        completed: list[tuple[Region, dict]] = []
        for gr in group_regions:
            if not has_bite(gr.mask):
                continue
            others = [o for o in group_regions if o.label != gr.label]
            contour, seam = label_boundary(gr, others)
            prim = complete_primitive(contour, seam, max_residual=_MAX_RESIDUAL, min_arc_deg=_MIN_ARC_DEG)
            if prim is not None:
                completed.append((gr, prim))
        if len(completed) != 2:
            continue                                      # v1 handles the two-disk case

        (ra, pa), (rb, pb) = completed
        # order left/right and (if symmetric) snap to an exact mirror pair
        if pa["params"]["cx"] > pb["params"]["cx"]:
            (ra, pa), (rb, pb) = (rb, pb), (ra, pa)
        if axis is not None and pa["kind"] == pb["kind"] == "circle":
            pa["params"], pb["params"] = _snap_pair(pa, pb, axis.x)[0]["params"], _snap_pair(pa, pb, axis.x)[1]["params"]

        prims = [
            {"kind": pa["kind"], "params": pa["params"], "color": ra.color_hex, "z": 0},
            {"kind": pb["kind"], "params": pb["params"], "color": rb.color_hex, "z": 1},
        ]
        # the lens region = a group member that is NOT one of the two completed crescents
        lens_region = next((g for g in group_regions if g.label not in {ra.label, rb.label}), None)
        lens = None
        if lens_region is not None and pa["kind"] == pb["kind"] == "circle":
            lens = {"mask_color": lens_region.color_hex, "lens_of": (0, 1)}

        if stack_agreement(prims, lens, group_regions, h, w) < _GATE_AGREEMENT:
            continue                                      # reject -> regions stay in `remaining`

        reconstructed.append(ScenePrimitive(pa["kind"], pa["params"], ra.color_hex, 0))
        reconstructed.append(ScenePrimitive(pb["kind"], pb["params"], rb.color_hex, 1))
        if lens is not None:
            d = intersection_lens_d(pa["params"], pb["params"])
            if d is not None:
                reconstructed.append(Shape("path", {"d": d, "color_hex": lens_region.color_hex, "z": 2}))
        consumed.update(g.label for g in group_regions)

    remaining = [r for r in regions if r.label not in consumed]
    return reconstructed, remaining
```

> The double `_snap_pair(...)` call is intentional in the minimal version; if simplifying, call it once into a local and read both halves.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_occlusion.py -k reconstruct_scene -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the whole occlusion suite, then commit**

Run: `uv run pytest tests/test_occlusion.py -q` → expect all PASS.

```bash
git add src/vectoreyes/occlusion.py tests/test_occlusion.py
SKIP_SIMPLE_GIT_HOOKS=1 git commit -m "feat(occlusion): reconstruct_scene orchestration + consistency gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Pipeline integration + acceptance

**Files:**
- Modify: `src/vectoreyes/pipeline.py`
- Create: `tests/test_acceptance_occlusion.py`

Wire `reconstruct_scene` into `idealize`: reconstruct first, emit reconstructed primitives/lenses, then run the existing straddler/pair/fit path on the *remaining* regions only.

- [ ] **Step 1: Write the failing acceptance + negative tests**

```python
# tests/test_acceptance_occlusion.py
import io
import numpy as np
from vectoreyes.pipeline import idealize, Options
from tests._render import render_svg, ssim


def _synthetic_mastercard(H=240, W=360, gap=58, r=104):
    """Two overlapping disks (distinct colours) with a third colour in the overlap."""
    yy, xx = np.ogrid[:H, :W]
    cx, cy = W // 2, H // 2
    da = (xx - (cx - gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    db = (xx - (cx + gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    img = np.full((H, W, 3), 255, np.uint8)
    img[da] = (235, 0, 27)         # red
    img[db] = (247, 158, 27)       # yellow
    img[da & db] = (255, 95, 0)    # orange overlap
    return img, cx


def test_synthetic_mastercard_reconstructs_to_circles_plus_lens():
    img, cx = _synthetic_mastercard()
    H, W = img.shape[:2]
    svg = idealize(img, options=Options())
    assert svg.count("<circle") == 2                      # two real disks recovered
    assert "<path" in svg                                 # the lens
    out = render_svg(svg, W, H)
    # exact bilateral symmetry about the vertical axis
    k = min(cx, W - cx)
    left = out[:, cx - k:cx][:, ::-1].astype(float)
    right = out[:, cx:cx + k].astype(float)
    assert float(ssim(left, right, channel_axis=2, data_range=255)) >= 0.99
    # faithful to the input
    assert float(ssim(out.astype(float), img.astype(float), channel_axis=2, data_range=255)) >= 0.95


def test_daikonic_unchanged_by_occlusion_pass():
    """The bite trigger / gate must NOT fire false occlusion on a non-overlapping mark."""
    from pathlib import Path
    from PIL import Image
    fix = Path(__file__).parent / "fixtures" / "daikonic" / "source.png"
    icon = np.asarray(Image.open(fix).convert("RGB"))[:392]
    svg = idealize(icon, options=Options())
    # the dome cap is still a 2-arc half-ellipse path and the leaves still mirror via <use>
    assert "<use" in svg
    assert svg.count("<circle") == 0                      # nothing got wrongly "completed" to a disk
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_acceptance_occlusion.py -q`
Expected: FAIL — `test_synthetic_mastercard...` fails (currently 0 `<circle>`, crescents emitted as paths).

- [ ] **Step 3: Implement the integration**

In `src/vectoreyes/pipeline.py`, add the import:

```python
from .occlusion import ScenePrimitive, reconstruct_scene
```

Then in `idealize`, after `corner_radius` is computed and BEFORE the `classify_regions` block, insert reconstruction and emit its output; run the existing classification on the remaining regions only. Replace the block from `straddlers: list[Region]` through the end of the draw loop with:

```python
    reconstructed, regions = reconstruct_scene(regions, axis, (h, w))

    straddlers: list[Region]
    pairs: list[tuple[Region, Region]]
    if axis is not None:
        straddlers, pairs = classify_regions(regions, axis)
    else:
        straddlers, pairs = list(regions), []

    body: list[str] = []
    eid = 0

    # 1) reconstructed occlusion primitives + lenses, painted in their own z-order
    for elem in sorted(reconstructed, key=lambda e: e.z if isinstance(e, ScenePrimitive) else e.params["z"]):
        if isinstance(elem, ScenePrimitive):
            shape = Shape(elem.kind, dict(elem.params))
            if opt.flatten:
                body.append(path_svg(shape_to_path_d(shape), elem.color_hex))
            else:
                body.append(shape_to_svg(shape, elem.color_hex, f"s{eid}"))
        else:  # lens Shape("path", {"d", "color_hex", "z"})
            body.append(path_svg(elem.params["d"], elem.params["color_hex"]))
        eid += 1

    # 2) everything else through the existing per-region path
    drawn = [(r, False) for r in straddlers] + [(canon, True) for canon, _ in pairs]
    drawn.sort(key=lambda rp: rp[0].area, reverse=True)
    for region, is_pair in drawn:
        shape = _fit_region(region, opt, axis if not is_pair else None, corner_radius)
        if shape is None:
            continue
        if opt.flatten:
            d = shape_to_path_d(shape)
            rule = shape.params.get("fill_rule")
            body.append(path_svg(d, region.color_hex, rule))
            if is_pair and axis is not None:
                body.append(path_svg(reflect_path_d(d, axis.x), region.color_hex, rule))
        else:
            elem_id = f"s{eid}"
            body.append(shape_to_svg(shape, region.color_hex, elem_id))
            if is_pair and axis is not None:
                body.append(mirror_use(elem_id, axis))
        eid += 1

    return render_svg_doc(w, h, body)
```

(Remove the old `straddlers`/`drawn`/loop/`return` that this replaces.)

- [ ] **Step 4: Run the acceptance + full suite**

Run: `uv run pytest tests/test_acceptance_occlusion.py -q` → expect PASS (2).
Run: `uv run pytest -q` → expect ALL pass (no regression in Daikonic/Mitsubishi/etc.).
Read the exit code; do NOT pipe through `tail` before committing.

- [ ] **Step 5: Visual smoke check (manual, not committed)**

Render the synthetic mark and eyeball the lens bulge direction:

```bash
uv run python -c "
import numpy as np, resvg_py, io
from PIL import Image
from tests.test_acceptance_occlusion import _synthetic_mastercard
from vectoreyes.pipeline import idealize, Options
img,_=_synthetic_mastercard(); H,W=img.shape[:2]
svg=idealize(img,options=Options())
png=resvg_py.svg_to_bytes(svg_string=svg,width=W,height=H)
Image.open(io.BytesIO(bytes(png))).convert('RGB').save('/tmp/occ_smoke.png')
print(svg[:400])
"
```

If the orange lens bulges the wrong way, flip the two sweep flags in `intersection_lens_d` (Task 5 note) and re-run Step 4.

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/pipeline.py tests/test_acceptance_occlusion.py
SKIP_SIMPLE_GIT_HOOKS=1 git commit -m "feat(pipeline): integrate occlusion reconstruction into idealize

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final review (after all tasks)

Dispatch a final code-reviewer over the whole `occlusion.py` + the `pipeline.py` diff. Specifically confirm:
- The consistency gate actually rejects (try the real rasterized Mastercard or a noisy fixture and confirm graceful fallback, no crash).
- `reconstruct_scene` returns `([], all_regions)` for every existing fixture except the synthetic Mastercard (no false positives) — `uv run pytest -q` covers the bundled ones.
- No new runtime dependency beyond scipy/shapely/skimage (already present).
- SPDX header present on `occlusion.py`.

Then finish via **superpowers:finishing-a-development-branch**.

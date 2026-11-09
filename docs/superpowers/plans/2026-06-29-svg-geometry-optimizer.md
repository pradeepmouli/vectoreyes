# SVG Geometry Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raster-level optimizations with a two-stage pipeline — faithful vectorize, then geometry-optimizer passes (primitives, clones, symmetry, simplify) each gated against the faithful ground truth so it cannot regress.

**Architecture:** Stage 1 emits a ground-truth object list (dual representation: exact `Shape` + flattened `shapely` polygon) plus each object's true raster mask. Stage 2 runs ordered passes over the objects; every proposed change is accepted only if the changed object's coverage stays within a symmetric-difference budget of its true mask. Reuses the existing front end (`color`/`segment`/`contour`/`fit`/`surface_merge`/`emit`) and the existing primitive/symmetry/simplify algorithms, relocated to operate on object geometry.

**Tech Stack:** Python ≥ 3.12, numpy, scipy.ndimage, shapely, scikit-image; pure-Python, TDD.

## Global Constraints

- Python ≥ 3.12, pure-Python. `rg` not `grep`. Determinism: all ordering value-based (sorted by id/area), no `Math.random`/time, no reordered float reductions.
- **No-regression gate is mandatory.** Every stage-2 change is accepted only if `coverage_symmetric_difference(changed_flat, true_mask) / norm <= BUDGET`. A change that fails the gate is rejected and the faithful object kept. `BUDGET` is a single coverage tolerance at rasterization-noise scale (start `BUDGET = 0.02`), grounded in sampling — never tuned per logo.
- **Faithful first.** Stage 1 performs NO primitive recognition, NO symmetry, NO clone detection. It emits faithful filled paths + gradient-merged surfaces only.
- Pass order is fixed: primitives → clones → symmetry → simplify.
- Curves are flattened to dense polylines for all geometry ops; the exact `Shape` is preserved for emission.
- Commit trailer EXACTLY (last line): `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Run tests: `PYTHONPATH=src ../../.venv/bin/python -m pytest <path> -q` from the worktree.

---

## File Structure

- `src/vectoreyes/optimizer/__init__.py` — package.
- `src/vectoreyes/optimizer/optobject.py` — `OptObject` dual-rep model + `flatten`.
- `src/vectoreyes/optimizer/faithful.py` — Stage 1: regions → `OptObject` list + true masks.
- `src/vectoreyes/optimizer/gate.py` — coverage gate (`coverage_residual`, `gate_ok`).
- `src/vectoreyes/optimizer/framework.py` — `optimize(objects, masks, passes)` runner.
- `src/vectoreyes/optimizer/passes/primitives.py` — pass 2a.
- `src/vectoreyes/optimizer/passes/clones.py` — pass 2b.
- `src/vectoreyes/optimizer/passes/symmetry.py` — pass 2c.
- `src/vectoreyes/optimizer/passes/simplify.py` — pass 2d.
- `tests/optimizer/test_*.py` — per-unit tests.

Reuse (do not duplicate): `Shape` (fit.py), `Fill`/`FlatFill`/`LinearGradientFill`/`RadialGradientFill` (candidate.py), `region_contours` (contour.py), `fit_path`/`recognize_primitive` (fit.py), `merge_surfaces` (surface_merge.py), `emit.shape_to_path_d`/`mirror_use`/`transform_path_d`/`path_svg` (emit.py), `symmetry` axis/test helpers (symmetry.py), `_fitcurve`/`rdp` (simplify).

---

### Task 1: `OptObject` model + flatten

**Files:**
- Create: `src/vectoreyes/optimizer/__init__.py` (empty), `src/vectoreyes/optimizer/optobject.py`
- Test: `tests/optimizer/test_optobject.py`

**Interfaces:**
- Consumes: `Shape` (`from ..fit import Shape`), `emit.shape_to_path_d`.
- Produces:
  - `flatten_points(shape: Shape, *, samples: int = 24) -> list[tuple[float, float]]` — dense outer-boundary points of a `Shape` (curves sampled). Reuses `emit.shape_to_path_d` then samples the `d`.
  - `to_polygon(shape: Shape, *, samples: int = 24) -> shapely.geometry.Polygon | shapely.geometry.MultiPolygon` — shapely polygon (shell + holes if the `d` has multiple subpaths), curves sampled, `.buffer(0)` to validate.
  - `OptObject` dataclass: `id: int`, `exact: Shape`, `fill` (the candidate Fill object), `z: int`, and a cached `flat` shapely geometry computed from `exact` (recomputed via `to_polygon` when `exact` changes — provide `with_exact(new_shape) -> OptObject` that returns a copy with `flat` refreshed).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from shapely.geometry import Polygon
from vectoreyes.fit import Shape
from vectoreyes.optimizer.optobject import to_polygon, flatten_points, OptObject
from vectoreyes.candidate import FlatFill

def test_to_polygon_circle_area_matches():
    circ = Shape("circle", {"cx": 50.0, "cy": 50.0, "r": 20.0})
    poly = to_polygon(circ, samples=64)
    assert abs(poly.area - np.pi * 20**2) / (np.pi * 20**2) < 0.01   # ~circle area

def test_to_polygon_path_with_hole():
    # outer square 0..40 with inner square 10..30 hole (evenodd path: two subpaths)
    d = "M0 0 L40 0 L40 40 L0 40 Z M10 10 L30 10 L30 30 L10 30 Z"
    poly = to_polygon(Shape("path", {"d": d}))
    assert abs(poly.area - (40*40 - 20*20)) < 4   # has the hole

def test_optobject_with_exact_refreshes_flat():
    o = OptObject(id=1, exact=Shape("rect", {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}),
                  fill=FlatFill("#000000"), z=0)
    assert abs(o.flat.area - 100) < 1
    o2 = o.with_exact(Shape("rect", {"x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0}))
    assert abs(o2.flat.area - 200) < 1 and abs(o.flat.area - 100) < 1   # original unchanged
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError: vectoreyes.optimizer`).

- [ ] **Step 3: Implement** `optobject.py`:

```python
from __future__ import annotations
import re
from dataclasses import dataclass, field, replace
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from ..fit import Shape
from ..emit import shape_to_path_d

_NUM = r'-?\d+(?:\.\d+)?'

def _sample_subpath(tokens, samples):
    cur = start = None; pts = []
    for k, a in tokens:
        v = [float(x) for x in re.findall(_NUM, a)]
        if k == 'M': cur = np.array(v[:2]); start = cur; pts.append(tuple(cur))
        elif k == 'L': cur = np.array(v[:2]); pts.append(tuple(cur))
        elif k == 'Q':
            c, p = np.array(v[:2]), np.array(v[2:4])
            for t in np.linspace(0, 1, samples)[1:]:
                pts.append(tuple((1-t)**2*cur + 2*(1-t)*t*c + t**2*p))
            cur = p
        elif k == 'C':
            c1, c2, p = np.array(v[:2]), np.array(v[2:4]), np.array(v[4:6])
            for t in np.linspace(0, 1, samples)[1:]:
                pts.append(tuple((1-t)**3*cur + 3*(1-t)**2*t*c1 + 3*(1-t)*t**2*c2 + t**3*p))
            cur = p
        elif k == 'Z': cur = start
    return pts

def flatten_points(shape: Shape, *, samples: int = 24):
    d = shape_to_path_d(shape)
    # split into subpaths at M; first subpath only (outer)
    toks = re.findall(rf'([MLQCZ])((?:\s*{_NUM}){{0,6}})', d)
    return _sample_subpath(toks, samples)

def to_polygon(shape: Shape, *, samples: int = 24):
    d = shape_to_path_d(shape)
    toks = re.findall(rf'([MLQCZ])((?:\s*{_NUM}){{0,6}})', d)
    # group into subpaths (each starts at M)
    subs, cur = [], []
    for t in toks:
        if t[0] == 'M' and cur:
            subs.append(cur); cur = [t]
        else:
            cur.append(t)
    if cur: subs.append(cur)
    rings = [_sample_subpath(s, samples) for s in subs]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return Polygon()
    # largest ring = shell, rest = holes
    polys = [Polygon(r) for r in rings]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    shell = max(polys, key=lambda p: p.area)
    holes = [p for p in polys if p is not shell]
    out = shell
    for h in holes:
        out = out.difference(h)
    return out if out.is_valid else out.buffer(0)

@dataclass(frozen=True)
class OptObject:
    id: int
    exact: Shape
    fill: object
    z: int
    flat: object = field(default=None, compare=False)
    def __post_init__(self):
        if self.flat is None:
            object.__setattr__(self, "flat", to_polygon(self.exact))
    def with_exact(self, new_shape: Shape) -> "OptObject":
        return OptObject(self.id, new_shape, self.fill, self.z)
```

- [ ] **Step 4: Run → PASS** (3 tests).
- [ ] **Step 5: Commit** `feat(optimizer): OptObject dual-rep model + shapely flatten`.

---

### Task 2: Stage-1 faithful vectorize → objects + true masks

**Files:**
- Create: `src/vectoreyes/optimizer/faithful.py`
- Test: `tests/optimizer/test_faithful.py`

**Interfaces:**
- Consumes: `_segment_image` (pipeline.py), `region_contours` (contour.py), `fit_path` (fit.py), `merge_surfaces`/`fit_fill` (surface_merge.py / fill_fit.py), `Options` (pipeline.py), `OptObject` (Task 1).
- Produces:
  - `faithful_objects(arr: np.ndarray, opt: Options) -> tuple[list[OptObject], dict[int, np.ndarray]]` — the ground-truth objects (faithful `fit_path` geometry + fills, gradient surfaces merged, z = back-to-front by area), and `true_masks[obj.id]` = the boolean raster mask of pixels that object is responsible for (its merged region mask). NO primitive recognition, NO symmetry.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from vectoreyes.pipeline import Options
from vectoreyes.optimizer.faithful import faithful_objects

def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy-cy)**2 + (xx-cx)**2) <= r*r

def test_faithful_single_disk_one_object_path():
    img = np.full((120, 120, 3), 255, np.uint8)
    img[_disk(120,120,60,60,40)] = (200, 30, 30)
    objs, masks = faithful_objects(img, Options())
    assert len(objs) == 1
    o = objs[0]
    assert o.exact.kind == "path"          # faithful: NO primitive recognition (not "circle")
    assert o.id in masks and masks[o.id].sum() > 0
    # flat polygon area ~ disk area
    assert abs(o.flat.area - np.pi*40**2) / (np.pi*40**2) < 0.1
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** `faithful.py`. Drive segmentation + fills exactly as `_render_body` does today MINUS symmetry/primitive: call `_segment_image`; per component build flat fills, `merge_surfaces`; for each surviving (region, fill), build the object: `shape = Shape("path", {"d": <fit_path(region_contours(region.mask)[0], epsilon, max_error)>.params['d']})` (handle holes by joining `significant_contours` subpaths with `fill_rule="evenodd"`), `z` = index in area-descending order, `true_masks[id] = region.mask`. Use `dataclasses`/existing helpers; do not run `recognize_primitive` or any symmetry function. Return `(objects, true_masks)`.

(Implementer: read `_render_body` lines that build `cands` for the non-symmetric, non-primitive path and lift only the faithful-path + fill portion. The geometry is always `fit_path`; never `recognize_primitive`/`select_geometry`'s primitive branch.)

- [ ] **Step 4: Run → PASS.** Also add a 2-color gradient test: a horizontal gradient strip → after `merge_surfaces`, ONE object with a gradient fill (assert `len(objs)==1` and fill is a gradient type).
- [ ] **Step 5: Commit** `feat(optimizer): faithful vectorize — objects + true masks (no primitives/symmetry)`.

---

### Task 3: Coverage gate

**Files:**
- Create: `src/vectoreyes/optimizer/gate.py`
- Test: `tests/optimizer/test_gate.py`

**Interfaces:**
- Consumes: shapely geometry, numpy mask.
- Produces:
  - `BUDGET = 0.02`
  - `rasterize(geom, shape_hw: tuple[int, int]) -> np.ndarray` — boolean coverage mask of a shapely polygon (use `skimage.draw.polygon` on exterior minus holes).
  - `coverage_residual(geom, true_mask: np.ndarray) -> float` — `|raster(geom) XOR true_mask|.sum() / true_mask.sum()` (normalized symmetric difference). When both are polygons the implementer MAY instead use `geom.symmetric_difference(true_poly).area / true_poly.area`; the mask form is the reference.
  - `gate_ok(geom, true_mask, *, budget: float = BUDGET) -> bool` — `coverage_residual <= budget`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from shapely.geometry import Point, Polygon
from vectoreyes.optimizer.gate import coverage_residual, gate_ok, BUDGET

def _disk_mask(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy-cy)**2 + (xx-cx)**2) <= r*r

def test_gate_accepts_matching_circle():
    H = W = 120
    true = _disk_mask(H, W, 60, 60, 40)
    circ = Point(60, 60).buffer(40, quad_segs=64)
    assert coverage_residual(circ, true) < 0.05
    assert gate_ok(circ, true)

def test_gate_rejects_wrong_shape():
    H = W = 120
    true = _disk_mask(H, W, 60, 60, 40)
    square = Polygon([(20,20),(100,20),(100,100),(20,100)])  # square over a disk
    assert coverage_residual(square, true) > BUDGET
    assert not gate_ok(square, true)
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `gate.py` using `skimage.draw.polygon` for `rasterize` (exterior coords → filled; subtract each hole's filled mask), XOR with `true_mask`, normalize by `true_mask.sum()`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `feat(optimizer): coverage gate (symmetric-difference vs true mask)`.

---

### Task 4: Optimizer framework

**Files:**
- Create: `src/vectoreyes/optimizer/framework.py`
- Test: `tests/optimizer/test_framework.py`

**Interfaces:**
- Consumes: `OptObject` (T1), `gate_ok` (T3).
- Produces:
  - A `Pass` protocol: a callable `pass_fn(objects: list[OptObject], masks: dict[int, np.ndarray]) -> list[Proposal]` where `Proposal` = `namedtuple("Proposal", "obj_ids new_objects")` (the object ids consumed, and the replacement object(s) — 1 for primitives/simplify/self-symmetry, 2→1+use for clones/pairs).
  - `optimize(objects, masks, passes, *, budget=BUDGET) -> list[OptObject]` — runs each pass in order; for each proposal, GATE every `new_object` whose geometry changed (`coverage_residual(new.flat, masks[orig_id]) <= budget` against the matching consumed object's mask); if ALL gates pass, apply (replace consumed ids with new objects, carry masks forward for new ids = union of consumed masks); else reject (keep originals). Deterministic: proposals applied in `sorted(obj_ids)` order.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from vectoreyes.fit import Shape
from vectoreyes.candidate import FlatFill
from vectoreyes.optimizer.optobject import OptObject
from vectoreyes.optimizer.framework import optimize, Proposal

def _rect_obj(i, w, h):
    return OptObject(i, Shape("rect", {"x":0.0,"y":0.0,"w":float(w),"h":float(h)}), FlatFill("#000"), 0)

def test_framework_accepts_good_rejects_bad():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20,20), bool)}; masks[1][0:10,0:10] = True
    # good pass: replace with identical rect (residual 0) -> accepted
    good = lambda os, ms: [Proposal((1,), [os[0]])]
    out = optimize(objs, masks, [good]); assert len(out) == 1
    # bad pass: replace with a tiny rect (huge residual) -> rejected, original kept
    bad = lambda os, ms: [Proposal((1,), [_rect_obj(1, 2, 2)])]
    out2 = optimize(objs, masks, [bad]); assert abs(out2[0].exact.params["w"] - 10) < 1e-9
```

- [ ] **Step 2: Run → FAIL.** **Step 3: Implement.** **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat(optimizer): pass framework with per-change coverage gate`.

---

### Task 5: Pass 2a — primitives

**Files:** Create `src/vectoreyes/optimizer/passes/__init__.py`, `passes/primitives.py`; Test `tests/optimizer/test_pass_primitives.py`.

**Interfaces:**
- Consumes: `recognize_primitive` (fit.py), `flatten_points` (T1), `Proposal`/`OptObject`.
- Produces: `primitives_pass(objects, masks, *, epsilon=1.5) -> list[Proposal]` — for each object, sample its `flat` exterior to points, call `recognize_primitive(points, epsilon=epsilon)`; if it returns a `Shape` (circle/ellipse/rect/polygon), propose replacing the object with `obj.with_exact(primitive_shape)`. The framework gate decides accept/reject.

- [ ] **Step 1: failing test** — a faithful-path disk object → `primitives_pass` proposes a `circle`; running it through `optimize([...], primitives_pass)` yields `exact.kind == "circle"`. A faithful-path on an irregular blob proposes nothing (or is gate-rejected).
- [ ] **Steps 2-4:** fail → implement (points = exterior coords of `obj.flat`; `recognize_primitive` already returns None when no primitive fits within ε) → pass.
- [ ] **Step 5: Commit** `feat(optimizer): primitives pass (path -> circle/ellipse/rect)`.

---

### Task 6: Pass 2b — clones

**Files:** Create `passes/clones.py`; Test `tests/optimizer/test_pass_clones.py`.

**Interfaces:**
- Consumes: `OptObject`, `emit.transform_path_d`, shapely affinity.
- Produces:
  - `clones_pass(objects, masks) -> list[Proposal]` — group objects whose `flat` are congruent up to **translation + rotation** (fill may differ): bucket by a rotation/translation-invariant descriptor (area within 1%, hu-moment vector within tol), then within a bucket verify by best-fit rigid transform (`skimage.measure.ransac`/`EuclideanTransform` on sampled boundary, or match centroids+principal angle) and confirm residual coverage ≤ budget. For each confirmed clone of a chosen canonical object, propose replacing it with a `<use>`-reference object: emit an `OptObject` whose `exact` is a `Shape("use", {"href": canonical_id, "transform": matrix, "fill": fill_hex})`. (Add `"use"` handling to `emit.shape_to_svg`/`shape_to_path_d`: a `use` shape emits `<use href="#id" transform=... fill=.../>`.)

- [ ] **Step 1: failing test** — two identical squares at different positions with different fills → `clones_pass` proposes a `use` for the second referencing the first, with a translation transform; congruent-up-to-rotation (one square rotated 30°) also matched; a square + a circle (not congruent) → no proposal.
- [ ] **Steps 2-4:** fail → implement (descriptor bucket + transform verify + gate) → pass. Extend `emit` for the `use` shape kind.
- [ ] **Step 5: Commit** `feat(optimizer): clones pass (translate+rotate dedup via <use>)`.

---

### Task 7: Pass 2c — symmetry (relocated to objects)

**Files:** Create `passes/symmetry.py`; Test `tests/optimizer/test_pass_symmetry.py`.

**Interfaces:**
- Consumes: `symmetry.py` axis-voting + the absolute test, adapted to operate on object `flat` polygons (reflect polygon, symmetric-difference ratio) instead of raster masks; `emit.mirror_use`/`reflect_path_d`.
- Produces:
  - `symmetry_pass(objects, masks) -> list[Proposal]` — (1) propose axes from object self-axes (a polygon symmetric about a vertical axis through its centroid) and from congruent mirror **pairs** (perpendicular bisector); cluster; pick the dominant **vertical** axis (reconstruction is vertical-only this pass — non-vertical detected but not reconstructed, consistent with the current spec deferral). (2) For each self-symmetric object about that axis: propose `obj.with_exact(<exact mirror of its left half about axis>)` — build the half from `reflect_path_d` of the object's own path. (3) For each mirror pair (A,B) about that axis: propose replacing B with a `mirror_use` of A. The framework gate (coverage vs each object's true mask) rejects any force-mirror that doesn't match (icloud).
  - Adapt the absolute test as `poly_symmetry_residual(poly, axis_x) = poly.symmetric_difference(reflect(poly, axis_x)).area / poly.area`; symmetric iff `<= budget`.

- [ ] **Step 1: failing tests** — (a) a self-symmetric object (vertically symmetric polygon) → proposes an exact-mirrored reconstruction accepted by the gate; (b) two mirror-twin objects → second becomes a `mirror_use` of the first; (c) an ASYMMETRIC object (icloud-like, residual 0.78) → proposed reconstruction is **gate-rejected**, object stays faithful; (d) a vertical pair beats a horizontal one for primary; non-vertical-only figure → no reconstruction (faithful). Determinism across runs.
- [ ] **Steps 2-4:** fail → implement (reuse `symmetry.py` clustering/voting; swap the mask-EDT test for the polygon symmetric-difference ratio) → pass.
- [ ] **Step 5: Commit** `feat(optimizer): symmetry pass on object geometry (gate-guarded, vertical primary)`.

---

### Task 8: Pass 2d — simplify

**Files:** Create `passes/simplify.py`; Test `tests/optimizer/test_pass_simplify.py`.

**Interfaces:**
- Consumes: `_fitcurve` (fit one curve), `rdp` (contour.py), `OptObject`.
- Produces: `simplify_pass(objects, masks) -> list[Proposal]` — for each path object, propose a reduced path: RDP-collapse near-collinear runs to lines; re-fit each curved run with the fewest Bézier segments (one curve where it fits within ε). Propose `obj.with_exact(Shape("path", {"d": reduced}))`. The gate rejects over-simplification.

- [ ] **Step 1: failing test** — a path with a long straight run split into many `L` segments → simplified to one `L`; a smooth arc split into many short `Q`/`C` → fewer segments; gate rejects a simplification that drops a real bump (coverage residual spikes).
- [ ] **Steps 2-4:** fail → implement → pass.
- [ ] **Step 5: Commit** `feat(optimizer): simplify pass (segment merge / curve reduction)`.

---

### Task 9: Integration + corpus acceptance

**Files:** Modify `src/vectoreyes/pipeline.py` (add an optimizer code path in `idealize`); Test `tests/optimizer/test_integration.py`.

**Interfaces:**
- Consumes: `faithful_objects` (T2), `optimize` (T4), the four passes (T5–T8), `emit` to serialize objects → SVG.
- Produces: a new `idealize` path (behind `Options.optimizer: bool = False` initially, default off so existing goldens are untouched) that runs `objects, masks = faithful_objects(arr, opt)` → `optimize(objects, masks, [primitives_pass, clones_pass, symmetry_pass, simplify_pass])` → serialize to SVG via `emit`.

- [ ] **Step 1: failing acceptance tests** (drive `idealize(..., options=Options(optimizer=True))`):
  - **icloud:** the cloud object is NOT mirrored (its symmetry proposal is gate-rejected) — its emitted path is the faithful asymmetric cloud (no `<use>` mirror on it).
  - **daikonic:** the radish bands are reconstructed exactly symmetric (a band's left/right halves are mirror-equal within budget) AND the "Daikonic" text objects are faithful (present, not dropped — object count for the text region labels unchanged).
  - **instagram:** the outer merged gradient object is detected symmetric (a vertical axis is found for it; its reconstruction is gate-accepted).
  - **a disk image:** the disk object becomes a `<circle>` (primitives pass).
  - Determinism: byte-identical `idealize(optimizer=True)` across two runs.
- [ ] **Step 2: run → FAIL. Step 3: implement** the `idealize` optimizer branch + object→SVG serialization. **Step 4:** run the acceptance tests + full suite `PYTHONPATH=src ../../.venv/bin/python -m pytest -q --ignore=tests/test_mcp_server.py --ignore=tests/test_mcp_image.py`. Existing goldens (optimizer off) must stay green; the new optimizer-on tests pass. STOP+report if any existing golden moves.
- [ ] **Step 5: Commit** `feat(pipeline): wire SVG geometry optimizer behind Options.optimizer`.

---

## Notes
- The optimizer path is gated behind `Options.optimizer` (default off) so it lands without disturbing the existing corpus; flipping the default to on (and retiring the raster symmetry/primitive paths) is a follow-up once the optimizer reaches parity.
- `BUDGET = 0.02` is the single coverage knob; if a real logo needs a different value, that is a finding to report, not a silent per-logo tune.
- Native coverage rasterizer and the optional geometric-merge pass are deferred (spec'd as follow-ons).
- Do not enable any pass before Task 4's gate is green — the gate is the safety net every pass depends on.

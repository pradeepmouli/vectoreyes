# Candidate / Fill Interface Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple geometry from fill via a typed `Candidate (geometry, fill)` + `Fill` sum type, collapsing the three emit loops in `pipeline._render_body` into one candidate-rendering loop — with **byte-identical SVG output**.

**Architecture:** New IO-free `candidate.py` (Fill sum type + Candidate dataclass). A new `build_candidates(...)` produces the candidate list in exact current paint order (occlusion-by-z → regions-by-area → gradients); a single emit loop replaces the three loops, owning gradient `<def>` registration and `s{eid}`/`g{N}` id-minting in the same order as today. A golden harness captured from the pre-refactor code proves nothing changed.

**Tech Stack:** Python, numpy, dataclasses, pytest. Reuses existing `fit.Shape`, `emit.*`, `_fit_region`, `detect_gradients`, `reconstruct_scene`.

**Spec:** `docs/superpowers/specs/2026-06-07-candidate-interface-design.md`
**Branch:** `feat/candidate-interface` (already created off `master`).

---

## Background the implementer needs

`pipeline._render_body(w, h, regions, opt, *, bake=None, rgb=None) -> (body, defs)` currently:
1. measures `silhouette`/`axis`/`corner_radius`, runs `reconstruct_scene`, then `detect_gradients` (when `rgb` given) → `gradient_fills, regions`,
2. classifies regions into `straddlers, pairs, loners`,
3. emits in **three loops**: (a) `reconstructed` occlusion prims/lenses sorted by z, (b) `drawn` regions sorted by area desc (straddlers with axis; pairs canonical + `<use>`/reflect mirror; loners), (c) `gradient_fills` footprints last — each loop maintaining the shared `eid` counter and `defs` list.

The refactor moves the per-element "decide geometry + fill" into `build_candidates` and renders all candidates in **one** loop. Output must be **byte-identical** — verified by the golden harness in Task 1.

**Exact current emit behaviour to preserve (verified):**
- `eid` increments once per emitted element, in order occlusion → regions → gradients. A region/gradient whose `_fit_region` returns `None` is skipped *and does not increment `eid`* (we reproduce this by **dropping** it in `build_candidates`, so it never reaches the loop).
- Occlusion `ScenePrimitive`: non-flatten → `shape_to_svg(shape, color, f"s{eid}")`; flatten → `emit(shape_to_path_d(shape), color, "evenodd" if kind=="annulus" else None)`.
- Occlusion **lens** (`Shape("path",{d,color_hex,z})`): **both modes** → `emit(d, color)` — a plain path with **no id** (the preserved quirk). `eid` still increments.
- Region (straddler/pair/loner): non-flatten → `shape_to_svg(shape, region.color_hex, f"s{eid}")`, then if pair `mirror_use(f"s{eid}", axis)`; flatten → `emit(d, color, rule)` then if pair `emit(reflect_path_d(d, axis.x), color, rule)` where `rule = shape.params.get("fill_rule")`.
- Gradient footprint: `gid = f"g{len(defs)}"` minted **before** appending the def; `_bake_gradient_geometry(geom, kind, bake)` applied when `bake` set; then rendered like a region with `fill=url(#gid)`.
- `emit` is the closure `def emit(d, fill, rule=None): return path_svg(transform_path_d(d, bake) if bake is not None else d, fill, rule)`.
- `shape_to_path_d(Shape("path",{"d":X})) == X` (confirmed), so the lens reproduces exactly through the flatten branch.

`Options` fields: `flatten: bool`, `no_symmetry: bool`, etc. `idealize` accepts a numpy array directly.

---

## Task 1: Byte-identical golden harness (capture from pre-refactor code)

**This task must land BEFORE any refactor commit** — the goldens freeze current behaviour as ground truth.

**Files:**
- Create: `tests/test_candidate_byte_identical.py`
- Create (generated): `tests/fixtures/golden/*.svg`

- [ ] **Step 1: Write the golden harness test**

Create `tests/test_candidate_byte_identical.py`. It reuses existing acceptance-test builders (DRY) so the cases exercise the real code paths (lens, annulus-evenodd, gradient defs, rectified bake), across both modes.

```python
"""Byte-identical regression net for the Candidate/Fill refactor.

Each case's golden SVG is captured from the pre-refactor code (Task 1) and
committed. The refactor (Tasks 2-3) must keep idealize output == golden, exactly.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from vectoreyes import Options, idealize
from tests._render import paint, disk
from tests.test_acceptance_annulus import _ring
from tests.test_acceptance_occlusion import _synthetic_mastercard
from tests.test_acceptance_gradient import _linear_img, _radial_img
from tests.test_acceptance_smooth_gradient import _smooth_linear_rect, _rotate_img

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
DAIKONIC = Path(__file__).parent / "fixtures" / "daikonic" / "source.png"


def _daikonic_icon():
    return np.asarray(Image.open(DAIKONIC).convert("RGB"), dtype=np.uint8)[:392]


def _ring_disk():
    h, w = 160, 200
    return paint([(_ring(70, 70, 45, 25, h, w), (37, 99, 235)),
                  (disk(135, 70, 38, h, w), (220, 30, 30))], h, w)


def _linear_grad():
    return _linear_img(200, 260, (40, 100), (220, 100),
                       [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])


def _radial_grad():
    return _radial_img(220, 220, (110, 110), 90,
                       [(0.0, (125, 211, 252)), (1.0, (29, 78, 216))])


def _rectified_grad():
    base = _smooth_linear_rect(160, 240, 40, 200, (85, 145, 225), (70, 125, 210))
    return _rotate_img(base, 30)


# (name, image factory, Options) — covers every source path x both modes + rectified.
CASES = [
    ("daikonic", _daikonic_icon, Options()),
    ("daikonic_flatten", _daikonic_icon, Options(flatten=True)),
    ("mastercard", lambda: _synthetic_mastercard()[0], Options()),
    ("mastercard_flatten", lambda: _synthetic_mastercard()[0], Options(flatten=True)),
    ("ring_disk", _ring_disk, Options()),
    ("ring_disk_flatten", _ring_disk, Options(flatten=True)),
    ("linear_grad", _linear_grad, Options()),
    ("linear_grad_flatten", _linear_grad, Options(flatten=True)),
    ("radial_grad", _radial_grad, Options()),
    ("rectified_grad", _rectified_grad, Options()),
    ("rectified_grad_flatten", _rectified_grad, Options(flatten=True)),
]


def _render(name):
    factory, opts = next((f, o) for n, f, o in CASES if n == name)
    return idealize(factory(), options=opts)


import pytest


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_byte_identical(name):
    golden = (GOLDEN / f"{name}.svg").read_text()
    assert _render(name) == golden, f"{name}: output diverged from golden"
```

- [ ] **Step 2: Generate the goldens from the current (pre-refactor) code**

Run this one-off generator (writes the committed golden files from current behaviour):

```bash
mkdir -p tests/fixtures/golden
.venv/bin/python -c "
from pathlib import Path
from tests.test_candidate_byte_identical import CASES, GOLDEN
from vectoreyes import idealize
GOLDEN.mkdir(parents=True, exist_ok=True)
for name, factory, opts in CASES:
    (GOLDEN / f'{name}.svg').write_text(idealize(factory(), options=opts))
    print('wrote', name)
"
```

Expected: prints `wrote <name>` for all 11 cases; 11 files appear in `tests/fixtures/golden/`.

- [ ] **Step 3: Run the harness — it must PASS on current code**

Run: `.venv/bin/pytest tests/test_candidate_byte_identical.py -v`
Expected: 11 passed (goldens were just captured from this same code).

- [ ] **Step 4: Commit the harness + goldens**

```bash
git add tests/test_candidate_byte_identical.py tests/fixtures/golden
git commit -m "test(pipeline): byte-identical golden harness for candidate refactor

Captures current idealize output for 11 cases (occlusion+lens, symmetry
pairs/<use>, annulus, linear/radial gradients, rectified path; flatten and
non-flatten) as committed goldens. Frozen from pre-refactor code as the
ground-truth regression net for the Candidate/Fill seam.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: `candidate.py` — Fill sum type + Candidate dataclass

**Files:**
- Create: `src/vectoreyes/candidate.py`
- Test: `tests/test_candidate.py`

- [ ] **Step 1: Write the failing unit test**

Create `tests/test_candidate.py`:

```python
from vectoreyes.candidate import (
    Candidate, FlatFill, LinearGradientFill, RadialGradientFill,
)
from vectoreyes.fit import Shape
from vectoreyes.types import Axis


def test_flat_fill_holds_hex():
    assert FlatFill("#ff0000").hex == "#ff0000"


def test_gradient_fills_hold_geometry_and_stops():
    lin = LinearGradientFill({"x1": 0, "y1": 0, "x2": 1, "y2": 0}, [(0.0, "#000")])
    rad = RadialGradientFill({"cx": 1, "cy": 1, "r": 2}, [(1.0, "#fff")])
    assert lin.geometry["x2"] == 1 and lin.stops[0][1] == "#000"
    assert rad.geometry["r"] == 2 and rad.stops[0][0] == 1.0


def test_candidate_defaults_mirror_none():
    c = Candidate(Shape("circle", {"cx": 1, "cy": 1, "r": 1}), FlatFill("#000"), "region")
    assert c.mirror is None and c.source == "region"


def test_candidate_carries_mirror_axis():
    c = Candidate(Shape("path", {"d": "M0 0"}), FlatFill("#000"), "region", mirror=Axis(5.0))
    assert c.mirror.x == 5.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_candidate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vectoreyes.candidate'`.

- [ ] **Step 3: Create `src/vectoreyes/candidate.py`**

```python
"""The (geometry, fill) candidate seam: decouples shape/path detection from
colour application. IO-free data (SVG emission stays in emit.py / pipeline.py)."""

from __future__ import annotations

from dataclasses import dataclass

from .fit import Shape
from .types import Axis


@dataclass
class FlatFill:
    """A solid colour fill."""
    hex: str


@dataclass
class LinearGradientFill:
    """A linear gradient fill. `geometry` = {x1, y1, x2, y2} in the element's frame."""
    geometry: dict
    stops: list


@dataclass
class RadialGradientFill:
    """A radial gradient fill. `geometry` = {cx, cy, r} in the element's frame."""
    geometry: dict
    stops: list


Fill = FlatFill | LinearGradientFill | RadialGradientFill


@dataclass
class Candidate:
    """One renderable element: a geometry paired with a fill.

    `source` records the producing strategy ("occlusion" | "lens" | "region" |
    "gradient") — provenance for later agent/user candidate selection, and the
    discriminator for the one legacy emit quirk (lens = plain path, no id).
    `mirror`, when set, means: emit the element AND its mirror twin about that axis.
    """
    geometry: Shape
    fill: Fill
    source: str
    mirror: Axis | None = None
```

- [ ] **Step 4: Run the unit test — PASS**

Run: `.venv/bin/pytest tests/test_candidate.py -v`
Expected: 4 passed.

- [ ] **Step 5: Confirm goldens still pass (no wiring yet → no change)**

Run: `.venv/bin/pytest tests/test_candidate_byte_identical.py -q`
Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/candidate.py tests/test_candidate.py
git commit -m "feat(candidate): Fill sum type + Candidate dataclass

IO-free (geometry, fill) seam decoupling shape detection from colour
application. FlatFill / LinearGradientFill / RadialGradientFill; Candidate
carries source provenance + optional mirror axis. No pipeline wiring yet.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: `build_candidates` + single emit loop in `_render_body`

**Files:**
- Modify: `src/vectoreyes/pipeline.py` (add `build_candidates`; rewrite the emit section of `_render_body`)

This is the behaviour-preserving refactor. The golden harness (Task 1) is the gate.

- [ ] **Step 1: Add the candidate imports to `pipeline.py`**

In the import block (after the `from .types import Axis, Region` line, around line 31), add:

```python
from .candidate import Candidate, Fill, FlatFill, LinearGradientFill, RadialGradientFill
```

- [ ] **Step 2: Add `build_candidates` above `_render_body`**

Insert this function immediately before `def _render_body(` (around line 202):

```python
def build_candidates(
    reconstructed: list, straddlers: list[Region], pairs: list,
    loners: list[Region], gradient_fills: list[tuple[Region, dict]],
    opt: Options, axis: Axis | None, corner_radius: float,
) -> list[Candidate]:
    """Decide geometry + fill per element and return the candidate list in exact
    paint order: occlusion (by z) -> regions (by area desc) -> gradients (detect
    order). Elements whose geometry fit returns None are dropped (matching the
    old per-loop `continue`, so the emit-time id sequence is unchanged)."""
    cands: list[Candidate] = []

    # 1) reconstructed occlusion primitives + lenses, in their own z-order
    for elem in sorted(
        reconstructed,
        key=lambda e: e.z if isinstance(e, ScenePrimitive) else e.params["z"],
    ):
        if isinstance(elem, ScenePrimitive):
            cands.append(Candidate(Shape(elem.kind, dict(elem.params)),
                                   FlatFill(elem.color_hex), "occlusion"))
        else:  # lens Shape("path", {"d", "color_hex", "z"})
            cands.append(Candidate(Shape("path", {"d": elem.params["d"]}),
                                   FlatFill(elem.params["color_hex"]), "lens"))

    # 2) regions: straddlers (fit half+mirror about axis), pairs (fit once +
    # <use>/reflect mirror), loners (as-is, no axis). Sorted by area descending.
    drawn = (
        [(r, axis, False) for r in straddlers]
        + [(canon, None, True) for canon, _ in pairs]
        + [(r, None, False) for r in loners]
    )
    drawn.sort(key=lambda rp: rp[0].area, reverse=True)
    for region, fit_axis, is_pair in drawn:
        shape = _fit_region(region, opt, fit_axis, corner_radius)
        if shape is None:
            continue
        cands.append(Candidate(shape, FlatFill(region.color_hex), "region",
                               mirror=axis if is_pair else None))

    # 3) gradient-filled footprints (after all flats/occlusion)
    for footprint, model in gradient_fills:
        shape = _fit_region(footprint, opt, None, corner_radius)
        if shape is None:
            continue
        g = model["geometry"]
        fill: Fill = (
            LinearGradientFill(g, model["stops"]) if model["kind"] == "linear"
            else RadialGradientFill(g, model["stops"])
        )
        cands.append(Candidate(shape, fill, "gradient"))

    return cands
```

- [ ] **Step 3: Replace the emit section of `_render_body`**

In `_render_body`, replace everything from the `if axis is not None:` region-classification block (around line 227) through the end of the gradient loop (the `return body, defs` at line 306) with:

```python
    if axis is not None:
        straddlers, pairs, loners = classify_regions(regions, axis)
    else:
        straddlers, pairs, loners = list(regions), [], []

    cands = build_candidates(
        reconstructed, straddlers, pairs, loners, gradient_fills, opt, axis, corner_radius
    )

    def emit(d: str, fill: str, rule: str | None = None) -> str:
        return path_svg(transform_path_d(d, bake) if bake is not None else d, fill, rule)

    def resolve_fill(fill: Fill) -> str:
        if isinstance(fill, FlatFill):
            return fill.hex
        g = fill.geometry
        if bake is not None:
            kind = "linear" if isinstance(fill, LinearGradientFill) else "radial"
            g = _bake_gradient_geometry(g, kind, bake)
        gid = f"g{len(defs)}"
        if isinstance(fill, LinearGradientFill):
            defs.append(linear_gradient_def(gid, g["x1"], g["y1"], g["x2"], g["y2"], fill.stops))
        else:
            defs.append(radial_gradient_def(gid, g["cx"], g["cy"], g["r"], fill.stops))
        return f"url(#{gid})"

    body: list[str] = []
    eid = 0
    for cand in cands:
        geom = cand.geometry
        fill = resolve_fill(cand.fill)
        if opt.flatten:
            d = shape_to_path_d(geom)
            rule = geom.params.get("fill_rule", "evenodd" if geom.kind == "annulus" else None)
            body.append(emit(d, fill, rule))
            if cand.mirror is not None:
                body.append(emit(reflect_path_d(d, cand.mirror.x), fill, rule))
        elif cand.source == "lens":
            body.append(emit(geom.params["d"], fill))
        else:
            elem_id = f"s{eid}"
            body.append(shape_to_svg(geom, fill, elem_id))
            if cand.mirror is not None:
                body.append(mirror_use(elem_id, cand.mirror))
        eid += 1

    return body, defs
```

Note: keep the lines *above* this block unchanged — `silhouette`/`axis`/`corner_radius`, `reconstruct_scene`, the `defs: list[str] = []` / `gradient_fills` init, and the `if rgb is not None: gradient_fills, regions = detect_gradients(regions, rgb)` call all stay exactly as they are. Delete the old `def emit`, the three old loops, and the old `gradient_fills: list[...] = []` only if duplicated — `defs` and `gradient_fills` must still be initialised before `build_candidates` is called.

- [ ] **Step 4: Run the golden harness — must stay byte-identical**

Run: `.venv/bin/pytest tests/test_candidate_byte_identical.py -v`
Expected: 11 passed. **If any case diverges**, diff the output against the golden to find the mismatch (most likely: gradient `gid` order, `eid` skip on `None` fits, the annulus `evenodd` rule, or the lens no-id branch). Fix the loop to match — do NOT regenerate the goldens.

Per the spec's pressure valve: only if reproducing a *specific* legacy quirk forces genuinely worse code may that single case be relaxed to a render-ΔE≈0 assertion, called out explicitly with a one-line rationale in the test + commit. Default is exact match.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass (previous count + the new candidate/golden tests).

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/pipeline.py
git commit -m "refactor(pipeline): unify emit via build_candidates + single loop

Collapse the three emit loops (occlusion, regions, gradients) in _render_body
into one candidate-rendering loop fed by build_candidates, which decides
geometry+fill per element in exact paint order. Fill is resolved (incl.
gradient def registration / id minting) in one place. Byte-identical output
(golden harness green); pure structural decoupling for the scorer/selector.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- New `candidate.py` with `Fill` sum type + `Candidate(geometry, fill, source, mirror)`, no `cost`, no `z` → Task 2.
- `Fill` pure data; emit owns def-registration/id-minting (approach A) → Task 3 `resolve_fill`.
- `build_candidates` produces ordered list (occlusion-z → regions-area → gradients) → Task 3 Step 2.
- Single emit loop replacing three loops; preserves flatten/non-flatten, lens-no-id quirk, mirror, `eid`/`gid` order → Task 3 Step 3.
- Byte-identical golden harness, captured pre-refactor, exact `==` → Task 1; pressure valve noted in Task 3 Step 4.
- Full suite green → Task 2 Step 5, Task 3 Step 5.
- No `color.py` / palette changes; `_fit_region`/`detect_gradients`/`reconstruct_scene`/def helpers reused as-is → Tasks reuse them, none modified.

**2. Placeholder scan:** No TBD/TODO; every code step has complete code; every run step has exact command + expected result.

**3. Type consistency:** `Candidate(geometry, fill, source, mirror=None)` and `FlatFill/LinearGradientFill/RadialGradientFill` are defined in Task 2 and used identically in Task 3. `build_candidates` signature in Task 3 Step 2 matches its call in Step 3. `resolve_fill`/`emit` closures match the preserved-behaviour spec. Golden `CASES` names in Task 1 match the parametrize and the generator.

# Gradient Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect genuine linear/radial gradients in a logo and emit one shape with an SVG `<linearGradient>`/`<radialGradient>` fill, instead of shattering the ramp into many flat quantized bands.

**Architecture:** A new `gradient.py` module. In the upright pipeline path, after occlusion reconstruction, group adjacent flat regions whose colors form an OKLab ramp into a *footprint*, fit a linear (then radial) color-vs-position model against the **original** RGB pixels under that footprint, and accept it only if it re-renders within a mean-OKLab-ΔE bar (else dissolve back into flat bands). Accepted footprints become one region fitted by the existing shape recognizers but emitted with a `fill="url(#gN)"` and a gradient `<defs>` entry. Reuses the occlusion group→fit→gate pattern; gradient is an orthogonal fill on any recognized shape.

**Tech Stack:** Python, NumPy, scikit-image; `color.srgb_to_oklab` for perceptual fitting/gating; the existing `region_adjacency`, `segment`/`Region`, shape recognizers, and `emit`/`render_svg_doc`.

---

## Background the implementer needs

Read `docs/superpowers/specs/2026-06-06-gradient-handling-design.md`. Facts about the existing code:

- **`Region`** (`src/vectoreyes/types.py`): dataclass `.label:int`, `.mask:np.ndarray` bool (H,W), `.color_hex:str`, `.area` property.
- **`color.srgb_to_oklab(rgb)`** (`src/vectoreyes/color.py`): takes a float `(N,3)` array in `[0,1]` → OKLab `(N,3)`. (See its use in `tests/_render.py::mean_delta_e`.)
- **`occlusion.region_adjacency(regions) -> dict[int,set[int]]`**: label → set of touching labels. Reuse for grouping.
- **Pipeline** (`src/vectoreyes/pipeline.py`): `idealize(image)` builds `arr` (uint8 RGB `(H,W,3)`), calls `_segment_image(arr,opt) -> (w,h,regions)`, then `_render_body(w,h,regions,opt) -> body:list[str]`, then `render_svg_doc(w,h,body)`. `_render_body` runs `reconstruct_scene` (occlusion) then fits/emits regions. There is also `_idealize_rectified` (any-axis symmetry) which runs `_render_body` in a rotated frame — **gradient detection must be OFF there** (see Task 7).
- **`emit.render_svg_doc(width,height,body)`** wraps a body list in `<svg>…</svg>`. `emit.path_svg(d, fill, fill_rule=None)` and `emit.shape_to_svg(shape, fill, elem_id)` take the fill string verbatim — so a gradient fill is just `fill="url(#g0)"`.

**Data shapes (consistent across all tasks):**
- A fitted gradient model is a dict: `{"kind": "linear"|"radial", "geometry": {...}, "stops": [(offset_float, "#rrggbb"), …]}`. Linear geometry `{"x1","y1","x2","y2"}`; radial geometry `{"cx","cy","r"}`.
- `detect_gradients(...)` returns `(fills, remaining)` where `fills` is `list[tuple[Region, dict]]` (footprint region + gradient model) and `remaining` is the leftover flat regions.

**Stops are sampled directly in sRGB** (median original color in each offset-neighborhood) so no OKLab→sRGB inverse is needed; OKLab is used only for the ramp test, the parameter fit, and the ΔE gate (all forward `srgb_to_oklab`).

---

## File Structure

- **Create** `src/vectoreyes/gradient.py` — all detection/fit/gate logic.
- **Modify** `src/vectoreyes/color.py` — add `mean_delta_e` (promoted from the test helper).
- **Modify** `src/vectoreyes/emit.py` — `linear_gradient_def`, `radial_gradient_def`, `render_svg_doc(defs=...)`.
- **Modify** `src/vectoreyes/pipeline.py` — thread the original RGB + gradient defs through `_render_body`; call `detect_gradients` in the upright path only.
- **Modify** `tests/_render.py` — re-export `mean_delta_e` from `color` (keep the test import working, DRY).
- **Create** `tests/test_gradient.py` — unit tests.
- **Create** `tests/test_acceptance_gradient.py` — end-to-end acceptance.

---

### Task 1: Promote `mean_delta_e` into the color module

**Files:**
- Modify: `src/vectoreyes/color.py`
- Modify: `tests/_render.py`
- Test: `tests/test_gradient.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_gradient.py`:

```python
# SPDX-License-Identifier: MIT
import numpy as np


def test_mean_delta_e_zero_for_identical_and_positive_for_different():
    from vectoreyes.color import mean_delta_e
    a = np.full((4, 4, 3), 120, np.uint8)
    assert mean_delta_e(a, a) == 0.0
    b = a.copy(); b[:, :, 0] = 200          # shift red channel
    assert mean_delta_e(a, b) > 0.02
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_gradient.py -k mean_delta_e -q`
Expected: FAIL — `ImportError: cannot import name 'mean_delta_e' from 'vectoreyes.color'`.

- [ ] **Step 3: Add `mean_delta_e` to `color.py`**

Append to `src/vectoreyes/color.py`:

```python
def mean_delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-pixel OKLab Euclidean distance between two uint8 RGB images/arrays
    (shape (...,3)). 0.0 == identical; perceptual color error."""
    la = srgb_to_oklab(a.reshape(-1, 3) / 255.0)
    lb = srgb_to_oklab(b.reshape(-1, 3) / 255.0)
    return float(np.linalg.norm(la - lb, axis=1).mean())
```

- [ ] **Step 4: Re-point the test helper (DRY)**

Replace the body of `mean_delta_e` in `tests/_render.py` with a re-export so there's one implementation:

```python
from vectoreyes.color import mean_delta_e  # re-exported; single source of truth
```

Remove the old local `def mean_delta_e(...)` in `tests/_render.py`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_gradient.py -k mean_delta_e -q`
Expected: PASS (1 passed). Then `uv run pytest -q` — full suite still green (the re-export keeps existing importers working).

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/color.py tests/_render.py tests/test_gradient.py
git commit -m "refactor(color): promote mean_delta_e (OKLab dE) into the color module"
```

---

### Task 2: Gradient `<defs>` emit + `render_svg_doc` defs

**Files:**
- Modify: `src/vectoreyes/emit.py`
- Test: `tests/test_gradient.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gradient.py`:

```python
def test_linear_gradient_def_emits_stops_and_coords():
    from vectoreyes.emit import linear_gradient_def
    d = linear_gradient_def("g0", 10, 20, 110, 20, [(0.0, "#ff0000"), (1.0, "#0000ff")])
    assert d.startswith("<linearGradient") and 'id="g0"' in d
    assert 'gradientUnits="userSpaceOnUse"' in d
    assert 'x1="10"' in d and 'x2="110"' in d
    assert d.count("<stop") == 2
    assert 'offset="0"' in d and 'stop-color="#ff0000"' in d
    assert 'offset="1"' in d and 'stop-color="#0000ff"' in d


def test_radial_gradient_def_emits_center_radius():
    from vectoreyes.emit import radial_gradient_def
    d = radial_gradient_def("g1", 50, 60, 40, [(0.0, "#ffffff"), (1.0, "#000000")])
    assert d.startswith("<radialGradient") and 'id="g1"' in d
    assert 'cx="50"' in d and 'cy="60"' in d and 'r="40"' in d
    assert d.count("<stop") == 2


def test_render_svg_doc_wraps_defs():
    from vectoreyes.emit import render_svg_doc
    out = render_svg_doc(100, 100, ['<rect/>'], defs=['<linearGradient id="g0"></linearGradient>'])
    assert "<defs>" in out and "</defs>" in out
    assert out.index("<defs>") < out.index("<rect/>")     # defs before body
    out2 = render_svg_doc(100, 100, ['<rect/>'])
    assert "<defs>" not in out2                            # no defs block when none given
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k "gradient_def or render_svg_doc_wraps" -q`
Expected: FAIL — `linear_gradient_def`/`radial_gradient_def` not defined; `render_svg_doc` has no `defs` kwarg.

- [ ] **Step 3: Implement in `emit.py`**

Add to `src/vectoreyes/emit.py` (near `path_svg`):

```python
def _gradient_stops(stops: list[tuple[float, str]]) -> str:
    return "".join(f'<stop offset="{_fmt(o)}" stop-color="{c}"/>' for o, c in stops)


def linear_gradient_def(elem_id: str, x1: float, y1: float, x2: float, y2: float,
                        stops: list[tuple[float, str]]) -> str:
    """A <linearGradient> in userSpaceOnUse coords (absolute px; no gradientTransform,
    survives --flatten)."""
    return (f'<linearGradient id="{elem_id}" gradientUnits="userSpaceOnUse" '
            f'x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}">'
            f'{_gradient_stops(stops)}</linearGradient>')


def radial_gradient_def(elem_id: str, cx: float, cy: float, r: float,
                        stops: list[tuple[float, str]]) -> str:
    """A <radialGradient> in userSpaceOnUse coords."""
    return (f'<radialGradient id="{elem_id}" gradientUnits="userSpaceOnUse" '
            f'cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}">'
            f'{_gradient_stops(stops)}</radialGradient>')
```

Change `render_svg_doc` to accept `defs`:

```python
def render_svg_doc(width: int, height: int, body: list[str], defs: list[str] | None = None) -> str:
    defs_block = f'  <defs>{"".join(defs)}</defs>\n  ' if defs else "  "
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n'
        + defs_block
        + "\n  ".join(body)
        + "\n</svg>\n"
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_gradient.py -k "gradient_def or render_svg_doc_wraps" -q`
Expected: PASS (3 passed). Then `uv run pytest -q` — full suite green (the `defs` default keeps every existing `render_svg_doc(w,h,body)` call working).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/emit.py tests/test_gradient.py
git commit -m "feat(emit): linear/radial gradient defs + render_svg_doc defs block"
```

---

### Task 3: Band grouping (`_ramp_groups`)

**Files:**
- Create: `src/vectoreyes/gradient.py`
- Test: `tests/test_gradient.py` (append)

A gradient's bands are spatially adjacent AND their flat colors progress along a 1-D path in OKLab. Group connected regions whose colors are ramp-consistent: collinear in OKLab (the colors fit a single line within tolerance) and at least 3 of them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gradient.py`:

```python
def _hstrip_regions(colors_hex, h=40, band_w=12):
    """Adjacent vertical bands left→right, one per color (a quantized ramp)."""
    from vectoreyes.types import Region
    w = band_w * len(colors_hex)
    regions = []
    for i, c in enumerate(colors_hex):
        m = np.zeros((h, w), bool)
        m[:, i * band_w:(i + 1) * band_w] = True
        regions.append(Region(label=i + 1, mask=m, color_hex=c))
    return regions


def test_ramp_groups_groups_a_monotonic_ramp():
    from vectoreyes.gradient import _ramp_groups
    # 4 adjacent bands stepping blue->magenta (a clear OKLab ramp)
    regions = _hstrip_regions(["#2563eb", "#7b3fc4", "#b13a9e", "#db2777"])
    groups = _ramp_groups(regions)
    assert len(groups) == 1 and len(groups[0]) == 4


def test_ramp_groups_rejects_flat_and_too_few():
    from vectoreyes.gradient import _ramp_groups
    flat = _hstrip_regions(["#2563eb", "#2563eb", "#2563eb"])   # no variation
    assert _ramp_groups(flat) == []
    two = _hstrip_regions(["#2563eb", "#db2777"])               # only 2 -> not a gradient
    assert _ramp_groups(two) == []


def test_ramp_groups_rejects_nonramp_colors():
    from vectoreyes.gradient import _ramp_groups
    # adjacent but colors are not collinear in OKLab (zig-zag hues)
    regions = _hstrip_regions(["#ff0000", "#00ff00", "#0000ff", "#00ff00"])
    assert _ramp_groups(regions) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k ramp_groups -q`
Expected: FAIL — `gradient` module / `_ramp_groups` not defined.

- [ ] **Step 3: Implement `gradient.py` (grouping)**

Create `src/vectoreyes/gradient.py`:

```python
# SPDX-License-Identifier: MIT
"""Gradient detection: recover a gradient's footprint from quantized bands, fit a
linear/radial colour model against the original image, and emit one gradient-filled
shape when it re-renders faithfully (see
docs/superpowers/specs/2026-06-06-gradient-handling-design.md)."""

from __future__ import annotations

import numpy as np

from .color import srgb_to_oklab
from .occlusion import region_adjacency
from .types import Region

_MIN_BANDS = 3
_RAMP_TOL = 0.06          # max OKLab distance of a band colour from the fitted ramp line
_GATE_DELTA_E = 0.05      # mean OKLab ΔE bar; a faithful gradient fit scores below this


def _hex_to_oklab(hex_colors: list[str]) -> np.ndarray:
    rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in hex_colors], float)
    return srgb_to_oklab(rgb / 255.0)


def _is_ramp(colors_oklab: np.ndarray) -> bool:
    """True if >=3 colours and they lie on a single line in OKLab within _RAMP_TOL."""
    if len(colors_oklab) < _MIN_BANDS:
        return False
    mean = colors_oklab.mean(axis=0)
    centred = colors_oklab - mean
    if np.abs(centred).max() < 1e-6:
        return False                                   # all equal -> flat, not a ramp
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    line = vt[0]                                        # principal colour direction
    proj = np.outer(centred @ line, line)
    resid = np.linalg.norm(centred - proj, axis=1).max()
    return resid <= _RAMP_TOL


def _ramp_groups(regions: list[Region]) -> list[list[Region]]:
    """Connected groups of >=3 adjacent regions whose flat colours form an OKLab ramp."""
    by_label = {r.label: r for r in regions}
    adj = region_adjacency(regions)
    seen: set[int] = set()
    groups: list[list[Region]] = []
    for r in regions:
        if r.label in seen:
            continue
        # grow the connected component
        comp: list[int] = []
        stack = [r.label]
        while stack:
            lab = stack.pop()
            if lab in seen:
                continue
            seen.add(lab)
            comp.append(lab)
            stack.extend(adj[lab] - seen)
        members = [by_label[l] for l in comp]
        if len(members) < _MIN_BANDS:
            continue
        if _is_ramp(_hex_to_oklab([m.color_hex for m in members])):
            groups.append(members)
    return groups
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_gradient.py -k ramp_groups -q`
Expected: PASS (3 passed).

If `test_ramp_groups_rejects_nonramp_colors` does not reject (the zig-zag happens to fit within `_RAMP_TOL`), the residual tolerance is too loose — keep `_RAMP_TOL = 0.06`; the gate (Task 6) is the real arbiter, but the unit expectation is that a clearly non-collinear hue set exceeds it. If needed, make the fixture's zig-zag more extreme (saturated R/G/B) rather than loosening the constant.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): ramp-consistent band grouping (_ramp_groups)"
```

---

### Task 4: Stop fitting + linear model (`_fit_stops`, `_reduce_stops`, `_fit_linear`)

**Files:**
- Modify: `src/vectoreyes/gradient.py`
- Test: `tests/test_gradient.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gradient.py`:

```python
def _linear_gradient_image(h, w, p0, p1, stops_rgb):
    """Render a ground-truth linear gradient (for fitting against)."""
    yy, xx = np.mgrid[:h, :w]
    d = np.array(p1, float) - np.array(p0, float)
    L = np.dot(d, d)
    t = (((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb])
    cols = np.array([c for _, c in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    return img.round().astype(np.uint8)


def test_fit_linear_recovers_axis_and_endpoints():
    from vectoreyes.gradient import _fit_linear
    h, w = 80, 120
    p0, p1 = (10, 40), (110, 40)                       # horizontal axis
    img = _linear_gradient_image(h, w, p0, p1, [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    ys, xs = np.mgrid[:h, :w]
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    oklab = _OKLAB(img)
    model = _fit_linear(pts, oklab, img.reshape(-1, 3))
    assert model is not None and model["kind"] == "linear"
    g = model["geometry"]
    # axis is ~horizontal: endpoints span x, ~constant y
    assert abs(g["y1"] - g["y2"]) < 3.0
    assert abs(abs(g["x2"] - g["x1"]) - 100) < 12
    assert len(model["stops"]) >= 2


def test_reduce_stops_drops_redundant_midpoints():
    from vectoreyes.gradient import _reduce_stops
    # a perfectly linear ramp in sRGB: midpoints are redundant
    stops = [(0.0, "#000000"), (0.25, "#404040"), (0.5, "#808080"),
             (0.75, "#bfbfbf"), (1.0, "#ffffff")]
    reduced = _reduce_stops(stops, max_delta_e=0.02)
    assert len(reduced) < len(stops) and reduced[0][0] == 0.0 and reduced[-1][0] == 1.0
```

Add this helper near the top of `tests/test_gradient.py` (after the imports):

```python
def _OKLAB(img_uint8):
    from vectoreyes.color import srgb_to_oklab
    return srgb_to_oklab(img_uint8.reshape(-1, 3) / 255.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k "fit_linear or reduce_stops" -q`
Expected: FAIL — `_fit_linear`/`_reduce_stops` not defined.

- [ ] **Step 3: Implement stop fitting + linear fit**

Append to `src/vectoreyes/gradient.py`:

```python
def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(round(v)) for v in np.clip(rgb, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def _fit_stops(t: np.ndarray, rgb: np.ndarray, k: int = 5) -> list[tuple[float, str]]:
    """Sample k evenly-spaced stops; each stop colour = median original RGB of pixels
    in its t-neighbourhood. `t` in [0,1], `rgb` is (N,3) uint8-ish."""
    edges = np.linspace(0.0, 1.0, k)
    half = 0.5 / (k - 1)
    stops: list[tuple[float, str]] = []
    for e in edges:
        sel = np.abs(t - e) <= half
        if not sel.any():
            sel = np.argsort(np.abs(t - e))[:32]       # nearest fallback
        stops.append((float(e), _rgb_to_hex(np.median(rgb[sel], axis=0))))
    return stops


def _stop_colors_oklab(stops: list[tuple[float, str]]) -> np.ndarray:
    return _hex_to_oklab([c for _, c in stops])


def _reduce_stops(stops: list[tuple[float, str]], *, max_delta_e: float) -> list[tuple[float, str]]:
    """Greedily drop interior stops whose removal keeps the piecewise-linear (in OKLab)
    reconstruction within max_delta_e of the full stop set."""
    kept = list(stops)
    changed = True
    while changed and len(kept) > 2:
        changed = False
        for i in range(1, len(kept) - 1):
            trial = kept[:i] + kept[i + 1:]
            offs = np.array([o for o, _ in trial])
            cols = _stop_colors_oklab(trial)
            ref_off = np.array([o for o, _ in kept])
            ref_cols = _stop_colors_oklab(kept)
            approx = np.column_stack([np.interp(ref_off, offs, cols[:, ch]) for ch in range(3)])
            if np.linalg.norm(approx - ref_cols, axis=1).max() <= max_delta_e:
                kept = trial
                changed = True
                break
    return kept


def _fit_linear(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Fit a linear gradient. pts:(N,2) xy, oklab:(N,3), rgb:(N,3). Returns a model or
    None if the points don't span an axis."""
    A = np.column_stack([pts, np.ones(len(pts))])
    coef, *_ = np.linalg.lstsq(A, oklab, rcond=None)   # rows: [p, q, r] per channel column
    G = coef[:2, :].T                                   # (3 channels, 2) per-channel gradient
    if np.abs(G).max() < 1e-8:
        return None
    _, _, vt = np.linalg.svd(G, full_matrices=False)
    u = vt[0]                                           # unit axis direction
    proj = pts @ u
    t0, t1 = float(proj.min()), float(proj.max())
    if t1 - t0 < 1e-6:
        return None
    tn = (proj - t0) / (t1 - t0)
    mean = pts.mean(axis=0)
    mt = float(mean @ u)
    p1 = mean + (t0 - mt) * u
    p2 = mean + (t1 - mt) * u
    stops = _reduce_stops(_fit_stops(tn, rgb), max_delta_e=_GATE_DELTA_E)
    return {"kind": "linear",
            "geometry": {"x1": float(p1[0]), "y1": float(p1[1]),
                         "x2": float(p2[0]), "y2": float(p2[1])},
            "stops": stops}
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_gradient.py -k "fit_linear or reduce_stops" -q`
Expected: PASS (2 passed). If `_fit_linear`'s endpoint-span assertion is off by more than the tolerance, widen the test tolerance slightly (the axis *direction* is what matters; endpoints are derived from the projection span) — do not change the fit.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): stop fitting, greedy stop reduction, linear model fit"
```

---

### Task 5: Radial model (`_fit_radial`)

**Files:**
- Modify: `src/vectoreyes/gradient.py`
- Test: `tests/test_gradient.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gradient.py`:

```python
def _radial_gradient_image(h, w, c, r, stops_rgb):
    yy, xx = np.mgrid[:h, :w]
    t = (np.hypot(xx - c[0], yy - c[1]) / r).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb]); cols = np.array([col for _, col in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    return img.round().astype(np.uint8)


def test_fit_radial_recovers_center():
    from vectoreyes.gradient import _fit_radial
    h, w = 120, 120
    c, r = (60, 60), 50
    img = _radial_gradient_image(h, w, c, r, [(0.0, (125, 211, 252)), (1.0, (29, 78, 216))])
    ys, xs = np.mgrid[:h, :w]
    # restrict to the disc so background doesn't dominate the fit
    inside = np.hypot(xs - c[0], ys - c[1]) <= r
    pts = np.column_stack([xs[inside], ys[inside]]).astype(float)
    oklab = _OKLAB(img[inside])
    model = _fit_radial(pts, oklab, img[inside].reshape(-1, 3))
    assert model is not None and model["kind"] == "radial"
    g = model["geometry"]
    assert abs(g["cx"] - 60) < 6 and abs(g["cy"] - 60) < 6
    assert g["r"] > 30
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_gradient.py -k fit_radial -q`
Expected: FAIL — `_fit_radial` not defined.

- [ ] **Step 3: Implement `_fit_radial`**

Append to `src/vectoreyes/gradient.py`:

```python
def _radial_spread(pts: np.ndarray, oklab: np.ndarray, c: np.ndarray, nbins: int = 16) -> float:
    """Mean within-bin colour variance when binning pixels by distance from c. Lower =
    more concentric (a better radial centre)."""
    r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    rmax = r.max()
    if rmax < 1e-6:
        return np.inf
    bins = np.clip((r / rmax * nbins).astype(int), 0, nbins - 1)
    total, count = 0.0, 0
    for b in range(nbins):
        sel = bins == b
        if sel.sum() >= 2:
            total += float(np.var(oklab[sel], axis=0).sum())
            count += 1
    return total / count if count else np.inf


def _fit_radial(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Fit a radial gradient: estimate the centre as the centroid of the extreme along
    the principal colour axis (try both ends; keep the more concentric), then fit stops
    vs normalized radius."""
    cmean = oklab.mean(axis=0)
    centred = oklab - cmean
    if np.abs(centred).max() < 1e-8:
        return None
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    s = centred @ vt[0]                                 # 1-D colour coordinate
    best = None
    for sel in (s >= np.quantile(s, 0.9), s <= np.quantile(s, 0.1)):
        if sel.sum() < 3:
            continue
        c = pts[sel].mean(axis=0)
        spread = _radial_spread(pts, oklab, c)
        if best is None or spread < best[0]:
            best = (spread, c)
    if best is None:
        return None
    c = best[1]
    r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    rmax = float(r.max())
    if rmax < 1e-6:
        return None
    tn = r / rmax
    stops = _reduce_stops(_fit_stops(tn, rgb), max_delta_e=_GATE_DELTA_E)
    return {"kind": "radial",
            "geometry": {"cx": float(c[0]), "cy": float(c[1]), "r": rmax},
            "stops": stops}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_gradient.py -k fit_radial -q`
Expected: PASS (1 passed). If the recovered centre is outside tolerance, the principal-colour-axis extreme may be picking a ring rather than the core — widen the quantile band (0.9/0.1 → 0.85/0.15) before relaxing the test, since the gate (Task 6) ultimately validates.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): radial model fit with concentricity-scored center"
```

---

### Task 6: Gate + orchestration (`_render_model`, `fit_gradient`, `detect_gradients`)

**Files:**
- Modify: `src/vectoreyes/gradient.py`
- Test: `tests/test_gradient.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gradient.py`:

```python
def test_fit_gradient_accepts_linear_rejects_flat():
    from vectoreyes.gradient import fit_gradient
    h, w = 80, 120
    img = _linear_gradient_image(h, w, (10, 40), (110, 40),
                                 [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    mask = np.ones((h, w), bool)
    model = fit_gradient(mask, img)
    assert model is not None and model["kind"] == "linear"
    flat = np.full((h, w, 3), (37, 99, 235), np.uint8)
    assert fit_gradient(mask, flat) is None             # flat -> no gradient


def test_detect_gradients_consumes_ramp_returns_remaining():
    from vectoreyes.gradient import detect_gradients
    from vectoreyes.types import Region
    h, w = 60, 160
    # left half: a 4-band blue->magenta linear ramp; right: one flat green block
    img = _linear_gradient_image(h, 80, (0, 30), (79, 30),
                                 [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    full = np.zeros((h, w, 3), np.uint8)
    full[:, :80] = img
    full[:, 80:] = (20, 160, 60)
    # build the quantized regions the way the pipeline would (4 ramp bands + 1 flat)
    regions = []
    for i in range(4):
        m = np.zeros((h, w), bool); m[:, i * 20:(i + 1) * 20] = True
        regions.append(Region(label=i + 1, mask=m,
                              color_hex="#%02x%02x%02x" % tuple(np.median(full[m], axis=0).astype(int))))
    gm = np.zeros((h, w), bool); gm[:, 80:] = True
    regions.append(Region(label=5, mask=gm, color_hex="#149c3c"))
    fills, remaining = detect_gradients(regions, full)
    assert len(fills) == 1                               # the ramp became one gradient fill
    assert fills[0][1]["kind"] == "linear"
    assert {r.label for r in remaining} == {5}           # the flat green block remains
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k "fit_gradient or detect_gradients" -q`
Expected: FAIL — `fit_gradient`/`detect_gradients` not defined.

- [ ] **Step 3: Implement the gate + orchestration**

Append to `src/vectoreyes/gradient.py`:

```python
def _interp_stops_rgb(t: np.ndarray, stops: list[tuple[float, str]]) -> np.ndarray:
    offs = np.array([o for o, _ in stops])
    cols = np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for _, c in stops], float)
    return np.column_stack([np.interp(t, offs, cols[:, ch]) for ch in range(3)])


def _model_t(model: dict, pts: np.ndarray) -> np.ndarray:
    g = model["geometry"]
    if model["kind"] == "linear":
        d = np.array([g["x2"] - g["x1"], g["y2"] - g["y1"]], float)
        L = float(d @ d) or 1.0
        return (((pts[:, 0] - g["x1"]) * d[0] + (pts[:, 1] - g["y1"]) * d[1]) / L).clip(0, 1)
    r = np.hypot(pts[:, 0] - g["cx"], pts[:, 1] - g["cy"]) / (g["r"] or 1.0)
    return r.clip(0, 1)


def _agreement_delta_e(model: dict, mask: np.ndarray, rgb_image: np.ndarray) -> float:
    """Mean OKLab ΔE between the rendered model and the original over the footprint."""
    ys, xs = np.where(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    t = _model_t(model, pts)
    rendered = _interp_stops_rgb(t, model["stops"])
    truth = rgb_image[ys, xs].astype(float)
    la = srgb_to_oklab(rendered / 255.0)
    lb = srgb_to_oklab(truth / 255.0)
    return float(np.linalg.norm(la - lb, axis=1).mean())


def fit_gradient(mask: np.ndarray, rgb_image: np.ndarray) -> dict | None:
    """Fit a linear (then radial) gradient to the original pixels under `mask`. Returns
    the model only if it re-renders within _GATE_DELTA_E, else None."""
    ys, xs = np.where(mask)
    if len(xs) < 3 * _MIN_BANDS:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    rgb = rgb_image[ys, xs].astype(float)
    oklab = srgb_to_oklab(rgb / 255.0)
    best = None
    for fit in (_fit_linear, _fit_radial):
        model = fit(pts, oklab, rgb)
        if model is None:
            continue
        de = _agreement_delta_e(model, mask, rgb_image)
        if de <= _GATE_DELTA_E and (best is None or de < best[0]):
            best = (de, model)
    return best[1] if best is not None else None


def detect_gradients(
    regions: list[Region], rgb_image: np.ndarray
) -> tuple[list[tuple[Region, dict]], list[Region]]:
    """Group ramp bands, fit+gate a gradient per footprint, and return
    (accepted [(footprint_region, model)], remaining flat regions)."""
    fills: list[tuple[Region, dict]] = []
    consumed: set[int] = set()
    for group in _ramp_groups(regions):
        mask = np.zeros(rgb_image.shape[:2], bool)
        for m in group:
            mask |= m.mask
        model = fit_gradient(mask, rgb_image)
        if model is None:
            continue                                   # dissolve back into flat bands
        rep = max(group, key=lambda r: r.area)
        footprint = Region(label=rep.label, mask=mask, color_hex=rep.color_hex)
        fills.append((footprint, model))
        consumed.update(m.label for m in group)
    remaining = [r for r in regions if r.label not in consumed]
    return fills, remaining
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_gradient.py -k "fit_gradient or detect_gradients" -q`
Expected: PASS (2 passed). If `fit_gradient` rejects the true linear ramp (ΔE just over the bar), the stop count or fit is slightly off — raise `k` in `_fit_stops` to 7 before loosening `_GATE_DELTA_E`; the flat case must still return None.

- [ ] **Step 5: Run the whole gradient unit suite + full suite**

Run (own statement): `uv run pytest tests/test_gradient.py -q` then `uv run pytest -q`
Expected: all green; the full suite is unchanged (gradient.py isn't wired into the pipeline yet).

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): consistency gate (mean dE) + fit_gradient + detect_gradients"
```

---

### Task 7: Pipeline integration

**Files:**
- Modify: `src/vectoreyes/pipeline.py`
- Test: covered by Task 8 acceptance (this task wires; verify via full suite + a smoke check)

Thread the original RGB and gradient `<defs>` through `_render_body`, and call `detect_gradients` only in the upright path (`bake is None`, RGB available).

- [ ] **Step 1: Change `_render_body` to detect gradients and return defs**

In `src/vectoreyes/pipeline.py`, add the import near the others:

```python
from .gradient import detect_gradients
from .emit import linear_gradient_def, radial_gradient_def
```

Change the signature and body of `_render_body`. Update the signature line:

```python
def _render_body(
    w: int, h: int, regions: list[Region], opt: Options, *,
    bake: Affine | None = None, rgb: np.ndarray | None = None,
) -> tuple[list[str], list[str]]:
```

Immediately after `reconstructed, regions = reconstruct_scene(regions, axis, (h, w))`, insert the gradient pass (only upright, i.e. `bake is None`, and with RGB available):

```python
    defs: list[str] = []
    gradient_fills: list[tuple[Region, dict]] = []
    if rgb is not None and bake is None:
        gradient_fills, regions = detect_gradients(regions, rgb)
```

Then re-run classification on the (reduced) `regions` exactly as before (`classify_regions` / the `else` branch stays). Before the `return`, emit the gradient fills (after the occlusion-prims pass, before or interleaved with flats — order among disjoint shapes is immaterial). Add this block just before `return body`:

```python
    # gradient-filled footprints: fit the outline with the normal recognizers, emit
    # with a fill="url(#gN)" and register the gradient def. axis=None (gradient marks
    # aren't force-mirrored in this cut).
    for footprint, model in gradient_fills:
        shape = _fit_region(footprint, opt, None, corner_radius)
        if shape is None:
            continue
        gid = f"g{len(defs)}"
        if model["kind"] == "linear":
            gg = model["geometry"]
            defs.append(linear_gradient_def(gid, gg["x1"], gg["y1"], gg["x2"], gg["y2"], model["stops"]))
        else:
            gg = model["geometry"]
            defs.append(radial_gradient_def(gid, gg["cx"], gg["cy"], gg["r"], model["stops"]))
        fill = f"url(#{gid})"
        if opt.flatten:
            body.append(emit(shape_to_path_d(shape), fill, shape.params.get("fill_rule")))
        else:
            body.append(shape_to_svg(shape, fill, f"s{eid}"))
        eid += 1
```

Change the final `return body` to `return body, defs`.

- [ ] **Step 2: Update `_render_body` callers**

In `idealize` (end of function), change:

```python
    body, defs = _render_body(w, h, regions, opt, rgb=arr)
    return render_svg_doc(w, h, body, defs)
```

In `_idealize_rectified`, both `_render_body` calls return a tuple now, and gradients are OFF there (no `rgb` passed → `rgb=None`):

```python
    if opt.flatten:
        body, _ = _render_body(rw, rh, regions, opt, bake=_rectify_affine(rho, w0, h0, rw, rh))
        return render_svg_doc(w0, h0, body)
    body, _ = _render_body(rw, rh, regions, opt)
    wrap = (...)
    return render_svg_doc(w0, h0, [wrap, *body, "</g>"])
```

(`detect_gradients` is skipped in the rectified frame because `rgb` is not passed — a gradient mark with a tilted symmetry axis falls back to flat bands there, an explicit non-goal.)

- [ ] **Step 3: Smoke-check against a generated sample**

Run (own statement):

```bash
uv run vectoreyes scratch/gradient-samples/01_linear_2stop.png -o /tmp/g01.svg && grep -c "linearGradient" /tmp/g01.svg
```

Expected: prints `1` (a `<linearGradient>` is now emitted) — vs. the pre-feature baseline of 5 flat bands. (If `scratch/gradient-samples/` is absent, regenerate via `scratch/gradient-samples/_gen.py` or skip this smoke step; Task 8 is the real gate.)

- [ ] **Step 4: Full suite (regression)**

Run (own statement): `uv run pytest -q`
Expected: all existing tests green — the flat path is unchanged; existing fixtures are flat so `detect_gradients` finds nothing and returns them all in `remaining`.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/pipeline.py
git commit -m "feat(pipeline): detect gradients in the upright path; emit gradient defs"
```

---

### Task 8: Acceptance + regression

**Files:**
- Create: `tests/test_acceptance_gradient.py`

End-to-end through `idealize`, with synthetic gradients whose params are known (mirroring the `scratch/gradient-samples` set). The proof: a `<linearGradient>`/`<radialGradient>` is emitted and the render matches within a ΔE bar; flat/non-ramp inputs are untouched.

- [ ] **Step 1: Write the acceptance tests**

Create `tests/test_acceptance_gradient.py`:

```python
"""Gradient reconstruction end-to-end through idealize. A genuine gradient becomes one
shape + one <linearGradient>/<radialGradient> that re-renders within a perceptual ΔE
bar; flat and non-ramp inputs stay flat (no <defs>)."""

import numpy as np

from vectoreyes import Options, idealize
from tests._render import render_svg, mean_delta_e


def _linear_img(h, w, p0, p1, stops_rgb, bg=(255, 255, 255), shape_mask=None):
    yy, xx = np.mgrid[:h, :w]
    d = np.array(p1, float) - np.array(p0, float); L = float(d @ d)
    t = (((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb]); cols = np.array([c for _, c in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    img = img.round().astype(np.uint8)
    m = shape_mask if shape_mask is not None else np.ones((h, w), bool)
    out = np.full((h, w, 3), bg, np.uint8); out[m] = img[m]
    return out


def _radial_img(h, w, c, r, stops_rgb, bg=(255, 255, 255)):
    yy, xx = np.mgrid[:h, :w]
    dist = np.hypot(xx - c[0], yy - c[1])
    t = (dist / r).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb]); cols = np.array([col for _, col in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    out = np.full((h, w, 3), bg, np.uint8)
    disc = dist <= r
    out[disc] = img.round().astype(np.uint8)[disc]
    return out


def test_linear_gradient_reconstructs():
    h, w = 200, 260
    img = _linear_img(h, w, (40, 100), (220, 100),
                      [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    svg = idealize(img, options=Options())
    assert svg.count("<linearGradient") == 1
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.06


def test_radial_gradient_reconstructs():
    h, w = 220, 220
    img = _radial_img(h, w, (110, 110), 90,
                      [(0.0, (125, 211, 252)), (1.0, (29, 78, 216))])
    svg = idealize(img, options=Options())
    assert svg.count("<radialGradient") == 1
    assert mean_delta_e(render_svg(svg, w, h), img) <= 0.07


def test_flat_logo_not_gradientified():
    h, w = 160, 160
    img = np.full((h, w, 3), 255, np.uint8)
    img[40:120, 40:120] = (37, 99, 235)               # one flat blue square
    svg = idealize(img, options=Options())
    assert "<defs>" not in svg and "url(#" not in svg


def test_two_color_nonramp_stays_flat():
    h, w = 160, 200
    img = np.full((h, w, 3), 255, np.uint8)
    img[30:130, 20:90] = (220, 30, 30)                # red block
    img[30:130, 110:180] = (30, 30, 220)              # blue block (not a ramp)
    svg = idealize(img, options=Options())
    assert "<linearGradient" not in svg and "<radialGradient" not in svg
```

- [ ] **Step 2: Run the acceptance tests**

Run: `uv run pytest tests/test_acceptance_gradient.py -q`
Expected: PASS (4 passed). If a gradient test fails on the ΔE bar but the `<…Gradient>` is emitted, the fit is firing but imperfect — bump `_fit_stops` `k` or revisit the fit per the Task 4/5 notes; do NOT loosen the acceptance ΔE beyond 0.07. If `test_flat_logo_not_gradientified` fails (a gradient wrongly fired), the gate is too lax — tighten `_GATE_DELTA_E` (e.g. 0.05 → 0.04). The two negatives (flat, non-ramp) are the critical guards.

- [ ] **Step 3: Full regression**

Run (own statement): `uv run pytest -q`
Expected: all green — every prior fixture (occlusion, annulus, polygon, symmetry, daikonic, mastercard) unchanged, plus the gradient unit + acceptance tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_acceptance_gradient.py
git commit -m "test(acceptance): gradient reconstruction (linear, radial) + flat/non-ramp negatives"
```

---

## Self-Review

**1. Spec coverage:**
- Original-RGB-fidelity detection → `detect_gradients(regions, rgb)` fits against `rgb_image` (Tasks 6–7). ✓
- Linear + radial, shared 1-D parameter → `_fit_linear`/`_fit_radial` + `_model_t` (Tasks 4–6). ✓
- Footprint via ramp band-grouping → `_ramp_groups` (Task 3). ✓
- Consistency gate (mean OKLab ΔE) with flat-band fallback → `fit_gradient` + `_agreement_delta_e` (Task 6). ✓
- Emit `<linearGradient>`/`<radialGradient>` in `userSpaceOnUse` + `render_svg_doc` defs → Task 2; wired Task 7. ✓
- Gradient = orthogonal fill on any recognized shape → footprint goes through `_fit_region`, fill swapped (Task 7). ✓
- Priority occlusion → gradient → flat; `(fills, remaining)` contract → Tasks 6–7. ✓
- Non-goals respected: conic absent; gradient OFF in rectified frame (Task 7); no forced symmetry (axis=None). ✓
- `mean_delta_e` promoted to package (DRY) → Task 1. ✓
- Testing: unit (grouping, linear, radial, gate) + acceptance (linear, radial, flat-negative, non-ramp-negative) + regression. ✓

**2. Placeholder scan:** No TBD/"handle edge cases"/vague steps; every code step is complete; gate-tuning notes name concrete bounds (matching the spec's tunable-threshold latitude). ✓

**3. Type consistency:** Model dict shape `{"kind","geometry","stops"}` is identical across `_fit_linear`/`_fit_radial`/`_model_t`/`_agreement_delta_e`/Task 7 emit. `detect_gradients -> (list[(Region,dict)], list[Region])` matches the Task 7 consumer. `_fit_stops`/`_reduce_stops`/`_hex_to_oklab` signatures are consistent. `render_svg_doc(w,h,body,defs)` and `_render_body(...) -> (body, defs)` match all callers. ✓

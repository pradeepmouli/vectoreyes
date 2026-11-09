# Layered Gradient Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a smooth multi-hue blob can't be reduced to one parametric gradient, emit a searched parametric gradient if one fits loosely, else a bilinear stretch-fill (`<pattern><image>`), else leave it as flat bands — so Firefox/Instagram stop fragmenting while faceted art (gdrive) stays crisp.

**Architecture:** Extend only the smooth-blob fallback branch of `detect_gradients`. A new decision ladder (`_fit_smooth_blob`) tries the existing strict parametric fit (parity), then a smoothness guard (band-count + median per-pixel ΔE), then a searched parametric tier, then an adaptive stretch-fill that downsamples the footprint and lets the renderer stretch it back. A new `RasterFill` fill type plugs into the existing `resolve_fill → url(#id)` plumbing as a `<pattern>` paint server.

**Tech Stack:** Python 3.12+, numpy, scipy, Pillow (already a dependency — used for downsample/upsample and PNG encoding), pytest.

## Global Constraints

- Python ≥ 3.12; pure-Python module changes only.
- DRY is the #1 rule (per repo CLAUDE.md): factor shared model-builders rather than copying fit code.
- The strict band-merge path and `fit_gradient`'s 0.05 acceptance are **untouched**; all existing `test_gradient*`, `test_emit`, `test_candidate*`, and acceptance tests must keep passing (parity by construction).
- New gradient thresholds are **starting values**, to be corpus-validated before the PR merges (faceted-art must not regress to blur).
- Determinism: no `Math.random`/time; fixed search grids and step sizes; stable input order.
- Commit trailers: implementer commits use `Co-Authored-By: Claude Sonnet 4.6`. (Controller adds the Opus trailer on the final review/PR.)

---

### Task 1: `RasterFill` type + `<pattern><image>` emission

**Files:**
- Modify: `src/vectoreyes/candidate.py:12-32` (add type, extend `Fill` union)
- Modify: `src/vectoreyes/emit.py:8` (import), add `pattern_image_def`, extend `resolve_fill` (`src/vectoreyes/emit.py:188-201`)
- Modify: `src/vectoreyes/score.py:17` (import), `src/vectoreyes/score.py:48-54` (add branch)
- Test: `tests/test_emit.py`

**Interfaces:**
- Produces:
  - `RasterFill(geometry: dict, png_b64: str)` — `geometry = {"x","y","w","h"}` (footprint bbox in the element's frame), `png_b64` is a bare base64 PNG string (no data-URI prefix).
  - `Fill = FlatFill | LinearGradientFill | RadialGradientFill | RasterFill`
  - `pattern_image_def(elem_id: str, x, y, w, h, png_b64: str, transform: tuple | None = None) -> str`
  - `resolve_fill(fill, defs, *, geometry: dict | None = None, transform: tuple | None = None) -> str` (new `transform` kwarg; used only for `RasterFill`, emitted as `patternTransform="matrix(...)"`)

- [ ] **Step 1: Write the failing emission test**

Add to `tests/test_emit.py`:

```python
def test_pattern_image_def_emits_stretched_image():
    from vectoreyes.emit import pattern_image_def
    s = pattern_image_def("g0", 10.0, 20.0, 100.0, 80.0, "AAAA")
    assert 'id="g0"' in s and 'patternUnits="userSpaceOnUse"' in s
    assert 'width="100" height="80"' in s
    assert 'href="data:image/png;base64,AAAA"' in s
    assert 'preserveAspectRatio="none"' in s
    assert "patternTransform" not in s            # no transform given


def test_pattern_image_def_emits_transform_matrix():
    from vectoreyes.emit import pattern_image_def
    s = pattern_image_def("g1", 0.0, 0.0, 4.0, 4.0, "BBBB", transform=(1.0, 0.0, 0.0, 1.0, 5.0, 6.0))
    assert 'patternTransform="matrix(1 0 0 1 5 6)"' in s


def test_resolve_fill_registers_pattern_for_rasterfill():
    from vectoreyes.candidate import RasterFill
    from vectoreyes.emit import resolve_fill
    defs = []
    out = resolve_fill(RasterFill({"x": 0.0, "y": 0.0, "w": 8.0, "h": 8.0}, "CCCC"), defs)
    assert out == "url(#g0)"
    assert len(defs) == 1 and defs[0].startswith("<pattern")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_emit.py -k "pattern_image_def or rasterfill" -v`
Expected: FAIL — `RasterFill` / `pattern_image_def` not defined.

- [ ] **Step 3: Add the `RasterFill` dataclass and extend the union**

In `src/vectoreyes/candidate.py`, after the `RadialGradientFill` dataclass (line 29) and before `Fill = ...` (line 32):

```python
@dataclass
class RasterFill:
    """A bilinear-stretched raster fill: a small NxN PNG stretched across
    `geometry` = {x, y, w, h} (the footprint bbox) and clipped by the path it
    fills. For smooth 2-D colour fields no parametric gradient expresses.
    `png_b64` is a bare base64 PNG (no data-URI prefix)."""
    geometry: dict
    png_b64: str


Fill = FlatFill | LinearGradientFill | RadialGradientFill | RasterFill
```

(Replace the old `Fill = FlatFill | LinearGradientFill | RadialGradientFill` line.)

- [ ] **Step 4: Add `pattern_image_def` and extend `resolve_fill`**

In `src/vectoreyes/emit.py`, update the import on line 8:

```python
from .candidate import FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
```

Add after `radial_gradient_def` (after line 178):

```python
def pattern_image_def(elem_id: str, x: float, y: float, w: float, h: float,
                      png_b64: str, transform: tuple | None = None) -> str:
    """A <pattern> paint server holding one stretched <image> spanning the bbox
    (preserveAspectRatio='none' => bilinear stretch). userSpaceOnUse + absolute
    coords so it survives --flatten; `transform` (an SVG affine a,b,c,d,e,f) is
    emitted as patternTransform to map the pattern frame to a baked frame."""
    pt = ""
    if transform is not None:
        a, b, c, d, e, f = transform
        pt = (f' patternTransform="matrix({_fmt(a)} {_fmt(b)} {_fmt(c)} '
              f'{_fmt(d)} {_fmt(e)} {_fmt(f)})"')
    return (f'<pattern id="{elem_id}" patternUnits="userSpaceOnUse"{pt} '
            f'x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}">'
            f'<image href="data:image/png;base64,{png_b64}" '
            f'x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(w)}" height="{_fmt(h)}" '
            f'preserveAspectRatio="none"/></pattern>')
```

Replace `resolve_fill` (lines 188-201) with:

```python
def resolve_fill(fill, defs: list[str], *, geometry: dict | None = None,
                 transform: tuple | None = None) -> str:
    """Resolve a Fill to an SVG fill attribute. FlatFill -> its hex. Gradient/raster
    fill -> register a <def> (id g{len(defs)}, minted BEFORE the append) and return
    url(#id). `geometry` overrides the gradient/raster coords (used when the caller
    baked them); `transform` is the raster patternTransform (gradients ignore it)."""
    if isinstance(fill, FlatFill):
        return fill.hex
    gid = f"g{len(defs)}"
    g = geometry if geometry is not None else fill.geometry
    if isinstance(fill, RasterFill):
        defs.append(pattern_image_def(gid, g["x"], g["y"], g["w"], g["h"],
                                      fill.png_b64, transform))
    elif isinstance(fill, LinearGradientFill):
        defs.append(linear_gradient_def(gid, g["x1"], g["y1"], g["x2"], g["y2"], fill.stops))
    else:
        defs.append(radial_gradient_def(gid, g["cx"], g["cy"], g["r"], fill.stops))
    return f"url(#{gid})"
```

- [ ] **Step 5: Make the `score.py` complexity site total**

In `src/vectoreyes/score.py`, update the import on line 17:

```python
from .candidate import Candidate, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
```

Replace the fill-cost block (lines 49-54) with:

```python
    if isinstance(f, FlatFill):
        fill = _FILL_FLAT
    elif isinstance(f, LinearGradientFill):
        fill = 4.0 + 2.0 * len(f.stops)
    elif isinstance(f, RadialGradientFill):
        fill = 3.0 + 2.0 * len(f.stops)
    else:  # RasterFill: one embedded image, fixed cost
        fill = 8.0
```

(Defensive: raster candidates are committed gradient candidates and don't enter geometry scoring, but the isinstance chain must stay total so a stray raster fill never crashes scoring.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_emit.py -v`
Expected: PASS (new tests + existing emit tests).

- [ ] **Step 7: Commit**

```bash
git add src/vectoreyes/candidate.py src/vectoreyes/emit.py src/vectoreyes/score.py tests/test_emit.py
git commit -m "feat(gradient): RasterFill type + <pattern><image> emission

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Searched parametric fit (fallback-only) + shared model builders

**Files:**
- Modify: `src/vectoreyes/gradient.py` (factor builders from `_fit_radial`/`_fit_linear` at lines 210-265; add searched fits + `_best_parametric`)
- Test: `tests/test_gradient.py`

**Interfaces:**
- Consumes (existing, unchanged): `_fit_stops`, `_reduce_stops`, `_stop_span`, `_per_pixel_delta_e`, `_principal_axis`, `_GATE_DELTA_E`, `_MIN_STOP_SPAN`, `srgb_to_oklab`.
- Produces:
  - `_radial_model_from_center(c: np.ndarray, pts: np.ndarray, rgb: np.ndarray) -> dict | None`
  - `_linear_model_from_axis(u: np.ndarray, pts: np.ndarray, rgb: np.ndarray) -> dict | None`
  - `_fit_radial_searched(pts, oklab, rgb) -> dict | None`
  - `_fit_linear_searched(pts, oklab, rgb) -> dict | None`
  - `_best_parametric(mask: np.ndarray, rgb_image: np.ndarray) -> tuple[dict, float, float] | None` — returns `(model, mean_dE, median_dE)` for the lowest-mean-ΔE searched model whose stop-span ≥ `_MIN_STOP_SPAN`, with **no** acceptance cut; `None` if neither fits or span too small.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_gradient.py`:

```python
def test_best_parametric_searched_beats_heuristic_on_offcenter_radial():
    # a radial gradient whose centre is in a corner (where the principal-axis-extreme
    # heuristic lands poorly); the searched fit must find a low-mean-ΔE radial model.
    from vectoreyes.gradient import _best_parametric
    h, w = 80, 80
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(xx - 5, yy - 5) / np.hypot(w, h)        # centre near (5,5) corner
    img = np.empty((h, w, 3))
    for ch, (a, b) in enumerate(((230, 30), (120, 60), (40, 210))):
        img[:, :, ch] = (a + r * (b - a))
    img = img.round().astype(np.uint8)
    out = _best_parametric(np.ones((h, w), bool), img)
    assert out is not None
    model, mean_de, median_de = out
    assert model["kind"] in ("radial", "linear")
    assert mean_de < 0.05 and median_de < 0.05          # a real gradient fits tightly


def test_best_parametric_returns_none_for_flat():
    from vectoreyes.gradient import _best_parametric
    img = np.full((40, 40, 3), (50, 100, 150), np.uint8)
    assert _best_parametric(np.ones((40, 40), bool), img) is None   # span below minimum
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k best_parametric -v`
Expected: FAIL — `_best_parametric` not defined.

- [ ] **Step 3: Factor the shared model builders (DRY) and refactor the existing fits**

In `src/vectoreyes/gradient.py`, add these builders just above `_fit_radial` (before line 210):

```python
def _radial_model_from_center(c: np.ndarray, pts: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Build a radial gradient model centred at `c` (xy) over `pts`/`rgb`."""
    r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    rmax = float(r.max())
    if rmax < 1e-6:
        return None
    stops = _reduce_stops(_fit_stops(r / rmax, rgb), max_delta_e=_GATE_DELTA_E)
    return {"kind": "radial",
            "geometry": {"cx": float(c[0]), "cy": float(c[1]), "r": rmax},
            "stops": stops}


def _linear_model_from_axis(u: np.ndarray, pts: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Build a linear gradient model along unit axis `u` over `pts`/`rgb`."""
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

Refactor the tail of `_fit_radial` (lines 230-239) so it ends by delegating to the builder (behaviour-identical — the parity tests guard this):

```python
    c = best_c
    return _radial_model_from_center(c, pts, rgb)
```

Refactor the tail of `_fit_linear` (lines 252-265) likewise:

```python
    u = vt[0]                                           # unit axis direction
    return _linear_model_from_axis(u, pts, rgb)
```

- [ ] **Step 4: Add the searched fits and `_best_parametric`**

Add after `_fit_linear` (after line 265). `_model_mean_de_on_points` is defined first
because the searched fits call it:

```python
def _model_mean_de_on_points(model: dict, pts: np.ndarray, rgb: np.ndarray) -> float:
    """Mean OKLab ΔE of a model's rendered colour vs `rgb` at `pts` (footprint pixels)."""
    rendered = _interp_stops_rgb(_model_t(model, pts), model["stops"])
    return float(np.linalg.norm(srgb_to_oklab(rendered / 255.0) - srgb_to_oklab(rgb / 255.0),
                                axis=1).mean())


def _search_centers(centers, pts: np.ndarray, rgb: np.ndarray):
    """Lowest-mean-ΔE radial model over candidate centres. Returns (de, model, cx, cy) or None."""
    best = None
    for cx, cy in centers:
        model = _radial_model_from_center(np.array([cx, cy]), pts, rgb)
        if model is None:
            continue
        de = _model_mean_de_on_points(model, pts, rgb)
        if best is None or de < best[0]:
            best = (de, model, cx, cy)
    return best


def _fit_radial_searched(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Radial fit via a deterministic centre search (coarse grid over the bbox extended
    +/-50%, then a finer local grid around the best). Beats the principal-axis-extreme
    heuristic for corner-anchored / clipped fields. `oklab` is unused (kept for a uniform
    fit signature)."""
    xs, ys = pts[:, 0], pts[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    gx = np.linspace(x0 - (x1 - x0) * 0.5, x1 + (x1 - x0) * 0.5, 13)
    gy = np.linspace(y0 - (y1 - y0) * 0.5, y1 + (y1 - y0) * 0.5, 13)
    coarse = _search_centers([(cx, cy) for cx in gx for cy in gy], pts, rgb)
    if coarse is None:
        return None
    _, _, bcx, bcy = coarse
    sx, sy = gx[1] - gx[0], gy[1] - gy[0]               # one coarse step
    rx = np.linspace(bcx - sx, bcx + sx, 7)
    ry = np.linspace(bcy - sy, bcy + sy, 7)
    fine = _search_centers([(cx, cy) for cx in rx for cy in ry], pts, rgb)
    return (fine or coarse)[1]


def _fit_linear_searched(pts: np.ndarray, oklab: np.ndarray, rgb: np.ndarray) -> dict | None:
    """Linear fit via an axis-angle search (5° coarse steps, then a finer local sweep).
    `oklab` is unused (kept for a uniform fit signature)."""
    def at(ang):
        m = _linear_model_from_axis(np.array([np.cos(ang), np.sin(ang)]), pts, rgb)
        return None if m is None else (_model_mean_de_on_points(m, pts, rgb), m, ang)
    best = None
    for ang in np.linspace(0.0, np.pi, 36, endpoint=False):
        r = at(ang)
        if r is not None and (best is None or r[0] < best[0]):
            best = r
    if best is None:
        return None
    step = np.pi / 36
    for ang in np.linspace(best[2] - step, best[2] + step, 7):
        r = at(ang)
        if r is not None and r[0] < best[0]:
            best = r
    return best[1]


def _best_parametric(mask: np.ndarray, rgb_image: np.ndarray) -> tuple[dict, float, float] | None:
    """Best searched parametric (linear or radial) model for the footprint, with its
    mean and median per-pixel ΔE. No acceptance cut (the ladder applies the gates).
    None if neither model travels at least _MIN_STOP_SPAN."""
    ys, xs = np.where(mask)
    if len(xs) < 3 * _MIN_BANDS:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    rgb = rgb_image[ys, xs].astype(float)
    oklab = srgb_to_oklab(rgb / 255.0)
    best = None
    for fit in (_fit_linear_searched, _fit_radial_searched):
        model = fit(pts, oklab, rgb)
        if model is None or _stop_span(model["stops"]) < _MIN_STOP_SPAN:
            continue
        pde = _per_pixel_delta_e(model, ys, xs, rgb_image)
        mean_de, median_de = float(pde.mean()), float(np.median(pde))
        if best is None or mean_de < best[1]:
            best = (model, mean_de, median_de)
    return best
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_gradient.py -k "best_parametric or fit_radial or fit_linear or fit_gradient" -v`
Expected: PASS — new tests AND the existing `test_fit_radial_recovers_center`, `test_fit_linear_recovers_axis_and_endpoints`, `test_fit_gradient_*` (parity for the builder refactor).

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): searched parametric fit + shared model builders

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Adaptive stretch-fill

**Files:**
- Modify: `src/vectoreyes/gradient.py` (imports + new constants + `_fit_stretch` and helpers)
- Test: `tests/test_gradient.py`

**Interfaces:**
- Consumes: `srgb_to_oklab`, numpy; PIL (`from PIL import Image`), `io`, `base64`.
- Produces:
  - `_STRETCH_GRID_STEPS = (8, 16, 24, 32, 48)`, `_STRETCH_TARGET = 0.05` (module constants).
  - `_png_b64(small: np.ndarray) -> str` — bare base64 PNG of an HxWx3 uint8 array.
  - `_fit_stretch(mask: np.ndarray, rgb_image: np.ndarray) -> dict | None` — returns a raster model `{"kind": "raster", "geometry": {"x","y","w","h"}, "png_b64": str}` (smallest grid meeting `_STRETCH_TARGET`, else the largest grid), or `None` if the bbox is degenerate (<2px a side).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_gradient.py`:

```python
def _diagonal_2d_field(h, w):
    """A separable 2-D field (horizontal hue x vertical luminance) that NO single
    linear/radial gradient fits: hue runs left->right, brightness runs top->bottom."""
    yy, xx = np.mgrid[:h, :w]
    tx = xx / (w - 1)
    ty = yy / (h - 1)
    img = np.empty((h, w, 3))
    img[:, :, 0] = 30 + tx * 200                       # R climbs with x
    img[:, :, 1] = 20 + ty * 200                       # G climbs with y
    img[:, :, 2] = 200 - tx * 160                      # B falls with x
    return img.round().astype(np.uint8)


def test_fit_stretch_returns_raster_model_under_target():
    from vectoreyes.gradient import _fit_stretch, _STRETCH_TARGET
    img = _diagonal_2d_field(96, 96)
    model = _fit_stretch(np.ones((96, 96), bool), img)
    assert model is not None and model["kind"] == "raster"
    g = model["geometry"]
    assert (g["x"], g["y"], g["w"], g["h"]) == (0.0, 0.0, 96.0, 96.0)
    assert isinstance(model["png_b64"], str) and len(model["png_b64"]) > 0


def test_fit_stretch_none_for_degenerate_bbox():
    from vectoreyes.gradient import _fit_stretch
    m = np.zeros((40, 40), bool)
    m[10, 5:9] = True                                  # 1px tall footprint
    assert _fit_stretch(m, np.zeros((40, 40, 3), np.uint8)) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k fit_stretch -v`
Expected: FAIL — `_fit_stretch` not defined.

- [ ] **Step 3: Add imports and constants**

In `src/vectoreyes/gradient.py`, extend the imports near the top (after line 10):

```python
import base64
import io

from PIL import Image
```

Add constants after `_BLOB_DOMINANCE` (after line 24):

```python
_STRETCH_GRID_STEPS = (8, 16, 24, 32, 48)   # NxN downsample sizes for the stretch-fill;
                                            # the renderer bilinearly stretches the grid
                                            # back over the footprint. Last entry is the cap.
_STRETCH_TARGET = 0.05                       # grow the grid until mean per-pixel ΔE <= this.
```

- [ ] **Step 4: Add `_png_b64` and `_fit_stretch`**

Add after `_dominant_blob_fraction` (after line 357):

```python
def _png_b64(small: np.ndarray) -> str:
    """Bare base64 PNG (no data-URI prefix) of an HxWx3 uint8 array."""
    buf = io.BytesIO()
    Image.fromarray(small.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fit_stretch(mask: np.ndarray, rgb_image: np.ndarray) -> dict | None:
    """Downsample the footprint bbox to NxN and let the renderer stretch it back
    (bilinear) to reproduce a smooth 2-D field one gradient can't. Grows N over
    _STRETCH_GRID_STEPS until the upsampled reconstruction's mean per-pixel ΔE over
    the footprint is <= _STRETCH_TARGET, else returns the largest grid. None if the
    bbox is degenerate (<2px a side)."""
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    if bw < 2 or bh < 2:
        return None
    crop = rgb_image[y0:y1, x0:x1]
    ry, rx = ys - y0, xs - x0
    truth = srgb_to_oklab(rgb_image[ys, xs].astype(float) / 255.0)
    geometry = {"x": float(x0), "y": float(y0), "w": float(bw), "h": float(bh)}
    best = None
    for n in _STRETCH_GRID_STEPS:
        small = np.asarray(Image.fromarray(crop).resize((n, n), Image.BILINEAR))
        up = np.asarray(Image.fromarray(small).resize((bw, bh), Image.BILINEAR)).astype(float)
        recon = up[ry, rx]
        de = float(np.linalg.norm(srgb_to_oklab(recon / 255.0) - truth, axis=1).mean())
        if best is None or de < best[0]:
            best = (de, small)
        if de <= _STRETCH_TARGET:
            break
    return {"kind": "raster", "geometry": geometry, "png_b64": _png_b64(best[1])}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_gradient.py -k fit_stretch -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): adaptive bilinear stretch-fill for 2-D colour fields

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Smooth-blob decision ladder in `detect_gradients`

**Files:**
- Modify: `src/vectoreyes/gradient.py` (new constants + `_fit_smooth_blob`; rewire the smooth-blob block at lines 392-400)
- Test: `tests/test_gradient.py`

**Interfaces:**
- Consumes: `fit_gradient`, `_best_parametric` (Task 2), `_fit_stretch` (Task 3), `_dominant_blob_fraction`, `_union_mask`, `_BLOB_DOMINANCE`.
- Produces:
  - `_SMOOTH_BAND_MIN = 10`, `_SMOOTH_MEDIAN_TOL = 0.05`, `_PARAM_FALLBACK_TOL = 0.07` (module constants).
  - `_fit_smooth_blob(leftover: list[Region], sil: np.ndarray, rgb_image: np.ndarray) -> dict | None` — the ladder. Returns a linear/radial model (strict OR loose parametric), a raster model, or `None` (stay flat).
  - `detect_gradients` return type is unchanged: `tuple[list[tuple[Region, dict]], list[Region]]`; raster models simply appear as `dict` with `kind == "raster"`.

- [ ] **Step 1: Write the failing ladder tests**

These tests call `_fit_smooth_blob` **directly**, NOT through `detect_gradients`. That is
deliberate: in full `detect_gradients` the band-merge path runs first and consumes any
field whose colours form a clean OKLab ramp (a plain linear/radial gradient), so such a
field never reaches the fallback. Calling `_fit_smooth_blob` directly isolates the ladder's
routing on a controlled footprint. The real full-pipeline raster proof (Firefox, whose
multi-hue arc evades band-merge) is the corpus validation in Task 6.

`_fit_smooth_blob` reads `leftover` only for its length (the band-count guard) and `sil`
for the pixels it fits, so a list of count-only dummy regions plus a real `sil`/`img` fully
exercises the ladder.

Add to `tests/test_gradient.py`:

```python
def _dummy_regions(n, shape):
    """n placeholder Regions — _fit_smooth_blob uses only len(leftover) (band-count guard)."""
    from vectoreyes.types import Region
    return [Region(label=i + 1, mask=np.zeros(shape, bool), color_hex="#000000") for i in range(n)]


def _mostly_linear_plus_corner(h, w):
    """A smooth horizontal linear gradient over ~75% of the canvas plus a contrasting flat
    corner (~25%) one gradient cannot fit. The Firefox signature: the bulk follows one model
    (low MEDIAN per-pixel ΔE) while the minority drives the MEAN above the parametric bound,
    so it routes to the raster stretch-fill tier."""
    yy, xx = np.mgrid[:h, :w]
    t = xx / (w - 1)
    img = np.empty((h, w, 3))
    for ch, (a, b) in enumerate(((30, 230), (60, 60), (220, 40))):   # blue -> orange ramp
        img[:, :, ch] = a + t * (b - a)
    img[(xx >= w * 0.5) & (yy >= h * 0.5)] = (20, 230, 40)           # contrasting corner
    return img.round().astype(np.uint8)


def _three_facets(h, w):
    """3 non-collinear flat facets (a gdrive-like field): no smooth model fits its median
    pixel, so the median guard rejects it even with the band-count guard bypassed."""
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros((h, w, 3), np.uint8)
    img[xx < w // 3] = (40, 160, 90)
    img[(xx >= w // 3) & (xx < 2 * w // 3)] = (250, 200, 60)
    img[xx >= 2 * w // 3] = (60, 110, 230)
    return img


def test_fit_smooth_blob_raster_for_2d_field():
    from vectoreyes.gradient import _fit_smooth_blob
    img = _mostly_linear_plus_corner(96, 96)
    model = _fit_smooth_blob(_dummy_regions(12, (96, 96)), np.ones((96, 96), bool), img)
    assert model is not None and model["kind"] == "raster"


def test_fit_smooth_blob_strict_parametric_for_clean_gradient():
    from vectoreyes.gradient import _fit_smooth_blob
    h, w = 60, 120
    img = np.empty((h, w, 3))
    t = np.linspace(0.0, 1.0, w)
    for ch, (a, b) in enumerate(((20, 220), (40, 40), (200, 90))):
        img[:, :, ch] = a + t * (b - a)
    img = img.round().astype(np.uint8)
    model = _fit_smooth_blob(_dummy_regions(12, (h, w)), np.ones((h, w), bool), img)
    assert model is not None and model["kind"] in ("linear", "radial")   # editable, not raster


def test_fit_smooth_blob_none_for_faceted_few_bands():
    # band-count guard: faceted art with too few bands stays flat (returns None).
    from vectoreyes.gradient import _fit_smooth_blob
    img = _three_facets(90, 90)
    model = _fit_smooth_blob(_dummy_regions(3, (90, 90)), np.ones((90, 90), bool), img)
    assert model is None


def test_fit_smooth_blob_none_for_faceted_median_guard():
    # median guard: even with the band-count guard bypassed (12 dummy bands), faceted art's
    # best parametric fit has median per-pixel ΔE > _SMOOTH_MEDIAN_TOL -> None (never blurred).
    from vectoreyes.gradient import _fit_smooth_blob
    img = _three_facets(90, 90)
    model = _fit_smooth_blob(_dummy_regions(12, (90, 90)), np.ones((90, 90), bool), img)
    assert model is None
```

> Fixture-tuning note (do NOT touch production thresholds — those are corpus-validated in
> Task 6): if `test_fit_smooth_blob_raster_for_2d_field` instead lands parametric, enlarge
> the corner or its contrast; if it lands `None`, shrink the corner. If
> `test_fit_smooth_blob_none_for_faceted_median_guard` wrongly produces a model, make the
> three facet colours more clearly non-collinear in colour space.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k fit_smooth_blob -v`
Expected: FAIL — `_fit_smooth_blob` not defined.

- [ ] **Step 3: Add the ladder constants and `_fit_smooth_blob`**

In `src/vectoreyes/gradient.py`, add constants after `_STRETCH_TARGET` (from Task 3):

```python
_SMOOTH_BAND_MIN = 10        # min quantized bands (len(leftover)) for the smooth-blob fallback:
                            # a posterized continuous tone has many; faceted art has few.
_SMOOTH_MEDIAN_TOL = 0.05    # max median per-pixel ΔE of the best parametric fit: the bulk of a
                            # gradient follows some smooth model even when its mean ΔE is high
                            # (the 2-D residual is a minority). Faceted art fails this.
_PARAM_FALLBACK_TOL = 0.07   # mean ΔE bound to still prefer the editable parametric gradient
                            # over a raster stretch-fill.
```

Add `_fit_smooth_blob` just above `detect_gradients` (before line 368):

```python
def _fit_smooth_blob(leftover: list[Region], sil: np.ndarray,
                     rgb_image: np.ndarray) -> dict | None:
    """Decide how to fill one smooth dominant blob the band-merge path didn't consume:

    1. strict parametric (existing behaviour) — a clean gradient, accepted at <=_GATE_DELTA_E.
    2. smoothness guard — band_count >= _SMOOTH_BAND_MIN AND median per-pixel ΔE of the best
       searched parametric fit <= _SMOOTH_MEDIAN_TOL. Fails -> None (stay flat; faceted art).
    3. loose parametric — best searched model with mean ΔE <= _PARAM_FALLBACK_TOL (editable).
    4. stretch-fill — a 2-D field no gradient expresses.
    """
    strict = fit_gradient(sil, rgb_image)
    if strict is not None:
        return strict
    if len(leftover) < _SMOOTH_BAND_MIN:
        return None
    bp = _best_parametric(sil, rgb_image)
    if bp is None:
        return None
    model, mean_de, median_de = bp
    if median_de > _SMOOTH_MEDIAN_TOL:
        return None                                  # faceted art -> stays flat
    if mean_de <= _PARAM_FALLBACK_TOL:
        return model                                 # editable parametric gradient
    return _fit_stretch(sil, rgb_image)              # 2-D field -> raster
```

- [ ] **Step 4: Rewire the smooth-blob block in `detect_gradients`**

In `detect_gradients`, replace the current smooth-blob block (lines 392-400, the `leftover = ...` through the inner `consumed.update(...)`) with:

```python
    leftover = [r for r in regions if r.label not in consumed]
    if leftover:
        sil = _union_mask(leftover, rgb_image.shape[:2])
        if _dominant_blob_fraction(sil) >= _BLOB_DOMINANCE:
            model = _fit_smooth_blob(leftover, sil, rgb_image)
            if model is not None:
                rep = max(leftover, key=lambda r: r.area)
                fills.append((Region(label=rep.label, mask=sil, color_hex=rep.color_hex), model))
                consumed.update(r.label for r in leftover)
```

(Only the `model = ...` call changes — the strict `fit_gradient` is now the first rung inside `_fit_smooth_blob`, so the existing accept path is preserved.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_gradient.py -v`
Expected: PASS — the new `fit_smooth_blob` tests AND all existing gradient tests, including `test_detect_gradients_smooth_single_blob_fits` (strict rung), `test_detect_gradients_smooth_rejects_flat_blob`, `test_detect_gradients_smooth_rejects_multiblob` (parity).

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): smooth-blob fallback ladder (strict -> guard -> loose -> stretch)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Pipeline wiring — raster model -> `RasterFill`, bake, end-to-end

**Files:**
- Modify: `src/vectoreyes/pipeline.py:27` (import `RasterFill`), `:236-241` (model→fill mapping), `:292-297` (`_fill_attr` raster bake)
- Test: `tests/test_acceptance_gradient.py`, plus a full-suite parity run.

**Interfaces:**
- Consumes: `RasterFill` (Task 1), raster model dicts (`{"kind":"raster","geometry":{x,y,w,h},"png_b64":...}`) from `detect_gradients` (Task 4), existing `resolve_fill(..., transform=...)` (Task 1), existing `bake` affine in `_render_body`.
- Produces: `idealize(...)` maps a raster model to a `RasterFill` and emits a `<pattern><image>`; unchanged output for everything else.

These tests inject a raster model by monkeypatching `detect_gradients`, isolating the
pipeline **wiring** (model → `RasterFill` → `<pattern>`) deterministically. The real
end-to-end proof that Firefox/Instagram actually route to raster is the corpus run in
Task 6 (a synthetic field can't reliably evade band-merge AND pass the median guard).
Note `idealize`'s signature: `idealize(image, *, options=...)` (options is keyword-only).

- [ ] **Step 1: Write the failing wiring test**

Add to `tests/test_acceptance_gradient.py`:

```python
def test_pipeline_emits_pattern_for_injected_raster_model(monkeypatch):
    import numpy as np
    import vectoreyes.pipeline as P
    from vectoreyes import Options, idealize
    img = np.full((40, 40, 3), (200, 80, 60), np.uint8)        # trivial one-region mark
    raster = {"kind": "raster",
              "geometry": {"x": 0.0, "y": 0.0, "w": 40.0, "h": 40.0},
              "png_b64": "iVBORw0KGgo="}                        # any non-empty base64
    monkeypatch.setattr(P, "detect_gradients",
                        lambda comp, rgb: ([(comp[0], raster)], []))   # force a raster fill
    svg = idealize(img, options=Options())
    assert "<pattern" in svg and "<image" in svg and 'preserveAspectRatio="none"' in svg
    assert 'href="data:image/png;base64,iVBORw0KGgo="' in svg


def test_pipeline_raster_survives_flatten(monkeypatch):
    import numpy as np
    import vectoreyes.pipeline as P
    from vectoreyes import Options, idealize
    img = np.full((40, 40, 3), (200, 80, 60), np.uint8)
    raster = {"kind": "raster",
              "geometry": {"x": 0.0, "y": 0.0, "w": 40.0, "h": 40.0},
              "png_b64": "iVBORw0KGgo="}
    monkeypatch.setattr(P, "detect_gradients",
                        lambda comp, rgb: ([(comp[0], raster)], []))
    svg = idealize(img, options=Options(flatten=True))
    assert "<pattern" in svg and "<image" in svg               # raster survives --flatten
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_acceptance_gradient.py -k "injected_raster or raster_survives" -v`
Expected: FAIL — `<pattern>` not emitted (raster model isn't mapped to a fill yet).

- [ ] **Step 3: Map the raster model to `RasterFill`**

In `src/vectoreyes/pipeline.py`, extend the import on line 27:

```python
from .candidate import Candidate, Fill, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
```

Replace the fill construction in `build_candidates` (lines 236-241) with:

```python
        g = model["geometry"]
        kind = model["kind"]
        fill: Fill
        if kind == "linear":
            fill = LinearGradientFill(g, model["stops"])
        elif kind == "radial":
            fill = RadialGradientFill(g, model["stops"])
        else:  # raster
            fill = RasterFill(g, model["png_b64"])
        cands.append(Candidate(shape, fill, "gradient"))
```

- [ ] **Step 4: Handle raster in `_fill_attr` (bake -> patternTransform)**

In `src/vectoreyes/pipeline.py`, replace `_fill_attr` (lines 292-297) with:

```python
    def _fill_attr(fill: Fill) -> str:
        if isinstance(fill, RasterFill):
            # userSpaceOnUse pattern: map to the baked frame via patternTransform
            # (bake is set only in flatten mode; None otherwise -> absolute coords).
            return resolve_fill(fill, defs, transform=bake)
        baked = None
        if not isinstance(fill, FlatFill) and bake is not None:
            kind = "linear" if isinstance(fill, LinearGradientFill) else "radial"
            baked = _bake_gradient_geometry(fill.geometry, kind, bake)
        return resolve_fill(fill, defs, geometry=baked)
```

- [ ] **Step 5: Run the new tests, then the full suite for parity**

Run: `uv run pytest tests/test_acceptance_gradient.py -k "injected_raster or raster_survives" -v`
Expected: PASS.

Run: `uv run pytest -q`
Expected: PASS — the full suite (no regressions; report the verbatim summary line, e.g. `NNN passed`).

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/pipeline.py tests/test_acceptance_gradient.py
git commit -m "feat(gradient): wire raster stretch-fill through the candidate pipeline

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Corpus threshold validation + visual confirmation

**Files:**
- No production code unless a threshold must move; if one does, modify the relevant constant in `src/vectoreyes/gradient.py` and note why in its comment.
- Scratch only (untracked): render the corpus before/after.

**Interfaces:**
- Consumes: the full corpus in `scratch/` (untracked brand assets) and the finished ladder.

- [ ] **Step 1: Render the whole corpus through the new pipeline**

Run vectoreyes over every corpus mark (Firefox, Instagram, gdrive, and the rest of the gallery) and record, per mark: `kind` of any emitted gradient/raster fill, element count, and whether it changed vs `master`. Use the existing corpus script pattern in `scratch/` (do not add scratch files to git).

- [ ] **Step 2: Assert the three anchor outcomes** (the real end-to-end raster proof)

This is where Firefox→raster is confirmed against a real mark (Task 5 only tested the wiring
with an injected model). Confirm: Firefox and Instagram now emit a single `raster` fill
(element count collapses from ~29/37 to a handful); gdrive stays flat facets (NO raster, NO
new gradient — its corners are a *separate* known issue, untouched here); every other corpus
mark is unchanged from `master` (parity).

- [ ] **Step 3: If any mark regresses, tune the constant, not the test**

If a genuine gradient mark fails to fire (parametric or raster) or a flat mark wrongly turns raster, adjust the responsible constant (`_SMOOTH_BAND_MIN`, `_SMOOTH_MEDIAN_TOL`, `_PARAM_FALLBACK_TOL`, `_STRETCH_TARGET`) within a documented range and re-run Step 1. Record the final values and the evidence (which marks bound each threshold) in the commit message. Re-run `uv run pytest -q` after any constant change.

- [ ] **Step 4: Visual spot-check + commit any tuning**

Render Firefox/Instagram SVGs and confirm the stretch-fill looks smooth (not banded) and gdrive's facets are still crisp. If constants changed:

```bash
git add src/vectoreyes/gradient.py
git commit -m "fix(gradient): tune smooth-blob thresholds against the corpus

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final Review

After all tasks: dispatch a whole-branch code review (Opus), then use superpowers:finishing-a-development-branch to open the PR. PR body ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

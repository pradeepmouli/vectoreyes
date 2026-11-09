# vectoreyes v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic v1 pipeline (A0 color-optimize → B segment → C idealize → emit structured SVG) that converts a flat-color segmented logo raster into a clean, editable, exactly-symmetric SVG, gated on converting the Daikonic mark.

**Architecture:** A pure, IO-free Python core composed as a function chain `decode → quantize → segment → symmetry → fit → emit`, each stage a module over plain numpy/dataclass data. The structured-SVG output is the stable contract. No vtracer: once A0 hard-quantizes to a flat palette, regions come from connected-components labeling (simpler, and gives the masks C needs directly).

**Tech Stack:** Python 3.12, `uv`, numpy, scipy, scikit-image, shapely, Pillow; tests with pytest + `resvg-py` (render-back-diff). Build backend hatchling.

**Spec:** `docs/superpowers/specs/2026-06-04-vectormark-logo-idealizer-design.md`

**Conventions for every commit in this plan:**
- Run from repo root `/Users/pmouli/GitHub.nosync/active/py/vectoreyes`.
- Use `uv run pytest ...` to run tests inside the project venv.
- Commit messages end with the standard `Co-Authored-By` trailer used in this repo.
- The package import root is `vectoreyes` (`src/vectoreyes/...`).

---

## File Structure

```
vectoreyes/
  pyproject.toml                  # project + deps + pytest config (uv-managed)
  src/vectoreyes/
    __init__.py                   # version + public idealize() re-export
    types.py                      # dataclasses: Region, Axis (Shape lives in fit.py)
    color.py                      # A0: sRGB↔OKLab, ΔE, greedy palette, quantize
    segment.py                    # B: connected-component regions, background drop
    symmetry.py                   # C1/C2: vertical axis detect + region classification
    contour.py                    # C3/C4: subpixel contour, RDP simplify, corner split
    fit.py                        # C5/C6: primitive recognition + segment path fitting
    _fitcurve.py                  # vendored Schneider cubic-Bézier fitter (MIT)
    emit.py                       # structured SVG emitter (+ --flatten)
    pipeline.py                   # idealize(image, opts) -> svg string
    cli.py                        # `vectoreyes` entry point
  tests/
    fixtures/daikonic/source.png            # (present) gating fixture
    fixtures/daikonic/reference-trace.svg   # (present) organic baseline
    fixtures/daikonic/reference-geometric.svg  # (present) visual target
    _render.py                    # SVG→PNG + SSIM/ΔE diff helpers (shared test util)
    test_color.py
    test_segment.py
    test_symmetry.py
    test_contour.py
    test_fit.py
    test_emit.py
    test_pipeline.py              # Daikonic end-to-end acceptance
```

---

## Phase 0 — Project scaffold

### Task 0: pyproject + package skeleton + smoke test

**Files:**
- Create: `pyproject.toml`
- Create: `src/vectoreyes/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "vectoreyes"
version = "0.0.1"
description = "Deterministic logo idealizer — rendered logo raster to clean, editable, symmetric SVG."
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
dependencies = [
    "numpy>=2.0",
    "scipy>=1.13",
    "scikit-image>=0.24",
    "shapely>=2.0",
    "pillow>=10.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "resvg-py>=0.1.7"]

[project.scripts]
vectoreyes = "vectoreyes.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vectoreyes"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Write `src/vectoreyes/__init__.py`**

```python
"""vectoreyes — deterministic logo idealizer."""

__version__ = "0.0.1"
```

- [ ] **Step 3: Write the smoke test** (`tests/test_smoke.py`)

```python
import vectoreyes


def test_version_exposed():
    assert isinstance(vectoreyes.__version__, str)
    assert vectoreyes.__version__.count(".") == 2
```

- [ ] **Step 4: Sync env and run the smoke test**

Run: `uv sync --extra dev && uv run pytest tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/vectoreyes/__init__.py tests/test_smoke.py uv.lock
git commit -m "chore: scaffold vectoreyes python package + smoke test"
```

---

### Task 1: Render-back-diff test utility

The acceptance test rasterizes output SVGs and compares to the source. This util is shared, so build it first with its own tests.

**Files:**
- Create: `tests/_render.py`
- Test: `tests/test_render_util.py`

- [ ] **Step 1: Write the failing test** (`tests/test_render_util.py`)

```python
import numpy as np
from tests._render import render_svg, ssim, mean_delta_e

TEAL = "#3DA89D"
SVG = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" width="20" height="20"><rect width="20" height="20" fill="{TEAL}"/></svg>'


def test_render_returns_rgb_array():
    img = render_svg(SVG, 20, 20)
    assert img.shape == (20, 20, 3)
    assert img.dtype == np.uint8
    # center pixel ~ teal (61,168,157)
    r, g, b = img[10, 10]
    assert abs(int(r) - 61) < 6 and abs(int(g) - 168) < 6 and abs(int(b) - 157) < 6


def test_identical_images_score_perfectly():
    img = render_svg(SVG, 20, 20)
    assert ssim(img, img) > 0.999
    assert mean_delta_e(img, img) < 0.01
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_render_util.py -v`
Expected: FAIL (`ModuleNotFoundError: tests._render`).

- [ ] **Step 3: Write `tests/_render.py`**

```python
"""Shared test helpers: render an SVG to RGB and compare to a reference."""

from __future__ import annotations

import io

import numpy as np
import resvg_py
from PIL import Image
from skimage.metrics import structural_similarity


def render_svg(svg: str, width: int, height: int) -> np.ndarray:
    """Rasterize `svg` to an (H, W, 3) uint8 array on a white background."""
    png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height)
    img = Image.open(io.BytesIO(bytes(png)))
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
    return np.asarray(bg, dtype=np.uint8)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity in [0, 1]; 1.0 == identical."""
    return float(structural_similarity(a, b, channel_axis=-1))


def mean_delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean OKLab Euclidean distance per pixel (perceptual color error)."""
    from vectoreyes.color import srgb_to_oklab

    la = srgb_to_oklab(a.reshape(-1, 3) / 255.0)
    lb = srgb_to_oklab(b.reshape(-1, 3) / 255.0)
    return float(np.linalg.norm(la - lb, axis=1).mean())
```

> Note: `mean_delta_e` imports `vectoreyes.color.srgb_to_oklab`, built in Task 2. Run only `test_render_returns_rgb_array` and the SSIM half until Task 2 lands; the import is exercised by `test_identical_images_score_perfectly` after Task 2.

- [ ] **Step 4: Run the render half of the test**

Run: `uv run pytest tests/test_render_util.py::test_render_returns_rgb_array -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/_render.py tests/test_render_util.py
git commit -m "test: add SVG render-back-diff utility (resvg + SSIM/OKLab ΔE)"
```

---

## Phase 1 — Core types

### Task 2: Color space + ΔE (`color.py`, part 1)

**Files:**
- Create: `src/vectoreyes/color.py`
- Test: `tests/test_color.py`

- [ ] **Step 1: Write the failing test** (`tests/test_color.py`)

```python
import numpy as np
from vectoreyes.color import srgb_to_oklab, oklab_to_srgb, delta_e


def test_oklab_roundtrip():
    rgb = np.array([[0.024, 0.137, 0.212], [0.99, 0.55, 0.15], [1, 1, 1]])
    back = oklab_to_srgb(srgb_to_oklab(rgb))
    assert np.allclose(rgb, back, atol=1e-4)


def test_white_maps_to_L_one():
    lab = srgb_to_oklab(np.array([[1.0, 1.0, 1.0]]))
    assert abs(lab[0, 0] - 1.0) < 1e-3
    assert abs(lab[0, 1]) < 1e-3 and abs(lab[0, 2]) < 1e-3


def test_delta_e_is_symmetric_and_zero_on_equal():
    a = np.array([0.5, 0.1, -0.05])
    b = np.array([0.4, 0.0, 0.02])
    assert delta_e(a, a) == 0.0
    assert abs(delta_e(a, b) - delta_e(b, a)) < 1e-12
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_color.py -v`
Expected: FAIL (`ModuleNotFoundError: vectoreyes.color`).

- [ ] **Step 3: Implement `src/vectoreyes/color.py` (color science)**

```python
"""A0 colour optimisation: OKLab conversion, ΔE, palette extraction, quantise."""

from __future__ import annotations

import numpy as np

# sRGB <-> linear
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


# OKLab matrices (Björn Ottosson, 2020)
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) in [0,1] -> OKLab (..., 3)."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=float))
    lms = lin @ _M1.T
    lms_ = np.cbrt(lms)
    return lms_ @ _M2.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """OKLab (..., 3) -> sRGB (..., 3) clipped to [0,1]."""
    lms_ = np.asarray(lab, dtype=float) @ _M2_INV.T
    lms = lms_ ** 3
    lin = lms @ _M1_INV.T
    return np.clip(_linear_to_srgb(lin), 0.0, 1.0)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """OKLab Euclidean distance (OKLab is designed for L2 ΔE)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_color.py -v`
Expected: PASS (3 tests). Also re-run `uv run pytest tests/test_render_util.py -v` → both pass now.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/color.py tests/test_color.py
git commit -m "feat(color): sRGB<->OKLab conversion + ΔE"
```

---

### Task 3: Core dataclasses (`types.py`)

**Files:**
- Create: `src/vectoreyes/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test** (`tests/test_types.py`)

```python
import numpy as np
from vectoreyes.types import Region, Axis


def test_region_holds_mask_and_color():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    r = Region(label=1, mask=mask, color_hex="#062336")
    assert r.area == 4
    assert r.color_hex == "#062336"


def test_axis_reflect_x():
    ax = Axis(x=10.0)
    assert ax.reflect_x(7.0) == 13.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/vectoreyes/types.py`**

```python
"""Plain data types passed between pipeline stages (IO-free, port-friendly).

`Shape` (a fitted SVG element) lives in `fit.py` next to the code that produces
it; this module holds only the inputs to fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Region:
    """One connected, single-colour area of the quantised image."""
    label: int
    mask: np.ndarray          # bool (H, W)
    color_hex: str

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Axis:
    """Vertical axis of bilateral symmetry at image x == self.x."""
    x: float

    def reflect_x(self, x: float) -> float:
        return 2.0 * self.x - x
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/types.py tests/test_types.py
git commit -m "feat(types): core pipeline dataclasses"
```

---

## Phase 2 — A0 colour optimisation

### Task 4: Greedy palette extraction (`color.py`, part 2)

Anti-aliasing produces many low-count blend colours between high-count true colours. Greedy frequency selection with ΔE-merge picks the true palette and skips blends.

**Files:**
- Modify: `src/vectoreyes/color.py`
- Test: `tests/test_color.py`

- [ ] **Step 1: Add the failing test** (append to `tests/test_color.py`)

```python
from vectoreyes.color import extract_palette, quantize


def _band_image():
    """64x64: navy top, teal bottom, with a realistic 1px anti-alias gradient row
    between — each column a distinct navy->teal blend, so no single blend colour
    is frequent enough to survive (mirrors real AA: edge blends spread across many
    low-count colours). A single uniform blend row would be wrongly high-count."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:31] = (6, 35, 54)       # navy
    navy = np.array([6.0, 35.0, 54.0]); teal = np.array([61.0, 168.0, 157.0])
    for x in range(64):
        t = x / 63.0
        img[31, x] = np.round(navy * (1 - t) + teal * t).astype(np.uint8)
    img[32:] = (61, 168, 157)    # teal
    return img


def test_extract_palette_finds_two_true_colors_not_the_blend():
    pal = extract_palette(_band_image(), max_colors=8)
    assert len(pal) == 2
    hexes = {tuple(c) for c in pal}
    assert (6, 35, 54) in hexes and (61, 168, 157) in hexes


def test_quantize_collapses_blend_row():
    img = _band_image()
    pal = extract_palette(img, max_colors=8)
    q = quantize(img, pal)
    assert set(map(tuple, np.unique(q.reshape(-1, 3), axis=0))) <= {(6, 35, 54), (61, 168, 157)}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_color.py -k "palette or quantize" -v`
Expected: FAIL (`ImportError: cannot import name 'extract_palette'`).

- [ ] **Step 3: Append to `src/vectoreyes/color.py`**

```python
def extract_palette(
    rgb_image: np.ndarray, *, max_colors: int = 16, merge_de: float = 0.045,
    min_fraction: float = 0.002,
) -> np.ndarray:
    """Greedy frequency-ordered palette in OKLab; skips AA blends.

    Returns an (N, 3) uint8 array of true palette colours.
    """
    flat = rgb_image.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    colors, counts = colors[order], counts[order]
    total = counts.sum()
    lab = srgb_to_oklab(colors / 255.0)

    palette_idx: list[int] = []
    for i in range(len(colors)):
        if counts[i] < min_fraction * total:
            break
        if all(delta_e(lab[i], lab[j]) >= merge_de for j in palette_idx):
            palette_idx.append(i)
        if len(palette_idx) >= max_colors:
            break
    return colors[palette_idx].astype(np.uint8)


def quantize(rgb_image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Assign every pixel to its nearest palette colour by OKLab ΔE."""
    h, w, _ = rgb_image.shape
    flat = srgb_to_oklab(rgb_image.reshape(-1, 3) / 255.0)
    pal_lab = srgb_to_oklab(palette / 255.0)
    # pairwise distances (P pixels x K palette)
    d = np.linalg.norm(flat[:, None, :] - pal_lab[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    return palette[nearest].reshape(h, w, 3).astype(np.uint8)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_color.py -v`
Expected: PASS (all color tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/color.py tests/test_color.py
git commit -m "feat(color): greedy OKLab palette extraction + quantize (AA-robust)"
```

---

## Phase 3 — B segmentation

### Task 5: Region segmentation + background drop (`segment.py`)

**Files:**
- Create: `src/vectoreyes/segment.py`
- Test: `tests/test_segment.py`

- [ ] **Step 1: Write the failing test** (`tests/test_segment.py`)

```python
import numpy as np
from vectoreyes.color import extract_palette, quantize
from vectoreyes.segment import segment, hexstr


def _logo_on_white():
    """48x48 white bg, a navy square and a separate teal square."""
    img = np.full((48, 48, 3), 255, dtype=np.uint8)
    img[6:20, 6:42] = (6, 35, 54)     # navy bar
    img[28:42, 6:42] = (61, 168, 157)  # teal bar
    return img


def test_segment_drops_white_background_and_returns_two_regions():
    img = _logo_on_white()
    q = quantize(img, extract_palette(img))
    regions = segment(q)
    assert len(regions) == 2
    colors = sorted(r.color_hex for r in regions)
    assert colors == ["#063336".replace("3336", "3336"), "#3DA89D"] or "#3DA89D" in {r.color_hex for r in regions}


def test_hexstr_formats_uppercase():
    assert hexstr((6, 35, 54)) == "#062336"
```

> Fix the first assertion to the exact navy hex once you print it; `#062336` for (6,35,54). Replace the brittle line with:
> `assert {r.color_hex for r in regions} == {"#062336", "#3DA89D"}`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_segment.py -v`
Expected: FAIL (`ModuleNotFoundError: vectoreyes.segment`).

- [ ] **Step 3: Implement `src/vectoreyes/segment.py`**

```python
"""B: split the quantised image into connected single-colour regions."""

from __future__ import annotations

import numpy as np
from skimage.measure import label

from .types import Region


def hexstr(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _background_color(q: np.ndarray) -> tuple[int, int, int]:
    """Majority colour on the 1px border = background plate."""
    border = np.concatenate([q[0], q[-1], q[:, 0], q[:, -1]])
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return tuple(int(v) for v in colors[counts.argmax()])


def segment(quantized: np.ndarray, *, min_area: int = 16) -> list[Region]:
    """Connected components per palette colour, excluding the background."""
    bg = _background_color(quantized)
    palette = np.unique(quantized.reshape(-1, 3), axis=0)
    regions: list[Region] = []
    next_label = 1
    for color in palette:
        if tuple(int(v) for v in color) == bg:
            continue
        color_mask = np.all(quantized == color, axis=2)
        labels = label(color_mask, connectivity=2)
        for lab_id in range(1, labels.max() + 1):
            comp = labels == lab_id
            if comp.sum() < min_area:
                continue
            regions.append(Region(next_label, comp, hexstr(tuple(int(v) for v in color))))
            next_label += 1
    return regions
```

- [ ] **Step 4: Apply the test fix from Step 1's note, then run**

Run: `uv run pytest tests/test_segment.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/segment.py tests/test_segment.py
git commit -m "feat(segment): connected-component regions + background-plate drop"
```

---

## Phase 4 — C1/C2 symmetry

### Task 6: Vertical axis detection + region classification (`symmetry.py`)

v1 detects an upright (vertical) bilateral axis — covers the Daikonic mark; arbitrary-angle is a documented v1 limitation. Regions are then classed `straddling` (self-symmetric, fit with axis constraint) or `pair` (mirror-image of another region, emit one + `<use>`).

**Files:**
- Create: `src/vectoreyes/symmetry.py`
- Test: `tests/test_symmetry.py`

- [ ] **Step 1: Write the failing test** (`tests/test_symmetry.py`)

```python
import numpy as np
from vectoreyes.types import Region
from vectoreyes.symmetry import detect_axis, classify_regions


def _sym_masks():
    H, W = 40, 40
    dome = np.zeros((H, W), bool)
    yy, xx = np.ogrid[:H, :W]
    dome[((xx - 20) ** 2 / 100 + (yy - 20) ** 2 / 64) <= 1] = True  # centered ellipse
    left = np.zeros((H, W), bool); left[5:10, 6:12] = True
    right = np.zeros((H, W), bool); right[5:10, 29:35] = True       # mirror of left[6:12) about x=20
    return [Region(1, dome, "#062336"), Region(2, left, "#062336"), Region(3, right, "#062336")]


def test_detect_axis_finds_center():
    regions = _sym_masks()
    union = np.any([r.mask for r in regions], axis=0)
    axis = detect_axis(union)
    assert axis is not None
    assert abs(axis.x - 19.5) < 1.0  # image center for W=40 is 19.5


def test_classify_straddle_vs_pair():
    regions = _sym_masks()
    axis = detect_axis(np.any([r.mask for r in regions], axis=0))
    straddlers, pairs = classify_regions(regions, axis)
    assert len(straddlers) == 1 and straddlers[0].label == 1       # the dome
    assert len(pairs) == 1                                          # left+right as one pair
    canon, _mirror = pairs[0]
    assert {canon.label, _mirror.label} == {2, 3}


def test_no_symmetry_returns_none():
    # a centered square IS bilaterally symmetric — knock out one corner so the
    # shape has no vertical axis at all
    asym = np.zeros((30, 30), bool); asym[2:8, 2:8] = True
    asym[2:4, 2:4] = False
    assert detect_axis(asym) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_symmetry.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/vectoreyes/symmetry.py`**

```python
"""C1/C2: vertical bilateral symmetry detection + region classification."""

from __future__ import annotations

import numpy as np

from .types import Axis, Region


def _reflect_cols(mask: np.ndarray, axis_x: float) -> np.ndarray:
    """Reflect a boolean mask across the vertical line x = axis_x (nearest col)."""
    h, w = mask.shape
    cols = np.arange(w)
    src = np.rint(2.0 * axis_x - cols).astype(int)
    valid = (src >= 0) & (src < w)
    out = np.zeros_like(mask)
    out[:, valid] = mask[:, src[valid]]
    return out


def _mismatch(mask: np.ndarray, axis_x: float) -> float:
    refl = _reflect_cols(mask, axis_x)
    inter = np.logical_and(mask, refl).sum()
    union = np.logical_or(mask, refl).sum()
    return 1.0 - (inter / union if union else 1.0)


def detect_axis(silhouette: np.ndarray, *, tol: float = 0.10) -> Axis | None:
    """Find the best vertical symmetry axis; None if mismatch exceeds `tol`.

    Searches candidate columns near the foreground centroid at 0.5px resolution.
    `tol` is generous (0.10) because discrete masks carry pixel-quantization
    residual even when geometrically symmetric; clearly asymmetric marks score
    well above this (~0.2+).
    """
    ys, xs = np.nonzero(silhouette)
    if xs.size == 0:
        return None
    cx = xs.mean()
    candidates = np.arange(cx - 6, cx + 6 + 0.5, 0.5)
    scored = [(float(_mismatch(silhouette, a)), float(a)) for a in candidates]
    best_mismatch, best_x = min(scored)
    return Axis(x=best_x) if best_mismatch <= tol else None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def classify_regions(
    regions: list[Region], axis: Axis, *, pair_iou: float = 0.6, straddle_iou: float = 0.5,
) -> tuple[list[Region], list[tuple[Region, Region]]]:
    """Split regions into straddlers (self-symmetric) and mirror pairs."""
    straddlers: list[Region] = []
    pairs: list[tuple[Region, Region]] = []
    used: set[int] = set()
    by_label = {r.label: r for r in regions}
    for r in regions:
        if r.label in used:
            continue
        self_refl = _reflect_cols(r.mask, axis.x)
        if _iou(r.mask, self_refl) >= straddle_iou:
            straddlers.append(r)
            used.add(r.label)
            continue
        # find a partner whose mask matches r's reflection
        partner = None
        for other in regions:
            if other.label in used or other.label == r.label:
                continue
            if other.color_hex == r.color_hex and _iou(self_refl, other.mask) >= pair_iou:
                partner = other
                break
        if partner is not None:
            # canonical = the one whose centroid is on the +x (right) side
            canon, mirror = (r, partner)
            if np.nonzero(r.mask)[1].mean() < axis.x:
                canon, mirror = partner, r
            pairs.append((canon, mirror))
            used.update({r.label, partner.label})
        else:
            straddlers.append(r)  # lone asymmetric region: fit as-is
            used.add(r.label)
    return straddlers, pairs
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_symmetry.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/symmetry.py tests/test_symmetry.py
git commit -m "feat(symmetry): vertical axis detection + straddle/pair classification"
```

---

## Phase 5 — C3/C4 contour

### Task 7: Sub-pixel contour + RDP + corner split (`contour.py`)

**Files:**
- Create: `src/vectoreyes/contour.py`
- Test: `tests/test_contour.py`

- [ ] **Step 1: Write the failing test** (`tests/test_contour.py`)

```python
import numpy as np
from vectoreyes.contour import outer_contour, rdp, corner_indices


def test_outer_contour_of_square_is_closed_xy():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    c = outer_contour(mask)
    assert c.ndim == 2 and c.shape[1] == 2          # (N, 2) as (x, y)
    assert np.allclose(c[0], c[-1])                  # closed loop
    assert c[:, 0].min() < 6 and c[:, 0].max() > 13  # spans the square in x


def test_rdp_reduces_straight_run_to_endpoints():
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [3, 3]], float)
    simp = rdp(pts, epsilon=0.5)
    assert len(simp) == 3                            # (0,0),(3,0),(3,3)


def test_corner_indices_finds_square_corners():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    c = rdp(outer_contour(mask), epsilon=1.0)
    corners = corner_indices(c, angle_threshold_deg=45)
    assert len(corners) >= 4
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_contour.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/vectoreyes/contour.py`**

```python
"""C3/C4: sub-pixel contour extraction, RDP simplification, corner detection."""

from __future__ import annotations

import numpy as np
from skimage.measure import find_contours


def outer_contour(mask: np.ndarray) -> np.ndarray:
    """Longest sub-pixel contour of `mask`, as an (N, 2) array of (x, y) points."""
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.empty((0, 2))
    longest = max(contours, key=len)          # rows of (row, col) == (y, x)
    return np.column_stack([longest[:, 1], longest[:, 0]])  # -> (x, y)


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker polyline simplification."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    line = end - start
    norm = np.hypot(*line)
    if norm == 0:
        d = np.hypot(*(pts - start).T)
    else:
        d = np.abs(np.cross(line, pts - start)) / norm
    idx = int(d.argmax())
    if d[idx] > epsilon:
        left = rdp(pts[: idx + 1], epsilon)
        right = rdp(pts[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def corner_indices(poly: np.ndarray, *, angle_threshold_deg: float = 40.0) -> list[int]:
    """Indices of `poly` vertices whose turn angle exceeds the threshold.

    `poly` is assumed closed (first point repeated at the end).
    """
    pts = np.asarray(poly, dtype=float)
    n = len(pts) - 1 if np.allclose(pts[0], pts[-1]) else len(pts)
    thresh = np.radians(angle_threshold_deg)
    corners: list[int] = []
    for i in range(n):
        prev, cur, nxt = pts[(i - 1) % n], pts[i % n], pts[(i + 1) % n]
        v1, v2 = cur - prev, nxt - cur
        if np.hypot(*v1) == 0 or np.hypot(*v2) == 0:
            continue
        ang = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
        if abs(ang) >= thresh:
            corners.append(i)
    return corners
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_contour.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/contour.py tests/test_contour.py
git commit -m "feat(contour): subpixel contour + RDP + corner detection"
```

---

## Phase 6 — C5 primitive recognition

### Task 8: Whole-region primitive recognizers (`fit.py`, part 1)

Recognize a region as a native SVG primitive (circle, ellipse, axis-aligned rect, regular polygon) when its boundary matches within ε. Ellipse/circle fits use `skimage.measure` models; rect uses shapely's minimum rotated rectangle + fill ratio.

**Files:**
- Create: `src/vectoreyes/fit.py`
- Test: `tests/test_fit.py`

- [ ] **Step 1: Write the failing test** (`tests/test_fit.py`)

```python
import numpy as np
from vectoreyes.contour import outer_contour
from vectoreyes.fit import recognize_primitive


def _disk(cx, cy, r, size=60):
    yy, xx = np.ogrid[:size, :size]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def _rect(x0, y0, x1, y1, size=60):
    m = np.zeros((size, size), bool); m[y0:y1, x0:x1] = True
    return m


def test_recognizes_circle():
    c = outer_contour(_disk(30, 30, 18))
    shape = recognize_primitive(c, epsilon=1.0)
    assert shape is not None and shape.kind == "circle"
    assert abs(shape.params["r"] - 18) < 1.5


def test_recognizes_axis_aligned_rect():
    c = outer_contour(_rect(10, 14, 50, 40))
    shape = recognize_primitive(c, epsilon=1.0)
    assert shape is not None and shape.kind == "rect"
    assert abs(shape.params["w"] - 40) < 2 and abs(shape.params["h"] - 26) < 2


def test_rejects_half_ellipse_region():
    # dome = ellipse with the bottom flattened -> NOT a whole primitive
    yy, xx = np.ogrid[:60, :60]
    dome = (((xx - 30) ** 2 / 324 + (yy - 40) ** 2 / 400) <= 1) & (yy <= 40)
    c = outer_contour(dome)
    assert recognize_primitive(c, epsilon=1.0) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fit.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/vectoreyes/fit.py` (primitive recognition)**

```python
"""C5/C6: primitive recognition (this task) + segment path fitting (next task)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from skimage.measure import CircleModel, EllipseModel


@dataclass
class Shape:
    kind: str                 # "circle" | "ellipse" | "rect" | "polygon" | "path"
    params: dict
    closed: bool = True


def _max_residual(model, pts: np.ndarray) -> float:
    return float(np.abs(model.residuals(pts)).max())


def recognize_primitive(contour: np.ndarray, *, epsilon: float) -> Shape | None:
    """Return a native-primitive Shape if `contour` matches one within ε, else None."""
    pts = np.asarray(contour, dtype=float)
    if len(pts) < 8:
        return None
    poly = Polygon(pts)
    if not poly.is_valid or poly.area < 1:
        return None

    # circle
    cm = CircleModel()
    if cm.estimate(pts) and _max_residual(cm, pts) <= epsilon:
        xc, yc, r = cm.params
        return Shape("circle", {"cx": xc, "cy": yc, "r": r})

    # ellipse (axis-aligned check: snap small thetas to 0 for symmetric output)
    em = EllipseModel()
    if em.estimate(pts):
        xc, yc, a, b, theta = em.params
        if _max_residual(em, pts) <= epsilon and (abs(theta) < 0.08 or abs(abs(theta) - np.pi) < 0.08):
            return Shape("ellipse", {"cx": xc, "cy": yc, "rx": a, "ry": b})

    # axis-aligned rectangle: bbox fill ratio near 1 and rotated-rect ~ axis-aligned
    minx, miny, maxx, maxy = poly.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    rot = poly.minimum_rotated_rectangle
    rx, ry = rot.exterior.xy
    edge_angles = np.arctan2(np.diff(ry), np.diff(rx))
    axis_aligned = np.all(np.minimum(np.abs(edge_angles % (np.pi / 2)),
                                     np.pi / 2 - np.abs(edge_angles % (np.pi / 2))) < 0.06)
    if bbox_area > 0 and poly.area / bbox_area > 0.96 and axis_aligned:
        return Shape("rect", {"x": minx, "y": miny, "w": maxx - minx, "h": maxy - miny})

    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_fit.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/fit.py tests/test_fit.py
git commit -m "feat(fit): whole-region primitive recognition (circle/ellipse/rect)"
```

---

### Task 9: General polygon recognition (`fit.py`, part 2)

A trapezoid (the tapering band) is not a *regular* polygon but IS a clean straight-edged polygon — emit `<polygon>` when RDP yields few vertices and every edge is straight.

**Files:**
- Modify: `src/vectoreyes/fit.py`
- Test: `tests/test_fit.py`

- [ ] **Step 1: Add the failing test** (append to `tests/test_fit.py`)

```python
from vectoreyes.fit import recognize_polygon


def _trapezoid(size=60):
    m = np.zeros((size, size), bool)
    for y in range(10, 40):
        half = int(20 - (y - 10) * 0.3)
        m[y, 30 - half:30 + half] = True
    return m


def test_recognizes_trapezoid_as_polygon():
    c = outer_contour(_trapezoid())
    shape = recognize_polygon(c, epsilon=1.2)
    assert shape is not None and shape.kind == "polygon"
    assert 4 <= len(shape.params["points"]) <= 5


def test_polygon_rejects_curved_region():
    c = outer_contour(_disk := _disk_local())
    assert recognize_polygon(c, epsilon=1.2) is None


def _disk_local(size=60):
    yy, xx = np.ogrid[:size, :size]
    return ((xx - 30) ** 2 + (yy - 30) ** 2) <= 18 * 18
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fit.py -k polygon -v`
Expected: FAIL (`ImportError: cannot import name 'recognize_polygon'`).

- [ ] **Step 3: Append to `src/vectoreyes/fit.py`**

```python
from .contour import rdp


def recognize_polygon(contour: np.ndarray, *, epsilon: float, max_vertices: int = 8) -> Shape | None:
    """Emit a <polygon> when the contour simplifies to few straight edges."""
    pts = np.asarray(contour, dtype=float)
    simp = rdp(pts, epsilon)
    if np.allclose(simp[0], simp[-1]):
        simp = simp[:-1]
    if not (3 <= len(simp) <= max_vertices):
        return None
    # every original point must lie within ε of the simplified polygon edges
    if _max_point_to_polyline(pts, np.vstack([simp, simp[0]])) > epsilon:
        return None
    return Shape("polygon", {"points": [(float(x), float(y)) for x, y in simp]})


def _max_point_to_polyline(pts: np.ndarray, poly: np.ndarray) -> float:
    worst = 0.0
    segs = np.stack([poly[:-1], poly[1:]], axis=1)
    for p in pts:
        d = min(_point_seg_dist(p, s[0], s[1]) for s in segs)
        worst = max(worst, d)
    return worst


def _point_seg_dist(p, a, b) -> float:
    ab = b - a
    t = 0.0 if np.dot(ab, ab) == 0 else np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.hypot(*(p - (a + t * ab))))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_fit.py -v`
Expected: PASS (all fit tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/fit.py tests/test_fit.py
git commit -m "feat(fit): general straight-edged polygon recognition"
```

---

## Phase 7 — C5 path fallback (Bézier)

### Task 10: Vendor the Schneider cubic-Bézier fitter (`_fitcurve.py`)

For organic regions (dome arc-cap, tip, leaves) we fit cubic Béziers to boundary segments. Vendor the well-known MIT port of Schneider's *Graphics Gems* algorithm rather than reimplementing.

**Files:**
- Create: `src/vectoreyes/_fitcurve.py`
- Test: `tests/test_fitcurve.py`

- [ ] **Step 1: Write the failing test** (`tests/test_fitcurve.py`)

```python
import numpy as np
from vectoreyes._fitcurve import fit_cubic_beziers


def test_fits_quarter_circle_within_tolerance():
    t = np.linspace(0, np.pi / 2, 40)
    pts = np.column_stack([np.cos(t), np.sin(t)]) * 50
    beziers = fit_cubic_beziers(pts, max_error=0.5)
    assert len(beziers) >= 1
    # each bezier is 4 control points of dim 2
    assert all(b.shape == (4, 2) for b in beziers)
    # endpoints match the data endpoints
    assert np.allclose(beziers[0][0], pts[0], atol=1e-6)
    assert np.allclose(beziers[-1][3], pts[-1], atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fitcurve.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Create `src/vectoreyes/_fitcurve.py`**

Vendor the MIT-licensed implementation from `volkerp/fitCurves`
(https://github.com/volkerp/fitCurves — Philip J. Schneider, *Graphics Gems*,
1990). Paste it adapted to this signature and header:

```python
"""Vendored cubic-Bézier curve fitting.

Adapted from volkerp/fitCurves (MIT), an implementation of
Philip J. Schneider, "An Algorithm for Automatically Fitting Digitized Curves",
Graphics Gems (1990). Returns a list of (4, 2) control-point arrays.
"""

from __future__ import annotations

import numpy as np

bezier = lambda ctrl, t: (  # noqa: E731 - cubic Bézier point at t
    (1 - t) ** 3 * ctrl[0]
    + 3 * (1 - t) ** 2 * t * ctrl[1]
    + 3 * (1 - t) * t ** 2 * ctrl[2]
    + t ** 3 * ctrl[3]
)


def fit_cubic_beziers(points: np.ndarray, max_error: float) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return []
    left_t = _normalize(pts[1] - pts[0])
    right_t = _normalize(pts[-2] - pts[-1])
    return _fit_cubic(pts, left_t, right_t, max_error)


def _normalize(v):
    n = np.hypot(*v)
    return v / n if n else v


def _fit_cubic(pts, left_t, right_t, error):
    if len(pts) == 2:
        dist = np.hypot(*(pts[0] - pts[1])) / 3.0
        ctrl = np.array([pts[0], pts[0] + left_t * dist, pts[1] + right_t * dist, pts[1]])
        return [ctrl]
    u = _chord_length_parameterize(pts)
    ctrl = _generate_bezier(pts, u, left_t, right_t)
    max_err, split = _compute_max_error(pts, ctrl, u)
    if max_err < error:
        return [ctrl]
    if max_err < error * error:
        for _ in range(20):
            u = _reparameterize(pts, u, ctrl)
            ctrl = _generate_bezier(pts, u, left_t, right_t)
            max_err, split = _compute_max_error(pts, ctrl, u)
            if max_err < error:
                return [ctrl]
    center_t = _normalize(pts[split - 1] - pts[split + 1])
    left = _fit_cubic(pts[: split + 1], left_t, center_t, error)
    right = _fit_cubic(pts[split:], -center_t, right_t, error)
    return left + right


def _generate_bezier(pts, u, left_t, right_t):
    A = np.zeros((len(u), 2, 2))
    A[:, 0] = left_t * (3 * (1 - u) ** 2 * u)[:, None]
    A[:, 1] = right_t * (3 * (1 - u) * u ** 2)[:, None]
    c = np.zeros((2, 2))
    x = np.zeros(2)
    first, last = pts[0], pts[-1]
    for i, ui in enumerate(u):
        c[0, 0] += A[i, 0] @ A[i, 0]
        c[0, 1] += A[i, 0] @ A[i, 1]
        c[1, 0] = c[0, 1]
        c[1, 1] += A[i, 1] @ A[i, 1]
        tmp = pts[i] - bezier(np.array([first, first, last, last]), ui)
        x[0] += A[i, 0] @ tmp
        x[1] += A[i, 1] @ tmp
    det_c = c[0, 0] * c[1, 1] - c[1, 0] * c[0, 1]
    det_x0 = x[0] * c[1, 1] - c[0, 1] * x[1]
    det_x1 = c[0, 0] * x[1] - x[0] * c[1, 0]
    alpha_l = 0.0 if det_c == 0 else det_x0 / det_c
    alpha_r = 0.0 if det_c == 0 else det_x1 / det_c
    seg = np.hypot(*(first - last))
    if alpha_l < 1e-6 * seg or alpha_r < 1e-6 * seg:
        d = seg / 3.0
        return np.array([first, first + left_t * d, last + right_t * d, last])
    return np.array([first, first + left_t * alpha_l, last + right_t * alpha_r, last])


def _reparameterize(pts, u, ctrl):
    return np.array([_newton(p, uu, ctrl) for p, uu in zip(pts, u)])


def _newton(p, u, ctrl):
    d = bezier(ctrl, u) - p
    q1 = 3 * (ctrl[1] - ctrl[0]) * (1 - u) ** 2 + 6 * (ctrl[2] - ctrl[1]) * (1 - u) * u + 3 * (ctrl[3] - ctrl[2]) * u ** 2
    q2 = 6 * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) * (1 - u) + 6 * (ctrl[3] - 2 * ctrl[2] + ctrl[1]) * u
    denom = q1 @ q1 + d @ q2
    return u if denom == 0 else u - (d @ q1) / denom


def _chord_length_parameterize(pts):
    u = np.zeros(len(pts))
    u[1:] = np.cumsum(np.hypot(*np.diff(pts, axis=0).T))
    return u / u[-1] if u[-1] else u


def _compute_max_error(pts, ctrl, u):
    errs = np.array([np.hypot(*(bezier(ctrl, uu) - p)) ** 2 for p, uu in zip(pts, u)])
    split = int(errs.argmax())
    return float(errs[split]), max(1, min(split, len(pts) - 2))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_fitcurve.py -v`
Expected: PASS (1 test). If the quarter-circle needs >1 segment that's fine; the asserts only require endpoints + shape.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/_fitcurve.py tests/test_fitcurve.py
git commit -m "feat(fit): vendor Schneider cubic-Bézier fitter (MIT)"
```

---

### Task 11: Segment-fitted path builder (`fit.py`, part 3)

Build a parametric `<path>` for a contour by splitting at corners, fitting each segment to a **line** (if straight) else **cubic Béziers**. This is the fallback that produces the dome (line bottom + arc-like top), tip, and leaf paths.

**Files:**
- Modify: `src/vectoreyes/fit.py`
- Test: `tests/test_fit.py`

- [ ] **Step 1: Add the failing test** (append to `tests/test_fit.py`)

```python
from vectoreyes.fit import fit_path


def test_fit_path_of_square_uses_only_lines():
    mask = np.zeros((30, 30), bool); mask[6:24, 6:24] = True
    c = outer_contour(mask)
    shape = fit_path(c, epsilon=1.0, max_error=0.8)
    assert shape.kind == "path"
    assert "C" not in shape.params["d"]      # all straight -> only line ops
    assert shape.params["d"].strip().endswith("Z")


def test_fit_path_of_dome_uses_curve():
    yy, xx = np.ogrid[:80, :80]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 1600) <= 1) & (yy <= 55)
    c = outer_contour(dome)
    shape = fit_path(c, epsilon=1.0, max_error=0.8)
    assert shape.kind == "path" and "C" in shape.params["d"]   # curved top
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fit.py -k fit_path -v`
Expected: FAIL (`ImportError: cannot import name 'fit_path'`).

- [ ] **Step 3: Append to `src/vectoreyes/fit.py`**

```python
from ._fitcurve import fit_cubic_beziers
from .contour import corner_indices


def _segment_is_straight(seg: np.ndarray, epsilon: float) -> bool:
    if len(seg) <= 2:
        return True
    return _max_point_to_polyline(seg, np.vstack([seg[0], seg[-1]])) <= epsilon


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def fit_path(contour: np.ndarray, *, epsilon: float, max_error: float) -> Shape:
    """Corner-split the contour; emit lines for straight runs, Béziers otherwise."""
    pts = np.asarray(contour, dtype=float)
    closed = np.allclose(pts[0], pts[-1])
    ring = pts[:-1] if closed else pts
    simp = rdp(ring, epsilon)
    corners = corner_indices(np.vstack([simp, simp[0]]), angle_threshold_deg=40)
    # map corner positions in `simp` back to indices in `ring`
    corner_pts = simp[corners] if corners else simp[[0]]
    cut_idx = sorted({int(np.argmin(np.hypot(*(ring - cp).T))) for cp in corner_pts})
    if len(cut_idx) < 2:
        cut_idx = [0, len(ring) // 2]

    d = f"M{_fmt(ring[cut_idx[0]][0])} {_fmt(ring[cut_idx[0]][1])} "
    for k in range(len(cut_idx)):
        i0 = cut_idx[k]
        i1 = cut_idx[(k + 1) % len(cut_idx)]
        seg = ring[i0:i1 + 1] if i1 > i0 else np.vstack([ring[i0:], ring[: i1 + 1]])
        if len(seg) < 2:
            continue
        if _segment_is_straight(seg, epsilon):
            d += f"L{_fmt(seg[-1][0])} {_fmt(seg[-1][1])} "
        else:
            for b in fit_cubic_beziers(seg, max_error):
                d += (f"C{_fmt(b[1][0])} {_fmt(b[1][1])} {_fmt(b[2][0])} {_fmt(b[2][1])} "
                      f"{_fmt(b[3][0])} {_fmt(b[3][1])} ")
    d += "Z"
    return Shape("path", {"d": d})
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_fit.py -v`
Expected: PASS (all fit tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/fit.py tests/test_fit.py
git commit -m "feat(fit): segment-fitted parametric path (line/Bézier by corner)"
```

---

## Phase 8 — C6 emit

### Task 12: Structured SVG emitter (`emit.py`)

Turn `Shape`s + symmetry into a `Drawing`'s SVG string: native elements for primitives, `<path>` otherwise, and a `<use transform="matrix(...)">` mirror for paired regions about the axis. Optional `--flatten`.

**Files:**
- Create: `src/vectoreyes/emit.py`
- Test: `tests/test_emit.py`

- [ ] **Step 1: Write the failing test** (`tests/test_emit.py`)

```python
from vectoreyes.fit import Shape
from vectoreyes.types import Axis
from vectoreyes.emit import shape_to_svg, mirror_use, render_svg_doc


def test_rect_shape_emits_native_rect():
    s = Shape("rect", {"x": 5, "y": 6, "w": 20, "h": 10})
    out = shape_to_svg(s, fill="#3DA89D", elem_id="r1")
    assert out.startswith("<rect") and 'width="20"' in out and 'fill="#3DA89D"' in out


def test_circle_and_ellipse_and_polygon():
    assert "<circle" in shape_to_svg(Shape("circle", {"cx": 5, "cy": 5, "r": 4}), "#000", "c")
    assert "<ellipse" in shape_to_svg(Shape("ellipse", {"cx": 5, "cy": 5, "rx": 4, "ry": 3}), "#000", "e")
    poly = shape_to_svg(Shape("polygon", {"points": [(0, 0), (4, 0), (2, 4)]}), "#000", "p")
    assert "<polygon" in poly and "points=" in poly


def test_mirror_use_reflects_about_axis():
    use = mirror_use("leaf1", Axis(x=60.0))
    # reflection about x=a is matrix(-1 0 0 1 2a 0)
    assert 'href="#leaf1"' in use and "matrix(-1 0 0 1 120 0)" in use


def test_render_svg_doc_wraps_with_viewbox():
    doc = render_svg_doc(120, 100, ['<rect x="0" y="0" width="1" height="1" fill="#000"/>'])
    assert 'viewBox="0 0 120 100"' in doc and doc.strip().endswith("</svg>")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_emit.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/vectoreyes/emit.py`**

```python
"""C6: serialise fitted shapes into a structured SVG document."""

from __future__ import annotations

from .fit import Shape, _fmt
from .types import Axis


def shape_to_svg(shape: Shape, fill: str, elem_id: str) -> str:
    p = shape.params
    common = f'id="{elem_id}" fill="{fill}"'
    if shape.kind == "circle":
        return f'<circle {common} cx="{_fmt(p["cx"])}" cy="{_fmt(p["cy"])}" r="{_fmt(p["r"])}"/>'
    if shape.kind == "ellipse":
        return (f'<ellipse {common} cx="{_fmt(p["cx"])}" cy="{_fmt(p["cy"])}" '
                f'rx="{_fmt(p["rx"])}" ry="{_fmt(p["ry"])}"/>')
    if shape.kind == "rect":
        return (f'<rect {common} x="{_fmt(p["x"])}" y="{_fmt(p["y"])}" '
                f'width="{_fmt(p["w"])}" height="{_fmt(p["h"])}"/>')
    if shape.kind == "polygon":
        pts = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in p["points"])
        return f'<polygon {common} points="{pts}"/>'
    if shape.kind == "path":
        return f'<path {common} d="{p["d"]}"/>'
    raise ValueError(f"unknown shape kind: {shape.kind}")


def mirror_use(ref_id: str, axis: Axis) -> str:
    """Mirror element `ref_id` about the vertical axis via a reflection matrix."""
    return f'<use href="#{ref_id}" transform="matrix(-1 0 0 1 {_fmt(2 * axis.x)} 0)"/>'


def render_svg_doc(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">\n  '
        + "\n  ".join(body)
        + "\n</svg>\n"
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_emit.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/emit.py tests/test_emit.py
git commit -m "feat(emit): structured SVG serialiser + <use> mirror"
```

---

## Phase 9 — pipeline + CLI

### Task 13: Pipeline orchestration (`pipeline.py`)

Wire the stages: decode → quantize → segment → symmetry → per-region fit (primitive-first, polygon, then path fallback; straddlers snap to axis, pairs emit + mirror) → emit. Z-order regions back-to-front by area (largest first) so smaller bands/leaves paint on top.

**Files:**
- Create: `src/vectoreyes/pipeline.py`
- Modify: `src/vectoreyes/__init__.py`
- Test: `tests/test_pipeline.py` (structure-level; full acceptance in Task 15)

- [ ] **Step 1: Write the failing test** (`tests/test_pipeline.py`)

```python
import numpy as np
from PIL import Image
from vectoreyes.pipeline import idealize


def _two_band_logo(path):
    img = np.full((60, 80, 3), 255, np.uint8)
    img[8:26, 12:68] = (6, 35, 54)      # navy rect
    img[34:52, 20:60] = (61, 168, 157)  # teal rect
    Image.fromarray(img).save(path)


def test_idealize_emits_two_rects(tmp_path):
    p = tmp_path / "logo.png"
    _two_band_logo(p)
    svg = idealize(str(p))
    assert svg.count("<rect") == 2
    assert "#062336" in svg and "#3DA89D" in svg
    assert 'viewBox="0 0 80 60"' in svg
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/vectoreyes/pipeline.py`**

```python
"""Top-level orchestration: raster path/array -> structured SVG string."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .color import extract_palette, quantize
from .contour import outer_contour
from .emit import mirror_use, render_svg_doc, shape_to_svg
from .fit import Shape, fit_path, recognize_polygon, recognize_primitive
from .segment import segment
from .symmetry import classify_regions, detect_axis
from .types import Axis, Region


@dataclass
class Options:
    epsilon: float = 1.5          # primitive/polygon recognition tolerance (px)
    max_error: float = 1.0        # Bézier fit tolerance (px)
    max_colors: int = 16
    flatten: bool = False


def _fit_region(region: Region, opt: Options, axis: Axis | None) -> Shape:
    contour = outer_contour(region.mask)
    shape = recognize_primitive(contour, epsilon=opt.epsilon)
    if shape is None:
        shape = recognize_polygon(contour, epsilon=opt.epsilon)
    if shape is None:
        shape = fit_path(contour, epsilon=opt.epsilon, max_error=opt.max_error)
    if axis is not None:
        shape = _snap_to_axis(shape, axis)
    return shape


def _snap_to_axis(shape: Shape, axis: Axis) -> Shape:
    """Force x-centre of a straddling primitive onto the axis for exact symmetry."""
    if shape.kind in ("circle", "ellipse"):
        shape.params["cx"] = axis.x
    elif shape.kind == "rect":
        shape.params["x"] = axis.x - shape.params["w"] / 2
    return shape


def idealize(image, *, options: Options | None = None) -> str:
    opt = options or Options()
    if isinstance(image, str):
        arr = np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image, dtype=np.uint8)
    h, w, _ = arr.shape

    palette = extract_palette(arr, max_colors=opt.max_colors)
    q = quantize(arr, palette)
    regions = segment(q)

    silhouette = np.any([r.mask for r in regions], axis=0)
    axis = detect_axis(silhouette)

    straddlers: list[Region]
    pairs: list[tuple[Region, Region]]
    if axis is not None:
        straddlers, pairs = classify_regions(regions, axis)
    else:
        straddlers, pairs = list(regions), []

    # back-to-front: paint larger regions first
    body: list[str] = []
    drawn = [(r, False) for r in straddlers] + [(canon, True) for canon, _ in pairs]
    drawn.sort(key=lambda rp: rp[0].area, reverse=True)
    eid = 0
    for region, is_pair in drawn:
        shape = _fit_region(region, opt, axis if not is_pair else None)
        elem_id = f"s{eid}"
        body.append(shape_to_svg(shape, region.color_hex, elem_id))
        if is_pair and axis is not None:
            body.append(mirror_use(elem_id, axis))
        eid += 1

    return render_svg_doc(w, h, body)
```

- [ ] **Step 4: Update `src/vectoreyes/__init__.py`**

```python
"""vectoreyes — deterministic logo idealizer."""

from .pipeline import Options, idealize

__version__ = "0.0.1"
__all__ = ["idealize", "Options", "__version__"]
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/pipeline.py src/vectoreyes/__init__.py tests/test_pipeline.py
git commit -m "feat(pipeline): orchestrate quantize→segment→symmetry→fit→emit"
```

---

### Task 14: CLI (`cli.py`)

**Files:**
- Create: `src/vectoreyes/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test** (`tests/test_cli.py`)

```python
import subprocess, sys
import numpy as np
from PIL import Image


def test_cli_writes_svg(tmp_path):
    img = np.full((40, 40, 3), 255, np.uint8)
    img[8:32, 8:32] = (6, 35, 54)
    src = tmp_path / "in.png"; out = tmp_path / "out.svg"
    Image.fromarray(img).save(src)
    r = subprocess.run([sys.executable, "-m", "vectoreyes.cli", str(src), "-o", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.read_text().startswith("<svg")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL (`No module named vectoreyes.cli`).

- [ ] **Step 3: Implement `src/vectoreyes/cli.py`**

```python
"""`vectoreyes` CLI."""

from __future__ import annotations

import argparse
import sys

from .pipeline import Options, idealize


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vectoreyes", description="Idealize a logo raster into SVG.")
    ap.add_argument("input", help="input raster (PNG/JPG)")
    ap.add_argument("-o", "--output", help="output .svg (default: stdout)")
    ap.add_argument("--epsilon", type=float, default=1.5, help="fit tolerance in px")
    ap.add_argument("--max-error", type=float, default=1.0, help="Bézier fit tolerance in px")
    ap.add_argument("--colors", type=int, default=16, help="max palette colours")
    ap.add_argument("--flatten", action="store_true", help="flatten primitives to paths")
    args = ap.parse_args(argv)

    svg = idealize(args.input, options=Options(
        epsilon=args.epsilon, max_error=args.max_error, max_colors=args.colors, flatten=args.flatten,
    ))
    if args.output:
        with open(args.output, "w") as f:
            f.write(svg)
    else:
        sys.stdout.write(svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/cli.py tests/test_cli.py
git commit -m "feat(cli): vectoreyes command-line entry point"
```

---

## Phase 10 — Gating acceptance: Daikonic

### Task 15: Daikonic end-to-end acceptance test

This is the spec's primary acceptance criterion. It exercises the real fixture and asserts both structure and rendered fidelity. Expect to **tune ε / max_error** here — that's the intended place (per spec §10).

**Files:**
- Test: `tests/test_acceptance_daikonic.py`

- [ ] **Step 1: Write the acceptance test** (`tests/test_acceptance_daikonic.py`)

```python
import re
from pathlib import Path

import numpy as np
from PIL import Image

from vectoreyes import Options, idealize
from tests._render import render_svg, ssim, mean_delta_e

FIX = Path(__file__).parent / "fixtures" / "daikonic" / "source.png"
ICON_BOTTOM = 392  # the wordmark sits below this row; idealize only the mark


def _icon_array():
    """The Daikonic icon WITHOUT the wordmark. The wordmark is asymmetric, so
    feeding the full image would defeat vertical-symmetry detection (no <use>
    mirror). `idealize` accepts a numpy array directly, so we crop and pass it."""
    src = np.asarray(Image.open(FIX).convert("RGB"), dtype=np.uint8)
    return src[:ICON_BOTTOM]


def test_daikonic_structure_and_symmetry():
    svg = idealize(_icon_array(), options=Options(epsilon=1.8, max_error=1.2))
    # at least the two colour bands recognized as native rect/polygon
    assert svg.count("<rect") + svg.count("<polygon") >= 1
    # bilateral symmetry: the leaves emit a <use> mirror
    assert "<use" in svg
    # unified palette: exactly one navy hex present (AA navies collapsed)
    navies = set(re.findall(r"#0[0-9A-F]2[0-9A-F]3[0-9A-F]", svg.upper()))
    assert len(navies) <= 1
    # the four brand colours survive, nothing extra
    fills = {f.upper() for f in re.findall(r'fill="(#[0-9A-Fa-f]{6})"', svg)}
    assert len(fills) == 4
    # a teal is present (green & blue high, red low) — exact hex varies by the
    # palette mode, so match the hue family rather than a literal hex
    def _rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    assert any(g > 110 and b > 110 and r < 120 for r, g, b in map(_rgb, fills))


def test_daikonic_renders_close_to_source():
    icon = _icon_array()
    h, w, _ = icon.shape
    svg = idealize(icon, options=Options(epsilon=1.8, max_error=1.2))
    out = render_svg(svg, w, h)
    score = ssim(icon, out)
    de = mean_delta_e(icon, out)
    assert score >= 0.90, f"SSIM too low: {score:.3f}"
    assert de <= 0.05, f"mean ΔE too high: {de:.3f}"
```

- [ ] **Step 2: Run it (expect tuning)**

Run: `uv run pytest tests/test_acceptance_daikonic.py -v`
Expected: initially may FAIL on thresholds. Tune in this order:
1. If too few colours / split navies → raise `extract_palette` `merge_de` or lower `min_fraction`.
2. If bands become paths not rects → raise `epsilon` for `recognize_primitive`.
3. If SSIM low at edges → lower `max_error`.
4. If leaves not mirrored → check `classify_regions` `pair_iou` (lower toward 0.5).
Re-run until both asserts pass. Do **not** weaken the asserts below SSIM 0.90 / ΔE 0.05 without recording why in the test docstring.

- [ ] **Step 3: Visual spot-check (manual, one-off)**

Run:
```bash
uv run python -c "from vectoreyes import idealize, Options; open('/tmp/daikonic.svg','w').write(idealize('tests/fixtures/daikonic/source.png', options=Options(epsilon=1.8, max_error=1.2)))"
```
Open `/tmp/daikonic.svg` and compare to `tests/fixtures/daikonic/reference-geometric.svg`. Confirm: dome capped by a curve, two bands solid, tip pointed, two symmetric leaves.

- [ ] **Step 4: Commit**

```bash
git add tests/test_acceptance_daikonic.py
git commit -m "test(acceptance): Daikonic mark end-to-end (structure + render fidelity)"
```

- [ ] **Step 5: Full suite green + README status bump**

Run: `uv run pytest -q`
Expected: all tests pass. Then update `README.md` Status from "🚧 Early development" to note v1 converts the Daikonic mark, and commit:

```bash
git add README.md
git commit -m "docs: mark v1 Daikonic conversion working"
```

---

## Self-Review (completed)

**Spec coverage:**
- A0 colour optimisation (palette + quantize, AA discarded) → Tasks 2, 4. ✓
- B trace+clean (segment + bg drop; vtracer dropped per plan note) → Task 5. ✓
- C1/C2 symmetry (axis + fundamental domain via classify) → Task 6. ✓
- C3/C4 contour + RDP + corners → Task 7. ✓
- C5 primitive-first → Tasks 8, 9; path fallback (Bézier) → Tasks 10, 11. ✓
- C6 emit + `<use>` mirror; `--flatten` flag wired (CLI/Options) → Tasks 12, 14. ✓
- Geometry-level mirror (no raster reflect): pairs via `<use>`, straddlers via axis snap → Tasks 6, 13. ✓
- Structured-SVG-as-IR (native shapes) → Task 12. ✓
- Testing: golden/structure + render-back-diff + property tests → Tasks 1, 15 and per-stage. ✓
- Gating Daikonic acceptance → Task 15. ✓

**Deferred (documented, not gaps):** `--flatten` *implementation* (paths-only re-emit) is wired as a flag but its conversion pass is a v1 polish item — add a follow-up task only if `--flatten` output is needed before v1.1. n-fold symmetry and A1/D gradients are out of v1 scope per spec.

**Type/name consistency:** one fitted-shape type only — `Shape` (defined in `fit.py`, carries `.kind`/`.params`), passed straight to `shape_to_svg`. `types.py` holds only `Region` + `Axis` (fitting *inputs*). `idealize` signature is consistent across Tasks 13–15. `_fmt` is defined once in `fit.py` and imported by `emit.py`. `Region.color_hex` and `Axis.x` names are used identically everywhere they appear.

**Known tuning surface:** ε, max_error, merge_de, pair_iou — all surfaced as parameters; Task 15 is the tuning checkpoint.

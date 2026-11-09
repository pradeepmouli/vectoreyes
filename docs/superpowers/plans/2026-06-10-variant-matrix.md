# Variant Matrix (`--variants`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `vectoreyes logo.png --variants` mode that idealizes a mark across an `epsilon × max_error` matrix and writes the variant SVGs + a JSON manifest + an annotated contact-sheet PNG.

**Architecture:** A new self-contained `src/vectoreyes/variants.py` drives the matrix by calling the existing `idealize()` once per cell. A small, backward-compatible change adds `idealize(..., report=True)` returning an `IdealizeReport` (the fitter strategies the scorer chose) by reading the `Candidate.strategy` data the pipeline already produces but discards. A `--variants` CLI flag selects the mode.

**Tech Stack:** numpy, Pillow, existing vectoreyes pipeline; `resvg-py` (optional `scoring` extra) for the contact sheet only.

**Spec:** `docs/superpowers/specs/2026-06-10-variant-matrix-design.md`

---

## File Structure

- **Create** `src/vectoreyes/variants.py` — matrix driver: `DEFAULT_EPSILONS`, `DEFAULT_MAX_ERRORS`, `Variant`, `generate_variants`, `write_variant_set`, `compose_contact_sheet`.
- **Modify** `src/vectoreyes/pipeline.py` — add `IdealizeReport` + `_report_from_cands`; `_render_body` returns candidates; `_idealize_rectified` returns candidates; `idealize(..., report=False)`.
- **Modify** `src/vectoreyes/cli.py` — `--variants`, `--out-dir`, `--epsilons`, `--max-errors`.
- **Test** `tests/test_pipeline_report.py`, `tests/test_variants.py`.

---

## Task 1: `idealize(..., report=True)` surfaces a strategy report

**Files:**
- Modify: `src/vectoreyes/pipeline.py`
- Test: `tests/test_pipeline_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_report.py
import numpy as np
from PIL import Image, ImageDraw

from vectoreyes.pipeline import IdealizeReport, idealize


def _disc(n=64):
    im = Image.new("RGB", (n, n), "white")
    ImageDraw.Draw(im).ellipse((10, 10, n - 10, n - 10), fill=(30, 100, 220))
    return np.asarray(im, dtype=np.uint8)


def test_idealize_default_returns_bare_string():
    out = idealize(_disc())
    assert isinstance(out, str) and out.startswith("<svg ")


def test_idealize_report_returns_strategy_counts():
    svg, report = idealize(_disc(), report=True)
    assert isinstance(svg, str) and svg.startswith("<svg ")
    assert isinstance(report, IdealizeReport)
    # a clean disc is recognised as a primitive (circle)
    assert report.strategies.get("primitive", 0) >= 1
    assert report.elements >= 1
    assert report.gradients == 0


def test_report_empty_for_blank_image():
    blank = np.full((40, 40, 3), 255, dtype=np.uint8)
    svg, report = idealize(blank, report=True)
    assert report.elements == 0 and dict(report.strategies) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_pipeline_report.py -q`
Expected: FAIL — `ImportError: cannot import name 'IdealizeReport'` / `idealize() got an unexpected keyword argument 'report'`.

- [ ] **Step 3: Add `IdealizeReport` + `_report_from_cands` to `pipeline.py`**

At the top of `src/vectoreyes/pipeline.py`, add to the imports:

```python
import types
from collections.abc import Mapping
```

(`dataclass` is already imported.) Add this dataclass and helper just above `def build_candidates(` (around line 135):

```python
@dataclass(frozen=True)
class IdealizeReport:
    """What the pipeline actually emitted for one idealize() run: the histogram of
    fitter strategies the scorer chose per region, the gradient-fill count, and the
    total emitted element count. Diagnostic annotation for the variant matrix."""

    strategies: Mapping[str, int]
    gradients: int
    elements: int

    @staticmethod
    def empty() -> "IdealizeReport":
        return IdealizeReport(types.MappingProxyType({}), 0, 0)


def _report_from_cands(cands: list[Candidate]) -> IdealizeReport:
    strategies: dict[str, int] = {}
    gradients = 0
    for c in cands:
        if c.source == "gradient":
            gradients += 1
        if c.strategy is not None:                 # None for occlusion / lens / gradient
            strategies[c.strategy] = strategies.get(c.strategy, 0) + 1
    return IdealizeReport(types.MappingProxyType(dict(strategies)), gradients, len(cands))
```

- [ ] **Step 4: Make `_render_body` return its candidates**

Change `_render_body`'s signature (line 201-204) return type and the final return (line 270):

```python
def _render_body(
    w: int, h: int, regions: list[Region], opt: Options, *,
    bake: Affine | None = None, rgb: np.ndarray | None = None,
) -> tuple[list[str], list[str], list[Candidate]]:
```

Change the final line from `return body, defs` to:

```python
    return body, defs, cands
```

- [ ] **Step 5: Thread candidates through `_idealize_rectified`**

Change `_idealize_rectified` (lines 297-318) to return `tuple[str | None, list[Candidate]]`. Replace its body's return points:

```python
def _idealize_rectified(arr: np.ndarray, opt: Options, rho: float, w0: int, h0: int) -> tuple[str | None, list[Candidate]]:
    """...(unchanged docstring)..."""
    rot = ndi.rotate(arr.astype(float), -rho, reshape=True, order=1, cval=255.0)
    rot = np.clip(rot, 0.0, 255.0).astype(np.uint8)
    rw, rh, regions = _segment_image(rot, opt)
    if not regions:
        return None, []
    if detect_axis(np.any([r.mask for r in regions], axis=0)) is None:
        return None, []
    if opt.flatten:
        body, defs, cands = _render_body(rw, rh, regions, opt,
                                         bake=_rectify_affine(rho, w0, h0, rw, rh), rgb=rot)
        return render_svg_doc(w0, h0, body, defs), cands
    body, defs, cands = _render_body(rw, rh, regions, opt, rgb=rot)
    wrap = (f'<g transform="translate({_fmt(w0 / 2)} {_fmt(h0 / 2)}) '
            f'rotate({_fmt(round(-rho, 3))}) translate({_fmt(-rw / 2)} {_fmt(-rh / 2)})">')
    return render_svg_doc(w0, h0, [wrap, *body, "</g>"], defs), cands
```

- [ ] **Step 6: Rewrite `idealize` with `report=` and a single return**

Replace the whole `idealize` function (lines 334-362) with:

```python
def idealize(image, *, options: Options | None = None, report: bool = False):
    """Idealize a raster mark into SVG. With `report=True`, returns
    `(svg, IdealizeReport)`; otherwise returns the SVG string (back-compatible)."""
    opt = options or Options()
    if isinstance(image, str):
        with Image.open(image) as im:
            arr = _flatten_on_white(im)
    else:
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:            # RGBA array -> composite on white
            arr = _flatten_on_white(Image.fromarray(arr, "RGBA"))
    h0, w0 = arr.shape[:2]

    w, h, regions = _segment_image(arr, opt)
    if not regions:
        svg, cands = render_svg_doc(w, h, []), []
    else:
        svg, cands = None, []
        # Any-axis symmetry: rectify a tilted mirror upright, idealize there, wrap back.
        if not opt.no_symmetry:
            silhouette = np.any([r.mask for r in regions], axis=0)
            if detect_axis(silhouette) is None:
                rho = detect_symmetry_rotation(silhouette)
                if rho is not None:
                    rectified, rcands = _idealize_rectified(arr, opt, rho, w0, h0)
                    if rectified is not None:
                        svg, cands = rectified, rcands
        if svg is None:
            body, defs, cands = _render_body(w, h, regions, opt, rgb=arr)
            svg = render_svg_doc(w, h, body, defs)

    return (svg, _report_from_cands(cands)) if report else svg
```

- [ ] **Step 7: Run the report tests + full suite**

Run: `uv run --extra dev pytest tests/test_pipeline_report.py -q && uv run --extra dev pytest -q`
Expected: PASS — all report tests green, no regressions (the `report=False` default keeps every existing `idealize()` caller returning a bare `str`).

- [ ] **Step 8: Commit**

```bash
git add src/vectoreyes/pipeline.py tests/test_pipeline_report.py
git commit -m "feat(pipeline): idealize(report=True) surfaces fitter-strategy report"
```

---

## Task 2: `generate_variants` over the matrix

**Files:**
- Create: `src/vectoreyes/variants.py`
- Test: `tests/test_variants.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_variants.py
import numpy as np
from PIL import Image, ImageDraw

from vectoreyes.variants import (
    DEFAULT_EPSILONS, DEFAULT_MAX_ERRORS, Variant, generate_variants,
)


def _mark(n=80):
    im = Image.new("RGB", (n, n), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((8, 8, n - 8, n - 8), fill=(30, 100, 220))
    d.rectangle((30, 30, 50, 70), fill=(220, 40, 40))
    return np.asarray(im, dtype=np.uint8)


def test_generate_variants_grid_shape_and_order():
    eps, mes = (0.5, 3.0), (0.5, 2.5)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    assert len(variants) == 4
    # row-major: epsilon outer, max_error inner
    assert [(v.epsilon, v.max_error) for v in variants] == [
        (0.5, 0.5), (0.5, 2.5), (3.0, 0.5), (3.0, 2.5),
    ]
    for v in variants:
        assert isinstance(v, Variant)
        assert v.svg.startswith("<svg ")
        assert v.error is None
        assert v.report.elements >= 1


def test_generate_variants_defaults_are_3x3():
    variants = generate_variants(_mark())
    assert len(variants) == len(DEFAULT_EPSILONS) * len(DEFAULT_MAX_ERRORS) == 9


def test_generate_variants_params_take_effect():
    variants = generate_variants(_mark(), epsilons=(0.3, 6.0), max_errors=(0.5,))
    # a very loose epsilon must change the geometry vs a very tight one
    assert variants[0].svg != variants[1].svg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_variants.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vectoreyes.variants'`.

- [ ] **Step 3: Create `variants.py` with `generate_variants`**

```python
# src/vectoreyes/variants.py
"""Whole-mark variant matrix: idealize one raster across an epsilon × max_error
grid, for SVG export, a JSON manifest, and an annotated contact sheet."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from .pipeline import IdealizeReport, Options, _flatten_on_white, idealize

DEFAULT_EPSILONS = (0.5, 1.5, 3.0)
DEFAULT_MAX_ERRORS = (0.5, 1.0, 2.5)


@dataclass(frozen=True)
class Variant:
    """One matrix cell: the (epsilon, max_error) used, the emitted SVG, the
    strategy report, and an error string if that cell failed to idealize."""

    epsilon: float
    max_error: float
    svg: str
    report: IdealizeReport
    error: str | None = None


def _as_rgb(image) -> np.ndarray:
    """Load any accepted image input to an (H, W, 3) uint8 array ONCE, so the grid
    does not re-read/re-flatten the source per cell."""
    if isinstance(image, str):
        with Image.open(image) as im:
            return _flatten_on_white(im)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return _flatten_on_white(Image.fromarray(arr, "RGBA"))
    return arr


def generate_variants(
    image, *, epsilons=DEFAULT_EPSILONS, max_errors=DEFAULT_MAX_ERRORS,
    base: Options | None = None,
) -> list[Variant]:
    """idealize() the image once per (epsilon, max_error) cell, row-major (epsilon
    outer, max_error inner). `base` supplies the non-geometry knobs held constant
    across the grid (max_colors, no_symmetry, …); epsilon/max_error are overridden
    per cell. A cell that raises is captured as a failed Variant, never aborting
    the grid."""
    arr = _as_rgb(image)
    base = base or Options()
    out: list[Variant] = []
    for eps in epsilons:
        for me in max_errors:
            opt = replace(base, epsilon=eps, max_error=me)
            try:
                svg, report = idealize(arr, options=opt, report=True)
                out.append(Variant(eps, me, svg, report))
            except Exception as exc:                       # one bad cell must not kill the grid
                out.append(Variant(eps, me, "", IdealizeReport.empty(), error=str(exc)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_variants.py -q`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/variants.py tests/test_variants.py
git commit -m "feat(variants): generate_variants over an epsilon x max_error grid"
```

---

## Task 3: `write_variant_set` — SVGs + manifest

**Files:**
- Modify: `src/vectoreyes/variants.py`
- Test: `tests/test_variants.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_variants.py`:

```python
import json

from vectoreyes.variants import write_variant_set


def test_write_variant_set_writes_svgs_and_manifest(tmp_path):
    variants = generate_variants(_mark(), epsilons=(0.5, 3.0), max_errors=(1.0,))
    write_variant_set(variants, tmp_path, source="mark.png")

    assert (tmp_path / "variant-e0_5-m1.svg").read_text().startswith("<svg ")
    assert (tmp_path / "variant-e3-m1.svg").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["source"] == "mark.png"
    assert manifest["axes"] == {"epsilon": [0.5, 3.0], "max_error": [1.0]}
    assert len(manifest["variants"]) == 2
    first = manifest["variants"][0]
    assert first["epsilon"] == 0.5 and first["max_error"] == 1.0
    assert first["file"] == "variant-e0_5-m1.svg"
    assert first["svg_bytes"] > 0
    assert isinstance(first["strategies"], dict) and first["elements"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_variants.py::test_write_variant_set_writes_svgs_and_manifest -q`
Expected: FAIL — `ImportError: cannot import name 'write_variant_set'`.

- [ ] **Step 3: Add `write_variant_set` to `variants.py`**

Add imports at the top of `variants.py`:

```python
import json
from pathlib import Path
```

and add:

```python
from .fit import _fmt


def _axis_tag(value: float) -> str:
    """Filename-safe axis token: 0.5 -> '0_5', 3.0 -> '3', 1.0 -> '1'."""
    return _fmt(value).replace(".", "_")


def variant_filename(v: Variant) -> str:
    return f"variant-e{_axis_tag(v.epsilon)}-m{_axis_tag(v.max_error)}.svg"


def write_variant_set(variants: list[Variant], out_dir, *, source: str) -> Path:
    """Write one SVG per successful cell plus manifest.json into out_dir (created if
    needed). A failed cell contributes a manifest entry with `error` and no `file`.
    Returns the out_dir path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    epsilons, max_errors = [], []
    for v in variants:
        if v.epsilon not in epsilons:
            epsilons.append(v.epsilon)
        if v.max_error not in max_errors:
            max_errors.append(v.max_error)

    entries = []
    for v in variants:
        entry = {"epsilon": v.epsilon, "max_error": v.max_error}
        if v.error is not None:
            entry["error"] = v.error
        else:
            fname = variant_filename(v)
            (out / fname).write_text(v.svg)
            entry.update(
                file=fname,
                svg_bytes=len(v.svg.encode()),
                strategies=dict(v.report.strategies),
                gradients=v.report.gradients,
                elements=v.report.elements,
            )
        entries.append(entry)

    manifest = {
        "source": source,
        "axes": {"epsilon": epsilons, "max_error": max_errors},
        "variants": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_variants.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/variants.py tests/test_variants.py
git commit -m "feat(variants): write_variant_set emits SVGs + annotated manifest"
```

---

## Task 4: `compose_contact_sheet` — annotated grid PNG (renderer-optional)

**Files:**
- Modify: `src/vectoreyes/variants.py`
- Test: `tests/test_variants.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_variants.py`:

```python
import io

import pytest

import vectoreyes.variants as V
from vectoreyes.score import SvgRendererUnavailable


def _renderer_available():
    try:
        import resvg_py  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not _renderer_available(), reason="needs resvg-py")
def test_contact_sheet_renders_grid_png():
    eps, mes = (0.5, 3.0), (0.5, 2.5)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    png = V.compose_contact_sheet(variants, epsilons=eps, max_errors=mes)
    assert isinstance(png, (bytes, bytearray)) and len(png) > 0
    img = Image.open(io.BytesIO(bytes(png)))
    assert img.width > 0 and img.height > 0


def test_contact_sheet_none_without_renderer(monkeypatch):
    def boom(*a, **k):
        raise SvgRendererUnavailable("no renderer")
    monkeypatch.setattr(V, "_rasterize", boom)
    eps, mes = (0.5,), (0.5,)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    assert V.compose_contact_sheet(variants, epsilons=eps, max_errors=mes) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_variants.py -k contact_sheet -q`
Expected: FAIL — `AttributeError: module 'vectoreyes.variants' has no attribute 'compose_contact_sheet'` / `_rasterize`.

- [ ] **Step 3: Add the contact-sheet composer to `variants.py`**

Add imports at the top of `variants.py`:

```python
from PIL import ImageDraw
from .score import SvgRendererUnavailable, _rasterize
```

and add:

```python
_TILE = 220          # rendered variant size (px)
_PAD = 10            # gap between tiles
_LABEL_H = 26        # caption strip height under each tile
_AXIS_W = 80         # left gutter for epsilon row labels
_AXIS_H = 24         # top strip for max_error column labels


def _histogram_caption(report: IdealizeReport) -> str:
    """Compact strategy histogram, e.g. 'prim×3 sym_poly×2 path×1'."""
    short = {"primitive": "prim", "sym_polygon": "sym_poly", "trapezoid": "trap",
             "symmetric": "sym", "polygon": "poly", "holed_symmetric": "holed_sym",
             "holed_path": "holed", "cap": "cap", "path": "path"}
    parts = [f"{short.get(k, k)}×{n}" for k, n in sorted(report.strategies.items())]
    if report.gradients:
        parts.append(f"grad×{report.gradients}")
    return "  ".join(parts) or "(none)"


def _render_tile(v: Variant) -> Image.Image:
    """One variant rendered into a _TILE×_TILE white tile, or a placeholder for a
    failed cell. Raises SvgRendererUnavailable if the renderer is missing."""
    tile = Image.new("RGB", (_TILE, _TILE), "white")
    if v.error is None and v.svg:
        arr = _rasterize(v.svg, _TILE, _TILE)          # may raise SvgRendererUnavailable
        tile = Image.fromarray(arr)
    else:
        ImageDraw.Draw(tile).text((8, _TILE // 2), "failed", fill=(180, 40, 40))
    return tile


def compose_contact_sheet(variants: list[Variant], *, epsilons, max_errors) -> bytes | None:
    """Render the variants into an annotated grid PNG (epsilon rows × max_error
    columns), each tile captioned with its strategy histogram and the axes labelled.
    Returns PNG bytes, or None if the SVG renderer is unavailable."""
    by_cell = {(v.epsilon, v.max_error): v for v in variants}
    cell_w = _TILE + _PAD
    cell_h = _TILE + _LABEL_H + _PAD
    sheet_w = _AXIS_W + len(max_errors) * cell_w + _PAD
    sheet_h = _AXIS_H + len(epsilons) * cell_h + _PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for ci, me in enumerate(max_errors):
        x = _AXIS_W + ci * cell_w + _PAD
        draw.text((x, 6), f"max_error={_fmt(me)}", fill=(20, 30, 40))

    try:
        for ri, eps in enumerate(epsilons):
            y0 = _AXIS_H + ri * cell_h + _PAD
            draw.text((8, y0 + _TILE // 2), f"ε={_fmt(eps)}", fill=(20, 30, 40))
            for ci, me in enumerate(max_errors):
                v = by_cell[(eps, me)]
                tile = _render_tile(v)                  # may raise SvgRendererUnavailable
                x = _AXIS_W + ci * cell_w + _PAD
                sheet.paste(tile, (x, y0))
                draw.text((x, y0 + _TILE + 4), _histogram_caption(v.report), fill=(60, 70, 80))
    except SvgRendererUnavailable:
        return None

    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()
```

Add `import io` to the top of `variants.py` if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_variants.py -k contact_sheet -q`
Expected: PASS — grid renders with resvg; returns `None` when `_rasterize` is monkeypatched to raise.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/variants.py tests/test_variants.py
git commit -m "feat(variants): annotated contact-sheet PNG (renderer-optional)"
```

---

## Task 5: CLI `--variants` mode

**Files:**
- Modify: `src/vectoreyes/cli.py`
- Test: `tests/test_variants.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_variants.py`:

```python
from vectoreyes.cli import main as cli_main


def test_cli_variants_writes_set(tmp_path):
    src = tmp_path / "mark.png"
    Image.fromarray(_mark()).save(src)
    out = tmp_path / "looks"
    rc = cli_main([str(src), "--variants", "--out-dir", str(out),
                   "--epsilons", "0.5,3", "--max-errors", "1"])
    assert rc == 0
    assert (out / "manifest.json").exists()
    assert (out / "variant-e0_5-m1.svg").exists()
    assert (out / "variant-e3-m1.svg").exists()


def test_cli_variants_rejects_bad_axis(tmp_path):
    src = tmp_path / "mark.png"
    Image.fromarray(_mark()).save(src)
    with pytest.raises(SystemExit):
        cli_main([str(src), "--variants", "--epsilons", "abc"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_variants.py -k cli_variants -q`
Expected: FAIL — `--variants`/`--out-dir`/`--epsilons` are unrecognised arguments.

- [ ] **Step 3: Wire `--variants` into `cli.py`**

Replace the contents of `src/vectoreyes/cli.py`'s `main` with:

```python
def _floats(text: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(t) for t in text.split(",") if t.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}")
    if not vals:
        raise argparse.ArgumentTypeError("at least one value required")
    return vals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vectoreyes", description="Idealize a logo raster into SVG.")
    ap.add_argument("input", help="input raster (PNG/JPG)")
    ap.add_argument("-o", "--output", help="output .svg (default: stdout)")
    ap.add_argument("--epsilon", type=float, default=1.5, help="fit tolerance in px")
    ap.add_argument("--max-error", type=float, default=1.0, help="Bézier fit tolerance in px")
    ap.add_argument("--colors", type=int, default=16, help="max palette colours")
    ap.add_argument("--flatten", action="store_true", help="flatten primitives to paths")
    ap.add_argument("--no-symmetry", action="store_true", help="disable symmetry detection")
    ap.add_argument("--variants", action="store_true",
                    help="write an epsilon × max_error matrix of variants to a directory")
    ap.add_argument("--out-dir", help="output directory for --variants (default: ./<stem>-variants)")
    ap.add_argument("--epsilons", type=_floats, help="--variants epsilon axis, e.g. 0.5,1.5,3")
    ap.add_argument("--max-errors", type=_floats, help="--variants max_error axis, e.g. 0.5,1,2.5")
    args = ap.parse_args(argv)

    base = Options(max_colors=args.colors, flatten=args.flatten, no_symmetry=args.no_symmetry)

    if args.variants:
        from pathlib import Path

        from .variants import (
            DEFAULT_EPSILONS, DEFAULT_MAX_ERRORS, compose_contact_sheet,
            generate_variants, write_variant_set,
        )

        epsilons = args.epsilons or DEFAULT_EPSILONS
        max_errors = args.max_errors or DEFAULT_MAX_ERRORS
        out_dir = Path(args.out_dir) if args.out_dir else Path(f"{Path(args.input).stem}-variants")

        variants = generate_variants(args.input, epsilons=epsilons, max_errors=max_errors, base=base)
        write_variant_set(variants, out_dir, source=args.input)
        png = compose_contact_sheet(variants, epsilons=epsilons, max_errors=max_errors)
        if png is not None:
            (out_dir / "contact-sheet.png").write_bytes(png)
        else:
            print("contact sheet skipped (install vectoreyes[scoring] to render it)", file=sys.stderr)
        print(f"{len(variants)} variants -> {out_dir}"
              f"{' (+contact-sheet.png)' if png is not None else ''}")
        return 0

    svg = idealize(args.input, options=Options(
        epsilon=args.epsilon, max_error=args.max_error, max_colors=args.colors,
        flatten=args.flatten, no_symmetry=args.no_symmetry,
    ))
    if args.output:
        with open(args.output, "w") as f:
            f.write(svg)
    else:
        sys.stdout.write(svg)
    return 0
```

(Keep the existing module imports at the top of `cli.py`: `argparse`, `sys`, and `from .pipeline import Options, idealize`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_variants.py -k cli_variants -q`
Expected: PASS — variants written; a bad `--epsilons` raises `SystemExit` (argparse error).

- [ ] **Step 5: Run the full suite**

Run: `uv run --extra dev pytest -q`
Expected: PASS — no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/cli.py tests/test_variants.py
git commit -m "feat(cli): --variants writes the epsilon x max_error matrix"
```

---

## Task 6: Docs

**Files:**
- Modify: `README.md` (CLI usage section), `docs/mcp.md` is unaffected.

- [ ] **Step 1: Add a `--variants` usage block to README**

Under the CLI usage section of `README.md`, add:

```markdown
### Variant matrix

Explore the geometric design space without re-running by hand:

\`\`\`bash
vectoreyes logo.png --variants                       # 3×3 epsilon × max_error grid -> ./logo-variants/
vectoreyes logo.png --variants --out-dir ./looks/
vectoreyes logo.png --variants --epsilons 0.5,2,4 --max-errors 0.5,2
\`\`\`

Writes `variant-e<ε>-m<max_error>.svg` per cell, a `manifest.json` (params +
the fitter strategies each variant used), and—if `vectoreyes[scoring]` is
installed—an annotated `contact-sheet.png`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document the --variants matrix in the README"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** report plumbing (Task 1), matrix (Task 2), SVGs+manifest (Task 3), annotated contact sheet + renderer-optional (Task 4), CLI + axis flags + per-cell failure capture (Tasks 2 & 5), docs (Task 6). All spec components mapped.
- **Type consistency:** `IdealizeReport(strategies, gradients, elements)` defined in Task 1 is used identically in the manifest (Task 3) and caption (Task 4); `Variant(epsilon, max_error, svg, report, error)` from Task 2 is consumed unchanged in Tasks 3–5; `_render_body`/`_idealize_rectified` 3-tuple return is consistent across Task 1 steps 4–6.
- **DRY:** the matrix reuses `idealize` (no pipeline re-implementation); the contact sheet reuses `score._rasterize`/`SvgRendererUnavailable`; filenames reuse `fit._fmt`.

# Symmetry-axis overlay (`--axes`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An opt-in `--axes` flag that overlays each detected bilateral mirror axis on the variant contact-sheet tiles.

**Architecture:** Surface the detected mirror axes from `idealize` as line segments in output-frame (viewBox) coordinates (`IdealizeReport.axes`, parallel to `strategies`; tilted marks mapped through `_rectify_affine`). The contact-sheet renderer injects those segments as `<line>` elements into each variant SVG before rasterizing, so resvg scales them with the artwork.

**Tech Stack:** numpy, Pillow, existing vectoreyes pipeline; `resvg-py` (`scoring` extra) for the contact sheet.

**Spec:** `docs/superpowers/specs/2026-06-10-axis-overlay-design.md`

---

## File Structure

- **Modify** `src/vectoreyes/pipeline.py` — `AxisLine` dataclass; `IdealizeReport.axes`; `_render_body` collects per-component frame axes; `idealize`/`_idealize_rectified` thread + map axes; `_report_from_cands` → `_build_report(cands, axes)`.
- **Modify** `src/vectoreyes/variants.py` — `_axis_line_svg`, `_inject_axes`, `_render_tile(draw_axes=)`, `compose_contact_sheet(draw_axes=)`.
- **Modify** `src/vectoreyes/cli.py` — `--axes` flag + stderr note.
- **Test** `tests/test_pipeline_report.py`, `tests/test_variants.py`.

---

## Task 1: Surface mirror axes on `IdealizeReport`

**Files:**
- Modify: `src/vectoreyes/pipeline.py`
- Test: `tests/test_pipeline_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline_report.py` (it already imports numpy, PIL Image/ImageDraw, `IdealizeReport`, `idealize`, and defines `_disc`):

```python
from tests._render import render_svg
from vectoreyes.pipeline import AxisLine, Options


def _arch_svg(deg):
    # rectangle + semicircle dome: exactly one mirror axis (vertical at deg=0)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240">'
            f'<g transform="rotate({deg} 120 120)">'
            f'<rect x="75" y="120" width="90" height="80" fill="#36c"/>'
            f'<path d="M75 120 A45 45 0 0 1 165 120 Z" fill="#36c"/></g></svg>')


def test_report_axes_upright_is_vertical():
    svg, report = idealize(_disc(), report=True)
    assert len(report.axes) == 1
    a = report.axes[0]
    assert isinstance(a, AxisLine)
    assert abs(a.x1 - a.x2) < 0.01          # vertical segment
    assert abs(a.x1 - 32) < 2.0             # disc centre (64/2)
    assert a.y2 > a.y1                       # spans a real vertical extent


def test_report_axes_tilted_is_non_vertical():
    arr = render_svg(_arch_svg(45), 240, 240)
    svg, report = idealize(arr, report=True)
    assert "rotate(" in svg                  # took the rectify path
    assert len(report.axes) >= 1
    assert any(abs(a.x1 - a.x2) > 1.0 for a in report.axes)   # at least one tilted line


def test_report_axes_empty_without_symmetry():
    svg, report = idealize(_disc(), options=Options(no_symmetry=True), report=True)
    assert report.axes == ()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_pipeline_report.py -q`
Expected: FAIL — `ImportError: cannot import name 'AxisLine'`.

- [ ] **Step 3: Add `AxisLine` and extend `IdealizeReport`**

In `src/vectoreyes/pipeline.py`, add `AxisLine` just above `class IdealizeReport`, and add the `axes` field + update `empty()`:

```python
@dataclass(frozen=True)
class AxisLine:
    """A detected mirror axis as a segment in output-frame (viewBox) coords."""
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class IdealizeReport:
    """What the pipeline actually emitted for one idealize() run: the histogram of
    fitter strategies the scorer chose per region, the gradient-fill count, the total
    emitted element count, and the detected mirror axes (one segment per component
    with a vertical mirror, in output-frame coords). Diagnostic annotation."""

    strategies: Mapping[str, int]
    gradients: int
    elements: int
    axes: tuple[AxisLine, ...]

    @staticmethod
    def empty() -> "IdealizeReport":
        return IdealizeReport(types.MappingProxyType({}), 0, 0, ())
```

- [ ] **Step 4: Collect per-component frame axes in `_render_body`**

In `_render_body`, add a `frame_axes` accumulator and record one segment per component that has an axis. Change the accumulator init (just after `cands: list[Candidate] = []`):

```python
    cands: list[Candidate] = []
    frame_axes: list[AxisLine] = []
```

Inside the `for comp in components:` loop, right after `axis = None if opt.no_symmetry else detect_axis(silhouette)`:

```python
        if axis is not None:
            ys = np.nonzero(silhouette)[0]
            frame_axes.append(AxisLine(axis.x, float(ys.min()), axis.x, float(ys.max())))
```

Change the return type to `tuple[list[str], list[str], list[Candidate], list[AxisLine]]` and the final `return body, defs, cands` to:

```python
    return body, defs, cands, frame_axes
```

- [ ] **Step 5: Add `_build_report` and an axis-mapper; map axes in `_idealize_rectified`**

Replace `_report_from_cands` with `_build_report` (it now also takes axes), and add `_map_axis`. (Find the existing `def _report_from_cands(` and replace the whole function.)

```python
def _map_axis(a: AxisLine, affine: Affine) -> AxisLine:
    x1, y1 = apply_affine_point(affine, a.x1, a.y1)
    x2, y2 = apply_affine_point(affine, a.x2, a.y2)
    return AxisLine(x1, y1, x2, y2)


def _build_report(cands: list[Candidate], axes: list[AxisLine]) -> IdealizeReport:
    strategies: dict[str, int] = {}
    gradients = 0
    for c in cands:
        if c.source == "gradient":
            gradients += 1
        if c.strategy is not None:                 # None for occlusion / lens / gradient
            strategies[c.strategy] = strategies.get(c.strategy, 0) + 1
    return IdealizeReport(types.MappingProxyType(dict(strategies)), gradients, len(cands), tuple(axes))
```

Add the import for `apply_affine_point` to the existing `from .emit import ...` line in pipeline.py (verify the current emit import with `rg -n "from .emit import" src/vectoreyes/pipeline.py`; add `apply_affine_point` to it).

Rewrite `_idealize_rectified` to thread frame axes and map them through the rectify affine (compute the affine once, use it for both the flatten bake and the axis mapping):

```python
def _idealize_rectified(arr: np.ndarray, opt: Options, rho: float, w0: int, h0: int) -> tuple[str | None, list[Candidate], list[AxisLine]]:
    """...(keep existing docstring)..."""
    rot = ndi.rotate(arr.astype(float), -rho, reshape=True, order=1, cval=255.0)
    rot = np.clip(rot, 0.0, 255.0).astype(np.uint8)
    rw, rh, regions = _segment_image(rot, opt)
    if not regions:
        return None, [], []
    if detect_axis(np.any([r.mask for r in regions], axis=0)) is None:
        return None, [], []
    affine = _rectify_affine(rho, w0, h0, rw, rh)
    if opt.flatten:
        body, defs, cands, frame_axes = _render_body(rw, rh, regions, opt, bake=affine, rgb=rot)
        doc = render_svg_doc(w0, h0, body, defs)
    else:
        body, defs, cands, frame_axes = _render_body(rw, rh, regions, opt, rgb=rot)
        wrap = (f'<g transform="translate({_fmt(w0 / 2)} {_fmt(h0 / 2)}) '
                f'rotate({_fmt(round(-rho, 3))}) translate({_fmt(-rw / 2)} {_fmt(-rh / 2)})">')
        doc = render_svg_doc(w0, h0, [wrap, *body, "</g>"], defs)
    axes = [_map_axis(a, affine) for a in frame_axes]
    return doc, cands, axes
```

- [ ] **Step 6: Thread axes through `idealize`**

Rewrite the body of `idealize` to carry `axes` alongside `cands` (the upright frame IS the output frame, so its axes pass through unmapped):

```python
    w, h, regions = _segment_image(arr, opt)
    if not regions:
        svg, cands, axes = render_svg_doc(w, h, []), [], []
    else:
        svg, cands, axes = None, [], []
        # Any-axis symmetry: rectify a tilted mirror upright, idealize there, wrap back.
        if not opt.no_symmetry:
            silhouette = np.any([r.mask for r in regions], axis=0)
            if detect_axis(silhouette) is None:
                rho = detect_symmetry_rotation(silhouette)
                if rho is not None:
                    rectified, rcands, raxes = _idealize_rectified(arr, opt, rho, w0, h0)
                    if rectified is not None:
                        svg, cands, axes = rectified, rcands, raxes
        if svg is None:
            body, defs, cands, axes = _render_body(w, h, regions, opt, rgb=arr)
            svg = render_svg_doc(w, h, body, defs)

    return (svg, _build_report(cands, axes)) if report else svg
```

- [ ] **Step 7: Run the report tests + full suite**

Run: `uv run --extra dev pytest tests/test_pipeline_report.py -q && uv run --extra dev pytest -p no:warnings 2>&1 | tail -1`
Expected: report-axes tests PASS; full suite PASS (report=False default still returns bare str, so existing callers and the variant tests are unaffected). Paste the exact final summary line.

If `test_report_axes_tilted_is_non_vertical` does not see `"rotate("` in the svg (rectify path not taken for the 45° arch), the fixture is wrong, not the code — but the 45° arch is the proven tilt fixture from `tests/test_symmetry_rotation.py`, so it should rectify.

- [ ] **Step 8: Commit**

```bash
git add src/vectoreyes/pipeline.py tests/test_pipeline_report.py
git commit -m "feat(pipeline): surface detected mirror axes on IdealizeReport

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Inject axes into the contact-sheet tiles

**Files:**
- Modify: `src/vectoreyes/variants.py`
- Test: `tests/test_variants.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_variants.py` (top-of-file imports already include `io`, `pytest`, `Image`, `import vectoreyes.variants as V`, `generate_variants`, `_mark`; add `from vectoreyes.variants import _axis_line_svg, _inject_axes` to the top import block and `from vectoreyes.pipeline import AxisLine` to the top):

```python
def test_axis_line_svg_emits_line():
    frag = _axis_line_svg(AxisLine(10.0, 0.0, 10.0, 40.0), 1.5)
    assert frag.startswith("<line ") and 'x1="10' in frag and "stroke" in frag


def test_inject_axes_inserts_before_closing_svg():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="40" height="40"><circle/></svg>'
    out = _inject_axes(svg, (AxisLine(20.0, 0.0, 20.0, 40.0),))
    assert "<line " in out
    assert out.rstrip().endswith("</svg>")
    assert out.index("<line ") < out.index("</svg>")


@pytest.mark.skipif(not _renderer_available(), reason="needs resvg-py")
def test_contact_sheet_with_axes_renders():
    eps, mes = (0.5, 3.0), (1.0,)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    png = V.compose_contact_sheet(variants, epsilons=eps, max_errors=mes, draw_axes=True)
    assert isinstance(png, (bytes, bytearray)) and len(png) > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_variants.py -k "axis or with_axes" -q`
Expected: FAIL — `cannot import name '_axis_line_svg'` / `compose_contact_sheet() got an unexpected keyword 'draw_axes'`.

- [ ] **Step 3: Add the injection helpers and `draw_axes` threading**

In `src/vectoreyes/variants.py`, add `import re` to the top imports and `from .pipeline import AxisLine` to the existing `from .pipeline import ...` line (verify with `rg -n "from .pipeline import" src/vectoreyes/variants.py`). Add the helpers (place them just above `_render_tile`):

```python
_AXIS_STROKE = "#e000a0"   # magenta — distinct from typical logo palettes


def _axis_line_svg(a: AxisLine, stroke_w: float) -> str:
    dash = max(6.0, stroke_w * 4.0)
    return (f'<line x1="{a.x1:.1f}" y1="{a.y1:.1f}" x2="{a.x2:.1f}" y2="{a.y2:.1f}" '
            f'stroke="{_AXIS_STROKE}" stroke-width="{stroke_w:.2f}" '
            f'stroke-dasharray="{dash:.1f},{dash * 0.7:.1f}" stroke-opacity="0.85"/>')


def _inject_axes(svg: str, axes: tuple[AxisLine, ...]) -> str:
    """Insert one dashed <line> per axis (already in viewBox coords) just before the
    closing </svg>, scaling stroke width to the viewBox so it renders at any size."""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    extent = max(float(m.group(1)), float(m.group(2))) if m else 100.0
    stroke_w = max(1.0, extent / 250.0)
    fragment = "".join(_axis_line_svg(a, stroke_w) for a in axes)
    return svg.replace("</svg>", fragment + "</svg>")
```

Change `_render_tile` to accept `draw_axes` and inject when asked:

```python
def _render_tile(v: Variant, *, draw_axes: bool = False) -> Image.Image:
    """One variant rendered into a _TILE×_TILE white tile, or a placeholder for a
    failed cell. Raises SvgRendererUnavailable if the renderer is missing."""
    tile = Image.new("RGB", (_TILE, _TILE), "white")
    if v.error is None and v.svg:
        svg = _inject_axes(v.svg, v.report.axes) if draw_axes and v.report.axes else v.svg
        arr = _rasterize(svg, _TILE, _TILE)          # may raise SvgRendererUnavailable
        tile = Image.fromarray(arr)
    else:
        ImageDraw.Draw(tile).text((8, _TILE // 2), "failed", fill=(180, 40, 40))
    return tile
```

Add `draw_axes` to `compose_contact_sheet` and pass it through. Change its signature line and the `_render_tile` call:

```python
def compose_contact_sheet(variants: list[Variant], *, epsilons, max_errors, draw_axes: bool = False) -> bytes | None:
```

and inside the tile loop change `tile = _render_tile(v)` to `tile = _render_tile(v, draw_axes=draw_axes)`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_variants.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/variants.py tests/test_variants.py
git commit -m "feat(variants): inject symmetry-axis <line>s into contact-sheet tiles

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: CLI `--axes` flag + README

**Files:**
- Modify: `src/vectoreyes/cli.py`, `README.md`
- Test: `tests/test_variants.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_variants.py`:

```python
def test_cli_variants_axes_writes_contact_sheet(tmp_path):
    src = tmp_path / "mark.png"
    Image.fromarray(_mark()).save(src)
    out = tmp_path / "looks"
    rc = cli_main([str(src), "--variants", "--axes", "--out-dir", str(out),
                   "--epsilons", "0.5,3", "--max-errors", "1"])
    assert rc == 0
    assert (out / "manifest.json").exists()
    # contact sheet only exists when the renderer is installed; the run must still succeed
    if _renderer_available():
        assert (out / "contact-sheet.png").exists()


def test_cli_axes_without_variants_is_noted(tmp_path, capsys):
    src = tmp_path / "mark.png"
    Image.fromarray(_mark()).save(src)
    out = tmp_path / "out.svg"
    rc = cli_main([str(src), "--axes", "-o", str(out)])
    assert rc == 0 and out.exists()
    assert "axes" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/test_variants.py -k "axes" -q`
Expected: FAIL — unrecognized argument `--axes`.

- [ ] **Step 3: Add `--axes` to `cli.py`**

Add the argument (next to `--variants`):

```python
    ap.add_argument("--axes", action="store_true",
                    help="overlay detected symmetry axes on the contact-sheet tiles (with --variants)")
```

In the `if args.variants:` block, pass `draw_axes` to the composer — change the `compose_contact_sheet(...)` call to:

```python
        png = compose_contact_sheet(variants, epsilons=epsilons, max_errors=max_errors, draw_axes=args.axes)
```

Immediately after `args = ap.parse_args(argv)` (before building `base`), add the stderr note for the ignored-flag case:

```python
    if args.axes and not args.variants:
        print("note: --axes only applies in --variants mode; ignoring it", file=sys.stderr)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/test_variants.py -q`
Expected: PASS.

- [ ] **Step 5: Add a README line**

In `README.md`, in the `### Variant matrix` section, after the code block, add:

```markdown
Add `--axes` to draw each detected symmetry axis over the contact-sheet tiles —
a quick visual check of what the symmetry detector found.
```

- [ ] **Step 6: Run the full suite**

Run: `uv run --extra dev pytest -p no:warnings 2>&1 | tail -1`
Expected: all pass. Paste the exact final summary line.

- [ ] **Step 7: Commit**

```bash
git add src/vectoreyes/cli.py README.md tests/test_variants.py
git commit -m "feat(cli): --axes overlays symmetry axes on the variant contact sheet

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** `AxisLine` + `IdealizeReport.axes` (Task 1); per-component collection + upright/rectified mapping via `_rectify_affine` (Task 1, steps 4–6); `<line>` injection in `_render_tile` (Task 2); `compose_contact_sheet(draw_axes=)` + CLI `--axes` + ignored-flag note (Tasks 2–3); tests for upright/tilted/no-symmetry/injection/CLI (all tasks). The per-component count is exercised implicitly by the loop; the upright/tilted single-axis tests pin the core behavior.
- **Type consistency:** `AxisLine(x1,y1,x2,y2)` from Task 1 is consumed unchanged in Task 2 (`_axis_line_svg`, `_inject_axes`); `_render_body`'s new 4-tuple return is updated at all three call sites in Task 1 (idealize + both `_idealize_rectified` branches); `_build_report(cands, axes)` replaces `_report_from_cands(cands)` at the single call site in `idealize`.
- **DRY:** axis mapping reuses `apply_affine_point` + the existing `_rectify_affine`; injection reuses the variant SVG (no second render path); no manual letterbox math.

# Symmetry-axis overlay (`--axes`) — design

**Goal:** An opt-in `--axes` flag that draws each detected bilateral mirror axis
over the variant contact-sheet tiles, so the symmetry the pipeline found is
visible at a glance (the diagnostic that would have made the carrot-tip fork
obvious).

**Architecture:** Surface the detected mirror axes from `idealize` as line
segments in **output-frame (viewBox) coordinates** (a new `IdealizeReport.axes`
field, parallel to `strategies`). The contact-sheet renderer injects those
segments as `<line>` elements into each variant SVG just before rasterizing, so
resvg scales them with the artwork — no pixel-space letterbox math.

**Tech stack:** existing numpy/Pillow pipeline; `resvg-py` (optional `scoring`
extra) for the contact sheet.

---

## Scope

In scope: per-component mirror-axis segments surfaced on `IdealizeReport`; SVG
`<line>` injection in the contact-sheet tiles; a `--axes` CLI flag.

Out of scope (explicitly):
- **Axes in `manifest.json`** — the manifest's top-level `axes` key already means
  the grid (epsilon/max_error) axes; adding symmetry axes there invites
  confusion. Symmetry axes are a contact-sheet visual only here.
- **A guide layer in the idealized SVG output** — the `AxisLine` representation
  makes this nearly free later, but it is not built now.
- **Single-shot (`non-variants`) overlay** — `--axes` is meaningful only with
  `--variants`.
- The **PCA / rotation** construction axis — only the accepted *mirror* axis is
  drawn (that is "the symmetry line").

## Components

### 1. `AxisLine` + `IdealizeReport.axes` (pipeline.py)

```python
@dataclass(frozen=True)
class AxisLine:
    """A detected mirror axis as a segment in output-frame (viewBox) coords."""
    x1: float
    y1: float
    x2: float
    y2: float
```

Extend `IdealizeReport` with `axes: tuple[AxisLine, ...]` (default not needed —
always constructed). Update `IdealizeReport.empty()` to pass `()`.

### 2. Per-component axis collection (pipeline.py `_render_body`)

`_render_body` already computes `axis = detect_axis(silhouette)` per component
(pipeline.py:247). For each component whose `axis is not None`, record a **vertical
segment in the current frame**: `x = axis.x`, spanning that component's silhouette
vertical extent `[ymin, ymax]` (from `np.nonzero(silhouette)`), i.e.
`AxisLine(axis.x, ymin, axis.x, ymax)`. Collect these into `frame_axes` and change
the return to `(body, defs, cands, frame_axes)`.

(`frame_axes` are in `_render_body`'s frame — the output frame for upright marks,
the rectified frame for tilted marks. The caller maps them to output coords.)

### 3. Frame → output mapping (pipeline.py `idealize` / `_idealize_rectified`)

- **Upright path** (`idealize`): `_render_body`'s frame IS the output frame, so
  `axes = frame_axes` unchanged.
- **Rectified path** (`_idealize_rectified`): map each `frame_axes` endpoint
  through `_rectify_affine(rho, w0, h0, rw, rh)` using `apply_affine_point`
  (emit.py:111). The vertical rectified segment becomes a correctly *tilted*
  segment in output coords. Return the mapped axes alongside `cands`.
- **No-regions / `no_symmetry` / no-axis component**: contributes no `AxisLine`.

The report assembly (`_report_from_cands`) gains the axes: rename/extend to
`_build_report(cands, axes)` returning `IdealizeReport(strategies, gradients,
elements, tuple(axes))`. `report=False` path is unaffected (still returns bare
`str`).

### 4. SVG `<line>` injection (variants.py `_render_tile`)

`_render_tile(v, *, draw_axes=False)`:
- When `draw_axes` and `v.report.axes`, build an SVG fragment of one `<line>` per
  axis, in viewBox coords, and insert it immediately before the closing `</svg>`
  of `v.svg`, then rasterize the augmented SVG.
- Line style: `stroke="#e000a0"` (magenta, distinct from logo palettes),
  `stroke-width` scaled to the viewBox (e.g. `max(w0,h0)/250`, min 1.0),
  `stroke-dasharray` proportional (e.g. `8,6` scaled), `stroke-opacity="0.85"`.
  Helper `_axis_line_svg(axis, stroke_w) -> str`.
- Insertion via `svg.replace("</svg>", fragment + "</svg>")` (the idealized SVG
  ends with a single `</svg>`).
- A failed/empty cell still draws the red "failed" placeholder (no axes).

### 5. `compose_contact_sheet` + CLI

- `compose_contact_sheet(variants, *, epsilons, max_errors, draw_axes=False)` —
  threads `draw_axes` to `_render_tile`. No other change.
- CLI `cli.py`: add `--axes` (store_true, help "overlay detected symmetry axes on
  the contact-sheet tiles (with --variants)"). Pass `draw_axes=args.axes` to
  `compose_contact_sheet`. If `args.axes and not args.variants`, print a stderr
  note that `--axes` only applies in `--variants` mode (it is otherwise ignored).

## Data flow

```
idealize(report=True)
  └─ _render_body → per-component (axis.x, ymin, ymax) as frame AxisLines
        ├─ upright:   axes = frame_axes
        └─ rectified: axes = [map endpoints via _rectify_affine]
  └─ IdealizeReport(strategies, gradients, elements, axes)
        └─ Variant.report.axes
              └─ _render_tile(draw_axes): inject <line>…</line> into v.svg → rasterize
```

## Error handling

- A component with no detected axis simply contributes no line (common).
- `apply_affine_point` on endpoints cannot fail for finite inputs; `rho`/affine
  are already validated upstream.
- If `v.svg` somehow lacks `</svg>` (only for a failed/empty cell, where
  `draw_axes` is skipped anyway), no injection occurs.

## Testing

- **Upright axis**: a vertically-symmetric mark → `idealize(report=True).axes`
  has one `AxisLine` with `x1 == x2` (vertical) at the mirror x (±1px).
- **Tilted axis**: a mark with only a tilted mirror (forces the rectify path) →
  one `AxisLine` with `x1 != x2` (the mapped segment is not vertical).
- **Per-component**: a two-gutter-separated-component mark, each symmetric →
  two `AxisLine`s.
- **No symmetry**: `Options(no_symmetry=True)` → `axes == ()`.
- **Injection**: `_render_tile(v, draw_axes=True)` for an axis-bearing variant
  produces a raster (renderer present); and the augmented SVG string contains
  `"<line "` (unit-test the fragment builder `_axis_line_svg` directly too).
- **Contact sheet**: `compose_contact_sheet(..., draw_axes=True)` returns valid
  PNG bytes; `draw_axes=False` path unchanged.
- **CLI**: `--variants --axes` writes a contact sheet; `--axes` without
  `--variants` prints the stderr note and does not error.

## Files

- Modify: `src/vectoreyes/pipeline.py` (`AxisLine`, `IdealizeReport.axes`,
  `_render_body` frame-axes, `idealize`/`_idealize_rectified` mapping,
  `_build_report`)
- Modify: `src/vectoreyes/variants.py` (`_render_tile` injection,
  `_axis_line_svg`, `compose_contact_sheet` `draw_axes`)
- Modify: `src/vectoreyes/cli.py` (`--axes`)
- Test: `tests/test_pipeline_report.py` (axes cases), `tests/test_variants.py`
  (injection + contact sheet + CLI)

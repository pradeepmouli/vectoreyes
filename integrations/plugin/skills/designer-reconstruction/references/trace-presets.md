# Trace starting-point presets

Select `vectoreyes_trace_drawing.options.preset`: `black-and-white`, `flat-logo`,
`gradient-artwork`, or `conservative`, in the order below. The server applies
the corresponding bundle before defaults; explicitly supplied options override
the bundle. The UI selector updates its controls, and responses record all
effective options. The JSON bundles below remain useful as explicit overrides.
Inspect source, v0, selected pass revisions,
region map, and difference before accepting a result. Input must be PNG.

## Black-and-white artwork / cutting-file baseline

```json
{
  "refine": "none",
  "max_sampling_colors": 2,
  "max_output_colors": 2,
  "allow_gradients": false,
  "background_cutouts": "subtract",
  "remove_background": "auto",
  "trace_level": "subpixel",
  "fit_strategy": "progressive_allow_lines",
  "min_region_size": 1,
  "min_region_fraction": 0,
  "max_hole_area": 0
}
```

Use only for genuinely two-color artwork. Subtract export makes the resolved
background color transparent; it does not retain white ink. If white is intended
paint, use `remove_background="off"` and `background_cutouts="off"` instead.
Set `background_color` explicitly only when the source's intended background is
known. One visible ink plus transparency is still a useful two-color source.

Zero relative filtering and hole cleanup preserve small details, but also
preserve specks. Inspect isolated shapes and thin bridges at the intended cutting
scale. This is a geometry baseline, not certification of fabrication readiness.
Refine the retained v0 with a plan, or use `refine="auto"` to inspect the pass-revision chain.
Do not set fitting tolerances to zero as a detail-preservation shortcut: that can
chase every raster irregularity and greatly increase command count and cost.

`corner_rounding=true` (default) protects detected rounded-corner intervals in
v0. Set it false to retain detected point boundaries rather than those intervals
when diagnosing over-rounding. Corner detection still runs and source curvature
is still fitted; use explicit corner declarations to request sharp geometry.
`corner_normalize` is a separate opt-in refinement pass, not trace rounding.
It only normalizes existing retained corners and never detects new ones.
Rounding classification uses source-contour inset and edge-fit confidence, not
fillet radius alone: shallow turns can amplify pixel noise into a large radius.
Geometric support thresholds are canvas-relative (1254-pixel reference), while
anti-alias allowances and fitting tolerances remain in pixels. Increasing image
size no longer lowers the relative minimum protected straight-edge length.

## Flat-color geometric logo

```json
{
  "refine": "auto",
  "max_sampling_colors": "auto",
  "max_output_colors": null,
  "allow_gradients": false,
  "background_cutouts": "subtract",
  "trace_level": "subpixel",
  "fit_strategy": "progressive_allow_lines"
}
```

When the source palette is known, replace sampling auto with that count; set an
output count separately if exact palette reduction is desired. Unlimited output
does not mean gradients are enabled. Check primitive, clone, and symmetry
diagnostics and inspect each selected pass revision against its parent; disable only a demonstrably harmful auto
pass using `refinement_passes`, or continue with explicit declarations from v0.
Default region filtering can discard small lettering: lower it if the source
contains important small features. Do not silently use cutting settings for a
full-color illustration.

## Gradient logo / illustration

```json
{
  "refine": "auto",
  "max_sampling_colors": "auto",
  "max_output_colors": null,
  "allow_gradients": true,
  "background_cutouts": "subtract",
  "trace_level": "subpixel",
  "fit_strategy": "progressive",
  "gradient_pretrace": true
}
```

The gradient pre-pass traces a simplified-color proxy while original pixels
remain authoritative for final paint. Compare with `gradient_pretrace=false`
if genuine interior divisions disappear. A sampling palette limit does not
limit final paint. A finite output palette caps gradient stops, not the shades
between them; use `allow_gradients=false` for strictly finite-color output.
Gradient or raster fills are not background cutouts merely because some sampled
pixels resemble the background. Inspect fill and geometry independently.

## Conservative tracing / regression diagnosis

```json
{
  "refine": "none",
  "max_sampling_colors": "auto",
  "trace_level": "subpixel",
  "fit_strategy": "quadratic",
  "gradient_pretrace": false
}
```

This removes automatic optimization, not tracing or the default initial stitch.
Use `initial_stitch=false` only to isolate shared-boundary changes. Compare with
progressive fitting using the same palette and preprocessing; do not change many
controls at once. Prefer reported source correspondence and explicit plans over
editing exported path strings.

## Transport failures are not geometry failures

The deployed Streamable HTTP endpoint returns tool POST results as JSON rather
than SSE, avoiding the Python SDK's 1 MiB SSE-event cap while retaining complete
panels and diagnostics. Other hosts/proxies can have independent response limits.
Do not lower palette/detail settings merely to fit a transport response.
Record the actual error and use [the direct fallback](direct-fallback.md) when
local execution is available. `DRAWING_NOT_FOUND` instead indicates expired or
unavailable state; a reconnect alone does not invalidate a retained drawing.

Drawing responses use `structuredContent` as the authoritative data; text is a
short receipt. If `delivery.reports_deferred` is true, fetch `artifact="report"`
for the version under review before constructing a plan. An artifact response
with `chunk` is not a complete export: concatenate base64-decoded `chunk.data`
bytes in offset order, requesting `next_offset` with identical arguments until
null; verify `total_bytes` and `sha256`, then parse the UTF-8 JSON as the complete
MCP result. The evaluation script and widget handle this automatically. Do not
reduce artwork detail or report success for an incomplete download.

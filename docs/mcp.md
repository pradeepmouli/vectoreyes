# VectorMark MCP integration

The server exposes five drawing tools with generated input/output schemas.
The retired `idealize_logo`, `idealize_logo_data`, and `render_idealized_logo`
names are not registered MCP tools.

## Installation and transports

Use a dedicated environment so client startup does not alter the developer's
environment through an implicit `uv run`. From the checkout:

```bash
uv venv /tmp/vectormark-mcp-env --python 3.13
uv pip install --python /tmp/vectormark-mcp-env/bin/python -e '.[server,scoring]'
npm --prefix integrations/mcp-app ci
npm --prefix integrations/mcp-app run build
/tmp/vectormark-mcp-env/bin/vectormark-mcp
```

Configure the MCP client to launch that environment's executable. Stdio is the
default. For HTTP:

```bash
VECTORMARK_MCP_TRANSPORT=streamable-http \
VECTORMARK_MCP_HOST=127.0.0.1 VECTORMARK_MCP_PORT=8000 \
  /tmp/vectormark-mcp-env/bin/vectormark-mcp
```

The endpoint is `/mcp`. Require HTTPS and authentication before external access;
a tunnel alone does not provide authentication.

FastMCP 4.1 does not expose the SDK's request-body-size option. The HTTP
startup adapter configures both the SDK manager and its body-limit middleware
to retain VectorMark's base64 image allowance. It fails at startup if those
integration points change, rather than silently reverting to a 4 MiB limit.

## Workflow

### Packaged reconstruction skill

The server uses FastMCP 4.1 and exposes the packaged
`skill://designer-reconstruction/SKILL.md` resource, its `_manifest`, and
supporting example plans. Clients can discover these using `resources/list`
and retrieve them with `resources/read`. Server instructions point to the
skill; hosts do not necessarily load skills automatically. Only VectorMark's
packaged guidance is exposed, never the operator's personal skills directory.
The Pydantic action and output models remain authoritative for tool schemas.

1. **`trace_drawing(image, options)`** creates a live drawing. With
   `refine="auto"`, `outputs` contains pre-refinement `v0` and optimized `v1`.
   With `refine="none"` (the default), it contains only `v0`. Both modes support
   subsequent refinement. Each output is an immutable canonical `Drawing`;
   `v1` is replayed from retained source evidence and concrete accepted actions,
   not from `v0` geometry or a second legacy scene.
2. Inspect the returned lineage review panel: source, selected retained
   baseline, actual parent (or an explicit no-parent state), current SVG,
   addressable region map, and amplified source/SVG difference. Evaluate
   geometry and paint separately. Auto refinement is a proposal, not proof that
   the reconstruction is complete.
3. **`refine_drawing(plan)`** applies typed actions. The plan contains
   `drawing_id`, `base_version`, and `ops`. Use region and command IDs from the
   selected version. Branch from either `v0` or `v1`, then use returned IDs.
   The submitted plan is retained separately from the effective recipe: the
   recipe contains the exact ordered, source-bound actions used to reconstruct
   the child from evidence. The selected version's report is also the capability
   surface for command-addressed corner edits: submit only paths whose
   `corner_edit.supported` value is true and only command boundaries marked
   `corner_boundary.editable`. Missing exact source/output correspondence is an
   explicit unsupported state, never an invitation to guess an endpoint.
4. **`retrace_drawing(drawing_id, base_version, geometry_guide, options, label)`**
   offers an alternative: a transparent, flat-color geometry guide whose pixel
   dimensions exactly match the retained trace canvas. It automatically refines
   guide geometry and fits final paint from the original retained raster, not
   the guide's label colors. A call from `v1` atomically retains `v1.0` as the
   guide baseline and `v1.0.0` as its optimized child. The response reports
   `requested_base_version=v1`, `parent_version=v1.0`,
   `baseline_version=v1.0`, and `version=v1.0.0`; `outputs` contains both new
versions with their actual parent links. Its guide-inclusive lineage review
   keeps the submitted guide visually separate from the requested base, newly
   retained guide baseline/actual parent, and current original-filled result.
5. **`get_drawing_artifact(drawing_id, version, artifact)`** retrieves `svg`,
   `preview`/`preview_png`, `review_panel`, `labeled_svg`, `raw_trace`, `plan`,
   or `versions`. Returned `drawing://...` references are retained artifact
   identifiers, not downloadable HTTP URLs. `versions` is metadata-only, and
   `plan` distinguishes submitted input from the effective replay recipe.
   Artifact reads render/project retained values; they do not rerun fitting,
   detection, optimization, or corner discovery.
6. **`render_drawing(result)`** displays a result containing `svg` in the MCP
   Apps widget. Use an SVG artifact response, not just a drawing ID. Rendering
   does not create a version.

Choose the best branch, which need not be the last one.

## Image references and options

Only PNG input is supported, including geometry guides. The server validates
the encoded image format regardless of filename or declared MIME type.
Export other formats as PNG before submitting; this does not undo existing
JPEG compression artifacts.

Provide exactly one input route, never a bare path string:

- `image`: a host file object with required string `download_url` and `file_id`,
  and optional string `mime_type` and `file_name`.
- `image_data`: a direct object containing exactly one of HTTPS `url`,
  `data_uri`, bare `base64`, or server-local `path`. This is not a file param.

Retracing uses the same contracts under `geometry_guide` and
`geometry_guide_data`. Local paths are allowed only in trusted stdio, not HTTP.
An agent-local file is not necessarily server-local. Both routes share image
validation, preprocessing, and tracing; they are not separate pipelines.

Illustrative trace arguments (replace the URL):

```json
{
  "image_data": {"url": "https://example.com/artwork.png"},
  "options": {
    "refine": "auto",
    "max_sampling_colors": "auto",
    "trace_level": "subpixel",
    "fit_strategy": "progressive"
  }
}
```

Use the live schema for the full settings contract. Key choices:

The packaged designer-reconstruction skill includes [use-case presets](../src/vectormark/assets/skills/designer-reconstruction/references/trace-presets.md)
for monochrome cutting artwork, flat logos, gradient artwork, and tracing
diagnostics. These are suggested options, not a new `preset` API field.

- `max_sampling_colors`: tracing/segmentation palette size or `"auto"`; default 16. This replaces `max_colors` without an alias.
- `max_output_colors`: final source-derived palette limit, default `null` (unlimited). With gradients enabled, it constrains stop colors, not interpolated shades.
- `allow_gradients`: defaults to true. Set false for flat fills only; combine with `max_output_colors=2` for monochrome artwork.
- `background_cutouts`: `"subtract"` (default), `"mask"`, or `"off"`; affects final SVG export only.
- `trace_level`: `"pixel"` (default) or `"subpixel"`.
- `fit_strategy`: `"quadratic"` (default), `"progressive"` (protected straight
  sides followed by Q/C fitting), or `"progressive_allow_lines"` (also permits
  lines during recursion).
- `simplify_tolerance` / `curve_tolerance`: contour and curve fitting tolerances.
  Progressive cubic and dual-quadratic shortcuts must also meet the requested
  source-sample curve tolerance; long sides do not receive a looser relative bound.
- `min_region_size` / `min_region_fraction` / `max_hole_area`: retention and
  small-hole cleanup controls; see the schema for units and limits.
- `gradient_pretrace`: optional simplified material proxy; original pixels
  remain available for final fill fitting.
- `corner_normalize`: optional normalization during auto refinement, not v0.
- `corner_rounding`: true by default; protects detected rounded-corner spans
  during trace/root fitting. False retains detected point boundaries instead.
  Detection remains enabled, and source curvature is still fitted; this is not
  a request to invent sharp geometry. Use corner plan declarations for that.
  Local corner evidence takes precedence over coarse recovery estimates.
  Sharp/rounded classification compares source-contour inset with the pixel AA
  allowance plus conditioned straight-edge fit uncertainty, before calculating
  an angle-corrected radius. Radius alone is not evidence of visible rounding.
- Geometric support lengths scale with the canvas's longer side against a
  1254-pixel reference: the 64-pixel protected-line threshold is 52.3 pixels
  at 1024. Pixel anti-alias allowances, local linelet-noise floors, and fitting error
  tolerances do not scale. Every region uses the whole canvas, not its own bounds.
- Corner tangent support grows independently on each side, bounded by the
  configured maximum extent and tangent stability. It stops before a distinct
  neighboring bend; opposing turns are not deduplicated as one AA transition.
- `support_weighted_corners`: optional shallow-corner support evidence.
- `remove_background`: `"auto"`, `"off"`, or `"on"`.

Geometry guides have a separate options schema. Do not assume every trace option
is accepted by `retrace_drawing`.

## Plans and detected axes

The [drawing contract](superpowers/specs/2026-07-10-agent-guided-drawing-mcp-design.md)
defines actions including `update_geometry` and `assign_clone`. Path operations
are nested in `update_geometry`. `fit.type` selects one fitted command;
`fit.strategy` selects recursive fitting of a command-bounded span.

Copy an accepted self/intrinsic `report.targets[].diagnostics.symmetry.axis`
directly into `set_symmetry`. Both use a point on the reflection line and its angle in degrees
from positive x (90 is vertical), with no client-side conversion:

```json
{
  "version": "vectormark.plan.v1",
  "drawing_id": "drw_example",
  "base_version": "v1",
  "ops": [{
    "op": "set_symmetry",
    "target": "r1",
    "axis": {"point": [40, 40], "angle_degrees": 90}
  }]
}
```

IDs and coordinates above are illustrative; the target must exist in the chosen
retained version.

Path and command addresses are version-bound. Before submitting corner edits,
inspect `report.targets[].geometry.corner_edit.supported`; when false, the
selected path has no exact retained source/output correspondence and corner
editing will fail closed. Editable commands are identified by
`geometry.commands[].corner_boundary.editable`. Add, replace, and remove corner
edits replay only retained pre-fit declarations; they never redetect a removed
corner from current geometry.

`mode="pair"` uses the same axis format but describes a relationship between
two regions, not self-symmetry of either region; do not blindly apply that axis
to a single region with `set_symmetry`.

## Lifetime and security

Drawings have a 30-minute idle lifetime in the server process. Drawing IDs are
not bound to transient transport sessions: reconnecting to the same live process
does not itself lose state. Process restart or idle expiry does. Versions, plans,
guides, and artifacts share that lifetime. Export required SVGs before expiry.
`[DRAWING_NOT_FOUND]` means the drawing/version is unavailable.

Treat drawing IDs as bearer capabilities, not user authorization. Protect a
shared server with an appropriate authentication/isolation boundary. URL
validation and HTTP path rejection do not replace access control.

## MCP Apps UI

### File delivery and plugin refresh

SVG artifact retrieval is UI-free and returns exact SVG text plus a standard
MCP `resource_link`. `resources/read` resolves its `drawing://.../vN.svg` URI
through the same canonical renderer. This URI is not an HTTP download URL.
Whether a host presents MCP resources as file attachments is host-dependent.
Clients with filesystem tools can write `structuredContent.svg` verbatim.

The widget's export action feature-detects ChatGPT `uploadFile` and
`getFileDownloadUrl`, uploads the exact selected artifact, and displays a
temporary HTTPS download link. It also supplies the exported file ID and URL
to model context. Missing host APIs produce an explicit error, not a claimed
successful iframe download. Ordinary non-ChatGPT hosts retain browser export.
Real ChatGPT/mobile download behavior must be verified in that host.

Packaged skills are imported snapshots: after updating them, run **Scan Tools**,
review the imported files, and submit a new plugin version. Restarting the MCP
server alone does not refresh the imported skill. Refresh scanned tool metadata
as well when debugging stale UI associations.

### Region and segment selection maps

`artifact: "labeled_svg"` colors whole regions, labels larger regions in place,
and lists every region ID in a side legend. `artifact: "path_map"` colors
individual drawable L/Q/C/A segments and lists their exact command IDs and
types in a side legend. The path map also includes a separate anchor legend:
point corners have diamond markers; rounded corners highlight their owned
segments and show circular entry/exit markers. Retained inflection, extremum,
and junction anchors are marked at their referenced command endpoints.
One legend entry represents one anchor, regardless of its command count.
Anchor elements carry `data-anchor-id` (path-qualified map identity),
`data-source-anchor-id` (exact retained anchor ID), `data-anchor-kind`, and
`data-command-ids`; markers identify their role with `data-anchor-role`.
No new anchors are detected while rendering. Neither map uses leader lines. Colors are deterministic
visual aids; use the explicit IDs, not hue guesses, for plan targets. Segment
SVG elements also carry `data-command-id` attributes.

Filter `path_map` with `region_ids` and `command_types` when inspecting a busy
drawing. Anchors remain visible when at least one referenced command matches
the command filter. Move commands have no drawn segment; use `command_map` for those.
These maps reuse retained geometry and never retrace or modify the drawing.

Corner plans address only the fitted commands and displayed features of the
selected `drawing_id` / `base_version`: for example `r4.p0.c5` and
`r4.p0.corner2`. Use the same fitted command ID for entry and exit to declare a
point corner; distinct IDs declare a rounded span. The report's
`corner_boundary.editable` flag indicates exact retained source correspondence.
The path's move command can address the starting node when an explicit closing
command has its exact receipt. Source provenance hashes and original-polyline
command IDs are internal, not public plan targets. `raw_trace` remains diagnostic
evidence; never copy its source IDs into a plan. The engine resolves fitted
addresses to the original polyline for a fresh fit, rejecting absent or ambiguous
correspondence rather than guessing. Refresh displayed IDs after each refit.

Request `get_drawing_artifact` with `artifact: "command_map"` to overlay
version-scoped command IDs at their endpoints on the retained vector drawing.
Optional `region_ids` selects leaf regions; `command_types` selects `move`,
`line`, `quadratic`, `cubic`, or `arc` commands. Omitted filters include all.
The response contains an SVG and, when rendering is available, a PNG view.
Use the labels to select commands for a refinement plan against that same
version. Primitive and clone geometries have no local path commands to label.
The map does not refit or modify geometry. Corner-boundary editability in the
drawing report is based on exact retained source endpoint correspondence, not
nearest-point guesses or matching source/output command counts.

UI-capable hosts discover `ui://vectormark/logo-widget.html` from tool metadata.
The React source is in `integrations/mcp-app/src`. Installed wheels serve the
built HTML from `vectormark/assets/mcp-app.html` using package resources; they
do not need a checkout or Node.js. Editable installs instead read
`integrations/mcp-app/dist/mcp-app.html`. Build that file before running from a
checkout: a missing editable build produces a placeholder, not the review UI.
The release workflow builds the frontend, includes it in both distributions,
and verifies the installed wheel's resource outside the checkout.

Text-only clients can inspect structured output and retrieve artifacts without
the widget. Rendering a supplied SVG does not certify it as safe for embedding
in other HTML contexts.

### Selectable automatic passes and background cutouts

`trace_drawing.options.refinement_passes` (also available on geometry-guide
retracing) has individual booleans for `occlusion`, `split_compound`,
`primitives`, `clones`, `symmetry`, `simplify`, `smooth`, and `stitch`.
All default to true; omitted fields keep those defaults. The canonical order
is unchanged. These controls affect only automatic refinement, not explicit
plan actions. `corner_normalize` remains a separate opt-in control (false).
`initial_stitch` defaults to true and controls the pre-refinement v0 stitch.
Final source-fill resolution is mandatory, not a switchable geometry pass.
Overlapping compound contours do not abort tracing. They remain addressable as
`compound_path` geometry, with their original path IDs, commands and fill rule.
`topology_pending=true` and `diagnostics.topology.pending` indicate that exterior
and cutout ownership is unresolved. Use plan geometry edits or `stitch` to refine
them; the topology classification is retried afterward. A pending diagnostic is
not a promise that stitching has repaired an internal contour overlap. Invalid
or unresolvable self-intersections still require correction rather than being
silently discarded.
The widget renders the pass names, defaults, and descriptions from the shared
Pydantic schema rather than maintaining a second pass catalog.

`background_color` optionally specifies a six-digit RGB color; otherwise the
boundary detector resolves it. `background_tolerance` defaults to 32 and uses
the same Euclidean RGB predicate for boundary detection and cutout matching.
With background removal enabled and a resolved color, finalized **flat-fill**
regions matching that color are negative space, including enclosed regions.
They are retained as addressable geometry. At final export, `subtract` removes
their overlap from earlier painted shapes and omits the background-colored
shape; `mask` does the same in SVG masks; `off` retains the original paint.
Later foreground shapes are never erased. Redundant shapes covering empty
space are simply omitted; non-intersecting paths are not subtracted. No export
mode mutates retained paths, regions, plans, or command addresses. Gradients and
raster fills are never classified as single-color cutouts. The background
policy survives refinement and geometry-guide retracing with the original
paint evidence, as do the independent output-fill controls.
`remove_background="off"` disables color cutouts. For cutting-machine export,
use `max_output_colors=2`, `allow_gradients=false`, and `background_cutouts="subtract"`
on black-and-white artwork. General artwork need not disable gradients.

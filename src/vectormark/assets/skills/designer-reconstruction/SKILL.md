---
name: designer-reconstruction
description: Reconstruct raster logos and illustrations as faithful editable SVG using VectorMark trace, inspection, semantic refinement plans, and optional geometry guides.
---

# Designer-quality reconstruction with VectorMark

The deliverable is an editable vector replacement for the source, not merely
a successful trace. Judge geometry and paint separately. Visible structural
differences need correction; ordinary antialiasing and gradient approximation
differences do not justify chasing individual pixels.

## Choose a starting point

If the user requests new artwork, use the host's available image-generation
model first and retain its actual PNG as the source. If no generation tool is
available, ask for a PNG rather than claiming an image was generated. The
plugin supplies vector tooling, not an image-generation model or API key.

Start with [use-case presets](references/trace-presets.md), then verify the
returned panel. These are suggested option bundles, not named server modes.
Use the live schema for accepted fields, enums, and defaults.

- `trace_drawing(image, options)` retains the original source. `refine="auto"`
  returns pre-refinement v0 and one normal revision per executed refinement
  pass, including unchanged results. Disabled passes have no revision. Use the
  returned `version` for the final result, not a hard-coded v1. Any retained
  revision can be inspected or used as a plan base. Use
  `refine="none"` when you need to inspect the baseline alone.
- Use the tool's generated options schema, not a copied enum list. A fixed
  `max_sampling_colors` is useful when the material count is known. Automatic colors
  are a proposal; inspect for lost material boundaries or color artifacts.
- Provide exactly one of `image` (host file object with required string
  `download_url` and `file_id`, optional string `mime_type` and `file_name`) or
  `image_data` (one `path`, `url`, `data_uri`, or `base64` source). Never send
  direct inputs in the host file field or supply both routes. Retracing uses
  `geometry_guide` versus `geometry_guide_data` under the same rules.
  Local paths require a trusted stdio server.
  Sources and geometry guides must be PNG. Export other formats to PNG before
  submitting; changing the filename alone is not sufficient.

## Inspect before declaring

Inspect the returned review panel: source, retained baseline, actual parent,
current output, region map, and amplified difference. For each region decide
whether geometry is a primitive or path, which corners/inflections are real,
whether other regions are clones, and whether self-symmetry is intentional.
Evaluate fills independently; a gradient must not redefine a silhouette.

Fetch `get_drawing_artifact` with `labeled_svg` for region addresses,
`path_map` for individual colored segments and exact command IDs in its side
legend, `command_map` for endpoint labels, and `raw_trace` for exact source
polyline addresses and corner spans. Use the selected version's reported
editability and correspondence; never guess a compressed command's source span.
Use IDs rather than inferring identity from color. Filter segment maps with
`region_ids` and `command_types`; move commands appear only in the endpoint map.
Plans always address the selected version's fitted commands and displayed
features (for example `r4.p0.c5` and `r4.p0.corner2`), never raw source IDs or
provenance hashes. Source correspondence is resolved internally. For corner
boundaries, check `corner_boundary.editable`; if exact correspondence is absent,
report the limitation rather than substituting a raw-polyline address.
Actually inspect the colored segment and corner/feature images at useful
region scale before planning edits and judging their results. Merely fetching
or embedding maps is not inspection. The required map checks, including
corner-pass and fit-pass review, are in [the evaluation loop](references/evaluation-loop.md#colored-map-review).

## Refine through declarations

Every `refine_drawing` plan includes the live `drawing_id`, chosen
`base_version`, and ordered `ops`. Returned versions can branch from any retained
version. Re-read child IDs after a split or refit. IDs in the examples are
illustrative and must be replaced with IDs from your selected version.

- `fit_primitive` replaces a whole region; `fit_path` refits a whole path.
- `update_geometry` carries path-local operations. `fit.target` selects one
  command; `fit.between=[A,B]` fits from the end of A to the end of B,
  replacing commands after A through B inclusive. A remains unchanged;
  adjacent IDs select B alone. Choose exactly one of `type` (one command) or `strategy`
  (recursive multi-command fitting). Explicit `type="cubic"` may inflect.
- Correct corner declarations before broad side refits. Plan fits propose
  geometry from the current curves. Acceptance measures curve-to-original
  material-polyline distance on the exactly corresponding retained span;
  original-to-curve distance is diagnostic only. Missing exact correspondence
  rejects the fit rather than substituting the current curves as its reference.
- `assign_clone` relates different regions with explicit transforms; self-
  symmetry is a separate `set_symmetry` declaration. Clones may have different
  paint. `stitch` reconciles shared region boundaries.
- Merge one branch ID to union its children, then split across a divider when
  semantic surfaces were over/under-segmented. Inspect the resulting child
  version before addressing those children.
- Auto refinement already runs detection. Do not routinely rerun `detect_*`;
  use it only to investigate a specific missed relationship visible in review.

Do not edit the exported SVG as a substitute for a supported plan. If plans
cannot express the required geometry, `retrace_drawing` accepts a flat-color,
transparent geometry guide with exactly the retained canvas dimensions. Guide
colors are labels; final fills come from the original source.

## Check and deliver

For iterative plan-based reconstruction, read
[the evaluation loop](references/evaluation-loop.md), including its active
region worklist and alternative-fit workflow. With local execution available,
the packaged runner saves every trial, exposes separate shape/fill review, and
generates an offline artwork handoff with a Review / Edit inspector. Remote
agents follow the same decision loop using the public MCP tools and available
artifact storage; local shell access and widget interaction are not required
to continue refining. Do not claim a local report was saved without file access.
The engine's public plans and artifact contracts remain authoritative.
For an authorized reconstruction, actively work through the region worklist:
inspect each result, try evidence-backed alternative fits after a rejection,
and move to another region when a local attempt is exhausted. A failed trial
or a generated report is not the end of the evaluation. Follow the reference's
bounded search and stopping rules; do not wait for user input after each trial.

`get_drawing_artifact(artifact="svg")` returns the exact SVG text plus an MCP
file resource link. Resource-capable clients can read that link; it is not an
HTTP URL or a `/mnt/data` path. If you have filesystem tools, save
`structuredContent.svg` verbatim and attach the resulting file. Do not claim
a downloadable file exists merely because SVG text was retrieved. In the
review widget, select SVG and use **Export selected file**, then the host's
**Download** link. If host file APIs are unavailable, report that limitation;
do not retrace, invent a URL, or edit the SVG to work around delivery.

If an internal MCP failure blocks conversion and local VectorMark execution is
available, read [the direct fallback](references/direct-fallback.md). This
produces an independent file, not a retained MCP version or a successful plan.

Inspect each accepted child's panel and source difference. Revert by branching
from a better retained version when a proposal worsens fidelity. Stop when the
vector has faithful geometry, appropriate primitives/relationships, clean
corners and joins, and correct paint. If an unsupported edit prevents that,
report the limitation rather than claiming designer fidelity.

Read [the simple-logo walkthrough](examples/simple-logo.md) for a concrete
trace → primitive declarations → local curve-refit example. Its companion JSON
plans are checked against the same action models used by the server.

Automatic geometry passes can be selected independently with
`options.refinement_passes`: occlusion, split_compound, primitives, clones,
symmetry, simplify, smooth, and stitch (all default true). Corner normalization
is independently opt-in; initial_stitch defaults true for v0. Omitted pass
fields use defaults. Final source paint is always resolved after geometry.

For transparent negative space, use the inferred boundary background color or
set `background_color` explicitly (for example `#ffffff`). The shared
`background_tolerance` controls both boundary and final flat-fill matching.
Matching single-color regions become export-only cutouts without refitting.
`background_cutouts="subtract"` (default) removes overlap from earlier painted
shapes and emits actual path holes; `"mask"` uses SVG masks; `"off"` retains paint.
Later foreground survives; redundant background regions are simply omitted.
`max_sampling_colors` controls segmentation only. `max_output_colors` limits
flat-fill and gradient-stop palette colors; `allow_gradients=false` requires
flat output. These settings survive refinement and guide retracing. For a
two-color cutting file use sampling/output limits of 2, gradients false, and
subtract export. Do not assume a small sampling palette disables gradients.
Gradients remain paint. `remove_background="off"` preserves those fills.
Inspect the report's `cutout` flags and verify transparency before delivery.

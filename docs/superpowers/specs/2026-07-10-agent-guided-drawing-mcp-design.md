# Agent-Guided Drawing MCP Design

**Date:** 2026-07-10  
**Area:** `src/vectormark/mcp_server.py` plus new trace, plan, and session-state modules.

## Goal

Replace the logo-specific MCP surface with one drawing-oriented entry point.
`trace_drawing` creates a retained trace that an agent can refine through
versioned plans, with optional automatic refinement as its first child version.

```text
trace_drawing(image, { refine: "auto", ... })
  -> drawing_id + pre-optimizer v0 + automatic v1 + review artifacts

trace_drawing(image, { refine: "none", ... })
  -> drawing_id + version v0 + trace paths + labeled region map

refine_drawing(plan)
  -> drawing_id + derived version + SVG + refinement report + preview
```

## Product boundary

The drawing workflow is stateful and MCP-server-local. It retains an immutable
`Drawing` snapshot for every version. A `DrawingRegion` carries canonical typed
geometry, fill, z-order, stable region ID, and region-level source provenance.
Raster masks, Skia footprints, and optimizer IDs are private engine evidence;
they are not fields on the scene contract.
`refine_drawing` accepts no image and reads the opaque `drawing_id` from the
plan. Both trace modes allocate a drawing ID and participate in versioning.

Each drawing belongs to the originating FastMCP `Context.session`. A drawing ID
is cryptographically random, valid only within that session, and expires after
30 minutes of inactivity. A successful refinement refreshes its idle expiry.
Unknown, expired, or cross-session IDs return the structured error code
`DRAWING_NOT_FOUND`. The MVP never writes drawing state to disk and does not
provide a portable trace import/export format.

## Trace modes

`refine` is required and has exactly two values:

- `auto`: retain pre-optimizer v0, run automatic optimization as immutable v1,
  and return both outputs plus the review artifacts. Later plans may branch
  from either version.
- `none`: retain and return only pre-optimizer v0 so an agent can inspect and
  refine the decomposition without automatic geometry decisions.

The shared image-reference, preprocess contract, and six trace options remain
unchanged. Automatic mode maps `max_colors`, `trace_level`,
`simplify_tolerance`, `curve_tolerance`, and `curve_type` to the current
idealizer; `min_region_size` supplies its absolute component-size floor. It does
not carry forward logo-only flattening or manual symmetry toggles.
`render_idealized_logo` is renamed `render_drawing` and accepts either result
shape.

## Retained trace contract

`trace_drawing` uses the existing secure image resolver and preprocessor,
then retains both deterministic raw paths and a compact agent-facing root
forest. Raw paths are produced by palette segmentation, contour extraction, and
path fitting; roots use the one-shot pipeline's size filtering and soft-surface
merge so an agent does not need to reason about every palette fragment. It does
**not** run primitive recognition, symmetry detection, clone detection, or
other optimizer passes.

Before `v0` is retained, its compact root forest is automatically
stitched so neighboring retained regions share reconciled boundaries. This does
not alter the on-demand `raw_trace` artifact. Agents may still invoke `stitch`
after later geometry, merge, or split edits create new neighboring edges.

The public path is named `trace_path`, not `raw_path`. The stored contour samples
are the raw pixel or coverage evidence; `trace_path` is a pre-simplified line and
Bézier representation of that evidence. It is deliberately free of semantic
interpretation while remaining compact enough for stable command IDs and agent
reasoning.

### Trace options

```json
{
  "max_colors": 16,
  "min_region_size": 16,
  "min_region_fraction": 0.02,
  "trace_level": "pixel",
  "simplify_tolerance": 1.5,
  "curve_tolerance": 1.0,
  "curve_type": "quadratic"
}
```

- `max_colors` is the palette-quantization ceiling.
- `min_region_size` is the minimum retained connected-component pixel count.
- `min_region_fraction` is the relative-to-largest-region floor used when
  constructing the agent-facing root regions. The complete raw trace remains
  available on demand, but small fragments do not become editable roots by
  default.
- `trace_level` is `pixel` for binary quantized-mask contours or `subpixel` for
  anti-alias-aware coverage contours. The result reports each region's effective
  level because the coverage safety guard may fall back to `pixel`.
- `simplify_tolerance` controls contour simplification and straight-run detection.
- `curve_tolerance` bounds fitted Bézier error.
- `curve_type` is `quadratic` by default or `cubic` for intentionally complex
  contours. Quadratics are the default because they resist raster stair-stepping.

Input crop, resize, and alpha handling remain in the existing preprocess options.
There are no trace options for symmetry, primitive recognition, fill inference,
surface merging, flattening, or shadow removal.

The returned structured result contains compact root data plus artifact
references; it does not embed large SVG strings or raw command lists in the
trace response. `raw_trace` is the on-demand artifact for the underlying raw
paths and stable command IDs:

```json
{
  "drawing_id": "drw_<opaque>",
  "version": "v0",
  "trace": {
    "width": 1024,
    "height": 1024,
    "regions": [
    {
      "id": "r1",
      "source_regions": ["r1", "r7"],
      "geometry": { "type": "path", "d": "M…Z" },
      "fill": { "type": "linear_gradient" }
    }
    ]
  },
  "artifacts": {
    "svg": "drawing://drw_<opaque>/v0.svg",
    "preview": "drawing://drw_<opaque>/v0.png",
    "labeled_svg": "drawing://drw_<opaque>/v0.labels.svg",
    "raw_trace": "drawing://drw_<opaque>/v0.trace.json"
  },
  "report": { "targets": [{ "id": "r1", "source_regions": ["r1"], "geometry": "path", "z": 0 }] }
}
```

Root IDs are deterministic for one trace: the retained one-shot surfaces sort
by descending area, segment label, and color. Raw command IDs include a
subpath index so holes and disconnected contours remain unambiguous.

The default labeled SVG renders editable roots and their region IDs. The raw
trace artifact contains the exhaustive raw-path region map for command-level
planning when a root needs a custom path.

## Refinement-plan contract

`refine_drawing` accepts a single typed plan:

```json
{
  "version": "vectormark.plan.v1",
  "drawing_id": "drw_<opaque>",
  "base_version": "v0",
  "label": "optional user-facing description",
  "defaults": { "epsilon": 1.5, "max_error": 1.0 },
  "ops": []
}
```

The server validates the plan before mutation: the drawing exists for the
current session; the base version exists; every referenced region, group,
command, and fill is known;
operation IDs are unique; and path source commands form a directed contiguous
run within one raw subpath. Validation errors are returned as `INVALID_PLAN`
with a JSON pointer and actionable message.

`defaults.epsilon` and `defaults.max_error` override the retained trace
defaults for the plan. `set_geometry`, `detect_primitives`, and
`detect_symmetry` may each supply either field directly; an operation value
overrides the plan default. `epsilon` governs simplification/primitive fitting
and `max_error` bounds Bézier fitting. Both must be finite and non-negative.
Detection diagnostics are included with each returned target. In particular,
`detect_symmetry` returns its accepted axis there, so the next versioned plan
can use that axis directly in `set_symmetry`.

### Version tree

Every trace begins at immutable version `v0`. A successful plan creates a new
immutable child of its base version; it never overwrites its parent. Version IDs
are branch paths: the first, second, and third refinements of `v0` are `v0.0`,
`v0.1`, and `v0.2`; the first child of `v0.1` is `v0.1.0`. The server assigns the
next child segment atomically, so agents provide only `base_version` and cannot
collide when submitting alternatives concurrently.

Each version stores its parent version, accepted plan, immutable `Drawing`
snapshot, private engine evidence, and optional plan `label`. SVG, preview,
labeled SVG, and the target report are rendered on demand from the `Drawing`.
There is no parallel `DrawingScene`, `VectorRegion`, or `RenderTarget` public
tree. Agents can branch from any retained
version and present sibling alternatives without retracing the original raster.
Each operation is a pure transform that returns fresh affected `DrawingRegion`
objects; the complete plan is atomically committed as one child drawing version.
No operation mutates the base version or creates a separately persisted partial
version.

### Drawing actions

Every top-level action is applied in the context of one `Drawing`; there is no
separate drawing-scope versus region-tree-scope action hierarchy. Its
parameters determine scope:

```text
target / targets omitted  → applicable drawing roots
target: "r1"              → r1 and its descendants
targets: ["r1", "r2"]     → explicit region or relationship set
```

The canonical typed action vocabulary is:

```text
detect_primitives(target?)
detect_symmetry(target?)
detect_clones(target?)
merge(id, regions[])
split(target, divider)
set_geometry(target, assignment)
set_fill(target, fill)
set_clone(source, target, transforms[])
set_symmetry(target, axis)
align(target, reference, axes)
stitch(targets?, epsilon?, max_error?)
set_z_order(targets[])
```

`set_geometry(assignment.type="path")` is a composite drawing action. The engine applies
its nested path actions in order, placing each returned `PathGeometry` into a
fresh drawing snapshot before invoking the next action. This lets a later path
action reference commands produced or retained by an earlier one.

The engine also exposes `apply_path_action(PathActionRequest) ->
PathActionResult` directly. The request contains the current `Drawing`, target
region ID, action, and fitting tolerances so cross-region command references
can be resolved without a parallel command-lineage map. The optimizer may call
this operation directly; it remains an orchestrator rather than a second path
implementation.

Drawing actions may lower to the same path machinery internally without
changing the public plan hierarchy. `merge` and `split` create a new outline
and refit it; `set_symmetry` half-fits, mirrors, and unions a path; `stitch`
reconciles corresponding region boundaries. `align`, `set_clone`, and fill
assignment remain region operations.

- `merge` creates a semantic target from regions, retains their combined
  source-region ancestry, and serializes their `SkPath` boolean union as one
  outline. This removes shared interior edges before SVG emission so adjacent
  same-fill regions cannot produce anti-aliased hairlines. It invalidates the
  merged target's fill, which is subsequently re-evaluated by the fill pass.
- `split` divides a target using either an infinite line from two points or a
  supplied divider path. It creates addressable child regions for the next
  versioned plan and invalidates their fills.
- `set_geometry` accepts either a primitive assignment (`circle`, `ellipse`,
  `rect`, `rounded_rect`, or `polygon`) or a path assignment containing the
  nested path actions below. The engine resolves that intent to concrete typed
  geometry; agents do not author final SVG coordinates.
- `set_clone` establishes a source/target relationship and applies one or more
  transforms. The target becomes `CloneGeometry`; its fill remains independent
  of the source, so visually related clones may use different fills.
- `set_symmetry` establishes self-symmetry for a target from an explicit axis.
  The resulting geometry is a `PathGeometryWithSymmetry`. Before seam polish,
  self-symmetry is materialized and unioned so the axis cannot emit a hairline.
- `stitch` is region-only. It reconciles shared boundaries between neighboring
  regions and may internally snap endpoints or adjust adjacent handles, but it
  is not serialized as independent path commands.
- `detect_*` is global when `target` is omitted and is restricted to a target
  root/subtree otherwise. Clone detection includes translation, rotation, and
  reflected affine clones; accepted clones can emit SVG `<use>`.
- `set_z_order` establishes final paint order for all retained emitted targets.

Every action returns a fresh drawing plus `affected_region_ids`,
`created_region_ids`, `retired_region_ids`, and diagnostics. The store assigns
the derived version ID around that result; actions never mutate their base
drawing.

`split` is structural and must be the final operation in a plan.
`detect_symmetry` is structural only when it needs to split a freeform
self-symmetric path. A global symmetry detection must be final; a terminal
block may contain several targeted `detect_symmetry` operations for distinct
regions. Inspect the returned report, then address any derived hyphenated
children in a new versioned plan. Explicitly selected intrinsically symmetric
geometry, such as a cap or rounded trapezoid, is retained as one region and
reported as `mode: "intrinsic"` rather than being split into a redundant pair.
Clone `<use>` relationships remain symbolic through retained trace modes
output unless explicit SVG flattening is requested. A self-symmetry mirror
`<use>` is the exception: it is baked to a concrete path immediately before
`simplify` or `stitch`, so polish passes see both sides of a shared seam and
cannot leave an anti-aliased hairline.

Automatic one-shot refinement runs the existing optimizer pass sequence,
including occlusion, compound splitting, symmetry, clones, simplification, and
stitching. Version v0 remains pre-optimizer; plans may opt into individual
automatic behaviors through ordered `detect_*` drawing actions.

### Path actions

For `set_geometry` with `type: "path"`, the assignment owns an ordered `ops`
array. These are the only nested actions because they address version-scoped
command IDs rather than drawing-region IDs:

```text
fit(target | between, type | strategy)
remove(target)
match(target, reference, transform?)
set_parallel(target, reference, distance?)
match_length(target, reference)
align(target, reference, axes)
snap(target, reference)
break(target)
reverse(target)
mirror(target, reference, axis)
simplify()
normalize_corners()
```

- `fit.type` selects the single-command fitter: `line`, `quadratic`, `cubic`,
  or `keep`. A cubic may inflect when that is the appropriate fit.
- `fit.strategy` selects recursive fitting across multiple commands:
  `quadratic`, `cubic`, `progressive`, or `progressive_allow_lines`.
- A `target` refits that retained command. `between: [left, right]` refits the
  commands strictly between ordered boundary commands: it begins at the end of
  `left` and ends at the start of `right`. `type` may be used with either form
  for one replacement curve; `strategy` applies to a multi-command span.
- `remove` removes a command and joins its neighbours. The resulting join is
  eligible for later path fitting and region-level `stitch`.
- `set_parallel` preserves the target's existing length unless `distance` is
  supplied; without a distance it fits parallel geometry within the retained
  region evidence.
- `simplify` and `normalize_corners` are path-only. Corner normalization may
  operate only on existing pre-fit corner provenance; it cannot invent a new
  corner from a post-fit tangent.

Adjacent fitted path segments are tangent-continuous by default; only `break`
requests a sharp join. Paths are closed by the `PathGeometry` invariant, so
there is no `close` operation. Arbitrary coordinate-authored insertion is not a
public operation: `fit` over a retained span is the coordinate-free mechanism
for replacing or creating commands.

## Execution model

The trace module produces an immutable canonical `Drawing`, private engine
evidence, and a public JSON-safe summary. The plan executor transforms the
cached `DrawingRegion` tree. The selected engine uses private evidence to fit
requested geometry and run named `detect_*` operations; it returns new typed
geometry rather than exposing optimizer records. It does not mutate the stored
trace. Each accepted plan creates an immutable version, so the agent may issue
multiple branches for one drawing.

The refinement result includes its new version ID, artifact references, and
target-level geometry/fill/provenance summaries. Call
`get_drawing_artifact` to retrieve the SVG, clean PNG preview, or labeled SVG.

### Trace-engine boundary and native follow-up

The tracing implementation is VectorMark's own pipeline: palette segmentation,
`find_contours`, and custom line/Bézier fitting. It is not Potrace or vtracer.
The MCP MVP wraps it behind a `TraceEngine.trace(rgb, options) -> TraceResult`
boundary. The initial implementation uses the existing Python/scikit-image
pipeline; retained `DrawingState` ensures every branch refines a trace once
rather than retracing it.

Trace-engine substitution is a Phase 2 performance project, after profiling
representative MCP drawings. A Rust implementation or an adapter for a
third-party tracer may replace segmentation, contour extraction, and curve
fitting behind the same `TraceEngine` interface. Any replacement must preserve
the public trace result, command-ID rules, and trace-options semantics, so it
does not alter the MCP tools, drawing store, version tree, or plan schema.

## Non-goals

- CLI commands or a portable trace file.
- Ribbons, arcs, path congruency, and a general constraint solver.
- Persistence across an MCP server restart or session end.
- Retaining state for automatic calls.

## Testing and acceptance

Unit tests cover deterministic IDs, command parsing, session isolation, sliding
TTL, expiration, plan validation, primitive fitting, path operation execution,
fill emission, z-order, and no mutation of stored state. MCP integration tests
call both tools over stdio using a data URI.

Given a flat folded-ribbon-style raster, acceptance requires a labeled region
map, two sibling plans from `v0` with distinct labels, a child plan from one
sibling, native circles for equal dots, a flat or gradient fill, and no automatic
symmetry or candidate inference during tracing.

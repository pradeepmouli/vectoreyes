# Agent-Guided Drawing MCP Design

**Date:** 2026-07-10  
**Area:** `src/vectoreyes/mcp_server.py` plus new trace, plan, and session-state modules.

## Goal

Replace the logo-specific MCP surface with one drawing-oriented entry point.
`trace_drawing` creates a retained trace that an agent can refine through
versioned plans, with optional automatic refinement as a chain of child revisions.

```text
trace_drawing(image, { refine: "auto", ... })
  -> drawing_id + trace v0 + one revision per executed refinement pass + review artifacts

trace_drawing(image, { refine: "none", ... })
  -> drawing_id + version v0 + trace paths + labeled region map

refine_drawing(plan)
  -> drawing_id + derived version + SVG + refinement report + preview
```

## Product boundary

The drawing workflow is stateful and MCP-server-local. It retains an immutable
`Drawing` snapshot for every version. A `DrawingRegion` carries canonical typed
geometry, fill, z-order, stable region ID, optional lowercase hyphenated name,
and region-level source provenance. Parent regions with children are groups;
there is no independent Group or Layer engine type. `set_region_name` is a
metadata-only action, performed before geometric revisions. Region inputs accept
stable ID strings or explicit `{"name":"orange-band"}` selectors; accepted
recipes resolve names to IDs in the selected version/ordered plan state.
Ambiguous names fail with matching IDs, and renaming cannot retarget old recipes.
Commands, paths and feature IDs remain version-scoped fitted addresses.

Cutouts are relationships between ordinary regions: `assign_cutout(source,
targets)` retains geometry and declares subtraction at export. The source owns
the stable target IDs in `cutout_targets`; an empty list removes the relationship.
Parent targets cover descendants. Self/subtree subtraction, nested operands and
cycles are rejected; referenced targets cannot be silently retired. This is
independent of color-inferred automatic background cutouts and paint order.
Editable SVG supplies standard titles, Inkscape labels and versioned metadata;
parent regions serialize as groups without changing paint order. Plain SVG
omits application extensions, and visual rendering always uses standard SVG.
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

- `auto`: retain trace v0 and one ordinary immutable revision per executed
  refinement pass, including no-change results. Disabled passes create no
  revision. Return all outputs and the selected final version; never assume v1.
  Later plans may branch from any retained version.
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
  "version": "vectoreyes.plan.v1",
  "drawing_id": "drw_<opaque>",
  "base_version": "v0",
  "label": "optional user-facing description",
  "settings": {
    "epsilon": 1.5,
    "max_error": 1.0,
    "fit_strategy": "quadratic"
  },
  "ops": []
}
```

The server validates the plan before mutation: the drawing exists for the
current session; the base version exists; every referenced region, group,
command, and fill is known;
operation IDs are unique; and path source commands form a directed contiguous
run within one raw subpath. Validation errors are returned as `INVALID_PLAN`
with a JSON pointer and actionable message.

Each drawing action has an optional `settings` object with the same fields.
Settings resolve in one order: retained trace settings, then plan settings,
then the action override. The executor passes the resulting complete
`ActionSettings` to the engine. Nested operations in `update_geometry` inherit
their parent action's resolved settings. `epsilon` governs
simplification/primitive fitting, `max_error` bounds Bézier fitting, and
`fit_strategy` selects `quadratic`, `progressive`, or
`progressive_allow_lines` for whole-region path fits. Numeric values must be
finite and non-negative.
Detection diagnostics are included with each returned target. In particular,
`detect_symmetry` returns its accepted axis there, so the next versioned plan
can use that axis directly in `set_symmetry`.
The shared axis value is `{"point": [x, y], "angle_degrees": degrees}`:
the point lies on the reflection line in drawing coordinates, and the angle
is measured from positive x (90 degrees is vertical). Detected self-symmetry,
intrinsic symmetry, and explicit symmetry use this same representation; no
client-side radians conversion is required.

### Version tree

Every trace begins at immutable version `v0`. A successful plan creates a new
immutable child of its base version; it never overwrites its parent. Version IDs
are branch paths: the first, second, and third refinements of `v0` are `v0.0`,
`v0.1`, and `v0.2`; the first child of `v0.1` is `v0.1.0`. The server assigns the
next child segment atomically, so agents provide only `base_version` and cannot
collide when submitting alternatives concurrently.

Each version stores its parent version, submitted recipe edits, effective plan
and declarations, resolved settings, immutable `Drawing` snapshot, references to
private engine evidence, and optional plan `label`. The effective recipe,
evidence, and retained resolved operation outcomes reproduce the snapshot;
the snapshot does not replace the recipe. Exploratory generation is not rerun
merely to read, render, or reproduce a published version.
SVG, preview,
labeled SVG, and the target report are rendered on demand from the `Drawing`.
There is no parallel `DrawingScene`, `VectorRegion`, or `RenderTarget` public
tree. Agents can branch from any retained
version and present sibling alternatives without retracing the original raster.
Each operation returns fresh affected `DrawingRegion` objects; generating a
proposal may be exploratory rather than deterministic. The complete selected
result is atomically committed as one child drawing version.
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
fit_path(target)
fit_primitive(target, type)
update_geometry(target, ops[])
set_fill(target, fill)
assign_clone(source, target, transforms[], symbolic=true)
reconstruct_occlusion(target, regions[], children[], base_z?)
normalize_topology()
set_symmetry(target, axis)
align(target, reference, axes)
stitch(targets?, cubic=false, post_symmetry=false, allow_underlap=true)
set_z_order(targets[])
```

`update_geometry` is a composite drawing action. The engine applies its nested
geometry actions in order, placing each returned geometry into a
fresh drawing snapshot before invoking the next action. This lets a later path
action reference commands produced or retained by an earlier one.

The engine also exposes `apply_path_action(PathActionRequest) ->
PathActionResult` directly. The request contains the current `Drawing`, target
region ID, action, and resolved `ActionSettings` so cross-region command references
can be resolved without a parallel command-lineage map. The optimizer may call
this operation directly; it remains an orchestrator rather than a second path
implementation.

Evidence-based drawing operations use
`apply_action(DrawingActionRequest[Evidence]) ->
DrawingActionResult[Evidence]`. The request pairs the immutable `Drawing` with
opaque, immutable engine evidence and resolved `ActionSettings`. The result returns
both a fresh drawing and fresh evidence. `trace_drawing` establishes that pair
as `TraceDrawingResult[Evidence]`. For the Python engine the evidence contains
raster masks, Skia footprints, source pixels, and optimizer bookkeeping; a
native engine may use a different private representation without changing the
scene, plan, store, or MCP contracts.

Drawing actions may lower to the same path machinery internally without
changing the public plan hierarchy. `merge` and `split` create a new outline
and refit it; `set_symmetry` mirrors and unions the retained path without refitting; `stitch`
reconciles corresponding region boundaries. `align`, `assign_clone`, and fill
assignment remain region operations.

- `merge` creates a semantic target from regions, retains their combined
  source-region ancestry, and serializes their `SkPath` boolean union as one
  outline. This removes shared interior edges before SVG emission so adjacent
  same-fill regions cannot produce anti-aliased hairlines. It invalidates the
  merged target's fill, which is subsequently re-evaluated by the fill pass.
- `split` divides a target using either an infinite line from two points or a
  supplied divider path. It creates addressable child regions for the next
  versioned plan and invalidates their fills.
- `fit_path` replaces a target's complete geometry with a newly fitted path.
- `fit_primitive` replaces a target's complete geometry with a fitted
  `circle`, `ellipse`, `rect`, `rounded_rect`, or `polygon`.
- `update_geometry` owns an ordered region-local path program. Its
  command-addressed operations update retained path geometry without changing
  the geometry representation. Agents do not author final SVG coordinates.
- `assign_clone` establishes a source/target relationship and applies one or more
  transforms. By default the target becomes `CloneGeometry`; `symbolic=false`
  materializes the transforms without fitting, with one child per instance when
  there are multiple transforms. Its fill remains independent of the source.
- `reconstruct_occlusion` publishes an explicit typed graph of completed
  primitives, retained source boundaries, and intersection lenses through the
  automatic reconstruction constructor. Child paint comes from named source
  regions; dependency and live-clone checks remain mandatory.
- `normalize_topology` is drawing-wide retained component/ownership normalization,
  including command correspondence and dependent clones, not an intentional split.
  These two structural declarations must be final in a plan; edit their returned
  regions in a subsequent revision.
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

`update_geometry` owns an ordered `ops` array. Its operations address
version-scoped path command IDs and are applied sequentially:

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
simplify(linelet_only=false, samples=16, cubic=false, max_boundary_error?)
smooth(max_boundary_error?)
normalize_corners()
```

- `fit.type` selects the single-command fitter: `line`, `quadratic`, `cubic`,
  or `keep`. A cubic may inflect when that is the appropriate fit.
- `fit.strategy` selects recursive fitting across multiple commands:
  `quadratic`, `cubic`, `progressive`, or `progressive_allow_lines`.
- A `target` refits that retained command. `between: [A, B]` begins at the end
  of `A` and ends at the end of `B`, replacing commands after `A` through `B`
  inclusive. `A` remains unchanged; the replacement preserves `B`'s endpoint.
  Adjacent IDs select `B` alone. Boundaries must be ordered in one subpath;
  a span may end at its explicit closing drawable command. `type` may be used
  with either form for one replacement curve; `strategy` permits multiple.
- `remove` removes a command and joins its neighbours. The resulting join is
  eligible for later path fitting and region-level `stitch`.
- `set_parallel` preserves the target's existing length unless `distance` is
  supplied; without a distance it fits parallel geometry within the retained
  region evidence.
- `simplify`, `smooth` and `normalize_corners` are path-only. Corner normalization may
  operate only on existing pre-fit corner provenance; it cannot invent a new
  corner from a post-fit tangent.
- Simplification and smoothing use the same discovery and execution as automatic
  passes. Their optional `max_boundary_error` defaults to resolved `max_error`;
  explicit zero is valid. Automatic linelet-only cleanup can be requested via
  `linelet_only=true`; the public broader simplification default is unchanged.
  Cleanup positions/repetitions among direct edits are preserved, and replay
  applies recorded operations/settings without rediscovery.

Corner diagnostics distinguish boundary anchors from owned geometry. Each
rounded span records `entry_command` and `exit_command` (command-end anchor
addresses), plus `commands` in directed path order. Ownership starts **after**
the entry command and includes the command arriving at the exit; it never
includes the following side. `curve_commands` identifies the Q/C members.
A point corner owns no drawable command. `anchor_commands` and `anchors`
describe node membership independently; touching corners may share a boundary
anchor and its `corner_ids`, but overlapping curve ownership is not normalized.

Normalization preserves both endpoints and unaffected side handles, handles
closed-path wraparound in directed order, and transfers the original pre-fit
corner IDs/classification to replacement commands. A single Q is already
normalized. Missing, stale, ambiguous, or overlapping provenance produces an
explicit skip, including on repeated no-ops. An unknown radius does not convert
a rounded declaration into a point. Reports expose `corner_counts` separately
from `commands.corner` (owned command count); neither is inferred from fitted
tangents.

Adjacent fitted path segments are tangent-continuous by default; only `break`
requests a sharp join. Paths are closed by the `PathGeometry` invariant, so
there is no `close` operation. Arbitrary coordinate-authored insertion is not a
public operation: `fit` over a retained span is the coordinate-free mechanism
for replacing or creating commands.

## Operation purpose and proposal semantics — 2026-10-16

Geometry operations have one or both purposes:

- **Reconstruction accuracy:** reproduce raster-evidenced geometry as closely
  as practical within explicit measurement budgets.
- **Design fidelity:** adjust or rebuild the result into a construction a
  designer would have authored, using defaults and optionally automatic,
  agent, or user intent.

Purpose, intent origin, and execution behavior are independent. Design-fidelity
operations provide defaults but need not receive a fully specified construction;
their inputs, interpretation, and outputs may differ across runs. An explicit
transform may still be deterministic while serving design fidelity. Accuracy
operations may also use exploratory solvers. One shared operation implementation
serves either purpose or both; do not create two geometry engines.

Separate **proposal generation** from **commitment**. An operation may propose
one or more constructions; assess actual candidate geometry rather than equality
with an earlier proposal. Report raster error, design-constraint satisfaction,
topology, and complexity separately. Design fidelity does not silently waive an
error limit: any accuracy trade-off must be explicit in the resolved policy.
Invalid geometry, non-finite results, and unresolved references remain hard errors.

Commit the selected exact canonical geometry, effective intent, resolved settings,
bindings, assessments, and operation outcomes atomically. Commitment is not a
claim of visual acceptance; a needs-refinement branch remains labelled as such.
Reproduction uses the retained result/resolutions without exploratory generation.
**Re-proposal** deliberately executes intent again, may produce a different valid
result, and never overwrites the prior committed version.

Bind a retained resolution to its operation/intent revision, evidence revision,
resolved settings, and upstream dependencies. Changed dependencies invalidate
reuse: return an actionable binding error or enter an explicitly requested
re-proposal workflow, never silently reuse a stale outcome or regenerate one
while reproducing a version. Seeds, when available, are provenance rather than
a universal reproducibility guarantee. Reuse canonical geometry and replay
receipts; do not introduce a second authoritative drawing tree.

These are execution contracts, not additional MCP tools or an assertion that
exploratory candidate selection has already been implemented.

## Execution model

The trace module produces an immutable canonical `Drawing`, private engine
evidence, and a public JSON-safe summary. The agreed target execution model is
recipe replay: the selected version supplies the inherited effective plan and
declarations, agent edits revise that recipe, and the engine reconstructs from
retained evidence and valid committed operation resolutions. Exploratory work
is regenerated only through explicit re-proposal. The parent's generated geometry is a comparison reference,
not an implicit constraint to preserve. Individual actions transform a working
`DrawingRegion` tree during replay; rendering never performs reconstruction.

Deleting a declaration removes its effect on reconstruction, rather than freezing
the geometry it previously produced. Explicit geometry deletion is separate.
Persist stable recipe identities and bindings to evidence or named operation
outputs; bind displayed version-scoped command selections before replay and
reject unresolved or ambiguous targets. Geometry changes refresh feature maps
and affected paint fits. Unaffected work may be cached only when equivalent to
full replay. Each accepted recipe creates one immutable child version atomically.

Recipe editing, stable binding, and full replay are refactoring requirements,
not claims that the current deployed tool supports them already. The detailed
contract and verification cases are in
[Slice 3 of the reconstruction plan](../plans/2026-09-07-designer-quality-reconstruction.md#agreed-reconstructionreplay-contract--2026-09-09).

The refinement result includes its new version ID, artifact references, and
target-level geometry/fill/provenance summaries. Call
`get_drawing_artifact` to retrieve the SVG, clean PNG preview, or labeled SVG.

### Trace-engine boundary and native follow-up

The tracing implementation is vectorEyes's own pipeline: palette segmentation,
`find_contours`, and custom line/Bézier fitting. It is not Potrace or vtracer.
The MCP MVP wraps it behind a `TraceEngine.trace(rgb, options) -> TraceResult`
boundary. The initial implementation uses the existing Python/scikit-image
pipeline; retained evidence allows recipe replay without repeating segmentation
or contour extraction unless source, guide, or relevant trace settings change.

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

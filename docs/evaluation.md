# Plan evaluation reports

`python -m vectormark.evaluation --help` exposes the script-first review loop.
Use the existing MCP endpoint; the runner does not start, stop or replace it.
Install the server and scoring extras for MCP client and PNG rendering support.

```sh
python -m vectormark.evaluation trace source.png output/run --options options.json
python -m vectormark.evaluation refine output/run plan.json --hypothesis "One curve per supported side"
python -m vectormark.evaluation report output/run
```

Open `output/run/index.html`: a clean artwork handoff leads into Review / Edit.
The pass-revision list selects normal retained versions and compares each with
its actual parent. Every executed refinement pass creates a revision, including
unchanged outcomes; disabled passes have no revision. The runner captures the
returned outputs from one trace request, not separate trace executions.
Parameters and per-region complexity appear below the artwork inspector.
Boundary detail also follows the inspector. The synchronized comparison includes
an absolute raster-to-clean-SVG RGB diff, amplified 4× on white; annotations and
neutral review coloring never enter it. Black indicates matching pixels. This
is a full-appearance diagnostic, not a combined shape/fill acceptance score.

The first page contains retained pass history and an editable copy of the captured
trace parameters/pass selections. Copy agent/MCP rerun instructions or download
the options JSON and original PNG. This portable handoff prepares a new trace;
it does not execute a request or modify existing drawings. Historical parameters
under Review / Edit remain the actual captured values, not pending edits.
The selected trial's **Parameters used** disclosure separates its submitted
request from retained parameters for the displayed artwork. Trace options
link to their captured response; effective recipe/action overrides and output
policy come from that version's evidence. Missing values stay unavailable.
For a failed trial, the displayed parent's parameters are not presented as
effective settings for an unpublished candidate.
The report is self-contained for offline inspection; SVG download targets the
exact saved file beside it. It does not execute plans from browser selections.

Review / Edit also shows **Per-region complexity**, comparing the displayed
version with its captured parent or baseline. It lists local paths, L/Q/C/A
segments, corner-owned/free commands and point/rounded corners, with downloadable
JSON. Select a region to focus the table. Missing corner evidence is unavailable,
not zero; primitive/clone geometry has no local path commands. Count deltas need
the same drawing, region ID and nonempty source-region lineage. Missing,
ambiguous or changed lineage stays unpaired rather than guessing merge/split
correspondence. Fewer commands alone are not a quality verdict.

Report regeneration derives inspection counts from captured production responses,
not from re-tracing the image or trusting an old derived inspection summary.
Saved response, recipe, source, SVG and inspection artifacts remain unchanged.
An SVG without its captured response is not a count-bearing production version.

The run records exact requests, failures, candidate versions, artifacts and
independent shape/fill verdicts. No version is accepted automatically. Use
`review --track shape|fill` and then `accept` only after actual image review.
`view` exports matching annotated SVG/PNG/JSON without requiring UI automation.
`capture` retries artifact retrieval for a known saved response without another
mutation. `reconcile` saves the live version manifest after unknown execution.

See the packaged [evaluation reference](../src/vectormark/assets/skills/designer-reconstruction/references/evaluation-loop.md)
for concrete commands and lifecycle limits, and the
[design specification](superpowers/specs/2026-10-14-plan-evaluation-report-design.md)
for the review model. Artifact generation and interactive display use the same
retained command/feature data and canonical map exports; neither redetects
features. Shape is neutral; fill source sampling uses the current geometry's
coverage. No blended overall quality score is computed.

Historical artifact folders can be imported with `import SNAPSHOT SOURCE RUN`.
They remain historical evidence, not restored engine sessions. Server restarts
expire drawing handles but do not delete the report files. Source settings and
server schema are saved; an unexposed server build identity is labeled unknown.

For Codex/Claude distribution, see [the shared agent plugin](plugins.md).
It packages the existing reconstruction skill; it does not duplicate the
engine or require the in-chat widget.

## Source-supported boundary detail

`DrawingEngine.assess_boundary_detail(BoundaryDetailRequest(...))` is a
diagnostic-only operation for one explicitly paired, region-local closed
boundary. Use the retained material polyline from trace evidence as `source`
and the current canonical path as `candidate`; both are `PathGeometry`.
Supply the region ID, both revision identities and `reference_kind`:
`retained_polyline`, `cleaned_polyline`, or `declared_geometry`. The first two
require line-only source commands. The operation does not retrieve evidence,
guess hole/component matches, retrace, fit, or modify the drawing. Call it for
each boundary whose correspondence is known. Review / Edit now includes a
**Source-supported boundary detail** section with a scale selector, per-boundary
unsupported/missing turning, signed mismatch, alignment RMS and explicit
assessment status. The existing region filter applies here too. Optional
read-only location markers show up to six sampled differences per boundary
on the source and current candidate, not detected corners or plan addresses.
The detail section is shape-only; it is hidden on the fill track.

Download full detail JSON for every sample and its provenance/settings. Inline
report data retains a bounded hotspot summary; full results are written to a
content-addressed derived JSON file alongside each version. Report regeneration
recomputes from captured response/raw-trace evidence without replacing captures
or retracing. Missing, duplicate, merged, changed-boundary-set and unsupported
symbolic geometry cases stay explicitly unassessed; boundaries are never
paired by list position. Root trace regions already assembled from material
regions remain valid: the retained root polyline is their reference.

Live MCP clients can retrieve the same JSON via
`get_drawing_artifact(drawing_id, version, artifact="boundary_detail")`.
`structuredContent.boundary_detail` holds the data, with no widget or image
content. `region_ids` optionally restricts the selected targets. The operation
is on demand and does not run automatically during trace/refine.

The default scales are 1, 2, 4 and 8 source pixels, with at most 0.25-pixel
arc-length spacing. `units_per_pixel` explicitly registers canvas lengths to
source pixels (isotropic only); no translation/rotation/scale fit is performed.
Native path measurement makes sampling independent of command subdivision,
within numerical resolution. Winding is normalized, and cyclic spatial
least-squares aligns the starting position. Normalized arc length provides
the remaining pairing; this is not semantic feature matching. The alignment
RMS and localized source/candidate positions are included so a poor pairing
is visible rather than mistaken for authoritative feature correspondence.

At each scale, Gaussian-smoothed tangents yield signed local turning per pixel.
Positive differences in absolute turning are reported as **unsupported detail**;
negative differences are **missing detail**. A separate signed-turn mismatch
exposes changes in bend direction. Each integral is in radians over source arc
length, not a count of commands or detected corners. `diagnostics()` exports
the measurements and all local samples as JSON. Points are in source pixels.
These are descriptive signals, not validated perceptual scores: displaced
features and arc-length misregistration can contribute to both directions.

Smaller-scale differences may reflect anti-aliasing or compression; persistence
across scales is evidence to review, not proof of design intent. Keep boundary
distance, topology and fill evaluation separate. Exact curve subdivision or
degree elevation must not materially change this diagnostic. A more complex
candidate matching a complex source is not penalized. No acceptance threshold,
minimum-command objective or blended quality score is introduced.

Degenerate boundaries, insufficient scale support and exhausted sampling
budgets produce `not_assessed` with an explicit reason, never a zero error or
silent coarsening. Invalid settings/provenance are request errors. Primitive,
clone and compound-boundary resolution and persistent corner/inflection
matching remain separate work; callers must not invent a pairing.

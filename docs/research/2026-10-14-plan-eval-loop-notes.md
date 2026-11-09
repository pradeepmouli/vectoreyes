# Plan-driven evaluation loop — working notes

Status: empirical notes, not a completed skill or an approved UI removal plan.
Updated: 2026-10-14.

## Intended outcome

An image-capable agent produces a faithful editable vector using declarative
vectorEyes operations. The loop must evaluate actual published versions, not
equate successful execution, fewer commands, or passing tests with visual quality.
The original source polylines remain authoritative for refitting.

## Recorded case: angled radish

Input: `corpus/input/radish.png`.
Execution: deployed compiled runtime over local Streamable HTTP MCP; not a
source-checkout fallback. Baseline uses progressive Q/C, subpixel, sampling
colors 8, output colors 64, gradients allowed, refine none, initial stitch on,
corner rounding on, corner normalization off, gradient pretrace off, max canvas
size 1254. Exact options are in the baseline request artifact.

Evidence bundle:
`/Users/pmouli/.codex/visualizations/2026/07/09/019f491c-25bb-7402-948f-c85a418c3aa7/radish-deployed-plan-eval/`.

Live drawing: `drw_d17YYhUW3yE3IcvSdslqFwhK`. This is an ephemeral handle,
not a replayable fixture; a server restart invalidates it.

| Region | Baseline commands, excluding move | Baseline corners |
| --- | ---: | --- |
| Blue body/leaves, r1 | 30 | 7 point |
| Teal, r2 | 13 | 4 point |
| Orange, r3 | 9 | 4 point |
| Red tip, r4 | 15 | 2 point + 1 rounded |

The intended lower-region corner counts are 4/4/3. Existing counts already
match; fewer commands require side refits, not indiscriminate corner deletion.

### Trials and dispositions

1. Traced fresh v0; inspected review panel, source, segment/corner map, report,
   and raw source correspondence. Saved artifacts before issuing changes.
2. Replaced red rounded corner2 with a point at source command c51, the
   minimum-x source vertex in its rounded interval, radius 0. Published v0.0.
   This is a trial source-vertex choice, not a general rule that minimum-x is
   the correct designed apex. Broad silhouette is retained; full side cleanup
   is not complete. Executed plan is `v0.0/request.json`.
3. Proposed single cubic on teal's long side: rejected by fitting error guard.
4. Proposed red-left single Q and red-top single C with max_error 3 pixels:
   cubic rejected by fitting guard. Do not increase tolerance repeatedly just
   to force a desired command count.
5. Red-left single Q alone from v0.0 passed the fit guard but failed version
   publication: `changed paint resolution input revision mismatch`.
6. Recursive cubic side-fit batch failed with the same publication error.
7. Red-left single Q directly from v0 exceeded its fit guard; a recursive
   quadratic directly from v0 reached the same paint revision failure. Thus
   the publication blocker is not limited to the corner-declaration child.

No side-refit version was accepted. No exported SVG coordinates were edited.
v0.0 is a corner experiment, not a completed optimal SVG.

The observed publication guard is in
`drawing_replay._replay_regions_with_producers`, where a changed-paint operation
compares its recorded input revision with the geometry input revision.
The upstream mismatch cause has not yet been established. Do not bypass the
guard, suppress refill, or call a local fallback a successful MCP plan.

## Candidate skill protocol

### Agreed review structure

- Whole drawing first, then targeted region crops. Do not automatically emit
  a full comparison for every region on every iteration.
- Shape and fill are completely separate evaluation tracks, with independent
  acceptance decisions. Do not combine them into one score.
- Shape review uses neutral, consistent paint to compare boundaries, corners,
  curves, holes, seams, symmetry, clones and topology. Show source raster crops
  alongside retained source polylines; the polylines are evidence, not ground
  truth. Include parent/candidate and per-region command/anchor diagnostics.
- Fill review holds geometry and sampling support fixed for the comparison,
  then assesses source color, assigned flat/gradient paint, palette and interior
  residuals. Treat boundary antialiasing separately. A geometry change may
  require fill to be re-evaluated, but not attributed as a fill-only improvement.
- The final full-color composite is a sanity check, not a substitute for either
  independent acceptance decision.
- Overview identifies suspect regions; a focused crop carries the detailed
  source/parent/candidate comparison and exact region/segment/feature IDs.

### Agreed annotation and interaction requirements

Annotations are mandatory plan-authoring evidence, not optional decoration.
Each targeted review bundles a clean comparison, annotated map, and
machine-readable source correspondence, editability and diagnostics.

Use a layered, non-editing inspector with independently selectable overlays:

- Regions: boundaries and exact region IDs.
- Paths: individual contours and cutouts with path IDs.
- Commands: segment types and exact version-scoped command IDs.
- Features: point corners, rounded-corner entry/exit spans, inflections and
  junctions, with exact feature IDs.
- Source correspondence: the retained source-polyline interval supporting a
  selected command or feature, when correspondence is available.

Default overview: region labels. Selecting a region enables its command and
feature overlays. Selecting an item highlights it and exposes metadata and its
exact plan address; selection alone never changes geometry. Do not invent
correspondence for items whose mapping is unavailable or ambiguous.

Every interactive view must also be obtainable as an annotated image plus JSON
without manipulating UI toggles. Interactive and scripted review must consume
the same retained data and annotation generation, not separate implementations.
This approves the design direction/defaults, not implementation or UI removal.

Selection is synchronized across source, parent and candidate views, with the
same crop and zoom. Highlight counterparts only where retained correspondence
proves the relationship; otherwise show an explicit unmapped state. Command IDs
are version-scoped: matching text or index alone is not cross-version lineage.
Expose one-to-many/many-to-one mappings rather than forcing one counterpart.

### Agreed trial record

Present each comparison as a trial card containing the design hypothesis,
exact submitted plan and base version, before/after evidence, per-region
diagnostics, and an accept/reject/blocked disposition with a reason. Keep shape
and fill judgments independent within the record. Preserve failed requests and
errors even when no candidate version was published; do not fabricate an after
view for those trials. An execution failure is recorded as such, not as a
negative visual-quality judgment. The cards form the review history and the
future skill's machine-readable execution record, including branch ancestry.

### Agreed user-facing delivery

The default user report is a finished-artwork handoff, not the agent's working
inspector. Show a large clean preview of the current accepted SVG, a primary
Download SVG action, a brief change summary and remaining limitations, and
clear accepted/needs-review/incomplete status. Execution success alone must
never imply artwork acceptance. If no accepted version exists, show that
explicitly and label any baseline preview as provisional.

Provide Compare with original using synchronized source/result views and
separate shape/fill modes. A Review / Edit entry point opens the detailed
workspace: region selection, required annotation overlays, per-region findings,
trial history, settings, plans, diagnostics and downloadable evidence.

The handoff shows the accepted version, not the latest experimental candidate.
Keep execution status, shape verdict and fill verdict separate, with an explicit
accepted-version pointer. Successful trials require review before promotion;
rejected candidates remain inspectable without replacing the current best.
When blocked, retain the best accepted result and explain the unfinished work.
Entering Review / Edit does not itself mutate the accepted drawing; actual
edits use explicit plans and create candidates for review.

1. Record input hash, build identity, live schema and all resolved options.
2. Trace once; save v0 SVG, source, review panel, region map, segment/anchor map,
   raw correspondence, report, and provenance. Reuse retained evidence.
3. Write a region-specific design hypothesis: real corners, corner extents,
   sides, plausible inflections, paint and relationships. Explain every edit.
4. Correct corner declarations before broad side refits. Use exact source IDs;
   do not guess source spans from compressed command indices.
5. Submit one bounded hypothesis per trial. Fetch fresh IDs after changes.
   Start with a single operation when diagnosing failures; batch only after
   addressing and interaction behavior is understood.
6. Save the exact submitted plan, base version, response or error, elapsed
   time, resolved settings, and expected versus observed effect for EVERY
   trial, including rejected requests. The initial scratch runner did not
   persist failed responses automatically; this is a required script upgrade.
7. Compare source, baseline, parent and child at full view and regional zoom.
   Check paint separately from silhouette; check seams, topology and corner
   ownership. Count commands and anchors per region, not only globally.
8. Accept only when fidelity is preserved or improved and complexity is
   justified. Branch from the better parent when a proposal regresses.
9. Export the exact accepted SVG and plan history; distinguish accepted,
   rejected, and blocked trials. Report remaining limitations honestly.

### Evaluation cautions

- Single-curve hypotheses are useful, not acceptance criteria by themselves.
  Inflections do not uniquely determine the necessary segment count: a
  non-inflecting arc can still require multiple Q/C segments at a given error.
- Distinguish geometric nearest-curve error from error at assigned sample
  parameters. Poor parameterization can reject an otherwise plausible curve;
  inspect the fitter before concluding that extra commands are necessary.
- Radius 0 is an explicit point declaration, not proof that adjacent sides
  have clean tangents or that the source apex was chosen correctly.
- A fit-guard rejection, schema/address error, replay failure, and transport
  failure require different responses. Classify them before changing settings.
- This case also exposed missing aggregate corner/command counts in the edited
  region's diagnostics, even though command and anchor objects are present.
  A future runner should flag the omission and label derived counts explicitly.

## MCP UI versus skill scripts

Working recommendation: make the evaluation workflow headless-first; retain
the UI as an optional thin viewer/export surface, not a prerequisite.

| Capability | Appropriate owner |
| --- | --- |
| Trace, source evidence, plan validation/replay, immutable versions | Engine/service |
| Schema discovery, trial orchestration, artifact persistence, run manifest | Skill scripts over the same public contracts |
| Visual judgment and proposed declarations | Image-capable agent, with human review when needed |
| Source/parent/child/diff maps and regional zooms | Shared artifact generators, usable without a widget |
| In-chat inspection, settings editing, version navigation, download | Optional MCP UI |

The successful trace, corner plan and artifact retrieval here required no
widget interaction. Scripts plus saved PNG/SVG/JSON and a linked static report
are sufficient for this local agent's mechanics. An agent still needs image
inspection; numeric scoring alone does not replace it.

Scripts do not replace remote access: ChatGPT/mobile hosts may lack local
execution, filesystem access, persistent processes, or an installed skill.
MCP remains a useful transport. Packaged-skill availability/execution must be
verified per host, not assumed from this local run.

The present failure is inside replay/paint provenance, so removing the UI or
wrapping calls in a script would not fix it. Conversely, artifact retrieval
must remain data-only and should not open extra widgets.

Avoid a second implementation: scripts call the same typed engine/service
operations, use generated schemas, and reuse artifact generation. They should
not reproduce fit logic, corner detection, fill resolution, or plan semantics.
Saved plans alone are not complete replay packages: durable source evidence,
settings, build identity, and rebinding/version identity need an explicit
export/import contract if sessions must survive restarts.

### Before simplifying the UI

- Finish one complete radish loop, including acceptance and rollback.
- Repeat with a geometric logo and a compound/cutout example.
- Verify end-to-end operation and actual downloads in target ChatGPT/mobile
  hosts without relying on local file paths.
- Compare artifact payload cost and review usability with and without widget
  metadata. Keep the same artifact generator and renderer in both routes.
- Then decide which controls to retain. No UI removal is authorized or
  implemented by these notes.

## Next work

Fix and verify the paint input revision mismatch, then resume this live
evaluation. Turn the proven protocol and recorded cases into the packaged
skill only after the loop completes; preserve failures as diagnostic examples.

# Designer-quality reconstruction: implementation plan

Date: 2026-09-07
Status: Stabilization, canonical corner ingress, explicit Python fitting, merged
source evidence, internal corner recipe replay, canonical value ownership, and
retained self-symmetry/pure export and topology ingress green (812 tests).
Canonical version-store integration and later slices remain pending.

### Current-state correction — 2026-10-16

The historical status above is not the current implementation boundary.
`DrawingVersion` now retains a canonical `Drawing`, effective recipe, submitted
edits, and replay receipt; MCP trace/refinement uses `propose_effective_recipe`.
Canonical storage is no longer an entirely pending prerequisite. Remaining
Slice 3 work is concrete shared engine execution and migration of transient
`VectorRegion` algorithms, not another authoritative store. Native compilation
and the packaged evaluation workflow have also advanced beyond the progress
notes below; neither alone completes the unified quality/fill or holdout gates.

## Goal

An LLM agent with an image-generation tool can generate artwork, pass it to
VectorMark, inspect the result, and iteratively produce a faithful, practical-to-edit
SVG that replaces the raster as the user's source of truth.

Initial acceptance domain: logos, icons, and flat/gradient illustrations.
Photographic texture, unrestricted effects, and editable text reconstruction are
not implicit promises. Unsupported artwork must be identified, not silently
flattened or embedded as raster and reported as successful vector reconstruction.

This plan completes and hardens the existing workflow. It does not replace its
public action vocabulary or move image generation into VectorMark.

## Non-negotiable invariants

1. One immutable canonical `Drawing` per version, containing `DrawingRegion`s.
   Engine evidence is private; it is not a second authoritative scene.
2. Rendering serializes retained geometry and fills. It does not fit, simplify,
   stitch, invent underlays, or otherwise repair the drawing.
3. `refine="none"` returns v0; `refine="auto"` retains v0 and its optimized v1.
   Plans may branch from either. No partially executed plan becomes a version.
4. Geometry checks use cached trace evidence and cumulative v0 comparisons.
   Appearance checks use the original image in the same coordinate system.
5. Geometry changes invalidate affected fill fits. Refitting paint cannot change
   geometry. Geometry-guide colors are labels, not final paint.
6. Corners originate in pre-fit evidence. Boundary anchors and the commands inside
   a corner have distinct ownership. No legacy post-fit corner inference fallback.
7. Self-symmetry union/materialization happens as a retained operation. Ordinary
   clones may remain symbolic and have independent fills.
8. All public schemas derive from the canonical typed models. Examples must
   validate against those schemas. No compatibility shims for unreleased APIs.
9. Reuse cached trace output for optimizer, rendering, and diagnostic changes.
   Invalidate caches when source, preprocessing, trace settings, or engine behavior
   changes; never compare stale artifacts as if they came from the current run.
10. Each version retains its effective reconstruction recipe (plan, declarations,
    resolved settings, and evidence references) alongside the generated drawing.
    Refinement revises that recipe and replays it; parent geometry is comparison
    output, not an implicit preservation constraint.

### V0 quality boundary — 2026-09-09

V0 is a faithful, inspectable reconstruction baseline, not a promise of finished
designer-quality geometry. Residual kinks or excessive commands may remain for
agent-directed correction when source evidence, addressable regions/features,
and honest diagnostics make those corrections possible. Do not indefinitely
tune automatic fitting to eliminate every visible imperfection before enabling
the correction/replay workflow. Invalid geometry, lost topology, inconsistent
corner provenance, or inaccessible edit targets remain correctness defects;
they are not excused as expected agent cleanup. Final-artifact quality is judged
after the iterative workflow, separately from v0 acceptance.

## Delivery sequence

| Slice | Deliverable | Depends on | Exit condition |
| --- | --- | --- | --- |
| 1 | Reproducible environment and current contract baseline | — | Fresh install and current MCP workflow tests pass |
| 2 | Correct corner ownership and provenance | 1 | Normalization cannot consume neighboring sides or erase metadata |
| 3 | Canonical engine execution and pure export | 1–2 | Plans, optimizer, reports, and SVG describe the same drawing |
| 4 | Shared quality assessment and accountable acceptance | 3 | Internal damage and cumulative drift are detectable and reported |
| 5 | Explicit fill fitting and completion status | 3–4 | Unsupported paint cannot masquerade as faithful output |
| 6 | Agent workflow, artifacts, and lifecycle hardening | 4–5 | Real MCP client can inspect, branch, refine, and retrieve reliably |
| 7 | Autonomous evaluation and release criteria | 6 | Holdout tasks converge without repository access or handholding |
| 8 | Profile-led optimization/native engine | 7 baseline | Measured improvement with contract and quality parity |

Deliver each slice independently. Do not combine a native port, fitter experiment,
and acceptance-rule change in the same review. Slice 3 can be several small PRs.

## Slice 1 — Reproducible baseline and contract reconciliation

Primary files: `pyproject.toml`, MCP launch/test configuration,
`drawing_actions.py`, `drawing_plan.py`, `mcp_server.py`, current contract tests,
`docs/mcp.md`, and the existing agent-guided drawing design.

Tasks:

- Upgrade dependencies before stabilizing, as approved. Establish a tested,
  locked resolution and migrate the MCP API as a separately verified change.
  Review minimum dependency versions against methods used by the code.
- Make test subprocesses use a dedicated environment with the server extra;
  tests must not recreate the developer's `.venv` through an implicit `uv run`.
- Migrate tests from retired `set_geometry`/`clone` and old validator signatures
  to `update_geometry`/`assign_clone` and the current canonical contract.
- Reproduce the ellipse-fitting exception from geometry-guide retracing. Reject
  numerically invalid candidates with a structured reason; do not let an optional
  primitive candidate abort the drawing or broadly swallow unrelated errors.
- Reconcile the spec with current types: options, action settings, v0/v1 lineage,
  branch naming, error codes, and drawing-ID access across transport reconnects.
- Keep the intended 30-minute idle live-drawing model. Do not reintroduce binding
  to transient MCP transport sessions. Persistent storage is a separate product
  decision, not a prerequisite for this local MVP.

Verification:

- Fresh install imports and starts the MCP server.
- Real stdio and HTTP client smoke: trace none; trace auto; valid refinement;
  geometry-guide retrace; retrieve SVG/report/preview; branch from v0 and v1.
- Both input and output schemas validate actual requests and responses.
- Current-contract suite is green. Classify every existing failure instead of
  deleting assertions merely to reach green.

Review baseline: 41 passed / 51 failed in the four focused plan/refine/server/state
test modules after temporarily constraining MCP. These are review observations,
not a target pass count or a current full-suite certification.

### Progress — dependency upgrade, 2026-09-07

- Upgraded the lockfile, including MCP 2.2.0, NumPy 2.5.3, SciPy 1.18.1,
  Pillow 12.3.0, resvg-py 0.5.0, and pytest 9.1.1. Added the direct Pydantic
  dependency used by the canonical models and retained explicit HTTPX for the
  existing URL resolver; MCP 2 uses a different HTTP client package internally.
- Migrated server imports, transport startup, package-version advertisement,
  and Python SDK field access. HTTP requests retain sufficient capacity for
  the supported 20 MiB decoded image plus base64/envelope overhead.
- Verification in an isolated Python 3.13 environment: 42 focused MCP,
  image-resolution, and drawing-state checks pass, with one known
  geometry-guide/ellipse failure deselected. Includes real stdio and HTTP,
  current and legacy protocol modes, reconnects, v0/v1 branches, artifact
  retrieval, image blocks, and uploads above the SDK's default 4 MiB limit.
- `uv pip check` passes; `pip-audit` reports no known vulnerabilities in installed
  dependencies (the editable local project is excluded). Wheel and sdist build.
- A fresh wheel installation outside the checkout exposes all five tools with
  output schemas and completes a v0 trace. The wheel currently lacks the built
  app HTML and serves the build-required placeholder; packaging the actual UI
  is a separate existing gap, not a successful app-rendering check.
- Remaining: numerically invalid ellipse candidate handling, broader stale
  action-contract tests, and public documentation/schema reconciliation.
  The updated four-module contract baseline is 47 passed / 47 failed; the
  broader suite is not green. Full test collection succeeds.
  Detected symmetry diagnostics still use `theta/cx/cy`, while explicit symmetry
  inputs use `point/angle_degrees`; current tests translate explicitly, but agent
  output/input consistency still needs attention.
- No tracing/fitting algorithm changes, deployed-server restart, UI dependency
  upgrade, or corpus regeneration belongs to this dependency change.

### Progress — stabilization and corpus references, 2026-09-08

- NumPy 2.5's always-complex `eig` result is incompatible with scikit-image
  0.26's ellipse estimator, including ordinary valid ellipses. Constrain NumPy
  to `<2.5` (locked to 2.4.6) until a supported estimator handles that contract;
  this supersedes the 2.5.3 selection above without reverting the other upgrades.
  See the [NumPy release note](https://numpy.org/doc/stable/release/2.5.0-notes.html).
- Optional ellipse estimation rejects the specific complex-modulo numerical
  failure and non-finite/non-real/non-positive parameters with structured log
  reasons. Unrelated estimator exceptions still propagate. Valid ellipse
  recognition and geometry-guide retracing are regression checks.
- Curated target SVGs and structural manifests added to Slice 7. Targets are
  reviewed independently, separate from generated snapshots; no corpus artwork
  or baseline has been generated or replaced by this planning update.

### Progress — contract reconciliation, 2026-09-08

- Current plan/refine/state/MCP/primitive-estimation baseline passed 101 checks
  in the isolated upgraded environment before the symmetry-contract changes.
- Detected self, intrinsic, and mirror-pair axes now serialize through the same
  typed `point` / `angle_degrees` value accepted by explicit actions. Direct
  refinement and MCP checks submit detected self axes unchanged; horizontal,
  vertical, oblique, and negative directions are covered. No retired radians
  payload or client-side translation is retained.
- Fixed loss of exact symmetry diagnostics when no fitted reconstruction is
  accepted: retain the unchanged path with intrinsic symmetry only when the
  reflected residual is effectively zero. Approximate rejected reconstructions
  are not promoted. Updated the stale primitive assertion to verify native
  geometry/fill preservation alongside the intrinsic diagnostic. All 25 focused
  symmetry checks pass; these changes do not adjust corner or fit thresholds.
- Replaced `docs/mcp.md`'s retired logo-tool instructions with the five current
  drawing tools, ImageRef examples, trace/guide settings, version branching,
  live-ID lifetime, and deployment boundaries. Its JSON examples validate
  against current types, and live tool discovery reports five output schemas.
- Expanded verification passes: 161 checks across plan/refine/state/MCP,
  primitive estimation, symmetry, optimizer framework, and image resolution.
  There are 24 existing NumPy/SciPy deprecation warnings; `git diff --check`
  passes. No tests in this focused run remain failing or deselected.
- Collected 19 downloaded brand reference SVGs, all explicitly user-approved,
  plus the local Daikonic draft in `corpus/reference/`. Raster registration and
  automated reference scoring remain pending; corpus tracing outputs unchanged.
- Slice 1 is not yet complete: standalone wheel UI packaging remains open,
  and focused checks do not certify the full repository test suite. Continue
  classifying failures before declaring the broader baseline stable.

### Progress — packaged UI and full-suite audit, 2026-09-08

- Closed the standalone-widget packaging gap above. Wheels load built HTML via
  package resources; editable installs retain the local frontend build path.
  Release automation builds the UI and checks installed-wheel resource bytes
  outside the checkout. Three focused resource-loading checks pass.
- Built an sdist and then its wheel, installed the wheel in a fresh environment,
  and retrieved the actual widget through stdio MCP from `/tmp`. Both archives
  contain the exact built HTML. A build without that asset fails as intended.
- Restricted sdist contents to explicit project files after discovering local
  graph/worktree data in the default archive. The rebuilt archive is 1,157,150
  bytes with 204 entries and contains no local graph, worktree, or node_modules
  directories. No distribution was uploaded or deployed.
- Full-suite audit: **610 passed, 17 failed**, 627 collected, no errors/skips,
  963 warnings. This was collected before the three new widget tests, which
  were run separately. See the [failure inventory](2026-09-08-stabilization-audit.md).
- Slice 1 remains open. Investigate cutout/annulus preservation first, then
  optimizer contract expectations and golden/report differences. These are
  observed failures, not yet established root causes; do not refresh goldens
  or remove assertions simply to make the suite pass.

### Progress — closed-seam merge correctness, 2026-09-08

- Fixed the material merge that erased a white counter inside a navy frame.
  A closed perimeter was incorrectly judged by an open-seam length/span noise
  heuristic. Closure now makes that inference inconclusive; source-color
  continuity remains available for gradient-band merging. High-color tracing
  reuses the same decision rather than a separate heuristic.
- Six additional checks cover hard closed boundaries, argument ordering,
  continuous-color closed boundaries, and noisy open boundaries. The focused
  run passes 18 tests; the full rerun is **620 passed / 16 failed**, 636 total,
  with the original hole-rendering failure resolved and no new failures.
  Evidence: `/tmp/vectormark-cutout-audit/{pytest.log,junit.xml}`.
- Annulus structural recovery and simplify boundary drift are distinct pending
  issues; their cutouts are not absent in the observed outputs. See the audit
  follow-up for measured evidence. No expected SVGs or corpus entries changed.

### Progress — cutout-aware simplification acceptance, 2026-09-09

- Compound-path simplification now checks each corresponding boundary against
  the displacement budget (`max_boundary_error`, or `max_error` by default),
  in addition to residual area. An unsafe refit falls back to safe local command
  normalization, then to the original boundary coordinates.
- Retain small cutouts and source subpath order. Validate component/hole counts,
  pairwise containment/intersection, and filled topology under the fill rule.
  Explicit covered-cutout removal remains a separate intentional operation.
- Single-boundary fitting retains its existing displacement acceptance policy.
  No fitting thresholds, corpus entries, or golden SVGs were changed.
- **30 focused tests pass.** The existing Apple Music compound fixture now stays
  within the 1 px boundary budget: exterior displacement **0.64 px**, cutout
  **0.58 px**, versus **1.90 px / 2.10 px** for the rejected refit. These are
  sampled boundary estimates, not analytic Hausdorff bounds. Weighted command
  cost falls from 115 to 82 while the command count remains 62.
- Cached-fixture SVG comparison:
  `/tmp/vectormark-cutout-boundary-review/index.html`. This is not a corpus retrace.
- Final full suite: **652 passed / 15 failed**, 667 total, no errors or skips.
  Compared with the preceding local-corner audit, the compound simplify failure
  is resolved and there are no new failing identities. Evidence:
  `/tmp/vectormark-cutout-boundary-final-{pytest.log,junit.xml}`.
  Remaining stabilization failures still need reconciliation before Slice 1
  can close.

### Progress — integration and retained-artwork stabilization, 2026-09-09

- Reconciled optimizer representation assertions while retaining geometry/paint
  checks; prevented compound splitting from dropping filtered tiny counters.
- Recovered annuli while retaining cavity artwork by source-region identity.
  A hard retained-pixel check now precedes tolerant appearance scoring; independent
  review caught and verified correction of the thin-overpaint case.
- Reviewed full suite: **665 passed / 5 failed**, 670 total, in 95.91 seconds.
  Remaining: four historical byte snapshots requiring fidelity/structure review
  and a tiny-disk primitive acceptance-policy mismatch. No new failing identities,
  golden replacements, corpus regeneration, commits, or deployment in this batch.
- Evidence and details are in the stabilization audit and
  `/tmp/vectormark-retained-annulus-reviewed-{pytest.log,junit.xml}`.

## Slice 2 — Corner ownership and provenance

### Stabilization handoff — semantic regression contracts, 2026-09-09

The historical Candidate/Fill byte snapshots now check renderer-normalized
appearance, visible paint ownership, and component/hole topology, with source
drift measured against each historical baseline. Equivalent primitive/path,
clone/expanded, CSS paint and gradient-stop encodings are covered; deterministic
serialization remains a separate check. Independent review approved the change
after correction of the negative controls and hole counting. **23 focused checks
pass; no production geometry or historical SVG references changed in this task.**
Subsequent integration exposed anti-aliased navy edges being assigned to teal by
nearest-color ownership. Requiring actual color support before paint-topology
measurement fixes that false positive; a real small teal region remains visible
to the check. The updated focused run passes **24 checks**, without changing
appearance budgets.
The tiny-disk primitive/framework acceptance mismatch remains independently
reproducible. These focused results are not a new full-suite certification.

Primary files: `optimizer/corners.py`, `fit.py`, `drawing_trace.py`, and canonical
path/anchor types.

Tasks:

- Represent a rounded corner with entry and exit anchors plus its directed set
  of owned commands. The command arriving at the entry and the command leaving
  the exit are not corner curves merely because they touch a boundary anchor.
- Handle wraparound at a closed path's start explicitly; do not use numeric
  min/max command indices as a substitute for directed span order.
- Preserve declared corners when normalization makes no change. After a rewrite,
  transfer provenance to replacement commands and preserve untouched anchors.
- Restrict normalization to declared spans and preserve their endpoints. Missing
  or ambiguous provenance means an explicit skip, not new inferred corners.
- Keep point-versus-rounded classification based on pre-fit evidence and the
  anti-aliasing margin. Unknown radius is not evidence of a point corner.
- Report logical corner count separately from corner-owned command count.

Verification:

- A rounded corner followed or preceded by Q/C cannot absorb that adjacent side.
- No-op normalization preserves all provenance; repeated normalization is stable
  within documented serialization precision.
- Closed-path wraparound and touching corner spans behave consistently.
- Cached radish, Apple Music, and Vbird traces show before/after per-region
  overlays; inspect the orange lower corner and note stem-to-band junction.

Do not change general corner detection thresholds in this slice. Fix ownership
first so subsequent detector experiments can be evaluated independently.

### Progress — corner ownership, 2026-09-08

- One shared pre-fit span binder now separates endpoint-anchor membership from
  directed command ownership. Adjacent side commands are excluded; point corners
  own no drawable commands. Original corner IDs and subpath scope survive
  rebinding, wraparound, and normalization, including no-op normalization.
- Normalization preserves declared endpoints and untouched command precision.
  Missing, stale, ambiguous, overlapping, or point-crossing spans are explicitly
  skipped rather than inferred from fitted tangents. Logical corner counts are
  reported separately from owned-command counts.
- All 13 focused ownership checks pass. The full-suite run collected before the
  final skip-stability check reports **632 passed / 16 failed**; failure identities
  exactly match the preceding audit, with no added failures. The final focused
  rerun includes the additional check. Evidence:
  `/tmp/vectormark-corner-final-{pytest.log,junit.xml}`.
- Historical cached radish, Apple Music, and Vbird paths have a linked-SVG
  comparison at `/tmp/vectormark-corner-ownership-review/index.html`. All seven
  reviewed region records retain their bound corner IDs. These comparisons use
  the recorded paths matching their provenance, not newly retraced corpus output.
- Detector thresholds remain unchanged. Over-wide or unresolved pre-fit corner
  declarations remain visible in the report; this change does not claim to fix
  their detection or complete the canonical typed-anchor migration in Slice 3.
  Corpus caches, approved references, and expected SVGs were not changed.

### Follow-up — localized pre-fit declarations, 2026-09-09

- User review of the cached overlays exposed two additional problems: all gaps
  between protected lines were declared as corners, and same-direction turns
  could join two corners across an ordinary curved side.
- Removed the line-gap declaration shortcut. Supported corner neighborhoods now
  split at weak-turn valleys and require each retained group to be compact
  relative to its exterior support. The caller's turn threshold caps the valley
  cutoff so a sharper neighbor does not erase a qualifying shallow corner.
  Existing source-corner and recovery detection thresholds are unchanged.
- Nine locality regressions cover line-gap metadata, unequal-strength corners,
  compact fillets versus broad arcs, and both winding directions. Together with
  ownership, fitter, and contour checks, **45 focused tests pass**.
- Final full-suite verification: **642 passed / 16 failed** in 97.40s. Failure
  identities exactly match the preceding corner-ownership audit; no new
  failures. Evidence: `/tmp/vectormark-local-corners-final-{pytest.log,junit.xml}`.
  `git diff --check` passes. Existing stabilization failures remain unresolved.
- `/tmp/vectormark-local-corners-review/index.html` compares identical outlines
  reconstructed from cached region pixels: polyline evidence, previous fit and
  ownership, updated fit and ownership, and updated corner normalization. This
  is not a fresh source trace or a full-optimizer/corpus rerun.
- The teal band's longest declared span decreases from 131.1px to 13.1px.
  Apple Music's longest span decreases from 270.3px to 40.0px, without changing
  its fitted outline. Orange-band and red-tip fits/declarations are unchanged
  on this evidence. Extra local declarations (including the teal upper-left)
  remain visible; perfect logical corner counts are not claimed.

## Slice 3 — One drawing model, one engine, pure export

### Agreed operation semantics — 2026-10-16

Apply the [operation purpose/proposal contract](../specs/2026-07-10-agent-guided-drawing-mcp-design.md#operation-purpose-and-proposal-semantics--2026-10-16)
to this existing cleanup, without a separate implementation plan or duplicate
action vocabulary. Operations serve reconstruction accuracy, design fidelity,
or both. Purpose is independent of intent origin and deterministic versus
exploratory execution. Design-fidelity inputs/outputs may vary; defaults are
not a promise of identical proposals.

The next bounded checkpoint in this slice is:

- [x] Carry resolved purpose, intent origin, and execution behavior through the
  existing action/engine contracts and per-operation diagnostics.
- [x] Extend the existing recipe/replay receipt to bind selected resolved
  outcomes to evidence, intent/settings, and upstream dependencies. Retain one
  canonical drawing; do not build a parallel candidate scene/store.
- [x] Keep reproduction separate from deliberate re-proposal. Reproduction
  cannot invoke an exploratory solver; re-proposal cannot mutate its parent.
  Invalidated bindings must fail explicitly rather than silently retarget.
- [x] Verify with a controlled proposer that returns different valid candidates
  on repeated calls: committed render/reproduction stays identical without
  calling it again, while explicit re-proposal can publish a different child.
  Cover sibling isolation, rejected proposals, changed evidence/settings,
  upstream edits, and failure before atomic publication.
- [x] Route existing deterministic actions through that same boundary before
  adding exploratory solvers. Do not change fitter defaults or add randomness
  merely to demonstrate the contract.

Slice 4 must assess actual proposals using independent raster, design-constraint,
topology, and complexity results. Any accuracy trade-off is explicit; design
fidelity is not an implicit tolerance override. Commitment and designer approval
remain different states. Existing deployed behavior is unchanged by this design.

Primary files: `drawing_engine.py`, canonical drawing/geometry modules,
`drawing_refine.py`, `drawing_state.py`, `optimizer/framework.py`, optimizer passes,
and SVG emission.

#### Implementation progress — 2026-10-16

The first foundation is implemented, not the entire checkpoint. Existing action
settings now carry typed `purpose` and `execution` values through plan/action
override resolution and retained public-operation diagnostics. Numerical fitting
budgets are unchanged. `propose_effective_recipe` explicitly generates a result;
`reproduce_effective_recipe` validates and returns the selected canonical drawing
without invoking geometry generation. The existing replay receipt now also binds
paint evidence, and publication shares that validation rather than duplicating it.

Temporary contract checks exercise a controlled varying proposer, exact retained
rendering, distinct sibling outcomes, dependency mismatches, failed publication
without child-number consumption, and public refinement metadata. The focused
existing suite passes (118 tests). The full source suite passes (895 tests,
951 warnings), the four temporary contract checks pass, and fresh scoped review
found no actionable findings. No compilation or deployment is part of this
checkpoint.

Follow-up implemented: captured agent batches and corner applications retain
resolved settings/origin; the common receipt projection includes automatic,
direct, corner, and downstream-batch operations. Low-level optimizer batches
inherit their owning recipe settings unless the drawing action supplies resolved
overrides. Corner fitting honors those overrides, and downstream rebasing
preserves them. Nine temporary contract checks and 64 focused tests pass; fresh
scoped review found no actionable findings. The full source suite passes again
(895 tests, 951 warnings). User authorized a checkpoint commit including required
engine/test prerequisites; unrelated UI, plugin, corpus and local tooling remain
outside that commit.

Selected-fit reuse is now implemented for primitive fits, whole-path fits and
direct path edits. Each operation retains one chosen canonical geometry value,
not a second drawing tree. Its dependency identity binds evidence/engine/paint,
intent, resolved settings, target bindings, upstream recipe and working input.
Extending a recipe reuses that result without fitting; changed dependencies fail
explicitly rather than silently regenerate. A fresh fit declaration deliberately
selects a fresh outcome. The dependency scope is conservative: unrelated upstream
changes can also invalidate reuse until target-local dependency analysis exists.

Canonical corner diagnostics and ordinary symmetry/topology preparation are
preserved across reuse. Fifteen retained contract tests pass, including controlled
varying proposals, path correspondence, stale settings/paint/upstream inputs and
the two review-driven metadata/materialization regressions. Post-fix full-suite
verification passes: 910 tests, 951 warnings. No fitting defaults or error budgets
changed. This checkpoint is source-tested; it is not a native rebuild/deployment
or a visual-quality acceptance.

#### Concrete operation entry point — 2026-10-17

`PythonDrawingEngine` now implements the existing typed engine boundary. It
owns source tracing, material-proxy/palette resolution, root fitting, initial
stitching, automatic refinement, and submitted plan/action execution. MCP keeps
image resolution/preparation, rendering, and atomic version publication. Source
and geometry-guide tracing use the same engine operations while preserving
their distinct geometry-input and original-paint policies.

The implementation delegates to the existing trace, fit, stitch and recipe
executors; it does not add another fitting algorithm, scene store, or action
vocabulary. Default corner-recipe fitting also enters through the engine.
Direct path actions require retained evidence and the caller's selected base
version, so correspondence validation cannot be bypassed by supplying only a
displayed drawing. Plans validate the existing receipt before execution.

| Entry point | Input | Output |
| --- | --- | --- |
| `trace_drawing` | `TraceDrawingRequest`: packed pixels, trace/output settings, initial-stitch control | `TraceDrawingResult`: canonical v0, evidence/recipe/receipt, diagnostics |
| `fit_path` | `FitPathRequest`: source line path, declarations and fit settings | `FitPathResult`: fitted path, source correspondence, diagnostics |
| `detect_path_features` | `DetectPathFeaturesRequest`: undecorated line path, existing trace options, whole-canvas extent, optional forced source command IDs | `DetectPathFeaturesResult`: unchanged commands with typed corners, supplementary fitting boundaries, explicit binding diagnostics |
| `apply_path_action` | `PathActionRequest`: drawing, evidence, base version, region, path action, settings | `PathActionResult`: updated single path and affected commands |
| `apply_action` | `DrawingActionRequest`: drawing, evidence, base version, action, settings | `DrawingActionResult`: canonical scene, updated receipts, region changes |
| `apply_plan` | Drawing, evidence and existing `DrawingPlan` | `PythonPlanResult`: drawing-action result plus resolved submitted corner edits |
| `refine_automatically` | Drawing, evidence, corner-normalization toggle and selectable passes | `DrawingActionResult`: canonical scene and retained automatic recipe/receipt |

This is an ownership/consolidation checkpoint, not completion of every internal
adapter migration: temporary `VectorRegion` execution adapters and automatic
feature discovery remain inside existing kernels. Automatic and explicit fits
continue sharing the established numeric fitting kernel; this checkpoint does
not replace automatic feature discovery with a new declaration algorithm or
change fitter settings/defaults. Native compilation and deployment are separate.

Verification: 19 engine contract tests pass, including exact v0 parity for
pixel/subpixel tracing, all three fit strategies and stitching on/off. Fresh
review found two source/guide policy differences; both were reproduced and
corrected with regression tests. Full source suite after those fixes and the
full-tree change-reporting correction: **929 passed, 951 warnings** in 117.01s.
No native rebuild, deployment or visual-quality acceptance is claimed.

Still pending: generic outcome reuse beyond fitted geometry, and removal of
temporary `VectorRegion` execution adapters inside the shared engine. These
checkpoints do not cache every corner, automatic or relationship result.

#### Independent pre-fit discovery — 2026-10-17

Automatic fitting now explicitly orchestrates discovery, the shared numeric
fitter, and evidence binding. Discovery returns an immutable source-ring and
boundary value and does not fit commands. The same detector is available through
`DrawingEngine.detect_path_features`; its canonical input/output retain the
original line commands and addresses. Existing source binding is shared with
automatic evidence rather than duplicated. Existing declarations are rejected
at detection ingress; explicit fitting remains the authoritative consumer of
declarations, including an intentionally empty declaration set.

`FitPathRequest.fit_boundary_command_ids` completes that handoff. `None` retains
existing protected-line discovery; a supplied tuple carries the selected
supplementary boundaries and disables fresh line discovery. An empty tuple is
authoritative. Valid corner boundaries are derived from the current anchors,
not duplicated in supplementary cuts, so deleting an anchor does not silently
leave its semantic partition protected. Unknown source IDs and excluded-boundary
conflicts fail explicitly, including the coincident move/closing-line alias.

Ordinary automatic outputs and receipt identities remain covered by ten exact
pre-refactor golden cases across quadratic/cubic/progressive fitting, rounded
and stepped source contours, and canvas scaling. No detector thresholds or
fitter defaults changed. One narrow correctness fix is intentional: recovery
cleanup can no longer delete an explicitly forced corner. Command-addressed
forcing also uses exact source indices instead of coordinate-based retargeting.

This exposes an engine capability, not another MCP plan vocabulary. Remaining
canonical execution-adapter migration is still open; native rebuild/deployment
and visual-quality acceptance are not part of this source checkpoint.

Verification: 85 focused checks pass, including ordinary-output golden receipts,
discovery without fitting, declaration removal, exact forced addresses and
closing-line aliases. Fresh review's forced-corner defect and the semantic-cut
handoff clarification both have RED-to-GREEN regressions. Final full source
suite: **950 passed, 951 warnings** in 127.57s. No deferred review minors.

#### Canonical primitive fitting kernel — 2026-10-17

The public primitive action now delegates to `fit_primitive_geometry`, a numeric
kernel consuming retained raster evidence and returning typed `PrimitiveGeometry`.
It no longer takes a `VectorRegion` or returns an untyped `Shape`. The five public
primitive kinds retain their existing fitting rules; unsupported private cap and
trapezoid branches are removed rather than preserved as unreleased compatibility.

The action executor still projects this result into its temporary region forest;
this checkpoint does not remove that scene adapter or migrate the remaining
operations. `DrawingEngine` remains the public operation entry point. Remaining
work includes canonical path-action execution and removal of the replay adapters.

Verification: **116 focused tests** and **964 full-suite tests** pass. The full
suite reports 951 warnings in 219.15s. Fresh scoped review found no actionable
defects; no native rebuild, deployment or visual-quality acceptance is claimed.

#### Canonical region-local actions — 2026-10-17

`fit_region_primitive` now accepts a canonical `DrawingRegion`, the existing
`FitPrimitiveAction`, retained raster evidence and resolved `ActionSettings`.
It calls the shared numeric kernel and returns a replacement region, preserving
identity, paint, z-order and source lineage. `set_region_fill` similarly accepts
the existing `SetFillAction` and changes only canonical paint, reusing the shared
fill deserializer. Both reject branch targets and mismatched region addresses.

Plans reach these implementations through the existing `DrawingEngine` entry
point. The working-region bridge remains temporarily responsible for projection;
it does not duplicate action semantics. Its first action reuses the validated
canonical view, while subsequent actions see the updated working scene. Shared
topology preparation, paint resolution and selected-outcome retention remain in
place. Explicit paint adopts canonical lowercase hex serialization; colors and
numeric fitting rules are unchanged. Remaining actions migrate incrementally to
this same separation; this checkpoint does not complete scene-adapter removal.

Review also exposed an existing radial-paint bridge gap: noncentral focal points
were accepted by the public action but dropped before output. The assignment and
working-to-canonical conversion now retain them, and the working SVG renderer
emits `fx`/`fy` as the canonical renderer already does. Two regressions were
verified RED-to-GREEN; gradients with no focal point retain their existing output.

Verification: **148 focused tests** pass. Final post-fix full source suite:
**976 passed, 951 warnings** in 116.30s. Scoped review found no newly introduced
defects; its focal-point observation was fixed with the regressions above. No
native rebuild, deployment or visual-quality acceptance is claimed.

#### Canonical drawing ordering — 2026-10-17

`set_drawing_z_order` accepts the canonical `Drawing` and existing
`SetZOrderAction`. It uses shared target expansion, requires complete leaf
coverage, and preserves each selected group's relative paint order with stable
tree-order ties. It returns a replacement drawing without changing geometry,
paint, annotations, lineage or tree structure. The temporary executor projects
only resulting leaf z values; its duplicate ordering algorithm is removed.

Whole-region path fitting remains on its existing implementation until its
result contract explicitly represents component splits and cutouts. The migration
must not silently restrict it to one fitted leaf or change topology behavior.

Verification: **125 focused tests** pass. Full source suite: **982 passed,
951 warnings** in 121.35s. Independent scoped review found no actionable defects
and passed 30 checks. No native rebuild, deployment or visual sign-off.

#### Topology-aware selected path-fit outcomes — 2026-10-17

A selected fit now retains its canonical `DrawingRegion` subtree rather than
assuming one leaf geometry. This fixes recipe proposal/replay when a raster
refit produces multiple components; each leaf retains canonical geometry,
diagnostics and non-geometric shape metadata. The operation's dependency binding
still covers the complete input scene, evidence, settings, paint and upstream
recipe. No working raster or `VectorRegion` is added to the retained outcome.

Branch application passes through the existing topology-normalization transaction
so clones referencing the former leaf expand consistently, then restores exact
selected canonical leaf geometry and annotations. Regression coverage includes
separate components with/without cutouts and dependent clones, repeated replay
without refitting, and subsequent paint edits addressed to the created children.
Retired source-path slots remain authoritative for component addresses, so a
removed redundant path does not renumber later children or their clone targets.
The selected root's topology receipt is retained exactly. A scoped review found
this retired-slot case; its failing regression now passes alongside the split,
cutout and clone cases (64 focused checks).
Numerical fitting is unchanged; migration of the fitter's execution wrapper is
still separate from this outcome-contract correction.

Final source verification: **987 passed, 951 warnings** in 116.92s. No native
rebuild, deployment or visual sign-off; render-ΔE scoring remains unavailable
in this test environment because the optional renderer is not installed.

#### Typed whole-region path fitting kernel — 2026-10-17

`fit_raster_paths` accepts retained two-dimensional raster evidence, a region
address and resolved `ActionSettings`, returning source-ordered typed
`PathGeometry` values. It shares the existing numeric fitter unchanged across
quadratic, progressive and progressive-with-lines strategies. It does not infer
component ownership, resolve paint, mutate the raster or publish region IDs.
The existing shared topology transaction remains responsible for component
splits, cutouts and clone expansion.

Whole-region plan fitting now delegates to this kernel; `_fit_target_path` is
only the temporary typed-to-working-scene projection. Numeric fitting no longer
depends on `_Target` or `VectorRegion`. This is the numeric extraction needed by
the canonical region action, not completion of that action's scene adapter
removal. Tests compare exact typed fits with the previous numeric implementation
for all strategies, disconnected components and holes; plan routing verifies one
fit with resolved overrides and selected replay without regeneration.

Verification: **128 focused checks** pass. Full source suite: **998 passed,
951 warnings** in 217.86s. Two no-refit tests now guard the extracted kernel
instead of the removed import; no unused compatibility alias was retained.
No native rebuild, deployment or visual acceptance is claimed. The optional
render-ΔE scoring dependency is still absent from this environment.

#### Canonical region path publication — 2026-10-17

`fit_region_path` now consumes a canonical `DrawingRegion`, existing
`FitPathDrawingAction`, retained raster and resolved settings. It invokes the
typed numeric kernel and returns a canonical leaf or component subtree, retaining
paint, z-order and lineage. `publish_region_paths` shares the existing filled-path
classifier and owns stable component slots, cutout ownership, retired paths and
topology receipts. The working normalizer delegates those decisions to it rather
than maintaining a parallel publisher.

Plan actions reach this operation through `DrawingEngine`. The old
`_fit_target_path` wrapper is removed. New results and dependency-bound selected
results share `apply_region_geometry` in the sanctioned scene adapter; clone
expansion and original path slots remain transaction-wide. Raw working-path text,
footprints, raster state and legacy corner provenance translation remain adapter
responsibilities during migration. A targeted regression caught fresh fits
skipping that existing corner ingress; fresh-fit application now explicitly
performs it, while retained selections preserve exact canonical annotations.
No legacy post-fit corner inference or alternate fitter is introduced.

This migrates the whole-region path action, not direct path edits, remaining
relationship/corner actions, or the entire working-scene model. Fitter thresholds
and defaults are unchanged. Native compilation/deployment remain separate.

Verification: **182 focused checks** pass. Final full source suite after the
corner-ingress correction: **1,011 passed, 951 warnings** in 116.20s. The optional
renderer for render-ΔE scoring remains absent; no native build, deployment or
visual-quality acceptance is claimed. Remaining canonical action migration and
the parent plan's final whole-branch review are still open.

#### Canonical direct path edits — 2026-10-17

`path_operations.apply_region_path_operations` consumes the canonical drawing,
typed `UpdateGeometryAction`, resolved settings and optional exact material-span
evidence. It returns an updated immutable region and a command-lineage/acceptance
receipt. Numeric input comes from typed commands rather than rounded SVG text.
`drawing_refine` partitions ordered cleanup stages and delegates direct edits;
the shared scene adapter still owns topology, clones, paint and publication.
The old scene-bound `_path_shape` executor and duplicated fitting helpers are
removed. Inclusive `between`, source-distance guards, anchor retirement/rebinding
and composed command correspondence retain their existing semantics.

Three schema-declared operations (`reverse`, `mirror`, `break`) previously did
nothing in this executor. They now fail explicitly before changing geometry;
their geometry semantics remain unimplemented. No compatibility no-op is kept.
The full-suite replay regression exposed empty corner diagnostics being
reconstructed after cleanup despite having no declarations to preserve. The
legacy rebind now leaves those cleanup diagnostics untouched. Dependency checks
are unchanged; this is not a general redesign of cleanup observation recording.

Verification: **1,029 passed, 951 warnings** in 123.09s in the final full source
suite; the focused reserved-operation/binding/corner group passed 42 checks.
Additional validation is recorded in the operation-outcomes progress ledger. Remaining
corner/relationship execution, generic outcome reuse, full adapter removal and
the parent final whole-branch review are still open. No native build, deployment
or visual acceptance is implied by this source checkpoint.

#### Typed corner-span fitting kernel — 2026-10-18

`corner_fitting.fit_corner_span` consumes ordered samples, fixed entry/exit
coordinates, an output command ID and the resolved error budget. It returns one
typed `QuadraticCommand`, or no candidate when the existing residual gate fails.
It preserves endpoint precision and the current two-decimal fitted-control
policy. Corner detection, ownership, sampling, topology, paint and publication
remain outside this numeric operation.

The existing normalization helper delegates its fitting arithmetic to this
kernel, so automatic proposals and their recorded replay share one implementation.
This is a prerequisite extraction, not completion of canonical corner-path
execution: the legacy helper still owns span ordering, SVG projection and
provenance rebinding. No corner thresholds, fitting defaults or new inference
fallbacks are introduced. The remaining corner/relationship migration and final
whole-branch review remain open.

Verification: **54 focused tests** and **1,043 full-suite tests** pass
(951 full-suite dependency warnings; 175.24s). No native rebuild, deployment,
push or visual-quality acceptance is part of this checkpoint.

#### Canonical corner-path normalization — 2026-10-18

`corner_normalization.normalize_path_corners` now consumes `PathGeometry` and
the resolved error budget, returning an immutable updated path and a receipt of
normalized/skipped features and surviving command endpoints. Exact canonical
`CornerAnchor` ownership determines the fitted spans; no corner discovery or
coordinate-based anchor matching occurs inside the operation. It preserves
unaffected command IDs/coordinates and corner identity/radius, handles shared
boundaries and wrapped serialization, and protects other feature anchors from
being consumed. Unsupported non-curve spans and fits below the closed-path
command minimum are skipped explicitly.

The previous SVG helper delegates normalization to this operation. It retains
only provenance ingress, malformed/overlapping legacy declaration filtering,
diagnostic projection and output serialization; the duplicate normalization
loop is removed. Automatic proposals, explicit path-normalization plans and
recorded replay therefore use the same canonical operation. Empty declarations
do not force unnecessary canonical ingress. `command_token` is the shared exact
numeric command projection used by direct path edits and corner normalization,
avoiding a second serializer or rounding the numeric input through SVG.

Remaining work includes migrating the provenance adapter/publication boundary,
relationship actions, generic selected-outcome reuse and complete working-scene
adapter removal. This source checkpoint does not change corner detection,
sampling density, fitting defaults or tolerances, and is not a native rebuild,
deployment or visual-quality acceptance.

Verification: **172 focused tests** pass. The final full source suite passes
**1,056 tests, 951 warnings** in 116.08s. The parent final whole-branch review
remains open.

#### Shared canonical corner provenance binding — 2026-10-18

`corner_provenance.bind_prefit_corner_anchors` publishes already-resolved pre-fit
declarations onto source-indexed canonical paths. It resolves actual command IDs
by their positions in the supplied path, preserving sparse retained IDs rather
than synthesizing `path.cN` addresses. It preserves existing features and returns
immutable updated paths, skipped-declaration diagnostics and canonical-to-source
feature-ID bindings. Source subpath indices remain distinct from canonical path
addresses, including retained component slots and cutouts.

Drawing publication and normalization ingress now share this binder; their
duplicate anchor-construction loops are removed. Existing overlap behavior is
explicit rather than silently unified: initial publication retains the first
valid owner, while normalization rejects every overlapping candidate. Source
feature names remain unchanged in normalization reports through the returned
ID bindings. Malformed declarations are skipped without attaching invalid anchors
or mutating the original paths.

This moves canonical anchor validation/publication out of the working-scene
adapter, not the source-coordinate matching or topology/paint transactions.
Those adapter responsibilities, relationship actions, generic outcome reuse and
final whole-branch review remain open. No fitting/detection defaults change.

Verification: **217 expanded focused tests** pass. The final full source suite
passes **1,069 tests, 951 warnings** in 116.07s. No native rebuild, deployment,
push or visual-quality acceptance is included.

#### Typed source-coordinate corner matching — 2026-10-18

`corner_matching.match_prefit_corner_spans` resolves pre-fit declarations against
typed command endpoints and returns immutable ownership/skip records. The SVG
diagnostic adapter delegates matching and only projects its existing output
shape. Shared `path_geometry.command_from_token` also serves closed-path ingress,
removing a duplicate command decoder without rounding numeric inputs.

Nearest-endpoint tolerance, ambiguous ties, explicit-closing aliases, directed
ownership and invalid-provenance rejection retain their existing policies. The
matcher does not impose filled-contour minimum sizes on diagnostic inputs or
infer undeclared corners. One intentional correction: arc endpoints participate
in matching; the previous segment walker silently omitted them.

TDD includes missing-operation failures, routing failures before delegation and
the previously failing arc-endpoint regression. All **71 focused corner tests**
pass. The full source suite passes **1,086 tests, 951 warnings** in 119.48s.
Relationship actions, generic outcome
reuse, remaining adapter removal and final whole-branch review remain open.
No detector/fitter defaults, native build, deployment or visual acceptance change.

#### Integrated relationships and selected-operation reuse — 2026-10-18

`drawing_relationships` owns explicit clone assignment, alignment and symmetry
over canonical drawings/regions. The old working-scene implementations are
deleted. Clone chains resolve to concrete sources with composed transforms;
multiple transforms remain a canonical clone relationship rather than silently
becoming unrelated path geometry. Alignment preserves primitive types and typed
command/anchor identities. Explicit symmetry preserves the existing best-half
selection and joined filled topology, including cutouts and retired references.
`geometry_native` shares disposable native path predicates, reflection and line
clipping between operations and export rather than duplicating their arithmetic.

`SelectedOperationOutcome` replaces the fit-only outcome contract. Every public
`RecipeOperation` retains changed canonical root subtrees and output ordering,
including related clone changes and structural merge/split changes. Unchanged
roots are not copied into a parallel scene. Dependency bindings still cover
evidence, paint, upstream recipe, exact working input, intent and settings;
stale selections fail instead of running a solver. Fresh proposals can differ,
but reuse cannot execute the selected action again.

The projection boundary no longer makes topology or fitting decisions when
restoring these outcomes. It retains source raster/coverage evidence for new
components and borrows unchanged evidence from the dependency-bound input rather
than copying it per geometry edit. Paint projection also lives at this boundary,
preserving unchanged paint values. No fitter/detector defaults change.

This advances all three concerns together; it does **not** finish removal of the
whole working-scene model. Automatic detection/batch execution, source-bound
corner replay, paint resolution and source-evidence bookkeeping still use that
boundary. Their migration and the parent final whole-branch review remain open.
No native rebuild, deployment, push or visual-quality acceptance is included.
Verification: **56 focused tests** pass. The final full source suite passes
**1,108 tests, 951 warnings** in 122.72s; the diff check is clean. This includes
no-solver replay for merge/split, paint, order and relationships, a controlled
varying clone proposer, exact fit/corner correspondence, new-component coverage,
borrowed raster evidence, invalid root ownership and stale-dependency rejection.

#### Canonical corner replay and source paint fitting — 2026-10-18

`drawing_corner_actions` owns selected-path corner recipe application, fitted
command/feature identity binding, corner diagnostics and exact-topology annotation
transfer. Replay retains its evidence and ancestry checks, then delegates the
geometry decision. The shared `publish_region_update` adapter publishes both
corner and relationship results, preserving untouched component paths and paint.

`paint_fitting.fit_source_paint` returns immutable canonical paint and sampling
diagnostics from source pixels and visibility. Both canonical region paint fitting
and transitional refill use this kernel. Source support, erosion, thin-detail
fallback and median-color decisions are unchanged. Paint fitting must not require
whole-scene geometry conversion: a newly merged working region can legitimately
contain disconnected paths before topology publication. This is not permission to
weaken canonical geometry validation or construct placeholder geometry.

The old duplicate corner-binding/diagnostic and paint-conversion implementations
are removed. SVG annotation ingress/publication, automatic batch execution,
global visibility/shared-gradient orchestration and source bookkeeping still need
migration; the working-scene adapter and parent final review are not complete.
No detector/fitter defaults changed. Verification: **206 expanded tests** and the
full **1,122-test suite** pass (951 warnings, 140.28s); diff check is clean.
Optional render-DeltaE scoring remains unavailable without resvg-py. No native
build, deployment, push or visual acceptance is claimed.

#### Consolidated publication boundary — 2026-10-18

Corner replay no longer constructs working `Shape` objects or calculates their
footprints. `drawing_adapter.restore_corner_annotations` performs that boundary
translation and delegates exact command-topology checks to the canonical corner
operation. Downstream raw SVG coordinates are retained rather than serialized
again. The shared replacement guard now reports an absent or ambiguous target
explicitly instead of leaking a failed target lookup.

Primitive decoding and projection now live only in `drawing_adapter`.
`publish_primitive_geometry` is shared by manual primitive fitting and automatic
primitive replay; replay publishes its retained numeric result without rerunning
recognition. Occlusion uses the same primitive conversion. The duplicate decoder
in canonical drawing ingress and both converters in automatic replay are deleted,
with no compatibility aliases or changes to fitting policies.

This consolidates publication, not all automatic execution. Clone assignment,
symmetry reconstruction, seam constraints, simplification and occlusion still
have working-region execution boundaries; their migration must preserve each
retained intent's input/output semantics rather than replace it with a fresh
detector call. Global paint orchestration and source evidence also remain open.
Verification: **81 focused tests**, **89 additional engine/topology/occlusion
tests**, and the fresh full **1,131-test suite** pass (953 warnings, 115.82s).
Standalone imports and diff check pass. Optional render-DeltaE scoring is still
unavailable without resvg-py. No native build, deployment or visual acceptance.

#### Shared retained relationship transforms — 2026-10-18

`geometry_transforms` now owns typed endpoint/control and arc-axis/sweep
transformation. Canonical path alignment and the existing SVG projection share
this numeric kernel. Typed geometry retains full precision and command/anchor
IDs; the working SVG boundary retains its previous rounding policy. Arc behavior
is unchanged: this does not add ellipse scaling/shear support to baked arcs.

Clone and mirror-pair publication moved from optimizer pass implementations to
the shared drawing adapter. Proposal generation and replay now call those same
publication functions with the selected source, transform and axis. Target paint,
source evidence, family diagnostics and same-batch snapshot reads are preserved.
Duplicate native matrix-convention wrappers and the optimizer's separate shape
transform module are removed. Dependent-clone refresh, topology expansion and
seam source materialization also use the shared transformation boundary.

The migration exposed and fixes stale canonical caches in baked copies: changing
the SVG path previously left `canonical_path_geometry` at the source location.
The adapter now transfers exact-topology annotations onto the transformed numeric
output, retaining sparse command IDs and cutouts. It cannot silently transfer
features across changed command topology. SVG spelling is normalized when a
canonical cache is present, without changing its prior rounded coordinates.

This completes the shared numeric/publication slice, not removal of every
working-region automatic operation. Self-symmetry reconstruction/clone ownership,
seam decisions, simplification, occlusion and global paint/source orchestration
still require migration. No detector thresholds or fitter defaults changed.
Verification: **320 expanded checks** before the cache correction, **62 targeted
checks** after it, **13 transform/cutout checks**, and the final full **1,144-test
suite** pass (953 warnings, 180.08s). Standalone imports and diff check pass.
Optional render-DeltaE scoring remains unavailable without resvg-py. No native
build, deployment, performance improvement or visual acceptance is claimed.

#### Retained self-symmetry and seam execution — 2026-10-18

`drawing_symmetry` owns application of a recorded self-symmetry fitter/axis,
intrinsic declaration metadata, half-path reconstruction and shared numeric
helpers. `drawing_seams` owns recorded axis/endpoint/cluster/underlap execution,
exact command correspondence and the shared boundary mutations. Both optimizer
passes now import those implementations for proposal construction; replay calls
the engine modules directly rather than importing execution from a pass. The
old implementations were moved, not retained behind compatibility aliases.

Self-symmetry still materializes through the retained topology/clone transaction;
seams still validate path identities, preserve paint/owner evidence and clip
underlap to the retained outer boundary. Candidate discovery/ranking remains in
optimizer passes. No detector, tolerance, fitter-default or acceptance changes.

This completes both execution extractions together, not the entire working-region
adapter migration: these modules still consume the existing `VectorRegion` and
retained automatic-action boundaries. Simplification, occlusion, global paint and
source-evidence orchestration remain open. Final validation is recorded below.

Verification: **213 focused tests** and the fresh full **1,147-test suite** pass
(953 warnings, 141.09s). Unused-import/undefined-name lint, standalone imports
and diff check pass. Optional render-DeltaE scoring remains unavailable without
resvg-py. No native rebuild, deployment, speedup or visual acceptance is claimed.

#### Shared simplification and occlusion execution — 2026-10-18

`drawing_simplification` now owns path resampling, candidate fitting, subpath
topology/boundary checks and retained simplification application. The optimizer
keeps target/coverage discovery and imports the shared implementation. Seam
cleanup and rounded-rectangle recognition use the same engine helpers rather
than reaching into another pass. Existing sampling, fitting and acceptance
budgets are unchanged; cutout boundaries retain their independent checks.

`drawing_occlusion` now validates and applies recorded primitive/source/lens
children. Missing, forward and cyclic dependencies retain their existing errors;
replay does not rerun scene discovery. Proposals and replay share child and branch
publication, including source-raster union, paint order and consumed-region
diagnostics. The old replay body and pass-owned implementations are removed.

Both operations still consume the existing working-region/evidence adapters.
This is execution consolidation, not completed canonical-adapter removal.
Smoothing, remaining topology/clone-refresh orchestration, global paint/source
bookkeeping and the parent final review remain open. No thresholds/defaults,
corpus references or cached visual reports changed. Final validation follows.

Verification: **199 focused tests** and the fresh full **1,151-test suite** pass
(953 warnings, 119.21s). Unused-import/undefined-name lint, standalone imports
and diff check pass. Optional render-DeltaE scoring remains unavailable without
resvg-py. No native rebuild, deployment, speedup or visual acceptance is claimed.

#### Remaining automatic execution consolidation — 2026-10-18

`drawing_smoothing` owns bounded tangent projection and retained smoothing
application. `drawing_topology` now owns the former working-forest normalizer,
exact path-owner correspondence and retained topology application alongside
canonical component publication. The old `optimizer/topology.py` is deleted,
and all callers use the shared engine owner without compatibility aliases.

Retained corner normalization delegates to `drawing_corner_actions`; proposals
and replay share the same declaration validation, normalization and publication.
`drawing_adapter` owns dependent-clone refresh and its per-clone publisher.
Symmetry materialization shares that publisher while retaining its scoped
discovery/no-op checks and existing footprint-refresh policy. Source bindings,
paint, diagnostics and forest-level publication remain unchanged.

The automatic dispatcher now only routes recorded operations and resolves their
inputs; it no longer imports optimizer passes or implements these geometry
mutations. This does not eliminate working-region/evidence adapters or global
paint/source bookkeeping. Those and the parent whole-branch review remain open.
No detector thresholds, fit budgets, defaults or corpus artifacts changed.
Final validation follows.

Verification: **296 focused tests** and the fresh full **1,155-test suite** pass
(953 warnings, 115.49s). Unused-import/undefined-name lint, standalone imports
and diff check pass. Optional render-DeltaE scoring remains unavailable without
resvg-py. No native rebuild, deployment, speedup or visual acceptance is claimed.

#### Shared drawing-paint orchestration and scoped evidence — 2026-10-18

`drawing_paint` owns geometry-change fill invalidation, visible-pixel sampling,
source refill, shared symmetry/gradient surfaces and terminal output limits.
Plan/replay orchestration calls this shared owner; `paint_fitting` remains the
geometry-independent numeric fitter. Output palette limits are still terminal,
not an intermediate recipe mutation. The old implementations are removed from
`drawing_refine`; geometry execution and source raster evidence are unchanged.

Regression tests exposed a real ownership defect: visibility and invalidation
used local leaf IDs, which may repeat under different roots or nested branches.
They now use full ancestry paths throughout refill and shared-surface handling.
Previously one child could receive another child's sampled paint, or an unchanged
child could be invalidated. Those cases are corrected without changing sampling,
gradient, output-palette or fitting thresholds. Coverage includes nested repeated
IDs, independent source colors, one changed child, self-symmetry surfaces, shared
gradients with white cutouts, and geometry/evidence-preserving output limits.

This consolidates global paint orchestration, not every initial trace/root
assembly or working-evidence projection. Those remaining adapters and the parent
whole-branch review are still open. Final validation follows.

Verification: **107 focused tests** and the fresh full **1,161-test suite** pass
(953 warnings, 165.87s). New paint-module/test unused-import and undefined-name
checks, standalone paint/refine/replay/MCP imports and diff check pass. Existing
unused imports elsewhere are not a whole-workspace lint sign-off. Optional
render-DeltaE scoring remains unavailable without resvg-py. No native rebuild,
deployment, speedup or visual acceptance is claimed.

#### Initial root assembly ownership — 2026-10-18

`drawing_assembly` now owns the active v0 root builder, raw-contour entry,
coverage-aware material-polyline union, corner-aware fitting and exact retained
fit-evidence publication. `PythonDrawingEngine.fit_trace`, report scripts and
tests use that owner directly; refinement orchestration no longer owns initial
construction. Existing observer stages and source ordering are unchanged.

Address publication moved into the shared drawing adapter, preserving existing
child slots, internal IDs and root-address validation. The unreferenced older
boundary-clipping assembly route and its unused surface/AA-completion helpers
were deleted rather than retained as compatibility alternatives. Renderer tests
now guard the actual active assembly fitter instead of an unused helper.

Existing root render hashes and source-contour correspondence checks still pass.
No contour, corner, fitting, material-merge or diagnostic-stage policies changed.
Remaining work is the broader working-evidence projection audit and the parent
whole-branch review; this is not a native build or visual acceptance. Final
validation follows.

Verification: **132 focused tests** and the fresh full **1,165-test suite** pass
(953 warnings, 124.33s). Targeted unused-import/undefined-name lint, standalone
assembly/refine/engine/MCP imports, both report-script compilation checks and
diff check pass. Existing root render hashes remain unchanged. Optional
render-DeltaE scoring remains unavailable without resvg-py. No native rebuild,
deployment, speedup or visual acceptance is claimed.

#### Working-evidence projection audit — 2026-10-18

Corner provenance rebinding and root identity restoration now belong to
`drawing_adapter`, not refinement orchestration. Rebinding uses full tree
ancestry rather than local leaf IDs, which can repeat across roots or nested
branches. Regression tests demonstrate that identical paths with distinct
declared corners previously inherited the wrong corner annotations.

Working evidence now uses the same public address index as canonical geometry,
including children whose public addresses are derived from their ancestry.
Previously those children collided under a `None` key: even a z-order-only
canonical update could replace a child's retained raster with its parent's.
Address-scoped raster/coverage borrowing preserves the selected evidence without
copying unchanged arrays. Root metadata publication changes only root identity
and source references, retaining all other selected fields.

These are publication/evidence fixes, not changes to fitting, corner detection,
thresholds or SVG repair. Region merge/split execution still resides in
refinement orchestration; that migration, the broader transient working-record
reduction and parent whole-branch review remain open. Full verification follows; no native
build, deployment or visual acceptance is claimed.

Verification: **90 focused tests** and the fresh full **1,172-test suite** pass
(953 warnings, 122.88s). Targeted unused-import/undefined-name lint, standalone
adapter/refine/replay/engine/MCP imports and diff check pass. Optional
render-DeltaE scoring remains unavailable without resvg-py.

#### Shared merge/split execution — 2026-10-18

`drawing_partition.merge_regions` and `split_region` now execute the existing
typed `MergeAction`/`SplitAction` contracts. Refinement dispatches to them rather
than owning a second region-construction implementation. The shared owner retains
the complete union/clipping operation, raster publication, ordering, paint and
source provenance. The existing line splitter remains the shared native geometry
kernel; open-path splitting retains its sampling and boundary-snapping policy.

Derived leaf targets and leaf replacement/removal moved to `drawing_adapter`.
All consumers share its existing public-region address rule; the duplicate leaf
address decoder was deleted. These remain private working-evidence projections,
not a new public scene model or a claim that `VectorRegion` migration is complete.

Direct engine tests cover scoped child merging, line and curved dividers,
footprint preservation, stable child addressing, retained source paint/evidence
and invalid-divider rejection. A test that depended on refinement incidentally
re-exporting `unary_union` now imports the geometry owner directly. No numeric
threshold, detector, output-color or export-repair policy changed.

Full validation and the parent whole-branch review follow. No native rebuild,
deployment or new corpus visual acceptance is claimed.

Verification: **148 focused tests** and the fresh full **1,177-test suite** pass
(953 warnings, 139.81s). The renderer-purity guard now intercepts rasterization
at its actual partition owner. Targeted unused-import/undefined-name lint,
standalone partition/refine/replay/engine/MCP imports and diff check pass.

#### Engine branch review fix pass — 2026-10-18

One fresh-context read-only review of `2481aef..efe898f` found three Important
correctness defects, all independently reproduced before production edits:

- Source edits did not refresh dependent clone footprints. Resolved single- and
  multi-transform clone geometry and changed branch footprints now refresh before
  paint invalidation. Component publication remains after paint to preserve merge
  fill inheritance. Nested-source, later-split and replay checks cover the fix.
- Direct path programs grouped actions by kind instead of applying submitted
  order. Execution now composes ordered single steps, current reference values,
  exact version-scoped command bindings and lineage. Retired references reject
  rather than accidentally selecting another command at the same ordinal.
- Component splitting retained a whole-region canonical path cache. Each child
  now receives only its own component paths and exact-bound annotations; both
  cutout-to-component edits and redundant-path retirement are covered.

Multi-transform clone tests also exposed duplicate typed clone data in generic
metadata and inappropriate path-only corner binding; canonical clone values now
stay in canonical geometry rather than being copied into that metadata.

The review was risk-based, not exhaustive coverage of all 170 changed files.
Numerical quality, all corner/lineage combinations, MCP security/transport, UI,
packaging, cutout-export behavior, native parity, performance and visual acceptance
are not certified. No Critical findings or deferred minors were reported. The
private working-record boundary and later plan slices remain explicitly open.
Full fix-pass validation follows; no deployment is authorized by this review.

Verification: all three findings have behavioral RED-to-GREEN regressions;
**203 focused tests** and the fresh full **1,189-test suite** pass (953 warnings,
131.34s). Changed execution modules/tests pass targeted unused-import and
undefined-name checks; replay's six existing unused imports remain outside this
fix. Standalone engine/replay/MCP imports and diff check pass. No native rebuild,
performance or visual certification is implied; resvg-py remains unavailable.

#### Canonical export and reporting boundary — 2026-10-19

Deleted the working-region renderer, its result type, region-map wrapper, and
parallel trace summary/paint/diagnostic projections from `drawing_refine`.
Corpus/debug consumers now project their still-private working results into a
canonical `Drawing` before calling `drawing_render`; no legacy export aliases
remain. The canonical serializer also enforces the existing unmaterialized
self-symmetry rejection without repairing geometry during export.

Trace diagnostics now reuse the canonical geometry summary and fill serializer
from the actual published v0 drawing, including its exact path correspondence.
A palette-limit regression showed why this matters: the old summary reported
pre-limit blue paint while the published drawing had already selected white.
The trace envelope remains, but command/geometry and paint data come from the
same canonical models as normal MCP reports.

SVG byte layout for corpus/debug output now follows the existing canonical
renderer, including its clone clips and geometry definition IDs. This is not
an algorithm or fitting change; topology, independent clone paint, cutouts,
and no-render-time-reconstruction tests cover the migration.

This retires the parallel presentation adapters, not every `VectorRegion`:
optimizer proposals, replay batches, and action publication
still use private working records. Migrating those evidence/execution boundaries
remains open; do not replace the old type with a compatibility alias or claim
that private execution migration is complete.

Verification: the adapter-removal and published-paint regressions failed before
their fixes; 167 focused tests and the full 1,192-test suite pass. Native wheel
compilation and compiled regression checks pass. A compiled MCP radish auto run
publishes both versions and retrieves all artifacts; its v0/v1 SVGs are byte-
identical to the preceding compiled run. No live deployment was performed.

#### Canonical source evidence — 2026-10-19

`GeometryEvidence` no longer retains a `VectorRegion` forest. It stores the
canonical source `Drawing`, immutable raster/coverage observations, original
source observations, numeric ownership, and non-geometric kernel annotations.
Its revision binds those values and the trace/fit receipts together. Snapshot
loads detach nested canonical values; disposable kernel inputs cannot alias the
retained command objects. No `trace_roots` compatibility accessor remains.

Source capture and selected-operation capture share the same annotation
projection. Clone source/transform data live only in canonical clone geometry,
not in parallel shape metadata. Replay correspondence reads source ownership
directly from retained observations, without reconstructing a working forest.

Initial stitching and legacy numerical execution still create disposable
working inputs explicitly. Seam/corner rewrites discard superseded canonical
caches when changing path data; receipt capture occurs after canonical ingress,
with strict dependency checks preserved. This removes duplicate retained scene
state, not the remaining transient optimizer kernels or their adapters.

Verification: 1,194 full-suite tests passed; the final clone-metadata change also
passes 66 focused tests. Native rebuild and four compiled evidence regressions
pass. The final compiled MCP radish auto run publishes v0/v1 and retrieves all
artifacts without errors, with both SVGs byte-identical to the preceding build.
No live deployment was performed.

#### Automatic primitive kernel and selected-leaf publication — 2026-10-19

`recognize_primitive_geometry(footprint, epsilon=...)` now belongs to
`primitive_fitting` and returns `PrimitiveGeometry | None`, with no working
region input. The optimizer primitive pass orchestrates this kernel, canonical
leaf publication, and the existing proposal quality policy. Recorded primitive
intents retain the typed geometry directly; replay never repeats recognition.
Primitive decoding moved out of the adapter into the same numeric module,
including the occlusion caller, with no compatibility re-export.

Deleted the primitive-only `publish_primitive_geometry` route. Canonical path
and primitive selections share `publish_selected_leaf`; source observations,
private IDs and paint remain on the working evidence, while selected diagnostics
replace rather than merge retired annotations. The shared root publisher reuses
this route when observations and paint are unchanged.

Ruling: retain the working Shape-to-footprint codec at this temporary boundary.
Switching to native Skia primitive footprints changed rounded-rectangle quality
decisions in the integration suite. Changing that policy is outside this
ownership migration. Canonical geometry remains authoritative; the generic
working projection and automatic relationship-publication dependencies remain open.

#### Canonical path smoothing — 2026-10-19

Smoothing now consumes typed `PathGeometry` values through `smooth_paths`;
`smooth_region` applies the same kernel without changing retained component
membership. Numeric tangent projection and its all-or-nothing error budget are
unchanged. Command IDs and anchors survive exact command correspondence,
including quadratic-to-cubic elevation. No raster evidence, working `Shape`,
or optimizer record enters the geometry kernel.

Automatic proposals and replay share `apply_smoothing_choice`. Their temporary
`BoundPathInput` codec binds source-ordered commands without prematurely
classifying intermediate disconnected paths as a single filled component. It
publishes matching SVG text and canonical cache while retaining paint, source
evidence, private identity, and shape metadata. Deleted `_smoothed_shape` and
the working-record `apply_smoothing` entry point, without aliases.

New proposal/replay tests reproduced a real stale-cache defect: smoothing
previously changed `d` while retaining the old canonical path. The full suite
also exposed a premature component-classification error in the first binding
implementation; a disconnected-path regression reproduced it, and the explicit
path input boundary fixes it without merging or splitting regions. Broader
working-evidence migration remains open.

Verification: five new regression/contract tests passed after observed failures;
72 focused/integration tests passed. Fresh full suite: 1,217 passed (953 existing
warnings); 113 selected tests passed against the rebuilt native package (98
dependency warnings), with compiled smoothing, binding, and replay modules
explicitly checked. Targeted F401/F821 and whitespace checks pass. Optional
resvg-py remains absent from the source test environment; native smoke testing
uses the isolated runtime with scoring dependencies. No commit or live deploy.
Native MCP radish (24.63s) and Vbird (87.38s) retrieved v0/v1 artifacts with empty
error reports; all four SVGs are byte-identical to the canonical-bindings build.
The verified wheel is installed only in the isolated smoke runtime.

#### Relationship migration status clarification — 2026-10-19

The combined follow-up below supersedes the working-record symmetry execution
boundary described in this historical checkpoint; clone discovery/selection
and its existing canonical relationship operation were not reimplemented.

Clone and symmetry operation ownership is already implemented: canonical
`assign_clone`/`set_region_symmetry` live in `drawing_relationships`, and retained
intrinsic/self-symmetry execution lives in `drawing_symmetry`. Do not repeat those
migrations. The remaining relationship work is representation-boundary cleanup:
at the preceding checkpoint, automatic proposals/replay still called `publish_clone_assignment`,
`publish_mirror_pair`, and `publish_clone_refresh` in `drawing_adapter`.

This remaining boundary is now implemented: `BoundRegionInputs` binds automatic
input snapshots and named-output descendants to canonical selections while
retaining separate source observations and full scoped addresses. Automatic
clone and mirrored-copy choices call the existing canonical `assign_clone`;
dependent clone refresh uses the same binding/publication route. Deleted all
three relationship-specific publication helpers, without compatibility aliases.

Symbolic dependencies within the selected inputs remain symbolic, preserving
canonical cycle detection and transform composition. References outside the
selected inputs use retained resolved geometry, without retracing. A typed
missing-source error distinguishes this partial-input case from malformed or
ambiguous references; full-scene resolution remains fail-closed. Independent
target paint, scoped references, detector thresholds, and the existing numeric
shape/footprint publication policy are preserved.

The generic working-evidence bridge and shape codec remain explicit migration
work. This does not claim complete `VectorRegion` retirement or a new clone or
symmetry algorithm.

Verification: 1,205 full-suite tests passed (953 dependency warnings); 169 tests,
including optimizer integration and recipe replay, passed against the rebuilt
native package. New recognition/topology/annotation tests were observed failing
before implementation. Radish and Vbird compiled MCP runs retrieved v0/v1
artifacts without errors, with SVG bytes unchanged from the prior partition
checkpoint. Targeted F401/F821 checks pass except two pre-existing unused
imports in the integration test module (its F821 check passes). No commit or
live deployment was performed.

Binding/publication follow-up verification: 1,212 full-suite tests passed (953
dependency warnings), and all 123 focused tests passed against both source and
the rebuilt native package. Seven new binding tests cover scoped local-ID
collisions, alias rejection, symbolic cycles, named-output descendants,
independent paint, baked reflection, and both clone footprint-refresh policies.
Targeted F401/F821 checks and whitespace validation pass. The source environment
lacks optional resvg-py, so its render-Delta-E scoring is disabled; native MCP
validation uses the isolated runtime with scoring dependencies installed.
Another 66 compiled integration/recipe tests passed (189 compiled tests total).
Compiled MCP radish (28.61s) and Vbird (86.62s) retrieved v0/v1 artifacts without
errors; all four SVGs are byte-identical to the preceding canonical-primitives
checkpoint. Wheel `289c4b1-canonical-bindings` was installed only into the
isolated smoke runtime. No commit or live deployment was performed.

#### Canonical merge/split execution — 2026-10-19

`drawing_partition.merge_regions` and `split_region` now consume and return a
canonical `Drawing`. Geometry decisions no longer construct `VectorRegion`
trees or allocate globally numbered split children. Merge membership follows
the actual region tree; split children use stable parent-local addresses.
Disconnected outputs pass through the shared canonical component publisher.

The shared drawing-delta publisher separately retains merged source observations
and creates component sampling observations. Unchanged siblings remain intact;
source paint is refitted on the resulting visible leaves. Deleted the now-unused
`replace_working_leaf` and `remove_working_leaves` implementations rather than
leaving compatibility entry points.

The generic working-evidence projection and automatic geometry kernels remain
open migration work. Numeric divider sampling still imports its existing helper
functions from the optimizer module; this is not complete optimizer independence.

Verification: 1,200 full-suite tests passed (953 existing dependency warnings),
45 merge/split/refinement tests passed against the installed native wheel, and
targeted lint/whitespace checks passed. Compiled MCP radish and Vbird auto runs
retrieved v0/v1 artifacts without errors; both SVG versions are byte-identical
to the preceding canonical-publication build. No live deployment was performed.

#### Canonical operation publication — 2026-10-19

Manual path/primitive fitting, direct path edits, fill and paint ordering now
publish canonical selected subtrees through the shared publication boundary.
Deleted the one-off `apply_region_geometry`, `_with_fill`, `_with_z`, and the
private detection compatibility alias. Pre-fit corner ingress now belongs to
`corner_provenance`, shared by direct canonical fitting and source projection;
the adapter's `_ingress_corners` implementation is removed.

Canonical source-leaf replacement expands dependent clones by retained component
slot, preserving transforms and independent paint. Shared publication owns new
component observation rasters and keeps unchanged siblings intact. Geometry-
preserving paint/order updates retain path receipts; clone reuse additionally
checks the resolved footprint so source edits cannot leave stale clone bounds.
Implicit sparse child addresses now remain stable instead of being renumbered
as a side effect of assigning paint.

This removes the parallel per-operation publication routes. The generic working
projection and automatic kernels still need their own migration; this is not a
claim that the `VectorRegion` working model has been retired completely.

Verification: 1,198 full-suite tests pass, alongside 153 focused and 72 additional
publication/provenance/clone tests. Native rebuild and the three new compiled
regressions pass. Compiled MCP radish trace/refine and a same-paint/order manual
plan retrieve all artifacts without errors; the manual plan preserves SVG bytes.
No live deployment was performed.

#### Combined canonical kernel and boundary pass — 2026-10-19

The combined migration replaces working-record execution in simplification,
declared-corner normalization, occlusion and self-symmetry. Simplification and
normalization consume typed source-ordered paths; occlusion and symmetry return
canonical regions, preserving real cutouts and disconnected components. Seam
execution consumes explicit shape, paint-order and exterior-boundary observations
instead of a region resolver. Terminal output-color/gradient arithmetic lives in
the shared paint engine. Discovery and quality selection remain orchestration.

Shared curve/arc sampling moved to `path_sampling`; corner diagnostic projection
moved out of the optimizer into `corner_diagnostics`. Deleted the old owners and
working-record kernel entry points without re-export aliases. Canonical path
binding preserves sparse command IDs, source path order, numeric values and
anchors through fitting and component publication. It validates geometry rather
than requiring identical numeric text spelling.

Cross-operation regressions found and corrected in this pass:

- Synthetic closure IDs could collide with sparse retained command IDs.
- Component publication reparsed and renumbered already-canonical commands.
- Nested immutable diagnostic mappings were stringified in JSON reports.
- Named-output occlusion inputs lacked their full scoped source address.
- A disconnected, already-materialized symmetry union was mistaken for an
  unresolved symmetry group during rendering.
- A symmetry union tried to resolve external instance sources inside its selected
  subtree, despite already having the retained resolved instance geometry.

Simplification retains a path when explicit corner ownership cannot survive exact
correspondence, rather than silently deleting that declaration. Retired command
IDs are not aliases for newly renumbered endpoints. Renderer and report tests
distinguish JSON lists from immutable tuples, color casing from paint changes,
and SVG definitions from painted elements. Detector thresholds, fit budgets and
sampling policy are unchanged.

This is a combined execution/boundary migration, not complete removal of the
private optimizer model. Generic working projection, topology/paint observation
bookkeeping, root assembly and recipe orchestration still have `VectorRegion`
consumers. The parent whole-branch review and later plan slices remain open.
Verification at the preceding frozen checkpoint: 1,244 source tests passed; the
full compiled run passed 1,240 tests with four existing Python/native evidence-
revision snapshot mismatches. An isolated installation of the previous verified
wheel produces exactly the same four native hashes. Fitted SVG and corner fields
are identical; these baselines were not replaced. Compiled MCP Radish (22.22s)
and Vbird (83.07s) retrieved v0/v1 artifacts without errors, and all four SVGs
are byte-identical to the preceding smoothing build. The external-instance fix
then passed 61 focused checks; final rebuilt-package results follow. No commit
or live deployment is implied.

Final frozen-code verification: **1,245 source tests passed** (953 warnings,
116.50s). Wheel `289c4b1-combined-closure` rebuilt successfully and was installed
only into the isolated smoke runtime. All twelve changed kernel/binding modules
were explicitly verified as compiled extensions before the full native run:
**1,241 passed, four unchanged receipt-hash failures** (953 warnings, 115.06s).
The final native hashes again match the previous verified native wheel exactly;
this is not a clean full native-suite signoff. Targeted F401/F821 and whitespace
checks pass. Optional resvg-py is absent from the source environment but present
in the native smoke runtime.

Final compiled MCP Radish (21.23s) and Vbird (81.69s) retrieved both trace v0 and
refine v1 with empty artifact error reports. All four SVGs are byte-identical to
the preceding smoothing build. Reports are retained under
`native-combined-closure/{radish,vbird}/index.html` in this session's visualization
directory. No live deployment, commit, corpus replacement or full working-model
retirement is claimed; the remaining generic bridge and parent review stay open.

Use the existing contract direction:

| Method | Input | Output |
| --- | --- | --- |
| `trace_drawing` | Source, trace options, drawing settings | v0 `Drawing`, immutable engine evidence, diagnostics |
| Recipe revision (internal contract) | Selected version's effective recipe, typed edits, version-scoped selections | Validated effective recipe and stable target bindings |
| Recipe replay (internal contract) | Retained evidence, effective recipe, resolved settings | Reconstructed `Drawing`, feature/command correspondence, action and quality diagnostics |
| `fit_path` | Ordered path evidence, protected anchors, resolved settings | Typed fitted geometry, anchor mapping, fit diagnostics |
| `apply_path_action` | Replay working drawing, target path, typed action, resolved settings | New path geometry, changed command IDs, diagnostics |
| `apply_action` | Replay working drawing, evidence, drawing action, resolved settings | New drawing/evidence, affected/created/retired region IDs, diagnostics |
| `render_drawing` | Drawing and presentation-only render options | SVG/preview; no drawing mutation |

Confirm field-level signatures against the existing types before implementation;
do not create duplicate request/result classes to match this table literally.

Tasks:

1. Implement the Python `DrawingEngine` using current algorithms. Resolve settings
   once using drawing defaults, plan settings, action overrides, then applicable
   nested overrides. Keep path and drawing action defaults distinct where the
   canonical settings contract requires it.
2. Route manual refinement through it, then route optimizer passes through the
   same operations. The optimizer orders proposals and selects outcomes; it does
   not implement alternate geometry transforms.
3. Make the store retain canonical drawings, effective recipes, submitted edits,
   and references to opaque immutable evidence.
   Remove the parallel `VectorRegion` execution model after its callers migrate.
   Internal adapters during migration must not become permanent public contracts.
4. Move symmetry materialization and union into an explicit retained action.
   Preserve clone references and paint-space behavior through source updates.
5. Eliminate render-only material underlays. Where a repair is appropriate, make
   it a real retained geometry operation with provenance and quality assessment.
   Do not preserve a visually attractive overlay by concealing it from plan targets.
6. Remove render-time atomic flattening. Keep render precision separate from
   fitting/cleanup settings.

### Agreed reconstruction/replay contract — 2026-09-09

This is target refactoring behavior, not a claim that recipe editing and replay
are already implemented in the MCP surface.

```text
selected version's effective recipe + agent edits
                         ↓
             validated effective recipe
                         +
             retained source/trace evidence
                         ↓
      replay geometry actions → fit final paint
                         ↓
     child drawing + refreshed maps + assessment
```

- `base_version` selects the inherited recipe and the version in which submitted
  selections were made. It does not require the child to preserve the parent's
  generated geometry. Keep v0 pre-optimizer and auto v1 semantics unchanged.
- Support adding, replacing, and removing declarations and recipe operations by
  stable identity, not only appending actions. Retain both the submitted edits
  and the resolved effective recipe, including automatic actions/settings, so a
  child is reproducible without replaying an undocumented history of mutations.
- Removing a corner declaration refits without that protected corner. Removing
  symmetry or a clone relationship reconstructs without that constraint; it does
  not freeze the previously reconstructed shape. An explicit geometry deletion
  remains a different action from removing a declaration.
- Region merge/split operations create named logical outputs for later actions.
  Validate ordering and dependencies before replay. Removing or changing an
  upstream operation must not silently retarget dependent operations; return an
  actionable unresolved-reference error unless the revision repairs them.
- Keep detected observations separate from active declarations. Record origin
  and application status independently: detected versus agent-declared, and
  candidate versus applied versus rejected. An explicit rejection must not be
  immediately undone by the same automatic detector on replay. Recompute derived
  observations after geometry changes rather than carrying stale detections.
- Feature selection may use the displayed version's region/path/command IDs.
  Resolve and retain a stable binding to source evidence, a declared feature, or
  a named operation output plus its directed span. Generated command numbers
  alone are not durable recipe targets. Report created/retired correspondence;
  ambiguous mappings fail explicitly instead of guessing a nearest command.
- Geometry guides select a new retained evidence revision with explicit original
  image coordinates. Rebind or invalidate declarations when that evidence changes;
  never assume old command IDs address a newly traced guide. Paint still comes
  from the original source, not guide label colors.
- Incremental replay may reuse unaffected cached work only when its result is
  equivalent to full replay. Cache keys include evidence revision, recipe and
  resolved settings, engine revision, and upstream dependencies. Rendering remains
  pure serialization of the resulting stored drawing.
- Commit a child atomically after validation/execution; failures leave the parent
  unchanged. A geometrically valid but visually imperfect manual result may be a
  labelled needs-refinement branch, following Slice 4's assessment policy.

Implement first as a bounded feature-review/correction slice for pre-fit corner
ownership on radish and Apple Music, then expand to shared boundaries and
clone/symmetry declarations. Reuse canonical types; finalize exact typed edit
signatures and field-level examples before changing the public schema.

### Execution-boundary audit — 2026-09-09

The current `DrawingEngine` is a protocol, not the active implementation boundary.
`DrawingVersion` retains generated `VectorRegion` forests; refinement copies and
changes those forests, adapting them to `Drawing` only for validation. This is
not recipe replay: removing a declaration would retain geometry produced while
that declaration was active.

The first implementation checkpoint therefore spans storage and execution:

1. Retain immutable evidence with an explicit revision, separately from each
   version's canonical `Drawing`.
2. Define typed corner declaration add/replace/remove edits, stable declaration
   IDs, and directed bindings to pre-fit evidence. Resolve display command IDs
   using the selected version; reject stale or ambiguous bindings.
3. Resolve the inherited effective recipe, then rebuild a fresh `Drawing` from
   evidence, not from parent output. Fit paint after geometry. Record submitted
   edits, resolved recipe/settings, evidence revision, and result atomically.
4. Migrate trace/auto version creation, refinement, and stored-artifact rendering
   through that boundary. Do not maintain parallel authoritative scene fields
   or add an unreleased legacy read fallback. Any temporary algorithm ingress
   adapter is internal evidence plumbing, not a second version scene.
5. Prove add-then-remove equivalence and sibling isolation on corner corrections
   before expanding to merge/split named outputs or clone/symmetry declarations.

The initial checkpoint uses full replay. Incremental execution is deferred until
it can be checked against that reference implementation. Merely adding recipe
fields around the current generated-geometry mutation loop does not complete it.

### Canonical corner ingress and fitter contracts — implementation order

Before changing version storage, close the verified adapter gap: it currently
parses an SVG path without attaching the pre-fit corner declarations already
available on the Python working region. Canonical `CornerAnchor` must preserve
the same directed ownership semantics as the shared pre-fit binder.

| Contract | Input | Output / invariant |
| --- | --- | --- |
| Canonical corner | Feature ID, entry command ID, exit command ID, ordered owned command IDs, optional measured radius | Point: equal entry/exit and no owned commands. Rounded span: distinct boundaries and exactly the directed intervening drawable commands, including the command arriving at the exit. |
| Corner membership view | Canonical corner | Referenced command IDs derived from boundaries and owned commands; do not store another independent membership list. |
| Pre-fit ingress binding | Recorded pre-fit declarations and the matching path | Typed corner anchors or explicit binding skips; no corner inference from the fitted result. |
| `DrawingEngine.fit_path` | Existing `FitPathRequest`: source polyline `PathGeometry` with explicit anchors, resolved `PathFitOptions` | Existing `FitPathResult`: fitted `PathGeometry`, transferred anchors/correspondence, fit diagnostics. |

Validate all references against the one `ClosedPath`. Preserve directed
wraparound and distinguish the explicit closing drawable command from the
equivalent `M` endpoint address. Reject ambiguous spans or an unaddressable
implicit closing edge instead of claiming ownership of the next side. An
unknown radius never implies a point corner. Keep non-corner anchor types and
their existing semantics; no compatibility constructor for unreleased corner
models is required.

The pure fitter consumes supplied feature declarations; it must not silently
rediscover a corner the effective recipe removed. Automatic feature detection
belongs before that call and contributes the initial declarations. Extract and
reuse the current detector/fitter code paths rather than introduce a second
implementation or change their thresholds during this refactoring. Source
polyline command addresses bind the retained evidence; output command addresses
remain version-scoped and require explicit correspondence after refitting.

Implementation checkpoints:

1. Canonical corner ownership and ingress binding, with typed round-trip,
   point/rounded/wraparound and adjacent-side exclusion checks.
2. Concrete Python `fit_path` using explicit pre-fit declarations and retained
   polyline evidence; prove add/remove/refit equivalence independently of store.
3. Canonical storage plus recipe revision/replay and artifact migration as one
   end-to-end boundary, preserving v0/v1 and immutable sibling branches. Steps
   1–2 are foundations, not completion of the agent correction flow.

### Progress — explicit Python fitting, 2026-09-09

`PythonPathFitter.fit_path(FitPathRequest)` now consumes retained line geometry
and explicit canonical corner declarations. Automatic tracing and the explicit
entry share the same numerical span fitter; detection/recovery remains outside
the explicit entry. Empty declarations are authoritative. Point and rounded
feature IDs, exact boundary coordinates, directed ownership, and source/output
span correspondence survive fitting, including a nonzero closed-path start.
The result requires typed correspondence rather than treating regenerated command
numbers as lineage. Unsupported source geometry/annotations and unaddressable
implicit closure fail explicitly; no legacy fallback or full-engine placeholder
was added.

Independent spec and quality review approved the task without findings. The final
frozen-code suite passes **721 tests**, with 949 existing warnings, in 104.96s.
Evidence: `/tmp/vectormark-explicit-path-fitter-frozen-{pytest.log,junit.xml}`.
No corpus, reference, golden, deployed-server, or committed-branch changes were
made. This is the fitting capability, not yet canonical version storage or replay.

Two evidence prerequisites remain before replay: detach nested preprocessing
buffers/models in stored snapshots, and retain the actual merged pre-fit
polylines. Boundary-root working rasters can be rasterizations of already-fitted
geometry and must not be mistaken for original source evidence.

### Progress — retained preprocessing isolation, 2026-09-09

The first evidence prerequisite above is complete. Stored and retrieved trace
snapshots now detach all four preprocessing rasters and deeply freeze gradient
models and background metadata. The existing freeze implementation is reused;
one extracted JSON converter serves both core trace artifacts and MCP output.
Source, returned-snapshot, public-dictionary, and sibling-version mutation checks
pass without changing tracing or paint values. Independent review approved the
task; the full suite passes **723 tests**, with 949 existing warnings, in 100.38s.
Evidence: `/tmp/vectormark-retained-evidence-{pytest.log,junit.xml}`.
Merged source-polyline retention is completed below; canonical recipe replay
remains pending.

### Progress — exact merged pre-fit evidence, 2026-09-09

Root construction now returns fitted regions together with the trace enriched by
the actual merged source polylines, pre-fit declarations, and emitted-run
correspondence. Source, fitted, and public target addresses are distinct. Evidence
is bound to normalized emitted leaf paths at the pre-stitch stage with exact
output digests; it does not pretend to address post-optimizer commands. A
single-material boundary root keeps the retained baseline path's own evidence.
All valid-input fitting and paint decisions remain unchanged; four deterministic
fixture SVG hashes match the pre-edit output exactly.

Store snapshots isolate the new typed evidence, pickle-based trace caches retain
it with a new cache format, and ordinary public output remains compact. Missing
source contours now fail explicitly instead of fabricating source evidence from
a potentially fitted SVG. Independent spec/quality review approved the task.
The full suite passes **740 tests**,949 existing warnings,96.90s; evidence:
`/tmp/vectormark-root-path-evidence-{pytest.log,junit.xml}`. No corpus, references,
goldens, commits, or deployed server were changed. This is a replay prerequisite,
not completion of canonical version storage or public recipe editing.

### Progress — source-based corner recipe replay, 2026-09-09

The internal `CornerRecipeExecutor` creates an effective declaration set once,
applies typed add/replace/remove edits transactionally, and replays retained source
commands through the explicit fitter. Empty declarations are authoritative; parent
fitted geometry is not an input. Derived recipe/output identities and actual
correspondence distinguish source references, stable features, and generated
commands. Invalid source observations remain explicit diagnostics rather than
silently becoming constraints.

Independent review found and verified fixes for two mutable aliases: canonical
path command lists and nested fit-result diagnostics. Commands now take tuple
ownership at construction; fit diagnostics reuse the extracted existing freezer.
No duplicate recipe-specific snapshot implementation was added. Reversible edits,
source/sibling isolation, strict JSON reporting, and exact endpoints are covered.
Final full suite: **756 passed**,949 existing warnings,99.22s. Evidence:
`/tmp/vectormark-corner-recipe-round1-{pytest.log,junit.xml}`. Task13 spec and code
quality are independently approved. Compatible artwork evidence caches were
unavailable; this verifies deterministic fixtures, not a new corpus run.

This is an internal path capability. Canonical version storage, drawing-level
recipe composition, fill invalidation, public edit selection, and pure artifact
rendering are still pending. No commit or deployment was performed.

Topology ingress also needs verification before canonical rendering becomes
authoritative: disconnected subpaths are sibling components, not subtractive
cutouts merely because they follow the first path in an SVG string. Preserve
the existing child-region addressing convention when making them explicit.

Read-only probes confirmed both prerequisites for authoritative canonical storage:

- `geometry_from_svg_d` currently assigns the first subpath as outer and the rest
  as cutouts. Two disconnected rectangles, or a hole serialized before its
  enclosing outline, therefore receive false semantic relationships even though
  evenodd rendering can still look correct. Classify topology independently of
  serialization order and preserve command/feature correspondence through ingress.
- Some frozen canonical dataclasses still accept mutable owned containers:
  nested diagnostics can alter
  `DrawingRegion`, and a gradient's input stop list can be emptied after validation.
  Task13 fixed the associated `ClosedPath` command-list alias. Normalize the
  remaining owned sequences and deeply detach JSON metadata before canonical
  drawings become retained version state. Reuse shared isolation machinery;
  do not introduce a second authoritative scene or another generic freezer.

These findings are storage/ingress concerns, not invitations to adjust automatic
fitting or rerun the corpus. The ownership finding is resolved below; topology
ingress remains part of the upcoming canonical integration boundary.

### Progress — canonical value ownership, 2026-09-09

Canonical cutouts, clone transforms, polygon vertices, gradient stops, region
children/provenance IDs, drawing roots, and engine result ID collections take
tuple ownership before validation. Drawing and engine-result diagnostics reuse
the existing shared deep freezer. No second generic snapshot format or legacy
wrapper was introduced. Generic engine evidence remains opaque, and existing
store defensive copies are retained until the authority migration.

Mutation, validation, JSON, and exact-value/SVG preservation checks pass.
Independent spec and quality review approved the task with no Critical or
Important findings. The final frozen-code suite passes **761 tests**,949 warnings,
in107.97s; evidence:
`/tmp/vectormark-canonical-ownership-{pytest.log,junit.xml}`. Eight known NumPy
cross-product warnings in the focused gate remain tracked debt. No geometry,
paint, topology, rendering, corpus, dependency, commit, or deployment change was
made in this batch. This secures canonical values; it does not make them the
store's authoritative representation yet.

Verification:

- Serialized drawing round-trip preserves geometry, anchors, fill, relationships,
  and IDs within the version.
- Repeated render is deterministic and cannot invoke a fitter or mutate evidence.
- Every visible painted object resolves to retained geometry or a declared clone
  instance. Definitions are allowed; unaddressable geometry repairs are not.
- Same operation/settings through a plan and optimizer produces equivalent output.
- Old versions remain unchanged; changed regions invalidate only relevant caches.
- Full replay and incremental replay of the same recipe/evidence/settings agree.
- Adding then removing a declaration reproduces the recipe without it, rather
  than preserving geometry generated while it was active.
- Replacing an earlier declaration does not leave its superseded effect in the
  child. Sibling recipes remain independent.
- Upstream merge/split edits and regenerated command IDs either resolve through
  explicit bindings or fail with precise diagnostics; no silent target drift.

## Slice 4 — Quality assessment and acceptance

Add one typed quality assessment service shared by automatic optimization, manual
refinement diagnostics, MCP results, and corpus/debug reports.

Proposed logical contract:

`assess_revision(reference, before, candidate, changes, policy) -> QualityReport`

`reference` includes immutable v0 geometry/evidence, original image, and explicit
image-to-drawing coordinates. `changes` supplies region lineage so legitimate
merge/split actions are not mistaken for missing regions.

`QualityReport` contains:

- Per-region exterior and interior boundary errors, with localized worst regions.
- Hole/topology/adjacency changes; unexpected gaps and overlaps.
- Per-region fill error and final composited appearance error.
- Constraint results for clones, symmetry, alignment, and shared endpoints.
- Free/corner command counts and complexity deltas, without rewarding broken shapes.
- Per-step and cumulative-v0 deviations, plus measurement scale/tolerances.
- Typed outcome, reasons, and actionable region/command references.

Acceptance rules:

- Always reject invalid geometry, unresolved references, and non-finite results.
- Automatic proposals must satisfy fidelity and structural constraints; reject a
  regression with its reason and retain the previous state.
- Explicit structural plans may intentionally change topology. Assess against the
  declared operation and ancestry, not an unconditional same-region-count rule.
- A valid manual/guide candidate that misses visual targets may be retained as a
  clearly labelled needs-refinement branch. It must not be labelled complete.
- Keep original-image appearance and v0 geometry assessments separate. A poor v0
  must not permanently forbid an agent's intentional improvement toward the source.
- Calibrate thresholds on a development set, lock them before holdout evaluation,
  and account for resolution/anti-aliasing. Do not invent a universal zero-pixel-
  difference requirement or a single weighted score that hides topology failures.

Required counterexamples include identical outer silhouette with a moved internal
boundary, small lost holes, accumulated drift, symmetry damage, altered clone
paint, and export-time differences. The reviewed x=5 to x=8 shared-boundary case
must no longer pass undetected at a zero budget.

## Slice 5 — Fill fitting as an accountable separate pipeline

Primary files: `fill_fit.py`, fill models, region refill orchestration, quality
reports, and SVG paint serialization.

Proposed result: `FillFitResult(fill, residuals, status, reason, sample_domain)`.
Use typed statuses rather than interpreting a flat fill as evidence of success.

- Distinguish a supported flat fit from fallback after an unsupported gradient.
- Retain original source pixels independently from segmentation/geometry guides.
- Sample the region's visible source domain using explicit coordinates and
  occlusion; diagnose empty/insufficient samples.
- Clip emitted paint to final retained geometry; fill fitting cannot rewrite it.
- Refit changed regions and affected relationships after geometry changes, not
  unrelated regions or the original trace.
- For unsupported complex paint, return enough evidence for an agent to split the
  region or acknowledge the limitation. Richer vector fill models can follow
  measured failures; raster embedding requires an explicit hybrid-output policy.

Exit: a poor gradient approximation cannot silently receive complete status;
geometry guides never leak their label colors into output; recolored clones retain
independent fills with correct transform behavior.

## Slice 6 — Agent workflow and lifecycle

Primary files: MCP server/models, drawing store, artifact generation, MCP UI,
README, `docs/mcp.md`, generation skill and examples.

- Generate descriptions/examples from or validate them against the typed schema.
  Remove stale CLI-only/one-shot advice from the generation skill.
- Teach the same loop everywhere: generate or receive source; trace auto; inspect
  all review views; select a targeted plan or geometry guide; compare the result;
  continue or return the best branch with explicit unresolved issues.
- Present a version-scoped feature/curve map alongside the source, rendered SVG,
  region map, and difference view. Show corner entry/exit and owned commands,
  inflections, primitives, symmetry axes, clone families, shared junctions and
  cutouts; distinguish free L/Q/C curves from corner/primitive geometry. Expose
  detection origin and application status rather than implying every observation
  is an enforced feature. Complex regions need addressable close-ups and a legend.
- Agent corrections revise the effective recipe as defined in Slice 3. Return
  the child recipe, regenerated feature/region maps, reference correspondence,
  operation outcomes, and separate geometry/fill assessments against parent and
  source. Diagnostic maps are views, not editable sources of drawing truth.
- A declared missing feature can be supplied by the agent; absence from detection
  is not proof of absence in the source. Inflections are normally derived from
  curves; an explicit inflection constraint is distinct from an observation.
- Inspect geometry and fill separately. Auto is a starting proposal, not proof
  that all primitives/clones/symmetries were found. Avoid reflexively rerunning
  every detection pass.
- Include the shared quality report in normal MCP output; keep large images/SVGs
  as retrievable artifacts. Localized crops should include scale and target IDs.
- Retain compact previews plus full-resolution source/diff inspection. A tiny
  panel must not be the only evidence available for small defects.
- Exercise image transfer between generation tooling and MCP: valid ImageRef,
  exact guide dimensions, alpha/background policy, and actionable invalid-input
  errors. A file path on the agent's machine is not assumed server-local.
- Add retry-safe submission identity for expensive mutating calls, with request
  digest checks so a retried plan does not accidentally create a sibling version.
- Test concurrent branching, transport reconnects, idle expiration, artifact
  retrieval, missing drawings, and bounded drawing/artifact memory usage.
- Surface expiration/liveness in output. Durable project export/restart recovery
  remains an optional follow-up to the explicitly ephemeral MVP.

Exit: a real client can complete the workflow without repo access, missing tools,
opaque references, hidden output changes, or pretending expired state is live.

## Slice 7 — Evaluate the actual product

Keep the current 24-image corpus as a regression set, but add a separate generated
artwork evaluation set. Do not tune repeatedly against the held-out cases.

### Curated target drawings

Aim for a reviewed target SVG for every corpus entry, introduced incrementally.
Keep three distinct artifacts: original raster, approved target SVG, and generated
candidate SVG. Never overwrite a target when refreshing generated corpus output.
The source raster remains the appearance authority; an attractive target that
redesigns the artwork is not ground truth for faithful reconstruction.

Proposed layout, reusing each corpus entry's existing identity:

```
corpus/reference/<entry>.svg
corpus/reference/<entry>.json
corpus/svg/trace_auto-<entry>.svg
```

The manifest records source hash, canvas/viewBox and coordinate registration,
target revision/hash, author (human or agent), review status (`draft` or
`approved`), reviewer, and notes on ambiguous source details. Agent-created
targets are drafts until reviewed. Approved targets must be genuine editable
vectors, not raster images embedded in SVG, with all assets available locally.
Draft and missing targets remain visible in the report but cannot silently
become acceptance baselines.

Structural expectations live alongside the target, keyed by semantic parts of
the artwork rather than generated command IDs: region adjacency and holes,
clone families/transforms, expected symmetries, important corner locations,
straight spans, and flat/gradient paint. Use explicit matching to generated
regions; missing or ambiguous matches are reported rather than assigned by
region number. Record useful complexity budgets where reviewed, not a universal
requirement for identical command counts or Q/C choices.

Compare candidate and target in two independent ways:

1. Appearance: render in the same registered coordinates, color/alpha policy,
   renderer/version, and evaluation scales. Compare against both source and
   target, including localized exterior/interior boundary and fill differences.
2. Editability/structure: check the reviewed relationships and topology without
   requiring identical SVG bytes, path ordering, IDs, or Bézier decomposition.
   Multiple equivalent vector constructions are valid.

Primitive recognition and clone extraction may intentionally change the SVG
representation: a fitted path may become a native primitive, and independently
painted paths may share one geometry definition through transformed instances.
Those are valid outcomes even when the approved reference uses another encoding.
Assess visible instantiated geometry and each instance's paint, while checking
reviewed clone/symmetry relationships in the retained drawing. Representation
differences alone are not fidelity failures.

Keep three test contracts distinct:

- Approved-reference conformance uses registered render/boundary/paint checks
  plus topology and reviewed structural expectations; never SVG byte equality.
- Algorithm regression checks use before/after fidelity and structure. Existing
  implementation snapshots must be evaluated under those invariants when an
  intentional algorithm change invalidates their exact serialization; do not
  blindly overwrite them or demand their old command decomposition.
- Byte equality is appropriate for repeated serialization of the same retained
  drawing, immutable-version checks, and narrowly scoped refactors explicitly
  intended to preserve serialization. It is not a general vectorization goal.

The main report links source, target, candidate, differences, structural checks,
and review provenance. Keep last-run regression snapshots separate from approved
targets: a stable incorrect result is not a successful reconstruction. Changes
to approved targets require explicit review with their own before/after diff.
Start with radish, Apple Music, Vbird, and Instagram, then expand across the
corpus. For autonomous holdout evaluation, targets and expectations are available
to the evaluator, not to the reconstruction agent.

### SVG understanding and editability evaluation

Research linked from VTracer informs this evaluation, not a replacement-engine
decision. [Text-Based Reasoning About Vector Graphics](https://arxiv.org/html/2404.06479v1)
uses a learned structured description between SVG paths and reasoning, and reports
failures involving verbose paths, small objects, and intentional constraints.
[SVG image-text bridging, supplementary limitations](https://openaccess.thecvf.com/content/WACV2025/supplemental/Cai_An_Investigation_on_WACV_2025_supplemental.pdf)
distinguishes recognition from reliable low-level editing: its tested models
failed rotation/scaling trials. These are historical model results, not an
assumption about today's agent capabilities.

Our design inference is to expose rendered evidence plus compact, addressable,
typed geometry and relationships. Keep measured facts, proposed relationships,
and accepted constraints distinct. Derive summaries from the canonical drawing
and diagnostics rather than maintaining a second scene description. The agent
selects intent and targets; the engine performs numerical geometry operations.

For a small frozen evaluation subset, compare raw SVG text, visual panels with
addressable IDs, and panels plus typed geometry/diagnostics. Hold the model,
source, initial drawing, tools, and interaction budget fixed; record actual token
use. Ask the agent to locate a specific part, identify clone/symmetry candidates,
and perform a localized edit without damaging unrelated parts. Separate target
selection, plan validity, executed geometry, and visual improvement failures.
Approved target annotations remain evaluator-only in blind runs.

[StarVector](https://arxiv.org/html/2312.11556v4) motivates evaluating primitive
usage and compactness alongside appearance, and reports weaknesses in pixel-only
quality ranking. For VectorMark, do not replace localized boundary/paint checks
with a perceptual score: semantic resemblance cannot excuse a missing cutout or
a misplaced shared endpoint. Likewise, do not require an exact target command
sequence when equivalent editable geometry is valid.

Evaluation groups:

- Geometric marks with repeats, rotation/reflection, and multiple independent fills.
- Organic silhouettes with shallow/rounded corners and inflections.
- Transparent holes and overlapping regions.
- Multiple gradient surfaces and intentionally subtle internal boundaries.
- Deliberately unsupported texture/effects to verify honest limitation reporting.

Run three modes on frozen source images: auto-only; agent plans after auto;
agent plans or geometry-guide retrace after auto. Separately exercise fresh image
generation to test the complete handoff without making generation variability the
only measure of reconstruction performance.

Use agents without repository/code access and the shipped instructions only.
Retain tool transcripts, source/guide artifacts, all branch versions, quality
reports, time, tool calls, and best-version selection. Bound calls/time and stop
when improvement stalls; do not force the last version to be the delivered one.

Report separately:

1. Appearance fidelity, including localized structural errors.
2. Editability: logical structure, relationships, useful anchors, unnecessary
   commands, no hidden raster or export-only repairs.
3. Autonomous success: completion rate, manual interventions, calls, latency,
   meaningful improvement over auto-only, and honest failure identification.

Designer/user review is a release gate, not something inferred solely from SSIM
or command count. Freeze quantitative targets before the holdout run and publish
failures beside successes. Corpus index stays at `corpus/index.html`, SVG links
under `corpus/svg`, with linked per-image debug reports and shared report schemas.

## Slice 8 — Performance and native follow-up

Profile cold trace, preprocessing, region construction, fitting, optimizer actions,
fill fitting, render/artifact generation, and snapshot copying separately. Measure
latency and peak memory for representative image sizes and long branch histories.

First optimize avoidable work: repeat fitting during render, duplicate raster
sampling, repeated full-history copies, redundant conversions, and uncached
artifact rendering. Preserve behavior; do not hide accuracy changes as speedups.

Only then select native kernels by measured cost. Segmentation, contour extraction,
path fitting, and geometry actions can sit behind the same `DrawingEngine` contract;
the optimizer remains an orchestrator. Python/native conformance compares topology,
provenance, constraints, and geometric/render tolerances, not byte-identical SVG
serialization. Preserve IDs for identical retained inputs/actions where promised;
do not promise identical command layouts when engines legitimately fit differently.

## First implementation checkpoint

Start with Slice 1, then the isolated corner fixes in Slice 2. Present:

- Fresh-install and current-contract test results.
- The exact runtime/schema changes, with no legacy compatibility restoration.
- Cached radish/Apple Music/Vbird before/after corner diagnostics and SVGs.
- Any unresolved failures before beginning the canonical execution migration.

Planning does not authorize deployment, corpus replacement, commits, PR creation,
or an automatic native rewrite. Those follow the user's implementation direction.

## Canonical topology normalization — 2026-10-19

Added `normalize_drawing_topology(Drawing)` for atomic component publication and
clone dependency traversal. Existing paths, sparse command IDs, annotations,
fills, transforms, and source component slots survive without refitting. Shared
`expand_clone_components` replaces duplicate region-replacement expansion and
fixes multi-transform working clones whose source becomes a component group.

Disposable normalization and correspondence receipt code now lives in
`optimizer/topology_binding.py` as `normalize_working_topology`; callers were
migrated without compatibility re-exports. This temporary boundary does not
complete working-model or generic bridge retirement. Seven new regression cases
observed genuine RED before implementation; the focused publication/replay group
passes 82 cases. Full source suite: 1,252 passed. Compiled suite: 1,248 passed
with the same four pre-existing receipt-fingerprint failures (all four audit
values unchanged from the preceding wheel). Compiled MCP radish/Vbird runs have
no artifact errors; all four trace/refine SVGs are byte-identical to the preceding
build. No live deployment, commit, or corpus replacement was performed.

## Canonical root assembly and evidence boundary — 2026-10-19

`assemble_drawing_roots` now builds a `Drawing` directly from the existing fit
results and source paint, binds pre-fit corner declarations, publishes components,
and retains immutable source observations and root-path evidence. It does not
construct `VectorRegion` records or round-trip through the working normalizer.
`build_geometry_evidence` now snapshots canonical drawing/metadata/observations;
the Python engine uses that path directly. Neither assembly nor recipe values
imports the working region class.

The remaining optimizer/report input codecs are explicitly scoped to
`optimizer/assembly_binding.py`; no old entry-point aliases are retained. The
working-to-evidence factory is needed only by optimizer-oriented test fixtures
and lives in `tests/_working_evidence.py`, not the packaged engine. Replay still
uses disposable inputs for its existing working orchestration; this change
removes the root-construction round trip, not the entire replay bridge.

Three new canonical assembly/snapshot cases observed RED-to-GREEN. The focused
assembly, engine, source evidence, topology and recipe group passes 120 tests.
The final full source suite passes 1,255 tests. Corpus-cache integration failures
were corrected using a cache-local immutable-mapping pickle codec; retained
canonical mappings remain immutable. One test now compares hex colors by value
rather than casing. The compiled wheel is installed only in the isolated test
runtime. MCP radish (21.82s) and Vbird (84.70s) have no artifact errors; all four
trace/refine SVGs are byte-identical to the canonical-topology checkpoint.
Final compiled suite: 1,251 passed, with exactly the four known receipt-fingerprint
failures; their audit values are unchanged. Targeted F401/F821 and diff checks
pass. No live deployment, commit, or corpus replacement. Full replay-loop and
optimizer working-model retirement remain unfinished.

## Canonical replay-loop state — 2026-10-19

The recipe loop now carries `ReplayState`: one canonical `Drawing` plus immutable
kernel annotations and source observations. Grouped automatic batches, corner
executors and remaining working handlers receive disposable projections only at
their boundary; their outputs are captured back into detached canonical values
before the next step. `ReplayExecution` no longer exposes a second working
forest. `replay_drawing` replaces the old working-valued entry point without an
alias. Remaining working consumers explicitly use `optimizer/replay_binding.py`.

Proposal capture crosses the same canonical boundary as execution between direct
operations/stages, preserving strict dependency rejection without false failures
from temporary representation differences. Numeric batch receipts remain strict.
The redundant SVG spelling in corner diagnostic `path` is observational for input
hashing; current geometry and all corner declarations remain fingerprinted.
Original-shape observations discard derived canonical cache objects rather than
retaining a second geometry authority. Capture deep-detaches cached commands.

Retain-only replay and terminal output-paint constraints operate directly on
canonical geometry without synthetic raster/working-record fields. The terminal
pass shares the existing numeric color limiter and does not change geometry.
Focused verification: 136 passed, including new canonical-only replay, semantic
identity, detached-command and output-palette checks. Full source: 1,260 passed.
Compiled: 1,256 passed with exactly the four known receipt-fingerprint failures,
whose audit values remain unchanged. All seven changed execution modules were
verified as native extensions before the compiled suite.

Compiled MCP radish/Vbird return no artifact errors; all four trace/refine SVGs
are byte-identical to the assembly checkpoint. Concurrent runs took 37.10s and
116.00s. A separate instrumented radish run took 24.37s: five working projections
totalled 5.6ms, five captures 8.3ms, and two terminal paint calls 1.94s. These are
not controlled before/after performance benchmarks. Targeted lint/diff checks
pass. No commit, live deployment, restart, or corpus replacement. This migrates
loop state, not every remaining working executor or proposal builder.

## Canonical source-paint replay — 2026-10-19

`drawing_paint` now owns one canonical implementation for source-paint
invalidation/refitting, visible surfaces, shared self-symmetry paint, shared root
gradients and terminal output constraints. It consumes `Drawing` and retained
source coverage, not `VectorRegion`, mutable `Shape`, or original-shape codecs.
Recipe all/changed source-paint steps retain canonical state directly. Geometry,
source observations and kernel annotations are not rewritten by paint replay.

Remaining working callers use `optimizer/paint_binding`: it binds existing
geometry/coverage, calls that same policy and publishes only fills/diagnostics.
Unchanged leaves preserve identity; modified leaves retain their original shape,
footprint and raster objects. No compatibility aliases are retained in the
canonical module. Existing private invalidation tests now verify the public
changed-refill result; their repeated-child-ID and ownership checks remain.

Four new canonical tests observed RED-to-GREEN, including replay without working
observations, occluded sampling, unchanged-fill preservation and source-support
sampling. Forty-nine focused paint/replay/topology checks pass. Final full
source: 1,264 passed (134.88s); compiled: 1,260 passed plus exactly the same four
known `test_fit_output_and_source_receipt_remain_identical` fingerprint failures
(2/3 × rounded/serif, 128.85s). Their four-case audit values are unchanged. An
obsolete test patch point moved to the shared numeric fitter, retaining its
zero-refit guard. All four changed execution modules were verified as native.

Compiled MCP Radish/Vbird succeed without artifact errors; all four v0/v1 SVGs
are byte-identical to the preceding replay checkpoint. Runs use the checkpoint's
exact recorded trace settings (Radish 8 colors; Vbird 64), taking 23.49s/93.58s.
An initial Vbird comparison used 8 colors and was not comparable; no engine fix
was needed. These are smoke timings, not controlled performance benchmarks.
Targeted F401/F821 and diff checks pass. The native build is isolated only.
Remaining geometry handlers/proposal builders and the parent review are not
declared complete. No commit, live deployment/restart or corpus replacement.

## Canonical selected-operation replay — 2026-10-19

`selected_outcomes` retains changed canonical roots and applies their ordered
delta directly to `ReplayState`. Unchanged roots retain identity; retired
subtree annotations and observations are removed. Borrowed raster/coverage is
resolved against the pinned prior state before later operations can consume it.
No working geometry, original-shape codec or fitting is needed to apply a saved
result. Public binding/command validation and generated-output registration now
consume canonical state/drawings as well.

Selected public-operation dependencies use the full canonical geometry,
annotations and source observations (not temporary drawing IDs). Existing
automatic/corner/changed-paint numeric receipt hashes remain strict and unchanged.
Unreleased recipes captured with the former selected-input hash must be regenerated;
no compatibility alias or weakened dependency check is retained.

Remaining working proposal handlers capture/project at the explicit
`optimizer/replay_binding.retain_working_operation_outcome` boundary. The former
private selected-outcome adapters were removed, not re-exported. This removes
the chosen-result replay round trip, not all working proposal execution.

Six new cases observed RED-to-GREEN; 103 focused replay tests pass. Full source:
1,270 passed (159.72s). Compiled: 1,266 passed and exactly the same four known
`test_fit_output_and_source_receipt_remain_identical` fingerprint failures
(2/3 × rounded/serif, 147.45s); their four-case audit values are unchanged. All
three changed execution modules were verified as native extensions.

Exact-settings MCP Radish/Vbird have no artifact errors and all four v0/v1 SVGs
are byte-identical to the paint checkpoint. Radish additionally replays a real
two-operation plan to v1.0 without changing its SVG; capture has no errors.
Smoke timings were 34.53s (Radish including that trial) and 99.75s (Vbird), not
controlled performance benchmarks. Targeted F401/F821 and diff checks pass.
The build is isolated; no commit or live deployment/restart. Remaining actual
handler/proposal boundaries and parent review stay open.

## Canonical proposal-loop state — 2026-10-19

`extend_recipe_with_plan` now carries `ReplayState` between actions and direct
stages, sharing canonical selected-result capture and source-paint handling with
replay. Empty plans do not project a working scene. Paint assignment and complete
paint ordering call the existing canonical actions directly; they require neither
original-shape records nor retained rasters. Geometry, detection and source-corner
preparation still use disposable projections at their actual handler boundary.
No fitting algorithm, tolerance or numeric receipt policy changes in this pass.

Two new cases observed RED-to-GREEN and 99 focused replay tests pass. Final source:
1,272 passed (161.82s). The final isolated native wheel has 1,268 passed and only
the same four known 2/3 × rounded/serif receipt-fingerprint failures (143.06s).
Their four-case audit values remain identical to the selected-result checkpoint;
the four execution/binding modules were verified as native extensions.

Exact-settings MCP Radish/Vbird capture has no artifact errors and all four v0/v1
SVGs are byte-identical to the selected-result checkpoint. Radish's real two-op
manual plan also replays to v1.0 with an identical SVG and no artifact errors.
Smoke timings were 29.56s (Radish including the manual trial) and 97.16s (Vbird),
not controlled benchmarks. Targeted F401/F821 and diff checks pass. The initial
trial wheel/source run were superseded by the final test-helper cleanup; only
the final wheel was installed, in the isolated runtime.
The selected-result working capture helper now has no production consumers and
has moved to `tests/_working_evidence.py`, without a packaged compatibility alias.
This completes proposal-loop state migration, not the remaining geometry/corner/
automatic handler projections. No commit or live deployment/restart.

## Canonical geometry proposal handlers — 2026-10-19

Region path/primitive fitting, alignment, clone assignment, explicit symmetry and
merge/split now call their existing canonical actions directly. The shared
`publish_drawing_state` owns observation inheritance and annotation invalidation;
the remaining working publication adapter delegates to it rather than duplicating
those rules. Source rasters are read independently of the original-shape codec.

Five new cases observed missing-codec failures before the change and then passed.
The broader run exposed the existing implicit unmaterialized self-symmetry
preparation requirement. That remains an explicit scoped working boundary until
its clone-remapping/numeric receipts migrate; it is not silently omitted. After
preserving it and verifying a null symmetry observation is not a relationship,
139 focused tests pass. The MCP executor-error test now injects into the canonical
primitive handler instead of the bypassed working dispatcher; its error assertions
are unchanged. Seven focused error/new-case checks pass.

Final source: 1,278 passed (262.80s). Final compiled wheel: 1,274 passed and exactly
the same four known 2/3 × rounded/serif receipt-fingerprint failures (232.26s).
The four-case audit is identical to the prior checkpoint. Five execution/binding
modules load as native extensions. Exact-settings MCP Radish/Vbird have no artifact
errors, with all four v0/v1 SVGs byte-identical to the canonical-proposal checkpoint;
Radish's two-operation manual plan also preserves its v1 SVG exactly. Timings under
concurrent validation load are 88.40s/128.27s, not performance benchmarks.
F401/F821 and diff checks pass. Initial trial runs were superseded by the null-marker
guard and updated error-injection point; final validation uses the corrected source
and `canonical-geometry-proposal-final` wheel in the isolated runtime only.

Direct path source correspondence, corner, automatic execution and unmaterialized
self-symmetry preparation boundaries remain open, as does the parent review. No
fitting policy change, commit or live deployment/restart.

## Canonical path correspondence and corner replay — 2026-10-19

Direct path operations now call the canonical path executor with retained source
spans and lineage metadata. The recipe's working public-action dispatcher was
deleted, not aliased. Display/source command and feature lookup consume ReplayState
directly, including source-bound corner preparation against base-version commands.
Corner recipes publish canonical geometry/state; their seam downstream batches
remain explicit temporary inputs, and existing numeric input receipts stay strict.
The corner recipe builder no longer uses working replay for ordinary selection,
raw execution, receipt rebinding or final validation.

Three new cases observed RED-to-GREEN for codec-free path removal/correspondence
and canonical corner publication. The topology-remap fixture now supplies its
already-captured canonical state to the changed private lookup; assertions are
unchanged. One new lineage expectation was corrected to the existing remove-op
renumbering semantics rather than changing the operation. The first full run found
seven guards on the deleted dispatcher and a real missing component `path_ids`
annotation. Guards now target canonical proposal dispatch with the same assertions;
shared publication derives IDs/fill rule for newly generated concrete path leaves,
without treating clones as paths. Existing component-cache assertions remain intact.
The final focused group has 129 passing tests.

Final source: 1,281 passed (145.16s). Compiled: 1,277 passed and exactly the same
four known 2/3 × rounded/serif receipt-fingerprint failures (134.87s). The four-case
audit is identical to the preceding checkpoint; seven execution/binding modules
were verified as native extensions. Exact-settings MCP Radish/Vbird have no
artifact errors and all four v0/v1 SVGs are byte-identical to the geometry-proposal
checkpoint. Radish additionally replays a real path `keep` plus fill plan to v1.0
with an identical SVG. Smoke timings are 29.94s/91.86s, not controlled benchmarks.
F401/F821 and diff checks pass. The initial trial build/runs were superseded by
the component-publication fix; the final `canonical-path-corner-final` wheel was
built and tested only in the isolated runtime.
Automatic batch execution and unmaterialized self-symmetry preparation remain open;
numeric receipt projection and downstream seam rebinding are not declared retired.
No commit or live deployment/restart.

## Canonical basic automatic replay — 2026-10-19

`drawing_automatic.replay_automatic_batches` owns the canonical recipe boundary.
Eligible root-leaf groups containing only retained primitive choices and corner
observations install those choices directly, without numeric action execution or
semantic detection. Input validation remains the exact numeric receipt check,
shared with the old executor. Accepted diagnostic payloads are shared as well.
Mixed, scoped, named-output-input and other geometry families remain explicit
`optimizer/replay_binding.replay_numeric_batches` calls; this is not full automatic
executor retirement or a performance claim. Producer registration now reads the
already-canonical ancestry map, removing its extra whole-scene capture helper.

Four new cases observed missing-entry-point RED and then passed, covering empty
programs, retained primitives, stale input rejection and geometry-preserving
observations with exact numeric-receipt parity. Fixture setup was corrected to
provide stable engine IDs and an explicit observation receipt rather than relying
on corner detection after canonical capture. 121 focused tests pass (6.90s).
Final source: 1,285 passed (130.17s). Compiled: 1,281 passed and only the same
four known 2/3 × rounded/serif receipt-fingerprint failures (131.62s); their
four-case audit values remain identical. Nine execution/binding modules load as
native extensions. Exact-settings MCP Radish/Vbird have no artifact errors and
all four v0/v1 SVGs are byte-identical to the path/corner checkpoint. Radish's
path-keep-plus-fill plan also replays to v1.0 with an identical SVG. Smoke timings
are 28.16s/92.40s, not controlled benchmarks. F401/F821 and diff checks pass.
The wheel is installed in the isolated test runtime only; no commit or live
deployment/restart. Other automatic geometry/proposal families, self-symmetry
preparation, numeric receipt projection and seam rebinding remain open, as does
the parent review.

## Remaining adapter inventory — 2026-10-20

This inventory supersedes earlier shorthand such as "remaining clone/symmetry
families". Their geometry implementations are already shared engine operations;
they must not be listed as algorithms still needing migration.

| Boundary | Actual remaining responsibility | Status |
| --- | --- | --- |
| `drawing_automatic.replay_automatic_batches` | Install selected geometry and diagnostics in canonical state | Direct single-leaf replay includes scoped children, primitive sequences, simplify, smooth, declared corner normalization, intrinsic symmetry and observations |
| `replay_identity.bind_automatic_inputs` and `canonical_input_revision` | Validate semantic input receipts and source-mask bindings | Direct from retained state; obsolete working-region receipt wrappers deleted |
| `optimizer/replay_binding.replay_numeric_batches` | Preserve multi-region transactions, named-output dependencies, replacement masks and complex publication | Explicit boundary around already-shared relationship, topology, occlusion and seam operations; not a second implementation of those algorithms |
| `drawing_replay._batch_inputs_from_start` and `_rebase_downstream_batches` | Rebind downstream seam endpoints after corner declarations | Still consumes working snapshots; current exact-geometry restriction remains unchanged |
| `drawing_refine` / optimizer framework and proposal passes | Discover, score and record automatic choices from raster observations | Working proposal boundary remains; canonical Drawing is still authoritative between recipe steps |
| `drawing_adapter` and optimizer binding modules | Project canonical selections and source observations into the above consumers | Remove consumers before deleting these codecs; relocating them is not retirement |

The direct path route and proposal route now share `select_automatic_paths`;
the operation settings and numerical kernels are not duplicated. Primitive
replacement explicitly retires old path annotations, matching working publication.
No fit thresholds, acceptance rules or detector behavior changed.

Ruling: keep whole-group numeric replay for scoped or otherwise unsupported
transactions. A two-parent fixture with repeated child IDs showed why: after a
child becomes a primitive, the historical receipt contains the preserved parent
footprint and mask snapshot, whereas canonical projection derives that footprint
from the children. Enabling the direct route caused the next exact receipt check
to reject the result. Do not bypass that check or change footprint semantics as
an incidental adapter cleanup. The scoped regression retains exact result and
receipt parity through the explicit transaction boundary.

The first full-suite run also exposed the same receipt coupling without a parent:
four integration checks failed after primitive replacement because a later empty
batch hashes the optimizer's SVG primitive footprint, not Skia's native conic
encoding. Primitive groups with any later batch therefore retain whole-group
numeric execution as well. This preserves the existing validation contract; it
does not retry after a validation failure or weaken stale-input detection.

This closes the redundant direct path-publication round trip, not all transient
working-record use. Full retirement requires one explicit receipt/transaction
contract change, followed by migration of the remaining consumers above; it does
not require reimplementing clone, symmetry, topology or seam geometry.

Verification: six additional regression cases cover the three direct path
choices, intrinsic symmetry, primitive-annotation retirement and repeated scoped
child IDs. The targeted integration rerun passed 55 tests; the final full source
suite passed 1,291 tests (965 warnings, 132.09s). F401/F821 and diff checks pass.
The optional renderer is absent in this source environment, so this is not a
render-scored quality or native-wheel certification. No native rebuild, commit,
deployment, server restart or corpus overwrite was performed for this cleanup.

### Semantic receipt follow-up — 2026-10-20

The earlier whole-group guards for scoped children and primitive sequences are
now removed. Their root cause was an input-identity contract that hashed derived
footprint encodings, including stale parent footprints. The new internal
`vectormark.replay.input.v2` payload hashes typed region inputs, selected geometry,
original source geometry, raster/coverage observations, fill, ordering, identity,
and semantic diagnostics, but not a disposable footprint. Numeric proposal
recording and canonical replay share that contract and its validation error.
No legacy-receipt fallback is retained; prior internal receipts must be recorded
again, consistent with the unreleased API policy.

`canonical_input_revision` and automatic input binding construct no VectorRegion
objects. The old `automatic_batch_masks` and `replay_state_input_revision` names
are deleted, with callers moved to the canonical owner. A shared leaf Shape codec
serves existing numerical consumers and receipt projection; cloning/paint codecs
are not duplicated. This retires receipt projection, not the remaining complex
transaction executor or proposal-discovery boundary.

Final verification: 1,298 source tests pass (138.59s); compiled wheel has 1,294
passing tests and only the same four known 2/3 × rounded/serif feature-fingerprint
failures (126.89s), with all four fingerprint pairs unchanged from the preceding
native build. Six changed modules were verified as native extensions; 88 focused
compiled checks pass. Lint and diff checks pass. Real in-process FastMCP requests
reused the exact Radish/Vbird inputs and settings: all four v0/v1 SVGs are
byte-identical to the preceding automatic-replay checkpoint. Smoke timings were
18.77s/82.18s, not controlled benchmarks. Reports/SVGs are separate artifacts under
`native-canonical-receipts`; corpus outputs were not overwritten. The verified
wheel is installed only in the isolated test runtime, not the live MCP server.

### Canonical seam rebinding and transaction outputs — 2026-10-20

Seam input templates, endpoint correspondence and downstream receipt rebasing now
retain ReplayState snapshots instead of projecting entire working scenes. Scoped
endpoint lookup uses full region ancestry. Exact command-topology rejection is
unchanged; annotation restoration publishes typed geometry without rounding its
coordinates or changing siblings. Intermediate seam templates deliberately skip
terminal topology normalization, matching their prior execution order.

ReplayInputResolver now owns revision validation and named-output addressing for
both canonical and remaining numerical transactions. Pass inputs remain snapshots;
named inputs read raw producer selections before accepted-operation diagnostics
are appended. Eligible leaf transactions now include symbolic clone assignment,
using the existing assign_clone operation and preserving independent target paint
and observations. No fitting or clone-detection algorithm changes are included.

This does not retire the working transaction boundary for structural multi-region
operations, inline clones, covering simplification, or proposal discovery.
The final source suite passes 1,305 tests (983 warnings, 130.70s). The compiled
suite passes 1,301 with only the same four known feature-fingerprint failures
(120.95s); their hash values match the preceding receipt build exactly. Six
changed modules are confirmed native extensions. Lint and diff checks pass.
Exact-settings FastMCP Radish and Vbird checks retain all four v0/v1 SVGs byte for
byte (18.60s/77.08s, not controlled performance measurements). Artifacts are
separate under `native-canonical-seam-multi`. The final wheel is installed only
in the isolated test runtime. No commit or live deployment is included.

### Structural automatic transaction publication — 2026-10-20

Canonical replay now publishes multiple selected replacements, retires consumed
roots, preserves pass-snapshot mask ownership and registers generated descendant
outputs. This enables covering simplification, inline clone copies, mirrored
pairs and occlusion reconstruction without a working scene. Existing geometry
operations and numeric affine projection policy are reused, not reimplemented.

The numerical proposal framework and canonical replay share one replacement-mask
assignment kernel, including preserved input masks, split rasterization and union
ownership. Generated-tree observation capture is also shared; its canonical path
does not allocate VectorRegion objects. An explicit empty metadata projection is
preserved rather than gaining public-path defaults during replay. Scoped output
observations retain their full parent ancestry, and newly generated named children
can feed later inline-copy operations in the same batch.

Eight regression cases cover these routes and compare geometry, source observations
and exact receipts with the existing executor while prohibiting working-scene
execution. The first four and two subsequent scope/named-source failures were
observed before their fixes. A final clone-instance case also exposed the need
to retain resolved instance-outline projection and corner-ingress diagnostics,
rather than reclassifying the copy as its concrete source primitive. Both
consumers now share the instance projection policy. A structural snapshot-isolation
case then verified that a later pass input restores original observations even
after that identity was replaced by a generated branch. Canonical transaction
inputs now bind immutable geometry/observation/metadata bundles; named outputs
capture their own bundles before publication diagnostics are added. Closure source
verification passes 1,313 tests (983 warnings, 144.97s); 64 focused compiled checks
pass. The full native suite has 1,309 passes and only the same four known
2/3 × rounded/serif feature-fingerprint failures (983 warnings, 141.53s), with
all twelve printed fingerprint tokens unchanged from the prior seam/multi build.
Five changed modules are confirmed native extensions. Exact-settings FastMCP
Radish/Vbird v0/v1 SVGs remain byte-identical (21.37s/92.18s; not controlled
benchmarks). Closure artifacts are under `native-canonical-structural-closure`;
the corrected wheel is installed only in the isolated runtime. Earlier structural
wheels and intermediate verification results are superseded. Lint and diff checks
pass. Structural self-symmetry, seam/topology and
dependent-reference refresh transaction families, plus proposal discovery, still
retain an explicit numerical boundary; this is not whole-model retirement.
No commit, live deployment or corpus overwrite is included.

### Remaining automatic transactions and terminology retirement — 2026-10-20

Checkpoint `27ec035` commits the accumulated engine, evaluation, plugin and
approved reference-artwork work. The subsequent retirement batch moves recorded
self-symmetry, seam constraints, retained topology and dependent-clone refresh
transactions onto canonical Drawing/ReplayState publication. The whole-scene
numeric replay fallback is deleted. Independent geometry operations remain owned
by DrawingEngine; automatic refinement selects proposals and their order.

The former `optimizer` package is now `refinement`, including active imports,
option/report names, scripts and tests. `VectorRegion` is now `RegionSample`, an
explicitly disposable numerical record for discovery/scoring and interoperability,
not a second retained drawing model. This naming cleanup does not claim those
remaining numerical projection boundaries have disappeared. No unreleased API
compatibility aliases are retained. Historical plan sections above are unchanged.

Eleven exact-receipt regression cases prohibit working-scene replay and cover
seams, direct/nested dependent clones, self-symmetry, single-/multi-transform clone
component expansion, hole-first paths, stale correspondence and diagnostic-only
corner retirement. Shared helpers own topology correspondence and retired-feature
mapping. A fresh-context reviewer found the hole-first and retired-feature cases;
both are now regression-tested and independently verified.

Final verification: 1,324 source tests pass (983 warnings, 125.80s). The frozen
compiled wheel has 1,320 passes and exactly the same four known 2/3 × rounded/serif
feature-fingerprint failures (983 warnings, 122.32s); all printed fingerprint
tokens match the earlier native baseline exactly. Six changed modules are
confirmed native extensions. UI build, packaged-skill consistency, focused lint
and staged whitespace checks pass. Exact-settings in-process FastMCP Radish and
Vbird runs preserve all four v0/v1 SVGs byte-for-byte (17.90s/78.01s, not controlled
benchmarks). Artifacts are separate under `native-canonical-retirement`; the
verified wheel is installed only in the isolated test runtime. No live deployment,
server restart, push or corpus overwrite is included.

### Native evidence-receipt parity — 2026-10-20

The four source/native feature-receipt failures were traced to scalar coercion,
not different fitted curves. At the automatic-evidence boundary Python retained
the caller's integer `canvas_extent=1254`, while Cython's `float` parameter changed
it to `1254.0`; JSON therefore hashed different settings. The receipt-producing
entry point now accepts `float | int`, preserving the caller's representation
while numerical detection still consumes a floating-point value. The existing
21-case feature tests pass in both runtimes without changing any baseline hashes.
Full verification passes 1,324 tests in each runtime (983 warnings;
121.00s source, 123.79s compiled). The verified wheel is installed only in the
isolated test runtime. No server deployment is included.

### Region-aware shape quality foundation — 2026-10-20

`DrawingEngine.assess_revision` now assesses canonical drawings against both their
immediate parent and a separately retained reference. Complete region/branch
correspondence is explicit for merges and splits; absent or duplicate lineage
cannot hide a missing material. The report includes per-region step/reference
geometry assessments plus total coverage. Cumulative reference budgets are
independent and optional, so an inaccurate baseline cannot silently prohibit an
intentional improvement. Successful geometry is labelled `geometry_passed`, never
full fidelity approval; paint remains explicitly `not_assessed`. Invalid metrics
serialize safely, and unresolved/cyclic references fail locally rather than
recursing indefinitely.

Automatic proposal acceptance now also checks stable material boundaries. A
shared-edge shift from x=5 to x=8 no longer passes simply because the union is
unchanged. Seam repair declares a separate per-material boundary policy using
its existing seam/junction tolerances, while retaining the previous joint
coverage budget. Closing a long gap intentionally removes internal union edges;
that union Hausdorff distance must not be confused with movement of either
material's boundary. Explicit caller limits remain binding.

This is a shape-quality foundation, not completion of Slice 4: original-raster
appearance, detailed design-constraint/complexity reports and broader evaluation
integration remain open. Canonical automatic proposal discovery and removal of
its remaining disposable numerical adapters are also still open. The old Slice 3
selected-outcome/reproposal checklist above is now reconciled with its existing
controlled-proposer and selected-fit regression tests; those were already done.

Verification: 1,338 source tests pass (149.13s) and 1,338 compiled tests pass
(138.92s), each with 983 warnings and no failures. Fourteen focused regression
cases cover the shape report, material gate and separate seam budgets. One
existing test spy was updated for the additional region-policy argument without
changing its assertions. Lint, whitespace and packaged-skill consistency checks
pass. Compiled in-process FastMCP Radish/Vbird runs preserve all four v0/v1 SVGs
byte-for-byte (19.01s/94.80s; not controlled benchmarks). Artifacts remain separate
under `native-region-quality`; no live server deployment or corpus overwrite.

### Canonical proposal selection and primitive discovery — 2026-10-20

Automatic selection and recorded replay now share `AutomaticBatchApplication`.
Preparation captures detached canonical region/observation/metadata bundles from
the pinned pass snapshot. Acceptance checks the binding and assessment, then
installs that exact prepared result without invoking the action again. Rejected
choices neither consume input identities nor mutate retained evidence. Named
outputs register atomically, so a later conflicting output cannot leave an
earlier name from the failed transaction visible.

`drawing_proposals` discovers and scores primitive candidates directly against
Drawing/ReplayState. Public `detect_primitives` plan operations now use this
route, including child scopes with repeated local IDs. Shared proposal identity
rules are extracted rather than copied from the existing numerical selector.
Optional actions with no candidate yield a typed rejection; invalid prepared
geometry fails assessment instead of escaping as a geometry-resolution error.

This does not retire every automatic detector or numerical adapter. The default
multi-pass discovery orchestrator and non-primitive public detection still use
their explicit sampling boundary. The remaining quality, accountable-fill and
agent/holdout evaluation items above remain open.

Verification: 1,361 tests pass in both source (164.18s) and compiled (165.84s)
runtimes, each with 985 warnings and no failures. Twenty-three new cases cover
prepared acceptance, shared identity rules, atomic outputs, canonical discovery,
public plan integration, scoped addressing and fail-closed geometry. Five changed
modules are confirmed native extensions. Focused lint and staged whitespace
checks pass. Exact-settings in-process FastMCP Radish/Vbird runs preserve all four
v0/v1 SVGs byte-for-byte (21.87s/117.52s; concurrent validation, not benchmarks).
The frozen wheel is installed only in the isolated test runtime; artifacts are
under `native-canonical-proposal-selection`. No live deployment, push, server
restart or corpus overwrite is included.

### Canonical path-cleanup discovery — 2026-10-20

Smoothing, simplification and corner normalization can now discover and select
choices directly against retained Drawing/ReplayState. Public `update_geometry`
cleanup stages use that route without projecting a sampling scene. Child-first
scope traversal is shared with primitive discovery; operation position,
repetition and action settings remain unchanged.

Simplification retains its existing preferred covered-cutout choice and
topology-preserving fallback. The fitter runs once during discovery; its detached
prepared result is then validated against the input revision, scope and rebound
intent/settings before acceptance. Covered-cutout geometry policy and retained
corner-evidence policy each have one shared owner across the remaining old and
new discovery paths. Stale corner evidence is observation-only, never redetected
from fitted curves. Clone sources remain conservatively protected by canonical
identity, at the cost of skipping some potential source simplification.

Replacement rasterization now reads canonical geometry and selected dependency
context directly rather than converting through Shape/RegionSample helpers.
The default multi-pass detector orchestrator, structural/relationship discovery
and the broader quality/fill/evaluation tasks remain open.

Verification: 1,378 source tests pass (165.83s) and 1,378 compiled tests pass
(159.38s), each with 985 warnings. Seventeen new guards cover discovery,
observation-only stale declarations, cutouts, clone-source protection, prepared
choice reuse/binding, public cleanup and direct replacement rasterization.
Focused lint and whitespace checks pass. Exact-settings FastMCP Radish/Vbird
v0/v1 SVGs are all byte-identical to the retained baseline (18.62s/111.22s;
concurrent checks, not benchmarks). A separate compiled Radish plan successfully
normalizes/simplifies r2/r3/r4 to v1.0 with the sampling projection forbidden;
its 2,524-byte SVG parses successfully. Artifacts are separate under
`native-canonical-path-discovery`; the wheel is installed only in the isolated
runtime. No live deployment, server restart, push or corpus overwrite.

### Accountable fill fitting — 2026-10-20

`fit_fill` now returns a typed `FillFitResult`, not an unexplained paint value.
Its selected fill is separate from the decision: flat-color evidence, accepted
gradient, forced flat, fallback flat, or empty support. Reasons distinguish the
existing dominant-color and low-spread heuristics from disabled gradients, absent
candidates, rejected error budgets and invalid numeric evidence. The report
retains sample counts, whether a gradient search ran, the configured gate and
available mean/median errors. Non-finite diagnostics serialize as null; invalid
gradient budgets do not run a search or approve a candidate.

Canonical root assembly, visible-source refitting and shared self/root-gradient
assignment retain that selection evidence in fill diagnostics. Existing numeric
consumers explicitly access the result's `.fill`; no unreleased compatibility
wrapper is retained. The selected fills and valid-input fitting thresholds are
unchanged. A recorded fitting decision is not an independent original-raster
appearance assessment or designer-quality approval: that evaluation work remains
open, as do the remaining automatic detector migrations.

Verification: 1,391 source tests pass (138.86s) and 1,391 compiled tests pass
(155.10s), each with 985 warnings. Thirteen new cases cover evidence/forced/
fallback/empty outcomes, accepted gradient metrics, non-finite errors, invalid
budgets and canonical/MCP propagation. The 103-case focused fill/paint/engine
group passes; the final 13 guards also pass separately in both runtimes after
strengthening the MCP propagation assertion. Existing numeric-fill tests now
explicitly inspect `.fill`; their geometry/color assertions are unchanged.
Four changed fill/paint modules are confirmed native extensions. Focused lint
and whitespace checks pass. Exact-settings FastMCP Radish/Vbird v0/v1 SVGs remain
byte-identical (17.84s/105.27s; not benchmarks). Artifacts are separate under
`native-fill-outcomes`; the wheel is installed only in the isolated runtime.
No push, live deployment, server restart or corpus overwrite.

### Canonical clone discovery and shared matcher — 2026-10-20

Clone congruence matching now has one geometry-only owner, `clone_detection`,
consuming typed footprint inputs and returning source/target/transform matches.
The remaining numerical pass delegates to it instead of owning a second
algorithm. Canonical discovery and public `detect_clones` plans use the same
matcher without constructing RegionSample inputs. Targeting either a source or
an instance retains the existing family-selection meaning; paint is independent
of the reused geometry, and scoped local IDs remain distinct.

Matching tolerances, reflection/rotation candidates and tie-breaking are
unchanged. A new real-geometry regression exposed a family-before-pair defect:
the strict-pair phase could reclassify an already selected family source,
creating a chain. Selected family sources are now protected from that later
phase. This preserves family priority; its conservative cost is foregoing a
strict-pair fallback when a preferred family's proposals later fail an outer
quality gate. Replay guards now prohibit the new matcher owner, not a retired
private function. Remaining symmetry/occlusion/seam discovery and default
multi-pass orchestration are still migration work.

Verification: 1,401 source tests pass (139.06s) and 1,401 compiled tests pass
(148.40s), each with 985 warnings. Ten new cases cover translation, rotation,
reflection, non-chained families, independent paint, source/instance targeting,
scoped IDs and public plan discovery without sampling projection. The initial
79-case integration group and final 20-case matcher/pass guard group pass.
Three matcher/discovery modules are confirmed native extensions; lint and
staged whitespace checks pass. Exact-settings FastMCP Radish/Vbird v0/v1 SVGs
remain byte-identical (22.08s/98.93s; not benchmarks). The final wheel and
artifacts under `native-canonical-clone-discovery` supersede the initial clone
build made before the family-source fix. No live deployment, push, server
restart or corpus overwrite.

### Canonical symmetry discovery and shared numeric checks — 2026-10-20

The symmetry axis candidates, geometric residuals, reflection matching and
self-reconstruction fitter selection now belong to `symmetry_detection`.
The numerical pass delegates to that owner; its tolerances, candidate ordering
and fitter preferences are unchanged. Canonical discovery and public
`detect_symmetry` plans operate on retained drawings without projecting a
RegionSample scene. Accepted recipes retain the chosen axis and fitter, not
instructions to repeat discovery during replay.

Mirror pairs retain independent paint. Intrinsic symmetry keeps the geometry
unchanged. Material partitions and implementation-only compound children keep
their existing eligibility restrictions; polygon primitives are not silently
promoted to path reconstruction. Child-first scopes preserve repeated local
IDs. Same-scope dependent clones participate in the source reconstruction's
quality transaction. A source with dependents outside that scope is
conservatively protected from automatic reconstruction until cross-scope
orchestration can assess the complete transaction; verified intrinsic symmetry
may still be declared. This may skip useful reconstruction, but does not claim
that a local quality check covers remote instances.

Fourteen new cases cover mirror/source/target selection, intrinsic and selected
self-symmetry, material eligibility, public plans without projection, polygon
eligibility, local and cross-scope clones, repeated IDs and numerical-selection
parity. All 1,415 source tests pass (147.63s) and 1,415 compiled tests pass
(149.51s), each with 985 warnings. The 155-case focused integration group
passes, and all 14 new cases also pass separately in the compiled runtime.
The shared kernel, discovery, public replay and numerical pass are confirmed
native extensions; focused lint and staged whitespace checks pass.
Exact-settings FastMCP Radish/Vbird trace and refine SVGs are all byte-identical
to the retained baseline (18.05s/105.06s; concurrent verification, not a
benchmark). The wheel is installed only in the isolated runtime, with separate
artifacts under `native-canonical-symmetry-discovery`.
Default multi-pass discovery still uses the numerical orchestrator; occlusion,
seam and cross-scope orchestration migration and the broader quality/evaluation
work remain open. No live deployment or corpus overwrite.

### Canonical occlusion discovery and shared graph construction — 2026-10-20

Occlusion proposal construction now has one raster-evidence owner,
`occlusion_detection`. Both the remaining numerical pass and canonical
discovery use it, retaining the existing adjacency/component traversal,
symmetry option, reconstruction thresholds, global layering and child-ID
allocation. Canonical discovery reads retained masks, paint and coverage
without constructing a RegionSample scene. The selected graph names completed
primitives, retained source artwork and dependent circle intersections; replay
executes those choices without rerunning discovery. Annulus cutouts and colored
cavity artwork retain their exact source dependencies.

Coverage decoding is shared by canonical receipts, the remaining projection
adapter and occlusion discovery, preserving tile origins and immutable values.
Numerical discovery receives detached raster/array-coverage inputs. Canonical
occlusion currently protects symbolic clone instances and live clone sources
from component retirement until explicit remapping can assess and publish the
complete relationship transaction. This may defer a useful reconstruction; it
does not claim that a local replacement safely retires external references.

Twelve new guards cover exact selection/evidence parity, symmetry disabling,
source immutability, replay without detection, material targeting, singleton
inputs, tiled/full coverage, scoped IDs, failed-quality rollback, clone-source
protection and annulus/cavity dependencies. All 1,427 source tests pass
(199.99s) and 1,427 compiled tests pass (193.77s), each with 1,035 warnings.
The 131-case integration group passes, and all 12 new guards pass separately
in the compiled runtime. Six changed core modules are confirmed native
extensions; focused lint and staged whitespace checks pass. Exact-settings
FastMCP Radish/Vbird trace and refine SVGs remain byte-identical to the retained
baseline (27.98s/122.72s; concurrent checks, not benchmarks). The wheel is
installed only in the isolated runtime, with separate artifacts under
`native-canonical-occlusion-discovery`.

Default multi-pass orchestration remains on its
numerical route; seam discovery, cross-scope/default orchestration and broader
quality/evaluation work remain open. No live deployment or corpus overwrite.

### Canonical seam discovery and full-ancestry publication — 2026-10-20

Seam discovery now has one geometry-only owner, `seam_detection`. The initial
numerical stitch pass and canonical/public stitch action share ordered pair
discovery, composite adjacency checks, paint ordering, endpoint clustering and
the existing execution math. Canonical discovery reads retained drawing values
without constructing a RegionSample scene, records the selected constraints,
and reuses the prepared execution during quality selection. Replay executes
those constraints without rediscovering seams. Topology checks remain
predicates before publication; they are not export-time geometry repairs.

Leaf publication now uses full ancestry rather than a bare local ID. A new
regression demonstrated that numerical replay could otherwise overwrite two
different nested leaves with the same edit. Targeted canonical discovery also
includes an enclosing owner when one of its descendants is selected. PathOps
reconstruction correspondence uses the same full-path key as execution rather
than incorrectly treating rebuilt commands as positional mutations.

Underlap still requires a shared owner and its actual boundary clip. No larger
clip is invented for standalone leaves. Sixteen new cases cover scene-free
discovery/replay, exact selection parity, material eligibility, public plans,
individual/multiple/nested targets, repeated local IDs, underlap constraints and
post-symmetry selection. Native validation caught and corrected an empty-child
fallback type mismatch that ordinary Python execution accepted.

All 1,443 source tests pass (139.87s) and all 1,443 compiled tests pass
(136.85s), each with 1,035 warnings. The 93-case focused integration group
passes; all 16 new guards also pass separately in the corrected compiled
runtime. Six changed core modules are confirmed native extensions. Focused
lint and staged whitespace checks pass. Exact-settings FastMCP Radish/Vbird
trace and refine SVGs remain byte-identical to the retained baseline
(18.09s/91.00s; concurrent checks, not benchmarks). The corrected wheel is
installed only in the isolated runtime, with separate artifacts under
`native-canonical-seam-verified`. No live deployment or corpus overwrite.

Default multi-pass orchestration and remaining explicit symmetry projection
are not retired by this checkpoint. Cross-scope orchestration and broader
appearance/constraints/complexity/evaluation work remain open.

### Default DrawingEngine refinement on canonical actions — 2026-10-20

`PythonDrawingEngine.refine_automatically` now obtains its pre-output-palette
input through canonical recipe replay and runs the independent canonical
discoverers, without materializing a RegionSample scene. The engine retains
their accepted batches and resolves original paint only after the complete
geometry sequence. The final output-palette projection remains downstream of
the recorded geometry input.

`refinement_pass_names` is the shared owner of stage order and toggles for this
route and the remaining numerical pipeline: occlusion, component publication,
primitives, clones, symmetry, linelet simplification, smoothing, optional corner
normalization/observation, then full-scene stitching. Canonical scope traversal
records the actual pass name and ordinal, so empty or observational stages do
not collide merely because their input geometry is unchanged.

Component discovery now shares preparation with strict topology replay. It
records surviving path ownership and reuses the prepared result for selection;
replay still rejects missing or stale correspondence. This is not permission
to infer post-fit corners or repair geometry during export.

All 1,453 source tests pass (156.40s) and all 1,453 compiled tests pass
(153.28s), each with 1,035 warnings. Ten new canonical-orchestration guards
pass in both runtimes; the 92-case focused integration group passes. The MCP
control-threading test now observes the real engine/canonical route instead of
patching the removed adapter import. Six changed core modules are confirmed
native extensions; focused lint and staged whitespace checks pass.
Exact-settings FastMCP Radish/Vbird trace and refine SVGs remain byte-identical
to the retained baseline (18.65s/87.06s; concurrent checks, not benchmarks).
The wheel is installed only in the isolated runtime, with separate artifacts
under `native-canonical-default-refinement`. No live deployment or corpus
overwrite.

The standalone numerical pipeline, its adapter-facing automatic wrapper,
remaining explicit self-symmetry projection and cross-scope clone transaction
migration remain open. In particular, canonical symmetry still conservatively
declines a local reconstruction whose clone dependents are outside its scope;
this checkpoint does not claim that the older outer clone-refresh transaction
has been ported. The broader appearance/constraints/complexity/evaluation gates
also remain open.

### Cross-scope self-symmetry and active wrapper closure — 2026-10-20

Self-symmetry discovery now lifts a nested source and all of its external
clone owners to their least common scope. The typed reconstruction records
the exact source, dependent clone addresses and replaced owner roots. One
prepared selection assesses both the combined silhouette and every affected
source/instance boundary before publication. Rejection publishes neither the
source rewrite nor a clone refresh. Strict replay requires every dependent
clone and owner; a same-numbered root cannot capture a nested source binding.
Clone transforms and independent fills remain retained values.

Canonical self-symmetry materialization now owns the explicit public-action
preparation boundary too. It unions retained vector halves, preserves cutouts,
publishes source observations/lineage through the same helper as selected
reconstruction, and remaps exact canonical clone source IDs. No temporary
RegionSample scene is built in this public proposal path. A disconnected
reconstruction that cannot remain a concrete symbolic clone source is declined
rather than retiring live source identity without a component-remap operation.
New cross-scope records require canonical replay; the numerical test kernel
rejects them rather than silently applying a nested edit as a root replacement.

The diagnostic automatic adapter, standalone pipeline, cached corpus report
and pass-panel generator now use canonical selection. RegionSample conversion
remains at their existing trace/report boundaries, not as a second refinement
algorithm or authoritative scene. Standalone tracing assigns stable drawing
IDs when roots are constructed. Cached reports bind existing trace labels to
those IDs without retracing. Numerical kernel/configuration helpers still
serve their direct tests and the separate initial-trace preparation boundary;
this is not a claim that every private adapter class has been deleted.

The standalone square fixture now retains a path when the existing canonical
primitive gate rejects a rectangle 1.685 px from that path under a 1 px budget.
The clone regression checks shared unpainted geometry and separate colors,
not an unconditional `<rect>` encoding. No budget was loosened to preserve that
old representation. Topology preparation now leaves native primitives alone
instead of treating them as compound paths.

All 1,463 source tests pass (180.23s) and all 1,463 compiled tests pass
(178.95s), each with 1,037 warnings. Nine new transaction/materialization
guards pass separately in both runtimes; the final cached-report guard also
passes and is included in both full runs. The 208-case focused group passes.
Eight changed core modules are confirmed native extensions; focused lint and
staged whitespace checks pass. Exact-settings compiled FastMCP Radish/Vbird
trace and refine SVGs remain byte-identical to the retained baseline
(19.26s/99.00s; concurrent checks, not benchmarks). The wheel is installed only
in the isolated runtime, with artifacts under `native-canonical-crossscope-wrapper`.
No live deployment, push or corpus overwrite.

A reduced-size Daikonic pass-panel probe initially rejected partially
overlapping fitted boundaries at 128/256 px. The numerical baseline failed
identically. The trace-overlap ingress follow-up below resolves this report
entry failure. Broader appearance, constraints, complexity and
agent/generated/holdout evaluation gates remain open.

### Trace-overlap ingress consistency — 2026-10-20

Reduced Daikonic tracing produces genuine, small overlaps between separate
fitted fragments (three pairs at 128 px, intersection areas 0.004–0.084 px²).
These are not float32 Boolean residue and are not evidence that the fragments
may be discarded. Canonical root assembly already retains such boundaries as
pending compound topology, but diagnostic/report entry points were attempting
strict component publication before refinement could run.

`capture_trace_replay_state` now owns the shared trace-report boundary: stable
addresses, overlap-deferred topology publication, detached replay capture and
canonical symmetry materialization. Diagnostic automatic refinement,
standalone conversion, cached corpus reporting and the pass panel all use it.
Input commands and explicit pending ownership are preserved. This does not
silently Boolean-normalize a trace, promise that stitching resolves every
overlap, or relax the default strict validation used for geometry edits.

Four regression cases fail on the original entry error and pass with the
shared boundary. Both actual 128 px and 256 px Daikonic pass panels complete
through all enabled passes. The 38-case focused topology/materialization group
passes in both runtimes. All 1,467 source tests pass (220.95s), and all 1,467
compiled tests pass (176.92s), each with the same 1,037 existing warnings.
The compiled 256 px panel is byte-identical to its source-runtime counterpart;
all three changed engine modules load as native extensions. The wheel is
installed only in the isolated verification runtime. No deployment or corpus
overwrite.

### Retire the standalone refinement pass panel — 2026-10-20

Removed the standalone pass-panel generator after the high-resolution Radish
check exposed that it uses the separate quadratic trace baseline and does not
retain pre-fit corner provenance. Its images therefore do not evaluate the
production progressive, corner-aware trace. The 1254 px Radish run retained four
paths and 79 drawable segments through every pass; corner counts were
unavailable, not genuinely zero. Earlier successful panel runs above establish
only that the diagnostic route completed, not production fitting quality.

Removed the panel-only overlap regression case; diagnostic, standalone and
cached-report overlap coverage remains. Existing generated images are retained
as historical artifacts, not current evaluation evidence. Future pass views
must render retained DrawingEngine/MCP states with exact settings and feature
provenance; the production review/evaluation artifacts remain available.

Validation: all 1,466 source tests pass (144.28s, 1,037 existing warnings).
The 37-case focused group also passes against the compiled engine (2.49s,
two existing warnings). No engine code changed, so no native rebuild or
deployment was required for this diagnostic-only retirement.

### Production-state complexity review — 2026-10-20

The existing evaluation report now compares local per-region path/command and
corner counts against the captured parent and baseline. It reuses exact stored
production responses, parameter snapshots and canonical annotation maps; no
alternative tracing, fitting or feature detection is introduced. Derived
inspection views are rebuilt from those responses without replacing immutable
capture files. Missing/null/stale corner provenance remains unavailable rather
than becoming zero corners or classifying all commands as free.

Counts include local paths, L/Q/C/A commands, corner-owned/free commands, and
point/rounded corners. The UI supports region filtering, parent/baseline
selection and JSON download. Clones and primitives explicitly have no local
path commands. Deltas require the same drawing ID, unique region IDs and
matching nonempty source-region lineage; other cases remain unpaired. This
does not yet supply general merge/split correspondence or a quality verdict.

The saved production progressive/subpixel 2 px Radish bundle was re-rendered
without re-tracing: its existing v0.4 plan revision shows navy 19→15 and red
6→5 drawable commands, with corner counts unchanged. All 30 retained response,
recipe, raw-trace, inspection and SVG artifacts remain byte-identical. This is
a historical plan comparison, not a new automatic-fitting benchmark. Browser
checks cover selection, missing comparisons, failed trials, JSON download and
mobile table overflow. Temporary regression checks follow the existing report
plan's no-new-permanent-tests constraint. Broader appearance, design-constraint,
fill-completion and autonomous evaluation work remains open.

Validation: all 1,466 source tests pass (159.94s, 1,037 existing warnings).
Eight temporary report checks pass against source and the final installed wheel
(3.11s for the wheel run); the packaged JS/CSS match source bytes. The wheel
re-renders the saved Radish bundle successfully. Browser checks include actual
JSON download, failed-trial parent display, missing-comparison handling, region
filtering and 390 px mobile overflow. Focused lint, JS syntax and whitespace
checks pass. No fitting defaults changed and no live deployment occurred.

### Source-supported boundary detail diagnostic — 2026-10-20

Simplicity is not an independent fidelity goal. The first quantitative detail
comparison is now a separate `DrawingEngine.assess_boundary_detail` operation:
one explicitly paired region-local boundary, a retained/cleaned/declared source
reference with revision identity, and the candidate canonical path. It measures
local turning at multiple source-pixel scales, exposing unsupported and missing
detail separately with positions and spatial-alignment RMS. Command counts do
not enter the measurement. Native arc-length sampling and winding/cyclic phase
normalization keep exact subdivision and degree elevation materially invariant.
Pixel registration, sample budget and scales are explicit; unusable measurements
are `not_assessed`, not zero or silently coarsened.

This is diagnostic infrastructure, not a validated perceptual objective or a
new automatic gate. It changes no fit/default/acceptance logic. Fine-scale raw
polyline turning includes raster noise; scale persistence and localized review
are necessary before interpreting a difference as design detail. Source-backed
corner/inflection matching, topology-aware boundary correspondence, primitive/
clone boundary resolution and browser/MCP report integration remain open.
Boundary distance, topology and fill retain their separate evaluation roles.

Validation: 19 new detail regressions plus 11 existing revision-quality tests
pass against both source and the isolated compiled wheel; the new metric module
is confirmed as a native extension. The complete source suite passes: 1,485
tests, 1,037 existing warnings, 153.03s. Source scoring remains unavailable
without the optional renderer, as before. Cached production Radish v0/v0.4
boundaries were assessed without retracing or replacing captured artifacts;
source/native summaries match to displayed precision. Targeted metric lint and
whitespace checks pass; existing unused evidence re-export imports in
`drawing_engine.py` remain unchanged. No full native-suite claim or deployment.

### Boundary detail report and MCP integration — 2026-10-20

The offline report and on-demand `boundary_detail` MCP artifact now share one
adapter and the existing DrawingEngine measurement. The raw-trace source-path
codec is shared too. Only exact retained region/path pairs with unchanged
boundary sets are measured; duplicates, missing evidence, merged source roots,
unsupported symbolic geometry and canvas mismatches remain explicitly
unassessed. No pairing by position, new segmentation or source retracing.

Review / Edit includes per-boundary unsupported/missing turning, signed
mismatch, alignment RMS, scale selection and the existing region filter.
Read-only location markers show sampled differences on source/current candidate,
not corners or command addresses; the whole section is hidden on the fill track.
Inline summaries are bounded, with full samples/provenance/settings available
in content-addressed derived JSON files. Captured response/source/SVG files are
not replaced. The live artifact is JSON-only, optionally region-filtered, and
does not automatically add expensive measurements to trace/refine responses.

Validation: all 1,485 source tests pass (1,037 existing warnings, 134.78s).
Thirteen temporary integration regressions follow the existing report plan;
these plus eight existing temporary report checks and thirty detail/quality
tests pass against the installed wheel (51 total, 10.30s). A real in-process
FastMCP client confirms text-only content and serverInfo-only metadata, no UI.
The compiled-wheel report writer re-renders the cached production Radish bundle;
all thirty immutable capture files remain byte-identical. Browser checks cover
scale/region selection, source/candidate markers, JSON download, hidden fill
track and 390px layout without page overflow. Adapter/report lint, JS syntax and
whitespace checks pass; four pre-existing unused imports in the MCP surface
remain unchanged. No full native-suite claim, live deployment or server restart.

### Pass results use ordinary revisions — 2026-10-21

The automatic refinement entry point now retains the computed state and exact
batches of every executed pass. MCP publishes each as an ordinary immutable
DrawingStore child, even when it changes neither geometry nor fill. Disabled
passes are checklist outcomes without revisions. Shared replay helpers bind
paint policy and source correspondence without rerunning discovery or replaying
every recipe prefix just to construct history. The final source-paint resolution
is its own revision. Consumers use the returned final version, not a v1 alias.

Conversion controls group optional tracing and refinement stages into checklists
while keeping fitter policies distinct. The existing offline report selects pass
revisions and compares actual parents; parameters and per-region complexity now
follow the inspector. The runner captures outputs from the one trace request,
not invented separate mutations. Existing shared plugin guidance reflects this
contract. Local radish has v0-v9 with every artifact capture successful; its final
SVG remains byte-identical to the previously deployed result. Desktop/mobile
checks confirm pass selection and bottom diagnostic placement. Full validation
and build evidence are maintained in the existing operation-outcomes ledger.

Validation: 1,487 source tests pass (1,037 existing warnings); the built native
wheel passes 82 focused engine/MCP/replay/capture checks. UI build, shared-plugin
check, scoped lint, JavaScript syntax and whitespace checks pass. The compiled
runtime remains isolated; there is no new deployment or server restart.

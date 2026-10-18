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
- [ ] Extend the existing recipe/replay receipt to bind selected resolved
  outcomes to evidence, intent/settings, and upstream dependencies. Retain one
  canonical drawing; do not build a parallel candidate scene/store.
- [x] Keep reproduction separate from deliberate re-proposal. Reproduction
  cannot invoke an exploratory solver; re-proposal cannot mutate its parent.
  Invalidated bindings must fail explicitly rather than silently retarget.
- [ ] Verify with a controlled proposer that returns different valid candidates
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

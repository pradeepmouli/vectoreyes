# Stabilization audit — 2026-09-08

Environment: isolated Python 3.13.15 at
`/tmp/vectormark-upgrade-20260907`, using the upgraded project dependencies.
The repository environment was not recreated. Full-suite result: **610 passed,
17 failed, 0 errors, 0 skipped**, 627 tests, 963 warnings, about 97 seconds.
The three new widget-resource tests were collected separately and all passed.

Local raw evidence: `/tmp/vectormark-full-audit/pytest.log` and
`/tmp/vectormark-full-audit/junit.xml`. These files are temporary; this inventory
retains the failing test identities. No baseline SVGs were updated.

## Failure inventory

These groupings describe observed assertions, not confirmed causes or evidence
that a failure was introduced by the latest packaging change.

| Module | Failing test | Observed mismatch |
| --- | --- | --- |
| `optimizer/test_integration.py` | `test_optimizer_recolored_clone_retains_a_use_with_its_own_fill` | Clone href points at geometry definition rather than expected region ID |
| same | `test_optimizer_threads_epsilon_into_primitives_pass` | Expected primitive invocation not recorded |
| same | `test_optimizer_no_symmetry_threads_into_occlusion_pass` | Expected occlusion invocation not recorded |
| same | `test_optimizer_pipeline_reconstructs_mastercard_as_scene_branch` | Two uses instead of one |
| same | `test_optimizer_trace_uses_quadratic_base_even_when_cubics_are_enabled` | Four cubic commands instead of zero |
| same | `test_compound_split_keeps_uncovered_cutout_subpaths_subtractive` | Expected retained path missing |
| same | `test_compound_split_materializes_gradient_cutout_as_white_occluder` | Rectangle instead of expected path |
| `optimizer/test_pass_simplify.py` | `test_simplify_pass_does_not_worsen_cutout_subpath_when_outer_improves` | Expected cutout command absent after simplification |
| `test_acceptance_annulus.py` | `test_ring_plus_disk_reconstructs_and_renders` | No annulus recovered; expected one |
| same | `test_two_overlapping_rings_reconstruct` | No annuli recovered; expected two |
| same | `test_three_ring_consistent_stack_reconstructs` | No annuli recovered; expected three |
| `test_candidate_byte_identical.py` | `test_byte_identical[daikonic]` | Golden byte mismatch |
| same | `test_byte_identical[daikonic_flatten]` | Golden byte mismatch |
| same | `test_byte_identical[ring_disk]` | Golden byte mismatch |
| same | `test_byte_identical[ring_disk_flatten]` | Golden byte mismatch |
| `test_corpus_html.py` | `test_generate_corpus_html_writes_render_svg_and_diagnostics` | Expected circle table cell absent |
| `test_pipeline.py` | `test_region_with_hole_uses_evenodd_and_renders_hole` | Output is a solid rectangle rather than expected subtractive geometry |

## Next investigation order

1. Reproduce cutout/annulus preservation failures and locate the first operation
   that loses subtractive geometry. Compare retained geometry with rendering;
   do not infer correctness from SVG command spelling alone.
2. Check optimizer invocation and clone-output expectations against the current
   canonical contract. Fix behavior where the invariant remains valid; update
   assertions only where an intentional contract change explains the mismatch.
3. Reassess byte-identical and report failures after geometry is correct. Keep
   approved reference SVGs independent of generated regression snapshots.
4. Rerun the full suite before closing Slice 1 and moving to corner ownership.

Packaging verification independently passed: frontend build, source-to-wheel
build, installed-wheel stdio widget retrieval from outside the checkout, exact
HTML byte comparison, and rejection of a missing widget asset. Source archive
inspection excludes local graph databases, nested worktrees, and node_modules.
No commit, publication, deployment, or corpus regeneration was performed.

## Follow-up — closed seams and cutouts

The solid-rectangle failure was traced through each operation: segmentation
retained the navy frame (1,456 pixels, with a 144-pixel hole) and white center
(144 pixels). Scene reconstruction preserved both. Material merging incorrectly
classified their clean closed boundary as noise and returned a solid navy region
of 1,600 pixels. Symmetry enabled/disabled produced the same failure.

The open-seam length/bounding-box-diagonal heuristic now treats closed contact
loops as inconclusive rather than noisy. Source-color evidence can still merge
continuous gradient bands. High-color preprocessing uses its ownership scan to
find candidates, then reuses the same classifier on cropped contact bounds.
The focused surface-merge and rendered-hole checks pass: **18 tests**, including
square/circular counters in either ordering, soft closed boundaries that should
merge, and noisy open boundaries that should still merge.

The three annulus failures are separate: inner white artwork is retained as a
region and counted as an occlusion seam. Annulus completion declines that inner
boundary, while the resulting layered paths/circles still render the hole. The
read-only diagnosis measured SSIM about 0.955, 0.961, and 0.941 for the three
fixtures, and the flattened-hole check passes. Native annulus recovery remains
an unresolved representation requirement; do not fix it by treating every white
region as disposable background.

The simplification failure also retains its cutout, but changes its geometry.
Measured on the fixture: whole-shape residual is 0.00647, cutout residual 0.0130,
and maximum cutout boundary displacement about 2.10 pixels. This needs separate
acceptance/tolerance investigation, not a blanket update to expected command
strings. No simplify or annulus behavior was changed in the closed-seam fix.

Full follow-up run: **620 passed, 16 failed**, 636 tests. Compared with the
original audit, only `test_region_with_hole_uses_evenodd_and_renders_hole` changed
from failing to passing; there are no new failing tests. The nine added cases
are the three widget checks and six closed/open-seam checks. Raw follow-up
evidence is at `/tmp/vectormark-cutout-audit/pytest.log` and `junit.xml`.

## Follow-up — cutout-aware simplify acceptance, 2026-09-09

The compound simplification failure is now fixed by validating corresponding
boundaries independently. Tiny loops are retained, and topology checks preserve
component/hole counts, containment, intersection relationships, and fill-rule
winding behavior. Unsafe refits fall back to local command normalization or the
original boundary; explicit covered-cutout removal remains intentional.

The focused suite passes **30 tests**, including nine new boundary/topology
cases. The original failing assertion now measures boundary displacement and
weighted command complexity rather than requiring one exact SVG spelling.

On the existing Apple Music regression fixture, the rejected refit's exterior
and cutout displacement were **1.90 px and 2.10 px**. Accepted cleanup measures
**0.64 px and 0.58 px**, both below its 1 px budget. These are sampled estimates.
Command count remains 62; weighted complexity decreases from 115 to 82.
The linked SVG comparison is `/tmp/vectormark-cutout-boundary-review/index.html`.
It reuses fixture geometry and does not regenerate corpus segmentation.

Final full suite: **652 passed, 15 failed**, 667 total, no errors or skips,
about 91 seconds. Relative to the local-corner audit, the compound simplify
failure is resolved and there are no new failing test identities. Remaining
failures are the seven optimizer integration, three annulus representation,
four golden-byte, and one corpus-report assertions recorded above.
Evidence: `/tmp/vectormark-cutout-boundary-final-pytest.log` and
`/tmp/vectormark-cutout-boundary-final-junit.xml`. `git diff --check` passes.

## Follow-up — optimizer integration reconciliation, 2026-09-09

All **32 optimizer integration tests pass**, with independent task review.
Six prior failures assumed outdated representation or pass positions: clone
instances now share a paint-independent geometry definition; pass wrappers are
selected by name, not index; quadratic tracing may include protected lines;
and compound children may legitimately become primitives. Replacement assertions
retain settings, paint, branch structure, and geometric-fidelity checks.

The seventh failure exposed a real bug: compound splitting filtered a tiny
subtractive subpath, then rendered children without it while the branch retained
the original footprint. Splitting now declines if it cannot retain all parsed
subpaths. This preserves small counters at the cost of foregoing partial
primitive optimization for that compound. No fitting threshold was changed.

The corpus-report failure remains open, not simply reclassified as stale HTML.
At the gallery's epsilon 1.0, the tiny-disk fixture produces a circle candidate
with maximum sampled radial residual about 0.653 px, but the framework rejects
its 4.78% symmetric-difference area against a separate 4% budget. At the API's
epsilon 1.5, the traced geometry differs and the circle is accepted. The report
accurately describes the retained path. Resolve this acceptance-policy mismatch
without silently loosening tolerances or replacing the expected shape assertion.

## Follow-up — retained artwork and annulus reconstruction, 2026-09-09

The combined suite now reports **665 passed, 5 failed**, 670 total, in 96.86
seconds. All seven integration failures and three annulus acceptance failures
from the prior 652/15 baseline are resolved; no new failing identities appeared.
The five remaining failures are the four byte-identical cases (`daikonic`,
`daikonic_flatten`, `ring_disk`, `ring_disk_flatten`) and the tiny-disk corpus
acceptance mismatch above. Golden SVGs were not changed.

The annulus fix distinguishes cavity-contained artwork from crossing occluders
using source region identity, preserves that artwork as contour paths, and
includes it in composite acceptance. Its focused suite passes 51 checks.
Independent review confirmed retained-artwork topology, but found that aggregate
acceptance could overlook thin overpaint after disagreement erosion. The corrected
gate now rejects any intersection of primitive/lens paint with a retained source
region before tolerant appearance scoring. A 0.5 px intrusion regression covers
the former false acceptance. All 51 focused checks still pass, without changing
fitting tolerances; independent re-review approved the correction.

Evidence: `/tmp/vectormark-retained-annulus-final-pytest.log` and
`/tmp/vectormark-retained-annulus-final-junit.xml`. The full run reports 945
warnings; those have not been silently suppressed. `git diff --check` passes.

The final reviewed-code rerun also reports **665 passed, 5 failed**, in 95.91
seconds, with the same five failing identities and 945 warnings. Evidence:
`/tmp/vectormark-retained-annulus-reviewed-pytest.log` and
`/tmp/vectormark-retained-annulus-reviewed-junit.xml`.

The four historical byte snapshots are not the approved corpus reference SVGs.
Per the clarified evaluation contract, primitive recognition and clone extraction
need not reproduce reference encoding. Evaluate these differences for geometry,
paint, topology, and relationships before revising the old regression assertions;
do not treat either changed bytes alone or a refreshed snapshot as a fidelity
verdict. Approved reference files remain unchanged.

## Follow-up — semantic checks and shared acceptance, 2026-09-09

The four historical byte snapshots now use renderer-normalized appearance,
paint and topology checks, with unchanged reference files. Equivalent primitive
and path encodings, clone expansion, CSS paint and redundant gradient stops are
positive controls; lost instances, paint changes, outline changes and missing
holes are independent negative controls. A subsequent correction excludes
unsupported antialias pixels from nearest-palette topology membership while
retaining real small colored components. Independent review approved all24
focused checks; no appearance budget was widened.

The fitter now retains a singleton pre-fit corner cut when adding its necessary
auxiliary closed-contour split. Both detected and explicitly supplied singleton
corners are covered, along with unchanged zero/multiple-cut behavior. Independent
review approved50 focused checks. This is provenance preservation, not a new
corner detector or smoothing heuristic. Residual v0 visual imperfections can be
agent refinement work; they do not by themselves require more automatic fitting.

The latest combined run is **688 passed, 2 failed**,690 total,95.30seconds.
Evidence: `/tmp/vectormark-geometry-policy-pytest.log` and
`/tmp/vectormark-geometry-policy-junit.xml`. Both failures are MCP reconnect
checks whose refinement targets hardcode r1 after optimization exposes split
children. Actual published addresses and retained geometry are under review;
this is not yet declared a harmless test mismatch.

Shared primitive geometry acceptance remains **under correction**, not complete.
Although its focused86 checks pass and the original tiny-disk assertion passes,
independent review found that early rejected candidates lose diagnostics and
fixed-stride boundary sampling can miss a5.012px localized spike entirely.
The fix must provide rejected assessments and a conservative, explicitly bounded
polyline metric without silently changing the simplifier's established policy.

The corrected and independently re-reviewed implementation now passes the full
suite: **693 passed**,949warnings,105.51seconds,exit0. Evidence:
`/tmp/vectormark-geometry-policy-round1-pytest.log` and corresponding JUnit XML.
All original stabilization failures are reconciled. The acceptance metric uses
a conservative continuous-polyline bound with declared0.05px precision; its
SkPath curve linearization remains an explicitly reported approximation.
Simplification retains its previous sampling policy. Rejected primitive
assessments now reach framework diagnostics without replacing source geometry.

Two follow-ups remain explicit: the current public fitter targets leaves rather
than retained branch containers (canonical action/replay contract work), and
conservative measurement cost warrants profiling before dense-drawing use.
Neither is declared solved by this green stabilization checkpoint.

## Follow-up — canonical corner ingress, 2026-09-09

Canonical corners now retain entry/exit boundaries separately from the ordered
commands they own. Membership is derived, and one directed-ownership helper is
shared with the existing pre-fit binder. The adapter transfers only matching
recorded pre-fit declarations, with child/cutout command addresses and explicit
skips for invalid, stale or ambiguous evidence. Geometry and fill remain
unchanged. Malformed declarations are handled individually, preserving valid
peers instead of aborting conversion.

Independent review approved the implementation after one malformed-provenance
correction. The final combined run passes **707 tests**,949warnings,105.38seconds.
Evidence: `/tmp/vectormark-canonical-corner-ingress-round1-pytest.log` and JUnit XML.
This closes the annotation-ingress gap only: concrete explicit-declaration
fitting, canonical version storage and full recipe replay are still pending.

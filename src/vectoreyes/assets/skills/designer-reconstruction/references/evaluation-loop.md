# Headless plan evaluation and artwork report

Use when iterating on a traced drawing and needing persistent evidence,
independent shape/fill judgments, and an artwork handoff with Review / Edit.
The decision loop applies to both direct MCP agents and local runners. The
commands below require local Python with `vectoreyes[server,scoring]` and an
accessible MCP endpoint. Without local execution, use the equivalent public
trace/refine/artifact tools and retain the worklist in available task notes;
do not assume shell access or fabricate saved bundle paths. In a source
checkout, prefix commands with `PYTHONPATH=src`. The report itself is offline
HTML and does not require widget interaction.

## Start and inspect

```sh
python -m vectoreyes.evaluation trace input.png output/run \
  --options options.json --title "My drawing" \
  --endpoint http://127.0.0.1:8000/mcp
```

Options are the public trace schema, not a separate enum list. Start with
`refine="none"` to inspect v0. The runner records the request before calling
the server, discovers tool schemas, and saves exact responses, maps, source
correspondence and generated views. The normalized source uses the existing
preprocessor so crops/downscaling stay aligned. Build identity is explicitly
unavailable if the server does not expose it; do not claim it was verified.

Open `output/run/index.html`. Review the whole drawing, then suspect regions.
Shape and fill are independent. Shape offers neutral geometry, stored corner
and command overlays, original source polylines, and exact addresses. Fill uses
the current geometry's coverage for source sampling; do not compare changed
coverage as though it were a paint improvement. No single overall score exists.

Get the live drawing ID from `run.json`, and exact targets from the selected
version's `inspection.json`, `path_map.svg` and raw evidence. Never reuse a
command index from a different version without retained correspondence.

## Colored map review

Visually inspect both the colored segment image and the colored corner/feature
image for the selected version. A combined overlay is sufficient only when
both remain legible; otherwise inspect them separately. Review the whole
drawing first, then region-scale crops with exact IDs and legends. Fetching,
exporting, or placing these images in the report does not count as reviewing
them. Counts and clean comparison crops cannot replace this check.

- Before planning: locate each intended corner and its actual span boundaries,
  identify the commands on each intervening side, and compare their placement
  with the source. Use exact IDs, not color alone, to address the plan.
- After a corner-declaration pass: inspect every affected region's colored
  corner spans and segment endpoints against the source and parent. Check
  missing or spurious corners, point versus rounded classification, and whether
  side endpoints meet the nearest boundary of each corner span.
- After a fitting pass: refresh the child version's maps and IDs. Check curve
  types and splits, unexpected short segments or wobble, retained corner spans,
  and whether a fit consumed a corner or crossed its intended boundaries.
  Recheck neighboring seams and unchanged regions in whole-drawing context.
- Record the inspected version and relevant map/crop evidence in the worklist
  and shape review. If maps are absent or unreadable, obtain usable evidence
  before passing shape review; do not infer success from command counts.

For a batched revision, review all targeted regions after the corner pass,
then submit the fitting pass and review all targeted regions again. Keep clean
source/parent/candidate comparisons alongside the maps to judge fidelity;
evaluate fill separately without diagnostic overlay colors.

## Submit and evaluate a trial

```sh
python -m vectoreyes.evaluation refine output/run plan.json \
  --hypothesis "Replace the noisy side with one source-fitted curve"
```

Use the public plan schema from the saved `tool-schemas.json`. Correct corner
declarations before broad side refits when their endpoints are wrong. Specify
`type` for one command or `strategy` for recursive fitting. A fit-guard failure
is evidence against that proposal at that tolerance, not permission to loosen
the guard repeatedly. Curve count alone does not establish quality.

For `between=[A,B]`, the span starts at the end of A and ends at the end of B:
commands after A through B are replaced, including B. Name the two corner
endpoints directly for a full side; do not stop one command before the corner.
Before a simplification sweep, enumerate every corner-to-corner side and its
commands. After each fit, refresh IDs and record which full sides remain
fragmented. A few successful local fits do not establish an exhausted minimum.

Plan fit input is the selected current curves. The acceptance reference is the
exact original material-polyline span: only curve-to-original distance gates
publication. The reverse distance is diagnostic, not another tolerance test.
Read the rejection's `reference` and `metric` fields; current-curve residuals
from older builds are not interchangeable with this source-relative measurement.
The adaptive measurement includes its stated precision bound. Missing exact
source correspondence is a capability error, not a geometric fit rejection.

Every request gets a trial file, including errors. Publication failure produces
no candidate. Unknown execution is not retried automatically. Use `reconcile`
to save the version manifest before deciding what happened. `capture` with an
exact saved response can retry artifact retrieval without retracing/refining.

```sh
python -m vectoreyes.evaluation view output/run --version v0.0 \
  --region r4 --layers commands features source --name tip-detail
```

This writes matching SVG, PNG and JSON under `views/`. Exact path bounds define
the crop; use `--viewport x y width height` for a shared custom comparison.
The report also exports the current annotated view without requiring a
headless agent to drive the browser controls.

## Drive the evaluation, not just the tool calls

Within an authorized reconstruction, continue the loop without asking the user
to approve each ordinary trial. A report is evidence for the next decision,
not the completion condition. Do not stop at the latest failed trial while
other supported corrections remain.

1. Inspect the whole drawing and create a region worklist. For each region
   note the visible defect, relevant corner/source evidence, proposed next
   action, and state: pending, improved, satisfactory, or blocked. Prioritize
   lost topology/corners and wrong silhouettes before command-count reduction.
   Keep separate fill tasks. Save this reasoning in `evaluation-notes.md`
   beside the run; trial JSON remains the authoritative execution record.
2. Choose one local hypothesis and branch from the best visually reviewed
   retained version, not automatically the latest trial. A working best-known
   version is not the explicit whole-artwork accepted pointer. Refresh exact
   IDs and correspondence for that base before submitting the plan.
3. After success, perform the colored-map review above, and inspect clean
   whole-drawing context and affected region crops against source and parent.
   Check neighboring seams, corners and paint too.
   Record independent verdicts with evidence. Retain an improvement as the
   working base; branch back from the better parent if it regresses fidelity.
   Never promote based on lower command counts alone.
4. After a fit-guard rejection, inspect why the proposal failed. Try a
   materially different, evidence-backed hypothesis: correct a misplaced
   corner declaration; shorten the span to its proper corner boundaries;
   try Q versus C; or allow recursive fitting / split at supported inflections.
   Use only operations and strategy values present in the live schema.
   More commands are acceptable when one curve cannot faithfully fit the side.
   Do not repeat the same failed request or ratchet tolerance upward just to
   force publication. A tolerance change needs a stated source-scale rationale.
5. Bound local search: absent a user-specified budget, try up to three distinct
   hypotheses for a defect in one sweep. If none improves it, record the best
   result and blocker, then move to the next region. Revisit only when new
   evidence or a corrected upstream declaration changes the hypothesis.
   Do not let one stubborn side consume the entire drawing's evaluation.
6. After the sweep, recheck the entire best-known drawing, including fill and
   topology. Stop when remaining differences are acceptable source uncertainty
   or no supported, evidence-backed correction remains within the budget.
   Explicitly report unresolved regions; request direction only for a material
   design choice, new authority, missing evidence, or an exhausted overall
   budget—not for an ordinary rejected fit.

For example, a rejected single-C teal-side fit should lead to inspecting its
corner boundaries and trying a justified alternative such as recursive fitting
or two supported spans. If those do not help, continue to the orange band or
leaves while retaining the successful red-tip version. Do not portray a failed
teal experiment as failure of the complete reconstruction.

Infrastructure errors are different: an unknown timeout must be reconciled
before another mutation. Retry artifact capture without resubmitting the plan
when publication is known. If the service or session blocks all regions, save
the worklist and evidence and report that global blocker; do not fabricate a
candidate or switch to external SVG edits.

## Independent review and explicit acceptance

```sh
python -m vectoreyes.evaluation review output/run t0003 --track shape \
  --verdict pass --reason "Source boundaries retained; three corners intact" \
  --reviewer agent --evidence versions/v0.0.0/annotated.png
python -m vectoreyes.evaluation review output/run t0003 --track fill \
  --verdict pass --reason "Interior paint matches source on fixed geometry" \
  --reviewer agent --evidence versions/v0.0.0/review_panel.png
python -m vectoreyes.evaluation accept output/run t0003
```

IDs above are illustrative. Inspect the actual evidence before recording either
pass, including the colored-map checks for shape; execution success is not a
review. Acceptance requires both passes and
captured artifacts. Rejected trials remain inspectable via `reject --reason`.
Use `report` to regenerate presentation without executing any engine operations.
Do not claim artwork accepted merely because its report was produced.

The handoff shows the accepted SVG, or a clearly provisional baseline if no
version is accepted. Download delivers exact saved SVG bytes. Review / Edit
is a non-mutating inspector: use public plans for actual edits.

## Lifecycle and recorded evidence

Bundles remain inspectable after restart; live drawing IDs do not. A bundle is
not an engine-session backup. The `import` command imports an existing recorded
snapshot for review, explicitly labeled as historical evidence—not a successful
new trace and not proof that its drawing handle is still live.

Verified on the angled radish with an isolated compiled MCP server: baseline
capture, explicit source-point corner replacement, and a single-Q red-left refit
published and all requested artifacts were retrieved. Red tip commands reduced
15→11 while retaining three point corners. A single-cubic teal proposal was
also recorded as a fit-guard rejection. This demonstrates execution and capture;
it does not certify that the complete radish is optimally fitted or accepted.
The earlier paint-revision publication error required a replay fix; the report
does not bypass engine provenance validation.

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
The selected trial's **Parameters used** disclosure separates its submitted
request from retained parameters for the displayed artwork. Trace options
link to their captured response; effective recipe/action overrides and output
policy come from that version's evidence. Missing values stay unavailable.
For a failed trial, the displayed parent's parameters are not presented as
effective settings for an unpublished candidate.
The report is self-contained for offline inspection; SVG download targets the
exact saved file beside it. It does not execute plans from browser selections.

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

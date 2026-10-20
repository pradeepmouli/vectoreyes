# Plan Evaluation Report Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user explicitly requested execution without another planning approval stop. Steps use checkboxes.

**Goal:** Persist and review plan trials with separate shape/fill decisions and a clean artwork handoff.
**Architecture:** A small evaluation package consumes the existing public MCP artifacts. It owns trial metadata and deterministic report files, never drawing geometry. A packaged HTML/JS viewer and headless PNG/SVG exports share the same inspection snapshot.
**Tech Stack:** Python/Pydantic, existing FastMCP client, Pillow/resvg, dependency-free browser JS.
**Spec:** `docs/superpowers/specs/2026-10-14-plan-evaluation-report-design.md`.

## Global constraints

- DRY: reuse public typed plans, canonical SVG exports and stored feature/source metadata.
- No new permanent tests; use temporary fixtures/checks plus existing suites.
- Preserve the dirty feature checkout and peer's replay correction. Do not deploy/restart live services as part of report implementation.
- Shape/fill verdicts are independent; no promotion on execution alone.
- Selection and report rendering never mutate the drawing.
- No legacy compatibility shims or invented correspondence.

## Review focus

1. Failed/unknown execution leaves no fabricated candidate and is not retried automatically.
2. Source-relative files, malicious SVG/JSON strings and unsafe artifact paths do not become executable report markup.
3. Many-to-one or absent correspondence is represented explicitly, not inferred from indices.
4. Acceptance requires successful execution and independently recorded shape/fill passes.
5. A server restart leaves saved evidence inspectable but expired drawing IDs cannot resume implicitly.

## Files and interfaces

- `src/vectormark/evaluation/models.py`: strict run/trial/view/verdict models and explicit promotion.
- `src/vectormark/evaluation/bundle.py`: atomic JSON and hashed artifact persistence; immutable trial requests/results.
- `src/vectormark/evaluation/inspection.py`: public response geometry/features/source evidence to inspection records; derived counts labeled.
- `src/vectormark/evaluation/views.py`: shared view specifications, neutral SVG and annotated SVG/PNG crop exports.
- `src/vectormark/evaluation/client.py`: public MCP capture and trial orchestration, preserving all failures.
- `src/vectormark/evaluation/report.py` and `assets/evaluation/*`: static offline handoff/inspector from saved bundle, no server dependency.
- `src/vectormark/evaluation/__main__.py`: script CLI for trace/refine/review/render/view and serve.
- Packaged reconstruction skill/reference and docs: verified commands and status boundaries.

### Task 1: Trial records and persistent bundle

Produces `EvaluationRun`, `TrialRecord`, `ReviewVerdict`, `ViewSpec`, `RunBundle`.
- [ ] Temporary RED checks: no implicit promotion; failed execution cannot accept; both track passes required; artifact traversal rejected.
- [ ] Implement models with strict extra-field rejection and scoped identifiers. Atomic manifest replace and content hashes; requests saved before execution.
- [ ] GREEN checks plus serialization roundtrip. Record verification in ledger.

### Task 2: Inspection and shared exports

Consumes public response/report, raw_trace, plan replay metadata and exported SVG. Produces `build_inspection(response, raw_trace, provenance)` and `export_view(bundle, version, spec)`.
- [ ] RED fixtures using saved radish: 4 regions, exact IDs, 4/4/3 corners, unavailable correspondence explicit, per-region Q/C counts.
- [ ] Normalize public geometry recursively without engine re-detection. Retain actual anchor/source records and correspondence verbatim, exposing no inferred mapping.
- [ ] Export neutral shape, original paint and stored annotation maps with consistent viewBox; region crops from exact SVG geometry bounds, no hardcoded orange viewport. Emit matching JSON and PNG.
- [ ] GREEN checks for source/candidate crop identity, SVG safety and geometry preservation.

### Task 3: Scripted MCP loop

Consumes Tasks1–2; produces `trace_run`, `refine_trial`, `capture_version` async operations and `python -m vectormark.evaluation` CLI.
- [ ] RED temporary checks for request saved before failure; response saved on success; partial artifact failure retained; no retries.
- [ ] Discover live tool schemas; trace PNG via base64. Explicit options disable cropping/downscaling unless requested and record normalized source canvas. Capture all version artifacts through existing endpoints.
- [ ] Refine only exact plans matching run drawing/base. Capture schema/fit/replay/transport errors with null candidate until outcome is known; reconcile manifest after unknown outcome.
- [ ] Add explicit per-track review and promotion CLI operations; neither changes engine versions.
- [ ] GREEN with live baseline/artifact retrieval and isolated fixed-runtime reproduction when available.

### Task 4: Handoff and inspector

Consumes saved models/inspection; produces `write_report(bundle)` self-contained offline HTML plus images/JSON.
- [ ] RED browser/temporary checks for provisional baseline, accepted pointer, no live mutation controls, safe text rendering.
- [ ] Implement approved handoff, download, Review/Edit, synchronized viewports and version selection, separate shape/fill modes, mandatory layer toggles/IDs and metadata.
- [ ] Draw layers from shared inspection/view exports. Selection uses stored correspondence; unmapped stays unmapped. Show execution/shape/fill independently in trial history.
- [ ] Browser-check real radish images, region/command/feature selection, crop exports, provisional download, keyboard/heading/contrast and failed-trial state.

### Task 5: Real loop and packaged instructions

- [ ] Execute bounded radish trial on fixed runtime without discarding prior evidence. Inspect actual child, record separate verdicts; reject visually worse changes.
- [ ] Verify geometric and cutout fixtures with existing source artifacts/checks. Do not claim entire corpus clean.
- [ ] Update packaged skill and docs with headless commands, report entry point, acceptance and error handling; only call actual executions verified examples.
- [ ] Run relevant existing suites and UI build; report incomplete global suites accurately. Review changed files independently of implementation pass.

## Execution record

Ledger: `.superpowers/sdd/2026-10-14-plan-evaluation-report/progress.md`.
No release/deployment is included. Do not commit unrelated existing changes.

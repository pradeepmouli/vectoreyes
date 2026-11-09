# Fidelity + Parsimony Scorer (Prototype) — Design Spec

**Status:** Approved (design). Slice 3 of the candidate-pipeline roadmap
(`docs/architecture/2026-06-07-candidate-pipeline-roadmap.md`).
**Branch:** `feat/scorer` (off `feat/candidate-interface` / PR #16 — needs the `Candidate` type;
rebase onto `master` once #16 lands).
**Date:** 2026-06-07

## Goal

Build a **standalone candidate scorer** that ranks the several ways an element could be rendered and
picks the best by **render-fidelity gated, then parsimony**, with the proven structural guards as
hard gates. Evaluate it in isolation against a labelled synthetic corpus (and the untracked
real-logo corpus, manually) to measure false-accept rates and tune the knobs — **before** wiring it
into `idealize` (that is slice 4). This is the roadmap's deliberately-isolated linchpin: ΔE-alone
over-accepts, proven repeatedly in the gradient work.

## Why this shape (decisions made in brainstorming)

- **Role: rank a candidate set** (not a single accept/reject). The set includes a "leave flat / do
  nothing" candidate, so accept/reject falls out of ranking, and the output is exactly what the
  slice-4 selector consumes.
- **Combination: lexicographic / fidelity-gate** (potrace's "fewest segments, then least penalty").
  Fidelity is a *gate* (render-ΔE ≤ tolerance), not a score to maximise; among qualifiers the
  **most parsimonious** wins; ΔE only breaks ties. This structurally cannot repeat the over-accept
  failure: a more-complex candidate with marginally-better ΔE cannot out-rank a simpler one that
  also clears the gate.
- **Parsimony: structured description-length** — a parameter/token count per candidate (geometry
  params + fill params). Decoupled from SVG text formatting; extensible; weights tunable.
- **Fidelity: SVG render-ΔE via resvg** — rasterize the candidate (bbox-only) and `mean_delta_e`
  vs the source pixels. The exact, proven fidelity measure. Makes `resvg-py` a runtime dependency
  (currently dev-only).
- **Manual selection:** the scorer returns the **full ranked list with score breakdowns**, not just
  a winner — enabling the slice-4 selector's post-evaluation override, and pre-execution strategy
  choice by passing a restricted candidate set.

## Architecture

New module `src/vectoreyes/score.py`. Pure scoring logic + the resvg-backed fidelity measure. Not
wired into `pipeline.py` in this slice.

```python
@dataclass
class ScoreBreakdown:
    delta_e: float             # render-ΔE fidelity (lower = better); inf if priors failed
    parsimony: float           # structured description-length (lower = better)
    priors_ok: bool            # passed all structural-prior hard gates
    reject_reason: str | None  # which prior failed (transparency for override)
    qualified: bool            # priors_ok AND delta_e <= fidelity_tol


def parsimony_cost(cand: Candidate) -> float: ...
def render_delta_e(cand: Candidate, source_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> float: ...
def structural_priors(cand: Candidate, region: Region) -> tuple[bool, str | None]: ...
def rank_candidates(
    cands: list[Candidate], source_rgb: np.ndarray, region: Region, *,
    fidelity_tol: float = 0.06,
) -> list[tuple[Candidate, ScoreBreakdown]]: ...
```

### `rank_candidates` algorithm

1. For each candidate build a `ScoreBreakdown`: run `structural_priors`; if `priors_ok`, compute
   `render_delta_e`; always compute `parsimony_cost`. `qualified = priors_ok and delta_e <=
   fidelity_tol`.
2. Order: **qualified** candidates first, sorted by `(parsimony asc, delta_e asc)`; then
   **disqualified** candidates, sorted by `(delta_e asc)` so the tail is still a sensible inspectable
   ranking.
3. Return the full list of `(candidate, breakdown)` best-first. The winner is element 0;
   `reject_reason` explains every disqualification. Determinism: stable sort, fixed key order,
   resvg is deterministic.

### `parsimony_cost` (structured description-length)

Geometry cost by descriptive parameters + fill cost:

| geometry | cost | fill | cost |
| --- | --- | --- | --- |
| circle | 3 (cx, cy, r) | flat | 1 (one colour) |
| rect | 4 | linear gradient | 4 (x1,y1,x2,y2) + 2×stops |
| ellipse | 5 | radial gradient | 3 (cx,cy,r) + 2×stops |
| polygon | 2 × vertices | | |
| annulus | 6 | | |
| path | per-command base + per-control-point (M/L = 2, C = 6, Q = 4, Z = 0) | | |

`parsimony_cost = geometry_cost + fill_cost`. Weights live as named module constants so they are
tunable during evaluation. Lower is simpler; a circle+flat (4) beats a many-Bézier path+gradient.

### `render_delta_e` (fidelity)

Build a minimal SVG for the single candidate (reuse `emit` helpers / `render_svg_doc`), rasterize
its bounding box with `resvg_py`, and `mean_delta_e` against the same bbox of `source_rgb`.
Bbox-only keeps it bounded. Returns mean OKLab ΔE (0 = identical). This is the same render-and-compare
the acceptance suite uses, lifted into `src/`.

### `structural_priors` (generalise the proven guards into hard gates)

Predicates keyed by candidate type; a failure disqualifies the candidate with a `reject_reason`:
- **Gradient candidate:** stops must travel ≥ `_MIN_STOP_SPAN` (0.02 OKLab) end-to-end — no
  near-degenerate "gradient" on a flat region (the proven Pinterest/Vimeo false positive); and its
  footprint must be a single dominant blob (≥ `_BLOB_DOMINANCE`, 0.85) — reuse/import the existing
  `gradient._stop_span` / `_dominant_blob_fraction` rather than re-deriving (DRY).
- **Primitive candidate (circle/ellipse/rect/polygon/annulus):** must actually match its source
  contour within the recogniser's residual (it was produced by `recognize_primitive`, so this is
  effectively already true; the prior is a guard against a forced primitive on a non-matching
  region in the eval generator).
- **Flat / smooth-path candidates:** no structural prior (always eligible; the fidelity gate and
  parsimony decide).

Priors run *before* fidelity so a disqualified candidate never costs a render.

## The candidate-variant generator (evaluation only)

To rank a set you need a set. Slice 3 adds a generator **for evaluation** (production
multi-candidate generation is slice 4). Given a `region` + `source_rgb`, produce the competing
candidates by forcing each strategy, reusing existing fitters:
- **flat:** `Candidate(_fit_region-style geometry, FlatFill(region.color_hex))` — the simplest
  faithful outline with a flat fill.
- **primitive-snap:** `recognize_primitive(contour)` → `Candidate(primitive, FlatFill(...))` when one
  is found.
- **smooth-path:** `fit_path(contour)` → `Candidate(path, FlatFill(...))`.
- **gradient:** `fit_gradient(region.mask, source_rgb)` → `Candidate(footprint geometry,
  Linear/RadialGradientFill)` when a model is returned.

Lives in the **test/eval harness** (e.g. `tests/_candidates.py`), **not** in `score.py` (which stays
a pure scorer) and **not** wired into `idealize`. Production multi-candidate generation is slice 4.

## Evaluation corpus & success metric

- **Committed synthetic labelled cases** (the automated gate, no brand assets): each is
  `(source_rgb, region, expected_winner)` where the right answer is known by construction:
  - true circle on white → expect **primitive circle + flat** (not a path).
  - true linear gradient rect / radial gradient disc → expect **gradient** (not flat).
  - **flat square → expect flat/primitive, NOT a gradient or complex candidate** (the over-accept
    regression — the headline guard).
  - organic noisy blob → expect **smooth-path**.
  - posterized ramp → expect **gradient** (band path).
  Assert `rank_candidates(...)[0].candidate` matches the expected winner kind/fill.
- **False-accept metric:** number of cases where a more-complex/wrong candidate out-ranks the
  correct simpler one. **Target: 0** on the labelled set.
- **Real-logo corpus (manual/local only):** a dev script (`scripts/` or an xfail-guarded,
  skip-if-missing test) that runs the scorer over the untracked `scratch/real-logos/` and prints
  rankings + breakdowns for inspection. Brand assets stay uncommitted; this is not a CI gate.

## Error / edge handling

- Empty candidate set → return `[]`.
- No candidate qualifies (all fail priors or exceed `fidelity_tol`) → the list is still returned,
  ranked, with `qualified=False` for all; the caller (slice 4) decides the fallback. For the eval,
  this is itself a recordable outcome.
- `render_delta_e` on a degenerate bbox (zero area) → treat as max ΔE (disqualify by fidelity), with
  a clear path, not an exception.
- Determinism preserved (stable sort, deterministic resvg, fixed weights).

## Dependencies

Move `resvg-py` from `dev` to runtime dependencies in `pyproject.toml` — `score.py` (production
code) imports it for `render_delta_e`. (Anticipated and accepted in brainstorming.)

## Testing

- **Unit:** `parsimony_cost` ordering on known shapes (circle+flat < polygon < many-Bézier
  path+gradient); `render_delta_e` ≈ 0 on an exact-match candidate and large on a mismatch;
  `structural_priors` rejects a near-degenerate gradient (stop-span < 0.02) and a multi-blob gradient
  footprint.
- **Headline:** the labelled synthetic eval — every case's top-ranked candidate equals the expected
  winner; **false-accept count == 0**, including the flat-square-stays-flat over-accept regression.
- **Isolation:** `score.py` is not imported by `pipeline.py`/`idealize` in this slice, so the
  existing suite is unaffected (it must stay green).

## Non-goals (slice 3)

- Wiring the scorer into `idealize` / replacing `_fit_region`'s decisions — that's slice 4
  (the selector harness), together with production multi-candidate generation.
- The apply/override *mechanism* for manual selection — slice 3 only guarantees the transparent
  ranked output that enables it.
- A `cost` field on `Candidate` in the wired pipeline — `parsimony_cost` is computed by the scorer
  here; whether to cache it on `Candidate` is a slice-4 decision.
- Component decomposition (slice 5).
- Tuning against the full real-logo set as a CI gate (brand-asset licensing — manual/local only).

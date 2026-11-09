# Selector Harness (4a — auto-selection) — Design Spec

**Status:** Approved (design). Slice 4a of the candidate-pipeline roadmap
(`docs/architecture/2026-06-07-candidate-pipeline-roadmap.md`).
**Branch:** `feat/selector` (off `master`; builds on the merged Candidate seam (slice 2) and
`score.py` (slice 3)).
**Date:** 2026-06-07

## Goal

Replace `_fit_region`'s hand-tuned **geometry priority cascade** with **generate-all-candidates →
score → pick**, using the slice-3 scorer. The simplest geometry that still renders the region
faithfully wins, instead of a fixed priority order. This is the **first slice that changes
`idealize`'s output** — there is no byte-identical gate; correctness is held by the acceptance suite
(parity) plus a before/after corpus eval.

**Scope (4a only):** geometry selection for the per-region path. The `detect_gradients` and occlusion
reconstruction passes are unchanged (they already feed the candidate/emit model from slice 2).
Manual selection (pre-execution restriction + post-evaluation override) is **slice 4b**, layered on
afterward.

## Why this is safer than it looks

The candidate generators are **exactly the cascade's existing fitters** (primitive, rounded
trapezoid, symmetric polygon, half-ellipse cap, symmetric fit, polygon, path, holed half-mirror). So
the selector can only ever pick a geometry the cascade was *already capable of producing* — the
change is purely **re-prioritization by score**, never a new output type. That bounds the regression
surface to "which known-good candidate wins," which the acceptance-suite parity gate catches well.

## Architecture

New module `src/vectoreyes/selector.py`. `_fit_region` is **deleted**; its fitter calls relocate into
the generator (no dead code, no duplication — the orchestration changes from first-non-None-wins to
collect-all-then-score).

```python
def generate_geometry_candidates(
    region: Region, opt: Options, axis: Axis | None, corner_radius: float,
) -> list[Shape]:
    """All geometry fits the cascade could produce for this region (non-None only)."""

def select_geometry(
    region: Region, opt: Options, axis: Axis | None, corner_radius: float,
    source_rgb: np.ndarray,
) -> Shape | None:
    """Generate candidates, score them, return the winning geometry (or None if none)."""
```

`pipeline.build_candidates` swaps its two `_fit_region(...)` calls (region path + gradient-footprint
geometry) for `select_geometry(..., source_rgb)`. Symmetry handling is unchanged: `select_geometry`
receives the same `fit_axis` the cascade did (straddler = axis → axis-aware candidates; pair/loner =
None), and `build_candidates` still sets `mirror=axis` for pairs.

### `generate_geometry_candidates` — the candidate set

Mirrors `_fit_region`'s applicability structure, but **collects** instead of returning the first hit:

- **Holed region** (`len(contours) > 1`): primitive/polygon recognition is skipped (they see only the
  outer ring). Candidates:
  - if `axis`: the symmetric half-mirror path — `symmetric_fit` per contour, joined, `fill_rule=evenodd`
    (only when every contour straddles cleanly, as today);
  - always: the faithful per-contour `fit_path`, joined, `fill_rule=evenodd`.
- **Single contour:**
  - `recognize_primitive` → if found, axis-snapped via `_snap_to_axis` when `axis` is set.
  - if `axis` (straddler): `rounded_trapezoid_fit`, `symmetric_polygon_fit`, `half_ellipse_cap_fit`,
    `symmetric_fit` — each added when non-`None`.
  - `recognize_polygon` (if found).
  - `fit_path` — always (the catch-all, guarantees a non-empty set).

`_snap_to_axis` moves with the generator (it is a geometry-construction helper).

### `select_geometry` — scoring

Wrap each `Shape` as `Candidate(shape, FlatFill(region.color_hex), "region")` (all candidates share
the region's flat fill, so ranking turns purely on geometry). Call
`score.rank_candidates(wrapped, source_rgb, region, fidelity_tol=opt.fidelity_tol)`. Return
`ranked[0].candidate.geometry`. Because all candidates carry the same fill, the lexicographic rule
reduces to: **ΔE ≤ τ gate → minimum parsimony → ΔE tiebreak** = "the simplest geometry that renders
the region faithfully." If no candidate clears the gate, `ranked[0]` is still the best-fidelity
option (always returned — matches the cascade always falling back to `fit_path`). Empty candidate
list (no usable contour) → `None` (region dropped, as today).

### Options

Add `fidelity_tol: float = 0.06` to `Options` (the scorer's render-ΔE gate; starts at
`rank_candidates`'s existing 0.06 default and may be retuned during evaluation — see below). No
other `Options` changes in 4a.

## bbox rendering (performance)

The cascade rendered nothing; the selector renders every candidate. Add an optional bounding box to
`score.render_delta_e`:

```python
def render_delta_e(cand, source_rgb, region, *, bbox: tuple[int,int,int,int] | None = None) -> float
```

When `bbox` is given (selector derives it from `region.mask` + a small margin, clamped to the
canvas), render the candidate SVG and compare only within that crop (mask restricted to the bbox).
Render cost then scales with element size, not canvas size. Behaviour is identical to full-canvas for
the compared pixels (same mask, same colours) — purely a speed optimization. `bbox=None` keeps the
current full-canvas behaviour, so the slice-3 score tests are unaffected. The selector always passes
the region bbox.

## Evaluation & regression strategy (the safety net)

Output changes, so the gate is **not** byte-identical:

1. **Parity gate (headline):** the entire existing acceptance suite stays green — daikonic
   (symmetry, `<use>`, 4 fills, exact-symmetric render), annulus, polygon corners, gradient +
   smooth-gradient ΔE, occlusion, rotation, segment. A break → **STOP and investigate that case**:
   if the scorer's pick is genuinely worse, fix the scorer (`fidelity_tol` / parsimony weights /
   priors); only if the pick is genuinely **better** and the test was over-specified do we update
   the test, with a one-line recorded rationale in the commit.
2. **Corpus before/after eval (dev tool, brand-safe):** before any selector change, capture
   current-master `idealize` metrics (render-ΔE & SSIM vs source, element/`<path>`/`<circle>` counts)
   over the committed synthetic fixtures + the untracked `scratch/real-logos/`. A
   `scripts/eval_selector.py` re-runs after the change and reports per-logo deltas. **Overall
   faithfulness (mean render-ΔE) must not regress**; element-count drops that *improve* parsimony
   without raising ΔE are wins. Skips cleanly when the corpus is absent; commits no brand assets.
3. **`fidelity_tol` tuning:** start at 0.06; if (1) fails or (2) shows a faithfulness regression,
   retune and record the final value and rationale in the commit. (A higher τ lets parsimony
   simplify more aggressively; a lower τ keeps fidelity tighter.)

Per-element faithfulness is bounded by `τ` by construction.

## Error / edge handling

- No usable contour → `select_geometry` returns `None` → region dropped (current behaviour).
- No candidate clears the fidelity gate → return the best-fidelity candidate (`ranked[0]`), never
  empty when candidates exist.
- bbox clamped to canvas bounds; a degenerate (zero-area) bbox falls back to full-canvas render.
- Determinism preserved: `generate_geometry_candidates` is deterministic; `rank_candidates` uses a
  stable sort and deterministic resvg; candidate construction order is fixed.

## Testing

- **Unit (`tests/test_selector.py`):** `generate_geometry_candidates` returns the expected set —
  clean disk → contains `circle` and `path`; holed ring → only `path`(s) with `evenodd`; straddler
  band → contains `rect`/trapezoid-path and `path`; straddler dome → contains the cap/sym-fit and
  `path`. `select_geometry` picks `circle` for a disk, the trapezoid for a band, a `path` for an
  organic straddler.
- **bbox (`tests/test_score.py` append):** `render_delta_e(..., bbox=region_bbox)` agrees with the
  full-canvas value within a small tolerance on a centered shape.
- **Parity (headline):** the full existing suite stays green (run `pytest -q`).
- **Before/after eval:** `scripts/eval_selector.py` runs and reports; manual inspection on the real
  corpus. Not a CI gate.

## Non-goals (4a)

- Manual selection (pre-execution restriction + post-evaluation override) — **slice 4b**.
- Folding flat-vs-gradient or occlusion into the scored model — geometry cascade only.
- New geometry fitters — the candidate set is exactly the existing cascade's fitters.
- A `cost` field cached on `Candidate` — parsimony is computed by the scorer.
- Component decomposition (slice 5).

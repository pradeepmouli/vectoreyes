# Candidate-Based Idealization Pipeline — Architecture & Roadmap

**Date:** 2026-06-07
**Status:** Vision / roadmap (not yet a build spec). Each numbered slice below gets its own
brainstorm → spec → plan → build cycle. This doc is the map, not the territory.
**Author:** captured from a design conversation following the gradient + smooth-gradient work.

## The core idea

Every element vectoreyes emits is a pair: **(geometry, fill)**.

- *Geometry* = the shape's outline: a primitive (`<circle>`, `<rect>`, ellipse, annulus,
  convex polygon) or a fitted smooth path (lines + Béziers).
- *Fill* = how that outline is painted: a flat colour, a linear/radial gradient, (later)
  a pattern.

Today these two axes are **entangled**. `_fit_region` picks geometry while
`segment()`/`extract_palette()` fix the colour, computed in lockstep per region. That
entanglement is why each new capability (gradients, symmetry, occlusion) has had to be
threaded invasively through `pipeline.py`, and why we cannot ask the obvious question:
*"of the several ways I could render this element, which is best?"* — we only ever generate
one candidate.

The vision is a **generate-and-select** pipeline built on four interlocking layers. They are
listed in dependency order, but the *delivery* order (below) is different — we build the
cheap, high-leverage, low-risk slices first and defer the linchpin until we have evidence.

## The four layers

### 1. Component decomposition (front-end)

Before any fitting, split the raster into **clearly-separate regions** by gutters of
background (connected-component analysis on the non-background mask, with a separation
threshold). Each component is idealized independently, then composed back.

*Why:* symmetry detection becomes **purer** — a wordmark beside a mark no longer drags the
symmetry axis off the mark; each component's own symmetry (or tilt) is detected in isolation.
It also unblocks the multi-blob gradient case the smooth-gradient work explicitly deferred
(a gradient mark *plus* a separate glyph): decompose first, then each blob is a clean
single-gradient silhouette.

*Risk:* low-medium. Connected components are deterministic; the open question is the
separation threshold and how to recompose (shared canvas coords, z-order). Touches
segmentation and the rectified path.

### 2. Per-element candidate generation (geometry × fill)

For each component/region, generate **multiple** candidate renderings instead of one:

- geometry candidates: primitive-snap, smooth-path fit, (later) potrace-style optimal polygon
- fill candidates: flat, linear gradient, radial gradient

The Cartesian product (filtered to the sensible ones) is the candidate set. This is the
"multiple algorithms" idea made concrete: algorithms don't compete globally, they each
*propose a candidate* and the selector decides per element.

*Risk:* medium. Mostly refactoring existing fitters behind a uniform `Candidate` interface
`(geometry, fill, cost)`; the work is API design, not new math.

### 3. Fidelity + parsimony scorer (the linchpin)

Score each candidate and pick the winner. **This is the riskiest, most load-bearing piece.**

The hard lesson from the gradient work, proven repeatedly: **ΔE (render fidelity) alone
over-accepts.** A flat shape trivially fits a degenerate near-constant gradient at ΔE ~0.000;
a perfectly-flat mark fits a tilted "gradient" via edge-smear. Fidelity must be paired with:

- **parsimony** — prefer fewer/simpler primitives, fewer stops, fewer control points
  (an explicit description-length penalty, à la potrace's lexicographic
  "fewest-segments-then-least-penalty");
- **structural priors** — the guards we already discovered empirically (min stop-span,
  single-dominant-blob connectivity, contiguity-bounded footprints) generalized into the
  scorer rather than bolted onto each detector.

The scorer is what makes "pick the optimal one per image" trustworthy instead of a
fidelity race to the most complex candidate.

The scorer must also be **transparent and overridable**: it returns the *full ranked list of
candidates with their score breakdown* (fidelity, parsimony, which structural prior fired), not just
a single winner. This is what lets an agent or user inspect *why* a candidate won and choose a
different one (see "Selection is automated **and** manual" under layer 4).

*Risk:* high. There is no off-the-shelf scorer; ΔE+parsimony weighting is a tuning problem
with real false-accept failure modes. Prototype against the existing scratch corpus before
committing the API.

### 4. Selector harness (integration)

The deterministic driver: for each element, generate candidates (2), score them (3), emit
the winner; compose components (1). Replaces the current `_fit_region` + ad-hoc gradient
gate with one uniform loop. The emit layer already renders `(geometry, fill)` generically
(the gradient work proved this), so this is plumbing once 1–3 exist.

**Selection is automated *and* manual (a first-class requirement, not just `cost` ranking).** The
selector supports human/agent choice *in addition to* automated scoring, at two points:
- **Pre-execution:** restrict which algorithm/strategy candidates are generated or run for an
  element (e.g. "only try primitive-snap and smooth-path here") — selection is a *separate stage*
  from generation, not hardwired into it.
- **Post-evaluation:** override the auto-scored winner after seeing the ranked candidates and their
  breakdowns. The scorer's transparent ranked output (layer 3) and each candidate's `source`
  provenance label (layer 2) are the enablers.

*Risk:* low once the pieces exist; it is glue.

## Separation of concerns: colour vs. shape

A cross-cutting principle the layers encode: **colour application and shape/path detection
are orthogonal and should be separate modules.** Geometry fitters should not know whether
the fill is flat or a gradient; fill detectors should not know whether the outline is a
circle or a path. The `Candidate = (geometry, fill, cost)` interface is the seam. This also
fixes the immediate palette bug independently of everything else (see slice 1 below) — colour
extraction is a self-contained concern that the rest of the pipeline consumes through a
stable interface.

## Delivery sequence

Cheapest, highest-leverage, lowest-risk first. Each is an independent slice that leaves the
tool working and testable; we learn before committing to the risky linchpin.

1. **Palette fix (perceptual clustering).** *In progress — first slice.* `extract_palette`
   drops AA-dispersed colours (thin/small marks like the settir blue) because it counts exact
   RGB shades and applies a frequency floor with an early break. Replace with greedy perceptual
   clustering *before* the floor. Self-contained colour-concern fix; no pipeline restructure.
   Validates the "colour is a separable concern" principle in the smallest possible change.

2. **Geometry/fill separation.** Introduce the `Candidate (geometry, fill, cost)` interface
   and refactor the existing fitters + gradient detector behind it. No behaviour change
   intended — pure decoupling, regression-gated. Sets up everything after.

3. **Scorer prototype.** Build fidelity+parsimony scoring as a standalone, evaluated against
   the scratch corpus. Deliberately *before* wiring it in, so we can measure false-accept rates
   and tune weights without destabilizing `idealize`. The riskiest piece, de-risked in isolation.

4. **Selector harness.** Wire generate → score → emit into one loop, replacing `_fit_region`'s
   ad-hoc decisions. Now multiple candidates genuinely compete per element.

5. **Component decomposition.** Add the gutter-based front-end; re-run symmetry/tilt per
   component; recompose. Unblocks the deferred multi-blob gradient case and purifies symmetry.

Order rationale: 1 is an isolated bugfix with immediate user value; 2 is prerequisite
refactoring; 3 is isolated so its risk can't sink the tool; 4 needs 2+3; 5 is the largest
structural change and benefits from the uniform candidate loop already existing.

## Non-goals (for this roadmap)

- ML / learned tracing (StarVector-class) — vectoreyes stays deterministic.
- Conic/sweep gradients — still a gradient non-goal.
- Rewriting the emit layer — it already renders `(geometry, fill)` generically.

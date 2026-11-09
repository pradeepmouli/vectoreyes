# SVG Geometry Optimizer — Design

**Status:** Approved (design) — pending implementation plan
**Date:** 2026-06-29

## Problem

vectoreyes currently performs its optimizations — primitive recognition, mirror
symmetry, gradient merging — against the **raster** (per-region pixel masks),
interleaved into the vectorization pass. This is the wrong layer and it has
produced a steady stream of regressions:

- **Wrong unit.** The thing an optimization should reason about is the **object
  that gets drawn**, not a raw color bin. The Instagram outer square is one
  drawn object (a merged gradient surface) but ~20 asymmetric diagonal color
  bands as raster regions — so region-level symmetry can't see its symmetry,
  while the pristine silhouette-level detector could. daikonic is the opposite:
  the radish is symmetric but its component silhouette is contaminated by the
  "Daikonic" wordmark. No single raster-level unit serves both.
- **No safety net.** Each optimization mutates output with no objective check
  that it stayed faithful to the input. Symmetry reconstruction silently dropped
  glyphs, force-mirrored asymmetric shapes (icloud), and lumped curves — each
  caught only by manual visual review, each fixed only to reveal the next.

The fix is structural: **vectorize faithfully first, then optimize the geometry
of the resulting objects, with every optimization gated against the faithful
result so it cannot regress.**

## Goal

A two-stage pipeline:

1. **Faithful vectorize** turns the raster into a ground-truth set of drawn
   objects (paths + fills + z-order) with no optimization.
2. A **geometry optimizer** runs ordered passes over those objects —
   primitives, clones, symmetry, simplify — each operating on exact geometry,
   each change accepted only if it stays within a coverage budget of the
   faithful ground truth. **No regression by construction.**

## Architecture

```
raster
  └─ STAGE 1: FAITHFUL VECTORIZE
       quantize → segment (regions) → region_contours → fit_path
       + gradient surface merge (surface_merge)
       → ground-truth OBJECTS  { id, exact, fill, z }
       → render once = GROUND-TRUTH coverage reference (per region/object)

  └─ STAGE 2: GEOMETRY OPTIMIZER  (passes over objects, exact geometry)
       2a primitives : flat → circle/ellipse/rect/polygon fit
       2b clones     : flat congruent (translate+rotate, fill may differ) → dedup <use>
       2c symmetry   : axis vote + absolute test on flat → self-symmetric reconstruct,
                       mirror PAIRS → <use> mirrored twin
       2d simplify   : merge segments → single curve; collinear → line; drop control pts
       [+ optional geometric merge: adjacent same-fill union]
       GATE (every proposed change, all passes):
         coverage of changed shape  vs  true region mask  →  symmetric-difference ≤ budget
         accept ⇒ keep optimization;  reject ⇒ keep faithful
```

### Object model (dual representation)

Each object carries two representations of its area:

- **`exact`** — the emit/fit representation: a path (cubic/quadratic/line
  segments, plus hole subpaths for interior counters) **or** a primitive
  (circle / ellipse / rect / polygon). This is what is written to SVG and what
  fitting/simplification produce.
- **`flat`** — a `shapely` `Polygon`/`MultiPolygon` (outer shell + interior
  holes), with any curves sampled to a dense polyline. This is what all
  geometry operations run on.

A region is therefore a **filled shape** — curved boundary, possibly with holes,
possibly a primitive — not a bare polygon. The historical distinction between
"region", "component", and "object" collapses: after tracing, everything is one
object type. Geometry ops use `flat`; results are applied back to `exact`.

### Stage 1 — Faithful vectorize

Reuses the existing front end unchanged in spirit, minus the optimizations:
`quantize` (color.py) → `segment` (segment.py) → `region_contours` (contour.py)
→ `fit_path` (fit.py) for each region's boundary → `surface_merge` for gradient
surfaces (gradient bands that form one smooth fill become one object). **No**
primitive recognition, **no** symmetry, **no** clone detection here. Output is
the ordered object list and a rendered ground-truth coverage reference per
object (its true region mask = the pixels that object is responsible for).

### Stage 2 — Geometry optimizer passes

Fixed order (each pass feeds cleaner geometry to the next):

- **2a primitives.** Fit each object's `flat` to a circle / ellipse / rect /
  axis-aligned polygon (reuse the `recognize_primitive` algorithm, now applied
  to the object's geometry rather than a fresh raster contour). On accept, set
  `exact` to the primitive.
- **2b clones.** Find objects whose `flat` are congruent up to **translation +
  rotation** (reflection is deferred to 2c). Fill may differ. Collapse to a
  shared definition referenced by `<use>` with a `fill` override. Catches
  repeated icons / squares / rotated repeats.
- **2c symmetry.** Reuse the region-level detection algorithms already built in
  `symmetry.py` (candidate-axis voting from object self-axes and pair
  bisectors, clustering, the absolute area-aware test) — but applied to object
  `flat` geometry. A self-symmetric object is reconstructed exactly about its
  axis; a mirror **pair** becomes one object plus a `<use>` mirrored twin.
  Because detection now runs on drawn objects (merged surfaces included), it
  catches Instagram's outer symmetry (one merged object) and daikonic's radish
  (distinct symmetric bands) alike.
- **2d simplify.** Reduce each object's path: merge adjacent segments into a
  single curve where one fits, collapse near-collinear runs to a line, drop
  redundant control points.
- **(optional) geometric merge.** Adjacent objects with identical fill whose
  union is simpler than the parts. Candidate pass; include only if a real case
  needs it (gradient merge already handled in Stage 1).

### The gate (no-regression safety net)

Every proposed change in every pass is gated identically:

1. Take the changed object's optimized `flat`.
2. Rasterize its **coverage** (fill the polygon to a boolean mask) — **no fill
   color, no gradient, no compositing**; geometry is unchanged-fill, so only
   coverage can move.
3. Compare to the object's **true region mask** (its Stage-1 pixels): the
   normalized symmetric-difference area.
4. `≤ budget` → accept the change; otherwise reject and keep the faithful
   object.

Where both sides are polygons, the comparison can be a pure `shapely`
symmetric-difference-area computation with no rasterization at all.

This unifies all passes under one criterion — it is the generalized form of the
absolute off/peri symmetry test, now serving every optimization:

- primitives: does the circle cover the region's pixels?
- symmetry: does the mirrored half cover the region's pixels?
- simplify: does the reduced curve cover the same pixels?

Every accepted optimization is therefore provably within `budget` coverage of
the faithful ground truth. A change that drops a glyph, force-mirrors an
asymmetric shape, or lumps a curve raises the symmetric-difference and is
rejected automatically.

### Gate rasterization performance

The gate is the inner loop of every optimization (hundreds of evaluations per
image), so its cost dominates. v1 keeps it cheap without a native dependency:

- **Coverage only** (no fill/gradient/composite) — already far cheaper than a
  full SVG render.
- **Local bbox** — rasterize/compare only the changed object's bounding box,
  not the full image.
- **Incremental** — cache the ground-truth reference; a change re-evaluates only
  its own object.
- A compiled (Rust/native) polygon-fill + symmetric-difference kernel is the
  prime **deferred** perf target (bigger payoff than the `idealize()` exact-algo
  wins, since it is the optimizer's inner loop). Out of scope for v1.

## Preserve / Retire

**Preserve (reuse):**

- Stage-1 front end: `color.py` (quantize/OKLab), `segment.py`, `contour.py`
  (`region_contours`, rdp), `fit.py:fit_path`, `_fitcurve.py`, fills
  (`fill_fit.py`, `gradient.py`, `surface_merge.py`), `occlusion.py` (z-order).
- The **algorithms** behind the optimizations: `recognize_primitive` (→ 2a),
  `symmetry.py` axis-voting + absolute test + reflection (→ 2c), curve/polyline
  reduction (→ 2d) — relocated to operate on object `flat` geometry.
- `score.py` render-ΔE machinery → the gate.
- conditioning (`pipeline.py`, opt-in input prep).

**Retire / rewrite:**

- Raster region-level symmetry wiring in `_render_body` (logic relocates to 2c).
- Primitive/symmetry interleaved into Stage-1 selection (Stage 1 emits faithful
  paths only).
- Mask-based adjacency / occlusion / symmetry-reflection → `shapely` polygon
  operations on `flat`.
- `pipeline.py` orchestration → rewritten two-stage.
- The component/region coupling (`components.py` decomposition) → revisited; at
  the object level the distinction largely vanishes.

## Testing

- **Per pass, on synthetic objects:** 2a recognizes a circle/rect object; 2b
  dedups a translated+rotated repeat with differing fill; 2c detects a
  self-symmetric object and a mirror pair, reconstructs exact; 2d reduces a
  multi-segment run to one curve. Each asserts the emitted geometry.
- **Gate, adversarial:** a deliberately bad change (mirror an asymmetric shape;
  fit a circle to a square; lump a real curve) is **rejected** by the coverage
  budget; a faithful change is accepted. This is the regression net and gets
  explicit RED→GREEN tests.
- **Corpus:** Stage-1 faithful output is the per-object ground truth. Assert
  every optimized logo's objects stay within `budget` of their true masks.
  Named cases that must hold: Instagram outer shape symmetric (one merged
  object); daikonic radish exactly symmetric **and** "Daikonic" text faithful
  (no dropped glyphs); icloud cloud NOT mirrored (asymmetric → gate rejects);
  burger_king smooth (full resolution); microsoft/gdrive no dropped glyphs.
- **Determinism:** byte-identical across repeated runs; pass order fixed; all
  ordering value-based.

## Scope

v1 = the foundation (Stage 1 faithful, dual-rep object model, optimizer
framework, coverage gate) **plus all four passes** (primitives, clones,
symmetry, simplify). Native gate rasterizer and the optional geometric-merge
pass are deferred follow-ons. This is a large single plan touching the pipeline
core; the implementation plan should sequence it so Stage 1 + the gate land and
are provable before any pass is enabled, then passes are added one at a time
behind the gate.

## Open implementation notes (for the plan)

- Symmetry on `flat`: reflect the shapely polygon and use symmetric-difference /
  IoU rather than mask-EDT reflection; the absolute test becomes a polygon-area
  ratio.
- Clone congruence: match by normalized shape descriptors (area, moments,
  turning function) then verify by best-fit transform + coverage gate.
- `budget`: a single coverage tolerance (normalized symmetric-difference), set
  at rasterization-noise scale; it is the one knob and it is grounded in
  sampling, not tuned per logo.

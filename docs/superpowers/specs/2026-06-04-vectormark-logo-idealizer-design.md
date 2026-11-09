# vectoreyes — deterministic logo idealizer (design)

**Date:** 2026-06-04
**Status:** Approved for planning
**Working name:** `vectoreyes` (alts: `logovector`, `brandvector` — placeholder until repo creation)

## 1. Summary

A **deterministic** tool that converts a *rendered* (raster) logo into a clean,
editable, exactly-symmetric SVG. Where conventional tracers (`vtracer`,
`potrace`, `autotrace`) chase pixel edges and emit anonymous `<path>` soup,
`vectoreyes` **recognizes structure**: it collapses anti-aliasing to a true
palette, detects symmetry, and fits *ideal primitives* (`<ellipse>`, `<rect>`,
regular polygons) to regions — falling back to fitted Bézier paths only where no
primitive matches. The output SVG doubles as a human-editable parametric model.

No model/LLM is in the loop. Every stage is a known, deterministic algorithm.

## 2. Goals / non-goals

**Goals**
- Turn a flat-color, segmented logo raster into a brand-grade SVG.
- Output is **semantic and editable**: native SVG primitives + `<use>` mirroring.
- **Exact** bilateral symmetry when the source is symmetric (n-fold in v1.1).
- Fully deterministic and reproducible (same input → same bytes).
- Architecture that ports cleanly from a Python prototype to a Rust core.

**Primary acceptance criterion (v1):** converting the **Daikonic mark**
(`tests/fixtures/daikonic/source.png`) end-to-end is a gating requirement, not
just a fixture. The output must:
- render within tolerance of the source (SSIM ≥ target / pixel-ΔE ≤ target);
- emit each region as a clean **parametric `<path>`** that captures its TRUE
  shape — the bands are **rounded trapezoids** (slight taper + filleted corners),
  NOT sharp rectangles; the dome is a half-ellipse cap; the tip and leaves are
  petal/teardrop paths. Whole-region native primitives (`<rect>`/`<ellipse>`/…)
  are emitted only when a region genuinely *is* one (sharp, untapered);
- be **exactly bilaterally symmetric** — axis-straddling regions (dome, bands,
  tip) are fit by their half-outline and mirrored (`refine.symmetric_fit`); the
  leaf pair is a `<use>` mirror. Fold-SSIM about the axis ≈ 1.0;
- use the unified flat palette (the three anti-aliased "navies" collapsed to one);
- be re-editable (deterministic re-render).

The reference SVGs in `tests/fixtures/daikonic/` are loose artifacts, NOT the
target — and SSIM-against-the-source-raster only sets a floor (it rewards
matching anti-aliasing). The real quality bar is exact symmetry + faithful,
clean shapes.

**Non-goals (v1)**
- Photographic / heavily shaded logos (gradients arrive in v1.1).
- Wordmark / text reconstruction (treat text regions as opaque shapes or skip).
- **Regions with holes / enclosed counters** (letter bowls, donuts): v1 keeps the
  outer contour only. Multi-contour `fill-rule="evenodd"` support is **v1.1** —
  most real logos with negative space need it, but the Daikonic mark has none.
- Interactive/GUI editing — output is plain SVG, edited with any tool.
- Beating a tracer on *arbitrary* raster art; the value is in *logo* structure.

## 3. Scope

| Version | Stages | Capability |
|---|---|---|
| **v1** | A0 → B → C | Flat-color segmented marks → structured SVG |
| **v1.1** | + A1 + D | Realistic logos via normalize→idealize→re-gradient |

v1.1 bolts on without touching the v1 core: A1 (per-region gradient detection)
and D (re-apply gradients to idealized primitives) sandwich the flat core.

## 4. Pipeline

```
INPUT raster (PNG)
  │
  ▼ A0  COLOR OPTIMIZATION                                     [v1, required]
  │      robust palette: histogram-peak / interior-sampling, OKLab space,
  │      ΔE-merge of near-duplicates → assign every pixel to nearest palette
  │      color. Anti-aliasing halo is DISCARDED as noise (hard quantize).
  │      → clean flat N-color image.   (gradient detection deferred to A1)
  │
  ▼ B   TRACE + CLEAN                                          [v1]
  │      vtracer per color layer · drop background plate (border-touching /
  │      max-area fill) · palette already unified by A0.
  │      → per-region masks/contours.
  │
  ▼ C   IDEALIZE  (the differentiator, fully deterministic)   [v1]
  │      C1 symmetry axis    PCA orientation on union silhouette → refine
  │                          offset by min reflection-residual (mask XOR) →
  │                          gate: none │ bilateral │ n-fold (residual/area<ε)
  │      C2 fundamental dom   bilateral→half, n-fold→wedge, none→whole
  │      C3 contour           marching-squares / Suzuki–Abe per region
  │      C4 simplify+corner   Ramer–Douglas–Peucker → curvature-split segments
  │      C5 fit               PRIMITIVE-FIRST: whole-region circle (Taubin),
  │                           ellipse (Fitzgibbon), rect/rounded-rect, regular
  │                           polygon — emit native shape if residual < ε.
  │                           PATH-FALLBACK: per-segment line / circular arc /
  │                           cubic Bézier (Schneider, Graphics Gems).
  │      C6 assemble          simplest-model-under-ε wins (line<arc<ellipse<
  │                           bézier); G1 joins at smooth pts, hard corners
  │                           where detected.
  │      symmetry application: GEOMETRY-LEVEL MIRROR (never reflect the raster)
  │        · reflected-pair regions → fit canonical, emit <use scale(-1,1)>
  │        · axis-straddling regions → fit ALL pixels with axis-constraint
  │          (ellipse cx=axis, rect centered); path-fallback averages the two
  │          half-contours then mirrors.
  │
  ▼ D   RE-APPLY FILLS                                         [v1.1]
  │      attach A1 fill models as <linearGradient>/<radialGradient> on the
  │      idealized primitives.
  ▼
OUTPUT  structured.svg   (+ optional flattened.svg via --flatten)
```

### A0 detail — why not k-means
Anti-aliasing produces many blend pixels along every edge; k-means is dragged by
them and can spawn phantom palette entries. True colors are dense **modes** in
color space; blends are a sparse ridge between modes. Extract the palette by
peak-finding / interior-sampling (erode region, sample core), merge by ΔE in
**OKLab**, then hard-assign. The AA halo carries sub-pixel edge information, but
we **discard it as noise** — C5's curve fitting averages out the resulting
±0.5px stair-stepping, so nothing meaningful is lost. The palette colors become
the output fill colors; palette size = region/layer count.

## 5. The IR *is* the SVG

No bespoke JSON intermediate. The idealizer writes **structured SVG**:
- recognized primitives → native `<ellipse>` / `<rect rx>` / `<circle>` /
  `<polygon>` (the parameters, e.g. `rx`, *are* the editable model);
- everything else → minimal `<path>`;
- symmetry → `<use transform="scale(-1,1)">` referencing the canonical half;
- fit metadata (residual, alternative candidates) → `data-*` attributes.

Editing by hand (tweak `rx`) re-renders deterministically. `--flatten` converts
to optimized `<path>`s via `svgo` for maximum portability. Known boundary: pure
SVG can't express deeper parametric constraints ("band width = silhouette(y)");
acceptable for flat segmented marks.

## 6. Module boundaries (port-ready)

Pure, IO-free core as a chain of functions over plain data:

```
decode → quantize(A0) → segment(B) → symmetry(C1/C2) → fit(C3–C6) → emit(SVG)
```

The only library-heavy edges are B (vtracer) and the SVG emitter. The
**structured-SVG output is the stable contract**, identical across the Python
prototype and the eventual Rust core. Each unit is independently testable: input
data → output data, no shared mutable state.

## 7. Language & port strategy

- **Prototype: Python.** `numpy`/`scipy`/`scikit-image` (regions, moments,
  contours), `shapely` (geometry), `Pillow` (raster), `vtracer` bindings. The
  geometry/CV work is where Python's ecosystem saves the most effort.
- **Destination: Rust core**, exposed via `napi-rs` (npm), `PyO3`/`maturin`
  (PyPI), and WASM (browser). Ported once algorithms are proven; the prototype's
  durable output is the **algorithm + golden test corpus**, not reused code.
- Keep the core pure/IO-free from day one so the port is mechanical.

## 8. Interface

- **CLI:** `vectoreyes in.png [-o out.svg] [--flatten] [--epsilon N]
  [--colors N] [--no-symmetry]`.
- **Library:** `idealize(image, opts) -> svg_string` plus the staged functions
  for advanced use.
- **Output:** `structured.svg` by default; `--flatten` adds/overrides with the
  path-only variant.

## 9. Testing (deterministic ⇒ golden + render-diff)

- **Golden corpus:** input PNG → committed expected SVG; byte-diff regression.
- **Render-back-and-diff:** rasterize output, compare to source by SSIM / pixel
  ΔE under threshold — catches "valid SVG that looks wrong."
- **Property tests:** symmetric input ⇒ exactly-symmetric output; idempotent
  re-trace; primitive-recognition recall on synthetic ellipses/rects/polys.
- **Fixture #1 (gating):** the Daikonic mark
  (`tests/fixtures/daikonic/source.png`), with `reference-geometric.svg` as the
  target and `reference-trace.svg` as the baseline (see §2 acceptance criterion).

## 10. Open items (resolve at plan time)

- **Repository:** standalone repo at `active/py/vectoreyes` (**resolved**).
- **Primitive vocabulary v1:** circle, ellipse, rect, rounded-rect, regular
  polygon (**resolved**); teardrop/tip stays Bézier path-fallback; capsule/
  stadium deferred.
- **n-fold symmetry:** v1 ships **bilateral only**; n-fold detection/wedge-
  replication deferred to **v1.1** (**resolved**). Pipeline stays n-fold-ready.
- **Packaging:** PyPI distribution name (verify `vectoreyes` availability);
  license MIT.
- **Default ε values** for symmetry gate, primitive recognition, RDP — tune on
  the corpus (Daikonic mark first).

## 11. Algorithm references

- Suzuki & Abe (1985) — border following (`cv2.findContours`).
- marching squares — sub-pixel contours (`skimage.measure.find_contours`).
- Ramer–Douglas–Peucker — polyline simplification.
- Taubin (1991) / Kåsa — circle fitting.
- Fitzgibbon, Pilu & Fisher (1999) — direct least-squares ellipse fitting.
- Schneider (1990), *Graphics Gems* — automatic cubic Bézier curve fitting.
- image moments / PCA — symmetry-axis orientation.
- OKLab (Ottosson 2020) — perceptual color distance for palette ΔE.

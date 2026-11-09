# Raster→Vector / Autotracing / Font Tracing — Algorithm Inventory

**Date:** 2026-06-06
**Purpose:** Survey of existing raster-tracing and raster-to-font tools and the specific
algorithms inside them, scored for what vectoreyes could leverage. vectoreyes is a
**deterministic** flat-color raster→SVG **idealizer** (numpy/scipy/scikit-image/shapely):
it segments colored regions, then per region either snaps to an ideal primitive
(circle/ellipse/rect/convex polygon/annulus) or fits a smooth outline (lines + Béziers,
corner detection, even-odd holes), plus symmetry and occlusion reconstruction.

**License is load-bearing.** vectoreyes is MIT; GPL/CPL code is disqualifying for code reuse.
As a general rule an algorithm/idea is not protected by copyright (only the specific source
is), so a GPL tool's *math* can typically be reimplemented clean-room while its *code* cannot
be copied — but this is a rule of thumb, not legal advice: patents can cover techniques, and
clean-room reimplementation must avoid copying the original expression. Check specifics before
relying on any single algorithm.

## Foundational positioning: idealize vs. reproduce

Font autotracers and vectoreyes share a **bottom layer** (raster region → clean outline:
contour follow → corner detect → Bézier fit → even-odd counters), but diverge on intent:

- **Autotracers reproduce.** A glyph's bowl carries hand-tuned optical corrections (overshoot,
  stroke modulation, ink traps). Regularizing it to a primitive would *destroy* the typeface,
  so font tools deliberately never snap to primitives, never enforce symmetry.
- **vectoreyes idealizes.** A logo's circle is *meant* to be a perfect circle drawn imperfectly;
  recognizing that and snapping to `<circle>` is the win.

Consequence: vectoreyes's distinctive layers — primitive recognition, symmetry detection, and
especially **occlusion reconstruction** (recovering hidden geometry) — have **no font-world
analog** (glyphs are tuned imperfections to preserve, and glyphs don't occlude each other).
Confirmed: **no permissive open tracer** does primitive-snapping or symmetry; only closed-source
(Vectorizer.AI) and ML (StarVector) do. vectoreyes's idealization layer has no off-the-shelf OSS
equivalent. A font is also "define a glyph once, reference by codepoint" with **composite glyphs**
(é = e + ´ with a transform) — exactly vectoreyes's `<use>` mirror for symmetry; the generalization
is **motif detection** (n-fold rotational symmetry = one motif + k rotated `<use>` refs).

## Verdict table

| Candidate | License | vectoreyes target | Action |
| --- | --- | --- | --- |
| **Taubin / Halíř–Flusser** low-bias circle/ellipse fit | free (math) | `occlusion._fit_circle`, `complete_annulus`; `fit.recognize_primitive` | **Consider** — partial-arc fits in occlusion are bias-prone |
| **`volkerp/fitCurves`** (Schneider cubic) | **MIT** | `_fitcurve.fit_quadratic_beziers` | Evaluate vs current quadratic fitter |
| **potrace** optimal-polygon + α-corner test | GPL (algo only) | `fit.fit_path` + `contour.corner_indices` | **Clean-room, later** — biggest fallback-quality lever |
| **VTracer** color-layer / stacked model | MIT | segmentation + occlusion z-stack | Reference / validation |
| **shapely `orient()`** winding normalization | BSD (already a dep) | annulus even-odd, fill-rule emit | Cheap DRY win |

## 1. Bitmap tracers

### potrace (Peter Selinger) — the reference design for the smooth-outline fallback
**License: GPL-2.0-or-later** (pypotrace and the pure-Python `tatarize/potrace` are also GPL —
reference-only reads, not dependencies). The *paper* fully documents the algorithm; reimplement
clean-room. Pipeline:

1. **Path decomposition (local).** Lattice-point graph; trace closed contours keeping black-on-left;
   a `turdsize` area threshold despeckles (area via shoelace ∮ y dx).
2. **Optimal polygon (global — the clever part).** A subpath is "straight" if one segment
   approximates it within distance ½ at every vertex; each "possible segment" gets a penalty =
   |vⱼ−vᵢ|·(stddev of point distances to the chord), in closed form from prefix sums; the optimal
   polygon is the min cycle ordered **lexicographically (fewest segments, then least penalty)** —
   non-local, immune to staircase artifacts. **This is strictly better than RDP/VW for raster input.**
3. **Vertex adjustment.** Per edge fit a least-squares line (covariance eigenvector); place each
   vertex at the intersection of its two adjacent fit lines, clamped to the unit square.
   *(vectoreyes's `complete_polygon` independently uses exactly this — TLS line per edge, intersect
   adjacent lines — so the approach is validated. The clamp is the one guard we omit.)*
4. **Corner detection + smoothing.** α = 4γ/3; corner iff α > αmax (default 1) → two segments,
   else a Bézier. Uniquely favors corners by **both sharp angle and long segment**. The constant
   0.55 ≈ 4/3(√2−1) is the optimal quarter-circle Bézier constant.
5. **Curve optimization (optional, global).** Merge consecutive Béziers under an equal-enclosed-area
   constraint within tolerance ε; shortest-path decompose.

### autotrace (Martin Weber)
**Program GPL-2.0+; libautotrace LGPL-2.1+** (link-only if you must; no maintained Python binding).
Local edge-follow + angle/curvature corner detection + Schneider Bézier fitting. Multi-color (per
layer) and a **centerline/skeleton mode** (the one differentiator worth noting). Selinger shows
potrace beats it on quality/speed/size.

### VTracer (VisionCortex) — the MIT multi-region color tracer
**License: MIT**, Rust core with **Python bindings** (`pip install vtracer`) and WASM. Hierarchical
color clustering; `stacked` mode (layers paint over each other → maps to even-odd/painter's-model
holes) vs `cutout`; walker traces outlines; splice-point + spline fit (`corner_threshold` in deg).
Claimed O(n). Best permissive reference for **deterministic multi-region layering**; does NOT snap
to primitives or detect symmetry.

### ImageMagick / mkbitmap
ImageMagick has no real tracer (pixel-blocks) — skip. mkbitmap (GPL) is a *preprocessor*
(invert → high-pass → scale → threshold); largely irrelevant for flat-color input, trivial to
reimplement with skimage if needed.

## 2. Curve fitting

### Schneider's algorithm (Graphics Gems, 1990) — the standard Bézier fitter
Least-squares piecewise cubic Bézier: estimate endpoint tangents → chord-length parameterize →
solve for the two interior control points → **Newton–Raphson reparameterization** → split at max
error and recurse (large error near an endpoint naturally forces a G¹ corner).

- Graphics Gems `FitCurves.c` — permissive (public-domain-ish), MIT-compatible.
- **`volkerp/fitCurves` — Python/NumPy, MIT** — small, auditable, directly vendorable. **This is the
  natural "smooth outline fallback" engine** if vectoreyes's current quadratic fitter
  (`_fitcurve.fit_quadratic_beziers`) ever needs upgrading; cubics fit with fewer segments.

Potrace's α=β single-parameter family is a simpler/faster alternative producing compact convex
segments by construction. Direct bezigon optimization (arXiv 1602.01913) is research-grade — skip.

## 3. Polyline simplification & corner detection

- **RDP** (distance tolerance) — already used in vectoreyes (`contour.rdp`). Fast; can leave spikes.
- **Visvalingam–Whyatt** (triangle-area tolerance) — perceptually smoother, ~3.5× slower. (BSD impls
  in `shapely.simplify`, `urschrei/simplification`.)
- **Corner detection:** angle threshold (autotrace/VTracer; vectoreyes's `contour.corner_indices`
  uses a 40° threshold) → potrace's α-test (angle **and** length; clean-room) → Curvature Scale Space
  (robust but heavy; reserve). For raster staircase boundaries, potrace's optimal-polygon beats RDP/VW.

## 4. Raster-to-font

No raster-to-font tool contributes a novel core tracer — they all **wrap potrace/autotrace**:

- **FontForge** (GPL-3.0): "Import → trace" shells out to potrace (preferred) or autotrace, then adds
  font-grade post-steps worth borrowing **conceptually**: **Simplify**, **Round to int** (em-grid
  quantization), autohinting, and **Correct Direction** (winding normalization: PostScript wants
  clockwise outers / non-zero, TrueType counter-clockwise). vectoreyes gets winding via shapely's
  `orient()` (BSD) — relevant to emitting correct `fill-rule` and hole orientation (cf. the annulus
  even-odd work).
- **mftrace** (GPL): METAFONT → bitmap → potrace/autotrace → Type1. Orchestration only.
- **Glyphs / Fontographer**: closed-source; wrap autotrace/potrace-class engines; same
  em-grid/winding concepts.

Net: raster-to-font adds **no new tracing algorithm**, but surfaces three font-grade post-steps
(grid quantization, winding normalization, overlap removal) — all achievable with shapely (BSD).

## 5. Shape / primitive fitting & symmetry (classical, deterministic)

- **Circle:** Kåsa (algebraic, fast, **biased toward small arcs**), **Taubin** (algebraic, removes
  most bias — near-geometric quality at algebraic speed), geometric LM (best, iterative). skimage
  `measure.CircleModel` (BSD) is already in the stack; deterministic with fixed input (don't use
  RANSAC unseeded).
- **Ellipse:** **Fitzgibbon** direct LSQ (guarantees an ellipse via 4ac−b²=1); **Halíř–Flusser** is
  the numerically stable reformulation to prefer; Taubin/HyperLS lower-bias variants. skimage
  `measure.EllipseModel` (BSD) already available.
- **Rect / convex polygon / annulus:** no tracer provides these — build deterministically with
  shapely (`minimum_rotated_rectangle`, polygon-with-holes) + skimage + numpy (vectoreyes already does).

**Bias note (highest practical relevance):** occlusion completers fit circles to *partial arcs*
(occluded disks/rings) — precisely where Kåsa/algebraic bias toward small arcs bites. **Taubin /
Halíř–Flusser would reduce that bias** in `occlusion._fit_circle` / `complete_annulus` at the same
speed. Flag if occluded-circle radii ever come out slightly small.

## Highest-leverage shortlist

1. **`volkerp/fitCurves` (MIT)** — vendorable Schneider cubic fitter; the smooth-outline fallback engine.
2. **skimage CircleModel/EllipseModel (BSD, already present)** — primitive branch; pair with
   **Taubin/Halíř–Flusser math** for low-bias fits on partial occluded arcs (most relevant to the
   occlusion code).
3. **Potrace optimal-polygon + α-corner test** — clean-room; the single biggest quality lever for the
   non-primitive `fit_path` fallback. Strictly better than RDP/VW on raster staircases.
4. **VTracer (MIT, Python bindings)** — reference for deterministic multi-region/even-odd layering;
   optionally a baseline to call.
5. **shapely (BSD)** — winding normalization (`orient()`), even-odd holes, simplification, rect/hull
   primitives. Borrow FontForge's *concepts*, implement with shapely.

## License watch-list

- **Avoid code reuse (GPL/LGPL):** potrace, pypotrace, tatarize/potrace, mkbitmap, mftrace,
  FontForge, autotrace program (libautotrace is LGPL — link-only).
- **Safe to vendor/use:** `volkerp/fitCurves` (MIT), Graphics Gems `FitCurves.c` (permissive),
  VTracer (MIT), scikit-image / shapely / scipy (BSD).
- **Algorithms are always free to reimplement** regardless of any reference implementation's license.

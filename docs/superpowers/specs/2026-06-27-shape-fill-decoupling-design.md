# Decoupling Shape from Fill in the vectoreyes Pipeline — Design

**Status:** approved (brainstorming)
**Date:** 2026-06-27
**Area:** `src/vectoreyes/pipeline.py`, `src/vectoreyes/gradient.py`, new fill/merge modules; `Region`/`Candidate` types unchanged.

## Goal

Make geometry (a shape's silhouette) and fill (flat / linear / radial / raster) **entirely independent**. A shape's boundary is derived once, from clean region masks, and is never rebuilt from quantized color bands. Fills are fit afterward, over the existing silhouette, from the source pixels. Within-shape shading is a *fill* property and never creates or moves a boundary.

## Motivation (observed failure)

On a gradient logo (the "V-bird": blue→navy and orange→red wing gradients, three dots, a connector, on near-white), the current pipeline:

- **Frays every silhouette edge** — the gradient path reconstructs a footprint from quantized bands and grows it over background (`_expand_footprint`), so the outer contour comes out ragged/"eaten away" instead of from the clean region mask.
- **Shatters a small feature** — the middle blue dot, which is the same blue family as the wing, gets absorbed by the wing's color-similarity merge and breaks into a noisy blob.
- **Bands gradients** — when detection does *not* fire, a smooth wing is emitted as 2–4 stacked flat paths.

Root cause: `Region` binds mask to color ("one connected single-colour area"), so the segmenter splits a gradient into many regions, and the gradient machinery rebuilds the silhouette from those bands. Geometry and fill are coupled at construction time (`build_candidates` pairs `Shape` + `Fill` per region/footprint).

The palette ceiling (`max_colors`) is **not** the lever — 16/64/256 produce byte-identical output; the defect is in the footprint/merge path, not quantization.

## Architecture: five stages, single responsibilities

```
arr ─► [1 segment] ─► Regions ─► [2 shape pass] ─► clean shapes
                                                      │
                                  [3 fill pass] ◄─────┘  fit per-shape fill from source
                                       │
                                  [4 continuity-merge] ── fill-informed: union masks + fuse gradients,
                                       │                  then RE-FIT geometry on the merged silhouette
                                  [5 emit] ─► SVG (one path/primitive per final shape)
```

1. **Segment** — *reuse as-is.* Clean flat `Region`s (mask + color). Color is a detection signal, not a fill commitment. Masks are clean because no footprint reconstruction runs here.

2. **Shape pass** — *reuse the existing flat machinery with the gradient path disabled.* Symmetry (`detect_axis`), occlusion (`reconstruct_scene`), corner-rounding (`region_corner_radius`), and geometry recognition (`select_geometry`) all run on clean flat regions, producing clean "pre-colored" shapes — exactly the path that was already good for flat logos. The gradient footprint code does **not** touch geometry.

3. **Fill pass (new), per shape** — for each shape, fit the best fill from the **source pixels under its existing silhouette**:
   - **Flat** — the region is ~uniform → its own `color_hex` is the fill (`FlatFill`).
   - **Gradient** — a smooth ramp → fit linear/radial via the searched `_best_parametric(mask, rgb)` and emit `LinearGradientFill`/`RadialGradientFill`.
   - **Raster** — last resort for non-parametric tone (`RasterFill`).
   Geometry is read-only in this stage; a fill decision never moves an edge.

4. **Continuity-merge (new), fill-informed — a hybrid criterion** keyed on the seam, never on color similarity:
   - **(B) gradient continuity (primary).** When both adjacent shapes fit gradients, merge if one gradient's color *at the shared seam* matches the other's within a ΔE (`seam_de`) — lead-meets-tail. This is the natural case for surfaces that segmented into wide-enough pieces.
   - **(A) soft-seam fallback.** When a *narrow* region devolved to a flat fill (too little color span to fit a gradient), there is no per-region gradient to compare, so merge instead when the **source** has no hard edge across the shared border — the median straddling-pixel OKLab ΔE is below `edge_de`. A within-gradient band seam is soft; a real object edge — even a same-color feature's border — spikes.
   - **Both paths additionally require the union to fit a parametric gradient** (so two genuinely distinct flats abutting never merge). The merged fill IS that union gradient.
   - **Merge:** union the two clean masks (the internal seam vanishes; the outer contour stays clean), and **re-fit geometry on the merged silhouette** via the full `select_geometry` recognition — so the merged surface can be promoted to a *primitive* (`<rect>`/`<polygon>`/`<circle>`), not just a freeform path ("what was a path could be a shape"). Re-run `region_corner_radius` on the union so corners are smooth over the whole surface, never kinked at an old band seam.
   - Iterate to a fixed point so a chain of bands collapses into one gradient.
   - A flat dot sitting on the gradient wing has a **hard source edge** at its border (it never matches a soft seam) and is uniform (the B path never applies) → it never merges and is never swallowed, even when its color lies on the wing's ramp.

5. **Emit** — *reuse.* One `<path>` or primitive per final shape; flat/gradient/raster fill in `<defs>`.

## Why this fixes both bugs

- **Edges:** the only sources of a silhouette are (a) a clean region mask and (b) a union of clean region masks. `_expand_footprint` is gone, so nothing grows a contour over background.
- **Dot:** the dot is its own region/shape; the continuity test keys on the *fill function* across the seam, so a same-color uniform feature is never merged into a ramp.

## Data model

`Region` (mask + color) and the `Candidate(shape, fill, kind, …)` / `Fill` hierarchy are **unchanged** — the decoupling is in the *staging*, not the types. A "surface" is simply a `Region` whose mask is the union of band masks and whose `color_hex` is a representative shade (used only for detection/flat fallback). The new work is two stages (fill, continuity-merge) plus retiring the old gradient path.

## Stages retired / replaced

`detect_gradients`, `_component_fill`, and especially `_expand_footprint` (gradient.py) are **replaced** by stages 3–4, not run in parallel. `merge_components`' color-step union-find is superseded by continuity-merge. `_best_parametric` (the searched linear/radial fit) is **kept and reused** by the fill pass. `fit_gradient` (the rung-1 heuristic) is no longer on the main path.

## Error handling / edge cases

- **Single-region gradient** (a smooth area that segmented as one region): the per-region fill fit (stage 3) handles it directly; no merge needed.
- **Coarsely-quantized bands** (hard-ish internal seams): continuity uses the same ΔE family already calibrated in the repo (the `SYM_TOL`/`MERGE_DE` ≈ 0.045 neighborhood); if a real gradient's bands fail continuity, it degrades to clean stacked flats (today's failure mode for *those* bands) — never to a frayed edge. Tuning the tolerance is a calibration task in the plan.
- **Radial vs linear:** the fill pass tries both via `_best_parametric` and takes the better fit; merge fuses same-kind gradients (linear∥linear, radial∥radial) and declines to merge mismatched kinds.
- **Occlusion / symmetry:** unchanged — they run in stage 2 on flat shapes, before fills exist, so the decoupling does not perturb them.

## Testing

- **Acceptance (the V-bird):** edges crisp (silhouette ΔE/edge-roughness vs source within tolerance), the middle blue dot intact as one clean circle, each wing one shape with a single fitted gradient (one `<path>`/primitive + one `linearGradient`), no stacked bands.
- **Unit:** continuity test (continuous bands merge; a flat feature on a ramp does not); per-region fill fit (uniform→flat, ramp→gradient); merged-silhouette geometry promotion (two band-rects → one primitive); corner re-fit on a union.
- **Corpus regression:** flat logos must be byte-identical or visibly unchanged (stage 2 is the old flat path); gradient logos (firefox, instagram, pokéball, the glossy set) compared before/after for crisper edges and fewer elements. The corpus is the gate.

## Risks

- **False merges** — continuity tolerance too loose fuses distinct surfaces. Mitigation: require *both* boundary-color agreement and consistent ramp direction; calibrate against the corpus.
- **False non-merges** — coarse bands fail continuity → residual banding (no worse than today, and never a frayed edge). Mitigation: tolerance calibration; raster fallback for a surface that reads as continuous tone but won't fit parametric.
- **Corpus shift** — gradient outputs change by design; flats must not. Validate both halves explicitly.
- **Performance** — per-region parametric fits + iterative merge add cost; bounded by region count and a fixed-point cap.

## Out of scope

Manual candidate selection (`SelectionPolicy`) and the render-ΔE scorer (`score.py`) are untouched; they continue to operate on the candidates this pipeline emits. The variant gallery and palette-default question are separate follow-ups.

# Perceptual-Clustering Palette Extraction — Design Spec

**Status:** Approved (design). First slice of the candidate-pipeline roadmap
(`docs/architecture/2026-06-07-candidate-pipeline-roadmap.md`).
**Branch:** `feat/perceptual-palette`
**Date:** 2026-06-07

## Goal

Make `extract_palette` recover colours carried by **thin or small antialiased marks** — colours
whose pixels are *dispersed across many near-identical shades* — so logos like the settir blue
node-graph mark keep their colour instead of collapsing to `[white, navy]`.

## Motivation

`extract_palette` counts **exact** RGB shades, sorts them by individual frequency, then walks the
list adding a shade to the palette if it is ≥ `merge_de` (OKLab) from every already-picked shade —
with an early `break` the moment a shade falls below the `min_fraction` floor:

```python
for i in range(len(colors)):
    if counts[i] < min_fraction * total:   # <-- the bug
        break
    if all(delta_e(lab[i], lab[j]) >= merge_de for j in palette_idx):
        palette_idx.append(i)
```

A thin antialiased mark spreads its colour over hundreds of slightly-different shades (edge blends
against the background). Each individual shade ranks *below* `min_fraction`, so the loop `break`s
before blue is ever considered — even though blue's **aggregate** pixel count is large. The greedy
`merge_de` clustering that would have grouped those shades never runs, because the floor fires first.

Measured on settir: palette returned `[white, navy]`; the blue (~22k px, AA-dispersed, median
≈ (1,131,253)) was dropped entirely. Render ΔE through `idealize` was 0.0096 with blue missing.

## The fix

Reorder the algorithm: **cluster first, aggregate counts per cluster, then apply the floor to
cluster totals** (not to per-shade crumbs). The representative colour for each cluster is its
**most-frequent member** (an exact colour that actually occurs — not a centroid that may render as
an off colour).

This is the smallest possible expression of the roadmap's "colour is a separable concern"
principle: a self-contained change to one function, no pipeline restructuring.

### Algorithm

`extract_palette(rgb_image, *, max_colors=16, merge_de=0.045, min_fraction=0.002)`:

1. **Coarse pre-bin (performance guard).** Quantize each channel to ~5 bits (`>> 3`, i.e. 32
   levels/channel) and `np.unique` on the binned image to get the distinct binned shades and their
   summed counts. This caps the clustering input at ≤ 32³ candidates regardless of image size or AA
   spread, keeping the O(shades²) greedy loop bounded. The pre-bin is used **only** to group shades
   for counting; the representative returned is a real (full-precision) colour (see step 4).
2. **Frequency sort.** Sort the binned shades by descending aggregate count (deterministic;
   ties broken by shade value to remove `np.unique` order dependence).
3. **Greedy perceptual clustering with a seed-floor (the floor is applied here, not at the end).**
   Walk bins in frequency order; for each, if it is within `merge_de` (OKLab) of an existing
   cluster's representative, absorb it (add its count). Otherwise it may **seed a new cluster only if
   its own aggregate count ≥ `min_fraction * total_pixels`**; a sub-floor bin that matches no
   existing cluster is dropped. Because we walk in frequency order, the bin that seeds a cluster is
   its most-frequent member → it is the representative.

   *Why the floor gates seeding rather than being applied once at the end* (revised during
   implementation): applying the floor only after building every cluster lets mid-gradient AA-blend
   bins chain together via `merge_de` into a spurious cluster that creeps just over the floor — which
   reintroduces exactly the blend-colour false positives the band-image regression forbids. Gating
   *seeding* by the floor keeps genuine marks (whose dominant bin is comfortably supra-floor) while
   refusing to let sub-floor blend crumbs originate a colour. Sub-floor bins are still **absorbed**
   into a real cluster they are perceptually close to, so AA edges of a true mark still count toward
   its weight.
4. **Representative = most-frequent member.** The seed bin of each cluster (highest count, since
   frequency-ordered) is the representative, mapped back to the **most-frequent full-precision colour
   within that bin**, so the returned colour is one that genuinely appears in the image, not the bin
   corner.
5. **Cap.** Every surviving cluster is supra-floor by construction (only supra-floor bins seed), so
   no separate end-of-pipeline floor pass is needed. Sort clusters by aggregate count descending;
   keep the top `max_colors`.
6. **Fallback.** If no bin reaches the floor (every bin sub-floor) but the image is non-empty, return
   the single most-frequent bin's representative (preserves current behaviour for degenerate inputs).
   Empty image → empty `(0, 3)` array.

### Why this design (vs. alternatives)

- **Centroid representative — rejected.** A cluster centroid can be a colour that never occurs (e.g.
  the average of blue-vs-white AA blends is a washed mid-blue). The most-frequent member is always a
  real colour and renders faithfully. (User-confirmed.)
- **k-means — rejected.** Non-deterministic without a fixed seed, needs a `k`, and is heavier than
  the greedy single-pass clustering. vectoreyes is deterministic by contract; the greedy pass is
  deterministic and already the codebase idiom.
- **No pre-bin — rejected.** `np.unique` on a photographic-AA logo can yield >100k exact shades; an
  O(shades²) greedy loop over that is too slow. The 5-bit pre-bin bounds it without perceptually
  meaningful loss (32 levels/channel is well below `merge_de`). (User-confirmed keep.)
- **Lowering `min_fraction` — rejected.** Doesn't fix the ordering bug; it just moves the cliff. The
  dispersed colour still never aggregates because clustering runs after the floor.

## Architecture & placement

Single function rewrite in `src/vectoreyes/color.py::extract_palette`. **Signature unchanged**
`(rgb_image, *, max_colors, merge_de, min_fraction)` so every caller (`segment.py`, pipeline,
tests) is unaffected. Return type unchanged: `(N, 3) uint8`, frequency-ordered. `srgb_to_oklab`,
`delta_e`, `quantize`, `mean_delta_e` are untouched and reused.

No other module changes. The fix is entirely internal to one function.

## Error / edge handling

- Empty / single-colour image → fallback returns the one most-frequent colour (step 6).
- All colours within `merge_de` of each other → one cluster, one representative.
- Determinism: frequency sort with value tiebreak removes `np.unique` ordering dependence;
  no `set` iteration; identical input → identical palette.

## Testing

Test-first (TDD), through the public `extract_palette` and end-to-end through `idealize`.
No committed brand assets — synthetic fixtures only.

1. **Thin-AA-line fixture (the core regression).** Synthesize a white canvas with a 1–2px-wide
   antialiased coloured line/mark (rasterize an AA stroke so the colour is dispersed across many
   shades, each individually below `min_fraction`). Assert the dispersed colour **appears** in the
   returned palette (nearest-palette ΔE to the true colour ≤ a small bar). Assert the *old*
   algorithm would have dropped it (documents the bug being fixed — e.g. a direct assertion that the
   colour's max single-shade fraction is below `min_fraction`).
2. **Representative is a real colour.** Assert every returned palette colour occurs in the input
   image (no synthesized centroids).
3. **Flat / posterized inputs unchanged.** A handful of distinct flat blocks → palette equals those
   block colours (no spurious extras, no drops). Guards against over-clustering.
4. **Determinism.** Same image twice → identical palette array.
5. **`max_colors` / `min_fraction` honoured.** More-than-`max_colors` distinct blocks → exactly
   `max_colors` returned, the most frequent ones. A block below the aggregate floor → excluded.
6. **Committed settir-style fixture through `idealize`.** A synthetic logo in the settir spirit
   (navy wordmark-ish blocks + a thin blue mark on white) committed to `tests/`, run through
   `idealize`, asserting the blue mark **keeps its colour** in the output (render ΔE improves vs. a
   computed no-blue baseline, or the emitted SVG contains the blue fill). Synthetic, not the brand
   asset.
7. **Full regression — including the gradient acceptance suite.** The entire existing test suite
   stays green. Gradient detection depends on palette behaviour (band-grouping reads
   `extract_palette`), so the gradient acceptance fixtures are an explicit part of the gate.

## Known limitation (from the seed-floor rule)

Recovery requires the mark's colour to have **at least one 5-bit bin whose aggregate count reaches
the floor**, not merely a supra-floor sum spread across many sub-floor bins. A genuine thin mark has
perceptually stable hue, so its shades concentrate in one bin neighbourhood and its dominant bin is
comfortably supra-floor (the settir-style fixture's blue peaks well above the floor) — so realistic
marks are recovered. The narrowing only drops a *hyper-dispersed* colour smeared so smoothly across
many bins that no single bin reaches the floor while the total does; such a colour is gradient-like
rather than a clean palette entry, and rejecting it is consistent with the band-image regression
(AA-blend smears must not become palette colours). This is the *safe* failure direction (a miss,
not a false colour).

## Non-goals

- Changing the palette **signature** or callers.
- Changing `merge_de` / `min_fraction` **defaults** (only the order in which `min_fraction` is
  applied changes).
- Touching `quantize`, segmentation, or the gradient path.
- The broader candidate-pipeline refactor (slices 2–5 of the roadmap) — this is slice 1 only.

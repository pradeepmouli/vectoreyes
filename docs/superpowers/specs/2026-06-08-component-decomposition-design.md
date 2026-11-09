# Component Decomposition (Roadmap Slice 5) — Design

**Status:** approved design, pre-implementation
**Date:** 2026-06-08
**Builds on:** the full candidate pipeline (slices 1–4b, all merged). Final roadmap slice.

## Goal

Split a mark into spatially-separated **components** via a gutter-based front-end, then
run the existing per-mark analysis (symmetry, corner-radius, occlusion, gradients,
candidate selection) **independently per component**. This:

1. Gives each component its **own vertical symmetry axis** (a mark whose whole
   silhouette has no mirror, but whose parts do, now idealizes each part symmetrically).
2. **Unblocks the deferred multi-blob gradient case** — `detect_gradients` per component
   sees a single blob, so two separately-gradient'd components each pass the
   `_dominant_blob_fraction` single-dominant-blob gate.

A mark with no qualifying gutter is **one component** and idealizes **byte-identically**
to today (parity by construction).

## Non-goals (YAGNI)

- **Per-component tilt.** The whole-mark tilt rectification at the top of `idealize`
  (`_idealize_rectified`) stays unchanged; decomposition runs *inside* `_render_body`,
  i.e. in whatever frame (upright or already-rectified) it receives. Components tilted
  differently from each other are out of scope.
- **An `Options` knob for the gutter threshold.** Use internal constants derived from
  mark scale (mirrors `min_region_fraction` / corner-radius precedent). No new API.
- **Local component coordinates / translation.** Components keep full-canvas coords.
- **Manual control of decomposition** (no policy/override for component boundaries).

## Architecture

The only genuinely new logic is a front-end `decompose_components`. Everything else is
the existing per-mark code, invoked in a loop. Decomposition slots **inside**
`_render_body`, which already encapsulates "given a region subset + raster, detect
symmetry/occlusion/gradients and emit."

### New module: `src/vectoreyes/components.py`

```python
def decompose_components(regions: list[Region], shape: tuple[int, int]) -> list[list[Region]]:
    """Partition regions into spatially-separated components via recursive X-Y cut on
    the union silhouette. Returns components in reading order (top→bottom, left→right).
    A mark with no qualifying gutter (or <=1 region) returns [regions] (one component),
    which is the parity path."""
```

Recursive X-Y cut:

```
decompose(regions):
    if len(regions) <= 1: return [regions]
    sil  = union of region masks
    bbox = tight bbox of sil
    cut  = best_gutter(sil, bbox)                 # widest qualifying full-span gap (H or V)
    if cut is None: return [regions]              # atomic block — base case
    side_a, side_b = partition regions by pixel-centroid side of `cut`
    return decompose(side_a) + decompose(side_b)  # recurse; reading order preserved
```

- **Gutter detection.** For a horizontal cut, project the silhouette onto the vertical
  axis (sum over columns within the bbox); rows with sum 0 are empty; a maximal run of
  empty rows strictly interior to the bbox is a candidate gutter. Transposed for a
  vertical cut. Choose the **widest** qualifying gutter across both axes; on a tie,
  prefer the cut that splits the blocks most evenly. Leading/trailing background (outside
  the tight bbox) is not a gutter.
- **Qualifying threshold (conservative, scale-relative).** A gutter qualifies iff its
  thickness ≥ `max(_GUTTER_ABS_FLOOR, _GUTTER_FRACTION × extent_along_cut_axis_of_block)`.
  Defaults: `_GUTTER_FRACTION = 0.3`, `_GUTTER_ABS_FLOOR = 6` (px). Tuned so obvious
  multi-element marks split while borderline intra-mark gaps do not (e.g. the two-band
  logo's ~8px gap on a ~44px-tall block ≈ 18% < 30% → no split). The constants are tuned
  against the existing goldens; any fixture that splits is surfaced and adjudicated.
- **Region assignment** by pixel-centroid side of the cut. The gutter is empty by
  construction, so no region's pixels lie in the cut band — every region falls cleanly to
  one side (no sliced regions; a component is always a clean partition of whole regions).
- **Reading order.** A horizontal cut yields top-then-bottom; a vertical cut yields
  left-then-right. The recursion therefore emits components in reading order, which
  becomes the `sN` order.

### `_render_body` change (`src/vectoreyes/pipeline.py`)

`_render_body` currently computes one mark-wide `axis`/`corner_radius`, runs
`reconstruct_scene`, `detect_gradients`, `classify_regions`, and `build_candidates` over
**all** regions, then a single emit loop. The change: decompose first, run that same
analysis **per component**, concatenate the per-component candidate lists in reading
order into one list, then run the **unchanged** single emit loop.

```python
def _render_body(w, h, regions, opt, *, bake=None, rgb=None):
    components = decompose_components(regions, (h, w))
    defs: list[str] = []
    all_cands: list[Candidate] = []
    for comp in components:
        silhouette = np.any([r.mask for r in comp], axis=0)
        axis = None if opt.no_symmetry else detect_axis(silhouette)
        corner_radius = opt.corner_radius if opt.corner_radius is not None else _mark_corner_radius(comp, axis)
        reconstructed, comp = reconstruct_scene(comp, axis, (h, w))
        grad_fills = []
        if rgb is not None:
            grad_fills, comp = detect_gradients(comp, rgb)
        if axis is not None:
            straddlers, pairs, loners = classify_regions(comp, axis)
        else:
            straddlers, pairs, loners = list(comp), [], []
        all_cands += build_candidates(reconstructed, straddlers, pairs, loners,
                                      grad_fills, opt, axis, corner_radius, rgb)
    # ... existing single emit loop over all_cands (unchanged): assigns s0..sN globally,
    #     applies bake / _fill_attr / mirror_use exactly as today.
    return body, defs
```

- **Continuous `sN`.** Concatenating per-component candidate lists then running the
  existing single emit loop assigns global continuous ids — 4b's `sN` addressing stays
  intact, the emit loop is untouched.
- **Per-component `axis`/`corner_radius`/occlusion/gradients** — the heart of the slice.
- The `bake` (flatten inverse transform) and `_fill_attr` logic stay in the single emit
  loop, so flatten composes unchanged.

## Parity

Slice-4a playbook. Gate = full acceptance suite + byte-identical golden harness
(`tests/test_candidate_byte_identical.py`). A mark with no qualifying gutter → one
component → identical inputs to today → byte-identical (no re-capture). Any golden that
diverges (a fixture with a qualifying gutter) is investigated case-by-case; genuine
improvements are re-baselined **and surfaced to the user** — never silently decided.

## Multi-blob gradient unblock

Automatic from per-component `detect_gradients` — no gradient code changes. A two-component
mark, each component a single gradient blob, now emits two gradients; today the second
fails `_dominant_blob_fraction` when both blobs are projected together.

## Error handling & edge cases

- **0 / 1 region** → `[regions]` unchanged.
- **No qualifying gutter** → single component (common case; parity path).
- **Nested layouts** (icon above a two-word row) → recursion yields icon, then the two
  words — handled by the recursive cut.
- **Occlusion across a gutter** — impossible: `reconstruct_scene` runs per component and a
  gutter is empty, so an occlusion group never spans one. Matches the existing
  per-canonical-group scoping.
- **`no_symmetry` / `flatten` / tilt-rectified frame** — decomposition runs on the regions
  in whatever frame `_render_body` receives; `bake` is applied uniformly in the single
  emit loop, so flatten and tilt-rectified output compose unchanged.

## Testing

### Unit (`tests/test_components.py`)
- Single blob → 1 component.
- Two widely-separated blobs (circle left, square right, wide gutter) → 2 components,
  reading order left→right.
- Vertical stack with a wide gap → 2 components, top→bottom.
- Borderline narrow gap (two-band-logo geometry) → **1 component** (threshold protects it).
- Nested (icon over a two-element row) → 3 components in reading order.
- Partition integrity: union of returned components == input regions, with no region in
  two components (clean partition).

### Integration (`tests/test_pipeline.py`)
- **Parity:** existing suite + goldens stay green (the gate).
- **Per-component vertical symmetry:** a mark whose whole silhouette has no vertical axis
  but each component does → each component emits mirrored geometry.
- **Multi-blob gradient:** two gradient blobs separated by a gutter → two gradient fills.
- **Continuous `sN`:** a two-component mark emits `s0…sN` with no gaps or collisions across
  the component boundary.

### Regression
- Byte-identical golden harness re-run unchanged (no re-capture unless a divergence is
  adjudicated an improvement and surfaced).

## Files

- **Create:** `src/vectoreyes/components.py`, `tests/test_components.py`
- **Modify:** `src/vectoreyes/pipeline.py` (`_render_body` loops over components; import
  `decompose_components`), `tests/test_pipeline.py` (per-component symmetry, multi-blob
  gradient, continuous-`sN`, parity tests).

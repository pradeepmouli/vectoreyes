# Gradient Handling Design

**Status:** Approved (design) — pending implementation plan
**Date:** 2026-06-06
**Depends on:** the palette/segment flow (`color.py`, `segment.py`), the shape recognizers
(`fit.recognize_primitive`/`recognize_polygon`/`fit_path`), the emit/doc layer (`emit.py`), and the
OKLab color utilities (`color.srgb_to_oklab`). Reuses the `region_adjacency` grouping idea and the
group→fit→consistency-gate→fall-back pattern from the occlusion pass.

## Problem

vectoreyes's color model is flat-fill-per-region: `extract_palette` → `quantize` → `segment`
produces `Region`s each with a single `color_hex`. A genuine gradient in the input is therefore
**shattered into many flat bands** — the "path soup / too many colors" failure mode. There is no way
to recognize that a set of bands is one smooth color ramp and emit a single shape with an SVG
gradient fill.

This feature adds **gradient detection and reconstruction**: recover a gradient's spatial footprint
from its quantized bands, fit a linear or radial color model against the **original** image, and —
when it re-renders faithfully — emit one shape with a `<linearGradient>`/`<radialGradient>` fill
instead of many flat bands.

## Goals & scope

- **Fidelity first.** Detection fits color-as-a-function-of-position against the **original
  pre-quantization RGB**, so genuine gradients (modern / AI-generated marks, the generate→idealize
  plugin pipeline) reproduce faithfully — not merely a re-collapse of whatever quantization kept.
- **Linear + radial** gradients. They share the fit machinery (a 1-D color ramp over a geometric
  parameter — axis projection for linear, distance-from-center for radial).
- **Safety contract (unchanged from occlusion):** a gradient is emitted **only** when its fit
  re-renders within a perceptual error bar; otherwise the footprint dissolves back into today's flat
  bands. A flat or non-ramp region can never be falsely gradient-ified.

### Explicit non-goals (YAGNI)

- **Conic gradients** (angle parameter) — a fast-follow reusing this machinery.
- **Gradient + occlusion interaction** — a region is claimed by occlusion **or** gradient, not both;
  occlusion runs first. A shape that is both gradient-filled and occluded is not reconstructed as such.
- **Alpha/transparency gradients** and **gradient meshes** (multi-axis).
- **Forcing gradient-axis symmetry.** A gradient footprint still mirrors via `<use>` when symmetric
  (the fill reference rides along), but the axis is not snapped to a symmetry line.
- Very complex ramps that don't gate simply **fall back to flat bands** — the contract, not a bug.

## Architecture & pipeline placement

A new `gradient.py` module, integrated in `pipeline.py` after segmentation and occlusion, operating
on the regions occlusion did not claim, with the **original RGB** threaded in.

```
load → extract_palette → quantize → segment (flat Regions)
   → reconstruct_scene (occlusion)                 [unchanged] -> (reconstructed, remaining)
   → detect_gradients(remaining, original_rgb)     [NEW]       -> (gradient_shapes, remaining')
        group adjacent ramp-colored regions -> footprint
        -> fit linear|radial model vs original pixels
        -> consistency gate (mean OKLab ΔE) -> accept | fall back
   → fit remaining' flat regions                   [unchanged]
   → emit (+ <defs> gradients)                      [extended]
```

Priority order per region: **occlusion member → gradient footprint → flat region**, each stage
consuming from the pool. `detect_gradients` mirrors `reconstruct_scene`'s contract: returns
`(gradient_shapes, remaining)`.

Two contained consequences:

1. **Thread the original RGB to the detector.** Today `_segment_image` returns `(w, h, regions)` and
   the original array is dropped after quantization. Carry the original `rgb` array through so
   `detect_gradients` can fit against true colors. Small plumbing change.
2. **Gradient is a fill, orthogonal to shape.** A footprint's outline still goes through the existing
   shape recognition; only its flat `fill` is swapped for `fill="url(#g0)"`. Any shape kind can carry
   a gradient — no new geometry.

## Components

### 1. Footprint recovery (band grouping)

Build adjacency over the leftover flat regions (`region_adjacency`). A gradient's bands are spatially
adjacent **and** their flat colors progress monotonically along a 1-D path in OKLab. Grow connected
groups whose member colors are **ramp-consistent** — near a single line/curve in OKLab, not doubling
back. Require **≥ 3 bands** (two flat colors are two regions, not a gradient). The ramp test is a
cheap pre-filter; the consistency gate is the real arbiter.

### 2. Gradient model fit (`fit_gradient`)

Fit against the **original RGB** pixels under the footprint mask, in OKLab.

- **Recover the per-pixel geometric parameter `t`:**
  - *Linear:* fit each OKLab channel as a plane `c ≈ a·x + b·y + c₀`; the shared gradient direction
    `u` (averaged across channels, weighted by variation) is the axis; `t = ((x,y)·u)` normalized to
    `[0,1]`.
  - *Radial:* estimate the center `(cx,cy)` (color extremum / isotropy point), then `t = ‖(x,y)−c‖`
    normalized. Try linear first; if the planar residual is high but a radial fit is tight, take
    radial.
- **Fit stops:** sort pixels by `t`, sample ~5 initial stops (each = median original color in its `t`
  neighborhood), then **greedily drop** stops whose removal keeps the piecewise-linear color error
  low — the minimal stop set that still reproduces the ramp.

### 3. Consistency gate

Render the fitted gradient (each footprint pixel ← its modeled color) and compute **mean OKLab ΔE**
vs the original under the footprint. Accept iff below a threshold; on reject the footprint dissolves
back into its flat bands. (The `mean_delta_e` helper in `tests/_render.py` is the same metric —
promote it into the `color` module so production and tests share it.)

`detect_gradients` returns `(gradient_shapes, remaining)`, each gradient shape carrying its
recognized outline plus the fitted fill `{kind: "linear"|"radial", stops, geometry}`.

### 4. Emit (gradient `<defs>`)

- `emit.linear_gradient_def(id, x1,y1,x2,y2, stops)` → `<linearGradient>` and
  `emit.radial_gradient_def(id, cx,cy,r, stops)` → `<radialGradient>`, each with
  `<stop offset="…" stop-color="#…"/>` entries, in **`userSpaceOnUse`** (absolute pixel coordinates)
  so no `gradientTransform` is needed and they survive `--flatten` unchanged.
- `render_svg_doc` gains an optional `defs: list[str]`; when non-empty it wraps them in
  `<defs>…</defs>` before the body.
- The footprint shape emits normally but is handed `fill="url(#g0)"` instead of a hex — `shape_to_svg`
  / `path_svg` already take the fill string, so no per-shape special-casing. Under `--flatten` it is a
  path with `fill="url(#g0)"` plus the def.

## Testing

**Unit** (`tests/test_gradient.py`)
- ramp-group detection: adjacent ramp bands group; a flat set / non-ramp set does not.
- `fit_gradient` linear: recovers a known axis + stops from a synthetic linear gradient (axis within
  tolerance, ΔE low).
- `fit_gradient` radial: recovers center + stops from a synthetic radial gradient.
- the gate **accepts** a true gradient and **rejects** a flat region and a two-color non-ramp region.
- greedy stop-reduction yields few stops on a smooth ramp.

**Acceptance** (`tests/test_acceptance_gradient.py`, through `idealize`)
- synthetic **linear-gradient** mark → one shape + one `<linearGradient>` with a small stop count,
  **mean ΔE ≤ threshold**.
- synthetic **radial-gradient** orb → one shape + one `<radialGradient>`.
- **regression:** a **flat** logo is NOT gradient-ified (no `<defs>` / `url(#…)`).
- a **multi-color non-ramp** region stays flat (no false merge).

**Regression:** the full existing suite stays green (gradients are additive; the flat path is
untouched).

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Flat / near-flat region falsely gradient-ified | ≥3-band requirement + ramp pre-filter + ΔE consistency gate (a flat fit has ~0 variation and a flat region already reproduces, so a gradient offers no gain and isn't emitted) |
| Multi-color non-ramp region merged | ramp-consistency (colors must lie on one 1-D OKLab path, monotonic in `t`) + the gate |
| Radial center mis-estimated | try linear first; accept radial only if its fit is tight and beats linear; the gate backstops |
| Banding re-introduced by too few stops | greedy stop reduction is bounded by the same ΔE error it must keep low |
| Gradient axis noisy on small footprints | normalized planar fit across all three OKLab channels; gate declines a poor fit |

## Acceptance summary

The feature is correct when: a synthetic linear gradient and a radial gradient each reconstruct to a
single gradient-filled shape that re-renders within a perceptual ΔE bar using few stops; flat and
non-ramp inputs are never falsely gradient-ified; and the existing suite is unchanged. The
consistency gate guarantees any gradient the pass emits re-renders to match the input.

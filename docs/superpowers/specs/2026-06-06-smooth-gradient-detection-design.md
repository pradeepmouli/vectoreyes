# Smooth-Gradient Detection — Design Spec

**Status:** Approved (design) — follow-up to PR #8 (`feat/gradients`).
**Branch:** `feat/gradients-smooth` (stacked on `feat/gradients`).
**Date:** 2026-06-06

## Goal

Detect a *smooth* (non-posterized) gradient that fills a logo's mark and emit it as one
gradient-filled shape, so real app-icon-style gradient logos (e.g. Telegram, Apple Music)
idealize to a gradient instead of collapsing to a flat colour.

## Motivation

PR #8 added gradient detection via **band-grouping**: it groups ≥3 adjacent quantized bands
whose flat colours form an OKLab ramp, fits a linear/radial model, gates on perceptual ΔE,
and emits a gradient def. A real-logo spike (Telegram, Apple Music, Slack, Microsoft, Google
Drive, Sketch, Instagram, Asana) showed band-grouping **never fires on smooth real gradients**:
`extract_palette`'s `merge_de` collapses a smooth ramp to 1–2 palette colours before grouping
runs, so the gradient never appears as ≥3 bands. The mark ends up as a single flat region.

Crucially, the spike also showed the **fit + gate engine is excellent**: fitting `fit_gradient`
directly to a region's *raw pixels* recovers Telegram at ΔE 0.0021 and Apple Music at 0.0027.
The bottleneck is purely the *front-end* (how candidates are formed), not the fit/gate.

## The detection rule (v1)

Detect a gradient when the mark is **essentially one gradient blob**:

1. `sil` = union mask of all regions not already consumed by band-grouping.
2. Label connected components of `sil`; `dom = largest_component_area / sil_area`.
3. Accept iff **`dom ≥ _BLOB_DOMINANCE` (0.85)** AND **`fit_gradient(sil, rgb_image)` returns a
   model** (which already requires mean OKLab ΔE ≤ `_GATE_DELTA_E`).
4. On accept: emit one gradient over `sil`, consume all the contributing regions.
5. Otherwise: leave `remaining` untouched.

### Why this rule, and why not the alternatives (spike evidence)

The single-dominant-blob guard — not fit quality — is what rejects flats. Telegram = 1 blob
(`dom=1.00`) → fit → accept; Asana = 8 disconnected glyphs (`dom=0.15`) → rejected structurally
before fit quality matters. On the 8-logo set this gives **8/8 correct**, zero false positives;
conic (Instagram) and flat-faceted (Drive, Sketch) are rejected by the gate returning `None`.

Rejected alternatives (all empirically false-positive-prone or unsound):
- **Per-region / per-component fitting** — a flat solid shape trivially fits a degenerate
  near-constant gradient (Microsoft squares dE 0.005, Slack blobs 0.001, Asana glyphs 0.004).
- **Variance / flat-residual threshold** — the *true* subtle gradient (Telegram, flat-residual
  0.011) ranks *below* false positives (Asana glyph 0.021, Instagram ring 0.096). No clean cut.
- **Compactness/fill guard** — flat squares are maximally compact (fill 1.00); useless.
- **Distinct-stop-span filter** — backwards: subtle true gradient spans 0.039, flat-multicolour
  false positive spans 0.55.

## Architecture & placement

All new logic lives in `src/vectoreyes/gradient.py`, inside `detect_gradients`, running **after**
the existing band-grouping loop on the **remaining** (unconsumed) regions. A mark is realistically
either posterized or smooth, so the two paths compose without overlap: band-grouping consumes
posterized ramps first; whatever remains is tested once as a smooth silhouette.

**No pipeline changes.** `detect_gradients` already returns `(fills, remaining)`, and the Task-7
emit loop in `_render_body` renders each `(footprint, model)` fill generically. The smooth path
appends one fill and removes the consumed regions — `pipeline.py` needs no edits.

### Components

- New private helper `_dominant_blob_fraction(mask) -> float` (or inline): `ndi.label` →
  largest-component / total foreground. Deterministic.
- New constant `_BLOB_DOMINANCE = 0.85`.
- Reuse unchanged: `fit_gradient`, `_agreement_delta_e`, the existing emit path.
- **No `_expand_footprint`** on this path — `sil` is already the complete footprint.

### Data flow

`detect_gradients(regions, rgb_image)`:
1. band-grouping loop → `fills`, `consumed` (existing behaviour).
2. `remaining = [r for r in regions if r.label not in consumed]`.
3. smooth path: `sil = union(remaining masks)`; if `_dominant_blob_fraction(sil) ≥ 0.85` and
   `model = fit_gradient(sil, rgb_image)` is not None → append `(Region(footprint=sil), model)`
   to `fills`, add those labels to `consumed`.
4. return `(fills, remaining-after-consumed)`.

## Error / edge handling

- No remaining regions, or multi-blob silhouette (`dom < 0.85`), or `fit_gradient` returns
  `None` → skip, return remaining unchanged.
- A small non-background distinct element *inside* the dominant blob is bounded by the gate: if
  it keeps mean ΔE ≤ `_GATE_DELTA_E` it is absorbed (rendered as the gradient colour there,
  within tolerance); otherwise the whole fit is rejected and the mark stays flat.
- Determinism preserved (`ndi.label` is deterministic; no set-order dependence introduced).

## Testing

End-to-end through `idealize`, synthetic fixtures (no committed brand assets). ΔE bars follow
PR #8's acceptance convention (render-ΔE, i.e. rasterize the output and compare to the input —
distinct from the internal `_GATE_DELTA_E` pixel-model gate):
- **Smooth radial disc** on white → exactly one `<radialGradient>`; render ΔE ≤ 0.07.
- **Smooth linear rounded-rect** → exactly one `<linearGradient>`; render ΔE ≤ 0.06.
- **Flat shape** → no `<defs>`, no `url(#`.
- **Two-blob (wordmark-like) smooth-coloured input** → no gradient (dominant-blob guard).
- **Regression:** full existing suite stays green; band-grouping fixtures unaffected.

Real brand logos (Telegram, Apple Music, …) remain in untracked `scratch/real-logos/` for manual
eval only — not committed, for brand-asset licensing reasons.

## Known limitation / future work

v1 only fires when the mark is essentially one gradient shape. A logo that is a gradient shape
**plus** a separate distinct element (multi-blob silhouette) is missed — the whole-silhouette fit
includes the extra element and fails the gate/blob guard. This is the *safe* failure direction
(miss, not false positive). Robust per-component detection is deferred: the spike showed it needs
a stronger discriminator than the simple guards tried (gate, flat-residual, compactness all
overlap between subtle true gradients and flat/conic false positives).

The rectified frame is produced by bilinear rotation (`ndi.rotate`), whose edge-smear can let a *contrived* high-contrast flat tilted mark be fit as a (faithful, low-ΔE) gradient. Real flat marks (e.g. the daikonic fixture) do not trigger this — their flat interiors make `fit_gradient` return None — and the ΔE gate keeps any such render faithful.

## Rectified-path gradient support (added after real-logo eval)

Real-logo testing showed Telegram (a gradient circle + diagonal paper plane) breaks vertical
symmetry, so `idealize` routes it through the **rectified** (tilted-symmetry) path — where PR #8
left gradients OFF — and it rendered flat, even though the upright fit was perfect (ΔE 0.0021).
Apple Music (no tilted symmetry) reached the upright path and worked. We therefore **reverse the
PR #8 non-goal** and support gradients in the rectified frame.

Approach (spike-verified):
- `_idealize_rectified` passes `rgb=rot` (the rotated image) to both `_render_body` calls and
  threads the returned `defs` into `render_svg_doc`.
- The gradient pass gate in `_render_body` relaxes from `rgb is not None and bake is None` to
  `rgb is not None`, so detection runs in the baked/rectified frame too.
- **Non-flatten:** gradient defs (`userSpaceOnUse`, rectified-frame coords) sit in `<defs>`; the
  gradient-filled shapes are inside the rotated `<g>`. A `userSpaceOnUse` gradient resolves in the
  referencing element's user space (inside the `<g>`), so gradient and shape rotate together —
  verified ΔE 0.0004 vs the rotated reference. No geometry transform needed.
- **Flatten:** path coords are baked to the original frame, so the gradient geometry is baked the
  same way via a new `_bake_gradient_geometry(geom, kind, bake)` helper (linear: transform both
  endpoints; radial: transform centre, keep `r` — the rectify affine is a rigid rotation+translation
  so radius is preserved). Keeps flatten's "no surviving transforms" philosophy.

## Non-goals

- Conic/sweep gradients (still unsupported; correctly gate-rejected).
- Multi-blob gradient-plus-element marks (future work, above).
- Changing `extract_palette` / `merge_de` (the smooth path fits raw pixels, so palette collapse
  is irrelevant — only the silhouette mask is needed, and that survives).

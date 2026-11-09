# Perceptual-Clustering Palette Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠️ Revised during implementation — read this first.** The literal "floor on cluster
> **totals** at the end" described below was found to create spurious AA-blend clusters (it broke the
> band regression). The **shipped** rule is a **seed-floor**: a bin may *seed* a new cluster only if
> its own aggregate count ≥ `min_fraction × total`; sub-floor bins may only be *absorbed* into an
> existing cluster. The design-of-record is the reconciled spec
> (`docs/superpowers/specs/2026-06-07-perceptual-palette-design.md`, §3 "Greedy perceptual clustering
> with a seed-floor") and the code in `src/vectoreyes/color.py`. Where this plan still says "floor on
> cluster totals" (the snippet in Task 1 Step 3 and the self-review), read "seed-floor" per the spec.

**Goal:** Rewrite `extract_palette` so a colour dispersed across many antialiased shades (a thin/small mark like the settir blue) is recovered, instead of being dropped by the per-shade frequency floor.

**Architecture:** Reorder the algorithm to **cluster perceptually → aggregate counts per cluster → gate cluster *seeding* by the `min_fraction` floor** (a bin seeds a new cluster only if its own aggregate ≥ floor; sub-floor bins are absorbed). A coarse 5-bit pre-bin caps the clustering input; each cluster's representative is its most-frequent member (a real colour, never a centroid). Single-function rewrite in `src/vectoreyes/color.py`; signature, return type, and all callers unchanged.

**Tech Stack:** Python, numpy, OKLab (`srgb_to_oklab`/`delta_e` already in `color.py`), pytest. End-to-end check via `idealize` + `tests/_render.py`.

**Spec:** `docs/superpowers/specs/2026-06-07-perceptual-palette-design.md`
**Roadmap context:** `docs/architecture/2026-06-07-candidate-pipeline-roadmap.md` (slice 1 of 5)
**Branch:** `feat/perceptual-palette` (already created, off `master`)

---

## Background the implementer needs

`extract_palette(rgb_image, *, max_colors=16, merge_de=0.045, min_fraction=0.002) -> (N,3) uint8`
returns a frequency-ordered palette. The **current** body (the code being replaced):

```python
flat = rgb_image.reshape(-1, 3)
colors, counts = np.unique(flat, axis=0, return_counts=True)
order = np.argsort(counts)[::-1]
colors, counts = colors[order], counts[order]
total = counts.sum()
lab = srgb_to_oklab(colors / 255.0)
palette_idx: list[int] = []
for i in range(len(colors)):
    if counts[i] < min_fraction * total:   # <-- BUG: breaks before dispersed colours aggregate
        break
    if all(delta_e(lab[i], lab[j]) >= merge_de for j in palette_idx):
        palette_idx.append(i)
    if len(palette_idx) >= max_colors:
        break
if not palette_idx and len(colors):
    palette_idx = [0]
return colors[palette_idx].astype(np.uint8)
```

The bug: shades are sorted by *individual* frequency; an antialiased colour spreads over hundreds
of near-identical shades, each below `min_fraction * total`, so the loop `break`s before that colour
is ever considered — even though its *aggregate* count is large.

**DO NOT change** `srgb_to_oklab`, `oklab_to_srgb`, `delta_e`, `mean_delta_e`, `quantize`, the
function signature, the return type (`(N,3) uint8`, frequency-ordered), or any caller. The fix is
internal to `extract_palette` only.

Existing tests that MUST stay green (they encode required behaviour):
- `tests/test_color.py::test_extract_palette_finds_two_true_colors_not_the_blend` — a navy/teal band
  image with a 1px AA blend row must return **exactly** `(6,35,54)` and `(61,168,157)`, no blend
  colour. (Guards against over-clustering and against adding AA blends.)
- `tests/test_color.py::test_quantize_collapses_blend_row`
- `tests/test_color.py::test_extract_palette_never_empty_on_gradient`
- `tests/test_acceptance_daikonic.py` — asserts exactly 4 fills and AA-navy collapse end-to-end.

`idealize` accepts a numpy array directly (`idealize(arr, options=Options(...))`) and returns an SVG
string. `Options` has a `no_symmetry` flag.

---

## Task 1: Rewrite `extract_palette` with perceptual clustering

**Files:**
- Modify: `src/vectoreyes/color.py:52-77` (the `extract_palette` body)
- Test: `tests/test_color.py` (append new tests; keep existing ones)

- [ ] **Step 1: Write the failing core-regression test (thin AA mark)**

Append to `tests/test_color.py`. Add `srgb_to_oklab` to the existing import on line 2 if not present
(line 2 already imports it). The fixture makes the blue genuinely un-recoverable by the old code —
every blue pixel is a slightly different AA shade, none reaching the floor.

```python
def _thin_aa_mark():
    """256x256 white canvas with a 3px-wide antialiased blue diagonal mark.
    Coverage varies smoothly per row, so the blue is dispersed across many AA
    shades — no single shade is frequent enough to survive the min_fraction
    floor, mirroring a real thin coloured mark (the settir blue)."""
    h = w = 256
    img = np.full((h, w, 3), 255, np.uint8)
    blue = np.array([1.0, 131.0, 253.0])
    white = np.array([255.0, 255.0, 255.0])
    covs = np.linspace(0.86, 0.99, h)
    for y in range(h):
        for dx, k in ((-1, 0.90), (0, 1.0), (1, 0.90)):
            x = y + dx
            if 0 <= x < w:
                cov = covs[y] * k
                img[y, x] = np.round(blue * cov + white * (1 - cov)).astype(np.uint8)
    return img


def test_thin_aa_color_no_single_shade_survives_floor():
    """Documents the bug: no single blue shade reaches the frequency floor, so
    only clustering (aggregate weight) can recover the colour."""
    img = _thin_aa_mark()
    flat = img.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    blueish = (colors[:, 2] > 200) & (colors[:, 0] < 120)
    assert blueish.any()
    assert counts[blueish].max() < 0.002 * len(flat)


def test_extract_palette_recovers_dispersed_thin_color():
    img = _thin_aa_mark()
    pal = extract_palette(img)
    pal_lab = srgb_to_oklab(pal / 255.0)
    true_blue = srgb_to_oklab(np.array([1.0, 131.0, 253.0]) / 255.0)
    nearest = min(delta_e(true_blue, c) for c in pal_lab)
    assert nearest <= 0.10, f"blue not recovered: nearest ΔE {nearest:.3f}; palette={pal.tolist()}"
```

- [ ] **Step 2: Run the core-regression test to verify it fails**

Run: `pytest tests/test_color.py::test_extract_palette_recovers_dispersed_thin_color -v`
Expected: FAIL — the old algorithm returns `[white]` (or white+navy), no blue, so `nearest` exceeds 0.10.
(`test_thin_aa_color_no_single_shade_survives_floor` should PASS — it characterizes the fixture.)

- [ ] **Step 3: Rewrite `extract_palette`**

Replace the body of `extract_palette` in `src/vectoreyes/color.py` (lines 52-77) with:

```python
def extract_palette(
    rgb_image: np.ndarray, *, max_colors: int = 16, merge_de: float = 0.045,
    min_fraction: float = 0.002,
) -> np.ndarray:
    """Perceptual-clustering palette in OKLab.

    Clusters near-identical shades *before* applying the frequency floor, so a
    colour dispersed across many antialiased shades (a thin/small mark) is kept
    by its aggregate weight instead of dropped per-shade. Each cluster's
    representative is its most-frequent member — a real colour, never a centroid.

    Returns an (N, 3) uint8 array, frequency-ordered.
    """
    flat = np.asarray(rgb_image, dtype=np.uint8).reshape(-1, 3)
    total = len(flat)
    if total == 0:
        return np.empty((0, 3), dtype=np.uint8)

    # Distinct full-precision colours and their pixel counts.
    colors, counts = np.unique(flat, axis=0, return_counts=True)

    # Coarse 5-bit pre-bin (32 levels/channel) caps clustering input regardless
    # of AA spread. Group full-precision colours by their bin.
    color_bin = colors >> 3
    bins, bin_inv = np.unique(color_bin, axis=0, return_inverse=True)
    bin_inv = bin_inv.ravel()
    nbins = len(bins)
    bin_total = np.zeros(nbins, dtype=np.int64)
    np.add.at(bin_total, bin_inv, counts)

    # Representative per bin = most-frequent full-precision colour in that bin.
    # Order colours by descending count, value-tiebroken for determinism; the
    # first colour seen for each bin (in this order) is its representative.
    order = np.lexsort((colors[:, 2], colors[:, 1], colors[:, 0], -counts))
    sorted_bins = bin_inv[order]
    uniq_bin, first_pos = np.unique(sorted_bins, return_index=True)
    rep_color_idx = np.empty(nbins, dtype=np.int64)
    rep_color_idx[uniq_bin] = order[first_pos]
    rep_color = colors[rep_color_idx]                       # (nbins, 3) full precision
    rep_lab = srgb_to_oklab(rep_color / 255.0)

    # Greedy perceptual clustering over bins, in descending aggregate-count
    # order (stable -> value-tiebroken via the value-sorted bin ids). The seed
    # bin of each cluster is its most-frequent member -> the representative.
    b_order = np.argsort(-bin_total, kind="stable")
    cluster_lab: list[np.ndarray] = []
    cluster_total: list[int] = []
    cluster_bin: list[int] = []
    for b in b_order:
        placed = False
        for k in range(len(cluster_lab)):
            if delta_e(rep_lab[b], cluster_lab[k]) < merge_de:
                cluster_total[k] += int(bin_total[b])
                placed = True
                break
        if not placed:
            cluster_lab.append(rep_lab[b])
            cluster_total.append(int(bin_total[b]))
            cluster_bin.append(int(b))

    # Floor on cluster TOTALS, then cap to max_colors, frequency-ordered.
    totals = np.array(cluster_total, dtype=np.int64)
    floor = min_fraction * total
    keep = [k for k in np.argsort(-totals, kind="stable") if totals[k] >= floor][:max_colors]
    if not keep:
        keep = [int(np.argmax(totals))]   # fallback: the single most-frequent colour
    return rep_color[[cluster_bin[k] for k in keep]].astype(np.uint8)
```

> **As-built deviation (see the banner at the top):** the shipped code does NOT apply the floor only
> here at the end — it gates cluster **seeding** (`if not placed and bin_total[b] >= floor:`), so
> sub-floor bins can only be absorbed, never seed. The end-of-loop floor above is then redundant
> (every seeded cluster is already supra-floor). The authoritative algorithm is the spec's §3 and
> `src/vectoreyes/color.py`.

- [ ] **Step 4: Run the core-regression test to verify it passes**

Run: `pytest tests/test_color.py::test_extract_palette_recovers_dispersed_thin_color -v`
Expected: PASS.

- [ ] **Step 5: Add the remaining spec tests (real representative, determinism, max_colors/floor)**

Append to `tests/test_color.py`:

```python
def test_palette_representatives_are_real_colors():
    img = _thin_aa_mark()
    pal = extract_palette(img)
    present = set(map(tuple, np.unique(img.reshape(-1, 3), axis=0)))
    for c in pal:
        assert tuple(int(v) for v in c) in present


def test_palette_is_deterministic():
    img = _thin_aa_mark()
    assert np.array_equal(extract_palette(img), extract_palette(img))


def test_palette_honours_max_colors():
    img = np.zeros((60, 60, 3), np.uint8)
    cols = [(200, 0, 0), (0, 200, 0), (0, 0, 200),
            (200, 200, 0), (0, 200, 200), (200, 0, 200)]
    for i, c in enumerate(cols):
        img[(i // 3) * 30:(i // 3) * 30 + 30, (i % 3) * 20:(i % 3) * 20 + 20] = c
    pal = extract_palette(img, max_colors=4)
    assert len(pal) == 4


def test_palette_excludes_below_floor_block():
    img = np.full((100, 100, 3), 255, np.uint8)
    img[:5, :1] = (200, 0, 0)        # 5 px / 10000 = 0.0005 < 0.002 floor
    pal = extract_palette(img)
    assert not any(tuple(int(v) for v in c) == (200, 0, 0) for c in pal)
```

- [ ] **Step 6: Run the full `test_color.py` suite**

Run: `pytest tests/test_color.py -v`
Expected: PASS — all new tests AND the pre-existing
`test_extract_palette_finds_two_true_colors_not_the_blend` (still exactly 2 colours),
`test_quantize_collapses_blend_row`, `test_extract_palette_never_empty_on_gradient`.

If `test_extract_palette_finds_two_true_colors_not_the_blend` fails (e.g. a blend bin survives the
floor as a third colour), the fixture's blend pixels are sub-floor by construction — investigate
clustering/floor logic, do NOT weaken that test.

- [ ] **Step 7: Run the full suite, including the gradient acceptance suite**

Run: `pytest -q`
Expected: PASS — entire suite green. Pay special attention to
`tests/test_acceptance_daikonic.py`, `tests/test_acceptance_gradient.py`,
`tests/test_acceptance_smooth_gradient.py`, and `tests/test_segment.py` (all consume
`extract_palette`). If `test_acceptance_daikonic.py` changes fill count, the clustering altered
brand-colour grouping — diagnose against the spec (the 4 brand colours must survive, AA navies
collapse) before adjusting anything.

- [ ] **Step 8: Commit**

```bash
git add src/vectoreyes/color.py tests/test_color.py
git commit -m "fix(color): perceptual-clustering palette recovers AA-dispersed colours

extract_palette now clusters near-identical shades, aggregates counts per
cluster, then applies the min_fraction floor to cluster TOTALS — so a colour
dispersed across many antialiased shades (a thin/small mark) survives instead
of being dropped by the per-shade frequency break. Representative per cluster
is its most-frequent member (a real colour). 5-bit pre-bin bounds the greedy
loop. Signature and callers unchanged.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: End-to-end acceptance — settir-style mark keeps its colour through `idealize`

**Files:**
- Test: `tests/test_acceptance_palette.py` (create)

This is the user-facing guarantee: a thin coloured mark beside larger blocks keeps its colour in the
emitted SVG. The synthetic logo is brand-free (no committed brand asset). With Task 1 in place this
passes; it would fail on the pre-fix code (blue quantized away → no blue fill).

- [ ] **Step 1: Write the end-to-end acceptance test**

Create `tests/test_acceptance_palette.py`:

```python
import re

import numpy as np

from vectoreyes import Options, idealize


def _settir_synth():
    """Synthetic settir-spirit logo (brand-free): three solid navy blocks (a
    wordmark stand-in) plus a thin antialiased blue diagonal mark on white. The
    blue is AA-dispersed like a real thin mark, so it only survives palette
    extraction via perceptual clustering."""
    h = w = 200
    img = np.full((h, w, 3), 255, np.uint8)
    navy = (10, 30, 70)
    img[150:170, 20:60] = navy
    img[150:170, 80:120] = navy
    img[150:170, 140:180] = navy
    blue = np.array([1.0, 131.0, 253.0])
    white = np.array([255.0, 255.0, 255.0])
    covs = np.linspace(0.86, 0.99, 100)
    for i in range(100):
        y = 20 + i
        for dx, k in ((-1, 0.90), (0, 1.0), (1, 0.90)):
            x = 40 + i + dx
            if 0 <= x < w and 0 <= y < h:
                cov = covs[i] * k
                img[y, x] = np.round(blue * cov + white * (1 - cov)).astype(np.uint8)
    return img


def _fills(svg):
    def rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    return [rgb(f) for f in re.findall(r'fill="(#[0-9A-Fa-f]{6})"', svg)]


def test_settir_style_mark_keeps_blue_through_idealize():
    svg = idealize(_settir_synth(), options=Options(no_symmetry=True))
    fills = _fills(svg)
    assert any(b > 180 and r < 120 and g > 90 for r, g, b in fills), \
        f"blue mark lost; fills={fills}"
```

- [ ] **Step 2: Run the acceptance test**

Run: `pytest tests/test_acceptance_palette.py -v`
Expected: PASS (the blue fill is present because Task 1 recovered the blue). It documents the
behaviour that would regress if the palette fix were reverted.

- [ ] **Step 3: Run the full suite once more**

Run: `pytest -q`
Expected: PASS — whole suite green including the new acceptance test.

- [ ] **Step 4: Commit**

```bash
git add tests/test_acceptance_palette.py
git commit -m "test(color): end-to-end acceptance — thin mark keeps colour through idealize

Synthetic brand-free settir-spirit logo (navy blocks + thin AA blue mark);
asserts the blue mark survives into the emitted SVG fills, guarding the
perceptual-palette fix end-to-end.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Cluster → aggregate → floor reorder → Task 1 Step 3 (the rewrite).
- 5-bit pre-bin performance guard → Step 3 (`color_bin = colors >> 3`).
- Representative = most-frequent member (not centroid) → Step 3 (`rep_color_idx`/`first_pos`) +
  `test_palette_representatives_are_real_colors` (Step 5).
- Signature unchanged → Step 3 keeps `(rgb_image, *, max_colors, merge_de, min_fraction)`.
- Determinism → Step 3 (lexsort + stable argsort, no sets) + `test_palette_is_deterministic`.
- Seed-floor (a bin seeds only if supra-floor; sub-floor absorbed) + cap → Step 3 (as-built; see
  banner) + `test_palette_honours_max_colors`, `test_palette_excludes_below_floor_block`.
- Fallback for degenerate input → Step 3 (`if not keep: keep = [argmax]`) + empty-image early return.
- Thin-AA-line fixture, dispersed colour appears, old-would-drop documentation → Step 1.
- Flat/posterized unchanged → existing `test_extract_palette_finds_two_true_colors_not_the_blend`
  is an explicit gate in Step 6.
- Committed settir-style fixture through idealize → Task 2.
- Full regression incl. gradient acceptance → Step 7 + Task 2 Step 3.

All spec sections map to a task. No gaps.

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step gives the
exact command and expected result.

**3. Type consistency:** `extract_palette` keeps signature and `(N,3) uint8` return throughout.
Helpers `_thin_aa_mark`/`_settir_synth`/`_fills` are each defined before use. `srgb_to_oklab`,
`delta_e` are imported in `tests/test_color.py` (lines 2-3) and re-used; Task 2 imports only
`Options`, `idealize`, `numpy`, `re`. Consistent.

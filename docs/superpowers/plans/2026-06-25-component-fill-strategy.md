# Per-Component Fill Strategy (Colour-Step Merge) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace gradient detection's band-merge + whole-blob smooth path with a unified colour-step region merge, then choose a fill strategy (flat / true gradient / raster) per merged component — so multi-hue smooth fields reconstruct cleanly while sharp features (outlines, facet edges) stay crisp.

**Architecture:** Within each gutter-component, agglomeratively merge spatially-adjacent regions whose OKLab colour step ≤ `MERGE_TOL` into single components (generalizes `_ramp_groups` from collinear ramps to any locally-smooth field). Each component then independently picks a fill: strict parametric gradient → searched parametric → raster stretch-fill → flat. Reuses the already-built primitives (`RasterFill`, `_best_parametric`, `_fit_stretch`, pipeline wiring).

**Tech Stack:** Python 3.12+, numpy, scipy, Pillow, pytest.

## Global Constraints

- Python ≥ 3.12; pure-Python changes. DRY is the #1 rule (reuse existing helpers; do not duplicate fit/merge logic).
- Determinism: union-find and merges in stable label order; fixed grids; no randomness/time.
- **Parity:** the end-to-end acceptance tests `tests/test_acceptance_gradient.py`, `tests/test_acceptance_smooth_gradient.py`, and the surviving `tests/test_gradient.py` cases must pass — marks that emit one `<linearGradient>`/`<radialGradient>` today must still do so. If the unified merge cannot preserve a specific existing-gradient output, keep `_ramp_groups` for the collinear case and apply the merge only to the residual (additive fallback) — flag this to the controller before doing it.
- `MERGE_TOL` and all thresholds are corpus-validate-before-merge starting values.
- Commit trailer exactly `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`, no other trailer.
- `score.py` imports `_BLOB_DOMINANCE` and `_dominant_blob_fraction` from `gradient.py` for its own gradient gate — do NOT remove those two symbols even though `detect_gradients` no longer uses them.

---

### Task 1: `merge_components` — colour-step agglomerative merge

**Files:**
- Modify: `src/vectoreyes/gradient.py` (add `MERGE_TOL` constant + `merge_components`)
- Test: `tests/test_gradient.py`

**Interfaces:**
- Consumes (existing): `region_adjacency` (from `.occlusion`), `_hex_to_oklab`, `Region`.
- Produces:
  - `MERGE_TOL = 0.15` (module constant)
  - `merge_components(regions: list[Region], *, tol: float = MERGE_TOL) -> list[list[Region]]` — groups of regions; each group is a connected set whose adjacent colour steps are all ≤ `tol`. A region with no small-step neighbour is its own singleton group. Deterministic order (groups sorted by their minimum label).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_gradient.py`:

```python
def test_merge_components_merges_small_steps_into_one():
    from vectoreyes.gradient import merge_components
    # 4 adjacent bands stepping blue->magenta (small OKLab steps between neighbours)
    regions = _hstrip_regions(["#2563eb", "#7b3fc4", "#b13a9e", "#db2777"])
    groups = merge_components(regions, tol=0.15)
    assert len(groups) == 1 and len(groups[0]) == 4


def test_merge_components_splits_at_large_step():
    from vectoreyes.gradient import merge_components
    # a small-step pair, then a large jump to a distinct hue, then another small-step pair
    regions = _hstrip_regions(["#2563eb", "#3a6ae0", "#11aa33", "#15b53a"])
    groups = merge_components(regions, tol=0.15)
    labels = sorted(sorted(r.label for r in g) for g in groups)
    assert labels == [[1, 2], [3, 4]]                 # split at the blue->green jump


def test_merge_components_singleton_when_isolated_by_large_steps():
    from vectoreyes.gradient import merge_components
    # zig-zag hues: every adjacency is a large step -> no merges -> all singletons
    regions = _hstrip_regions(["#ff0000", "#00ff00", "#0000ff", "#ffff00"])
    groups = merge_components(regions, tol=0.15)
    assert sorted(len(g) for g in groups) == [1, 1, 1, 1]


def test_merge_components_transitive_chain():
    from vectoreyes.gradient import merge_components
    # a long chain of small steps merges end-to-end even though the ends are far apart
    regions = _hstrip_regions(["#2563eb", "#5a4fd0", "#8a44b4", "#b13a9e", "#db2777"])
    groups = merge_components(regions, tol=0.15)
    assert len(groups) == 1 and len(groups[0]) == 5
```

`_hstrip_regions` already exists in this test file (horizontal adjacent bands). Reuse it.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k merge_components -v`
Expected: FAIL — `merge_components` not defined.

- [ ] **Step 3: Add `MERGE_TOL` and `merge_components`**

In `src/vectoreyes/gradient.py`, add the constant near the other module constants (after `_STRETCH_TARGET`):

```python
MERGE_TOL = 0.15   # max OKLab colour step between two spatially-adjacent regions for them
                   # to merge into one vector component. Below this = one smooth field (gradient
                   # bands, within-facet shading); above = a real boundary (facet edge, outline).
                   # Corpus: within-field steps <=0.12, boundary steps >=0.27 -> clean gap.
```

Add the function (place it near `_ramp_groups`, which it generalizes — above `detect_gradients`):

```python
def merge_components(regions: list[Region], *, tol: float = MERGE_TOL) -> list[list[Region]]:
    """Agglomeratively merge spatially-adjacent regions whose OKLab colour step is <= tol
    into single components (union-find over region_adjacency). Generalizes _ramp_groups from
    collinear ramps to any locally-smooth field: a region with no small-step neighbour is its
    own singleton group. Deterministic (groups ordered by their minimum label)."""
    by_label = {r.label: r for r in regions}
    adj = region_adjacency(regions)
    colors = {r.label: _hex_to_oklab([r.color_hex])[0] for r in regions}
    parent = {r.label: r.label for r in regions}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)        # attach to lower label (deterministic)

    for r in regions:
        for n in sorted(adj[r.label]):
            if n > r.label and n in by_label:
                if float(np.linalg.norm(colors[r.label] - colors[n])) <= tol:
                    union(r.label, n)

    groups: dict[int, list[Region]] = {}
    for r in regions:
        groups.setdefault(find(r.label), []).append(r)
    return [g for _, g in sorted(groups.items(), key=lambda kv: min(m.label for m in kv[1]))]
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_gradient.py -k merge_components -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): colour-step agglomerative region merge

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: `_component_fill` — per-component fill decision

**Files:**
- Modify: `src/vectoreyes/gradient.py` (re-add `_PARAM_FALLBACK_TOL`; add `_field_spread`, `_component_fill`)
- Test: `tests/test_gradient.py`

**Interfaces:**
- Consumes (existing): `fit_gradient`, `_best_parametric`, `_fit_stretch`, `_stop_span`, `_MIN_STOP_SPAN`, `srgb_to_oklab`.
- Produces:
  - `_PARAM_FALLBACK_TOL = 0.07` (module constant)
  - `_field_spread(mask: np.ndarray, rgb_image: np.ndarray) -> float` — max OKLab distance of the masked original pixels from their mean (≈0 for a flat region).
  - `_component_fill(mask: np.ndarray, rgb_image: np.ndarray) -> dict | None` — a fill model dict (`kind` in `{"linear","radial","raster"}`) or `None` (the field is flat: caller renders a solid colour). Decision order: strict `fit_gradient` → flat-gate (`_field_spread < _MIN_STOP_SPAN` → None) → `_best_parametric` (None → None) → mean ΔE ≤ `_PARAM_FALLBACK_TOL` → that gradient → else `_fit_stretch`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_gradient.py` (`_linear_gradient_image` already exists in this file; `_2d_field` below is self-contained):

```python
def _2d_field(h, w):
    """A smooth field that no single linear/radial gradient fits under the param bound:
    horizontal hue ramp plus a contrasting corner."""
    yy, xx = np.mgrid[:h, :w]
    t = xx / (w - 1)
    img = np.empty((h, w, 3))
    for ch, (a, b) in enumerate(((30, 230), (60, 60), (220, 40))):
        img[:, :, ch] = a + t * (b - a)
    img[(xx >= w * 0.5) & (yy >= h * 0.5)] = (20, 230, 40)
    return img.round().astype(np.uint8)


def test_component_fill_strict_gradient_for_clean_ramp():
    from vectoreyes.gradient import _component_fill
    h, w = 60, 120
    img = _linear_gradient_image(h, w, (0, 30), (119, 30),
                                 [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    model = _component_fill(np.ones((h, w), bool), img)
    assert model is not None and model["kind"] in ("linear", "radial")


def test_component_fill_none_for_flat():
    from vectoreyes.gradient import _component_fill
    img = np.full((40, 40, 3), (50, 100, 150), np.uint8)
    assert _component_fill(np.ones((40, 40), bool), img) is None     # flat -> solid colour


def test_component_fill_raster_for_2d_field():
    from vectoreyes.gradient import _component_fill
    img = _2d_field(96, 96)
    model = _component_fill(np.ones((96, 96), bool), img)
    assert model is not None and model["kind"] == "raster"
```

> Fixture note: if `test_component_fill_raster_for_2d_field` lands `linear`/`radial` instead of `raster`, increase the corner contrast/size in `_2d_field`; if it lands `None`, decrease it. Tune the fixture, never the thresholds.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_gradient.py -k component_fill -v`
Expected: FAIL — `_component_fill` not defined.

- [ ] **Step 3: Re-add `_PARAM_FALLBACK_TOL`, add `_field_spread` and `_component_fill`**

In `src/vectoreyes/gradient.py`, add the constant near the other thresholds (after `_STRETCH_TARGET` / `MERGE_TOL`):

```python
_PARAM_FALLBACK_TOL = 0.07   # max mean per-pixel ΔE for a merged component to prefer an
                            # editable parametric gradient over a raster stretch-fill.
```

Add the functions just above `detect_gradients`:

```python
def _field_spread(mask: np.ndarray, rgb_image: np.ndarray) -> float:
    """Max OKLab distance of the original pixels under `mask` from their mean colour.
    A cheap flat-vs-varying gate (~0 for a flat region)."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    okl = srgb_to_oklab(rgb_image[ys, xs].astype(float) / 255.0)
    return float(np.linalg.norm(okl - okl.mean(axis=0), axis=1).max())


def _component_fill(mask: np.ndarray, rgb_image: np.ndarray) -> dict | None:
    """Pick a fill model for one component's footprint: strict parametric gradient, else a
    searched parametric gradient (mean ΔE <= _PARAM_FALLBACK_TOL), else a raster stretch-fill.
    None when the field is too flat to be anything but a solid colour (caller renders flat)."""
    strict = fit_gradient(mask, rgb_image)
    if strict is not None:
        return strict
    if _field_spread(mask, rgb_image) < _MIN_STOP_SPAN:
        return None                                  # flat -> solid colour
    bp = _best_parametric(mask, rgb_image)
    if bp is None:
        return None                                  # near-flat (span guard) -> solid colour
    model, mean_de, _median = bp
    if mean_de <= _PARAM_FALLBACK_TOL:
        return model                                 # editable gradient
    return _fit_stretch(mask, rgb_image)             # 2-D field -> raster
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_gradient.py -k component_fill -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): per-component fill decision (gradient/raster/flat)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Rework `detect_gradients` to merge → per-component fill; remove dead ramp-grouping

**Files:**
- Modify: `src/vectoreyes/gradient.py` (rewrite `detect_gradients`; remove now-dead `_ramp_groups`, `_trim_to_ramp`, `_is_ramp`, `_is_strict_ramp`, `_ramp_fit` if unreferenced)
- Modify: `tests/test_gradient.py` (replace the three `_ramp_groups` unit tests; update the dissolve test)
- Modify: `tests/test_acceptance_smooth_gradient.py` (drop `_ramp_groups`-based assertions; keep the output assertions)
- Test: full suite (parity)

**Interfaces:**
- Consumes: `merge_components` (Task 1), `_component_fill` (Task 2), `_union_mask`, `_expand_footprint`, `Region`.
- Produces: `detect_gradients(regions, rgb_image) -> tuple[list[tuple[Region, dict]], list[Region]]` — unchanged signature. An *eligible* component (≥`_MIN_BANDS` merged, or a dominant single blob) that fits becomes a `(footprint_region, model)` fill; every other region — ineligible groups, eligible-but-unfittable groups, unmerged singletons — stays in `remaining` as its original region(s). Reuses existing `_MIN_BANDS` / `_BLOB_DOMINANCE` constants.

- [ ] **Step 1: Update the existing tests to the new behavior (write them first, expect failure)**

In `tests/test_gradient.py`:

Delete `test_ramp_groups_groups_a_monotonic_ramp`, `test_ramp_groups_rejects_flat_and_too_few`, and `test_ramp_groups_rejects_nonramp_colors` (their behavior is now covered by the `merge_components` tests from Task 1).

Replace `test_detect_gradients_dissolves_unfittable_group_back_to_flats` with (the zig-zag colour layout now simply does not merge — adjacent bands are large colour steps — so each falls through as its own flat region; same end result, new mechanism):

```python
def test_detect_gradients_zigzag_bands_stay_flat():
    from vectoreyes.color import oklab_to_srgb, srgb_to_oklab
    from vectoreyes.gradient import detect_gradients
    from vectoreyes.types import Region
    # Bands whose colours zig-zag along a line in OKLab so every spatially-adjacent pair is
    # a LARGE colour step -> merge_components never joins them -> all stay flat in `remaining`.
    l0 = srgb_to_oklab(np.array([20, 50, 250])[None] / 255.0)[0]
    l1 = srgb_to_oklab(np.array([250, 20, 90])[None] / 255.0)[0]

    def hex_at(o):
        rgb = (np.clip(oklab_to_srgb((l0 + o * (l1 - l0))[None])[0], 0, 1) * 255).round().astype(int)
        return "#%02x%02x%02x" % tuple(rgb)

    spatial = [0.0, 1.0, 0.2, 0.8, 0.4, 0.6]
    h, band_w = 40, 18
    w = band_w * len(spatial)
    img = np.zeros((h, w, 3), np.uint8)
    regions = []
    for i, o in enumerate(spatial):
        hx = hex_at(o)
        m = np.zeros((h, w), bool)
        m[:, i * band_w:(i + 1) * band_w] = True
        img[m] = (int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16))
        regions.append(Region(label=i + 1, mask=m, color_hex=hx))
    fills, remaining = detect_gradients(regions, img)
    assert fills == []
    assert {r.label for r in remaining} == {1, 2, 3, 4, 5, 6}
```

(`test_detect_gradients_consumes_ramp_returns_remaining` is unchanged — the 4 small-step ramp bands merge and fit one linear gradient; the large-step green block stays flat. Leave it as-is; it is a parity check.)

In `tests/test_acceptance_smooth_gradient.py`: remove the two `assert _ramp_groups(regions) == []` lines (in `test_smooth_linear_rect_via_smooth_path` and `test_smooth_radial_disc_via_smooth_path`) and the now-unused `from vectoreyes.gradient import _ramp_groups` import. Keep every other assertion (the `svg.count("<linearGradient") == 1` / `<radialGradient>` and the ΔE bounds) — those are the real parity checks.

- [ ] **Step 2: Run to verify the suite fails as expected**

Run: `uv run pytest tests/test_gradient.py tests/test_acceptance_smooth_gradient.py -q`
Expected: FAIL — the deleted/edited tests reference removed names, and `test_detect_gradients_zigzag_bands_stay_flat` exercises behavior not yet implemented (the old `detect_gradients` still groups via `_ramp_groups`). This confirms the tests bind to the new behavior.

- [ ] **Step 3: Rewrite `detect_gradients`**

Replace the entire body of `detect_gradients` in `src/vectoreyes/gradient.py` with:

```python
def detect_gradients(
    regions: list[Region], rgb_image: np.ndarray
) -> tuple[list[tuple[Region, dict]], list[Region]]:
    """Merge spatially-adjacent regions into smooth-field components (colour-step merge),
    then choose a fill per *eligible* component. Returns (fills, remaining).

    Fill-eligibility gate (restores the two guards the colour-step merge alone drops, so
    adjacent distinct-but-similar flat shapes and mildly-noisy single flats are NOT
    over-fit): a group qualifies only if it is a genuine merged field (len(group) >=
    _MIN_BANDS) OR a single dominant blob (its area >= _BLOB_DOMINANCE * total foreground).
    An eligible component that fits a gradient/raster becomes a (footprint_region, model)
    fill (gradient footprints grow into model-matching background via _expand_footprint).
    Every other region — ineligible groups, eligible-but-unfittable (near-flat) groups,
    and unmerged singletons — stays in `remaining` as its original region(s)."""
    fills: list[tuple[Region, dict]] = []
    consumed: set[int] = set()
    shape = rgb_image.shape[:2]
    total_fg = float(sum(r.area for r in regions)) or 1.0
    for group in merge_components(regions):
        eligible = (len(group) >= _MIN_BANDS
                    or sum(r.area for r in group) >= _BLOB_DOMINANCE * total_fg)
        if not eligible:
            continue                                 # leave regions in `remaining` as-is
        mask = _union_mask(group, shape)
        model = _component_fill(mask, rgb_image)
        if model is None:
            continue                                 # not a field -> regions stay flat
        if model["kind"] in ("linear", "radial"):
            mask = _expand_footprint(model, mask, rgb_image)
        rep = max(group, key=lambda r: r.area)
        fills.append((Region(label=rep.label, mask=mask, color_hex=rep.color_hex), model))
        consumed.update(m.label for m in group)
    remaining = [r for r in regions if r.label not in consumed]
    return fills, remaining
```

`_MIN_BANDS` and `_BLOB_DOMINANCE` are existing module constants (no new thresholds). `Region.area` is the existing region-area attribute.

- [ ] **Step 4: Remove the now-dead ramp-grouping helpers**

Verify each is unreferenced after the rewrite, then delete it:

Run: `rg -n "_ramp_groups|_trim_to_ramp|_is_strict_ramp|_is_ramp|_ramp_fit" src/ tests/`
For each of `_ramp_groups`, `_trim_to_ramp`, `_is_strict_ramp`, `_is_ramp`, `_ramp_fit` that now appears ONLY at its own definition (no other references in `src/`), delete the function. Do NOT delete `_principal_axis` (used by `_fit_radial`/`_fit_linear`), `_dominant_blob_fraction`, or `_BLOB_DOMINANCE` (the latter two are imported by `score.py`). If any ramp helper is still referenced somewhere unexpected, leave it and note it in the report.

- [ ] **Step 5: Run the targeted tests, then the full suite for parity**

Run: `uv run pytest tests/test_gradient.py tests/test_acceptance_gradient.py tests/test_acceptance_smooth_gradient.py -q`
Expected: PASS — including the unchanged `test_detect_gradients_consumes_ramp_returns_remaining` (parity) and the smooth-gradient acceptance tests (one `<linearGradient>`/`<radialGradient>` preserved).

Run: `uv run pytest -q`
Expected: PASS — full suite, no regressions (paste the verbatim summary line). If a parity test fails because the unified merge cannot reproduce an existing gradient output, STOP and report — the additive fallback (keep `_ramp_groups` for collinear, merge the residual) is the contingency, and it needs controller sign-off.

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py tests/test_acceptance_smooth_gradient.py
git commit -m "feat(gradient): detect_gradients via colour-step merge + per-component fill

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Corpus validation + visual confirmation + `MERGE_TOL` tuning

**Files:**
- Modify only if a threshold must move: `src/vectoreyes/gradient.py` (`MERGE_TOL` / `_PARAM_FALLBACK_TOL`), with the rationale in its comment.
- Scratch only (untracked): render the corpus before/after.

**Interfaces:**
- Consumes: the corpus in `scratch/real-logos/*.png` (untracked brand assets) and the finished merge + per-component fill.

- [ ] **Step 1: Render the corpus and record per-mark fill structure**

For each source logo (apple_music, appstore, asana, burger_king, dropbox, firefox, gdrive, icloud, instagram, mastercard, microsoft, photoshop, pinterest, sketch, slack, telegram, vimeo, visa), run `idealize(img, options=Options())` and record per mark: counts of `<pattern>` (raster), `<linearGradient>`, `<radialGradient>`, and total element count. (Do not add scratch files to git.)

- [ ] **Step 2: Assert the anchor outcomes**

Confirm: firefox and instagram emit per-component **parametric gradient** fields (validated: firefox → linear + radial, instagram → radial + linear — NOT raster, so no blur) AND keep their sharp features (the white camera outline) as separate crisp vector elements; element count is small, not the old 29/37 shards; gdrive stays crisp facets (each facet its own component, flat or a small gradient, sharp edges, NO blur); icloud/telegram still emit their gradient; every flat mark (apple_music, dropbox, mastercard, microsoft, photoshop, pinterest, sketch, slack, vimeo, visa, appstore, asana, burger_king) gains NO spurious fill. (Raster remains the safety net in `_component_fill` for fields that don't fit a parametric gradient; it is fine if no corpus mark currently triggers it.)

- [ ] **Step 3: Visual spot-check**

Rasterize the idealized firefox, instagram, and gdrive SVGs (use `tests/_render.render_svg`) and compare against the sources. Confirm: instagram's white camera outline is a crisp vector line (not a glow); firefox's structure is preserved; gdrive's facet edges are sharp. Load the source with the SAME background convention as the renderer (composite RGBA onto white) so ΔE is meaningful — do not compare a black-flattened source against a white-composited render.

- [ ] **Step 4: Tune `MERGE_TOL` only if a mark regresses**

If a mark over-merges (a sharp feature merged into the field) lower `MERGE_TOL`; if a smooth field fragments (bands not merging) raise it — within the validated 0.13–0.25 window. Re-run Step 1 and `uv run pytest -q` after any change. Record the final value and which marks bound it in the commit message.

- [ ] **Step 5: Commit any tuning**

```bash
git add src/vectoreyes/gradient.py
git commit -m "fix(gradient): tune MERGE_TOL against the corpus

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

(If no tuning was needed, skip the commit and note the corpus result in the final review.)

---

---

### Task 5: Thin-band fillability gate (keep faceted logos crisp)

**Files:**
- Modify: `src/vectoreyes/gradient.py` (add `_THIN_BAND_TOL`; extract `_group_is_fillable`; use it in `detect_gradients`)
- Test: `tests/test_gradient.py`

**Context:** Corpus validation (Task 4) found the `>= _MIN_BANDS OR dominant` gate still smears faceted logos whose facets are similar-coloured (Sketch's top facets merge into a 4-band group and fit a radial at ΔE 0.037 — *better* than real gradients — so neither colour, band-count, nor fit-quality rejects it). The only signal that separates a finely-quantized gradient from coarse facets is band thinness: a gradient's bands are thin (firefox avg 0.03 of the mark), facets are chunky (Sketch avg 0.20). Validated across the corpus: this flips only Sketch (radial→flat) and App Store (near-dominant low-contrast → flat, its prior rendering); every real gradient and flat mark is unchanged.

**Interfaces:**
- Produces: `_THIN_BAND_TOL = 0.10` (constant); `_group_is_fillable(group: list[Region], total_fg: float) -> bool`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gradient.py`:

```python
def _regions_with_areas(areas):
    from vectoreyes.types import Region
    out = []
    for i, a in enumerate(areas):
        m = np.zeros((1, 1000), bool); m[0, :a] = True       # a True pixels -> Region.area == a
        out.append(Region(label=i + 1, mask=m, color_hex="#000000"))
    return out


def test_group_is_fillable_dominant_thin_chunky():
    from vectoreyes.gradient import _group_is_fillable
    # dominant single blob (90% of fg) -> fillable
    assert _group_is_fillable(_regions_with_areas([900]), 1000.0) is True
    # 10 thin bands, 80% of fg, each 8% -> avg 0.008 < 0.10 -> finely-quantized -> fillable
    assert _group_is_fillable(_regions_with_areas([80] * 10), 1000.0) is True
    # 4 chunky facets, 80% of fg, each 20% -> avg 0.20 >= 0.10 and not dominant -> NOT fillable
    assert _group_is_fillable(_regions_with_areas([200] * 4), 1000.0) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_gradient.py -k group_is_fillable -v`
Expected: FAIL — `_group_is_fillable` not defined.

- [ ] **Step 3: Add `_THIN_BAND_TOL` and `_group_is_fillable`; use it in `detect_gradients`**

Add the constant near the other thresholds:

```python
_THIN_BAND_TOL = 0.10   # max mean band-area fraction (group_area / total_fg / band count) for a
                       # >= _MIN_BANDS group to count as a finely-quantized continuous tone rather
                       # than a few chunky similar-coloured facets. Keeps faceted logos (Sketch)
                       # crisp instead of smearing them into a gradient that ΔE cannot tell apart.
```

Add the helper just above `detect_gradients`:

```python
def _group_is_fillable(group: list[Region], total_fg: float) -> bool:
    """A merged group is eligible for a gradient/raster fill if it is a dominant blob
    (covers >= _BLOB_DOMINANCE of the foreground) OR a finely-quantized continuous tone
    (>= _MIN_BANDS bands, each thin: mean band-area fraction < _THIN_BAND_TOL). The
    thinness test keeps faceted logos (few chunky similar-coloured facets) crisp instead
    of smearing them into a gradient that ΔE cannot distinguish from a real one."""
    af = sum(r.area for r in group) / total_fg
    return (af >= _BLOB_DOMINANCE
            or (len(group) >= _MIN_BANDS and af / len(group) < _THIN_BAND_TOL))
```

In `detect_gradients`, replace the inline `eligible = (...)` expression and its `if not eligible:` with:

```python
        if not _group_is_fillable(group, total_fg):
            continue                                 # leave regions in `remaining` as-is
```

(Keep `total_fg = float(sum(r.area for r in regions)) or 1.0` as the per-call foreground total.)

- [ ] **Step 4: Run the test, then the full suite for parity**

Run: `uv run pytest tests/test_gradient.py -k group_is_fillable -v`
Expected: PASS.

Run: `uv run pytest -q`
Expected: PASS — full suite, no regressions (paste the verbatim summary line).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/gradient.py tests/test_gradient.py
git commit -m "feat(gradient): thin-band fillability gate keeps faceted logos crisp

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final Review

After all tasks: dispatch a whole-branch code review (Opus) covering the full branch (the retained Task 1/2/3/5 primitives from the prior plan + this merge rework + the revert), then use superpowers:finishing-a-development-branch to open the PR. PR body ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

# Variant matrix (`--variants`) — design

**Goal:** A CLI mode that idealizes one mark across an `epsilon × max_error`
geometry matrix and emits the variants as SVGs + a JSON manifest + an annotated
contact-sheet PNG, so a user or agent can compare whole-mark looks and pick one.

**Architecture:** A new self-contained `src/vectoreyes/variants.py` drives the
matrix by calling the existing `idealize()` once per cell. A small, backward-
compatible change to the pipeline surfaces a per-variant **strategy report** (the
fitter strategies the scorer chose), which the manifest and contact sheet
annotate. A new `--variants` CLI flag selects this mode.

**Tech stack:** existing numpy/Pillow pipeline; `resvg-py` (optional `scoring`
extra) for the contact sheet only.

---

## Scope

In scope: whole-mark variants over the two visual geometry tolerances
(`epsilon`, `max_error`); SVG set; manifest; annotated contact sheet; strategy
report plumbing.

Out of scope (explicitly):
- **Per-element strategy selection** — already covered by the slice-4b
  `Options.selection` API; the matrix is whole-mark.
- **A fill / solid-vs-gradient axis** — gradients have no on/off toggle today;
  not adding one here.
- **`flatten` as an axis** — it changes SVG *structure* but renders pixel-
  identical, so it is useless for a visual matrix.
- **The skill update** (one-shot flow, variants usage, design-system
  incorporation) — a separate follow-on PR after this lands.

## Components

### 1. Strategy report (pipeline plumbing)

Each `Candidate` already carries `.strategy` (selector.py:176). Surface an
aggregate without changing existing callers:

- New frozen dataclass in `pipeline.py`:
  ```python
  @dataclass(frozen=True)
  class IdealizeReport:
      strategies: Mapping[str, int]   # e.g. {"primitive": 3, "sym_polygon": 2, "path": 1}
      gradients: int                  # count of gradient fills emitted
      elements: int                   # total emitted geometry elements
  ```
- `build_candidates(...)` already returns the `Candidate` list; `_render_body`
  collects `c.strategy` for region candidates (skip occlusion/lens/gradient
  pseudo-strategies — those are counted separately: `gradients` from
  `fill == "gradient"`).
- `idealize(image, *, options=None, report=False)`:
  - `report=False` (default) → returns `str` (unchanged; all current callers
    keep working).
  - `report=True` → returns `tuple[str, IdealizeReport]`.
- Aggregation sums strategy counts across all components (multi-component marks).

### 2. `variants.py`

```python
DEFAULT_EPSILONS = (0.5, 1.5, 3.0)
DEFAULT_MAX_ERRORS = (0.5, 1.0, 2.5)

@dataclass(frozen=True)
class Variant:
    epsilon: float
    max_error: float
    svg: str
    report: IdealizeReport

def generate_variants(image, *, epsilons=DEFAULT_EPSILONS,
                      max_errors=DEFAULT_MAX_ERRORS, base=None) -> list[Variant]:
    """idealize() the image once per (epsilon, max_error) cell, in row-major
    order (epsilon outer, max_error inner). `base` is an optional Options whose
    other fields (max_colors, no_symmetry, …) are held constant across the grid;
    epsilon/max_error are overridden per cell."""
```

- Reuses `idealize(..., report=True)` per cell — no re-implementation of
  segmentation/symmetry/components (DRY).
- `base` lets the caller hold non-geometry knobs constant (e.g. a fixed palette)
  while sweeping geometry.

### 3. Output: `write_variant_set(variants, out_dir, *, source)`

- `out_dir/variant-e{epsilon}-m{max_error}.svg` per cell (epsilon/max_error
  formatted with `_fmt`, `.` → `_` so filenames are clean, e.g.
  `variant-e0_5-m1.svg`).
- `out_dir/manifest.json`:
  ```json
  {
    "source": "logo.png",
    "axes": {"epsilon": [0.5, 1.5, 3.0], "max_error": [0.5, 1.0, 2.5]},
    "variants": [
      {
        "epsilon": 0.5, "max_error": 0.5,
        "file": "variant-e0_5-m0_5.svg",
        "svg_bytes": 4210,
        "strategies": {"primitive": 3, "sym_polygon": 2, "path": 1},
        "gradients": 0, "elements": 6
      }
    ]
  }
  ```

### 4. Contact sheet: `compose_contact_sheet(variants, *, epsilons, max_errors) -> bytes | None`

- Renders each variant SVG to a tile via the shared `score` renderer; lays them
  out in an `len(epsilons) × len(max_errors)` grid.
- Row labels: `ε=<value>`; column labels: `max_error=<value>`.
- Each tile is captioned with its strategy histogram (compact, e.g.
  `prim×3 sym_poly×2 path×1`).
- Returns `None` (and the caller warns once) when the renderer is unavailable —
  see Renderer fallback.
- Composition logic (tile crop/paste/grid) is a clean rewrite modeled on
  `scratch/show_options.py`, living in `variants.py` (no dependency on scratch).

### 5. CLI (`cli.py`)

- Add `--variants` (store_true). When set, the single-SVG path is bypassed:
  - `--out-dir DIR` (default `./<input-stem>-variants/`).
  - `--epsilons "0.5,1.5,3"` and `--max-errors "0.5,1,2.5"` (comma lists; default
    to the module defaults). Parsed to float tuples; a bad value is a usage error.
  - Non-geometry flags (`--colors`, `--no-symmetry`) form the `base` Options held
    constant across the grid; `--epsilon`/`--max-error` are ignored in variants
    mode (the matrix supplies them) — warn if both given.
  - Writes the variant set + manifest; renders + writes
    `contact-sheet.png` when the renderer is available.
  - Prints the out-dir and a one-line summary (N variants, sheet written? y/n).

## Renderer fallback (DRY)

The contact sheet is the only renderer-dependent part. Reuse the existing
`score` cascade: attempt render; on `SvgRendererUnavailable`, skip the PNG, warn
once (`install vectoreyes[scoring]` to enable the contact sheet), and still emit
the SVGs + manifest. So `--variants` works in the minimal install, minus the
sheet.

## Data flow

```
input raster
  └─ generate_variants  ── per cell ─▶ idealize(report=True) ─▶ (svg, IdealizeReport)
        └─ list[Variant]
              ├─ write_variant_set ─▶ variant-*.svg + manifest.json
              └─ compose_contact_sheet ─▶ contact-sheet.png  (or None → warn)
```

## Error handling

- Empty / unparseable `--epsilons` / `--max-errors` → argparse usage error.
- `idealize` raising for a pathological cell must not abort the whole grid: catch
  per cell, record the cell as failed in the manifest (`"error": "<message>"`,
  no `file`), and continue. The contact sheet shows a placeholder tile for a
  failed cell.
- No regions found for a cell → that variant's SVG is the empty-doc `idealize`
  already returns; report has empty strategies. Not an error.

## Testing

- **Strategy report**: a synthetic 2-region mark → `idealize(report=True)` returns
  counts matching the emitted geometry (e.g. a circle region → `primitive`); the
  default `report=False` still returns a bare `str` (back-compat).
- **generate_variants**: a 2×2 grid yields 4 variants in row-major order with the
  expected epsilon/max_error per cell; each `svg` starts with `<svg `.
- **Distinctness**: across a wide grid, at least two variants differ in their SVG
  (params actually take effect).
- **Manifest**: `write_variant_set` writes one SVG per cell + a manifest whose
  `variants[i]` matches each `Variant` (epsilon, max_error, file, strategies).
- **Contact sheet**: with the renderer available, `compose_contact_sheet` returns
  PNG bytes of the expected grid dimensions; monkeypatching the renderer
  unavailable returns `None` (and the CLI path warns once, writes no PNG).
- **CLI**: `vectoreyes logo.png --variants --out-dir tmp` writes N SVGs +
  `manifest.json` (+ `contact-sheet.png` when renderer present); `--epsilons`
  with a bad value exits non-zero.

## Files

- Create: `src/vectoreyes/variants.py`
- Modify: `src/vectoreyes/pipeline.py` (add `IdealizeReport`, `report=` on
  `idealize`, strategy aggregation in `_render_body`)
- Modify: `src/vectoreyes/cli.py` (`--variants` + axis flags + out-dir)
- Test: `tests/test_variants.py`, plus a `report=` case in the pipeline tests

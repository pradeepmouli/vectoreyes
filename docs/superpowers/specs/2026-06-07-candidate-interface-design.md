# Candidate / Fill Interface — Design Spec

**Status:** Approved (design). Slice 2 of the candidate-pipeline roadmap
(`docs/architecture/2026-06-07-candidate-pipeline-roadmap.md`).
**Branch:** `feat/candidate-interface` (off `master`; independent of the palette PR #14 — touches
`pipeline.py`/`fit.py`/`gradient.py`/`emit.py` + a new `candidate.py`, not `color.py`).
**Date:** 2026-06-07

## Goal

Decouple **shape/path detection (geometry)** from **colour application (fill)** by introducing a
typed `Candidate (geometry, fill)` interface and a typed `Fill` sum type, then collapsing the three
parallel emit loops in `_render_body` into a single candidate-rendering loop. This is a **pure
structural refactor: byte-identical SVG output** for every existing fixture. It is the prerequisite
seam for the scorer (slice 3) and the selector harness (slice 4).

## Motivation

Today `_render_body` entangles geometry and fill, and emits in three places:

- **Geometry** is produced by `_fit_region(region, opt, axis, corner_radius) -> Shape` (shared by
  flat regions and gradient footprints).
- **Fill** is represented three different ways, decided at three different sites: flat =
  `region.color_hex` (a string on `Region`); gradient = `detect_gradients` → `(footprint, model)` +
  a `<linearGradient>`/`<radialGradient>` def + `url(#gN)`; occlusion =
  `ScenePrimitive.color_hex` / lens `params["color_hex"]`.
- **Emit** is three parallel loops (occlusion prims/lenses by z; flat/pair/loner regions by area;
  gradient footprints last), each independently re-deriving the fill source, the flatten-vs-`<g>`
  branch, the id-minting, and the mirror/`<use>` branch.

That triplicated emit logic is the DRY pain and the geometry↔fill entanglement. There is no single
"what fill does this element get" abstraction, and you cannot ask "of several ways to render this
element, which is best?" — there is no candidate object to compare.

## Architecture

New module `src/vectoreyes/candidate.py` holding IO-free data types (matching the `types.py`
convention that pipeline-stage data is IO-free; SVG emission stays in `emit.py`/`pipeline.py`):

```python
@dataclass
class FlatFill:
    hex: str

@dataclass
class LinearGradientFill:
    geometry: dict          # {x1, y1, x2, y2}
    stops: list

@dataclass
class RadialGradientFill:
    geometry: dict          # {cx, cy, r}
    stops: list

Fill = FlatFill | LinearGradientFill | RadialGradientFill

@dataclass
class Candidate:
    geometry: Shape         # existing fit.Shape (kind + params)
    fill: Fill
    source: str             # provenance: "occlusion" | "lens" | "region" | "gradient"
    mirror: Axis | None = None   # if set, emit the element AND its <use>/reflected twin
```

- **`Fill` is pure data** (approach A): the emit loop owns the gradient `<def>` side-effect, the
  `g{len(defs)}` id-minting, and the bake of gradient geometry. This keeps `Fill` a value and
  centralises emission/id-minting in one place, which is what makes byte-identical output natural.
- **`source`** records which strategy produced the candidate. It has no behavioural effect in slice
  2 except one preserved quirk (lenses emit as a plain path with no id even in non-flatten mode — see
  Data flow). It is the cheap enabler for slices 3–4, where candidates must be *identifiable* so an
  agent or user can select among them (see Forward context).
- **`mirror`** folds the pair/`<use>` logic into the candidate, replacing a dedicated emit branch.
- **No `z` field.** Paint order is carried by **list order** — `build_candidates` returns the list
  already in the exact current paint order; the emit loop iterates in order without re-sorting. List
  order as the contract is simpler and more robustly byte-identical than a z-key plus sort.
- **No `cost` field** in slice 2. `cost` is added in slice 3 together with its first reader (the
  scorer). YAGNI: no field nothing reads.

## Components & data flow

### `build_candidates(...) -> list[Candidate]`

A new function (in `pipeline.py`, or `candidate.py` if it stays IO-free) that runs **after** symmetry
detection, occlusion reconstruction, and gradient detection — exactly the state `_render_body` has
today before its emit loops — and produces the ordered candidate list by concatenating three groups
in the current paint order:

1. **Occlusion** — iterate `reconstructed` sorted by z (as today). A `ScenePrimitive` →
   `Candidate(Shape(kind, params), FlatFill(color_hex), source="occlusion")`. A lens `Shape("path",
   {d, color_hex, z})` → `Candidate(Shape("path", {"d": d}), FlatFill(color_hex), source="lens")`.
2. **Regions** — build the `drawn` list (straddlers with axis; pairs canonical with `mirror=axis`;
   loners with no axis), sorted by area descending (as today). Each →
   `Candidate(_fit_region(region, opt, fit_axis, corner_radius), FlatFill(region.color_hex),
   source="region", mirror=axis if is_pair else None)`. Candidates whose `_fit_region` returns
   `None` are dropped (preserves the current `if shape is None: continue`).
3. **Gradients** — for each `(footprint, model)` in `gradient_fills` (detect order):
   `Candidate(_fit_region(footprint, opt, None, corner_radius), Linear/RadialGradientFill(
   model["geometry"], model["stops"]), source="gradient")`. `None`-geometry dropped.

### The single emit loop

Replaces the three loops. Iterates `build_candidates(...)` in order, maintaining the `eid` counter
and the `defs` list. Per candidate:

- **Resolve fill** → fill attribute string:
  - `FlatFill` → `#hex`.
  - `Linear/RadialGradientFill` → bake geometry via `_bake_gradient_geometry` when `bake` is set;
    mint `gid = f"g{len(defs)}"` **before** appending (exactly as today: the index is the def count
    prior to this gradient); then append `linear_gradient_def`/`radial_gradient_def` to `defs`; use
    `url(#{gid})`.
- **Render geometry**, preserving today's exact branches:
  - `opt.flatten` → baked plain path: `emit(shape_to_path_d(geom), fill, geom.params.get("fill_rule"))`;
    if `mirror` set → also `emit(reflect_path_d(d, axis.x), fill, rule)`.
  - non-flatten, **identified** sources (`occlusion`, `region`, `gradient`) → `shape_to_svg(geom,
    fill, f"s{eid}")`; if `mirror` set → also `mirror_use(f"s{eid}", axis)`.
  - non-flatten, **lens** source → plain path `emit(geom.params["d"], fill)` with **no id** (the one
    preserved quirk; derive "plain-path-no-id" from `source == "lens"`).
- **`eid += 1`** for every candidate (matches today: `eid` increments in every branch, including the
  invisible flatten case and the id-less lens case).

`_render_body` shrinks to: measure silhouette/axis/corner_radius → `reconstruct_scene` →
`detect_gradients` (when `rgb`) → classify regions → `cands = build_candidates(...)` → single emit
loop → `return body, defs`.

## Error / edge handling

No new failure modes — this is restructuring. `_fit_region` returning `None` still skips the element
(handled in `build_candidates`). Empty regions / no candidates → empty body, as today. Determinism is
unchanged: the candidate list is built in the same deterministic order the three loops use today.

## Testing

**Byte-identical golden harness (the gate):**

1. *Before* refactoring, capture the current `idealize` SVG output for a representative set of
   committed fixtures — covering each source path and both modes: occlusion (annulus/polygon
   fixtures), symmetry pairs + straddlers (daikonic), gradients (gradient + smooth-gradient
   acceptance fixtures), flatten and non-flatten, and the rectified path. Store as golden strings
   (a new `tests/test_candidate_byte_identical.py`, goldens captured from `master` HEAD before any
   refactor commit).
2. A test asserts the refactored `idealize` output **`==`** the golden, per fixture/mode. Any
   difference fails — this is the proof of "no behaviour change".
3. The entire existing suite (158 tests) stays green.

The goldens are generated from the pre-refactor code so they encode current behaviour exactly; they
are committed so the equality assertion is reproducible. Because the bar is byte-identical, the
golden set does not need ΔE/SSIM tolerances — exact string match.

**Pressure valve:** byte-identical is the *default*. If faithfully reproducing a specific legacy
quirk (e.g. the lens-no-id case) would force genuinely worse code, that *single* case may be relaxed
to a render-ΔE≈0 / SSIM≈1 assertion instead of exact match — but it must be called out explicitly in
the test and the commit, with a one-line rationale, rather than loosening the whole bar. Default to
exact match; relax only with a named reason.

## Forward context (slices 3–4) — captured here so it isn't lost

Per design discussion, candidate **selection** in later slices is **not purely automated**. It must
support **agent/user-chosen options in addition to automated scoring**, at two points:

- **Pre-execution:** choose which algorithm/strategy candidates to *generate or run* for an element
  (e.g. "only try primitive-snap and smooth-path here").
- **Post-evaluation:** *override* the auto-scored winner after seeing the scored candidates.

The slice-2 implications, already in this design: candidates carry a `source` label so they are
identifiable for presentation/selection, and selection is architecturally a *separate stage* from
generation (build → [score] → [select] → emit), not hardwired into generation. The scorer (slice 3)
and selector (slice 4) specs will make manual selection a first-class mode alongside `cost`-based
ranking. **The roadmap doc will be updated to record this once it is on `master` (it currently lives
on PR #14's branch); tracked as a follow-up to avoid a cross-branch edit of that file here.**

## Non-goals

- Any output change — byte-identical is the bar.
- The scorer, `cost`, multi-candidate generation, or selection logic (slices 3–4).
- Folding occlusion into a *different* model than today — occlusion candidates are built from the
  existing `ScenePrimitive`/lens data unchanged; only their emission is unified.
- Touching `color.py` / the palette work (PR #14).
- Changing `_fit_region`, `detect_gradients`, `reconstruct_scene`, or the gradient def helpers — they
  are reused as-is; only their call sites are reorganised behind the candidate seam.

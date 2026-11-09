# Manual Candidate Selection (Roadmap Slice 4b) — Design

**Status:** approved design, pre-implementation
**Date:** 2026-06-08
**Builds on:** slice 4a scored geometry auto-selection (`selector.py`, `score.py`), merged in #20.

## Goal

Let an agent or user steer geometry selection *in addition to* the automated scorer,
at two points per the roadmap's layer 4:

1. **Pre-execution restriction** — restrict which strategy candidates are kept for an
   element before scoring ("only try primitive-snap and smooth-path here").
2. **Post-evaluation override** — override the auto-scored winner after seeing the
   ranked candidates ("for this element, use the symmetric path, not the polygon").

Selection is a **separate stage** from generation, declarative, and addressed
**per element by the stable `sN` id** the emit layer already stamps. When no policy
is supplied, output is byte-identical to 4a (parity gate green by construction).

## Non-goals (YAGNI)

- Denylist restriction (allowlist covers the roadmap example; a global "only these"
  uses `default`).
- A CLI surface (programmatic `Options.selection` is the agent's consumer; CLI later).
- Cross-element / callback selection logic (Approach C, rejected — declarative
  allow/force covers stated needs).
- Spatial/color element addressing (Approach: id-based, chosen).
- Manual *fill* selection (this slice is geometry; fill candidates are a later slice).

## Architecture

Approach A: a declarative `SelectionPolicy` carried on `Options`, threaded
`idealize → _render_body → build_candidates → select_geometry`. Generation stays
pure; selection filters/forces as separate stages. Each geometry candidate carries
a fitter-level **strategy** provenance label so both stages can name what they act on.

### New module: `src/vectoreyes/selection.py`

User-facing config + the strategy vocabulary. `selector.py` imports it (no cycle:
`selection.py` imports nothing from the pipeline).

```python
# Strategy provenance labels — one per fitter in generate_geometry_candidates.
PRIMITIVE   = "primitive"        # recognize_primitive -> circle/rect/ellipse
TRAPEZOID   = "trapezoid"        # rounded_trapezoid_fit
SYM_POLYGON = "sym_polygon"      # symmetric_polygon_fit
CAP         = "cap"              # half_ellipse_cap_fit
SYMMETRIC   = "symmetric"        # symmetric_fit
POLYGON     = "polygon"          # recognize_polygon
PATH        = "path"             # fit_path
HOLED_SYM   = "holed_symmetric"  # multi-contour mirrored halves (even-odd)
HOLED_PATH  = "holed_path"       # multi-contour per-contour fit (even-odd)

KNOWN_STRATEGIES = frozenset({PRIMITIVE, TRAPEZOID, SYM_POLYGON, CAP, SYMMETRIC,
                              POLYGON, PATH, HOLED_SYM, HOLED_PATH})

@dataclass(frozen=True)
class ElementSelection:
    allow: frozenset[str] | None = None   # pre-execution allowlist; None = all strategies
    force: str | None = None              # post-evaluation forced strategy

@dataclass(frozen=True)
class SelectionPolicy:
    by_id: Mapping[str, ElementSelection] = field(default_factory=dict)
    default: ElementSelection | None = None  # applied to elements with no by_id entry

    def for_id(self, eid: str) -> ElementSelection | None:
        return self.by_id.get(eid, self.default)
```

Validation: a helper `validate_strategies(sel: ElementSelection)` raises `ValueError`
if any label in `allow` or `force` is not in `KNOWN_STRATEGIES` (catches typos
loudly). Called once at the top of `select_geometry` whenever `element is not None`,
before the stages — so an unknown `force` label fails even when `allow` is None.

### `GeomCandidate` (in `selector.py`)

```python
@dataclass(frozen=True)
class GeomCandidate:
    strategy: str
    shape: Shape
```

`generate_geometry_candidates` returns `list[GeomCandidate]` in the same
cascade-priority order as 4a (`candidates[0].shape` == old `_fit_region` pick). Each
fitter's result is wrapped with its label where it is appended; the straddler
symmetry-gating is unchanged (it decides which fitters *run*; labeling is orthogonal).

### `Candidate.strategy` (in `candidate.py`)

One backward-compatible field so the label survives through scoring for the
post-eval force:

```python
strategy: str | None = None   # fitter provenance; None for occlusion/lens/gradient
```

Distinct from the existing `source` (element *category*: region/gradient/occlusion/lens).
This is the finer-grained realization of the roadmap's "candidate source provenance label."

### `Options.selection`

```python
selection: SelectionPolicy | None = None
```

## Data flow

`select_geometry(region, opt, axis, corner_radius, source_rgb, element=None)`
becomes a 3-stage pipeline; stages 2–3 are no-ops when `element is None`:

```
0. if element is not None: validate_strategies(element)  # ValueError on unknown label
                                                         # (covers allow AND force)

1. cands = generate_geometry_candidates(...)            # list[GeomCandidate], 4a order
   if not cands: return None

2. PRE-RESTRICTION (element.allow is not None):
       kept = [gc for gc in cands if gc.strategy in element.allow]
       if not kept:
           warnings.warn(f"selection {eid}: allow={set(element.allow)} removed all "
                         f"candidates; ignoring restriction", UserWarning)
           kept = cands
       cands = kept

3a. no source_rgb (no scoring):
       if element and element.force:
           hit = next((gc for gc in cands if gc.strategy == element.force), None)
           if hit is None:
               warnings.warn(f"selection {eid}: force='{element.force}' not among "
                             f"{[gc.strategy for gc in cands]}; using '{cands[0].strategy}'",
                             UserWarning)
               hit = cands[0]
           return hit.shape
       return cands[0].shape                            # 4a fallback

3b. score:
       wrapped = [Candidate(gc.shape, FlatFill(region.color_hex), "region",
                            strategy=gc.strategy) for gc in cands]
       ranked = rank_candidates(wrapped, source_rgb, region, fidelity_tol=tol, bbox=bbox)
       if element and element.force:
           hit = next((c for c, _ in ranked if c.strategy == element.force), None)
           if hit is None:
               warnings.warn(f"selection {eid}: force='{element.force}' not among "
                             f"{[c.strategy for c,_ in ranked]}; using auto winner "
                             f"'{ranked[0][0].strategy}'", UserWarning)
               hit = ranked[0][0]
           return hit.geometry
       return ranked[0][0].geometry                     # 4a winner
```

### Threading & addressing

`Options.selection` flows `idealize → _render_body → build_candidates`. Inside
`build_candidates`, each region/gradient element's eventual id is `f"s{len(cands)}"`
computed immediately before its `select_geometry` call — the id↔paint-order-index 1:1
relationship (verified: mirror twins reuse the same id via `<use>`, so they don't
perturb the count; occlusion/lens candidates are appended first, so region ids start
after them, exactly matching the emitted `sN`). That id keys `policy.for_id(eid)`;
the resulting `ElementSelection | None` is passed to `select_geometry`.

When `opt.selection is None`, `build_candidates` passes `element=None` to every call →
stages 2–3 skipped → `ranked[0]` wins → identical to 4a.

## Error handling

Both "manual choice couldn't be honored" cases are **non-fatal**, surfaced via
`warnings.warn(..., UserWarning)` (capturable via `pytest.warns`, visible on stderr,
non-breaking to the `idealize -> str` signature):

| Case | Behavior |
|------|----------|
| `allow` filters out every candidate | Warn (id + allow-set), ignore restriction, score full set |
| `force` strategy absent from candidates | Warn (id + force + available + auto winner), use auto winner |

**Hard error** (not a warning): an unknown strategy label in `allow` or `force`
raises `ValueError` at validation — a typo must fail loudly, not warn per element.
Validation is *lazy* — it runs per element when that element's policy is first
consulted in `select_geometry`. A typo in a `by_id` entry whose element is never
emitted (or in `default` when nothing is scored) therefore won't raise. This is the
deliberate per-element-on-use contract; it is not a global up-front check.

Note also that only fitted elements (regions, gradient footprints) are
policy-addressable. Reconstructed-occlusion and lens primitives advance the same `sN`
index counter (occlusion gets a visible `id="sN"`; lens emits a plain path with no
`id`), but neither is produced by the strategy selector, so a policy keyed to one of
those indices is silently ignored (out of scope for this slice).

## Parity gate

`selection is None` ⟹ `for_id` never consulted ⟹ stages 2–3 skipped ⟹ `ranked[0]`
wins ⟹ byte-identical to 4a. The full acceptance suite + byte-identical golden
harness stay green with **no re-capture**. Output changes only when a policy is
supplied *and* a choice is actually exercised.

## Testing

### Unit (`tests/test_selection.py`)
- Allowlist keeps only allowed strategies; winner drawn from them.
- Allowlist empties the set → `pytest.warns(UserWarning)` + auto winner (no crash).
- Force a present strategy not equal to `ranked[0]` → that strategy's shape wins
  (fixture where auto picks `primitive`, force `path`).
- Force an absent strategy → `pytest.warns` + auto winner returned.
- Unknown label in `allow`/`force` → `pytest.raises(ValueError)`.
- `GeomCandidate` ordering: `candidates[0].shape` == 4a cascade pick (re-asserted
  against the new return type).

### Integration (`tests/test_pipeline.py`)
- **Parity:** `idealize(img, options=Options(selection=None))` == `idealize(img)` —
  byte-identical no-op proof.
- **End-to-end override:** a disk auto-fits `circle`; `selection=SelectionPolicy(
  by_id={"s0": ElementSelection(force="path")})` emits `<path>` not `<circle>` —
  proves `sN` addressing reaches emit.
- **Pre-restriction e2e:** restrict an element to `allow={"path"}` and verify the
  emitted geometry (choose an element where the symmetry gate permits the path
  candidate).

### Regression
- 4a selector tests updated for the `GeomCandidate` return type (`c.shape.kind`,
  plus new `c.strategy` assertions).
- Byte-identical golden harness re-run unchanged — no re-capture.

## Files

- **Create:** `src/vectoreyes/selection.py`, `tests/test_selection.py`
- **Modify:** `src/vectoreyes/selector.py` (GeomCandidate, labeled generation,
  3-stage `select_geometry`), `src/vectoreyes/candidate.py` (`strategy` field),
  `src/vectoreyes/pipeline.py` (`Options.selection`, thread + `sN` addressing in
  `build_candidates`), `tests/test_selector.py` (return-type update),
  `tests/test_pipeline.py` (parity + override + restriction).

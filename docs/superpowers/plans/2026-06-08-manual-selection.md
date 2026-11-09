# Manual Candidate Selection (Slice 4b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an agent/user steer geometry selection per element (by stable `sN` id) — restrict which strategies are considered (pre-execution) and override the scored winner (post-evaluation) — while producing byte-identical 4a output when no policy is supplied.

**Architecture:** A declarative `SelectionPolicy` on `Options`, read inside `build_candidates` and passed per element to a 3-stage `select_geometry` (validate → allow-filter → score → force). Geometry candidates carry a fitter-level `strategy` provenance label so both stages can name what they act on. Generation stays pure; selection is a separate stage. Parity is by construction: `selection is None` skips every new stage.

**Tech Stack:** Python 3, numpy, dataclasses, `warnings`, pytest. Tests run with `.venv/bin/python -m pytest`.

**Branch:** `feat/manual-selection` off `master`.

---

## Pre-flight

- [ ] **Create the branch**

```bash
cd /Users/pmouli/GitHub.nosync/active/py/vectoreyes
git checkout master && git pull --ff-only
git checkout -b feat/manual-selection
.venv/bin/python -m pytest -q   # baseline: expect "197 passed"
```

## File Structure

- **Create** `src/vectoreyes/selection.py` — user-facing config: strategy vocabulary constants, `ElementSelection`, `SelectionPolicy`, `validate_strategies`. Imports nothing from the pipeline (no cycle).
- **Create** `tests/test_selection.py` — unit tests for the policy + the 3-stage `select_geometry` behavior.
- **Modify** `src/vectoreyes/candidate.py` — add `Candidate.strategy: str | None = None`.
- **Modify** `src/vectoreyes/selector.py` — add `GeomCandidate`, label generation, rewrite `select_geometry` to the 3-stage pipeline.
- **Modify** `src/vectoreyes/pipeline.py` — add `Options.selection`, thread `sN` addressing in `build_candidates`.
- **Modify** `tests/test_selector.py` — update for the `GeomCandidate` return type.
- **Modify** `tests/test_pipeline.py` — parity + override + restriction integration tests.

---

### Task 1: `selection.py` — vocabulary, policy types, validation

**Files:**
- Create: `src/vectoreyes/selection.py`
- Test: `tests/test_selection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_selection.py`:

```python
import pytest

from vectoreyes.selection import (
    ElementSelection, SelectionPolicy, validate_strategies,
    KNOWN_STRATEGIES, PRIMITIVE, PATH, SYMMETRIC,
)


def test_known_strategies_has_all_nine_labels():
    assert KNOWN_STRATEGIES == {
        "primitive", "trapezoid", "sym_polygon", "cap", "symmetric",
        "polygon", "path", "holed_symmetric", "holed_path",
    }


def test_for_id_returns_specific_then_default_then_none():
    sel = ElementSelection(force=PATH)
    deflt = ElementSelection(allow=frozenset({PRIMITIVE}))
    policy = SelectionPolicy(by_id={"s3": sel}, default=deflt)
    assert policy.for_id("s3") is sel        # specific wins
    assert policy.for_id("s9") is deflt       # falls back to default
    assert SelectionPolicy().for_id("s0") is None  # no entry, no default


def test_validate_accepts_known_labels():
    validate_strategies(ElementSelection(allow=frozenset({PRIMITIVE, PATH}), force=SYMMETRIC))


def test_validate_rejects_unknown_allow_label():
    with pytest.raises(ValueError, match="symetric"):
        validate_strategies(ElementSelection(allow=frozenset({"symetric"})))


def test_validate_rejects_unknown_force_label_even_when_allow_none():
    with pytest.raises(ValueError, match="blob"):
        validate_strategies(ElementSelection(force="blob"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_selection.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'vectoreyes.selection'`

- [ ] **Step 3: Write the implementation**

Create `src/vectoreyes/selection.py`:

```python
"""User-facing manual-selection policy (slice 4b). An agent/user restricts which
geometry strategies are considered for an element (pre-execution) and/or overrides
the auto-scored winner (post-evaluation), addressed per element by the stable `sN`
id the emit layer stamps. Imports nothing from the pipeline (kept dependency-free so
Options can hold it without a cycle)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# Strategy provenance labels — one per fitter in generate_geometry_candidates.
PRIMITIVE = "primitive"          # recognize_primitive -> circle/rect/ellipse
TRAPEZOID = "trapezoid"          # rounded_trapezoid_fit
SYM_POLYGON = "sym_polygon"      # symmetric_polygon_fit
CAP = "cap"                      # half_ellipse_cap_fit
SYMMETRIC = "symmetric"          # symmetric_fit
POLYGON = "polygon"              # recognize_polygon
PATH = "path"                    # fit_path
HOLED_SYM = "holed_symmetric"    # multi-contour mirrored halves (even-odd)
HOLED_PATH = "holed_path"        # multi-contour per-contour fit (even-odd)

KNOWN_STRATEGIES = frozenset({
    PRIMITIVE, TRAPEZOID, SYM_POLYGON, CAP, SYMMETRIC,
    POLYGON, PATH, HOLED_SYM, HOLED_PATH,
})


@dataclass(frozen=True)
class ElementSelection:
    """One element's manual policy. `allow` (None = all) restricts which strategies
    are scored; `force` (None = auto) names the strategy whose candidate should win."""
    allow: frozenset[str] | None = None
    force: str | None = None


@dataclass(frozen=True)
class SelectionPolicy:
    """Per-element selection keyed by emit-time id (`s0`, `s1`, ...), with an optional
    `default` applied to elements that have no explicit entry."""
    by_id: Mapping[str, ElementSelection] = field(default_factory=dict)
    default: ElementSelection | None = None

    def for_id(self, eid: str) -> ElementSelection | None:
        return self.by_id.get(eid, self.default)


def validate_strategies(sel: ElementSelection) -> None:
    """Raise ValueError if `allow` or `force` names a strategy outside KNOWN_STRATEGIES
    (catches typos loudly instead of silently warning per element)."""
    labels = set(sel.allow or ())
    if sel.force is not None:
        labels.add(sel.force)
    unknown = labels - KNOWN_STRATEGIES
    if unknown:
        raise ValueError(
            f"unknown selection strategy {sorted(unknown)}; "
            f"known: {sorted(KNOWN_STRATEGIES)}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_selection.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/selection.py tests/test_selection.py
git commit -m "feat(selection): policy types + strategy vocabulary (slice 4b)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: `Candidate.strategy` field

**Files:**
- Modify: `src/vectoreyes/candidate.py:35-47`
- Test: `tests/test_selection.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_selection.py`:

```python
from vectoreyes.candidate import Candidate, FlatFill
from vectoreyes.fit import Shape


def test_candidate_strategy_defaults_none_and_is_settable():
    c0 = Candidate(Shape("circle", {}), FlatFill("#000000"), "region")
    assert c0.strategy is None                       # backward compatible default
    c1 = Candidate(Shape("path", {"d": "M0 0"}), FlatFill("#000000"),
                   "region", strategy="path")
    assert c1.strategy == "path"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_selection.py::test_candidate_strategy_defaults_none_and_is_settable -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'strategy'`

- [ ] **Step 3: Write the implementation**

In `src/vectoreyes/candidate.py`, change the `Candidate` body (lines 44-47) to add the field and document it:

```python
    geometry: Shape
    fill: Fill
    source: str
    mirror: Axis | None = None
    strategy: str | None = None   # fitter provenance (slice 4b); None for occlusion/lens/gradient
```

Also extend the docstring's `source` paragraph with a sentence:

```python
    `strategy` is the finer-grained fitter provenance (e.g. "symmetric" vs "path")
    used by manual selection; `source` stays the coarse element category.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_selection.py -q && .venv/bin/python -m pytest -q`
Expected: both PASS (full suite still 198 passed — Task 1 added 5, Task 2 added 1, minus nothing; exact count not critical, must be all green)

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/candidate.py tests/test_selection.py
git commit -m "feat(candidate): optional strategy provenance field (slice 4b)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: `GeomCandidate` + labeled generation (return-type change, behavior unchanged)

**Files:**
- Modify: `src/vectoreyes/selector.py:30-130`
- Modify: `tests/test_selector.py` (update for new return type)
- Test: `tests/test_selector.py`

This task changes `generate_geometry_candidates` to return `list[GeomCandidate]` and updates `select_geometry`'s two internal uses so behavior is **identical** to 4a. The 3-stage selection logic comes in Task 4.

- [ ] **Step 1: Update the existing selector tests for the new return type**

In `tests/test_selector.py`, the three `generate_geometry_candidates` tests currently read `c.kind` / `c.params` directly on results. Change them to use `.shape` and assert `.strategy`.

Replace `test_candidates_for_disk_include_circle_and_path_first_is_circle`:

```python
def test_candidates_for_disk_include_circle_and_path_first_is_circle():
    h = w = 80
    region = Region(1, _disk(40, 40, 25, h, w), "#1e64eb")
    cands = generate_geometry_candidates(region, Options(), None, 0.0)
    kinds = [c.shape.kind for c in cands]
    assert "circle" in kinds and "path" in kinds
    assert cands[0].shape.kind == "circle"        # cascade-priority order preserved
    assert cands[0].strategy == "primitive"        # the producing fitter is labeled
```

Replace `test_candidates_nonempty_for_organic_blob`:

```python
def test_candidates_nonempty_for_organic_blob():
    h = w = 80
    mask = np.zeros((h, w), bool)
    mask[20:60, 20:60] = True
    mask[20:35, 20:35] = False           # a bite -> not a clean primitive
    region = Region(1, mask, "#222222")
    cands = generate_geometry_candidates(region, Options(), None, 0.0)
    assert cands and cands[-1].shape.kind == "path"    # fit_path is the final fallback
    assert cands[-1].strategy == "path"
```

In `test_straddler_with_symmetric_candidate_excludes_nonsymmetric_fallback`, change the
exclusion checks to read through `.shape`:

```python
    fp_d = fit_path(contour, epsilon=Options().epsilon, max_error=Options().max_error).params["d"]
    assert not any(c.shape.params.get("d") == fp_d and "fill_rule" not in c.shape.params
                   for c in cands)
    poly = recognize_polygon(contour, epsilon=Options().epsilon)
    if poly is not None:
        assert not any(c.shape.kind == "polygon" and c.shape.params.get("points") == poly.params["points"]
                       for c in cands)
```

In `test_holed_straddler_emits_single_symmetric_evenodd_candidate`, update the three result reads:

```python
    cands = generate_geometry_candidates(region, Options(), Axis(59.5), 2.0)
    assert len(cands) == 1
    assert cands[0].shape.kind == "path"
    assert cands[0].shape.params.get("fill_rule") == "evenodd"
    assert cands[0].strategy == "holed_symmetric"
```

In `test_primitive_only_straddler_excludes_nonsymmetric_fallback`, update result reads:

```python
    cands = sel.generate_geometry_candidates(region, Options(), Axis(49.0), 0.0)
    assert any(c.shape.kind == "circle" for c in cands)
    contour = [c for c in region_contours(mask) if len(c) >= 3][0]
    fp_d = fit_path(contour, epsilon=Options().epsilon, max_error=Options().max_error).params["d"]
    assert not any(c.shape.params.get("d") == fp_d and "fill_rule" not in c.shape.params
                   for c in cands)
    poly = recognize_polygon(contour, epsilon=Options().epsilon)
    if poly is not None:
        assert not any(c.shape.kind == "polygon" and c.shape.params.get("points") == poly.params["points"]
                       for c in cands)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_selector.py -q`
Expected: FAIL — `AttributeError: 'Shape' object has no attribute 'shape'` (results are still bare Shapes).

- [ ] **Step 3: Write the implementation**

In `src/vectoreyes/selector.py`, add the import and the `GeomCandidate` dataclass after the existing imports (the file already imports `Shape`, `Candidate`, `FlatFill`, the fitters, `rank_candidates`, `Axis`, `Region`):

```python
from dataclasses import dataclass

from .selection import (
    PRIMITIVE, TRAPEZOID, SYM_POLYGON, CAP, SYMMETRIC, POLYGON, PATH,
    HOLED_SYM, HOLED_PATH, ElementSelection, validate_strategies,
)


@dataclass(frozen=True)
class GeomCandidate:
    """A generated geometry paired with the fitter (strategy) that produced it."""
    strategy: str
    shape: Shape
```

Rewrite `generate_geometry_candidates` to return labeled candidates (same order, same gating):

```python
def generate_geometry_candidates(
    region: Region, opt, axis: Axis | None, corner_radius: float,
) -> list[GeomCandidate]:
    """All geometry fits the cascade could produce for this region, in cascade
    priority order (candidates[0].shape == the old _fit_region pick), non-None only,
    each tagged with its producing strategy.

    For a straddler (axis set) the non-symmetric fallbacks (recognize_polygon,
    fit_path) are added ONLY when no symmetric candidate exists — so the scorer can
    never pick a cheaper non-symmetric geometry over a valid symmetric one."""
    contours = [c for c in region_contours(region.mask) if len(c) >= 3]
    if not contours:
        return []

    if len(contours) > 1:                       # holed / counter
        if axis is not None:
            halves = [
                symmetric_fit(c, axis.x, corner_radius=corner_radius,
                              epsilon=opt.epsilon, max_error=opt.max_error)
                for c in contours
            ]
            if all(s is not None for s in halves):
                d = " ".join(s.params["d"] for s in halves)
                return [GeomCandidate(HOLED_SYM, Shape("path", {"d": d, "fill_rule": "evenodd"}))]
        d = " ".join(
            fit_path(c, epsilon=opt.epsilon, max_error=opt.max_error).params["d"]
            for c in contours
        )
        return [GeomCandidate(HOLED_PATH, Shape("path", {"d": d, "fill_rule": "evenodd"}))]

    contour = contours[0]
    cands: list[GeomCandidate] = []

    prim = recognize_primitive(contour, epsilon=opt.epsilon)
    if prim is not None:
        cands.append(GeomCandidate(PRIMITIVE, _snap_to_axis(prim, axis) if axis is not None else prim))

    sym: list[GeomCandidate] = []
    if axis is not None:
        trap = rounded_trapezoid_fit(contour, axis.x, radius=corner_radius, max_error=opt.max_error)
        if trap is not None:
            sym.append(GeomCandidate(TRAPEZOID, trap))
        poly = symmetric_polygon_fit(contour, axis.x, epsilon=opt.epsilon)
        if poly is not None:
            sym.append(GeomCandidate(SYM_POLYGON, poly))
        cap = half_ellipse_cap_fit(contour, axis.x, corner_radius=corner_radius, max_error=opt.max_error)
        if cap is not None:
            sym.append(GeomCandidate(CAP, cap))
        s = symmetric_fit(contour, axis.x, corner_radius=corner_radius,
                          epsilon=opt.epsilon, max_error=opt.max_error)
        if s is not None:
            sym.append(GeomCandidate(SYMMETRIC, s))
    cands.extend(sym)

    has_symmetry_preserving = bool(sym) or (axis is not None and prim is not None)

    if axis is None or not has_symmetry_preserving:
        gpoly = recognize_polygon(contour, epsilon=opt.epsilon)
        if gpoly is not None:
            cands.append(GeomCandidate(POLYGON, gpoly))
        cands.append(GeomCandidate(PATH, fit_path(contour, epsilon=opt.epsilon, max_error=opt.max_error)))

    return cands
```

Update `select_geometry` to consume the new type, with behavior unchanged (the `element` param + stages arrive in Task 4). Replace the body after `if not cands: return None`:

```python
def select_geometry(
    region: Region, opt, axis: Axis | None, corner_radius: float,
    source_rgb: np.ndarray | None,
) -> Shape | None:
    """Generate geometry candidates, score them (simplest faithful geometry wins),
    return the winning Shape. Without `source_rgb` fall back to candidates[0].shape =
    the cascade-priority pick. None if no candidate."""
    cands = generate_geometry_candidates(region, opt, axis, corner_radius)
    if not cands:
        return None
    if source_rgb is None:
        return cands[0].shape
    wrapped = [Candidate(gc.shape, FlatFill(region.color_hex), "region", strategy=gc.strategy)
               for gc in cands]
    bbox = _region_bbox(region.mask)
    tol = getattr(opt, "fidelity_tol", 0.06)
    ranked = rank_candidates(wrapped, source_rgb, region, fidelity_tol=tol, bbox=bbox)
    return ranked[0][0].geometry
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_selector.py -q && .venv/bin/python -m pytest -q`
Expected: both PASS (full suite all green — behavior identical, only the internal return type changed).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/selector.py tests/test_selector.py
git commit -m "refactor(selector): label geometry candidates with producing strategy (slice 4b)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: 3-stage `select_geometry` (validate → allow-filter → score → force)

**Files:**
- Modify: `src/vectoreyes/selector.py` (`select_geometry`)
- Test: `tests/test_selection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_selection.py`:

```python
import warnings

import numpy as np

from vectoreyes.pipeline import Options
from vectoreyes.types import Region
from vectoreyes.selector import select_geometry, generate_geometry_candidates


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _disk_region_and_src():
    h = w = 100
    mask = _disk(50, 50, 32, h, w)
    src = np.full((h, w, 3), 255, np.uint8)
    src[mask] = (30, 100, 235)
    return Region(1, mask, "#1e64eb"), src


def test_allow_restricts_winner_to_allowed_strategy():
    region, src = _disk_region_and_src()
    # auto would pick "primitive" (circle); restrict to path -> a path must win
    sel = ElementSelection(allow=frozenset({PATH}))
    shape = select_geometry(region, Options(), None, 0.0, src, element=sel)
    assert shape.kind == "path"


def test_allow_empty_set_warns_and_falls_back_to_auto():
    region, src = _disk_region_and_src()
    sel = ElementSelection(allow=frozenset({SYMMETRIC}))  # no symmetric cand for a plain disk
    with pytest.warns(UserWarning, match="removed all candidates"):
        shape = select_geometry(region, Options(), None, 0.0, src, element=sel, eid="s0")
    assert shape.kind == "circle"                         # auto winner survives


def test_force_present_strategy_overrides_auto_winner():
    region, src = _disk_region_and_src()
    sel = ElementSelection(force=PATH)                    # auto picks circle; force path
    shape = select_geometry(region, Options(), None, 0.0, src, element=sel)
    assert shape.kind == "path"


def test_force_absent_strategy_warns_and_returns_auto_winner():
    region, src = _disk_region_and_src()
    sel = ElementSelection(force=SYMMETRIC)               # not generated for a plain disk
    with pytest.warns(UserWarning, match="not among"):
        shape = select_geometry(region, Options(), None, 0.0, src, element=sel, eid="s0")
    assert shape.kind == "circle"


def test_unknown_force_label_raises_valueerror():
    region, src = _disk_region_and_src()
    with pytest.raises(ValueError, match="blob"):
        select_geometry(region, Options(), None, 0.0, src, element=ElementSelection(force="blob"))


def test_force_works_without_source_rgb():
    region, _ = _disk_region_and_src()
    sel = ElementSelection(force=PATH)
    shape = select_geometry(region, Options(), None, 0.0, None, element=sel)
    assert shape.kind == "path"


def test_element_none_is_pure_passthrough():
    region, src = _disk_region_and_src()
    assert select_geometry(region, Options(), None, 0.0, src).kind == "circle"
    assert select_geometry(region, Options(), None, 0.0, src, element=None).kind == "circle"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_selection.py -k "allow or force or passthrough or unknown_force" -q`
Expected: FAIL — `TypeError: select_geometry() got an unexpected keyword argument 'element'`

- [ ] **Step 3: Write the implementation**

In `src/vectoreyes/selector.py`, add `import warnings` at the top, then replace `select_geometry` with the 3-stage version:

```python
def select_geometry(
    region: Region, opt, axis: Axis | None, corner_radius: float,
    source_rgb: np.ndarray | None, *,
    element: ElementSelection | None = None, eid: str = "?",
) -> Shape | None:
    """Generate geometry candidates, optionally apply a manual `element` policy, score,
    and return the winning Shape. With `element` None this is the 4a auto-selector
    (pure pass-through). `eid` only labels warning messages.

    Stages (skipped when element is None): validate the policy's strategy labels;
    pre-restriction keeps only `allow`ed strategies (warn + restore if that empties the
    set); post-override returns the highest-ranked candidate whose strategy == `force`
    (warn + auto winner if absent)."""
    if element is not None:
        validate_strategies(element)

    cands = generate_geometry_candidates(region, opt, axis, corner_radius)
    if not cands:
        return None

    if element is not None and element.allow is not None:
        kept = [gc for gc in cands if gc.strategy in element.allow]
        if not kept:
            warnings.warn(
                f"selection {eid}: allow={sorted(element.allow)} removed all candidates "
                f"(have {[gc.strategy for gc in cands]}); ignoring restriction",
                UserWarning, stacklevel=2,
            )
        else:
            cands = kept

    force = element.force if element is not None else None

    if source_rgb is None:
        if force is not None:
            hit = next((gc for gc in cands if gc.strategy == force), None)
            if hit is None:
                warnings.warn(
                    f"selection {eid}: force='{force}' not among "
                    f"{[gc.strategy for gc in cands]}; using '{cands[0].strategy}'",
                    UserWarning, stacklevel=2,
                )
                hit = cands[0]
            return hit.shape
        return cands[0].shape

    wrapped = [Candidate(gc.shape, FlatFill(region.color_hex), "region", strategy=gc.strategy)
               for gc in cands]
    bbox = _region_bbox(region.mask)
    tol = getattr(opt, "fidelity_tol", 0.06)
    ranked = rank_candidates(wrapped, source_rgb, region, fidelity_tol=tol, bbox=bbox)

    if force is not None:
        hit = next((c for c, _ in ranked if c.strategy == force), None)
        if hit is None:
            warnings.warn(
                f"selection {eid}: force='{force}' not among "
                f"{[c.strategy for c, _ in ranked]}; using auto winner "
                f"'{ranked[0][0].strategy}'",
                UserWarning, stacklevel=2,
            )
            hit = ranked[0][0]
        return hit.geometry
    return ranked[0][0].geometry
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_selection.py -q && .venv/bin/python -m pytest -q`
Expected: both PASS (the `element`-None default keeps every existing call green).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/selector.py tests/test_selection.py
git commit -m "feat(selector): manual selection stages in select_geometry (slice 4b)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: `Options.selection` + `sN` addressing in `build_candidates` + integration tests

**Files:**
- Modify: `src/vectoreyes/pipeline.py:78-86` (Options), `src/vectoreyes/pipeline.py:132-184` (build_candidates)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
def test_selection_none_is_byte_identical_to_default():
    # parity: supplying selection=None must not change output at all
    img = _rounded_band_img()
    a = idealize(img, options=Options())
    b = idealize(img, options=Options(selection=None))
    assert a == b


def test_force_path_on_s0_emits_path_not_circle():
    from vectoreyes.selection import SelectionPolicy, ElementSelection
    h = w = 100
    img = np.full((h, w, 3), 255, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    img[(xx - 50) ** 2 + (yy - 50) ** 2 <= 32 ** 2] = (30, 100, 235)
    auto = idealize(img, options=Options())
    assert "<circle" in auto                                  # auto picks a circle
    policy = SelectionPolicy(by_id={"s0": ElementSelection(force="path")})
    forced = idealize(img, options=Options(selection=policy))
    assert "<circle" not in forced and "<path" in forced     # sN addressing reached emit


def test_default_policy_restricts_all_elements():
    from vectoreyes.selection import SelectionPolicy, ElementSelection
    h = w = 100
    img = np.full((h, w, 3), 255, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    img[(xx - 50) ** 2 + (yy - 50) ** 2 <= 32 ** 2] = (30, 100, 235)
    policy = SelectionPolicy(default=ElementSelection(force="path"))
    out = idealize(img, options=Options(selection=policy))
    assert "<circle" not in out and "<path" in out           # default reached the un-keyed element
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k "selection or force_path or default_policy" -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'selection'`

- [ ] **Step 3a: Add the Options field**

In `src/vectoreyes/pipeline.py`, add to the `Options` dataclass (after `fidelity_tol`):

```python
    selection: "SelectionPolicy | None" = None  # manual candidate selection (slice 4b)
```

Add the import near the other `from .` imports at the top of `pipeline.py`:

```python
from .selection import SelectionPolicy
```

(With the import present, the quotes on the annotation are optional — use the bare type `SelectionPolicy | None = None`.)

- [ ] **Step 3b: Thread `sN` addressing through `build_candidates`**

In `build_candidates`, the region/gradient loops call `select_geometry`. Each element's emit id is `f"s{len(cands)}"` at the moment before its candidate is appended. Read the policy from `opt.selection` and pass `element` + `eid`.

Replace the region loop (currently lines ~161-166):

```python
    for region, fit_axis, is_pair in drawn:
        eid = f"s{len(cands)}"
        element = opt.selection.for_id(eid) if opt.selection is not None else None
        shape = select_geometry(region, opt, fit_axis, corner_radius, source_rgb,
                                element=element, eid=eid)
        if shape is None:
            continue
        cands.append(Candidate(shape, FlatFill(region.color_hex), "region",
                               mirror=axis if is_pair else None))
```

Replace the gradient loop's `select_geometry` call (currently line ~174):

```python
    for footprint, model in gradient_fills:
        eid = f"s{len(cands)}"
        element = opt.selection.for_id(eid) if opt.selection is not None else None
        shape = select_geometry(footprint, opt, None, corner_radius, source_rgb,
                                element=element, eid=eid)
        if shape is None:
            continue
```

> **Invariant note for the implementer:** `select_geometry` returning `None` (a dropped element) does NOT append, so the next element's `len(cands)` is unchanged — meaning a dropped element consumes no `sN`. This matches the emit loop, which only increments `eid` for candidates that made it into `cands`. Do not pre-reserve ids for dropped elements.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q && .venv/bin/python -m pytest -q`
Expected: both PASS. The parity test proves `selection=None` is byte-identical; the full suite (including the byte-identical golden harness `tests/test_candidate_byte_identical.py`) stays green with no re-capture.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): thread SelectionPolicy with sN addressing (slice 4b)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Full suite green**

Run: `.venv/bin/python -m pytest -q`
Expected: all passed (197 baseline + new selection/pipeline tests), including `tests/test_candidate_byte_identical.py` with no golden re-capture.

- [ ] **Parity sanity (no policy = no change)** — already asserted by `test_selection_none_is_byte_identical_to_default`; confirm it is in the run above.

- [ ] Dispatch the final whole-branch code review, then use `superpowers:finishing-a-development-branch`.

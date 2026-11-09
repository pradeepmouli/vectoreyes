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
    validate_strategies(ElementSelection(allow=frozenset({PRIMITIVE, PATH}), force=SYMMETRIC))  # should not raise


def test_validate_rejects_unknown_allow_label():
    with pytest.raises(ValueError, match="symetric"):
        validate_strategies(ElementSelection(allow=frozenset({"symetric"})))


def test_validate_rejects_unknown_force_label_even_when_allow_none():
    with pytest.raises(ValueError, match="blob"):
        validate_strategies(ElementSelection(force="blob"))


from vectoreyes.candidate import Candidate, FlatFill
from vectoreyes.fit import Shape


def test_candidate_strategy_defaults_none_and_is_settable():
    c0 = Candidate(Shape("circle", {}), FlatFill("#000000"), "region")
    assert c0.strategy is None                       # backward compatible default
    c1 = Candidate(Shape("path", {"d": "M0 0"}), FlatFill("#000000"),
                   "region", strategy="path")
    assert c1.strategy == "path"


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
    shape, _ = select_geometry(region, Options(), None, 0.0, src)
    assert shape.kind == "circle"  # auto baseline
    # auto would pick "primitive" (circle); restrict to path -> a path must win
    sel = ElementSelection(allow=frozenset({PATH}))
    shape, _ = select_geometry(region, Options(), None, 0.0, src, element=sel)
    assert shape.kind == "path"


def test_allow_empty_set_warns_and_falls_back_to_auto():
    region, src = _disk_region_and_src()
    sel = ElementSelection(allow=frozenset({SYMMETRIC}))  # no symmetric cand for a plain disk
    with pytest.warns(UserWarning, match="removed all candidates"):
        shape, _ = select_geometry(region, Options(), None, 0.0, src, element=sel, eid="s0")
    assert shape.kind == "circle"                         # auto winner survives


def test_force_present_strategy_overrides_auto_winner():
    region, src = _disk_region_and_src()
    sel = ElementSelection(force=PATH)                    # auto picks circle; force path
    shape, _ = select_geometry(region, Options(), None, 0.0, src, element=sel)
    assert shape.kind == "path"


def test_force_absent_strategy_warns_and_returns_auto_winner():
    region, src = _disk_region_and_src()
    sel = ElementSelection(force=SYMMETRIC)               # not generated for a plain disk
    with pytest.warns(UserWarning, match="not among"):
        shape, _ = select_geometry(region, Options(), None, 0.0, src, element=sel, eid="s0")
    assert shape.kind == "circle"


def test_unknown_force_label_raises_valueerror():
    region, src = _disk_region_and_src()
    with pytest.raises(ValueError, match="blob"):
        select_geometry(region, Options(), None, 0.0, src, element=ElementSelection(force="blob"))


def test_force_works_without_source_rgb():
    region, _ = _disk_region_and_src()
    sel = ElementSelection(force=PATH)
    shape, _ = select_geometry(region, Options(), None, 0.0, None, element=sel)
    assert shape.kind == "path"


def test_element_none_is_pure_passthrough():
    region, src = _disk_region_and_src()
    shape1, _ = select_geometry(region, Options(), None, 0.0, src)
    assert shape1.kind == "circle"
    shape2, _ = select_geometry(region, Options(), None, 0.0, src, element=None)
    assert shape2.kind == "circle"


def test_force_after_allow_operates_on_narrowed_set():
    region, src = _disk_region_and_src()
    sel = ElementSelection(allow=frozenset({PATH}), force=PATH)   # path survives allow; force finds it
    shape, _ = select_geometry(region, Options(), None, 0.0, src, element=sel)
    assert shape.kind == "path"


def test_force_absent_after_allow_warns_and_falls_back_within_narrowed_set():
    region, src = _disk_region_and_src()
    # allow={PATH} narrows to path only; force=SYMMETRIC is absent from the narrowed set ->
    # warn + fall back to the narrowed auto winner (path), NOT the original circle.
    sel = ElementSelection(allow=frozenset({PATH}), force=SYMMETRIC)
    with pytest.warns(UserWarning, match="not among"):
        shape, _ = select_geometry(region, Options(), None, 0.0, src, element=sel)
    assert shape.kind == "path"   # fallback drawn from the narrowed set, proving stage order

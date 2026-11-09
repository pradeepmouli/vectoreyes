import numpy as np

from vectoreyes.contour import region_contours
from vectoreyes.fit import fit_path, recognize_polygon
from vectoreyes.pipeline import Options
from vectoreyes.types import Axis, Region
from vectoreyes.selector import generate_geometry_candidates


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def test_candidates_for_disk_include_circle_and_path_first_is_circle():
    h = w = 80
    region = Region(1, _disk(40, 40, 25, h, w), "#1e64eb")
    cands = generate_geometry_candidates(region, Options(), None, 0.0)
    kinds = [c.shape.kind for c in cands]
    assert "circle" in kinds and "path" in kinds
    assert cands[0].shape.kind == "circle"        # cascade-priority order preserved
    assert cands[0].strategy == "primitive"        # the producing fitter is labeled


def test_candidates_nonempty_for_organic_blob():
    h = w = 80
    mask = np.zeros((h, w), bool)
    mask[20:60, 20:60] = True
    mask[20:35, 20:35] = False           # a bite -> not a clean primitive
    region = Region(1, mask, "#222222")
    cands = generate_geometry_candidates(region, Options(), None, 0.0)
    assert cands and cands[-1].shape.kind == "path"    # fit_path is the final fallback
    assert cands[-1].strategy == "path"


def test_straddler_with_symmetric_candidate_excludes_nonsymmetric_fallback():
    # a centered vertical bar (axis x=40): symmetric fits exist, so the non-symmetric
    # recognize_polygon/fit_path fallbacks must NOT be appended (symmetry preserved —
    # the branch's key invariant: no non-symmetric candidate can win over a valid
    # symmetric one for a straddler).
    h = w = 80
    mask = np.zeros((h, w), bool)
    mask[20:60, 30:50] = True
    region = Region(1, mask, "#333333")
    cands = generate_geometry_candidates(region, Options(), Axis(40.0), 2.0)
    assert cands                                  # at least one symmetric candidate exists

    # The bare non-symmetric fit_path fallback (a path with no fill_rule) must be
    # ABSENT from the candidate set: compute what it would be and assert exclusion.
    contour = [c for c in region_contours(mask) if len(c) >= 3][0]
    fp_d = fit_path(contour, epsilon=Options().epsilon, max_error=Options().max_error).params["d"]
    assert not any(c.shape.params.get("d") == fp_d and "fill_rule" not in c.shape.params
                   for c in cands)
    # and no non-symmetric recognize_polygon fallback slipped in either
    poly = recognize_polygon(contour, epsilon=Options().epsilon)
    if poly is not None:
        assert not any(c.shape.kind == "polygon" and c.shape.params.get("points") == poly.params["points"]
                       for c in cands)


def test_holed_straddler_emits_single_symmetric_evenodd_candidate():
    # A centered annulus is a holed straddler whose every contour fits symmetrically.
    # The selector must return EXACTLY ONE candidate — the mirrored-half even-odd path —
    # never also the per-contour fit_path (which the scorer could pick and break exact
    # symmetry). Locks the early-return in the holed branch.
    h = w = 120
    yy, xx = np.ogrid[:h, :w]
    r2 = (xx - 59.5) ** 2 + (yy - 59.5) ** 2
    mask = (r2 <= 46 ** 2) & (r2 >= 22 ** 2)
    region = Region(1, mask, "#142850")
    cands = generate_geometry_candidates(region, Options(), Axis(59.5), 2.0)
    assert len(cands) == 1
    assert cands[0].shape.kind == "path"
    assert cands[0].shape.params.get("fill_rule") == "evenodd"
    assert cands[0].strategy == "holed_symmetric"


def test_primitive_only_straddler_excludes_nonsymmetric_fallback(monkeypatch):
    # When NO refine fit succeeds (sym empty) but a primitive IS recognized, the
    # snapped primitive is the symmetry-preserving candidate — so the non-symmetric
    # recognize_polygon/fit_path fallbacks must still be excluded. Forcing the four
    # refine fitters to None is the only way to reach the prim-clause of
    # has_symmetry_preserving (on natural straddlers symmetric_fit always succeeds).
    import vectoreyes.selector as sel
    for name in ("symmetric_fit", "rounded_trapezoid_fit",
                 "half_ellipse_cap_fit", "symmetric_polygon_fit"):
        monkeypatch.setattr(sel, name, lambda *a, **k: None)

    h = w = 100
    mask = _disk(49, 49, 30, h, w)
    region = Region(1, mask, "#142850")
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


from vectoreyes.selector import select_geometry


def test_select_picks_circle_for_clean_disk():
    h = w = 100
    src = np.full((h, w, 3), 255, np.uint8)
    mask = _disk(50, 50, 32, h, w)
    src[mask] = (30, 100, 235)
    region = Region(1, mask, "#1e64eb")
    shape, strategy = select_geometry(region, Options(), None, 0.0, src)
    assert shape is not None and shape.kind == "circle"
    assert strategy == "primitive"


def test_select_falls_back_to_first_candidate_without_source():
    h = w = 80
    region = Region(1, _disk(40, 40, 25, h, w), "#1e64eb")
    shape, strategy = select_geometry(region, Options(), None, 0.0, None)
    assert shape is not None and shape.kind == "circle"   # candidates[0]
    assert strategy == "primitive"


def test_select_returns_none_when_no_contour():
    region = Region(1, np.zeros((20, 20), bool), "#000000")
    shape, strategy = select_geometry(region, Options(), None, 0.0, None)
    assert shape is None and strategy is None

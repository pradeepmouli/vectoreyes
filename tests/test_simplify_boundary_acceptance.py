import numpy as np
import pytest

from vectormark import drawing_simplification as simplify_module


def _path(d):
    return d


def _parts(shape, **kwargs):
    return simplify_module._candidate_parts(
        simplify_module._path_subpaths(shape), epsilon=1, max_error=1,
        cubic=False, preserve_subpaths=True, **kwargs,
    )


def test_tiny_cutout_is_not_dropped_relative_to_large_exterior():
    outer = "M0 0 L500 0 L1000 0 L1000 1000 L0 1000 Z"
    hole = "M20 20 L20.5 20 L20.5 20.5 L20 20.5 Z"
    result = _parts(_path(f"{outer} {hole}"))
    assert len(result) == 2
    assert simplify_module._same_subpath_topology(_path(f"{outer} {hole}"), _path(" ".join(result)))


@pytest.mark.parametrize("limit, expected_shift", [(None, 0.5), (0.1, 0.0)])
def test_boundary_override_applies_to_each_cutout(monkeypatch, limit, expected_shift):
    outer = "M0 0 L1000 0 L1000 1000 L0 1000 Z"
    hole = "M100 100 L200 100 L200 200 L100 200 Z"
    shifted = "M100.5 100 L200.5 100 L200.5 200 L100.5 200 Z"
    monkeypatch.setattr(simplify_module, "_simplified_subpath_d",
                        lambda tokens, points, **kw: shifted if tokens[0][1][0] == 100 else outer)
    parts = _parts(_path(f"{outer} {hole}"), max_boundary_error=limit)
    error = simplify_module._boundary_max_deviation(_path(hole), _path(parts[1]))
    assert error == pytest.approx(expected_shift)


def test_nearby_cutouts_cannot_substitute_for_each_others_boundary(monkeypatch):
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    holes = ["M10 10 L20 10 L20 20 L10 20 Z", "M22 10 L32 10 L32 20 L22 20 Z"]
    replacements = {0: outer, 10: holes[1], 22: holes[0]}
    monkeypatch.setattr(simplify_module, "_simplified_subpath_d",
                        lambda tokens, points, **kw: replacements[tokens[0][1][0]])
    parts = _parts(_path(" ".join([outer, *holes])))
    for source, accepted in zip(holes, parts[1:], strict=True):
        assert simplify_module._boundary_max_deviation(_path(source), _path(accepted)) == 0


def test_damaged_exterior_cannot_use_cutouts_area_to_pass(monkeypatch):
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    hole = "M20 20 L80 20 L80 80 L20 80 Z"
    bad = "M0 -3 L100 0 L100 100 L0 100 Z"
    monkeypatch.setattr(simplify_module, "_simplified_subpath_d",
                        lambda tokens, points, **kw: bad if tokens[0][1][0] == 0 else hole)
    parts = _parts(_path(f"{outer} {hole}"))
    assert simplify_module._boundary_max_deviation(_path(outer), _path(parts[0])) == 0


def test_rejects_cutout_crossing_exterior_even_within_distance_budget(monkeypatch):
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    hole = "M0.2 20 L1 20 L1 21 L0.2 21 Z"
    crossing = "M-0.2 20 L0.6 20 L0.6 21 L-0.2 21 Z"
    shape = _path(f"{outer} {hole}")
    assert simplify_module._boundary_max_deviation(_path(hole), _path(crossing)) < 1
    monkeypatch.setattr(simplify_module, "_candidate_parts", lambda *args, **kw: [outer, crossing])
    assert simplify_module.simplify_path_data(shape, epsilon=1, max_error=1, samples=16, fill_rule='evenodd') is None


def test_retained_boundary_keeps_high_precision_coordinates(monkeypatch):
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    hole = "M10.123456 10 L20.123456 10 L20.123456 20 L10.123456 20 Z"
    bad = "M50 50 L60 50 L60 60 L50 60 Z"
    monkeypatch.setattr(simplify_module, "_simplified_subpath_d", lambda *args, **kw: bad)
    parts = _parts(_path(f"{outer} {hole}"), max_boundary_error=0)
    before = simplify_module._parse_subpaths(hole)[0]
    after = simplify_module._parse_subpaths(parts[1])[0]
    for (_, a), (_, b) in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fill_rule, expected", [("nonzero", False), ("evenodd", True)])
def test_topology_check_honors_fill_rule_when_cutout_winding_changes(fill_rule, expected):
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    hole = "M20 20 L20 80 L80 80 L80 20 Z"
    reversed_hole = "M20 20 L80 20 L80 80 L20 80 Z"
    before = f"{outer} {hole}"
    after = f"{outer} {reversed_hole}"
    assert simplify_module._same_subpath_topology(before, after, fill_rule=fill_rule) is expected

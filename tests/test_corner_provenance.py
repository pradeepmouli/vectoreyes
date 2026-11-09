"""Source-position declarations bind once to actual canonical command IDs."""

from dataclasses import replace

import pytest

from vectoreyes.path_geometry import CornerAnchor, InflectionAnchor, parse_svg_paths


def _path():
    return parse_svg_paths("M0 0 Q2 0 3 1 Q4 2 4 4 Q4 6 5 7 L0 8 L0 0 Z", region_id="r1")[0]


def _span(name="p0.corner0", entry=0, exit=2, commands=(1, 2), subpath=0):
    return dict(corner_id=name, subpath=subpath, entry_command=entry,
        exit_command=exit, commands=commands, radius=None)


def test_prefit_binding_uses_actual_sparse_command_ids_without_mutating_inputs():
    from vectoreyes.corner_provenance import bind_prefit_corner_anchors

    path = _path()
    path = replace(path, path=replace(path.path, commands=tuple(
        replace(command, id=f"r1.p0.c{index * 3}")
        for index, command in enumerate(path.path.commands))))
    source = _span()
    result, skipped, bindings = bind_prefit_corner_anchors({0: path}, (source,), overlap_policy="retain_first")
    anchor = result[0].anchors[0]
    assert anchor == CornerAnchor("r1.p0.corner0", "r1.p0.c0", "r1.p0.c6", ("r1.p0.c3", "r1.p0.c6"))
    assert path.anchors == ()
    assert source == _span()
    assert skipped == ()
    assert bindings == {anchor.id: "p0.corner0"}
    with pytest.raises(TypeError):
        result[0] = path


@pytest.mark.parametrize("policy, kept, rejected", [
    ("retain_first", ("r1.p0.corner0",), 1), ("reject_all", (), 2),
])
def test_prefit_binding_keeps_overlap_policy_explicit(policy, kept, rejected):
    from vectoreyes.corner_provenance import bind_prefit_corner_anchors

    result, skipped, bindings = bind_prefit_corner_anchors({0: _path()},
        (_span(), _span("p0.corner1", 1, 3, (2, 3))), overlap_policy=policy)
    assert tuple(anchor.id for anchor in result[0].anchors) == kept
    assert len(skipped) == rejected
    assert {item["reason"] for item in skipped} == {"overlapping_corner_ownership"}
    assert set(bindings) == set(kept)


def test_prefit_binding_preserves_other_features_and_untouched_paths():
    from vectoreyes.corner_provenance import bind_prefit_corner_anchors

    path = _path()
    inflection = InflectionAnchor("r1.p0.inflection", (path.path.commands[3].id,))
    path = replace(path, anchors=(inflection,))
    untouched = replace(_path(), path=replace(_path().path, id="r1.p9"))
    result, skipped, _ = bind_prefit_corner_anchors({0: path, 1: untouched},
        (_span(),), overlap_policy="retain_first")
    assert result[0].anchors[0] is inflection
    assert result[0].anchors[1].id == "r1.p0.corner0"
    assert result[1] is untouched
    assert skipped == ()


@pytest.mark.parametrize("change", [
    dict(entry_command=-1), dict(exit_command=99), dict(entry_command=True),
    dict(commands=(1,)), dict(radius=-1), dict(subpath=4), dict(corner_id=""),
])
def test_prefit_binding_skips_invalid_declarations(change):
    from vectoreyes.corner_provenance import bind_prefit_corner_anchors

    path = _path()
    result, skipped, bindings = bind_prefit_corner_anchors({0: path},
        ({**_span(), **change},), overlap_policy="retain_first")
    assert result[0] is path
    assert skipped[0]["reason"] == "invalid_corner_anchor"
    assert bindings == {}


def test_prefit_binding_keeps_source_subpath_separate_from_path_address():
    from vectoreyes.corner_provenance import bind_prefit_corner_anchors

    path = parse_svg_paths("M0 0 L5 0 L5 5 L0 5 Z", region_id="r2", path_ids=("r2.p7",))[0]
    source = _span("p3.point", entry=1, exit=1, commands=(), subpath=3)
    result, skipped, bindings = bind_prefit_corner_anchors({3: path}, (source,), overlap_policy="retain_first")
    assert result[3].anchors == (CornerAnchor("r2.p7.point", "r2.p7.c1", "r2.p7.c1"),)
    assert skipped == ()
    assert bindings == {"r2.p7.point": "p3.point"}


def test_drawing_publication_and_normalization_share_prefit_binder(monkeypatch):
    import vectoreyes.corner_provenance as provenance
    from vectoreyes.corner_provenance import ingress_prefit_corners
    from vectoreyes.corner_diagnostics import normalize_corners_path_d, path_corner_diagnostics

    calls = []
    execute = provenance.bind_prefit_corner_anchors
    def record(*args, **kwargs):
        calls.append(kwargs["overlap_policy"])
        return execute(*args, **kwargs)
    monkeypatch.setattr(provenance, "bind_prefit_corner_anchors", record)
    d = "M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z"
    geometry = parse_svg_paths(d, region_id="r1")[0]
    declared = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (14, 4)),))
    published, _ = ingress_prefit_corners(geometry, d, "r1", declared)
    normalized, _ = normalize_corners_path_d(d, max_error=1, diagnostics=declared)
    assert calls == ["retain_first", "reject_all"]
    assert len(published.anchors) == 1
    assert normalized != d

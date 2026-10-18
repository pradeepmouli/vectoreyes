"""Pre-fit evidence matching is independent of SVG text and scene state."""

import pytest

from vectormark.path_geometry import command_token, parse_svg_paths


def _point(point=(10, 0), **changes):
    return dict(corner_id="p0.corner0", source="prefit", kind="point",
        subpath=0, point=point, **changes)


def _commands(d="M0 0 L10 0 L10 10 L0 10 Z"):
    return parse_svg_paths(d, region_id="r1")[0].path.commands


def test_numeric_command_token_inverse_round_trips_every_variant():
    from vectormark.path_geometry import command_from_token

    for command in _commands("M.123456789 0 L1 0 Q1 1 2 2 C2 3 3 3 4 4 A1 2 15 0 1 5 5 Z"):
        kind, values = command_token(command)
        assert command_from_token(kind, values, command_id=command.id) == command


@pytest.mark.parametrize("tolerance, accepted", [(0.02, True), (0.01, False)])
def test_typed_matcher_retains_nearest_endpoint_tolerance(tolerance, accepted):
    from vectormark.corner_matching import match_prefit_corner_spans

    source = _point((10.015, 0))
    spans, skipped = match_prefit_corner_spans({0: _commands()}, (source,), tolerance=tolerance)
    assert bool(spans) is accepted
    if accepted:
        assert spans[0]["entry_command"] == spans[0]["exit_command"] == 1
        assert spans[0]["commands"] == ()
    else:
        assert skipped[0]["reason"] == "missing_or_ambiguous_endpoint"
    assert source["point"] == (10.015, 0)


def test_typed_matcher_rejects_equally_close_endpoints():
    from vectormark.corner_matching import match_prefit_corner_spans

    spans, skipped = match_prefit_corner_spans({0: _commands("M0 0 L2 0 L2 2 L0 2 Z")},
        (_point((1, 0)),), tolerance=2)
    assert spans == ()
    assert skipped[0]["reason"] == "missing_or_ambiguous_endpoint"


@pytest.mark.parametrize("explicit_close", [True, False])
def test_typed_matcher_distinguishes_wrapped_ownership_from_implicit_closure(explicit_close):
    from vectormark.corner_matching import match_prefit_corner_spans

    d = "M2 0 Q4 0 4 2 L4 10 L-10 10 L-10 -2 L0 -2 " + ("Q0 0 2 0 " if explicit_close else "") + "Z"
    declaration = dict(corner_id="p0.rounded", source="prefit", kind="curve", subpath=0,
        start_point=(0, -2), end_point=(4, 2))
    spans, skipped = match_prefit_corner_spans({0: _commands(d)}, (declaration,), tolerance=0.02)
    if explicit_close:
        assert spans[0]["commands"] == (6, 1)
        assert spans[0]["entry_command"] == 5
        assert spans[0]["exit_command"] == 1
        assert skipped == ()
    else:
        assert spans == ()
        assert skipped[0]["reason"] == "implicit_closing_edge"


def test_typed_matcher_merges_move_and_explicit_closing_endpoint_alias():
    from vectormark.corner_matching import match_prefit_corner_spans

    spans, skipped = match_prefit_corner_spans({0: _commands("M0 0 L10 0 L10 10 L0 0 Z")},
        (_point((0, 0)),), tolerance=0.02)
    assert spans[0]["entry_command"] == 0
    assert skipped == ()


@pytest.mark.parametrize("changes", [dict(source="postfit"), dict(kind="unknown"),
    dict(corner_id=""), dict(subpath=True), dict(point=(float("nan"), 0)), dict(point=(1,))])
def test_typed_matcher_rejects_invalid_source_evidence(changes):
    from vectormark.corner_matching import match_prefit_corner_spans

    spans, skipped = match_prefit_corner_spans({0: _commands()},
        ({**_point(), **changes},), tolerance=0.02)
    assert spans == ()
    assert skipped[0]["reason"] == "invalid_corner_provenance"


def test_typed_matcher_does_not_infer_missing_declarations():
    from vectormark.corner_matching import match_prefit_corner_spans

    assert match_prefit_corner_spans({0: _commands()}, (), tolerance=0.02) == ((), ())


def test_typed_matcher_rejects_missing_source_subpath():
    from vectormark.corner_matching import match_prefit_corner_spans

    spans, skipped = match_prefit_corner_spans({0: _commands()},
        ({**_point(), "subpath": 3},), tolerance=0.02)
    assert spans == ()
    assert skipped[0]["reason"] == "missing_subpath"


def test_svg_diagnostics_delegate_endpoint_matching_to_typed_operation(monkeypatch):
    import vectormark.corner_matching as matching
    from vectormark.optimizer.corners import path_corner_diagnostics

    calls = []
    execute = matching.match_prefit_corner_spans
    def record(paths, declarations, **kwargs):
        calls.append(paths)
        return execute(paths, declarations, **kwargs)
    monkeypatch.setattr(matching, "match_prefit_corner_spans", record)
    result = path_corner_diagnostics("M0 0 L10 0 L10 10 L0 10 Z", declared_corners=((10, 0),))
    assert len(calls) == 1
    assert calls[0][0][1].end == (10, 0)
    assert result["spans"][0]["entry_command"] == 1


def test_arc_endpoint_can_receive_declared_point_corner():
    from vectormark.optimizer.corners import path_corner_diagnostics

    result = path_corner_diagnostics("M0 0 A5 5 0 0 1 10 0 L10 10 L0 10 Z", declared_corners=((10, 0),))
    assert result["spans"]
    assert result["spans"][0]["entry_command"] == 1

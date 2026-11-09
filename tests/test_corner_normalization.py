"""Canonical normalization acts on exact declared ownership, not nearby points."""

from dataclasses import replace

import pytest

from vectoreyes.corner_ownership import directed_corner_ownership
from vectoreyes.path_geometry import CornerAnchor, InflectionAnchor, QuadraticCommand, parse_svg_paths


def test_numeric_command_tokens_preserve_every_command_variant_without_rounding():
    from vectoreyes.path_geometry import command_token

    path = parse_svg_paths("M0.123456789 0 L1 0 Q1 1 2 2 C2 3 3 3 4 4 A1 2 15 0 1 5 5 Z", region_id="r1")[0]
    assert tuple(command_token(command) for command in path.path.commands) == (
        ("M", (0.123456789, 0)), ("L", (1, 0)), ("Q", (1, 1, 2, 2)),
        ("C", (2, 3, 3, 3, 4, 4)), ("A", (1, 2, 15, 0, 1, 5, 5)),
    )


def _path(d, spans=(), points=()):
    geometry = parse_svg_paths(d, region_id="r1")[0]
    ids = tuple(command.id for command in geometry.path.commands)
    anchors = tuple(CornerAnchor(id=f"corner{i}", entry_command_id=ids[entry],
        exit_command_id=ids[exit], owned_command_ids=directed_corner_ownership(
            ids, ids[entry], ids[exit],
            explicit_closing_drawable=geometry.path.commands[-1].end == geometry.path.start),
        radius=None) for i, (entry, exit) in enumerate(spans))
    anchors += tuple(CornerAnchor(id=f"point{i}", entry_command_id=ids[index],
        exit_command_id=ids[index]) for i, index in enumerate(points))
    return replace(geometry, anchors=anchors)


def test_canonical_corner_preserves_boundaries_side_commands_and_identity():
    from vectoreyes.corner_normalization import normalize_path_corners

    original = _path("M0 0 L10.001 0 Q12.001 0 13.001 1 Q14.001 2 14.001 4 C18.003 7 20.004 9 20.005 12 L0 12 Z", [(1, 3)])
    result, receipt = normalize_path_corners(original, max_error=1)
    assert len(result.path.commands) == len(original.path.commands) - 1
    assert result.path.commands[1] is original.path.commands[1]
    assert result.path.commands[3:] == original.path.commands[4:]
    curve = result.path.commands[2]
    assert isinstance(curve, QuadraticCommand)
    assert curve.id == original.path.commands[3].id
    assert curve.end == original.path.commands[3].end
    assert result.anchors[0] == replace(original.anchors[0], owned_command_ids=(curve.id,))
    assert receipt["normalized"] == ("corner0",)
    assert original.path.commands[2].id not in receipt["commands"]
    assert normalize_path_corners(result, max_error=1)[0] is result


def test_canonical_corner_does_not_infer_undeclared_features():
    from vectoreyes.corner_normalization import normalize_path_corners

    original = _path("M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z")
    result, receipt = normalize_path_corners(original, max_error=1)
    assert result is original
    assert receipt["normalized"] == ()
    assert receipt["skipped"] == ()


def test_canonical_wrapped_corner_rotates_start_and_keeps_other_features():
    from vectoreyes.corner_normalization import normalize_path_corners

    original = _path("M2 0 Q4 0 4 2 L4 10 L-10 10 L-10 -2 L0 -2 Q0 0 2 0 Z", [(5, 1)], [3])
    result, receipt = normalize_path_corners(original, max_error=2)
    assert result.path.start == original.path.commands[5].end
    assert tuple(command.id for command in result.path.drawable_commands) == tuple(
        original.path.commands[index].id for index in [1, 2, 3, 4, 5])
    assert result.path.commands[2:] == original.path.commands[2:6]
    assert result.anchors[1] == original.anchors[1]
    assert result.anchors[0].owned_command_ids == (original.path.commands[1].id,)
    assert original.path.commands[0].id not in receipt["commands"]


@pytest.mark.parametrize("feature", ["point", "inflection"])
def test_canonical_corner_does_not_consume_protected_anchor(feature):
    from vectoreyes.corner_normalization import normalize_path_corners

    original = _path("M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z", [(1, 3)], [2] if feature == "point" else [])
    if feature == "inflection":
        original = replace(original, anchors=(*original.anchors,
            InflectionAnchor("inflection", (original.path.commands[2].id,))))
    result, receipt = normalize_path_corners(original, max_error=10)
    assert result is original
    assert receipt["skipped"][0]["reason"] == "protected_anchor_inside_span"


def test_canonical_corner_at_serialization_origin_is_protected():
    from vectoreyes.corner_normalization import normalize_path_corners

    original = _path("M2 0 Q4 0 4 2 L4 10 L-10 10 L-10 -2 L0 -2 Q0 0 2 0 Z", [(5, 1)], [0])
    result, receipt = normalize_path_corners(original, max_error=10)
    assert result is original
    assert receipt["skipped"][0]["reason"] == "protected_anchor_inside_span"


def test_adjacent_canonical_corners_keep_shared_boundary():
    from vectoreyes.corner_normalization import normalize_path_corners

    original = _path("M0 0 Q2 0 3 1 Q4 2 4 4 Q4 6 5 7 Q6 8 8 8 L0 8 Z", [(0, 2), (2, 4)])
    result, receipt = normalize_path_corners(original, max_error=1)
    assert len(result.path.commands) == 4
    assert result.anchors[0].exit_command_id == result.anchors[1].entry_command_id
    assert result.anchors[0].owned_command_ids == (original.path.commands[2].id,)
    assert result.anchors[1].owned_command_ids == (original.path.commands[4].id,)
    assert receipt["normalized"] == ("corner0", "corner1")


@pytest.mark.parametrize("d", ["M0 0 L10 0 Q14 0 14 4 L0 12 Z", "M0 0 L10 0 L14 2 L14 4 L0 12 Z"])
def test_single_quadratic_and_noncurve_spans_are_not_refit(d):
    from vectoreyes.corner_normalization import normalize_path_corners

    original = _path(d, [(1, 2)] if "Q" in d else [(1, 3)])
    result, receipt = normalize_path_corners(original, max_error=10)
    assert result is original
    assert receipt["normalized"] == ()
    if "Q" not in d:
        assert receipt["skipped"][0]["reason"] == "non_curve_span"


def test_svg_normalizer_routes_through_canonical_corner_operation(monkeypatch):
    import vectoreyes.corner_normalization as canonical
    from vectoreyes.corner_diagnostics import normalize_corners_path_d, path_corner_diagnostics

    calls = []
    operation = canonical.normalize_path_corners
    def record(geometry, **kwargs):
        calls.append(geometry)
        return operation(geometry, **kwargs)
    monkeypatch.setattr(canonical, "normalize_path_corners", record)
    d = "M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z"
    declared = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (14, 4)),))
    result, _ = normalize_corners_path_d(d, max_error=1, diagnostics=declared)
    assert len(calls) == 1
    assert len(calls[0].anchors) == 1
    assert result != d


def test_empty_legacy_declaration_does_not_require_canonical_path_ingress():
    from vectoreyes.corner_diagnostics import normalize_corners_path_d, path_corner_diagnostics

    d = "M0 0 Q1 1 0 0 Z"
    declaration = path_corner_diagnostics(d)
    result, receipt = normalize_corners_path_d(d, max_error=1, diagnostics=declaration)
    assert result == d
    assert receipt["normalized"] == []


def test_canonical_corner_cannot_collapse_below_closed_path_minimum():
    from vectoreyes.corner_normalization import normalize_path_corners

    original = _path("M0 0 Q2 0 2 2 Q0 2 0 0 Z", [(0, 2)])
    result, receipt = normalize_path_corners(original, max_error=10)
    assert result is original
    assert receipt["skipped"][0]["reason"] == "minimum_path_commands"

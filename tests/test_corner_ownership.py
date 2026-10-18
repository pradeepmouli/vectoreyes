import copy

import pytest

from vectormark.optimizer.corners import (
    normalize_corners_path_d,
    path_corner_diagnostics,
    rebind_prefit_corner_diagnostics,
)
from vectormark.optimizer.vector_region import _parse_subpaths


@pytest.mark.parametrize("side", ["Q5 -3 10 0", "C3 -2 7 -2 10 0"])
def test_corner_owns_only_curves_between_entry_and_exit(side):
    d = f"M0 0 {side} Q12 0 13 1 Q14 2 14 4 C18 7 20 9 20 12 L0 12 Z"
    declared = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (14, 4)),))
    snapshot = copy.deepcopy(declared)
    corner = declared["spans"][0]
    assert corner["entry_command"] == 1
    assert corner["exit_command"] == 3
    assert corner["commands"] == corner["curve_commands"] == [2, 3]
    assert corner["anchor_commands"] == [1, 2, 3]
    assert declared["commands"]["corner"] == 2
    assert declared["corner_counts"]["total"] == 1
    result, diagnostics = normalize_corners_path_d(d, max_error=1, diagnostics=declared)
    before, after = _parse_subpaths(d)[0], _parse_subpaths(result)[0]
    assert after[1] == before[1]
    assert after[3:] == before[4:]
    assert after[2][1][-2:] == [14, 4]
    assert diagnostics["spans"][0]["commands"] == [2]
    assert diagnostics["path"] == result
    assert diagnostics["corner_counts"]["total"] == 1
    again, rebound = normalize_corners_path_d(result, max_error=1, diagnostics=diagnostics)
    assert again == result
    assert rebound["spans"] == diagnostics["spans"]
    assert declared == snapshot


def test_noop_preserves_point_and_rounded_provenance_with_unknown_radius():
    d = "M0 0 L10 0 Q14 0 14 4 L14 12 L0 12 Z"
    declared = path_corner_diagnostics(d, declared_corners=((0, 0),),
        declared_corner_spans=(((10, 0), (14, 4)),))
    declared["spans"][1]["radius"] = None
    result, diagnostics = normalize_corners_path_d(d, max_error=0.1, diagnostics=declared)
    assert result == d
    assert diagnostics["spans"] == declared["spans"]
    assert diagnostics["corner_counts"] == {"total": 2, "point": 1, "curve": 1}
    assert diagnostics["commands"]["corner"] == 1  # the point owns no edge


def test_wrapped_corner_uses_directed_order_and_preserves_other_sides():
    d = "M2 0 Q4 0 4 2 L4 10 L-10 10 L-10 -2 L0 -2 Q0 0 2 0 Z"
    declared = path_corner_diagnostics(d, declared_corners=((-10, 10),),
        declared_corner_spans=(((0, -2), (4, 2)),))
    rounded = declared["spans"][1]
    assert rounded["commands"] == [6, 1]
    result, diagnostics = normalize_corners_path_d(d, max_error=2, diagnostics=declared)
    tokens = _parse_subpaths(result)[0]
    assert tokens[0] == ("M", [0, -2])
    assert tokens[1][0] == "Q" and tokens[1][1][-2:] == [4, 2]
    assert tokens[2:6] == _parse_subpaths(d)[0][2:6]
    assert diagnostics["corner_counts"]["total"] == 2
    assert {s["corner_id"] for s in diagnostics["spans"]} == {s["corner_id"] for s in declared["spans"]}
    assert normalize_corners_path_d(result, max_error=2, diagnostics=diagnostics)[0] == result


def test_touching_corners_share_anchor_but_not_owned_commands():
    d = "M0 0 Q2 0 3 1 Q4 2 4 4 Q4 6 5 7 Q6 8 8 8 L0 8 Z"
    declared = path_corner_diagnostics(d, declared_corner_spans=(((0, 0), (4, 4)), ((4, 4), (8, 8))))
    assert [s["commands"] for s in declared["spans"]] == [[1, 2], [3, 4]]
    shared = next(a for a in declared["anchors"] if a["command_index"] == 2)
    assert len(shared["corner_ids"]) == 2
    result, diagnostics = normalize_corners_path_d(d, max_error=1, diagnostics=declared)
    assert len(diagnostics["normalized"]) == 2
    assert [s["commands"] for s in diagnostics["spans"]] == [[1], [2]]
    assert _parse_subpaths(result)[0][-2] == ("L", [0, 8])


def test_overlapping_corner_ownership_is_skipped():
    d = "M0 0 Q2 0 3 1 Q4 2 4 4 Q4 6 5 7 L0 8 Z"
    declared = path_corner_diagnostics(d, declared_corner_spans=(((0, 0), (4, 4)), ((3, 1), (5, 7))))
    result, diagnostics = normalize_corners_path_d(d, max_error=10, diagnostics=declared)
    assert result == d
    assert {s["reason"] for s in diagnostics["skipped"]} == {"overlapping_corner_ownership"}


def test_unmatched_or_stale_provenance_does_not_infer_corners():
    d = "M0 0 L10 0 Q14 0 14 4 L0 10 Z"
    assert normalize_corners_path_d(d, max_error=1)[1]["spans"] == []
    declared = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (14, 4)),))
    changed = d.replace("14 4", "15 4")
    assert normalize_corners_path_d(changed, max_error=1, diagnostics=declared)[0] == changed
    rebound = rebind_prefit_corner_diagnostics(changed, declared, tolerance=0.02)
    assert rebound["spans"] == []
    assert rebound["skipped"][0]["reason"] == "missing_or_ambiguous_endpoint"


def test_rebind_is_subpath_scoped_and_preserves_corner_ids():
    d = "M0 0 L10 0 Q14 0 14 4 L0 10 Z M0 0 L10 0 Q14 0 14 4 L0 10 Z"
    declared = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (14, 4)),))
    declared["spans"] = [declared["spans"][1]]
    result = rebind_prefit_corner_diagnostics(d, declared, tolerance=0.02)
    assert len(result["spans"]) == 1
    assert result["spans"][0]["corner_id"] == "p1.rounded_corner0"


def test_ambiguous_repeated_endpoint_is_not_guessed():
    d = "M0 0 Q2 0 4 4 Q2 4 0 0 Q2 0 4 4 L0 10 Z"
    declared = path_corner_diagnostics(d, declared_corner_spans=(((0, 0), (4, 4)),))
    assert declared["spans"] == []
    assert declared["skipped"][0]["reason"] == "missing_or_ambiguous_endpoint"


def test_normalization_preserves_high_precision_side_and_boundary_coordinates():
    d = "M0 0 L10.001 0 Q12.001 0 13.001 1 Q14.001 2 14.001 4 C18.003 7 20.004 9 20.005 12 L0 12 Z"
    declared = path_corner_diagnostics(d, declared_corner_spans=(((10.001, 0), (14.001, 4)),))
    result, _ = normalize_corners_path_d(d, max_error=1, diagnostics=declared)
    before, after = _parse_subpaths(d)[0], _parse_subpaths(result)[0]
    assert after[1] == before[1]
    assert after[2][1][-2:] == [14.001, 4]
    assert after[3:] == before[4:]


def test_point_at_serialization_start_cannot_be_consumed_by_wrapped_corner():
    d = "M2 0 Q4 0 4 2 L4 10 L-10 10 L-10 -2 L0 -2 Q0 0 2 0 Z"
    declared = path_corner_diagnostics(d, declared_corners=((2, 0),),
        declared_corner_spans=(((0, -2), (4, 2)),))
    result, diagnostics = normalize_corners_path_d(d, max_error=2, diagnostics=declared)
    assert result == d
    assert diagnostics["skipped"][0]["reason"] == "protected_point_inside_span"


def test_optimizer_pass_preserves_provenance_across_repeated_calls():
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.passes.corners import corner_normalize_pass
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon

    d = "M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z"
    diagnostics = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (14, 4)),))
    shape = Shape("path", {"d": d})
    obj = VectorRegion(id=1, current=shape, original=shape, footprint=to_polygon(shape),
        fill=FlatFill("#112233"), z=0, diagnostics={"corners": diagnostics})
    first = corner_normalize_pass([obj], {}, max_error=1, enabled=True)[0].new_objects[0]
    second = corner_normalize_pass([first], {}, max_error=1, enabled=True)[0].new_objects[0]
    assert first.current == second.current
    assert first.diagnostics["corners"]["spans"] == second.diagnostics["corners"]["spans"]
    assert first.diagnostics["corners"]["path"] == first.current.params["d"]


def test_no_provenance_skip_survives_repeated_noop():
    d = "M0 0 Q5 0 10 10 L0 10 Z"
    first_d, first = normalize_corners_path_d(d, max_error=1)
    second_d, second = normalize_corners_path_d(first_d, max_error=1, diagnostics=first)
    assert second_d == first_d == d
    assert second == first


def test_rebind_without_declarations_preserves_cleanup_diagnostics():
    from vectormark.candidate import FlatFill
    from vectormark.drawing_refine import _rebind_corner_provenance
    from vectormark.fit import Shape
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon

    d = "M0 0 Q5 0 10 10 L0 10 Z"
    shape = Shape("path", {"d": d})
    before = VectorRegion(1, shape, FlatFill("#112233"), 0,
        footprint=to_polygon(shape), diagnostics={"corners": path_corner_diagnostics(d)})
    cleanup = {"path": d, "spans": (), "skipped": ({"reason": "missing_or_stale_provenance"},)}
    after = before.with_diagnostics({"corners": cleanup})
    result = _rebind_corner_provenance((before,), (after,), tolerance=1)
    assert result[0] is after

import math

import numpy as np
import pytest

import vectormark.fit as fit
import vectormark.python_path_fitter as path_fitter
from vectormark.corner_ownership import directed_corner_ownership
from vectormark.drawing_engine import FitPathRequest, PathFitOptions
from vectormark.path_geometry import (
    ClosedPath,
    CornerAnchor,
    InflectionAnchor,
    LineCommand,
    MoveCommand,
    PathGeometry,
)
from vectormark.python_path_fitter import PythonPathFitter


def _geometry(points, anchors=()):
    commands = [MoveCommand("source.c0", tuple(points[0]))]
    commands.extend(
        LineCommand(f"source.c{index}", tuple(point))
        for index, point in enumerate(points[1:], 1)
    )
    commands.append(LineCommand(f"source.c{len(points)}", tuple(points[0])))
    return PathGeometry(ClosedPath("source", tuple(commands)), tuple(anchors))


def _request(geometry, strategy="quadratic", epsilon=0.25, max_error=0.5):
    return FitPathRequest(geometry, PathFitOptions(epsilon, max_error, strategy))


def _boundary(result, feature_id, role):
    return next(
        boundary
        for boundary in result.correspondence.boundaries
        if boundary.feature_id == feature_id and boundary.role == role
    )


def test_point_corner_is_exact_and_has_typed_correspondence(monkeypatch):
    points = [(0.123456789, 0), (8, 0), (12.987654321, 4), (8, 10), (0, 10)]
    anchor = CornerAnchor("apex", "source.c2", "source.c2")
    source = _geometry(points, (anchor,))
    monkeypatch.setattr(fit, "corner_indices", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("explicit fitting must not invoke corner detection")
    ))

    result = PythonPathFitter().fit_path(_request(source))

    boundary = _boundary(result, "apex", "point")
    assert result.geometry.endpoint(boundary.output_command_id) == points[2]
    assert result.geometry.path.start == points[2]
    assert result.geometry.anchors[0].id == "apex"
    assert result.geometry.anchors[0].entry_command_id == boundary.output_command_id
    assert all(span.source_command_ids and span.output_command_ids for span in result.correspondence.spans)


def test_rounded_corner_owns_exact_declared_run_and_not_adjacent_sides():
    points = [(0, 0), (10.000000001, 0), (12, 0.5), (14.000000001, 4), (14, 12), (0, 12)]
    anchor = CornerAnchor(
        "rounded",
        "source.c1",
        "source.c3",
        ("source.c2", "source.c3"),
        3.0,
    )
    result = PythonPathFitter().fit_path(_request(_geometry(points, (anchor,))))
    fitted = result.geometry.anchors[0]

    assert result.geometry.endpoint(fitted.entry_command_id) == points[1]
    assert result.geometry.endpoint(fitted.exit_command_id) == points[3]
    assert fitted.id == anchor.id and fitted.radius == anchor.radius
    expected = directed_corner_ownership(
        tuple(command.id for command in result.geometry.path.commands),
        fitted.entry_command_id,
        fitted.exit_command_id,
        explicit_closing_drawable=True,
    )
    assert fitted.owned_command_ids == expected
    corner_span = next(span for span in result.correspondence.spans if span.kind == "corner")
    assert corner_span.source_command_ids == anchor.owned_command_ids
    assert corner_span.output_command_ids == fitted.owned_command_ids
    assert any("source.c4" in span.source_command_ids for span in result.correspondence.spans if span.kind == "side")


def test_progressive_raw_line_cuts_cannot_split_declared_rounded_span(monkeypatch):
    points = [(0, 0), (10, 0), (11, 0.25), (13, 1.5), (14, 4), (14, 80), (0, 80)]
    anchor = CornerAnchor(
        "rounded",
        "source.c1",
        "source.c4",
        ("source.c2", "source.c3", "source.c4"),
    )
    monkeypatch.setattr(path_fitter, "_protected_line_spans", lambda _ring: [(2, 3), (4, 6)])

    result = PythonPathFitter().fit_path(_request(
        _geometry(points, (anchor,)), strategy="progressive_allow_lines",
    ))

    corner = next(span for span in result.correspondence.spans if span.kind == "corner")
    assert corner.source_command_ids == anchor.owned_command_ids
    assert not any(
        span.source_command_ids in {("source.c3",), ("source.c4",)}
        for span in result.correspondence.spans
    )


def test_wrapped_corner_preserves_move_and_explicit_close_alias_bindings():
    points = [(2, 0), (4, 2), (4, 10), (-10, 10), (-10, -2), (0, -2)]
    anchor = CornerAnchor(
        "wrapped",
        "source.c4",
        "source.c1",
        ("source.c5", "source.c6", "source.c1"),
    )
    result = PythonPathFitter().fit_path(_request(_geometry(points, (anchor,))))
    fitted = result.geometry.anchors[0]

    assert fitted.owned_command_ids == next(
        span.output_command_ids for span in result.correspondence.spans if span.kind == "corner"
    )
    assert result.geometry.path.start == points[1]
    assert result.geometry.path.commands[-1].end == result.geometry.path.start
    assert result.geometry.endpoint(_boundary(result, "wrapped", "entry").output_command_id) == points[4]
    assert result.geometry.endpoint(_boundary(result, "wrapped", "exit").output_command_id) == points[1]


def test_move_and_explicit_close_alias_bind_to_distinct_output_commands():
    points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    source = _geometry(points, (
        CornerAnchor("move-point", "source.c0", "source.c0"),
        CornerAnchor("close-point", "source.c4", "source.c4"),
    ))
    result = PythonPathFitter().fit_path(_request(source))
    move = _boundary(result, "move-point", "point")
    close = _boundary(result, "close-point", "point")

    assert move.output_command_id == result.geometry.path.commands[0].id
    assert close.output_command_id == result.geometry.path.commands[-1].id
    assert move.output_command_id != close.output_command_id
    assert result.geometry.endpoint(move.output_command_id) == points[0]
    assert result.geometry.endpoint(close.output_command_id) == points[0]


def test_empty_declarations_are_authoritative_and_repeated_fit_is_deterministic(monkeypatch):
    theta = np.linspace(0, 2 * np.pi, 17, endpoint=False)
    source = _geometry(list(zip(10 + 7 * np.cos(theta), 12 + 4 * np.sin(theta), strict=True)))
    monkeypatch.setattr(fit, "corner_indices", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("explicit fitting must not invoke corner detection")
    ))
    fitter = PythonPathFitter()

    first = fitter.fit_path(_request(source))
    second = fitter.fit_path(_request(source))

    assert first == second
    assert first.geometry.anchors == ()
    assert first.correspondence.boundaries == ()
    assert first.correspondence.spans
    assert source == _geometry(list(zip(10 + 7 * np.cos(theta), 12 + 4 * np.sin(theta), strict=True)))


@pytest.mark.parametrize("strategy", ["quadratic", "progressive", "progressive_allow_lines"])
def test_all_path_fit_strategies_are_supported(strategy):
    points = [(0, 0), (80, 0), (100, 20), (100, 100), (0, 100)]
    result = PythonPathFitter().fit_path(_request(_geometry(points), strategy=strategy))
    assert result.diagnostics["strategy"] == strategy
    assert result.correspondence.spans


@pytest.mark.parametrize("strategy", ["quadratic", "progressive", "progressive_allow_lines"])
def test_explicit_points_match_equivalent_automatic_fit(strategy):
    points = np.asarray([(0, 0), (80, 0), (80, 60), (0, 60)], dtype=float)
    contour = np.vstack((points, points[0]))
    automatic, corner_points, corner_spans = fit._fit_path_d_with_corners(
        contour,
        epsilon=0.25,
        max_error=0.5,
        progressive=strategy != "quadratic",
        progressive_allow_lines=strategy == "progressive_allow_lines",
    )
    assert set(corner_points) == set(map(tuple, points))
    assert corner_spans == ()
    anchors = tuple(
        CornerAnchor(f"corner-{index}", f"source.c{index}", f"source.c{index}")
        for index in range(len(points))
    )
    explicit = PythonPathFitter().fit_path(_request(_geometry(points, anchors), strategy=strategy))
    assert explicit.geometry.to_svg_d() == automatic.replace(" ", " ").strip()


def test_replacing_and_removing_declaration_refits_original_evidence():
    points = [(0, 0), (8, 0), (12, 1), (15, 5), (13, 10), (5, 13), (0, 9)]
    original = _geometry(points)
    first = PythonPathFitter().fit_path(_request(_geometry(
        points, (CornerAnchor("old", "source.c2", "source.c2"),),
    ), max_error=2.0))
    replaced = PythonPathFitter().fit_path(_request(_geometry(
        points, (CornerAnchor("new", "source.c4", "source.c4"),),
    ), max_error=2.0))
    removed = PythonPathFitter().fit_path(_request(original, max_error=2.0))
    independent = PythonPathFitter().fit_path(_request(_geometry(points), max_error=2.0))

    assert first.geometry != replaced.geometry
    assert first.geometry.path != removed.geometry.path
    assert {anchor.id for anchor in replaced.geometry.anchors} == {"new"}
    assert removed == independent
    assert removed.geometry.anchors == ()
    assert first.geometry.anchors[0].id == "old"


def test_invalid_options_source_and_annotations_fail_explicitly():
    with pytest.raises(ValueError, match="positive"):
        PathFitOptions(math.nan, 1)
    with pytest.raises(ValueError, match="positive"):
        PathFitOptions(1, math.inf)

    points = [(0, 0), (10, 0), (10, 10), (0, 10)]
    implicit = PathGeometry(ClosedPath("implicit", (
        MoveCommand("c0", points[0]),
        LineCommand("c1", points[1]),
        LineCommand("c2", points[2]),
        LineCommand("c3", points[3]),
    )))
    with pytest.raises(ValueError, match="addressable closing"):
        PythonPathFitter().fit_path(_request(implicit))

    annotated = _geometry(points)
    annotated = PathGeometry(
        annotated.path,
        (InflectionAnchor("unsupported", ("source.c1",)),),
    )
    with pytest.raises(ValueError, match="non-corner"):
        PythonPathFitter().fit_path(_request(annotated))

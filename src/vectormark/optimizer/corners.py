"""Corner provenance and conservative corner-local normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .._fitcurve import fit_quadratic_once
from ..fit import Shape, _fmt, _quadratic_max_residual
from .vector_region import _parse_subpaths, _sample_subpath

_CURVE_COMMANDS = {"Q", "C"}


def _segments(tokens: list[tuple[str, list[float]]]) -> list[dict[str, object]]:
    cursor: np.ndarray | None = None
    subpath_start: np.ndarray | None = None
    segments: list[dict[str, object]] = []
    for command_index, (command, values) in enumerate(tokens):
        if command == "M":
            cursor = np.asarray(values[:2], dtype=float)
            subpath_start = cursor.copy()
            continue
        if cursor is None:
            continue
        start = cursor.copy()
        if command == "Z":
            end = subpath_start.copy() if subpath_start is not None else start.copy()
            start_tangent = end - start
            end_tangent = start_tangent
        elif command == "L":
            end = np.asarray(values[:2], dtype=float)
            start_tangent = end - start
            end_tangent = start_tangent
        elif command == "Q":
            control = np.asarray(values[:2], dtype=float)
            end = np.asarray(values[2:4], dtype=float)
            start_tangent = control - start
            end_tangent = end - control
        elif command == "C":
            first = np.asarray(values[:2], dtype=float)
            second = np.asarray(values[2:4], dtype=float)
            end = np.asarray(values[4:6], dtype=float)
            start_tangent = first - start
            end_tangent = end - second
        else:
            continue
        segments.append(
            {
                "command_index": command_index,
                "command": command,
                "start": start,
                "end": end,
                "start_tangent": start_tangent,
                "end_tangent": end_tangent,
            }
        )
        cursor = end
    return segments


def path_corner_diagnostics(
    d: str,
    *,
    declared_corners: Sequence[tuple[float, float]] = (),
    declared_corner_spans: Sequence[tuple[tuple[float, float], tuple[float, float]]] = (),
    corner_tolerance: float = 0.02,
) -> dict[str, object]:
    """Classify command-end anchors that are part of point or curved corners.

    The result is intentionally command-indexed: an SVG command owns the
    anchor at its endpoint, so there is no parallel anchor-address namespace.
    Curved-corner members share a ``corner_id`` and include both boundary
    anchors plus every curve command in the detected run.
    """
    anchors: dict[tuple[int, int], dict[str, object]] = {}
    spans: list[dict[str, object]] = []
    corner_number = 0
    total_commands = 0

    def mark(subpath: int, command_index: int, corner_id: str) -> None:
        anchors[(subpath, command_index)] = {
            "subpath": subpath,
            "command_index": command_index,
            "anchor_kind": "corner",
            "corner_id": corner_id,
        }

    # The path fitter's selected contour corners are authoritative.  An
    # unannotated path has no normalizable corners: do not invent a different
    # set by inspecting its fitted tangents.
    declared = tuple((float(x), float(y)) for x, y in declared_corners)
    declared_spans = tuple(
        ((float(start[0]), float(start[1])), (float(end[0]), float(end[1])))
        for start, end in declared_corner_spans
    )
    for subpath_index, tokens in enumerate(_parse_subpaths(d)):
        total_commands += sum(command not in {"M", "Z"} for command, _values in tokens)
        segments = _segments(tokens)
        if not segments:
            continue
        endpoints: list[tuple[int, np.ndarray]] = []
        if tokens and tokens[0][0] == "M":
            endpoints.append((0, np.asarray(tokens[0][1][:2], dtype=float)))
        endpoints.extend(
            (int(segment["command_index"]), np.asarray(segment["end"], dtype=float))
            for segment in segments
            if segment["command"] != "Z"
        )
        for number, point in enumerate(declared):
            matching = [
                (command_index, endpoint)
                for command_index, endpoint in endpoints
                if float(np.linalg.norm(endpoint - point)) <= corner_tolerance
            ]
            if not matching:
                continue
            command_index, _endpoint = min(matching, key=lambda item: item[0])
            corner_id = f"p{subpath_index}.corner{number}"
            corner_number += 1
            mark(subpath_index, command_index, corner_id)
            spans.append({
                "corner_id": corner_id,
                "kind": "point",
                "subpath": subpath_index,
                "commands": [command_index],
                "source": "prefit",
                "point": point,
            })
        command_by_index = {int(segment["command_index"]): segment for segment in segments}
        endpoint_index = {
            index: endpoint
            for index, endpoint in endpoints
        }
        drawable_indices = [
            index for index, (command, _values) in enumerate(tokens)
            if command != "Z"
        ]
        for number, (start_point, end_point) in enumerate(declared_spans):
            starts = [
                index for index, endpoint in endpoint_index.items()
                if float(np.linalg.norm(endpoint - start_point)) <= corner_tolerance
            ]
            ends = [
                index for index, endpoint in endpoint_index.items()
                if float(np.linalg.norm(endpoint - end_point)) <= corner_tolerance
            ]
            if not starts or not ends:
                continue
            start_index, end_index = min(starts), min(ends)
            start_position = drawable_indices.index(start_index)
            end_position = drawable_indices.index(end_index)
            if end_position < start_position:
                member_indices = drawable_indices[start_position:] + drawable_indices[:end_position + 1]
            else:
                member_indices = drawable_indices[start_position:end_position + 1]
            # A rounded corner is bounded by *two* protected straight sides.
            # The endpoint command owns the curve arriving at the exit anchor;
            # include the following side command as well so a stem-to-band
            # transition retains both of its declared boundaries.  Only Q/C
            # members are refitted by corner normalization below.
            exit_position = (end_position + 1) % len(drawable_indices)
            exit_index = drawable_indices[exit_position]
            if exit_index not in member_indices:
                member_indices.append(exit_index)
            curve_commands = [
                index for index in member_indices
                if index in command_by_index and command_by_index[index]["command"] in _CURVE_COMMANDS
            ]
            if not curve_commands:
                continue
            corner_id = f"p{subpath_index}.rounded_corner{number}"
            for command_index in member_indices:
                mark(subpath_index, command_index, corner_id)
            spans.append({
                "corner_id": corner_id,
                "kind": "curve",
                "subpath": subpath_index,
                "commands": member_indices,
                "curve_commands": curve_commands,
                "source": "prefit",
                "start_point": start_point,
                "end_point": end_point,
            })

    ordered_anchors = [anchors[key] for key in sorted(anchors)]
    corner_commands = {
        (int(anchor["subpath"]), int(anchor["command_index"]))
        for anchor in ordered_anchors
        if int(anchor["command_index"]) != 0
    }
    return {
        "commands": {
            "total": total_commands,
            "corner": len(corner_commands),
            "free": total_commands - len(corner_commands),
        },
        "anchor_counts": {
            "total": total_commands + sum(1 for tokens in _parse_subpaths(d) if tokens),
            "corner": len(ordered_anchors),
        },
        "anchors": ordered_anchors,
        "spans": spans,
    }


def rebind_prefit_corner_diagnostics(
    d: str,
    diagnostics: Mapping[str, object],
    *,
    tolerance: float,
) -> dict[str, object]:
    """Carry existing pre-fit corner provenance across a path rewrite.

    A rewrite may move an endpoint slightly or expand one command into several.
    It may not create corner semantics.  Rebind only the explicitly recorded
    points/spans to their nearest retained endpoints in the rewritten path.
    """
    spans = diagnostics.get("spans", ())
    if not isinstance(spans, Sequence):
        return {"path": d}
    points = [
        tuple(float(value) for value in span["point"])
        for span in spans
        if isinstance(span, Mapping) and span.get("source") == "prefit" and "point" in span
    ]
    rounded_spans = [
        (
            tuple(float(value) for value in span["start_point"]),
            tuple(float(value) for value in span["end_point"]),
        )
        for span in spans
        if (
            isinstance(span, Mapping)
            and span.get("source") == "prefit"
            and "start_point" in span
            and "end_point" in span
        )
    ]
    rebound = path_corner_diagnostics(
        d,
        declared_corners=points,
        declared_corner_spans=rounded_spans,
        corner_tolerance=tolerance,
    )
    rebound["path"] = d
    return rebound


def _subpath_d(tokens: list[tuple[str, list[float]]]) -> str:
    return " ".join(
        command if not values else f"{command}{' '.join(_fmt(value) for value in values)}"
        for command, values in tokens
    )


def normalize_corners_path_d(
    d: str,
    *,
    max_error: float,
    diagnostics: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Collapse recognised rounded-corner runs to one Q when the fit is faithful."""
    diagnostics = dict(diagnostics) if diagnostics is not None else path_corner_diagnostics(d)
    tokens_by_subpath = _parse_subpaths(d)
    normalized_spans: list[str] = []
    curve_spans = sorted(
        (span for span in diagnostics["spans"] if span["kind"] == "curve"),
        key=lambda span: (int(span["subpath"]), max(int(command) for command in span["curve_commands"])),
        reverse=True,
    )
    for span in curve_spans:
        subpath_index = int(span["subpath"])
        curve_commands = [int(command) for command in span["curve_commands"]]
        tokens = tokens_by_subpath[subpath_index]
        start, end = min(curve_commands), max(curve_commands)
        if not all(tokens[index][0] in _CURVE_COMMANDS for index in range(start, end + 1)):
            continue
        first_start = _segments(tokens)
        segment_by_command = {int(segment["command_index"]): segment for segment in first_start}
        source = [
            ("M", [float(value) for value in segment_by_command[start]["start"]]),
            *tokens[start:end + 1],
        ]
        samples = np.asarray(_sample_subpath(source, 16), dtype=float)
        if len(samples) < 3:
            continue
        quadratic, _squared_error, _split = fit_quadratic_once(samples, max_error)
        if _quadratic_max_residual(samples, quadratic) > max_error:
            continue
        end_point = quadratic[2]
        tokens[start:end + 1] = [
            ("Q", [float(quadratic[1][0]), float(quadratic[1][1]), float(end_point[0]), float(end_point[1])])
        ]
        normalized_spans.append(str(span["corner_id"]))

    normalized_d = " ".join(_subpath_d(tokens) for tokens in tokens_by_subpath)
    result = path_corner_diagnostics(normalized_d)
    result["normalized"] = normalized_spans
    return normalized_d, result


def anchor_annotations(diagnostics: Mapping[str, object], subpath: int) -> dict[int, dict[str, object]]:
    """Return trace-command annotations for one subpath."""
    anchors = diagnostics.get("anchors", ())
    if not isinstance(anchors, list):
        return {}
    return {
        int(anchor["command_index"]): dict(anchor)
        for anchor in anchors
        if isinstance(anchor, Mapping) and int(anchor.get("subpath", -1)) == subpath
    }

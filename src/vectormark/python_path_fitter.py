"""Concrete explicit-declaration path fitting for the Python engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .corner_ownership import directed_corner_ownership
from .drawing_engine import (
    FitBoundaryCorrespondence,
    FitPathCorrespondence,
    FitPathRequest,
    FitPathResult,
    FitSpanCorrespondence,
    PathFitOptions,
)
from .fit import _fit_path_d_from_cuts, _inside_closed_span, _protected_line_spans
from .path_fit_evidence import FitEndpointCorrespondence
from .path_geometry import (
    ClosedPath,
    CornerAnchor,
    LineCommand,
    PathGeometry,
    geometry_from_svg_d,
)

_STRATEGIES = {"quadratic", "progressive", "progressive_allow_lines"}


def validate_path_fit_options(options: PathFitOptions) -> None:
    """Validate the runtime strategy carried by otherwise typed fit options."""
    if options.strategy not in _STRATEGIES:
        raise ValueError(f"unsupported path-fit strategy: {options.strategy!r}")


def validate_explicit_fit_source(source: PathGeometry) -> None:
    """Validate one immutable line source and its complete corner declarations."""
    if any(not isinstance(command, LineCommand) for command in source.path.drawable_commands):
        raise ValueError("explicit path fitting requires source LineCommands")
    if source.path.drawable_commands[-1].end != source.path.start:
        raise ValueError(
            "explicit path fitting requires an addressable closing LineCommand"
        )
    if any(not isinstance(anchor, CornerAnchor) for anchor in source.anchors):
        raise ValueError("explicit path fitting does not support non-corner anchors")

    source_commands = source.path.commands
    ring = np.asarray([command.end for command in source_commands[:-1]], dtype=float)
    if len(ring) < 3:
        raise ValueError("explicit path fitting requires at least three source vertices")
    edges = np.roll(ring, -1, axis=0) - ring
    if np.any(np.linalg.norm(edges, axis=1) == 0):
        raise ValueError("explicit path fitting rejects zero-length source lines")

    command_ids = tuple(command.id for command in source_commands)
    node_by_command = {
        command.id: index
        for index, command in enumerate(source_commands[:-1])
    }
    node_by_command[source_commands[-1].id] = 0
    rounded_ranges: list[tuple[int, int]] = []
    point_nodes: set[int] = set()
    for anchor in source.anchors:
        expected = directed_corner_ownership(
            command_ids,
            anchor.entry_command_id,
            anchor.exit_command_id,
            explicit_closing_drawable=True,
        )
        if expected != anchor.owned_command_ids:
            raise ValueError("corner ownership does not match its directed boundaries")
        start = node_by_command[anchor.entry_command_id]
        end = node_by_command[anchor.exit_command_id]
        if start == end:
            point_nodes.add(start)
        else:
            rounded_ranges.append((start, end))

    for point in point_nodes:
        if any(_inside_closed_span(point, start, end) for start, end in rounded_ranges):
            raise ValueError("point corner cannot lie inside a rounded corner span")


class PythonPathFitter:
    """Fit one retained source polyline using only its explicit declarations."""

    def fit_path(self, request: FitPathRequest) -> FitPathResult:
        source = request.source
        strategy = request.options.strategy
        validate_path_fit_options(request.options)
        validate_explicit_fit_source(source)

        source_commands = source.path.commands
        ring = np.asarray([command.end for command in source_commands[:-1]], dtype=float)

        command_ids = tuple(command.id for command in source_commands)
        node_by_command = {
            command.id: index
            for index, command in enumerate(source_commands[:-1])
        }
        node_by_command[source_commands[-1].id] = 0
        if not set(request.excluded_command_ids).issubset(node_by_command):
            raise ValueError("excluded fitting commands must belong to the retained source")
        excluded_indices = frozenset(
            node_by_command[command_id] for command_id in request.excluded_command_ids
        )

        cuts: set[int] = set()
        rounded_ranges: list[tuple[int, int]] = []
        rounded_by_range: dict[tuple[int, int], CornerAnchor] = {}
        for anchor in source.anchors:
            start = node_by_command[anchor.entry_command_id]
            end = node_by_command[anchor.exit_command_id]
            cuts.update((start, end))
            if start != end:
                rounded_ranges.append((start, end))
                rounded_by_range[(start, end)] = anchor

        progressive = strategy != "quadratic"
        if progressive:
            for start, end in _protected_line_spans(ring):
                cuts.update(index for index in (start, end) if index not in excluded_indices)
        for start, end in rounded_ranges:
            cuts = {
                index
                for index in cuts
                if index in {start, end} or not _inside_closed_span(index, start, end)
            }
            cuts.update((start, end))

        command_end_indices: list[int | None] = []
        d, runs = _fit_path_d_from_cuts(
            ring,
            sorted(cuts),
            epsilon=request.options.epsilon,
            max_error=request.options.max_error,
            progressive=progressive,
            progressive_allow_lines=strategy == "progressive_allow_lines",
            excluded_indices=excluded_indices,
            command_end_indices=command_end_indices,
        )
        parsed = geometry_from_svg_d(d, region_id=f"{source.path.id}.fit")
        if not isinstance(parsed, PathGeometry):
            raise RuntimeError("single-path fitter emitted multiple paths")

        output_commands = [
            replace(command, id=f"{source.path.id}.fit.c{index}")
            for index, command in enumerate(parsed.path.commands)
        ]
        output_commands[0] = replace(output_commands[0], end=tuple(ring[runs[0][0]]))
        run_output_ids: list[tuple[str, ...]] = []
        cursor = 1
        ending_output_id: dict[int, str] = {}
        for _start, end, count in runs:
            if count <= 0:
                raise ValueError("fitted source span emitted no output commands")
            span_commands = output_commands[cursor:cursor + count]
            if len(span_commands) != count:
                raise RuntimeError("fitted-span command accounting is inconsistent")
            span_commands[-1] = replace(span_commands[-1], end=tuple(ring[end]))
            output_commands[cursor:cursor + count] = span_commands
            ids = tuple(command.id for command in span_commands)
            run_output_ids.append(ids)
            ending_output_id[end] = ids[-1]
            cursor += count
        if cursor != len(output_commands):
            raise RuntimeError("unbound fitted output commands remain")
        if len(command_end_indices) != len(output_commands) - 1:
            raise RuntimeError("fitted endpoint accounting is inconsistent")
        endpoint_bindings = []
        for index, node in enumerate(command_end_indices, start=1):
            if node is None:
                continue
            output_commands[index] = replace(output_commands[index], end=tuple(ring[node]))
            endpoint_bindings.append(FitEndpointCorrespondence(
                source_commands[node or len(ring)].id, output_commands[index].id,
            ))

        fitted_path = ClosedPath(source.path.id, tuple(output_commands))

        def output_boundary_id(source_command_id: str) -> str:
            if source_command_id == source_commands[0].id:
                if runs[0][0] != 0:
                    raise ValueError("declared source move has no fitted output binding")
                return fitted_path.commands[0].id
            if source_command_id == source_commands[-1].id:
                if runs[-1][1] != 0:
                    raise ValueError("declared source close has no fitted output binding")
                return fitted_path.commands[-1].id
            try:
                return ending_output_id[node_by_command[source_command_id]]
            except KeyError as error:
                raise ValueError("declared boundary has no fitted output binding") from error

        fitted_anchors: list[CornerAnchor] = []
        boundaries: list[FitBoundaryCorrespondence] = []
        for anchor in source.anchors:
            entry_output = output_boundary_id(anchor.entry_command_id)
            exit_output = output_boundary_id(anchor.exit_command_id)
            if anchor.entry_command_id == anchor.exit_command_id:
                boundaries.append(FitBoundaryCorrespondence(
                    anchor.id, "point", anchor.entry_command_id, entry_output,
                ))
                fitted_anchors.append(CornerAnchor(
                    anchor.id, entry_output, entry_output, radius=anchor.radius,
                ))
                continue
            boundaries.extend((
                FitBoundaryCorrespondence(
                    anchor.id, "entry", anchor.entry_command_id, entry_output,
                ),
                FitBoundaryCorrespondence(
                    anchor.id, "exit", anchor.exit_command_id, exit_output,
                ),
            ))
            owned_output = directed_corner_ownership(
                tuple(command.id for command in fitted_path.commands),
                entry_output,
                exit_output,
                explicit_closing_drawable=True,
            )
            fitted_anchors.append(CornerAnchor(
                anchor.id,
                entry_output,
                exit_output,
                owned_output,
                anchor.radius,
            ))

        def source_span_ids(start: int, end: int) -> tuple[str, ...]:
            ids: list[str] = []
            current = start
            while current != end:
                ids.append(source_commands[current + 1].id)
                current = (current + 1) % len(ring)
            return tuple(ids)

        spans: list[FitSpanCorrespondence] = []
        for (start, end, _count), output_ids in zip(runs, run_output_ids, strict=True):
            corner = rounded_by_range.get((start, end))
            spans.append(FitSpanCorrespondence(
                kind="corner" if corner is not None else "side",
                source_command_ids=source_span_ids(start, end),
                output_command_ids=output_ids,
                feature_id=corner.id if corner is not None else None,
            ))

        geometry = PathGeometry(fitted_path, tuple(fitted_anchors))
        correspondence = FitPathCorrespondence(tuple(boundaries), tuple(spans), tuple(endpoint_bindings))
        return FitPathResult(
            geometry,
            correspondence,
            diagnostics={
                "strategy": strategy,
                "semantic_corner_count": len(source.anchors),
                "fitted_span_count": len(spans),
            },
        )

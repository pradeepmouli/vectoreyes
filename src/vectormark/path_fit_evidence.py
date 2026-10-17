"""Private, immutable source evidence and shared path-fit correspondence.

This module depends only on canonical geometry, never drawing orchestration.
Output addresses describe the recorded fit stage, not a later drawing version.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Literal, NamedTuple
from collections.abc import Sequence

from .corner_ownership import directed_corner_ownership
from .path_geometry import ClosedPath, CornerAnchor, LineCommand, MoveCommand, PathGeometry, Point


@dataclass(frozen=True)
class FitBoundaryCorrespondence:
    """One declared feature boundary retained by an emitted command endpoint."""

    feature_id: str
    role: Literal["point", "entry", "exit"]
    source_command_id: str
    output_command_id: str


@dataclass(frozen=True)
class FitSpanCorrespondence:
    """One directed source run and the commands fitted from it."""

    kind: Literal["corner", "side"]
    source_command_ids: tuple[str, ...]
    output_command_ids: tuple[str, ...]
    feature_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_command_ids", tuple(self.source_command_ids))
        object.__setattr__(self, "output_command_ids", tuple(self.output_command_ids))
        if not self.source_command_ids or not self.output_command_ids:
            raise ValueError("fitted-span correspondence must bind non-empty spans")
        if self.kind == "corner" and not self.feature_id:
            raise ValueError("corner-span correspondence requires a feature ID")
        if self.kind == "side" and self.feature_id is not None:
            raise ValueError("side-span correspondence cannot carry a feature ID")


@dataclass(frozen=True)
class FitEndpointCorrespondence:
    """An emitted endpoint selected from an exact pre-serialization source node."""

    source_command_id: str
    output_command_id: str


@dataclass(frozen=True)
class FitPathCorrespondence:
    """Typed source/output bindings for one explicit path fit."""

    boundaries: tuple[FitBoundaryCorrespondence, ...]
    spans: tuple[FitSpanCorrespondence, ...]
    endpoints: tuple[FitEndpointCorrespondence, ...] = ()

    @property
    def endpoint_bindings(self) -> tuple[FitEndpointCorrespondence, ...]:
        """All exact endpoint receipts, including each directed run's end."""
        return (
            *self.endpoints,
            *(FitEndpointCorrespondence(item.source_command_id, item.output_command_id)
              for item in self.boundaries),
            *(FitEndpointCorrespondence(item.source_command_ids[-1], item.output_command_ids[-1])
              for item in self.spans),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "spans", tuple(self.spans))
        object.__setattr__(self, "endpoints", tuple(self.endpoints))
        if not self.spans:
            raise ValueError("path-fit correspondence must describe every fitted span")


class FittedRun(NamedTuple):
    """Actual directed run emitted by the shared numerical fitting loop."""

    start: int
    end: int
    command_count: int


@dataclass(frozen=True)
class FeatureBindingDiagnostic:
    declaration: CornerAnchor
    reason: str


@dataclass(frozen=True)
class AutomaticFitEvidence:
    revision: str
    source: PathGeometry
    output_path_id: str
    output_digest: str
    correspondence: FitPathCorrespondence
    diagnostics: tuple[FeatureBindingDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def matches_output(self, d: str) -> bool:
        return self.output_digest == sha256(d.encode()).hexdigest()


@dataclass(frozen=True)
class SourceFeatureBinding:
    geometry: PathGeometry
    rounded: tuple[tuple[tuple[int, int], CornerAnchor], ...]
    diagnostics: tuple[FeatureBindingDiagnostic, ...]


def source_node_command_id(path: ClosedPath, index: int) -> str:
    """A source ring's node zero is its addressable closing line, not M."""
    return path.commands[index or -1].id


def bind_source_path_features(
    path: ClosedPath,
    point_indices: tuple[int, ...],
    corner_ranges: tuple[tuple[int, int], ...],
) -> SourceFeatureBinding:
    """Bind detected declarations once, preserving explicit ambiguity diagnostics."""
    command_ids = tuple(command.id for command in path.commands)

    def source_boundary(index: int) -> str:
        return source_node_command_id(path, index)

    declarations = [
        CornerAnchor(f"{path.id}.corner{index}", source_boundary(node), source_boundary(node))
        for index, node in enumerate(point_indices)
    ]
    rounded = {}
    for start, end in corner_ranges:
        entry, exit = source_boundary(start), source_boundary(end)
        anchor = CornerAnchor(
            f"{path.id}.corner{len(declarations)}", entry, exit,
            directed_corner_ownership(command_ids, entry, exit, explicit_closing_drawable=True),
        )
        declarations.append(anchor)
        rounded[(start, end)] = anchor

    anchors = []
    diagnostics = []
    for anchor in declarations:
        try:
            PathGeometry(path, (*anchors, anchor))
        except ValueError as error:
            diagnostics.append(FeatureBindingDiagnostic(anchor, f"ambiguous_source_declaration: {error}"))
        else:
            anchors.append(anchor)
    return SourceFeatureBinding(PathGeometry(path, tuple(anchors)), tuple(rounded.items()), tuple(diagnostics))


def automatic_fit_evidence(
    ring: Sequence[Point],
    point_indices: tuple[int, ...],
    corner_ranges: tuple[tuple[int, int], ...],
    runs: tuple[FittedRun, ...],
    d: str,
    *,
    settings: tuple[object, ...],
    command_end_indices: tuple[int | None, ...] = (),
) -> AutomaticFitEvidence:
    """Bind detector indices to exact source lines and actual emitted runs.

    Node zero uses the explicit closing L, never the coincident M alias.
    No fitted coordinates are searched, rounded, or used as source evidence.
    """
    points = tuple((float(point[0]), float(point[1])) for point in ring)
    payload = ("vectormark.automatic-evidence.v1/python-fit.v1", points,
                point_indices, corner_ranges, runs, settings, d, command_end_indices)
    revision = sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    source_id = f"e{revision}.source.p0"
    output_id = f"e{revision}.fit.p0"
    commands = (
        MoveCommand(f"{source_id}.c0", points[0]),
        *(LineCommand(f"{source_id}.c{index}", point)
          for index, point in enumerate((*points[1:], points[0]), start=1)),
    )
    path = ClosedPath(source_id, commands)
    command_ids = tuple(command.id for command in commands)

    def source_boundary(index: int) -> str:
        return source_node_command_id(path, index)

    binding = bind_source_path_features(path, point_indices, corner_ranges)
    anchors = binding.geometry.anchors
    rounded = dict(binding.rounded)
    diagnostics = list(binding.diagnostics)

    cursor = 1
    spans = []
    ending_ids: dict[int, list[str]] = {}
    for start, end, count in runs:
        if count <= 0:
            raise ValueError("fitted source span emitted no output commands")
        output_ids = tuple(f"{output_id}.c{index}" for index in range(cursor, cursor + count))
        cursor += count
        ending_ids.setdefault(end, []).append(output_ids[-1])
        source_ids = directed_corner_ownership(
            command_ids, source_boundary(start), source_boundary(end), explicit_closing_drawable=True,
        )
        anchor = rounded.get((start, end))
        if anchor not in anchors:
            anchor = None
        spans.append(FitSpanCorrespondence(
            "corner" if anchor else "side", source_ids, output_ids, anchor.id if anchor else None,
        ))
    if cursor - 1 != len(re.findall(r"[LQC]", d)):
        raise ValueError("fitted-span command accounting is inconsistent")
    if command_end_indices and len(command_end_indices) != cursor - 1:
        raise ValueError("fitted endpoint accounting is inconsistent")
    endpoints = tuple(
        FitEndpointCorrespondence(source_boundary(node), f"{output_id}.c{index}")
        for index, node in enumerate(command_end_indices, start=1)
        if node is not None
    )

    node_by_id = {source_boundary(index): index for index in range(len(points))}
    boundaries = []
    for anchor in anchors:
        roles = (("point", anchor.entry_command_id),) if anchor.entry_command_id == anchor.exit_command_id else (
            ("entry", anchor.entry_command_id), ("exit", anchor.exit_command_id),
        )
        for role, source_command in roles:
            output_ids = ending_ids.get(node_by_id[source_command], [])
            if len(output_ids) != 1:
                reason = "missing_fitted_boundary" if not output_ids else "ambiguous_fitted_boundary"
                diagnostics.append(FeatureBindingDiagnostic(anchor, f"{reason}: {role}"))
                continue
            boundaries.append(FitBoundaryCorrespondence(anchor.id, role, source_command, output_ids[0]))
    return AutomaticFitEvidence(
        revision, binding.geometry, output_id, sha256(d.encode()).hexdigest(),
        FitPathCorrespondence(tuple(boundaries), tuple(spans), endpoints), tuple(diagnostics),
    )

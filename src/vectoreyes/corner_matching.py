"""Match pre-fit corner evidence to typed source commands, without SVG state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .corner_ownership import directed_corner_ownership
from .immutable_value import freeze_value
from .path_geometry import CubicCommand, MoveCommand, PathCommand, QuadraticCommand


def match_prefit_corner_spans(
    paths: Mapping[int, Sequence[PathCommand]], declarations: Sequence[object], *, tolerance: float,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    """Resolve existing source declarations, never discover replacement corners.

    Source command sequences need not be valid filled contours: matching must
    not impose a new topology or minimum-command requirement on diagnostics.
    Closure aliasing, nearest-distance tolerance and tie rejection retain the
    existing policy. Every typed drawable command contributes its endpoint.
    """
    spans, skipped = [], []
    for declaration in declarations:
        corner_id = declaration.get("corner_id") if isinstance(declaration, Mapping) else None

        def invalid_provenance() -> None:
            skip = {"reason": "invalid_corner_provenance"}
            if isinstance(corner_id, str) and corner_id:
                skip["corner_id"] = corner_id
            skipped.append(skip)

        if not isinstance(declaration, Mapping):
            invalid_provenance()
            continue
        kind, subpath = declaration.get("kind"), declaration.get("subpath")
        if (declaration.get("source") != "prefit" or not isinstance(corner_id, str) or not corner_id
            or not isinstance(subpath, int) or isinstance(subpath, bool) or kind not in {"point", "curve"}):
            invalid_provenance()
            continue
        fields = ("point",) if kind == "point" else ("start_point", "end_point")
        points = {}
        for field in fields:
            value = declaration.get(field)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
                invalid_provenance()
                break
            try:
                point = (float(value[0]), float(value[1]))
            except (TypeError, ValueError):
                invalid_provenance()
                break
            if not all(np.isfinite(coordinate) for coordinate in point):
                invalid_provenance()
                break
            points[field] = point
        if len(points) != len(fields):
            continue
        span = {**declaration, **points}
        if subpath not in paths:
            skipped.append({"corner_id": corner_id, "reason": "missing_subpath"})
            continue
        commands = tuple(paths[subpath])
        if not commands or not isinstance(commands[0], MoveCommand):
            continue
        endpoints = {index: np.asarray(command.end, dtype=float) for index, command in enumerate(commands)}
        explicit_close = len(commands) > 1 and np.linalg.norm(endpoints[len(commands) - 1] - endpoints[0]) <= 1e-9
        if explicit_close:
            del endpoints[len(commands) - 1]

        def locate(point):
            matches = sorted((float(np.linalg.norm(end - np.asarray(point, dtype=float))), index)
                for index, end in endpoints.items())
            if not matches or matches[0][0] > tolerance:
                return None
            if len(matches) > 1 and abs(matches[1][0] - matches[0][0]) <= 1e-9:
                return None
            return matches[0][1]

        reason = None
        owned = []
        if kind == "point":
            entry = exit = locate(points["point"])
        else:
            entry, exit = locate(points["start_point"]), locate(points["end_point"])
            if entry is not None and exit is not None and entry != exit:
                try:
                    ids = tuple(command.id for command in commands)
                    positions = {command.id: index for index, command in enumerate(commands)}
                    owned = [positions[item] for item in directed_corner_ownership(
                        ids, ids[entry], ids[exit], explicit_closing_drawable=explicit_close)]
                except ValueError as error:
                    reason = "implicit_closing_edge" if "implicit closing" in str(error) else "ambiguous_span"
            elif entry == exit and entry is not None:
                reason = "ambiguous_span"
        if entry is None or exit is None:
            reason = "missing_or_ambiguous_endpoint"
        if reason:
            skipped.append({"corner_id": corner_id, "reason": reason})
            continue
        span.update(entry_command=entry, exit_command=exit, commands=owned,
            curve_commands=[index for index in owned if isinstance(commands[index], (QuadraticCommand, CubicCommand))],
            anchor_commands=list(dict.fromkeys([entry, *owned, exit])))
        spans.append(span)
    return tuple(freeze_value(span) for span in spans), tuple(freeze_value(item) for item in skipped)

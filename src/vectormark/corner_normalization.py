"""Corner-local canonical path operation, independent of drawing/paint state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from . import corner_fitting
from .immutable_value import freeze_value
from .path_geometry import CornerAnchor, CubicCommand, PathGeometry, QuadraticCommand, command_token
from .optimizer.vector_region import _sample_subpath


def normalize_path_corners(
    geometry: PathGeometry, *, max_error: float,
) -> tuple[PathGeometry, Mapping[str, object]]:
    """Reduce declared curved corners to one Q without moving boundaries.

    Only exact canonical ownership is consumed. Other feature endpoints remain
    protected, and retained command IDs/coordinates survive unchanged. A wrapped
    replacement rotates M, not the outline. The receipt maps surviving endpoint
    identities; consumed interior endpoints are deliberately not remapped.
    """
    commands = geometry.path.commands
    by_id = {command.id: command for command in commands}
    positions = {command.id: index for index, command in enumerate(commands)}
    replacements = {}
    normalized = []
    skipped = []
    for anchor in geometry.anchors:
        if not isinstance(anchor, CornerAnchor) or not anchor.owned_command_ids:
            continue
        owned = anchor.owned_command_ids
        reason = None
        if any(not isinstance(by_id[item], (QuadraticCommand, CubicCommand)) for item in owned):
            reason = "non_curve_span"
        elif any(
            geometry.path.boundary_command_id(item) in owned[:-1]
            for other in geometry.anchors if other is not anchor
            for item in other.command_ids
        ):
            reason = "protected_anchor_inside_span"
        elif len(owned) == 1 and isinstance(by_id[owned[0]], QuadraticCommand):
            continue
        elif len(commands) - len(owned) + 1 < 3:
            reason = "minimum_path_commands"
        if reason:
            skipped.append({"corner_id": anchor.id, "reason": reason})
            continue
        start = geometry.endpoint(anchor.entry_command_id)
        end = geometry.endpoint(anchor.exit_command_id)
        source = [("M", list(start)), *[
            (kind, list(values)) for item in owned
            for kind, values in (command_token(by_id[item]),)
        ]]
        samples = _sample_subpath(source, 16)
        fitted = corner_fitting.fit_corner_span(samples, start=start, end=end,
            command_id=owned[-1], max_error=max_error)
        if fitted is None:
            skipped.append({"corner_id": anchor.id, "reason": "fit_residual"})
            continue
        replacements[owned[0]] = (anchor, fitted)
        normalized.append(anchor.id)

    result = geometry
    lineage = {command.id: command.id for command in commands}
    if replacements:
        order = list(commands[1:])
        wrap = next((anchor for anchor, _ in replacements.values()
            if any(positions[right] < positions[left]
                for left, right in zip(anchor.owned_command_ids, anchor.owned_command_ids[1:]))), None)
        move = commands[0]
        if wrap is not None:
            offset = next(i for i, command in enumerate(order) if command.id == wrap.owned_command_ids[0])
            order = order[offset:] + order[:offset]
            move = replace(move, end=geometry.endpoint(wrap.entry_command_id))
            lineage.pop(move.id)
        consumed = {item for anchor, _ in replacements.values() for item in anchor.owned_command_ids}
        output = [move]
        for command in order:
            if command.id in replacements:
                output.append(replacements[command.id][1])
            elif command.id not in consumed:
                output.append(command)
        for anchor, fitted in replacements.values():
            for item in anchor.owned_command_ids[:-1]:
                lineage.pop(item)
        anchors = tuple(replace(anchor, owned_command_ids=(anchor.owned_command_ids[-1],))
            if anchor.id in normalized else anchor for anchor in geometry.anchors)
        result = replace(geometry, path=replace(geometry.path, commands=tuple(output)), anchors=anchors)
    return result, freeze_value({"normalized": tuple(normalized), "skipped": tuple(skipped), "commands": lineage})

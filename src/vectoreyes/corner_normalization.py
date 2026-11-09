"""Corner-local canonical path operation, independent of drawing/paint state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from . import corner_fitting
from .immutable_value import freeze_value
from .path_geometry import CornerAnchor, CubicCommand, PathGeometry, QuadraticCommand, command_token
from .path_sampling import _sample_subpath


def normalize_declared_paths(
    paths: tuple[PathGeometry, ...], *, declaration: Mapping[str, object], max_error: float,
    source_text: str | None = None,
) -> tuple[tuple[PathGeometry, ...], Mapping[str, object]]:
    """Bind retained pre-fit declarations to actual typed command identities."""
    from .corner_diagnostics import (
        _subpath_d, rebind_prefit_corner_diagnostics, retained_corner_declaration_matches,
    )
    from .corner_provenance import bind_prefit_corner_anchors

    d = source_text if source_text is not None else ' '.join(path.to_svg_d() for path in paths)
    if not retained_corner_declaration_matches(d, declaration):
        raise ValueError('corner operation declaration does not match replay target')
    bound = rebind_prefit_corner_diagnostics(d, declaration, tolerance=0.02)
    spans = bound['spans']
    skipped = list(bound['skipped'])
    output = []
    normalized = []
    for index, path in enumerate(paths):
        # The supplied declaration is authoritative, including an empty set.
        path = replace(path, anchors=tuple(anchor for anchor in path.anchors
                                          if not isinstance(anchor, CornerAnchor)))
        local = [span for span in spans if span['subpath'] == index]
        if not local:
            output.append(path)
            continue
        selected, binding_skips, feature_ids = bind_prefit_corner_anchors(
            {index: path}, local, overlap_policy='reject_all')
        skipped.extend(dict(item) for item in binding_skips)
        updated, receipt = normalize_path_corners(selected[index], max_error=max_error)
        normalized.extend(feature_ids[item] for item in receipt['normalized'])
        skipped.extend({**item, 'corner_id': feature_ids[item['corner_id']],
            'reason': 'protected_point_inside_span' if item['reason'] == 'protected_anchor_inside_span'
            else item['reason']} for item in receipt['skipped'])
        output.append(updated)
    normalized_d = ' '.join(_subpath_d([
        *( (kind, list(values)) for command in path.path.commands
           for kind, values in (command_token(command),)), ('Z', [])]) for path in output) if normalized else d
    result = rebind_prefit_corner_diagnostics(normalized_d, bound, tolerance=0.02)
    result['normalized'] = normalized
    prior = declaration.get('skipped', ())
    result['skipped'] = list({(item.get('corner_id'), item.get('reason')): dict(item)
        for item in [*(prior if isinstance(prior, Sequence) else ()), *skipped]
        if isinstance(item, Mapping) and item.get('corner_id') not in normalized
        and not (spans and item.get('reason') == 'missing_or_stale_provenance')}.values())
    return tuple(output), result


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

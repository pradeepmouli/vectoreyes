"""Canonical direct path operations; scene preparation remains engine-owned."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from ._fitcurve import fit_cubic_once, fit_quadratic_once
from .drawing import Drawing, DrawingRegion
from .drawing_actions import ActionSettings, DrawingPlan, UpdateGeometryAction
from .drawing_plan import validate_plan
from .drawing_trace import TraceCommand, svg_path_commands
from .emit import apply_affine_point
from .fit import _curved_run_d, _fmt
from .immutable_value import freeze_value
from .path_geometry import (
    CompoundPathGeometry, CornerAnchor, PathGeometry, PathGeometryWithCutouts,
    parse_svg_paths, command_token,
)
from .path_topology import component_paths, replace_component_paths
from .path_sampling import _sample_subpath
from .quality import polyline_boundary_distance


def _command_stream(geometry: PathGeometry | PathGeometryWithCutouts | CompoundPathGeometry) -> tuple[TraceCommand, ...]:
    """Adapt typed commands to transient numeric rows without rounding values."""
    rows = []
    for path in component_paths(geometry):
        for command in path.path.commands:
            kind, values = command_token(command)
            rows.append(TraceCommand(command.id, kind, tuple(values)))
        closing_id = f"{path.path.id}.z"
        identities = {command.id for command in path.path.commands}
        while closing_id in identities:
            closing_id += '_'
        rows.append(TraceCommand(closing_id, "Z", ()))
    return tuple(rows)


def _segment_references(drawing: Drawing) -> dict[str, tuple[TraceCommand, np.ndarray]]:
    references = {}
    for root in drawing.regions:
        for region in root.leaves():
            if not isinstance(region.geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
                continue
            commands = _command_stream(region.geometry)
            for index, command in enumerate(commands):
                if command.command not in {"M", "Z"}:
                    references[command.id] = (command, np.asarray(commands[index - 1].values[-2:], dtype=float))
    return references


def _affine_tuple(spec: Mapping[str, object]) -> tuple[float, float, float, float, float, float]:
    return tuple(float(spec[key]) for key in ("a", "b", "c", "d", "tx", "ty"))  # type: ignore[return-value]


def apply_region_path_operations(
    drawing: Drawing, action: UpdateGeometryAction, *, settings: ActionSettings,
    fit_sources: Mapping[str, np.ndarray] | None = None,
    source_text: str | None = None,
    previous_lineage: Mapping[str, object] | None = None,
) -> tuple[DrawingRegion, Mapping[str, object]]:
    """Execute the ordered program, binding version-scoped selections per step."""
    from .drawing_relationships import replace_region

    if len(action.ops) <= 1:
        return _apply_single_path_operation(drawing, action, settings=settings,
            fit_sources=fit_sources, source_text=source_text, previous_lineage=previous_lineage)
    validate_plan(DrawingPlan(version="vectoreyes.plan.v1", drawing_id=drawing.drawing_id,
        base_version="v0", ops=(action,)), drawing)
    original = drawing.region(action.target)
    if not isinstance(original.geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        raise ValueError("direct path operations require a concrete path-region leaf")
    bindings = {command.id: command.id for command in _command_stream(original.geometry)}
    input_ids = frozenset(bindings)
    metrics: dict[str, object] = {}
    current = drawing

    def bind(identity: str) -> str:
        if identity not in input_ids:
            return identity
        if identity not in bindings:
            raise ValueError(f"path command {identity!r} was retired by an earlier operation")
        return bindings[identity]

    for operation in action.ops:
        changes: dict[str, object] = {}
        for field in ("target", "reference", "between"):
            value = getattr(operation, field, None)
            if isinstance(value, str):
                changes[field] = bind(value)
            elif isinstance(value, tuple):
                changes[field] = tuple(bind(identity) for identity in value)
        step = action.model_copy(update={"ops": (operation.model_copy(update=changes),)})
        sources = None if fit_sources is None else {
            bindings.get(identity, identity): points for identity, points in fit_sources.items()
            if identity not in input_ids or identity in bindings
        }
        updated, receipt = _apply_single_path_operation(current, step, settings=settings,
            fit_sources=sources)
        immediate = receipt["path_command_lineage"]["commands"]
        bindings = {old: immediate[intermediate] for old, intermediate in bindings.items()
            if intermediate in immediate}
        metrics.update(receipt.get("fit_acceptance", {}))
        current = replace_region(current, updated)
    result = current.region(action.target)
    lineage = bindings
    input_d = source_text if source_text is not None else original.geometry.to_svg_d()
    if previous_lineage is not None:
        input_d = str(previous_lineage["input_d"])
        lineage = {old: bindings[intermediate] for old, intermediate in previous_lineage["commands"].items()
            if intermediate in bindings}
    final = dict(receipt)
    final["path_command_lineage"] = {
        "input_d": input_d, "output_d": result.geometry.to_svg_d(), "commands": lineage,
    }
    if metrics:
        final["fit_acceptance"] = metrics
    result = replace(result, diagnostics={**result.diagnostics, "fit_acceptance": metrics})
    return result, freeze_value(final)


def _apply_single_path_operation(
    drawing: Drawing, action: UpdateGeometryAction, *, settings: ActionSettings,
    fit_sources: Mapping[str, np.ndarray] | None = None,
    source_text: str | None = None,
    previous_lineage: Mapping[str, object] | None = None,
) -> tuple[DrawingRegion, Mapping[str, object]]:
    """Apply one direct stage to canonical geometry and return its binding receipt.

    The engine chooses original-material spans. This operation does not own
    raster state, paint resolution, scene topology, or version publication.
    Source text is retained only as the identity of the lineage input; numeric
    commands come directly from typed values, never rounded SVG serialization.
    """
    region = drawing.region(action.target)
    annotated = region.geometry
    if not isinstance(annotated, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        raise ValueError("direct path operations require a concrete path-region leaf")
    if any(op.op in {"simplify", "normalize_corners", "smooth"} for op in action.ops):
        raise ValueError("cleanup operations require their own ordered execution stage")
    unsupported = tuple(op.op for op in action.ops if op.op in {"reverse", "mirror", "break"})
    if unsupported:
        raise ValueError(f"path operation(s) not implemented: {', '.join(unsupported)}; no geometry was changed")
    validate_plan(DrawingPlan(version="vectoreyes.plan.v1", drawing_id=drawing.drawing_id,
        base_version="v0", ops=(action,)), drawing)
    commands = _command_stream(annotated)
    source_path_ids = {command.id: path.path.id for path in component_paths(annotated)
                       for command in path.path.commands}
    references = _segment_references(drawing)
    geometry = {"ops": tuple(op.model_dump(exclude_none=True) for op in action.ops)}
    epsilon, max_error = settings.epsilon, settings.max_error
    fit_metrics: dict[str, object] = {}
    def source_for(command_id: str) -> np.ndarray | None:
        if fit_sources is None:
            return None
        if command_id not in fit_sources:
            raise ValueError(f"fit {command_id} has no exact original material-polyline span")
        return fit_sources[command_id]

    fits = {
        op["target"]: _fit_mode(op)
        for op in geometry["ops"]
        if op["op"] == "fit" and "target" in op
    }
    length_matches = {
        op["target"]: op["reference"]
        for op in geometry["ops"]
        if op["op"] == "match_length"
    }
    matches = {
        op["target"]: (op["reference"], _affine_tuple(op["transform"]))
        for op in geometry["ops"]
        if op["op"] == "match"
    }
    parallels = {
        op["target"]: (op["reference"], op.get("distance"))
        for op in geometry["ops"]
        if op["op"] == "set_parallel"
    }
    alignments = {
        op["target"]: (op["reference"], set(op["axes"]))
        for op in geometry["ops"]
        if op["op"] == "align"
    }
    snaps = {
        op["target"]: op["reference"]
        for op in geometry["ops"]
        if op["op"] == "snap"
    }
    removals = {op["target"] for op in geometry["ops"] if op["op"] == "remove"}
    values_by_command = {command.id: tuple(command.values) for command in commands}
    command_index = {command.id: index for index, command in enumerate(commands)}
    raw_starts: dict[str, tuple[float, float]] = {}
    raw_current: tuple[float, float] | None = None
    raw_subpath_start: tuple[float, float] | None = None
    for command in commands:
        if command.command == "M":
            raw_current = (float(command.values[0]), float(command.values[1]))
            raw_subpath_start = raw_current
            continue
        if raw_current is not None:
            raw_starts[command.id] = raw_current
        if command.command == "Z":
            raw_current = raw_subpath_start
        else:
            raw_current = (float(command.values[-2]), float(command.values[-1]))

    span_fits: dict[str, tuple[tuple[str, ...], tuple[float, float], str, bool]] = {}
    skipped_by_span: set[str] = set()
    for op in geometry["ops"]:
        if op["op"] != "fit" or "between" not in op:
            continue
        left_id, right_id = op["between"]
        left_index, right_index = command_index[left_id], command_index[right_id]
        interior = tuple(
            command.id
            for command in commands[left_index + 1:right_index + 1]
            if command.command not in {"M", "Z"}
        )
        # ``drawing_plan.validate_plan`` has already proved that this is a
        # non-empty, target-local, ordered span in one subpath.
        assert interior and right_id in raw_starts
        kind, recursive = _fit_mode(op)
        end = tuple(float(value) for value in values_by_command[right_id][-2:])
        span_fits[interior[0]] = (interior, end, kind, recursive)
        skipped_by_span.update(interior[1:])
    parallel_lines: set[str] = set()
    for index, command in enumerate(commands):
        if command.id not in parallels:
            continue
        reference_id, requested_distance = parallels[command.id]
        reference_command, reference_start = references[reference_id]
        reference_vector = np.asarray(reference_command.values[-2:], dtype=float) - reference_start
        reference_length = float(np.linalg.norm(reference_vector))
        if reference_length == 0.0:
            raise ValueError(f"cannot make a segment parallel to zero-length {reference_command.id!r}")
        unit = reference_vector / reference_length
        normal = np.array([-unit[1], unit[0]])
        previous = commands[index - 1]
        start = np.asarray(values_by_command[previous.id][-2:], dtype=float)
        values = values_by_command[command.id]
        samples = np.array([start, *[values[offset:offset + 2] for offset in range(0, len(values), 2)]], dtype=float)
        segment_length = float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
        if segment_length == 0.0:
            raise ValueError(f"cannot fit zero-length segment {command.id!r} as parallel")
        along = (samples - reference_start) @ unit
        distance = float(requested_distance) if requested_distance is not None else float(((samples - reference_start) @ normal).mean())
        fitted_start = reference_start + unit * (float(along.mean()) - segment_length / 2) + normal * distance
        fitted_end = reference_start + unit * (float(along.mean()) + segment_length / 2) + normal * distance
        previous_values = list(values_by_command[previous.id])
        previous_values[-2:] = [float(fitted_start[0]), float(fitted_start[1])]
        values_by_command[previous.id] = tuple(previous_values)
        values_by_command[command.id] = (float(fitted_end[0]), float(fitted_end[1]))
        parallel_lines.add(command.id)
    pieces: list[str] = []
    command_lineage: dict[str, str] = {}
    output_index = -1
    output_path_id = ""

    def append_piece(piece: str, source_id: str) -> None:
        nonlocal output_index, output_path_id
        if piece.startswith("M"):
            output_path_id = source_path_ids[source_id]
            output_index = -1
        pieces.append(piece)
        output_index += sum(char in "MLQCA" for char in piece)
        command_lineage[source_id] = f"{output_path_id}.c{output_index}"

    current: tuple[float, float] | None = None
    subpath_start: tuple[float, float] | None = None
    for index, command in enumerate(commands):
        if command.command == "M":
            values = values_by_command[command.id]
            append_piece(f"M{_fmt(values[0])} {_fmt(values[1])}", command.id)
            current = (float(values[0]), float(values[1]))
            subpath_start = current
            continue
        if command.command == "Z":
            pieces.append("Z")
            current = subpath_start
            continue
        if command.id in skipped_by_span or command.id in removals:
            continue
        if current is None:
            raise ValueError("path segment appears before its move command")
        start = current
        if (span := span_fits.get(command.id)) is not None:
            interior, end, kind, recursive = span
            samples = _sample_subpath(
                [("M", list(start)), *[
                    (commands[command_index[interior_id]].command,
                     list(values_by_command[interior_id]))
                    for interior_id in interior
                ]],
                samples=32,
            )
            if not np.allclose(samples[-1], end):
                samples.append(end)
            sample_points = np.asarray(samples, dtype=float)
            source_points = source_for(interior[-1])
            if recursive:
                fitted = _recursive_fit_d(sample_points, kind, max_error=max_error, epsilon=epsilon)
                fitted = _validate_fit_source(start, fitted, source_points, max_error, f"{interior[0]}..{interior[-1]}", diagnostics=fit_metrics)
            else:
                fitted = _single_fit_d(
                    sample_points, kind, max_error=max_error,
                    address=f"{interior[0]}..{interior[-1]}", source_points=source_points, diagnostics=fit_metrics,
                )
            append_piece(fitted, interior[-1])
            current = end
            continue
        kind, recursive = fits.get(command.id, ("keep", False))
        output_command = "L" if command.id in parallel_lines else command.command
        values = values_by_command[command.id]
        if (match := matches.get(command.id)) is not None:
            reference_id, matrix = match
            reference_command, _ = references[reference_id]
            values = tuple(
                coordinate
                for point in zip(reference_command.values[::2], reference_command.values[1::2], strict=True)
                for coordinate in apply_affine_point(matrix, float(point[0]), float(point[1]))
            )
            output_command = reference_command.command
        if (reference_id := length_matches.get(command.id)) is not None:
            _, reference_start = references[reference_id]
            reference_command = references[reference_id][0]
            reference_vector = np.asarray(reference_command.values[-2:], dtype=float) - reference_start
            target_vector = np.asarray(values[-2:], dtype=float) - np.asarray(start, dtype=float)
            target_length = float(np.linalg.norm(target_vector))
            if target_length == 0.0:
                raise ValueError(f"cannot match length of zero-length segment {command.id!r}")
            scale = float(np.linalg.norm(reference_vector)) / target_length
            values = tuple(
                float(start[coordinate % 2]) + (value - start[coordinate % 2]) * scale
                for coordinate, value in enumerate(values)
            )
        if (alignment := alignments.get(command.id)) is not None:
            reference_id, axes = alignment
            reference_command, _ = references[reference_id]
            aligned = list(values)
            if "x" in axes:
                aligned[-2] = float(reference_command.values[-2])
            if "y" in axes:
                aligned[-1] = float(reference_command.values[-1])
            values = tuple(aligned)
        if (reference_id := snaps.get(command.id)) is not None:
            # Shared corners are the common case: move the target endpoint to
            # the reference endpoint on both axes, without translating the
            # region or exposing coordinate authoring to the caller.
            reference_command, _ = references[reference_id]
            snapped = list(values)
            snapped[-2:] = reference_command.values[-2:]
            values = tuple(snapped)
        # A retained command already in the requested family is an exact,
        # zero-residual one-command fit; numerical refitting can only lose it.
        exact_family = {"line": "L", "quadratic": "Q", "cubic": "C"}
        if kind == "keep" or (not recursive and output_command == exact_family.get(kind)):
            fitted = output_command + " ".join(_fmt(value) for value in values)
            if kind != "keep":
                fitted = _validate_fit_source(start, fitted, source_for(command.id), max_error, command.id, diagnostics=fit_metrics)
            append_piece(fitted, command.id)
        else:
            points = np.asarray(_sample_subpath(
                [("M", list(start)), (output_command, list(values))],
                samples=32,
            ), dtype=float)
            source_points = source_for(command.id)
            if recursive:
                fitted = _recursive_fit_d(points, kind, max_error=max_error, epsilon=epsilon)
                fitted = _validate_fit_source(start, fitted, source_points, max_error, command.id, diagnostics=fit_metrics)
            else:
                fitted = _single_fit_d(points, kind, max_error=max_error, address=command.id, source_points=source_points, diagnostics=fit_metrics)
            append_piece(fitted, command.id)
        current = (float(values[-2]), float(values[-1]))
    params: dict[str, object] = {"d": " ".join(pieces)}
    if fit_metrics:
        params["fit_acceptance"] = fit_metrics
    params["path_ids"] = tuple(path.path.id for path in component_paths(annotated))
    if isinstance(annotated, PathGeometryWithCutouts):
        params["fill_rule"] = "evenodd"
    elif isinstance(annotated, CompoundPathGeometry):
        params["fill_rule"] = annotated.fill_rule
    immediate = dict(command_lineage)
    input_d = source_text if source_text is not None else annotated.to_svg_d()
    if isinstance(previous_lineage, Mapping):
        input_d = str(previous_lineage["input_d"])
        command_lineage = {
            old: command_lineage[intermediate]
            for old, intermediate in previous_lineage["commands"].items()
            if intermediate in command_lineage
        }
    params["path_command_lineage"] = {
        "input_d": input_d,
        "output_d": params["d"],
        "commands": command_lineage,
    }
    if isinstance(annotated, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        originals = component_paths(annotated)
        parsed = parse_svg_paths(str(params["d"]), region_id=region.id, path_ids=tuple(path.path.id for path in originals))
        # Only endpoint/owned-command bindings proven by the executor survive.
        # A refit that explicitly consumes a corner retires that declaration.
        rebound_paths = []
        for path, original in zip(parsed, originals, strict=True):
            ids = tuple(command.id for command in path.path.commands)
            rebound_anchors = []
            for anchor in original.anchors:
                if isinstance(anchor, CornerAnchor):
                    if not all(item in immediate for item in (anchor.entry_command_id, anchor.exit_command_id)):
                        continue
                    entry = immediate[anchor.entry_command_id]
                    exit = immediate[anchor.exit_command_id]
                    owned = []
                    index = ids.index(entry)
                    while index != ids.index(exit):
                        index = (index + 1) % len(ids)
                        if index != 0:
                            owned.append(ids[index])
                    rebound_anchors.append(replace(anchor,
                        entry_command_id=entry, exit_command_id=exit,
                        owned_command_ids=tuple(owned)))
                elif all(item in immediate for item in anchor.command_ids):
                    rebound_anchors.append(replace(anchor,
                        command_ids=tuple(immediate[item] for item in anchor.command_ids)))
            rebound_paths.append(replace(path, anchors=tuple(rebound_anchors)))
        rebound = tuple(rebound_paths)
        params["canonical_path_geometry"] = replace_component_paths(annotated, rebound)
    updated = replace(region, geometry=params["canonical_path_geometry"], diagnostics={
        **region.diagnostics, "geometry": {"explicit": "path"}, "fit_acceptance": fit_metrics,
    })
    receipt = freeze_value({key: value for key, value in params.items()
                            if key not in {"d", "canonical_path_geometry"}})
    return updated, receipt


def _fit_mode(op: Mapping[str, object]) -> tuple[str, bool]:
    """Return the requested fit family and whether it may recursively split."""
    if "type" in op:
        return str(op["type"]), False
    return str(op["strategy"]), True


def _single_fit_d(
    points: np.ndarray, kind: str, *, max_error: float, address: str = "",
    source_points: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> str:
    """Emit exactly one command or reject a forced fit that exceeds tolerance."""
    start, end = points[0], points[-1]

    def reject(residual: float, split: int) -> None:
        worst = points[split]
        raise ValueError(
            f"single {kind} fit exceeds max_error; "
            f"target={address or 'unspecified'}; "
            f"measured_error_px={float(np.sqrt(residual)):.6g}; "
            f"max_error_px={max_error:.6g}; "
            f"worst_sample={split}/{len(points) - 1}; "
            f"worst_point=({worst[0]:.6g}, {worst[1]:.6g}); "
            "metric=parameterized_sample_distance; "
            f"use strategy='{kind}' to allow recursive splitting or inspect the span boundaries"
        )

    if kind == "line":
        return _validate_fit_source(start, f"L{_fmt(end[0])} {_fmt(end[1])}", source_points, max_error, address, diagnostics=diagnostics)
    if kind == "quadratic":
        curve, residual, split = fit_quadratic_once(points, max_error)
        if source_points is None and residual > max_error * max_error:
            reject(residual, split)
        return _validate_fit_source(start, f"Q{_fmt(curve[1][0])} {_fmt(curve[1][1])} {_fmt(curve[2][0])} {_fmt(curve[2][1])}", source_points, max_error, address, diagnostics=diagnostics)
    if kind == "cubic":
        # ``type`` is an explicit geometric instruction: unlike automatic
        # cubic fitting, it deliberately permits an S-curve when that is what
        # best represents the requested retained command or span. Its controls
        # are unconstrained unless the plan explicitly adds tangent constraints.
        curve, residual, split = fit_cubic_once(
            points, max_error, guard_inflections=False, constrain_tangents=False
        )
        if source_points is None and residual > max_error * max_error:
            reject(residual, split)
        return _validate_fit_source(start, (
            f"C{_fmt(curve[1][0])} {_fmt(curve[1][1])} {_fmt(curve[2][0])} {_fmt(curve[2][1])} "
            f"{_fmt(curve[3][0])} {_fmt(curve[3][1])}"
        ), source_points, max_error, address, diagnostics=diagnostics)
    raise ValueError(f"unsupported one-command fit type {kind!r}")


def _validate_fit_source(
    start: Sequence[float], d: str, source_points: np.ndarray | None,
    max_error: float, address: str, *, diagnostics: dict[str, object] | None = None,
) -> str:
    """Check an emitted L/Q/C span against its exact original material span."""
    if source_points is None:
        return d
    # Subdivide Bézier control polygons until their chord error is bounded.
    # This also validates the rounded emitted coordinates, not hidden controls.
    flatten_error = min(0.01, max_error / 20.0)
    vertices = [np.asarray(start, dtype=float)]

    def flatten(control: np.ndarray, depth: int = 0) -> None:
        a, b = control[0], control[-1]
        chord = b - a
        denom = float(chord @ chord)
        t = np.clip(((control - a) @ chord) / denom, 0, 1) if denom else np.zeros(len(control))
        error = float(np.max(np.linalg.norm(control - (a + t[:, None] * chord), axis=1)))
        if error <= flatten_error:
            vertices.append(b)
            return
        if depth >= 24:
            raise ValueError("fit source validation could not bound curve linearization")
        levels = [control]
        while len(levels[-1]) > 1:
            levels.append((levels[-1][:-1] + levels[-1][1:]) / 2.0)
        flatten(np.asarray([level[0] for level in levels]), depth + 1)
        flatten(np.asarray([level[-1] for level in reversed(levels)]), depth + 1)

    for command in svg_path_commands(f"M{_fmt(start[0])} {_fmt(start[1])} {d}", "fit-check"):
        if command.command == "M":
            vertices[0] = np.asarray(command.values, dtype=float)
        elif command.command in {"L", "Q", "C"}:
            control = np.vstack([vertices[-1], np.asarray(command.values).reshape(-1, 2)])
            flatten(control)
        else:
            raise ValueError("source fit guard supports only open L/Q/C spans")
    original = [list(map(tuple, source_points))]
    candidate = [list(map(tuple, vertices))]
    measured = polyline_boundary_distance(
        original, candidate,
        closed=False, precision=flatten_error, direction="replacement_to_source",
    ) + flatten_error
    reverse = polyline_boundary_distance(
        original, candidate,
        closed=False, precision=flatten_error, direction="source_to_replacement",
    ) + flatten_error
    if diagnostics is not None:
        diagnostics[address] = {
            "reference": "original_material_polyline",
            "acceptance_direction": "curve_to_original",
            "curve_to_original_px": measured,
            "original_to_curve_px": reverse,
            "reverse_is_diagnostic_only": True,
            "max_error_px": max_error,
            "measurement_precision_px": 2 * flatten_error,
        }
    if not math.isfinite(measured) or measured > max_error:
        raise ValueError(
            f"fit exceeds max_error; target={address}; "
            f"measured_error_px={measured:.6g}; max_error_px={max_error:.6g}; "
            "reference=original_material_polyline; metric=curve_to_original_distance_upper_bound; "
            f"original_to_curve_diagnostic_px={reverse:.6g}; "
            f"measurement_precision_px={2 * flatten_error:.6g}; "
            "inspect source span boundaries or use recursive fitting"
        )
    return d


def _recursive_fit_d(points: np.ndarray, strategy: str, *, max_error: float, epsilon: float) -> str:
    """Emit one or more commands according to a residual-driven fit strategy."""
    return _curved_run_d(
        points,
        max_error,
        cubic=strategy == "cubic",
        line_epsilon=epsilon,
        progressive=strategy in {"progressive", "progressive_allow_lines"},
        progressive_allow_lines=strategy == "progressive_allow_lines",
    ).strip()

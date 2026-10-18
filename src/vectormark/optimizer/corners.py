"""Corner provenance and conservative corner-local normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import numpy as np

from .. import corner_normalization, corner_provenance
from ..corner_ownership import directed_corner_ownership
from ..fit import Shape
from ..path_geometry import command_token, parse_svg_paths
from .vector_region import _parse_subpaths

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
    """Bind pre-fit declarations; boundary anchors do not own adjacent sides."""
    declarations = []
    for subpath, _tokens in enumerate(_parse_subpaths(d)):
        declarations.extend({
            "corner_id": f"p{subpath}.corner{number}", "kind": "point",
            "subpath": subpath, "source": "prefit", "point": tuple(point),
        } for number, point in enumerate(declared_corners))
        declarations.extend({
            "corner_id": f"p{subpath}.rounded_corner{number}", "kind": "curve",
            "subpath": subpath, "source": "prefit",
            "start_point": tuple(start), "end_point": tuple(end),
        } for number, (start, end) in enumerate(declared_corner_spans))
    return _bind_prefit_spans(d, declarations, tolerance=corner_tolerance)


def _bind_prefit_spans(d: str, declarations: Sequence, *, tolerance: float) -> dict[str, object]:
    """One ownership definition shared by initial binding and path rewrites."""
    paths = _parse_subpaths(d)
    spans, skipped = [], []
    anchors: dict[tuple[int, int], dict[str, object]] = {}
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
        kind = declaration.get("kind")
        subpath_value = declaration.get("subpath")
        if (
            declaration.get("source") != "prefit"
            or not isinstance(corner_id, str)
            or not corner_id
            or not isinstance(subpath_value, int)
            or isinstance(subpath_value, bool)
            or kind not in {"point", "curve"}
        ):
            invalid_provenance()
            continue

        point_fields = ("point",) if kind == "point" else ("start_point", "end_point")
        normalized_points: dict[str, tuple[float, float]] = {}
        for field in point_fields:
            value = declaration.get(field)
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != 2
            ):
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
            normalized_points[field] = point
        if len(normalized_points) != len(point_fields):
            continue

        span = dict(declaration)
        span.update(normalized_points)
        subpath = subpath_value
        reason = None
        if not 0 <= subpath < len(paths):
            skipped.append({"corner_id": span["corner_id"], "reason": "missing_subpath"})
            continue
        tokens = paths[subpath]
        segments = _segments(tokens)
        if not segments or not tokens or tokens[0][0] != "M":
            continue
        drawable = [int(s["command_index"]) for s in segments if s["command"] != "Z"]
        endpoints = {0: np.asarray(tokens[0][1], dtype=float)}
        endpoints.update({int(s["command_index"]): s["end"] for s in segments if s["command"] != "Z"})
        # M and the explicit closing endpoint are two addresses for one node.
        if drawable and np.linalg.norm(endpoints[drawable[-1]] - endpoints[0]) <= 1e-9:
            del endpoints[drawable[-1]]

        def locate(point):
            matches = sorted((float(np.linalg.norm(end - np.asarray(point, dtype=float))), index)
                             for index, end in endpoints.items())
            if not matches or matches[0][0] > tolerance:
                return None
            if len(matches) > 1 and abs(matches[1][0] - matches[0][0]) <= 1e-9:
                return None
            return matches[0][1]

        if span["kind"] == "point":
            entry = exit = locate(span["point"])
            commands = []
        else:
            entry, exit = locate(span["start_point"]), locate(span["end_point"])
            commands = []
            if entry is not None and exit is not None and entry != exit:
                explicit_close = bool(drawable) and np.linalg.norm(
                    segments[drawable[-1]]["end"] - np.asarray(tokens[0][1], dtype=float)
                ) <= 1e-9
                try:
                    commands = list(directed_corner_ownership(
                        tuple(str(index) for index in [0, *drawable]), str(entry), str(exit),
                        explicit_closing_drawable=explicit_close,
                    ))
                    commands = [int(index) for index in commands]
                except ValueError as error:
                    reason = "implicit_closing_edge" if "implicit closing" in str(error) else "ambiguous_span"
            elif entry == exit and entry is not None:
                reason = "ambiguous_span"
        if entry is None or exit is None:
            reason = "missing_or_ambiguous_endpoint"
        if reason:
            skipped.append({"corner_id": span["corner_id"], "reason": reason})
            continue
        span.update(entry_command=entry, exit_command=exit, commands=commands,
                    curve_commands=[i for i in commands if tokens[i][0] in _CURVE_COMMANDS],
                    anchor_commands=list(dict.fromkeys([entry, *commands, exit])))
        spans.append(span)
        for index in span["anchor_commands"]:
            key = (subpath, index)
            anchor = anchors.setdefault(key, {"subpath": subpath, "command_index": index,
                "anchor_kind": "corner", "corner_id": span["corner_id"], "corner_ids": []})
            if span["corner_id"] not in anchor["corner_ids"]:
                anchor["corner_ids"].append(span["corner_id"])
    return recount_corner_diagnostics(d, {
        "path": d, "anchors": [anchors[key] for key in sorted(anchors)],
        "spans": spans, "skipped": skipped,
    })


def recount_corner_diagnostics(d: str, diagnostics: Mapping[str, object]) -> dict[str, object]:
    """Derive counters from the retained commands and exact declared ownership."""
    paths = _parse_subpaths(d)
    total = sum(command not in {"M", "Z"} for tokens in paths for command, _ in tokens)
    spans = diagnostics.get("spans", ())
    anchors = diagnostics.get("anchors", ())
    owned = {(span["subpath"], index) for span in spans for index in span.get("commands", ())}
    return {
        **diagnostics,
        "commands": {"total": total, "corner": len(owned), "free": total - len(owned)},
        "anchor_counts": {"total": total + len(paths), "corner": len(anchors)},
        "corner_counts": {"total": len(spans), "point": sum(s.get("kind") == "point" for s in spans),
                          "curve": sum(s.get("kind") == "curve" for s in spans)},
    }


def source_ordered_corner_diagnostics(d: str, evidence: Mapping[str, object]) -> dict[str, object]:
    """Flatten the root fitter's per-path declarations by original path index.

    The producer emits local subpath zero in each record. This is identity
    transport only: stale/malformed records are explicit skips, never inferred
    from coordinates or from the path after topology classification.
    """
    if "spans" in evidence or "subpaths" not in evidence or evidence.get("path") != d:
        return dict(evidence)
    records = evidence["subpaths"]
    raw = tuple(part.strip() for part in re.findall(r"M[^M]*", d))
    skipped = list(evidence.get("skipped", ()))
    spans, anchors = [], []
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or len(records) != len(raw):
        return {**evidence, "spans": [], "anchors": [], "skipped": [*skipped, {"reason": "invalid_corner_provenance"}]}
    for index, (record, path_d) in enumerate(zip(records, raw, strict=True)):
        if not isinstance(record, Mapping) or record.get("path") != path_d:
            skipped.append({"subpath": index, "reason": "missing_or_stale_provenance"})
            continue
        skipped.extend(record.get("skipped", ()))
        for field, output in (("spans", spans), ("anchors", anchors)):
            for declaration in record.get(field, ()):
                if not isinstance(declaration, Mapping) or declaration.get("subpath") != 0:
                    skipped.append({"subpath": index, "reason": "invalid_corner_provenance"})
                    continue
                identity = str(declaration.get("corner_id", ""))
                if identity.startswith("p0."):
                    identity = f"p{index}." + identity[3:]
                output.append({**declaration, "subpath": index, "corner_id": identity})
    return {**evidence, "spans": spans, "anchors": anchors, "skipped": skipped}


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
    return _bind_prefit_spans(d, spans if isinstance(spans, Sequence) else (), tolerance=tolerance)


def _subpath_d(tokens: list[tuple[str, list[float]]]) -> str:
    # Only newly fitted controls are rounded. Retained endpoints/side handles
    # must not drift merely because another corner in this path was normalized.
    return " ".join(
        command if not values else f"{command}{' '.join(str(float(value)) for value in values)}"
        for command, values in tokens
    )


def retained_corner_declaration_matches(
    d: str, diagnostics: object
) -> bool:
    """Whether diagnostics are pre-fit corner evidence for this exact path."""
    return isinstance(diagnostics, Mapping) and diagnostics.get("path") == d


def normalize_corners_path_d(
    d: str,
    *,
    max_error: float,
    diagnostics: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Fit only declared owned curves; preserve their endpoints and provenance."""
    if not retained_corner_declaration_matches(d, diagnostics):
        result = path_corner_diagnostics(d)
        result["skipped"] = [{"reason": "missing_or_stale_provenance"}]
        result["normalized"] = []
        return d, result
    # Rebind from pre-fit endpoint evidence, not caller-provided command ranges.
    bound = rebind_prefit_corner_diagnostics(d, diagnostics, tolerance=0.02)
    paths = _parse_subpaths(d)
    spans = bound["spans"]
    skipped = list(bound["skipped"])
    normalized = []
    output = []
    for subpath, tokens in enumerate(paths):
        local_spans = [s for s in spans if s["subpath"] == subpath]
        if not local_spans:
            output.append(tokens)
            continue
        path_id = f"p{subpath}"
        path = parse_svg_paths(_subpath_d(tokens), region_id="corner", path_ids=(path_id,))[0]
        bound_paths, binding_skips, feature_ids = corner_provenance.bind_prefit_corner_anchors(
            {subpath: path}, local_spans, overlap_policy="reject_all",
        )
        skipped.extend(dict(item) for item in binding_skips)
        canonical, receipt = corner_normalization.normalize_path_corners(
            bound_paths[subpath], max_error=max_error,
        )
        normalized.extend(feature_ids[item] for item in receipt["normalized"])
        skipped.extend({**item, "corner_id": feature_ids[item["corner_id"]], "reason": "protected_point_inside_span"
            if item["reason"] == "protected_anchor_inside_span" else item["reason"]}
            for item in receipt["skipped"])
        output.append([*( (kind, list(values)) for command in canonical.path.commands
            for kind, values in (command_token(command),)), ("Z", [])])
    normalized_d = " ".join(_subpath_d(tokens) for tokens in output) if normalized else d
    result = rebind_prefit_corner_diagnostics(normalized_d, bound, tolerance=0.02)
    result["normalized"] = normalized
    # An unresolved declaration may already have been removed from the bound
    # spans. Retain its explanation on subsequent no-ops rather than silently
    # presenting the same unannotated path as successfully processed.
    prior_skips = diagnostics.get("skipped", ())
    all_skips = [*(prior_skips if isinstance(prior_skips, Sequence) else ()), *skipped]
    result["skipped"] = list({
        (item.get("corner_id"), item.get("reason")): dict(item)
        for item in all_skips
        if isinstance(item, Mapping)
        and item.get("corner_id") not in normalized
        and not (spans and item.get("reason") == "missing_or_stale_provenance")
    }.values())
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

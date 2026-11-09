"""Shared saved-report/MCP adapter; all numeric measurements belong to the engine."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
import json
from math import isfinite
from typing import Any

from ..boundary_detail import BoundaryDetailPolicy, BoundaryDetailRequest
from ..drawing_trace import RootPathEvidence
from ..path_geometry import ClosedPath, LineCommand, MoveCommand, PathGeometry, parse_svg_paths
from ..python_drawing_engine import PythonDrawingEngine
from .inspection import _paths


def retained_source_paths(records: Sequence[RootPathEvidence]) -> list[dict[str, Any]]:
    """One codec for the existing raw-trace artifact and detail diagnostics."""
    return [{"region_id": record.region_id, "path_id": record.path_id,
        "source_path_id": record.fit.source.path.id,
        "commands": [{"id": command.id, "end": list(command.end)} for command in record.fit.source.path.commands],
        "corners": [{"id": anchor.id, "entry_command_id": anchor.entry_command_id,
            "exit_command_id": anchor.exit_command_id, "owned_command_ids": list(anchor.owned_command_ids)}
            for anchor in record.fit.source.anchors]} for record in records]


def _source_geometry(record: dict[str, Any]) -> PathGeometry:
    commands = record["commands"]
    points = [tuple(command["end"]) for command in commands]
    if len(points) < 4 or points[0] != points[-1] or any(
        len(point) != 2 or not all(isfinite(value) for value in point) for point in points
    ):
        raise ValueError("source must be a finite, explicitly closed polyline")
    return PathGeometry(ClosedPath(record["source_path_id"], tuple(
        (MoveCommand if index == 0 else LineCommand)(command["id"], point)
        for index, (command, point) in enumerate(zip(commands, points)))))


def build_detail_report(response: dict[str, Any], raw_trace: dict[str, Any], *,
                        policy: BoundaryDetailPolicy | None = None) -> dict[str, Any]:
    """Measure only exact region/path pairs; never guess reordered holes or splits."""
    policy = policy or BoundaryDetailPolicy()
    records = raw_trace.get("source_paths") or []
    source_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_index[record.get("region_id"), record.get("path_id")].append(record)
    source_regions = {key[0] for key in source_index}
    targets = response.get("report", {}).get("targets", [])
    target_ids = Counter(target.get("id") for target in targets)
    trace = raw_trace.get("trace") or {}
    mismatch = any(response.get(axis) is not None and response[axis] != trace.get(axis) for axis in ("width", "height"))
    engine = PythonDrawingEngine()
    regions = []
    for target in targets:
        identity, geometry = target.get("id"), target.get("geometry") or {}
        paths = _paths(geometry)
        reason = None
        if mismatch:
            reason = "source_canvas_mismatch"
        elif target_ids[identity] != 1:
            reason = "ambiguous_candidate_region"
        elif geometry.get("type") not in {"path", "path_with_cutouts"}:
            reason = "no_local_boundary_pair"
        elif len(source_regions.intersection(target.get("source_regions") or [])) > 1:
            reason = "merged_source_regions"
        elif not paths or any((identity, path.get("id")) not in source_index for path in paths):
            reason = "missing_source_boundary"
        elif any(len(source_index[identity, path.get("id")]) != 1 for path in paths):
            reason = "ambiguous_source_boundary"
        elif len({path.get("id") for path in paths}) != len(paths):
            reason = "ambiguous_candidate_boundary"
        elif {path.get("id") for path in paths} != {key[1] for key in source_index if key[0] == identity}:
            reason = "boundary_set_changed"
        boundaries = []
        if reason is None:
            for path in paths:
                record = source_index[identity, path["id"]][0]
                try:
                    source = _source_geometry(record)
                except (KeyError, TypeError, ValueError):
                    reason = "invalid_source_boundary"
                    break
                try:
                    candidates = parse_svg_paths(path["d"], region_id=identity)
                    if len(candidates) != 1:
                        raise ValueError("candidate must contain exactly one boundary")
                    candidate = replace(candidates[0], path=replace(candidates[0].path, id=path["id"]))
                    revision = "sha256:" + sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
                    assessment = engine.assess_boundary_detail(BoundaryDetailRequest(
                        region_id=identity, source=source, candidate=candidate, source_revision=revision,
                        candidate_revision=response["version"], reference_kind="retained_polyline", policy=policy))
                    boundaries.append(assessment.diagnostics())
                except (KeyError, TypeError, ValueError):
                    reason = "invalid_candidate_boundary"
                    break
        if reason:
            boundaries = []
        measured = bool(boundaries) and all(item["status"] == "assessed" for item in boundaries)
        regions.append({"id": identity, "status": "assessed" if measured else "not_assessed",
            "reason": reason, "boundaries": boundaries})
    return {"schema_version": "vectoreyes.boundary-detail.v1", "drawing_id": response.get("drawing_id"),
        "version": response.get("version"), "reference": "retained material polylines; trace pixel coordinates",
        "regions": regions}


def detail_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Bound inline report size; full per-sample results remain in the JSON artifact."""
    regions = []
    for region in report["regions"]:
        boundaries = []
        for boundary in region["boundaries"]:
            scales = []
            for scale in boundary["scales"]:
                samples = scale["samples"]
                # Preserve both directions, not just whichever has larger magnitude.
                indices = set()
                for key in ("unsupported_curvature_per_px", "missing_curvature_per_px"):
                    indices.update(sorted(range(len(samples)), key=lambda i: samples[i][key], reverse=True)[:6])
                scales.append({**{k: v for k, v in scale.items() if k != "samples"},
                    "hotspots": [samples[index] for index in sorted(indices)]})
            boundaries.append({**boundary, "scales": scales})
        regions.append({**region, "boundaries": boundaries})
    return {**report, "regions": regions}

"""Inspection records from public retained metadata, with no feature detection."""
from __future__ import annotations

from collections import Counter
from typing import Any


def _paths(value: object) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("commands"), list) and "id" in value:
            return [value]
        return [path for child in value.values() for path in _paths(child)]
    if isinstance(value, list):
        return [path for child in value for path in _paths(child)]
    return []


def _fields(value: object) -> Any:
    """Unwrap the engine's tagged provenance JSON without interpreting IDs."""
    if isinstance(value, list):
        return [_fields(item) for item in value]
    if isinstance(value, dict):
        body = value.get("fields", value) if "type" in value and "fields" in value else value
        return {key: _fields(child) for key, child in body.items()}
    return value


def build_inspection(response: dict[str, Any], raw_trace: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    regions = []
    for target in response.get("report", {}).get("targets", []):
        paths = _paths(target.get("geometry", {}))
        commands = [c for path in paths for c in path.get("commands", [])]
        anchors = [a for path in paths for a in path.get("anchors", [])]
        corners = [a for a in anchors if a.get("type") == "corner"]
        corner_owned = {identity for a in corners for identity in a.get("owned_command_ids", [])}
        kinds = Counter(c.get("type") for c in commands if c.get("type") != "move")
        regions.append({
            "id": target["id"], "root_id": target.get("root_id"), "source_regions": target.get("source_regions", []),
            "geometry": target.get("geometry"), "paths": paths,
            "fill": target.get("fill"), "diagnostics": target.get("diagnostics", {}),
            "counts": {"origin": "derived_from_retained_objects", "commands": sum(kinds.values()),
                       "by_type": dict(kinds), "corners": len(corners),
                       "corner_commands": len(corner_owned), "free_commands": sum(kinds.values())-len(corner_owned),
                       "point_corners": sum(a.get("entry_command_id") == a.get("exit_command_id") for a in corners),
                       "rounded_corners": sum(a.get("entry_command_id") != a.get("exit_command_id") for a in corners)},
            "path_editability": "retained_commands" if paths else "no_local_path_commands",
        })
    replay = _fields(provenance.get("plan", provenance).get("replay", {}))
    correspondences = replay.get("path_correspondence", []) if isinstance(replay, dict) else []
    return {"schema_version": "vectormark.inspection.v1", "drawing_id": response.get("drawing_id"),
            "version": response.get("version"), "width": response.get("width") or raw_trace.get("trace", {}).get("width"),
            "height": response.get("height") or raw_trace.get("trace", {}).get("height"),
            "regions": regions, "source_paths": raw_trace.get("source_paths", []),
            "correspondence": correspondences,
            "correspondence_status": "retained" if correspondences else "unmapped"}

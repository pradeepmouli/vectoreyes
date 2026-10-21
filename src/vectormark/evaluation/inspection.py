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
        anchors = [a for path in paths for a in (path.get("anchors") or [])]
        corners = [a for a in anchors if a.get("type") == "corner"]
        corner_owned = {identity for a in corners for identity in a.get("owned_command_ids", [])}
        kinds = Counter(c.get("type") for c in commands if c.get("type") != "move")
        skipped = (target.get("diagnostics", {}).get("corners") or {}).get("skipped") or []
        corner_evidence = bool(paths) and all(isinstance(p.get("anchors"), list) for p in paths)
        if any(item.get("reason") == "missing_or_stale_provenance" for item in skipped):
            corner_evidence = False
        regions.append({
            "id": target["id"], "root_id": target.get("root_id"), "source_regions": target.get("source_regions", []),
            "geometry": target.get("geometry"), "paths": paths,
            "fill": target.get("fill"), "diagnostics": target.get("diagnostics", {}),
              "counts": {"origin": "derived_from_retained_objects", "paths": len(paths),
                         "commands": sum(kinds.values()), "by_type": dict(kinds),
                         "corner_evidence": "retained" if corner_evidence else "unavailable",
                         "corners": len(corners) if corner_evidence else None,
                         "corner_commands": len(corner_owned) if corner_evidence else None,
                         "free_commands": sum(kinds.values())-len(corner_owned) if corner_evidence else None,
                         "point_corners": sum(a.get("entry_command_id") == a.get("exit_command_id") for a in corners) if corner_evidence else None,
                         "rounded_corners": sum(a.get("entry_command_id") != a.get("exit_command_id") for a in corners) if corner_evidence else None},
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


def compare_inspections(reference: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compare local retained complexity, never infer merge/split counterparts."""
    result = {"reference_version": reference.get("version"), "version": current.get("version"),
              "status": "unavailable", "regions": []}
    if not reference.get("drawing_id") or reference.get("drawing_id") != current.get("drawing_id"):
        return result
    before = {row['id']: row for row in reference.get('regions', [])}
    after = {row['id']: row for row in current.get('regions', [])}
    if len(before) != len(reference.get('regions', [])) or len(after) != len(current.get('regions', [])):
        return result
    for identity in dict.fromkeys((*after, *before)):
        old, new = before.get(identity), after.get(identity)
        status = ('unmatched_reference' if new is None else 'unmatched_current' if old is None
                  else 'lineage_unavailable' if not old.get('source_regions') or not new.get('source_regions')
                  else 'matched' if set(old.get('source_regions', [])) == set(new.get('source_regions', []))
                  else 'lineage_changed')
        delta = None
        if status == 'matched':
            delta = {key: new['counts'][key] - old['counts'][key]
                     if isinstance(new['counts'].get(key), int) and isinstance(old['counts'].get(key), int) else None
                     for key in ('paths', 'commands', 'corners', 'point_corners',
                                 'rounded_corners', 'corner_commands', 'free_commands')}
        result['regions'].append({'id': identity, 'status': status,
            'before': old['counts'] if old else None, 'after': new['counts'] if new else None, 'delta': delta})
    result['status'] = 'compared'
    return result

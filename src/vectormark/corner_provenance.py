"""Bind already-resolved pre-fit declarations to canonical path commands."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import Literal

from .immutable_value import freeze_value
from .path_geometry import CornerAnchor, PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry, parse_svg_paths


def bind_matching_prefit_corners(
    paths: Sequence[PathGeometry], *, d: str, evidence: object,
) -> tuple[tuple[PathGeometry, ...], Mapping[str, object]]:
    """Attach matching source declarations before canonical topology publication."""
    from .corner_diagnostics import source_ordered_corner_diagnostics, rebind_prefit_corner_diagnostics
    if not isinstance(evidence, Mapping) or evidence.get("path") != d:
        diagnostics = dict(evidence) if isinstance(evidence, Mapping) else {"path": d, "spans": []}
        diagnostics["skipped"] = [
            *(dict(item) for item in diagnostics.get("skipped", ()) if isinstance(item, Mapping)),
            {"reason": "missing_or_stale_provenance"},
        ]
        return tuple(paths), diagnostics
    evidence = source_ordered_corner_diagnostics(d, evidence)
    rebound = rebind_prefit_corner_diagnostics(d, evidence, tolerance=0.02)
    rebound["skipped"] = [
        *(dict(item) for item in evidence.get("skipped", ()) if isinstance(item, Mapping)),
        *(dict(item) for item in rebound.get("skipped", ()) if isinstance(item, Mapping)),
    ]
    bound, skipped, _feature_ids = bind_prefit_corner_anchors(
        dict(enumerate(paths)), rebound["spans"], overlap_policy="retain_first")
    rebound["skipped"].extend(dict(item) for item in skipped)
    return tuple(bound[index] for index in range(len(paths))), rebound


def ingress_prefit_corners(
    geometry: PathGeometry | PathGeometryWithCutouts | CompoundPathGeometry,
    d: str, region_id: str, evidence: object, *, path_ids: Sequence[str] | None = None,
) -> tuple[PathGeometry | PathGeometryWithCutouts | CompoundPathGeometry, Mapping[str, object]]:
    """Bind declarations in source order, then retain canonical component order."""
    from .path_topology import component_paths, replace_component_paths
    by_id = {path.path.id: path for path in component_paths(geometry)}
    source = parse_svg_paths(d, region_id=region_id, path_ids=path_ids)
    paths, diagnostics = bind_matching_prefit_corners(
        tuple(by_id[path.path.id] for path in source), d=d, evidence=evidence)
    by_id = {path.path.id: path for path in paths}
    return replace_component_paths(geometry,
        tuple(by_id[path.path.id] for path in component_paths(geometry))), diagnostics


def bind_prefit_corner_anchors(
    paths: Mapping[int, PathGeometry], declarations: Sequence[Mapping[str, object]], *,
    overlap_policy: Literal["retain_first", "reject_all"],
) -> tuple[Mapping[int, PathGeometry], tuple[Mapping[str, object], ...], Mapping[str, str]]:
    """Publish source-indexed declarations without inferring new corners.

    Source matching is the caller's responsibility. Indices select actual typed
    commands, not reconstructed command names. Publication preserves existing
    features and returns canonical-to-source feature IDs for diagnostic output.
    Initial ingress retains the first valid owner; normalization rejects every
    overlapping candidate before changing geometry. Those policies are explicit.
    """
    if overlap_policy not in {"retain_first", "reject_all"}:
        raise ValueError("unknown corner overlap policy")
    result = dict(paths)
    skipped: list[dict[str, object]] = []
    candidates: list[tuple[int, str, CornerAnchor]] = []
    for span in declarations:
        source_id = span.get("corner_id") if isinstance(span, Mapping) else None
        try:
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("corner provenance requires a non-empty feature ID")
            subpath = span["subpath"]
            if type(subpath) is not int:
                raise ValueError("corner provenance requires an integer source subpath")
            path = paths[subpath]
            def command_id(index: object) -> str:
                if type(index) is not int or not 0 <= index < len(path.path.commands):
                    raise ValueError("corner provenance command index is outside its source path")
                return path.path.commands[index].id
            prefix = f"p{subpath}."
            local_id = source_id[len(prefix):] if source_id.startswith(prefix) else source_id
            anchor = CornerAnchor(id=f"{path.path.id}.{local_id}",
                entry_command_id=command_id(span["entry_command"]),
                exit_command_id=command_id(span["exit_command"]),
                owned_command_ids=tuple(command_id(index) for index in span["commands"]),
                radius=span.get("radius"))
            PathGeometry(path.path, (anchor,))
            candidates.append((subpath, source_id, anchor))
        except (KeyError, TypeError, ValueError) as error:
            skipped.append({**({"corner_id": source_id} if isinstance(source_id, str) and source_id else {}),
                "reason": "invalid_corner_anchor", "detail": str(error)})

    ownership = Counter((subpath, command_id) for subpath, _source, anchor in candidates
        for command_id in anchor.owned_command_ids)
    owned = {index: {command_id for anchor in path.anchors if isinstance(anchor, CornerAnchor)
        for command_id in anchor.owned_command_ids} for index, path in paths.items()}
    feature_ids: dict[str, str] = {}
    for subpath, source_id, anchor in candidates:
        if owned[subpath].intersection(anchor.owned_command_ids) or (
            overlap_policy == "reject_all" and any(
                ownership[subpath, item] > 1 for item in anchor.owned_command_ids)
        ):
            skipped.append({"corner_id": source_id, "reason": "overlapping_corner_ownership"})
            continue
        path = result[subpath]
        try:
            updated = replace(path, anchors=(*path.anchors, anchor))
        except ValueError as error:
            skipped.append({"corner_id": source_id, "reason": "invalid_corner_anchor", "detail": str(error)})
            continue
        result[subpath] = updated
        owned[subpath].update(anchor.owned_command_ids)
        feature_ids[anchor.id] = source_id
    return MappingProxyType(result), tuple(freeze_value(item) for item in skipped), freeze_value(feature_ids)

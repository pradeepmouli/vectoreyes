"""Shared merge/split execution over retained Python-engine region evidence.

The drawing engine owns invocation and publication. Refinement only dispatches
these typed actions; no independent partition implementation lives there.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .drawing_actions import MergeAction, SplitAction
from .drawing_adapter import (
    WorkingLeafTarget,
    working_leaf_targets,
    replace_working_leaf,
    remove_working_leaves,
)
from .fit import Shape, _fmt
from .geometry_native import split_path_by_line as _split_footprint_by_line
from .optimizer.gate import rasterize
from .optimizer.vector_region import VectorRegion, _parse_subpaths, _sample_subpath
from .skia_geometry import SkPath, unary_union


def merge_regions(
    regions: Sequence[VectorRegion], action: MergeAction
) -> tuple[VectorRegion, ...]:
    """Create a semantic target with a SkPath-unioned, seam-free outline."""
    targets = working_leaf_targets(regions)
    member_ids = _expand_merge_member_ids(targets, action.regions)
    members = tuple(targets[region_id] for region_id in member_ids)
    member_regions = tuple(member.region for member in members)
    first = member_regions[0]
    assert first.current is not None and first.fill is not None
    raster = np.zeros_like(first.raster, dtype=bool)
    for member in member_regions:
        raster |= member.raster
    footprints = [member.footprint for member in member_regions]
    if not all(isinstance(footprint, SkPath) for footprint in footprints):
        raise ValueError("merge requires polygonal vector-region footprints")
    footprint = unary_union(
        [footprint for footprint in footprints if isinstance(footprint, SkPath)]
    )
    shape = Shape("path", {"d": footprint.to_svg_d()})
    merged = VectorRegion.from_shape(
        id=max(region.id for region in _all_regions(regions)) + 1,
        shape=shape,
        fill=first.fill,
        z=min(member.z for member in member_regions),
        raster=raster,
        footprint=footprint,
        color_hex=first.color_hex,
        drawing_id=action.id,
        source_regions=tuple(
            sorted({source for member in members for source in member.source_regions})
        ),
        diagnostics={"merge": {"unioned": True, "sources": member_ids}},
    )
    return remove_working_leaves(regions, frozenset(member_ids)) + (merged,)


def _expand_merge_member_ids(
    targets: Mapping[str, WorkingLeafTarget], requested_ids: Sequence[str]
) -> tuple[str, ...]:
    """Resolve a branch/root reference to every leaf currently retained below it."""
    resolved: list[str] = []
    for requested in requested_ids:
        members = (
            (requested,)
            if requested in targets
            else tuple(
                sorted(
                    target_id
                    for target_id in targets
                    if target_id.startswith(f"{requested}-")
                )
            )
        )
        if not members:
            raise KeyError(requested)
        for member in members:
            if member in resolved:
                raise ValueError(
                    f"merge region {requested!r} resolves to an already selected child"
                )
            resolved.append(member)
    return tuple(resolved)


def split_region(
    regions: Sequence[VectorRegion], action: SplitAction
) -> tuple[VectorRegion, ...]:
    """Replace one retained target with two exact vector clips across an explicit divider."""
    targets = working_leaf_targets(regions)
    target_id = action.target
    target = targets[target_id]
    source = target.region
    if not source.is_leaf or source.current is None or source.fill is None:
        raise ValueError("split requires a filled leaf target")
    if not isinstance(source.footprint, SkPath):
        raise ValueError("split requires a polygonal vector-region footprint")

    divider = action.divider
    if divider.type == "line":
        first, second = divider.points
        parts = _split_footprint_by_line(
            source.footprint,
            (float(first[0]), float(first[1])),
            (float(second[0]), float(second[1])),
        )
    elif divider.type == "path":
        parts = _split_footprint_by_path(source.footprint, divider.d)
    else:  # guarded by plan validation; keeps this executor safe for direct callers.
        raise ValueError("divider type must be 'line' or 'path'")

    if len(parts) != 2 or any(part.is_empty or part.area <= 1e-6 for part in parts):
        raise ValueError(
            "divider does not divide the target into two non-empty regions"
        )

    divider_data = divider.model_dump(exclude_none=True)
    next_id = max(region.id for region in _all_regions(regions)) + 1
    children: list[VectorRegion] = []
    for index, footprint in enumerate(parts):
        shape = Shape("path", {"d": footprint.to_svg_d()})
        children.append(
            VectorRegion.from_shape(
                id=source.id if index == 0 else next_id,
                shape=shape,
                fill=source.fill,
                z=source.z + index * 0.001,
                footprint=footprint,
                raster=rasterize(footprint, source.raster.shape),
                source_label=source.source_label,
                color_hex=source.color_hex,
                source_regions=source.source_regions,
                coverage=source.coverage,
                diagnostics={
                    "split": {
                        "source": target_id,
                        "divider": divider_data,
                        "part": index + 1,
                    }
                },
            )
        )
    branch = VectorRegion.branch(
        id=source.id,
        children=tuple(children),
        z=source.z,
        raster=source.raster,
        footprint=source.footprint,
        fill=source.fill,
        source_label=source.source_label,
        color_hex=source.color_hex,
        drawing_id=source.drawing_id,
        source_regions=source.source_regions,
        diagnostics={
            "split": {"source": target_id, "divider": divider_data, "children": 2}
        },
    )
    return replace_working_leaf(regions, target_id, lambda _region: branch)


def _split_footprint_by_path(footprint: SkPath, d: str) -> tuple[SkPath, SkPath]:
    """Split one simple footprint using an open divider whose endpoints touch its boundary."""
    subpaths = _parse_subpaths(d)
    if (
        len(subpaths) != 1
        or not subpaths[0]
        or subpaths[0][0][0] != "M"
        or any(kind == "Z" for kind, _ in subpaths[0])
    ):
        raise ValueError("path divider must contain exactly one open SVG subpath")
    sampled_divider = _sample_subpath(subpaths[0], samples=32)
    if len(sampled_divider) < 2:
        raise ValueError("path divider must contain a drawable segment")
    rings = footprint.linearized_subpaths
    if len(rings) != 1:
        raise ValueError(
            "path divider currently requires a simple target without holes or disconnected islands"
        )
    ring = rings[0]
    start, start_segment, start_t, start_distance = _project_to_ring(
        sampled_divider[0], ring
    )
    end, end_segment, end_t, end_distance = _project_to_ring(sampled_divider[-1], ring)
    min_x, min_y, max_x, max_y = footprint.bounds
    endpoint_tolerance = max(
        1.0, min(3.0, math.hypot(max_x - min_x, max_y - min_y) * 0.005)
    )
    if max(start_distance, end_distance) > endpoint_tolerance:
        raise ValueError("path divider endpoints must lie on the target boundary")
    boundary = _forward_ring_arc(
        ring, end, end_segment, end_t, start, start_segment, start_t
    )
    divider_d = _snapped_open_divider_d(subpaths[0], start, end)
    boundary_d = " ".join(f"L{_fmt(x)} {_fmt(y)}" for x, y in boundary[1:])
    clip = SkPath.from_svg_d(f"{divider_d} {boundary_d} Z")
    first = footprint.intersection(clip)
    second = footprint.difference(first)
    return first, second


def _snapped_open_divider_d(
    tokens: Sequence[tuple[str, list[float]]],
    start: tuple[float, float],
    end: tuple[float, float],
) -> str:
    """Serialize one open divider while snapping only its boundary endpoints."""
    normalized = [(kind, list(values)) for kind, values in tokens]
    normalized[0][1][:2] = [start[0], start[1]]
    final_kind, final_values = normalized[-1]
    endpoint_offset = {"L": 0, "Q": 2, "C": 4, "A": 5}.get(final_kind)
    if endpoint_offset is None:
        raise ValueError("path divider must end with a drawable SVG command")
    final_values[endpoint_offset : endpoint_offset + 2] = [end[0], end[1]]
    parts: list[str] = []
    for kind, values in normalized:
        parts.append(kind + " ".join(_fmt(value) for value in values))
    return " ".join(parts)


def _project_to_ring(
    point: tuple[float, float], ring: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float], int, float, float]:
    px, py = point
    best: tuple[tuple[float, float], int, float, float] | None = None
    for index, start in enumerate(ring):
        end = ring[(index + 1) % len(ring)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        t = (
            0.0
            if length_sq <= 1e-12
            else max(
                0.0, min(1.0, ((px - start[0]) * dx + (py - start[1]) * dy) / length_sq)
            )
        )
        projected = (start[0] + t * dx, start[1] + t * dy)
        distance = math.hypot(px - projected[0], py - projected[1])
        if best is None or distance < best[3]:
            best = (projected, index, t, distance)
    assert best is not None
    return best


def _forward_ring_arc(
    ring: Sequence[tuple[float, float]],
    start: tuple[float, float],
    start_segment: int,
    start_t: float,
    end: tuple[float, float],
    end_segment: int,
    end_t: float,
) -> list[tuple[float, float]]:
    """Return the forward boundary arc, including projected start/end points."""
    if start_segment == end_segment and start_t <= end_t:
        return [start, end]
    points = [start]
    index = (start_segment + 1) % len(ring)
    while index != (end_segment + 1) % len(ring):
        points.append(ring[index])
        index = (index + 1) % len(ring)
    points.append(end)
    return points


def _all_regions(regions: Sequence[VectorRegion]) -> tuple[VectorRegion, ...]:
    all_regions: list[VectorRegion] = []
    for region in regions:
        all_regions.append(region)
        all_regions.extend(region.leaves()) if region.is_branch else None
    return tuple(all_regions)

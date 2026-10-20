"""Shared tangent smoothing and retained operation execution.

The optimizer selects targets; this module owns the existing bounded projection.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from .drawing import DrawingRegion
from .fit import _parse_path_commands, _smooth_quadratic_path_d
from .path_geometry import (
    CompoundPathGeometry, PathGeometry, PathGeometryWithCutouts, parse_svg_paths,
)
from .path_topology import component_paths, replace_component_paths


def _local_quadratic_deviation(original_d: str, candidate_d: str) -> float:
    """Bound smoothing drift in O(n), including exact Q-to-C elevation."""
    original = _parse_path_commands(original_d)
    candidate = _parse_path_commands(candidate_d)
    if original is None or candidate is None or len(original) != len(candidate):
        return float("inf")

    maximum = 0.0
    original_cursor: np.ndarray | None = None
    candidate_cursor: np.ndarray | None = None
    for (command, values), (candidate_command, candidate_values) in zip(
        original, candidate, strict=True
    ):
        if command == "M" and candidate_command == "M":
            original_cursor = np.asarray(values[:2], dtype=float)
            candidate_cursor = np.asarray(candidate_values[:2], dtype=float)
            maximum = max(
                maximum, float(np.linalg.norm(original_cursor - candidate_cursor))
            )
            continue
        if (
            command == "Q"
            and candidate_command == "C"
            and original_cursor is not None
            and candidate_cursor is not None
        ):
            # Sample the original quadratic against the elevated-and-adjusted
            # cubic.  The endpoints remain fixed; 17 samples bound the local
            # shape change without introducing a global solver.
            start = original_cursor
            control = np.asarray(values[:2], dtype=float)
            end = np.asarray(values[2:4], dtype=float)
            c_start = candidate_cursor
            first = np.asarray(candidate_values[:2], dtype=float)
            second = np.asarray(candidate_values[2:4], dtype=float)
            c_end = np.asarray(candidate_values[4:6], dtype=float)
            for t in np.linspace(0.0, 1.0, 17):
                quadratic = (
                    (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end
                )
                cubic = (
                    (1.0 - t) ** 3 * c_start
                    + 3.0 * (1.0 - t) ** 2 * t * first
                    + 3.0 * (1.0 - t) * t**2 * second
                    + t**3 * c_end
                )
                maximum = max(maximum, float(np.linalg.norm(quadratic - cubic)))
            original_cursor = end
            candidate_cursor = c_end
            continue
        if command != candidate_command or len(values) != len(candidate_values):
            return float("inf")
        if command == "Q":
            endpoint_shift = float(
                np.linalg.norm(
                    np.asarray(values[2:4]) - np.asarray(candidate_values[2:4])
                )
            )
            control_shift = float(
                np.linalg.norm(
                    np.asarray(values[:2]) - np.asarray(candidate_values[:2])
                )
            )
            maximum = max(maximum, endpoint_shift + control_shift / 2.0)
        elif values:
            maximum = max(
                maximum,
                float(
                    np.max(np.abs(np.asarray(values) - np.asarray(candidate_values)))
                ),
            )
        if command == "L":
            original_cursor = np.asarray(values[:2], dtype=float)
            candidate_cursor = np.asarray(candidate_values[:2], dtype=float)
        elif command == "Q":
            original_cursor = np.asarray(values[2:4], dtype=float)
            candidate_cursor = np.asarray(candidate_values[2:4], dtype=float)
        elif command == "C":
            original_cursor = np.asarray(values[4:6], dtype=float)
            candidate_cursor = np.asarray(candidate_values[4:6], dtype=float)
    return maximum


def smooth_region(region: DrawingRegion, *, max_boundary_error: float) -> DrawingRegion | None:
    """Project path tangents without changing retained component membership."""
    geometry = region.geometry
    if not isinstance(geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        return None
    paths = smooth_paths(component_paths(geometry), max_boundary_error=max_boundary_error)
    return None if paths is None else replace(region, geometry=replace_component_paths(geometry, paths))


def smooth_paths(paths: tuple[PathGeometry, ...], *, max_boundary_error: float) -> tuple[PathGeometry, ...] | None:
    """Project retained path tangents within one shared error budget.

    Partial control moves are intentionally forbidden: they make a second
    smoothing pass move the same join again.  The underlying path projection
    sets the requested tangent exactly and is idempotent.
    """
    if not paths:
        return None
    current_d = ' '.join(path.to_svg_d() for path in paths)
    candidate_d = _smooth_quadratic_path_d(current_d)
    if candidate_d == current_d:
        return None
    if _local_quadratic_deviation(current_d, candidate_d) > max_boundary_error:
        return None
    parsed = parse_svg_paths(candidate_d, region_id='smoothing',
                             path_ids=tuple(path.path.id for path in paths))
    updated_paths = []
    for original, updated in zip(paths, parsed, strict=True):
        # Tangent projection may elevate Q to C, but cannot add/remove commands.
        # Keep command addresses and every anchor across that exact correspondence.
        commands = tuple(replace(new, id=old.id) for old, new in zip(
            original.path.commands, updated.path.commands, strict=True))
        updated_paths.append(replace(original, path=replace(original.path, commands=commands)))
    return tuple(updated_paths)

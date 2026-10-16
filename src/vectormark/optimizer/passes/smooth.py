from __future__ import annotations

import numpy as np
from ...fit import Shape, _parse_path_commands, _smooth_quadratic_path_d
from ..automatic_replay import SmoothPathOperation, draft_input_ref, operation_intent
from ..framework import Proposal
from ..vector_region import VectorRegion, to_polygon


def _local_quadratic_deviation(original_d: str, candidate_d: str) -> float:
    """Bound smoothing drift in O(n), including exact Q-to-C elevation."""
    original = _parse_path_commands(original_d)
    candidate = _parse_path_commands(candidate_d)
    if original is None or candidate is None or len(original) != len(candidate):
        return float("inf")

    maximum = 0.0
    original_cursor: np.ndarray | None = None
    candidate_cursor: np.ndarray | None = None
    for (command, values), (candidate_command, candidate_values) in zip(original, candidate, strict=True):
        if command == "M" and candidate_command == "M":
            original_cursor = np.asarray(values[:2], dtype=float)
            candidate_cursor = np.asarray(candidate_values[:2], dtype=float)
            maximum = max(maximum, float(np.linalg.norm(original_cursor - candidate_cursor)))
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
                quadratic = (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end
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
            endpoint_shift = float(np.linalg.norm(np.asarray(values[2:4]) - np.asarray(candidate_values[2:4])))
            control_shift = float(np.linalg.norm(np.asarray(values[:2]) - np.asarray(candidate_values[:2])))
            maximum = max(maximum, endpoint_shift + control_shift / 2.0)
        elif values:
            maximum = max(maximum, float(np.max(np.abs(np.asarray(values) - np.asarray(candidate_values)))))
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


def _smoothed_shape(shape: Shape, *, max_boundary_error: float) -> Shape | None:
    """Return one exact tangent projection when it fits the error budget.

    Partial control moves are intentionally forbidden: they make a second
    smoothing pass move the same join again.  The underlying path projection
    sets the requested tangent exactly and is idempotent.
    """
    if shape.kind != "path":
        return None
    current_d = str(shape.params.get("d", ""))
    if not current_d:
        return None

    candidate = Shape("path", {**shape.params, "d": _smooth_quadratic_path_d(current_d)})
    if candidate == shape:
        return None
    if _local_quadratic_deviation(current_d, str(candidate.params["d"])) > max_boundary_error:
        return None
    return candidate


def smooth_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    max_error: float = 1.0,
) -> list[Proposal]:
    """Smooth tangent-continuous quadratic joins without moving the boundary too far."""
    del masks
    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None or obj.current.kind != "path":
            continue
        shape = _smoothed_shape(obj.current, max_boundary_error=max_error)
        if shape is None:
            continue
        target_ref = draft_input_ref(obj.id)
        action = SmoothPathOperation(target_ref, max_error)
        proposals.append(Proposal(
            (obj.id,),
            [obj.with_current(
                shape,
                footprint=to_polygon(shape),
            )],
            intent=operation_intent(action, (target_ref,), (obj.id,)),
        ))
    return proposals

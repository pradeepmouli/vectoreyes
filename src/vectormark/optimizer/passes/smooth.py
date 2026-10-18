from __future__ import annotations

import numpy as np
from ...drawing_smoothing import _smoothed_shape
from ..automatic_replay import SmoothPathOperation, draft_input_ref, operation_intent
from ..framework import Proposal
from ..vector_region import VectorRegion, to_polygon


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

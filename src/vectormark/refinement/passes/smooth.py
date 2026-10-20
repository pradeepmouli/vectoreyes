from __future__ import annotations

import numpy as np
from ...drawing_adapter import bind_path_input
from ..automatic_replay import (
    SmoothPathOperation, apply_smoothing_choice, draft_input_ref, operation_intent,
)
from ..framework import Proposal
from ..region_sample import RegionSample


def smooth_pass(
    objects: list[RegionSample],
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
        target_ref = draft_input_ref(obj.id)
        action = SmoothPathOperation(target_ref, max_error)
        selected = apply_smoothing_choice(action, bind_path_input(obj), required=False)
        if not selected:
            continue
        proposals.append(Proposal(
            (obj.id,),
            selected,
            intent=operation_intent(action, (target_ref,), (obj.id,)),
        ))
    return proposals

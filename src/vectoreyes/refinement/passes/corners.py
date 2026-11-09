"""Optional rounded-corner canonicalization pass."""

from __future__ import annotations

import numpy as np

from ...drawing_adapter import bind_path_input
from ..automatic_replay import apply_corner_normalization_choice
from ..automatic_replay import (
    CornerDiagnosticsObservation,
    NormalizeCornersOperation,
    draft_input_ref,
    operation_intent,
)
from ...corner_diagnostics import retained_corner_evidence
from ..framework import Proposal
from ..region_sample import RegionSample


def corner_normalize_pass(
    objects: list[RegionSample],
    masks: dict[int, np.ndarray],
    *,
    max_error: float,
    enabled: bool = False,
) -> list[Proposal]:
    """Attach corner diagnostics and optionally reduce each curved corner to one Q."""
    del masks
    proposals: list[Proposal] = []
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None or obj.current.kind != "path":
            continue
        d = str(obj.current.params.get("d", ""))
        authoritative, declared = retained_corner_evidence(d, obj.diagnostics.get("corners"))
        target_ref = draft_input_ref(obj.id)
        if enabled and authoritative:
            replacement, = apply_corner_normalization_choice(
                NormalizeCornersOperation(target_ref, max_error, declared), bind_path_input(obj))
            intent = operation_intent(
                NormalizeCornersOperation(target_ref, max_error, declared),
                (target_ref,),
                (obj.id,),
            )
        else:
            replacement = obj.with_diagnostics({"corners": declared})
            intent = CornerDiagnosticsObservation("", target_ref, declared)
        proposals.append(
            Proposal(
                (obj.id,),
                [replacement],
                intent=intent,
            )
        )
    return proposals

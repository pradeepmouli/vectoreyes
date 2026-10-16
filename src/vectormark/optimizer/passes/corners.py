"""Optional rounded-corner canonicalization pass."""

from __future__ import annotations

import numpy as np

from ...fit import Shape
from ..automatic_replay import (
    CornerDiagnosticsObservation,
    NormalizeCornersOperation,
    draft_input_ref,
    operation_intent,
)
from ..corners import (
    normalize_corners_path_d,
    recount_corner_diagnostics,
    retained_corner_declaration_matches,
)
from ..framework import Proposal
from ..vector_region import VectorRegion, to_polygon


def corner_normalize_pass(
    objects: list[VectorRegion],
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
        existing = obj.diagnostics.get("corners")
        declared = existing if retained_corner_declaration_matches(d, existing) else None
        target_ref = draft_input_ref(obj.id)
        if enabled and declared is not None:
            normalized_d, diagnostics = normalize_corners_path_d(
                d, max_error=max_error, diagnostics=declared
            )
            shape = Shape("path", {**obj.current.params, "d": normalized_d})
            replacement = obj.with_current(
                shape,
                footprint=to_polygon(shape),
                diagnostics={"corners": diagnostics},
            )
            intent = operation_intent(
                NormalizeCornersOperation(target_ref, max_error, declared),
                (target_ref,),
                (obj.id,),
            )
        else:
            diagnostics = declared if declared is not None else recount_corner_diagnostics(
                d,
                {
                    "path": d,
                    "anchors": [],
                    "spans": [],
                    "skipped": [{"reason": "missing_or_stale_provenance"}],
                },
            )
            replacement = obj.with_diagnostics({"corners": diagnostics})
            intent = CornerDiagnosticsObservation("", target_ref, diagnostics)
        proposals.append(
            Proposal(
                (obj.id,),
                [replacement],
                intent=intent,
            )
        )
    return proposals

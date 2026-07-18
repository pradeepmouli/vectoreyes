"""Optional rounded-corner canonicalization pass."""

from __future__ import annotations

import numpy as np

from ...fit import Shape
from ..corners import normalize_corners_path_d, path_corner_diagnostics
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
        declared = (
            existing
            if isinstance(existing, dict) and existing.get("path") == d
            else None
        )
        normalized_d, diagnostics = (
            normalize_corners_path_d(d, max_error=max_error, diagnostics=declared)
            if enabled and declared is not None
            else (d, declared if declared is not None else path_corner_diagnostics(d))
        )
        shape = Shape("path", {**obj.current.params, "d": normalized_d})
        proposals.append(
            Proposal(
                (obj.id,),
                [
                    obj.with_current(
                        shape,
                        footprint=to_polygon(shape),
                        diagnostics={"corners": diagnostics},
                    )
                ],
            )
        )
    return proposals

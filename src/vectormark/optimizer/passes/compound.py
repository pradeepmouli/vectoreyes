"""Expose actual painted components; transparent cutouts stay subtractive."""

import numpy as np

from ..framework import Proposal
from ..automatic_replay import (
    NormalizeRetainedTopologyOperation,
    draft_input_ref,
    operation_intent,
)
from ...drawing_topology import normalize_retained_topology, topology_path_correspondence
from ..vector_region import VectorRegion


def split_compound_pass(
    objects: list[VectorRegion], masks: dict[int, np.ndarray],
) -> list[Proposal]:
    normalized = normalize_retained_topology(objects, allow_external=True)
    changed = [(old, new) for old, new in zip(objects, normalized, strict=True) if old is not new]
    if not changed:
        return []
    # Source expansion and dependent clones are one atomic proposal.
    changed_ids = tuple(int(old.id) for old, _ in changed)
    roots = tuple(draft_input_ref(obj.id) for obj in objects)
    action = NormalizeRetainedTopologyOperation(
        roots,
        changed_ids,
        topology_path_correspondence(objects, normalized),
    )
    return [Proposal(
        changed_ids,
        [new for _, new in changed],
        intent=operation_intent(action, roots, changed_ids),
    )]

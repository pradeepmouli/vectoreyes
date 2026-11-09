from __future__ import annotations

import numpy as np

from ...skia_geometry import SkPath
from ...clone_detection import CloneDetectionInput, find_clone_matches
from ...drawing_adapter import bind_region_inputs
from ..automatic_replay import AssignCloneOperation, draft_input_ref, operation_intent
from ..framework import Proposal
from ..region_sample import RegionSample


def _clone_proposal(
    canonical: RegionSample, target: RegionSample, *,
    matrix: tuple[float, float, float, float, float, float], transformed: SkPath,
    symbolic: bool, family_size: int | None = None,
) -> Proposal:
    from ..automatic_replay import apply_relationship_choice
    source_ref, target_ref = draft_input_ref(canonical.id), draft_input_ref(target.id)
    action = AssignCloneOperation(source_ref, target_ref, matrix, symbolic, family_size)
    replacement = apply_relationship_choice(action,
        bind_region_inputs({'source': canonical, 'target': target}), footprint=transformed)[0]
    return Proposal((target.id,), [replacement],
        intent=operation_intent(action, (source_ref, target_ref), (target.id,)))


def clones_pass(
    objects: list[RegionSample], masks: dict[int, np.ndarray], *, symbolic: bool = True,
) -> list[Proposal]:
    """Adapt disposable discovery inputs to the shared geometric matcher."""
    del masks
    inputs = []
    by_id = {obj.id: obj for obj in objects}
    for obj in sorted(objects, key=lambda current: int(current.id)):
        if not obj.is_leaf or obj.current is None:
            continue
        symmetry = obj.diagnostics.get('symmetry')
        if obj.fill is None or obj.current.kind == 'use' or (isinstance(symmetry, dict) and symmetry.get('mode') == 'self'):
            continue
        flat = obj.footprint
        if isinstance(flat, SkPath) and not flat.is_empty:
            inputs.append(CloneDetectionInput(obj.id, flat))
    return [_clone_proposal(by_id[match.source_id], by_id[match.target_id],
        matrix=match.matrix, transformed=match.transformed, symbolic=symbolic,
        family_size=match.family_size) for match in find_clone_matches(tuple(inputs))]

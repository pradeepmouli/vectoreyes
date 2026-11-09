from __future__ import annotations

import numpy as np

from ...candidate import FlatFill
from ...occlusion_detection import OcclusionDetectionInput, discover_occlusion_intents
from ..automatic_replay import apply_occlusion_choice
from ..framework import Proposal
from ..region_sample import RegionSample


def _flat_hex(region: RegionSample) -> str | None:
    if isinstance(region.fill, FlatFill):
        return region.fill.hex
    return region.color_hex


def occlusion_pass(
    objects: list[RegionSample], masks: dict[int, np.ndarray], *, no_symmetry: bool = False,
) -> list[Proposal]:
    leaves = {int(obj.id): obj for obj in objects if obj.is_leaf}
    inputs = tuple(OcclusionDetectionInput(identity, obj.raster, color, obj.z, obj.coverage)
        for identity, obj in sorted(leaves.items())
        if (color := _flat_hex(obj)) is not None and obj.raster.size)
    if len(inputs) < 2:
        return []
    shape_hw = next(iter(masks.values())).shape if masks else inputs[0].raster.shape
    proposals = []
    for intent in discover_occlusion_intents(inputs, shape_hw=shape_hw,
            next_id=max((int(obj.id) for obj in objects), default=-1)+1, no_symmetry=no_symmetry):
        sources = {ref: leaves[int(ref.ancestry[-1])] for ref in intent.action.component}
        branch, = apply_occlusion_choice(intent.action, sources.__getitem__)
        proposals.append(Proposal(tuple(int(ref.ancestry[-1]) for ref in intent.action.component),
            [branch], intent=intent))
    return proposals

from __future__ import annotations

import numpy as np

from ...drawing_adapter import publish_seam_shape, replace_private_leaves
from ...drawing_seams import _SEAM_TOL, _JUNCTION_TOL
from ...quality import GeometryPolicy
from ...seam_detection import SeamDetectionInput, discover_seam_choices
from ..automatic_replay import PassInputRegionRef
from ..framework import Proposal
from ..region_sample import RegionSample
from ..symmetry_materialize import index_region_paths


def seams_pass(
    objects: list[RegionSample], masks: dict[int, np.ndarray], *,
    epsilon: float = 1.0, max_error: float = 1.0, cubic: bool = False,
    post_symmetry: bool = False, allow_underlap: bool = True,
) -> list[Proposal]:
    owners = {int(owner.id): owner for owner in objects}
    paths = index_region_paths(objects)
    inputs = []
    for path, region in paths.items():
        owner = owners[path[0]]
        source_path = None
        if region.current is not None and region.current.kind == 'use':
            source_id = int(region.current.params.get('href_obj_id', -1))
            source_path = next((key for key, item in reversed(tuple(paths.items()))
                if key[0] == path[0] and item.is_leaf and int(item.id) == source_id), None)
        inputs.append(SeamDetectionInput(PassInputRegionRef('', '', tuple(path)),
            region.current, region.footprint,
            (float(owner.z), float(region.z), int(owner.id), int(region.id)),
            owner.footprint, region.diagnostics, source_path))
    proposals = []
    for choice in discover_seam_choices(tuple(inputs), epsilon=epsilon, max_error=max_error,
            cubic=cubic, post_symmetry=post_symmetry, allow_underlap=allow_underlap):
        leaves = {path: publish_seam_shape(paths[path], update.shape,
            update.paired_with, update.candidate, selected=update.selected)
            for path, update in choice.updates.items()}
        replacements = [replace_private_leaves(owners[identity], leaves) for identity in choice.owner_ids]
        from ..topology_binding import normalize_working_topology
        try:
            normalize_working_topology(replacements, allow_external=True)
        except ValueError:
            continue
        proposals.append(Proposal(choice.owner_ids, replacements,
            region_geometry_policy=GeometryPolicy(max_boundary_error_px=max(_SEAM_TOL, _JUNCTION_TOL),
                max_perimeter_residual_error_px=max(_SEAM_TOL, _JUNCTION_TOL), preserve_topology=False),
            intent=choice.intent))
    return proposals

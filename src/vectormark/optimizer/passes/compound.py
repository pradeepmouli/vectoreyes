"""Expose actual painted components; transparent cutouts stay subtractive."""

import numpy as np

from ..framework import Proposal
from ..automatic_replay import (
    NormalizeRetainedTopologyOperation,
    PassInputRegionRef,
    TopologyPathCorrespondence,
    draft_input_ref,
    operation_intent,
)
from ...path_geometry import parse_svg_paths
from .seams import seam_path_revision
from ..symmetry_materialize import index_region_paths
from ..topology import normalize_retained_topology
from ..vector_region import VectorRegion


def topology_path_correspondence(
    before: list[VectorRegion], after: tuple[VectorRegion, ...]
) -> tuple[TopologyPathCorrespondence, ...]:
    """Bind every surviving path to its exact normalized owner and ordinal."""
    def paths(regions):
        result: dict[str, list[tuple[tuple[int, ...], int, str, int]]] = {}
        for ancestry, region in index_region_paths(regions).items():
            if not region.is_leaf or region.current.kind != "path":
                continue
            d = str(region.current.params.get("d", ""))
            address = region.drawing_id or "-".join(str(part) for part in ancestry)
            parsed = parse_svg_paths(
                d,
                region_id=address,
                path_ids=region.current.params.get("path_ids"),
            )
            for index, path in enumerate(parsed):
                result.setdefault(path.path.id, []).append((
                    tuple(ancestry),
                    index,
                    seam_path_revision(d, index),
                    len(path.path.commands),
                ))
        return result

    inputs = paths(before)
    outputs = paths(after)
    records: list[TopologyPathCorrespondence] = []
    for path_id, input_values in inputs.items():
        output_values = outputs.get(path_id, [])
        if not output_values:
            continue
        if len(input_values) != 1 or len(output_values) != 1:
            raise ValueError("retained topology path ownership is ambiguous")
        input_ancestry, input_index, input_revision, input_count = input_values[0]
        output_ancestry, output_index, output_revision, output_count = output_values[0]
        if input_revision != output_revision or input_count != output_count:
            raise ValueError("retained topology rewrote a surviving path")
        records.append(TopologyPathCorrespondence(
            target=PassInputRegionRef("", "", input_ancestry),
            input_subpath_index=input_index,
            input_path_id=path_id,
            input_path_revision=input_revision,
            output_ancestry=output_ancestry,
            output_subpath_index=output_index,
            output_path_id=path_id,
            output_path_revision=output_revision,
            command_count=input_count,
        ))
    return tuple(records)


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

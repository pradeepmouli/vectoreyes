from __future__ import annotations

import numpy as np

from ...candidate import FlatFill
from ...drawing_adapter import primitive_geometry_from_shape
from ...drawing_occlusion import (
    _shape_from_scene_primitive, materialize_occlusion_child, materialize_occlusion_branch,
)
from ...fit import Shape
from ...occlusion import ScenePrimitive, reconstruct_scene, region_adjacency
from ...symmetry import detect_axis
from ...types import Region
from ..automatic_replay import (
    AnnulusPrimitive,
    AutomaticIntent,
    CompletedOcclusionPrimitive,
    IntersectionLens,
    NamedRegionOutput,
    ReconstructOcclusionOperation,
    RetainedOcclusionSource,
    draft_input_ref,
)
from ..framework import Proposal
from ..vector_region import VectorRegion


def _flat_hex(region: VectorRegion) -> str | None:
    if isinstance(region.fill, FlatFill):
        return region.fill.hex
    return region.color_hex


def _as_trace_region(region: VectorRegion) -> Region | None:
    color_hex = _flat_hex(region)
    if color_hex is None or region.raster.size == 0:
        return None
    return Region(
        label=int(region.id),
        mask=np.asarray(region.raster, dtype=bool).copy(),
        color_hex=color_hex,
        coverage=region.coverage,
    )


def _typed_occlusion_primitive(item: ScenePrimitive):
    if item.kind == "annulus":
        return AnnulusPrimitive(
            (float(item.params["cx"]), float(item.params["cy"])),
            float(item.params["r_outer"]),
            float(item.params["r_inner"]),
        )
    return primitive_geometry_from_shape(_shape_from_scene_primitive(item))


def occlusion_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    no_symmetry: bool = False,
) -> list[Proposal]:
    leaves = [obj for obj in sorted(objects, key=lambda current: int(current.id)) if obj.is_leaf]
    trace_regions = [_as_trace_region(obj) for obj in leaves]
    region_by_id = {region.label: region for region in trace_regions if region is not None}
    if len(region_by_id) < 2:
        return []

    shape_hw = next(iter(masks.values())).shape if masks else next(iter(region_by_id.values())).mask.shape
    silhouette = np.zeros(shape_hw, dtype=bool)
    for region in region_by_id.values():
        if region.mask.shape == shape_hw:
            silhouette |= region.mask
    axis = None if no_symmetry else detect_axis(silhouette)
    adjacency = region_adjacency(list(region_by_id.values()))
    seen: set[int] = set()
    proposals: list[Proposal] = []
    next_id = max([int(obj.id) for obj in objects], default=-1) + 1

    for root_id in sorted(region_by_id):
        if root_id in seen:
            continue
        component_ids: set[int] = set()
        stack = [root_id]
        while stack:
            obj_id = stack.pop()
            if obj_id in component_ids:
                continue
            component_ids.add(obj_id)
            stack.extend(adjacency[obj_id] - component_ids)
        seen.update(component_ids)
        if len(component_ids) < 2:
            continue

        component_regions = [region_by_id[obj_id] for obj_id in sorted(component_ids)]
        reconstructed, remaining = reconstruct_scene(component_regions, axis, shape_hw)
        if not reconstructed:
            continue
        remaining_ids = {int(region.label) for region in remaining}
        consumed_ids = tuple(sorted(component_ids - remaining_ids))
        if len(consumed_ids) < 2:
            continue

        base_z = min(float(obj.z) for obj in leaves if int(obj.id) in consumed_ids)
        children: list[VectorRegion] = []
        child_records: list[tuple[ScenePrimitive | Shape, int, str]] = []
        primitive_id_index = 0
        for item_index, item in enumerate(reconstructed):
            if isinstance(item, ScenePrimitive) and primitive_id_index < len(consumed_ids):
                child_id = consumed_ids[primitive_id_index]
                primitive_id_index += 1
            else:
                child_id = next_id
                next_id += 1
            children.append(materialize_occlusion_child(item, id=child_id, shape_hw=shape_hw, z_offset=base_z))
            child_records.append((item, child_id, f"child_{item_index}"))

        branch = materialize_occlusion_branch(
            [obj for obj in leaves if int(obj.id) in consumed_ids], children,
            branch_id=consumed_ids[0], base_z=base_z, shape_hw=shape_hw,
        )
        refs_by_id = {obj_id: draft_input_ref(obj_id) for obj_id in consumed_ids}
        primitive_output_by_source: dict[int, str] = {}
        specs = []
        provenance_complete = True
        for item, child_id, output_name in child_records:
            source_label = (
                item.source_label
                if isinstance(item, ScenePrimitive)
                else item.params.get("source_label")
            )
            if source_label is None or int(source_label) not in refs_by_id:
                provenance_complete = False
                break
            source_ref = refs_by_id[int(source_label)]
            if isinstance(item, ScenePrimitive):
                primitive_output_by_source[int(source_label)] = output_name
                specs.append(CompletedOcclusionPrimitive(
                    source_ref,
                    output_name,
                    child_id,
                    _typed_occlusion_primitive(item),
                    int(item.z),
                ))
            elif "lens_of_labels" in item.params:
                left_label, right_label = (int(value) for value in item.params["lens_of_labels"])
                left_output = primitive_output_by_source.get(left_label)
                right_output = primitive_output_by_source.get(right_label)
                if left_output is None or right_output is None:
                    provenance_complete = False
                    break
                specs.append(IntersectionLens(
                    source_ref,
                    output_name,
                    child_id,
                    left_output,
                    right_output,
                    int(item.params.get("z", 0)),
                ))
            else:
                specs.append(RetainedOcclusionSource(
                    source_ref,
                    output_name,
                    child_id,
                    int(item.params.get("z", -1)),
                ))
        if not provenance_complete:
            raise ValueError(
                "occlusion reconstruction lacks exact source/output provenance"
            )
        component_refs = tuple(refs_by_id[obj_id] for obj_id in consumed_ids)
        action = ReconstructOcclusionOperation(
            component_refs,
            int(branch.id),
            float(base_z),
            tuple(int(value) for value in shape_hw),
            tuple(specs),
        )
        intent = AutomaticIntent(
            "",
            action,
            component_refs,
            (
                NamedRegionOutput("branch", (int(branch.id),)),
                *(NamedRegionOutput(name, (int(branch.id), child_id)) for _, child_id, name in child_records),
            ),
        )
        proposals.append(Proposal(consumed_ids, [branch], intent=intent))

    return proposals

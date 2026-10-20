"""Raster-evidence occlusion discovery; selected graphs are independent of a scene codec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import Shape
from .occlusion import ScenePrimitive, reconstruct_scene, region_adjacency
from .primitive_fitting import primitive_geometry_from_shape
from .symmetry import detect_axis
from .types import CoverageField, Region
from .refinement.automatic_replay import (
    AnnulusPrimitive, AutomaticIntent, CompletedOcclusionPrimitive, IntersectionLens,
    NamedRegionOutput, ReconstructOcclusionOperation, RetainedOcclusionSource, draft_input_ref,
)


@dataclass(frozen=True)
class OcclusionDetectionInput:
    id: int
    raster: np.ndarray
    color_hex: str
    z: float
    coverage: CoverageField | np.ndarray | None = None


def _typed_occlusion_primitive(item: ScenePrimitive):
    if item.kind == "annulus":
        return AnnulusPrimitive(
            (float(item.params["cx"]), float(item.params["cy"])),
            float(item.params["r_outer"]),
            float(item.params["r_inner"]),
        )
    return primitive_geometry_from_shape(Shape(item.kind, dict(item.params)))


def discover_occlusion_intents(
    inputs: tuple[OcclusionDetectionInput, ...], *, shape_hw: tuple[int, int],
    next_id: int, no_symmetry: bool = False,
) -> tuple[AutomaticIntent, ...]:
    """Choose completed primitives and exact source/output dependencies once."""
    region_by_id = {}
    z_by_id = {}
    for item in sorted(inputs, key=lambda current: current.id):
        if not item.raster.size:
            continue
        coverage = item.coverage
        if isinstance(coverage, np.ndarray):
            coverage = coverage.copy()
        region_by_id[item.id] = Region(item.id, np.asarray(item.raster, dtype=bool).copy(),
            item.color_hex, coverage)
        z_by_id[item.id] = float(item.z)
    if len(region_by_id) < 2:
        return ()
    silhouette = np.zeros(shape_hw, dtype=bool)
    for region in region_by_id.values():
        if region.mask.shape == shape_hw:
            silhouette |= region.mask
    axis = None if no_symmetry else detect_axis(silhouette)
    adjacency = region_adjacency(list(region_by_id.values()))
    seen: set[int] = set()
    proposals: list[AutomaticIntent] = []

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

        base_z = min(z_by_id[identity] for identity in consumed_ids)
        child_records: list[tuple[ScenePrimitive | Shape, int, str]] = []
        primitive_id_index = 0
        for item_index, item in enumerate(reconstructed):
            if isinstance(item, ScenePrimitive) and primitive_id_index < len(consumed_ids):
                child_id = consumed_ids[primitive_id_index]
                primitive_id_index += 1
            else:
                child_id = next_id
                next_id += 1
            child_records.append((item, child_id, f"child_{item_index}"))

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
            consumed_ids[0],
            float(base_z),
            tuple(int(value) for value in shape_hw),
            tuple(specs),
        )
        intent = AutomaticIntent(
            "", action, component_refs,
            (NamedRegionOutput("branch", (action.branch_id,)),
             *(NamedRegionOutput(name, (action.branch_id, child_id)) for _, child_id, name in child_records)),
        )
        proposals.append(intent)
    return tuple(proposals)

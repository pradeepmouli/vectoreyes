"""Recorded occlusion reconstruction and shared child/branch publication.

Scene discovery stays in the optimizer; this module validates and applies the
selected primitive/source/lens dependencies without detecting a different scene.
"""

from __future__ import annotations

import numpy as np
from .candidate import FlatFill
from .drawing_adapter import shape_from_primitive_geometry
from .fit import Shape
from .occlusion import ScenePrimitive, _retained_region_shape, intersection_lens_d
from .types import Region
from .optimizer.automatic_replay import (
    AnnulusPrimitive,
    CompletedOcclusionPrimitive,
    IntersectionLens,
    RetainedOcclusionSource,
    ReconstructOcclusionOperation,
    RegionResolver,
)
from .optimizer.gate import rasterize
from .optimizer.vector_region import VectorRegion, to_polygon


def _shape_from_scene_primitive(primitive: ScenePrimitive) -> Shape:
    return Shape(primitive.kind, dict(primitive.params))


def materialize_occlusion_child(
    item: ScenePrimitive | Shape,
    *,
    id: int,
    shape_hw: tuple[int, int],
    z_offset: float = 0.0,
) -> VectorRegion:
    if isinstance(item, ScenePrimitive):
        shape = _shape_from_scene_primitive(item)
        fill = FlatFill(item.color_hex)
        z = z_offset + float(item.z)
        raster = rasterize(to_polygon(shape), shape_hw)
        color_hex = item.color_hex
    else:
        shape = item
        color_hex = str(item.params["color_hex"])
        fill = FlatFill(color_hex)
        z = z_offset + float(item.params.get("z", 0))
        raster = rasterize(to_polygon(shape), shape_hw)

    return VectorRegion(
        id=id,
        current=shape,
        fill=fill,
        z=z,
        raster=raster,
        footprint=to_polygon(shape),
        color_hex=color_hex,
        diagnostics={"occlusion": {"reconstructed_child": True}},
    )


def materialize_occlusion_branch(
    sources: list[VectorRegion],
    children: list[VectorRegion],
    *,
    branch_id: int,
    base_z: float,
    shape_hw: tuple[int, int],
) -> VectorRegion:
    """Publish completed children with their source raster and paint order."""
    rasters = [
        np.asarray(source.raster, dtype=bool)
        for source in sources
        if source.raster.size
    ]
    branch_raster = (
        np.logical_or.reduce(rasters) if rasters else np.zeros(shape_hw, dtype=bool)
    )
    return VectorRegion.branch(
        id=branch_id,
        children=children,
        z=base_z,
        raster=branch_raster,
        diagnostics={
            "occlusion": {
                "accepted": True,
                "consumed_ids": [int(source.id) for source in sources],
                "children": len(children),
            }
        },
    )


def apply_occlusion_reconstruction(
    action: ReconstructOcclusionOperation,
    resolver: RegionResolver,
) -> list[VectorRegion]:
    """Apply recorded children after validating all dependency relationships."""
    output_positions: dict[str, int] = {}
    for index, child in enumerate(action.children):
        if child.output_name in output_positions:
            raise ValueError(f"duplicate occlusion child output: {child.output_name!r}")
        output_positions[child.output_name] = index
    dependencies = {
        child.output_name: (child.left_output, child.right_output)
        for child in action.children
        if isinstance(child, IntersectionLens)
    }
    for output_name, inputs in dependencies.items():
        for dependency in inputs:
            if dependency not in output_positions:
                raise ValueError(
                    f"occlusion child {output_name!r} has absent dependency {dependency!r}"
                )
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(output_name: str) -> None:
        if output_name in visiting:
            raise ValueError("occlusion child dependency cycle")
        if output_name in visited:
            return
        visiting.add(output_name)
        for dependency in dependencies.get(output_name, ()):
            visit(dependency)
        visiting.remove(output_name)
        visited.add(output_name)

    for output_name in dependencies:
        visit(output_name)
    for output_name, inputs in dependencies.items():
        if any(
            output_positions[dependency] >= output_positions[output_name]
            for dependency in inputs
        ):
            raise ValueError(f"occlusion child {output_name!r} has forward dependency")

    children: list[VectorRegion] = []
    by_name: dict[str, VectorRegion] = {}
    source_id_by_name: dict[str, int] = {}
    for child in action.children:
        source = resolver.resolve(child.source)
        color_hex = (
            source.fill.hex if isinstance(source.fill, FlatFill) else source.color_hex
        )
        if color_hex is None:
            raise ValueError("occlusion replay source has no flat paint")
        if isinstance(child, CompletedOcclusionPrimitive):
            if isinstance(child.primitive, AnnulusPrimitive):
                shape = Shape(
                    "annulus",
                    {
                        "cx": child.primitive.center[0],
                        "cy": child.primitive.center[1],
                        "r_outer": child.primitive.outer_radius,
                        "r_inner": child.primitive.inner_radius,
                    },
                )
            else:
                shape = shape_from_primitive_geometry(child.primitive)
            item = ScenePrimitive(
                shape.kind, dict(shape.params), color_hex, child.z, int(source.id)
            )
        elif isinstance(child, RetainedOcclusionSource):
            if source.raster.size == 0:
                raise ValueError(
                    "retained occlusion replay source has no raster evidence"
                )
            item = _retained_region_shape(
                Region(
                    label=int(source.id),
                    mask=np.asarray(source.raster, dtype=bool).copy(),
                    color_hex=color_hex,
                    coverage=source.coverage,
                )
            )
            item.params["z"] = child.z
        else:
            if child.left_output not in by_name or child.right_output not in by_name:
                raise ValueError(
                    "occlusion lens has an absent or forward child dependency"
                )
            left = by_name[child.left_output].current
            right = by_name[child.right_output].current
            if (
                left is None
                or right is None
                or left.kind != "circle"
                or right.kind != "circle"
            ):
                raise ValueError("occlusion lens operands must be completed circles")
            d = intersection_lens_d(left.params, right.params)
            if d is None:
                raise ValueError("recorded occlusion lens operands no longer intersect")
            item = Shape(
                "path",
                {
                    "d": d,
                    "color_hex": color_hex,
                    "z": child.z,
                    "source_label": int(source.id),
                    "lens_of_labels": (
                        source_id_by_name[child.left_output],
                        source_id_by_name[child.right_output],
                    ),
                },
            )
        materialized = materialize_occlusion_child(
            item,
            id=child.output_id,
            shape_hw=action.shape_hw,
            z_offset=action.base_z,
        )
        children.append(materialized)
        by_name[child.output_name] = materialized
        source_id_by_name[child.output_name] = int(source.id)
    sources = [resolver.resolve(ref) for ref in action.component]
    return [
        materialize_occlusion_branch(
            sources,
            children,
            branch_id=action.branch_id,
            base_z=action.base_z,
            shape_hw=action.shape_hw,
        )
    ]

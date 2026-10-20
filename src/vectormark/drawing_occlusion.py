"""Recorded occlusion reconstruction and shared child/branch publication.

Scene discovery stays in the optimizer; this module validates and applies the
selected primitive/source/lens dependencies without detecting a different scene.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from collections.abc import Mapping
from .drawing import DrawingRegion
from .drawing_fill import FlatFill as DrawingFlatFill
from .path_geometry import (
    CirclePrimitive,
    PrimitiveGeometry,
    geometry_from_svg_d,
    parse_svg_paths,
)
from .occlusion import intersection_lens_d
from .optimizer.automatic_replay import (
    AnnulusPrimitive,
    CompletedOcclusionPrimitive,
    IntersectionLens,
    RetainedOcclusionSource,
    ReconstructOcclusionOperation,
)


@dataclass(frozen=True)
class OcclusionSource:
    """Source observation, separate from selected reconstruction geometry."""

    id: int
    region_id: str
    color_hex: str
    raster: np.ndarray

    def __post_init__(self):
        pixels = np.asarray(self.raster, dtype=bool).copy()
        pixels.setflags(write=False)
        object.__setattr__(self, "raster", pixels)


def validate_occlusion_children(action: ReconstructOcclusionOperation) -> None:
    """Prove child dependency order before touching any source observation."""
    positions = {}
    for index, child in enumerate(action.children):
        if child.output_name in positions:
            raise ValueError(f"duplicate occlusion child output: {child.output_name!r}")
        positions[child.output_name] = index
    dependencies = {
        child.output_name: (child.left_output, child.right_output)
        for child in action.children
        if isinstance(child, IntersectionLens)
    }
    for name, inputs in dependencies.items():
        for dependency in inputs:
            if dependency not in positions:
                raise ValueError(
                    f"occlusion child {name!r} has absent dependency {dependency!r}"
                )
    active, visited = set(), set()

    def visit(name):
        if name in active:
            raise ValueError("occlusion child dependency cycle")
        if name in visited:
            return
        active.add(name)
        for dependency in dependencies.get(name, ()):
            visit(dependency)
        active.remove(name)
        visited.add(name)

    for name in dependencies:
        visit(name)
    for name, inputs in dependencies.items():
        if any(positions[dependency] >= positions[name] for dependency in inputs):
            raise ValueError(f"occlusion child {name!r} has forward dependency")


def reconstruct_occlusion(
    action: ReconstructOcclusionOperation,
    sources: Mapping[object, OcclusionSource],
    *,
    region_id: str,
) -> DrawingRegion:
    """Execute the selected primitive/source/lens graph on canonical geometry."""
    from .emit import _annulus_path_d
    from .occlusion import retained_region_path_data
    from .drawing_topology import publish_region_paths

    validate_occlusion_children(action)
    children, by_name = [], {}
    for child in action.children:
        source = sources[child.source]
        address = f"{region_id}-{child.output_id}"
        paths = None
        if isinstance(child, CompletedOcclusionPrimitive):
            geometry = child.primitive
            if isinstance(geometry, AnnulusPrimitive):
                geometry = geometry_from_svg_d(
                    _annulus_path_d(
                        *geometry.center, geometry.outer_radius, geometry.inner_radius
                    ),
                    region_id=address,
                    fill_rule="evenodd",
                )
        elif isinstance(child, RetainedOcclusionSource):
            if not source.raster.size:
                raise ValueError(
                    "retained occlusion replay source has no raster evidence"
                )
            paths = parse_svg_paths(
                retained_region_path_data(source.raster), region_id=address
            )
            geometry = paths[0]
        else:
            left, right = (
                by_name[child.left_output].geometry,
                by_name[child.right_output].geometry,
            )
            if not (
                isinstance(left, PrimitiveGeometry)
                and isinstance(right, PrimitiveGeometry)
                and isinstance(left.primitive, CirclePrimitive)
                and isinstance(right.primitive, CirclePrimitive)
            ):
                raise ValueError("occlusion lens operands must be completed circles")
            left, right = left.primitive, right.primitive
            d = intersection_lens_d(
                {"cx": left.center[0], "cy": left.center[1], "r": left.radius},
                {"cx": right.center[0], "cy": right.center[1], "r": right.radius},
            )
            if d is None:
                raise ValueError("recorded occlusion lens operands no longer intersect")
            geometry = geometry_from_svg_d(d, region_id=address)
        selected = DrawingRegion(
            address,
            geometry,
            DrawingFlatFill(source.color_hex),
            z=action.base_z + child.z,
            source_region_ids=(source.region_id,),
            diagnostics={"occlusion": {"reconstructed_child": True}},
        )
        if paths is not None:
            selected = publish_region_paths(
                selected, paths, fill_rule="nonzero", defer_overlaps=True
            )
        children.append(selected)
        by_name[child.output_name] = selected
    return DrawingRegion(
        region_id,
        children=tuple(children),
        z=action.base_z,
        source_region_ids=tuple(
            dict.fromkeys(sources[ref].region_id for ref in action.component)
        ),
        diagnostics={
            "occlusion": {
                "accepted": True,
                "consumed_ids": [sources[ref].id for ref in action.component],
                "children": len(children),
            }
        },
    )

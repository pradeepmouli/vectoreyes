"""Canonical region-local actions shared by drawing orchestration.

These operations do not own scene storage, versioning or evidence selection.
The engine resolves retained evidence and settings before invoking them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .drawing import Drawing, DrawingRegion
from .drawing_actions import ActionSettings, FitPathDrawingAction, FitPrimitiveAction, SetFillAction, SetZOrderAction
from .drawing_plan import PlanValidationError, _expand_merge_regions
from .drawing_fill import deserialize_fill
from .primitive_fitting import fit_primitive_geometry
from .skia_geometry import SkPath
from . import drawing_topology, path_fitting


def _require_leaf_target(region: DrawingRegion, target_id: str) -> None:
    if target_id != region.id:
        raise ValueError(f"action target {target_id!r} does not match region {region.id!r}")
    if not region.is_leaf:
        raise ValueError("region-local action requires a drawing-region leaf")


def fit_region_primitive(
    region: DrawingRegion,
    action: FitPrimitiveAction,
    *,
    footprint: SkPath,
    settings: ActionSettings,
) -> DrawingRegion:
    """Replace a leaf's geometry using resolved settings and retained evidence.

    Paint is retained here so the orchestration layer can resolve it through its
    shared changed-geometry paint pass, rather than fitting paint twice.
    """
    _require_leaf_target(region, action.target)
    geometry = fit_primitive_geometry(
        footprint, action.type, epsilon=settings.epsilon, target_id=region.id,
    )
    return replace(region, geometry=geometry, diagnostics={
        **region.diagnostics, "geometry": {"explicit": action.type},
    })


def set_region_fill(region: DrawingRegion, action: SetFillAction) -> DrawingRegion:
    """Assign canonical paint without modifying geometry or its annotations."""
    _require_leaf_target(region, action.target)
    return replace(region, fill=deserialize_fill(action.fill.model_dump()))


def fit_region_path(
    region: DrawingRegion, action: FitPathDrawingAction, *,
    raster: np.ndarray, settings: ActionSettings,
) -> DrawingRegion:
    """Fit retained contours and publish a canonical leaf or component subtree."""
    _require_leaf_target(region, action.target)
    paths = path_fitting.fit_raster_paths(raster, region_id=region.id, settings=settings)
    from .corner_provenance import bind_matching_prefit_corners
    paths, corners = bind_matching_prefit_corners(paths,
        d=' '.join(path.to_svg_d() for path in paths), evidence=region.diagnostics.get('corners'))
    selected = replace(region, diagnostics={**region.diagnostics, 'geometry': {'explicit': 'path'},
        'corners': corners})
    return drawing_topology.publish_region_paths(selected, paths, fill_rule='evenodd')


def set_drawing_z_order(drawing: Drawing, action: SetZOrderAction) -> Drawing:
    """Assign a complete leaf paint order without modifying the region tree."""
    leaves = tuple(leaf for root in drawing.regions for leaf in root.leaves())
    by_id = {leaf.id: leaf for leaf in leaves}
    expanded = _expand_merge_regions(action.targets, set(by_id), "/ops")
    if set(expanded) != set(by_id):
        raise PlanValidationError("/ops", "z order must include every retained leaf exactly once")
    positions = {leaf.id: index for index, leaf in enumerate(leaves)}
    ordered = tuple(
        member for selected in action.targets
        for member in sorted(
            (key for key in expanded if key == selected or key.startswith(f"{selected}-")),
            key=lambda key: (by_id[key].z, positions[key]),
        )
    )
    z_by_id = {region_id: z for z, region_id in enumerate(ordered)}
    def visit(region: DrawingRegion) -> DrawingRegion:
        if region.is_leaf:
            return replace(region, z=z_by_id[region.id])
        return replace(region, children=tuple(visit(child) for child in region.children))
    return replace(drawing, regions=tuple(visit(root) for root in drawing.regions))

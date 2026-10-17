"""Canonical region-local actions shared by drawing orchestration.

These operations do not own scene storage, versioning or evidence selection.
The engine resolves retained evidence and settings before invoking them.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .drawing import DrawingRegion
from .drawing_actions import ActionSettings, FitPrimitiveAction, SetFillAction
from .drawing_fill import deserialize_fill
from .primitive_fitting import fit_primitive_geometry


def _require_leaf_target(region: DrawingRegion, target_id: str) -> None:
    if target_id != region.id:
        raise ValueError(f"action target {target_id!r} does not match region {region.id!r}")
    if not region.is_leaf:
        raise ValueError("region-local action requires a drawing-region leaf")


def fit_region_primitive(
    region: DrawingRegion,
    action: FitPrimitiveAction,
    *,
    raster: np.ndarray,
    settings: ActionSettings,
) -> DrawingRegion:
    """Replace a leaf's geometry using resolved settings and retained evidence.

    Paint is retained here so the orchestration layer can resolve it through its
    shared changed-geometry paint pass, rather than fitting paint twice.
    """
    _require_leaf_target(region, action.target)
    geometry = fit_primitive_geometry(
        raster, action.type, epsilon=settings.epsilon, target_id=region.id,
    )
    return replace(region, geometry=geometry, diagnostics={
        **region.diagnostics, "geometry": {"explicit": action.type},
    })


def set_region_fill(region: DrawingRegion, action: SetFillAction) -> DrawingRegion:
    """Assign canonical paint without modifying geometry or its annotations."""
    _require_leaf_target(region, action.target)
    return replace(region, fill=deserialize_fill(action.fill.model_dump()))

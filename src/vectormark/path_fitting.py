"""Whole-region path fitting, independent of the working region scene."""

from __future__ import annotations

import numpy as np

from .contour import region_contours
from .drawing_actions import ActionSettings
from .fit import fit_path
from .path_geometry import PathGeometry, parse_svg_paths


def fit_raster_paths(
    raster: np.ndarray, *, region_id: str, settings: ActionSettings,
) -> tuple[PathGeometry, ...]:
    """Fit retained contours in source order without assigning filled topology.

    These are the same numeric fits used by whole-region plan actions. Separate
    components and cutouts remain ordered paths for the shared topology pass;
    current displayed geometry never replaces the retained raster evidence.
    """
    if not region_id:
        raise ValueError("path fitting requires a non-empty region ID")
    mask = np.asarray(raster)
    if mask.ndim != 2:
        raise ValueError("path fitting requires a two-dimensional raster")
    contours = [contour for contour in region_contours(mask) if len(contour) >= 3]
    if not contours:
        raise ValueError(f"cannot fit an empty drawing region {region_id!r}")
    paths = [str(fit_path(
        contour, epsilon=settings.epsilon, max_error=settings.max_error,
        cubic=False, progressive=settings.fit_strategy != "quadratic",
        progressive_allow_lines=settings.fit_strategy == "progressive_allow_lines",
        prefer_simple_curves=False,
    ).params["d"]) for contour in contours]
    return parse_svg_paths(" ".join(paths), region_id=region_id)

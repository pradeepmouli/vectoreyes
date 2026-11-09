"""Disposable refinement paint inputs; all paint policy lives in drawing_paint."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ..candidate import FlatFill, LinearGradientFill, RadialGradientFill
from ..drawing import Drawing
from ..drawing_adapter import (
    drawing_from_region_samples, region_sample_address_index, working_fill_from_spec,
)
from ..drawing_fill import serialize_fill
from ..drawing_paint import constrain_drawing_output_paint, refill_drawing_from_source
from ..output_settings import DrawingOutputSettings
from ..paint_fitting import canonical_fill
from .symmetry_materialize import index_region_paths
from .region_sample import RegionSample


def _bind(regions: Sequence[RegionSample], rgb: np.ndarray) -> tuple[Drawing, dict[str, np.ndarray]]:
    drawing = drawing_from_region_samples('paint-input', width=rgb.shape[1], height=rgb.shape[0], regions=regions)
    addresses = region_sample_address_index(regions)
    masks = {addresses[path]: region.raster for path, region in index_region_paths(regions).items()}
    return drawing, masks


def _publish(regions: Sequence[RegionSample], drawing: Drawing) -> tuple[RegionSample, ...]:
    addresses = region_sample_address_index(regions)
    def visit(region: RegionSample, parent: tuple[int, ...] = ()) -> RegionSample:
        path = (*parent, region.id)
        if region.is_branch:
            children = tuple(visit(child, path) for child in region.children)
            return region if all(a is b for a, b in zip(children, region.children, strict=True)) else region.with_children(children)
        selected = drawing.region(addresses[path])
        if canonical_fill(region.fill) == selected.fill and region.diagnostics == selected.diagnostics:
            return region
        fill = None if selected.fill is None else working_fill_from_spec(serialize_fill(selected.fill))
        if isinstance(fill, FlatFill):
            fill = FlatFill(fill.hex.upper())
        elif isinstance(fill, (LinearGradientFill, RadialGradientFill)):
            fill = replace(fill, stops=[(offset, value.upper()) for offset, value in fill.stops])
        return region.with_current(region.current, fill=fill, footprint=region.footprint,
            diagnostics=selected.diagnostics)
    return tuple(visit(root) for root in regions)


def refill_regions(
    regions: Sequence[RegionSample], rgb: np.ndarray, *, allow_gradients: bool = True,
) -> tuple[RegionSample, ...]:
    drawing, masks = _bind(regions, rgb)
    return _publish(regions, refill_drawing_from_source(drawing, rgb,
        source_masks=masks, allow_gradients=allow_gradients))


def refill_all_regions_from_source(
    regions: Sequence[RegionSample], rgb: np.ndarray, *,
    share_root_gradients: bool = True, allow_gradients: bool = True,
) -> tuple[RegionSample, ...]:
    drawing, masks = _bind(regions, rgb)
    return _publish(regions, refill_drawing_from_source(drawing, rgb, source_masks=masks,
        invalidate_all=True, share_root_gradients=share_root_gradients, allow_gradients=allow_gradients))


def refill_changed_regions_from_source(
    before: Sequence[RegionSample], current: Sequence[RegionSample], rgb: np.ndarray, *,
    allow_gradients: bool = True,
) -> tuple[RegionSample, ...]:
    previous, _ = _bind(before, rgb)
    drawing, masks = _bind(current, rgb)
    return _publish(current, refill_drawing_from_source(drawing, rgb, source_masks=masks,
        before=previous, allow_gradients=allow_gradients))


def apply_output_fill_settings(
    regions: Sequence[RegionSample], rgb: np.ndarray, settings: DrawingOutputSettings,
) -> tuple[RegionSample, ...]:
    if settings.allow_gradients and settings.max_output_colors is None:
        return tuple(regions)
    drawing, _ = _bind(regions, rgb)
    return _publish(regions, constrain_drawing_output_paint(drawing, rgb, settings))

"""Canonical source-paint orchestration, independent of geometry execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from . import paint_fitting
from .color import extract_palette
from .drawing import Drawing, DrawingRegion, _walk_regions
from .drawing_fill import Fill, FlatFill, LinearGradientFill, RadialGradientFill
from .geometry_native import region_path
from .optimizer.gate import rasterize
from .output_settings import DrawingOutputSettings


def _map_regions(drawing: Drawing, visit: Callable[[DrawingRegion], DrawingRegion]) -> Drawing:
    def walk(region: DrawingRegion) -> DrawingRegion:
        if region.is_branch:
            children = tuple(walk(child) for child in region.children)
            if any(a is not b for a, b in zip(children, region.children, strict=True)):
                region = replace(region, children=children)
        return visit(region)
    roots = tuple(walk(root) for root in drawing.regions)
    return drawing if all(a is b for a, b in zip(roots, drawing.regions, strict=True)) else replace(drawing, regions=roots)


def _index(drawing: Drawing) -> dict[str, DrawingRegion]:
    return {region.id: region for root in drawing.regions for region in _walk_regions(root)}


def source_paint_masks(observations: Mapping[str, Mapping[str, object]]) -> dict[str, np.ndarray]:
    """Decode retained coverage alone; no working shape or original-geometry codec."""
    masks = {}
    for identity, info in observations.items():
        if info.get('raster') is not None:
            masks[identity] = np.frombuffer(info['raster'], dtype=bool).reshape(info['raster_shape'])
    for identity, info in observations.items():
        source = info.get('raster_source')
        if source is not None:
            if source not in masks:
                raise ValueError(f'missing retained source coverage: {source}')
            masks[identity] = masks[source]
    return masks


def _visible_leaf_masks(drawing: Drawing, canvas_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    indexed = _index(drawing)
    ordered = sorted((leaf for root in drawing.regions for leaf in root.leaves()), key=lambda leaf: leaf.z)
    occluded = np.zeros(canvas_shape, dtype=bool)
    masks = {}
    for leaf in reversed(ordered):
        full = rasterize(region_path(leaf, indexed), canvas_shape)
        masks[leaf.id] = full & ~occluded
        occluded |= full
    return masks


def _with_fill(region: DrawingRegion, fill: Fill | None, diagnostics: Mapping[str, object]) -> DrawingRegion:
    return replace(region, fill=fill, diagnostics={**region.diagnostics, 'fill': diagnostics})


def _source_mask(identity: str, source_masks: Mapping[str, np.ndarray]) -> np.ndarray:
    # Missing material observations use the existing visible-footprint fallback.
    return source_masks.get(identity, np.empty((0, 0), dtype=bool))


def refill_drawing_from_source(
    drawing: Drawing, rgb: np.ndarray, *, source_masks: Mapping[str, np.ndarray],
    before: Drawing | None = None, invalidate_all: bool = False,
    share_root_gradients: bool = True, allow_gradients: bool = True,
) -> Drawing:
    """Resolve missing/all/changed paint while retaining canonical geometry.

    Shared self-symmetry and root gradients are all-source policies. A changed
    pass preserves fills on unaffected leaves and only refits invalidated ones.
    """
    indexed = _index(drawing)
    prior = {} if before is None else _index(before)
    gradient_roots = {root.id for root in drawing.regions if any(
        isinstance(leaf.fill, (LinearGradientFill, RadialGradientFill)) for leaf in root.leaves())}

    def invalidate(region: DrawingRegion) -> DrawingRegion:
        if region.is_branch:
            return region
        if invalidate_all:
            compound = region.diagnostics.get('compound')
            if isinstance(compound, Mapping) and compound.get('fill_source_id') is None and region.fill is not None:
                return region
            reason = 'geometry_guide'
        elif before is not None:
            previous = prior.get(region.id)
            if (previous is not None and previous.is_leaf
                    and previous.geometry == region.geometry
                    and region_path(previous, prior).to_svg_d() == region_path(region, indexed).to_svg_d()):
                return region
            reason = 'geometry_changed'
        else:
            return region
        return _with_fill(region, None, {'status': 'invalidated', 'reason': reason})

    cleared = _map_regions(drawing, invalidate)
    visible = _visible_leaf_masks(cleared, rgb.shape[:2])

    def share_self(region: DrawingRegion) -> DrawingRegion:
        if region.is_leaf:
            return region
        symmetry = region.diagnostics.get('symmetry')
        leaves = region.leaves()
        if (not isinstance(symmetry, Mapping) or symmetry.get('mode') != 'self'
                or any(leaf.fill is not None for leaf in leaves)
                or any(leaf.source_region_ids != leaves[0].source_region_ids for leaf in leaves[1:])):
            return region
        union = np.zeros(rgb.shape[:2], bool)
        for leaf in leaves:
            union |= visible[leaf.id]
        samples, domain = paint_fitting.source_sample_mask(_source_mask(region.id, source_masks), union)
        fitted = paint_fitting.canonical_fill(paint_fitting.fit_fill(samples, rgb,
            flat_hex=paint_fitting.sampled_flat_hex(samples, rgb), allow_gradients=allow_gradients))
        def assign(node: DrawingRegion) -> DrawingRegion:
            if node.is_leaf:
                return _with_fill(node, fitted, {'status': 'refit', 'mask': domain,
                    'sampled_pixels': int(samples.sum()), 'shared_self_symmetry': True})
            return replace(node, children=tuple(assign(child) for child in node.children))
        return assign(region)

    if invalidate_all:
        cleared = _map_regions(cleared, share_self)

    def refill(region: DrawingRegion) -> DrawingRegion:
        if region.is_branch or region.fill is not None:
            return region
        fitted = paint_fitting.fit_source_paint(rgb,
            source_mask=_source_mask(region.id, source_masks), visible_mask=visible[region.id],
            allow_gradients=allow_gradients)
        return _with_fill(region, fitted.fill, fitted.diagnostics)

    refilled = _map_regions(cleared, refill)
    if not invalidate_all or not share_root_gradients or not allow_gradients:
        return refilled
    return _refit_shared_root_gradients(refilled, rgb, gradient_roots, visible)


def _visible_material_fraction(mask: np.ndarray, rgb: np.ndarray) -> float:
    pixels = rgb[mask]
    return 0.0 if len(pixels) == 0 else float(np.mean(np.any(pixels < 240, axis=1)))


def _refit_shared_root_gradients(
    drawing: Drawing, rgb: np.ndarray, gradient_roots: set[str],
    visible_masks: Mapping[str, np.ndarray],
) -> Drawing:
    def refit_root(root: DrawingRegion) -> DrawingRegion:
        if root.id not in gradient_roots:
            return root
        leaves = root.leaves()
        cutouts = {leaf.id: FlatFill(paint_fitting.sampled_flat_hex(visible_masks[leaf.id], rgb))
            for leaf in leaves if visible_masks[leaf.id].any()
            and _visible_material_fraction(visible_masks[leaf.id], rgb) < .5}
        eligible = {leaf.id for leaf in leaves
            if _visible_material_fraction(visible_masks[leaf.id], rgb) >= .5}
        if len(eligible) < 2:
            return root
        mask = np.zeros(rgb.shape[:2], bool)
        for identity in eligible:
            mask |= visible_masks[identity]
        material_mask = mask & np.any(rgb < 240, axis=2)
        if int(material_mask.sum()) < 3:
            return root
        fill = paint_fitting.canonical_fill(paint_fitting.fit_fill(material_mask, rgb,
            flat_hex=paint_fitting.sampled_flat_hex(material_mask, rgb)))
        if not isinstance(fill, (LinearGradientFill, RadialGradientFill)):
            return root
        def visit(region: DrawingRegion) -> DrawingRegion:
            if region.is_branch:
                return replace(region, children=tuple(visit(child) for child in region.children))
            if region.id not in eligible and region.id not in cutouts:
                return region
            return _with_fill(region, cutouts.get(region.id, fill), {
                'status': 'visible_cutout' if region.id in cutouts else 'shared_root_gradient', 'root': root.id})
        return visit(root)
    return replace(drawing, regions=tuple(refit_root(root) for root in drawing.regions))


def constrain_drawing_output_paint(
    drawing: Drawing, rgb: np.ndarray, settings: DrawingOutputSettings,
) -> Drawing:
    """Apply terminal paint limits without working geometry or source records."""
    if settings.allow_gradients and settings.max_output_colors is None:
        return drawing
    palette = (extract_palette(rgb, max_sampling_colors=settings.max_output_colors, min_fraction=0.0)
        if settings.max_output_colors is not None else None)
    indexed = _index(drawing)
    def visit(region: DrawingRegion) -> DrawingRegion:
        if region.is_branch:
            return region
        fill = paint_fitting.constrain_output_paint(region.fill, rgb=rgb,
            sample_mask=rasterize(region_path(region, indexed), rgb.shape[:2]),
            settings=settings, palette=palette)
        return region if fill == region.fill else replace(region, fill=fill)
    return _map_regions(drawing, visit)

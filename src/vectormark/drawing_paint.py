"""Drawing-wide source-paint orchestration, independent of geometry execution.

Visibility, invalidation, shared surfaces and terminal output limits live here;
the numeric source-paint fitter remains shared with canonical region actions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import numpy as np
from scipy import ndimage
from .candidate import (
    Fill,
    FlatFill,
    LinearGradientFill,
    RadialGradientFill,
    RasterFill,
)
from .drawing_adapter import working_fill_from_spec as _fill
from .drawing_fill import serialize_fill
from .output_settings import DrawingOutputSettings
from .color import extract_palette, quantize
from .skia_geometry import SkPath
from .fill_fit import fit_fill
from . import paint_fitting
from .paint_fitting import sampled_flat_hex as _sampled_flat_hex
from .optimizer.gate import rasterize
from .optimizer.vector_region import VectorRegion, to_polygon
from .optimizer.symmetry_materialize import index_region_paths

DrawingRegions = tuple[VectorRegion, ...]
RegionPath = tuple[int, ...]


def _geometry_key(region: VectorRegion) -> tuple[object, ...]:
    """Stable enough comparison for deciding whether a retained leaf moved."""
    assert region.current is not None
    footprint_d = (
        region.footprint.to_svg_d() if isinstance(region.footprint, SkPath) else None
    )
    return (region.current.kind, repr(region.current.params), footprint_d)


def _invalidate_changed_fills(
    before: Sequence[VectorRegion], current: Sequence[VectorRegion]
) -> DrawingRegions:
    """Clear fills on new or geometrically changed leaves, preserving all others."""
    prior = {
        path: _geometry_key(leaf)
        for path, leaf in index_region_paths(before).items()
        if leaf.is_leaf
    }

    def visit(region: VectorRegion, path: RegionPath) -> VectorRegion:
        if region.is_leaf:
            assert region.current is not None
            if prior.get(path) == _geometry_key(region):
                return region
            return region.with_current(
                region.current,
                fill=None,
                footprint=region.footprint,
                diagnostics={
                    "fill": {"status": "invalidated", "reason": "geometry_changed"}
                },
            )
        children = tuple(visit(child, (*path, child.id)) for child in region.children)
        return region if children == region.children else region.with_children(children)

    return tuple(visit(region, (region.id,)) for region in current)


def refill_changed_regions_from_source(
    before: Sequence[VectorRegion],
    current: Sequence[VectorRegion],
    rgb: np.ndarray,
    *,
    allow_gradients: bool = True,
) -> DrawingRegions:
    """Apply the shared fill resolver only to geometry changed since *before*."""
    return refill_regions(
        _invalidate_changed_fills(before, current), rgb, allow_gradients=allow_gradients
    )


def _visible_leaf_masks(
    regions: Sequence[VectorRegion],
    canvas_shape: tuple[int, int],
) -> dict[RegionPath, np.ndarray]:
    """Return each leaf's visible paint surface after later siblings occlude it.

    Drawing geometry frequently uses overlapping shapes to create a ring or a
    cutout.  The fill stage must sample the portion that remains visible in
    SVG paint order, not the complete construction shape underneath it.
    """
    ordered = [
        (path, leaf)
        for path, leaf in index_region_paths(regions).items()
        if leaf.is_leaf
    ]
    ordered.sort(key=lambda item: item[1].z)
    occluded = np.zeros(canvas_shape, dtype=bool)
    masks: dict[RegionPath, np.ndarray] = {}
    for path, leaf in reversed(ordered):
        footprint = (
            leaf.footprint
            if isinstance(leaf.footprint, SkPath)
            else to_polygon(leaf.current)
        )
        full_mask = rasterize(footprint, canvas_shape)
        masks[path] = full_mask & ~occluded
        occluded |= full_mask
    return masks


def _fill_sample_mask(
    region: VectorRegion,
    visible_mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return pixels that are visible now and supported by the source material.

    ``footprint`` is the final geometric claim and can intentionally move beyond
    the original trace after primitive fitting, symmetry, or stitching.
    ``raster`` remains the leaf's traced material evidence.  Sampling their
    intersection prevents adjacent paint and the white-composited AA fringe
    from teaching a leaf the wrong fill.

    A deliberate plan can move a leaf far enough that the old material overlap
    is no longer useful.  Fall back to its visible footprint in that case so
    the fill pass still has meaningful source samples.
    """
    return paint_fitting.source_sample_mask(region.raster, visible_mask)


def apply_output_fill_settings(
    regions: Sequence[VectorRegion],
    rgb: np.ndarray,
    settings: DrawingOutputSettings,
) -> DrawingRegions:
    """Constrain final paint without touching geometry or source evidence."""
    if settings.allow_gradients and settings.max_output_colors is None:
        return tuple(regions)
    palette = (
        extract_palette(
            rgb, max_sampling_colors=settings.max_output_colors, min_fraction=0.0
        )
        if settings.max_output_colors is not None
        else None
    )

    def flat_source_color(region: VectorRegion) -> str:
        # Select the representative paint from the finalized shape itself.
        # Its old segmentation raster may contain only a boundary fringe after
        # compound splitting; that fringe must not choose the flat output color.
        footprint = (
            region.footprint
            if isinstance(region.footprint, SkPath)
            else to_polygon(region.current)
        )
        sample = rasterize(footprint, rgb.shape[:2])
        core = ndimage.binary_erosion(sample, iterations=2)
        return _sampled_flat_hex(core if core.any() else sample, rgb)

    def color(value: str) -> str:
        if palette is None:
            return value
        sample = np.asarray(
            [[[int(value[i : i + 2], 16) for i in (1, 3, 5)]]], dtype=np.uint8
        )
        red, green, blue = quantize(sample, palette)[0, 0]
        return f"#{red:02X}{green:02X}{blue:02X}"

    def visit(region: VectorRegion) -> VectorRegion:
        if not region.is_leaf:
            return region.with_children(
                tuple(visit(child) for child in region.children)
            )
        assert region.current is not None
        fill = region.fill
        if not settings.allow_gradients and not isinstance(fill, FlatFill):
            fill = FlatFill(flat_source_color(region))
        if isinstance(fill, FlatFill):
            fill = FlatFill(color(fill.hex))
        elif isinstance(fill, (LinearGradientFill, RadialGradientFill)):
            stops = [(offset, color(value)) for offset, value in fill.stops]
            if len({value for _offset, value in stops}) == 1:
                fill = FlatFill(stops[0][1])
            else:
                fill = replace(fill, stops=stops)
        elif isinstance(fill, RasterFill) and palette is not None:
            # A raster paint cannot honor a finite vector palette. Resolve it
            # to the same source-derived representative used for flat output.
            fill = FlatFill(color(flat_source_color(region)))
        return region.with_current(
            region.current, fill=fill, footprint=region.footprint
        )

    return tuple(visit(region) for region in regions)


def refill_regions(
    regions: Sequence[VectorRegion], rgb: np.ndarray, *, allow_gradients: bool = True
) -> DrawingRegions:
    """Run the shared fill stage for leaves invalidated by a geometry mutation.

    Callers deliberately leave an invalidated leaf with ``fill=None``.  This
    function is the sole point that turns its finalized vector footprint back
    into a source-derived FlatFill, LinearGradientFill, or RadialGradientFill.
    """
    canvas_shape = rgb.shape[:2]
    visible_masks = _visible_leaf_masks(regions, canvas_shape)

    def visit(region: VectorRegion, path: RegionPath) -> VectorRegion:
        if region.is_leaf:
            if region.fill is not None:
                return region
            assert region.current is not None
            footprint = region.footprint
            if not isinstance(footprint, SkPath):
                footprint = to_polygon(region.current)
            fitted = paint_fitting.fit_source_paint(
                rgb,
                source_mask=region.raster,
                visible_mask=visible_masks[path],
                allow_gradients=allow_gradients,
            )
            fill = _fill(serialize_fill(fitted.fill))
            if isinstance(fill, FlatFill):
                # Preserve the numeric fitter's median-color output spelling.
                fill = FlatFill(fill.hex.upper())
            return region.with_current(
                region.current,
                fill=fill,
                footprint=footprint,
                diagnostics={"fill": fitted.diagnostics},
            )
        children = tuple(visit(child, (*path, child.id)) for child in region.children)
        return region if children == region.children else region.with_children(children)

    return tuple(visit(region, (region.id,)) for region in regions)


def refill_all_regions_from_source(
    regions: Sequence[VectorRegion],
    rgb: np.ndarray,
    *,
    share_root_gradients: bool = True,
    allow_gradients: bool = True,
) -> DrawingRegions:
    """Discard every existing fill and re-fit finalized geometry from *rgb*.

    Geometry-guide retracing deliberately uses a palette-labelled image to
    establish boundaries.  Its colors are temporary segmentation labels, never
    a candidate final paint source.
    """
    gradient_roots = {
        root.id
        for root in regions
        if any(
            isinstance(leaf.fill, (LinearGradientFill, RadialGradientFill))
            for leaf in root.leaves()
        )
    }

    def clear(region: VectorRegion) -> VectorRegion:
        if region.is_leaf:
            assert region.current is not None
            compound = region.diagnostics.get("compound")
            # ``split_compound`` materializes a canvas/cutout plate when a
            # nested subpath has no source paint owner.  That fill is semantic
            # render evidence, not a stale source-derived fit: clearing it
            # makes later refill sample an empty footprint and emit black.
            if (
                isinstance(compound, Mapping)
                and compound.get("fill_source_id") is None
                and region.fill is not None
            ):
                return region
            return region.with_current(
                region.current,
                fill=None,
                footprint=region.footprint,
                diagnostics={
                    "fill": {"status": "invalidated", "reason": "geometry_guide"}
                },
            )
        children = tuple(clear(child) for child in region.children)
        return region.with_children(children)

    cleared = tuple(clear(region) for region in regions)
    # A ``self`` branch represents one material surface split only so the
    # optimizer can construct a reflected half.  Re-fit its paint once from
    # the finalized union and assign that paint to both implementation leaves.
    # This is deliberately distinct from ``root_self``: the latter retains
    # genuine material children and must continue to fit them independently.
    shared = _refill_self_symmetry_surfaces(
        cleared, rgb, allow_gradients=allow_gradients
    )
    refilled = refill_regions(shared, rgb, allow_gradients=allow_gradients)
    if not share_root_gradients or not allow_gradients:
        return refilled
    visible_masks = _visible_leaf_masks(refilled, rgb.shape[:2])
    return _refit_shared_root_gradients(refilled, rgb, gradient_roots, visible_masks)


def _refill_self_symmetry_surfaces(
    regions: Sequence[VectorRegion],
    rgb: np.ndarray,
    *,
    allow_gradients: bool = True,
) -> DrawingRegions:
    """Restore one source-derived fill to leaves created only for reflection."""
    visible_masks = _visible_leaf_masks(regions, rgb.shape[:2])

    def with_fill(
        region: VectorRegion, fill: Fill, sample_mask: np.ndarray, sample_domain: str
    ) -> VectorRegion:
        if region.is_leaf:
            assert region.current is not None
            return region.with_current(
                region.current,
                fill=fill,
                footprint=region.footprint,
                diagnostics={
                    "fill": {
                        "status": "refit",
                        "mask": sample_domain,
                        "sampled_pixels": int(sample_mask.sum()),
                        "shared_self_symmetry": True,
                    }
                },
            )
        return region.with_children(
            tuple(
                with_fill(child, fill, sample_mask, sample_domain)
                for child in region.children
            )
        )

    def visit(region: VectorRegion, path: RegionPath) -> VectorRegion:
        if region.is_leaf:
            return region
        children = tuple(visit(child, (*path, child.id)) for child in region.children)
        updated = (
            region if children == region.children else region.with_children(children)
        )
        symmetry = updated.diagnostics.get("symmetry")
        leaves = updated.leaves()
        if (
            not isinstance(symmetry, Mapping)
            or symmetry.get("mode") != "self"
            or not leaves
            # ``self`` children are implementation halves of one traced root.
            # If provenance ever differs, they are independently addressable
            # source regions and must remain candidates for clone detection
            # (which deliberately preserves per-region fills).
            or any(
                leaf.source_regions != leaves[0].source_regions for leaf in leaves[1:]
            )
            or any(leaf.fill is not None for leaf in leaves)
        ):
            return updated
        visible = np.zeros(rgb.shape[:2], dtype=bool)
        for local_path, leaf in index_region_paths((updated,)).items():
            if leaf.is_leaf:
                visible |= visible_masks.get(
                    (*path[:-1], *local_path), np.zeros_like(visible)
                )
        sample_mask, sample_domain = _fill_sample_mask(updated, visible)
        fill = fit_fill(
            sample_mask,
            rgb,
            flat_hex=_sampled_flat_hex(sample_mask, rgb),
            allow_gradients=allow_gradients,
        )
        return with_fill(updated, fill, sample_mask, sample_domain)

    return tuple(visit(region, (region.id,)) for region in regions)


def _visible_material_fraction(mask: np.ndarray, rgb: np.ndarray) -> float:
    """Return the fraction of a visible surface occupied by non-white source paint."""
    pixels = rgb[mask]
    if len(pixels) == 0:
        return 0.0
    return float(np.mean(np.any(pixels < 240, axis=1)))


def _refit_shared_root_gradients(
    regions: Sequence[VectorRegion],
    rgb: np.ndarray,
    gradient_roots: set[int],
    visible_masks: Mapping[RegionPath, np.ndarray],
) -> DrawingRegions:
    """Apply a shared source gradient across compatible finalized root leaves.

    Material quantization can split one continuous gradient into several
    palette-labelled leaves.  The leaves' geometry remains independent, but
    their union provides the sample domain needed to recover the original
    paint.  White cutouts and one-colour material groups are intentionally
    excluded.
    """
    canvas_shape = rgb.shape[:2]

    def refit_root(root: VectorRegion) -> VectorRegion:
        if root.id not in gradient_roots:
            return root
        leaves = [
            (path, leaf)
            for path, leaf in index_region_paths((root,)).items()
            if leaf.is_leaf
        ]
        visible_cutouts = [
            (path, leaf)
            for path, leaf in leaves
            if visible_masks[path].any()
            and _visible_material_fraction(visible_masks[path], rgb) < 0.5
        ]
        eligible = [
            (path, leaf)
            for path, leaf in leaves
            if _visible_material_fraction(visible_masks[path], rgb) >= 0.5
        ]
        if len(eligible) < 2:
            return root
        mask = np.zeros(canvas_shape, dtype=bool)
        for path, leaf in eligible:
            mask |= visible_masks[path]
        # Rasterization includes the outer path edge.  Those pixels can belong
        # to a white source background rather than the coloured material (in
        # particular for adjacent, separately traced gradient bands), so they
        # must not bias the gradient model itself.  The resulting fill is
        # nevertheless applied to each leaf's full finalized footprint below.
        material_mask = mask & np.any(rgb < 240, axis=2)
        if int(material_mask.sum()) < 3:
            return root
        fill = fit_fill(
            material_mask,
            rgb,
            flat_hex=_sampled_flat_hex(material_mask, rgb),
        )
        if not isinstance(fill, (LinearGradientFill, RadialGradientFill)):
            return root
        eligible_ids = {path for path, leaf in eligible}
        cutout_fills = {
            path: FlatFill(_sampled_flat_hex(visible_masks[path], rgb))
            for path, leaf in visible_cutouts
        }

        def visit(region: VectorRegion, path: RegionPath) -> VectorRegion:
            if region.is_leaf:
                if path not in eligible_ids and path not in cutout_fills:
                    return region
                assert region.current is not None
                is_cutout = path in cutout_fills
                return region.with_current(
                    region.current,
                    fill=cutout_fills.get(path, fill),
                    footprint=region.footprint,
                    diagnostics={
                        "fill": {
                            "status": "visible_cutout"
                            if is_cutout
                            else "shared_root_gradient",
                            "root": root.drawing_id,
                        }
                    },
                )
            children = tuple(
                visit(child, (*path, child.id)) for child in region.children
            )
            return (
                region
                if children == region.children
                else region.with_children(children)
            )

        return visit(root, (root.id,))

    return tuple(refit_root(root) for root in regions)

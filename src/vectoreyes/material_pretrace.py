"""Gradient-neutral material preprocessing for palette inference and tracing.

The source raster remains authoritative for geometry and final fill fitting.
This module builds a second raster whose only job is to make continuous colour
fields look like materials rather than dozens of quantized palette bands.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .color import (
    extract_palette,
    infer_max_sampling_colors,
    oklab_to_srgb,
    quantize,
    srgb_to_oklab,
    white_canvas_mask,
)
from .gradient import fit_gradient
from .segment import hexstr, segment
from .surface_merge import (
    _MAX_CREDIBLE_SEAM_TORTUOSITY,
    _MIN_CREDIBLE_SEAM_LENGTH,
    _region_owner,
    _soft_adjacencies,
    seam_is_noncredible,
)
from .types import Region


@dataclass(frozen=True)
class GradientSupport:
    """One contiguous region whose original pixels fit one gradient model."""

    id: str
    area: int
    model: Mapping[str, object]
    representative_rgb: tuple[int, int, int]


@dataclass(frozen=True)
class MaterialPretraceResult:
    """Reviewable raster stages and the gradient-neutral trace input."""

    high_color_quantized: np.ndarray
    shape_merged: np.ndarray
    color_field_merged: np.ndarray
    material_proxy: np.ndarray
    shape_region_count: int
    color_region_count: int
    gradient_supports: tuple[GradientSupport, ...]
    resolved_max_sampling_colors: int


def _representative_rgb(mask: np.ndarray, rgb: np.ndarray) -> tuple[int, int, int]:
    pixels = rgb[mask]
    if not len(pixels):
        return (0, 0, 0)
    mean_lab = srgb_to_oklab(pixels / 255.0).mean(axis=0)
    representative = np.rint(oklab_to_srgb(mean_lab) * 255.0).astype(np.uint8)
    return tuple(int(value) for value in representative)


def _merge_edges(
    regions: list[Region],
    edges: set[tuple[int, int]],
    rgb: np.ndarray,
) -> list[Region]:
    if not edges:
        return list(regions)
    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in sorted(edges):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    groups: dict[int, list[int]] = {}
    for index in range(len(regions)):
        groups.setdefault(find(index), []).append(index)

    merged: list[Region] = []
    ordered = sorted(
        groups.values(),
        key=lambda indices: (-sum(regions[index].area for index in indices), indices),
    )
    for label, indices in enumerate(ordered, start=1):
        mask = np.zeros_like(regions[indices[0]].mask)
        for index in indices:
            mask |= regions[index].mask
        color = _representative_rgb(mask, rgb)
        merged.append(Region(label, mask, hexstr(color)))
    return merged


def _noncredible_edges(regions: list[Region]) -> set[tuple[int, int]]:
    """Measure every shared seam in one ownership-image scan.

    This is the high-colour equivalent of the normal seam-credibility test.
    Applying one full-canvas dilation per palette pair is unnecessarily
    expensive here; encoded ownership pairs expose the same long, tortuous
    boundaries in one pass.
    """
    if len(regions) < 2:
        return set()
    owner = _region_owner(regions)
    count = len(regions)
    encoded_parts: list[np.ndarray] = []
    xs_parts: list[np.ndarray] = []
    ys_parts: list[np.ndarray] = []
    for left, right in (
        (owner[:, :-1], owner[:, 1:]),
        (owner[:-1, :], owner[1:, :]),
    ):
        valid = (left >= 0) & (right >= 0) & (left != right)
        if not valid.any():
            continue
        ys, xs = np.nonzero(valid)
        low = np.minimum(left[valid], right[valid])
        high = np.maximum(left[valid], right[valid])
        encoded_parts.append(low.astype(np.int64) * count + high)
        ys_parts.append(ys)
        xs_parts.append(xs)
    if not encoded_parts:
        return set()

    encoded = np.concatenate(encoded_parts)
    xs = np.concatenate(xs_parts)
    ys = np.concatenate(ys_parts)
    order = np.argsort(encoded, kind="stable")
    encoded, xs, ys = encoded[order], xs[order], ys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(encoded)) + 1]
    ends = np.r_[starts[1:], len(encoded)]
    result: set[tuple[int, int]] = set()
    for start, end in zip(starts, ends, strict=True):
        length = float(end - start)
        if length < _MIN_CREDIBLE_SEAM_LENGTH:
            continue
        span = max(
            float(np.hypot(float(np.ptp(xs[start:end])), float(np.ptp(ys[start:end])))),
            1.0,
        )
        if length / span > _MAX_CREDIBLE_SEAM_TORTUOSITY:
            pair = int(encoded[start])
            left, right = pair // count, pair % count
            # The ownership scan is a cheap candidate screen, not a second
            # seam classifier. Reuse the normal component/closure rules on
            # only this pair's contact bounds, avoiding full-canvas dilation.
            y0, y1 = max(0, int(ys[start:end].min()) - 1), int(ys[start:end].max()) + 3
            x0, x1 = max(0, int(xs[start:end].min()) - 1), int(xs[start:end].max()) + 3
            if seam_is_noncredible(
                regions[left].mask[y0:y1, x0:x1],
                regions[right].mask[y0:y1, x0:x1],
            ):
                result.add((left, right))
    return result


def _merge_continuous_color_fields(regions: list[Region], rgb: np.ndarray) -> list[Region]:
    """Merge each soft edge once while its accumulated support fits one gradient."""
    if len(regions) < 2:
        return list(regions)

    # Gradient-model selection is scale invariant enough for a merge gate. A
    # bounded proxy avoids repeatedly fitting million-pixel candidate masks;
    # final support models and representative colours are still full-size.
    step = max(1, int(np.ceil(max(rgb.shape[:2]) / 256.0)))
    preview_rgb = rgb[::step, ::step]
    preview_masks = [region.mask[::step, ::step].copy() for region in regions]
    parent = list(range(len(regions)))
    members = {index: {index} for index in range(len(regions))}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in sorted(_soft_adjacencies(regions, rgb, edge_de=0.06)):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        candidate = preview_masks[left_root] | preview_masks[right_root]
        if fit_gradient(candidate, preview_rgb) is None:
            continue
        parent[right_root] = left_root
        members[left_root] |= members.pop(right_root)
        preview_masks[left_root] = candidate

    merge_edges: set[tuple[int, int]] = set()
    for group in members.values():
        anchor = min(group)
        merge_edges.update((anchor, index) for index in group if index != anchor)
    return _merge_edges(regions, merge_edges, rgb)


def _region_raster(regions: list[Region], background: np.ndarray) -> np.ndarray:
    raster = np.asarray(background, dtype=np.uint8).copy()
    for region in regions:
        color = np.array(
            [int(region.color_hex[index:index + 2], 16) for index in (1, 3, 5)],
            dtype=np.uint8,
        )
        raster[region.mask] = color
    return raster


def preprocess_material_trace(
    rgb: np.ndarray,
    *,
    high_max_sampling_colors: int = 64,
    min_region_size: int = 16,
) -> MaterialPretraceResult:
    """Build a high-colour trace proxy while keeping distinct gradients separate."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("material pretrace expects an RGB image")

    white_canvas = white_canvas_mask(image)
    if white_canvas is None:
        palette = extract_palette(image, max_sampling_colors=high_max_sampling_colors, min_fraction=0.0)
    else:
        background_pixels = image[white_canvas]
        colors, counts = np.unique(background_pixels, axis=0, return_counts=True)
        background = colors[int(counts.argmax())][None, :]
        foreground = extract_palette(
            image[~white_canvas],
            max_sampling_colors=high_max_sampling_colors,
            min_fraction=0.0,
        )
        palette = np.vstack((background, foreground))

    high_quantized = quantize(image, palette)
    high_regions = segment(high_quantized, min_area=min_region_size)
    shape_regions = _merge_edges(
        high_regions,
        _noncredible_edges(high_regions),
        image,
    )
    color_regions = _merge_continuous_color_fields(shape_regions, image)

    proxy = image.copy()
    supports: list[GradientSupport] = []
    for index, region in enumerate(color_regions, start=1):
        # The proxy carries material topology, not source shading. Paint every
        # established support with one representative colour so antialiased
        # fringe pixels cannot be quantized back into narrow border regions by
        # the downstream trace. Final fill fitting still uses ``image``.
        representative = _representative_rgb(region.mask, image)
        proxy[region.mask] = representative
        model = fit_gradient(region.mask, image)
        if model is None:
            continue
        supports.append(
            GradientSupport(
                id=f"m{index}",
                area=region.area,
                model=model,
                representative_rgb=representative,
            )
        )

    resolved = infer_max_sampling_colors(proxy, exclude_mask=white_canvas)
    background = high_quantized.copy()
    for region in high_regions:
        background[region.mask] = high_quantized[region.mask]
    return MaterialPretraceResult(
        high_color_quantized=high_quantized,
        shape_merged=_region_raster(shape_regions, background),
        color_field_merged=_region_raster(color_regions, background),
        material_proxy=proxy,
        shape_region_count=len(shape_regions),
        color_region_count=len(color_regions),
        gradient_supports=tuple(supports),
        resolved_max_sampling_colors=resolved,
    )

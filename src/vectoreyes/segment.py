"""B: split the quantised image into connected single-colour regions."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label

from .types import Region


def hexstr(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _background_color(q: np.ndarray) -> tuple[int, int, int]:
    """Majority colour on the 1px border = background plate."""
    border = np.concatenate([q[0], q[-1], q[:, 0], q[:, -1]])
    colors, counts = np.unique(border, axis=0, return_counts=True)
    return tuple(int(v) for v in colors[counts.argmax()])


def segment(quantized: np.ndarray, *, min_area: int = 16) -> list[Region]:
    """Connected components per palette colour, excluding only the canvas plate.

    The border-majority colour is the canvas, but that same colour can be
    intentional artwork inside another surface (for example a white glyph on
    a transparent image composited onto white).  Retain its enclosed connected
    components and discard only components that reach the image border.
    """
    bg = _background_color(quantized)
    palette = np.unique(quantized.reshape(-1, 3), axis=0)
    regions: list[Region] = []
    next_label = 1
    for color in palette:
        is_background_color = tuple(int(v) for v in color) == bg
        color_mask = np.all(quantized == color, axis=2)
        labels = label(color_mask, connectivity=2)
        for lab_id in range(1, labels.max() + 1):
            comp = labels == lab_id
            if is_background_color and (
                comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()
            ):
                continue
            if comp.sum() < min_area:
                continue
            regions.append(Region(next_label, comp, hexstr(tuple(int(v) for v in color))))
            next_label += 1
    return regions


def fill_tiny_isolated_holes(mask: np.ndarray, *, max_area: int = 4) -> tuple[np.ndarray, int]:
    """Fill topology-noise pinholes without changing meaningful counters.

    Raster backgrounds can leak one or two near-background pixels into a solid
    foreground after anti-aliasing or resampling.  Those pixels are too small
    to carry drawing intent, but each becomes its own SVG subpath if left in a
    geometry mask.  Larger holes remain untouched for the colour-aware pass
    below, where their source material can be evaluated.
    """
    if max_area <= 0:
        return mask, 0

    inverse_labels = label(~mask, connectivity=1)
    border_labels = set(np.concatenate((
        inverse_labels[0], inverse_labels[-1], inverse_labels[:, 0], inverse_labels[:, -1],
    )).tolist())
    cleaned = mask.copy()
    filled = 0
    for component_id in range(1, int(inverse_labels.max()) + 1):
        if component_id in border_labels:
            continue
        hole = inverse_labels == component_id
        area = int(hole.sum())
        if area > max_area:
            continue
        cleaned[hole] = True
        filled += area
    return cleaned, filled


def fill_small_compatible_holes(
    mask: np.ndarray,
    rgb: np.ndarray,
    *,
    max_area: int,
    max_color_distance: float = 24.0,
) -> tuple[np.ndarray, int]:
    """Fill small enclosed mask holes whose pixels match their local surface.

    Quantizing a smooth raster gradient can create tiny palette islands.  When
    those islands are not part of the final merged surface, they become
    counters in an otherwise solid mask even though the source contains no
    visible hole.  A true counter normally has a contrasting local colour, so
    preserve it by comparing each enclosed component with its immediate mask
    neighbourhood.

    ``max_area`` is deliberately explicit: setting it to zero disables this
    root-level cleanup without changing raw segmentation or trace provenance.
    """
    if max_area <= 0:
        return mask, 0
    if mask.shape != rgb.shape[:2]:
        raise ValueError("mask and RGB image dimensions must match")

    # Foreground regions use 8-connectivity.  Use the complementary
    # 4-connectivity for background so a diagonally pinched palette island is
    # still treated as an enclosed counter by the path contour topology.
    inverse_labels = label(~mask, connectivity=1)
    border_labels = set(np.concatenate((
        inverse_labels[0], inverse_labels[-1], inverse_labels[:, 0], inverse_labels[:, -1],
    )).tolist())
    cleaned = mask.copy()
    filled = 0
    for component_id in range(1, int(inverse_labels.max()) + 1):
        if component_id in border_labels:
            continue
        hole = inverse_labels == component_id
        area = int(hole.sum())
        if area > max_area:
            continue
        ring = binary_dilation(hole, structure=np.ones((3, 3), dtype=bool)) & mask
        if not ring.any():
            continue
        source = np.median(rgb[hole], axis=0)
        surrounding = np.median(rgb[ring], axis=0)
        if float(np.linalg.norm(source - surrounding)) > max_color_distance:
            continue
        cleaned[hole] = True
        filled += area
    return cleaned, filled

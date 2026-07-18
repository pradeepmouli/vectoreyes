# SPDX-License-Identifier: MIT
"""Material-region construction from source-colour boundaries.

Adjacent quantized pieces belong to one geometry surface when their source seam
has no hard colour edge.  Fill fitting is deliberately separate: it runs only
after a material mask and its path geometry have been established.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .candidate import Fill, LinearGradientFill, RadialGradientFill
from .color import srgb_to_oklab
from .fill_fit import fit_fill
from .gradient import _interp_stops_rgb, _model_t
from .types import Region

_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
_MIN_CREDIBLE_SEAM_LENGTH = 24.0
_MAX_CREDIBLE_SEAM_TORTUOSITY = 2.5


def _kind(fill: Fill) -> str | None:
    if isinstance(fill, LinearGradientFill):
        return "linear"
    if isinstance(fill, RadialGradientFill):
        return "radial"
    return None


def _model(fill: Fill) -> dict:
    return {"kind": _kind(fill), "geometry": fill.geometry, "stops": fill.stops}


def seam_pairs(mask_a: np.ndarray, mask_b: np.ndarray, rgb: np.ndarray):
    """Colors of every 4-adjacent pixel pair straddling the A|B boundary.

    Returns (colors_a, colors_b), each (N,3) uint8 — colors_a[k] in mask_a is 4-adjacent
    to colors_b[k] in mask_b."""
    cols_a, cols_b = [], []
    H, W = mask_a.shape
    for dy, dx in _NEIGHBORS:
        # b_shift[y,x] = mask_b[y+dy, x+dx]; valid region trims the wrapped edge.
        b_shift = np.zeros_like(mask_b)
        ys = slice(max(0, -dy), H - max(0, dy))
        xs = slice(max(0, -dx), W - max(0, dx))
        ys2 = slice(max(0, dy), H - max(0, -dy))
        xs2 = slice(max(0, dx), W - max(0, -dx))
        b_shift[ys, xs] = mask_b[ys2, xs2]
        border = mask_a & b_shift
        if not border.any():
            continue
        ay, ax = np.where(border)
        cols_a.append(rgb[ay, ax])
        cols_b.append(rgb[ay + dy, ax + dx])
    if not cols_a:
        return np.empty((0, 3), np.uint8), np.empty((0, 3), np.uint8)
    return np.concatenate(cols_a), np.concatenate(cols_b)


def seam_is_soft(mask_a: np.ndarray, mask_b: np.ndarray, rgb: np.ndarray,
                 *, edge_de: float = 0.06) -> bool:
    """(A) True iff the masks are 4-adjacent and the source color steps smoothly across
    their shared border (median straddling-pair OKLab ΔE < edge_de).
    Same-color-at-seam features are not distinguishable here; the merge's
    union-fits-gradient guard handles them."""
    ca, cb = seam_pairs(mask_a, mask_b, rgb)
    if len(ca) == 0:
        return False
    de = np.linalg.norm(srgb_to_oklab(ca / 255.0) - srgb_to_oklab(cb / 255.0), axis=1)
    return float(np.median(de)) < edge_de


def seam_is_noncredible(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """Whether a shared seam is too noisy to justify separate regions.

    A clean line, smooth arc, or closed shape has a modest ratio between its
    traced length and end-to-end span.  A palette staircase has many boundary
    steps while making little geometric progress.  The latter is evidence of a
    segmentation artifact independent of source colour, and is the first-pass
    reason to union two regions.
    """
    seam = mask_a & ndimage.binary_dilation(mask_b, structure=_EIGHT_CONNECTED)
    labels, count = ndimage.label(seam, structure=_EIGHT_CONNECTED)
    for component_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == component_id)
        length = float(len(xs))
        if length < _MIN_CREDIBLE_SEAM_LENGTH:
            continue
        span = max(float(np.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))), 1.0)
        if length / span > _MAX_CREDIBLE_SEAM_TORTUOSITY:
            return True
    return False


def _region_owner(regions: list[Region]) -> np.ndarray:
    """Return the one-to-one component ownership image for ``regions``."""
    owner = np.full(regions[0].mask.shape, -1, dtype=np.int32)
    for index, region in enumerate(regions):
        owner[region.mask] = index
    return owner


def _adjacent_region_pairs(owner: np.ndarray) -> set[tuple[int, int]]:
    """Return the distinct 8-adjacent component pairs in an ownership image."""
    pairs: set[tuple[int, int]] = set()
    for left, right in (
        (owner[:, :-1], owner[:, 1:]),
        (owner[:-1, :], owner[1:, :]),
        (owner[:-1, :-1], owner[1:, 1:]),
        (owner[:-1, 1:], owner[1:, :-1]),
    ):
        valid = (left >= 0) & (right >= 0) & (left != right)
        if not valid.any():
            continue
        low = np.minimum(left[valid], right[valid])
        high = np.maximum(left[valid], right[valid])
        pairs.update(zip(low.tolist(), high.tolist(), strict=True))
    return pairs


def seam_band(mask_a: np.ndarray, mask_b: np.ndarray, *, width: int = 2):
    """(ys, xs) of mask_a pixels within `width` px of mask_b — the A-side of the seam."""
    band = mask_a & ndimage.binary_dilation(mask_b, iterations=width)
    return np.where(band)


def _rendered_oklab(model: dict, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    pts = np.column_stack([xs, ys]).astype(float)
    return srgb_to_oklab(_interp_stops_rgb(_model_t(model, pts), model["stops"]) / 255.0)


def gradients_continuous(fill_a: Fill, mask_a: np.ndarray, fill_b: Fill, mask_b: np.ndarray,
                         *, seam_de: float = 0.045) -> bool:
    """(B) True iff both fills are gradients over adjacent masks whose models render to
    agreeing colours across the shared seam (mean OKLab ΔE < seam_de) — one gradient's
    colour at the seam matches the other's. False if either fill is flat (A handles that)."""
    if _kind(fill_a) is None or _kind(fill_b) is None:
        return False
    ys_a, xs_a = seam_band(mask_a, mask_b)
    ys_b, xs_b = seam_band(mask_b, mask_a)
    if len(xs_a) == 0 or len(xs_b) == 0:
        return False                                     # not adjacent
    ma, mb = _model(fill_a), _model(fill_b)
    de1 = np.linalg.norm(_rendered_oklab(ma, ys_b, xs_b) - _rendered_oklab(mb, ys_b, xs_b), axis=1).mean()
    de2 = np.linalg.norm(_rendered_oklab(mb, ys_a, xs_a) - _rendered_oklab(ma, ys_a, xs_a), axis=1).mean()
    return max(float(de1), float(de2)) < seam_de


_GRADIENT = (LinearGradientFill, RadialGradientFill)


def _soft_adjacencies(
    regions: list[Region],
    rgb: np.ndarray,
    *,
    edge_de: float,
) -> set[tuple[int, int]]:
    """Return component-index pairs separated by a soft source-colour seam.

    Constructing this graph from one ownership image makes the surface merge
    proportional to actual region adjacencies, rather than all possible region
    pairs.  It also means a small quantized island is a normal graph node: it
    can join a continuous material surface without becoming a path of its own.
    """
    if not regions:
        return set()
    owner = _region_owner(regions)

    pair_ids: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    wide_pair_ids: list[np.ndarray] = []
    wide_deltas: list[np.ndarray] = []
    count = len(regions)
    for left_owner, right_owner, left_rgb, right_rgb in (
        (owner[:, :-1], owner[:, 1:], rgb[:, :-1], rgb[:, 1:]),
        (owner[:-1, :], owner[1:, :], rgb[:-1, :], rgb[1:, :]),
    ):
        valid = (left_owner >= 0) & (right_owner >= 0) & (left_owner != right_owner)
        if not valid.any():
            continue
        left = left_owner[valid]
        right = right_owner[valid]
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        pair_ids.append(low.astype(np.int64) * count + high)
        left_oklab = srgb_to_oklab(left_rgb[valid] / 255.0)
        right_oklab = srgb_to_oklab(right_rgb[valid] / 255.0)
        deltas.append(np.linalg.norm(left_oklab - right_oklab, axis=1))
    # A hard antialiased boundary can have several individually small colour
    # steps.  Looking one pixel beyond each side of its seam exposes the sharp
    # total change, while a genuinely continuous gradient remains smooth.
    for left_owner, right_owner, left_rgb, right_rgb in (
        (owner[:, 1:-2], owner[:, 2:-1], rgb[:, :-3], rgb[:, 3:]),
        (owner[1:-2, :], owner[2:-1, :], rgb[:-3, :], rgb[3:, :]),
    ):
        valid = (left_owner >= 0) & (right_owner >= 0) & (left_owner != right_owner)
        if not valid.any():
            continue
        left = left_owner[valid]
        right = right_owner[valid]
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        wide_pair_ids.append(low.astype(np.int64) * count + high)
        left_oklab = srgb_to_oklab(left_rgb[valid] / 255.0)
        right_oklab = srgb_to_oklab(right_rgb[valid] / 255.0)
        wide_deltas.append(np.linalg.norm(left_oklab - right_oklab, axis=1))
    if not pair_ids:
        return set()

    ids = np.concatenate(pair_ids)
    values = np.concatenate(deltas)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    ends = np.r_[starts[1:], len(ids)]
    wide_by_id: dict[int, float] = {}
    if wide_pair_ids:
        wide_ids = np.concatenate(wide_pair_ids)
        wide_values = np.concatenate(wide_deltas)
        wide_order = np.argsort(wide_ids, kind="stable")
        wide_ids = wide_ids[wide_order]
        wide_values = wide_values[wide_order]
        wide_starts = np.r_[0, np.flatnonzero(np.diff(wide_ids)) + 1]
        wide_ends = np.r_[wide_starts[1:], len(wide_ids)]
        wide_by_id = {
            int(wide_ids[start]): float(np.median(wide_values[start:end]))
            for start, end in zip(wide_starts, wide_ends, strict=True)
        }
    soft: set[tuple[int, int]] = set()
    for start, end in zip(starts, ends, strict=True):
        encoded = int(ids[start])
        if float(np.median(values[start:end])) >= edge_de:
            continue
        if wide_by_id.get(encoded, 0.0) >= edge_de * 2.0:
            continue
        soft.add((encoded // count, encoded % count))
    return soft


def _noncredible_adjacencies(regions: list[Region]) -> set[tuple[int, int]]:
    """Return adjacent region pairs with shape evidence against their seam."""
    if len(regions) < 2:
        return set()
    return {
        (left, right)
        for left, right in _adjacent_region_pairs(_region_owner(regions))
        if seam_is_noncredible(regions[left].mask, regions[right].mask)
    }


def material_groups(
    regions: list[Region],
    rgb: np.ndarray,
    *,
    edge_de: float = 0.06,
) -> list[tuple[int, ...]]:
    """Return contiguous material groups without fitting a fill or changing geometry.

    The indices address ``regions`` directly.  Keeping this result separate from
    the grouped masks lets callers fit their geometry first and union those
    fitted paths only afterwards.  First merge across non-credible shared
    seams; only when geometry has no such merge candidates, fall back to source
    colour continuity for palette/gradient fragmentation.
    """
    if len(regions) < 2:
        return [tuple(range(len(regions)))] if regions else []
    shape_edges = _noncredible_adjacencies(regions)
    merge_edges = shape_edges or _soft_adjacencies(regions, rgb, edge_de=edge_de)
    if not merge_edges:
        return [(index,) for index in range(len(regions))]

    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in sorted(merge_edges):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    groups: dict[int, list[int]] = {}
    for index in range(len(regions)):
        groups.setdefault(find(index), []).append(index)
    return sorted(
        (tuple(indices) for indices in groups.values()),
        key=lambda indices: (-sum(regions[index].area for index in indices), indices),
    )


def merge_material_regions(
    regions: list[Region],
    rgb: np.ndarray,
    *,
    edge_de: float = 0.06,
) -> list[Region]:
    """Compatibility helper returning the masks for :func:`material_groups`."""
    merged: list[Region] = []
    for indices in material_groups(regions, rgb, edge_de=edge_de):
        members = [regions[index] for index in indices]
        representative = max(members, key=lambda region: (region.area, -region.label))
        mask = np.zeros_like(representative.mask)
        for member in members:
            mask |= member.mask
        merged.append(Region(representative.label, mask, representative.color_hex))
    return sorted(merged, key=lambda region: (-region.area, region.label, region.color_hex))


def merge_surfaces(
    filled: list[tuple[Region, Fill]],
    rgb: np.ndarray,
    *,
    seam_de: float = 0.045,
    edge_de: float = 0.06,
) -> list[tuple[Region, Fill]]:
    """Compatibility wrapper: material geometry first, then fit each material fill."""
    del seam_de
    materials = merge_material_regions([region for region, _fill in filled], rgb, edge_de=edge_de)
    return [
        (region, fit_fill(region.mask, rgb, flat_hex=region.color_hex))
        for region in materials
    ]

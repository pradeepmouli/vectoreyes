"""Gutter-based component decomposition (slice 5). Recursive X-Y cut: split a mark on
the widest full-span whitespace gutter (horizontal or vertical) wider than a
conservative scale-relative threshold, recurse on each block, and return the components
in reading order. A mark with no qualifying gutter is a single component — the parity
path that keeps single-component output byte-identical to the pre-slice-5 pipeline."""

from __future__ import annotations

import numpy as np

from .types import Region

# Conservative, scale-relative: a gutter must be at least this fraction of the block's
# extent along the cut axis (or the absolute floor, whichever is larger) to split.
# Tuned so obvious multi-element marks split but borderline intra-mark gaps do not.
_GUTTER_FRACTION = 0.3
_GUTTER_ABS_FLOOR = 6


def _interior_gaps(occ: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of empty (False) cells strictly between the first and last occupied
    cell of `occ` (a 1D bool occupancy profile). Each run is returned as [start, end)."""
    idx = np.flatnonzero(occ)
    if len(idx) == 0:
        return []
    first, last = int(idx[0]), int(idx[-1])
    gaps: list[tuple[int, int]] = []
    run_start: int | None = None
    for i in range(first, last + 1):
        if not occ[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            gaps.append((run_start, i))
            run_start = None
    return gaps


def _best_gutter(sil: np.ndarray) -> tuple[str, float] | None:
    """The widest qualifying full-span gutter in the silhouette. Returns ("h", y_cut) or
    ("v", x_cut) (cut position in pixels), or None if no gutter qualifies. On a width
    tie, prefers the cut closest to the block centre (most even split)."""
    ys, xs = np.where(sil)
    if len(ys) == 0:
        return None
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    block = sil[r0:r1, c0:c1]

    cands: list[tuple[int, float, str, float]] = []

    row_occ = block.any(axis=1)                       # occupancy per row
    h_min = max(_GUTTER_ABS_FLOOR, _GUTTER_FRACTION * (r1 - r0))
    h_centre = (r0 + r1) / 2
    for s, e in _interior_gaps(row_occ):
        width = e - s
        if width >= h_min:
            cut = r0 + (s + e) / 2
            cands.append((width, abs(cut - h_centre), "h", cut))

    col_occ = block.any(axis=0)                       # occupancy per column
    v_min = max(_GUTTER_ABS_FLOOR, _GUTTER_FRACTION * (c1 - c0))
    v_centre = (c0 + c1) / 2
    for s, e in _interior_gaps(col_occ):
        width = e - s
        if width >= v_min:
            cut = c0 + (s + e) / 2
            cands.append((width, abs(cut - v_centre), "v", cut))

    if not cands:
        return None
    cands.sort(key=lambda t: (-t[0], t[1]))           # widest first, then most centred on tie
    _, _, axis, cut = cands[0]
    return (axis, cut)


def _partition(regions: list[Region], axis: str, cut: float) -> tuple[list[Region], list[Region]]:
    """Split regions by the side of `cut` their pixel-centroid lies on. The gutter is
    empty, so no region's pixels lie in the cut band — every region falls cleanly to one
    side. `axis`=="h" splits above/below (row centroid); "v" splits left/right (col)."""
    a: list[Region] = []
    b: list[Region] = []
    for r in regions:
        rr, cc = np.where(r.mask)
        centroid = rr.mean() if axis == "h" else cc.mean()
        (a if centroid < cut else b).append(r)
    return a, b


def decompose_components(regions: list[Region], shape: tuple[int, int]) -> list[list[Region]]:
    """Partition `regions` into spatially-separated components by recursive X-Y cut on
    the union silhouette, in reading order (top->bottom, left->right). Returns
    [regions] (one component) when there is <=1 region or no qualifying gutter — the
    parity path."""
    if len(regions) <= 1:
        return [regions]
    sil = np.zeros(shape, bool)
    for r in regions:
        sil |= r.mask
    gutter = _best_gutter(sil)
    if gutter is None:
        return [regions]
    axis, cut = gutter
    a, b = _partition(regions, axis, cut)
    if not a or not b:                                # degenerate guard (no real split)
        return [regions]
    return decompose_components(a, shape) + decompose_components(b, shape)

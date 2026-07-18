"""A0 colour optimisation: OKLab conversion, ΔE, palette extraction, quantise."""

from __future__ import annotations

import numpy as np

# sRGB <-> linear
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


# OKLab matrices (Björn Ottosson, 2020)
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) in [0,1] -> OKLab (..., 3)."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=float))
    lms = lin @ _M1.T
    lms_ = np.cbrt(lms)
    return lms_ @ _M2.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """OKLab (..., 3) -> sRGB (..., 3) clipped to [0,1]."""
    lms_ = np.asarray(lab, dtype=float) @ _M2_INV.T
    lms = lms_ ** 3
    lin = lms @ _M1_INV.T
    return np.clip(_linear_to_srgb(lin), 0.0, 1.0)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """OKLab Euclidean distance (OKLab is designed for L2 ΔE)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def extract_palette(
    rgb_image: np.ndarray, *, max_colors: int = 16, merge_de: float = 0.045,
    min_fraction: float = 0.002,
) -> np.ndarray:
    """Perceptual-clustering palette in OKLab.

    Clusters near-identical shades *before* applying the frequency floor, so a
    colour dispersed across many antialiased shades (a thin/small mark) is kept
    by its aggregate weight instead of dropped per-shade. Each cluster's
    representative is its most-frequent member — a real colour, never a centroid.

    Returns an (N, 3) uint8 array, frequency-ordered.
    """
    flat = np.asarray(rgb_image, dtype=np.uint8).reshape(-1, 3)
    total = len(flat)
    if total == 0:
        return np.empty((0, 3), dtype=np.uint8)

    # Distinct full-precision colours and their pixel counts.
    colors, counts = np.unique(flat, axis=0, return_counts=True)

    # Coarse 5-bit pre-bin (32 levels/channel) caps clustering input regardless
    # of AA spread. Group full-precision colours by their bin.
    color_bin = colors >> 3
    bins, bin_inv = np.unique(color_bin, axis=0, return_inverse=True)
    bin_inv = bin_inv.ravel()
    nbins = len(bins)
    bin_total = np.zeros(nbins, dtype=np.int64)
    np.add.at(bin_total, bin_inv, counts)

    # Representative per bin = most-frequent full-precision colour in that bin.
    # Order colours by descending count, value-tiebroken for determinism; the
    # first colour seen for each bin (in this order) is its representative.
    order = np.lexsort((colors[:, 2], colors[:, 1], colors[:, 0], -counts))
    sorted_bins = bin_inv[order]
    uniq_bin, first_pos = np.unique(sorted_bins, return_index=True)
    rep_color_idx = np.empty(nbins, dtype=np.int64)
    rep_color_idx[uniq_bin] = order[first_pos]
    rep_color = colors[rep_color_idx]                       # (nbins, 3) full precision
    rep_lab = srgb_to_oklab(rep_color / 255.0)

    # Greedy perceptual clustering over bins, in descending aggregate-count
    # order (stable -> value-tiebroken via the value-sorted bin ids). The seed
    # bin of each cluster is its most-frequent member -> the representative.
    # A new cluster is only seeded by a bin whose own total meets the floor;
    # sub-floor bins can only be absorbed into an existing cluster (AA shades).
    # Trade-off: a colour whose *most-frequent* bin is still sub-floor (e.g. a
    # hue smeared over many distant bins) is dropped — it has no dominant
    # representative to anchor a cluster. Safe direction: a miss, not a false
    # colour. See the spec's "Known limitation (from the seed-floor rule)".
    b_order = np.argsort(-bin_total, kind="stable")
    floor = min_fraction * total
    cluster_lab: list[np.ndarray] = []
    cluster_total: list[int] = []
    cluster_bin: list[int] = []
    for b in b_order:
        placed = False
        for k in range(len(cluster_lab)):
            if delta_e(rep_lab[b], cluster_lab[k]) < merge_de:
                cluster_total[k] += int(bin_total[b])
                placed = True
                break
        if not placed and bin_total[b] >= floor:
            cluster_lab.append(rep_lab[b])
            cluster_total.append(int(bin_total[b]))
            cluster_bin.append(int(b))

    # Cap to max_colors, frequency-ordered.
    if not cluster_total:
        # All bins are sub-floor: fall back to the most-frequent bin.
        fb = int(b_order[0])
        return rep_color[fb:fb + 1].astype(np.uint8)
    totals = np.array(cluster_total, dtype=np.int64)
    keep = list(np.argsort(-totals, kind="stable")[:max_colors])
    return rep_color[[cluster_bin[k] for k in keep]].astype(np.uint8)


def mean_delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-pixel OKLab Euclidean distance between two uint8 RGB images/arrays
    (shape (...,3)). 0.0 == identical; perceptual color error."""
    la = srgb_to_oklab(a.reshape(-1, 3) / 255.0)
    lb = srgb_to_oklab(b.reshape(-1, 3) / 255.0)
    return float(np.linalg.norm(la - lb, axis=1).mean())


def white_canvas_mask(rgb_image: np.ndarray, *, tolerance: float = 32.0) -> np.ndarray | None:
    """Return a dominant near-white border canvas, if the image has one.

    This is deliberately narrower than general background detection: it only
    reserves an almost-white opaque canvas so automatic palette sizing can
    count foreground materials rather than the export matte.
    """
    image = np.asarray(rgb_image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("white canvas detection expects an RGB image")
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    colors, counts = np.unique(border, axis=0, return_counts=True)
    background = colors[int(counts.argmax())].astype(float)
    if float(background.min()) < 240.0:
        return None
    mask = np.linalg.norm(image.astype(float) - background, axis=2) <= tolerance
    return mask if int(mask.sum()) else None


def infer_max_colors(
    rgb_image: np.ndarray,
    *,
    candidates: tuple[int, ...] = (4, 8, 16, 32, 64),
    sample_size: int = 16_384,
    minimum_improvement: float = 0.003,
    exclude_mask: np.ndarray | None = None,
) -> int:
    """Choose the smallest palette before quantization improvements flatten.

    Palette extraction is deterministic and frequency ordered, so candidate
    palettes are prefixes of one maximum palette.  This lets the selector score
    every palette size in a single incremental nearest-colour sweep over a
    bounded deterministic sample rather than repeatedly quantizing the image.
    """
    levels = tuple(sorted({int(level) for level in candidates if int(level) >= 2}))
    if not levels:
        raise ValueError("palette candidates must contain an integer >= 2")
    image = np.asarray(rgb_image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("palette inference expects an RGB image")
    pixels = image.reshape(-1, 3)
    if exclude_mask is not None:
        mask = np.asarray(exclude_mask, dtype=bool)
        if mask.shape != image.shape[:2]:
            raise ValueError("palette exclusion mask dimensions must match the image")
        retained = pixels[~mask.reshape(-1)]
        if len(retained):
            pixels = retained
    palette = extract_palette(pixels, max_colors=levels[-1], min_fraction=0.0)
    if len(palette) <= levels[0]:
        return int(len(palette))

    effective_levels = tuple(sorted({min(level, len(palette)) for level in levels}))
    if len(pixels) > sample_size:
        sample = pixels[np.linspace(0, len(pixels) - 1, sample_size, dtype=np.intp)]
    else:
        sample = pixels
    sample_lab = srgb_to_oklab(sample / 255.0)
    palette_lab = srgb_to_oklab(palette / 255.0)
    best = np.full(len(sample_lab), np.inf, dtype=float)
    errors: dict[int, float] = {}
    level_index = 0
    for index, color in enumerate(palette_lab, start=1):
        best = np.minimum(best, np.linalg.norm(sample_lab - color, axis=1))
        while level_index < len(effective_levels) and index >= effective_levels[level_index]:
            errors[effective_levels[level_index]] = float(best.mean())
            level_index += 1
    while level_index < len(effective_levels):
        errors[effective_levels[level_index]] = float(best.mean())
        level_index += 1

    for current, following in zip(effective_levels, effective_levels[1:]):
        if errors[current] - errors[following] <= minimum_improvement:
            return current
    return effective_levels[-1]


def quantize(rgb_image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Assign every pixel to its nearest palette colour by OKLab ΔE."""
    h, w, _ = rgb_image.shape
    flat = srgb_to_oklab(rgb_image.reshape(-1, 3) / 255.0)
    pal_lab = srgb_to_oklab(palette / 255.0)
    # pairwise distances (P pixels x K palette)
    d = np.linalg.norm(flat[:, None, :] - pal_lab[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    return palette[nearest].reshape(h, w, 3).astype(np.uint8)

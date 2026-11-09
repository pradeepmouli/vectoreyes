"""Shared appearance-difference visualization; never an acceptance score."""

import numpy as np
from PIL import Image


def raster_difference(source: Image.Image, candidate: Image.Image) -> Image.Image:
    """Absolute RGB difference, amplified fourfold, on a shared white canvas."""
    if source.size != candidate.size:
        raise ValueError("difference requires matching source and candidate canvases")
    def composite(image: Image.Image) -> np.ndarray:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", image.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return np.asarray(background, dtype=np.int16)
    difference = np.abs(composite(source) - composite(candidate))
    return Image.fromarray(np.clip(difference * 4, 0, 255).astype(np.uint8))

"""The (geometry, fill) candidate seam: decouples shape/path detection from
colour application. IO-free data (SVG emission stays in emit.py / pipeline.py)."""

from __future__ import annotations

from dataclasses import dataclass

from .fit import Shape
from .types import Axis


@dataclass
class FlatFill:
    """A solid colour fill."""
    hex: str


@dataclass
class LinearGradientFill:
    """A linear gradient fill. `geometry` = {x1, y1, x2, y2} in the element's frame."""
    geometry: dict
    stops: list


@dataclass
class RadialGradientFill:
    """A radial gradient fill. `geometry` = {cx, cy, r} in the element's frame."""
    geometry: dict
    stops: list


@dataclass
class RasterFill:
    """A bilinear-stretched raster fill: a small NxN PNG stretched across
    `geometry` = {x, y, w, h} (the footprint bbox) and clipped by the path it
    fills. For smooth 2-D colour fields no parametric gradient expresses.
    `png_b64` is a bare base64 PNG (no data-URI prefix)."""
    geometry: dict
    png_b64: str


Fill = FlatFill | LinearGradientFill | RadialGradientFill | RasterFill


@dataclass
class Candidate:
    """One renderable element: a geometry paired with a fill.

    `source` records the coarse element category ("occlusion" | "lens" | "region" |
    "gradient") — provenance for later agent/user candidate selection, and the
    discriminator for the one legacy emit quirk (lens = plain path, no id).
    `mirror`, when set, means: emit the element AND its mirror twin about that axis.
    `strategy` is the finer-grained fitter provenance (e.g. "symmetric" vs "path")
    used by manual selection; `source` stays the coarse element category.
    """
    geometry: Shape
    fill: Fill
    source: str
    mirror: Axis | None = None
    strategy: str | None = None   # fitter provenance (slice 4b); None for occlusion/lens/gradient
    region_label: int | None = None  # originating region label (None for occlusion/lens/gradient)

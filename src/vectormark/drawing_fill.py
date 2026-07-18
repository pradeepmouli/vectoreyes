"""Immutable, typed fill values for canonical drawings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

from .path_geometry import Point, _point


_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?$")


def _color(value: str) -> str:
    if not _HEX_COLOR.fullmatch(value):
        raise ValueError("colors must be #RRGGBB or #RRGGBBAA")
    return value.lower()


@dataclass(frozen=True)
class GradientStop:
    """One ordered color stop in a parametric gradient."""

    offset: float
    color: str

    def __post_init__(self) -> None:
        if not isfinite(float(self.offset)) or not 0.0 <= self.offset <= 1.0:
            raise ValueError("gradient-stop offsets must be finite and in [0, 1]")
        object.__setattr__(self, "color", _color(self.color))


def _gradient_stops(stops: tuple[GradientStop, ...]) -> tuple[GradientStop, ...]:
    if not stops:
        raise ValueError("gradients require at least one stop")
    if any(next_stop.offset < stop.offset for stop, next_stop in zip(stops, stops[1:], strict=False)):
        raise ValueError("gradient stops must be sorted by offset")
    return stops


@dataclass(frozen=True)
class FlatFill:
    """A solid color fill."""

    color: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "color", _color(self.color))


@dataclass(frozen=True)
class LinearGradientFill:
    """A linear gradient defined directly in drawing coordinates."""

    start: Point
    end: Point
    stops: tuple[GradientStop, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _point(self.start))
        object.__setattr__(self, "end", _point(self.end))
        object.__setattr__(self, "stops", _gradient_stops(self.stops))


@dataclass(frozen=True)
class RadialGradientFill:
    """A radial gradient with an optional focal point."""

    center: Point
    radius: float
    stops: tuple[GradientStop, ...]
    focal_point: Point | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center))
        if not isfinite(float(self.radius)) or self.radius < 0:
            raise ValueError("radial-gradient radius must be finite and non-negative")
        object.__setattr__(self, "stops", _gradient_stops(self.stops))
        if self.focal_point is not None:
            object.__setattr__(self, "focal_point", _point(self.focal_point))


@dataclass(frozen=True)
class RasterFill:
    """A raster color field clipped to the final region geometry."""

    origin: Point
    width: float
    height: float
    png_base64: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _point(self.origin))
        if not all(isfinite(float(value)) and value > 0 for value in (self.width, self.height)):
            raise ValueError("raster-fill dimensions must be finite and positive")
        if not self.png_base64:
            raise ValueError("raster fills require PNG data")


Fill: TypeAlias = FlatFill | LinearGradientFill | RadialGradientFill | RasterFill


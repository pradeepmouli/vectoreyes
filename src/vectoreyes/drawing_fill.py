"""Immutable, typed fill values for canonical drawings."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace as dataclass_replace
from math import isfinite
from typing import TypeAlias

from .path_geometry import AffineTransform, Point, _point


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


def _gradient_stops(stops: Iterable[GradientStop]) -> tuple[GradientStop, ...]:
    stops = tuple(stops)
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
    """A linear gradient whose affine maps paint-local to drawing coordinates."""

    start: Point
    end: Point
    stops: tuple[GradientStop, ...]
    transform: AffineTransform = AffineTransform()

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _point(self.start))
        object.__setattr__(self, "end", _point(self.end))
        object.__setattr__(self, "stops", _gradient_stops(self.stops))
        if not isinstance(self.transform, AffineTransform):
            raise TypeError("linear-gradient transform must be AffineTransform")


@dataclass(frozen=True)
class RadialGradientFill:
    """A radial gradient with an exact paint-local to drawing affine."""

    center: Point
    radius: float
    stops: tuple[GradientStop, ...]
    focal_point: Point | None = None
    transform: AffineTransform = AffineTransform()

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _point(self.center))
        if not isfinite(float(self.radius)) or self.radius < 0:
            raise ValueError("radial-gradient radius must be finite and non-negative")
        object.__setattr__(self, "stops", _gradient_stops(self.stops))
        if self.focal_point is not None:
            object.__setattr__(self, "focal_point", _point(self.focal_point))
        if not isinstance(self.transform, AffineTransform):
            raise TypeError("radial-gradient transform must be AffineTransform")


@dataclass(frozen=True)
class RasterFill:
    """A raster color field clipped to the final region geometry."""

    origin: Point
    width: float
    height: float
    png_base64: str
    transform: AffineTransform = AffineTransform()

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _point(self.origin))
        if not all(isfinite(float(value)) and value > 0 for value in (self.width, self.height)):
            raise ValueError("raster-fill dimensions must be finite and positive")
        if not self.png_base64:
            raise ValueError("raster fills require PNG data")
        if not isinstance(self.transform, AffineTransform):
            raise TypeError("raster-fill transform must be AffineTransform")


Fill: TypeAlias = FlatFill | LinearGradientFill | RadialGradientFill | RasterFill


def compose_affine(outer: AffineTransform, inner: AffineTransform) -> AffineTransform:
    """Compose SVG-compatible transforms as ``outer(inner(point))``."""
    return AffineTransform(
        a=outer.a * inner.a + outer.c * inner.b,
        b=outer.b * inner.a + outer.d * inner.b,
        c=outer.a * inner.c + outer.c * inner.d,
        d=outer.b * inner.c + outer.d * inner.d,
        tx=outer.a * inner.tx + outer.c * inner.ty + outer.tx,
        ty=outer.b * inner.tx + outer.d * inner.ty + outer.ty,
    )


def transform_fill(fill: Fill, transform: AffineTransform) -> Fill:
    """Transform paint exactly without rewriting its local gradient geometry."""
    if isinstance(fill, FlatFill):
        return fill
    if not isinstance(transform, AffineTransform):
        raise TypeError("paint transform must be AffineTransform")
    return dataclass_replace(fill, transform=compose_affine(transform, fill.transform))


def _transform_payload(transform: AffineTransform) -> dict[str, float]:
    return {
        "a": transform.a,
        "b": transform.b,
        "c": transform.c,
        "d": transform.d,
        "tx": transform.tx,
        "ty": transform.ty,
    }


def _transform_from_payload(value: object) -> AffineTransform:
    if value is None:
        return AffineTransform()
    if not isinstance(value, Mapping):
        raise TypeError("paint transform must be a mapping")
    return AffineTransform(*(float(value[key]) for key in ("a", "b", "c", "d", "tx", "ty")))


def serialize_fill(fill: Fill) -> dict[str, object]:
    """Serialize one canonical fill through the shared public JSON shape."""
    if isinstance(fill, FlatFill):
        return {"type": "flat", "color": fill.color}
    if isinstance(fill, RasterFill):
        return {
            "type": "raster",
            "origin": fill.origin,
            "width": fill.width,
            "height": fill.height,
            "png_base64": fill.png_base64,
            "transform": _transform_payload(fill.transform),
        }
    stops = tuple({"offset": stop.offset, "color": stop.color} for stop in fill.stops)
    common = {"stops": stops, "transform": _transform_payload(fill.transform)}
    if isinstance(fill, LinearGradientFill):
        return {"type": "linear_gradient", "start": fill.start, "end": fill.end, **common}
    if isinstance(fill, RadialGradientFill):
        return {
            "type": "radial_gradient",
            "center": fill.center,
            "radius": fill.radius,
            "focal_point": fill.focal_point,
            **common,
        }
    raise TypeError(f"unsupported canonical fill value: {type(fill).__name__}")


def deserialize_fill(spec: Mapping[str, object]) -> Fill:
    """Deserialize the shared public JSON fill shape into immutable values."""
    kind = spec.get("type")
    if kind == "flat":
        return FlatFill(str(spec["color"]))
    transform = _transform_from_payload(spec.get("transform"))
    if kind == "raster":
        return RasterFill(
            spec["origin"],  # type: ignore[arg-type]
            float(spec["width"]),
            float(spec["height"]),
            str(spec["png_base64"]),
            transform,
        )
    raw_stops = spec.get("stops")
    if not isinstance(raw_stops, (tuple, list)):
        raise TypeError("gradient stops must be a sequence")
    if not all(isinstance(stop, Mapping) for stop in raw_stops):
        raise TypeError("gradient stops must contain mappings")
    stops = tuple(
        GradientStop(float(stop["offset"]), str(stop["color"]))
        for stop in raw_stops
    )
    if kind == "linear_gradient":
        return LinearGradientFill(spec["start"], spec["end"], stops, transform)  # type: ignore[arg-type]
    if kind == "radial_gradient":
        return RadialGradientFill(
            spec["center"],  # type: ignore[arg-type]
            float(spec["radius"]),
            stops,
            spec.get("focal_point"),  # type: ignore[arg-type]
            transform,
        )
    raise ValueError(f"unsupported canonical fill type: {kind!r}")

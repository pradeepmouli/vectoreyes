"""Typed engine boundary for tracing and deterministic geometry operations.

The optimizer chooses *which* operation to run over a ``Drawing``.  This
module defines the engine operations that execute those choices.  The Python
tracer can implement this protocol today; a native implementation can replace
it without changing drawing, plan, or MCP contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

from .drawing import Drawing
from .drawing_actions import DrawingAction, DrawingActionResult
from .drawing_trace import TraceOptions
from .path_geometry import PathGeometry


PixelFormat = Literal["rgb8", "rgba8"]


@dataclass(frozen=True)
class RasterImage:
    """Packed source pixels supplied to a tracing engine.

    The source alpha channel, when present, is part of the trace input rather
    than a downstream geometry constraint.  This lets a trace establish the
    baseline outline and still permits later fitting and optimization to move
    that outline.
    """

    width: int
    height: int
    pixel_format: PixelFormat
    pixels: bytes

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("raster dimensions must be positive")
        channels = 3 if self.pixel_format == "rgb8" else 4
        if len(self.pixels) != self.width * self.height * channels:
            raise ValueError("packed raster length does not match dimensions and pixel format")


@dataclass(frozen=True)
class TraceDrawingRequest:
    """Input for the source-raster to v0-drawing pipeline."""

    source: RasterImage
    options: TraceOptions


@dataclass(frozen=True)
class TraceDrawingResult:
    """The v0 drawing plus diagnostics from the deterministic trace pipeline."""

    drawing: Drawing
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class PathFitOptions:
    """Deterministic curve-fitting controls for a single closed line path."""

    epsilon: float
    max_error: float
    strategy: Literal["quadratic", "progressive", "progressive_allow_lines"] = "quadratic"

    def __post_init__(self) -> None:
        if self.epsilon <= 0 or self.max_error <= 0:
            raise ValueError("path-fit epsilon and max_error must be positive")


@dataclass(frozen=True)
class FitPathRequest:
    """Refit one line-path while preserving its region-owned provenance."""

    source: PathGeometry
    options: PathFitOptions


@dataclass(frozen=True)
class FitPathResult:
    """One newly fitted path and observable fitting diagnostics."""

    geometry: PathGeometry
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class DrawingEngine(Protocol):
    """Deterministic pre-optimizer and geometry-operation implementation."""

    def trace_drawing(self, request: TraceDrawingRequest) -> TraceDrawingResult:
        """Trace source pixels through root fit and initial stitch into v0."""

    def fit_path(self, request: FitPathRequest) -> FitPathResult:
        """Refit a closed line path using the supplied fitting strategy."""

    def apply_action(self, drawing: Drawing, action: DrawingAction) -> DrawingActionResult:
        """Apply one deterministic action and return a fresh drawing snapshot."""

"""Typed numeric fitting for one declared corner span, without scene state."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from ._fitcurve import fit_quadratic_once
from .fit import _fmt, _quadratic_max_residual
from .path_geometry import Point, QuadraticCommand


def fit_corner_span(
    points: Sequence[Sequence[float]] | np.ndarray,
    *,
    start: Point,
    end: Point,
    command_id: str,
    max_error: float,
) -> QuadraticCommand | None:
    """Fit one Q inside fixed boundaries, or reject its rounded control.

    Ownership, sample selection and publication belong to the caller. This
    kernel neither detects corners nor changes their entry/exit coordinates.
    The existing chord-parameter residual and two-decimal control policy are
    retained; preserved endpoint coordinates are not rounded.
    """
    if not math.isfinite(max_error) or max_error < 0:
        raise ValueError("corner max_error must be finite and non-negative")
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return None
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError("corner samples must be finite two-dimensional points")
    if len(pts) < 3:
        return None
    fitted, _error, _split = fit_quadratic_once(pts, max_error)
    control = tuple(float(_fmt(value)) for value in fitted[1])
    command = QuadraticCommand(id=command_id, end=end, control=control)
    candidate = np.asarray([start, command.control, command.end], dtype=float)
    if not np.isfinite(candidate).all():
        raise ValueError("corner boundaries must be finite two-dimensional points")
    if _quadratic_max_residual(pts, candidate) > max_error:
        return None
    return command

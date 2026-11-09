"""Plain data types passed between pipeline stages (IO-free, port-friendly).

`Shape` (a fitted SVG element) lives in `fit.py` next to the code that produces
it; this module holds only the inputs to fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class CoverageField:
    """Owned, immutable coverage tile with an image-space (x, y) origin."""
    values: np.ndarray
    origin: tuple[int, int]
    canvas_shape: tuple[int, int]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        x, y = self.origin
        height, width = self.canvas_shape
        if values.ndim != 2 or not values.size:
            raise ValueError("coverage tiles must be nonempty two-dimensional fields")
        if x < 0 or y < 0 or y + values.shape[0] > height or x + values.shape[1] > width:
            raise ValueError("coverage tile lies outside its canvas")
        owned = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
        object.__setattr__(self, "values", owned)
        object.__setattr__(self, "origin", (int(x), int(y)))
        object.__setattr__(self, "canvas_shape", (int(height), int(width)))

    def __reduce__(self):
        # Reconstruct through the constructor so deserialized arrays remain owned
        # and immutable, rather than inheriting NumPy's mutable pickle storage.
        return CoverageField, (self.values, self.origin, self.canvas_shape)


@dataclass
class Region:
    """One connected, single-colour area of the quantised image."""
    label: int
    mask: np.ndarray          # bool (H, W) — all non-contour consumers use this
    color_hex: str
    coverage: CoverageField | np.ndarray | None = None

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Axis:
    """Vertical axis of bilateral symmetry at image x == self.x."""
    x: float

    def reflect_x(self, x: float) -> float:
        return 2.0 * self.x - x

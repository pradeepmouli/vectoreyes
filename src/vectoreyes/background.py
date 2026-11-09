"""One background-color predicate shared by tracing and final SVG cutouts."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BackgroundColor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    tolerance: float = Field(32.0, ge=0, le=442, allow_inf_nan=False)


def background_matches(colors: np.ndarray, background: BackgroundColor) -> np.ndarray:
    """Euclidean RGB distance, identical for pixels and retained fill colors."""
    rgb = np.asarray([int(background.color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return np.linalg.norm(np.asarray(colors, dtype=float) - rgb, axis=-1) <= background.tolerance

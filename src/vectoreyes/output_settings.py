"""Independent final-paint and SVG export policy."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DrawingOutputSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_output_colors: int | None = Field(
        None, ge=1, le=256,
        description="Maximum source-derived palette for flat fills and gradient stops; null is unlimited. Gradient interpolation can create intermediate colors. Use allow_gradients=false for a strictly finite-color output.",
    )
    allow_gradients: bool = Field(
        True, description="Allow final gradient fills. False samples a flat source color for every finalized region, independently of tracing colors.",
    )
    background_cutouts: Literal["subtract", "mask", "off"] = Field(
        "subtract", description="Final SVG export only: subtract creates actual path holes in earlier paint; mask uses SVG masks; off retains background-colored paint. Editable geometry is unchanged.",
    )

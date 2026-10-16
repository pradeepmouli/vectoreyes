"""Shared automatic pass controls for native orchestration and MCP schemas."""

from pydantic import BaseModel, ConfigDict, Field


class RefinementPasses(BaseModel):
    """Enable individual geometry passes without changing their canonical order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    occlusion: bool = Field(True, description="Reconstruct occluded region geometry.")
    split_compound: bool = Field(True, description="Expose compound exteriors and cutouts as semantic regions before detection.")
    primitives: bool = Field(True, description="Recognize and fit geometric primitives.")
    clones: bool = Field(True, description="Detect congruent region families including rotation and reflection.")
    symmetry: bool = Field(True, description="Detect and enforce self-symmetry on canonical geometry.")
    simplify: bool = Field(True, description="Apply the constrained linelet simplifier, not a broad refit.")
    smooth: bool = Field(True, description="Smooth eligible consecutive curve joins.")
    stitch: bool = Field(True, description="Repair shared region boundaries after the selected geometry passes.")

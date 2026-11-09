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
    bands: bool = Field(False, description="Reconstruct source-bound constant-width band interiors while retaining fixed corners.")
    stitch: bool = Field(True, description="Repair shared region boundaries after the selected geometry passes.")


def refinement_pass_names(settings: RefinementPasses, *, no_symmetry: bool = False,
                          include_disabled: bool = False) -> tuple[str, ...]:
    """One ordered schedule shared by canonical and numerical orchestration."""
    stages = (
        ('occlusion_pass', settings.occlusion),
        ('split_compound_pass', settings.split_compound),
        ('primitives_pass', settings.primitives),
        ('clones_pass', settings.clones),
        ('symmetry_pass', settings.symmetry and not no_symmetry),
        ('linelet_simplify_pass', settings.simplify),
        ('smooth_pass', settings.smooth),
        ('corner_normalize_pass', True),
        ('bands_pass', settings.bands),
        ('seams_pass', settings.stitch),
    )
    return tuple(name for name, enabled in stages if enabled or include_disabled)

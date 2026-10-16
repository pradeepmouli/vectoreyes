"""Canonical typed actions for transforming an immutable :class:`Drawing`.

The Pydantic models in this module are the single static contract for plan
input, engine dispatch, and generated MCP JSON Schema.  Validation that needs
a retained drawing (region existence, command ownership, ordering, and span
contiguity) belongs to the action executor.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

AxisName: TypeAlias = Literal["x", "y"]
PrimitiveKind: TypeAlias = Literal["circle", "ellipse", "rect", "rounded_rect", "polygon"]
FitType: TypeAlias = Literal["line", "quadratic", "cubic", "keep"]
FitStrategy: TypeAlias = Literal["quadratic", "cubic", "progressive", "progressive_allow_lines"]
PathFitStrategy: TypeAlias = Literal["quadratic", "progressive", "progressive_allow_lines"]
OperationPurpose: TypeAlias = Literal["accuracy", "design_fidelity", "both"]
ExecutionBehavior: TypeAlias = Literal["deterministic", "exploratory"]
OperationOrigin: TypeAlias = Literal["trace", "automatic", "agent"]
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ActionContract(BaseModel):
    """Strict, frozen base for schema-generating action values."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ActionSettings(ActionContract):
    """Fully resolved settings supplied to one drawing action execution."""

    epsilon: float = Field(ge=0)
    max_error: float = Field(ge=0)
    fit_strategy: PathFitStrategy
    purpose: OperationPurpose = "accuracy"
    execution: ExecutionBehavior = "deterministic"


class ActionSettingsOverride(ActionContract):
    """Partial settings inherited over retained and plan-level settings."""

    epsilon: float | None = Field(None, ge=0)
    max_error: float | None = Field(None, ge=0)
    fit_strategy: PathFitStrategy | None = None
    purpose: OperationPurpose | None = None
    execution: ExecutionBehavior | None = None


class DrawingActionContract(ActionContract):
    """Base for top-level actions with an optional settings override."""

    settings: ActionSettingsOverride = Field(default_factory=ActionSettingsOverride)


def _unique(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


class FitPathAction(ActionContract):
    """Refit current curves; gate curve-to-original-material-polyline distance.

    Original-to-curve distance is diagnostic only. Exact retained endpoint
    correspondence is required to select the original material span.
    """

    op: Literal["fit"]
    target: str | None = Field(None, min_length=1, description="One retained version-scoped command ID.")
    between: tuple[str, str] | None = Field(
        None,
        description="Ordered [A, B] retained command IDs in one subpath. Fit from the end of A to the end of B, replacing commands after A through B inclusive. Adjacent IDs select B alone; A is unchanged.",
    )
    type: FitType | None = Field(None, description="One replacement command; an explicit cubic may inflect.")
    strategy: FitStrategy | None = Field(None, description="Recursive fitting that may emit multiple commands.")

    @model_validator(mode="after")
    def validate_forms(self) -> "FitPathAction":
        if (self.target is None) == (self.between is None):
            raise ValueError("fit requires exactly one of target or between")
        if (self.type is None) == (self.strategy is None):
            raise ValueError("fit requires exactly one of type or strategy")
        if self.strategy is not None and self.between is None:
            raise ValueError("fit.strategy requires a command-bounded span")
        if self.type == "keep" and self.target is None:
            raise ValueError("fit.type=keep requires one target command")
        return self


class RemovePathAction(ActionContract):
    op: Literal["remove"]
    target: str = Field(min_length=1, description="Retained command removed before its neighbours are rejoined.")


class MatchPathAction(ActionContract):
    op: Literal["match"]
    target: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    transform: "AffineTransformInput"


class SetParallelPathAction(ActionContract):
    op: Literal["set_parallel"]
    target: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    distance: float | None = Field(
        None,
        description="Signed perpendicular offset; omit to infer the best parallel offset from retained geometry.",
    )


class MatchLengthPathAction(ActionContract):
    op: Literal["match_length"]
    target: str = Field(min_length=1)
    reference: str = Field(min_length=1)


class AlignPathAction(ActionContract):
    op: Literal["align"]
    target: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    axes: tuple[AxisName, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def unique_axes(self) -> "AlignPathAction":
        _unique(self.axes, label="align axes")
        return self


class SnapPathAction(ActionContract):
    op: Literal["snap"]
    target: str = Field(min_length=1)
    reference: str = Field(min_length=1)


class BreakPathAction(ActionContract):
    op: Literal["break"]
    target: str = Field(min_length=1, description="Fitted command after which tangent continuity is not enforced.")


class ReversePathAction(ActionContract):
    op: Literal["reverse"]
    target: str = Field(min_length=1)


class MirrorPathAction(ActionContract):
    op: Literal["mirror"]
    target: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    axis: "SymmetryAxisInput"


class SimplifyPathAction(ActionContract):
    op: Literal["simplify"]


class NormalizeCornersPathAction(ActionContract):
    op: Literal["normalize_corners"]


PathAction = Annotated[
    FitPathAction
    | RemovePathAction
    | MatchPathAction
    | SetParallelPathAction
    | MatchLengthPathAction
    | AlignPathAction
    | SnapPathAction
    | BreakPathAction
    | ReversePathAction
    | MirrorPathAction
    | SimplifyPathAction
    | NormalizeCornersPathAction,
    Field(discriminator="op"),
]


class GradientStopInput(ActionContract):
    offset: float = Field(ge=0, le=1)
    color: str

    @model_validator(mode="after")
    def validate_color(self) -> "GradientStopInput":
        if not _HEX_COLOR.fullmatch(self.color):
            raise ValueError("gradient stop color must be #RRGGBB or #RRGGBBAA")
        return self


class FlatFillAssignment(ActionContract):
    type: Literal["flat"]
    color: str

    @model_validator(mode="after")
    def validate_color(self) -> "FlatFillAssignment":
        if not _HEX_COLOR.fullmatch(self.color):
            raise ValueError("flat fill color must be #RRGGBB or #RRGGBBAA")
        return self


class LinearGradientFillAssignment(ActionContract):
    type: Literal["linear_gradient"]
    start: tuple[float, float]
    end: tuple[float, float]
    stops: tuple[GradientStopInput, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_stops(self) -> "LinearGradientFillAssignment":
        if any(right.offset < left.offset for left, right in zip(self.stops, self.stops[1:], strict=False)):
            raise ValueError("gradient stops must be sorted by offset")
        return self


class RadialGradientFillAssignment(ActionContract):
    type: Literal["radial_gradient"]
    center: tuple[float, float]
    radius: float = Field(gt=0)
    stops: tuple[GradientStopInput, ...] = Field(min_length=1)
    focal_point: tuple[float, float] | None = None

    @model_validator(mode="after")
    def ordered_stops(self) -> "RadialGradientFillAssignment":
        if any(right.offset < left.offset for left, right in zip(self.stops, self.stops[1:], strict=False)):
            raise ValueError("gradient stops must be sorted by offset")
        return self


class RasterFillAssignment(ActionContract):
    type: Literal["raster"]
    origin: tuple[float, float]
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    png_base64: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_png(self) -> "RasterFillAssignment":
        try:
            data = base64.b64decode(self.png_base64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("raster fill must contain valid base64 PNG data") from None
        if not data.startswith(_PNG_SIGNATURE):
            raise ValueError("raster fill must contain PNG data")
        return self


FillAssignment = Annotated[
    FlatFillAssignment
    | LinearGradientFillAssignment
    | RadialGradientFillAssignment
    | RasterFillAssignment,
    Field(discriminator="type"),
]


class SymmetryAxisInput(ActionContract):
    """Reflection line shared by detected-axis diagnostics and explicit actions."""

    point: tuple[float, float] = Field(description="A point [x, y] on the reflection line in drawing coordinates.")
    angle_degrees: float = Field(description="Line direction in degrees from the positive x axis; 90 is vertical.")


class AffineTransformInput(ActionContract):
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    tx: float = 0.0
    ty: float = 0.0


class LineDivider(ActionContract):
    type: Literal["line"]
    points: tuple[tuple[float, float], tuple[float, float]]

    @model_validator(mode="after")
    def distinct_points(self) -> "LineDivider":
        if self.points[0] == self.points[1]:
            raise ValueError("line divider requires two distinct points")
        return self


class PathDivider(ActionContract):
    type: Literal["path"]
    d: str = Field(min_length=1, description="Open SVG divider whose endpoints lie on the target boundary.")


SplitDivider = Annotated[LineDivider | PathDivider, Field(discriminator="type")]


class DetectPrimitivesAction(DrawingActionContract):
    op: Literal["detect_primitives"]
    target: str | None = Field(None, min_length=1)


class DetectSymmetryAction(DrawingActionContract):
    op: Literal["detect_symmetry"]
    target: str | None = Field(None, min_length=1)


class DetectClonesAction(DrawingActionContract):
    op: Literal["detect_clones"]
    target: str | None = Field(None, min_length=1)


class MergeAction(DrawingActionContract):
    op: Literal["merge"]
    id: str = Field(min_length=1, description="ID assigned to the merged region.")
    regions: tuple[str, ...] = Field(
        min_length=1,
        description="Regions to union; one branch ID means all retained children of that branch.",
    )

    @model_validator(mode="after")
    def unique_regions(self) -> "MergeAction":
        _unique(self.regions, label="merge regions")
        return self


class SplitAction(DrawingActionContract):
    op: Literal["split"]
    target: str = Field(min_length=1)
    divider: SplitDivider


class FitPathDrawingAction(DrawingActionContract):
    """Replace one region's complete geometry with a newly fitted path."""

    op: Literal["fit_path"]
    target: str = Field(min_length=1)


class FitPrimitiveAction(DrawingActionContract):
    """Replace one region's complete geometry with a fitted primitive."""

    op: Literal["fit_primitive"]
    target: str = Field(min_length=1)
    type: PrimitiveKind


class UpdateGeometryAction(DrawingActionContract):
    op: Literal["update_geometry"]
    target: str = Field(min_length=1)
    ops: tuple[PathAction, ...] = Field(min_length=1)


class AddCornerAction(DrawingActionContract):
    """Add one source-resolved corner declaration to an exact displayed path."""

    op: Literal["add_corner"]
    target: str = Field(min_length=1)
    path_id: str = Field(min_length=1)
    feature_id: str = Field(min_length=1)
    entry_command_id: str = Field(min_length=1, description="Short fitted command ID displayed for this target/path in base_version, for example r1.p0.c4. Source provenance IDs are internal and are not accepted.")
    exit_command_id: str = Field(min_length=1, description="Short fitted command ID in the same path and base version; equal to entry for a point corner, distinct for a rounded span.")
    radius: float | None = Field(None, ge=0)


class ReplaceCornerAction(DrawingActionContract):
    """Move/resize one retained corner while preserving its feature identity."""

    op: Literal["replace_corner"]
    target: str = Field(min_length=1)
    path_id: str = Field(min_length=1)
    feature_id: str = Field(min_length=1, description="Displayed corner feature ID from the selected base version, for example r1.p0.corner2; not an internal source ID.")
    entry_command_id: str = Field(min_length=1, description="Short fitted command ID displayed for this target/path in base_version, for example r1.p0.c4. Source provenance IDs are internal and are not accepted.")
    exit_command_id: str = Field(min_length=1, description="Short fitted command ID in the same path and base version; equal to entry for a point corner, distinct for a rounded span.")
    radius: float | None = Field(None, ge=0)


class RemoveCornerAction(DrawingActionContract):
    """Remove one retained active corner declaration by stable feature ID."""

    op: Literal["remove_corner"]
    target: str = Field(min_length=1)
    path_id: str = Field(min_length=1)
    feature_id: str = Field(min_length=1, description="Displayed corner feature ID from the selected base version, for example r1.p0.corner2; not an internal source ID.")


class SetFillAction(DrawingActionContract):
    op: Literal["set_fill"]
    target: str = Field(min_length=1)
    fill: FillAssignment


class AssignCloneAction(DrawingActionContract):
    op: Literal["assign_clone"]
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    transforms: tuple[AffineTransformInput, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_regions(self) -> "AssignCloneAction":
        if self.source == self.target:
            raise ValueError("clone source and target must differ")
        return self


class SetSymmetryAction(DrawingActionContract):
    op: Literal["set_symmetry"]
    target: str = Field(min_length=1)
    axis: SymmetryAxisInput


class AlignAction(DrawingActionContract):
    op: Literal["align"]
    target: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    axes: tuple[AxisName, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_relationship(self) -> "AlignAction":
        if self.target == self.reference:
            raise ValueError("align target and reference must differ")
        _unique(self.axes, label="align axes")
        return self


class StitchAction(DrawingActionContract):
    op: Literal["stitch"]
    targets: tuple[str, ...] | None = Field(
        None,
        min_length=1,
        description="Explicit regions to reconcile; omit to stitch applicable drawing roots.",
    )
    @model_validator(mode="after")
    def unique_targets(self) -> "StitchAction":
        if self.targets is not None:
            _unique(self.targets, label="stitch targets")
        return self


class SetZOrderAction(DrawingActionContract):
    op: Literal["set_z_order"]
    targets: tuple[str, ...] = Field(min_length=1, description="Complete region paint order, back to front.")

    @model_validator(mode="after")
    def unique_targets(self) -> "SetZOrderAction":
        _unique(self.targets, label="z-order targets")
        return self


DrawingAction = Annotated[
    DetectPrimitivesAction
    | DetectSymmetryAction
    | DetectClonesAction
    | MergeAction
    | SplitAction
    | FitPathDrawingAction
    | FitPrimitiveAction
    | UpdateGeometryAction
    | AddCornerAction
    | ReplaceCornerAction
    | RemoveCornerAction
    | SetFillAction
    | AssignCloneAction
    | SetSymmetryAction
    | AlignAction
    | StitchAction
    | SetZOrderAction,
    Field(discriminator="op"),
]


class DrawingPlan(ActionContract):
    """One immutable, version-branching program over a retained drawing."""

    version: Literal["vectormark.plan.v1"]
    drawing_id: str = Field(min_length=1, description="Live drawing ID returned by trace_drawing.")
    base_version: str = Field(
        pattern=r"^v\d+(?:\.\d+)*$",
        description="Existing immutable version to branch from, such as v0 or v0.1.",
    )
    label: str | None = Field(None, max_length=200)
    settings: ActionSettingsOverride = Field(default_factory=ActionSettingsOverride)
    ops: tuple[DrawingAction, ...] = Field(
        description="Ordered drawing actions; region-local geometry actions live inside update_geometry."
    )
MatchPathAction.model_rebuild()
MirrorPathAction.model_rebuild()

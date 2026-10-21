"""Typed engine boundary for tracing and deterministic geometry operations.

The refinement chooses *which* operation to run over a ``Drawing``.  This
module defines the engine operations that execute those choices.
``PythonDrawingEngine`` is the concrete entry point; a native implementation
can replace its capabilities without changing drawing, plan, or MCP contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Generic, Literal, Protocol, TypeVar

from .drawing import Drawing
from .boundary_detail import BoundaryDetailAssessment, BoundaryDetailRequest
from .drawing_actions import ActionSettings, DrawingAction, DrawingPlan, PathAction, PathFitStrategy
from .drawing_trace import TraceOptions
from .immutable_value import freeze_value
from .path_geometry import PathGeometry
from .output_settings import DrawingOutputSettings
from .refinement_settings import RefinementPasses
from .quality import GeometryPolicy
from .revision_quality import QualityReport, RegionMatch
from .path_fit_evidence import FeatureBindingDiagnostic, FitBoundaryCorrespondence, FitPathCorrespondence, FitSpanCorrespondence


PixelFormat = Literal["rgb8", "rgba8"]
EvidenceT = TypeVar("EvidenceT")


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

    drawing_id: str
    source: RasterImage
    options: TraceOptions
    min_region_fraction: float = 0.02
    initial_stitch: bool = True
    gradient_pretrace: bool = False
    output_settings: DrawingOutputSettings = field(default_factory=DrawingOutputSettings)

    def __post_init__(self) -> None:
        if not self.drawing_id:
            raise ValueError("trace request drawing ID must not be empty")
        if not isfinite(self.min_region_fraction) or not 0 <= self.min_region_fraction < 1:
            raise ValueError("minimum region fraction must be at least zero and less than one")


@dataclass(frozen=True)
class TraceDrawingResult(Generic[EvidenceT]):
    """The v0 drawing plus diagnostics from the deterministic trace pipeline."""

    drawing: Drawing
    evidence: EvidenceT
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))


@dataclass(frozen=True)
class PathFitOptions:
    """Deterministic curve-fitting controls for a single closed line path."""

    epsilon: float
    max_error: float
    strategy: PathFitStrategy = "quadratic"

    def __post_init__(self) -> None:
        if not all(isfinite(float(value)) and value > 0 for value in (self.epsilon, self.max_error)):
            raise ValueError("path-fit epsilon and max_error must be positive")


@dataclass(frozen=True)
class FitPathRequest:
    """Refit one line-path while preserving its region-owned provenance.

    ``None`` discovers ordinary protected-line cuts as before. A supplied
    tuple carries authoritative pre-fit cuts, distinct from corner anchors;
    an empty tuple intentionally supplies no additional cuts.
    """

    source: PathGeometry
    options: PathFitOptions
    excluded_command_ids: tuple[str, ...] = ()
    fit_boundary_command_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_command_ids", tuple(self.excluded_command_ids))
        if self.fit_boundary_command_ids is not None:
            object.__setattr__(self, "fit_boundary_command_ids", tuple(self.fit_boundary_command_ids))


@dataclass(frozen=True)
class FitPathResult:
    """One newly fitted path and observable fitting diagnostics."""

    geometry: PathGeometry
    correspondence: FitPathCorrespondence
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))


@dataclass(frozen=True)
class DetectPathFeaturesRequest:
    """Discover features on an undecorated source polyline, without fitting it.

    The existing trace options supply detector settings; fitting error and
    palette settings are irrelevant here. Canvas extent is the whole drawing's
    longer side, not the target region's bounds.
    """

    source: PathGeometry
    options: TraceOptions
    canvas_extent: float
    forced_corner_command_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isfinite(self.canvas_extent) or self.canvas_extent <= 0:
            raise ValueError("feature canvas extent must be finite and positive")
        object.__setattr__(self, "forced_corner_command_ids", tuple(self.forced_corner_command_ids))


@dataclass(frozen=True)
class DetectPathFeaturesResult:
    """Original polyline with typed corners and separate non-semantic fit cuts."""

    geometry: PathGeometry
    fit_boundary_command_ids: tuple[str, ...]
    binding_diagnostics: tuple[FeatureBindingDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_boundary_command_ids", tuple(self.fit_boundary_command_ids))
        object.__setattr__(self, "binding_diagnostics", tuple(self.binding_diagnostics))


@dataclass(frozen=True)
class PathActionRequest(Generic[EvidenceT]):
    """Apply one command-addressed action to a region's current path.

    The complete drawing is supplied so relationship actions can resolve
    command references owned by other regions. Evidence validates the current
    scene and resolves material-polyline correspondence; ``base_version`` is
    the caller's selected version, not a version allocated by the engine.
    ``region_id`` identifies the path being changed; the engine must reject a
    branch or non-path target.
    """

    drawing: Drawing
    region_id: str
    action: PathAction
    settings: ActionSettings
    evidence: EvidenceT
    base_version: str

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("path-action region ID must not be empty")


@dataclass(frozen=True)
class PathActionResult:
    """Fresh path geometry returned by one path-action execution."""

    geometry: PathGeometry
    affected_command_ids: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "affected_command_ids", tuple(self.affected_command_ids))
        if len(set(self.affected_command_ids)) != len(self.affected_command_ids):
            raise ValueError("affected command IDs must be unique")
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))


@dataclass(frozen=True)
class DrawingActionRequest(Generic[EvidenceT]):
    """Execute one drawing action against a scene and its engine evidence."""

    drawing: Drawing
    evidence: EvidenceT
    action: DrawingAction
    settings: ActionSettings
    base_version: str


@dataclass(frozen=True)
class DrawingActionResult(Generic[EvidenceT]):
    """Fresh scene and evidence returned by one drawing action."""

    drawing: Drawing
    evidence: EvidenceT
    affected_region_ids: tuple[str, ...] = ()
    created_region_ids: tuple[str, ...] = ()
    retired_region_ids: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "affected_region_ids", tuple(self.affected_region_ids))
        object.__setattr__(self, "created_region_ids", tuple(self.created_region_ids))
        object.__setattr__(self, "retired_region_ids", tuple(self.retired_region_ids))
        for label, values in (
            ("affected region IDs", self.affected_region_ids),
            ("created region IDs", self.created_region_ids),
            ("retired region IDs", self.retired_region_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique")
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))


@dataclass(frozen=True)
class DrawingPassResult(Generic[EvidenceT]):
    """An executed pass carries a normal revision result, even when unchanged."""

    name: str
    status: Literal["changed", "unchanged", "disabled"]
    result: DrawingActionResult[EvidenceT] | None = None


@dataclass(frozen=True)
class AutomaticRefinementResult(DrawingActionResult[EvidenceT]):
    """Final result plus ordered, independently replayable pass revisions."""

    passes: tuple[DrawingPassResult[EvidenceT], ...] = ()


class DrawingEngine(Protocol[EvidenceT]):
    """Deterministic pre-refinement and geometry-operation implementation."""

    def trace_drawing(self, request: TraceDrawingRequest) -> TraceDrawingResult[EvidenceT]:
        """Trace source pixels through root fit and initial stitch into v0."""

    def fit_path(self, request: FitPathRequest) -> FitPathResult:
        """Refit a closed line path using the supplied fitting strategy."""

    def detect_path_features(self, request: DetectPathFeaturesRequest) -> DetectPathFeaturesResult:
        """Annotate source polyline features without fitting or changing geometry."""

    def apply_path_action(self, request: PathActionRequest[EvidenceT]) -> PathActionResult:
        """Apply one nested path action using the current drawing for references."""

    def apply_action(self, request: DrawingActionRequest[EvidenceT]) -> DrawingActionResult[EvidenceT]:
        """Apply one drawing action, delegating nested path actions as needed."""

    def apply_plan(
        self, drawing: Drawing, evidence: EvidenceT, plan: DrawingPlan,
    ) -> DrawingActionResult[EvidenceT]:
        """Apply ordered actions against one retained version and its receipts."""

    def assess_revision(
        self, reference: Drawing, before: Drawing, candidate: Drawing, *,
        policy: GeometryPolicy, reference_policy: GeometryPolicy | None = None,
        matches: Sequence[RegionMatch] | None = None,
    ) -> QualityReport:
        """Assess region-aware shape fidelity independently from paint."""

    def assess_boundary_detail(self, request: BoundaryDetailRequest) -> BoundaryDetailAssessment:
        """Compare source-supported turning independently from command counts."""

    def refine_automatically(
        self, drawing: Drawing, evidence: EvidenceT, *,
        corner_normalize: bool = False, refinement_passes: RefinementPasses | None = None,
    ) -> AutomaticRefinementResult[EvidenceT]:
        """Orchestrate selectable passes through the existing geometry operations."""

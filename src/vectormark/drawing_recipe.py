"""Immutable evidence, effective-recipe, and replay receipt values."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
import json
from typing import Literal, TypeAlias

import numpy as np
from pydantic import BaseModel

from .candidate import (
    FlatFill as LegacyFlatFill,
    LinearGradientFill as LegacyLinearGradientFill,
    RadialGradientFill as LegacyRadialGradientFill,
    RasterFill as LegacyRasterFill,
)
from .corner_recipe import CornerRecipe, CornerRecipeEdit
from .drawing import Canvas, Drawing
from .background import BackgroundColor
from .types import CoverageField
from .output_settings import DrawingOutputSettings
from .drawing_actions import (
    ActionSettings,
    OperationOrigin,
    AddCornerAction,
    AlignAction,
    AssignCloneAction,
    FitPathDrawingAction,
    FitPrimitiveAction,
    MergeAction,
    RemoveCornerAction,
    ReplaceCornerAction,
    SetFillAction,
    SetSymmetryAction,
    SetZOrderAction,
    SplitAction,
    StitchAction,
    UpdateGeometryAction,
)
from .drawing_engine import RasterImage
from .drawing_trace import GeometryTraceRegion, RootPathEvidence, TraceRegion, TraceResult
from .fit import Shape
from .immutable_value import freeze_value
from .material_pretrace import GradientSupport, MaterialPretraceResult
from .optimizer.automatic_replay import ReplayBatch, input_revision
from .optimizer.vector_region import VectorRegion
from .path_fit_evidence import FitPathCorrespondence
from .path_geometry import (
    AffineTransform, PrimitiveGeometry, PathGeometry,
    PathGeometryWithCutouts, CompoundPathGeometry,
)


ENGINE_REVISION = "python-drawing.v1"
EvidenceKind = Literal["source", "geometry-guide"]
RecipeOrigin = OperationOrigin
ApplicationStatus = Literal["applied"]
PaintResolutionScope = Literal["retain", "all", "changed"]


ConcreteDrawingAction: TypeAlias = (
    MergeAction
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
    | SetZOrderAction
)
RecipeAction: TypeAlias = ConcreteDrawingAction


def _stable(value: object) -> object:
    if isinstance(value, BaseModel):
        return _stable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _stable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "ndarray": {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "sha256": sha256(contiguous.tobytes()).hexdigest(),
            }
        }
    if isinstance(value, np.generic):
        return _stable(value.item())
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {item.name: _stable(getattr(value, item.name)) for item in fields(value)},
        }
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError(f"unsupported recipe identity value: {type(value).__qualname__}")


def _digest(value: object) -> str:
    encoded = json.dumps(
        _stable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PaintEvidence:
    revision: str
    source: RasterImage
    canvas: Canvas
    background: BackgroundColor | None = None
    output_settings: DrawingOutputSettings = field(default_factory=DrawingOutputSettings)

    def __post_init__(self) -> None:
        if (self.canvas.width, self.canvas.height) != (
            float(self.source.width),
            float(self.source.height),
        ):
            raise ValueError("paint evidence canvas must match its source")


@dataclass(frozen=True)
class GeometryEvidence:
    revision: str
    kind: EvidenceKind
    engine_revision: str
    trace: TraceResult = field(compare=False, repr=False)
    geometry_source: RasterImage = field(repr=False)
    trace_roots: tuple[VectorRegion, ...] = field(compare=False, repr=False)
    geometry_to_paint: AffineTransform = AffineTransform()

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_roots", tuple(self.trace_roots))
        if (self.geometry_source.width, self.geometry_source.height) != (
            self.trace.width,
            self.trace.height,
        ):
            raise ValueError("geometry evidence dimensions must match its trace")
        if not isinstance(self.geometry_to_paint, AffineTransform):
            raise TypeError("geometry-to-paint must be AffineTransform")
        target_paths = tuple(
            (root.region_id, root.path_id) for root in self.root_paths
        )
        source_paths = tuple(
            (root.region_id, root.fit.source.path.id) for root in self.root_paths
        )
        for name, values in (("target path", target_paths), ("source path", source_paths)):
            if len(set(values)) != len(values):
                raise ValueError(
                    f"geometry evidence composite {name} identities must be unique"
                )

    @property
    def root_paths(self) -> tuple[RootPathEvidence, ...]:
        return tuple(self.trace.root_path_evidence)


def _readonly_array(value: np.ndarray) -> np.ndarray:
    snapshot = np.array(value, copy=True)
    snapshot.setflags(write=False)
    return snapshot


def _snapshot_shape(shape: Shape | None) -> Shape | None:
    return None if shape is None else Shape(shape.kind, freeze_value(shape.params))


def _snapshot_legacy_fill(fill: object) -> object:
    if isinstance(fill, LegacyFlatFill):
        return LegacyFlatFill(fill.hex)
    if isinstance(fill, LegacyLinearGradientFill):
        return LegacyLinearGradientFill(dict(fill.geometry), list(fill.stops))
    if isinstance(fill, LegacyRadialGradientFill):
        return LegacyRadialGradientFill(dict(fill.geometry), list(fill.stops))
    if isinstance(fill, LegacyRasterFill):
        return LegacyRasterFill(dict(fill.geometry), fill.png_b64)
    return None


def _snapshot_region(region: VectorRegion) -> VectorRegion:
    raster = _readonly_array(region.raster)
    coverage = region.coverage if isinstance(region.coverage, CoverageField) else None if region.coverage is None else _readonly_array(region.coverage)
    diagnostics = freeze_value(region.diagnostics)
    if region.is_leaf:
        assert region.current is not None
        return VectorRegion(
            id=region.id,
            current=_snapshot_shape(region.current),
            original=_snapshot_shape(region.original),
            fill=_snapshot_legacy_fill(region.fill),
            z=region.z,
            footprint=region.footprint,
            raster=raster,
            source_label=region.source_label,
            color_hex=region.color_hex,
            drawing_id=region.drawing_id,
            source_regions=region.source_regions,
            coverage=coverage,
            diagnostics=diagnostics,
        )
    return VectorRegion.branch(
        id=region.id,
        original=_snapshot_shape(region.original),
        footprint=region.footprint,
        children=tuple(_snapshot_region(child) for child in region.children),
        z=region.z,
        raster=raster,
        fill=_snapshot_legacy_fill(region.fill),
        source_label=region.source_label,
        color_hex=region.color_hex,
        drawing_id=region.drawing_id,
        source_regions=region.source_regions,
        diagnostics=diagnostics,
    )


def _snapshot_material_pretrace(
    pretrace: MaterialPretraceResult | None,
) -> MaterialPretraceResult | None:
    if pretrace is None:
        return None
    arrays = tuple(
        _readonly_array(array)
        for array in (
            pretrace.high_color_quantized,
            pretrace.shape_merged,
            pretrace.color_field_merged,
            pretrace.material_proxy,
        )
    )
    return MaterialPretraceResult(
        high_color_quantized=arrays[0],
        shape_merged=arrays[1],
        color_field_merged=arrays[2],
        material_proxy=arrays[3],
        shape_region_count=pretrace.shape_region_count,
        color_region_count=pretrace.color_region_count,
        gradient_supports=tuple(
            GradientSupport(
                id=support.id,
                area=support.area,
                model=freeze_value(support.model),
                representative_rgb=support.representative_rgb,
            )
            for support in pretrace.gradient_supports
        ),
        resolved_max_sampling_colors=pretrace.resolved_max_sampling_colors,
    )


def _snapshot_trace(trace: TraceResult) -> TraceResult:
    regions = tuple(
        TraceRegion(
            id=region.id,
            source_label=region.source_label,
            color=region.color,
            mask=_readonly_array(region.mask),
            contours=tuple(_readonly_array(contour) for contour in region.contours),
            trace_path=deepcopy(region.trace_path),
            effective_trace_level=region.effective_trace_level,
        )
        for region in trace.regions
    )
    geometry_regions = tuple(
        GeometryTraceRegion(
            id=region.id,
            mask=_readonly_array(region.mask),
            contours=tuple(_readonly_array(contour) for contour in region.contours),
            trace_path=deepcopy(region.trace_path),
        )
        for region in trace.geometry_regions
    )
    return TraceResult(
        width=trace.width,
        height=trace.height,
        options=trace.options,
        regions=regions,
        region_map_svg=trace.region_map_svg,
        geometry_regions=geometry_regions,
        background=freeze_value(trace.background),
        material_pretrace=_snapshot_material_pretrace(trace.material_pretrace),
        root_path_evidence=deepcopy(trace.root_path_evidence),
    )


def snapshot_geometry_evidence(evidence: GeometryEvidence) -> GeometryEvidence:
    """Detach every mutable trace/optimizer input while preserving its identity."""
    trace = _snapshot_trace(evidence.trace)
    roots = tuple(_snapshot_region(root) for root in evidence.trace_roots)
    revision = _digest(
        (
            "geometry-evidence.v1",
            evidence.kind,
            evidence.engine_revision,
            trace.to_public_dict(include_region_map=True),
            trace.root_path_evidence,
            evidence.geometry_source,
            input_revision(list(roots)),
            evidence.geometry_to_paint,
        )
    )
    if revision != evidence.revision:
        raise ValueError("geometry evidence revision does not match retained inputs")
    return GeometryEvidence(
        revision=revision,
        kind=evidence.kind,
        engine_revision=evidence.engine_revision,
        trace=trace,
        geometry_source=evidence.geometry_source,
        trace_roots=roots,
        geometry_to_paint=evidence.geometry_to_paint,
    )


def build_paint_evidence(source: RasterImage, *, background: BackgroundColor | None = None, output_settings: DrawingOutputSettings | None = None) -> PaintEvidence:
    canvas = Canvas(float(source.width), float(source.height))
    settings = output_settings or DrawingOutputSettings()
    revision = _digest(("paint.v3", source, canvas, background, settings))
    return PaintEvidence(revision, source, canvas, background, settings)


def snapshot_paint_evidence(evidence: PaintEvidence) -> PaintEvidence:
    """Validate immutable paint ownership before publication."""
    expected = _digest(("paint.v3", evidence.source, evidence.canvas, evidence.background, evidence.output_settings))
    if evidence.revision != expected:
        raise ValueError("paint evidence revision does not match retained inputs")
    return PaintEvidence(expected, evidence.source, evidence.canvas, evidence.background, evidence.output_settings)


def build_geometry_evidence(
    *,
    kind: EvidenceKind,
    trace: TraceResult,
    geometry_source: RasterImage,
    trace_roots: tuple[VectorRegion, ...],
    geometry_to_paint: AffineTransform = AffineTransform(),
    engine_revision: str = ENGINE_REVISION,
) -> GeometryEvidence:
    trace_snapshot = _snapshot_trace(trace)
    roots = tuple(_snapshot_region(root) for root in trace_roots)
    revision = _digest(
        (
            "geometry-evidence.v1",
            kind,
            engine_revision,
            trace_snapshot.to_public_dict(include_region_map=True),
            trace_snapshot.root_path_evidence,
            geometry_source,
            input_revision(list(roots)),
            geometry_to_paint,
        )
    )
    return GeometryEvidence(
        revision,
        kind,
        engine_revision,
        trace_snapshot,
        geometry_source,
        roots,
        geometry_to_paint,
    )


@dataclass(frozen=True)
class VersionPathSelection:
    version_id: str
    evidence_revision: str
    recipe_revision: str
    region_ancestry: tuple[int, ...]
    path_id: str
    entry_command_id: str
    exit_command_id: str


@dataclass(frozen=True)
class SourcePathBinding:
    evidence_revision: str
    region_ancestry: tuple[int, ...]
    target_path_id: str
    source_path_id: str
    source_entry_command_id: str
    source_exit_command_id: str
    source_owned_command_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedCornerEdit:
    selection: VersionPathSelection
    binding: SourcePathBinding
    edit: CornerRecipeEdit


@dataclass(frozen=True)
class EvidenceTargetBinding:
    role: str
    evidence_revision: str
    region_ancestry: tuple[int, ...]
    target_region_id: str
    source_path_id: str | None = None
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class NamedOutputBinding:
    role: str
    operation_id: str
    output_name: str
    target_region_id: str
    command_id: str | None = None
    descendant_ancestry: tuple[int, ...] = ()
    depends_on: tuple[str, ...] = ()


RecipeTargetBinding: TypeAlias = EvidenceTargetBinding | NamedOutputBinding


@dataclass(frozen=True)
class SelectedGeometryOutcome:
    """One chosen geometry value, not a second drawing or execution scene."""

    dependency_revision: str
    target_region_id: str
    geometry: PrimitiveGeometry | PathGeometry | PathGeometryWithCutouts | CompoundPathGeometry
    shape_metadata: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dependency_revision or not self.target_region_id:
            raise ValueError("selected geometry bindings must not be empty")
        object.__setattr__(self, "shape_metadata", freeze_value(self.shape_metadata))
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))


@dataclass(frozen=True)
class RecipeOperation:
    id: str
    origin: RecipeOrigin
    action: RecipeAction
    settings: ActionSettings
    bindings: tuple[RecipeTargetBinding, ...]
    status: ApplicationStatus = "applied"
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False)
    selected_geometry: SelectedGeometryOutcome | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("recipe operation ID must not be empty")
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))


@dataclass(frozen=True)
class ResolveOriginalPaintOperation:
    """Resolve every final leaf fill from retained original paint evidence."""

    scope: PaintResolutionScope
    share_root_gradients: bool
    input_revision: str | None = None

    def __post_init__(self) -> None:
        if self.scope == "changed" and not self.input_revision:
            raise ValueError("changed paint resolution requires an input revision")
        if self.scope in {"retain", "all"} and self.input_revision is not None:
            raise ValueError(f"{self.scope} paint resolution has no changed-input revision")


@dataclass(frozen=True)
class ApplyCornerRecipeOperation:
    """Apply one source-bound path recipe to an exact retained target path."""

    recipe: CornerRecipe
    evidence_revision: str
    region_ancestry: tuple[int, ...]
    target_region_id: str
    target_path_id: str
    input_revision: str
    downstream_batches: tuple[ReplayBatch, ...] = ()
    settings: ActionSettings | None = None
    origin: OperationOrigin = "agent"

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_ancestry", tuple(self.region_ancestry))
        object.__setattr__(self, "downstream_batches", tuple(self.downstream_batches))
        if (
            not self.region_ancestry
            or not self.target_region_id
            or not self.target_path_id
            or not self.input_revision
        ):
            raise ValueError("corner recipe application binding must not be empty")


RecipeStep: TypeAlias = (
    RecipeOperation
    | ReplayBatch
    | ResolveOriginalPaintOperation
    | ApplyCornerRecipeOperation
)


@dataclass(frozen=True)
class EffectiveDrawingRecipe:
    evidence_revision: str
    engine_revision: str
    settings: ActionSettings
    steps: tuple[RecipeStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.evidence_revision or not self.engine_revision:
            raise ValueError("effective recipe bindings must not be empty")

    @property
    def operations(self) -> tuple[RecipeOperation, ...]:
        """Applied public actions projected from the authoritative step stream."""
        return tuple(step for step in self.steps if isinstance(step, RecipeOperation))

    @property
    def automatic_batches(self) -> tuple[ReplayBatch, ...]:
        """Accepted automatic batches projected from the ordered step stream."""
        return tuple(step for step in self.steps if isinstance(step, ReplayBatch))

    @property
    def corner_recipes(self) -> tuple[CornerRecipe, ...]:
        """Active corner recipes projected from their ordered applications."""
        return tuple(
            step.recipe
            for step in self.steps
            if isinstance(step, ApplyCornerRecipeOperation)
        )

    @property
    def revision(self) -> str:
        return _digest(
            (
                "effective-recipe.v1",
                self.evidence_revision,
                self.engine_revision,
                self.settings,
                self.steps,
            )
        )


@dataclass(frozen=True)
class PathReplayCorrespondence:
    binding: SourcePathBinding
    output_revision: str
    output_region_id: str
    output_region_ancestry: tuple[int, ...]
    output_path_id: str
    correspondence: FitPathCorrespondence


@dataclass(frozen=True)
class ReplayRecord:
    evidence_revision: str
    recipe_revision: str
    engine_revision: str
    output_revision: str
    paint_revision: str
    path_correspondence: tuple[PathReplayCorrespondence, ...] = ()
    action_results: tuple[Mapping[str, object], ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_correspondence", tuple(self.path_correspondence))
        object.__setattr__(
            self,
            "action_results",
            tuple(freeze_value(result) for result in self.action_results),
        )
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))


def validate_replay_record(
    *, drawing: Drawing, recipe: EffectiveDrawingRecipe, replay: ReplayRecord,
    evidence: GeometryEvidence, paint: PaintEvidence,
) -> None:
    """Validate a selected exact outcome without invoking geometry generation."""
    if recipe.engine_revision != evidence.engine_revision:
        raise ValueError("effective recipe engine revision mismatch")
    if recipe.evidence_revision != evidence.revision:
        raise ValueError("effective recipe evidence revision mismatch")
    if replay.evidence_revision != evidence.revision:
        raise ValueError("replay evidence revision mismatch")
    if replay.recipe_revision != recipe.revision:
        raise ValueError("replay recipe revision mismatch")
    if replay.engine_revision != recipe.engine_revision:
        raise ValueError("replay engine revision mismatch")
    if replay.paint_revision != paint.revision:
        raise ValueError("replay paint revision mismatch")
    if replay.output_revision != drawing_revision(drawing):
        raise ValueError("replay output revision mismatch")


def geometry_outcome_dependencies(
    operation: RecipeOperation, *, evidence: GeometryEvidence, paint: PaintEvidence,
    upstream: EffectiveDrawingRecipe, input_revision: str,
) -> str:
    """Conservatively bind a chosen fit to all inputs that could affect it."""
    return _digest((
        "selected-geometry.v1", evidence.revision, evidence.engine_revision,
        paint.revision, upstream.revision, input_revision, operation.id,
        operation.origin, operation.action, operation.settings, operation.bindings,
    ))


def drawing_revision(drawing: Drawing) -> str:
    return _digest(("drawing.v1", drawing))


def recipe_json_value(value: object) -> object:
    """Return the same deterministic JSON value used by recipe identity."""
    return _stable(value)

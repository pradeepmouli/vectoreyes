from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
from typing import Literal, TYPE_CHECKING

import numpy as np

from ..fit import Shape
from ..drawing_actions import ActionSettings, OperationOrigin
from ..immutable_value import freeze_value
from ..path_geometry import PrimitiveGeometry
from ..quality import GeometryAssessment
from ..skia_geometry import SkPath
from .region_sample import RegionSample
from .symmetry_materialize import index_region_paths

if TYPE_CHECKING:
    from ..drawing_adapter import BoundPathInput, BoundRegionInputs
    from ..drawing import DrawingRegion
    from .framework import Pass


@dataclass(frozen=True)
class PassInputRegionRef:
    batch_id: str
    input_revision: str
    ancestry: tuple[int, ...]


@dataclass(frozen=True)
class OperationOutputRegionRef:
    operation_id: str
    output_name: str
    descendant_ancestry: tuple[int, ...] = ()


RegionRef = PassInputRegionRef | OperationOutputRegionRef


@dataclass(frozen=True)
class NamedRegionOutput:
    name: str
    ancestry: tuple[int, ...]


@dataclass(frozen=True)
class FitPrimitiveOperation:
    target: PassInputRegionRef
    primitive: PrimitiveGeometry
    epsilon: float


@dataclass(frozen=True)
class TopologyPathCorrespondence:
    target: PassInputRegionRef
    input_subpath_index: int
    input_path_id: str
    input_path_revision: str
    output_ancestry: tuple[int, ...]
    output_subpath_index: int
    output_path_id: str
    output_path_revision: str
    command_count: int


@dataclass(frozen=True)
class NormalizeRetainedTopologyOperation:
    roots: tuple[PassInputRegionRef, ...]
    affected_ids: tuple[int, ...]
    path_correspondence: tuple[TopologyPathCorrespondence, ...]
    allow_external: Literal[True] = True


@dataclass(frozen=True)
class AssignCloneOperation:
    source: PassInputRegionRef
    target: PassInputRegionRef
    transform: tuple[float, float, float, float, float, float]
    symbolic: bool
    family_size: int | None = None


@dataclass(frozen=True)
class SimplifyLineletOperation:
    target: PassInputRegionRef
    covering: tuple[PassInputRegionRef, ...]
    epsilon: float
    max_error: float
    samples: int
    cubic: bool
    linelet_only: bool
    max_boundary_error: float | None
    preserve_subpaths: bool
    check_geometry: bool


@dataclass(frozen=True)
class SmoothPathOperation:
    target: PassInputRegionRef
    max_boundary_error: float


@dataclass(frozen=True)
class NormalizeCornersOperation:
    target: PassInputRegionRef
    max_error: float
    declaration: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, Mapping):
            raise ValueError("corner operation requires retained pre-fit declaration")
        object.__setattr__(self, "declaration", freeze_value(self.declaration))


@dataclass(frozen=True)
class DeclareIntrinsicSymmetryOperation:
    target: PassInputRegionRef
    axis: tuple[float, float, float]
    residual: float


@dataclass(frozen=True)
class MirrorRegionOperation:
    source: PassInputRegionRef
    target: PassInputRegionRef
    axis: tuple[float, float, float]
    transform: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ReconstructSelfSymmetryOperation:
    target: PassInputRegionRef
    dependent_clones: tuple[PassInputRegionRef, ...]
    axis: tuple[float, float, float]
    residual: float
    fit_kind: Literal["ellipse_cap", "symmetric_half", "path_half", "geometry_path"]
    side: Literal["left", "right"]
    corner_radius: float
    epsilon: float
    max_error: float
    cubic: bool
    owner_roots: tuple[PassInputRegionRef, ...] = ()


@dataclass(frozen=True)
class PathEndpointRef:
    region: PassInputRegionRef
    path_id: str
    command_id: str
    endpoint: Literal["exit"] = "exit"


@dataclass(frozen=True)
class EndpointAssignment:
    members: tuple[PathEndpointRef, ...]
    point: tuple[float, float]


@dataclass(frozen=True)
class EndpointConstraintStage:
    assignments: tuple[EndpointAssignment, ...]


@dataclass(frozen=True)
class AxisSeamConstraint:
    left: PassInputRegionRef
    right: PassInputRegionRef
    axis: Literal["x", "y", "vertices"]
    left_side: Literal["left", "right", "top", "bottom", ""]
    right_side: Literal["left", "right", "top", "bottom", ""]
    left_value: float
    right_value: float
    seam: float
    span_min: float
    span_max: float
    before_cleanup: EndpointConstraintStage | None
    after_cleanup: EndpointConstraintStage | None
    selected: Literal["midpoint", "vertex_midpoint"]


@dataclass(frozen=True)
class PathOpsUnderlapConstraint:
    left: PassInputRegionRef
    right: PassInputRegionRef
    bottom: Literal["left", "right"]
    tolerance: float


SeamPairConstraint = AxisSeamConstraint | PathOpsUnderlapConstraint


@dataclass(frozen=True)
class VertexClusterConstraint:
    stage: EndpointConstraintStage


@dataclass(frozen=True)
class SeamCommandIndexCorrespondence:
    input_index: int
    output_index: int


@dataclass(frozen=True)
class SeamPathCorrespondence:
    target: PassInputRegionRef
    subpath_index: int
    input_path_revision: str
    output_path_revision: str
    commands: tuple[SeamCommandIndexCorrespondence, ...]


@dataclass(frozen=True)
class ApplySeamConstraintsOperation:
    roots: tuple[PassInputRegionRef, ...]
    pairs: tuple[SeamPairConstraint, ...]
    clusters: tuple[VertexClusterConstraint, ...]
    clustered: tuple[PassInputRegionRef, ...]
    epsilon: float
    max_error: float
    cubic: bool
    path_correspondence: tuple[SeamPathCorrespondence, ...]


@dataclass(frozen=True)
class DependentCloneRefresh:
    clone: PassInputRegionRef
    source: PassInputRegionRef
    previous_source_id: int
    previous_source_path: tuple[int, ...]


@dataclass(frozen=True)
class RefreshDependentClonesOperation:
    roots: tuple[PassInputRegionRef, ...]
    clones: tuple[DependentCloneRefresh, ...]


@dataclass(frozen=True)
class AnnulusPrimitive:
    center: tuple[float, float]
    outer_radius: float
    inner_radius: float


@dataclass(frozen=True)
class CompletedOcclusionPrimitive:
    source: PassInputRegionRef
    output_name: str
    output_id: int
    primitive: PrimitiveGeometry | AnnulusPrimitive
    z: int


@dataclass(frozen=True)
class RetainedOcclusionSource:
    source: PassInputRegionRef
    output_name: str
    output_id: int
    z: int


@dataclass(frozen=True)
class IntersectionLens:
    source: PassInputRegionRef
    output_name: str
    output_id: int
    left_output: str
    right_output: str
    z: int


OcclusionChild = CompletedOcclusionPrimitive | RetainedOcclusionSource | IntersectionLens


@dataclass(frozen=True)
class ReconstructOcclusionOperation:
    component: tuple[PassInputRegionRef, ...]
    branch_id: int
    base_z: float
    shape_hw: tuple[int, int]
    children: tuple[OcclusionChild, ...]


AutomaticAction = (
    FitPrimitiveOperation
    | NormalizeRetainedTopologyOperation
    | AssignCloneOperation
    | SimplifyLineletOperation
    | SmoothPathOperation
    | NormalizeCornersOperation
    | DeclareIntrinsicSymmetryOperation
    | MirrorRegionOperation
    | ReconstructSelfSymmetryOperation
    | ReconstructOcclusionOperation
    | ApplySeamConstraintsOperation
    | RefreshDependentClonesOperation
)


@dataclass(frozen=True)
class AutomaticIntent:
    operation_id: str
    action: AutomaticAction
    inputs: tuple[RegionRef, ...]
    outputs: tuple[NamedRegionOutput, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))


@dataclass(frozen=True)
class CornerDiagnosticsObservation:
    operation_id: str
    target: PassInputRegionRef
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))


ProposalIntent = AutomaticIntent | CornerDiagnosticsObservation


@dataclass(frozen=True)
class AutomaticOperationRecord:
    intent: AutomaticIntent
    pass_name: str
    proposal_ids: tuple[int, ...]
    assessment: GeometryAssessment | None

    @property
    def affected(self) -> tuple[PassInputRegionRef, ...]:
        proposal_ids = set(self.proposal_ids)
        return tuple(
            ref for ref in self.intent.inputs
            if isinstance(ref, PassInputRegionRef)
            and ref.ancestry
            and int(ref.ancestry[-1]) in proposal_ids
        )

    @property
    def declared_outputs(self) -> tuple[OperationOutputRegionRef, ...]:
        return tuple(
            OperationOutputRegionRef(self.intent.operation_id, output.name)
            for output in self.intent.outputs
        )

    @property
    def created_candidates(self) -> tuple[OperationOutputRegionRef, ...]:
        affected_paths = {tuple(ref.ancestry) for ref in self.affected}
        return tuple(
            OperationOutputRegionRef(self.intent.operation_id, output.name)
            for output in self.intent.outputs
            if tuple(output.ancestry) not in affected_paths
        )

    @property
    def would_retire(self) -> tuple[PassInputRegionRef, ...]:
        output_paths = {tuple(output.ancestry) for output in self.intent.outputs}
        return tuple(ref for ref in self.affected if tuple(ref.ancestry) not in output_paths)


@dataclass(frozen=True)
class AutomaticObservationRecord:
    intent: CornerDiagnosticsObservation
    pass_name: str
    proposal_ids: tuple[int, ...]
    assessment: GeometryAssessment | None

    @property
    def affected(self) -> tuple[PassInputRegionRef, ...]:
        return (self.intent.target,)

    @property
    def declared_outputs(self) -> tuple[OperationOutputRegionRef, ...]:
        return ()

    @property
    def created_candidates(self) -> tuple[OperationOutputRegionRef, ...]:
        return ()

    @property
    def would_retire(self) -> tuple[PassInputRegionRef, ...]:
        return ()


AutomaticRecord = AutomaticOperationRecord | AutomaticObservationRecord


@dataclass(frozen=True)
class ReplayBatch:
    batch_id: str
    input_revision: str
    scope: tuple[int, ...]
    entries: tuple[AutomaticRecord, ...]
    # Low-level refinement capture has no drawing settings context. Its owner
    # supplies resolved settings, or inherits the containing recipe settings.
    settings: ActionSettings | None = None
    origin: OperationOrigin = "automatic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", tuple(self.scope))
        object.__setattr__(self, "entries", tuple(self.entries))

    @property
    def operations(self) -> tuple[AutomaticOperationRecord, ...]:
        return tuple(
            record for record in self.entries
            if isinstance(record, AutomaticOperationRecord)
        )

    @property
    def observations(self) -> tuple[AutomaticObservationRecord, ...]:
        return tuple(
            record for record in self.entries
            if isinstance(record, AutomaticObservationRecord)
        )


@dataclass(frozen=True)
class AutomaticActionOutcome:
    operation: AutomaticRecord
    status: Literal["accepted", "rejected", "rolled_back"]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))

    @property
    def intent(self) -> ProposalIntent:
        return self.operation.intent

    @property
    def assessment(self) -> GeometryAssessment | None:
        return self.operation.assessment

    @property
    def affected(self) -> tuple[PassInputRegionRef, ...]:
        return self.operation.affected

    @property
    def created(self) -> tuple[OperationOutputRegionRef, ...]:
        return self.operation.created_candidates if self.status == "accepted" else ()

    @property
    def retired(self) -> tuple[PassInputRegionRef, ...]:
        return self.operation.would_retire if self.status == "accepted" else ()


@dataclass(frozen=True)
class RefinementResult:
    objects: tuple[RegionSample, ...]
    batches: tuple[ReplayBatch, ...]
    outcomes: tuple[AutomaticActionOutcome, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "batches", tuple(self.batches))
        object.__setattr__(self, "outcomes", tuple(self.outcomes))


def draft_input_ref(obj_id: int) -> PassInputRegionRef:
    """Create a local producer binding; the framework binds its batch revision."""
    return PassInputRegionRef("", "", (int(obj_id),))


def primitive_intent(obj_id: int, primitive: PrimitiveGeometry, *, epsilon: float) -> AutomaticIntent:
    target = draft_input_ref(obj_id)
    return AutomaticIntent(
        operation_id="",
        action=FitPrimitiveOperation(target, primitive, float(epsilon)),
        inputs=(target,),
        outputs=(NamedRegionOutput("target", (int(obj_id),)),),
    )


def operation_intent(
    action: AutomaticAction,
    inputs: tuple[PassInputRegionRef, ...],
    outputs: tuple[int, ...],
) -> AutomaticIntent:
    return AutomaticIntent(
        operation_id="",
        action=action,
        inputs=inputs,
        outputs=tuple(
            NamedRegionOutput(f"region_{obj_id}", (int(obj_id),)) for obj_id in outputs
        ),
    )


def _bind_ref(ref: RegionRef, batch_id: str, input_revision: str) -> RegionRef:
    if isinstance(ref, PassInputRegionRef):
        return replace(ref, batch_id=batch_id, input_revision=input_revision)
    return ref


def bind_intent(
    intent: ProposalIntent,
    *,
    batch_id: str,
    input_revision: str,
    ordinal: int,
) -> ProposalIntent:
    if isinstance(intent, CornerDiagnosticsObservation):
        target = _bind_ref(intent.target, batch_id, input_revision)
        operation_payload = json.dumps(
            _stable({
                "batch_id": batch_id,
                "ordinal": ordinal,
                "target": target,
                "diagnostics": intent.diagnostics,
            }),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return replace(
            intent,
            operation_id=hashlib.sha256(operation_payload.encode("utf-8")).hexdigest()[:20],
            target=target,
        )
    inputs = tuple(_bind_ref(ref, batch_id, input_revision) for ref in intent.inputs)
    action = intent.action
    if isinstance(action, FitPrimitiveOperation):
        action = replace(action, target=_bind_ref(action.target, batch_id, input_revision))
    elif isinstance(action, NormalizeRetainedTopologyOperation):
        action = replace(
            action,
            roots=tuple(_bind_ref(ref, batch_id, input_revision) for ref in action.roots),
            path_correspondence=tuple(
                replace(
                    item,
                    target=_bind_ref(item.target, batch_id, input_revision),
                )
                for item in action.path_correspondence
            ),
        )
    elif isinstance(action, AssignCloneOperation):
        action = replace(
            action,
            source=_bind_ref(action.source, batch_id, input_revision),
            target=_bind_ref(action.target, batch_id, input_revision),
        )
    elif isinstance(action, SimplifyLineletOperation):
        action = replace(
            action,
            target=_bind_ref(action.target, batch_id, input_revision),
            covering=tuple(_bind_ref(ref, batch_id, input_revision) for ref in action.covering),
        )
    elif isinstance(action, (SmoothPathOperation, NormalizeCornersOperation)):
        action = replace(action, target=_bind_ref(action.target, batch_id, input_revision))
    elif isinstance(action, DeclareIntrinsicSymmetryOperation):
        action = replace(action, target=_bind_ref(action.target, batch_id, input_revision))
    elif isinstance(action, MirrorRegionOperation):
        action = replace(
            action,
            source=_bind_ref(action.source, batch_id, input_revision),
            target=_bind_ref(action.target, batch_id, input_revision),
        )
    elif isinstance(action, ReconstructSelfSymmetryOperation):
        action = replace(
            action,
            target=_bind_ref(action.target, batch_id, input_revision),
            dependent_clones=tuple(
                _bind_ref(ref, batch_id, input_revision) for ref in action.dependent_clones
            ),
            owner_roots=tuple(_bind_ref(ref, batch_id, input_revision) for ref in action.owner_roots),
        )
    elif isinstance(action, ReconstructOcclusionOperation):
        def bind_child(child: OcclusionChild) -> OcclusionChild:
            return replace(
                child,
                source=_bind_ref(child.source, batch_id, input_revision),
            )
        action = replace(
            action,
            component=tuple(_bind_ref(ref, batch_id, input_revision) for ref in action.component),
            children=tuple(bind_child(child) for child in action.children),
        )
    elif isinstance(action, ApplySeamConstraintsOperation):
        def bind_endpoint(ref: PathEndpointRef) -> PathEndpointRef:
            return replace(ref, region=_bind_ref(ref.region, batch_id, input_revision))

        def bind_stage(stage: EndpointConstraintStage | None) -> EndpointConstraintStage | None:
            if stage is None:
                return None
            return replace(stage, assignments=tuple(
                replace(assignment, members=tuple(bind_endpoint(ref) for ref in assignment.members))
                for assignment in stage.assignments
            ))

        bound_pairs: list[SeamPairConstraint] = []
        for constraint in action.pairs:
            constraint = replace(
                constraint,
                left=_bind_ref(constraint.left, batch_id, input_revision),
                right=_bind_ref(constraint.right, batch_id, input_revision),
            )
            if isinstance(constraint, AxisSeamConstraint):
                constraint = replace(
                    constraint,
                    before_cleanup=bind_stage(constraint.before_cleanup),
                    after_cleanup=bind_stage(constraint.after_cleanup),
                )
            bound_pairs.append(constraint)
        action = replace(
            action,
            roots=tuple(_bind_ref(ref, batch_id, input_revision) for ref in action.roots),
            pairs=tuple(bound_pairs),
            clusters=tuple(
                replace(cluster, stage=bind_stage(cluster.stage))
                for cluster in action.clusters
            ),
            clustered=tuple(
                _bind_ref(ref, batch_id, input_revision) for ref in action.clustered
            ),
            path_correspondence=tuple(
                replace(
                    item,
                    target=_bind_ref(item.target, batch_id, input_revision),
                )
                for item in action.path_correspondence
            ),
        )
    elif isinstance(action, RefreshDependentClonesOperation):
        action = replace(
            action,
            roots=tuple(_bind_ref(ref, batch_id, input_revision) for ref in action.roots),
            clones=tuple(
                replace(
                    clone,
                    clone=_bind_ref(clone.clone, batch_id, input_revision),
                    source=_bind_ref(clone.source, batch_id, input_revision),
                )
                for clone in action.clones
            ),
        )
    operation_payload = json.dumps(
        _stable({
            "batch_id": batch_id,
            "ordinal": ordinal,
            "action": action,
            "inputs": inputs,
        }),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    operation_id = hashlib.sha256(operation_payload.encode("utf-8")).hexdigest()[:20]
    return replace(intent, operation_id=operation_id, action=action, inputs=inputs)


def _semantic_diagnostics(value: Mapping[object, object]) -> dict[object, object]:
    observational_fields = {
        "accepted",
        "proposal_ids",
        "proposal_region_paths",
        "geometry_assessment",
        "children_optimized",
        "reason",
    }
    return {
        key: ({field: value for field, value in item.items() if field != 'path'}
              if key == 'corners' and isinstance(item, Mapping) else item)
        for key, item in value.items()
        if not (
            isinstance(item, Mapping)
            and ("geometry_assessment" in item or item.get("children_optimized") is True)
            and set(item).issubset(observational_fields)
        )
    }


def _stable(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": value.shape, "digest": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return _stable(value.item())
    if isinstance(value, SkPath):
        state = value.__getstate__()
        return {"type": "SkPath", "d": state["d"], "fill_type": int(state["fill_type"])}
    if isinstance(value, Shape):
        return {"kind": value.kind, "params": _stable(value.params)}
    if isinstance(value, RegionSample):
        footprint = getattr(value.footprint, "to_svg_path", None)
        return {
            "id": int(value.id),
            "current": _stable(value.current),
            "original": _stable(value.original),
            "fill": _stable(value.fill),
            "z": float(value.z),
            "raster": _stable(value.raster),
            "footprint": _stable(value.footprint),
            "color_hex": value.color_hex,
            "children": [_stable(child) for child in value.children],
            "source_label": value.source_label,
            "drawing_id": value.drawing_id,
            "source_regions": _stable(value.source_regions),
            "coverage": _stable(value.coverage),
            "diagnostics": _stable(_semantic_diagnostics(value.diagnostics)),
        }
    if isinstance(value, Mapping):
        pairs = [(_stable(key), _stable(item)) for key, item in value.items()]
        return {"type": "mapping", "items": sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_stable(item) for item in value]
        return {"type": "set", "items": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite values cannot participate in automatic replay identity")
        return {"type": "float", "hex": value.hex()}
    if is_dataclass(value) and type(value).__module__.startswith("vectormark"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: _stable(getattr(value, field.name)) for field in fields(value)},
        }
    raise TypeError(f"unsupported automatic replay identity value: {type(value).__qualname__}")


def input_revision(
    objects: list[RegionSample], masks: Mapping[int, np.ndarray] | None = None
) -> str:
    from ..replay_identity import RegionInput, input_records_revision

    def record(obj: RegionSample) -> RegionInput:
        return RegionInput(int(obj.id), obj.current, obj.original, obj.fill, float(obj.z),
            obj.raster, obj.color_hex, tuple(record(child) for child in obj.children),
            obj.source_label, obj.drawing_id, tuple(obj.source_regions), obj.coverage, obj.diagnostics)
    if masks is None:
        masks = {
            int(obj.id): np.asarray(obj.raster, dtype=bool)
            for obj in objects
            if obj.raster.size
        }
    return input_records_revision(tuple(record(obj) for obj in objects), masks)


def batch_identity(pass_name: str, scope: tuple[int, ...], revision: str, ordinal: int) -> str:
    return hashlib.sha256(f"{pass_name}:{scope}:{revision}:{ordinal}".encode("utf-8")).hexdigest()[:20]


def _select_bound_clone(bindings: BoundRegionInputs, matrix: tuple[float, ...]) -> DrawingRegion:
    from ..drawing_actions import AssignCloneAction
    from ..drawing_relationships import assign_clone
    return assign_clone(bindings.drawing, AssignCloneAction(op='assign_clone',
        source='source', target='target', transforms=(dict(zip(
            ('a', 'b', 'c', 'd', 'tx', 'ty'), matrix, strict=True)),)))


def apply_occlusion_choice(action: ReconstructOcclusionOperation, resolve, *, path_for=None) -> list[RegionSample]:
    """Bind observations, execute canonical reconstruction, publish the selected tree."""
    from ..candidate import FlatFill
    from ..drawing_adapter import publish_generated_tree
    from ..drawing_occlusion import OcclusionSource, reconstruct_occlusion, validate_occlusion_children

    validate_occlusion_children(action)
    references = tuple(dict.fromkeys((*action.component, *(child.source for child in action.children))))
    working = {ref: resolve(ref) for ref in references}
    def address(ref):
        source = working[ref]
        if isinstance(source.drawing_id, str):
            return source.drawing_id
        if isinstance(ref, PassInputRegionRef):
            # Public tree addresses are derived at canonical ingress from the
            # full scoped ancestry, never a bare repeated leaf ID.
            return 'r' + '-'.join(str(part) for part in ref.ancestry)
        if path_for is not None:
            return 'r' + '-'.join(str(part) for part in path_for(ref))
        raise ValueError('occlusion named output has no retained drawing address')
    sources = {}
    for ref, source in working.items():
        color = source.fill.hex if isinstance(source.fill, FlatFill) else source.color_hex
        if color is None:
            raise ValueError('occlusion replay source has no flat paint')
        sources[ref] = OcclusionSource(int(source.id), address(ref), color, source.raster)
    branch_ref = next(ref for ref in action.component if working[ref].id == action.branch_id)
    selected = reconstruct_occlusion(action, sources, region_id=address(branch_ref))
    identities = {selected.id: action.branch_id}
    def index(region):
        for child in region.children:
            identities[child.id] = int(child.id.rsplit('-', 1)[1])
            index(child)
    index(selected)
    rasters = [np.asarray(working[ref].raster, dtype=bool) for ref in action.component if working[ref].raster.size]
    raster = np.logical_or.reduce(rasters) if rasters else np.zeros(action.shape_hw, dtype=bool)
    return [publish_generated_tree(selected, identities=identities, raster_shape=action.shape_hw,
                                   retained_rasters={selected.id: raster})]


def apply_corner_normalization_choice(
    action: NormalizeCornersOperation, binding: BoundPathInput,
) -> list[RegionSample]:
    """Normalize only the selected declaration, then publish its exact values."""
    from ..drawing_automatic import select_automatic_paths
    selected = select_automatic_paths(action, binding.paths, source_text=str(binding.target.current.params['d']))
    return [binding.publish(selected.paths, path_text=selected.path_text).with_diagnostics(selected.diagnostics or {})]


def apply_simplification_choice(
    action: SimplifyLineletOperation, binding: BoundPathInput, *,
    covering: tuple[RegionSample, ...] = (), required: bool = True,
) -> list[RegionSample]:
    """Execute retained path settings; scene coverage is already selected."""
    from ..drawing_automatic import select_automatic_paths
    original = binding.target.original
    selected = select_automatic_paths(action, binding.paths, source_text=str(binding.target.current.params['d']),
        original=str(original.params['d']) if original is not None and original.kind == 'path' else None,
        fill_rule=binding.target.current.params.get('fill_rule'))
    if selected is None:
        if required:
            raise ValueError('recorded simplify operation no longer applies to its bound input')
        return []
    return [binding.publish(selected.paths, preserve_subpaths=selected.preserve_subpaths), *covering]


def apply_smoothing_choice(
    action: SmoothPathOperation, binding: BoundPathInput, *, required: bool = True,
) -> list[RegionSample]:
    """Publish canonical smoothing against its separate retained observation."""
    from ..drawing_automatic import select_automatic_paths

    selected = select_automatic_paths(action, binding.paths, source_text=str(binding.target.current.params['d']))
    if selected is None:
        if required:
            raise ValueError("recorded smooth operation no longer applies to its bound input")
        return []
    return [binding.publish(selected.paths)]


def apply_relationship_choice(
    action: AssignCloneOperation | MirrorRegionOperation, bindings: BoundRegionInputs, *,
    footprint: SkPath | None = None,
) -> list[RegionSample]:
    """Orchestrate existing canonical assignment and generic evidence publication."""
    matrix = action.transform
    selected = _select_bound_clone(bindings, matrix)
    source, target = bindings.working['source'], bindings.working['target']
    if isinstance(action, AssignCloneOperation):
        from ..replay_inputs import clone_report
        report = clone_report(int(source.id), type(target.fill).__name__, matrix, action.family_size)
        return [bindings.publish(selected, symbolic=action.symbolic,
            diagnostics={'clones': report}, footprint=footprint)]
    from ..replay_inputs import mirror_report
    return [source, bindings.publish(selected, symbolic=False, footprint=footprint,
        diagnostics={'symmetry': mirror_report(int(source.id), action.axis)})]


def apply_clone_reference_update(
    bindings: BoundRegionInputs, *, previous_source_id: int,
    previous_source_path: tuple[int, ...], refresh_footprint: bool = True,
) -> RegionSample:
    """Rebind an already discovered scoped source through canonical assignment."""
    from ..drawing_adapter import transform_working_shape
    from .region_sample import to_polygon
    clone, source = bindings.working['target'], bindings.working['source']
    if clone.current is None or clone.current.kind != 'use':
        raise ValueError('dependent clone refresh requires bound clone and source geometry')
    matrix = tuple(clone.current.params['transform'])
    selected = _select_bound_clone(bindings, matrix)
    from ..replay_inputs import clone_reference_diagnostics
    diagnostics = clone_reference_diagnostics(clone.diagnostics, source_id=int(source.id),
        source_path=bindings.paths['source'], previous_source_id=previous_source_id,
        previous_source_path=previous_source_path)
    footprint = (to_polygon(transform_working_shape(bindings.shapes['source'], matrix))
        if refresh_footprint else clone.footprint)
    return bindings.publish(selected, diagnostics=diagnostics, footprint=footprint,
        shape_metadata=clone.current.params)


from ..replay_inputs import ReplayInputResolver


class RegionResolver(ReplayInputResolver[RegionSample]):
    """Resolve revision-bound inputs and named outputs through one path index."""

    def __init__(self, objects: list[RegionSample], batch: ReplayBatch) -> None:
        super().__init__(index_region_paths(objects), batch, lambda region: index_region_paths([region]))

    def bind_inputs(self, references: Mapping[str, RegionRef]) -> BoundRegionInputs:
        from ..drawing_adapter import bind_region_inputs
        regions = {key: self.resolve(ref) for key, ref in references.items()}
        paths = {key: self.path(ref) for key, ref in references.items()}
        return bind_region_inputs(regions, paths=paths)

def apply_seam_constraints(action, resolver):
    from ..drawing_seams import SeamInput, execute_seam_constraints
    from ..drawing_adapter import transform_working_shape, publish_seam_shape, replace_private_leaves

    inputs = {}
    def bind(ref):
        key = tuple(ref.ancestry)
        if key in inputs:
            return
        region = resolver.resolve(ref)
        owner = resolver.resolve(PassInputRegionRef(ref.batch_id, ref.input_revision, (key[0],)))
        shape = region.current
        if shape is None:
            raise ValueError('seam replay target has no geometry')
        if shape.kind == 'use':
            source_id = int(shape.params.get('href_obj_id', -1))
            sources = [leaf for leaf in owner.leaves() if int(leaf.id) == source_id]
            if len(sources) != 1 or sources[0].current is None:
                raise ValueError('seam replay clone source is absent or ambiguous')
            shape = transform_working_shape(sources[0].current, shape.params['transform'])
        inputs[key] = SeamInput(shape, int(region.id), int(owner.id),
            (float(owner.z), float(region.z), int(owner.id), int(region.id)), owner.footprint)
    for pair in action.pairs:
        bind(pair.left)
        bind(pair.right)
    updates = execute_seam_constraints(action, inputs)
    replacements = {}
    for key, update in updates.items():
        ref = PassInputRegionRef(action.roots[0].batch_id, action.roots[0].input_revision, key)
        region = resolver.resolve(ref)
        replacements.setdefault(key[0], {})[tuple(key)] = publish_seam_shape(region,
            update.shape, update.paired_with, update.candidate, selected=update.selected)
    return [replace_private_leaves(resolver.resolve(root), replacements.get(root.ancestry[0], {}))
        for root in action.roots]


def apply_intrinsic_symmetry_choice(target, axis, residual):
    from ..drawing_symmetry import apply_intrinsic_symmetry
    from .symmetry_materialize import _canonical_symmetry_input

    selected = apply_intrinsic_symmetry(_canonical_symmetry_input(target), axis, residual)
    return target.with_diagnostics(selected.diagnostics)


def apply_self_symmetry_choice(snapshot, target, dependent_clones, **settings):
    from ..drawing_symmetry import reconstruct_self_symmetry
    from ..drawing_adapter import publish_collapsed_region
    from ..geometry_native import region_path
    from .symmetry_materialize import (
        _canonical_symmetry_input, _geometry_root_footprint, remap_symmetry_clones,
    )

    selected = reconstruct_self_symmetry(_canonical_symmetry_input(target),
        _geometry_root_footprint(target, target.footprint), **settings)
    replacement = publish_collapsed_region(selected, target, footprint=region_path(selected, {}))
    if not dependent_clones:
        return [replacement]
    remaining = [obj for obj in snapshot if int(obj.id) != int(target.id)]
    remapped = remap_symmetry_clones([replacement, *remaining],
        {(int(target.id),): (int(target.id),)}, before=snapshot)
    dependent_ids = {int(obj.id) for obj in dependent_clones}
    return [replacement, *(obj for obj in remapped[1:] if int(obj.id) in dependent_ids)]


def _execute_action(
    action: AutomaticAction,
    snapshot: list[RegionSample],
    batch: ReplayBatch,
    resolver: RegionResolver,
) -> list[RegionSample]:
    if isinstance(action, FitPrimitiveOperation):
        from ..drawing import DrawingRegion
        from ..drawing_adapter import publish_selected_leaf
        target = resolver.resolve(action.target)
        selected = DrawingRegion(id=target.drawing_id or f'automatic-{target.id}',
            geometry=action.primitive, diagnostics=target.diagnostics)
        return [publish_selected_leaf(selected, target)]
    if isinstance(action, NormalizeRetainedTopologyOperation):
        from .topology_binding import apply_retained_topology

        return apply_retained_topology(action, snapshot)
    if isinstance(action, AssignCloneOperation):
        return apply_relationship_choice(action,
            resolver.bind_inputs({'source': action.source, 'target': action.target}))
    if isinstance(action, SimplifyLineletOperation):
        from ..drawing_adapter import bind_path_input

        return apply_simplification_choice(action, bind_path_input(resolver.resolve(action.target)),
            covering=tuple(resolver.resolve(ref) for ref in action.covering))
    if isinstance(action, SmoothPathOperation):
          from ..drawing_adapter import bind_path_input

          return apply_smoothing_choice(action, bind_path_input(resolver.resolve(action.target)))
    if isinstance(action, NormalizeCornersOperation):
        from ..drawing_adapter import bind_path_input

        return apply_corner_normalization_choice(action, bind_path_input(resolver.resolve(action.target)))
    if isinstance(action, DeclareIntrinsicSymmetryOperation):
        from ..drawing_symmetry import Axis2D

        target = resolver.resolve(action.target)
        return [apply_intrinsic_symmetry_choice(target, Axis2D(*action.axis), action.residual)]
    if isinstance(action, MirrorRegionOperation):
        return apply_relationship_choice(action,
            resolver.bind_inputs({'source': action.source, 'target': action.target}))
    if isinstance(action, ReconstructSelfSymmetryOperation):
        if action.owner_roots:
            raise ValueError('cross-scope source transactions require canonical replay')
        from ..drawing_symmetry import Axis2D

        target = resolver.resolve(action.target)
        dependent = tuple(resolver.resolve(ref) for ref in action.dependent_clones)
        return apply_self_symmetry_choice(
            snapshot,
            target,
            dependent,
            axis=Axis2D(*action.axis),
            residual=action.residual,
            fit_kind=action.fit_kind,
            side=action.side,
            corner_radius=action.corner_radius,
            epsilon=action.epsilon,
            max_error=action.max_error,
            cubic=action.cubic,
        )
    if isinstance(action, ReconstructOcclusionOperation):
        return apply_occlusion_choice(action, resolver.resolve, path_for=resolver.path)
    if isinstance(action, ApplySeamConstraintsOperation):
        return apply_seam_constraints(action, resolver)
    if isinstance(action, RefreshDependentClonesOperation):
        from ..drawing_adapter import refresh_dependent_clones

        return refresh_dependent_clones(action, resolver)
    raise TypeError(f"unsupported automatic replay action: {type(action).__name__}")


def _execute_intent(
    intent: ProposalIntent,
    snapshot: list[RegionSample],
    batch: ReplayBatch,
    resolver: RegionResolver,
) -> list[RegionSample]:
    if isinstance(intent, CornerDiagnosticsObservation):
        target = resolver.resolve(intent.target)
        return [target.with_diagnostics({"corners": intent.diagnostics})]
    return _execute_action(intent.action, snapshot, batch, resolver)


def _replace_scope(
    roots: list[RegionSample], scope: tuple[int, ...], replacement_children: list[RegionSample]
) -> list[RegionSample]:
    if not scope:
        return replacement_children
    head, *tail = scope
    matches = [index for index, root in enumerate(roots) if int(root.id) == int(head)]
    if len(matches) != 1:
        raise ValueError(f"automatic replay scope is not unique: {scope!r}")
    index = matches[0]
    root = roots[index]
    if not tail:
        replacement = root.with_children(replacement_children)
    else:
        replacement = root.with_children(_replace_scope(list(root.children), tuple(tail), replacement_children))
    return [*roots[:index], replacement, *roots[index + 1 :]]


def _scope_objects(roots: list[RegionSample], scope: tuple[int, ...]) -> list[RegionSample]:
    current = roots
    for obj_id in scope:
        matches = [obj for obj in current if int(obj.id) == int(obj_id)]
        if len(matches) != 1:
            raise ValueError(f"automatic replay scope is not unique: {scope!r}")
        current = list(matches[0].children)
    return current


def _bound_batch_input(
    objects: list[RegionSample], batch: ReplayBatch, root_masks: Mapping[int, np.ndarray],
) -> tuple[list[RegionSample], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Validate the shared semantic receipt and bind exact raster inputs."""
    from .framework import _masks_from_objects
    from ..replay_identity import validate_batch_revision
    scoped = _scope_objects(objects, batch.scope)
    current_masks = _masks_from_objects(scoped, root_masks)
    pass_masks = {obj_id: mask.copy() for obj_id, mask in current_masks.items()}
    validate_batch_revision(input_revision(scoped, pass_masks), batch)
    return scoped, current_masks, pass_masks


def replay_batches(
    objects: list[RegionSample],
    batches: tuple[ReplayBatch, ...],
    *,
    masks: Mapping[int, np.ndarray] | None = None,
) -> list[RegionSample]:
    """Replay committed concrete actions without invoking semantic detectors."""
    roots = list(objects)
    root_masks = {
        int(obj_id): np.asarray(mask, dtype=bool).copy()
        for obj_id, mask in (masks or {}).items()
    }
    for batch in batches:
        scoped, current_masks, pass_masks = _bound_batch_input(roots, batch, root_masks)
        from .framework import _assign_replacement_masks
        snapshot = list(scoped)
        pass_original_by_id = {int(obj.id): obj for obj in snapshot}
        current = list(scoped)
        resolver = RegionResolver(snapshot, batch)
        for committed in batch.entries:
            replacements = _execute_intent(committed.intent, snapshot, batch, resolver)
            consumed = {int(obj_id) for obj_id in committed.proposal_ids}
            current = [obj for obj in current if int(obj.id) not in consumed]
            current.extend(replacements)
            replacement_masks = _assign_replacement_masks(
                replacements,
                committed.proposal_ids,
                pass_original_by_id=pass_original_by_id,
                pass_masks=pass_masks,
                current_masks=current_masks,
            )
            from .framework import _annotate_replacement
            if committed.assessment is None:
                raise ValueError("committed automatic operation requires an assessment")
            current = [
                _annotate_replacement(
                    obj,
                    pass_name=committed.pass_name,
                    proposal_ids=committed.proposal_ids,
                    raster=replacement_masks[int(obj.id)],
                    assessment=committed.assessment,
                ) if int(obj.id) in {int(item.id) for item in replacements} else obj
                for obj in current
            ]
            current.sort(key=lambda obj: (float(obj.z), int(obj.id)))
            if isinstance(committed.intent, AutomaticIntent):
                by_path = index_region_paths(replacements)
                for output in committed.intent.outputs:
                    try:
                        region = by_path[output.ancestry]
                    except KeyError as exc:
                        raise ValueError(
                            f"automatic replay named output is absent: {output.name!r}"
                        ) from exc
                    resolver.register(committed.intent.operation_id, output, region)
        for obj_id in tuple(root_masks):
            if obj_id in {int(obj.id) for obj in scoped}:
                root_masks.pop(obj_id, None)
        root_masks.update({obj_id: mask.copy() for obj_id, mask in current_masks.items()})
        roots = _replace_scope(roots, batch.scope, current)
    return roots


def select_refinements_with_intent(
    objects: list[RegionSample],
    masks: dict[int, np.ndarray],
    passes: list[Pass],
    **kwargs,
) -> RefinementResult:
    from .framework import select_refinements_with_intent as _optimize_with_intent
    return _optimize_with_intent(objects, masks, passes, **kwargs)

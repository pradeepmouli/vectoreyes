"""Bind public structural declarations to existing retained engine operations."""

from .drawing import _walk_regions
from .drawing_actions import (
    AnnulusPrimitiveInput, CompletedOcclusionInput, NormalizeTopologyAction,
    OcclusionLensInput, ReconstructOcclusionAction, OcclusionPrimitiveInput,
)
from .drawing_proposals import CanonicalDetectionResult, CanonicalProposal, _select_region_scopes, select_canonical_pass
from .path_geometry import (
    CirclePrimitive, EllipsePrimitive, PolygonPrimitive, PrimitiveGeometry,
    RectanglePrimitive, RoundedRectanglePrimitive, CloneGeometry,
)
from .quality import GeometryPolicy
from .refinement.automatic_replay import (
    AnnulusPrimitive, CompletedOcclusionPrimitive, IntersectionLens,
    ReconstructOcclusionOperation, RetainedOcclusionSource, draft_input_ref, operation_intent,
    OcclusionChild,
)
from .replay_state import ReplayState, retained_region_raster


def _primitive(value: OcclusionPrimitiveInput) -> PrimitiveGeometry | AnnulusPrimitive:
    if isinstance(value, AnnulusPrimitiveInput):
        return AnnulusPrimitive(value.center, value.outer_radius, value.inner_radius)
    if value.type == 'circle':
        return PrimitiveGeometry(CirclePrimitive(value.center, value.radius))
    if value.type == 'ellipse':
        return PrimitiveGeometry(EllipsePrimitive(value.center, value.radius_x, value.radius_y))
    if value.type == 'rect':
        return PrimitiveGeometry(RectanglePrimitive(value.origin, value.width, value.height))
    if value.type == 'rounded_rect':
        return PrimitiveGeometry(RoundedRectanglePrimitive(value.origin, value.width, value.height,
            value.radius_x, value.radius_y))
    return PrimitiveGeometry(PolygonPrimitive(value.vertices))


def reconstruct_declared_occlusion(state: ReplayState, action: ReconstructOcclusionAction) -> CanonicalDetectionResult:
    """Bind a graph without discovery; share construction, assessment and publication."""
    inputs = {item.id: item for root in state.drawing.regions for item in _walk_regions(root)}
    names = tuple(dict.fromkeys((*action.regions, *(child.source for child in action.children))))
    if any(name not in inputs for name in names):
        raise ValueError('occlusion source is not retained')
    if any(not inputs[name].is_leaf for name in names):
        raise ValueError('occlusion sources must be retained leaves')
    scope = tuple(state.region_evidence[action.target]['ancestry'])[:-1]
    if any(tuple(state.region_evidence[name]['ancestry'])[:-1] != scope for name in names):
        raise ValueError('occlusion sources must belong to one sibling scope')
    referenced = {item.geometry.source_region_id for item in inputs.values() if isinstance(item.geometry, CloneGeometry)}
    if any(name in referenced or isinstance(inputs[name].geometry, CloneGeometry) for name in action.regions):
        raise ValueError('occlusion cannot retire a live clone source or instance')
    created = {f'{action.target}-{index}' for index in range(1, len(action.children)+1)}
    if created.intersection(set(inputs) - set(action.regions)):
        raise ValueError('occlusion child address collides with a retained region')
    refs = {name: draft_input_ref(int(state.region_evidence[name]['id'])) for name in names}
    children: list[OcclusionChild] = []
    for index, child in enumerate(action.children, 1):
        ref = refs[child.source]
        selected: OcclusionChild
        if isinstance(child, CompletedOcclusionInput):
            selected = CompletedOcclusionPrimitive(ref, child.id, index, _primitive(child.primitive), child.z)
        elif isinstance(child, OcclusionLensInput):
            selected = IntersectionLens(ref, child.id, index, child.left, child.right, child.z)
        else:
            selected = RetainedOcclusionSource(ref, child.id, index, child.z)
        children.append(selected)
    ids = tuple(int(state.region_evidence[name]['id']) for name in action.regions)
    branch_id = int(state.region_evidence[action.target]['id'])
    internal = ReconstructOcclusionOperation(tuple(refs[name] for name in action.regions), branch_id,
        min(inputs[name].z for name in action.regions) if action.base_z is None else action.base_z,
        retained_region_raster(state, action.target).shape, tuple(children))
    # Explicit declarations choose the graph. Match/admission heuristics belong
    # to detection; structural and numerical validity still apply here.
    proposal = CanonicalProposal(ids, operation_intent(internal, tuple(refs.values()), (branch_id,)),
        geometry_policy=GeometryPolicy(preserve_topology=False))
    result = select_canonical_pass(state, (proposal,), pass_name='reconstruct_occlusion', scope=scope)
    if not result.batch.operations:
        raise ValueError(f'declared occlusion was rejected: {result.outcomes}')
    return CanonicalDetectionResult(result.state, (result.batch,), result.outcomes)


def normalize_declared_topology(state: ReplayState, action: NormalizeTopologyAction) -> CanonicalDetectionResult:
    """Capture the automatic topology operation and its exact ownership receipt."""
    from .drawing_refinement import discover_topology_proposals
    return _select_region_scopes(state,
        lambda snapshot, scope: discover_topology_proposals(snapshot, scope=scope),
        target_id=None, pass_name='normalize_topology')

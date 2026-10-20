import pytest

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.automatic_replay import (
    DependentCloneRefresh,
    NormalizeCornersOperation,
    NormalizeRetainedTopologyOperation,
    PassInputRegionRef,
    RefreshDependentClonesOperation,
    SmoothPathOperation,
)
from vectormark.optimizer.symmetry_materialize import (
    index_region_paths,
    resolve_source_path,
)
from vectormark.optimizer.vector_region import VectorRegion, to_polygon


class Resolver:
    def __init__(self, roots):
        self.regions = index_region_paths(roots)

    def resolve(self, ref):
        return self.regions[ref.ancestry]


def test_engine_smoothing_respects_budget_and_preserves_source():
    from vectormark.drawing_adapter import bind_path_input
    from vectormark.optimizer.automatic_replay import apply_smoothing_choice

    shape = Shape("path", {"d": "M0 0 Q2 9.8 10 10 Q18 9.8 20 0 L0 0 Z"})
    target = VectorRegion(1, shape, FlatFill("#123456"), footprint=to_polygon(shape))
    ref = PassInputRegionRef("", "", (1,))
    bindings = bind_path_input(target)
    (changed,) = apply_smoothing_choice(SmoothPathOperation(ref, 1.1), bindings)
    assert changed.current != shape
    assert changed.fill is target.fill and changed.original is target.original
    assert target.current == shape
    with pytest.raises(ValueError, match="recorded smooth operation no longer applies"):
        apply_smoothing_choice(SmoothPathOperation(ref, 0.0), bindings)


def test_engine_topology_applies_exact_recorded_ownership_and_rejects_stale_receipt():
    from vectormark.optimizer.topology_binding import (
        apply_retained_topology,
        normalize_working_topology,
        topology_path_correspondence,
    )

    shape = Shape(
        "path",
        {
            "d": "M0 0 L4 0 L4 4 L0 4 Z M8 0 L12 0 L12 4 L8 4 Z",
            "path_ids": ("r1.p0", "r1.p1"),
        },
    )
    target = VectorRegion(1, shape, FlatFill("#123456"), drawing_id="r1")
    normalized = normalize_working_topology([target], allow_external=True)
    receipt = topology_path_correspondence([target], normalized)
    action = NormalizeRetainedTopologyOperation(
        (PassInputRegionRef("", "", (1,)),), (1,), receipt
    )
    (changed,) = apply_retained_topology(action, [target])
    assert len(changed.children) == 2
    assert [child.footprint.bounds for child in changed.children] == [
        (0.0, 0.0, 4.0, 4.0),
        (8.0, 0.0, 12.0, 4.0),
    ]
    assert all(child.fill is target.fill for child in changed.children)
    assert target.is_leaf and target.current is shape
    stale = NormalizeRetainedTopologyOperation(action.roots, (1,), ())
    with pytest.raises(ValueError, match="topology path correspondence is stale"):
        apply_retained_topology(stale, [target])


def test_engine_corner_normalization_preserves_authoritative_empty_declaration():
    from vectormark.drawing_adapter import bind_path_input
    from vectormark.optimizer.automatic_replay import apply_corner_normalization_choice
    from vectormark.corner_diagnostics import path_corner_diagnostics

    shape = Shape("path", {"d": "M0 0 L12 0 L12 12 L0 12 Z"})
    target = VectorRegion(1, shape, FlatFill("#123456"))
    declaration = path_corner_diagnostics(shape.params["d"])
    action = NormalizeCornersOperation(
        PassInputRegionRef("", "", (1,)), 1.0, declaration
    )
    binding = bind_path_input(target)
    (changed,) = apply_corner_normalization_choice(action, binding)
    assert changed.current.params['d'] == shape.params['d']
    assert changed.current.params['canonical_path_geometry'].anchors == ()
    assert changed.diagnostics["corners"]["spans"] == []
    assert target.diagnostics == {}
    stale = NormalizeCornersOperation(
        action.target, 1.0, {**declaration, "path": "M0 0 L1 0 L1 1 Z"}
    )
    with pytest.raises(ValueError, match="declaration does not match replay target"):
        apply_corner_normalization_choice(stale, binding)


def test_engine_refreshes_cross_scope_clone_without_mutating_source_or_parent():
    from vectormark.drawing_adapter import refresh_dependent_clones

    source = VectorRegion(
        10, Shape("rect", {"x": 0, "y": 0, "w": 4, "h": 6}), FlatFill("#123456")
    )
    clone = VectorRegion(
        20,
        Shape("use", {"href_obj_id": 99, "transform": (1.0, 0.0, 0.0, 1.0, 5.0, 0.0)}),
        FlatFill("#abcdef"),
        footprint=source.footprint,
        diagnostics={"clone": {"source_id": 99}},
    )
    roots = [
        VectorRegion.branch(id=1, children=(source,)),
        VectorRegion.branch(id=2, children=(clone,)),
    ]
    refs = tuple(PassInputRegionRef("", "", (i,)) for i in (1, 2))
    action = RefreshDependentClonesOperation(
        refs,
        (
            DependentCloneRefresh(
                PassInputRegionRef("", "", (2, 20)),
                PassInputRegionRef("", "", (1, 10)),
                99,
                (1, 99),
            ),
        ),
    )
    updated = refresh_dependent_clones(action, Resolver(roots))
    assert updated[0] is roots[0]
    changed = updated[1].children[0]
    assert changed.footprint.bounds == (5.0, 0.0, 9.0, 6.0)
    assert changed.fill is clone.fill
    assert changed.diagnostics["clone"]["source_id"] == 10
    assert changed.diagnostics["clone_reference_remap"][
        "previous_source_region_path"
    ] == (1, 99)
    assert resolve_source_path(
        index_region_paths(updated), (2, 20), changed.current.params
    ) == (1, 10)
    assert roots[1].children[0] is clone and clone.current.params["href_obj_id"] == 99

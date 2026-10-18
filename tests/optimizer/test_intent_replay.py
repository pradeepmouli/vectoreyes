import numpy as np
import pytest
from dataclasses import replace

from vectormark.optimizer.automatic_replay import (
    AutomaticActionOutcome,
    CornerDiagnosticsObservation,
    ApplySeamConstraintsOperation,
    AssignCloneOperation,
    AxisSeamConstraint,
    FitPrimitiveOperation,
    NormalizeCornersOperation,
    ReconstructOcclusionOperation,
    DeclareIntrinsicSymmetryOperation,
    MirrorRegionOperation,
    PathEndpointRef,
    PathOpsUnderlapConstraint,
    ReconstructSelfSymmetryOperation,
    RefreshDependentClonesOperation,
    SimplifyLineletOperation,
    SmoothPathOperation,
    draft_input_ref,
    operation_intent,
    primitive_intent,
    optimize_with_intent,
    input_revision,
    replay_batches,
)
from vectormark.optimizer.passes.primitives import primitives_pass
from vectormark.optimizer.trace import trace_regions
from vectormark.pipeline import Options


def _assert_semantic_regions_equal(actual, expected):
    assert input_revision(list(actual)) == input_revision(list(expected))
    assert [region.id for region in actual] == [region.id for region in expected]
    for left, right in zip(actual, expected, strict=True):
        assert left.current == right.current
        assert left.original == right.original
        assert left.fill == right.fill
        assert left.z == right.z
        assert np.array_equal(left.raster, right.raster)
        assert left.source_label == right.source_label
        assert left.drawing_id == right.drawing_id
        assert left.source_regions == right.source_regions
        assert left.coverage == right.coverage
        assert left.footprint.symmetric_difference(right.footprint).area < 1e-9
        _assert_semantic_regions_equal(left.children, right.children)


def _disk_image() -> np.ndarray:
    yy, xx = np.ogrid[:120, :120]
    mask = (yy - 60) ** 2 + (xx - 60) ** 2 <= 40**2
    image = np.full((120, 120, 3), 255, np.uint8)
    image[mask] = (200, 30, 30)
    return image


def test_actual_optimizer_records_and_replays_accepted_primitive_without_detection(monkeypatch):
    objects, masks = trace_regions(_disk_image(), Options())
    result = optimize_with_intent(objects, masks, [primitives_pass])

    assert len(result.batches) == 1
    assert len(result.batches[0].operations) == 1
    assert isinstance(result.batches[0].operations[0].intent.action, FitPrimitiveOperation)

    def fail_detector(*_args, **_kwargs):
        raise AssertionError("semantic primitive detection ran during replay")

    monkeypatch.setattr("vectormark.optimizer.passes.primitives.recognize_primitive", fail_detector)
    replayed = replay_batches(objects, result.batches, masks=masks)

    assert replayed[0].current == result.objects[0].current
    assert replayed[0].fill == result.objects[0].fill
    assert replayed[0].footprint.symmetric_difference(result.objects[0].footprint).area == 0.0


def test_capture_rejects_an_accepted_geometry_proposal_without_concrete_intent():
    objects, masks = trace_regions(_disk_image(), Options())

    def unrecorded(pass_objects, pass_masks):
        return [
            replace(proposal, intent=None)
            for proposal in primitives_pass(pass_objects, pass_masks)
        ]

    unrecorded.__name__ = "unrecorded_primitives"
    with pytest.raises(ValueError, match="concrete automatic intent"):
        optimize_with_intent(objects, masks, [unrecorded])
    assert objects[0].current.kind == "path"


def test_rejected_proposal_has_typed_outcome_but_no_active_replay_operation():
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.vector_region import VectorRegion

    shape = Shape(
        "path",
        {"d": "M20 70 C20 42 56 20 100 20 C144 20 180 42 180 70 L20 70 Z"},
    )
    source = VectorRegion(1, shape, FlatFill("#000000"), 0)
    masks = {1: np.zeros((120, 220), dtype=bool)}
    result = optimize_with_intent([source], masks, [primitives_pass])

    assert any(
        isinstance(outcome, AutomaticActionOutcome) and outcome.status == "rejected"
        for outcome in result.outcomes
    )
    rejected = next(outcome for outcome in result.outcomes if outcome.status == "rejected")
    assert rejected.created == ()
    assert rejected.retired == ()
    assert rejected.operation.declared_outputs
    assert all(not batch.operations for batch in result.batches)
    replayed = replay_batches([source], result.batches, masks=masks)
    assert replayed[0].current == source.current


def test_rejected_diagnostics_do_not_break_the_next_batch_input_identity():
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.vector_region import VectorRegion

    shape = Shape(
        "path",
        {"d": "M20 70 C20 42 56 20 100 20 C144 20 180 42 180 70 L20 70 Z"},
    )
    source = VectorRegion(1, shape, FlatFill("#000000"), 0)
    masks = {1: np.zeros((120, 220), dtype=bool)}
    result = optimize_with_intent([source], masks, [primitives_pass, primitives_pass])

    assert [outcome.status for outcome in result.outcomes] == ["rejected", "rejected"]
    replayed = replay_batches([source], result.batches, masks=masks)
    assert replayed[0].current == source.current


def test_semantic_diagnostic_sibling_remains_part_of_input_identity():
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.vector_region import VectorRegion

    shape = Shape("rect", {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0})
    first = VectorRegion(
        1,
        shape,
        FlatFill("#000000"),
        0,
        diagnostics={
            "symmetry_pass": {
                "accepted": True,
                "geometry_assessment": {"accepted": True},
                "axis": {"angle_degrees": 0.0},
            }
        },
    )
    second = first.with_diagnostics(
        diagnostics={
            "symmetry_pass": {
                "accepted": True,
                "geometry_assessment": {"accepted": True},
                "axis": {"angle_degrees": 90.0},
            }
        }
    )

    assert input_revision([first]) != input_revision([second])


def test_clone_intent_replays_source_identity_transform_and_independent_paint(monkeypatch):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.gate import rasterize
    from vectormark.optimizer.passes.clones import clones_pass
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon

    def square(obj_id, x, fill):
        shape = Shape("path", {"d": f"M{x} 10 L{x + 12} 10 L{x + 12} 22 L{x} 22 Z"})
        return VectorRegion(obj_id, shape, FlatFill(fill), 0, footprint=to_polygon(shape))

    source = square(1, 10, "#112233")
    target = square(2, 50, "#abcdef")
    objects = [source, target]
    masks = {obj.id: rasterize(obj.footprint, (48, 80)) for obj in objects}
    result = optimize_with_intent(objects, masks, [clones_pass])
    assert result.objects[1].current.params["href_obj_id"] == source.id

    monkeypatch.setattr(
        "vectormark.optimizer.passes.clones._best_transform",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("clone detection ran")),
    )
    replayed = replay_batches(objects, result.batches, masks=masks)
    _assert_semantic_regions_equal(replayed, result.objects)
    assert replayed[1].fill == FlatFill("#abcdef")
    assert replayed[1].current.params["href_obj_id"] == 1


def test_accepted_operation_preserves_external_masks_for_the_next_batch(monkeypatch):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.gate import rasterize
    from vectormark.optimizer.passes.clones import clones_pass
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon

    def square(obj_id, x, fill):
        shape = Shape("path", {"d": f"M{x} 10 L{x + 12} 10 L{x + 12} 22 L{x} 22 Z"})
        return VectorRegion(obj_id, shape, FlatFill(fill), obj_id, footprint=to_polygon(shape))

    source = square(1, 10, "#112233")
    target = square(2, 50, "#abcdef")
    objects = [source, target]
    masks = {obj.id: rasterize(obj.footprint, (48, 80)) for obj in objects}
    assert all(obj.raster.size == 0 for obj in objects)

    result = optimize_with_intent(objects, masks, [clones_pass, clones_pass])
    assert [
        [type(operation.intent.action).__name__ for operation in batch.operations]
        for batch in result.batches
    ] == [["AssignCloneOperation"], []]

    monkeypatch.setattr(
        "vectormark.optimizer.passes.clones._best_transform",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("clone detection ran")),
    )
    replayed = replay_batches(objects, result.batches, masks=masks)
    _assert_semantic_regions_equal(replayed, result.objects)


def test_same_pass_reads_resolve_against_the_recorded_input_snapshot():
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.framework import Proposal
    from vectormark.optimizer.gate import rasterize
    from vectormark.drawing_adapter import publish_clone_assignment
    from vectormark.drawing_adapter import transform_working_shape
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon

    points = [
        (10.0 + 10.0 * np.cos(theta), 10.0 + 10.0 * np.sin(theta))
        for theta in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    ]
    polygon_d = " ".join(
        [f"M{points[0][0]} {points[0][1]}"]
        + [f"L{x} {y}" for x, y in points[1:]]
        + ["Z"]
    )
    source_shape = Shape("path", {"d": polygon_d})
    source = VectorRegion(
        1, source_shape, FlatFill("#112233"), 0, footprint=to_polygon(source_shape)
    )
    matrix = (1.0, 0.0, 0.0, 1.0, 40.0, 0.0)
    target_shape = transform_working_shape(source_shape, matrix)
    target = VectorRegion(
        2, target_shape, FlatFill("#abcdef"), 1, footprint=to_polygon(target_shape)
    )
    primitive = Shape("circle", {"cx": 10.0, "cy": 10.0, "r": 10.0})

    def snapshot_reader(objects, _masks):
        by_id = {obj.id: obj for obj in objects}
        source_ref = draft_input_ref(1)
        target_ref = draft_input_ref(2)
        clone_action = AssignCloneOperation(source_ref, target_ref, matrix, True)
        return [
            Proposal(
                (1,),
                [source.with_current(primitive, footprint=to_polygon(primitive))],
                intent=primitive_intent(1, primitive, epsilon=1.0),
            ),
            Proposal(
                (2,),
                [publish_clone_assignment(by_id[1], by_id[2], matrix=matrix, symbolic=True)],
                intent=operation_intent(clone_action, (source_ref, target_ref), (2,)),
            ),
        ]

    objects = [source, target]
    masks = {obj.id: rasterize(obj.footprint, (32, 72)) for obj in objects}
    result = optimize_with_intent(objects, masks, [snapshot_reader])
    assert [type(op.intent.action) for op in result.batches[0].operations] == [
        FitPrimitiveOperation,
        AssignCloneOperation,
    ]
    replayed = replay_batches(objects, result.batches, masks=masks)
    _assert_semantic_regions_equal(replayed, result.objects)
    wrong_current_read = transform_working_shape(result.objects[0].current, matrix)
    assert to_polygon(wrong_current_read).symmetric_difference(result.objects[1].footprint).area > 0.1


def test_topology_normalization_replays_disconnected_named_children(monkeypatch):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.passes.compound import split_compound_pass
    from vectormark.optimizer.vector_region import VectorRegion

    d = "M0 0 L8 0 L8 8 L0 8 Z M12 0 L20 0 L20 8 L12 8 Z"
    source = VectorRegion.from_shape(
        id=1,
        shape=Shape("path", {"d": d, "fill_rule": "evenodd"}),
        fill=FlatFill("#123456"),
        z=0,
        raster=np.zeros((24, 24), bool),
        drawing_id="r1",
        source_regions=("r1",),
    )
    result = optimize_with_intent([source], {1: source.raster}, [split_compound_pass])
    assert result.objects[0].is_branch
    action = result.batches[0].operations[0].intent.action
    assert {
        (
            item.target.ancestry,
            item.input_subpath_index,
            item.output_ancestry,
            item.output_subpath_index,
        )
        for item in action.path_correspondence
    } == {
        ((1,), 0, (1, 1), 0),
        ((1,), 1, (1, 2), 0),
    }

    monkeypatch.setattr(
        "vectormark.optimizer.passes.compound.split_compound_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compound detection ran")),
    )
    replayed = replay_batches([source], result.batches)
    _assert_semantic_regions_equal(replayed, result.objects)

    stale_path = replace(
        action.path_correspondence[0], output_path_revision="stale"
    )
    stale_action = replace(
        action,
        path_correspondence=(stale_path, *action.path_correspondence[1:]),
    )
    batch = result.batches[0]
    record = batch.operations[0]
    stale_record = replace(
        record, intent=replace(record.intent, action=stale_action)
    )
    stale_batch = replace(
        batch,
        entries=tuple(
            stale_record if entry is record else entry
            for entry in batch.entries
        ),
    )
    with pytest.raises(
        ValueError, match="retained topology path correspondence is stale"
    ):
        replay_batches([source], (stale_batch, *result.batches[1:]))


def test_linelet_simplify_and_smooth_replay_their_recorded_settings(monkeypatch):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.gate import rasterize
    from vectormark.optimizer.passes.simplify import simplify_pass
    from vectormark.optimizer.passes.smooth import smooth_pass
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon

    simplify_d = (
        "M247 417.5 Q241.23 417.72 237.5 413 "
        "L237.5 405 L242 400.5 Q256.31 399.74 253.5 413 "
        "L247 417.5 Z"
    )
    simplify_shape = Shape("path", {"d": simplify_d})
    simplify_source = VectorRegion(
        1, simplify_shape, FlatFill("#123456"), 0, footprint=to_polygon(simplify_shape)
    )
    configured_simplify = lambda objects, masks: simplify_pass(
        objects, masks, epsilon=1.0, max_error=1.0, linelet_only=True,
        max_boundary_error=1.0,
    )
    configured_simplify.__name__ = "linelet_simplify_pass"
    simplify_result = optimize_with_intent(
        [simplify_source],
        {1: rasterize(simplify_source.footprint, (460, 300))},
        [configured_simplify],
    )
    simplify_action = simplify_result.batches[0].operations[0].intent.action
    assert isinstance(simplify_action, SimplifyLineletOperation)
    assert simplify_action.linelet_only is True
    assert simplify_action.epsilon == 1.0
    assert simplify_action.max_boundary_error == 1.0
    assert simplify_result.objects[0].current.params["d"] != simplify_d
    monkeypatch.setattr(
        "vectormark.optimizer.passes.simplify.simplify_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("simplify pass ran")),
    )
    _assert_semantic_regions_equal(
        replay_batches(
            [simplify_source], simplify_result.batches,
            masks={1: rasterize(simplify_source.footprint, (460, 300))},
        ),
        simplify_result.objects,
    )

    smooth_shape = Shape(
        "path", {"d": "M0 0 Q2 9.8 10 10 Q18 9.8 20 0 L0 0 Z"}
    )
    smooth_source = VectorRegion(
        2, smooth_shape, FlatFill("#123456"), 0, footprint=to_polygon(smooth_shape)
    )
    configured_smooth = lambda objects, masks: smooth_pass(objects, masks, max_error=1.1)
    configured_smooth.__name__ = "smooth_pass"
    smooth_result = optimize_with_intent(
        [smooth_source],
        {2: rasterize(smooth_source.footprint, (32, 32))},
        [configured_smooth],
    )
    smooth_action = smooth_result.batches[0].operations[0].intent.action
    assert isinstance(smooth_action, SmoothPathOperation)
    assert smooth_action.max_boundary_error == 1.1
    assert smooth_result.objects[0].current.params["d"] != smooth_shape.params["d"]
    monkeypatch.setattr(
        "vectormark.optimizer.passes.smooth.smooth_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("smooth pass ran")),
    )
    _assert_semantic_regions_equal(
        replay_batches(
            [smooth_source], smooth_result.batches,
            masks={2: rasterize(smooth_source.footprint, (32, 32))},
        ),
        smooth_result.objects,
    )


@pytest.fixture
def captured_corner_normalization():
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.corners import path_corner_diagnostics
    from vectormark.optimizer.passes.corners import corner_normalize_pass
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon

    d = "M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z"
    declaration = path_corner_diagnostics(
        d, declared_corner_spans=(((10, 0), (14, 4)),)
    )
    shape = Shape("path", {"d": d})
    source = VectorRegion(
        1,
        shape,
        FlatFill("#112233"),
        0,
        original=shape,
        footprint=to_polygon(shape),
        diagnostics={"corners": declaration},
    )
    configured = lambda objects, masks: corner_normalize_pass(
        objects, masks, max_error=1.0, enabled=True
    )
    configured.__name__ = "corner_normalize_pass"
    mask = np.ones((16, 24), dtype=bool)
    result = optimize_with_intent(
        [source], {source.id: mask}, [configured]
    )
    return source, mask, result


def test_enabled_corner_normalization_replays_recorded_prefit_declaration(
    monkeypatch, captured_corner_normalization
):
    source, mask, result = captured_corner_normalization
    d = source.current.params["d"]
    corner_action = result.batches[0].operations[0].intent.action
    assert isinstance(corner_action, NormalizeCornersOperation)
    assert corner_action.max_error == 1.0
    assert corner_action.declaration is not None
    assert result.objects[0].current.params["d"] != d
    monkeypatch.setattr(
        "vectormark.optimizer.corners.path_corner_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("corner detection ran")),
    )
    replayed = replay_batches([source], result.batches, masks={source.id: mask})
    _assert_semantic_regions_equal(replayed, result.objects)


def test_corner_operation_rejects_missing_or_stale_declaration_before_detection(
    monkeypatch, captured_corner_normalization
):
    source, mask, result = captured_corner_normalization
    batch = result.batches[0]
    record = batch.operations[0]
    action = record.intent.action
    assert isinstance(action, NormalizeCornersOperation)

    with pytest.raises(ValueError, match="requires retained pre-fit declaration"):
        replace(action, declaration=None)

    stale_action = replace(
        action,
        declaration={**action.declaration, "path": "M0 0 L1 0 L1 1 Z"},
    )
    stale_record = replace(
        record, intent=replace(record.intent, action=stale_action)
    )
    stale_batches = (replace(batch, entries=(stale_record,)), *result.batches[1:])
    monkeypatch.setattr(
        "vectormark.optimizer.corners.path_corner_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("corner detection ran")),
    )

    with pytest.raises(ValueError, match="does not match replay target"):
        replay_batches([source], stale_batches, masks={source.id: mask})


@pytest.mark.parametrize(
    "retained_corners",
    [
        None,
        {
            "path": "M0 0 L1 0 L1 1 Z",
            "anchors": [],
            "spans": [],
            "skipped": [],
        },
    ],
    ids=["missing", "stale"],
)
def test_enabled_corner_without_retained_declaration_is_observation_only(
    monkeypatch, retained_corners
):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.passes.corners import corner_normalize_pass
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon

    shape = Shape("path", {"d": "M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z"})
    source = VectorRegion(
        1,
        shape,
        FlatFill("#112233"),
        0,
        footprint=to_polygon(shape),
        diagnostics=(
            {"corners": retained_corners} if retained_corners is not None else {}
        ),
    )
    masks = {1: np.ones((16, 24), dtype=bool)}
    configured = lambda objects, pass_masks: corner_normalize_pass(
        objects, pass_masks, max_error=1.0, enabled=True
    )
    configured.__name__ = "corner_normalize_pass"
    result = optimize_with_intent([source], masks, [configured])
    assert all(not batch.operations for batch in result.batches)
    assert len(result.batches[0].observations) == 1
    assert isinstance(
        result.batches[0].observations[0].intent, CornerDiagnosticsObservation
    )
    observation_outcome = next(
        outcome
        for outcome in result.outcomes
        if isinstance(outcome.intent, CornerDiagnosticsObservation)
    )
    assert observation_outcome.status == "accepted"
    assert observation_outcome.created == ()
    assert observation_outcome.retired == ()
    assert len(observation_outcome.affected) == 1
    assert result.objects[0].current == source.current
    assert result.objects[0].diagnostics["corners"]["skipped"] == [
        {"reason": "missing_or_stale_provenance"}
    ]

    monkeypatch.setattr(
        "vectormark.optimizer.corners.path_corner_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("corner detection ran")),
    )
    replayed = replay_batches([source], result.batches, masks=masks)
    _assert_semantic_regions_equal(replayed, result.objects)


def test_authoritative_empty_corner_declaration_replays_without_redetection(monkeypatch):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.corners import path_corner_diagnostics
    from vectormark.optimizer.passes.corners import corner_normalize_pass
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon

    shape = Shape("path", {"d": "M0 0 L12 0 L12 12 L0 12 Z"})
    declaration = path_corner_diagnostics(shape.params["d"])
    assert declaration["spans"] == []
    source = VectorRegion(
        2,
        shape,
        FlatFill("#112233"),
        0,
        footprint=to_polygon(shape),
        diagnostics={"corners": declaration},
    )
    masks = {2: np.ones((16, 16), dtype=bool)}
    configured = lambda objects, pass_masks: corner_normalize_pass(
        objects, pass_masks, max_error=1.0, enabled=True
    )
    configured.__name__ = "corner_normalize_pass"
    result = optimize_with_intent([source], masks, [configured])
    action = result.batches[0].operations[0].intent.action
    assert isinstance(action, NormalizeCornersOperation)
    assert action.declaration is not None

    monkeypatch.setattr(
        "vectormark.optimizer.corners.path_corner_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("corner detection ran")),
    )
    replayed = replay_batches([source], result.batches, masks=masks)
    _assert_semantic_regions_equal(replayed, result.objects)


def test_occlusion_replays_completed_sources_and_lens_dependency_without_detection(monkeypatch):
    from vectormark.optimizer.passes.occlusion import occlusion_pass

    height, width, gap, radius = 240, 360, 58, 104
    yy, xx = np.ogrid[:height, :width]
    cx, cy = width // 2, height // 2
    left = (xx - (cx - gap)) ** 2 + (yy - cy) ** 2 <= radius**2
    right = (xx - (cx + gap)) ** 2 + (yy - cy) ** 2 <= radius**2
    image = np.full((height, width, 3), 255, np.uint8)
    image[left] = (235, 0, 27)
    image[right] = (247, 158, 27)
    image[left & right] = (255, 95, 0)
    objects, masks = trace_regions(image, Options())

    result = optimize_with_intent(objects, masks, [occlusion_pass])
    committed = [
        operation
        for batch in result.batches
        for operation in batch.operations
        if isinstance(operation.intent.action, ReconstructOcclusionOperation)
    ]
    assert len(committed) == 1
    action = committed[0].intent.action
    assert len(action.children) == 3
    assert all(not hasattr(child, "d") for child in action.children)

    monkeypatch.setattr(
        "vectormark.optimizer.passes.occlusion.reconstruct_scene",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("occlusion detection ran")),
    )
    replayed = replay_batches(objects, result.batches, masks=masks)
    _assert_semantic_regions_equal(replayed, result.objects)
    assert [child.current.kind for child in replayed[0].children] == ["circle", "circle", "path"]


def test_occlusion_replay_rejects_absent_forward_and_cyclic_child_dependencies():
    from vectormark.optimizer.passes.occlusion import occlusion_pass

    height, width, gap, radius = 240, 360, 58, 104
    yy, xx = np.ogrid[:height, :width]
    cx, cy = width // 2, height // 2
    left = (xx - (cx - gap)) ** 2 + (yy - cy) ** 2 <= radius**2
    right = (xx - (cx + gap)) ** 2 + (yy - cy) ** 2 <= radius**2
    image = np.full((height, width, 3), 255, np.uint8)
    image[left] = (235, 0, 27)
    image[right] = (247, 158, 27)
    image[left & right] = (255, 95, 0)
    objects, masks = trace_regions(image, Options())
    result = optimize_with_intent(objects, masks, [occlusion_pass])
    batch = next(batch for batch in result.batches if batch.operations)
    committed = batch.operations[0]
    action = committed.intent.action
    assert isinstance(action, ReconstructOcclusionOperation)
    first, second, lens = action.children

    malformed = {
        "absent dependency": replace(
            action, children=(first, second, replace(lens, left_output="absent"))
        ),
        "forward dependency": replace(action, children=(lens, first, second)),
        "dependency cycle": replace(
            action,
            children=(
                first,
                second,
                replace(lens, output_name="lens_a", left_output="lens_b", right_output=first.output_name),
                replace(lens, output_name="lens_b", left_output="lens_a", right_output=second.output_name),
            ),
        ),
    }
    for message, bad_action in malformed.items():
        bad_committed = replace(
            committed, intent=replace(committed.intent, action=bad_action)
        )
        bad_batches = tuple(
            replace(candidate, entries=(bad_committed,)) if candidate is batch else candidate
            for candidate in result.batches
        )
        with pytest.raises(ValueError, match=message):
            replay_batches(objects, bad_batches, masks=masks)


def test_stitch_replays_axis_cluster_and_pathops_constraints_without_discovery(monkeypatch):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.gate import rasterize
    from vectormark.optimizer.passes.seams import seams_pass
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon
    from vectormark.skia_geometry import SkPath

    def region(obj_id, d, z=0):
        shape = Shape("path", {"d": d})
        return VectorRegion(obj_id, shape, FlatFill(f"#{obj_id:06x}"), z, footprint=to_polygon(shape))

    axis_objects = [
        region(1, "M10 10 L49.2 10 L49.2 70 L10 70 Z"),
        region(2, "M50.4 10 L90 10 L90 70 L50.4 70 Z", 1),
    ]
    axis_masks = {obj.id: rasterize(obj.footprint, (80, 120)) for obj in axis_objects}
    axis_result = optimize_with_intent(axis_objects, axis_masks, [seams_pass])
    axis_action = axis_result.batches[0].operations[0].intent.action
    assert isinstance(axis_action, ApplySeamConstraintsOperation)
    assert isinstance(axis_action.pairs[0], AxisSeamConstraint)
    assert axis_action.path_correspondence
    assert all(
        len(item.commands) >= 4 for item in axis_action.path_correspondence
    )

    cluster_objects = [
        region(11, "M10 10 L49.2 10 L49.1 49.4 L10 49.4 Z"),
        region(12, "M49.8 10 L90 10 L90 49.8 L49.8 49.8 Z", 1),
        region(13, "M10 50.3 L50.3 50.3 L90 90 L10 90 Z", 2),
    ]
    cluster_masks = {obj.id: rasterize(obj.footprint, (100, 110)) for obj in cluster_objects}
    cluster_result = optimize_with_intent(cluster_objects, cluster_masks, [seams_pass])
    cluster_action = cluster_result.batches[0].operations[0].intent.action
    assert isinstance(cluster_action, ApplySeamConstraintsOperation)
    assert cluster_action.clusters
    assert all(
        isinstance(endpoint, PathEndpointRef)
        for cluster in cluster_action.clusters
        for assignment in cluster.stage.assignments
        for endpoint in assignment.members
    )

    under = region(21, "M0 0 L100 0 L0 100 Z")
    top = region(22, "M100 1 L100 100 L1 100 Z", 1)
    branch = VectorRegion.branch(
        id=20,
        children=[under, top],
        z=0,
        footprint=SkPath(shell=[(0, 0), (100, 0), (100, 100), (0, 100)]),
    )
    pathops_masks = {20: rasterize(branch.footprint, (110, 110))}
    pathops_result = optimize_with_intent([branch], pathops_masks, [seams_pass])
    pathops_operations = [
        operation
        for batch in pathops_result.batches
        for operation in batch.operations
        if isinstance(operation.intent.action, ApplySeamConstraintsOperation)
    ]
    assert len(pathops_operations) == 1
    pathops_action = pathops_operations[0].intent.action
    assert isinstance(pathops_action, ApplySeamConstraintsOperation)
    assert isinstance(pathops_action.pairs[0], PathOpsUnderlapConstraint)
    pathops_pair = pathops_action.pairs[0]
    bottom = (
        tuple(pathops_pair.left.ancestry)
        if pathops_pair.bottom == "left"
        else tuple(pathops_pair.right.ancestry)
    )
    top = (
        tuple(pathops_pair.right.ancestry)
        if pathops_pair.bottom == "left"
        else tuple(pathops_pair.left.ancestry)
    )
    bottom_receipt = next(
        item
        for item in pathops_action.path_correspondence
        if tuple(item.target.ancestry) == bottom
    )
    top_receipt = next(
        item
        for item in pathops_action.path_correspondence
        if tuple(item.target.ancestry) == top
    )
    bottom_map = {
        item.input_index: item.output_index for item in bottom_receipt.commands
    }
    assert bottom_map[2] == 3
    assert bottom_map.get(2) != 2
    assert 0 not in bottom_map
    assert top_receipt.commands

    monkeypatch.setattr(
        "vectormark.optimizer.passes.seams._candidate_for_pair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("seam pair discovery ran")),
    )
    monkeypatch.setattr(
        "vectormark.optimizer.passes.seams._cluster_leaf_vertices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("seam cluster discovery ran")),
    )
    _assert_semantic_regions_equal(
        replay_batches(axis_objects, axis_result.batches, masks=axis_masks), axis_result.objects
    )
    _assert_semantic_regions_equal(
        replay_batches(cluster_objects, cluster_result.batches, masks=cluster_masks),
        cluster_result.objects,
    )
    _assert_semantic_regions_equal(
        replay_batches([branch], pathops_result.batches, masks=pathops_masks),
        pathops_result.objects,
    )


def test_seam_correspondence_retains_unaffected_commands_but_not_replaced_cubic():
    from vectormark.fit import Shape
    from vectormark.optimizer.automatic_replay import PassInputRegionRef
    from vectormark.drawing_seams import (
        _cleanup_inflecting_cubics,
        _path_correspondence,
    )

    before = Shape(
        "path",
        {
            "d": (
                "M166.88 484.26 "
                "C164.99 482.5 163.09 480.72 161.19 478.95 "
                "L156.96 472.09 L166.88 484.26 Z"
            )
        },
    )
    after = _cleanup_inflecting_cubics(
        before, max_error=1.0, line_epsilon=1.0
    )
    (receipt,) = _path_correspondence(
        PassInputRegionRef("", "", (1,)),
        before,
        after,
        positional_mutation=True,
    )
    by_input = {item.input_index: item.output_index for item in receipt.commands}

    assert 0 in by_input
    assert 1 not in by_input
    assert 2 in by_input and 3 in by_input


def test_symmetry_modes_replay_exact_axis_winner_without_candidate_selection(monkeypatch):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.skia_geometry import SkPath
    from vectormark.optimizer.gate import rasterize
    from vectormark.optimizer.passes.symmetry import symmetry_pass
    from vectormark.optimizer.vector_region import VectorRegion

    def path_region(obj_id, polygon):
        coords = list(polygon.exterior.coords)
        d = " ".join(
            [f"M{coords[0][0]} {coords[0][1]}"]
            + [f"L{x} {y}" for x, y in coords[1:]]
            + ["Z"]
        )
        return VectorRegion(
            obj_id, Shape("path", {"d": d}), FlatFill("#112233"), 0, footprint=polygon
        )

    left = path_region(1, SkPath(shell=[(10, 20), (22, 20), (22, 34), (10, 34)]))
    right = path_region(2, SkPath(shell=[(58, 20), (70, 20), (70, 34), (58, 34)]))
    pair_objects = [left, right]
    pair_masks = {obj.id: rasterize(obj.footprint, (96, 96)) for obj in pair_objects}
    pair_result = optimize_with_intent(pair_objects, pair_masks, [symmetry_pass])
    pair_action = pair_result.batches[0].operations[0].intent.action
    assert isinstance(pair_action, MirrorRegionOperation)

    native = VectorRegion(
        3,
        Shape("rect", {"x": 20.0, "y": 20.0, "w": 40.0, "h": 40.0}),
        FlatFill("#445566"),
        0,
        footprint=SkPath(shell=[(20, 20), (60, 20), (60, 60), (20, 60)]),
    )
    intrinsic_result = optimize_with_intent(
        [native], {3: rasterize(native.footprint, (96, 96))}, [symmetry_pass]
    )
    intrinsic_action = intrinsic_result.batches[0].operations[0].intent.action
    assert isinstance(intrinsic_action, DeclareIntrinsicSymmetryOperation)
    intrinsic_diagnostics = intrinsic_result.objects[0].diagnostics["symmetry"]
    assert tuple(intrinsic_diagnostics["axis"]["point"]) == pytest.approx(
        intrinsic_action.axis[1:]
    )
    assert intrinsic_diagnostics["axis"]["angle_degrees"] == pytest.approx(
        np.degrees(intrinsic_action.axis[0])
    )
    assert intrinsic_diagnostics["residual"] == pytest.approx(intrinsic_action.residual)

    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    ellipse = SkPath(shell=[
        (48.0 + 20.0 * np.cos(theta), 48.0 + 30.0 * np.sin(theta))
        for theta in angles
    ])
    self_source = path_region(4, ellipse)
    self_result = optimize_with_intent(
        [self_source], {4: rasterize(ellipse, (96, 96))}, [symmetry_pass]
    )
    self_action = self_result.batches[0].operations[0].intent.action
    assert isinstance(self_action, ReconstructSelfSymmetryOperation)
    assert self_action.fit_kind in {"ellipse_cap", "symmetric_half", "path_half", "geometry_path"}

    monkeypatch.setattr(
        "vectormark.optimizer.passes.symmetry._best_self_reconstruction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("self selection ran")),
    )
    monkeypatch.setattr(
        "vectormark.optimizer.passes.symmetry._best_self_axis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("axis selection ran")),
    )
    monkeypatch.setattr(
        "vectormark.optimizer.passes.symmetry._mirror_pair_axis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("pair selection ran")),
    )
    _assert_semantic_regions_equal(
        replay_batches(pair_objects, pair_result.batches, masks=pair_masks), pair_result.objects
    )
    _assert_semantic_regions_equal(
        replay_batches(
            [native], intrinsic_result.batches,
            masks={3: rasterize(native.footprint, (96, 96))},
        ),
        intrinsic_result.objects,
    )
    intrinsic_batch = intrinsic_result.batches[0]
    intrinsic_record = intrinsic_batch.operations[0]
    wrong_axis_action = replace(
        intrinsic_action,
        axis=(intrinsic_action.axis[0] + 3.0, *intrinsic_action.axis[1:]),
    )
    wrong_axis_record = replace(
        intrinsic_record,
        intent=replace(intrinsic_record.intent, action=wrong_axis_action),
    )
    wrong_axis_batches = (
        replace(intrinsic_batch, entries=(wrong_axis_record,)),
        *intrinsic_result.batches[1:],
    )
    with pytest.raises(AssertionError):
        _assert_semantic_regions_equal(
            replay_batches(
                [native], wrong_axis_batches,
                masks={3: rasterize(native.footprint, (96, 96))},
            ),
            intrinsic_result.objects,
        )
    _assert_semantic_regions_equal(
        replay_batches(
            [self_source], self_result.batches,
            masks={4: rasterize(ellipse, (96, 96))},
        ),
        self_result.objects,
    )


@pytest.mark.parametrize(
    ("expected_kind", "source_kind"),
    [
        ("symmetric_half", "ellipse32"),
        ("ellipse_cap", "cap16"),
        ("geometry_path", "cap8"),
        ("path_half", "retained_path"),
    ],
)
def test_each_self_symmetry_winner_replays_only_its_recorded_fitter(
    monkeypatch, expected_kind, source_kind
):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.gate import rasterize
    from vectormark.optimizer.passes.symmetry import symmetry_pass
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon
    from vectormark.skia_geometry import SkPath

    if source_kind == "ellipse32":
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        footprint = SkPath(shell=[
            (48.0 + 20.0 * np.cos(theta), 48.0 + 30.0 * np.sin(theta))
            for theta in angles
        ])
        shape_hw = (96, 96)
    elif source_kind in {"cap16", "cap8"}:
        count = 16 if source_kind == "cap16" else 8
        angles = np.linspace(0.0, np.pi, count)
        footprint = SkPath(shell=[
            (20.0, 50.0),
            (60.0, 50.0),
            *[
                (40.0 + 20.0 * np.cos(theta), 50.0 - 30.0 * np.sin(theta))
                for theta in angles
            ][1:],
        ])
        shape_hw = (128, 128)
    else:
        shape = Shape("path", {
            "d": (
                "M390 302.5 L256 302.5 Q245.79 281.27 240.5 258 "
                "Q241.25 257.25 242 256.5 L404 256.5 Q404.75 257.25 405.5 258 "
                "Q400.2 281.26 390 302.5 Z"
            )
        })
        footprint = to_polygon(shape)
        shape_hw = (340, 460)

    if source_kind != "retained_path":
        points = list(footprint.exterior.coords)
        d = " ".join(
            [f"M{points[0][0]} {points[0][1]}"]
            + [f"L{x} {y}" for x, y in points[1:]]
            + ["Z"]
        )
        shape = Shape("path", {"d": d})
    source = VectorRegion(
        41, shape, FlatFill("#112233"), 0, footprint=footprint
    )
    masks = {41: rasterize(footprint, shape_hw)}
    configured = lambda objects, pass_masks: symmetry_pass(
        objects, pass_masks, epsilon=1.0, max_error=1.0, cubic=True
    )
    configured.__name__ = "symmetry_pass"
    result = optimize_with_intent([source], masks, [configured])
    action = result.batches[0].operations[0].intent.action
    assert isinstance(action, ReconstructSelfSymmetryOperation)
    assert action.fit_kind == expected_kind
    assert action.cubic is True

    monkeypatch.setattr(
        "vectormark.optimizer.passes.symmetry._best_self_reconstruction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("winner selection ran")),
    )
    monkeypatch.setattr(
        "vectormark.optimizer.passes.symmetry._best_self_axis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("axis selection ran")),
    )
    replayed = replay_batches([source], result.batches, masks=masks)
    _assert_semantic_regions_equal(replayed, result.objects)


@pytest.mark.parametrize("reject_outer", [False, True])
def test_recursive_intent_commits_or_rolls_back_as_one_outer_clone_transaction(
    reject_outer, monkeypatch
):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.gate import rasterize
    from vectormark.optimizer.passes.symmetry import symmetry_pass
    from vectormark.drawing_adapter import transform_working_shape
    from vectormark.optimizer.vector_region import VectorRegion, to_polygon
    from vectormark.skia_geometry import SkPath

    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    source_footprint = SkPath(shell=[
        (48.0 + 20.0 * np.cos(theta), 48.0 + 30.0 * np.sin(theta))
        for theta in angles
    ])
    points = list(source_footprint.exterior.coords)
    source_shape = Shape("path", {"d": " ".join(
        [f"M{points[0][0]} {points[0][1]}"]
        + [f"L{x} {y}" for x, y in points[1:]]
        + ["Z"]
    )})
    source_mask = rasterize(source_footprint, (120, 180))
    source = VectorRegion(
        11, source_shape, FlatFill("#123456"), 0,
        footprint=source_footprint, raster=source_mask,
    )
    native_shape = Shape("rect", {"x": 8.0, "y": 88.0, "w": 18.0, "h": 18.0})
    native = VectorRegion(
        12, native_shape, FlatFill("#334455"), 1,
        footprint=to_polygon(native_shape),
        raster=rasterize(to_polygon(native_shape), (120, 180)),
    )
    branch_mask = source.raster | native.raster
    branch = VectorRegion.branch(id=10, children=[source, native], z=0, raster=branch_mask)
    matrix = (1.0, 0.0, 0.0, 1.0, 80.0, 0.0)
    clone_footprint = to_polygon(transform_working_shape(source_shape, matrix))
    if reject_outer:
        clone_footprint = SkPath(shell=[(130, 0), (132, 0), (132, 2), (130, 2)])
    clone = VectorRegion(
        2,
        Shape("use", {"href_obj_id": 11, "transform": matrix}),
        FlatFill("#abcdef"),
        2,
        footprint=clone_footprint,
        raster=rasterize(clone_footprint, (120, 180)),
        diagnostics={"clones": {"matched_source": 11}},
    )
    objects = [branch, clone]
    masks = {10: branch_mask, 2: clone.raster}
    result = optimize_with_intent(objects, masks, [symmetry_pass])

    if reject_outer:
        assert not any(batch.operations for batch in result.batches)
        assert [outcome.status for outcome in result.outcomes].count("rolled_back") == 2
        assert any(
            outcome.status == "rejected"
            and isinstance(outcome.intent.action, RefreshDependentClonesOperation)
            for outcome in result.outcomes
        )
        assert result.objects[0].children[0] is source
        assert result.objects[0].children[1] is native
        assert result.objects[1].current == clone.current
        assert result.objects[1].footprint == clone.footprint
    else:
        actions = [
            operation.intent.action
            for batch in result.batches
            for operation in batch.operations
        ]
        assert any(isinstance(action, ReconstructSelfSymmetryOperation) for action in actions)
        assert any(isinstance(action, DeclareIntrinsicSymmetryOperation) for action in actions)
        assert any(isinstance(action, RefreshDependentClonesOperation) for action in actions)
        assert result.objects[1].current.params["href_region_path"] == (10, 11)

    monkeypatch.setattr(
        "vectormark.optimizer.passes.symmetry._best_self_reconstruction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("recursive selection ran")),
    )
    monkeypatch.setattr(
        "vectormark.optimizer.passes.symmetry._best_self_axis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("recursive axis ran")),
    )
    replayed = replay_batches(objects, result.batches, masks=masks)
    _assert_semantic_regions_equal(replayed, result.objects)

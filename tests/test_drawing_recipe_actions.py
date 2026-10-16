from __future__ import annotations

import base64
from dataclasses import replace
import io
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest

import vectormark.mcp_server as mcp_server_module
import numpy as np

from vectormark.candidate import FlatFill
from vectormark.drawing_actions import ActionSettings, DrawingPlan
from vectormark.drawing_adapter import drawing_from_vector_regions
from vectormark.drawing_engine import RasterImage
from vectormark.drawing_recipe import (
    EffectiveDrawingRecipe,
    RecipeOperation,
    build_geometry_evidence,
    build_paint_evidence,
)
from vectormark.drawing_replay import (
    _path_replay_records,
    _source_command_for_display,
    extend_recipe_with_plan,
    raster_pixels,
    propose_effective_recipe,
)
from vectormark.drawing_refine import refine, refill_changed_regions_from_source
from vectormark.drawing_render import drawing_summary
from vectormark.drawing_state import DrawingStore
from vectormark.drawing_trace import RootPathEvidence, TraceOptions, TraceResult
from vectormark.fit import Shape
from vectormark.mcp_server import (
    ImageRef,
    TraceDrawingOptions,
    get_drawing_artifact,
    refine_drawing,
    trace_drawing,
)
from vectormark.optimizer.automatic_replay import (
    ApplySeamConstraintsOperation,
    AxisSeamConstraint,
    PathOpsUnderlapConstraint,
    ReplayBatch,
)
from vectormark.optimizer.corners import path_corner_diagnostics
from vectormark.optimizer.framework import optimize_with_intent
from vectormark.optimizer.gate import rasterize
from vectormark.optimizer.passes.compound import split_compound_pass
from vectormark.optimizer.passes.seams import seams_pass
from vectormark.optimizer.vector_region import VectorRegion, to_polygon
from vectormark.path_fit_evidence import automatic_fit_evidence
from vectormark.path_geometry import CloneGeometry
from vectormark.skia_geometry import SkPath


def _data_uri(image: Image.Image) -> str:
    data = io.BytesIO()
    image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def _single_region_uri() -> str:
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((12, 12, 52, 52), fill=(20, 100, 240))
    return _data_uri(image)


def _disk_region_uri() -> str:
    image = Image.new("RGB", (120, 120), "white")
    ImageDraw.Draw(image).ellipse((20, 20, 100, 100), fill=(200, 30, 30))
    return _data_uri(image)


def _two_regions_uri() -> str:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 16, 24, 48), fill=(240, 40, 20))
    draw.rectangle((40, 16, 60, 48), fill=(20, 100, 240))
    return _data_uri(image)


def _three_regions_uri() -> str:
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 16, 24, 48), fill=(240, 40, 20))
    draw.rectangle((36, 16, 56, 48), fill=(40, 220, 80))
    draw.rectangle((68, 16, 88, 48), fill=(20, 100, 240))
    return _data_uri(image)


@pytest.fixture(autouse=True)
def _fresh_store(monkeypatch):
    store = DrawingStore()
    monkeypatch.setattr(mcp_server_module, "_DRAWINGS", store)
    return store


def _trace(uri: str):
    ctx = SimpleNamespace(session=object())
    result = trace_drawing(
        ImageRef(data_uri=uri), TraceDrawingOptions(refine="none"), ctx
    ).structured_content
    assert result is not None
    return ctx, result


def _refine(ctx, traced, ops):
    result = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": "v0",
            "ops": ops,
        },
        ctx,
    ).structured_content
    assert result is not None
    return result


def _assert_stored_replay(traced, version_id):
    version, paint, evidence = mcp_server_module._DRAWINGS.load_replay(
        traced["drawing_id"], version_id
    )
    replayed = propose_effective_recipe(
        drawing_id=traced["drawing_id"],
        evidence=evidence,
        paint=paint,
        recipe=version.effective_recipe,
    )
    assert replayed.drawing == version.drawing
    assert replayed.replay.output_revision == version.replay.output_revision
    return version


def _direct_recipe_fixture(regions, *, width: int, height: int):
    pixels = np.full((height, width, 3), 255, np.uint8)
    source = RasterImage(width, height, "rgb8", pixels.tobytes())
    evidence = build_geometry_evidence(
        kind="source",
        trace=TraceResult(width, height, TraceOptions(), (), "<svg/>"),
        geometry_source=source,
        trace_roots=tuple(regions),
    )
    paint = build_paint_evidence(source)
    recipe = EffectiveDrawingRecipe(
        evidence.revision,
        evidence.engine_revision,
        ActionSettings(epsilon=1.0, max_error=1.0, fit_strategy="quadratic"),
    )
    return evidence, paint, recipe


def _retained_path_region(
    obj_id: int,
    region_id: str,
    points: list[tuple[float, float]],
    d: str,
    *,
    z: float = 0,
):
    fit = automatic_fit_evidence(
        points,
        (),
        (),
        tuple((index, index + 1, 1) for index in range(len(points) - 1)),
        d,
        settings=(1.0, 1.0),
    )
    shape = Shape("path", {"d": d})
    region = VectorRegion(
        obj_id,
        shape,
        FlatFill("#123456"),
        z,
        footprint=to_polygon(shape),
        drawing_id=region_id,
        source_regions=(region_id,),
    )
    return region, RootPathEvidence(region_id, f"{region_id}.p0", fit)


def _seam_recipe_execution(regions, root_paths, masks):
    retained_regions = tuple(
        replace(region, raster=np.asarray(masks[region.id], dtype=bool))
        for region in regions
    )
    optimized = optimize_with_intent(list(retained_regions), masks, [seams_pass])
    pixels = np.full((110, 120, 3), 255, np.uint8)
    source = RasterImage(120, 110, "rgb8", pixels.tobytes())
    trace = TraceResult(
        120,
        110,
        TraceOptions(),
        (),
        "<svg/>",
        root_path_evidence=tuple(root_paths),
    )
    evidence = build_geometry_evidence(
        kind="source",
        trace=trace,
        geometry_source=source,
        trace_roots=retained_regions,
    )
    paint = build_paint_evidence(source)
    recipe = EffectiveDrawingRecipe(
        evidence.revision,
        evidence.engine_revision,
        ActionSettings(epsilon=1.0, max_error=1.0, fit_strategy="quadratic"),
        steps=optimized.batches,
    )
    return optimized, propose_effective_recipe(
        drawing_id="drw_seam_receipt",
        evidence=evidence,
        paint=paint,
        recipe=recipe,
    )


def _direct_plan_drawing(drawing_id, evidence, paint, plan):
    regions = refine(
        evidence.trace,
        evidence.trace_roots,
        plan,
        rgb=raster_pixels(paint.source),
    )
    return drawing_from_vector_regions(
        drawing_id,
        width=paint.canvas.width,
        height=paint.canvas.height,
        regions=regions,
    )


def _sequential_update_drawing(drawing_id, evidence, paint, plan):
    regions = evidence.trace_roots
    action = plan.ops[0]
    for nested in action.ops:
        stage = DrawingPlan(
            version="vectormark.plan.v1",
            drawing_id=drawing_id,
            base_version=plan.base_version,
            settings=plan.settings,
            ops=(action.model_copy(update={"ops": (nested,)}),),
        )
        regions = refine(evidence.trace, regions, stage, rgb=None)
    regions = refill_changed_regions_from_source(
        evidence.trace_roots, regions, raster_pixels(paint.source)
    )
    return drawing_from_vector_regions(
        drawing_id,
        width=paint.canvas.width,
        height=paint.canvas.height,
        regions=regions,
    )


@pytest.mark.parametrize(
    "action, expected_type",
    [
        ({"op": "fit_path", "target": "r1"}, RecipeOperation),
        (
            {"op": "fit_primitive", "target": "r1", "type": "rect"},
            RecipeOperation,
        ),
        (
            {
                "op": "set_fill",
                "target": "r1",
                "fill": {"type": "flat", "color": "#EA4C89"},
            },
            RecipeOperation,
        ),
        (
            {
                "op": "set_symmetry",
                "target": "r1",
                "axis": {"point": [32, 32], "angle_degrees": 90},
            },
            RecipeOperation,
        ),
        (
            {
                "op": "update_geometry",
                "target": "r1",
                "ops": [{"op": "remove", "target": "r1.p0.c2"}],
            },
            RecipeOperation,
        ),
        ({"op": "detect_primitives", "target": "r1"}, ReplayBatch),
        ({"op": "detect_symmetry", "target": "r1"}, ReplayBatch),
    ],
)
def test_single_region_public_action_replays_from_retained_evidence(
    action, expected_type, monkeypatch
):
    uri = _disk_region_uri() if action["op"] == "detect_primitives" else _single_region_uri()
    ctx, traced = _trace(uri)
    refined = _refine(ctx, traced, [action])
    version = mcp_server_module._DRAWINGS.get_version(
        traced["drawing_id"], refined["version"]
    )
    assert any(
        isinstance(step, expected_type)
        for step in version.effective_recipe.steps
    )
    if action["op"] == "detect_primitives":
        assert any(batch.operations for batch in version.effective_recipe.automatic_batches)
        monkeypatch.setattr(
            "vectormark.optimizer.passes.primitives.recognize_primitive",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("primitive detection ran during recipe replay")
            ),
        )
    if action["op"] == "detect_symmetry":
        assert any(batch.operations for batch in version.effective_recipe.automatic_batches)
        monkeypatch.setattr(
            "vectormark.optimizer.passes.symmetry._best_self_reconstruction",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("symmetry selection ran during recipe replay")
            ),
        )
    _assert_stored_replay(traced, refined["version"])


def test_merge_named_outputs_are_bound_for_follow_up_fill_and_order():
    ctx, traced = _trace(_two_regions_uri())
    refined = _refine(
        ctx,
        traced,
        [
            {"op": "merge", "id": "joined", "regions": ["r1", "r2"]},
            {
                "op": "set_fill",
                "target": "joined-1",
                "fill": {"type": "flat", "color": "#123456"},
            },
            {"op": "set_z_order", "targets": ["joined-2", "joined-1"]},
        ],
    )
    version = _assert_stored_replay(traced, refined["version"])
    operations = version.effective_recipe.operations
    assert [operation.action.op for operation in operations] == [
        "merge",
        "set_fill",
        "set_z_order",
    ]
    assert type(operations[1].bindings[0]).__name__ == "NamedOutputBinding"
    assert all(
        type(binding).__name__ == "NamedOutputBinding"
        for binding in operations[2].bindings
    )


def test_split_named_output_is_bound_for_follow_up_fill():
    ctx, traced = _trace(_single_region_uri())
    refined = _refine(
        ctx,
        traced,
        [
            {
                "op": "split",
                "target": "r1",
                "divider": {"type": "line", "points": [[32, 0], [32, 64]]},
            },
            {
                "op": "set_fill",
                "target": "r1-1",
                "fill": {"type": "flat", "color": "#123456"},
            },
        ],
    )
    version = _assert_stored_replay(traced, refined["version"])
    operations = version.effective_recipe.operations
    assert [operation.action.op for operation in operations] == ["split", "set_fill"]
    assert type(operations[1].bindings[0]).__name__ == "NamedOutputBinding"


def test_clone_align_and_z_order_replay_with_independent_target_paint():
    ctx, traced = _trace(_two_regions_uri())
    refined = _refine(
        ctx,
        traced,
        [
            {"op": "align", "target": "r2", "reference": "r1", "axes": ["y"]},
            {
                "op": "assign_clone",
                "source": "r1",
                "target": "r2",
                "transforms": [{"tx": 36}],
            },
            {"op": "set_z_order", "targets": ["r2", "r1"]},
        ],
    )
    version = _assert_stored_replay(traced, refined["version"])
    assert [operation.action.op for operation in version.effective_recipe.operations] == [
        "align",
        "assign_clone",
        "set_z_order",
    ]
    source, target = version.drawing.regions
    assert source.fill != target.fill


def test_public_clone_chain_flattens_to_concrete_source_and_renders_all_artifacts():
    ctx, traced = _trace(_three_regions_uri())
    first = _refine(
        ctx,
        traced,
        [{
            "op": "assign_clone",
            "source": "r1",
            "target": "r2",
            "transforms": [{"tx": 32}],
        }],
    )
    second = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": first["version"],
            "ops": [{
                "op": "assign_clone",
                "source": "r2",
                "target": "r3",
                "transforms": [{"tx": 32}],
            }],
        },
        ctx,
    ).structured_content
    assert second is not None
    version = _assert_stored_replay(traced, second["version"])
    by_id = {region.id: region for region in version.drawing.regions}
    assert isinstance(by_id["r2"].geometry, CloneGeometry)
    assert isinstance(by_id["r3"].geometry, CloneGeometry)
    assert by_id["r2"].geometry.source_region_id == "r1"
    assert by_id["r3"].geometry.source_region_id == "r1"
    assert by_id["r2"].geometry.transforms[0].tx == pytest.approx(32)
    assert by_id["r3"].geometry.transforms[0].tx == pytest.approx(64)
    assert by_id["r3"].fill != by_id["r1"].fill
    assert by_id["r3"].fill != by_id["r2"].fill

    for artifact in (
        "svg",
        "preview",
        "review_panel",
        "labeled_svg",
        "raw_trace",
        "plan",
        "versions",
    ):
        result = get_drawing_artifact(
            traced["drawing_id"], second["version"], artifact, ctx
        )
        assert not result.is_error


def test_public_clone_cycle_is_rejected_without_publishing_a_child():
    ctx, traced = _trace(_three_regions_uri())
    first = _refine(
        ctx,
        traced,
        [{
            "op": "assign_clone",
            "source": "r1",
            "target": "r2",
            "transforms": [{"tx": 32}],
        }],
    )
    before = mcp_server_module._DRAWINGS.get_manifest(traced["drawing_id"])

    with pytest.raises(Exception, match="cyclic retained clone reference"):
        refine_drawing(
            {
                "version": "vectormark.plan.v1",
                "drawing_id": traced["drawing_id"],
                "base_version": first["version"],
                "ops": [{
                    "op": "assign_clone",
                    "source": "r2",
                    "target": "r1",
                    "transforms": [{"tx": -32}],
                }],
            },
            ctx,
        )

    assert mcp_server_module._DRAWINGS.get_manifest(traced["drawing_id"]) == before


def test_detect_clones_replays_accepted_concrete_action_without_detection(monkeypatch):
    ctx, traced = _trace(_two_regions_uri())
    refined = _refine(ctx, traced, [{"op": "detect_clones"}])
    version = mcp_server_module._DRAWINGS.get_version(
        traced["drawing_id"], refined["version"]
    )
    assert any(batch.operations for batch in version.effective_recipe.automatic_batches)
    monkeypatch.setattr(
        "vectormark.optimizer.passes.clones.clones_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("clone detection ran during recipe replay")
        ),
    )
    _assert_stored_replay(traced, refined["version"])


def test_stitch_replays_accepted_constraints_without_pair_discovery(monkeypatch):
    def region(region_id: int, d: str, z: float) -> VectorRegion:
        shape = Shape("path", {"d": d})
        return VectorRegion(
            region_id,
            shape,
            FlatFill(f"#{region_id:06x}"),
            z,
            raster=rasterize(to_polygon(shape), (80, 120)),
            footprint=to_polygon(shape),
            drawing_id=f"r{region_id}",
            source_regions=(f"r{region_id}",),
        )

    regions = (
        region(1, "M10 10 L49.2 10 L49.2 70 L10 70 Z", 0),
        region(2, "M50.4 10 L90 10 L90 70 L50.4 70 Z", 1),
    )
    evidence, paint, base = _direct_recipe_fixture(regions, width=120, height=80)
    plan = DrawingPlan.model_validate({
        "version": "vectormark.plan.v1",
        "drawing_id": "drw_stitch",
        "base_version": "v0",
        "ops": [{"op": "stitch", "targets": ["r1", "r2"]}],
    })
    recipe = extend_recipe_with_plan(
        evidence=evidence, paint=paint, base_recipe=base, plan=plan
    ).recipe
    assert len(recipe.automatic_batches) == 1
    assert recipe.automatic_batches[0].operations
    expected = propose_effective_recipe(
        drawing_id="drw_stitch", evidence=evidence, paint=paint, recipe=recipe
    ).drawing
    monkeypatch.setattr(
        "vectormark.optimizer.passes.seams._candidate_for_pair",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("seam pair discovery ran during recipe replay")
        ),
    )
    replayed = propose_effective_recipe(
        drawing_id="drw_stitch", evidence=evidence, paint=paint, recipe=recipe
    ).drawing
    assert replayed == expected


def test_pathops_seam_report_keeps_only_exact_commands_and_unaffected_sibling():
    under, under_path = _retained_path_region(
        21,
        "r20-21",
        [(0, 0), (100, 0), (0, 100)],
        "M0 0 L100 0 L0 100 Z",
    )
    top, top_path = _retained_path_region(
        22,
        "r20-22",
        [(100, 1), (100, 100), (1, 100)],
        "M100 1 L100 100 L1 100 Z",
        z=1,
    )
    branch = VectorRegion.branch(
        id=20,
        children=[under, top],
        z=0,
        footprint=SkPath(shell=[(0, 0), (100, 0), (100, 100), (0, 100)]),
        drawing_id="r20",
        source_regions=("r20",),
    )
    mask = rasterize(branch.footprint, (110, 120))
    branch = replace(branch, raster=mask)
    optimized, execution = _seam_recipe_execution(
        (branch,), (under_path, top_path), {20: mask}
    )
    seam = next(
        operation.intent.action
        for batch in optimized.batches
        for operation in batch.operations
        if isinstance(operation.intent.action, ApplySeamConstraintsOperation)
    )
    assert isinstance(seam.pairs[0], PathOpsUnderlapConstraint)

    bottom = next(
        record
        for record in execution.replay.path_correspondence
        if record.output_region_id == "r20-21"
    )
    exact_pairs = {
        (source_id, output_id)
        for span in bottom.correspondence.spans
        for source_id, output_id in zip(
            span.source_command_ids, span.output_command_ids, strict=True
        )
    }
    source_c2 = under_path.fit.source.path.commands[2].id
    assert (source_c2, "r20-21.p0.c3") in exact_pairs
    assert (source_c2, "r20-21.p0.c2") not in exact_pairs

    targets = {
        target["id"]: target
        for target in drawing_summary(
            execution.drawing, execution.replay.path_correspondence
        )["targets"]
    }
    bottom_commands = {
        command["id"]: command["corner_boundary"]["editable"]
        for command in targets["r20-21"]["geometry"]["commands"]
    }
    assert bottom_commands["r20-21.p0.c2"] is False
    assert bottom_commands["r20-21.p0.c3"] is True
    top_commands = {
        command["id"]: command["corner_boundary"]["editable"]
        for command in targets["r20-22"]["geometry"]["commands"]
    }
    assert top_commands["r20-22.p0.c1"] is True
    assert top_commands["r20-22.p0.c2"] is True


def test_axis_seam_report_keeps_exactly_mapped_command_boundaries_editable():
    left, left_path = _retained_path_region(
        1,
        "r1",
        [(10, 10), (49.2, 10), (49.2, 70), (10, 70)],
        "M10 10 L49.2 10 L49.2 70 L10 70 Z",
    )
    right, right_path = _retained_path_region(
        2,
        "r2",
        [(50.4, 10), (90, 10), (90, 70), (50.4, 70)],
        "M50.4 10 L90 10 L90 70 L50.4 70 Z",
        z=1,
    )
    masks = {
        region.id: rasterize(region.footprint, (110, 120))
        for region in (left, right)
    }
    optimized, execution = _seam_recipe_execution(
        (left, right), (left_path, right_path), masks
    )
    seam = next(
        operation.intent.action
        for batch in optimized.batches
        for operation in batch.operations
        if isinstance(operation.intent.action, ApplySeamConstraintsOperation)
    )
    assert isinstance(seam.pairs[0], AxisSeamConstraint)
    targets = {
        target["id"]: target
        for target in drawing_summary(
            execution.drawing, execution.replay.path_correspondence
        )["targets"]
    }
    for region_id in ("r1", "r2"):
        commands = targets[region_id]["geometry"]["commands"]
        assert all(command["corner_boundary"]["editable"] for command in commands[1:])


def test_seam_cubic_cleanup_hides_replaced_command_but_keeps_exact_following_line():
    left, left_path = _retained_path_region(
        1,
        "r1",
        [(10, 10), (49.2, 70), (10, 70)],
        "M10 10 C48.9 18 40 35 49.2 70 L10 70 Z",
    )
    right, right_path = _retained_path_region(
        2,
        "r2",
        [(50.4, 10), (90, 10), (90, 70), (50.4, 70)],
        "M50.4 10 L90 10 L90 70 L50.4 70 Z",
        z=1,
    )
    left_trace = Shape(
        "path", {"d": "M10 10 C48.9 18 40 35 49.8 70 L10 70 Z"}
    )
    right_trace = Shape(
        "path", {"d": "M49.8 10 L90 10 L90 70 L49.8 70 Z"}
    )
    masks = {
        1: rasterize(to_polygon(left_trace), (110, 120)),
        2: rasterize(to_polygon(right_trace), (110, 120)),
    }
    optimized, execution = _seam_recipe_execution(
        (left, right), (left_path, right_path), masks
    )
    seam = next(
        operation.intent.action
        for batch in optimized.batches
        for operation in batch.operations
        if isinstance(operation.intent.action, ApplySeamConstraintsOperation)
    )
    assert isinstance(seam.pairs[0], AxisSeamConstraint)
    left_record = next(
        record
        for record in execution.replay.path_correspondence
        if record.output_region_id == "r1"
    )
    exact_pairs = {
        (source_id, output_id)
        for span in left_record.correspondence.spans
        for source_id, output_id in zip(
            span.source_command_ids, span.output_command_ids, strict=True
        )
    }
    source_cubic = left_path.fit.source.path.commands[1].id
    source_following_line = left_path.fit.source.path.commands[2].id
    assert not any(source_id == source_cubic for source_id, _ in exact_pairs)
    assert (source_following_line, "r1.p0.c3") in exact_pairs

    left_summary = next(
        target
        for target in drawing_summary(
            execution.drawing, execution.replay.path_correspondence
        )["targets"]
        if target["id"] == "r1"
    )
    editable = {
        command["id"]: command["corner_boundary"]["editable"]
        for command in left_summary["geometry"]["commands"]
    }
    assert editable["r1.p0.c1"] is False
    assert editable["r1.p0.c3"] is True


def test_topology_receipt_keeps_surviving_paths_editable_after_owner_remap():
    first_d = "M0 0 L8 0 L8 8 L0 8 Z"
    second_d = "M12 0 L20 0 L20 8 L12 8 Z"
    _, first_path = _retained_path_region(
        1,
        "r1",
        [(0, 0), (8, 0), (8, 8), (0, 8)],
        first_d,
    )
    _, second_path = _retained_path_region(
        1,
        "r1",
        [(12, 0), (20, 0), (20, 8), (12, 8)],
        second_d,
    )
    second_path = replace(second_path, path_id="r1.p1")
    shape = Shape(
        "path",
        {
            "d": f"{first_d} {second_d}",
            "fill_rule": "evenodd",
            "path_ids": ("r1.p0", "r1.p1"),
        },
    )
    source_region = VectorRegion.from_shape(
        id=1,
        shape=shape,
        fill=FlatFill("#123456"),
        z=0,
        raster=np.zeros((24, 24), bool),
        drawing_id="r1",
        source_regions=("r1",),
    )
    optimized = optimize_with_intent(
        [source_region], {1: source_region.raster}, [split_compound_pass]
    )
    pixels = np.full((24, 24, 3), 255, np.uint8)
    raster = RasterImage(24, 24, "rgb8", pixels.tobytes())
    evidence = build_geometry_evidence(
        kind="source",
        trace=TraceResult(
            24,
            24,
            TraceOptions(),
            (),
            "<svg/>",
            root_path_evidence=(first_path, second_path),
        ),
        geometry_source=raster,
        trace_roots=(source_region,),
    )
    paint = build_paint_evidence(raster)
    recipe = EffectiveDrawingRecipe(
        evidence.revision,
        evidence.engine_revision,
        ActionSettings(epsilon=1.0, max_error=1.0, fit_strategy="quadratic"),
        steps=optimized.batches,
    )
    regions = tuple(optimized.objects)
    drawing = drawing_from_vector_regions(
        "drw_topology_receipt",
        width=24,
        height=24,
        regions=regions,
    )
    path_correspondence = _path_replay_records(
        evidence, recipe, regions, drawing, []
    )
    records = {
        record.output_path_id: record
        for record in path_correspondence
    }
    assert records["r1.p0"].output_region_id == "r1-1"
    assert records["r1.p0"].output_region_ancestry == (1, 1)
    assert records["r1.p1"].output_region_id == "r1-2"
    assert records["r1.p1"].output_region_ancestry == (1, 2)
    targets = drawing_summary(
        drawing, path_correspondence
    )["targets"]
    assert all(
        target["geometry"]["corner_edit"]["supported"]
        for target in targets
    )
    assert all(
        command["corner_boundary"]["editable"]
        for target in targets
        for command in target["geometry"]["commands"][1:]
    )
    assert _source_command_for_display(
        evidence,
        regions,
        "r1-1",
        "r1.p0",
        "r1.p0.c1",
        [],
        recipe,
    ) == first_path.fit.source.path.commands[1].id


def test_update_geometry_preserves_cleanup_positions_repetitions_and_settings():
    d = (
        "M247 417.5 Q241.23 417.72 237.5 413 "
        "L237.5 405 L242 400.5 Q256.31 399.74 253.5 413 "
        "L247 417.5 Z"
    )
    shape = Shape("path", {"d": d})
    source = VectorRegion(
        1,
        shape,
        FlatFill("#123456"),
        0,
        raster=rasterize(to_polygon(shape), (460, 300)),
        footprint=to_polygon(shape),
        drawing_id="r1",
        source_regions=("r1",),
    )
    evidence, paint, base = _direct_recipe_fixture((source,), width=300, height=460)
    plan = DrawingPlan.model_validate({
        "version": "vectormark.plan.v1",
        "drawing_id": "drw_order",
        "base_version": "v0",
        "ops": [{
            "op": "update_geometry",
            "target": "r1",
            "settings": {"epsilon": 1.0, "max_error": 1.0},
            "ops": [
                {"op": "simplify"},
                {"op": "fit", "target": "r1.p0.c1", "type": "keep"},
                {"op": "simplify"},
                {"op": "fit", "target": "r1.p0.c1", "type": "keep"},
            ],
        }],
    })
    recipe = extend_recipe_with_plan(
        evidence=evidence, paint=paint, base_recipe=base, plan=plan
    ).recipe
    semantic_steps = tuple(
        step
        for step in recipe.steps
        if isinstance(step, (RecipeOperation, ReplayBatch))
    )
    assert tuple(type(step) for step in semantic_steps) == (
        ReplayBatch,
        RecipeOperation,
        ReplayBatch,
        RecipeOperation,
    )
    assert [step.action.ops[0].op for step in semantic_steps[1::2]] == ["fit", "fit"]
    assert all(
        step.settings.epsilon == 1.0 and step.settings.max_error == 1.0
        for step in semantic_steps[1::2]
    )
    assert semantic_steps[0].operations
    replayed = propose_effective_recipe(
        drawing_id="drw_order", evidence=evidence, paint=paint, recipe=recipe
    ).drawing
    assert replayed == _sequential_update_drawing(
        "drw_order", evidence, paint, plan
    )


@pytest.mark.parametrize(
    ("path_actions", "expected_roles"),
    [
        (
            [{"op": "fit", "target": "r1.p0.c2", "type": "keep"}],
            {"ops.0.target"},
        ),
        (
            [{"op": "fit", "target": "r1.p0.c2", "type": "quadratic"}],
            {"ops.0.target"},
        ),
        (
            [{
                "op": "fit",
                "between": ["r1.p0.c1", "r1.p0.c3"],
                "strategy": "quadratic",
            }],
            {"ops.0.between.0", "ops.0.between.1"},
        ),
        ([{"op": "remove", "target": "r1.p0.c2"}], {"ops.0.target"}),
        (
            [{
                "op": "match",
                "target": "r1.p0.c2",
                "reference": "r1.p0.c1",
                "transform": {"tx": 6},
            }],
            {"ops.0.target", "ops.0.reference"},
        ),
        (
            [{
                "op": "set_parallel",
                "target": "r1.p0.c2",
                "reference": "r1.p0.c1",
            }],
            {"ops.0.target", "ops.0.reference"},
        ),
        (
            [{
                "op": "match_length",
                "target": "r1.p0.c2",
                "reference": "r1.p0.c1",
            }],
            {"ops.0.target", "ops.0.reference"},
        ),
        (
            [{
                "op": "align",
                "target": "r1.p0.c2",
                "reference": "r1.p0.c1",
                "axes": ["y"],
            }],
            {"ops.0.target", "ops.0.reference"},
        ),
        (
            [{
                "op": "snap",
                "target": "r1.p0.c2",
                "reference": "r1.p0.c1",
            }],
            {"ops.0.target", "ops.0.reference"},
        ),
        (
            [
                {"op": "fit", "target": "r1.p0.c2", "type": "keep"},
                {"op": "break", "target": "r1.p0.c2"},
            ],
            {"ops.0.target", "ops.1.target"},
        ),
        ([{"op": "reverse", "target": "r1.p0.c2"}], {"ops.0.target"}),
        (
            [{
                "op": "mirror",
                "target": "r1.p0.c2",
                "reference": "r1.p0.c1",
                "axis": {"point": [3, 1.5], "angle_degrees": 90},
            }],
            {"ops.0.target", "ops.0.reference"},
        ),
    ],
)
def test_nested_path_action_replay_binds_every_command_reference(
    path_actions, expected_roles
):
    shape = Shape("path", {"d": "M0 0 L6 0 L6 3 L0 3 Z"})
    source = VectorRegion(
        1,
        shape,
        FlatFill("#123456"),
        0,
        raster=rasterize(to_polygon(shape), (12, 12)),
        footprint=to_polygon(shape),
        drawing_id="r1",
        source_regions=("r1",),
    )
    evidence, paint, base = _direct_recipe_fixture((source,), width=12, height=12)
    plan = DrawingPlan.model_validate({
        "version": "vectormark.plan.v1",
        "drawing_id": "drw_nested",
        "base_version": "v0",
        "ops": [{
            "op": "update_geometry",
            "target": "r1",
            "ops": path_actions,
        }],
    })
    if any(item["op"] == "fit" and item.get("type") != "keep" for item in path_actions):
        # This hand-authored fixture has no original material-polyline evidence.
        # Retained fits must not silently validate against their current curves.
        with pytest.raises(ValueError, match="source correspondence is absent"):
            extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan)
        return
    recipe = extend_recipe_with_plan(
        evidence=evidence, paint=paint, base_recipe=base, plan=plan
    ).recipe
    operation = recipe.operations[0]
    assert {binding.role for binding in operation.bindings} == {
        "target",
        *expected_roles,
    }
    replayed = propose_effective_recipe(
        drawing_id="drw_nested", evidence=evidence, paint=paint, recipe=recipe
    ).drawing
    assert replayed == _direct_plan_drawing(
        "drw_nested", evidence, paint, plan
    )
    first = path_actions[0]
    if first["op"] in {"remove", "match", "match_length", "align", "snap"} or (
        first["op"] == "fit" and first.get("type") not in {None, "keep"}
    ):
        baseline = propose_effective_recipe(
            drawing_id="drw_nested", evidence=evidence, paint=paint, recipe=base
        ).drawing
        assert replayed != baseline


def test_public_corner_normalization_replays_retained_declaration_without_detection(
    monkeypatch,
):
    d = "M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z"
    shape = Shape("path", {"d": d})
    declaration = path_corner_diagnostics(
        d, declared_corner_spans=(((10, 0), (14, 4)),)
    )
    source = VectorRegion(
        1,
        shape,
        FlatFill("#123456"),
        0,
        original=shape,
        raster=rasterize(to_polygon(shape), (16, 24)),
        footprint=to_polygon(shape),
        drawing_id="r1",
        source_regions=("r1",),
        diagnostics={"corners": declaration},
    )
    evidence, paint, base = _direct_recipe_fixture((source,), width=24, height=16)
    plan = DrawingPlan.model_validate({
        "version": "vectormark.plan.v1",
        "drawing_id": "drw_normalize",
        "base_version": "v0",
        "ops": [{
            "op": "update_geometry",
            "target": "r1",
            "ops": [{"op": "normalize_corners"}],
        }],
    })
    recipe = extend_recipe_with_plan(
        evidence=evidence, paint=paint, base_recipe=base, plan=plan
    ).recipe
    assert len(recipe.automatic_batches) == 1
    assert recipe.automatic_batches[0].operations
    monkeypatch.setattr(
        "vectormark.optimizer.corners.path_corner_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("corner detection ran during recipe replay")
        ),
    )
    propose_effective_recipe(
        drawing_id="drw_normalize", evidence=evidence, paint=paint, recipe=recipe
    )

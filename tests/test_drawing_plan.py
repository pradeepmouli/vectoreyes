import base64

import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.drawing_adapter import drawing_from_region_samples
from vectormark.drawing_trace import (
    TraceCommand,
    TraceOptions,
    TracePath,
    TraceRegion,
    TraceResult,
)
from vectormark.fit import Shape
from vectormark.refinement.region_sample import RegionSample


def _trace() -> TraceResult:
    path = TracePath(
        d="M 0 0 Q 1 0 2 0 Q 3 0 4 0 Q 5 0 6 0 Z",
        fill_rule="nonzero",
        commands=(
            TraceCommand("r1.p0.c0", "M", (0.0, 0.0)),
            TraceCommand("r1.p0.c1", "Q", (1.0, 0.0, 2.0, 0.0)),
            TraceCommand("r1.p0.c2", "Q", (3.0, 0.0, 4.0, 0.0)),
            TraceCommand("r1.p0.c3", "Q", (5.0, 0.0, 6.0, 0.0)),
            TraceCommand("r1.p0.c4", "Z", ()),
        ),
    )
    region = TraceRegion(
        id="r1",
        source_label=1,
        color="#112233",
        mask=np.ones((1, 1), dtype=bool),
        contours=(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),),
        trace_path=path,
        effective_trace_level="pixel",
    )
    return TraceResult(1, 1, TraceOptions(), (region,), "<svg/>")


def _two_region_trace() -> TraceResult:
    r1 = _trace().regions[0]
    r2 = TraceRegion(
        id="r2",
        source_label=2,
        color="#445566",
        mask=np.ones((1, 1), dtype=bool),
        contours=(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),),
        trace_path=TracePath(
            d=r1.trace_path.d,
            fill_rule=r1.trace_path.fill_rule,
            commands=tuple(
                TraceCommand(command.id.replace("r1.", "r2."), command.command, command.values)
                for command in r1.trace_path.commands
            ),
        ),
        effective_trace_level="pixel",
    )
    return TraceResult(1, 1, TraceOptions(), (r1, r2), "<svg/>")


def _scene(*target_ids: str) -> tuple[RegionSample, ...]:
    return tuple(
        RegionSample.from_shape(
            id=index + 1,
            shape=Shape("path", {"d": "M 0 0 Q 1 0 2 0 Q 3 0 4 0 Q 5 0 6 0 Z"}),
            fill=FlatFill("#112233"),
            z=index,
            raster=np.ones((1, 1), dtype=bool),
            drawing_id=target_id,
            source_regions=(target_id,),
        )
        for index, target_id in enumerate(target_ids)
    )


def _grouped_scene(*source_regions: str) -> tuple[RegionSample, ...]:
    return (
        RegionSample.from_shape(
            id=1,
            shape=Shape("path", {"d": "M 0 0 Q 1 0 2 0 Q 3 0 4 0 Q 5 0 6 0 Z"}),
            fill=FlatFill("#112233"),
            z=0,
            raster=np.ones((1, 1), dtype=bool),
            drawing_id="g1",
            source_regions=source_regions,
        ),
    )


def _validate_plan(plan, trace: TraceResult, regions: tuple[RegionSample, ...]) -> None:
    from vectormark.drawing_plan import validate_plan

    validate_plan(
        plan,
        drawing_from_region_samples(
            "drw_x",
            width=trace.width,
            height=trace.height,
            regions=regions,
        ),
    )


def _path_plan(segment: str = "r1.p0.c1") -> dict[str, object]:
    return {
        "version": "vectormark.plan.v1",
        "drawing_id": "drw_x",
        "base_version": "v0",
        "ops": [
            {
                "op": "update_geometry",
                "target": "r1",
                "ops": [
                        {"op": "fit", "target": segment, "type": "quadratic"},
                        {"op": "break", "target": segment},
                    ],
            }
        ],
    }


def test_plan_accepts_version_scoped_segment_fit_and_break():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(_path_plan())

    _validate_plan(plan, _trace(), _scene("r1"))


def test_plan_accepts_native_detection_and_explicit_relationship_operations():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {"op": "detect_primitives", "target": "r1"},
                {"op": "detect_clones", "target": "r2"},
                {"op": "assign_clone", "source": "r1", "target": "r2", "transforms": [{"tx": 5}]},
                {"op": "set_symmetry", "target": "r2", "axis": {"point": [1, 1], "angle_degrees": 0}},
                {"op": "detect_symmetry"},
            ],
        }
    )

    _validate_plan(plan, _two_region_trace(), _scene("r1", "r2"))


def test_plan_accepts_a_terminal_block_of_distinct_targeted_symmetry_operations():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {"op": "set_fill", "target": "r1", "fill": {"type": "flat", "color": "#112233"}},
                {"op": "detect_symmetry", "target": "r1"},
                {"op": "detect_symmetry", "target": "r2"},
            ],
        }
    )

    _validate_plan(plan, _two_region_trace(), _scene("r1", "r2"))


def test_plan_rejects_repeated_or_non_terminal_targeted_symmetry():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    repeated = parse_plan(
        {
            "version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
            "ops": [{"op": "detect_symmetry", "target": "r1"}, {"op": "detect_symmetry", "target": "r1"}],
        }
    )
    with pytest.raises(PlanValidationError, match="already has a symmetry"):
        _validate_plan(repeated, _trace(), _scene("r1"))

    global_after_target = parse_plan(
        {
            "version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
            "ops": [{"op": "detect_symmetry", "target": "r1"}, {"op": "detect_symmetry"}],
        }
    )
    with pytest.raises(PlanValidationError, match="cannot follow targeted"):
        _validate_plan(global_after_target, _two_region_trace(), _scene("r1", "r2"))


def test_plan_accepts_global_fitting_defaults_and_detect_override():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "settings": {"epsilon": 0.25, "max_error": 0.75},
            "ops": [{"op": "detect_primitives", "target": "r1", "settings": {"epsilon": 0.5}}],
        }
    )

    assert plan.settings.model_dump(exclude_none=True) == {"epsilon": 0.25, "max_error": 0.75}
    _validate_plan(plan, _trace(), _scene("r1"))


def test_plan_accepts_global_and_targeted_stitch_operations():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {"op": "stitch", "settings": {"epsilon": 0.5}},
                {"op": "stitch", "targets": ["r1"], "settings": {"max_error": 0.75}},
            ],
        }
    )

    _validate_plan(plan, _trace(), _scene("r1"))


@pytest.mark.parametrize("settings", [{"epsilon": -0.1}, {"max_error": float("inf")}, {"unknown": 1}])
def test_plan_rejects_invalid_global_fitting_settings(settings):
    from vectormark.drawing_plan import PlanValidationError, parse_plan

    with pytest.raises(PlanValidationError, match="/settings"):
        parse_plan({
            "version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
            "settings": settings, "ops": [],
        })


@pytest.mark.parametrize("operation", ["split", "detect_symmetry"])
def test_structural_operations_require_a_version_boundary(operation: str):
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    structural_op = (
        {"op": "split", "target": "r1", "divider": {"type": "line", "points": [[0, 0], [1, 1]]}}
        if operation == "split"
        else {"op": operation, "target": "r1"}
    )
    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [structural_op, {"op": "set_fill", "target": "r1", "fill": {"type": "flat", "color": "#112233"}}],
        }
    )

    with pytest.raises(PlanValidationError, match="final operation"):
        _validate_plan(plan, _trace(), _scene("r1"))


def test_plan_reports_pointer_for_unknown_retained_segment():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    plan = parse_plan(_path_plan("r1.p0.c99"))

    with pytest.raises(PlanValidationError, match="/ops/0/ops/0/target"):
        _validate_plan(plan, _trace(), _scene("r1"))


def test_path_geometry_rejects_segment_owned_by_another_region():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    plan = parse_plan(_path_plan("r2.p0.c1"))

    with pytest.raises(PlanValidationError, match="/ops/0/ops/0/target"):
        _validate_plan(plan, _two_region_trace(), _scene("r1", "r2"))


def test_path_geometry_uses_retained_target_commands_not_raw_trace_provenance():
    from vectormark.drawing_plan import parse_plan, validate_plan

    retained = RegionSample.from_shape(
        id=1,
        shape=Shape("path", {"d": "M0 0 L8 0 L8 8 L0 8 Z M2 2 L6 2 L6 6 L2 6 Z", "fill_rule": "evenodd"}),
        fill=FlatFill("#112233"),
        z=0,
        raster=np.ones((8, 8), dtype=bool),
        drawing_id="r1",
        source_regions=("r1", "r2"),
    )
    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {
                    "op": "update_geometry",
                    "target": "r1",
                    "ops": [
                            {"op": "fit", "target": "r1.p1.c1", "type": "keep"},
                        ],
                }
            ],
        }
    )

    _validate_plan(plan, _trace(), (retained,))


def test_path_geometry_requires_a_newly_merged_target_to_be_refined_in_its_child_version():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {"op": "merge", "id": "g1", "regions": ["r1", "r2"]},
                {
                    "op": "update_geometry",
                    "target": "g1",
                    "ops": [
                            {"op": "fit", "target": "r1.p0.c1", "type": "line"},
                            {"op": "fit", "target": "r2.p0.c1", "type": "line"},
                        ],
                },
                {"op": "set_z_order", "targets": ["g1"]},
            ],
        }
    )

    with pytest.raises(PlanValidationError, match="retained command"):
        _validate_plan(plan, _two_region_trace(), _scene("r1", "r2"))


def test_path_geometry_uses_its_own_retained_segment_ids():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {
                    "op": "update_geometry",
                    "target": "g1",
                    "ops": [
                            {"op": "fit", "target": "g1.p0.c1", "type": "line"},
                        ],
                },
                {"op": "set_z_order", "targets": ["g1"]},
            ],
        }
    )

    _validate_plan(plan, _two_region_trace(), _grouped_scene("r1", "r2"))


def test_path_geometry_rejects_a_segment_fitted_twice():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan()
    payload["ops"][0]["ops"] = [
        {"op": "fit", "target": "r1.p0.c1", "type": "line"},
        {"op": "fit", "target": "r1.p0.c1", "type": "line"},
    ]

    with pytest.raises(PlanValidationError, match="/ops/0/ops/1/target"):
        _validate_plan(parse_plan(payload), _trace(), _scene("r1"))


@pytest.mark.parametrize(
    ("payload", "pointer"),
    [
        (
            {
                "version": "vectormark.plan.v1",
                "drawing_id": "drw_x",
                "base_version": "v0",
                "ops": [{"op": "fit_primitive", "target": "missing", "type": "circle"}],
            },
            "/ops/0/target",
        ),
        (
            {
                "version": "vectormark.plan.v1",
                "drawing_id": "drw_x",
                "base_version": "v0",
                "ops": [
                    {"op": "merge", "id": "g1", "regions": ["r1"]},
                    {"op": "merge", "id": "g1", "regions": ["r1"]},
                ],
            },
            "/ops/1/id",
        ),
    ],
)
def test_plan_reports_operation_pointer_for_unknown_targets_and_duplicate_ids(payload, pointer):
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    with pytest.raises(PlanValidationError, match=pointer):
        _validate_plan(parse_plan(payload), _trace(), _scene("r1"))


def test_plan_rejects_invalid_fill_and_non_finite_numbers():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    with pytest.raises(PlanValidationError, match="/ops/0/fill"):
        parse_plan(
            {
                "version": "vectormark.plan.v1",
                "drawing_id": "drw_x",
                "base_version": "v0",
                "ops": [{"op": "set_fill", "target": "r1", "fill": {"type": "flat", "color": "blue"}}],
            }
        )

    with pytest.raises(PlanValidationError, match="/ops/0/fill/start/0"):
        parse_plan(
            {
                "version": "vectormark.plan.v1",
                "drawing_id": "drw_x",
                "base_version": "v0",
                "ops": [
                    {
                        "op": "set_fill",
                        "target": "r1",
                        "fill": {
                            "type": "linear_gradient",
                            "start": [float("nan"), 0],
                            "end": [1, 1],
                            "stops": [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#ffffff"}],
                        },
                    }
                ],
            }
        )


def test_plan_requires_each_base_target_once_in_z_order():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    omitted = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [{"op": "set_z_order", "targets": ["r1"]}],
        }
    )
    with pytest.raises(PlanValidationError, match="/ops/0/targets"):
        _validate_plan(omitted, _trace(), _scene("r1", "r2"))

    with pytest.raises(PlanValidationError, match="/ops/0"):
        parse_plan(
            {
                "version": "vectormark.plan.v1",
                "drawing_id": "drw_x",
                "base_version": "v0",
                "ops": [{"op": "set_z_order", "targets": ["r1", "r1"]}],
            }
        )


def test_z_order_uses_the_final_target_created_by_grouping():
    from vectormark.drawing_plan import parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {"op": "merge", "id": "g1", "regions": ["r1"]},
                {"op": "set_z_order", "targets": ["g1"]},
            ],
        }
    )

    _validate_plan(plan, _trace(), _scene("r1"))


def test_z_order_must_be_the_final_operation():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    plan = parse_plan(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": "drw_x",
            "base_version": "v0",
            "ops": [
                {"op": "set_z_order", "targets": ["r1", "r2"]},
                {"op": "merge", "id": "g1", "regions": ["r1", "r2"]},
            ],
        }
    )

    with pytest.raises(PlanValidationError, match="set_z_order must be the final operation"):
        _validate_plan(plan, _two_region_trace(), _scene("r1", "r2"))


def test_parse_plan_recursively_freezes_nested_path_op_sequences():
    from vectormark.drawing_plan import parse_plan, validate_plan

    payload = _path_plan()
    path_ops = tuple(payload["ops"][0]["ops"])
    payload["ops"][0]["ops"] = path_ops

    plan = parse_plan(payload)
    path_ops[0]["target"] = "changed-after-parsing"

    assert plan.ops[0].ops[0].target == "r1.p0.c1"
    _validate_plan(plan, _trace(), _scene("r1"))


@pytest.mark.parametrize(
    ("path_ops", "pointer"),
    [
        (
            [
                {"op": "fit", "target": "r1.p0.c1", "type": "line"},
                {"op": "break", "target": "r1.p0.c99"},
            ],
            "/ops/0/ops/1/target",
        ),
        (
            [
                {"op": "fit", "target": "r1.p0.c1", "type": "line"},
                {"op": "break", "target": "r1.p0.c1"},
                {"op": "break", "target": "r1.p0.c1"},
            ],
            "/ops/0/ops/2/target",
        ),
    ],
)
def test_path_break_must_target_an_unbroken_fitted_segment(path_ops, pointer):
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan()
    payload["ops"][0]["ops"] = path_ops

    with pytest.raises(PlanValidationError, match=pointer):
        _validate_plan(parse_plan(payload), _trace(), _scene("r1"))


@pytest.mark.parametrize(
    "png_b64",
    [
        "this is not base64!",
        base64.b64encode(b"not a PNG").decode("ascii"),
    ],
)
def test_raster_fill_requires_base64_encoded_png_data(png_b64):
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    with pytest.raises(PlanValidationError, match="/ops/0/fill"):
        parse_plan(
            {
                "version": "vectormark.plan.v1",
                "drawing_id": "drw_x",
                "base_version": "v0",
                "ops": [
                    {
                        "op": "set_fill",
                        "target": "r1",
                        "fill": {
                            "type": "raster",
                            "origin": [0, 0],
                            "width": 1,
                            "height": 1,
                            "png_base64": png_b64,
                        },
                    }
                ],
            }
        )


def test_path_geometry_requires_at_least_one_operation():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan()
    payload["ops"][0]["ops"] = []

    with pytest.raises(PlanValidationError, match="/ops/0/ops"):
        _validate_plan(parse_plan(payload), _trace(), _scene("r1"))


def test_plan_requires_path_dependencies():
    from vectormark.drawing_plan import PlanValidationError, parse_plan, validate_plan

    payload = _path_plan()
    path_ops = payload["ops"][0]["ops"]
    path_ops[0]["target"] = "missing"

    with pytest.raises(PlanValidationError, match="/ops/0/ops/0/target"):
        _validate_plan(parse_plan(payload), _trace(), _scene("r1"))

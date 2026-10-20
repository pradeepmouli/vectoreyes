from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json

import numpy as np
import pytest

from vectormark.candidate import FlatFill as LegacyFlatFill
from vectormark.drawing import Canvas, Drawing, DrawingRegion
from vectormark.drawing_actions import ActionSettings, DrawingPlan
from vectormark.drawing_engine import RasterImage
from vectormark.drawing_fill import FlatFill
from tests._working_evidence import capture_working_geometry_evidence
from vectormark.drawing_recipe import (
    EffectiveDrawingRecipe,
    ReplayRecord,
    build_paint_evidence,
    drawing_revision,
)
from vectormark.drawing_state import DrawingNotFound, DrawingStore, VersionCandidate
from vectormark.drawing_trace import TraceOptions, TracePath, TraceRegion, TraceResult
from vectormark.fit import Shape
from vectormark.material_pretrace import GradientSupport, MaterialPretraceResult
from vectormark.optimizer.vector_region import VectorRegion
from vectormark.path_geometry import PrimitiveGeometry, RectanglePrimitive


@dataclass
class FakeClock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


def _source(width: int = 2, height: int = 2) -> RasterImage:
    return RasterImage(width, height, "rgb8", bytes(width * height * 3))


def _regions(width: int = 2, height: int = 2) -> tuple[VectorRegion, ...]:
    return (
        VectorRegion.from_shape(
            id=1,
            shape=Shape("rect", {"x": 0.0, "y": 0.0, "w": width, "h": height}),
            fill=LegacyFlatFill("#112233"),
            z=0,
            raster=np.ones((height, width), dtype=bool),
            drawing_id="r1",
            source_regions=("r1",),
        ),
    )


def _trace(width: int = 2, height: int = 2) -> TraceResult:
    return TraceResult(width, height, TraceOptions(), (), "<svg/>")


def _mutable_trace() -> TraceResult:
    mask = np.array([[True, False], [False, False]])
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return TraceResult(
        width=2,
        height=2,
        options=TraceOptions(),
        regions=(TraceRegion(
            id="r1",
            source_label=1,
            color="#ffffff",
            mask=mask,
            contours=(contour,),
            trace_path=TracePath(d="M0 0Z", fill_rule="nonzero", commands=()),
            effective_trace_level="pixel",
        ),),
        region_map_svg="<svg/>",
        background={"mode": "inferred_alpha", "metadata": {"samples": [1, 2]}},
        material_pretrace=MaterialPretraceResult(
            high_color_quantized=np.full((2, 2, 3), 1, dtype=np.uint8),
            shape_merged=np.full((2, 2, 3), 2, dtype=np.uint8),
            color_field_merged=np.full((2, 2, 3), 3, dtype=np.uint8),
            material_proxy=np.full((2, 2, 3), 4, dtype=np.uint8),
            shape_region_count=5,
            color_region_count=6,
            gradient_supports=(GradientSupport(
                id="gradient-1",
                area=3,
                model={
                    "kind": "linear",
                    "stops": [{"offset": 0.0, "rgb": [1, 2, 3]}],
                },
                representative_rgb=(4, 5, 6),
            ),),
            resolved_max_sampling_colors=7,
        ),
    )


def _evidence(trace: TraceResult | None = None):
    trace = _trace() if trace is None else trace
    return capture_working_geometry_evidence(
        kind="source",
        trace=trace,
        geometry_source=_source(trace.width, trace.height),
        trace_roots=_regions(trace.width, trace.height),
    )


def _drawing(drawing_id: str, color: str = "#112233") -> Drawing:
    return Drawing(
        drawing_id,
        Canvas(2.0, 2.0),
        (DrawingRegion(
            "r1",
            geometry=PrimitiveGeometry(
                RectanglePrimitive((0.0, 0.0), 2.0, 2.0)
            ),
            fill=FlatFill(color),
            source_region_ids=("r1",),
        ),),
    )


def _candidate(
    drawing_id: str,
    evidence,
    *,
    color: str = "#112233",
    label: str | None = None,
    baseline_version_id: str | None = None,
    baseline_candidate_index: int | None = None,
    submitted_plan: DrawingPlan | None = None,
) -> VersionCandidate:
    recipe = EffectiveDrawingRecipe(
        evidence.revision,
        evidence.engine_revision,
        ActionSettings(epsilon=1.0, max_error=1.5, fit_strategy="quadratic"),
    )
    drawing = _drawing(drawing_id, color)
    return VersionCandidate(
        drawing=drawing,
        evidence=evidence,
        submitted_plan=submitted_plan,
        submitted_edits=(),
        effective_recipe=recipe,
        replay=ReplayRecord(
            evidence.revision,
            recipe.revision,
            evidence.engine_revision,
            drawing_revision(drawing),
            build_paint_evidence(_source()).revision,
        ),
        label=label,
        baseline_version_id=baseline_version_id,
        baseline_candidate_index=baseline_candidate_index,
    )


def _create(store: DrawingStore, evidence=None):
    evidence = _evidence() if evidence is None else evidence
    drawing_id = store.new_drawing_id()
    (root,) = store.create(
        drawing_id,
        build_paint_evidence(_source()),
        (_candidate(drawing_id, evidence, baseline_candidate_index=0),),
    )
    return drawing_id, evidence, root


def test_store_creates_one_immutable_canonical_root_version(fake_clock):
    store = DrawingStore(now=fake_clock)
    drawing_id, _evidence_value, root = _create(store)
    assert drawing_id.startswith("drw_")
    assert root.id == "v0" and root.parent_id is None
    assert root.submitted_plan is None and root.drawing.drawing_id == drawing_id
    assert not hasattr(root, "regions")
    assert tuple(item.id for item in store.get_manifest(drawing_id).versions) == ("v0",)


def test_store_retains_frozen_typed_plan_and_detached_evidence(fake_clock):
    store = DrawingStore(now=fake_clock)
    drawing_id, evidence, root = _create(store)
    plan = DrawingPlan.model_validate({
        "version": "vectormark.plan.v1",
        "drawing_id": drawing_id,
        "base_version": "v0",
        "ops": [{
            "op": "set_fill",
            "target": "r1",
            "fill": {"type": "flat", "color": "#223344"},
        }],
    })
    (child,) = store.append(
        drawing_id,
        root.id,
        (_candidate(
            drawing_id,
            evidence,
            color="#223344",
            baseline_version_id=root.id,
            submitted_plan=plan,
        ),),
    )
    assert child.submitted_plan == plan
    assert child.drawing.regions[0].fill == FlatFill("#223344")
    loaded, _paint, loaded_evidence = store.load_replay(drawing_id, child.id)
    assert loaded == child and loaded_evidence == evidence


def test_store_branches_from_any_retained_version(fake_clock):
    store = DrawingStore(now=fake_clock)
    drawing_id, evidence, root = _create(store)
    (first,) = store.append(
        drawing_id, root.id,
        (_candidate(drawing_id, evidence, baseline_version_id=root.id),),
    )
    (second,) = store.append(
        drawing_id, root.id,
        (_candidate(drawing_id, evidence, baseline_version_id=root.id),),
    )
    (child,) = store.append(
        drawing_id, second.id,
        (_candidate(drawing_id, evidence, baseline_version_id=root.id),),
    )
    assert (first.id, second.id, child.id) == ("v0.0", "v0.1", "v0.1.0")
    assert child.parent_id == second.id


def test_store_allocates_sibling_versions_atomically(fake_clock):
    store = DrawingStore(now=fake_clock)
    drawing_id, evidence, root = _create(store)
    with ThreadPoolExecutor(max_workers=8) as executor:
        versions = list(executor.map(
            lambda _: store.append(
                drawing_id,
                root.id,
                (_candidate(drawing_id, evidence, baseline_version_id=root.id),),
            )[0].id,
            range(32),
        ))
    assert sorted(versions, key=lambda value: int(value.rsplit(".", 1)[1])) == [
        f"v0.{index}" for index in range(32)
    ]


def test_store_capability_crosses_request_contexts_and_expires(fake_clock):
    store = DrawingStore(now=fake_clock, idle_ttl_seconds=1800)
    drawing_id, _evidence_value, root = _create(store)
    assert store.get_version(drawing_id, root.id) == root
    fake_clock.advance(1800)
    with pytest.raises(DrawingNotFound) as error:
        store.get_version(drawing_id, root.id)
    assert error.value.__cause__ is None


def test_store_not_found_errors_hide_internal_key_errors(fake_clock):
    store = DrawingStore(now=fake_clock)
    drawing_id, evidence, root = _create(store)
    with pytest.raises(DrawingNotFound) as missing:
        store.get_version(drawing_id, "v404")
    assert missing.value.__cause__ is None
    with pytest.raises(DrawingNotFound) as parent:
        store.append(
            drawing_id,
            "v404",
            (_candidate(drawing_id, evidence, baseline_version_id=root.id),),
        )
    assert parent.value.__cause__ is None


def test_store_reads_and_appends_refresh_sliding_expiry(fake_clock):
    store = DrawingStore(now=fake_clock, idle_ttl_seconds=1800)
    drawing_id, evidence, root = _create(store)
    fake_clock.advance(1799)
    assert store.get_version(drawing_id, root.id) == root
    fake_clock.advance(1799)
    (child,) = store.append(
        drawing_id,
        root.id,
        (_candidate(
            drawing_id,
            evidence,
            label="first edit",
            baseline_version_id=root.id,
        ),),
    )
    fake_clock.advance(1799)
    assert store.get_version(drawing_id, child.id).label == "first edit"


def test_store_rejects_invalid_labels_without_consuming_child_number(fake_clock):
    class StringSubclass(str):
        pass

    store = DrawingStore(now=fake_clock)
    drawing_id, evidence, root = _create(store)
    for label in (StringSubclass("subclass"), object()):
        with pytest.raises(TypeError, match="label"):
            store.append(
                drawing_id,
                root.id,
                (_candidate(
                    drawing_id,
                    evidence,
                    label=label,  # type: ignore[arg-type]
                    baseline_version_id=root.id,
                ),),
            )
    (version,) = store.append(
        drawing_id,
        root.id,
        (_candidate(
            drawing_id,
            evidence,
            label="valid",
            baseline_version_id=root.id,
        ),),
    )
    assert version.id == "v0.0"


def test_store_detaches_all_nested_geometry_evidence(fake_clock):
    store = DrawingStore(now=fake_clock)
    trace = _mutable_trace()
    source_region = trace.regions[0]
    assert trace.material_pretrace is not None
    source_pretrace = trace.material_pretrace
    source_model = source_pretrace.gradient_supports[0].model
    source_background = trace.background
    evidence = _evidence(trace)
    drawing_id, _evidence_value, root = _create(store, evidence)
    (child,) = store.append(
        drawing_id,
        root.id,
        (_candidate(drawing_id, evidence, baseline_version_id=root.id),),
    )

    source_region.mask[...] = False
    source_region.contours[0][0, 0] = 99.0
    for array in (
        source_pretrace.high_color_quantized,
        source_pretrace.shape_merged,
        source_pretrace.color_field_merged,
        source_pretrace.material_proxy,
    ):
        array[...] = 99
    source_model["stops"][0]["rgb"][0] = 99  # type: ignore[index]
    source_background["metadata"]["samples"][0] = 99  # type: ignore[index]

    for version_id in (root.id, child.id):
        _version, _paint, snapshot = store.load_replay(drawing_id, version_id)
        region = snapshot.trace.regions[0]
        assert region.mask[0, 0] and region.contours[0][0, 0] == 0.0
        assert not region.mask.flags.writeable
        pretrace = snapshot.trace.material_pretrace
        assert pretrace is not None
        arrays = (
            pretrace.high_color_quantized,
            pretrace.shape_merged,
            pretrace.color_field_merged,
            pretrace.material_proxy,
        )
        assert [int(array[0, 0, 0]) for array in arrays] == [1, 2, 3, 4]
        assert all(not array.flags.writeable for array in arrays)
        assert pretrace.gradient_supports[0].model["stops"][0]["rgb"] == (1, 2, 3)  # type: ignore[index]
        assert snapshot.trace.background["metadata"]["samples"] == (1, 2)  # type: ignore[index]

    _version, _paint, public_snapshot = store.load_replay(drawing_id, root.id)
    public_trace = public_snapshot.trace.to_public_dict()
    json.dumps(public_trace)
    public_trace["geometry"]["background"]["metadata"]["samples"][0] = 77
    assert public_snapshot.trace.background["metadata"]["samples"] == (1, 2)  # type: ignore[index]

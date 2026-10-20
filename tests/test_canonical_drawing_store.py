from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from vectormark.candidate import FlatFill as LegacyFlatFill
from vectormark.drawing import Canvas, Drawing, DrawingRegion
from vectormark.drawing_actions import ActionSettings
from vectormark.drawing_engine import RasterImage
from vectormark.drawing_fill import FlatFill
from tests._working_evidence import capture_working_geometry_evidence
from vectormark.refinement.assembly_binding import working_regions_from_evidence
from vectormark.drawing_recipe import (
    EffectiveDrawingRecipe,
    ReplayRecord,
    build_paint_evidence,
    drawing_revision,
    snapshot_geometry_evidence,
    )
from vectormark.drawing_state import DrawingNotFound, DrawingStore, VersionCandidate
from vectormark.drawing_trace import RootPathEvidence, TraceOptions, TraceResult
from vectormark.fit import Shape
from vectormark.refinement.region_sample import RegionSample
from vectormark.path_fit_evidence import FittedRun, automatic_fit_evidence
from vectormark.path_geometry import PrimitiveGeometry, RectanglePrimitive


def _source() -> RasterImage:
    return RasterImage(2, 2, "rgb8", bytes(range(12)))


def _trace() -> TraceResult:
    return TraceResult(2, 2, TraceOptions(), (), "<svg/>")


def _roots() -> tuple[RegionSample, ...]:
    return (
        RegionSample.from_shape(
            id=1,
            shape=Shape("rect", {"x": 0.0, "y": 0.0, "w": 2.0, "h": 2.0}),
            fill=LegacyFlatFill("#112233"),
            z=0,
            raster=np.ones((2, 2), dtype=bool),
            drawing_id="r1",
            source_regions=("r1",),
        ),
    )


def _drawing(drawing_id: str, color: str = "#112233") -> Drawing:
    return Drawing(
        drawing_id,
        Canvas(2.0, 2.0),
        (
            DrawingRegion(
                "r1",
                geometry=PrimitiveGeometry(RectanglePrimitive((0.0, 0.0), 2.0, 2.0)),
                fill=FlatFill(color),
                source_region_ids=("r1",),
            ),
        ),
    )


def _candidate(
    drawing_id: str,
    evidence,
    *,
    color: str = "#112233",
    baseline_version_id: str | None = None,
    baseline_candidate_index: int | None = None,
) -> VersionCandidate:
    settings = ActionSettings(epsilon=1.0, max_error=1.5, fit_strategy="quadratic")
    recipe = EffectiveDrawingRecipe(
        evidence_revision=evidence.revision,
        engine_revision=evidence.engine_revision,
        settings=settings,
    )
    drawing = _drawing(drawing_id, color)
    replay = ReplayRecord(
        evidence_revision=evidence.revision,
        recipe_revision=recipe.revision,
        engine_revision=evidence.engine_revision,
        output_revision=drawing_revision(drawing),
        paint_revision=build_paint_evidence(_source()).revision,
    )
    return VersionCandidate(
        drawing=drawing,
        evidence=evidence,
        submitted_plan=None,
        submitted_edits=(),
        effective_recipe=recipe,
        replay=replay,
        baseline_version_id=baseline_version_id,
        baseline_candidate_index=baseline_candidate_index,
    )


def _evidence(kind="source"):
    return capture_working_geometry_evidence(
        kind=kind,
        trace=_trace(),
        geometry_source=_source(),
        trace_roots=_roots(),
    )


def test_store_atomically_publishes_canonical_source_and_guide_chains():
    store = DrawingStore()
    drawing_id = store.new_drawing_id()
    source_evidence = _evidence()
    paint = build_paint_evidence(_source())
    v0, v1 = store.create(
        drawing_id,
        paint,
        (
            _candidate(drawing_id, source_evidence, baseline_candidate_index=0),
            _candidate(drawing_id, source_evidence, color="#223344", baseline_candidate_index=0),
        ),
    )
    guide = _evidence("geometry-guide")
    baseline, optimized = store.append(
        drawing_id,
        v1.id,
        (
            _candidate(drawing_id, guide, color="#334455", baseline_candidate_index=0),
            _candidate(drawing_id, guide, color="#445566", baseline_candidate_index=0),
        ),
    )

    assert (v0.id, v1.id, baseline.id, optimized.id) == (
        "v0", "v1", "v1.0", "v1.0.0"
    )
    assert (v1.parent_id, baseline.parent_id, optimized.parent_id) == (
        "v0", "v1", "v1.0"
    )
    assert baseline.baseline_version_id == optimized.baseline_version_id == "v1.0"
    selected = store.get_version(drawing_id, optimized.id)
    assert selected.drawing.drawing_id == drawing_id
    assert not hasattr(selected, "regions")
    manifest = store.get_manifest(drawing_id)
    assert tuple(item.id for item in manifest.versions) == ("v0", "v1", "v1.0", "v1.0.0")
    assert not hasattr(manifest.versions[0], "drawing")

    (baseline_sibling,) = store.append(
        drawing_id,
        baseline.id,
        (_candidate(drawing_id, guide, color="#556677", baseline_version_id=baseline.id),),
    )
    assert baseline_sibling.id == "v1.0.1"
    assert store.get_version(drawing_id, optimized.id) == optimized


def test_store_validates_complete_chain_before_publication_or_counter_allocation():
    store = DrawingStore()
    drawing_id = store.new_drawing_id()
    evidence = _evidence()
    paint = build_paint_evidence(_source())

    bad_second = replace(
        _candidate(drawing_id, evidence, baseline_candidate_index=0),
        drawing=_drawing("wrong"),
    )
    with pytest.raises(ValueError, match="drawing identity"):
        store.create(
            drawing_id,
            paint,
            (
                _candidate(drawing_id, evidence, baseline_candidate_index=0),
                bad_second,
            ),
        )
    with pytest.raises(DrawingNotFound):
        store.get_manifest(drawing_id)

    (v0,) = store.create(
        drawing_id,
        paint,
        (_candidate(drawing_id, evidence, baseline_candidate_index=0),),
    )
    with pytest.raises(ValueError, match="drawing identity"):
        store.append(
            drawing_id,
            v0.id,
            (
                _candidate(drawing_id, evidence, baseline_candidate_index=0),
                bad_second,
            ),
        )
    assert tuple(item.id for item in store.get_manifest(drawing_id).versions) == ("v0",)

    (child,) = store.append(
        drawing_id,
        v0.id,
        (_candidate(drawing_id, evidence, baseline_candidate_index=0),),
    )
    assert child.id == "v0.0"


def test_store_prepares_exact_source_ids_before_atomic_publication():
    store = DrawingStore()
    drawing_id = store.new_drawing_id()
    evidence = _evidence()
    paint = build_paint_evidence(_source())
    candidates = (
        _candidate(drawing_id, evidence, baseline_candidate_index=0),
        _candidate(
            drawing_id,
            evidence,
            color="#223344",
            baseline_candidate_index=0,
        ),
    )

    def fail(versions):
        assert tuple(version.id for version in versions) == ("v0", "v1")
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        store.create_prepared(drawing_id, paint, candidates, prepare=fail)
    with pytest.raises(DrawingNotFound):
        store.get_manifest(drawing_id)

    versions, prepared = store.create_prepared(
        drawing_id,
        paint,
        candidates,
        prepare=lambda proposed: tuple(version.id for version in proposed),
    )
    assert tuple(version.id for version in versions) == prepared == ("v0", "v1")


def test_store_failed_append_preparation_does_not_publish_or_consume_child_number():
    store = DrawingStore()
    drawing_id = store.new_drawing_id()
    evidence = _evidence()
    paint = build_paint_evidence(_source())
    (root,) = store.create(
        drawing_id,
        paint,
        (_candidate(drawing_id, evidence, baseline_candidate_index=0),),
    )
    candidates = (
        _candidate(drawing_id, evidence, baseline_candidate_index=0),
        _candidate(
            drawing_id,
            evidence,
            color="#223344",
            baseline_candidate_index=0,
        ),
    )

    def fail(versions):
        assert tuple(version.id for version in versions) == ("v0.0", "v0.0.0")
        raise RuntimeError("panel failed")

    with pytest.raises(RuntimeError, match="panel failed"):
        store.append_prepared(
            drawing_id, root.id, candidates, prepare=fail
        )
    assert tuple(item.id for item in store.get_manifest(drawing_id).versions) == (
        "v0",
    )

    versions, prepared = store.append_prepared(
        drawing_id,
        root.id,
        candidates,
        prepare=lambda proposed: tuple(version.id for version in proposed),
    )
    assert tuple(version.id for version in versions) == prepared == (
        "v0.0",
        "v0.0.0",
    )


def test_store_capability_is_cross_context_and_expires_after_idle_timeout():
    now = [0.0]
    store = DrawingStore(now=lambda: now[0], idle_ttl_seconds=1800)
    drawing_id = store.new_drawing_id()
    evidence = _evidence()
    (root,) = store.create(
        drawing_id,
        build_paint_evidence(_source()),
        (_candidate(drawing_id, evidence, baseline_candidate_index=0),),
    )

    assert store.get_version(drawing_id, root.id) == root
    now[0] = 1800
    with pytest.raises(DrawingNotFound):
        store.get_version(drawing_id, root.id)


def test_store_rejects_duplicate_and_mismatched_drawing_identity():
    store = DrawingStore()
    drawing_id = store.new_drawing_id()
    evidence = _evidence()
    paint = build_paint_evidence(_source())
    candidate = _candidate(drawing_id, evidence, baseline_candidate_index=0)
    store.create(drawing_id, paint, (candidate,))

    with pytest.raises(ValueError, match="duplicate drawing"):
        store.create(drawing_id, paint, (candidate,))
    with pytest.raises(ValueError, match="drawing identity"):
        store.create(
            store.new_drawing_id(), paint, (candidate,)
        )


def test_store_rejects_output_with_copied_replay_receipt():
    store = DrawingStore()
    drawing_id = store.new_drawing_id()
    evidence = _evidence()
    candidate = _candidate(drawing_id, evidence, baseline_candidate_index=0)
    forged = replace(candidate, drawing=_drawing(drawing_id, "#abcdef"))

    with pytest.raises(ValueError, match="output revision"):
        store.create(
            drawing_id,
            build_paint_evidence(_source()),
            (forged,),
        )
    with pytest.raises(DrawingNotFound):
        store.get_manifest(drawing_id)


def test_geometry_evidence_owns_roots_and_hashes_private_root_path_evidence():
    fit = automatic_fit_evidence(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
        (),
        (),
        (FittedRun(0, 1, 1), FittedRun(1, 2, 1), FittedRun(2, 3, 1)),
        "M0 0 L2 0 L2 2 L0 2 Z",
        settings=(1.0, 1.5, "quadratic"),
    )
    trace = replace(
        _trace(),
        root_path_evidence=(RootPathEvidence("r1", "r1.p0", fit),),
    )
    roots = _roots()
    evidence = capture_working_geometry_evidence(
        kind="source", trace=trace, geometry_source=_source(), trace_roots=roots
    )
    roots[0].raster[:] = False
    object.__setattr__(trace, "root_path_evidence", ())

    assert evidence.region_evidence['r1']['raster'] == bytes([1, 1, 1, 1])
    assert len(evidence.root_paths) == 1
    without_private_evidence = capture_working_geometry_evidence(
        kind="source",
        trace=_trace(),
        geometry_source=_source(),
        trace_roots=_roots(),
    )
    assert evidence.revision != without_private_evidence.revision


def test_geometry_evidence_retains_canonical_scene_and_detached_observations():
    roots = _roots()
    evidence = capture_working_geometry_evidence(
        kind="source", trace=_trace(), geometry_source=_source(), trace_roots=roots,
    )
    assert not hasattr(evidence, "trace_roots")
    assert isinstance(evidence.drawing, Drawing)
    assert evidence.region_evidence["r1"]["raster"] == bytes([1, 1, 1, 1])
    roots[0].raster[:] = False
    roots[0].current.params["w"] = 99
    detached = snapshot_geometry_evidence(evidence)
    assert detached.drawing == evidence.drawing
    assert detached.drawing is not evidence.drawing
    primitive = detached.drawing.region("r1").geometry.primitive
    object.__setattr__(primitive, "width", 99)
    assert evidence.drawing.region("r1").geometry.primitive.width == 2
    with pytest.raises(ValueError, match="revision"):
        snapshot_geometry_evidence(detached)
    with pytest.raises(TypeError):
        evidence.region_evidence["r1"]["raster"] = b""


def test_disposable_kernel_geometry_does_not_alias_retained_evidence():
    root = _roots()[0].with_current(Shape("path", {"d": "M0 0 L2 0 L2 2 L0 2 Z"}))
    evidence = capture_working_geometry_evidence(
        kind="source", trace=_trace(), geometry_source=_source(), trace_roots=(root,),
    )
    working = working_regions_from_evidence(evidence)[0]
    command = working.current.params['canonical_path_geometry'].path.commands[1]
    object.__setattr__(command, "end", (99., 99.))
    assert evidence.drawing.region("r1").geometry.path.commands[1].end == (2., 0.)
    assert snapshot_geometry_evidence(evidence).revision == evidence.revision


def test_geometry_evidence_keeps_clone_relationship_only_in_canonical_geometry():
    root = _roots()[0]
    clone = replace(root, id=2, drawing_id="r2", current=Shape("use", {
        "href_obj_id": 1, "transform": (1., 0., 0., 1., 3., 0.),
    }))
    evidence = capture_working_geometry_evidence(
        kind="source", trace=_trace(), geometry_source=_source(), trace_roots=(root, clone),
    )
    assert not {"href_obj_id", "href_region_path", "transform", "d"} & evidence.shape_metadata["r2"].keys()
    restored = working_regions_from_evidence(evidence)
    assert restored[1].current.params["href_obj_id"] == 1
    assert restored[1].current.params["transform"] == (1., 0., 0., 1., 3., 0.)

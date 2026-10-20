"""Private root replay evidence must precede fitting and survive retention."""

from vectormark.drawing_adapter import drawing_from_vector_regions as _export_drawing, normalize_drawing_addresses as _export_addresses
from vectormark.drawing_render import render_drawing as _canonical_render
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import pickle

import numpy as np
import pytest


from vectormark.optimizer.assembly_binding import assemble_working_roots
from vectormark.contour import region_contours
import vectormark.drawing_assembly as assembly
from vectormark.drawing_actions import ActionSettings
from vectormark.drawing_engine import RasterImage
from tests._working_evidence import capture_working_geometry_evidence
from vectormark.drawing_recipe import (
    EffectiveDrawingRecipe,
    build_paint_evidence,
)
from vectormark.drawing_replay import propose_effective_recipe
from vectormark.drawing_trace import PythonTraceEngine, TraceOptions
from vectormark.drawing_state import DrawingStore, VersionCandidate
from vectormark.path_geometry import LineCommand, geometry_from_svg_d
import vectormark.fit as fit
from vectormark.path_fit_evidence import automatic_fit_evidence


def _fixture(variant):
    y, x = np.mgrid[:64, :64]
    mask = (x - 32) ** 2 + (y - 32) ** 2 < 25 ** 2
    rgb = np.full((64, 64, 3), 255, np.uint8)
    rgb[mask] = (180, 30, 40)
    if variant != "single":
        rgb[mask & (x > 32)] = (20, 80, 170)
    boundary = variant in {"boundary", "single"}
    trace = PythonTraceEngine().trace(
        rgb,
        TraceOptions(min_region_size=2, source_has_alpha=boundary,
                     remove_background="auto" if boundary else "off"),
        alpha=mask.astype(np.uint8) * 255 if boundary else None,
    )
    return trace, rgb


@pytest.mark.parametrize("variant,expected", [
    ("merged", "0053de9b0e441840803127b0ce8fe7cf8db5f2ccd2251f1eab730406a42f2dbc"),
    ("boundary", "0053de9b0e441840803127b0ce8fe7cf8db5f2ccd2251f1eab730406a42f2dbc"),
    ("single", "26eec8e444f01227845e79743bf2954ea1bab376a7374a29e716738ad0865190"),
    ("raw", "0053de9b0e441840803127b0ce8fe7cf8db5f2ccd2251f1eab730406a42f2dbc"),
])
def test_root_variants_retain_evidence_without_changing_render(variant, expected):
    trace, rgb = _fixture(variant)
    result = assemble_working_roots(trace, None if variant == "raw" else rgb, min_region_fraction=0)
    assert trace.root_path_evidence == ()
    assert result.trace.root_path_evidence
    svg = _canonical_render(_export_drawing("export", width=(result.trace).width, height=(result.trace).height, regions=_export_addresses(result.regions))).svg
    assert hashlib.sha256(svg.encode()).hexdigest() == expected
    assert "root_path_evidence" not in result.trace.to_public_dict()
    json.dumps(result.trace.to_public_dict())
    recovered = pickle.loads(pickle.dumps(result.trace))
    assert recovered.root_path_evidence == result.trace.root_path_evidence


def test_boundary_evidence_is_actual_fit_input_not_retained_raster(monkeypatch):
    trace, rgb = _fixture("single")
    # The new material path only differs from its hard classification when
    # there is genuine partial edge coverage in the source image.
    from PIL import Image
    rgb = np.asarray(Image.fromarray(rgb).resize((128, 128), Image.Resampling.BICUBIC)
                     .resize((64, 64), Image.Resampling.LANCZOS))
    trace = PythonTraceEngine().trace(rgb, replace(trace.options, trace_level="subpixel", max_sampling_colors=2))
    captured = []
    original = assembly._fit_contours_with_corners

    def capture(contours, trace):
        captured.append(tuple(contour.copy() for contour in contours))
        return original(contours, trace)

    monkeypatch.setattr(assembly, "_fit_contours_with_corners", capture)
    result = assemble_working_roots(trace, rgb, min_region_fraction=0)
    leaves = [leaf for root in result.regions for leaf in root.leaves()]
    records = result.trace.root_path_evidence
    assert [record.region_id for record in records] == ["r1"]
    assert [record.path_id for record in records] == ["r1.p0"]
    differs = False
    for leaf, record in zip(leaves, records, strict=True):
        source = record.fit.source.path
        matches = [
            contours for contours in captured
            if np.array_equal([command.end for command in source.commands], contours[0])
        ]
        # Drawing addresses are normalized by provenance after fitting; fitting
        # order is not a second region-identity contract.
        assert len(matches) == 1
        expected = matches[0][0]
        np.testing.assert_array_equal([command.end for command in source.commands], expected)
        raster_contour = region_contours(leaf.raster)[0]
        differs |= not np.array_equal(expected, raster_contour)
        assert record.matches_output(leaf.current.params["d"], stage="root-pre-stitch", revision=record.fit.revision)
        assert not record.matches_output(leaf.current.params["d"], stage="optimized", revision=record.fit.revision)
        assert not record.matches_output(leaf.current.params["d"] + " ", stage=record.stage, revision=record.fit.revision)
        assert not record.matches_output(leaf.current.params["d"], stage=record.stage, revision="other")
    assert differs, "regression fixture must distinguish retained raster from fitting input"


def test_single_material_retains_baseline_evidence_and_declarations():
    trace, rgb = _fixture("single")
    result = assemble_working_roots(trace, rgb, min_region_fraction=0)
    baseline = trace.geometry_regions[0].trace_path
    assert result.trace.root_path_evidence[0].fit == baseline.path_evidence[0]
    from vectormark.immutable_value import freeze_value
    assert result.regions[0].diagnostics["corners"] == freeze_value({
        **baseline.corner_diagnostics, "path": baseline.d,
    })


def test_no_rgb_root_rejects_missing_source_instead_of_using_possibly_fitted_svg():
    trace, _ = _fixture("raw")
    missing = replace(trace.regions[0], contours=())
    assert missing.trace_path.d  # An SVG alone is not the missing fitting input.
    with pytest.raises(ValueError, match="requires retained source contours"):
        assemble_working_roots(replace(trace, regions=(missing,)))


def test_subpath_evidence_keeps_hole_order_and_independent_path_addresses():
    trace, rgb = _fixture("single")
    rgb[27:37, 27:37] = 255
    alpha = np.where(np.any(rgb != 255, axis=2), 255, 0).astype(np.uint8)
    trace = PythonTraceEngine().trace(
        rgb, replace(trace.options, max_hole_area=0), alpha=alpha,
    )
    result = assemble_working_roots(trace, rgb, min_region_fraction=0)
    records = result.trace.root_path_evidence
    assert [record.path_id for record in records] == ["r1.p0", "r1.p1"]
    for record, contour in zip(records, trace.geometry_regions[0].contours, strict=True):
        np.testing.assert_array_equal([command.end for command in record.fit.source.path.commands], contour)
    assert records[0].fit.source.path.id != records[1].fit.source.path.id


def _isolated_detector(monkeypatch, corners=(), rounded=(), lines=()):
    monkeypatch.setattr(fit, "rdp", lambda points, _epsilon: np.asarray(points))
    monkeypatch.setattr(fit, "_suppress_cornerlet_vertices", lambda points, **_kwargs: points)
    # A protected interval alone is not rounded evidence. Supply an actual
    # classified corner inside each explicitly mocked rounded interval.
    monkeypatch.setattr(fit, "corner_indices", lambda poly, **kwargs: list(corners) + [
        (start + ((end - start) % (len(poly) - 1)) // 2) % (len(poly) - 1)
        for start, end in rounded
    ])
    monkeypatch.setattr(fit, "corner_curve_radius_at", lambda *args, **kwargs: 1.0 if rounded else None)
    monkeypatch.setattr(fit, "corner_curve_span_at", lambda ring, index, **kwargs: next((
        (start, end) for start, end in rounded
        if index == start or index == end or fit._inside_closed_span(index, start, end)
    ), None))
    monkeypatch.setattr(fit, "_protected_line_spans", lambda _ring, **_kwargs: list(lines))
    monkeypatch.setattr(fit, "_protected_corner_ranges", lambda *args, **kwargs: list(rounded))


def _precise_ring():
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    return np.column_stack((40.123456789 + 30 * np.cos(theta), 50.987654321 + 20 * np.sin(theta)))


@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize("explicit_close", [False, True])
def test_actual_run_mapping_preserves_wrapped_spans_and_precise_source(monkeypatch, rounded, explicit_close):
    ring = _precise_ring()
    _isolated_detector(monkeypatch, corners=() if rounded else (5,), rounded=((16, 2),) if rounded else ())
    contour = np.vstack((ring, ring[0])) if explicit_close else ring
    stages = fit.fit_path_stages(contour, epsilon=1, max_error=1, progressive=rounded)
    evidence = stages.baseline_evidence
    source = evidence.source
    np.testing.assert_array_equal([command.end for command in source.path.commands], np.vstack((ring, ring[0])))
    assert all(isinstance(command, LineCommand) for command in source.path.drawable_commands)
    assert all(a.end != b.end for a, b in zip(source.path.commands, source.path.commands[1:]))
    assert len(source.anchors) == 1
    anchor = source.anchors[0]
    assert source.endpoint(anchor.entry_command_id) == tuple(ring[16 if rounded else 5])
    parsed = geometry_from_svg_d(stages.baseline_d, region_id="display")
    outputs = [command for span in evidence.correspondence.spans for command in span.output_command_ids]
    assert outputs == [f"{evidence.output_path_id}.c{i}" for i in range(1, len(parsed.path.commands))]
    sources = [command for span in evidence.correspondence.spans for command in span.source_command_ids]
    assert len(sources) == len(set(sources)) == len(ring)
    assert set(sources) == {command.id for command in source.path.drawable_commands}
    assert all(".source." in boundary.source_command_id and ".fit." in boundary.output_command_id
               for boundary in evidence.correspondence.boundaries)
    if rounded:
        owned = next(span for span in evidence.correspondence.spans if span.kind == "corner")
        assert owned.source_command_ids == anchor.owned_command_ids
        assert owned.source_command_ids == tuple(f"{source.path.id}.c{i}" for i in (17, 18, 19, 20, 1, 2))
    else:
        assert all(span.kind == "side" for span in evidence.correspondence.spans)
        # Singleton's opposite cut is a fitting boundary, not a second feature.
        assert len(evidence.correspondence.spans) == 2
    assert evidence.diagnostics == ()
    with pytest.raises(FrozenInstanceError):
        source.path.commands[0].end = (0, 0)


def test_missing_semantic_fit_boundary_is_explicit_and_declaration_survives(monkeypatch):
    _isolated_detector(monkeypatch, corners=(5,), lines=((2, 8),))
    evidence = fit.fit_path_stages(_precise_ring(), epsilon=1, max_error=1, progressive=True).baseline_evidence
    assert len(evidence.source.anchors) == 1
    assert not evidence.correspondence.boundaries
    assert evidence.diagnostics[0].declaration == evidence.source.anchors[0]
    assert evidence.diagnostics[0].reason == "missing_fitted_boundary: point"


def test_ambiguous_source_declaration_has_explicit_lossless_diagnostic():
    ring = _precise_ring()
    d, runs = fit._fit_path_d_from_cuts(ring, [2, 4, 16, 18], epsilon=1, max_error=1)
    evidence = automatic_fit_evidence(ring, (), ((16, 2), (18, 4)), runs, d, settings=(1, 1))
    assert len(evidence.source.anchors) == 1
    diagnostic = evidence.diagnostics[0]
    assert diagnostic.reason.startswith("ambiguous_source_declaration:")
    assert evidence.source.endpoint(diagnostic.declaration.entry_command_id) == tuple(ring[18])
    assert evidence.source.endpoint(diagnostic.declaration.exit_command_id) == tuple(ring[4])
    assert diagnostic.declaration.owned_command_ids


def test_semantic_origin_uses_addressable_closing_line_not_move_alias(monkeypatch):
    _isolated_detector(monkeypatch, corners=(0,))
    evidence = fit.fit_path_stages(_precise_ring(), epsilon=1, max_error=1).baseline_evidence
    source = evidence.source.path
    anchor = evidence.source.anchors[0]
    assert anchor.entry_command_id == source.commands[-1].id
    assert anchor.entry_command_id != source.commands[0].id
    assert evidence.correspondence.boundaries[0].source_command_id == source.commands[-1].id
    assert evidence.correspondence.boundaries[0].output_command_id == evidence.correspondence.spans[-1].output_command_ids[-1]


def test_revision_includes_exact_source_declarations_and_resolved_settings(monkeypatch):
    ring = _precise_ring()
    _isolated_detector(monkeypatch)
    first = fit.fit_path_stages(ring, epsilon=1, max_error=1).baseline_evidence
    settings = fit.fit_path_stages(ring, epsilon=1, max_error=1.001).baseline_evidence
    moved = ring.copy()
    moved[3, 0] += 0.000000001
    source = fit.fit_path_stages(moved, epsilon=1, max_error=1).baseline_evidence
    declared = fit.fit_path_stages(ring, epsilon=1, max_error=1, forced_corners=ring[[5]]).baseline_evidence
    assert len({item.revision for item in (first, settings, source, declared)}) == 4


def test_store_detaches_nested_source_evidence_across_input_public_and_siblings():
    trace, rgb = _fixture("boundary")
    rooted = assemble_working_roots(trace, rgb, min_region_fraction=0)
    store = DrawingStore()
    drawing_id = store.new_drawing_id()
    source = RasterImage(
        rooted.trace.width,
        rooted.trace.height,
        "rgb8",
        rgb.tobytes(),
    )
    evidence = capture_working_geometry_evidence(
        kind="source",
        trace=rooted.trace,
        geometry_source=source,
        trace_roots=rooted.regions,
    )
    paint = build_paint_evidence(source)
    recipe = EffectiveDrawingRecipe(
        evidence.revision,
        evidence.engine_revision,
        ActionSettings(epsilon=1.0, max_error=1.0, fit_strategy="quadratic"),
    )
    execution = propose_effective_recipe(
        drawing_id=drawing_id,
        evidence=evidence,
        paint=paint,
        recipe=recipe,
    )

    def candidate(*, evidence_value, baseline_version_id=None, baseline_candidate_index=None):
        return VersionCandidate(
            execution.drawing,
            evidence_value,
            None,
            (),
            recipe,
            execution.replay,
            baseline_version_id=baseline_version_id,
            baseline_candidate_index=baseline_candidate_index,
        )

    (root,) = store.create(
        drawing_id,
        paint,
        (candidate(evidence_value=evidence, baseline_candidate_index=0),),
    )
    (child,) = store.append(
        drawing_id,
        root.id,
        (candidate(evidence_value=None, baseline_version_id=root.id),),
    )
    (sibling,) = store.append(
        drawing_id,
        root.id,
        (candidate(evidence_value=None, baseline_version_id=root.id),),
    )
    original = rooted.trace.root_path_evidence[0].fit.source.path.commands[0].end
    output_ids = rooted.trace.root_path_evidence[0].fit.correspondence.spans[0].output_command_ids
    baseline_revision = rooted.trace.geometry_regions[0].trace_path.path_evidence[0].revision
    _root_version, _root_paint, root_snapshot = store.load_replay(
        drawing_id, root.id
    )
    _child_version, _child_paint, child_snapshot = store.load_replay(
        drawing_id, child.id
    )
    # Even bypassing a frozen dataclass on caller-owned and loaded snapshots
    # cannot alter retained evidence or another load.
    object.__setattr__(rooted.trace.root_path_evidence[0].fit.source.path.commands[0], "end", (999, 999))
    object.__setattr__(root_snapshot.trace.root_path_evidence[0].fit.source.path.commands[0], "end", (888, 888))
    object.__setattr__(child_snapshot.trace.root_path_evidence[0].fit.source.path.commands[0], "end", (777, 777))
    object.__setattr__(child_snapshot.trace.root_path_evidence[0].fit.correspondence.spans[0], "output_command_ids", ("tampered",))
    object.__setattr__(root_snapshot.trace.geometry_regions[0].trace_path.path_evidence[0], "revision", "tampered")
    for version_id in (root.id, child.id, sibling.id):
        _version, _paint, retained = store.load_replay(drawing_id, version_id)
        assert retained.trace.root_path_evidence[0].fit.source.path.commands[0].end == original
        assert retained.trace.root_path_evidence[0].fit.correspondence.spans[0].output_command_ids == output_ids
        assert retained.trace.geometry_regions[0].trace_path.path_evidence[0].revision == baseline_revision

"""Topology truth must survive retained publication, not only SVG rendering."""

from dataclasses import asdict as _trace_options
from vectoreyes.drawing_render import trace_summary as _canonical_trace
from vectoreyes.drawing_adapter import drawing_from_region_samples as _export_drawing, normalize_drawing_addresses as _export_addresses
from vectoreyes.drawing_render import render_drawing as _canonical_render
import numpy as np
import pytest

from vectoreyes import drawing_refine
from vectoreyes.candidate import FlatFill
from vectoreyes.drawing_plan import parse_plan
from vectoreyes.drawing_trace import TraceOptions, TraceResult
from vectoreyes.fit import Shape
from vectoreyes.refinement.passes.compound import split_compound_pass
from vectoreyes.refinement.region_sample import RegionSample
from vectoreyes.path_geometry import PathGeometryWithCutouts, geometry_from_svg_d
from vectoreyes.path_geometry import parse_svg_paths, ArcCommand, CubicCommand
from vectoreyes.path_topology import classify_filled_paths, component_paths
from vectoreyes.refinement.topology_binding import normalize_working_topology


OUTER = "M0 0 L12 0 L12 12 L0 12 Z"
HOLE = "M4 4 L8 4 L8 8 L4 8 Z"
DISCONNECTED = "M0 0 L8 0 L8 8 L0 8 Z M12 0 L20 0 L20 8 L12 8 Z"


def _region(d, *, fill_rule="evenodd"):
    return RegionSample.from_shape(
        id=1, shape=Shape("path", {"d": d, "fill_rule": fill_rule}),
        fill=FlatFill("#123456"), z=0, raster=np.zeros((24, 48), bool),
        drawing_id="r1", source_regions=("r1",),
    )


def _trace():
    return TraceResult(48, 24, TraceOptions(), (), "")


def test_path_edit_can_move_cutout_outside_and_publish_component_local_caches():
    from vectoreyes.drawing_engine import RasterImage
    from vectoreyes.drawing_actions import ActionSettings
    from tests._working_evidence import capture_working_geometry_evidence
    from vectoreyes.drawing_recipe import EffectiveDrawingRecipe, build_paint_evidence
    from vectoreyes.drawing_replay import extend_recipe_with_plan, propose_effective_recipe

    source = RasterImage(48, 24, 'rgb8', bytes([255]) * 48 * 24 * 3)
    root = _region('M0 0 L10 0 L10 10 L0 10 Z M2 2 L4 2 L4 4 L2 4 Z')
    reference = RegionSample.from_shape(id=2,
        shape=Shape('path', {'d': 'M20 0 L20 10 L24 3 L24 1 Z'}),
        fill=FlatFill('#123456'), z=1, raster=np.zeros((24, 48), bool), drawing_id='r2')
    evidence = capture_working_geometry_evidence(kind='source', trace=_trace(), geometry_source=source, trace_roots=(root, reference))
    paint = build_paint_evidence(source)
    base = EffectiveDrawingRecipe(evidence.revision, evidence.engine_revision,
        ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    plan = parse_plan({'version': 'vectoreyes.plan.v1', 'drawing_id': 'drw_test', 'base_version': 'v0', 'ops': [
        {'op': 'update_geometry', 'target': 'r1', 'ops': [
            {'op': 'set_parallel', 'target': 'r1.p1.c1', 'reference': 'r2.p0.c1', 'distance': -2},
            {'op': 'snap', 'target': 'r1.p1.c2', 'reference': 'r2.p0.c2'},
            {'op': 'snap', 'target': 'r1.p1.c3', 'reference': 'r2.p0.c3'},
        ]},
    ]})
    recipe = extend_recipe_with_plan(evidence=evidence, paint=paint, base_recipe=base, plan=plan).recipe
    replay = propose_effective_recipe(evidence=evidence, paint=paint, recipe=recipe, drawing_id='drw_test')
    from vectoreyes.refinement.replay_binding import working_regions_from_state
    actual = next(region for region in working_regions_from_state(replay) if region.drawing_id == 'r1')
    assert actual.is_branch and len(actual.children) == 2
    for child in actual.children:
        cache = child.current.params['canonical_path_geometry']
        assert cache.to_svg_d() == child.current.params['d']
        assert tuple(path.path.id for path in component_paths(cache)) == child.current.params['path_ids']


def test_redundant_path_retirement_projects_only_surviving_canonical_cache():
    from vectoreyes.path_geometry import CompoundPathGeometry

    paths = parse_svg_paths(f'{OUTER} {HOLE}', region_id='r1')
    cache = CompoundPathGeometry(paths, 'nonzero', 'retained input')
    d = cache.to_svg_d()
    root = _region(d, fill_rule='nonzero')
    root = root.with_current(Shape('path', {'d': d, 'fill_rule': 'nonzero',
        'canonical_path_geometry': cache}), footprint=root.footprint)
    result = normalize_working_topology((root,))
    actual = result[0].current.params['canonical_path_geometry']
    assert len(component_paths(actual)) == 1
    assert actual.to_svg_d() == result[0].current.params['d']


def test_hole_first_keeps_original_path_identity_but_corrects_topology():
    geometry = geometry_from_svg_d(f"{HOLE} {OUTER}", region_id="r1", fill_rule="evenodd")
    assert isinstance(geometry, PathGeometryWithCutouts)
    assert geometry.outer.path.id == "r1.p1"
    assert geometry.cutouts[0].path.id == "r1.p0"
    assert geometry.outer.path.start == (0, 0)


def test_manual_publication_creates_real_disconnected_children_and_retains_parent():
    source = _region(DISCONNECTED)
    plan = parse_plan({"version": "vectoreyes.plan.v1", "drawing_id": "drawing",
                       "base_version": "v0", "ops": []})
    result = drawing_refine.refine(_trace(), (source,), plan)
    assert result[0].is_branch
    assert tuple(leaf.drawing_id for leaf in result[0].leaves()) == ("r1-1", "r1-2")
    assert source.is_leaf and source.current.params["d"] == DISCONNECTED
    rendered = _canonical_render(_export_drawing("export", width=(_trace()).width, height=(_trace()).height, regions=_export_addresses(result)))
    assert {target["id"] for target in rendered.report["targets"]} == {"r1-1", "r1-2"}


def test_compound_pass_never_replaces_transparent_hole_with_white_cover():
    source = _region(f"{OUTER} {HOLE}")
    proposals = split_compound_pass([source], {1: source.raster})
    assert not proposals


@pytest.mark.parametrize("rule,inner,components,holes,redundant", [
    ("evenodd", HOLE, 1, 1, 0),
    ("nonzero", HOLE, 1, 0, 1),
    ("nonzero", "M4 4 L4 8 L8 8 L8 4 Z", 1, 1, 0),
])
def test_effective_winding_decides_subtraction(rule, inner, components, holes, redundant):
    paths = parse_svg_paths(f"{inner} {OUTER}", region_id="r")
    result = classify_filled_paths(paths, fill_rule=rule)
    assert len(result.components) == components
    assert len(component_paths(result.components[0])) - 1 == holes
    assert len(result.redundant_paths) == redundant


def test_nested_island_and_multiple_holes_are_independent_of_source_order():
    d = ("M1 1 L3 1 L3 3 L1 3 Z " + HOLE + " "
         "M5 5 L7 5 L7 7 L5 7 Z " + OUTER)
    result = classify_filled_paths(parse_svg_paths(d, region_id="r"), fill_rule="evenodd")
    assert [[path.path.id for path in component_paths(component)] for component in result.components] == [
        ["r.p2"], ["r.p3", "r.p0", "r.p1"],
    ]


def test_curve_and_arc_commands_survive_exactly_without_refitting():
    arc = "M1 0 A1 1 0 0 1 1 2 A1 1 0 0 1 1 0 Z"
    cubic = "M10 0 C12 0 12 2 10 2 L10 0 Z"
    parsed = parse_svg_paths(f"{arc} {cubic}", region_id="r")
    result = classify_filled_paths(parsed, fill_rule="evenodd")
    assert result.components == parsed
    assert isinstance(result.components[0].path.commands[1], ArcCommand)
    assert isinstance(result.components[1].path.commands[1], CubicCommand)


def test_inconsistent_native_predicate_never_silently_retires_tiny_exterior():
    d = "M0 0 L.00000002 0 L.00000002 .00000002 L0 .00000002 Z " + HOLE
    paths = parse_svg_paths(d, region_id="r")
    with pytest.raises(ValueError, match="inconsistent isolated exterior predicate"):
        classify_filled_paths(paths, fill_rule="evenodd")


@pytest.mark.parametrize("d", [
    f"{OUTER} M6 6 L16 6 L16 16 L6 16 Z",  # partial positive-area overlap
    f"{OUTER} {OUTER}",
    "M0 0 L8 8 L0 8 L8 0 Z M20 0 L24 0 L24 4 L20 4 Z",  # unresolved compound self-crossing
    f"M0 0 L4 0 L8 0 Z {HOLE}",  # zero-area member of compound
])
def test_unresolved_boundary_changes_fail_atomically(d):
    source = _region(d)
    with pytest.raises(ValueError, match="retained boolean normalization"):
        normalize_working_topology((source,))
    assert source.current.params["d"] == d and source.is_leaf


def test_point_touching_exteriors_still_retain_both_paths():
    paths = parse_svg_paths("M0 0 L4 0 L4 4 L0 4 Z M4 4 L8 4 L8 8 L4 8 Z", region_id="r")
    assert classify_filled_paths(paths, fill_rule="evenodd").components == paths


def test_single_path_command_contract_preserves_retraces_but_not_wrong_fill_rule():
    line = "M0 0 L4 0 L8 0 Z"
    assert geometry_from_svg_d(line, region_id="r").to_svg_d() == line
    twice = "M0 0 L4 0 L4 4 L0 4 L0 0 L4 0 L4 4 L0 4 L0 0 Z"
    assert geometry_from_svg_d(twice, region_id="r").path.id == "r.p0"
    with pytest.raises(ValueError, match="single-path fill rules differ"):
        geometry_from_svg_d(twice, region_id="r", fill_rule="evenodd")


def test_groups_are_reported_separately_and_typed_mcp_models_accept_them():
    from vectoreyes.drawing_adapter import drawing_from_region_samples
    from vectoreyes.drawing_render import render_drawing as render_canonical_drawing
    from vectoreyes.mcp_server import DrawingReportOutput, TraceSummaryOutput
    regions = normalize_working_topology((_region(DISCONNECTED),))
    drawing = drawing_from_region_samples(
        "drawing", width=48, height=24, regions=regions
    )
    report = render_canonical_drawing(drawing).report
    summary = _canonical_trace(_export_drawing("export", width=(_trace()).width, height=(_trace()).height, regions=_export_addresses(regions)), _trace_options((_trace()).options))
    assert report["groups"][0]["id"] == "r1"
    assert report["groups"][0]["child_ids"] == ["r1-1", "r1-2"]
    assert report["groups"][0]["diagnostics"]["topology"]["created_target_ids"] == ["r1-1", "r1-2"]
    assert isinstance(report["targets"][0]["geometry"], dict)
    typed_report = DrawingReportOutput.model_validate(report)
    typed_summary = TraceSummaryOutput.model_validate(summary)
    assert typed_report.groups == typed_summary.groups
    assert typed_report.groups[0].id == "r1"
    assert typed_summary.groups[0].child_ids == ["r1-1", "r1-2"]


def test_clone_expansion_preserves_exact_sources_paint_and_repeat_identity():
    from vectoreyes.refinement.symmetry_materialize import index_region_paths, resolve_source_path
    from vectoreyes.drawing_adapter import drawing_from_region_samples
    from vectoreyes.path_geometry import CloneGeometry
    from vectoreyes.candidate import LinearGradientFill
    from tests._render import render_svg
    source = _region(DISCONNECTED)
    fill = LinearGradientFill({"x1": 0, "y1": 0, "x2": 20, "y2": 0}, [(0, "#ff0000"), (1, "#0000ff")])
    clone = RegionSample(2, Shape("use", {"href_obj_id": 1, "transform": (1, 0, 0, 1, 22, 0)}),
                         fill, 1, source.footprint, raster=source.raster, drawing_id="r2", diagnostics={"clone": {"source": "r1"}})
    before = (source, clone)
    # Independent unsplit reference; canonical regions require normalized components.
    expected = (f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="24">'
                '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="20" y2="0">'
                '<stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#0000ff"/>'
                f'</linearGradient></defs><path d="{DISCONNECTED}" fill="#123456"/>'
                '<path d="M22 0 L30 0 L30 8 L22 8 Z M34 0 L42 0 L42 8 L34 8 Z" fill="url(#g)"/></svg>')
    result = normalize_working_topology(before)
    assert result[0].original == source.original and result[1].original == clone.original
    assert tuple(leaf.drawing_id for leaf in result[1].leaves()) == ("r2-1", "r2-2")
    index = index_region_paths(result)
    for ordinal, leaf in enumerate(result[1].children, start=1):
        assert resolve_source_path(index, (2, ordinal), leaf.current.params) == (1, ordinal)
        assert leaf.fill is fill
    canonical = drawing_from_region_samples("drawing", width=48, height=24, regions=result)
    assert canonical.region("r2-2").geometry == CloneGeometry("r1-2", canonical.region("r2-2").geometry.transforms)
    actual = _canonical_render(_export_drawing("export", width=(_trace()).width, height=(_trace()).height, regions=_export_addresses(result))).svg
    np.testing.assert_array_equal(render_svg(actual, 48, 24), render_svg(expected, 48, 24))
    repeated = normalize_working_topology(result)
    assert all(a is b for a, b in zip(result, repeated, strict=True))
    assert source.is_leaf and clone.is_leaf


def test_component_command_ids_drive_actual_manual_lookup_and_summary():
    from vectoreyes.drawing_adapter import drawing_from_region_samples
    result = normalize_working_topology((_region(DISCONNECTED),))
    summary = _canonical_trace(_export_drawing("export", width=(_trace()).width, height=(_trace()).height, regions=_export_addresses(result)), _trace_options((_trace()).options))
    assert summary["regions"][1]["geometry"]["commands"][1]["id"] == "r1.p1.c1"
    assert drawing_from_region_samples("drawing", width=48, height=24, regions=result).region("r1-2").geometry.path.commands[1].id == "r1.p1.c1"
    plan = parse_plan({"version": "vectoreyes.plan.v1", "drawing_id": "drawing", "base_version": "v0",
                      "ops": [{"op": "update_geometry", "target": "r1-2", "ops": [
                          {"op": "align", "target": "r1.p1.c1", "reference": "r1.p0.c1", "axes": ["y"]},
                      ]}]})
    refined = drawing_refine.refine(_trace(), result, plan)
    assert refined[0].children[0].current == result[0].children[0].current
    assert _canonical_trace(_export_drawing("export", width=(_trace()).width, height=(_trace()).height, regions=_export_addresses(refined)), _trace_options((_trace()).options))["regions"][1]["geometry"]["commands"][1]["id"] == "r1.p1.c1"


def test_hole_first_prefit_corners_remain_owned_by_original_paths():
    from vectoreyes.corner_diagnostics import path_corner_diagnostics
    from vectoreyes.drawing_adapter import drawing_from_region_samples
    d = f"{HOLE} {OUTER} M20 0 L24 0 L24 4 L20 4 Z"
    evidence = path_corner_diagnostics(d, declared_corners=((4, 4), (0, 0), (20, 0)))
    source = _region(d).with_diagnostics({"corners": evidence})
    result = normalize_working_topology((source,))
    drawing = drawing_from_region_samples("drawing", width=48, height=24, regions=result)
    geometry = drawing.region("r1-2").geometry
    assert geometry.outer.path.id == "r1.p1"
    assert geometry.outer.anchors[0].entry_command_id.startswith("r1.p1.")
    assert geometry.cutouts[0].anchors[0].entry_command_id.startswith("r1.p0.")
    assert drawing.region("r1-3").geometry.anchors[0].entry_command_id.startswith("r1.p2.")
    assert source.diagnostics["corners"] == evidence


def test_nested_root_producer_diagnostics_keep_declarations_on_component_children():
    from vectoreyes.corner_diagnostics import path_corner_diagnostics
    from vectoreyes.drawing_adapter import drawing_from_region_samples
    parts = (HOLE, OUTER, "M20 0 L24 0 L24 4 L20 4 Z")
    evidence = {"path": " ".join(parts), "subpaths": tuple(
        path_corner_diagnostics(d, declared_corners=(point,))
        for d, point in zip(parts, ((4, 4), (0, 0), (20, 0)), strict=True)
    )}
    result = normalize_working_topology((_region(evidence["path"]).with_diagnostics({"corners": evidence}),))
    drawing = drawing_from_region_samples("drawing", width=48, height=24, regions=result)
    assert drawing.region("r1-2").geometry.outer.anchors[0].entry_command_id == "r1.p1.c0"
    assert drawing.region("r1-2").geometry.cutouts[0].anchors[0].entry_command_id == "r1.p0.c0"
    assert drawing.region("r1-3").geometry.anchors[0].entry_command_id == "r1.p2.c0"


def test_redundant_nonzero_paths_and_features_are_explicitly_retired():
    from vectoreyes.corner_diagnostics import path_corner_diagnostics
    d = f"{HOLE} {OUTER}"
    source = _region(d, fill_rule="nonzero").with_diagnostics({"corners": path_corner_diagnostics(d, declared_corners=((4, 4),))})
    result = normalize_working_topology((source,))[0]
    assert result.current.params["path_ids"] == ("r1.p1",)
    assert result.diagnostics["topology"]["retired_path_ids"] == ("r1.p0",)
    assert result.diagnostics["topology"]["retired_feature_ids"]
    assert result.original.params["d"] == d


def test_root_publication_redistributes_actual_prefit_evidence_to_components():
    from dataclasses import replace
    from vectoreyes.drawing_trace import TraceRegion, TracePath
    contours = tuple(np.array(points, dtype=float) for points in (
        [(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)],
        [(12, 0), (20, 0), (20, 8), (12, 8), (12, 0)],
    ))
    trace = replace(_trace(), regions=(TraceRegion(
        id="r1", source_label=1, color="#123456", mask=_region(DISCONNECTED).raster,
        contours=contours, trace_path=TracePath(DISCONNECTED, "evenodd", ()),
        effective_trace_level="pixel",
    ),))
    from vectoreyes.refinement.assembly_binding import assemble_working_roots
    result = assemble_working_roots(trace)
    assert result.regions[0].is_branch
    records = result.trace.root_path_evidence
    assert [(record.region_id, record.path_id) for record in records] == [("r1-1", "r1.p0"), ("r1-2", "r1.p1")]
    for record, child, contour in zip(records, result.regions[0].children, contours, strict=True):
        np.testing.assert_array_equal([command.end for command in record.fit.source.path.commands], contour)
        assert record.fit.matches_output(child.current.params["d"])
        assert record.fit.correspondence.spans
    assert trace.root_path_evidence == ()


def test_auto_callback_and_stitch_publish_identical_target_trees():
    from vectoreyes.refinement_settings import RefinementPasses
    source = _region(DISCONNECTED)
    callbacks = []
    passes = RefinementPasses(**{name: name == 'split_compound' for name in RefinementPasses.model_fields})
    result = drawing_refine.auto_refine(_trace(), (source,), refinement_passes=passes,
        on_pass=lambda name, regions: callbacks.append(regions))
    assert callbacks
    for regions in (*callbacks, result, drawing_refine.stitch_regions(_trace(), (source,))):
        assert tuple(leaf.drawing_id for root in regions for leaf in root.leaves()) == ("r1-1", "r1-2")
        assert {target["id"] for target in _canonical_render(_export_drawing("export", width=(_trace()).width, height=(_trace()).height, regions=_export_addresses(regions))).report["targets"]} == {"r1-1", "r1-2"}
        assert regions[0].original == source.original


def test_empty_plan_with_rgb_preserves_shared_gradient_without_refilling(monkeypatch):
    from vectoreyes.candidate import LinearGradientFill
    from tests._render import render_svg
    fill = LinearGradientFill({"x1": 0, "y1": 0, "x2": 20, "y2": 0}, [(0, "#ff0000"), (1, "#0000ff")])
    source = _region(DISCONNECTED).with_current(_region(DISCONNECTED).current, fill=fill)
    before = (f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="24">'
              '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="20" y2="0">'
              '<stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#0000ff"/>'
              f'</linearGradient></defs><path d="{DISCONNECTED}" fill="url(#g)"/></svg>')
    def forbidden(*args, **kwargs):
        raise AssertionError("pure topology normalization refitted paint or geometry")
    import vectoreyes.drawing_assembly as assembly
    monkeypatch.setattr(assembly, "fit_fill", forbidden)
    import vectoreyes.paint_fitting as paint_fitting
    monkeypatch.setattr(paint_fitting, "fit_fill", forbidden)
    monkeypatch.setattr(paint_fitting, "fit_source_paint", forbidden)
    import vectoreyes.path_fitting as fitting
    monkeypatch.setattr(fitting, "fit_raster_paths", forbidden)
    plan = parse_plan({"version": "vectoreyes.plan.v1", "drawing_id": "drawing", "base_version": "v0", "ops": []})
    result = drawing_refine.refine(_trace(), (source,), plan, rgb=np.zeros((24, 48, 3), np.uint8))
    assert all(child.fill is fill for child in result[0].children)
    np.testing.assert_array_equal(render_svg(before, 48, 24), render_svg(_canonical_render(_export_drawing("export", width=(_trace()).width, height=(_trace()).height, regions=_export_addresses(result))).svg, 48, 24))


def test_component_holes_remain_transparent_on_contrasting_backgrounds():
    from tests._render import render_svg
    import xml.etree.ElementTree as ET
    source = _region(f"{HOLE} {OUTER} M20 0 L24 0 L24 4 L20 4 Z")
    result = normalize_working_topology((source,))
    svg = _canonical_render(_export_drawing("export", width=(_trace()).width, height=(_trace()).height, regions=_export_addresses(result))).svg
    for color, pixel in (("#123456", (18, 52, 86)), ("#ef7090", (239, 112, 144))):
        root = ET.fromstring(svg)
        root.insert(0, ET.Element("{http://www.w3.org/2000/svg}rect", {"width": "48", "height": "24", "fill": color}))
        np.testing.assert_array_equal(render_svg(ET.tostring(root, encoding="unicode"), 48, 24)[6, 6], pixel)


def test_snapshot_preserves_branch_original_and_detaches_its_params():
    from vectoreyes.drawing_engine import RasterImage
    from tests._working_evidence import capture_working_geometry_evidence
    from vectoreyes.refinement.assembly_binding import working_regions_from_evidence
    source = _region(DISCONNECTED)
    normalized = normalize_working_topology((source,))
    evidence = capture_working_geometry_evidence(
        kind="source",
        trace=_trace(),
        geometry_source=RasterImage(48, 24, "rgb8", bytes(48 * 24 * 3)),
        trace_roots=normalized,
    )
    retained = working_regions_from_evidence(evidence)
    source.original.params["d"] = OUTER
    assert retained[0].original.params["d"] == DISCONNECTED
    assert all(child.original.params["d"] == DISCONNECTED for child in retained[0].children)


def test_retained_component_reports_recount_corner_and_command_aggregates():
    from vectoreyes.corner_diagnostics import path_corner_diagnostics
    evidence = path_corner_diagnostics(DISCONNECTED, declared_corners=((0, 0), (12, 0)))
    source = _region(DISCONNECTED).with_diagnostics({"corners": evidence})
    result = normalize_working_topology((source,))
    targets = _canonical_render(_export_drawing("export", width=(_trace()).width, height=(_trace()).height, regions=_export_addresses(result))).report["targets"]
    for target in targets:
        corners = target["diagnostics"]["corners"]
        assert corners["commands"] == {"total": 3, "corner": 0, "free": 3}
        assert corners["corner_counts"] == {"total": 1, "point": 1, "curve": 0}
        assert corners["anchor_counts"] == {"total": 4, "corner": 1}
    assert source.diagnostics["corners"]["commands"]["total"] == 6

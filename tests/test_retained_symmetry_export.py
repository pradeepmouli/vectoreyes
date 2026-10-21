from vectormark.drawing_adapter import drawing_from_region_samples as _export_drawing, normalize_drawing_addresses as _export_addresses
from vectormark.drawing_render import render_drawing as _canonical_render
from types import SimpleNamespace
from typing import cast
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from vectormark import drawing_refine
from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.refinement.automatic_replay import ProposalIntent
from vectormark.refinement.region_sample import RegionSample, to_polygon


def _forest():
    # Two halves of a frame, with a cutout straddling the axis x=10.
    left = Shape("path", {"d": "M2 2 L10 2 L10 6 L6 6 L6 14 L10 14 L10 18 L2 18 Z"})
    right = Shape("path", {"d": "M10 2 L18 2 L18 18 L10 18 L10 14 L14 14 L14 6 L10 6 Z"})
    mask = np.zeros((24, 48), dtype=bool)
    source = RegionSample(11, left, FlatFill("#123456"), raster=mask,
                          drawing_id="r1-1", source_regions=("r1",))
    mirror = RegionSample(12, Shape("use", {"href_obj_id": 11, "transform": (-1, 0, 0, 1, 20, 0)}),
                          source.fill, raster=mask, footprint=to_polygon(right),
                          drawing_id="r1-2", source_regions=("r1",),
                          diagnostics={"symmetry": {"mode": "self_mirror"}})
    branch = RegionSample.branch(id=1, children=(source, mirror), raster=mask,
                                 drawing_id="r1", source_regions=("r1",),
                                 diagnostics={"symmetry": {"mode": "self"}})
    clone = RegionSample(2, Shape("use", {"href_obj_id": 11, "transform": (1, 0, 0, 1, 22, 0)}),
                         FlatFill("#fedcba"), raster=mask, footprint=source.footprint,
                         drawing_id="r2", source_regions=("r2",), diagnostics={"clone": {"source": "r1-1"}})
    return branch, clone


def test_materialization_retains_full_union_hole_identity_and_clone():
    from vectormark.refinement.symmetry_materialize import materialize_self_symmetry
    before = _forest()
    result = materialize_self_symmetry(before)
    joined, clone = result
    assert joined.is_leaf and joined.id == 1
    assert joined.footprint.bounds == (2.0, 2.0, 18.0, 18.0)
    assert joined.footprint.area == pytest.approx(192)
    assert joined.current.params["fill_rule"] == "evenodd"
    assert to_polygon(joined.current).symmetric_difference(before[0].footprint).area < 1e-8
    assert len(to_polygon(joined.current).interiors) == 1
    # No contour contains the former coincident vertical axis edge.
    for subpath in joined.footprint.linearized_subpaths:
        for start, end in zip(subpath, (*subpath[1:], subpath[0])):
            assert not (start[0] == end[0] == 10 and start[1] != end[1])
    assert joined.source_regions == ("r1",)
    assert joined.diagnostics["symmetry"]["materialized"] is True
    assert joined.diagnostics["materialization"]["retired_region_ids"] == (11, 12)
    assert clone.current.params["href_obj_id"] == 1
    assert clone.footprint.bounds == (24.0, 2.0, 40.0, 18.0)
    assert clone.fill == before[1].fill
    assert before[0].is_branch and before[1].current.params["href_obj_id"] == 11
    repeated = materialize_self_symmetry(result)
    assert all(a is b for a, b in zip(result, repeated))
    from vectormark.drawing_trace import TraceOptions, TraceResult
    from vectormark.drawing_plan import parse_plan, PlanValidationError
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
                       "ops": [{"op": "set_fill", "target": "r1-2", "fill": {"type": "flat", "color": "#ffffff"}}]})
    with pytest.raises(PlanValidationError):
        drawing_refine.refine(TraceResult(48, 24, TraceOptions(), (), ""), result, plan)


def test_render_is_geometry_and_raster_free_and_reports_every_painted_target(monkeypatch):
    from vectormark.refinement.symmetry_materialize import materialize_self_symmetry
    regions = materialize_self_symmetry(_forest())
    def forbidden(*args, **kwargs):
        raise AssertionError("renderer reconstructed geometry")
    import vectormark.fit as fit_module
    import vectormark.path_fitting as fitting
    monkeypatch.setattr(fit_module, "atomic_flatten_path_d", forbidden)
    monkeypatch.setattr(fitting, "fit_raster_paths", forbidden)
    import vectormark.drawing_assembly as assembly
    monkeypatch.setattr(assembly, "unary_union", forbidden)
    monkeypatch.setattr(assembly, "_fit_contours_with_corners", forbidden)
    import vectormark.refinement.gate as gate
    monkeypatch.setattr(gate, "rasterize", forbidden)
    trace = SimpleNamespace(width=48, height=24)
    first = _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(regions)))
    assert first.svg == _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(regions))).svg
    root = ET.fromstring(first.svg)
    painted = {e.attrib["id"] for e in root if "fill" in e.attrib}
    assert painted == {target["id"] for target in first.report["targets"]} == {"r1", "r2"}
    definition = next(e for e in root.iter() if e.attrib.get("id") == "geometry-r1")
    assert definition.attrib["d"] == regions[0].current.params["d"]


def test_render_rejects_unmaterialized_self_symmetry():
    with pytest.raises(ValueError, match="materializ"):
        _canonical_render(_export_drawing("export", width=(SimpleNamespace(width=48, height=24)).width, height=(SimpleNamespace(width=48, height=24)).height, regions=_export_addresses(_forest())))


def test_shared_materials_have_no_hidden_surface_underlays(monkeypatch):
    regions = tuple(leaf.with_diagnostics({"surface_fill": {"shared": True, "group": 1}})
                    for leaf in _forest()[0].leaves())
    regions = (regions[0], regions[1].with_current(
        Shape("path", {"d": "M20 2 L22 2 L22 18 L20 18 Z"}), diagnostics={"symmetry": {}}))
    def forbidden(*args, **kwargs):
        raise AssertionError("renderer fitted an undisclosed surface")
    import vectormark.drawing_assembly as assembly
    monkeypatch.setattr(assembly, "_fit_contours_with_corners", forbidden)
    rendered = _canonical_render(_export_drawing("export", width=(SimpleNamespace(width=48, height=24)).width, height=(SimpleNamespace(width=48, height=24)).height, regions=_export_addresses(regions)))
    assert "surface-" not in rendered.svg
    assert len(rendered.report["targets"]) == 2


@pytest.mark.parametrize("stale_clone", [False, True])
def test_cross_subtree_clone_refresh_is_an_assessed_transaction(stale_clone, monkeypatch):
    from vectormark.refinement import framework
    from vectormark.drawing_adapter import transform_working_shape
    from vectormark.refinement.symmetry_materialize import union_self_symmetry
    branch, old_clone = _forest()
    source = RegionSample(11, Shape("path", {"d": branch.footprint.to_svg_d(), "fill_rule": "evenodd", "tag": "before"}),
                          branch.fill or branch.children[0].fill, footprint=branch.footprint,
                          raster=branch.raster, drawing_id="r1-1", source_regions=("r1",))
    root = RegionSample.branch(id=1, children=(source,), drawing_id="r1", raster=branch.raster)
    clone_footprint = to_polygon(transform_working_shape(source.current, (1, 0, 0, 1, 22, 0)))
    clone = old_clone.with_current(old_clone.current, footprint=old_clone.footprint if stale_clone else clone_footprint)
    inputs = [root, clone]
    def reconstruct(objects, masks):
        return [framework.Proposal((obj.id,), [union_self_symmetry(
            obj, (obj.footprint,), diagnostics={"mode": "self"},
        )], intent=cast(ProposalIntent, None)) for obj in objects if obj.id == 11]
    assessed = []
    real_assess = framework._replacement_preserves_scene
    def capture(replacements, sources, *, policy, region_policy=None):
        assessed.append(tuple(sorted(obj.id for obj in replacements)))
        return real_assess(replacements, sources, policy=policy, region_policy=region_policy)
    monkeypatch.setattr(framework, "_replacement_preserves_scene", capture)
    published = []
    result = framework.select_refinements(inputs, {obj.id: obj.raster for obj in inputs}, [reconstruct],
                                on_pass=lambda name, values: published.append(values))
    assert (2, 11) in assessed
    assert len(published) == 1 and published[0] == result
    after_source = result[0].children[0]
    if stale_clone:
        assert after_source is source
        assert result[1].current == clone.current
        assert result[1].footprint is clone.footprint
        assert result[0].diagnostics["reconstruct"]["reason"] == "self_symmetry_clone_transaction"
    else:
        assert after_source.diagnostics["symmetry"]["materialized"]
        assert result[1].current.params["href_obj_id"] == 11
        assert result[1].footprint.symmetric_difference(clone_footprint).area < 1e-8
        rendered = _canonical_render(_export_drawing("export", width=(SimpleNamespace(width=48, height=24)).width, height=(SimpleNamespace(width=48, height=24)).height, regions=_export_addresses(result)))
        assert 'href="#geometry-r1-1"' in rendered.svg
        from vectormark.pipeline import _render_refinement_body
        # The legacy pipeline uses the same retained identity across subtrees.
        body, defs = _render_refinement_body(result)
        assert 'id="s1-11-geometry"' in "".join(defs)
        assert 'href="#s1-11-geometry"' in "".join(body)
    assert source.current.params["tag"] == "before"


def test_automatic_callback_and_later_polish_share_retained_materialization():
    from vectormark.refinement_settings import RefinementPasses
    from vectormark.refinement.symmetry_materialize import materialize_self_symmetry
    branch, _ = _forest()
    joined = materialize_self_symmetry((branch,))[0]
    original = joined.with_current(Shape("path", {**joined.current.params, "tag": "before"}),
                                   diagnostics={"symmetry": {}})
    trace = SimpleNamespace(width=48, height=24, options=SimpleNamespace(
        simplify_tolerance=1.0, curve_tolerance=1.0, trace_level="pixel",
        max_sampling_colors=8, min_region_size=1))
    callbacks = []
    passes = RefinementPasses(**{name: name in {'symmetry', 'smooth'} for name in RefinementPasses.model_fields})
    output = drawing_refine.auto_refine(trace, (original,), refinement_passes=passes, on_pass=lambda name, regions: callbacks.append(
        (regions, _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(regions))))))
    assert len(callbacks) == 2  # Disabled corner normalization is not executed.
    assert all(regions[0].is_leaf and regions[0].footprint.area == pytest.approx(192)
        and regions[0].diagnostics['symmetry']['mode'] == 'intrinsic'
        and regions[0].diagnostics['materialization']['operation'] == 'self_symmetry_union'
        for regions, _ in callbacks)
    assert all(len(render.report["targets"]) == 1 for _, render in callbacks)
    assert all(output[0].current == regions[0].current for regions, _ in callbacks)


def test_repeated_singleton_union_preserves_counter_and_detached_ownership():
    from vectormark.skia_geometry import SkPath, unary_union
    original = _forest()[0].footprint
    copies = [original]
    for _ in range(4):
        copies.append(unary_union([SkPath(), copies[-1], SkPath()]))
    assert all(path.area == pytest.approx(192) and len(path.interiors) == 1 for path in copies)
    assert all(path.to_svg_d() == original.to_svg_d() for path in copies)
    assert len({id(path) for path in copies}) == len(copies)
    copies[-1]._path.reset()
    assert copies[-1].is_empty
    assert all(path.area == pytest.approx(192) for path in copies[:-1])


def test_manual_symmetry_uses_shared_union_and_rejects_outside_axis_without_mutation():
    from vectormark.drawing_trace import TraceOptions, TraceResult
    from vectormark.drawing_plan import parse_plan
    shape = Shape("path", {"d": "M36 8 L56 8 L56 24 L36 24 Z"})
    source = RegionSample(2, shape, FlatFill("#123456"), raster=np.zeros((32, 64), bool),
                          drawing_id="r2", source_regions=("r2",))
    trace = TraceResult(64, 32, TraceOptions(), (), "")
    def plan(x):
        return parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
                           "ops": [{"op": "set_symmetry", "target": "r2",
                                    "axis": {"point": [x, 16], "angle_degrees": 90}}]})
    output = drawing_refine.refine(trace, (source,), plan(46))
    assert output[0].diagnostics["materialization"]["operation"] == "self_symmetry_union"
    assert output[0].diagnostics["symmetry"]["materialized"]
    assert output[0].footprint.symmetric_difference(source.footprint).area < 1e-8
    with pytest.raises(ValueError, match="empty|divide"):
        drawing_refine.refine(trace, (source,), plan(30))
    assert source.current is shape and not source.diagnostics


def test_pipeline_renderer_requires_retained_geometry_and_cannot_fit(monkeypatch):
    from vectormark import fit, skia_geometry
    from vectormark import drawing_simplification
    from vectormark.refinement.symmetry_materialize import materialize_self_symmetry
    from vectormark.pipeline import _render_refinement_body
    before = _forest()
    retained = materialize_self_symmetry(before)
    def forbidden(*args, **kwargs):
        raise AssertionError("pipeline renderer reconstructed geometry")
    monkeypatch.setattr(skia_geometry, "unary_union", forbidden)
    monkeypatch.setattr(fit, "atomic_flatten_path_d", forbidden)
    monkeypatch.setattr(drawing_simplification, "simplify_paths", forbidden)
    first = _render_refinement_body(retained)
    assert first == _render_refinement_body(retained)
    with pytest.raises(ValueError, match="materializ"):
        _render_refinement_body(before)


def test_top_level_symmetry_source_assesses_nested_clone_leaf_not_parent_exterior(monkeypatch):
    from vectormark.refinement import framework
    from vectormark.refinement.symmetry_materialize import materialize_self_symmetry, remap_symmetry_clones
    branch, clone = _forest()
    joined = materialize_self_symmetry((branch,))[0]
    source = joined.with_current(Shape("path", {**joined.current.params, "tag": "before"}),
                                 diagnostics={"symmetry": {}})
    clone = clone.with_current(Shape("use", {**clone.current.params, "href_obj_id": 1}), footprint=clone.footprint)
    # This exterior intentionally equals the eventual full clone, so measuring
    # it instead of the current stale leaf would wrongly accept the change.
    future_clone = remap_symmetry_clones((joined, clone), {(1,): (1,)}, before=(source, clone))[1]
    clone_root = RegionSample.branch(id=20, children=(clone,), footprint=future_clone.footprint,
                                     raster=clone.raster, drawing_id="r2")
    def symmetric(objects, masks):
        if not any(obj.id == 1 for obj in objects):
            return []
        remapped_root = remap_symmetry_clones((joined, clone_root), {(1,): (1,)}, before=(source, clone_root))[1]
        return [framework.Proposal(
            (1, 20), [joined, remapped_root], intent=cast(ProposalIntent, None)
        )]
    result = framework.select_refinements([source, clone_root], {1: source.raster, 20: clone.raster}, [symmetric])
    assert result[0].current == source.current
    assert result[1].children[0] is clone
    assert result[0].diagnostics["symmetric"]["accepted"] is False


def test_real_recursive_compound_producer_preserves_colliding_local_ids_at_export():
    from vectormark.refinement.framework import select_refinements
    from vectormark.refinement.passes.compound import split_compound_pass
    from vectormark.pipeline import _render_refinement_body
    mask = np.zeros((40, 40), dtype=bool)
    compound = RegionSample(1, Shape("path", {"d": "M0 0 L8 0 L8 8 L0 8 Z M12 0 L20 0 L20 8 L12 8 Z", "fill_rule": "evenodd"}),
                            FlatFill("#123456"), raster=mask)
    branch = RegionSample.branch(id=10, children=(compound,), raster=mask)
    unrelated = RegionSample(2, Shape("rect", {"x": 0, "y": 20, "w": 8, "h": 8}), FlatFill("#654321"), raster=mask)
    out = select_refinements([branch, unrelated], {10: mask, 2: mask}, [split_compound_pass])
    assert sorted(leaf.id for root in out for leaf in root.leaves()) == [1, 2, 2]
    body, _ = _render_refinement_body(out)
    ids = {Element.attrib["id"] for Element in ET.fromstring("<svg>" + "".join(body) + "</svg>")}
    assert ids == {"s2", "s10-1-1", "s10-1-2"}


def test_clone_local_source_shadows_unrelated_cross_scope_materialization():
    from vectormark.refinement.symmetry_materialize import remap_symmetry_clones, union_self_symmetry
    mask = np.zeros((40, 40), dtype=bool)
    outer = RegionSample(1, Shape("rect", {"x": 0, "y": 0, "w": 8, "h": 8}), FlatFill("#123456"), raster=mask)
    local = RegionSample(1, Shape("rect", {"x": 12, "y": 0, "w": 8, "h": 8}), FlatFill("#123456"), raster=mask)
    clone = RegionSample(2, Shape("use", {"href_obj_id": 1, "transform": (1, 0, 0, 1, 0, 12)}),
                         FlatFill("#654321"), footprint=local.footprint, raster=mask, diagnostics={"clones": {"matched_source": 1}})
    branch = RegionSample.branch(id=10, children=(local, clone), raster=mask)
    changed = union_self_symmetry(outer, (outer.footprint,), diagnostics={"mode": "self"})
    out = remap_symmetry_clones((changed, branch), {(1,): (1,)}, before=(outer, branch))
    assert out[1] is branch and out[1].children[1] is clone
    from vectormark.pipeline import _render_refinement_body
    body, defs = _render_refinement_body(out)
    assert 'href="#s10-1-geometry"' in "".join(body)
    assert 'id="s10-1-geometry"' in "".join(defs)
    assert 'href="#s1-geometry"' not in "".join(body)


def test_automatic_vector_halves_do_not_retire_live_clone_identity(monkeypatch):
    import importlib
    symmetry = importlib.import_module("vectormark.refinement.passes.symmetry")
    detection = importlib.import_module("vectormark.symmetry_detection")
    from vectormark.refinement.framework import select_refinements
    from vectormark.drawing_adapter import transform_working_shape
    mask = np.zeros((32, 48), dtype=bool)
    shape = Shape("path", {"d": "M2 2 L18 2 L18 18 L2 18 Z"})
    source = RegionSample(1, shape, FlatFill("#123456"), raster=mask, drawing_id="r1", source_regions=("r1",))
    matrix = (1, 0, 0, 1, 22, 0)
    clone = RegionSample(2, Shape("use", {"href_obj_id": 1, "transform": matrix}), FlatFill("#654321"),
                           footprint=to_polygon(transform_working_shape(shape, matrix)), raster=mask,
                         drawing_id="r2", source_regions=("r2",), diagnostics={"clones": {"matched_source": 1}})
    half_shape = Shape("path", {"d": "M2 2 L10 2 L10 18 L2 18 Z"})
    half = to_polygon(half_shape)
    mirror = to_polygon(Shape("path", {"d": "M10 2 L18 2 L18 18 L10 18 Z"}))
    monkeypatch.setattr(detection, "_best_self_reconstruction", lambda *args, **kwargs:
                        detection._SelfReconstruction(
                            symmetry.Axis2D(np.pi / 2, 10, 10),
                            half,
                            mirror,
                            half_shape,
                            0.0,
                            "path_half",
                            "left",
                            0.0,
                        ))
    out = select_refinements([source, clone], {1: mask, 2: mask}, [symmetry.symmetry_pass])
    assert [region.id for region in out] == [1, 2]
    history = out[0].diagnostics["materialization"]
    assert history["consumed_region_ids"] == (1,)
    assert history["retired_region_ids"] == ()
    assert history["source_region_remap"] == ((1, 1),)
    assert history["consumed_target_ids"] == ("r1",)
    assert out[1].current.kind == "use" and out[1].current.params["href_obj_id"] == 1


def test_real_branch_collapse_pins_nested_clone_against_local_id_shadowing():
    from vectormark.refinement.symmetry_materialize import materialize_self_symmetry
    from vectormark.drawing_adapter import drawing_from_region_samples
    from vectormark.drawing_trace import TraceOptions, TraceResult
    from vectormark.drawing_plan import parse_plan
    from vectormark.pipeline import _render_refinement_body
    branch, clone = _forest()
    local = RegionSample(1, Shape("rect", {"x": 0, "y": 20, "w": 4, "h": 4}), FlatFill("#aabbcc"),
                         raster=branch.raster, drawing_id="r3-1", source_regions=("r3",))
    host = RegionSample.branch(id=20, children=(local, clone), drawing_id="r3", raster=branch.raster)
    before = (branch, host)
    out = materialize_self_symmetry(before)
    retained_clone = out[1].children[1]
    assert retained_clone.current.params["href_obj_id"] == 1
    assert retained_clone.current.params["href_region_path"] == ("..", 1)
    assert retained_clone.footprint.bounds == (24.0, 2.0, 40.0, 18.0)
    assert out[1].children[0] is local
    assert out[0].diagnostics["materialization"]["retired_region_ids"] == (11, 12)
    assert out[0].diagnostics["materialization"]["source_region_path_remap"] == (((1, 11), (1,)), ((1, 12), (1,)))
    body, defs = _render_refinement_body(out)
    assert 'href="#s1-geometry"' in "".join(body)
    assert 'id="s1-geometry"' in "".join(defs)
    trace = TraceResult(48, 24, TraceOptions(), (), "")
    rendered = _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(out)))
    assert 'href="#geometry-r1"' in rendered.svg
    canonical = drawing_from_region_samples("drw_x", width=48, height=24, regions=out)
    assert canonical.regions[1].children[1].geometry.source_region_id == "r1"
    plan = parse_plan({"version": "vectormark.plan.v1", "drawing_id": "drw_x", "base_version": "v0",
                       "ops": [{"op": "set_fill", "target": "r3-2", "fill": {"type": "flat", "color": "#ffffff"}}]})
    refined = drawing_refine.refine(trace, out, plan)
    assert 'href="#geometry-r1"' in _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(refined))).svg
    assert before[0].is_branch and before[1].children[1] is clone
    assert "href_region_path" not in clone.current.params


@pytest.mark.parametrize("bad_path", [("..", "..", 11), ("bad", 11), (1, "..", 11), (True,), (), (123, 11)])
def test_final_export_rejects_invalid_explicit_paths_without_input_mutation(bad_path):
    from vectormark.pipeline import _render_refinement_body
    branch, clone = _forest()
    source = branch.children[0]
    # Use a valid local source ID so no bare-reference fallback may hide the
    # invalid explicit path or silently bind some other retained geometry.
    bad = clone.with_current(Shape("use", {**clone.current.params, "href_obj_id": source.id,
                                          "href_region_path": bad_path}), footprint=clone.footprint)
    with pytest.raises(ValueError, match="reference|source"):
        _render_refinement_body((source, bad))
    assert bad.current.params["href_region_path"] == bad_path
    assert "href_region_path" not in clone.current.params


def test_ambiguous_external_reference_is_rejected_not_arbitrarily_selected():
    from vectormark.pipeline import _render_refinement_body
    from vectormark.refinement.symmetry_materialize import remap_symmetry_clones
    branch, clone = _forest()
    source = branch.children[0]
    first = RegionSample.branch(id=10, children=(source,), raster=source.raster)
    second = RegionSample.branch(id=20, children=(source.with_current(source.current),), raster=source.raster)
    before = (first, second, clone)
    with pytest.raises(ValueError, match="ambiguous"):
        _render_refinement_body(before)
    with pytest.raises(ValueError, match="ambiguous"):
        remap_symmetry_clones(before, {(10, 11): (10, 11)}, before=before)
    assert before[0].children[0] is source and before[2] is clone


def test_recursive_transaction_does_not_update_shadowed_clone_in_other_subtree():
    from vectormark.refinement.framework import Proposal, select_refinements
    from vectormark.refinement.symmetry_materialize import union_self_symmetry
    mask = np.zeros((40, 40), dtype=bool)
    source = RegionSample(1, Shape("rect", {"x": 0, "y": 0, "w": 8, "h": 8, "reconstruct": True}),
                          FlatFill("#123456"), raster=mask)
    local = RegionSample(1, Shape("rect", {"x": 12, "y": 0, "w": 8, "h": 8}), FlatFill("#123456"), raster=mask)
    clone = RegionSample(2, Shape("use", {"href_obj_id": 1, "transform": (1, 0, 0, 1, 0, 12)}),
                         FlatFill("#654321"), footprint=local.footprint, raster=mask, diagnostics={"clones": {"matched_source": 1}})
    first = RegionSample.branch(id=10, children=(source,), raster=mask)
    second = RegionSample.branch(id=20, children=(local, clone), raster=mask)
    def reconstruct(objects, masks):
        return [Proposal(
            (obj.id,),
            [union_self_symmetry(obj, (obj.footprint,), diagnostics={"mode": "self"})],
            intent=cast(ProposalIntent, None),
        )
                for obj in objects if obj.current is not None and obj.current.params.get("reconstruct")]
    out = select_refinements([first, second], {10: mask, 20: mask}, [reconstruct])
    assert out[0].children[0].diagnostics["symmetry"]["materialized"]
    assert out[1] is second and out[1].children[1] is clone
    assert first.children[0] is source


def test_remap_binds_before_new_local_id_can_shadow_original_source():
    from vectormark.refinement.symmetry_materialize import materialize_self_symmetry, remap_symmetry_clones
    branch, clone = _forest()
    joined = materialize_self_symmetry((branch,))[0]
    clone = clone.with_current(Shape("use", {**clone.current.params, "href_obj_id": 1}), footprint=clone.footprint)
    host = RegionSample.branch(id=20, children=(clone,), raster=clone.raster)
    newcomer = RegionSample(1, Shape("rect", {"x": 0, "y": 20, "w": 4, "h": 4}), FlatFill("#aabbcc"), raster=clone.raster)
    candidate = (joined, host.with_children((newcomer, clone)))
    out = remap_symmetry_clones(candidate, {(1,): (1,)}, before=(joined, host))
    assert out[1].children[1].current.params["href_region_path"] == ("..", 1)
    assert out[1].children[1].footprint.bounds == (24.0, 2.0, 40.0, 18.0)
    assert out[1].children[0] is newcomer
    assert host.children == (clone,) and "href_region_path" not in clone.current.params

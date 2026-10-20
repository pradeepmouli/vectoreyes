import math

import numpy as np
import pytest

from vectormark.skia_geometry import SkPath
from vectormark.candidate import FlatFill, LinearGradientFill
from vectormark.refinement.framework import select_refinements
from vectormark.refinement.gate import rasterize
from vectormark.refinement.region_sample import RegionSample, to_polygon
from vectormark.refinement.passes.symmetry import symmetry_pass
from vectormark.fit import Shape
from vectormark.drawing_actions import SetSymmetryAction
from vectormark.drawing_symmetry import (
    Axis2D,
    _intrinsic_symmetry_diagnostics,
    _self_symmetry_diagnostics,
)


@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, -30.0])
@pytest.mark.parametrize("reporter,mode", [
    (_self_symmetry_diagnostics, "self"),
    (_intrinsic_symmetry_diagnostics, "intrinsic"),
])
def test_detected_axis_is_an_explicit_action_value(angle, reporter, mode):
    detected = Axis2D(math.radians(angle), 12.5, -3.25)
    report = reporter(detected, 0.001)["symmetry"]
    action = SetSymmetryAction.model_validate({
        "op": "set_symmetry", "target": "r1", "axis": report["axis"],
    })

    assert report["mode"] == mode
    assert action.axis.point == (12.5, -3.25)
    assert action.axis.angle_degrees == pytest.approx(angle)
    assert action.model_dump(mode="json")["axis"] == report["axis"]
    assert report["residual"] == 0.001


def _mask_for_polygon(poly: SkPath, shape_hw: tuple[int, int] = (96, 96)) -> np.ndarray:
    return rasterize(poly, shape_hw)


def _path_from_poly(poly: SkPath) -> Shape:
    coords = list(poly.exterior.coords)
    body = " ".join(f"L{x} {y}" for x, y in coords[1:])
    return Shape("path", {"d": f"M{coords[0][0]} {coords[0][1]} {body} Z"})


def _obj(
    obj_id: int,
    poly: SkPath,
    *,
    fill: object = FlatFill("#112233"),
    z: int = 0,
) -> RegionSample:
    return RegionSample(obj_id, _path_from_poly(poly), fill, z, poly)


def _rect(
    obj_id: int,
    *,
    bounds: tuple[float, float, float, float],
    fill: object = FlatFill("#112233"),
) -> RegionSample:
    minx, miny, maxx, maxy = bounds
    poly = SkPath(shell=[(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    return _obj(obj_id, poly, fill=fill)


def test_symmetry_pass_replaces_mirror_pair_with_baked_path():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#112233"))
    objects = [right, left]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    proposals = symmetry_pass(objects, masks)
    out = select_refinements(objects, masks, [symmetry_pass])

    assert proposals[0].obj_ids == (1, 2)
    assert [obj.id for obj in proposals[0].new_objects] == [1, 2]
    assert [obj.id for obj in out] == [1, 2]
    assert out[0].current.kind == "path"
    assert out[1].current.kind == "path"
    assert out[1].footprint.symmetric_difference(right.footprint).area < 1e-6
    assert out[1].diagnostics["symmetry"]["mode"] == "pair"
    assert out[1].diagnostics["symmetry"]["matched_source"] == 1


def test_symmetry_pass_reconstructs_self_symmetric_object_as_single_region():
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    symmetric_contour = SkPath(shell=
        [(48.0 + 20.0 * np.cos(theta), 48.0 + 30.0 * np.sin(theta)) for theta in angles]
    )
    obj = _obj(1, symmetric_contour)
    masks = {obj.id: _mask_for_polygon(obj.footprint)}

    out = select_refinements([obj], masks, [symmetry_pass])

    assert len(out) == 1
    region = out[0]
    assert region.id == 1
    assert region.is_leaf
    assert region.current.kind == "path"
    assert region.diagnostics["symmetry"]["materialized"] is True
    assert region.diagnostics["symmetry"]["mode"] == "self"


def test_symmetry_pass_does_not_auto_reconstruct_a_material_surface_leaf():
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    symmetric_contour = SkPath(shell=
        [(48.0 + 20.0 * np.cos(theta), 48.0 + 30.0 * np.sin(theta)) for theta in angles]
    )
    obj = _obj(1, symmetric_contour).with_diagnostics(
        diagnostics={"material_surface": {"root": "g1"}}
    )
    masks = {obj.id: _mask_for_polygon(obj.footprint)}

    assert symmetry_pass([obj], masks) == []
    out = select_refinements([obj], masks, [symmetry_pass])
    assert out == [obj]


def test_symmetry_pass_accepts_a_valid_specialized_self_fit_before_generic_cubic_fallback():
    shape = Shape(
        "path",
        {
            "d": (
                "M390 302.5 L256 302.5 "
                "Q245.79 281.27 240.5 258 "
                "Q241.25 257.25 242 256.5 "
                "L404 256.5 "
                "Q404.75 257.25 405.5 258 "
                "Q400.2 281.26 390 302.5 Z"
            )
        },
    )
    obj = RegionSample(1, shape, FlatFill("#fd8e27"), 0, to_polygon(shape))

    out = select_refinements(
        [obj],
        {obj.id: _mask_for_polygon(obj.footprint, (340, 460))},
        [lambda objects, masks: symmetry_pass(objects, masks, epsilon=1.0, max_error=1.0, cubic=True)],
    )

    assert out[0].is_leaf and out[0].diagnostics["symmetry"]["materialized"]


def test_symmetry_pass_selects_accurate_generic_path_fallback_over_heuristics():
    shape = Shape(
        "path",
        {
            "d": (
                "M322.92 256.5 "
                "L400.35 256.5 "
                "C403.61 256.5 405.46 259.02 404.49 262.13 "
                "L393.68 296.87 "
                "C392.71 299.98 389.29 302.5 386.03 302.5 "
                "L322.92 302.5 "
                "L259.81 302.5 "
                "C256.55 302.5 253.13 299.98 252.16 296.87 "
                "L241.34 262.13 "
                "C240.38 259.02 242.23 256.5 245.49 256.5 "
                "L322.92 256.5 Z"
            )
        },
    )
    obj = RegionSample(1, shape, FlatFill("#fd8e27"), 0, to_polygon(shape))

    out = select_refinements(
        [obj],
        {obj.id: _mask_for_polygon(obj.footprint, (340, 460))},
        [lambda objects, masks: symmetry_pass(objects, masks, epsilon=1.0, max_error=1.0, cubic=True)],
    )

    assert out[0].is_leaf and out[0].diagnostics["symmetry"]["materialized"]


def test_symmetry_pass_smooths_daikonic_like_cap_instead_of_preserving_facets():
    shape = Shape(
        "path",
        {
            "d": (
                "M323.08 118.6 "
                "C314.99 119.45 306.15 120.31 297.31 121.17 "
                "C288.54 123.94 279.77 126.72 271 129.5 "
                "L253.5 144 L242.5 162 L237.5 179 "
                "Q237.75 181.25 238 183.5 "
                "L323.08 183.5 L408.17 183.5 "
                "Q408.42 181.25 408.67 179 "
                "L403.67 162 L392.67 144 L375.17 129.5 "
                "C366.4 126.72 357.63 123.94 348.86 121.17 "
                "C340.02 120.31 331.18 119.45 323.08 118.6 Z"
            )
        },
    )
    obj = RegionSample(1, shape, FlatFill("#062337"), 0, to_polygon(shape))

    out = select_refinements(
        [obj],
        {obj.id: _mask_for_polygon(obj.footprint, (240, 480))},
        [lambda objects, masks: symmetry_pass(objects, masks, epsilon=1.0, max_error=1.0, cubic=True)],
    )

    assert out[0].is_leaf and out[0].diagnostics["symmetry"]["materialized"]
    d = str(out[0].current.params["d"])
    # The retained path now contains both fitted halves, not one source half.
    assert d.count("C") == 2
    assert d.count("L") == 2
    assert "Q" not in d


def test_symmetry_pass_preserves_flat_top_width_on_daikonic_like_tip():
    shape = Shape(
        "path",
        {
            "d": (
                "M322.92 314.5 "
                "L384.5 315 "
                "Q367.72 352.4 333 379.5 "
                "Q328.43 382.04 322.92 382.5 "
                "Q317.41 382.04 312.84 379.5 "
                "Q278.12 352.4 261.34 315 "
                "L322.92 314.5 Z"
            )
        },
    )
    obj = RegionSample(1, shape, FlatFill("#f23325"), 0, to_polygon(shape))

    out = select_refinements(
        [obj],
        {obj.id: _mask_for_polygon(obj.footprint, (420, 420))},
        [lambda objects, masks: symmetry_pass(objects, masks, epsilon=1.0, max_error=1.0, cubic=True)],
    )

    assert out[0].is_leaf and out[0].diagnostics["symmetry"]["materialized"]
    source = out[0]
    minx, _miny, maxx, _maxy = source.footprint.bounds
    d = str(source.current.params["d"])
    assert maxx - minx > 100.0
    assert "L" in d
    assert "C" not in d
    assert d.count("Q") <= 4


def test_symmetric_half_fit_fits_prefixed_half_only_once(monkeypatch):
    import vectormark.refine as refine_module

    shape = Shape(
        "path",
        {
            "d": (
                "M322.92 314.5 "
                "L384.5 315 "
                "Q367.72 352.4 333 379.5 "
                "Q328.43 382.04 322.92 382.5 "
                "Q317.41 382.04 312.84 379.5 "
                "Q278.12 352.4 261.34 315 "
                "L322.92 314.5 Z"
            )
        },
    )
    contour = np.asarray(to_polygon(shape).exterior.coords, dtype=float)
    calls: list[np.ndarray] = []
    real_fit = refine_module._fit_open_segments

    def spy_fit(points, *args, **kwargs):
        calls.append(np.asarray(points, dtype=float))
        return real_fit(points, *args, **kwargs)

    monkeypatch.setattr(refine_module, "_fit_open_segments", spy_fit)

    out = refine_module.symmetric_half_fit(
        contour,
        322.92,
        side="left",
        epsilon=1.0,
        max_error=1.0,
    )

    assert out is not None
    assert len(calls) == 1
    assert abs(float(calls[0][0][0]) - 322.92) > 1.0


def test_symmetry_pass_records_gradient_self_symmetry_without_rewriting_geometry():
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    symmetric_contour = SkPath(shell=
        [(48.0 + 20.0 * np.cos(theta), 48.0 + 30.0 * np.sin(theta)) for theta in angles]
    )
    obj = _obj(
        1,
        symmetric_contour,
        fill=LinearGradientFill(
            {"x1": 0.0, "y1": 0.0, "x2": 96.0, "y2": 96.0},
            [(0.0, "#000000"), (1.0, "#ffffff")],
        ),
    )

    out = select_refinements([obj], {obj.id: _mask_for_polygon(obj.footprint)}, [symmetry_pass])

    assert len(out) == 1
    assert out[0].is_leaf
    assert out[0].current == obj.current
    assert out[0].footprint.symmetric_difference(obj.footprint).area < 1e-6
    assert out[0].diagnostics["symmetry"]["mode"] == "intrinsic"
    assert out[0].diagnostics["symmetry"]["accepted"] is True


def test_symmetry_pass_preserves_self_symmetric_native_primitive():
    poly = SkPath(shell=[(20.0, 20.0), (60.0, 20.0), (60.0, 60.0), (20.0, 60.0)])
    obj = RegionSample(
        1,
        Shape("rect", {"x": 20.0, "y": 20.0, "w": 40.0, "h": 40.0}),
        FlatFill("#112233"),
        0,
        poly,
    )

    out = select_refinements([obj], {obj.id: _mask_for_polygon(poly)}, [symmetry_pass])

    assert len(out) == 1 and out[0].is_leaf
    assert out[0].current == obj.current
    assert out[0].fill == obj.fill
    assert out[0].footprint.symmetric_difference(poly).area < 1e-6
    assert out[0].diagnostics["symmetry"]["mode"] == "intrinsic"
    assert out[0].diagnostics["symmetry"]["accepted"] is True


def test_symmetry_pass_records_simple_self_symmetric_path_without_rewriting():
    poly = SkPath(shell=[(20.0, 60.0), (40.0, 20.0), (60.0, 60.0)])
    obj = _obj(1, poly)

    out = select_refinements([obj], {obj.id: _mask_for_polygon(poly)}, [symmetry_pass])

    assert len(out) == 1
    assert out[0].is_leaf
    assert out[0].current == obj.current
    assert out[0].fill == obj.fill
    assert out[0].footprint.symmetric_difference(poly).area < 1e-6
    assert out[0].diagnostics["symmetry"]["mode"] == "intrinsic"
    assert out[0].diagnostics["symmetry"]["accepted"] is True


def test_rejected_approximate_reconstruction_is_not_reported_as_intrinsic(monkeypatch):
    import vectormark.refinement.passes.symmetry as module

    poly = SkPath(shell=[(20.0, 60.0), (40.1, 20.0), (60.0, 60.0)])
    obj = _obj(1, poly)
    intrinsic = module._best_self_axis(poly)
    assert intrinsic is not None and intrinsic[1] > 1e-9
    monkeypatch.setattr(module, "_best_self_reconstruction", lambda *args, **kwargs: None)

    out = select_refinements([obj], {obj.id: _mask_for_polygon(poly)}, [symmetry_pass])

    assert out[0].current == obj.current
    assert "symmetry" not in out[0].diagnostics


def test_mirror_pair_axis_uses_the_shared_public_format():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0))
    objects = [left, right]
    out = select_refinements(objects, {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}, [symmetry_pass])

    axis = out[1].diagnostics["symmetry"]["axis"]
    assert set(axis) == {"point", "angle_degrees"}
    assert axis["point"] == pytest.approx([40.0, 27.0])
    assert axis["angle_degrees"] == pytest.approx(90.0)


def test_symmetry_pass_matches_non_flat_fill_pair_by_geometry():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(
        2,
        bounds=(58.0, 20.0, 70.0, 34.0),
        fill=LinearGradientFill(
            {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0},
            [(0.0, "#000000"), (1.0, "#ffffff")],
        ),
    )
    objects = [left, right]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = select_refinements(objects, masks, [symmetry_pass])

    assert [obj.current.kind for obj in out] == ["path", "path"]
    assert out[1].footprint.symmetric_difference(right.footprint).area < 1e-6
    assert out[1].diagnostics["symmetry"]["mode"] == "pair"
    assert out[1].fill == right.fill


def test_symmetry_pass_matches_recolored_pair_by_geometry():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#abcdef"))
    objects = [left, right]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = select_refinements(objects, masks, [symmetry_pass])

    assert [obj.current.kind for obj in out] == ["path", "path"]
    assert out[1].footprint.symmetric_difference(right.footprint).area < 1e-6
    assert out[1].diagnostics["symmetry"]["mode"] == "pair"
    assert out[1].fill == right.fill


def test_symmetry_pass_is_deterministic_for_unordered_inputs():
    left = _rect(3, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(9, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#112233"))
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in [left, right]}

    first = symmetry_pass([right, left], masks)
    second = symmetry_pass([left, right], masks)

    assert first == second
    assert first[0].obj_ids == (3, 9)
    assert first[0].new_objects[1].current.kind == "path"
    assert first[0].new_objects[1].footprint.symmetric_difference(right.footprint).area < 1e-6


def test_symmetry_pass_does_not_use_raster_mask_as_acceptance_gate():
    left = _rect(1, bounds=(10.0, 20.0, 22.0, 34.0), fill=FlatFill("#112233"))
    right = _rect(2, bounds=(58.0, 20.0, 70.0, 34.0), fill=FlatFill("#112233"))
    masks = {
        left.id: _mask_for_polygon(left.footprint),
        right.id: _mask_for_polygon(SkPath(shell=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])),
    }

    proposals = symmetry_pass([left, right], masks)
    out = select_refinements([left, right], masks, [symmetry_pass])

    assert any(2 in proposal.obj_ids for proposal in proposals)
    assert [obj.current.kind for obj in out] == ["path", "path"]

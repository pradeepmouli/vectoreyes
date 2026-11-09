from pathlib import Path
import math

import numpy as np
from PIL import Image

from vectoreyes.candidate import FlatFill, LinearGradientFill, RadialGradientFill
from vectoreyes.emit import shape_to_path_d
from vectoreyes.fit import Shape
from vectoreyes.refinement.framework import select_refinements
from vectoreyes.refinement.gate import rasterize
from vectoreyes.refinement.passes.compound import split_compound_pass
from vectoreyes.refinement.passes.primitives import primitives_pass
import vectoreyes.occlusion_detection as occlusion_module
import vectoreyes.refinement.passes.primitives as primitives_module
from vectoreyes.refinement.region_sample import RegionSample, to_polygon
from vectoreyes.refinement.trace import _trace_shape_from_contours
from vectoreyes.pipeline import Options, _refinement_passes, _refinement_report, _prefer_refinement_svg, _render_refinement_body, idealize
import vectoreyes.pipeline as pipeline_module
from tests._render import render_svg
from vectoreyes.emit import render_svg_doc


def _painted_on(regions, background, width=140, height=140):
    body, defs = _render_refinement_body(regions)
    return render_svg(render_svg_doc(width, height, [
        f'<rect width="{width}" height="{height}" fill="{background}"/>', *body,
    ], defs), width, height)


def _assert_same_paint(before, after):
    for background in ("#123456", "#ef7090"):
        np.testing.assert_array_equal(_painted_on(before, background), _painted_on(after, background))


def _disk(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _disk_image() -> np.ndarray:
    img = np.full((120, 120, 3), 255, np.uint8)
    img[_disk(120, 120, 60, 60, 40)] = (200, 30, 30)
    return img


def _asymmetric_cloud_image() -> np.ndarray:
    img = np.full((96, 128, 3), 255, np.uint8)
    yy, xx = np.ogrid[:96, :128]
    mask = (
        (((xx - 42) ** 2) / 22**2 + ((yy - 52) ** 2) / 18**2 <= 1)
        | (((xx - 66) ** 2) / 30**2 + ((yy - 42) ** 2) / 24**2 <= 1)
        | (((xx - 88) ** 2) / 20**2 + ((yy - 56) ** 2) / 16**2 <= 1)
    )
    mask[58:72, 32:102] = True
    mask[38:48, 88:110] = False
    img[mask] = (20, 70, 150)
    return img


def _two_colored_square_image() -> np.ndarray:
    img = np.full((80, 120, 3), 255, np.uint8)
    img[20:40, 16:36] = (17, 34, 51)
    img[20:40, 70:90] = (171, 205, 239)
    return img


def _synthetic_mastercard_image() -> np.ndarray:
    h, w, gap, r = 240, 360, 58, 104
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    left = (xx - (cx - gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    right = (xx - (cx + gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    img = np.full((h, w, 3), 255, np.uint8)
    img[left] = (235, 0, 27)
    img[right] = (247, 158, 27)
    img[left & right] = (255, 95, 0)
    return img


def _dense_rounded_rect_d(x0: float, y0: float, x1: float, y1: float, radius: float, *, samples: int = 10) -> str:
    commands = [f"M{x0 + radius} {y0}", f"L{x1 - radius} {y0}"]
    arcs = (
        (x1 - radius, y0 + radius, -math.pi / 2.0, 0.0),
        (x1 - radius, y1 - radius, 0.0, math.pi / 2.0),
        (x0 + radius, y1 - radius, math.pi / 2.0, math.pi),
        (x0 + radius, y0 + radius, math.pi, math.pi * 1.5),
    )
    straight_ends = (
        (x1, y1 - radius),
        (x0 + radius, y1),
        (x0, y0 + radius),
        (x0 + radius, y0),
    )
    for (cx, cy, start, end), line_end in zip(arcs, straight_ends, strict=True):
        for theta in np.linspace(start, end, samples + 1)[1:]:
            commands.append(f"L{cx + radius * math.cos(float(theta))} {cy + radius * math.sin(float(theta))}")
        if line_end != straight_ends[-1]:
            commands.append(f"L{line_end[0]} {line_end[1]}")
    commands.append("Z")
    return " ".join(commands)


def test_refinement_option_defaults_off_for_existing_pipeline():
    img = _disk_image()

    assert idealize(img, options=Options()) == idealize(img, options=Options(refinement=False))


def test_refinement_pipeline_turns_disk_into_circle_and_is_deterministic():
    img = _disk_image()
    opt = Options(refinement=True)

    first = idealize(img, options=opt)
    second = idealize(img, options=opt)

    assert first == second
    assert "<circle" in first


def test_refinement_pipeline_uses_refinement_output_even_when_noisier_than_trace(monkeypatch):
    monkeypatch.setattr(pipeline_module, "_prefer_refinement_svg", lambda _trace, _optimized: False)

    svg = idealize(_disk_image(), options=Options(refinement=True))

    assert "<circle" in svg


def test_refinement_report_includes_object_diagnostics():
    _svg, report = idealize(_disk_image(), options=Options(refinement=True), report=True)

    data = report.diagnostics.to_dict()
    assert report.elements == 1
    assert report.strategies["refinement_circle"] == 1
    assert data["refinement_objects"][0]["shape"] == "circle"


def test_refinement_structural_counts_do_not_reject_refinement_output():
    trace = '<svg><path d="M0 0 Z"/></svg>'
    larger = '<svg><path d="M0 0 Z"/><path d="M1 1 Z"/></svg>'
    longer = '<svg><path d="M0 0 L1 1 L2 2 L3 3 Z"/></svg>'

    assert _prefer_refinement_svg(trace, larger)
    assert _prefer_refinement_svg(trace, longer)
    assert _prefer_refinement_svg(trace, trace)


def test_refinement_prefer_allows_small_element_increase_when_path_segments_drop():
    trace = '<svg><path d="M0 0 L10 0 L10 10 L0 10 Z"/></svg>'
    optimized = '<svg><circle r="1"/><use href="#s0"/></svg>'

    assert _prefer_refinement_svg(trace, optimized)


def test_refinement_prefer_ignores_element_and_path_segment_counts():
    trace = '<svg><path d="M0 0 L10 0 L10 10 L0 10 L0 20 L10 20 L10 30 L0 30 Z"/></svg>'
    many_primitives = '<svg><circle r="1"/><circle r="2"/><use href="#s0"/></svg>'
    simple_long_path = '<svg><path d="M1000.123 1000.456 L2000.789 2000.123 Z"/></svg>'
    more_path_segments = '<svg><path d="M0 0 L1 0 L2 0 L3 0 L4 0 Z"/></svg>'
    fewer_path_segments_longer_text = (
        '<svg><path d="M1000.123 1000.456 L2000.789 2000.123 '
        'L3000.456 3000.789 Z"/></svg>'
    )

    assert _prefer_refinement_svg(trace, many_primitives)
    assert _prefer_refinement_svg(trace, fewer_path_segments_longer_text)
    assert _prefer_refinement_svg(simple_long_path, more_path_segments)


def test_refinement_flatten_emits_paths_without_primitives_or_transforms():
    svg = idealize(_disk_image(), options=Options(refinement=True, flatten=True))

    assert "<path" in svg
    assert "<circle" not in svg
    assert "<use" not in svg
    assert "transform=" not in svg


def test_refinement_report_includes_leaf_pass_diagnostics_in_object_section():
    obj = RegionSample(
        1,
        Shape("path", {"d": "M0 0 L10 0 L10 10 L0 10 Z"}),
        FlatFill("#111111"),
        0,
        diagnostics={"symmetry": {"accepted": True, "mode": "self"}},
    )

    report = _refinement_report([obj], Options(refinement=True))
    diagnostics = report.diagnostics.to_dict()

    assert diagnostics["refinement_objects"][0]["diagnostics"]["symmetry"]["mode"] == "self"


def test_refinement_simplifies_compound_children_after_split():
    outer = _dense_rounded_rect_d(0.0, 0.0, 120.0, 120.0, 20.0, samples=12)
    hole = _dense_rounded_rect_d(28.0, 28.0, 92.0, 92.0, 14.0, samples=12)
    shape = Shape("path", {"d": f"{outer} {hole}", "fill_rule": "evenodd"})
    fill = LinearGradientFill({"x1": 0.0, "y1": 0.0, "x2": 120.0, "y2": 120.0}, [(0.0, "#111111"), (1.0, "#eeeeee")])
    region = RegionSample(1, shape, fill, 0)

    out = select_refinements(
        [region],
        {region.id: rasterize(region.footprint, (140, 140))},
        _refinement_passes(Options(refinement=True)),
    )

    assert all(leaf.fill == fill for root in out for leaf in root.leaves())
    for background, rgb in (("#123456", (18, 52, 86)), ("#ef7090", (239, 112, 144))):
        np.testing.assert_array_equal(_painted_on(out, background)[60, 60], rgb)
    assert not out[0].footprint._path.contains(60, 60)


def test_refinement_recolored_clone_retains_a_use_with_its_own_fill():
    import xml.etree.ElementTree as ET
    svg = idealize(_two_colored_square_image(), options=Options(refinement=True))

    root = ET.fromstring(svg)
    geometry = next(element for element in root.iter() if element.get('id') == 's0-geometry')
    assert geometry.tag.rsplit('}', 1)[-1] in {'rect', 'path'}
    assert 'fill' not in geometry.attrib
    assert '<use id="s0" href="#s0-geometry"' in svg
    assert '<use id="s1" href="#s0-geometry"' in svg
    assert svg.lower().count('fill="#112233"') == 1
    assert svg.lower().count('fill="#abcdef"') == 1


def test_refinement_no_symmetry_skips_symmetry_pass():
    assert "symmetry_pass" in {getattr(pass_fn, "__name__", "") for pass_fn in _refinement_passes(Options(refinement=True))}
    assert "symmetry_pass" not in {
        getattr(pass_fn, "__name__", "")
        for pass_fn in _refinement_passes(Options(refinement=True, no_symmetry=True))
    }


def test_refinement_threads_epsilon_into_primitives_pass(monkeypatch):
    seen: list[float] = []

    def _recording_recognize(points, *, epsilon):
        seen.append(epsilon)
        return None

    monkeypatch.setattr(primitives_module, "recognize_primitive_geometry", _recording_recognize)
    obj = RegionSample(
        1,
        Shape("path", {"d": "M0 0 L20 0 L20 20 L0 20 Z"}),
        FlatFill("#111111"),
        0,
    )
    pass_fn = next(
        pass_fn
        for pass_fn in _refinement_passes(Options(refinement=True, epsilon=0.25))
        if pass_fn.__name__ == "primitives_pass"
    )

    pass_fn([obj], {obj.id: rasterize(obj.footprint, (32, 32))})

    assert seen == [0.25]


def test_refinement_no_symmetry_threads_into_occlusion_pass(monkeypatch):
    seen_axes = []

    def _recording_reconstruct_scene(regions, axis, shape_hw):
        seen_axes.append(axis)
        return [], regions

    monkeypatch.setattr(occlusion_module, "reconstruct_scene", _recording_reconstruct_scene)
    left = RegionSample(
        1,
        Shape("path", {"d": "M0 0 L10 0 L10 10 L0 10 Z"}),
        FlatFill("#111111"),
        0,
        raster=np.pad(np.ones((10, 10), dtype=bool), ((0, 22), (0, 22))),
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M10 0 L20 0 L20 10 L10 10 Z"}),
        FlatFill("#111111"),
        1,
        raster=np.pad(np.ones((10, 10), dtype=bool), ((0, 22), (10, 12))),
    )
    masks = {
        left.id: left.raster,
        right.id: right.raster,
    }
    pass_fn = next(
        pass_fn
        for pass_fn in _refinement_passes(Options(refinement=True, no_symmetry=True))
        if pass_fn.__name__ == "occlusion_pass"
    )

    pass_fn([left, right], masks)

    assert seen_axes == [None]


def test_refinement_pipeline_does_not_mirror_asymmetric_single_object():
    svg = idealize(_asymmetric_cloud_image(), options=Options(refinement=True))

    assert "<path" in svg
    assert "<use" not in svg


def test_refinement_pipeline_keeps_daikonic_fixture_objects_present():
    src = Path(__file__).parents[1] / "fixtures" / "daikonic" / "source.png"
    arr = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)

    svg = idealize(arr, options=Options(refinement=True, epsilon=1.8, max_error=1.2))

    element_count = sum(svg.count(token) for token in ("<path", "<circle", "<ellipse", "<rect", "<use"))
    assert element_count >= 8
    assert len(svg) > 1000


def test_refinement_pipeline_reconstructs_mastercard_as_scene_branch():
    svg, report = idealize(_synthetic_mastercard_image(), options=Options(refinement=True), report=True)

    assert svg.count("<circle") == 1
    assert svg.count("<use") == 2
    assert svg.count("<path") == 1
    assert svg.count('href="#s0-0-geometry"') == 2
    assert 'fill="#f79e1b"' in svg.lower()
    assert 'fill="#eb001b"' in svg.lower()
    assert 'fill="#ff5f00"' in svg.lower()
    diagnostics = report.diagnostics.to_dict()
    assert diagnostics["refinement_fallback"] is None
    assert diagnostics["refinement_regions"][0]["kind"] == "branch"
    assert diagnostics["refinement_regions"][0]["children"] == 3
    assert diagnostics["refinement_objects"][0]["diagnostics"]["occlusion"]["reconstructed_child"] is True
    assert [obj["shape"] for obj in diagnostics["refinement_objects"]] == ["circle", "use", "path"]
    assert diagnostics["refinement_objects"][2]["diagnostics"]["symmetry"]["materialized"] is True


def test_refinement_inlines_same_gradient_fill_mirror_to_preserve_paint_space():
    fill = RadialGradientFill({"cx": 20.0, "cy": 20.0, "r": 30.0}, [(0.0, "#000000"), (1.0, "#ffffff")])
    source = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L20 10 L20 30 L10 30 Z"}),
        fill,
        0,
    )
    mirror = RegionSample(
        2,
        Shape("use", {"href_obj_id": 1, "transform": (-1.0, 0.0, 0.0, 1.0, 60.0, 0.0)}),
        fill,
        0,
        footprint=source.footprint,
    )

    body, defs = _render_refinement_body([source, mirror])
    svg_body = "".join(body)

    assert len(defs) == 1
    assert "<use" not in svg_body
    assert svg_body.count("<path") == 2
    assert svg_body.count('fill="url(#g0)"') == 2


def test_refinement_retains_recolored_clone_as_use_with_its_own_fill():
    source = RegionSample(
        1,
        Shape("circle", {"cx": 10.0, "cy": 10.0, "r": 8.0}),
        FlatFill("#FF0000"),
        0,
    )
    clone = RegionSample(
        2,
        Shape("use", {"href_obj_id": 1, "transform": (1.0, 0.0, 0.0, 1.0, 20.0, 0.0)}),
        FlatFill("#0000FF"),
        1,
        footprint=source.footprint,
        diagnostics={"clones": {"accepted": True, "matched_source": 1}},
    )

    body, defs = _render_refinement_body([source, clone])

    assert defs == ['<circle id="s1-geometry" cx="10" cy="10" r="8"/>']
    assert body[0].startswith('<use id="s1" href="#s1-geometry"')
    assert 'fill="#FF0000"' in body[0]
    assert body[1].startswith('<use id="s2" href="#s1-geometry"')
    assert 'fill="#0000FF"' in body[1]


def test_refinement_emits_tree_ids_for_branch_children():
    fill = FlatFill("#111111")
    source = RegionSample(
        1,
        Shape("path", {"d": "M0 0 L10 0 L10 10 L0 10 Z"}),
        fill,
        0,
    )
    child = RegionSample(
        3,
        Shape("circle", {"cx": 5.0, "cy": 5.0, "r": 3.0}),
        FlatFill("#FFFFFF"),
        0,
    )
    mirror = RegionSample(
        4,
        Shape("use", {"href_obj_id": 1, "transform": (-1.0, 0.0, 0.0, 1.0, 20.0, 0.0)}),
        fill,
        0,
        footprint=source.footprint,
    )
    branch = RegionSample.branch(id=1, children=[source, child, mirror], z=0)

    body, _defs = _render_refinement_body([branch])
    svg_body = "".join(body)

    assert 'id="s1-1"' in svg_body
    assert 'id="s1-3"' in svg_body
    assert 'href="#s1-1"' not in svg_body
    assert svg_body.count("<path") == 2


def test_refinement_orders_branch_mirror_before_sibling_covers():
    fill = FlatFill("#111111")
    source = RegionSample(
        1,
        Shape("path", {"d": "M0 0 L10 0 L10 10 L0 10 Z"}),
        fill,
        0,
    )
    cover = RegionSample(
        3,
        Shape("circle", {"cx": 5.0, "cy": 5.0, "r": 3.0}),
        FlatFill("#FFFFFF"),
        0.4,
    )
    mirror = RegionSample(
        5,
        Shape("use", {"href_obj_id": 1, "transform": (-1.0, 0.0, 0.0, 1.0, 20.0, 0.0)}),
        fill,
        0.1,
        footprint=source.footprint,
    )
    branch = RegionSample.branch(id=1, children=[source, cover, mirror], z=0)

    body, _defs = _render_refinement_body([branch])

    assert body[0].startswith('<path id="s1-1"')
    assert body[1].startswith('<path')
    assert body[2].startswith('<circle id="s1-3"')


def test_refinement_combines_self_symmetry_branch_on_final_output():
    fill = FlatFill("#111111")
    source = RegionSample(
        1,
        Shape("path", {"d": "M0 0 L10 0 L10 20 L0 20 Z"}),
        fill,
        0,
    )
    mirror = RegionSample(
        2,
        Shape("use", {"href_obj_id": 1, "transform": (-1.0, 0.0, 0.0, 1.0, 20.0, 0.0)}),
        fill,
        0.1,
        footprint=RegionSample(3, Shape("path", {"d": "M10 0 L20 0 L20 20 L10 20 Z"}), fill).footprint,
    )
    branch = RegionSample.branch(
        id=1,
        children=[source, mirror],
        z=0,
        footprint=RegionSample(4, Shape("path", {"d": "M0 0 L20 0 L20 20 L0 20 Z"}), fill).footprint,
        diagnostics={"symmetry": {"accepted": True, "mode": "self"}},
    )

    from vectoreyes.refinement.symmetry_materialize import materialize_self_symmetry
    retained = materialize_self_symmetry((branch,))
    assert retained[0].is_leaf
    body, _defs = _render_refinement_body(retained)
    svg_body = "".join(body)

    assert svg_body.count("<path") == 1
    assert "<use" not in svg_body
    assert svg_body.count("M") == 1
    assert "Z Z" not in svg_body
    stitched = to_polygon(Shape("path", {"d": body[0].split(' d="', 1)[1].split('"', 1)[0]}))
    assert stitched.symmetric_difference(branch.footprint).area < 1e-6


def test_refinement_self_symmetry_uses_union_footprints_not_drifting_fitted_halves():
    fill = FlatFill("#111111")
    source = RegionSample(
        1,
        Shape("path", {"d": "M0 0 L9.75 0 L9.75 20 L0 20 Z"}),
        fill,
        0,
        footprint=RegionSample(10, Shape("path", {"d": "M0 0 L10 0 L10 20 L0 20 Z"}), fill).footprint,
    )
    mirror = RegionSample(
        2,
        Shape("use", {"href_obj_id": 1, "transform": (-1.0, 0.0, 0.0, 1.0, 20.0, 0.0)}),
        fill,
        0.1,
        footprint=RegionSample(11, Shape("path", {"d": "M10 0 L20 0 L20 20 L10 20 Z"}), fill).footprint,
    )
    branch = RegionSample.branch(
        id=1,
        children=[source, mirror],
        z=0,
        diagnostics={"symmetry": {"accepted": True, "mode": "self"}},
    )

    from vectoreyes.refinement.symmetry_materialize import materialize_self_symmetry
    retained = materialize_self_symmetry((branch,))
    assert retained[0].is_leaf
    body, _defs = _render_refinement_body(retained)
    emitted = to_polygon(Shape("path", {"d": body[0].split(' d="', 1)[1].split('"', 1)[0]}))

    assert body[0].count("M") == 1
    assert emitted.symmetric_difference(branch.footprint).area < 1e-6


def test_refinement_self_symmetry_preserves_retained_quadratic_tip():
    from vectoreyes.refinement.passes.symmetry import symmetry_pass
    shape = Shape(
        "path",
        {
            "d": (
                "M326 382.5 "
                "Q319.38 382.99 314 380.5 "
                "Q278.75 353.47 261.5 315 "
                "L384.5 315 "
                "Q369.58 347.78 340 374.5 "
                "L326 382.5 Z"
            )
        },
    )
    region = RegionSample(1, shape, FlatFill("#f23325"), 0)
    out = select_refinements(
        [region],
        {region.id: rasterize(region.footprint, (420, 420))},
        [symmetry_pass],
    )

    body, _defs = _render_refinement_body(out)
    svg_body = "".join(body)

    assert "L322.94 382." not in svg_body
    assert "C" not in svg_body
    assert svg_body.count("Q") <= 4
    assert "Q" in svg_body
    assert out[0].footprint.symmetric_difference(region.footprint).area / region.footprint.area < 0.06


def test_refinement_trace_uses_quadratic_base_even_when_cubics_are_enabled():
    contour = []
    for theta in np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False):
        contour.append(
            (
                40.0 + 25.0 * np.cos(float(theta)) + 5.0 * np.sin(2.0 * float(theta)),
                40.0 + 15.0 * np.sin(float(theta)),
            )
        )

    contour_array = np.asarray(contour, dtype=float)
    shape = _trace_shape_from_contours(
        [contour_array],
        Options(refinement=True, cubic_paths=True, epsilon=1.0, max_error=1.0),
    )

    d = str(shape.params["d"])
    assert "C" not in d
    assert d.count("Q") >= 4
    # This fixture is curved everywhere; quadratic-base tracing must not need
    # an artificial line just to satisfy a command-count expectation.
    assert "L" not in d
    contour_path = " ".join(
        [f"M{contour_array[0, 0]} {contour_array[0, 1]}"]
        + [f"L{x} {y}" for x, y in contour_array[1:]]
        + ["Z"]
    )
    source = to_polygon(Shape("path", {"d": contour_path}))
    traced = to_polygon(shape)
    assert traced.symmetric_difference(source).area / source.area < 0.03


def test_compound_split_keeps_uncovered_cutout_subpaths_subtractive():
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    hole = shape_to_path_d(Shape("circle", {"cx": 70.0, "cy": 50.0, "r": 12.0}))
    speck = "M10 10 L10.5 10 L10 10.5 Z"
    shape = Shape("path", {"d": f"{outer} {hole} {speck}", "fill_rule": "evenodd"})
    region = RegionSample(1, shape, FlatFill("#111111"), 0)

    out = select_refinements(
        [region],
        {region.id: rasterize(region.footprint, (120, 120))},
        [split_compound_pass, primitives_pass],
    )

    assert len(out) == 1
    assert out[0].current == shape
    body, defs = _render_refinement_body(out)
    assert defs == []
    assert len(body) == 1
    assert speck in body[0]
    assert 'fill-rule="evenodd"' in body[0]


def test_compound_split_keeps_disjoint_subpaths_parent_filled():
    left = "M0 50 L30 0 L40 50 Z"
    center = "M50 50 L80 0 L110 50 Z"
    right = "M120 50 L150 0 L160 50 Z"
    shape = Shape("path", {"d": f"{center} {left} {right}", "fill_rule": "evenodd"})
    region = RegionSample(1, shape, FlatFill("#FDAD00"), 0)
    source_mask = (
        rasterize(to_polygon(Shape("path", {"d": left})), (80, 180))
        | rasterize(to_polygon(Shape("path", {"d": center})), (80, 180))
        | rasterize(to_polygon(Shape("path", {"d": right})), (80, 180))
    )

    out = select_refinements(
        [region],
        {region.id: source_mask},
        [split_compound_pass],
    )

    assert len(out) == 1
    branch = out[0]
    assert branch.is_branch
    assert len(branch.children) == 3
    assert [child.fill for child in branch.children] == [region.fill, region.fill, region.fill]


def test_compound_split_preserves_same_winding_nonzero_painted_interior():
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    inner = "M25 25 L75 25 L75 75 L25 75 Z"
    # Both contours have the same winding: this is filled, not a hole.
    shape = Shape("path", {"d": f"{outer} {inner}"})
    region = RegionSample(1, shape, FlatFill("#C837AB"), 0)

    out = select_refinements(
        [region],
        {region.id: rasterize(region.footprint, (120, 120))},
        # New compound children are visited by the following pass.  This
        # mirrors the production pipeline, which runs primitive fitting again
        # after exposing compound subpaths as independent leaves.
        [split_compound_pass, primitives_pass, primitives_pass],
    )

    assert len(out) == 1
    assert out[0].is_leaf
    _assert_same_paint([region], out)
    assert out[0].footprint._path.contains(50, 50)


def test_compound_split_ignores_source_mask_that_incorrectly_fills_a_hole():
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    inner = "M30 30 L70 30 L70 70 L30 70 Z"
    shape = Shape("path", {"d": f"{outer} {inner}", "fill_rule": "evenodd"})
    region = RegionSample(1, shape, FlatFill("#FDAD00"), 0)
    source_mask = rasterize(to_polygon(Shape("path", {"d": outer})), (120, 120))

    out = select_refinements(
        [region],
        {region.id: source_mask},
        [split_compound_pass],
    )

    assert len(out) == 1
    assert out[0].is_leaf and out[0].current == shape
    _assert_same_paint([region], out)
    assert not out[0].footprint._path.contains(50, 50)


def test_compound_split_carries_visible_region_fill_for_uncovered_subpath():
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    inner = "M30 30 L70 30 L70 70 L30 70 Z"
    shape = Shape("path", {"d": f"{outer} {inner}", "fill_rule": "evenodd"})
    region = RegionSample(1, shape, FlatFill("#FDAD00"), 1)
    source_mask = rasterize(to_polygon(shape), (120, 120))
    visible = RegionSample(
        2,
        Shape("path", {"d": inner}),
        FlatFill("#0055CC"),
        0,
        raster=rasterize(to_polygon(Shape("path", {"d": inner})), (120, 120)),
    )

    out = select_refinements(
        [visible, region],
        {region.id: source_mask, visible.id: visible.raster},
        [split_compound_pass],
    )

    retained = next(obj for obj in out if obj.id == region.id)
    assert retained.is_leaf and retained.current == shape
    assert len(out) == 2
    _assert_same_paint([visible, region], out)
    np.testing.assert_array_equal(_painted_on(out, "#123456")[50, 50], (0, 85, 204))


def test_gradient_compound_split_preserves_transparent_cutout():
    outer = "M0 0 L100 0 L100 100 L0 100 Z"
    inner = "M25 25 L75 25 L75 75 L25 75 Z"
    shape = Shape("path", {"d": f"{outer} {inner}", "fill_rule": "evenodd"})
    fill = RadialGradientFill({"cx": 50.0, "cy": 50.0, "r": 70.0}, [(0.0, "#000000"), (1.0, "#ffffff")])
    region = RegionSample(1, shape, fill, 0)

    out = select_refinements(
        [region],
        {region.id: rasterize(region.footprint, (120, 120))},
        [split_compound_pass],
    )

    assert len(out) == 1
    assert out[0].is_leaf and out[0].fill == fill
    _assert_same_paint([region], out)
    np.testing.assert_array_equal(_painted_on(out, "#123456")[50, 50], (18, 52, 86))


def test_compound_split_preserves_transparency_through_stacked_gradient_holes():
    lower_outer = "M0 0 L100 0 L100 100 L0 100 Z"
    lower_inner = "M10 10 L90 10 L90 90 L10 90 Z"
    lower_shape = Shape("path", {"d": f"{lower_outer} {lower_inner}", "fill_rule": "evenodd"})
    lower_fill = RadialGradientFill({"cx": 50.0, "cy": 50.0, "r": 70.0}, [(0.0, "#000000"), (1.0, "#ffffff")])
    lower = RegionSample(1, lower_shape, lower_fill, 0)

    top_outer = "M20 20 L80 20 L80 80 L20 80 Z"
    top_hole = shape_to_path_d(Shape("circle", {"cx": 50.0, "cy": 50.0, "r": 16.0}))
    top_shape = Shape("path", {"d": f"{top_outer} {top_hole}", "fill_rule": "evenodd"})
    top_fill = LinearGradientFill({"x1": 20.0, "y1": 20.0, "x2": 80.0, "y2": 80.0}, [(0.0, "#111111"), (1.0, "#eeeeee")])
    top = RegionSample(2, top_shape, top_fill, 1)

    out = select_refinements(
        [lower, top],
        {
            lower.id: rasterize(lower.footprint, (120, 120)),
            top.id: rasterize(top.footprint, (120, 120)),
        },
        [split_compound_pass, primitives_pass],
    )

    assert all(region.is_leaf for region in out)
    assert [region.fill for region in out] == [lower_fill, top_fill]
    _assert_same_paint([lower, top], out)
    for background, rgb in (("#123456", (18, 52, 86)), ("#ef7090", (239, 112, 144))):
        np.testing.assert_array_equal(_painted_on(out, background)[50, 50], rgb)

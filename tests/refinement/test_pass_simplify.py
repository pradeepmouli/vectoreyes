import math

import numpy as np

from vectormark.candidate import FlatFill
from vectormark.fit import Shape, minimum_line_length
from vectormark.refinement.framework import select_refinements
from vectormark.refinement.gate import rasterize
from vectormark.refinement.region_sample import RegionSample
from vectormark.path_sampling import _parse_subpaths
from vectormark.refinement.passes.simplify import simplify_pass
from vectormark.pipeline import Options, _refinement_passes
import vectormark.drawing_simplification as simplify_module


def _command_count(d: str) -> int:
    return sum(1 for ch in d if ch in "MLQCAZ")


def _line_lengths(d: str) -> list[float]:
    lengths = []
    current = start = None
    for subpath in _parse_subpaths(d):
        for command, values in subpath:
            if command == "M":
                current = (values[0], values[1])
                start = current
            elif command == "L" and current is not None:
                end = (values[0], values[1])
                lengths.append(math.dist(current, end))
                current = end
            elif command == "Q":
                current = (values[2], values[3])
            elif command == "C":
                current = (values[4], values[5])
            elif command == "Z":
                current = start
    return lengths


def _obj_from_d(d: str, obj_id: int = 1) -> RegionSample:
    return RegionSample(obj_id, Shape("path", {"d": d}), FlatFill("#000000"), 0)


def _obj_from_path(shape: Shape, obj_id: int = 1) -> RegionSample:
    return RegionSample(obj_id, shape, FlatFill("#000000"), 0)


def _mask_for_obj(obj: RegionSample, shape_hw: tuple[int, int] = (96, 96)) -> np.ndarray:
    return rasterize(obj.footprint, shape_hw)


def _optimized_obj(
    obj: RegionSample,
    *,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    cubic: bool = False,
) -> RegionSample:
    out = select_refinements(
        [obj],
        {obj.id: _mask_for_obj(obj)},
        [lambda objects, masks: simplify_pass(objects, masks, epsilon=epsilon, max_error=max_error, cubic=cubic)],
    )
    return out[0]


def _optimized_d(
    obj: RegionSample,
    *,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    cubic: bool = False,
) -> str:
    return str(_optimized_obj(obj, epsilon=epsilon, max_error=max_error, cubic=cubic).current.params["d"])


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


def test_simplify_pass_refits_dense_rounded_rect_to_quadratic_border():
    d = _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)
    obj = _obj_from_d(d)

    simplified_d = _optimized_d(obj, epsilon=1.0, max_error=1.0)

    assert _command_count(simplified_d) < _command_count(d)
    assert simplified_d.count("Q") == 8
    assert "C" not in simplified_d


def test_simplify_pass_preserves_holes_when_refitting_outer_border():
    outer = _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)
    hole = "M35 35 L50 35 L42 50 Z"
    obj = _obj_from_path(Shape("path", {"d": f"{outer} {hole}", "fill_rule": "evenodd"}))

    optimized = _optimized_obj(obj, epsilon=1.0, max_error=1.0)
    simplified_d = str(optimized.current.params["d"])

    assert optimized.current.params.get("fill_rule") == "evenodd"
    assert simplified_d.count("M") == 2
    assert simplified_d.count("Q") == 8
    assert "L50 35" in simplified_d


def test_simplify_pass_does_not_worsen_cutout_subpath_when_outer_improves():
    outer = (
        "M387 497.5 Q318 497.5 249 497.5 Q208.5 497.5 168 497.5 "
        "Q138.48 498.25 110 496.5 Q82.94 495.98 60 485.5 "
        "Q49.76 479.58 40 472.5 Q3.21 440.02 1.5 393 "
        "Q-0.25 381.52 0.5 369 Q-1.25 341.02 -0.5 312 "
        "Q-0.5 266.5 -0.5 221 Q-0.5 168 -0.5 115 "
        "Q1.2 104.79 1.5 94 Q4.32 81.93 7.5 70 "
        "Q13.38 56.53 21.5 44 Q56.67 0.93 111 0.5 "
        "Q179.49 -1.24 249 -0.5 Q289.5 -0.5 330 -0.5 "
        "Q357.02 -1.25 383 0.5 Q410.99 0.69 435 10.5 "
        "Q445.1 16.25 455 22.5 Q494.59 55.21 496.5 105 "
        "Q498.25 116.98 497.5 130 Q497.5 157.5 497.5 185 "
        "Q497.5 231 497.5 277 Q498.19 337.99 496.5 398 "
        "Q493.56 414.94 488.5 431 Q483.32 441.13 477.5 451 "
        "Q447.21 491.42 399 495.5 Q393 496.5 387 497.5 Z"
    )
    cutout = (
        "M195.5 193 Q197.12 187.95 202 185.5 Q269.25 172.6 336 158.5 "
        "Q339.11 158.73 342 159.5 Q343.47 161.16 344.5 163 "
        "Q344.5 224.5 344.5 286 Q340.08 293.48 332 295.5 "
        "Q321.61 296.57 312 299.5 Q283.43 302.1 269.5 327 "
        "Q267.93 334.47 266.5 342 Q267.82 349.08 269.5 356 "
        "Q283.68 384.52 317 380.5 Q327.06 378.14 337 375.5 "
        "Q357.18 365.14 363.5 344 Q362.84 341.02 364.5 339 "
        "Q364.5 212 364.5 85 Q362.62 78.19 356 75.5 L183 110.5 "
        "Q178.7 114.31 176.5 119 Q176.5 217 176.5 315 "
        "Q174.62 318.84 173.5 323 Q167.26 329.19 159 330.5 "
        "Q151.11 331.06 144 333.5 Q120.9 335.16 104.5 353 "
        "Q99.53 361.89 97.5 372 Q97.39 401.61 124 413.5 "
        "Q132.21 415.21 141 415.5 Q145.5 414.5 150 413.5 "
        "Q170.03 411.95 184.5 397 Q191.41 386.79 194.5 375 "
        "Q196.24 284.51 195.5 193 Z"
    )
    obj = _obj_from_path(Shape("path", {"d": f"{outer} {cutout}", "fill_rule": "evenodd"}))

    proposals = simplify_pass([obj], {obj.id: _mask_for_obj(obj, (512, 512))})
    simplified_d = str(proposals[0].new_objects[0].current.params["d"])

    # Q -> L is a real simplification even when command count is unchanged.
    # More importantly, improvement of the exterior cannot pay for a distorted
    # cutout. Assert geometry rather than one fitter's exact command spelling.
    assert simplify_module._path_command_cost_d(simplified_d) < simplify_module._path_command_cost_d(f"{outer} {cutout}")
    parts = simplify_module._parse_subpaths(simplified_d)
    assert len(parts) == 2
    assert proposals[0].new_objects[0].current.params["fill_rule"] == "evenodd"
    for source_d, tokens in zip((outer, cutout), parts, strict=True):
        before = source_d
        after = simplify_module._subpath_d(tokens)
        assert simplify_module._boundary_max_deviation(before, after) <= 1.0
    assert simplify_module._same_subpath_topology(obj.current.params['d'], proposals[0].new_objects[0].current.params['d'], fill_rule='evenodd')


def test_simplify_forced_corners_include_sharp_curve_joins():
    tokens = [
        ("M", [156.0, 303.5]),
        ("L", [0.0, 303.5]),
        ("Q", [-0.65, 292.26, 3.5, 282.0]),
        ("Q", [78.85, 152.15, 153.5, 22.0]),
        ("Q", [160.57, 12.82, 170.0, 6.5]),
        ("L", [249.5, 142.0]),
        ("Z", []),
    ]

    corners = simplify_module._forced_corners(tokens)

    assert corners is not None
    assert (170.0, 6.5) in {tuple(point) for point in corners.tolist()}
    assert (3.5, 282.0) not in {tuple(point) for point in corners.tolist()}
    assert (153.5, 22.0) not in {tuple(point) for point in corners.tolist()}


def test_simplify_forced_corners_include_actual_sharp_curve_turn():
    tokens = [
        ("M", [0.0, 0.0]),
        ("Q", [20.0, 0.0, 20.0, 20.0]),
        ("Q", [40.0, 20.0, 40.0, 40.0]),
        ("Z", []),
    ]

    corners = simplify_module._forced_corners(tokens)

    assert corners is not None
    assert (20.0, 20.0) in {tuple(point) for point in corners.tolist()}


def test_simplify_does_not_force_smooth_daikonic_bottom_tip_join_to_line():
    d = (
        "M326 382.5 "
        "Q319.38 382.99 314 380.5 "
        "Q278.75 353.47 261.5 315 "
        "L384.5 315 "
        "Q369.58 347.78 340 374.5 "
        "L326 382.5 Z"
    )
    obj = _obj_from_d(d)

    simplified_d = _optimized_d(obj, epsilon=1.0, max_error=1.0)

    assert "M326 382.5 L314 380.5" not in simplified_d
    assert "Q" in simplified_d


def test_simplify_pass_drops_cutout_when_later_path_covers_it():
    outer = _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)
    hole = "M35 35 L50 35 L42 50 Z"
    background = _obj_from_path(Shape("path", {"d": f"{outer} {hole}", "fill_rule": "evenodd"}), obj_id=1)
    cover = RegionSample(2, Shape("path", {"d": hole}), FlatFill("#FFFFFF"), 1)

    out = select_refinements(
        [background, cover],
        {
            background.id: _mask_for_obj(background),
            cover.id: np.zeros_like(_mask_for_obj(cover)),
        },
        [lambda objects, masks: simplify_pass(objects, masks, epsilon=1.0, max_error=1.0)],
    )

    by_id = {obj.id: obj for obj in out}
    background_d = str(by_id[1].current.params["d"])
    assert by_id[1].current.params.get("fill_rule") is None
    assert background_d.count("M") == 1
    assert background_d.count("Q") == 8
    assert by_id[2].current.params["d"] == hole


def test_simplify_pass_keeps_referenced_source_subpaths():
    outer = _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)
    hole = "M35 35 L50 35 L42 50 Z"
    source = _obj_from_path(Shape("path", {"d": f"{outer} {hole}", "fill_rule": "evenodd"}), obj_id=1)
    cover = RegionSample(2, Shape("path", {"d": hole}), FlatFill("#FFFFFF"), 1)
    clone = RegionSample(
        3,
        Shape("use", {"href_obj_id": 1, "transform": (1.0, 0.0, 0.0, 1.0, 110.0, 0.0)}),
        FlatFill("#000000"),
        2,
        footprint=source.footprint,
    )

    proposals = simplify_pass(
        [source, cover, clone],
        {source.id: _mask_for_obj(source), cover.id: _mask_for_obj(cover), clone.id: _mask_for_obj(clone)},
        epsilon=1.0,
        max_error=1.0,
    )

    assert all(1 not in proposal.obj_ids for proposal in proposals)


def test_simplify_pass_collapses_long_straight_line_runs():
    top = " ".join(f"L{x} 10" for x in range(12, 71, 2))
    d = f"M10 10 {top} L70 40 L10 40 Z"
    obj = _obj_from_d(d)

    simplified_d = _optimized_d(obj)

    assert _command_count(simplified_d) < _command_count(d)
    assert "L70 10" in simplified_d
    assert "L70 40" in simplified_d
    assert "L10 40" in simplified_d


def test_simplified_path_data_threads_samples_to_subpath_sampling(monkeypatch):
    seen: list[int] = []
    original = simplify_module._sample_subpath

    def _recording_sample(tokens, *, samples):
        seen.append(samples)
        return original(tokens, samples=samples)

    monkeypatch.setattr(simplify_module, "_sample_subpath", _recording_sample)

    simplify_module.simplify_path_data(
        "M0 0 L20 0 L20 20 L0 20 Z",
        epsilon=1.0,
        max_error=1.0,
        samples=7,
    )

    assert seen == [7]


def test_simplify_pass_reduces_sampled_smooth_arc_to_curves():
    arc_points = []
    cx, cy, radius = 40.0, 40.0, 25.0
    for theta in np.linspace(math.pi, 0.0, 33):
        x = cx + radius * math.cos(float(theta))
        y = cy - radius * math.sin(float(theta))
        arc_points.append((x, y))

    commands = [f"M{arc_points[0][0]} {arc_points[0][1]}"]
    commands.extend(f"L{x} {y}" for x, y in arc_points[1:])
    commands.extend(["L65 40", "L15 40", "Z"])
    d = " ".join(commands)
    obj = _obj_from_d(d)

    simplified_d = _optimized_d(obj, epsilon=1.0, max_error=1.0)

    assert _command_count(simplified_d) < _command_count(d)
    assert "Q" in simplified_d


def test_simplify_pass_removes_short_linelets_from_curve_bearing_path():
    d = (
        "M329 113.5 L325.5 113 "
        "Q324.29 92.35 335.5 75 "
        "Q347.49 60 365 55.5 "
        "Q376.51 53.49 384.5 60 "
        "Q387.73 66.28 387.5 74 "
        "Q385.37 82.36 381.5 90 "
        "Q365.98 107.55 344 109.5 "
        "Q336.5 111.5 329 113.5 Z"
    )
    obj = _obj_from_d(d)

    simplified_d = _optimized_d(obj, epsilon=1.0, max_error=1.0)

    assert _command_count(simplified_d) < _command_count(d)
    assert all(length >= minimum_line_length(1.0) for length in _line_lengths(simplified_d))


def test_simplify_pass_removes_medium_linelets_from_curved_corpus_path():
    d = (
        "M247 417.5 Q241.23 417.72 237.5 413 "
        "L237.5 405 L242 400.5 "
        "Q256.31 399.74 253.5 413 "
        "L247 417.5 Z"
    )
    obj = _obj_from_d(d)

    simplified_d = _optimized_d(obj, epsilon=1.0, max_error=1.0)

    assert _command_count(simplified_d) <= _command_count(d)
    assert all(length >= minimum_line_length(1.0) for length in _line_lengths(simplified_d))


def test_simplify_pass_does_not_introduce_line_facets_into_all_curve_path():
    d = (
        "M319 113.5 "
        "Q310.09 112.49 302 109.5 "
        "Q277.89 107.5 262.5 88 "
        "Q259.31 81.29 257.5 74 "
        "Q257.05 65.28 262.5 59 "
        "Q270.27 53.67 281 55.5 "
        "Q287.66 57.61 294 60.5 "
        "Q316.77 76.92 320.5 104 "
        "Q320.15 108.78 319 113.5 Z"
    )
    obj = _obj_from_d(d)

    proposals = simplify_pass([obj], {obj.id: _mask_for_obj(obj, (140, 360))}, epsilon=1.0, max_error=1.0, cubic=True)

    assert proposals
    simplified = proposals[0].new_objects[0].current
    simplified_d = simplified.params["d"]
    assert "L" not in simplified_d
    assert _command_count(simplified_d) <= _command_count(d)


def test_simplify_pass_tries_quadratics_before_cubics():
    arc_points = []
    cx, cy, radius = 40.0, 40.0, 25.0
    for theta in np.linspace(math.pi, 0.0, 33):
        x = cx + radius * math.cos(float(theta))
        y = cy - radius * math.sin(float(theta))
        arc_points.append((x, y))

    commands = [f"M{arc_points[0][0]} {arc_points[0][1]}"]
    commands.extend(f"L{x} {y}" for x, y in arc_points[1:])
    commands.extend(["L65 40", "L15 40", "Z"])
    d = " ".join(commands)
    obj = _obj_from_d(d)

    quadratic_d = _optimized_d(obj, epsilon=1.0, max_error=1.0, cubic=False)
    cubic_enabled_d = _optimized_d(obj, epsilon=1.0, max_error=1.0, cubic=True)

    assert _command_count(cubic_enabled_d) < _command_count(d)
    assert _command_count(cubic_enabled_d) <= _command_count(quadratic_d)
    assert "Q" in cubic_enabled_d


def test_refinement_pipeline_keeps_only_the_constrained_linelet_simplifier():
    pass_names = [getattr(pass_fn, "__name__", pass_fn.__class__.__name__) for pass_fn in _refinement_passes(Options(refinement=True))]

    assert "simplify_pass" not in pass_names
    assert pass_names.count("linelet_simplify_pass") == 1


def test_refinement_detects_repetition_before_lossy_simplification():
    pass_names = [getattr(pass_fn, "__name__", pass_fn.__class__.__name__) for pass_fn in _refinement_passes(Options(refinement=True))]

    assert pass_names.count("simplify_pass") == 0
    assert pass_names.count("linelet_simplify_pass") == 1
    assert "straighten_pass" not in pass_names
    assert pass_names[-1:] == ["seams_pass"]


def test_simplify_rejects_a_local_bulge_beyond_max_error():
    """A small relative-area change can still visibly bow a long flat edge."""
    d = (
        "M406 243.5 L240 243.5 Q238.75 242.25 237.5 241 "
        "Q234.97 220.64 235.5 199 L409 197.5 Q409.75 198.25 410.5 199 "
        "Q411.13 220.17 408.5 240 Q407.47 241.84 406 243.5 Z"
    )
    obj = _obj_from_d(d)

    assert simplify_pass(
        [obj],
        {obj.id: _mask_for_obj(obj, (512, 512))},
        epsilon=1.5,
        max_error=1.0,
        max_boundary_error=1.0,
    ) == []


def test_simplify_pass_rejects_paths_more_complex_than_original_trace():
    current = Shape("path", {"d": _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)})
    original = Shape("path", {"d": "M8 8 L88 8 L88 88 L8 88 Z"})
    obj = RegionSample(1, current, FlatFill("#000000"), 0, original=original)

    proposals = simplify_pass([obj], {obj.id: _mask_for_obj(obj)}, epsilon=1.0, max_error=1.0)

    assert proposals == []


def test_simplify_pass_rejects_geometrically_bad_shorter_path():
    d = "M10 10 L50 10 L50 30 L62 30 L62 42 L50 42 L50 60 L10 60 Z"
    obj = _obj_from_d(d)
    proposals = simplify_pass([obj], {obj.id: _mask_for_obj(obj)}, epsilon=20.0)

    assert proposals == []


def test_optimize_accepts_geometrically_close_simplification_without_raster_gate():
    d = _dense_rounded_rect_d(8.0, 8.0, 88.0, 88.0, 18.0, samples=12)
    obj = _obj_from_d(d)
    bad_mask = np.zeros((96, 96), dtype=bool)
    proposals = simplify_pass([obj], {obj.id: bad_mask}, epsilon=1.0, max_error=1.0)

    assert proposals

    out = select_refinements(
        [obj],
        {obj.id: bad_mask},
        [lambda objects, masks: simplify_pass(objects, masks, epsilon=1.0, max_error=1.0)],
    )

    assert out[0].current != obj.current

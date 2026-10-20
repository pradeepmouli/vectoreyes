from types import SimpleNamespace

import numpy as np

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.refinement.framework import select_refinements
from vectormark.refinement.gate import rasterize
from vectormark.refinement.passes.seams import seams_pass
import vectormark.seam_detection as seams_module
from vectormark.refinement.region_sample import RegionSample, to_polygon
from vectormark.pipeline import Options, _refinement_passes
from vectormark._fitcurve import cubic_inflects


def _mask(shape: Shape, shape_hw: tuple[int, int] = (80, 120)) -> np.ndarray:
    return rasterize(to_polygon(shape), shape_hw)


def test_regions_close_rejects_separated_bounds_without_exact_path_distance():
    class Footprint:
        is_empty = False

        def __init__(self, bounds):
            self.bounds = bounds

        def distance(self, _other):
            raise AssertionError("far bounds must not require exact path distance")

    left = SimpleNamespace(footprint=Footprint((0.0, 0.0, 10.0, 10.0)))
    right = SimpleNamespace(footprint=Footprint((20.0, 0.0, 30.0, 10.0)))

    assert not seams_module._regions_close(left, right, tol=2.0)


def test_regions_close_rejects_overlapping_bounds_with_no_near_path_endpoints():
    class Footprint:
        is_empty = False
        bounds = (0.0, 0.0, 100.0, 100.0)

        def distance(self, _other):
            raise AssertionError("separated path endpoints must reject before exact distance")

    left = SimpleNamespace(
        shape=Shape("path", {"d": "M0 0 L0 100 L100 100 L100 0 Z"}),
        footprint=Footprint(),
    )
    right = SimpleNamespace(
        shape=Shape("path", {"d": "M40 40 L40 60 L60 60 L60 40 Z"}),
        footprint=Footprint(),
    )

    assert not seams_module._regions_close(left, right, tol=2.0)


def test_seams_pass_closes_adjacent_path_gap_to_shared_boundary():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L49.2 10 L49.2 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M50.4 10 L90 10 L90 70 L50.4 70 Z"}),
        FlatFill("#222222"),
        1,
    )
    left_trace = Shape("path", {"d": "M10 10 L49.8 10 L49.8 70 L10 70 Z"})
    right_trace = Shape("path", {"d": "M49.8 10 L90 10 L90 70 L49.8 70 Z"})

    out = select_refinements([left, right], {1: _mask(left_trace), 2: _mask(right_trace)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    assert "L49.8 10 L49.8 70" in by_id[1].current.params["d"]
    assert by_id[2].current.params["d"].startswith("M49.8 10")
    assert by_id[1].footprint.boundary.distance(by_id[2].footprint.boundary) == 0
    assert by_id[1].diagnostics["seams"]["selected"] == "midpoint"


def test_seams_pass_projects_shared_line_endpoints_exactly_after_trace_jitter():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L49 10 L49 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M51.4 10 L90 10 L90 70 L51.4 70 Z"}),
        FlatFill("#222222"),
        1,
    )

    out = select_refinements([left, right], {1: _mask(left.current), 2: _mask(right.current)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    assert "L50.2 10 L50.2 70" in by_id[1].current.params["d"]
    assert by_id[2].current.params["d"].startswith("M50.2 10")
    assert by_id[1].footprint.boundary.distance(by_id[2].footprint.boundary) == 0


def test_seams_pass_does_not_stitch_nearby_distinct_surface_fill_groups():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L49.2 10 L49.2 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
        diagnostics={"surface_fill": {"group": 1}},
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M50.4 10 L90 10 L90 70 L50.4 70 Z"}),
        FlatFill("#ffffff"),
        1,
        diagnostics={"surface_fill": {"group": 2}},
    )

    out = select_refinements([left, right], {1: _mask(left.current), 2: _mask(right.current)}, [seams_pass])

    assert [obj.current.params["d"] for obj in out] == [left.current.params["d"], right.current.params["d"]]


def test_seams_pass_adjusts_primitive_and_path_to_close_gap():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L49.7 10 L49.7 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = RegionSample(
        2,
        Shape("rect", {"x": 50.3, "y": 10.0, "w": 39.7, "h": 60.0}),
        FlatFill("#222222"),
        1,
    )
    left_trace = Shape("path", {"d": "M10 10 L50 10 L50 70 L10 70 Z"})
    right_trace = Shape("rect", {"x": 50.0, "y": 10.0, "w": 40.0, "h": 60.0})

    out = select_refinements([left, right], {1: _mask(left_trace), 2: _mask(right_trace)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    assert "L50 10 L50 70" in by_id[1].current.params["d"]
    assert by_id[2].current.params["x"] == 50.0
    assert by_id[2].current.params["w"] == 40.0
    assert by_id[1].footprint.boundary.distance(by_id[2].footprint.boundary) == 0


def test_seams_pass_true_ups_nearby_slanted_vertices_to_midpoints():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L49 20 L49 70 L10 60 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M50 20.6 L90 30 L90 80 L50 70.6 Z"}),
        FlatFill("#222222"),
        1,
    )
    left_trace = Shape("path", {"d": "M10 10 L49.5 20.3 L49.5 70.3 L10 60 Z"})
    right_trace = Shape("path", {"d": "M49.5 20.3 L90 30 L90 80 L49.5 70.3 Z"})

    out = select_refinements([left, right], {1: _mask(left_trace), 2: _mask(right_trace)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    assert "L49.5 20.3 L49.5 70.3" in by_id[1].current.params["d"]
    assert by_id[2].current.params["d"].startswith("M49.5 20.3")
    assert "L49.5 70.3" in by_id[2].current.params["d"]
    assert by_id[1].diagnostics["seams"]["selected"] == "vertex_midpoint"


def test_seams_pass_does_not_snap_curve_control_handles_to_seam():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 C48.9 18 40 35 49.2 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M50.4 10 L90 10 L90 70 L50.4 70 Z"}),
        FlatFill("#222222"),
        1,
    )
    left_trace = Shape("path", {"d": "M10 10 C48.9 18 40 35 49.8 70 L10 70 Z"})
    right_trace = Shape("path", {"d": "M49.8 10 L90 10 L90 70 L49.8 70 Z"})

    out = select_refinements([left, right], {1: _mask(left_trace), 2: _mask(right_trace)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    d = by_id[1].current.params["d"]
    assert "49.8 70" in d
    assert "C49.8 18" not in d
    for subpath in seams_module._parse_subpaths(d):
        current = None
        for command, values in subpath:
            if command == "M":
                current = np.array(values[:2], dtype=float)
            elif command == "L":
                current = np.array(values[:2], dtype=float)
            elif command == "Q":
                current = np.array(values[2:4], dtype=float)
            elif command == "C" and current is not None:
                ctrl = np.array([current, values[:2], values[2:4], values[4:6]], dtype=float)
                assert cubic_inflects(ctrl) == []
                current = np.array(values[4:6], dtype=float)


def test_seams_pass_removes_inflecting_cubics_after_vertex_edits():
    shape = Shape(
        "path",
        {
            "d": (
                "M166.88 484.26 "
                "C164.99 482.5 163.09 480.72 161.19 478.95 "
                "L156.96 472.09 L166.88 484.26 Z"
            )
        },
    )
    cleaned = seams_module._cleanup_inflecting_cubics(shape, max_error=1.0, line_epsilon=1.0)

    assert cleaned != shape
    for subpath in seams_module._parse_subpaths(cleaned.params["d"]):
        current = None
        for command, values in subpath:
            if command == "M":
                current = np.array(values[:2], dtype=float)
            elif command == "L":
                current = np.array(values[:2], dtype=float)
            elif command == "Q":
                current = np.array(values[2:4], dtype=float)
            elif command == "C" and current is not None:
                ctrl = np.array([current, values[:2], values[2:4], values[4:6]], dtype=float)
                assert cubic_inflects(ctrl) == []
                current = np.array(values[4:6], dtype=float)


def test_seams_pass_snaps_branch_child_against_sibling_region():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L49.2 10 L49.2 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    branch = RegionSample.branch(id=10, children=[left], z=0)
    right = RegionSample(
        2,
        Shape("path", {"d": "M50.4 10 L90 10 L90 70 L50.4 70 Z"}),
        FlatFill("#222222"),
        1,
    )
    left_trace = Shape("path", {"d": "M10 10 L49.8 10 L49.8 70 L10 70 Z"})
    right_trace = Shape("path", {"d": "M49.8 10 L90 10 L90 70 L49.8 70 Z"})

    out = select_refinements([branch, right], {10: _mask(left_trace), 2: _mask(right_trace)}, [seams_pass])

    by_id = {obj.id: obj for obj in out}
    child = by_id[10].children[0]
    assert "L49.8 10 L49.8 70" in child.current.params["d"]
    assert by_id[2].current.params["d"].startswith("M49.8 10")


def test_seams_pass_checks_composite_adjacency_before_descending(monkeypatch):
    left_near = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L49.2 10 L49.2 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    left_far = RegionSample(
        3,
        Shape("path", {"d": "M10 100 L30 100 L30 120 L10 120 Z"}),
        FlatFill("#333333"),
        0.1,
    )
    right_near = RegionSample(
        2,
        Shape("path", {"d": "M50.4 10 L90 10 L90 70 L50.4 70 Z"}),
        FlatFill("#222222"),
        1,
    )
    right_far = RegionSample(
        4,
        Shape("path", {"d": "M100 100 L120 100 L120 120 L100 120 Z"}),
        FlatFill("#444444"),
        1.1,
    )
    left_branch = RegionSample.branch(id=10, children=[left_near, left_far], z=0)
    right_branch = RegionSample.branch(id=20, children=[right_near, right_far], z=1)
    checked_pairs: list[tuple[int, int]] = []
    original_regions_close = seams_module._regions_close

    def _recording_regions_close(a, b, *, tol):
        checked_pairs.append((int(a.id), int(b.id)))
        return original_regions_close(a, b, tol=tol)

    monkeypatch.setattr(seams_module, "_regions_close", _recording_regions_close)

    seams_pass(
        [left_branch, right_branch],
        {
            10: rasterize(left_branch.footprint, (140, 140)),
            20: rasterize(right_branch.footprint, (140, 140)),
        },
    )

    assert checked_pairs.index((10, 20)) < checked_pairs.index((1, 2))
    assert (1, 2) in checked_pairs


def test_seams_pass_recursively_snaps_adjacent_children_within_composite_region():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L49.2 10 L49.2 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M50.4 10 L90 10 L90 70 L50.4 70 Z"}),
        FlatFill("#222222"),
        1,
    )
    branch = RegionSample.branch(id=10, children=[left, right], z=0)
    left_trace = Shape("path", {"d": "M10 10 L49.8 10 L49.8 70 L10 70 Z"})
    right_trace = Shape("path", {"d": "M49.8 10 L90 10 L90 70 L49.8 70 Z"})

    out = select_refinements(
        [branch],
        {10: _mask(left_trace) | _mask(right_trace)},
        [seams_pass],
    )

    out_branch = out[0]
    by_child_id = {child.id: child for child in out_branch.children}
    assert "L49.8 10 L49.8 70" in by_child_id[1].current.params["d"]
    assert by_child_id[2].current.params["d"].startswith("M49.8 10")


def test_seams_pass_applies_multiple_seams_touching_same_region_in_one_pass():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L29.2 10 L29.2 70 L10 70 Z"}),
        FlatFill("#111111"),
        0,
    )
    center = RegionSample(
        2,
        Shape("path", {"d": "M30.4 10 L59.2 10 L59.2 70 L30.4 70 Z"}),
        FlatFill("#222222"),
        1,
    )
    right = RegionSample(
        3,
        Shape("path", {"d": "M60.4 10 L90 10 L90 70 L60.4 70 Z"}),
        FlatFill("#333333"),
        2,
    )
    left_trace = Shape("path", {"d": "M10 10 L29.8 10 L29.8 70 L10 70 Z"})
    center_trace = Shape("path", {"d": "M29.8 10 L59.8 10 L59.8 70 L29.8 70 Z"})
    right_trace = Shape("path", {"d": "M59.8 10 L90 10 L90 70 L59.8 70 Z"})

    out = select_refinements(
        [left, center, right],
        {1: _mask(left_trace), 2: _mask(center_trace), 3: _mask(right_trace)},
        [seams_pass],
    )

    by_id = {obj.id: obj for obj in out}
    assert "L29.8 10 L29.8 70" in by_id[1].current.params["d"]
    assert by_id[2].current.params["d"].startswith("M29.8 10")
    assert "L59.8 10 L59.8 70" in by_id[2].current.params["d"]
    assert by_id[3].current.params["d"].startswith("M59.8 10")


def test_seams_pass_clusters_multi_region_junction_vertices_to_one_point():
    left = RegionSample(
        1,
        Shape("path", {"d": "M10 10 L49.2 10 L49.1 49.4 L10 49.4 Z"}),
        FlatFill("#111111"),
        0,
    )
    upper = RegionSample(
        2,
        Shape("path", {"d": "M49.8 10 L90 10 L90 49.8 L49.8 49.8 Z"}),
        FlatFill("#222222"),
        1,
    )
    lower = RegionSample(
        3,
        Shape("path", {"d": "M10 50.3 L50.3 50.3 L90 90 L10 90 Z"}),
        FlatFill("#333333"),
        2,
    )
    traces = {
        1: Shape("path", {"d": "M10 10 L49.8 10 L49.8 49.8 L10 49.8 Z"}),
        2: Shape("path", {"d": "M49.8 10 L90 10 L90 49.8 L49.8 49.8 Z"}),
        3: Shape("path", {"d": "M10 49.8 L49.8 49.8 L90 90 L10 90 Z"}),
    }

    out = select_refinements(
        [left, upper, lower],
        {obj_id: _mask(shape) for obj_id, shape in traces.items()},
        [seams_pass],
    )

    coordinates = []
    for obj in out:
        for subpath in seams_module._parse_subpaths(obj.current.params["d"]):
            for command, values in subpath:
                if command in {"M", "L"} and len(values) >= 2:
                    x, y = float(values[0]), float(values[1])
                    if 49.0 <= x <= 51.0 and 49.0 <= y <= 51.0:
                        coordinates.append((round(x, 6), round(y, 6)))

    assert len(set(coordinates)) == 1


def test_seams_pass_clusters_trace_delta_junction_vertices():
    left = RegionSample(
        1,
        Shape("path", {"d": "M0 0 L100.9 160.4 L0 161.25 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M0 161.25 L101.5 162 L0 260 Z"}),
        FlatFill("#222222"),
        1,
    )

    out = select_refinements(
        [left, right],
        {
            1: _mask(left.current, (280, 140)),
            2: _mask(right.current, (280, 140)),
        },
        [seams_pass],
    )

    coordinates = []
    for obj in out:
        for subpath in seams_module._parse_subpaths(obj.current.params["d"]):
            for command, values in subpath:
                if command in {"M", "L"} and len(values) >= 2:
                    x, y = float(values[0]), float(values[1])
                    if 99.0 <= x <= 103.0 and 159.0 <= y <= 163.0:
                        coordinates.append((round(x, 6), round(y, 6)))

    assert len(set(coordinates)) == 1


def test_seams_pass_clusters_unmodified_incident_junction_vertex():
    left = RegionSample(
        1,
        Shape("path", {"d": "M0 160 L100 160 L50 220 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M100.8 160.6 L200 160 L150 220 Z"}),
        FlatFill("#222222"),
        1,
    )
    top = RegionSample(
        3,
        Shape("path", {"d": "M50 80 L100.4 159.8 L150 80 Z"}),
        FlatFill("#333333"),
        2,
    )

    out = select_refinements(
        [left, right, top],
        {
            1: _mask(left.current, (240, 240)),
            2: _mask(right.current, (240, 240)),
            3: _mask(top.current, (240, 240)),
        },
        [seams_pass],
    )

    coordinates = []
    for obj in out:
        for subpath in seams_module._parse_subpaths(obj.current.params["d"]):
            for command, values in subpath:
                if command in {"M", "L"} and len(values) >= 2:
                    x, y = float(values[0]), float(values[1])
                    if 99.0 <= x <= 102.0 and 159.0 <= y <= 162.0:
                        coordinates.append((round(x, 6), round(y, 6)))

    assert len(coordinates) == 3
    assert len(set(coordinates)) == 1


def test_seams_pass_clusters_sketch_like_bottom_junction_with_duplicate_tip_segment():
    center = RegionSample(
        1,
        Shape("path", {"d": "M250 450.5 L247.5 450 L101 160.77 L396.74 161.26 L250 450.5 Z"}),
        FlatFill("#111111"),
        0,
    )
    right = RegionSample(
        2,
        Shape("path", {"d": "M252 447.5 L396.74 161.26 L498 161.75 L252 447.5 Z"}),
        FlatFill("#222222"),
        1,
    )
    left = RegionSample(
        3,
        Shape("path", {"d": "M246 447.5 L0.25 161.25 L101 160.77 L246.5 447 L246 447.5 Z"}),
        FlatFill("#333333"),
        2,
    )

    out = select_refinements(
        [center, right, left],
        {
            1: _mask(center.current, (480, 520)),
            2: _mask(right.current, (480, 520)),
            3: _mask(left.current, (480, 520)),
        },
        [seams_pass],
    )

    coordinates = []
    for obj in out:
        for subpath in seams_module._parse_subpaths(obj.current.params["d"]):
            for command, values in subpath:
                if command in {"M", "L"} and len(values) >= 2:
                    x, y = float(values[0]), float(values[1])
                    if 244.0 <= x <= 254.0 and 445.0 <= y <= 452.0:
                        coordinates.append((round(x, 6), round(y, 6)))

    assert len(coordinates) == 8
    assert len(set(coordinates)) == 1


def test_refinement_detects_clones_before_symmetry_and_runs_one_final_full_scene_stitch():
    pass_names = [getattr(pass_fn, "__name__", pass_fn.__class__.__name__) for pass_fn in _refinement_passes(Options(refinement=True))]

    assert pass_names.count("clones_pass") == 1
    assert pass_names.count("seams_pass") == 1
    assert pass_names[-6:] == [
        "clones_pass",
        "symmetry_pass",
        "linelet_simplify_pass",
        "smooth_pass",
        "corner_normalize_pass",
        "seams_pass",
    ]


def test_refinement_final_stitch_reconciles_all_shared_boundaries(monkeypatch):
    import vectormark.refinement.passes as passes

    calls: list[bool] = []

    def record_seams_pass(objects, masks, **kwargs):
        del objects, masks
        calls.append(kwargs["post_symmetry"])
        return []

    monkeypatch.setattr(passes, "seams_pass", record_seams_pass)
    final_stitch = _refinement_passes(Options(refinement=True))[-1]

    assert final_stitch([], {}) == []
    assert calls == [False]

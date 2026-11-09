import numpy as np
from vectoreyes.contour import outer_contour
from vectoreyes._fitcurve import cubic_inflects
from vectoreyes.fit import Shape, minimum_line_length, recognize_primitive


def _disk(cx, cy, r, size=60):
    yy, xx = np.ogrid[:size, :size]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def _rect(x0, y0, x1, y1, size=60):
    m = np.zeros((size, size), bool); m[y0:y1, x0:x1] = True
    return m


def test_recognizes_circle():
    c = outer_contour(_disk(30, 30, 18))
    shape = recognize_primitive(c, epsilon=1.0)
    assert shape is not None and shape.kind == "circle"
    assert abs(shape.params["r"] - 18) < 1.5


def test_recognizes_circle_despite_one_anti_aliased_outlier():
    angles = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
    contour = np.array([(30.0 + 18.0 * np.cos(angle), 30.0 + 18.0 * np.sin(angle)) for angle in angles])
    contour[0] += np.array((1.7, 0.0))
    contour = np.vstack((contour, contour[0]))

    shape = recognize_primitive(contour, epsilon=1.5)

    assert shape is not None and shape.kind == "circle"


def test_recognizes_axis_aligned_rect():
    c = outer_contour(_rect(10, 14, 50, 40))
    shape = recognize_primitive(c, epsilon=1.0)
    assert shape is not None and shape.kind == "rect"
    assert abs(shape.params["w"] - 40) < 2 and abs(shape.params["h"] - 26) < 2


def test_rejects_half_ellipse_region():
    # dome = ellipse with the bottom flattened -> NOT a whole primitive
    yy, xx = np.ogrid[:60, :60]
    dome = (((xx - 30) ** 2 / 324 + (yy - 40) ** 2 / 400) <= 1) & (yy <= 40)
    c = outer_contour(dome)
    assert recognize_primitive(c, epsilon=1.0) is None


from vectoreyes.fit import recognize_polygon


def _trapezoid(size=60):
    m = np.zeros((size, size), bool)
    for y in range(10, 40):
        half = int(20 - (y - 10) * 0.3)
        m[y, 30 - half:30 + half] = True
    return m


def test_recognizes_trapezoid_as_polygon():
    c = outer_contour(_trapezoid())
    shape = recognize_polygon(c, epsilon=1.2)
    assert shape is not None and shape.kind == "polygon"
    assert 4 <= len(shape.params["points"]) <= 5


def test_polygon_rejects_curved_region():
    c = outer_contour(_disk_local())
    assert recognize_polygon(c, epsilon=1.2) is None


def _disk_local(size=60):
    yy, xx = np.ogrid[:size, :size]
    return ((xx - 30) ** 2 + (yy - 30) ** 2) <= 18 * 18


from vectoreyes.fit import fit_path
from vectoreyes.path_sampling import _parse_subpaths


def _command_count(d: str) -> int:
    return sum(1 for ch in d if ch in "MLQCAZ")


def _line_lengths(d: str) -> list[float]:
    import math

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


def _cubic_controls(d: str):
    controls = []
    current = start = None
    for subpath in _parse_subpaths(d):
        for command, values in subpath:
            if command == "M":
                current = np.array(values[:2], dtype=float)
                start = current
            elif command == "L":
                current = np.array(values[:2], dtype=float)
            elif command == "Q":
                current = np.array(values[2:4], dtype=float)
            elif command == "C" and current is not None:
                ctrl = np.array(
                    [
                        current,
                        values[:2],
                        values[2:4],
                        values[4:6],
                    ],
                    dtype=float,
                )
                controls.append(ctrl)
                current = np.array(values[4:6], dtype=float)
            elif command == "Z":
                current = start
    return controls


def _quadratic_join_dots(d: str) -> list[float]:
    dots = []
    previous = None
    current = start = None
    for subpath in _parse_subpaths(d):
        for command, values in subpath:
            if command == "M":
                current = np.array(values[:2], dtype=float)
                start = current
                previous = None
            elif command == "Q" and current is not None:
                ctrl = np.array(values[:2], dtype=float)
                end = np.array(values[2:4], dtype=float)
                if previous is not None and np.allclose(previous[2], current):
                    incoming = current - previous[1]
                    outgoing = ctrl - current
                    incoming_norm = np.hypot(*incoming)
                    outgoing_norm = np.hypot(*outgoing)
                    if incoming_norm > 1e-9 and outgoing_norm > 1e-9:
                        dots.append(float((incoming @ outgoing) / (incoming_norm * outgoing_norm)))
                previous = (current, ctrl, end)
                current = end
            elif command == "L":
                current = np.array(values[:2], dtype=float)
                previous = None
            elif command == "C":
                current = np.array(values[4:6], dtype=float)
                previous = None
            elif command == "Z":
                current = start
                previous = None
    return dots


def test_fit_path_of_square_uses_only_lines():
    mask = np.zeros((30, 30), bool); mask[6:24, 6:24] = True
    c = outer_contour(mask)
    shape = fit_path(c, epsilon=1.0, max_error=0.8)
    assert shape.kind == "path"
    assert "C" not in shape.params["d"]      # all straight -> only line ops
    assert shape.params["d"].strip().endswith("Z")


def test_fit_path_of_dome_uses_curve():
    yy, xx = np.ogrid[:80, :80]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 1600) <= 1) & (yy <= 55)
    c = outer_contour(dome)
    shape = fit_path(c, epsilon=1.0, max_error=0.8)
    # curved top -> inflection-free quadratic arcs (Q), never cubic (C) by default
    assert shape.kind == "path" and "Q" in shape.params["d"] and "C" not in shape.params["d"]
    # opt-in cubic flag emits inflection-guarded cubic runs even when quadratics are shorter
    cshape = fit_path(c, epsilon=1.0, max_error=0.8, cubic=True)
    assert cshape.kind == "path"
    cubics = _cubic_controls(cshape.params["d"])
    assert cubics
    assert all(cubic_inflects(ctrl) == [] for ctrl in cubics)


def test_fit_path_does_not_emit_short_linelets_between_forced_corners_on_smooth_curve():
    angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    contour = np.array(
        [(50.0 + 30.0 * np.cos(theta), 50.0 + 30.0 * np.sin(theta)) for theta in angles]
        + [(80.0, 50.0)],
        dtype=float,
    )

    shape = fit_path(
        contour,
        epsilon=1.0,
        max_error=1.0,
        forced_corners=np.array([contour[0], contour[1]], dtype=float),
    )

    assert all(length >= minimum_line_length(1.0) for length in _line_lengths(shape.params["d"]))


def test_fit_path_keeps_nearly_linear_spans_as_curves_inside_curved_runs():
    points = np.array(
        [
            (249.66, -0.73),
            (237.53, -0.74),
            (224.65, -0.54),
            (216.37, -0.42),
            (208.09, -0.21),
            (202.56, -0.07),
            (197.04, 0.11),
            (182.93, -1.07),
            (171.25, 7.05),
            (249.66, -0.73),
        ],
        dtype=float,
    )

    shape = fit_path(points, epsilon=1.5, max_error=1.0)

    assert "Q" in shape.params["d"]
    assert "L" in shape.params["d"]


def test_fit_path_cubic_option_emits_inflection_guarded_cubics_for_smooth_runs():
    top = [(10.0 + 80.0 * t, 10.0 + 30.0 * (t**3)) for t in np.linspace(0.0, 1.0, 80)]
    contour = np.array([*top, (90.0, 70.0), (10.0, 70.0), top[0]], dtype=float)

    shape = fit_path(contour, epsilon=0.5, max_error=0.8, cubic=True)
    cubics = _cubic_controls(shape.params["d"])

    assert cubics
    assert all(cubic_inflects(ctrl) == [] for ctrl in cubics)


def test_fit_path_does_not_introduce_medium_line_facets_on_smooth_leaf():
    from vectoreyes.refinement.region_sample import flatten_points

    source = Shape(
        "path",
        {
            "d": (
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
        },
    )
    contour = np.asarray(flatten_points(source, samples=12), dtype=float)
    contour = np.vstack([contour, contour[0]])

    fitted = fit_path(contour, epsilon=1.0, max_error=1.0, prefer_simple_curves=True)

    assert fitted.params["d"].count("L") == 0


def test_fit_path_smooths_quadratic_joins_on_smooth_leaf():
    from vectoreyes.refinement.region_sample import flatten_points

    source = Shape(
        "path",
        {
            "d": (
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
        },
    )
    contour = np.asarray(flatten_points(source, samples=12), dtype=float)
    contour = np.vstack([contour, contour[0]])

    fitted = fit_path(contour, epsilon=1.0, max_error=1.0, prefer_simple_curves=True)

    assert fitted.params["d"].count("L") == 0
    assert _quadratic_join_dots(fitted.params["d"])
    # A dense quadratic run has shared controls at adjacent joins.  Those
    # joins are deliberately left alone rather than receiving order-dependent
    # partial smoothing; the raw fit remains smooth enough without facets.
    assert min(_quadratic_join_dots(fitted.params["d"])) > 0.9


def test_quadratic_to_line_smoothing_only_adjusts_quadratic_control():
    from vectoreyes.fit import _smooth_quadratic_path_d

    d = "M0 0 Q10 10 20 0 L30 0 L40 0 Z"
    smoothed = _smooth_quadratic_path_d(d)

    assert smoothed.startswith("M0 0 Q5.86 0 20 0")
    assert "L30 0" in smoothed
    assert "L40 0" in smoothed
    assert smoothed.endswith("Z")
    assert _smooth_quadratic_path_d(smoothed) == smoothed

"""Published approximations need a bound on the analytic normal offsets."""

import numpy as np
import pytest


def api():
    try:
        from vectoreyes.bands import approximate_band_offsets
    except ImportError:
        pytest.fail("certified SVG offset approximation is missing")
    return approximate_band_offsets


def evaluate(control, t):
    p = np.asarray(control)
    t = np.asarray(t)[:, None]
    return (1-t)**3*p[0] + 3*(1-t)**2*t*p[1] + 3*(1-t)*t*t*p[2] + t**3*p[3]


def test_straight_band_is_exact_without_segment_proliferation():
    result = api()(((0, 0), (30, 0), (70, 0), (100, 0)), 8, tolerance=0.02)
    assert len(result.first) == len(result.second) == 1
    assert np.asarray(result.first)[..., 1] == pytest.approx(-4)
    assert np.asarray(result.second)[..., 1] == pytest.approx(4)
    assert result.error_bound <= 0.02


def test_exact_polyline_compaction_preserves_turns_and_reversals():
    import vectoreyes.band_offsets as offsets
    compact = getattr(offsets, 'exact_polyline_vertices', None)
    assert compact is not None, 'shared exact source-guard compaction is missing'
    assert compact(((0, 0), (1, 0), (2, 0), (1, 0), (1, 1), (1, 2))) == (
        (0, 0), (2, 0), (1, 0), (1, 2))


def test_curved_svg_offsets_stay_within_the_certified_error():
    control = ((0, 0), (25, 0), (70, 60), (100, 80))
    result = api()(control, 12, tolerance=0.03)
    assert result.error_bound <= 0.03
    assert len(result.first) == len(result.second) == len(result.parameter_ranges)
    p = np.asarray(control, dtype=float)
    for index, (start, end) in enumerate(result.parameter_ranges):
        local = np.linspace(0, 1, 1001)
        t = (start + local * (end-start))[:, None]
        center = evaluate(control, t[:, 0])
        derivative = 3*(1-t)**2*(p[1]-p[0]) + 6*(1-t)*t*(p[2]-p[1]) + 3*t*t*(p[3]-p[2])
        normal = np.c_[-derivative[:, 1], derivative[:, 0]] / np.linalg.norm(derivative, axis=1)[:, None]
        for sign, segments in ((-1, result.first), (1, result.second)):
            error = np.linalg.norm(evaluate(segments[index], local) - (center + sign*6*normal), axis=1).max()
            assert error <= result.error_bound + 1e-8
    for segments in (result.first, result.second):
        for before, after in zip(segments, segments[1:]):
            assert before[-1] == after[0]
            a = np.asarray(before[-1]) - before[-2]
            b = np.asarray(after[1]) - after[0]
            assert abs(a[0]*b[1] - a[1]*b[0]) < 1e-8


def test_cusp_is_not_silently_exported():
    with pytest.raises(ValueError, match="singular|certif"):
        api()(((0, 0), (1, 0), (-1, 0), (0, 0)), 4, tolerance=0.02)


@pytest.mark.parametrize("width,tolerance", [(0, 0.1), (-1, 0.1), (2, 0), (2, float('nan'))])
def test_invalid_materialization_settings(width, tolerance):
    with pytest.raises(ValueError):
        api()(((0, 0), (1, 0), (2, 0), (3, 0)), width, tolerance=tolerance)


def test_certification_has_an_explicit_work_limit():
    with pytest.raises(ValueError, match="work limit"):
        api()(((0, 0), (25, 0), (70, 60), (100, 80)), 12, tolerance=0.03, max_nodes=1)


@pytest.mark.parametrize("max_nodes", [0, -1, 1.5, True])
def test_invalid_certification_work_limit(max_nodes):
    with pytest.raises(ValueError, match="work limit"):
        api()(((0, 0), (1, 0), (2, 0), (3, 0)), 2, max_nodes=max_nodes)

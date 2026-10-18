"""The corner kernel fits inside fixed boundaries, without scene state."""

import numpy as np
import pytest

from vectormark._fitcurve import fit_quadratic_once
from vectormark.fit import _fmt, _quadratic_max_residual
from vectormark.path_geometry import QuadraticCommand


@pytest.mark.parametrize("max_error", [0, 0.1, 1, 3])
def test_corner_span_matches_existing_rounded_control_acceptance(max_error):
    from vectormark.corner_fitting import fit_corner_span

    points = np.array([[10.001, 0], [12, 0.2], [13.5, 1.8], [14.003, 4]])
    original = points.copy()
    fitted, _, _ = fit_quadratic_once(points, max_error)
    control = tuple(float(_fmt(value)) for value in fitted[1])
    expected = np.array([points[0], control, points[-1]])
    result = fit_corner_span(points, start=tuple(points[0]), end=tuple(points[-1]),
        command_id="r1.p0.c3", max_error=max_error)
    if _quadratic_max_residual(points, expected) > max_error:
        assert result is None
    else:
        assert result == QuadraticCommand(id="r1.p0.c3", end=tuple(points[-1]), control=control)
    np.testing.assert_array_equal(points, original)


def test_corner_span_cannot_move_boundaries_to_reduce_error():
    from vectormark.corner_fitting import fit_corner_span

    points = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    assert fit_corner_span(points, start=(0, 3), end=(2, 2),
        command_id="r1.p0.c1", max_error=0.1) is None


@pytest.mark.parametrize("points", [[], [[0, 0]], [[0, 0], [1, 1]]])
def test_corner_span_without_enough_samples_is_not_fit(points):
    from vectormark.corner_fitting import fit_corner_span

    assert fit_corner_span(points, start=(0, 0), end=(1, 1),
        command_id="r1.p0.c1", max_error=1) is None


@pytest.mark.parametrize("points", [[[0, 0], [1, float("nan")], [2, 2]], [[0], [1], [2]]])
def test_corner_span_rejects_invalid_sample_coordinates(points):
    from vectormark.corner_fitting import fit_corner_span

    with pytest.raises(ValueError, match="finite two-dimensional points"):
        fit_corner_span(points, start=(0, 0), end=(2, 2), command_id="r1.p0.c1", max_error=1)


@pytest.mark.parametrize("max_error", [-1, float("nan"), float("inf")])
def test_corner_span_rejects_invalid_error_budget(max_error):
    from vectormark.corner_fitting import fit_corner_span

    with pytest.raises(ValueError, match="finite and non-negative"):
        fit_corner_span([[0, 0], [1, 1], [2, 2]], start=(0, 0), end=(2, 2),
            command_id="r1.p0.c1", max_error=max_error)


def test_corner_normalization_delegates_to_typed_span_kernel(monkeypatch):
    import vectormark.corner_fitting as fitting
    from vectormark.optimizer.corners import normalize_corners_path_d, path_corner_diagnostics

    calls = []
    execute = fitting.fit_corner_span
    def record(points, **kwargs):
        result = execute(points, **kwargs)
        calls.append((kwargs, result))
        return result
    monkeypatch.setattr(fitting, "fit_corner_span", record)
    d = "M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z"
    declared = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (14, 4)),))
    result, diagnostics = normalize_corners_path_d(d, max_error=1, diagnostics=declared)
    assert len(calls) == 1
    assert calls[0][0]["start"] == (10, 0)
    assert calls[0][0]["end"] == (14, 4)
    assert isinstance(calls[0][1], QuadraticCommand)
    assert result != d
    assert diagnostics["normalized"] == [declared["spans"][0]["corner_id"]]

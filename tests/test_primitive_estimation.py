"""Optional numerical candidates must not break the retained drawing."""

from types import SimpleNamespace

import numpy as np
import pytest

from vectoreyes.fit import Shape, recognize_primitive
from vectoreyes.primitive_fitting import _polygon_exterior_points
from vectoreyes.refinement.region_sample import to_polygon


def _ellipse_points():
    angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    return np.column_stack((30.0 + 20.0 * np.cos(angles), 30.0 + 10.0 * np.sin(angles)))


def test_guide_contour_does_not_crash_or_change_during_primitive_recognition():
    # Actual guide-retrace contour that triggers complex phi %= pi in skimage
    # 0.26. Keep the geometry compact and let the same Skia sampling reproduce it.
    shape = Shape("path", {"d": (
        "M38 48.5 L20 48.5 Q13.9 44.66 15.5 38 L15.5 20 "
        "Q19.34 13.9 26 15.5 L44 15.5 Q50.1 19.34 48.5 26 "
        "L48.5 43 Q46.14 49.76 38 48.5 Z"
    )})
    points, _footprint = _polygon_exterior_points(to_polygon(shape))
    before = points.copy()

    assert recognize_primitive(points, epsilon=0.1) is None

    np.testing.assert_array_equal(points, before)


def test_complex_modulo_estimator_failure_is_reported_and_rejected(monkeypatch, caplog):
    def complex_modulo(_points):
        # The exact numerical operation that fails inside skimage; avoid
        # depending on platform-specific eig roundoff to trigger the failure.
        angle = np.complex128(0)
        angle %= np.pi

    monkeypatch.setattr("vectoreyes.fit.EllipseModel.from_estimate", complex_modulo)

    assert recognize_primitive(_ellipse_points(), epsilon=0.1) is None
    rejection = next(record for record in caplog.records if getattr(record, "primitive", None) == "ellipse")
    assert rejection.reason == "complex_parameters"


def test_valid_ellipse_candidate_still_recognized():
    shape = recognize_primitive(_ellipse_points(), epsilon=0.1)

    assert shape is not None and shape.kind == "ellipse"
    assert shape.params == pytest.approx({"cx": 30, "cy": 30, "rx": 20, "ry": 10})


@pytest.mark.parametrize("params", [
    {"center": (np.nan, 30), "axis_lengths": (20, 10), "theta": 0},
    {"center": (30, 30), "axis_lengths": (-20, 10), "theta": 0},
    {"center": (30, 30), "axis_lengths": (20, 10), "theta": 1j},
])
def test_invalid_ellipse_parameters_are_rejected(monkeypatch, caplog, params):
    monkeypatch.setattr("vectoreyes.fit.EllipseModel.from_estimate", lambda _points: SimpleNamespace(**params))

    assert recognize_primitive(_ellipse_points(), epsilon=0.1) is None

    rejection = next(record for record in caplog.records if getattr(record, "primitive", None) == "ellipse")
    assert rejection.reason == "invalid_parameters"


def test_unrelated_estimator_type_error_is_not_hidden(monkeypatch):
    def fail(_points):
        raise TypeError("unexpected estimator failure")

    monkeypatch.setattr("vectoreyes.fit.EllipseModel.from_estimate", fail)
    with pytest.raises(TypeError, match="unexpected estimator failure"):
        recognize_primitive(_ellipse_points(), epsilon=0.1)

"""Constant-width evidence is normal separation, not a matching axis extent."""

import numpy as np
import pytest


def api():
    try:
        from vectoreyes.bands import BandRequest, detect_band, reconstruct_band
    except ImportError:
        pytest.fail("constant-width band detection and source-guarded reconstruction are missing")
    return BandRequest, detect_band, reconstruct_band


def arc_pair(width=8.0):
    t = np.linspace(-0.8, 0.8, 101)
    return tuple(map(tuple, np.c_[np.cos(t), np.sin(t)] * 100)), tuple(map(tuple, np.c_[np.cos(t), np.sin(t)] * (100 + width)))


def test_curved_normal_offsets_are_detected_with_reversed_boundary_order():
    Request, detect, _ = api()
    a, b = arc_pair()
    result = detect(Request(a, b[::-1]))
    assert result.accepted, result.reason
    assert result.width == pytest.approx(8, abs=0.05)
    assert result.width_variation < 0.1


def test_rotation_does_not_change_band_evidence():
    Request, detect, _ = api()
    a, b = arc_pair()
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    first = detect(Request(a, b))
    second = detect(Request(tuple(map(tuple, np.asarray(a) @ rotation)), tuple(map(tuple, np.asarray(b) @ rotation))))
    assert second.accepted
    assert second.width == pytest.approx(first.width)


def test_tangent_support_ignores_subpixel_staircase_directions():
    Request, detect, _ = api()
    t = np.linspace(-0.8, 0.8, 401)
    directions = np.c_[np.cos(t), np.sin(t)]
    a = directions * 100
    b = directions * (108 + 0.08 * (-1.0) ** np.arange(len(t)))[:, None]
    result = detect(Request(tuple(map(tuple, a)), tuple(map(tuple, b))))
    assert result.accepted, result.reason
    assert result.width == pytest.approx(8, abs=0.2)


def test_taper_is_not_silently_classified_as_constant_width():
    Request, detect, _ = api()
    x = np.linspace(0, 100, 101)
    result = detect(Request(tuple(zip(x, x * 0)), tuple(zip(x, 5 + x * 0.1)), width_tolerance=0.5))
    assert not result.accepted
    assert result.reason == "width_variation"


def test_crossing_boundaries_are_rejected():
    Request, detect, _ = api()
    x = np.linspace(0, 100, 101)
    result = detect(Request(tuple(zip(x, x * 0)), tuple(zip(x, x * 0.2 - 10))))
    assert not result.accepted


def test_reconstruction_uses_both_original_polylines_as_guards():
    Request, _, reconstruct = api()
    a, b = arc_pair()
    result = reconstruct(Request(a, b, max_error=0.3))
    assert result.accepted, result.reason
    assert max(result.source_errors) <= 0.3
    assert len(result.centerline_controls) == 4
    # A declared wrong width cannot pass just because the current fit is close.
    rejected = reconstruct(Request(a, b, max_error=0.3), width=20)
    assert not rejected.accepted
    assert rejected.reason == "source_fidelity"


def test_reconstruction_returns_source_guarded_svg_curves_not_sample_only_evidence():
    Request, _, reconstruct = api()
    a, b = arc_pair()
    result = reconstruct(Request(a, b, max_error=0.3))
    assert result.accepted, result.reason
    assert result.offsets is not None
    assert result.offsets.error_bound <= 0.02
    assert result.metric == "certified_svg_and_analytic_offset_to_source_upper_bound"
    # Check the actual returned cubic geometry independently, not the old
    # uniformly sampled analytic offset used before SVG materialization.
    for source, curves, bound in zip((a, b), (result.offsets.first, result.offsets.second), result.source_errors):
        t = np.linspace(0, 1, 1001)[:, None]
        points = np.vstack([(1-t)**3*np.array(c)[0] + 3*(1-t)**2*t*np.array(c)[1]
                            + 3*(1-t)*t*t*np.array(c)[2] + t**3*np.array(c)[3] for c in curves])
        starts = np.asarray(source[:-1])
        delta = np.diff(np.asarray(source), axis=0)
        parameter = np.clip(np.sum((points[:, None] - starts) * delta, axis=2) / np.sum(delta*delta, axis=1), 0, 1)
        distances = np.linalg.norm(points[:, None] - (starts + parameter[..., None]*delta), axis=2).min(axis=1)
        assert distances.max() <= bound


def test_negative_signed_width_keeps_material_side_ownership():
    Request, _, reconstruct = api()
    a, b = arc_pair()
    result = reconstruct(Request(a[::-1], b[::-1], max_error=0.3))
    assert result.accepted, result.reason
    assert np.linalg.norm(np.asarray(result.offsets.first[0][0]) - a[-1]) < 0.3
    assert np.linalg.norm(np.asarray(result.offsets.second[0][0]) - b[-1]) < 0.3


def test_engine_is_the_entry_point_for_the_same_detection_and_proposal():
    Request, detect, reconstruct = api()
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    a, b = arc_pair()
    request = Request(a, b)
    engine = PythonDrawingEngine()
    assert engine.detect_band(request) == detect(request)
    assert engine.reconstruct_band(request) == reconstruct(request)


def test_explicit_width_can_bypass_detection_admission_but_not_source_guard():
    Request, detect, reconstruct = api()
    t = np.linspace(-0.5, 0.5, 101)
    directions = np.c_[np.cos(t), np.sin(t)]
    a = directions * 100
    b = directions * (108 + 0.08 * np.sin(t * 10))[:, None]
    request = Request(tuple(map(tuple, a)), tuple(map(tuple, b)), width_tolerance=0.01, max_error=0.3)
    assert not detect(request).accepted
    result = reconstruct(request, width=8)
    assert result.accepted, result.reason


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_error_budget_is_rejected(value):
    Request, _, _ = api()
    a, b = arc_pair()
    with pytest.raises(ValueError):
        Request(a, b, max_error=value)

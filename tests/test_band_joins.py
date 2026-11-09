"""Band interiors must reconnect without moving existing corner endpoints."""

import numpy as np
import pytest


def api():
    try:
        from vectoreyes.band_joins import join_band_offsets
    except ImportError:
        pytest.fail("source-guarded, fixed-endpoint band joins are missing")
    return join_band_offsets


def test_straight_band_joins_keep_all_four_corner_endpoints_exact():
    from vectoreyes.band_offsets import approximate_band_offsets
    offsets = approximate_band_offsets(((10, 0), (35, 0), (65, 0), (90, 0)), 8)
    x = np.linspace(0, 100, 101)
    first = tuple(zip(x, np.full_like(x, -4)))
    second = tuple(zip(x, np.full_like(x, 4)))
    result = api()(offsets, first, second, max_error=0.1)
    assert result.accepted, result.reason
    assert result.first[0][0] == first[0]
    assert result.first[-1][-1] == first[-1]
    assert result.second[0][0] == second[0]
    assert result.second[-1][-1] == second[-1]
    assert max(result.source_errors) <= 0.1
    assert len(result.first) == len(offsets.first) + 2


def test_joined_curves_have_exact_position_and_tangent_continuity():
    from vectoreyes.bands import BandRequest, reconstruct_band
    t = np.linspace(-0.8, 0.8, 161)
    first = np.c_[np.cos(t), np.sin(t)] * 100
    second = np.c_[np.cos(t), np.sin(t)] * 108
    proposal = reconstruct_band(BandRequest(tuple(map(tuple, first[20:-20])), tuple(map(tuple, second)), max_error=0.3))
    assert proposal.accepted
    result = api()(proposal.offsets, tuple(map(tuple, first)), tuple(map(tuple, second[::-1])), max_error=0.3)
    assert result.accepted, result.reason
    for curves in (result.first, result.second):
        for left, right in zip(curves, curves[1:]):
            assert left[-1] == right[0]
            a = np.asarray(left[-1]) - left[-2]
            b = np.asarray(right[1]) - right[0]
            assert float(a[0]*b[1] - a[1]*b[0]) == pytest.approx(0, abs=1e-8)
            assert a @ b > 0


def test_incompatible_join_cannot_bypass_the_source_guard():
    from vectoreyes.band_offsets import approximate_band_offsets
    offsets = approximate_band_offsets(((10, 0), (35, 0), (65, 0), (90, 0)), 8)
    x = np.linspace(0, 100, 101)
    first = tuple(zip(x, np.full_like(x, -20)))
    second = tuple(zip(x, np.full_like(x, 20)))
    result = api()(offsets, first, second, max_error=0.2)
    assert not result.accepted
    assert result.reason == "source_fidelity"
    assert max(result.source_errors) > 0.2


def test_retained_endpoints_are_not_replaced_by_nearby_source_samples():
    from vectoreyes.band_offsets import approximate_band_offsets
    offsets = approximate_band_offsets(((10, 0), (35, 0), (65, 0), (90, 0)), 8)
    x = np.linspace(0, 100, 101)
    sources = (tuple(zip(x, np.full_like(x, -4))), tuple(zip(x, np.full_like(x, 4))))
    retained = (((0.0, -4.003), (100.0, -4.002)), ((0.0, 4.001), (100.0, 4.004)))
    result = api()(offsets, *sources, max_error=0.1, endpoints=retained)
    assert result.accepted
    for curves, ends in zip((result.first, result.second), retained):
        assert curves[0][0] == ends[0]
        assert curves[-1][-1] == ends[-1]


def test_declared_endpoints_do_not_redefine_the_source_guard():
    from vectoreyes.band_offsets import approximate_band_offsets
    offsets = approximate_band_offsets(((10, 0), (35, 0), (65, 0), (90, 0)), 8)
    x = np.linspace(0, 100, 101)
    sources = (tuple(zip(x, np.full_like(x, -4))), tuple(zip(x, np.full_like(x, 4))))
    retained = (((0.0, -20.0), (100.0, -20.0)), ((0.0, 20.0), (100.0, 20.0)))
    result = api()(offsets, *sources, max_error=0.1, endpoints=retained)
    assert not result.accepted
    assert result.reason == "source_fidelity"


def test_engine_exposes_the_same_join_primitive():
    from vectoreyes.band_offsets import approximate_band_offsets
    from vectoreyes.bands import BandRequest
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    offsets = approximate_band_offsets(((10, 0), (35, 0), (65, 0), (90, 0)), 8)
    x = np.linspace(0, 100, 101)
    first = tuple(zip(x, np.full_like(x, -4)))
    second = tuple(zip(x, np.full_like(x, 4)))
    request = BandRequest(first, second, max_error=0.1)
    assert PythonDrawingEngine().join_band(request, offsets) == api()(offsets, first, second, max_error=0.1)

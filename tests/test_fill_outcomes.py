import json

import numpy as np
import pytest

from tests.test_fill_fit import _hramp, _solid
from vectoreyes.candidate import FlatFill, LinearGradientFill, RadialGradientFill
from vectoreyes.fill_fit import fit_fill


def test_disabled_gradients_are_a_forced_choice_not_supported_flat_evidence(monkeypatch):
    mask, rgb = _hramp(h=8, w=20)
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled gradient search must not execute')
    monkeypatch.setattr('vectoreyes.fill_fit._best_parametric', forbidden)
    result = fit_fill(mask, rgb, flat_hex='#123456', allow_gradients=False)
    assert result.fill == FlatFill('#123456')
    assert result.status == 'forced_flat'
    assert result.reason == 'gradients_disabled'
    assert result.diagnostics()['gradient_searched'] is False


def test_uniform_flat_selection_records_its_evidence():
    mask, rgb = _solid()
    result = fit_fill(mask, rgb, flat_hex='#1E78C8')
    assert result.fill == FlatFill('#1E78C8')
    assert result.status == 'flat_evidence'
    assert result.reason == 'dominant_rgb'
    assert result.diagnostics()['dominant_rgb_fraction'] == 1.


def test_low_spread_flat_evidence_is_distinct_from_dominant_rgb():
    mask, rgb = _hramp(h=8, w=20, c0=(100, 100, 100), c1=(102, 100, 100))
    result = fit_fill(mask, rgb, flat_hex='#646464')
    assert result.status == 'flat_evidence'
    assert result.reason == 'low_oklab_spread'
    assert result.diagnostics()['dominant_rgb_fraction'] < .9
    assert result.diagnostics()['oklab_spread'] < .005


def test_empty_support_does_not_claim_a_successful_fit():
    mask, rgb = _solid()
    result = fit_fill(np.zeros_like(mask), rgb, flat_hex='#1E78C8')
    assert result.status == 'empty_support'
    assert result.sampled_pixels == 0
    assert result.fill == FlatFill('#1E78C8')


@pytest.mark.parametrize('candidate,reason', [
    (None, 'no_gradient_candidate'),
    (({'kind': 'linear', 'geometry': {'x1': 0., 'y1': 0., 'x2': 20., 'y2': 0.},
       'stops': [(0., '#000000'), (1., '#ffffff')]}, .2, .15), 'gradient_error_exceeded'),
    (({'kind': 'linear', 'geometry': {'x1': 0., 'y1': 0., 'x2': 20., 'y2': 0.},
       'stops': [(0., '#000000'), (1., '#ffffff')]}, float('nan'), .15), 'invalid_gradient_error'),
])
def test_failed_gradient_is_an_accountable_fallback(monkeypatch, candidate, reason):
    mask, rgb = _hramp(h=8, w=20)
    monkeypatch.setattr('vectoreyes.fill_fit._best_parametric', lambda *args: candidate)
    result = fit_fill(mask, rgb, flat_hex='#123456', max_gradient_de=.02)
    assert result.fill == FlatFill('#123456')
    assert result.status == 'fallback_flat'
    assert result.reason == reason
    assert result.diagnostics()['gradient_searched'] is True
    json.dumps(result.diagnostics(), allow_nan=False)


def test_accepted_gradient_reports_the_measured_gate():
    mask, rgb = _hramp()
    result = fit_fill(mask, rgb, flat_hex='#000000', max_gradient_de=.04)
    assert isinstance(result.fill, (LinearGradientFill, RadialGradientFill))
    assert result.status == 'gradient_accepted'
    assert result.mean_delta_e <= .04
    assert result.diagnostics()['max_gradient_delta_e'] == .04


@pytest.mark.parametrize('limit', [float('nan'), float('inf'), -.1])
def test_invalid_gradient_budget_cannot_approve_a_candidate(monkeypatch, limit):
    mask, rgb = _hramp(h=8, w=20)
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid gradient budget must not trigger a search')
    monkeypatch.setattr('vectoreyes.fill_fit._best_parametric', forbidden)
    result = fit_fill(mask, rgb, flat_hex='#123456', max_gradient_de=limit)
    assert result.status == 'fallback_flat'
    assert result.reason == 'invalid_gradient_budget'
    assert result.diagnostics()['gradient_searched'] is False
    json.dumps(result.diagnostics(), allow_nan=False)


def test_canonical_paint_result_exposes_the_actual_selection():
    from vectoreyes.paint_fitting import fit_source_paint
    mask, rgb = _hramp(h=8, w=20)
    result = fit_source_paint(rgb, source_mask=mask, visible_mask=mask, allow_gradients=False)
    assert result.diagnostics['selection']['status'] == 'forced_flat'
    assert result.diagnostics['selection']['reason'] == 'gradients_disabled'


def test_v0_root_retains_fill_selection_evidence():
    from tests.test_drawing_recipe_actions import _single_region_uri, _trace
    from vectoreyes import mcp_server
    _, response = _trace(_single_region_uri())
    version = mcp_server._DRAWINGS.get_version(response['drawing_id'], response['version'])
    reported = {target['id']: target for target in response['outputs'][0]['report']['targets']}
    for root in version.drawing.regions:
        assert root.diagnostics['fill']['selection']['status'] in {'flat_evidence', 'gradient_accepted', 'forced_flat', 'fallback_flat'}
        assert reported[root.id]['diagnostics']['fill']['selection'] == root.diagnostics['fill']['selection']

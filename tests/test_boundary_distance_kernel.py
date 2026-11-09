"""Correctness and deterministic work bounds for boundary certification."""

import math

import pytest

import vectoreyes.quality as quality


def test_segment_invariants_are_computed_once_per_boundary(monkeypatch):
    original = quality.np.einsum
    denominator_calls = []

    def counted(expression, left, right, **kwargs):
        if expression == 'ij,ij->i' and left is right:
            denominator_calls.append(len(left))
        return original(expression, left, right, **kwargs)

    monkeypatch.setattr(quality.np, 'einsum', counted)
    source = [((0., 0.), (20., 0.), (20., 10.), (0., 10.))]
    replacement = [((0., 1.), (20., 1.), (20., 11.), (0., 11.))]
    measured = quality.polyline_boundary_distance(source, replacement, precision=.05)
    assert 1. <= measured <= 1.05
    assert len(denominator_calls) <= 2, 'segment invariants rebuilt for sampled points'


@pytest.mark.parametrize('direction', ['both', 'source_to_replacement', 'replacement_to_source'])
@pytest.mark.parametrize('legacy', [None, 8])
def test_open_parallel_segments_keep_the_same_distance_bound(direction, legacy):
    measured = quality.polyline_boundary_distance(
        [((0., 0.), (10., 0.))], [((0., 3.), (10., 3.))],
        closed=False, precision=.01, direction=direction, max_samples_per_ring=legacy)
    assert 3. <= measured <= 3.01


def test_asymmetric_directions_and_degenerate_edges_remain_distinct():
    source = [((0., 0.), (0., 0.), (10., 0.))]
    replacement = [((0., 0.), (2., 0.))]
    assert quality.polyline_boundary_distance(source, replacement, closed=False,
        direction='source_to_replacement') == 8.
    assert quality.polyline_boundary_distance(source, replacement, closed=False,
        direction='replacement_to_source') <= .05
    assert math.isinf(quality.polyline_boundary_distance([], replacement))

"""Nonintersection is a boundary predicate, not a filled PathOps count."""

import pytest


def certify(d, **kwargs):
    import importlib.util
    assert importlib.util.find_spec('vectoreyes.boundary_topology') is not None, 'boundary certificate is missing'
    import vectoreyes.boundary_topology as topology
    from vectoreyes.path_geometry import parse_svg_paths
    return topology.certify_simple_boundaries(parse_svg_paths(d, region_id='r'), **kwargs)


@pytest.mark.parametrize('d', [
    'M0 0 L10 0 L10 10 L0 10 Z',
    'M0 0 L10 0 L10 3 L3 3 L3 10 L0 10 Z',
    'M0 0 C2 4 8 4 10 0 C8 -4 2 -4 0 0 Z',
    'M0 0 C2 0 3 2 5 2 C7 2 8 0 10 0 L10 10 L0 10 Z',
    'M0 0 L8 0 Q10 0 10 2 L10 8 Q10 10 8 10 L0 10 Z',
    'M0 0 L10 0 L10 10 L0 10 Z M2 2 L2 4 L4 4 L4 2 Z',
    'M0 0 L10 0 L10 10 L0 10 Z M20 0 L30 0 L30 10 L20 10 Z',
])
def test_simple_nonconvex_curved_and_cutout_boundaries_are_admissible(d):
    certify(d)


@pytest.mark.parametrize('d', [
    'M0 0 L10 10 L0 10 L10 0 Z',
    'M0 0 C3 3 -1 3 2 0 L2 -2 L0 -2 Z',
    'M9 0 L10 0 C8.4 4 8.4 6 10 10 L9 10 Z',
    'M0 0 L10 0 L10 10 L0 10 Z M8 2 L8 4 L12 4 L12 2 Z',
])
def test_crossed_boundaries_are_not_repaired_or_simplified_into_acceptance(d):
    with pytest.raises(ValueError, match='nonintersection'):
        certify(d)


def test_uncertifiable_work_budget_is_a_rejection_not_a_coarse_acceptance():
    with pytest.raises(ValueError, match='work limit'):
        certify('M0 0 C3 3 -1 3 2 0 L2 -2 L0 -2 Z', work_limit=1)


def test_svg_rounding_cannot_collapse_a_safe_gap_into_a_boundary_contact():
    with pytest.raises(ValueError, match='nonintersection'):
        certify('M0 0 L10 0 L10 10 L0 10 Z M0.0000001 2 L1 2 L1 3 L0.0000001 3 Z')


def test_unsupported_arcs_are_explicitly_unassessed_not_flattened():
    with pytest.raises(ValueError, match='requires L/Q/C'):
        certify('M0 0 L10 0 A5 5 0 0 1 10 10 L0 10 Z')

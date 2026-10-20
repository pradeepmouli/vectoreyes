from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.quality import GeometryPolicy
from vectormark.refinement.framework import _replacement_preserves_scene
from vectormark.refinement.region_sample import RegionSample


def leaf(identity, left, right):
    return RegionSample(identity, Shape('rect', {'x': left, 'y': 0, 'w': right-left, 'h': 10}),
        FlatFill('#123456'), drawing_id=f'r{identity}')


def test_automatic_gate_cannot_hide_a_moved_material_edge_in_a_common_union():
    before = [leaf(1, 0, 5), leaf(2, 5, 10)]
    after = [leaf(1, 0, 8), leaf(2, 8, 10)]
    report = _replacement_preserves_scene(after, before, policy=GeometryPolicy(max_boundary_error_px=0))
    assert not report.accepted
    assert 'region:r1:boundary_error_exceeded' in report.reasons
    assert 'region:r2:boundary_error_exceeded' in report.reasons
    assert report.symmetric_difference_area == 0


def test_explicit_structural_replacement_can_change_region_count_without_changing_coverage():
    report = _replacement_preserves_scene([leaf(3, 0, 10)], [leaf(1, 0, 5), leaf(2, 5, 10)],
        policy=GeometryPolicy(max_boundary_error_px=0))
    assert report.accepted


def test_gap_repair_has_separate_coverage_and_material_boundary_budgets():
    before = [leaf(1, 0, 4), leaf(2, 6, 10)]
    after = [leaf(1, 0, 5), leaf(2, 5, 10)]
    report = _replacement_preserves_scene(after, before,
        policy=GeometryPolicy(max_relative_area_error=.3, preserve_topology=False),
        region_policy=GeometryPolicy(max_boundary_error_px=1.1))
    assert report.accepted
    # Closing a long gap removes inner union boundaries, which is not the
    # distance that should constrain movement of either material's edge.
    assert report.max_boundary_error_px > 1.1
    assert all(item.accepted for _, item in report.region_assessments)
    bad = [leaf(1, 0, 8), leaf(2, 8, 10)]
    rejected = _replacement_preserves_scene(bad, before,
        policy=GeometryPolicy(max_relative_area_error=.3, preserve_topology=False),
        region_policy=GeometryPolicy(max_boundary_error_px=1.1))
    assert not rejected.accepted

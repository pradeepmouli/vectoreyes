from dataclasses import replace

import pytest

from vectormark.drawing import Canvas, Drawing, DrawingRegion
from vectormark.drawing_fill import FlatFill
from vectormark.path_geometry import CloneGeometry, RectanglePrimitive, PrimitiveGeometry, geometry_from_svg_d
from vectormark.quality import GeometryPolicy


def rectangle(identity, left, right):
    return DrawingRegion(identity, PrimitiveGeometry(RectanglePrimitive((left, 0), right-left, 10)),
        FlatFill('#123456'))


def drawing(*regions):
    return Drawing('test', Canvas(20, 10), regions)


def test_shifted_internal_boundary_fails_even_with_identical_outer_silhouette():
    from vectormark.revision_quality import assess_revision
    before = drawing(rectangle('left', 0, 5), rectangle('right', 5, 10))
    candidate = drawing(rectangle('left', 0, 8), rectangle('right', 8, 10))
    report = assess_revision(before, before, candidate, policy=GeometryPolicy(max_boundary_error_px=0))
    assert report.coverage.accepted
    assert not report.geometry_accepted
    assert {item.candidate_ids[0] for item in report.regions if not item.step.accepted} == {'left', 'right'}
    assert max(item.step.max_boundary_error_px for item in report.regions) == pytest.approx(3, abs=.05)


def test_step_and_cumulative_reference_budgets_are_independent():
    from vectormark.revision_quality import assess_revision
    original = drawing(rectangle('r1', 0, 5))
    before = drawing(rectangle('r1', 0, 6))
    candidate = drawing(rectangle('r1', 0, 7))
    report = assess_revision(original, before, candidate,
        policy=GeometryPolicy(max_boundary_error_px=1.1),
        reference_policy=GeometryPolicy(max_boundary_error_px=1.5))
    assert report.regions[0].step.accepted
    assert not report.regions[0].reference.accepted
    assert not report.geometry_accepted


def test_explicit_merge_correspondence_is_not_a_missing_region():
    from vectormark.revision_quality import RegionMatch, assess_revision
    before = drawing(rectangle('left', 0, 5), rectangle('right', 5, 10))
    candidate = drawing(rectangle('merged', 0, 10))
    report = assess_revision(before, before, candidate,
        matches=(RegionMatch(('left', 'right'), ('merged',), ('left', 'right')),),
        policy=GeometryPolicy(max_boundary_error_px=0))
    assert report.geometry_accepted
    assert report.regions[0].step.relative_area_error == 0
    # Geometry success must not silently claim that paint has been assessed.
    assert report.status == 'geometry_passed'
    assert report.fill_status == 'not_assessed'


def test_undeclared_merge_does_not_hide_retired_regions():
    from vectormark.revision_quality import assess_revision
    before = drawing(rectangle('left', 0, 5), rectangle('right', 5, 10))
    report = assess_revision(before, before, drawing(rectangle('merged', 0, 10)), policy=GeometryPolicy())
    assert not report.geometry_accepted
    assert 'invalid_region_correspondence' in report.reasons


def test_small_lost_cutout_is_rejected_despite_small_area_difference():
    from vectormark.revision_quality import assess_revision
    before = drawing(DrawingRegion('r1', geometry_from_svg_d(
        'M0 0 L10 0 L10 10 L0 10 Z M4 4 L4 5 L5 5 L5 4 Z', region_id='r1')))
    report = assess_revision(before, before, drawing(rectangle('r1', 0, 10)), policy=GeometryPolicy())
    assert not report.geometry_accepted
    assert 'topology_changed' in report.regions[0].step.reasons


def test_unresolved_clone_is_a_localized_quality_failure_not_a_crash():
    from vectormark.revision_quality import assess_revision
    from vectormark.path_geometry import AffineTransform
    before = drawing(rectangle('r1', 0, 5))
    candidate = drawing(DrawingRegion('r1', CloneGeometry('missing', (AffineTransform(1, 0, 0, 1, 0, 0),))))
    report = assess_revision(before, before, candidate, policy=GeometryPolicy())
    assert not report.geometry_accepted
    assert 'invalid_geometry:r1' in report.reasons


def test_fill_change_is_not_conflated_with_shape_error():
    from vectormark.revision_quality import assess_revision
    before = drawing(rectangle('r1', 0, 5))
    candidate = replace(before, regions=(replace(before.regions[0], fill=FlatFill('#abcdef')),))
    report = assess_revision(before, before, candidate, policy=GeometryPolicy(max_boundary_error_px=0))
    assert report.geometry_accepted
    assert report.fill_status == 'not_assessed'


def test_canvas_registration_mismatch_cannot_pass_as_same_geometry():
    from vectormark.revision_quality import assess_revision
    before = drawing(rectangle('r1', 0, 5))
    candidate = replace(before, canvas=Canvas(40, 20))
    report = assess_revision(before, before, candidate, policy=GeometryPolicy())
    assert not report.geometry_accepted
    assert 'canvas_registration_mismatch' in report.reasons


def test_duplicate_lineage_members_cannot_double_count_evidence():
    from vectormark.revision_quality import RegionMatch, assess_revision
    before = drawing(rectangle('r1', 0, 5))
    matches = (RegionMatch(('r1',), ('r1',), ('r1',)),) * 2
    report = assess_revision(before, before, before, matches=matches, policy=GeometryPolicy())
    assert not report.geometry_accepted
    assert 'invalid_region_correspondence' in report.reasons


def test_engine_assessment_is_region_aware_and_invalid_reports_are_json_safe():
    import json
    from vectormark.python_drawing_engine import PythonDrawingEngine
    from vectormark.path_geometry import AffineTransform
    before = drawing(rectangle('r1', 0, 5))
    candidate = drawing(DrawingRegion('r1', CloneGeometry('missing', (AffineTransform(1, 0, 0, 1, 0, 0),))))
    report = PythonDrawingEngine().assess_revision(before, before, candidate, policy=GeometryPolicy())
    assert not report.geometry_accepted
    serialized = json.dumps(report.diagnostics(), allow_nan=False)
    assert 'invalid_geometry:r1' in serialized


def test_clone_reference_to_its_parent_branch_fails_without_recursive_resolution():
    from vectormark.revision_quality import assess_revision
    from vectormark.path_geometry import AffineTransform
    before = drawing(DrawingRegion('group', children=(rectangle('r1', 0, 5),)))
    candidate = drawing(DrawingRegion('group', children=(DrawingRegion('r1',
        CloneGeometry('group', (AffineTransform(1, 0, 0, 1, 0, 0),))),)))
    report = assess_revision(before, before, candidate, policy=GeometryPolicy())
    assert not report.geometry_accepted
    assert 'invalid_geometry:r1' in report.reasons

import math

import numpy as np

from vectormark.quality import GeometryPolicy, assess_geometry, symmetric_boundary_distance
from vectormark.skia_geometry import SkPath


def test_geometry_policy_limits_are_conjunctive_and_typed():
    source = SkPath(shell=[(0, 0), (100, 0), (100, 100), (0, 100)])
    replacement = SkPath(shell=[(0, 0), (100, 0), (100, 99), (0, 99)])
    assessment = assess_geometry(
        source,
        replacement,
        GeometryPolicy(max_relative_area_error=0.02, max_boundary_error_px=0.5),
    )
    assert not assessment.accepted
    assert assessment.relative_area_error <= 0.02
    assert assessment.max_boundary_error_px > 0.5
    assert assessment.reasons == ("boundary_error_exceeded",)


def test_geometry_policy_rejects_dropped_component_and_counter():
    disk = SkPath.make_circle(20, 20, 10)
    component = SkPath.make_circle(50, 20, 1)
    with_component = disk.union(component)
    dropped_component = assess_geometry(
        with_component, disk, GeometryPolicy(max_boundary_error_px=100)
    )
    assert "topology_changed" in dropped_component.reasons

    ring = disk.difference(SkPath.make_circle(20, 20, 1))
    dropped_counter = assess_geometry(ring, disk, GeometryPolicy(max_boundary_error_px=100))
    assert "topology_changed" in dropped_counter.reasons


def test_geometry_policy_fails_closed_for_bad_limits_and_geometry():
    disk = SkPath.make_circle(20, 20, 10)
    assert assess_geometry(disk, disk, GeometryPolicy(max_boundary_error_px=-1)).reasons == (
        "invalid_tolerance",
    )
    assert not assess_geometry(SkPath(), disk, GeometryPolicy()).accepted
    assert math.isinf(symmetric_boundary_distance(SkPath(), disk))


def test_boundary_distance_cannot_skip_a_localized_vertex_defect():
    angles = np.linspace(0.0, 2.0 * math.pi, 128, endpoint=False)
    points = [(10.0 * math.cos(a), 10.0 * math.sin(a)) for a in angles]
    points[1] = (15.0 * math.cos(angles[1]), 15.0 * math.sin(angles[1]))
    source = SkPath(shell=points)
    replacement = SkPath(shell=points[:1] + points[2:])

    assert symmetric_boundary_distance(source, replacement) > 5.0
    assert symmetric_boundary_distance(source, replacement, max_samples_per_ring=64) == 0.0


def test_boundary_distance_bounds_discrepancy_inside_a_long_segment():
    source = SkPath(shell=[(0, 0), (20, 0), (20, -20), (0, -20)])
    replacement = SkPath(shell=[(0, 0), (10, 3), (20, 0), (20, -20), (0, -20)])

    assert symmetric_boundary_distance(source, replacement, precision=0.01) >= 3.0


def test_identical_geometry_has_zero_boundary_distance():
    rectangle = SkPath(shell=[(0, 0), (1000, 0), (1000, 10), (0, 10)])
    assert symmetric_boundary_distance(rectangle, rectangle) == 0.0

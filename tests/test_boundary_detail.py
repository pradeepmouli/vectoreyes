"""Source-supported detail must not depend on SVG command count."""

import json
from dataclasses import replace

import numpy as np
import pytest

from vectormark.boundary_detail import (
    BoundaryDetailPolicy, BoundaryDetailRequest, assess_boundary_detail,
)
from vectormark.path_geometry import parse_svg_paths
from vectormark.python_drawing_engine import PythonDrawingEngine


def path(d):
    return parse_svg_paths(d, region_id="r1")[0]


def request(source, candidate, **kwargs):
    return BoundaryDetailRequest(
        region_id="r1", source=source, candidate=candidate,
        source_revision="source-v0", candidate_revision="v1",
        reference_kind=kwargs.pop("reference_kind", "retained_polyline"), **kwargs,
    )


def circle(wobble=0, reverse=False, shift=0):
    t = np.arange(720) * (2 * np.pi / 720)
    radius = 80 + wobble * np.sin(20 * t)
    points = np.column_stack((radius * np.cos(t), radius * np.sin(t)))
    points = np.roll(points[::-1] if reverse else points, shift, axis=0)
    return path("M" + " L".join(f"{x:.9f} {y:.9f}" for x, y in points) + " Z")


def test_start_winding_and_line_subdivision_do_not_invent_detail():
    source = path("M0 0 L100 0 L100 100 L0 100 Z")
    candidate = path("M100 50 L100 0 L50 0 L0 0 L0 50 L0 100 L100 100 Z")
    report = PythonDrawingEngine().assess_boundary_detail(request(source, candidate))
    assert report.status == "assessed"
    assert max(scale.unsupported_turn_radians for scale in report.scales) < 1e-5
    assert max(scale.missing_turn_radians for scale in report.scales) < 1e-5
    assert report.alignment_rms_px < 1e-4


def test_exact_quadratic_degree_elevation_does_not_invent_detail():
    source = path("M0 0 Q30 60 90 0 L90 -30 L0 -30 Z")
    candidate = path("M0 0 C20 40 50 40 90 0 L90 -30 L0 -30 Z")
    report = assess_boundary_detail(request(source, candidate, reference_kind="declared_geometry"))
    assert report.status == "assessed"
    assert max(scale.unsupported_turn_radians for scale in report.scales) < .002
    assert max(scale.missing_turn_radians for scale in report.scales) < .002


def test_wobble_is_excess_detail_and_reversing_reference_reports_missing_detail():
    smooth, wavy = circle(), circle(2)
    added = assess_boundary_detail(request(smooth, wavy))
    lost = assess_boundary_detail(request(wavy, smooth))
    assert added.scales[0].unsupported_turn_radians > 5
    assert lost.scales[0].missing_turn_radians > 5
    assert added.scales[0].unsupported_turn_radians > added.scales[-1].unsupported_turn_radians
    assert max(p.unsupported_curvature_per_px for p in added.scales[0].samples) > .05
    assert all(len(p.source_point) == len(p.candidate_point) == 2 for p in added.scales[0].samples)
    json.dumps(added.diagnostics(), allow_nan=False)


def test_source_supported_wobble_is_not_penalized_for_being_complex():
    report = assess_boundary_detail(request(circle(2), circle(2, reverse=True, shift=181)))
    assert max(s.unsupported_turn_radians + s.missing_turn_radians for s in report.scales) < .001


def test_corner_lost_to_a_round_boundary_is_localized_not_hidden_by_command_counts():
    square = path("M-70 -70 L70 -70 L70 70 L-70 70 Z")
    report = assess_boundary_detail(request(square, circle()))
    assert report.scales[0].missing_turn_radians > 3
    missing = max(report.scales[0].samples, key=lambda p: p.missing_curvature_per_px)
    assert abs(abs(missing.source_point[0]) - 70) < 2
    assert abs(abs(missing.source_point[1]) - 70) < 2


def test_sampling_budget_is_explicit_instead_of_silently_coarsening():
    report = assess_boundary_detail(request(circle(), circle(), policy=BoundaryDetailPolicy(max_samples=64)))
    assert report.status == "not_assessed"
    assert report.reason == "sample_budget_exceeded"
    assert not report.scales


def test_scale_units_follow_explicit_pixel_registration():
    source = path("M0 0 L100 0 L100 100 L0 100 Z")
    candidate = path("M0 0 L100 0 L100 100 L10 100 Z")
    base = assess_boundary_detail(request(source, candidate))
    scaled = assess_boundary_detail(request(
        path("M0 0 L200 0 L200 200 L0 200 Z"),
        path("M0 0 L200 0 L200 200 L20 200 Z"),
        policy=BoundaryDetailPolicy(units_per_pixel=2),
    ))
    assert [s.unsupported_turn_radians for s in scaled.scales] == pytest.approx(
        [s.unsupported_turn_radians for s in base.scales], abs=1e-5)
    assert scaled.scales[0].samples[0].source_point == base.scales[0].samples[0].source_point


@pytest.mark.parametrize("kwargs", [
    {"scales_px": ()}, {"scales_px": (0,)}, {"scales_px": (float('nan'),)},
    {"sample_spacing_px": 0}, {"sample_spacing_px": 1},
    {"units_per_pixel": float('inf')}, {"max_samples": 63},
])
def test_invalid_sampling_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BoundaryDetailPolicy(**kwargs)


def test_missing_provenance_is_not_silently_labeled_raster_evidence():
    original = request(circle(), circle())
    with pytest.raises(ValueError):
        replace(original, source_revision="")


def test_fitted_reference_cannot_claim_to_be_a_retained_material_polyline():
    with pytest.raises(ValueError):
        request(path("M0 0 Q50 100 100 0 L0 0 Z"), circle())


@pytest.mark.parametrize("d,reason", [
    ("M0 0 L0 0 L0 0 Z", "degenerate_boundary"),
    ("M0 0 L2 0 L2 2 L0 2 Z", "scale_exceeds_boundary_support"),
])
def test_unmeasurable_boundaries_are_not_reported_as_zero_error(d, reason):
    report = assess_boundary_detail(request(path(d), path(d)))
    assert report.status == "not_assessed"
    assert report.reason == reason
    assert report.alignment_rms_px is None
    json.dumps(report.diagnostics(), allow_nan=False)


def test_exact_cubic_subdivision_does_not_invent_detail():
    source = path("M0 0 C0 80 100 80 100 0 L100 -30 L0 -30 Z")
    candidate = path("M0 0 C0 40 25 60 50 60 C75 60 100 40 100 0 L100 -30 L0 -30 Z")
    report = assess_boundary_detail(replace(request(circle(), candidate), source=source,
        reference_kind="declared_geometry"))
    assert max(s.unsupported_turn_radians + s.missing_turn_radians for s in report.scales) < .002

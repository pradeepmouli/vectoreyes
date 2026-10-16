"""Corner declarations describe local transitions, not arbitrary fitted spans."""

import numpy as np
import pytest

import vectormark.fit as fit
from vectormark.optimizer.vector_region import _parse_subpaths


def _sample_ring(points):
    return np.vstack([
        np.linspace(a, b, int(np.ceil(np.linalg.norm(b - a))) + 1)[:-1]
        for a, b in zip(points, np.roll(points, -1, axis=0), strict=True)
    ])


def _contains(index, span):
    start, end = span
    return start <= index <= end if start <= end else index >= start or index <= end


def _path_endpoints(d):
    endpoints = []
    for subpath in _parse_subpaths(d):
        for command, values in subpath:
            if command in {"M", "L"}:
                endpoints.append(tuple(values[:2]))
            elif command == "Q":
                endpoints.append(tuple(values[2:4]))
            elif command == "C":
                endpoints.append(tuple(values[4:6]))
    return endpoints


def _has_endpoint(d, point):
    return any(np.allclose(endpoint, point, atol=0.005) for endpoint in _path_endpoints(d))


def _singleton_ring():
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    ring = np.column_stack((40.0 + 30.0 * np.cos(angles), 50.0 + 20.0 * np.sin(angles)))
    return np.vstack((ring, ring[0]))


@pytest.mark.parametrize("progressive", [False, True])
def test_singleton_detected_cut_remains_a_fitted_endpoint(monkeypatch, progressive):
    contour = _singleton_ring()
    ring = contour[:-1]
    retained_index = 5
    auxiliary_index = (retained_index + len(ring) // 2) % len(ring)
    monkeypatch.setattr(fit, "rdp", lambda points, _epsilon: np.asarray(points))
    monkeypatch.setattr(fit, "_suppress_cornerlet_vertices", lambda points, **_kwargs: points)
    monkeypatch.setattr(fit, "corner_indices", lambda *args, **kwargs: [retained_index])
    monkeypatch.setattr(fit, "corner_curve_radius_at", lambda *args, **kwargs: None)
    monkeypatch.setattr(fit, "_protected_line_spans", lambda _ring, **_kwargs: [])

    d, points, spans = fit._fit_path_d_with_corners(
        contour, epsilon=1.0, max_error=1.0, progressive=progressive,
    )

    assert _has_endpoint(d, ring[retained_index])
    assert points == (tuple(ring[retained_index]),)
    assert tuple(ring[auxiliary_index]) not in points
    assert spans == ()


@pytest.mark.parametrize("progressive", [False, True])
def test_singleton_forced_cut_remains_a_fitted_endpoint(monkeypatch, progressive):
    contour = _singleton_ring()
    ring = contour[:-1]
    retained_index = 7
    auxiliary_index = (retained_index + len(ring) // 2) % len(ring)
    # With no detections the fitter uses simp[0] as a non-semantic fitting
    # seed. Rotate that isolated detector input so the forced point coincides
    # with the seed: the resulting fitting-boundary set is genuinely singleton.
    original_rdp = fit.rdp
    monkeypatch.setattr(
        fit, "rdp", lambda points, epsilon: (
            np.roll(np.asarray(points), -retained_index, axis=0)
            if len(points) == len(ring) else original_rdp(points, epsilon)
        ),
    )
    monkeypatch.setattr(fit, "corner_indices", lambda *args, **kwargs: [])
    monkeypatch.setattr(fit, "_protected_line_spans", lambda _ring, **_kwargs: [])

    d, points, spans = fit._fit_path_d_with_corners(
        contour, epsilon=1.0, max_error=1.0, progressive=progressive,
        forced_corners=np.asarray([ring[retained_index]]),
    )

    assert _has_endpoint(d, ring[retained_index])
    assert points == (tuple(ring[retained_index]),)
    assert tuple(ring[auxiliary_index]) not in points
    assert spans == ()


def test_zero_and_multiple_cut_boundary_selection_remains_unchanged(monkeypatch):
    contour = _singleton_ring()
    ring = contour[:-1]
    monkeypatch.setattr(fit, "corner_indices", lambda *args, **kwargs: [])

    no_corner_d, no_corner_points, _ = fit._fit_path_d_with_corners(
        contour, epsilon=1.0, max_error=1.0,
    )
    assert _path_endpoints(no_corner_d)[0] == tuple(ring[0])
    assert tuple(ring[len(ring) // 2]) in _path_endpoints(no_corner_d)
    assert no_corner_points == ()

    forced_indices = (3, 8)
    multiple_d, multiple_points, _ = fit._fit_path_d_with_corners(
        contour, epsilon=1.0, max_error=1.0,
        forced_corners=np.asarray([ring[index] for index in forced_indices]),
    )
    assert _path_endpoints(multiple_d)[0] == tuple(ring[0])
    assert all(_has_endpoint(multiple_d, ring[index]) for index in forced_indices)
    assert set(multiple_points) == {tuple(ring[index]) for index in forced_indices}


def test_protected_line_gaps_do_not_invent_corner_declarations(monkeypatch):
    arc = np.column_stack((
        200 + 50 * np.cos(np.linspace(-np.pi / 2, np.pi / 2, 19)),
        50 + 50 * np.sin(np.linspace(-np.pi / 2, np.pi / 2, 19)),
    ))
    ring = _sample_ring(np.vstack(([0, 0], arc, [0, 100])))
    assert len(fit._protected_line_spans(ring)) >= 2
    # With no detector evidence, ordinary fitting boundaries must not become
    # semantic corners. In particular, the semicircular end is not one corner.
    monkeypatch.setattr(fit, "corner_indices", lambda *args, **kwargs: [])
    d, points, spans = fit._fit_path_d_with_corners(
        np.vstack((ring, ring[0])), epsilon=1, max_error=1,
        progressive=True, support_weighted_corners=False,
    )
    assert "L" in d
    assert points == ()
    assert spans == ()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("following", [[-160, 20], [-170, 143]])
def test_two_supported_corners_do_not_swallow_curved_side(reverse, following):
    points = np.array([
        [-135, -105], [0, 0], [-6, 12], [-29, 47],
        [-42, 62], [-70, 88], following, [-240, -110],
    ], dtype=float)
    if reverse:
        points = points[::-1]
    ring = _sample_ring(points)
    spans = fit._supported_corner_ranges(
        ring, points, epsilon=1, min_turn_degrees=10, min_support_length=64,
    )
    indices = [int(np.argmin(np.linalg.norm(ring - p, axis=1)))
               for p in ([0, 0], [-29, 47], [-70, 88])]
    first, middle, last = indices
    assert any(_contains(first, span) for span in spans)
    assert any(_contains(last, span) for span in spans)
    assert not any(_contains(middle, span) for span in spans)
    assert not any(_contains(first, span) and _contains(last, span) for span in spans)


@pytest.mark.parametrize("radius, expected", [(10, True), (100, False)])
@pytest.mark.parametrize("reverse", [False, True])
def test_uniform_curvature_requires_local_extent(radius, expected, reverse):
    theta = np.linspace(-np.pi / 2, 0, 10)
    arc = np.column_stack((radius * np.cos(theta), radius + radius * np.sin(theta)))
    points = np.vstack(([-120, 0], arc, [radius, radius + 120], [-120, radius + 120]))
    if reverse:
        points = points[::-1]
    ring = _sample_ring(points)
    spans = fit._supported_corner_ranges(
        ring, points, epsilon=1, min_turn_degrees=10, min_support_length=64,
    )
    middle = int(np.argmin(np.linalg.norm(ring - arc[4], axis=1)))
    # A compact fillet is retained; an equally turning but broad side arc is
    # left for normal fitting instead of being marked for normalization.
    assert any(_contains(middle, span) for span in spans) is expected

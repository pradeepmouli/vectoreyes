import pytest


@pytest.mark.parametrize("command, midpoint", [
    ("Q2 4 4 0", (2, 2)),
    ("C0 4 4 4 4 0", (2, 3)),
    ("A2 2 0 0 1 4 0", (2, -2)),
])
def test_geometry_sampling_keeps_curve_samples_and_open_closure_policy(command, midpoint):
    from vectoreyes.path_sampling import _parse_subpaths, _sample_subpath

    (tokens,) = _parse_subpaths(f"M0 0 {command} Z")
    points = _sample_subpath(tokens, samples=3)

    assert len(points) == 3
    assert points[0] == (0, 0)
    assert points[1] == pytest.approx(midpoint)
    assert points[2] == pytest.approx((4, 0))


def test_geometry_sampling_preserves_disconnected_source_order_and_precision():
    from vectoreyes.path_sampling import _parse_subpaths

    paths = _parse_subpaths("M1e-3 2.123456 L3 4 Z M8 9 L10 11 Z")

    assert paths == [
        [("M", [0.001, 2.123456]), ("L", [3, 4]), ("Z", [])],
        [("M", [8, 9]), ("L", [10, 11]), ("Z", [])],
    ]

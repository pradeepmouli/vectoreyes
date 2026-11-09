import numpy as np

from vectoreyes.softlabel import alpha_unmix, soft_label_field


def test_alpha_unmix_vectorizes_over_leading_dimensions():
    red = np.array(
        [
            [[255.0, 0.0, 0.0], [255.0, 0.0, 0.0]],
            [[255.0, 0.0, 0.0], [255.0, 0.0, 0.0]],
        ]
    )
    blue = np.array(
        [
            [[0.0, 0.0, 255.0], [0.0, 0.0, 255.0]],
            [[0.0, 0.0, 255.0], [0.0, 0.0, 255.0]],
        ]
    )
    rgb = np.array(
        [
            [[255.0, 0.0, 0.0], [127.5, 0.0, 127.5]],
            [[63.75, 0.0, 191.25], [0.0, 0.0, 255.0]],
        ]
    )

    coverage = alpha_unmix(rgb, red, blue)

    assert coverage.shape == (2, 2)
    np.testing.assert_allclose(coverage, [[1.0, 0.5], [0.25, 0.0]])


def test_soft_label_field_unmixes_boundary_pixels_without_python_scalar_assumptions():
    rgb = np.zeros((3, 5, 3), dtype=float)
    rgb[:, :2] = [255.0, 0.0, 0.0]
    rgb[:, 2] = [127.5, 0.0, 127.5]
    rgb[:, 3:] = [0.0, 0.0, 255.0]
    palette = np.array([[255.0, 0.0, 0.0], [0.0, 0.0, 255.0]])

    labels = soft_label_field(rgb, palette)

    assert labels.shape == (3, 5, 2)
    np.testing.assert_allclose(labels[:, 2, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(labels.sum(axis=2), 1.0, atol=1e-6)

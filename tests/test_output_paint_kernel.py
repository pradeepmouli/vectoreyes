import numpy as np

from vectoreyes.drawing_fill import FlatFill, GradientStop, LinearGradientFill
from vectoreyes.output_settings import DrawingOutputSettings


def test_output_paint_limits_use_source_samples_without_geometry_access():
    from vectoreyes.paint_fitting import constrain_output_paint

    gradient = LinearGradientFill((0, 0), (10, 0), (
        GradientStop(0, '#ff0000'), GradientStop(1, '#0000ff')))
    rgb = np.full((10, 10, 3), (20, 100, 240), dtype=np.uint8)
    result = constrain_output_paint(gradient, rgb=rgb, sample_mask=np.ones((10, 10), bool),
        settings=DrawingOutputSettings(allow_gradients=False))

    assert result == FlatFill('#1464f0')
    assert gradient.stops[0].color == '#ff0000'


def test_output_palette_can_collapse_gradient_stops_to_one_flat_fill():
    from vectoreyes.paint_fitting import constrain_output_paint

    gradient = LinearGradientFill((0, 0), (10, 0), (
        GradientStop(0, '#ff0000'), GradientStop(1, '#fe0000')))
    palette = np.array([[255, 0, 0]], dtype=np.uint8)
    result = constrain_output_paint(gradient, rgb=np.zeros((10, 10, 3), np.uint8),
        sample_mask=np.ones((10, 10), bool), settings=DrawingOutputSettings(max_output_colors=1),
        palette=palette)

    assert result == FlatFill('#ff0000')

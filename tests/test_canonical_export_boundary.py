"""Only canonical drawings may cross the SVG/report serialization boundary."""

from vectormark import drawing_refine


def test_working_region_executor_has_no_parallel_export_adapter():
    assert not hasattr(drawing_refine, "render_drawing")
    assert not hasattr(drawing_refine, "labeled_drawing_svg")
    assert not hasattr(drawing_refine, "RenderedDrawing")
    assert not hasattr(drawing_refine, "drawing_summary")


def test_trace_diagnostics_describe_published_paint():
    import numpy as np
    from vectormark.drawing_engine import RasterImage, TraceDrawingRequest
    from vectormark.drawing_fill import serialize_fill
    from vectormark.drawing_trace import TraceOptions
    from vectormark.output_settings import DrawingOutputSettings
    from vectormark.python_drawing_engine import PythonDrawingEngine

    rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
    rgb[6:26, 6:26] = (20, 80, 140)
    result = PythonDrawingEngine().trace_drawing(TraceDrawingRequest(
        "diagnostics", RasterImage(32, 32, "rgb8", rgb.tobytes()),
        TraceOptions(max_sampling_colors=2, remove_background="off"),
        min_region_fraction=0, initial_stitch=False,
        output_settings=DrawingOutputSettings(max_output_colors=1, allow_gradients=False),
    ))
    assert result.diagnostics["regions"]
    for item in result.diagnostics["regions"]:
        assert item["fill"] == serialize_fill(result.drawing.region(item["id"]).fill)

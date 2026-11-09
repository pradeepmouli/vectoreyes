from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
from vectoreyes.drawing_fill import (
    GradientStop,
    LinearGradientFill,
    RadialGradientFill,
)
from vectoreyes.drawing_render import render_drawing
from vectoreyes.path_geometry import (
    AffineTransform,
    CloneGeometry,
    PathGeometryWithSymmetry,
    PolygonPrimitive,
    PrimitiveGeometry,
    RectanglePrimitive,
    SymmetryAxis,
    geometry_from_svg_d,
)


def test_renderer_emits_exact_affine_gradient_transforms_and_target_paint():
    linear = LinearGradientFill(
        (0.0, 0.0),
        (10.0, 0.0),
        (GradientStop(0.0, "#000000"), GradientStop(1.0, "#ffffff")),
        AffineTransform(1.0, 0.25, 0.5, 1.5, 3.0, 4.0),
    )
    radial = RadialGradientFill(
        (4.0, 6.0),
        3.0,
        (GradientStop(0.0, "#ff0000"), GradientStop(1.0, "#0000ff")),
        focal_point=(3.0, 5.0),
        transform=AffineTransform(2.0, 0.4, 0.75, 1.25, 11.0, -2.0),
    )
    drawing = Drawing(
        "drw_render",
        Canvas(40.0, 30.0),
        (
            DrawingRegion(
                "source",
                geometry=PrimitiveGeometry(
                    RectanglePrimitive((0.0, 0.0), 10.0, 10.0)
                ),
                fill=linear,
            ),
            DrawingRegion(
                "target",
                geometry=CloneGeometry(
                    "source", (AffineTransform(1.0, 0.3, 0.4, 2.0, 9.0, 7.0),)
                ),
                fill=radial,
                z=1,
            ),
        ),
    )

    svg = render_drawing(drawing).svg

    assert 'gradientTransform="matrix(1 0.25 0.5 1.5 3 4)"' in svg
    assert 'gradientTransform="matrix(2 0.4 0.75 1.25 11 -2)"' in svg
    assert 'fx="3" fy="5"' in svg
    assert 'id="target"' in svg and 'fill="url(#paint-1)"' in svg
    assert 'href="#geometry-source" transform="matrix(1 0.3 0.4 2 9 7)"' in svg
    assert '<rect id="source" fill="url(#paint-0)"' in svg


def test_clone_clip_preserves_drawing_space_gradient_appearance():
    import io

    import numpy as np
    from PIL import Image
    import resvg_py

    gradient = LinearGradientFill(
        (20.0, 0.0),
        (30.0, 0.0),
        (GradientStop(0.0, "#000000"), GradientStop(1.0, "#ffffff")),
    )
    concrete = Drawing(
        "drw_concrete",
        Canvas(40.0, 20.0),
        (
            DrawingRegion(
                "target",
                geometry=PrimitiveGeometry(
                    RectanglePrimitive((20.0, 0.0), 10.0, 10.0)
                ),
                fill=gradient,
            ),
        ),
    )
    clone = Drawing(
        "drw_clone",
        Canvas(40.0, 20.0),
        (
            DrawingRegion(
                "source",
                geometry=PrimitiveGeometry(
                    RectanglePrimitive((0.0, 0.0), 10.0, 10.0)
                ),
                fill=None,
            ),
            DrawingRegion(
                "target",
                geometry=CloneGeometry(
                    "source", (AffineTransform(tx=20.0),)
                ),
                fill=gradient,
                z=1,
            ),
        ),
    )

    def pixels(drawing):
        png = resvg_py.svg_to_bytes(
            svg_string=render_drawing(drawing).svg, width=40, height=20
        )
        return np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGBA"))

    # The source has no paint, so the visible clone must match the equivalent
    # concrete target under the target's drawing-space gradient exactly.
    assert np.array_equal(pixels(clone), pixels(concrete))


def test_symmetry_clip_preserves_drawing_space_gradient_appearance():
    import io

    import numpy as np
    from PIL import Image
    import resvg_py

    gradient = LinearGradientFill(
        (0.0, 0.0),
        (20.0, 0.0),
        (GradientStop(0.0, "#000000"), GradientStop(1.0, "#ffffff")),
    )
    concrete = Drawing(
        "drw_concrete_symmetry",
        Canvas(24.0, 14.0),
        (
            DrawingRegion(
                "target",
                geometry=PrimitiveGeometry(
                    RectanglePrimitive((0.0, 0.0), 20.0, 10.0)
                ),
                fill=gradient,
            ),
        ),
    )
    symmetric = Drawing(
        "drw_symmetric",
        Canvas(24.0, 14.0),
        (
            DrawingRegion(
                "target",
                geometry=PathGeometryWithSymmetry(
                    geometry_from_svg_d(
                        "M0 0 L10 0 L10 10 L0 10 Z", region_id="target-half"
                    ),
                    SymmetryAxis((10.0, 0.0), 90.0),
                ),
                fill=gradient,
            ),
        ),
    )

    def pixels(drawing):
        png = resvg_py.svg_to_bytes(
            svg_string=render_drawing(drawing).svg, width=24, height=14
        )
        return np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGBA"))

    assert np.array_equal(pixels(symmetric), pixels(concrete))


def test_general_affine_clone_preserves_radial_paint_to_one_edge_quantum():
    import io

    import numpy as np
    from PIL import Image
    import resvg_py

    transform = AffineTransform(1.5, 0.25, 0.75, 2.0, 25.0, 10.0)
    gradient = RadialGradientFill(
        (4.0, 4.0),
        5.6,
        (GradientStop(0.0, "#000000"), GradientStop(1.0, "#ffffff")),
        focal_point=(2.4, 3.2),
        transform=transform,
    )

    def point(x, y):
        return (
            transform.a * x + transform.c * y + transform.tx,
            transform.b * x + transform.d * y + transform.ty,
        )

    concrete = Drawing(
        "drw_affine_concrete",
        Canvas(64.0, 48.0),
        (
            DrawingRegion(
                "target",
                geometry=PrimitiveGeometry(
                    PolygonPrimitive(tuple(point(x, y) for x, y in ((0, 0), (8, 0), (8, 8), (0, 8))))
                ),
                fill=gradient,
            ),
        ),
    )
    clone = Drawing(
        "drw_affine_clone",
        Canvas(64.0, 48.0),
        (
            DrawingRegion(
                "source",
                geometry=PrimitiveGeometry(RectanglePrimitive((0.0, 0.0), 8.0, 8.0)),
                fill=None,
            ),
            DrawingRegion(
                "target",
                geometry=CloneGeometry("source", (transform,)),
                fill=gradient,
                z=1,
            ),
        ),
    )

    def pixels(drawing):
        png = resvg_py.svg_to_bytes(
            svg_string=render_drawing(drawing).svg, width=64, height=48
        )
        return np.asarray(Image.open(io.BytesIO(bytes(png))).convert("RGBA"))

    actual, expected = pixels(clone), pixels(concrete)
    assert np.array_equal(actual[..., 3], expected[..., 3])
    opaque = actual[..., 3] == 255
    assert np.array_equal(actual[opaque, :3], expected[opaque, :3])
    alpha = actual[..., 3:4].astype(np.float64) / 255.0
    premultiplied_delta = np.abs(
        actual[..., :3].astype(np.float64) * alpha
        - expected[..., :3].astype(np.float64) * alpha
    )
    # resvg rounds the two equivalent clip-edge rasterizations independently;
    # their field and alpha agree, with at most one 8-bit premultiplied quantum.
    assert premultiplied_delta.max() <= 1.01
    assert np.count_nonzero(np.any(premultiplied_delta > 0, axis=2)) <= 8

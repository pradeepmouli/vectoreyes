import numpy as np
import pytest

from vectoreyes.path_geometry import (
    CirclePrimitive, EllipsePrimitive, PolygonPrimitive, PrimitiveGeometry,
    RectanglePrimitive, RoundedRectanglePrimitive,
)
from vectoreyes.primitive_fitting import fit_primitive_geometry
from vectoreyes.skia_geometry import SkPath
from vectoreyes.geometry_native import region_path
from vectoreyes.drawing import DrawingRegion


def test_recognition_returns_canonical_primitive_without_working_scene():
    from vectoreyes.primitive_fitting import recognize_primitive_geometry
    from vectoreyes.skia_geometry import SkPath
    footprint = SkPath.from_svg_d('M2 3 L22 3 L22 15 L2 15 Z')
    before = footprint.to_svg_d()
    assert recognize_primitive_geometry(footprint, epsilon=0.1) == PrimitiveGeometry(
        RectanglePrimitive((2, 3), 20, 12))
    assert footprint.to_svg_d() == before


@pytest.mark.parametrize('d', [
    'M0 0 L20 0 L20 20 L0 20 Z M5 5 L5 15 L15 15 L15 5 Z',
    'M0 0 L4 0 L4 4 L0 4 Z M10 0 L14 0 L14 4 L10 4 Z',
])
def test_primitive_recognition_preserves_holes_and_disconnected_regions(d):
    from vectoreyes.primitive_fitting import recognize_primitive_geometry
    from vectoreyes.skia_geometry import SkPath
    footprint = SkPath.from_svg_d(d)
    assert recognize_primitive_geometry(footprint, epsilon=2) is None


@pytest.mark.parametrize(('kind', 'expected'), [
    ('rect', RectanglePrimitive((3., 2.), 8., 6.)),
    ('rounded_rect', RoundedRectanglePrimitive((3., 2.), 8., 6., 0., 0.)),
    ('circle', CirclePrimitive((7., 5.), 3.)),
    ('ellipse', EllipsePrimitive((7., 5.), 4., 3.)),
])
def test_primitive_fitter_returns_canonical_geometry_without_mutating_vector(kind, expected):
    footprint = region_path(DrawingRegion('r1', PrimitiveGeometry(expected)), {})
    before = footprint.to_svg_d()
    result = fit_primitive_geometry(footprint, kind, epsilon=.1, target_id='r1')
    fitted = region_path(DrawingRegion('r1', result), {})
    assert type(result.primitive) is type(expected)
    assert fitted.symmetric_difference(footprint).area < .001
    assert footprint.to_svg_d() == before


def test_polygon_primitive_uses_retained_contour():
    footprint = SkPath.from_svg_d('M3 2 L11 2 L11 8 L3 8 Z')
    result = fit_primitive_geometry(footprint, 'polygon', epsilon=1., target_id='r1')
    assert isinstance(result, PrimitiveGeometry)
    assert isinstance(result.primitive, PolygonPrimitive)
    assert len(result.primitive.vertices) == 4


def test_primitive_fitter_rejects_empty_vector():
    with pytest.raises(ValueError, match='non-empty, hole-free vector component'):
        fit_primitive_geometry(SkPath.from_svg_d(''), 'circle', epsilon=1., target_id='r1')


def test_primitive_fitter_rejects_unsupported_kind():
    with pytest.raises(ValueError, match='unsupported primitive'):
        fit_primitive_geometry(SkPath.from_svg_d('M0 0 L8 0 L8 8 L0 8 Z'),
            'cap', epsilon=1., target_id='r1')


def test_primitive_fitter_rejects_raster_input():
    with pytest.raises(ValueError, match='vector component'):
        fit_primitive_geometry(np.ones((8, 8, 3)), 'rect', epsilon=1., target_id='r1')


def test_rounded_primitive_retains_measured_radius():
    import vectoreyes.primitive_fitting as fitting
    footprint = region_path(DrawingRegion('r1', PrimitiveGeometry(
        RoundedRectanglePrimitive((0, 0), 40, 30, 6., 6.))), {})
    result = fitting.fit_primitive_geometry(footprint, 'rounded_rect', epsilon=.1, target_id='r1')
    assert result == fitting.recognize_primitive_geometry(footprint, epsilon=.1)
    assert result.primitive.radius_x == pytest.approx(6, abs=.1)

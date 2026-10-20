import numpy as np
import pytest

from vectormark.path_geometry import (
    CirclePrimitive, EllipsePrimitive, PolygonPrimitive, PrimitiveGeometry,
    RectanglePrimitive, RoundedRectanglePrimitive,
)
from vectormark.primitive_fitting import fit_primitive_geometry


def test_recognition_returns_canonical_primitive_without_working_scene():
    from vectormark.primitive_fitting import recognize_primitive_geometry
    from vectormark.skia_geometry import SkPath
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
    from vectormark.primitive_fitting import recognize_primitive_geometry
    from vectormark.skia_geometry import SkPath
    footprint = SkPath.from_svg_d(d)
    assert recognize_primitive_geometry(footprint, epsilon=2) is None


@pytest.mark.parametrize(('kind', 'expected'), [
    ('rect', RectanglePrimitive((3., 2.), 8., 6.)),
    ('rounded_rect', RoundedRectanglePrimitive((3., 2.), 8., 6., 0., 0.)),
    ('circle', CirclePrimitive((7., 5.), 3.)),
    ('ellipse', EllipsePrimitive((7., 5.), 4., 3.)),
])
def test_primitive_fitter_returns_canonical_geometry_without_mutating_raster(kind, expected):
    raster = np.zeros((12, 16), dtype=bool)
    raster[2:8, 3:11] = True
    before = raster.copy()
    assert fit_primitive_geometry(raster, kind, epsilon=1., target_id='r1') == PrimitiveGeometry(expected)
    np.testing.assert_array_equal(raster, before)


def test_polygon_primitive_uses_retained_contour():
    raster = np.zeros((12, 16), dtype=bool)
    raster[2:8, 3:11] = True
    result = fit_primitive_geometry(raster, 'polygon', epsilon=1., target_id='r1')
    assert isinstance(result, PrimitiveGeometry)
    assert isinstance(result.primitive, PolygonPrimitive)
    assert len(result.primitive.vertices) == 4


def test_primitive_fitter_rejects_empty_raster():
    with pytest.raises(ValueError, match='empty drawing region'):
        fit_primitive_geometry(np.zeros((8, 8), dtype=bool), 'circle', epsilon=1., target_id='r1')


def test_primitive_fitter_rejects_unsupported_kind():
    with pytest.raises(ValueError, match='unsupported primitive'):
        fit_primitive_geometry(np.ones((8, 8), dtype=bool), 'cap', epsilon=1., target_id='r1')


def test_primitive_fitter_rejects_non_planar_raster():
    with pytest.raises(ValueError, match='two-dimensional raster'):
        fit_primitive_geometry(np.ones((8, 8, 3)), 'rect', epsilon=1., target_id='r1')


def test_rounded_primitive_retains_measured_radius(monkeypatch):
    import vectormark.primitive_fitting as fitting
    monkeypatch.setattr(fitting, 'region_corner_radius', lambda mask: 2.)
    result = fitting.fit_primitive_geometry(np.ones((8, 10), dtype=bool),
        'rounded_rect', epsilon=1., target_id='r1')
    assert result == PrimitiveGeometry(RoundedRectanglePrimitive((0, 0), 10, 8, 2., 2.))

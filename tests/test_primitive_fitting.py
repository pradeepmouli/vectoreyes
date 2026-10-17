import numpy as np
import pytest

from vectormark.path_geometry import (
    CirclePrimitive, EllipsePrimitive, PolygonPrimitive, PrimitiveGeometry,
    RectanglePrimitive, RoundedRectanglePrimitive,
)
from vectormark.primitive_fitting import fit_primitive_geometry


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

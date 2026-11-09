import math

import pytest

from vectoreyes.drawing import DrawingRegion
from vectoreyes.drawing_fill import FlatFill
from vectoreyes.drawing_symmetry import Axis2D, set_symmetry
from vectoreyes.geometry_native import region_path
from vectoreyes.path_geometry import geometry_from_svg_d
from vectoreyes.skia_geometry import SkPath


@pytest.mark.parametrize('side', ['positive', 'negative'])
def test_symmetry_mirrors_retained_curves_and_cutouts_without_fitting(side):
    d = ('M10 0 C3 0 0 3 0 10 C0 17 3 20 10 20 '
         'C18 20 21 16 21 10 C21 4 18 0 10 0 Z '
         'M3 8 L5 8 L5 12 L3 12 Z')
    source = DrawingRegion('r1', geometry_from_svg_d(d, region_id='r1', fill_rule='evenodd'),
        FlatFill('#123456'))
    footprint = region_path(source, {})
    clip = SkPath(shell=[(-100, -100), (10, -100), (10, 100), (-100, 100)])
    half = footprint.intersection(clip) if side == 'positive' else footprint.difference(clip)
    expected = half.union(half.affine_transform((-1, 0, 0, 1, 20, 0)))

    result = set_symmetry(source, footprint, axis=Axis2D(math.pi / 2, 10, 10), side=side)

    actual = region_path(result, {})
    assert actual.symmetric_difference(expected).area < 0.01
    assert actual.symmetric_difference(actual.affine_transform((-1, 0, 0, 1, 20, 0))).area < 1e-6
    assert result.fill == source.fill
    assert source.geometry.to_svg_d() == geometry_from_svg_d(d, region_id='r1', fill_rule='evenodd').to_svg_d()


def test_symmetry_detection_never_fits_a_half(monkeypatch):
    from vectoreyes import symmetry_detection as detection

    def forbidden(*args, **kwargs):
        pytest.fail('symmetry detection must not fit a half')

    for name in ('_symmetric_refine_half_shapes', '_fit_geometry_to_path_shape', 'fit_path_half_fit'):
        monkeypatch.setattr(detection, name, forbidden, raising=False)
    footprint = SkPath.from_svg_d('M0 0 L20 0 L20 20 L0 20 Z')
    candidate = detection._best_self_reconstruction(footprint)

    assert candidate is not None
    assert candidate.residual < 1e-6

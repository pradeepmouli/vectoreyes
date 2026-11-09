import pytest

from vectoreyes.drawing import DrawingRegion
from vectoreyes.drawing_actions import ActionSettings, FitPrimitiveAction
from vectoreyes.drawing_region_actions import fit_region_primitive
from vectoreyes.geometry_native import region_path
from vectoreyes.path_geometry import (
    CirclePrimitive, EllipsePrimitive, PrimitiveGeometry, RectanglePrimitive,
    RoundedRectanglePrimitive,
)
from vectoreyes.primitive_fitting import recognize_primitive_geometry
from vectoreyes.primitive_fitting import fit_primitive_geometry
from vectoreyes.skia_geometry import SkPath


@pytest.mark.parametrize(('kind', 'primitive'), [
    ('rect', RectanglePrimitive((3.25, 4.5), 24, 18)),
    ('circle', CirclePrimitive((32.25, 24.5), 12)),
    ('ellipse', EllipsePrimitive((32.25, 24.5), 18, 10)),
    ('ellipse', EllipsePrimitive((32.25, 24.5), 10, 18)),
    ('rounded_rect', RoundedRectanglePrimitive((3.25, 4.5), 40, 30, 6, 6)),
])
def test_explicit_and_automatic_primitive_parameters_agree(kind, primitive):
    original = DrawingRegion('r1', PrimitiveGeometry(primitive))
    footprint = region_path(original, {'r1': original})
    automatic = recognize_primitive_geometry(footprint, epsilon=.2)
    assert automatic is not None
    result = fit_region_primitive(original,
        FitPrimitiveAction(op='fit_primitive', target='r1', type=kind),
        footprint=footprint,
        settings=ActionSettings(epsilon=.2, max_error=1, fit_strategy='progressive'))
    assert result.geometry == automatic
    assert original.geometry == PrimitiveGeometry(primitive)


def test_explicit_type_selection_does_not_require_automatic_type_admission():
    footprint = SkPath.from_svg_d('M0 0 L20 0 L20 10 L0 10 Z')
    assert isinstance(recognize_primitive_geometry(footprint, epsilon=.1).primitive,
        RectanglePrimitive)
    assert isinstance(fit_primitive_geometry(footprint, 'circle', epsilon=.1,
        target_id='r1').primitive, CirclePrimitive)


@pytest.mark.parametrize('d', [
    'M0 0 L20 0 L20 20 L0 20 Z M5 5 L5 15 L15 15 L15 5 Z',
    'M0 0 L4 0 L4 4 L0 4 Z M10 0 L14 0 L14 4 L10 4 Z',
])
@pytest.mark.parametrize('kind', ['rect', 'rounded_rect', 'circle', 'ellipse', 'polygon'])
def test_explicit_primitive_does_not_silently_drop_holes_or_components(d, kind):
    footprint = SkPath.from_svg_d(d)
    before = footprint.to_svg_d()
    with pytest.raises(ValueError, match='one non-empty, hole-free vector component'):
        fit_primitive_geometry(footprint, kind, epsilon=1, target_id='r1')
    assert footprint.to_svg_d() == before

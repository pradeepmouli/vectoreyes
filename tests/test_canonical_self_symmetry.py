import math

import pytest

from vectoreyes.drawing import DrawingRegion
from vectoreyes.drawing_fill import FlatFill
from vectoreyes.path_geometry import PathGeometryWithCutouts, geometry_from_svg_d
from vectoreyes.skia_geometry import SkPath


def test_self_symmetry_reconstructs_canonical_region_without_changing_paint_or_source():
    from vectoreyes.drawing_symmetry import Axis2D, set_symmetry

    points = [(48 + 20 * math.cos(t * math.tau / 32),
        48 + 30 * math.sin(t * math.tau / 32)) for t in range(32)]
    d = 'M' + ' L'.join(f'{x} {y}' for x, y in points) + ' Z'
    source = DrawingRegion(id='r1', geometry=geometry_from_svg_d(d, region_id='r1'),
        fill=FlatFill('#123456'), source_region_ids=('source',))

    result = set_symmetry(source, SkPath.from_svg_d(d),
        axis=Axis2D(math.pi / 2, 48, 48), residual=0, side='positive')

    assert result.id == source.id
    assert result.fill == source.fill
    assert result.source_region_ids == ('source',)
    assert result.diagnostics['symmetry']['materialized'] is True
    assert source.diagnostics == {}


def test_canonical_symmetry_union_preserves_hole_and_rejects_mixed_paint():
    from vectoreyes.drawing_symmetry import union_symmetric_regions

    ring = SkPath.from_svg_d('M0 0 L20 0 L20 20 L0 20 Z M5 5 L5 15 L15 15 L15 5 Z')
    child = DrawingRegion(id='r1-a', geometry=geometry_from_svg_d(ring.to_svg_d(),
        region_id='r1-a', fill_rule='evenodd'), fill=FlatFill('#123456'))
    source = DrawingRegion(id='r1', children=(child,))

    result = union_symmetric_regions(source, (ring,), diagnostics={'mode': 'self'})

    assert isinstance(result.geometry, PathGeometryWithCutouts)
    assert result.fill == child.fill
    assert result.diagnostics['materialization']['consumed_target_ids'] == ('r1-a',)
    other = DrawingRegion(id='r1-b', geometry=child.geometry, fill=FlatFill('#abcdef'))
    from dataclasses import replace
    with pytest.raises(ValueError, match='one shared fill'):
        union_symmetric_regions(replace(source, children=(child, other)), (ring,), diagnostics={})


def test_materialized_disconnected_symmetry_union_renders_as_real_components():
    from vectoreyes.drawing import Drawing, Canvas
    from vectoreyes.drawing_render import render_drawing
    from vectoreyes.drawing_symmetry import union_symmetric_regions

    first = SkPath.from_svg_d('M0 0 L10 0 L10 10 L0 10 Z')
    second = first.affine_transform((1, 0, 0, 1, 20, 0))
    source = DrawingRegion('r1', geometry_from_svg_d(first.to_svg_d(), region_id='r1'),
        fill=FlatFill('#123456'))

    result = union_symmetric_regions(source, (first, second), diagnostics={'mode': 'self'})
    rendered = render_drawing(Drawing(drawing_id='test', canvas=Canvas(30, 10), regions=(result,)))

    assert result.is_branch and len(result.children) == 2
    from xml.etree import ElementTree
    painted = ElementTree.fromstring(rendered.svg).findall('{http://www.w3.org/2000/svg}path')
    assert [element.attrib['id'] for element in painted] == ['r1-1', 'r1-2']
    assert sum(SkPath.from_svg_d(child.geometry.to_svg_d()).area for child in result.children) == 200

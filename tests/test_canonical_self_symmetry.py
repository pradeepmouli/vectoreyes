import math

import pytest

from vectormark.drawing import DrawingRegion
from vectormark.drawing_fill import FlatFill
from vectormark.path_geometry import PathGeometryWithCutouts, geometry_from_svg_d
from vectormark.skia_geometry import SkPath


def test_self_symmetry_reconstructs_canonical_region_without_changing_paint_or_source():
    from vectormark.drawing_symmetry import Axis2D, reconstruct_self_symmetry

    points = [(48 + 20 * math.cos(t * math.tau / 32),
        48 + 30 * math.sin(t * math.tau / 32)) for t in range(32)]
    d = 'M' + ' L'.join(f'{x} {y}' for x, y in points) + ' Z'
    source = DrawingRegion(id='r1', geometry=geometry_from_svg_d(d, region_id='r1'),
        fill=FlatFill('#123456'), source_region_ids=('source',))

    result = reconstruct_self_symmetry(source, SkPath.from_svg_d(d),
        axis=Axis2D(math.pi / 2, 48, 48), residual=0, fit_kind='symmetric_half',
        side='left', corner_radius=0, epsilon=1, max_error=1, cubic=False)

    assert result.id == source.id
    assert result.fill == source.fill
    assert result.source_region_ids == ('source',)
    assert result.diagnostics['symmetry']['materialized'] is True
    assert source.diagnostics == {}


def test_canonical_symmetry_union_preserves_hole_and_rejects_mixed_paint():
    from vectormark.drawing_symmetry import union_symmetric_regions

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
    from vectormark.drawing import Drawing, Canvas
    from vectormark.drawing_render import render_drawing
    from vectormark.drawing_symmetry import union_symmetric_regions

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

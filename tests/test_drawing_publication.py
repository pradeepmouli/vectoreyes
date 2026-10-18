from dataclasses import replace

import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion
from vectormark.path_geometry import (
    CirclePrimitive, CornerAnchor, EllipsePrimitive, PolygonPrimitive,
    PrimitiveGeometry, RectanglePrimitive, RoundedRectanglePrimitive,
    geometry_from_svg_d,
)


@pytest.mark.parametrize('primitive,kind,params', [
    (CirclePrimitive((3, 4), 2), 'circle', {'cx': 3, 'cy': 4, 'r': 2}),
    (EllipsePrimitive((3, 4), 2, 1), 'ellipse', {'cx': 3, 'cy': 4, 'rx': 2, 'ry': 1}),
    (RectanglePrimitive((1, 2), 4, 5), 'rect', {'x': 1, 'y': 2, 'w': 4, 'h': 5}),
    (RoundedRectanglePrimitive((1, 2), 4, 5, 1, 2), 'rect',
     {'x': 1, 'y': 2, 'w': 4, 'h': 5, 'rx': 1, 'ry': 2}),
    (PolygonPrimitive(((0, 0), (8, 0), (0, 8))), 'polygon',
     {'points': ((0, 0), (8, 0), (0, 8))}),
])
def test_retained_primitive_publication_preserves_evidence_and_paint(primitive, kind, params):
    from vectormark.drawing_adapter import publish_primitive_geometry, primitive_geometry_from_shape
    original = Shape('rect', {'x': 0, 'y': 0, 'w': 8, 'h': 8})
    raster = np.ones((10, 10), dtype=bool)
    target = VectorRegion(id=7, current=original, original=original, raster=raster,
        fill=FlatFill('#ABCDEF'), z=3, diagnostics={'retained': True})
    updated = publish_primitive_geometry(target, PrimitiveGeometry(primitive))
    assert updated.current == Shape(kind, params)
    assert primitive_geometry_from_shape(updated.current) == PrimitiveGeometry(primitive)
    assert updated.original is target.original
    assert np.array_equal(updated.raster, target.raster) and updated.fill is target.fill
    assert updated.z == 3 and updated.id == 7
    assert updated.diagnostics['retained'] is True
    assert updated.footprint.area > 0
    assert target.current is original


def annotated():
    path = geometry_from_svg_d('M0 0 L10 0 L10 10 L0 10 Z', region_id='r1')
    command = path.path.commands[1].id
    return replace(path, anchors=(CornerAnchor('r1.p0.corner0', command, command),))


def test_automatic_primitive_replay_publishes_selected_values_without_refitting(monkeypatch):
    from vectormark import drawing_adapter
    from vectormark.optimizer.automatic_replay import optimize_with_intent, replay_batches
    from vectormark.optimizer.passes.primitives import primitives_pass
    from vectormark.optimizer.trace import trace_regions
    from vectormark.pipeline import Options
    yy, xx = np.ogrid[:100, :100]
    image = np.full((100, 100, 3), 255, np.uint8)
    image[(yy - 50) ** 2 + (xx - 50) ** 2 <= 30 ** 2] = (200, 30, 30)
    regions, masks = trace_regions(image, Options())
    accepted = optimize_with_intent(regions, masks, [primitives_pass])
    calls = []
    publish = drawing_adapter.publish_primitive_geometry
    def record(target, geometry, **kwargs):
        calls.append(geometry)
        return publish(target, geometry, **kwargs)
    monkeypatch.setattr(drawing_adapter, 'publish_primitive_geometry', record)
    def forbidden(*args, **kwargs):
        pytest.fail('retained primitive replay ran recognition')
    monkeypatch.setattr('vectormark.optimizer.passes.primitives.recognize_primitive', forbidden)
    replayed = replay_batches(regions, accepted.batches, masks=masks)
    assert replayed[0].current == accepted.objects[0].current
    assert replayed[0].fill == accepted.objects[0].fill
    assert calls == [accepted.batches[0].operations[0].intent.action.primitive]


def test_corner_publication_keeps_downstream_precision_and_siblings():
    from vectormark.drawing_adapter import restore_corner_annotations
    raw = 'M0 0 L10.123456789 0 L10 10 L0 10 Z'
    shape = Shape('path', {'d': raw, 'path_ids': ('r1.p0',), 'debug': 'preserved'})
    target = VectorRegion(id=1, current=shape, fill=None, drawing_id='r1', raster=np.ones((12, 12), dtype=bool))
    sibling = VectorRegion(id=2, current=Shape('rect', {'x': 20, 'y': 0, 'w': 2, 'h': 2}), fill=None, drawing_id='r2')
    result = restore_corner_annotations((target, sibling), target_region_id='r1', annotated=annotated())
    assert result[1] is sibling
    assert result[0].current.params['d'] == raw
    assert result[0].current.params['debug'] == 'preserved'
    canonical = result[0].current.params['canonical_path_geometry']
    assert canonical.path.commands[1].end == (10.123456789, 0)
    assert canonical.anchors == annotated().anchors
    assert 'canonical_path_geometry' not in target.current.params


@pytest.mark.parametrize('shape,reason', [
    (Shape('rect', {'x': 0, 'y': 0, 'w': 10, 'h': 10}), 'not a path'),
    (Shape('path', {'d': 'M0 0 Q5 0 10 0 L10 10 L0 10 Z'}), 'command topology'),
])
def test_corner_publication_rejects_incompatible_downstream_geometry(shape, reason):
    from vectormark.drawing_adapter import restore_corner_annotations
    target = VectorRegion(id=1, current=shape, fill=None, drawing_id='r1')
    with pytest.raises(ValueError, match=reason):
        restore_corner_annotations((target,), target_region_id='r1', annotated=annotated())

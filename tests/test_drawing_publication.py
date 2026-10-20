from dataclasses import replace

import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.drawing import Canvas
from vectormark.fit import Shape
from vectormark.refinement.region_sample import RegionSample
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
    from vectormark.drawing import DrawingRegion
    from vectormark.drawing_adapter import publish_selected_leaf
    from vectormark.primitive_fitting import primitive_geometry_from_shape
    original = Shape('rect', {'x': 0, 'y': 0, 'w': 8, 'h': 8})
    raster = np.ones((10, 10), dtype=bool)
    target = RegionSample(id=7, current=original, original=original, raster=raster,
        fill=FlatFill('#ABCDEF'), z=3, diagnostics={'retained': True})
    selected = DrawingRegion(id='r7', geometry=PrimitiveGeometry(primitive), diagnostics=target.diagnostics)
    updated = publish_selected_leaf(selected, target)
    assert updated.current == Shape(kind, params)
    assert primitive_geometry_from_shape(updated.current) == PrimitiveGeometry(primitive)
    assert updated.original is target.original
    assert np.array_equal(updated.raster, target.raster) and updated.fill is target.fill
    assert updated.z == 3 and updated.id == 7
    assert updated.diagnostics['retained'] is True
    assert updated.footprint.area > 0
    assert target.current is original


def test_selected_leaf_publication_retains_canonical_path_and_source_identity():
    from vectormark.drawing import DrawingRegion
    from vectormark.drawing_adapter import publish_selected_leaf
    path = annotated()
    target = RegionSample(id=7, current=Shape('rect', {'x': 0, 'y': 0, 'w': 8, 'h': 8}),
        fill=FlatFill('#ABCDEF'), source_regions=('material-1',),
        drawing_id=None, raster=np.ones((12, 12), dtype=bool))
    updated = publish_selected_leaf(DrawingRegion(id='selected', geometry=path), target)
    assert updated.current.params['canonical_path_geometry'] is path
    assert updated.current.params['d'] == 'M0 0 L10 0 L10 10 L0 10 Z'
    assert updated.footprint.area == pytest.approx(100)
    assert updated.source_regions == ('material-1',)
    assert updated.drawing_id is None and updated.id == 7
    assert updated.fill is target.fill and np.array_equal(updated.raster, target.raster)


def test_selected_leaf_publication_does_not_restore_retired_corner_diagnostics():
    from vectormark.drawing import DrawingRegion
    from vectormark.drawing_adapter import publish_selected_leaf
    target = RegionSample(id=7, current=Shape('rect', {'x': 0, 'y': 0, 'w': 8, 'h': 8}),
        fill=FlatFill('#ABCDEF'), diagnostics={'corners': {'stale': True}, 'retained': True})
    selected = DrawingRegion(id='r7', geometry=PrimitiveGeometry(CirclePrimitive((4, 4), 3)),
        diagnostics={'retained': True})
    updated = publish_selected_leaf(selected, target)
    assert updated.diagnostics == {'retained': True}
    assert target.diagnostics['corners'] == {'stale': True}


def annotated():
    path = geometry_from_svg_d('M0 0 L10 0 L10 10 L0 10 Z', region_id='r1')
    command = path.path.commands[1].id
    return replace(path, anchors=(CornerAnchor('r1.p0.corner0', command, command),))


def test_automatic_primitive_replay_publishes_selected_values_without_refitting(monkeypatch):
    from vectormark.refinement.automatic_replay import select_refinements_with_intent, replay_batches
    from vectormark.refinement.passes.primitives import primitives_pass
    from vectormark.refinement.trace import trace_regions
    from vectormark.pipeline import Options
    yy, xx = np.ogrid[:100, :100]
    image = np.full((100, 100, 3), 255, np.uint8)
    image[(yy - 50) ** 2 + (xx - 50) ** 2 <= 30 ** 2] = (200, 30, 30)
    regions, masks = trace_regions(image, Options())
    accepted = select_refinements_with_intent(regions, masks, [primitives_pass])
    def forbidden(*args, **kwargs):
        pytest.fail('retained primitive replay ran recognition')
    monkeypatch.setattr('vectormark.refinement.passes.primitives.recognize_primitive_geometry', forbidden)
    monkeypatch.setattr('vectormark.primitive_fitting.recognize_primitive', forbidden)
    replayed = replay_batches(regions, accepted.batches, masks=masks)
    assert replayed[0].current == accepted.objects[0].current
    assert replayed[0].fill == accepted.objects[0].fill
    from vectormark.primitive_fitting import primitive_geometry_from_shape
    assert primitive_geometry_from_shape(replayed[0].current) == accepted.batches[0].operations[0].intent.action.primitive


def test_corner_publication_keeps_downstream_precision_and_siblings():
    from vectormark.drawing_replay import restore_corner_annotations
    from vectormark.refinement.replay_binding import capture_replay_state
    raw = 'M0 0 L10.123456789 0 L10 10 L0 10 Z'
    shape = Shape('path', {'d': raw, 'path_ids': ('r1.p0',), 'debug': 'preserved'})
    target = RegionSample(id=1, current=shape, fill=None, drawing_id='r1', raster=np.ones((12, 12), dtype=bool))
    sibling = RegionSample(id=2, current=Shape('rect', {'x': 20, 'y': 0, 'w': 2, 'h': 2}), fill=None, drawing_id='r2')
    state = capture_replay_state((target, sibling), Canvas(30, 12))
    result = restore_corner_annotations(state, target_region_id='r1', annotated=annotated())
    assert result.drawing.region('r2') is state.drawing.region('r2')
    assert result.shape_metadata['r1']['debug'] == 'preserved'
    assert result.region_evidence == state.region_evidence
    canonical = result.drawing.region('r1').geometry
    assert canonical.path.commands[1].end == (10.123456789, 0)
    assert canonical.anchors == annotated().anchors
    assert 'canonical_path_geometry' not in target.current.params


@pytest.mark.parametrize('shape,reason', [
    (Shape('rect', {'x': 0, 'y': 0, 'w': 10, 'h': 10}), 'not a path'),
    (Shape('path', {'d': 'M0 0 Q5 0 10 0 L10 10 L0 10 Z'}), 'command topology'),
])
def test_corner_publication_rejects_incompatible_downstream_geometry(shape, reason):
    from vectormark.drawing_replay import restore_corner_annotations
    from vectormark.refinement.replay_binding import capture_replay_state
    target = RegionSample(id=1, current=shape, fill=None, drawing_id='r1')
    state = capture_replay_state((target,), Canvas(12, 12))
    with pytest.raises(ValueError, match=reason):
        restore_corner_annotations(state, target_region_id='r1', annotated=annotated())


@pytest.mark.parametrize('branch', [False, True])
def test_root_metadata_publication_keeps_selected_geometry_and_evidence(branch):
    from vectormark.drawing_adapter import restore_working_root_metadata

    original = RegionSample(1, Shape('rect', {'x': 0, 'y': 0, 'w': 8, 'h': 8}),
        None, drawing_id='r7', source_regions=('material3',))
    child = RegionSample(2, Shape('circle', {'cx': 4, 'cy': 4, 'r': 3}),
        FlatFill('#123456'), diagnostics={'selected': True})
    selected = RegionSample.branch(id=1, children=(child,)) if branch else child
    result = restore_working_root_metadata(selected, original)
    assert result.drawing_id == 'r7'
    assert result.source_regions == ('material3',)
    assert result.current is selected.current
    assert result.children == selected.children
    assert result.diagnostics == selected.diagnostics
    assert np.array_equal(result.raster, selected.raster)
    assert selected.drawing_id is None
    assert restore_working_root_metadata(selected, None) is selected


@pytest.mark.parametrize('explicit_addresses', [False, True])
def test_working_evidence_uses_the_same_scoped_addresses_as_canonical_geometry(explicit_addresses):
    from vectormark.drawing_adapter import working_region_evidence

    shape = Shape('rect', {'x': 0, 'y': 0, 'w': 2, 'h': 2})
    children = tuple(RegionSample(1, shape, None,
        drawing_id=f'r{index}-1' if explicit_addresses else None,
        raster=np.full((3, 3), index == 1, dtype=bool),
        coverage=np.full((3, 3), float(index == 1))) for index in (1, 2))
    roots = tuple(RegionSample.branch(id=index, drawing_id=f'r{index}', children=(child,))
        for index, child in enumerate(children, 1))
    evidence = working_region_evidence(roots, baseline=roots)
    assert set(evidence) == {'r1', 'r1-1', 'r2', 'r2-1'}
    for address in evidence:
        assert evidence[address]['raster_source'] == address
        assert evidence[address]['raster'] is None
        assert evidence[address]['coverage_source'] == address
        assert evidence[address]['coverage'] is None


def test_canonical_child_publication_preserves_implicit_address_raster():
    from vectormark.drawing_adapter import drawing_from_region_samples, publish_region_update

    shape = Shape('rect', {'x': 0, 'y': 0, 'w': 2, 'h': 2})
    left = RegionSample(1, shape, None, raster=np.ones((3, 3), dtype=bool))
    right = RegionSample(2, shape, None, raster=np.zeros((3, 3), dtype=bool))
    root = RegionSample.branch(id=10, drawing_id='r1', children=(left, right))
    drawing = drawing_from_region_samples('scope', width=3, height=3, regions=(root,))
    selected = replace(drawing.region('r1-2'), z=5)
    result = publish_region_update(selected, (root,))
    assert np.array_equal(result[0].children[0].raster, left.raster)
    assert np.array_equal(result[0].children[1].raster, right.raster)
    assert result[0].children[1].z == 5

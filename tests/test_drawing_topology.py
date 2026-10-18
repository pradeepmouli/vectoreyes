import pytest

from vectormark.drawing import DrawingRegion
from vectormark.drawing_fill import FlatFill
from vectormark.path_geometry import PathGeometryWithCutouts, parse_svg_paths


def test_working_normalizer_uses_canonical_publication(monkeypatch):
    import numpy as np
    import vectormark.drawing_topology as publication
    from vectormark.candidate import FlatFill as WorkingFill
    from vectormark.fit import Shape
    from vectormark.optimizer.topology import normalize_retained_topology
    from vectormark.optimizer.vector_region import VectorRegion
    source = VectorRegion.from_shape(id=1, shape=Shape('path', {
        'd': 'M0 0 L4 0 L4 4 L0 4 Z M8 0 L12 0 L12 4 L8 4 Z'}),
        fill=WorkingFill('#123456'), z=0, raster=np.ones((16, 16), dtype=bool), drawing_id='r1')
    calls = []
    execute = publication.publish_region_paths
    def record(*args, **kwargs):
        calls.append(args[0].id)
        return execute(*args, **kwargs)
    monkeypatch.setattr(publication, 'publish_region_paths', record)
    result = normalize_retained_topology((source,))
    assert result[0].is_branch
    assert calls == ['r1']


@pytest.mark.parametrize('hole_first', [False, True])
def test_canonical_path_publication_preserves_path_ownership_and_region_metadata(hole_first):
    from vectormark.drawing_topology import publish_region_paths
    outer = 'M0 0 L8 0 L8 8 L0 8 Z'
    hole = 'M2 2 L4 2 L4 4 L2 4 Z'
    other = 'M10 0 L14 0 L14 4 L10 4 Z'
    paths = parse_svg_paths(' '.join((hole, outer, other) if hole_first else (outer, hole, other)), region_id='r2-1')
    source = DrawingRegion('r2-1', paths[0], fill=FlatFill('#123456'), z=9,
        source_region_ids=('source',), diagnostics={'label': 3})
    result = publish_region_paths(source, paths, fill_rule='evenodd')
    assert result.is_branch
    assert [child.id for child in result.children] == ['r2-1-2' if hole_first else 'r2-1-1', 'r2-1-3']
    assert isinstance(result.children[0].geometry, PathGeometryWithCutouts)
    assert result.children[0].geometry.outer is paths[1 if hole_first else 0]
    for child in result.children:
        assert child.fill is source.fill and child.z == source.z
        assert child.source_region_ids == source.source_region_ids
        assert child.diagnostics['label'] == 3
    receipt = result.diagnostics['topology']
    assert receipt['created_target_ids'] == tuple(child.id for child in result.children)
    assert dict(receipt['path_owner_remap'])[paths[0].path.id] == result.children[0].id
    assert source.is_leaf


def test_canonical_publication_retires_redundant_paths_without_renumbering():
    from vectormark.drawing_topology import publish_region_paths
    paths = parse_svg_paths('M0 0 L8 0 L8 8 L0 8 Z M2 2 L4 2 L4 4 L2 4 Z M10 0 L14 0 L14 4 L10 4 Z', region_id='r1')
    region = DrawingRegion('r1', paths[0])
    first = publish_region_paths(region, paths, fill_rule='nonzero')
    assert [child.id for child in first.children] == ['r1-1', 'r1-3']
    assert first.diagnostics['topology']['retired_path_ids'] == ('r1.p1',)
    again = publish_region_paths(region, (paths[0], paths[2]), fill_rule='nonzero',
        source_path_order=tuple(path.path.id for path in paths))
    assert [child.id for child in again.children] == ['r1-1', 'r1-3']


@pytest.mark.parametrize('order', [('r1.p0', 'r1.p0'), ('absent',)])
def test_canonical_publication_rejects_invalid_original_path_slots(order):
    from vectormark.drawing_topology import publish_region_paths
    paths = parse_svg_paths('M0 0 L4 0 L4 4 L0 4 Z', region_id='r1')
    region = DrawingRegion('r1', paths[0])
    with pytest.raises(ValueError, match='component path order'):
        publish_region_paths(region, paths, fill_rule='nonzero', source_path_order=order)


def test_canonical_publication_keeps_unresolved_geometry_explicit():
    from vectormark.drawing_topology import publish_region_paths
    from vectormark.path_geometry import CompoundPathGeometry
    paths = parse_svg_paths('M0 0 L4 0 L4 4 L0 4 Z M2 2 L6 2 L6 6 L2 6 Z', region_id='r1')
    region = DrawingRegion('r1', paths[0])
    with pytest.raises(ValueError, match='overlapping'):
        publish_region_paths(region, paths, fill_rule='evenodd')
    result = publish_region_paths(region, paths, fill_rule='evenodd', defer_overlaps=True)
    assert isinstance(result.geometry, CompoundPathGeometry)
    assert result.diagnostics['topology']['pending'] is True

from dataclasses import replace

import pytest

from vectormark.drawing import Canvas, Drawing, DrawingRegion
from vectormark.drawing_fill import FlatFill
from vectormark.path_geometry import (
    AffineTransform, CloneGeometry, CompoundPathGeometry, CornerAnchor,
    PathGeometryWithCutouts, parse_svg_paths,
)


def source_region():
    paths = parse_svg_paths(
        'M0 0 L10 0 L10 10 L0 10 Z M2 2 L2 4 L4 4 L4 2 Z '
        'M20 0 L30 0 L30 10 L20 10 Z', region_id='r1')
    paths = tuple(replace(path, path=replace(path.path, commands=tuple(
        replace(command, id=f'{path.path.id}.c{3 * index}')
        for index, command in enumerate(path.path.commands)))) for path in paths)
    paths = (replace(paths[0], anchors=(CornerAnchor('corner', 'r1.p0.c3', 'r1.p0.c3'),)), *paths[1:])
    return DrawingRegion('r1', CompoundPathGeometry(paths, 'evenodd', 'input'),
        FlatFill('#123456'), source_region_ids=('source',)), paths


def test_canonical_normalization_splits_components_without_refitting_commands_or_holes():
    from vectormark.drawing_topology import normalize_drawing_topology

    source, paths = source_region()
    drawing = Drawing('drawing', Canvas(80, 80), (source,))
    result = normalize_drawing_topology(drawing)

    first, second = result.region('r1').children
    assert (first.id, second.id) == ('r1-1', 'r1-3')
    assert isinstance(first.geometry, PathGeometryWithCutouts)
    assert first.geometry.outer is paths[0]
    assert first.geometry.cutouts == (paths[1],)
    assert second.geometry is paths[2]
    assert first.source_region_ids == ('source',)
    assert source.geometry.paths == paths and drawing.region('r1').is_leaf


@pytest.mark.parametrize('transforms', [
    (AffineTransform(1, 0, 0, 1, 40, 0),),
    (AffineTransform(1, 0, 0, 1, 40, 0), AffineTransform(1, 0, 0, 1, 60, 0)),
])
def test_canonical_normalization_expands_forward_clones_and_preserves_their_paint(transforms):
    from vectormark.drawing_topology import normalize_drawing_topology

    source, _ = source_region()
    clone = DrawingRegion('copy', CloneGeometry('r1', transforms), FlatFill('#abcdef'), z=9)
    result = normalize_drawing_topology(Drawing('drawing', Canvas(100, 80), (clone, source)))

    children = result.region('copy').children
    assert [child.id for child in children] == ['copy-1', 'copy-3']
    assert [child.geometry.source_region_id for child in children] == ['r1-1', 'r1-3']
    assert all(child.geometry.transforms == transforms for child in children)
    assert all(child.fill == FlatFill('#abcdef') and child.z == 9 for child in children)


def test_canonical_normalization_is_idempotent_and_keeps_unchanged_values():
    from vectormark.drawing_topology import normalize_drawing_topology

    source, _ = source_region()
    result = normalize_drawing_topology(Drawing('drawing', Canvas(80, 80), (source,)))

    assert normalize_drawing_topology(result) is result


def test_canonical_normalization_rejects_cycles_before_publishing_partial_output():
    from vectormark.drawing_topology import normalize_drawing_topology

    source, _ = source_region()
    first = DrawingRegion('a', CloneGeometry('b', (AffineTransform(),)))
    second = DrawingRegion('b', CloneGeometry('a', (AffineTransform(),)))
    drawing = Drawing('drawing', Canvas(80, 80), (source, first, second))

    with pytest.raises(ValueError, match='cyclic'):
        normalize_drawing_topology(drawing)
    assert drawing.region('r1').is_leaf


def test_canonical_normalization_requires_explicit_external_dependency_permission():
    from vectormark.drawing_topology import normalize_drawing_topology

    clone = DrawingRegion('copy', CloneGeometry('outside', (AffineTransform(),)))
    drawing = Drawing('drawing', Canvas(80, 80), (clone,))

    with pytest.raises(ValueError, match='source.*retained'):
        normalize_drawing_topology(drawing)
    assert normalize_drawing_topology(drawing, allow_external=True) is drawing


def test_working_publication_expands_multi_transform_clone_over_selected_components():
    from vectormark.candidate import FlatFill as WorkingFill
    from vectormark.optimizer.topology_binding import normalize_working_topology
    from vectormark.fit import Shape
    from vectormark.optimizer.vector_region import VectorRegion
    from vectormark.skia_geometry import unary_union

    source, _ = source_region()
    working = VectorRegion(1, Shape('path', {'d': source.geometry.to_svg_d(), 'fill_rule': 'evenodd'}),
        WorkingFill('#123456'), drawing_id='r1')
    transforms = (AffineTransform(1, 0, 0, 1, 40, 0), AffineTransform(1, 0, 0, 1, 60, 0))
    footprint = unary_union([working.footprint.affine_transform((1, 0, 0, 1, x, 0)) for x in (40, 60)])
    clone = VectorRegion(2, Shape('path', {'d': footprint.to_svg_d(),
        'canonical_clone_geometry': CloneGeometry('r1', transforms)}), WorkingFill('#abcdef'),
        footprint=footprint, drawing_id='copy')

    result = normalize_working_topology((clone, working))

    assert result[0].is_branch
    assert [child.current.params['canonical_clone_geometry'].source_region_id
        for child in result[0].children] == ['r1-1', 'r1-3']
    assert all(child.fill is clone.fill for child in result[0].children)

import numpy as np
import pytest

from vectoreyes.candidate import FlatFill
from vectoreyes.drawing_actions import MergeAction, SplitAction
from vectoreyes.fit import Shape
from vectoreyes.refinement.region_sample import RegionSample
from vectoreyes.drawing_adapter import drawing_from_region_samples, publish_drawing_update


def leaf(identity, x, *, drawing_id=None, source="source"):
    raster = np.zeros((20, 40), dtype=bool)
    raster[2:12, int(x) : int(x) + 10] = True
    return RegionSample.from_shape(
        id=identity,
        shape=Shape("rect", {"x": x, "y": 2, "w": 10, "h": 10}),
        fill=FlatFill("#123456"),
        z=identity,
        raster=raster,
        drawing_id=drawing_id,
        source_regions=(source,),
    )


def test_partition_actions_consume_canonical_drawing_without_working_regions():
    from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
    from vectoreyes.drawing_fill import FlatFill as CanonicalFill
    from vectoreyes.path_geometry import PrimitiveGeometry, RectanglePrimitive
    from vectoreyes.geometry_native import region_path
    from vectoreyes.drawing_partition import merge_regions, split_region
    source = DrawingRegion('orange', PrimitiveGeometry(RectanglePrimitive((0, 0), 10, 10)),
        fill=CanonicalFill('#123456'), source_region_ids=('material-a',))
    neighbor = DrawingRegion('teal', PrimitiveGeometry(RectanglePrimitive((10, 0), 10, 10)),
        fill=source.fill, source_region_ids=('material-b',))
    drawing = Drawing('drawing', Canvas(20, 10), (source, neighbor))
    merged = merge_regions(drawing, MergeAction(op='merge', id='band', regions=('orange', 'teal')))
    assert tuple(region.id for region in merged.regions) == ('band',)
    band = merged.region('band')
    assert band.source_region_ids == ('material-a', 'material-b')
    assert region_path(band, {'band': band}).area == pytest.approx(200)
    split = split_region(drawing, SplitAction(op='split', target='orange',
        divider={'type': 'line', 'points': ((5, 0), (5, 10))}))
    assert split.region('teal') is neighbor
    pieces = split.region('orange').children
    assert tuple(piece.id for piece in pieces) == ('orange-1', 'orange-2')
    assert all(piece.fill is source.fill and piece.source_region_ids == ('material-a',) for piece in pieces)
    assert sum(region_path(piece, {}).area for piece in pieces) == pytest.approx(100)
    assert drawing.regions == (source, neighbor)


def test_canonical_merge_publication_retains_union_of_source_observations():
    from vectoreyes.drawing_adapter import drawing_from_region_samples, publish_drawing_update
    from vectoreyes.drawing_partition import merge_regions
    left, right = leaf(1, 0, drawing_id='r1', source='a'), leaf(2, 10, drawing_id='r2', source='b')
    drawing = drawing_from_region_samples('merge', width=40, height=20, regions=(left, right))
    result = publish_drawing_update(merge_regions(drawing,
        MergeAction(op='merge', id='band', regions=('r1', 'r2'))), (left, right))
    assert len(result) == 1 and result[0].drawing_id == 'band'
    np.testing.assert_array_equal(result[0].raster, left.raster | right.raster)
    assert result[0].source_regions == ('a', 'b')
    assert result[0].footprint.area == pytest.approx(200)


def test_merge_engine_removes_only_selected_scoped_children():
    from vectoreyes.drawing_partition import merge_regions

    selected, sibling = leaf(1, 0, source="a"), leaf(2, 20, source="b")
    root = RegionSample.branch(id=10, drawing_id="r1", children=(selected, sibling))
    other = leaf(1, 10, drawing_id="r2", source="c")
    drawing = drawing_from_region_samples('partition', width=40, height=20, regions=(root, other))
    result = publish_drawing_update(merge_regions(
        drawing, MergeAction(op="merge", id="merged", regions=("r1-1", "r2"))
    ), (root, other))
    assert result[0].children == (sibling,)
    assert result[0].children[0] is sibling
    assert result[1].drawing_id == "merged"
    assert result[1].source_regions == ("a", "c")
    assert result[1].footprint.area == pytest.approx(200)
    assert result[1].fill == selected.fill
    assert np.array_equal(result[1].raster, selected.raster | other.raster)
    assert root.children == (selected, sibling)


@pytest.mark.parametrize(
    "divider",
    [
        {"type": "line", "points": ((5, 0), (5, 20))},
        {"type": "path", "d": "M5 2 Q7 7 5 12"},
    ],
)
def test_split_engine_preserves_scope_union_and_source_evidence(divider):
    from vectoreyes.drawing_adapter import working_leaf_targets
    from vectoreyes.drawing_partition import split_region

    source, sibling = leaf(1, 0, source="a"), leaf(2, 20, source="b")
    root = RegionSample.branch(id=10, drawing_id="r1", children=(source, sibling))
    drawing = drawing_from_region_samples('partition', width=40, height=20, regions=(root,))
    result = publish_drawing_update(split_region(
        drawing, SplitAction(op="split", target="r1-1", divider=divider)
    ), (root,))
    split = result[0].children[0]
    first, second = split.children
    assert result[0].children[1] is sibling
    assert split.footprint.symmetric_difference(source.footprint).area == 0
    assert (
        first.footprint.union(second.footprint)
        .symmetric_difference(source.footprint)
        .area
        < 1e-6
    )
    assert first.footprint.intersection(second.footprint).area < 1e-6
    assert first.source_regions == second.source_regions == ("a",)
    assert first.fill == second.fill == source.fill
    assert first.id == 1 and second.id == 2
    assert set(working_leaf_targets(result)) == {"r1-1-1", "r1-1-2", "r1-2"}
    assert np.array_equal(split.raster, source.raster)
    assert root.children == (source, sibling)


@pytest.mark.parametrize(
    "divider,message",
    [
        ({"type": "line", "points": ((30, 0), (30, 20))}, "two non-empty"),
        ({"type": "path", "d": "M5 5 Q7 7 5 9"}, "endpoints must lie"),
    ],
)
def test_split_engine_rejects_invalid_divider_without_mutating_input(divider, message):
    from vectoreyes.drawing_partition import split_region

    source = leaf(1, 0, drawing_id="r1")
    shape = source.current
    drawing = drawing_from_region_samples('partition', width=40, height=20, regions=(source,))
    geometry = drawing.region('r1').geometry
    with pytest.raises(ValueError, match=message):
        split_region(drawing, SplitAction(op="split", target="r1", divider=divider))
    assert drawing.region('r1').geometry is geometry
    assert source.current is shape
    assert source.is_leaf


@pytest.mark.parametrize('divider', [
    'M6 8 Q0 10 6 12',
    'M6 12 Q0 10 6 8',
    'M10 0 L10 20',
])
def test_path_split_preserves_existing_holes(divider):
    from vectoreyes.drawing_partition import split_region
    from vectoreyes.geometry_native import region_path
    from vectoreyes.skia_geometry import SkPath

    shape = Shape('path', {'d': (
        'M0 0 L20 0 L20 20 L0 20 Z '
        'M6 6 L6 14 L14 14 L14 6 Z '
        'M2 2 L2 4 L4 4 L4 2 Z'
    ), 'fill_rule': 'evenodd'})
    source = RegionSample(1, shape, FlatFill('#123456'), 0, drawing_id='r1')
    drawing = drawing_from_region_samples('holes', width=30, height=20, regions=(source,))
    original = region_path(drawing.region('r1'), {})

    result = split_region(drawing, SplitAction(
        op='split', target='r1', divider={'type': 'path', 'd': divider},
    ))

    children = result.region('r1').children
    a, b = (region_path(child, {}) for child in children)
    assert not a.is_empty and not b.is_empty
    assert a.intersection(b).area < 1e-6
    assert a.union(b).symmetric_difference(original).area < 1e-6
    holes = SkPath.from_svg_d('M6 6 L14 6 L14 14 L6 14 Z M2 2 L4 2 L4 4 L2 4 Z')
    assert a.union(b).intersection(holes).area < 1e-6
    assert all(child.fill == drawing.region('r1').fill for child in children)
    assert drawing.region('r1').is_leaf


def test_path_split_does_not_discard_disconnected_geometry():
    from vectoreyes.drawing_partition import _split_footprint_by_path
    from vectoreyes.skia_geometry import SkPath

    footprint = SkPath.from_svg_d(
        'M0 0 L20 0 L20 20 L0 20 Z M25 0 L28 0 L28 3 L25 3 Z'
    )
    a, b = _split_footprint_by_path(footprint, 'M10 0 L10 20')
    assert a.union(b).symmetric_difference(footprint).area < 1e-6
    assert a.intersection(b).area < 1e-6


def test_path_split_rejects_endpoints_on_different_boundary_rings():
    from vectoreyes.drawing_partition import _split_footprint_by_path
    from vectoreyes.skia_geometry import SkPath

    footprint = SkPath.from_svg_d(
        'M0 0 L20 0 L20 20 L0 20 Z M6 6 L6 14 L14 14 L14 6 Z'
    )
    with pytest.raises(ValueError, match='same boundary'):
        _split_footprint_by_path(footprint, 'M0 10 L6 10')


@pytest.mark.parametrize('divider', ['M6 10 Q0 0 10 6', 'M10 6 Q0 0 6 10'])
def test_curved_hole_split_does_not_leave_remote_boundary_slivers(divider):
    from vectoreyes.drawing_partition import _split_footprint_by_path
    from vectoreyes.skia_geometry import SkPath

    source = SkPath.from_svg_d(
        'M0 0 L20 0 L20 20 L0 20 Z '
        'M6 10 C6 12.209139 7.790861 14 10 14 '
        'C8 12 8 8 10 6 '
        'C7.790861 6 6 7.790861 6 10 Z'
    )
    a, b = _split_footprint_by_path(source, divider)
    detail = min((a, b), key=lambda path: path.area)
    assert detail.bounds[2] <= 10.001
    assert detail.bounds[3] <= 10.001
    assert a.union(b).symmetric_difference(source).area < 1e-5
    assert a.intersection(b).area < 1e-5


def test_split_publication_preserves_native_evenodd_cutouts(monkeypatch):
    from vectoreyes import drawing_partition
    from vectoreyes.geometry_native import region_path
    from vectoreyes.skia_geometry import SkPath

    # PathOps may return even-odd contours with matching winding directions.
    left = SkPath.from_svg_d(
        'M0 0 L10 0 L10 20 L0 20 Z M2 2 L4 2 L4 4 L2 4 Z',
        fill_rule='evenodd',
    )
    right = SkPath.from_svg_d('M10 0 L20 0 L20 20 L10 20 Z')
    shape = Shape('path', {'d': (
        'M0 0 L20 0 L20 20 L0 20 Z M2 2 L4 2 L4 4 L2 4 Z'
    ), 'fill_rule': 'evenodd'})
    drawing = drawing_from_region_samples('parity', width=20, height=20, regions=(
        RegionSample(1, shape, FlatFill('#123456'), 0, drawing_id='r1'),
    ))
    monkeypatch.setattr(drawing_partition, '_split_footprint_by_path', lambda *_: (left, right))

    result = drawing_partition.split_region(drawing, SplitAction(
        op='split', target='r1', divider={'type': 'path', 'd': 'M10 0 L10 20'},
    ))

    actual = region_path(result.region('r1-1'), {})
    assert actual.symmetric_difference(left).area < 1e-6

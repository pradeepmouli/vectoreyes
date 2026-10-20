import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.drawing_actions import MergeAction, SplitAction
from vectormark.fit import Shape
from vectormark.refinement.region_sample import RegionSample
from vectormark.drawing_adapter import drawing_from_region_samples, publish_drawing_update


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
    from vectormark.drawing import Canvas, Drawing, DrawingRegion
    from vectormark.drawing_fill import FlatFill as CanonicalFill
    from vectormark.path_geometry import PrimitiveGeometry, RectanglePrimitive
    from vectormark.geometry_native import region_path
    from vectormark.drawing_partition import merge_regions, split_region
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
    from vectormark.drawing_adapter import drawing_from_region_samples, publish_drawing_update
    from vectormark.drawing_partition import merge_regions
    left, right = leaf(1, 0, drawing_id='r1', source='a'), leaf(2, 10, drawing_id='r2', source='b')
    drawing = drawing_from_region_samples('merge', width=40, height=20, regions=(left, right))
    result = publish_drawing_update(merge_regions(drawing,
        MergeAction(op='merge', id='band', regions=('r1', 'r2'))), (left, right))
    assert len(result) == 1 and result[0].drawing_id == 'band'
    np.testing.assert_array_equal(result[0].raster, left.raster | right.raster)
    assert result[0].source_regions == ('a', 'b')
    assert result[0].footprint.area == pytest.approx(200)


def test_merge_engine_removes_only_selected_scoped_children():
    from vectormark.drawing_partition import merge_regions

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
    from vectormark.drawing_adapter import working_leaf_targets
    from vectormark.drawing_partition import split_region

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
    from vectormark.drawing_partition import split_region

    source = leaf(1, 0, drawing_id="r1")
    shape = source.current
    drawing = drawing_from_region_samples('partition', width=40, height=20, regions=(source,))
    geometry = drawing.region('r1').geometry
    with pytest.raises(ValueError, match=message):
        split_region(drawing, SplitAction(op="split", target="r1", divider=divider))
    assert drawing.region('r1').geometry is geometry
    assert source.current is shape
    assert source.is_leaf

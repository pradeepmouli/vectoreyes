import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.drawing_actions import MergeAction, SplitAction
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion


def leaf(identity, x, *, drawing_id=None, source="source"):
    raster = np.zeros((20, 40), dtype=bool)
    raster[2:12, int(x) : int(x) + 10] = True
    return VectorRegion.from_shape(
        id=identity,
        shape=Shape("rect", {"x": x, "y": 2, "w": 10, "h": 10}),
        fill=FlatFill("#123456"),
        z=identity,
        raster=raster,
        drawing_id=drawing_id,
        source_regions=(source,),
    )


def test_merge_engine_removes_only_selected_scoped_children():
    from vectormark.drawing_partition import merge_regions

    selected, sibling = leaf(1, 0, source="a"), leaf(2, 20, source="b")
    root = VectorRegion.branch(id=10, drawing_id="r1", children=(selected, sibling))
    other = leaf(1, 10, drawing_id="r2", source="c")
    result = merge_regions(
        (root, other), MergeAction(op="merge", id="merged", regions=("r1-1", "r2"))
    )
    assert result[0].children == (sibling,)
    assert result[0].children[0] is sibling
    assert result[1].drawing_id == "merged"
    assert result[1].source_regions == ("a", "c")
    assert result[1].footprint.area == pytest.approx(200)
    assert result[1].fill is selected.fill
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
    root = VectorRegion.branch(id=10, drawing_id="r1", children=(source, sibling))
    result = split_region(
        (root,), SplitAction(op="split", target="r1-1", divider=divider)
    )
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
    assert first.fill is second.fill is source.fill
    assert first.id == 1 and second.id == 11
    assert set(working_leaf_targets(result)) == {"r1-1-1", "r1-1-11", "r1-2"}
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
    with pytest.raises(ValueError, match=message):
        split_region((source,), SplitAction(op="split", target="r1", divider=divider))
    assert source.current is shape
    assert source.is_leaf

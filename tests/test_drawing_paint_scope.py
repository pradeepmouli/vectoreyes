import numpy as np

from vectoreyes.refinement import paint_binding as paint
from vectoreyes.candidate import FlatFill
from vectoreyes.fit import Shape
from vectoreyes.refinement.gate import rasterize
from vectoreyes.refinement.region_sample import RegionSample, to_polygon


def forest(*, fills=(None, None)):
    roots = []
    for root_id, x, fill in zip((10, 20), (1, 11), fills, strict=True):
        shape = Shape("rect", dict(x=x, y=1, w=7, h=7))
        footprint = to_polygon(shape)
        leaf = RegionSample(
            1,
            shape,
            fill,
            footprint=footprint,
            raster=rasterize(footprint, (10, 20)),
            drawing_id=f"r{root_id}-1",
        )
        roots.append(RegionSample.branch(id=root_id, children=(leaf,), drawing_id=f'r{root_id}'))
    return tuple(roots)


def test_refill_keeps_source_paint_separate_for_repeated_local_child_ids():
    roots = forest()
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    rgb[:, :10] = (200, 20, 30)
    rgb[:, 10:] = (20, 40, 210)
    result = paint.refill_regions(roots, rgb, allow_gradients=False)
    assert [root.children[0].fill.hex.upper() for root in result] == [
        "#C8141E",
        "#1428D2",
    ]
    assert all(
        new.children[0].current is old.children[0].current
        for old, new in zip(roots, result, strict=True)
    )
    assert all(root.children[0].fill is None for root in roots)


def test_invalidation_preserves_unchanged_children_with_same_local_id():
    roots = forest(fills=(FlatFill("#C8141E"), FlatFill("#1428D2")))
    result = paint.refill_changed_regions_from_source(roots, roots,
        np.zeros((10, 20, 3), np.uint8), allow_gradients=False)
    assert result[0] is roots[0]
    assert result[1] is roots[1]


def test_invalidation_only_clears_the_changed_scoped_child():
    roots = forest(fills=(FlatFill("#C8141E"), FlatFill("#1428D2")))
    leaf = roots[1].children[0]
    changed = leaf.with_current(Shape("rect", dict(x=12, y=1, w=6, h=7)))
    current = (roots[0], roots[1].with_children((changed,)))
    result = paint.refill_changed_regions_from_source(roots, current,
        np.full((10, 20, 3), [20, 40, 210], np.uint8), allow_gradients=False)
    assert result[0] is roots[0]
    assert result[1].children[0].fill == FlatFill('#1428D2')
    assert result[1].children[0].diagnostics['fill']['status'] == 'refit'
    assert leaf.fill is not None


def test_self_symmetry_shared_paint_uses_its_own_scoped_visible_surface():
    roots = tuple(
        root.with_diagnostics({"symmetry": {"mode": "self"}}) for root in forest()
    )
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    rgb[:, :10] = (200, 20, 30)
    rgb[:, 10:] = (20, 40, 210)
    result = paint.refill_all_regions_from_source(roots, rgb, allow_gradients=False)
    assert [root.children[0].fill.hex.upper() for root in result] == [
        "#C8141E",
        "#1428D2",
    ]
    assert all(
        root.children[0].diagnostics["fill"]["shared_self_symmetry"] for root in result
    )


def test_terminal_output_limits_do_not_change_geometry_or_source_evidence():
    from vectoreyes.candidate import LinearGradientFill
    from vectoreyes.output_settings import DrawingOutputSettings

    gradient = LinearGradientFill(
        dict(x1=0, y1=0, x2=20, y2=0), [(0.0, "#FF0000"), (1.0, "#0000FF")]
    )
    roots = forest(fills=(gradient, FlatFill("#0000FF")))
    rgb = np.zeros((10, 20, 3), dtype=np.uint8)
    rgb[:, :10] = (255, 0, 0)
    rgb[:, 10:] = (0, 0, 255)
    result = paint.apply_output_fill_settings(
        roots, rgb, DrawingOutputSettings(allow_gradients=False, max_output_colors=2)
    )
    assert all(isinstance(root.children[0].fill, FlatFill) for root in result)
    for old, new in zip(roots, result, strict=True):
        assert new.children[0].current is old.children[0].current
        np.testing.assert_array_equal(new.children[0].raster, old.children[0].raster)
    assert roots[0].children[0].fill is gradient

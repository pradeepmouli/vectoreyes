from dataclasses import replace

import numpy as np

from vectormark.candidate import FlatFill
from vectormark.drawing import DrawingRegion
from vectormark.fit import Shape
from vectormark.optimizer.passes.smooth import smooth_pass
from vectormark.optimizer.vector_region import VectorRegion
from vectormark.path_geometry import (
    CornerAnchor, PathGeometryWithCutouts, parse_svg_paths,
)


D = "M0 0 Q2 9.8 10 10 Q18 9.8 20 0 L0 0 Z"
SMOOTHED = "M0 0 Q2 10 10 10 Q18 10 20 0 L0 0 Z"


def test_smoothing_publishes_updated_canonical_cache_not_only_svg_text():
    (geometry,) = parse_svg_paths(D, region_id="r7")
    shape = Shape("path", {"d": D, "canonical_path_geometry": geometry, "note": "retain"})
    target = VectorRegion(7, shape, FlatFill("#123456"))

    (proposal,) = smooth_pass([target], {}, max_error=1.1)
    (changed,) = proposal.new_objects

    assert changed.current.params["d"] == SMOOTHED
    assert changed.current.params["canonical_path_geometry"].to_svg_d() == SMOOTHED
    assert changed.current.params["note"] == "retain"
    assert changed.fill is target.fill and changed.original is target.original
    assert target.current.params["canonical_path_geometry"].to_svg_d() == D


def test_canonical_smoothing_retains_custom_command_ids_and_anchors():
    from vectormark.drawing_smoothing import smooth_region

    (geometry,) = parse_svg_paths(D, region_id="r7")
    commands = tuple(replace(command, id=f"custom-{index}")
                     for index, command in enumerate(geometry.path.commands))
    anchor = CornerAnchor("corner", "custom-3", "custom-3", radius=0)
    geometry = replace(geometry, path=replace(geometry.path, commands=commands), anchors=(anchor,))
    target = DrawingRegion("r7", geometry, z=3, source_region_ids=("source",),
                           diagnostics={"retained": True})

    changed = smooth_region(target, max_boundary_error=1.1)

    assert changed.geometry.to_svg_d() == SMOOTHED
    assert tuple(command.id for command in changed.geometry.path.commands) == (
        "custom-0", "custom-1", "custom-2", "custom-3")
    assert changed.geometry.anchors == (anchor,)
    assert changed.id == "r7" and changed.z == 3
    assert changed.source_region_ids == ("source",)
    assert changed.diagnostics == {"retained": True}
    assert smooth_region(changed, max_boundary_error=1.1) is None
    assert smooth_region(target, max_boundary_error=0) is None
    assert target.geometry.to_svg_d() == D


def test_canonical_smoothing_keeps_cutout_membership_and_unmodified_paths():
    from vectormark.drawing_smoothing import smooth_region

    outer, hole = parse_svg_paths(D + " M8 2 L12 2 L10 4 Z", region_id="r7")
    geometry = PathGeometryWithCutouts(outer, (hole,))
    target = DrawingRegion("r7", geometry)

    changed = smooth_region(target, max_boundary_error=1.1)

    assert isinstance(changed.geometry, PathGeometryWithCutouts)
    assert changed.geometry.outer.to_svg_d() == SMOOTHED
    assert changed.geometry.cutouts == (hole,)
    assert changed.geometry.fill_rule == "evenodd"


def test_smoothing_replay_retains_published_canonical_geometry():
    from vectormark.optimizer.automatic_replay import replay_batches
    from vectormark.optimizer.framework import optimize_with_intent

    (geometry,) = parse_svg_paths(D, region_id="r7")
    shape = Shape("path", {"d": D, "canonical_path_geometry": geometry})
    target = VectorRegion(7, shape, FlatFill("#123456"))
    masks = {7: np.ones((24, 24), dtype=bool)}
    optimized = optimize_with_intent([target], masks,
        [lambda objects, masks: smooth_pass(objects, masks, max_error=1.1)])
    replayed = replay_batches([target], optimized.batches, masks=masks)

    assert replayed[0].current.params["d"] == SMOOTHED
    assert replayed[0].current.params["canonical_path_geometry"].to_svg_d() == SMOOTHED


def test_smoothing_accepts_disconnected_intermediate_paths_without_repartitioning():
    d = D + " M30 0 L34 0 L34 4 L30 4 Z"
    target = VectorRegion(7, Shape("path", {"d": d}), FlatFill("#123456"))

    (proposal,) = smooth_pass([target], {}, max_error=1.1)
    (changed,) = proposal.new_objects

    assert changed.is_leaf and changed.id == 7
    assert changed.current.params["d"] == SMOOTHED + " M30 0 L34 0 L34 4 L30 4 Z"
    assert changed.fill is target.fill
    assert target.current.params["d"] == d

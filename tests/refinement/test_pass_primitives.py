import numpy as np

from vectormark.skia_geometry import SkPath
from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.refinement.trace import trace_regions
from vectormark.refinement.framework import select_refinements
from vectormark.refinement.region_sample import RegionSample
from vectormark.refinement.passes.primitives import primitives_pass
from vectormark.pipeline import Options


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def _disk_image():
    img = np.full((120, 120, 3), 255, np.uint8)
    img[_disk(120, 120, 60, 60, 40)] = (200, 30, 30)
    return img


def _irregular_blob_image():
    img = np.full((120, 120, 3), 255, np.uint8)
    mask = np.zeros((120, 120), dtype=bool)
    mask[25:95, 25:95] = True
    mask[25:50, 25:50] = False
    mask[60:105, 70:95] = True
    img[mask] = (40, 40, 40)
    return img


def test_primitives_pass_proposes_circle_for_trace_disk_path():
    objects, masks = trace_regions(_disk_image(), Options())

    assert len(objects) == 1
    assert objects[0].current.kind == "path"

    proposals = primitives_pass(objects, masks)

    assert len(proposals) == 1
    assert proposals[0].obj_ids == (objects[0].id,)
    assert len(proposals[0].new_objects) == 1
    assert proposals[0].new_objects[0].current.kind == "circle"


def test_optimize_applies_primitives_pass_to_trace_disk():
    objects, masks = trace_regions(_disk_image(), Options())
    original_fill = objects[0].fill

    out = select_refinements(objects, masks, [primitives_pass])

    assert len(out) == 1
    assert out[0].current.kind == "circle"
    assert out[0].fill == original_fill
    assessment = out[0].diagnostics["primitives_pass"]["geometry_assessment"]
    assert assessment["accepted"] is True
    assert assessment["policy"]["max_boundary_error_px"] == 1.5
    assert assessment["boundary_measurement"]


def test_primitives_pass_skips_irregular_blob():
    objects, masks = trace_regions(_irregular_blob_image(), Options())

    assert len(objects) == 1
    assert objects[0].current.kind == "path"

    proposals = primitives_pass(objects, masks)
    out = select_refinements(objects, masks, [primitives_pass])

    assert proposals == []
    assert len(out) == 1
    assert out[0].current.kind == "path"


def test_primitives_pass_rejects_partial_oval_segment_as_full_ellipse():
    segment = Shape(
        "path",
        {
            "d": (
                "M20 70 "
                "C20 42 56 20 100 20 "
                "C144 20 180 42 180 70 "
                "L20 70 Z"
            )
        },
    )
    obj = RegionSample(1, segment, FlatFill("#000000"), 0)

    proposals = primitives_pass([obj], {1: np.zeros((120, 220), dtype=bool)})
    out = select_refinements([obj], {1: np.zeros((120, 220), dtype=bool)}, [primitives_pass])

    assert len(proposals) == 1
    assert out[0].current.kind == "path"
    assessment = out[0].diagnostics["primitives_pass"]["geometry_assessment"]
    assert assessment["accepted"] is False
    assert "boundary_error_exceeded" in assessment["reasons"]


def test_primitives_pass_accepts_quantized_small_circular_cutout():
    flash = Shape(
        "path",
        {
            "d": (
                "M351.5 174 Q370.14 167.06 368.5 147 L366.5 142 "
                "L357 132.5 L341.5 131 L328.5 139 L324.5 149 "
                "Q324.17 168.06 342 174.5 L351.5 174 Z"
            )
        },
    )
    obj = RegionSample(1, flash, FlatFill("#FFFFFF"), 0)

    out = select_refinements([obj], {1: np.zeros((220, 220), dtype=bool)}, [primitives_pass])

    assert out[0].current.kind == "circle"


def test_primitives_pass_skips_multipolygon_to_preserve_all_components():
    large = SkPath.make_circle(40.0, 40.0, 18.0)
    small = SkPath(shell=[(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)])
    obj = RegionSample(
        id=7,
        current=Shape("path", {"d": "M0 0 L1 0 L1 1 L0 1 Z"}),
        fill=FlatFill("#000000"),
        z=0,
        footprint=small.union(large),
    )

    proposals = primitives_pass([obj], {7: np.zeros((80, 80), dtype=bool)})
    out = select_refinements([obj], {7: np.zeros((80, 80), dtype=bool)}, [primitives_pass])

    assert proposals == []
    assert out[0].current.kind == "path"
    assert abs(out[0].footprint.area - obj.footprint.area) < 1e-6


def test_primitives_pass_skips_polygon_with_counter():
    outer = SkPath.make_circle(40.0, 40.0, 24.0)
    inner = SkPath.make_circle(40.0, 40.0, 5.0)
    ring = outer.difference(inner)
    obj = RegionSample(
        id=8,
        current=Shape("path", {"d": "M0 0 L1 0 L1 1 L0 1 Z"}),
        fill=FlatFill("#000000"),
        z=0,
        footprint=ring,
    )

    proposals = primitives_pass([obj], {8: np.zeros((80, 80), dtype=bool)})

    assert proposals == []


def test_primitives_pass_skips_empty_and_nonpolygon_geometry():
    empty = RegionSample(
        id=1,
        current=Shape("path", {"d": "M0 0 L1 0 L1 1 L0 1 Z"}),
        fill=FlatFill("#000000"),
        z=0,
        footprint=SkPath(),
    )
    nonpolygon = RegionSample(
        id=2,
        current=Shape("path", {"d": "M0 0 L1 0 L1 1 L0 1 Z"}),
        fill=FlatFill("#000000"),
        z=0,
        footprint=SkPath(),
    )

    proposals = primitives_pass(
        [nonpolygon, empty],
        {
            1: np.zeros((4, 4), dtype=bool),
            2: np.zeros((4, 4), dtype=bool),
        },
    )

    assert proposals == []

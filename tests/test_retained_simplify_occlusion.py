import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.automatic_replay import (
    CompletedOcclusionPrimitive,
    IntersectionLens,
    PassInputRegionRef,
    ReconstructOcclusionOperation,
    SimplifyLineletOperation,
)
from vectormark.optimizer.vector_region import VectorRegion, to_polygon
from vectormark.path_geometry import CirclePrimitive, PrimitiveGeometry


class Resolver:
    def __init__(self, *regions):
        self.regions = {region.id: region for region in regions}

    def resolve(self, ref):
        return self.regions[ref.ancestry[0]]


def test_engine_simplification_applies_recorded_settings_and_preserves_covering_region():
    from vectormark.drawing_simplification import apply_simplification

    d = (
        "M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 "
        "Q256.31 399.74 253.5 413 L247 417.5 Z"
    )
    shape = Shape("path", {"d": d})
    target = VectorRegion(1, shape, FlatFill("#123456"), footprint=to_polygon(shape))
    covering = VectorRegion(
        2, Shape("rect", {"x": 0, "y": 0, "w": 1, "h": 1}), FlatFill("#abcdef")
    )
    action = SimplifyLineletOperation(
        PassInputRegionRef("", "", (1,)),
        (PassInputRegionRef("", "", (2,)),),
        1.0,
        1.0,
        16,
        False,
        True,
        1.0,
        True,
        True,
    )
    changed, retained = apply_simplification(action, Resolver(target, covering))
    assert changed.current.params["d"] != d
    assert changed.fill is target.fill
    assert changed.original is target.original
    assert retained is covering
    assert target.current.params["d"] == d


def test_engine_simplification_rejects_an_inapplicable_recorded_operation():
    from vectormark.drawing_simplification import apply_simplification

    target = VectorRegion(
        1, Shape("rect", {"x": 0, "y": 0, "w": 1, "h": 1}), FlatFill("#123456")
    )
    action = SimplifyLineletOperation(
        PassInputRegionRef("", "", (1,)), (), 1.0, 1.0, 16, False, True, 1.0, True, True
    )
    with pytest.raises(
        ValueError, match="recorded simplify operation no longer applies"
    ):
        apply_simplification(action, Resolver(target))
    assert target.current.kind == "rect"


def test_engine_occlusion_materializes_recorded_circles_and_their_lens():
    from vectormark.drawing_occlusion import apply_occlusion_reconstruction

    sources = [
        VectorRegion(
            i,
            Shape("circle", {"cx": x, "cy": 16, "r": 8}),
            FlatFill(color),
            raster=np.zeros((32, 32), dtype=bool),
        )
        for i, x, color in ((1, 12, "#ff0000"), (2, 20, "#ffff00"), (3, 16, "#ff8000"))
    ]
    sources[0].raster[10, 10] = True
    refs = [PassInputRegionRef("", "", (i,)) for i in (1, 2, 3)]
    action = ReconstructOcclusionOperation(
        tuple(refs),
        1,
        4.0,
        (32, 32),
        (
            CompletedOcclusionPrimitive(
                refs[0],
                "left",
                1,
                PrimitiveGeometry(CirclePrimitive((12.0, 16.0), 8.0)),
                0,
            ),
            CompletedOcclusionPrimitive(
                refs[1],
                "right",
                2,
                PrimitiveGeometry(CirclePrimitive((20.0, 16.0), 8.0)),
                1,
            ),
            IntersectionLens(refs[2], "lens", 3, "left", "right", 2),
        ),
    )
    (branch,) = apply_occlusion_reconstruction(action, Resolver(*sources))
    assert [child.current.kind for child in branch.children] == [
        "circle",
        "circle",
        "path",
    ]
    assert [child.fill.hex for child in branch.children] == [
        "#ff0000",
        "#ffff00",
        "#ff8000",
    ]
    assert [child.z for child in branch.children] == [4.0, 5.0, 6.0]
    left, right, lens = (child.footprint for child in branch.children)
    assert lens.symmetric_difference(left.intersection(right)).area < 0.1
    assert branch.raster[10, 10]
    assert branch.diagnostics["occlusion"]["consumed_ids"] == [1, 2, 3]
    assert all(source.is_leaf and source.diagnostics == {} for source in sources)


def test_engine_occlusion_rejects_missing_dependency_before_resolving_sources():
    from vectormark.drawing_occlusion import apply_occlusion_reconstruction

    ref = PassInputRegionRef("", "", (1,))
    action = ReconstructOcclusionOperation(
        (ref,),
        1,
        0.0,
        (32, 32),
        (IntersectionLens(ref, "lens", 1, "missing", "missing", 0),),
    )
    with pytest.raises(ValueError, match="absent dependency"):
        apply_occlusion_reconstruction(action, Resolver())

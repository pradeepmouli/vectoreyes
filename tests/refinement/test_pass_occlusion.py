import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.occlusion import ScenePrimitive
from vectormark.refinement.framework import select_refinements
from vectormark.refinement.passes.occlusion import occlusion_pass
import vectormark.refinement.passes.occlusion as occlusion_module
from vectormark.refinement.passes.primitives import primitives_pass
from vectormark.refinement.trace import trace_regions
from vectormark.refinement.region_sample import RegionSample
from vectormark.pipeline import Options


def _synthetic_mastercard(h=240, w=360, gap=58, r=104):
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    left = (xx - (cx - gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    right = (xx - (cx + gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    img = np.full((h, w, 3), 255, np.uint8)
    img[left] = (235, 0, 27)
    img[right] = (247, 158, 27)
    img[left & right] = (255, 95, 0)
    return img


def _adjacent_regions():
    left_mask = np.pad(np.ones((12, 12), dtype=bool), ((0, 20), (0, 20)))
    right_mask = np.pad(np.ones((12, 12), dtype=bool), ((0, 20), (12, 8)))
    left = RegionSample(
        10,
        Shape("path", {"d": "M0 0 L12 0 L12 12 L0 12 Z"}),
        FlatFill("#111111"),
        5,
        raster=left_mask,
    )
    right = RegionSample(
        20,
        Shape("path", {"d": "M12 0 L24 0 L24 12 L12 12 Z"}),
        FlatFill("#222222"),
        6,
        raster=right_mask,
    )
    return left, right, left_mask, right_mask


def test_occlusion_pass_replaces_adjacent_traces_with_scene_branch():
    regions, masks = trace_regions(_synthetic_mastercard(), Options())

    optimized = select_refinements(regions, masks, [primitives_pass, occlusion_pass])

    assert len(optimized) == 1
    branch = optimized[0]
    assert branch.is_branch
    assert branch.current is None
    assert branch.diagnostics["occlusion"]["accepted"] is True
    assert branch.diagnostics["occlusion"]["children"] == 3
    assert [leaf.current.kind for leaf in branch.leaves()] == ["circle", "circle", "path"]


def test_occlusion_pass_preserves_global_z_for_reconstructed_children(monkeypatch):
    def _recording_reconstruct_scene(regions, axis, shape_hw):
        return [
            ScenePrimitive("rect", {"x": 0.0, "y": 0.0, "w": 12.0, "h": 12.0}, "#111111", 0, 10),
            ScenePrimitive("rect", {"x": 12.0, "y": 0.0, "w": 12.0, "h": 12.0}, "#222222", 1, 20),
        ], []

    monkeypatch.setattr(occlusion_module, "reconstruct_scene", _recording_reconstruct_scene)
    left, right, left_mask, right_mask = _adjacent_regions()

    proposals = occlusion_pass([left, right], {left.id: left_mask, right.id: right_mask})

    assert len(proposals) == 1
    assert proposals[0].intent is not None
    branch = proposals[0].new_objects[0]
    assert branch.z == 5
    assert [child.z for child in branch.children] == [5, 6]


def test_occlusion_pass_rejects_incomplete_source_provenance_before_mutation(monkeypatch):
    def _unbound_reconstruct_scene(regions, axis, shape_hw):
        return [
            ScenePrimitive(
                "rect",
                {"x": 0.0, "y": 0.0, "w": 12.0, "h": 12.0},
                "#111111",
                0,
            ),
            ScenePrimitive(
                "rect",
                {"x": 12.0, "y": 0.0, "w": 12.0, "h": 12.0},
                "#222222",
                1,
            ),
        ], []

    monkeypatch.setattr(occlusion_module, "reconstruct_scene", _unbound_reconstruct_scene)
    left, right, left_mask, right_mask = _adjacent_regions()
    before = (left, right)

    with pytest.raises(
        ValueError, match="occlusion reconstruction lacks exact source/output provenance"
    ):
        occlusion_pass(
            [left, right], {left.id: left_mask, right.id: right_mask}
        )

    assert (left, right) == before

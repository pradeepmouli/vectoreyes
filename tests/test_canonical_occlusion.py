import numpy as np
import pytest

from vectormark.geometry_native import region_path
from vectormark.refinement.automatic_replay import (
    AnnulusPrimitive, CompletedOcclusionPrimitive, IntersectionLens,
    PassInputRegionRef, ReconstructOcclusionOperation, RetainedOcclusionSource,
)
from vectormark.path_geometry import CirclePrimitive, PathGeometryWithCutouts, PrimitiveGeometry


def test_canonical_occlusion_builds_completed_circles_and_dependent_lens():
    from vectormark.drawing_occlusion import OcclusionSource, reconstruct_occlusion

    left, right = PassInputRegionRef('', '', (1,)), PassInputRegionRef('', '', (2,))
    action = ReconstructOcclusionOperation((left, right), 1, 4, (24, 24), (
        CompletedOcclusionPrimitive(left, 'left', 1, PrimitiveGeometry(CirclePrimitive((8, 8), 6)), 0),
        CompletedOcclusionPrimitive(right, 'right', 2, PrimitiveGeometry(CirclePrimitive((12, 8), 6)), 1),
        IntersectionLens(left, 'lens', 3, 'left', 'right', 2),
    ))
    sources = {left: OcclusionSource(1, 'source-left', '#ff0000', np.ones((24, 24), bool)),
               right: OcclusionSource(2, 'source-right', '#0000ff', np.ones((24, 24), bool))}

    result = reconstruct_occlusion(action, sources, region_id='r1')

    assert result.is_branch and len(result.children) == 3
    assert isinstance(result.children[0].geometry.primitive, CirclePrimitive)
    assert result.children[0].fill.color == '#ff0000'
    assert result.children[1].fill.color == '#0000ff'
    assert result.children[2].z == 6
    assert 0 < region_path(result.children[2], {}).area < region_path(result.children[0], {}).area


def test_canonical_occlusion_annulus_retains_real_cutout_geometry():
    from vectormark.drawing_occlusion import OcclusionSource, reconstruct_occlusion

    ref = PassInputRegionRef('', '', (1,))
    action = ReconstructOcclusionOperation((ref,), 1, 0, (32, 32), (
        CompletedOcclusionPrimitive(ref, 'ring', 2, AnnulusPrimitive((16, 16), 10, 5), 0),))
    result = reconstruct_occlusion(action, {
        ref: OcclusionSource(1, 'source', '#123456', np.ones((32, 32), bool))}, region_id='r1')
    geometry = result.children[0].geometry

    assert isinstance(geometry, PathGeometryWithCutouts)
    assert region_path(result.children[0], {}).area == pytest.approx(np.pi * 75, rel=0.01)


def test_canonical_occlusion_rejects_missing_child_dependency_before_source_lookup():
    from vectormark.drawing_occlusion import reconstruct_occlusion

    ref = PassInputRegionRef('', '', (1,))
    action = ReconstructOcclusionOperation((ref,), 1, 0, (16, 16), (
        IntersectionLens(ref, 'lens', 2, 'absent', 'also-absent', 0),))

    with pytest.raises(ValueError, match='absent dependency'):
        reconstruct_occlusion(action, {}, region_id='r1')


def test_canonical_occlusion_retained_source_keeps_disconnected_components():
    from vectormark.drawing_occlusion import OcclusionSource, reconstruct_occlusion

    mask = np.zeros((24, 24), bool)
    mask[2:6, 2:6] = True
    mask[15:19, 15:19] = True
    ref = PassInputRegionRef('', '', (1,))
    action = ReconstructOcclusionOperation((ref,), 1, 0, mask.shape, (
        RetainedOcclusionSource(ref, 'source', 2, 0),))

    result = reconstruct_occlusion(action, {
        ref: OcclusionSource(1, 'source', '#123456', mask)}, region_id='r1')

    assert len(result.children[0].leaves()) == 2
    assert result.children[0].source_region_ids == ('source',)


def test_automatic_occlusion_publication_keeps_annulus_as_editable_cutout():
    from vectormark.candidate import FlatFill
    from vectormark.drawing_adapter import drawing_from_region_samples
    from vectormark.fit import Shape
    from vectormark.refinement.automatic_replay import apply_occlusion_choice
    from vectormark.refinement.region_sample import RegionSample

    ref = PassInputRegionRef('', '', (1,))
    source = RegionSample(1, Shape('circle', {'cx': 16, 'cy': 16, 'r': 10}),
        FlatFill('#123456'), raster=np.ones((32, 32), bool), drawing_id='r1')
    action = ReconstructOcclusionOperation((ref,), 1, 0, (32, 32), (
        CompletedOcclusionPrimitive(ref, 'ring', 2, AnnulusPrimitive((16, 16), 10, 5), 0),))

    (branch,) = apply_occlusion_choice(action, {ref: source}.__getitem__)
    drawing = drawing_from_region_samples('result', width=32, height=32, regions=(branch,))

    assert isinstance(drawing.regions[0].children[0].geometry, PathGeometryWithCutouts)
    assert branch.children[0].fill.hex == '#123456'
    assert branch.children[0].footprint.area == pytest.approx(np.pi * 75, rel=0.01)


def test_generated_occlusion_diagnostics_remain_structured_in_report():
    import json
    from types import MappingProxyType
    from vectormark.pipeline import _to_json_safe

    diagnostics = MappingProxyType({'occlusion': MappingProxyType({
        'reconstructed_child': True, 'consumed_ids': (1, 2, 3),
    })})

    result = json.loads(json.dumps(_to_json_safe(diagnostics)))

    assert result == {'occlusion': {
        'reconstructed_child': True, 'consumed_ids': [1, 2, 3],
    }}


def test_occlusion_replay_accepts_named_output_with_scoped_ancestry():
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.refinement.automatic_replay import (
        OperationOutputRegionRef, apply_occlusion_choice,
    )
    from vectormark.refinement.region_sample import RegionSample

    ref = OperationOutputRegionRef('earlier', 'leaf', (7,))
    source = RegionSample(7, Shape('circle', {'cx': 8, 'cy': 8, 'r': 5}),
        FlatFill('#123456'), raster=np.ones((16, 16), bool))
    action = ReconstructOcclusionOperation((ref,), 7, 0, (16, 16), (
        CompletedOcclusionPrimitive(ref, 'disk', 2, PrimitiveGeometry(CirclePrimitive((8, 8), 5)), 0),))

    (result,) = apply_occlusion_choice(action, {ref: source}.__getitem__,
        path_for=lambda selected: (3, 9, *selected.descendant_ancestry))

    assert result.drawing_id == 'r3-9-7'
    assert result.children[0].source_regions == ('r3-9-7',)

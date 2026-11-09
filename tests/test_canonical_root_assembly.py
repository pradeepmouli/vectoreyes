from dataclasses import replace

import numpy as np
import pytest

from vectoreyes.drawing import Drawing
from vectoreyes.drawing_engine import RasterImage
from vectoreyes.drawing_trace import PythonTraceEngine, TraceOptions


@pytest.mark.parametrize('with_rgb', [False, True])
def test_root_assembly_publishes_canonical_paths_and_detached_source_observations(with_rgb):
    from vectoreyes.drawing_assembly import assemble_drawing_roots
    from vectoreyes.drawing_recipe import build_geometry_evidence
    from vectoreyes.path_topology import component_paths

    rgb = np.full((32, 48, 3), 255, np.uint8)
    rgb[5:25, 4:20] = (200, 30, 40)
    rgb[7:27, 28:44] = (20, 90, 150)
    trace = PythonTraceEngine().trace(rgb, TraceOptions(min_region_size=1, max_sampling_colors=3))
    result = assemble_drawing_roots(trace, rgb if with_rgb else None,
        min_region_fraction=0, allow_gradients=False)

    assert isinstance(result.drawing, Drawing)
    assert [region.id for region in result.drawing.regions] == ['r1', 'r2']
    assert [region.fill.color for region in result.drawing.regions] == ['#c81e28', '#145a96']
    for region in result.drawing.regions:
        observation = result.region_evidence[region.id]
        assert observation['raster_shape'] == (32, 48)
        assert isinstance(observation['raster'], bytes)
        for path in component_paths(region.geometry):
            record = next(item for item in result.trace.root_path_evidence if item.path_id == path.path.id)
            assert record.region_id == region.id
            assert record.matches_output(path.to_svg_d(), stage=record.stage, revision=record.fit.revision)
    source = RasterImage(48, 32, 'rgb8', rgb.tobytes())
    evidence = build_geometry_evidence(kind='source', trace=result.trace, geometry_source=source,
        drawing=result.drawing, shape_metadata=result.shape_metadata, region_evidence=result.region_evidence)
    assert evidence.drawing == result.drawing
    saved = evidence.region_evidence['r1']['raster']
    rgb[:] = 0
    assert evidence.region_evidence['r1']['raster'] == saved


def test_canonical_evidence_snapshot_is_detached_without_working_publication():
    from vectoreyes.drawing import Canvas, DrawingRegion
    from vectoreyes.drawing_recipe import build_geometry_evidence
    from vectoreyes.path_geometry import geometry_from_svg_d

    geometry = geometry_from_svg_d('M1 1 L6 1 L6 6 L1 6 Z', region_id='r1')
    rgb = np.zeros((8, 8, 3), np.uint8)
    trace = PythonTraceEngine().trace(rgb, TraceOptions(remove_background='off'))
    drawing = Drawing('retained', Canvas(8, 8), (DrawingRegion('r1', geometry),))
    annotations = {'r1': {'source': [1, 2]}}
    result = build_geometry_evidence(kind='source', trace=trace,
        geometry_source=RasterImage(8, 8, 'rgb8', rgb.tobytes()), drawing=drawing,
        shape_metadata=annotations, region_evidence={})
    annotations['r1']['source'].append(3)
    assert result.shape_metadata['r1']['source'] == (1, 2)
    assert result.drawing.regions[0].geometry is not geometry
    assert result.drawing == drawing
    assert replace(result, revision='other').drawing == drawing

from dataclasses import replace

import numpy as np

from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
from vectoreyes.drawing_actions import ActionSettings
from vectoreyes.drawing_engine import RasterImage
from vectoreyes.drawing_fill import FlatFill
from vectoreyes.drawing_recipe import (
    EffectiveDrawingRecipe, ResolveOriginalPaintOperation,
    build_geometry_evidence, build_paint_evidence,
)
from vectoreyes.drawing_trace import PythonTraceEngine, TraceOptions
from vectoreyes.path_geometry import geometry_from_svg_d


def inputs():
    rgb = np.zeros((8, 8, 3), np.uint8)
    source = RasterImage(8, 8, 'rgb8', rgb.tobytes())
    trace = PythonTraceEngine().trace(rgb, TraceOptions(remove_background='off'))
    geometry = geometry_from_svg_d('M1 1 L6 1 L6 6 L1 6 Z', region_id='r1')
    drawing = Drawing('source', Canvas(8, 8), (DrawingRegion('r1', geometry, FlatFill('#123456')),))
    # Canonical-only execution must not need synthetic working-record fields.
    evidence = build_geometry_evidence(kind='source', trace=trace, geometry_source=source,
        drawing=drawing, shape_metadata={}, region_evidence={})
    paint = build_paint_evidence(source)
    recipe = EffectiveDrawingRecipe(evidence.revision, evidence.engine_revision,
        ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'),
        steps=(ResolveOriginalPaintOperation(scope='retain', share_root_gradients=True),))
    return evidence, paint, recipe


def test_retain_only_replay_needs_no_working_geometry_or_raster_codec():
    from vectoreyes.drawing_replay import replay_drawing

    evidence, paint, recipe = inputs()
    result = replay_drawing(evidence, paint, recipe)
    assert isinstance(result.drawing, Drawing)
    assert result.drawing.region('r1').fill == FlatFill('#123456')
    assert result.drawing.region('r1').geometry == evidence.drawing.region('r1').geometry
    assert result.region_evidence == {} and result.shape_metadata == {}


def test_replay_execution_retains_canonical_output_without_disposable_regions():
    from vectoreyes.drawing_replay import propose_effective_recipe

    evidence, paint, recipe = inputs()
    result = propose_effective_recipe(drawing_id='result', evidence=evidence, paint=paint, recipe=recipe)
    assert result.drawing.drawing_id == 'result'
    assert result.drawing.region('r1').geometry == evidence.drawing.region('r1').geometry
    assert result.replay.evidence_revision == evidence.revision
    assert result.region_evidence == {} and result.shape_metadata == {}


def test_replay_input_identity_ignores_svg_spelling_but_not_geometry_or_paint():
    from vectoreyes.candidate import FlatFill as WorkingFill
    from vectoreyes.fit import Shape
    from vectoreyes.refinement.replay_binding import capture_replay_state
    from vectoreyes.replay_identity import canonical_input_revision
    from vectoreyes.refinement.region_sample import RegionSample

    mask = np.ones((8, 8), dtype=bool)
    first = RegionSample.from_shape(id=1, shape=Shape('path', {'d': 'M1 1 L6 1 L6 6 L1 6 Z'}),
        fill=WorkingFill('#123456'), z=0, raster=mask, drawing_id='r1')
    equivalent = replace(first, current=Shape('path', {'d': 'M 1.0 1.0 L 6.0 1.0 L 6.0 6.0 L 1.0 6.0 Z'}))
    changed = replace(first, current=Shape('path', {'d': 'M1 1 L7 1 L6 6 L1 6 Z'}))
    def revision(region):
        return canonical_input_revision(capture_replay_state((region,), Canvas(8, 8)))
    assert revision(first) == revision(equivalent)
    assert revision(first) != revision(changed)
    assert revision(first) != revision(replace(first, fill=WorkingFill('#654321')))


def test_replay_capture_detaches_cached_command_objects_from_working_inputs():
    from vectoreyes.candidate import FlatFill as WorkingFill
    from vectoreyes.fit import Shape
    from vectoreyes.refinement.replay_binding import capture_replay_state
    from vectoreyes.refinement.region_sample import RegionSample

    geometry = geometry_from_svg_d('M1 1 L6 1 L6 6 L1 6 Z', region_id='r1')
    region = RegionSample.from_shape(id=1, shape=Shape('path', {
        'd': geometry.to_svg_d(), 'canonical_path_geometry': geometry}),
        fill=WorkingFill('#123456'), z=0, raster=np.ones((8, 8), dtype=bool), drawing_id='r1')
    state = capture_replay_state((region,), Canvas(8, 8))
    assert state.drawing.region('r1').geometry is not geometry
    object.__setattr__(geometry.path.commands[1], 'end', (7., 7.))
    assert state.drawing.region('r1').geometry.path.commands[1].end == (6., 1.)


def test_terminal_palette_projection_uses_canonical_geometry_without_working_observations():
    from vectoreyes.drawing_replay import replay_drawing
    from vectoreyes.output_settings import DrawingOutputSettings

    evidence, paint, recipe = inputs()
    paint = build_paint_evidence(paint.source, output_settings=DrawingOutputSettings(
        max_output_colors=1, allow_gradients=False))
    before = replay_drawing(evidence, paint, recipe, finalize_output=False)
    after = replay_drawing(evidence, paint, recipe)
    assert before.drawing.region('r1').fill == FlatFill('#123456')
    assert after.drawing.region('r1').fill == FlatFill('#000000')
    assert after.drawing.region('r1').geometry == before.drawing.region('r1').geometry
    assert after.region_evidence == {}


def test_all_source_paint_replay_needs_only_canonical_geometry_and_pixels():
    from vectoreyes.drawing_replay import replay_drawing

    evidence, paint, recipe = inputs()
    recipe = replace(recipe, steps=(ResolveOriginalPaintOperation(
        scope='all', share_root_gradients=False),))
    result = replay_drawing(evidence, paint, recipe, finalize_output=False)
    assert result.drawing.region('r1').fill == FlatFill('#000000')
    assert result.drawing.region('r1').geometry == evidence.drawing.region('r1').geometry
    assert result.drawing.region('r1').diagnostics['fill']['mask'] == 'visible_footprint'
    assert result.region_evidence == {} and result.shape_metadata == {}

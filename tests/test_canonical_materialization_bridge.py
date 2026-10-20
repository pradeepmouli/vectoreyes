from types import SimpleNamespace

from vectormark.drawing import Canvas
from vectormark.refinement.replay_binding import capture_replay_state


def test_materialization_keeps_cutout_and_remaps_exact_canonical_clone(monkeypatch):
    from tests.test_retained_symmetry_export import _forest
    from vectormark.drawing_materialization import materialize_self_symmetry_state
    from vectormark.geometry_native import region_path
    state = capture_replay_state(_forest(), Canvas(48, 24))
    def forbidden(*args, **kwargs):
        raise AssertionError('canonical materialization projected a sampling scene')
    monkeypatch.setattr('vectormark.refinement.region_sample.RegionSample.__init__', forbidden)
    result = materialize_self_symmetry_state(state)
    source = result.drawing.region('r1')
    assert source.is_leaf
    assert source.diagnostics['symmetry']['materialized']
    assert region_path(source, {'r1': source}).area == 192
    assert result.drawing.region('r2').geometry.source_region_id == 'r1'
    assert result.drawing.region('r2').fill == state.drawing.region('r2').fill
    assert materialize_self_symmetry_state(result) == result
    assert state.drawing.region('r1').is_branch


def test_public_geometry_action_materializes_without_a_working_scene(monkeypatch):
    from tests.test_retained_symmetry_export import _forest
    from tests.test_canonical_geometry_proposals import inputs
    from vectormark.drawing_actions import DrawingPlan
    from vectormark.drawing_replay import bind_recipe_operation, _propose_public_operation
    evidence, paint, recipe = inputs()
    state = capture_replay_state(_forest(), Canvas(48, 24))
    action = DrawingPlan.model_validate({'version': 'vectormark.plan.v1',
        'drawing_id': 'test', 'base_version': 'v0',
        'ops': [{'op': 'align', 'target': 'r1', 'reference': 'r2', 'axes': ['x']}]}).ops[0]
    operation = bind_recipe_operation(operation_id='align', action=action,
        settings=recipe.settings, evidence_revision=evidence.revision, state=state)
    def forbidden(*args, **kwargs):
        raise AssertionError('public materialization projected a sampling scene')
    monkeypatch.setattr('vectormark.refinement.region_sample.RegionSample.__init__', forbidden)
    result = _propose_public_operation(evidence, paint, operation, state, recipe=recipe, corner_replays=[])
    assert result.drawing.region('r1').is_leaf
    assert result.drawing.region('r2').geometry.source_region_id == 'r1'


def test_diagnostic_auto_adapter_uses_canonical_selection(monkeypatch):
    from tests.test_canonical_seam_discovery import regions
    from vectormark import drawing_refine
    def forbidden(*args, **kwargs):
        raise AssertionError('diagnostic adapter used the numerical selection loop')
    monkeypatch.setattr(drawing_refine, 'select_refinements_with_intent', forbidden)
    trace = SimpleNamespace(width=300, height=440, material_pretrace=None,
        options=SimpleNamespace(simplify_tolerance=1., curve_tolerance=1.,
            trace_level='pixel', max_sampling_colors=2, min_region_size=1))
    result = drawing_refine.auto_refine_with_intent(trace, regions())
    assert result.regions and result.batches


def test_standalone_pipeline_uses_canonical_selection(monkeypatch):
    from tests.refinement.test_integration import _disk_image
    from vectormark.pipeline import idealize, Options
    def forbidden(*args, **kwargs):
        raise AssertionError('standalone pipeline used the numerical selection loop')
    monkeypatch.setattr('vectormark.refinement.framework.select_refinements', forbidden)
    svg = idealize(_disk_image(), options=Options(refinement=True))
    assert '<circle' in svg


def test_cached_corpus_refinement_reuses_trace_and_canonical_selection(monkeypatch, tmp_path):
    import pickle
    import numpy as np
    from scripts import generate_corpus_html as report
    from tests.test_canonical_seam_discovery import regions
    from vectormark.pipeline import Options
    from dataclasses import replace
    objects = [replace(region, drawing_id=None) for region in regions()]
    options = Options(refinement=True)
    cache_path = tmp_path / 'trace.pkl'
    cache_path.write_bytes(pickle.dumps({'version': report.TRACE_CACHE_VERSION,
        'trace_options': report._trace_options(options), 'objects': objects,
        'masks': {region.id: region.raster for region in objects}}))
    monkeypatch.setattr(report, '_trace_cache_path', lambda *args: cache_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('cached corpus run retraced or used numerical selection')
    monkeypatch.setattr('vectormark.refinement.trace.trace_regions', forbidden)
    monkeypatch.setattr('vectormark.refinement.framework.select_refinements', forbidden)
    svg, result = report._idealize_refinement_with_trace_cache(np.full((440, 300, 3), 255, dtype=np.uint8),
        options=options, entry=SimpleNamespace(name='cached'), cache_dir=tmp_path)
    assert '<svg' in svg and result.elements >= 1

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from tests.test_drawing_corner_store_replay import _source_store
from vectormark.candidate import FlatFill
from vectormark.drawing import Canvas
from vectormark.drawing_recipe import ApplyCornerRecipeOperation
from vectormark.drawing_engine import PathFitOptions
from vectormark.corner_recipe import CornerRecipeExecutor
from vectormark.drawing_relationships import replace_region
from vectormark.fit import Shape
from vectormark.optimizer.automatic_replay import ReplayBatch, PassInputRegionRef, PathEndpointRef
from vectormark.optimizer.replay_binding import capture_replay_state
from vectormark.optimizer.vector_region import VectorRegion
from vectormark.path_geometry import geometry_from_svg_d
from vectormark.replay_state import ReplayState
from vectormark.replay_identity import canonical_input_revision


def nested_state():
    def branch(identity):
        leaf = VectorRegion.from_shape(id=1, drawing_id=f'r{identity}-1',
            shape=Shape('path', {'d': 'M0 0 L12 0 L12 12 L0 12 Z'}),
            fill=FlatFill('#112233'), z=0, raster=np.ones((16, 16), dtype=bool))
        return VectorRegion.branch(id=identity, drawing_id=f'r{identity}', children=(leaf,))
    return capture_replay_state((branch(10), branch(20)), Canvas(16, 16))


def test_downstream_templates_and_receipts_use_canonical_snapshots(monkeypatch):
    from vectormark.drawing_replay import _batch_inputs_from_start, _rebase_downstream_batches
    state = nested_state()
    batch = ReplayBatch('template', canonical_input_revision(state, (10,)), (10,), ())
    def forbidden(*args, **kwargs):
        raise AssertionError('seam rebinding projected a working scene')
    monkeypatch.setattr('vectormark.drawing_replay.working_regions_from_state', forbidden)
    templates = _batch_inputs_from_start((batch,), state)
    assert templates[0][1] is state
    rebased, = _rebase_downstream_batches(templates, state,
        target_region_id='r10-1', output_revision='edited')
    assert rebased.scope == (10,)
    assert rebased.input_revision == canonical_input_revision(state, (10,))
    assert rebased.batch_id != batch.batch_id


def test_endpoint_rebinding_resolves_full_ancestry_not_repeated_leaf_id():
    from vectormark.drawing_replay import _rebind_initial_endpoint_refs
    from vectormark.drawing_seams import _command_id
    before = nested_state()
    old_d = before.drawing.region('r10-1').geometry.to_svg_d()
    geometry = geometry_from_svg_d('M0 0 L15 0 L12 12 L0 12 Z', region_id='r10-1')
    changed = replace_region(before.drawing, replace(before.drawing.region('r10-1'), geometry=geometry))
    after = replace(before, drawing=changed)
    ref = PathEndpointRef(PassInputRegionRef('batch', 'revision', (1,)),
        sha256(old_d.encode()).hexdigest(), _command_id(0, 1, 'L', [12., 0.]))
    rebound = _rebind_initial_endpoint_refs(ref, batch_scope=(10,),
        template_input=before, replay_input=after)
    assert rebound.path_id == sha256(geometry.to_svg_d().encode()).hexdigest()
    assert rebound.command_id == _command_id(0, 1, 'L', [15., 0.])
    assert after.drawing.region('r20-1') == before.drawing.region('r20-1')


def test_downstream_rebase_still_rejects_changed_target_geometry():
    from vectormark.drawing_replay import _rebase_downstream_batches
    before = nested_state()
    geometry = geometry_from_svg_d('M0 0 L15 0 L12 12 L0 12 Z', region_id='r10-1')
    after = replace(before, drawing=replace_region(before.drawing,
        replace(before.drawing.region('r10-1'), geometry=geometry)))
    batch = ReplayBatch('template', canonical_input_revision(before), (), ())
    with pytest.raises(ValueError, match='no exact endpoint rebinding'):
        _rebase_downstream_batches(((batch, before),), after,
            target_region_id='r10-1', output_revision='edited')


def test_corner_downstream_annotation_restore_needs_no_working_projection(monkeypatch):
    from vectormark.drawing_replay import _execute_corner_recipe
    _, base, _, evidence = _source_store()
    state = ReplayState(evidence.drawing, evidence.shape_metadata, evidence.region_evidence)
    source = evidence.root_paths[0]
    recipe = CornerRecipeExecutor().create(source.fit, PathFitOptions(1., 1.5, 'quadratic'))
    application = ApplyCornerRecipeOperation(recipe=recipe, evidence_revision=evidence.revision,
        region_ancestry=(1,), target_region_id='r1', target_path_id=source.path_id,
        input_revision=canonical_input_revision(state), settings=base.effective_recipe.settings)
    expected, _ = _execute_corner_recipe(application, evidence, state)
    application = replace(application, downstream_batches=(ReplayBatch('empty',
        canonical_input_revision(expected), (), ()),))
    def forbidden(*args, **kwargs):
        raise AssertionError('annotation restoration projected a working scene')
    monkeypatch.setattr('vectormark.drawing_replay.working_regions_from_state', forbidden)
    actual, _ = _execute_corner_recipe(application, evidence, state)
    assert actual.drawing == expected.drawing
    assert actual.region_evidence == expected.region_evidence

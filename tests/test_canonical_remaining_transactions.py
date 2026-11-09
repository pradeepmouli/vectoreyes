from dataclasses import replace

import numpy as np
import pytest

from tests.test_canonical_multi_replay import forbid_working_execution
from tests.test_canonical_structural_replay import committed, leaf, numeric_expected
from vectoreyes.candidate import FlatFill
from vectoreyes.drawing import Canvas
from vectoreyes.drawing_automatic import replay_automatic_batches
from vectoreyes.fit import Shape
from vectoreyes.refinement.automatic_replay import (
    ApplySeamConstraintsOperation, AxisSeamConstraint, DependentCloneRefresh,
    PassInputRegionRef, ReconstructSelfSymmetryOperation, RefreshDependentClonesOperation,
    ReplayBatch,
)
from vectoreyes.refinement.framework import select_refinements_with_intent
from vectoreyes.refinement.passes.compound import split_compound_pass
from vectoreyes.refinement.replay_binding import capture_replay_state, working_regions_from_state
from vectoreyes.refinement.region_sample import RegionSample
from vectoreyes.replay_identity import canonical_input_revision


def check_canonical(monkeypatch, state, batch):
    expected = numeric_expected(state, batch)
    following = ReplayBatch('following', canonical_input_revision(expected), (), ())
    forbid_working_execution(monkeypatch)
    actual = replay_automatic_batches(state, (batch, following))
    assert actual.drawing == expected.drawing
    assert actual.region_evidence == expected.region_evidence
    assert canonical_input_revision(actual) == canonical_input_revision(expected)
    return actual


def test_seam_transaction_keeps_exact_selected_boundary_and_observations(monkeypatch):
    state = capture_replay_state((leaf(1, Shape('rect', {'x': 0, 'y': 0, 'w': 10, 'h': 10}), '#123456'),
        leaf(2, Shape('rect', {'x': 10.4, 'y': 0, 'w': 10, 'h': 10}), '#abcdef')), Canvas(300, 440))
    left, right = (PassInputRegionRef('', '', (identity,)) for identity in (1, 2))
    action = ApplySeamConstraintsOperation((left, right), (
        AxisSeamConstraint(left, right, 'x', 'right', 'left', 10, 10.4,
            10.2, 0, 10, None, None, 'midpoint'),), (), (), 1., 1., False, ())
    result = check_canonical(monkeypatch, state, committed(state, action, (1, 2), (1, 2)))
    assert result.drawing.region('r1').geometry.primitive.width == 10.2


@pytest.mark.parametrize('dependent', ['none', 'leaf', 'nested'])
def test_self_symmetry_transaction_uses_recorded_fit_and_preserves_receipt(monkeypatch, dependent):
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    points = [(48 + 20 * np.cos(theta), 48 + 30 * np.sin(theta)) for theta in angles]
    d = 'M' + ' L'.join(f'{x} {y}' for x, y in points) + ' Z'
    source = leaf(1, Shape('path', {'d': d}), '#123456')
    roots = [source]
    if dependent != 'none':
        clone = RegionSample(7 if dependent == 'nested' else 2,
            Shape('use', {'href_obj_id': 1, 'href_region_path': ('..', 1) if dependent == 'nested' else (1,),
                'transform': (1, 0, 0, 1, 100, 0)}), FlatFill('#abcdef'),
            footprint=source.footprint.affine_transform((1, 0, 0, 1, 100, 0)), raster=source.raster,
            drawing_id='r2-7' if dependent == 'nested' else 'r2')
        roots.append(RegionSample.branch(id=2, children=(clone,), drawing_id='r2')
            if dependent == 'nested' else clone)
    state = capture_replay_state(roots, Canvas(300, 440))
    action = ReconstructSelfSymmetryOperation(PassInputRegionRef('', '', (1,)),
        () if dependent == 'none' else (PassInputRegionRef('', '', (2,)),),
        (np.pi / 2, 48., 48.), 0., 'positive')
    ids = (1,) if dependent == 'none' else (1, 2)
    result = check_canonical(monkeypatch, state, committed(state, action, ids, ids))
    assert result.drawing.region('r1').fill == state.drawing.region('r1').fill


@pytest.mark.parametrize('clone', ['none', 'single', 'multiple'])
def test_topology_transaction_publishes_named_components_and_checks_correspondence(monkeypatch, clone):
    from vectoreyes.path_geometry import AffineTransform, CloneGeometry, CompoundPathGeometry, parse_svg_paths
    shape = Shape('path', {'d': 'M0 0 L8 0 L8 8 L0 8 Z M12 0 L20 0 L20 8 L12 8 Z',
        'fill_rule': 'evenodd', 'path_ids': ('r1.p0', 'r1.p1')})
    shape = Shape('path', {**shape.params, 'canonical_path_geometry': CompoundPathGeometry(
        parse_svg_paths(shape.params['d'], region_id='r1', path_ids=shape.params['path_ids']),
        'evenodd', 'pending component publication')})
    source = leaf(1, shape, '#123456')
    roots = [source]
    if clone != 'none':
        footprint = source.footprint.affine_transform((1, 0, 0, 1, 30, 0))
        instance = Shape('use', {'href_obj_id': 1, 'transform': (1, 0, 0, 1, 30, 0)})
        if clone == 'multiple':
            footprint = footprint.union(source.footprint.affine_transform((1, 0, 0, 1, 60, 0)))
            instance = Shape('path', {'d': footprint.to_svg_d(), 'canonical_clone_geometry':
                CloneGeometry('r1', (AffineTransform(1, 0, 0, 1, 30, 0), AffineTransform(1, 0, 0, 1, 60, 0)))})
        roots.append(RegionSample(2, instance, FlatFill('#abcdef'), drawing_id='r2',
            footprint=footprint, raster=source.raster))
    state = capture_replay_state(roots, Canvas(300, 440))
    samples = list(working_regions_from_state(state))
    recorded = select_refinements_with_intent(samples, {obj.id: obj.raster for obj in samples}, [split_compound_pass])
    batch = recorded.batches[0]
    expected = numeric_expected(state, batch)
    forbid_working_execution(monkeypatch)
    actual = replay_automatic_batches(state, (batch,))
    assert actual.drawing == expected.drawing
    assert actual.region_evidence == expected.region_evidence
    assert canonical_input_revision(actual) == canonical_input_revision(expected)
    assert len(actual.drawing.region('r1').children) == 2
    operation = batch.operations[0]
    invalid = replace(operation, intent=replace(operation.intent,
        action=replace(operation.intent.action, path_correspondence=())))
    with pytest.raises(ValueError, match='correspondence is stale'):
        replay_automatic_batches(state, (replace(batch, entries=(invalid,)),))


def test_refresh_transaction_rebinds_recorded_source_without_copying_its_paint(monkeypatch):
    old = leaf(1, Shape('rect', {'x': 0, 'y': 0, 'w': 4, 'h': 6}), '#123456')
    new = leaf(3, Shape('rect', {'x': 20, 'y': 0, 'w': 4, 'h': 6}), '#445566')
    clone = RegionSample(2, Shape('use', {'href_obj_id': 1, 'transform': (1, 0, 0, 1, 5, 0)}),
        FlatFill('#abcdef'), drawing_id='r2', footprint=old.footprint.affine_transform((1, 0, 0, 1, 5, 0)),
        raster=old.raster, diagnostics={'clones': {'matched_source': 1}})
    state = capture_replay_state((old, clone, new), Canvas(300, 440))
    refs = {identity: PassInputRegionRef('', '', (identity,)) for identity in (1, 2, 3)}
    action = RefreshDependentClonesOperation((refs[2],), (
        DependentCloneRefresh(refs[2], refs[3], 1, (1,)),))
    result = check_canonical(monkeypatch, state, committed(state, action, (2,), (2,)))
    assert result.drawing.region('r2').geometry.source_region_id == 'r3'
    assert result.drawing.region('r2').fill == state.drawing.region('r2').fill


def test_topology_retires_corner_evidence_with_its_redundant_source_path(monkeypatch):
    from vectoreyes.corner_diagnostics import path_corner_diagnostics
    from vectoreyes.path_geometry import CompoundPathGeometry, parse_svg_paths
    d = 'M4 4 L8 4 L8 8 L4 8 Z M0 0 L12 0 L12 12 L0 12 Z'
    shape = Shape('path', {'d': d, 'fill_rule': 'nonzero',
        'path_ids': ('r1.p0', 'r1.p1'), 'canonical_path_geometry': CompoundPathGeometry(
            parse_svg_paths(d, region_id='r1'), 'nonzero', 'pending component publication')})
    source = leaf(1, shape, '#123456').with_diagnostics({
        'corners': path_corner_diagnostics(d, declared_corners=((4, 4),))})
    state = capture_replay_state((source,), Canvas(300, 440))
    samples = list(working_regions_from_state(state))
    recorded = select_refinements_with_intent(samples, {obj.id: obj.raster for obj in samples}, [split_compound_pass])
    result = check_canonical(monkeypatch, state, recorded.batches[0])
    assert result.drawing.region('r1').diagnostics['topology']['retired_feature_ids']


@pytest.mark.parametrize('fill_rule', ['evenodd', 'nonzero'])
def test_hole_first_topology_keeps_source_ordinals_and_feature_retirement(monkeypatch, fill_rule):
    from vectoreyes.corner_diagnostics import path_corner_diagnostics
    from vectoreyes.path_geometry import CompoundPathGeometry, parse_svg_paths
    d = 'M4 4 L8 4 L8 8 L4 8 Z M0 0 L12 0 L12 12 L0 12 Z M20 0 L24 0 L24 4 L20 4 Z'
    shape = Shape('path', {'d': d, 'fill_rule': fill_rule,
        'path_ids': ('r1.p0', 'r1.p1', 'r1.p2'), 'canonical_path_geometry': CompoundPathGeometry(
            parse_svg_paths(d, region_id='r1'), fill_rule, 'pending component publication')})
    source = leaf(1, shape, '#123456').with_diagnostics({'corners':
        path_corner_diagnostics(d, declared_corners=((4, 4), (0, 0), (20, 0)))})
    state = capture_replay_state((source,), Canvas(300, 440))
    samples = list(working_regions_from_state(state))
    recorded = select_refinements_with_intent(samples, {obj.id: obj.raster for obj in samples}, [split_compound_pass])
    check_canonical(monkeypatch, state, recorded.batches[0])

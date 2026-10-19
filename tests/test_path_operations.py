from dataclasses import replace

import numpy as np
import pytest

from vectormark.drawing import Canvas, Drawing, DrawingRegion
from vectormark.drawing_actions import ActionSettings, UpdateGeometryAction
from vectormark.drawing_fill import FlatFill
from vectormark.path_geometry import CornerAnchor, PathGeometry, geometry_from_svg_d


def fixture():
    geometry = geometry_from_svg_d('M0 0 L4 0 L8 1 L12 0 L12 10 L0 10 Z', region_id='r1')
    geometry = replace(geometry, anchors=(CornerAnchor('corner', 'r1.p0.c4', 'r1.p0.c4'),))
    region = DrawingRegion('r1', geometry, FlatFill('#123456'), z=4, source_region_ids=('source',))
    reference = DrawingRegion('r2', geometry_from_svg_d('M20 20 L30 20 L30 30 L20 30 Z', region_id='r2'))
    return Drawing('drw_test', Canvas(48, 48), (region, reference))


@pytest.mark.parametrize('ops,expected', [
    ([{'op': 'snap', 'target': 'r1.p0.c2', 'reference': 'r2.p0.c1'},
      {'op': 'align', 'target': 'r1.p0.c2', 'reference': 'r2.p0.c3', 'axes': ['x']}], (20, 20)),
    ([{'op': 'align', 'target': 'r1.p0.c2', 'reference': 'r2.p0.c1', 'axes': ['x']},
      {'op': 'align', 'target': 'r1.p0.c2', 'reference': 'r2.p0.c3', 'axes': ['y']}], (30, 30)),
    ([{'op': 'snap', 'target': 'r1.p0.c1', 'reference': 'r2.p0.c1'},
      {'op': 'snap', 'target': 'r1.p0.c2', 'reference': 'r1.p0.c1'}], (30, 20)),
])
def test_direct_actions_execute_in_order_and_read_current_references(ops, expected):
    from vectormark.path_operations import apply_region_path_operations

    drawing = fixture()
    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1', 'ops': ops})
    result, _ = apply_region_path_operations(drawing, action,
        settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    assert result.geometry.path.commands[2].end == expected
    assert drawing.region('r1').geometry.path.commands[2].end == (8, 1)


def test_ordered_action_rebinds_original_command_after_removal():
    from vectormark.path_operations import apply_region_path_operations

    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1', 'ops': [
        {'op': 'remove', 'target': 'r1.p0.c1'},
        {'op': 'snap', 'target': 'r1.p0.c3', 'reference': 'r2.p0.c1'},
    ]})
    result, receipt = apply_region_path_operations(fixture(), action,
        settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    assert result.geometry.path.commands[2].end == (30, 20)
    assert receipt['path_command_lineage']['commands']['r1.p0.c3'] == 'r1.p0.c2'
    assert 'r1.p0.c1' not in receipt['path_command_lineage']['commands']


def test_ordered_action_cannot_retarget_a_retired_reference_by_ordinal():
    from vectormark.path_operations import apply_region_path_operations

    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1', 'ops': [
        {'op': 'remove', 'target': 'r1.p0.c1'},
        {'op': 'snap', 'target': 'r1.p0.c3', 'reference': 'r1.p0.c1'},
    ]})
    with pytest.raises(ValueError, match='retired by an earlier operation'):
        apply_region_path_operations(fixture(), action,
            settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))


@pytest.mark.parametrize('op', [
    {'op': 'remove', 'target': 'r1.p0.c2'},
    {'op': 'snap', 'target': 'r1.p0.c2', 'reference': 'r2.p0.c1'},
    {'op': 'align', 'target': 'r1.p0.c2', 'reference': 'r2.p0.c1', 'axes': ['x']},
    {'op': 'match_length', 'target': 'r1.p0.c1', 'reference': 'r2.p0.c1'},
    {'op': 'set_parallel', 'target': 'r1.p0.c1', 'reference': 'r2.p0.c1', 'distance': 0},
    {'op': 'match', 'target': 'r1.p0.c1', 'reference': 'r2.p0.c1',
     'transform': {'a': 1, 'b': 0, 'c': 0, 'd': 1, 'tx': 0, 'ty': 0}},
])
def test_canonical_path_operations_preserve_region_and_receipts(op):
    from vectormark.path_operations import apply_region_path_operations
    drawing = fixture()
    original = drawing.region('r1')
    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1', 'ops': [op]})
    result, receipt = apply_region_path_operations(drawing, action,
        settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    assert isinstance(result.geometry, PathGeometry)
    assert result.id == original.id and result.z == original.z
    assert result.fill is original.fill and result.source_region_ids == original.source_region_ids
    lineage = receipt['path_command_lineage']['commands']
    if op['op'] == 'remove':
        assert 'r1.p0.c2' not in lineage
        assert len(result.geometry.path.commands) == len(original.geometry.path.commands) - 1
    elif op['op'] == 'snap':
        assert result.geometry.path.commands[2].end == (30, 20)
    elif op['op'] == 'align':
        assert result.geometry.path.commands[2].end == (30, 1)
    elif op['op'] == 'match_length':
        assert result.geometry.path.commands[1].end == (10, 0)
    elif op['op'] == 'match':
        assert result.geometry.path.commands[1].end == (30, 20)
    else:
        assert result.geometry.path.commands[0].end[1] == 20
        assert result.geometry.path.commands[1].end[1] == 20
    assert original.geometry.path.commands[2].end == (8, 1)


def test_canonical_between_is_end_to_end_and_rebinds_untouched_corner():
    from vectormark.path_operations import apply_region_path_operations
    drawing = fixture()
    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1',
        'ops': [{'op': 'fit', 'between': ['r1.p0.c0', 'r1.p0.c3'], 'type': 'cubic'}]})
    result, receipt = apply_region_path_operations(drawing, action,
        settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    assert result.geometry.path.start == (0, 0)
    assert result.geometry.path.commands[1].end == (12, 0)
    assert result.geometry.path.commands[2].end == (12, 10)
    assert result.geometry.anchors[0].entry_command_id == 'r1.p0.c2'
    assert receipt['path_command_lineage']['commands']['r1.p0.c3'] == 'r1.p0.c1'


def test_canonical_fit_reports_missing_exact_source_span():
    from vectormark.path_operations import apply_region_path_operations
    drawing = fixture()
    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1',
        'ops': [{'op': 'fit', 'target': 'r1.p0.c1', 'type': 'line'}]})
    with pytest.raises(ValueError, match='no exact original material-polyline span'):
        apply_region_path_operations(drawing, action, fit_sources={},
            settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))


def test_canonical_fit_retains_directional_acceptance_diagnostics():
    from vectormark.path_operations import apply_region_path_operations
    drawing = fixture()
    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1',
        'ops': [{'op': 'fit', 'target': 'r1.p0.c1', 'type': 'line'}]})
    result, receipt = apply_region_path_operations(drawing, action,
        fit_sources={'r1.p0.c1': np.array([[0, 0], [4, 0]], dtype=float)},
        settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    metric = result.diagnostics['fit_acceptance']['r1.p0.c1']
    assert metric['acceptance_direction'] == 'curve_to_original'
    assert metric['reverse_is_diagnostic_only'] is True
    assert metric['max_error_px'] == 2
    assert receipt['fit_acceptance'] == result.diagnostics['fit_acceptance']


def test_adjacent_between_equals_single_target_fit():
    from vectormark.path_operations import apply_region_path_operations
    drawing = fixture()
    results = []
    for selection in ({'target': 'r1.p0.c1'}, {'between': ['r1.p0.c0', 'r1.p0.c1']}):
        action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1',
            'ops': [{'op': 'fit', 'type': 'line', **selection}]})
        results.append(apply_region_path_operations(drawing, action,
            settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive')))
    assert results[0][0].geometry == results[1][0].geometry
    assert results[0][1]['path_command_lineage'] == results[1][1]['path_command_lineage']


def test_cutout_edit_preserves_original_path_addresses_and_source_text_identity():
    from vectormark.path_operations import apply_region_path_operations
    from vectormark.path_geometry import PathGeometryWithCutouts
    d = 'M2 2 L4 2 L4 4 L2 4 Z M0 0 L10 0 L10 10 L0 10 Z'
    geometry = geometry_from_svg_d(d, region_id='r1', fill_rule='evenodd')
    region = DrawingRegion('r1', geometry)
    drawing = Drawing('drw_test', Canvas(16, 16), (region,))
    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1',
        'ops': [{'op': 'remove', 'target': 'r1.p0.c2'}]})
    result, receipt = apply_region_path_operations(drawing, action, source_text=d,
        settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))
    assert isinstance(result.geometry, PathGeometryWithCutouts)
    assert result.geometry.outer == geometry.outer
    assert result.geometry.cutouts[0].path.id == 'r1.p0'
    assert receipt['path_ids'] == ('r1.p1', 'r1.p0')
    assert receipt['path_command_lineage']['input_d'] == d


def test_command_lineage_composes_across_direct_stages():
    from vectormark.path_operations import apply_region_path_operations
    drawing = fixture()
    settings = ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive')
    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1',
        'ops': [{'op': 'remove', 'target': 'r1.p0.c2'}]})
    first, receipt = apply_region_path_operations(drawing, action, settings=settings)
    updated = replace(drawing, regions=(first, drawing.region('r2')))
    second, final = apply_region_path_operations(updated, action, settings=settings,
        previous_lineage=receipt['path_command_lineage'])
    lineage = final['path_command_lineage']
    assert lineage['input_d'] == drawing.region('r1').geometry.to_svg_d()
    assert 'r1.p0.c2' not in lineage['commands'] and 'r1.p0.c3' not in lineage['commands']
    assert lineage['commands']['r1.p0.c4'] == 'r1.p0.c2'


def test_numeric_command_rows_do_not_round_canonical_controls():
    from vectormark.path_operations import _command_stream
    geometry = geometry_from_svg_d('M0 0 Q1.123456789 2.987654321 4 0 L4 4 L0 4 Z', region_id='r1')
    assert _command_stream(geometry)[1].values[:2] == (1.123456789, 2.987654321)


@pytest.mark.parametrize('ops', [
    [{'op': 'reverse', 'target': 'r1.p0.c1'}],
    [{'op': 'mirror', 'target': 'r1.p0.c1', 'reference': 'r2.p0.c1',
      'axis': {'point': [0, 0], 'angle_degrees': 90}}],
    [{'op': 'fit', 'target': 'r1.p0.c1', 'type': 'keep'}, {'op': 'break', 'target': 'r1.p0.c1'}],
])
def test_unimplemented_path_operations_never_report_silent_success(ops):
    from vectormark.path_operations import apply_region_path_operations
    drawing = fixture()
    action = UpdateGeometryAction.model_validate({'op': 'update_geometry', 'target': 'r1', 'ops': ops})
    with pytest.raises(ValueError, match='not implemented'):
        apply_region_path_operations(drawing, action,
            settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))

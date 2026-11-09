from vectoreyes.fit import Shape
from vectoreyes.refinement.automatic_replay import (
    ApplySeamConstraintsOperation, AxisSeamConstraint, PassInputRegionRef,
)
from vectoreyes.skia_geometry import SkPath


def test_seam_execution_consumes_explicit_numeric_inputs_without_region_resolver():
    from vectoreyes.drawing_seams import SeamInput, execute_seam_constraints

    left, right = PassInputRegionRef('', '', (1,)), PassInputRegionRef('', '', (2,))
    action = ApplySeamConstraintsOperation((left, right), (
        AxisSeamConstraint(left, right, 'x', 'right', 'left', 10, 10.4,
            10.2, 0, 10, None, None, 'midpoint'),), (), (), 1, 1, False, ())
    boundary = SkPath.from_svg_d('M0 0 L21 0 L21 10 L0 10 Z')
    inputs = {
        (1,): SeamInput(Shape('rect', {'x': 0, 'y': 0, 'w': 10, 'h': 10}),
            1, 1, (0, 0, 1, 1), boundary),
        (2,): SeamInput(Shape('rect', {'x': 10.4, 'y': 0, 'w': 10, 'h': 10}),
            2, 2, (0, 0, 2, 2), boundary),
    }

    updates = execute_seam_constraints(action, inputs)

    assert updates[(1,)].shape.params['w'] == 10.2
    assert updates[(2,)].shape.params['x'] == 10.2
    assert inputs[(1,)].shape.params['w'] == 10


def test_seam_execution_preserves_inflecting_cubic_away_from_shared_edge():
    from vectoreyes.drawing_seams import SeamInput, execute_seam_constraints
    from vectoreyes.fit import _parse_path_commands

    left, right = PassInputRegionRef('', '', (1,)), PassInputRegionRef('', '', (2,))
    action = ApplySeamConstraintsOperation((left, right), (
        AxisSeamConstraint(left, right, 'x', 'right', 'left', 10, 10.4,
            10.2, 0, 20, None, None, 'midpoint'),), (), (), 1, 1, False, ())
    shape = Shape('path', {'d': 'M0 0 L10 0 L10 20 L0 20 C-5 15 5 5 0 0 Z'})
    inputs = {
        (1,): SeamInput(shape, 1, 1, (0, 0, 1, 1), SkPath()),
        (2,): SeamInput(Shape('rect', {'x': 10.4, 'y': 0, 'w': 10, 'h': 20}),
            2, 2, (0, 0, 2, 2), SkPath()),
    }

    updates = execute_seam_constraints(action, inputs)

    commands = _parse_path_commands(updates[(1,)].shape.params['d'])
    assert ('C', [-5.0, 15.0, 5.0, 5.0, 0.0, 0.0]) in commands
    assert ('L', [10.2, 0.0]) in commands
    assert ('L', [10.2, 20.0]) in commands
    assert updates[(2,)].shape.params['x'] == 10.2
    assert inputs[(1,)].shape == shape

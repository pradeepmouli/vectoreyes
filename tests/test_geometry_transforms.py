from dataclasses import replace

import pytest

from vectoreyes.path_geometry import (
    AffineTransform, ArcCommand, CornerAnchor, CubicCommand, LineCommand,
    MoveCommand, QuadraticCommand, geometry_from_svg_d,
)


@pytest.mark.parametrize('command,expected', [
    (MoveCommand('p.c0', (2, 3)), MoveCommand('p.c0', (7, 22))),
    (LineCommand('p.c1', (2, 3)), LineCommand('p.c1', (7, 22))),
    (QuadraticCommand('p.c1', (2, 3), (4, 5)),
     QuadraticCommand('p.c1', (7, 22), (5, 24))),
    (CubicCommand('p.c1', (2, 3), (4, 5), (6, 7)),
     CubicCommand('p.c1', (7, 22), (5, 24), (3, 26))),
])
def test_transform_command_rotates_every_control_without_renaming(command, expected):
    from vectoreyes.geometry_transforms import transform_command
    assert transform_command(command, AffineTransform(0, 1, -1, 0, 10, 20)) == expected


def test_reflection_moves_arc_axis_and_reverses_sweep():
    from vectoreyes.geometry_transforms import transform_command
    original = ArcCommand('p.c8', (5, 7), 8, 3, 30, True, False)
    result = transform_command(original, AffineTransform(-1, 0, 0, 1, 20, 0))
    assert result.end == (15, 7)
    assert result.rotation_degrees == pytest.approx(150)
    assert result.sweep is True and result.large_arc is True
    assert result.rx == 8 and result.ry == 3 and result.id == original.id
    assert original.end == (5, 7) and original.sweep is False


def test_path_transform_retains_sparse_ids_and_corner_annotations_without_rounding():
    from vectoreyes.geometry_transforms import transform_path_geometry
    path = geometry_from_svg_d('M0 0 Q2 0 4 0 L4 4 L0 4 Z', region_id='r1')
    commands = tuple(replace(command, id=f'r1.p0.c{index * 7}')
        for index, command in enumerate(path.path.commands))
    path = replace(path, path=replace(path.path, commands=commands),
        anchors=(CornerAnchor('corner', commands[1].id, commands[1].id),))
    result = transform_path_geometry(path, AffineTransform(tx=0.123456789, ty=2))
    assert result.path.commands[0].end == (0.123456789, 2)
    assert result.path.commands[1].control == (2.123456789, 2)
    assert [item.id for item in result.path.commands] == [item.id for item in commands]
    assert result.anchors is path.anchors and result.path.id == path.path.id
    assert path.path.commands[0].end == (0, 0)


def test_svg_projection_uses_the_same_numeric_command_transform(monkeypatch):
    from vectoreyes import geometry_transforms
    from vectoreyes.emit import transform_path_d
    seen = []
    execute = geometry_transforms.transform_command
    def record(command, transform):
        seen.append(type(command))
        return execute(command, transform)
    monkeypatch.setattr(geometry_transforms, 'transform_command', record)
    result = transform_path_d('M2 3 Q4 5 2 3 C4 5 6 7 2 3 Z', (0, 1, -1, 0, 10, 20))
    assert result == 'M 7 22 Q 5 24 7 22 C 5 24 3 26 7 22 Z'
    assert seen == [MoveCommand, QuadraticCommand, CubicCommand]

"""Typed numeric transforms shared by retained operations and SVG projection."""

from dataclasses import replace
from math import atan2, cos, degrees, radians, sin

from .path_geometry import (
    AffineTransform, ArcCommand, CubicCommand, LineCommand, MoveCommand,
    PathCommand, PathGeometry, Point, QuadraticCommand,
)


def transform_point(point: Point, transform: AffineTransform) -> Point:
    x, y = point
    return (transform.a * x + transform.c * y + transform.tx,
            transform.b * x + transform.d * y + transform.ty)


def transform_command(command: PathCommand, transform: AffineTransform) -> PathCommand:
    """Preserve command identity while applying the retained transform.

    Arc axes and sweep follow the existing rigid rotation/reflection policy;
    this does not add ellipse scaling or shear support to the baked-arc path.
    """
    end = transform_point(command.end, transform)
    if isinstance(command, (MoveCommand, LineCommand)):
        return replace(command, end=end)
    if isinstance(command, QuadraticCommand):
        return replace(command, end=end, control=transform_point(command.control, transform))
    if isinstance(command, CubicCommand):
        return replace(command, end=end,
            control_1=transform_point(command.control_1, transform),
            control_2=transform_point(command.control_2, transform))
    if isinstance(command, ArcCommand):
        theta = radians(command.rotation_degrees)
        axis_x = transform.a * cos(theta) + transform.c * sin(theta)
        axis_y = transform.b * cos(theta) + transform.d * sin(theta)
        reflected = transform.a * transform.d - transform.b * transform.c < 0.0
        return replace(command, end=end,
            rotation_degrees=(degrees(atan2(axis_y, axis_x)) + 360.0) % 360.0,
            sweep=not command.sweep if reflected else command.sweep)
    raise TypeError(f'unsupported path command: {type(command).__name__}')


def transform_path_geometry(path: PathGeometry, transform: AffineTransform) -> PathGeometry:
    """Transform typed values without serializing, refitting or changing anchors."""
    return replace(path, path=replace(path.path, commands=tuple(
        transform_command(command, transform) for command in path.path.commands)))

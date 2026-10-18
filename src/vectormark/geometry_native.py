"""Disposable native geometry predicates shared by engine operations and export."""

from collections.abc import Mapping
from math import cos, pi, sin

import skia

from .drawing import DrawingRegion
from .path_geometry import (
    AffineTransform, CirclePrimitive, CloneGeometry, CompoundPathGeometry,
    EllipsePrimitive, PathGeometry, PathGeometryWithCutouts, PathGeometryWithSymmetry,
    PolygonPrimitive, PrimitiveGeometry, RectanglePrimitive, RoundedRectanglePrimitive,
)
from .skia_geometry import SkPath, unary_union


def reflection(point: tuple[float, float], angle_degrees: float) -> AffineTransform:
    theta = angle_degrees * pi / 180.0
    ux, uy = cos(theta), sin(theta)
    a, b, d = 2 * ux * ux - 1, 2 * ux * uy, 2 * uy * uy - 1
    px, py = point
    return AffineTransform(a, b, b, d, px - a * px - b * py, py - b * px - d * py)


def affine_tuple(transform: AffineTransform) -> tuple[float, float, float, float, float, float]:
    return (transform.a, transform.b, transform.c, transform.d, transform.tx, transform.ty)


def region_path(leaf: DrawingRegion, by_id: Mapping[str, DrawingRegion]) -> SkPath:
    """Resolve canonical geometry without building a second mutable scene."""
    geometry = leaf.geometry
    if leaf.is_branch:
        return unary_union([region_path(child, by_id) for child in leaf.children])
    if isinstance(geometry, CloneGeometry):
        source = by_id.get(geometry.source_region_id)
        if source is None or isinstance(source.geometry, CloneGeometry):
            raise ValueError('clone source must identify a concrete retained region')
        path = region_path(source, by_id)
        return unary_union([path.affine_transform(affine_tuple(transform)) for transform in geometry.transforms])
    if isinstance(geometry, PathGeometryWithSymmetry):
        half = SkPath.from_svg_d(geometry.half.to_svg_d())
        return half.union(half.affine_transform(affine_tuple(reflection(geometry.axis.point, geometry.axis.angle_degrees))))
    if isinstance(geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        return SkPath.from_svg_d(geometry.to_svg_d(), fill_rule=geometry.fill_rule if isinstance(geometry, CompoundPathGeometry)
            else 'evenodd' if isinstance(geometry, PathGeometryWithCutouts) else None)
    if isinstance(geometry, PrimitiveGeometry):
        primitive = geometry.primitive
        path = skia.Path()
        if isinstance(primitive, CirclePrimitive):
            path.addCircle(*primitive.center, primitive.radius)
        elif isinstance(primitive, EllipsePrimitive):
            x, y = primitive.center
            path.addOval(skia.Rect.MakeLTRB(x - primitive.radius_x, y - primitive.radius_y, x + primitive.radius_x, y + primitive.radius_y))
        elif isinstance(primitive, (RectanglePrimitive, RoundedRectanglePrimitive)):
            rect = skia.Rect.MakeXYWH(*primitive.origin, primitive.width, primitive.height)
            if isinstance(primitive, RoundedRectanglePrimitive):
                path.addRoundRect(rect, primitive.radius_x, primitive.radius_y)
            else:
                path.addRect(rect)
        elif isinstance(primitive, PolygonPrimitive):
            return SkPath(shell=list(primitive.vertices))
        else:
            raise TypeError(f'unsupported primitive: {type(primitive).__name__}')
        return SkPath.from_skia(path)
    raise TypeError(f'unsupported canonical geometry: {type(geometry).__name__}')


def split_path_by_line(footprint: SkPath, first: tuple[float, float], second: tuple[float, float]) -> tuple[SkPath, SkPath]:
    """Clip against complementary half-planes; never discard whole components."""
    dx, dy = second[0] - first[0], second[1] - first[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length <= 1e-9:
        raise ValueError('line divider points must be distinct')
    direction = (dx / length, dy / length)
    normal = (-direction[1], direction[0])
    min_x, min_y, max_x, max_y = footprint.bounds
    extent = max(((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5 * 4, 1.0)
    start = (first[0] - direction[0] * extent, first[1] - direction[1] * extent)
    end = (first[0] + direction[0] * extent, first[1] + direction[1] * extent)
    positive = SkPath([start, end, (end[0] + normal[0] * extent * 2, end[1] + normal[1] * extent * 2),
        (start[0] + normal[0] * extent * 2, start[1] + normal[1] * extent * 2)])
    negative = SkPath([start, (start[0] - normal[0] * extent * 2, start[1] - normal[1] * extent * 2),
        (end[0] - normal[0] * extent * 2, end[1] - normal[1] * extent * 2), end])
    return footprint.intersection(positive), footprint.intersection(negative)

"""Canonical clone, alignment and explicit symmetry actions."""

from dataclasses import replace
from collections.abc import Iterator
from math import cos, radians, sin

from .drawing import Drawing, DrawingRegion
from .drawing_actions import AlignAction, AssignCloneAction, SetSymmetryAction
from .drawing_fill import compose_affine
from .geometry_transforms import transform_path_geometry
from .geometry_native import affine_tuple, reflection, region_path, split_path_by_line
from .path_geometry import (
    AffineTransform, CirclePrimitive, CloneGeometry, CompoundPathGeometry, Geometry,
    EllipsePrimitive, PathGeometry, PathGeometryWithCutouts, PathGeometryWithSymmetry,
    PolygonPrimitive, PrimitiveGeometry, RectanglePrimitive, RoundedRectanglePrimitive,
    parse_svg_paths,
)
from .drawing_topology import publish_region_paths


def _index(drawing: Drawing) -> dict[str, DrawingRegion]:
    def walk(region: DrawingRegion) -> Iterator[DrawingRegion]:
        yield region
        for child in region.children:
            yield from walk(child)
    return {region.id: region for root in drawing.regions for region in walk(root)}


def replace_region(drawing: Drawing, selected: DrawingRegion) -> Drawing:
    """Replace one canonical subtree and retarget references to retired members."""
    original = drawing.region(selected.id)
    retained = {region.id for region in selected.leaves()}
    retired = {region.id for region in original.leaves()} - retained
    def visit(region: DrawingRegion) -> DrawingRegion:
        if region.id == selected.id:
            return selected
        if isinstance(region.geometry, CloneGeometry) and region.geometry.source_region_id in retired:
            return replace(region, geometry=replace(region.geometry, source_region_id=selected.id))
        if region.is_branch:
            children = tuple(visit(child) for child in region.children)
            if any(first is not second for first, second in zip(children, region.children)):
                return replace(region, children=children)
        return region
    return replace(drawing, regions=tuple(visit(root) for root in drawing.regions))


def assign_clone(drawing: Drawing, action: AssignCloneAction) -> DrawingRegion:
    source, target = drawing.region(action.source), drawing.region(action.target)
    if not source.is_leaf or not target.is_leaf:
        raise ValueError('clone assignment requires retained leaf geometry')
    transforms = tuple(AffineTransform(**item.model_dump()) for item in action.transforms)
    visited = {target.id}
    while True:
        if source.id in visited:
            raise ValueError('cyclic retained clone reference: clone source chain contains a cycle')
        visited.add(source.id)
        if not isinstance(source.geometry, CloneGeometry):
            break
        transforms = tuple(compose_affine(outer, inner) for outer in transforms for inner in source.geometry.transforms)
        source = drawing.region(source.geometry.source_region_id)
    return replace(target, geometry=CloneGeometry(source.id, transforms), diagnostics={**target.diagnostics,
        'clone': {'source_id': source.id, 'transforms': tuple(affine_tuple(item) for item in transforms)}})


def _translated_geometry(geometry: Geometry, dx: float, dy: float) -> Geometry:
    def point(value: tuple[float, float]) -> tuple[float, float]:
        return (value[0] + dx, value[1] + dy)
    if isinstance(geometry, PathGeometry):
        return transform_path_geometry(geometry, AffineTransform(tx=dx, ty=dy))
    if isinstance(geometry, PathGeometryWithCutouts):
        return replace(geometry, outer=_translated_geometry(geometry.outer, dx, dy),
            cutouts=tuple(_translated_geometry(item, dx, dy) for item in geometry.cutouts))
    if isinstance(geometry, CompoundPathGeometry):
        return replace(geometry, paths=tuple(_translated_geometry(item, dx, dy) for item in geometry.paths))
    if isinstance(geometry, CloneGeometry):
        return replace(geometry, transforms=tuple(compose_affine(AffineTransform(tx=dx, ty=dy), item) for item in geometry.transforms))
    if isinstance(geometry, PathGeometryWithSymmetry):
        return replace(geometry, half=_translated_geometry(geometry.half, dx, dy),
            axis=replace(geometry.axis, point=point(geometry.axis.point)))
    if isinstance(geometry, PrimitiveGeometry):
        primitive = geometry.primitive
        if isinstance(primitive, (CirclePrimitive, EllipsePrimitive)):
            primitive = replace(primitive, center=point(primitive.center))
        elif isinstance(primitive, (RectanglePrimitive, RoundedRectanglePrimitive)):
            primitive = replace(primitive, origin=point(primitive.origin))
        elif isinstance(primitive, PolygonPrimitive):
            primitive = replace(primitive, vertices=tuple(point(item) for item in primitive.vertices))
        else:
            raise TypeError(f'unsupported primitive: {type(primitive).__name__}')
        return replace(geometry, primitive=primitive)
    raise TypeError(f'unsupported canonical geometry: {type(geometry).__name__}')


def align_region(drawing: Drawing, action: AlignAction) -> DrawingRegion:
    target, reference = drawing.region(action.target), drawing.region(action.reference)
    if not target.is_leaf:
        raise ValueError('alignment requires a retained leaf target')
    by_id = _index(drawing)
    first, second = region_path(target, by_id).bounds, region_path(reference, by_id).bounds
    dx = (second[0] + second[2] - first[0] - first[2]) / 2 if 'x' in action.axes else 0
    dy = (second[1] + second[3] - first[1] - first[3]) / 2 if 'y' in action.axes else 0
    return replace(target, geometry=_translated_geometry(target.geometry, dx, dy), diagnostics={**target.diagnostics,
        'align': {'reference': reference.id, 'axes': tuple(sorted(action.axes))}})


def set_region_symmetry(drawing: Drawing, action: SetSymmetryAction) -> DrawingRegion:
    target = drawing.region(action.target)
    by_id = _index(drawing)
    footprint = region_path(target, by_id)
    axis = action.axis
    theta = radians(axis.angle_degrees)
    first = axis.point
    halves = split_path_by_line(footprint, first, (first[0] + cos(theta), first[1] + sin(theta)))
    if any(half.is_empty for half in halves):
        raise ValueError('symmetry axis must divide the target into two non-empty halves')
    matrix = affine_tuple(reflection(axis.point, axis.angle_degrees))
    candidates = []
    for half in halves:
        joined = half.union(half.affine_transform(matrix))
        residual = footprint.symmetric_difference(joined).area / max(footprint.area, 1.0)
        candidates.append((residual, joined))
    residual, joined = min(candidates, key=lambda item: item[0])
    leaves = target.leaves()
    if any(leaf.fill != leaves[0].fill for leaf in leaves):
        raise ValueError('self-symmetry materialization requires one shared fill')
    consumed = tuple(leaf.id for leaf in leaves)
    diagnostics = {**target.diagnostics, 'symmetry': {'mode': 'explicit_self',
        'axis': axis.model_dump(), 'residual': float(residual), 'materialized': True},
        'materialization': {'operation': 'self_symmetry_union', 'consumed_region_ids': consumed,
            'retired_region_ids': tuple(item for item in consumed if item != target.id),
            'source_region_remap': tuple((item, target.id) for item in consumed), 'consumed_target_ids': consumed}}
    selected = replace(target, geometry=leaves[0].geometry, children=(), fill=leaves[0].fill,
        source_region_ids=tuple(dict.fromkeys((*target.source_region_ids,
            *(source for leaf in leaves for source in leaf.source_region_ids)))), diagnostics=diagnostics)
    return publish_region_paths(selected, parse_svg_paths(joined.to_svg_d(), region_id=target.id), fill_rule='evenodd')

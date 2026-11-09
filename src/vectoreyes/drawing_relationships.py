"""Canonical clone, alignment and explicit symmetry actions."""

from dataclasses import replace
from collections.abc import Iterator
from math import radians, isclose, hypot

from .drawing import Drawing, DrawingRegion
from .drawing_actions import AlignAction, AssignCloneAction, SetSymmetryAction
from .drawing_fill import compose_affine
from .geometry_transforms import transform_path_geometry
from .geometry_native import affine_tuple, region_path
from .path_geometry import (
    AffineTransform, CirclePrimitive, CloneGeometry, CompoundPathGeometry, Geometry,
    EllipsePrimitive, PathGeometry, PathGeometryWithCutouts, PathGeometryWithSymmetry,
    PolygonPrimitive, PrimitiveGeometry, RectanglePrimitive, RoundedRectanglePrimitive,
    geometry_from_svg_d,
    CornerAnchor, JunctionAnchor,
)


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
        if (isinstance(region.geometry, CloneGeometry)
                and region.geometry.source_region_id == selected.id
                and original.is_leaf and selected.is_branch):
            from .drawing_topology import expand_clone_components
            return expand_clone_components(region, selected)
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
    geometry: Geometry = CloneGeometry(source.id, transforms)
    if not action.symbolic:
        if len(transforms) > 1:
            children = tuple(replace(target, id=f'{target.id}-{index}',
                geometry=_bake_clone_geometry(drawing, source, replace(target, id=f'{target.id}-{index}'), (matrix,)))
                for index, matrix in enumerate(transforms, 1))
            return replace(target, geometry=None, fill=None, children=children, diagnostics={**target.diagnostics,
                'clone': {'source_id': source.id, 'transforms': tuple(affine_tuple(item) for item in transforms),
                          'symbolic': False}})
        geometry = _bake_clone_geometry(drawing, source, target, transforms)
    return replace(target, geometry=geometry, diagnostics={**target.diagnostics,
        'clone': {'source_id': source.id, 'transforms': tuple(affine_tuple(item) for item in transforms),
                  'symbolic': action.symbolic}})


def _bake_clone_geometry(
    drawing: Drawing, source: DrawingRegion, target: DrawingRegion,
    transforms: tuple[AffineTransform, ...],
) -> Geometry:
    """Materialize existing transforms, sharing the numeric relationship kernel."""
    from .drawing_adapter import _geometry, shape_from_region_geometry, transform_working_shape
    if len(transforms) == 1 and isinstance(source.geometry, PrimitiveGeometry) and isinstance(source.geometry.primitive, CirclePrimitive):
        matrix = transforms[0]
        sx, sy = hypot(matrix.a, matrix.b), hypot(matrix.c, matrix.d)
        if isclose(sx, sy, abs_tol=1e-9) and isclose(matrix.a*matrix.c + matrix.b*matrix.d, 0, abs_tol=1e-9):
            from .geometry_transforms import transform_point
            circle = source.geometry.primitive
            return PrimitiveGeometry(CirclePrimitive(transform_point(circle.center, matrix), circle.radius*sx))
    if len(transforms) == 1 and not isinstance(source.geometry, PathGeometryWithSymmetry):
        matrix = transforms[0]
        rigid = (isclose(matrix.a**2 + matrix.b**2, 1, abs_tol=1e-9)
            and isclose(matrix.c**2 + matrix.d**2, 1, abs_tol=1e-9)
            and isclose(matrix.a*matrix.c + matrix.b*matrix.d, 0, abs_tol=1e-9))
        if rigid:
            selected = transform_working_shape(shape_from_region_geometry(source, {}), affine_tuple(matrix))
            return readdress_baked_geometry(_geometry(selected.kind, selected.params, target.id, {}), target.id)
    # Symmetry geometry and non-rigid transforms use the native path engine;
    # no curve fitting or rasterization is involved.
    instance = replace(target, geometry=CloneGeometry(source.id, transforms))
    return geometry_from_svg_d(region_path(instance, _index(drawing)).to_svg_d(),
        region_id=target.id, defer_overlaps=True)


def readdress_baked_geometry(geometry: Geometry, region_id: str) -> Geometry:
    """Give copied commands and anchors their new owner without altering values."""
    from .path_topology import component_paths, replace_component_paths
    if not isinstance(geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        return geometry
    paths = component_paths(geometry)
    commands = {command.id: f'{region_id}.p{index}.c{position}'
        for index, path in enumerate(paths) for position, command in enumerate(path.path.commands)}
    anchors = {anchor.id: f'{region_id}.p{index}.{anchor.kind}{position}'
        for index, path in enumerate(paths) for position, anchor in enumerate(path.anchors)}
    rebound = []
    for index, path in enumerate(paths):
        features = []
        for anchor in path.anchors:
            if isinstance(anchor, CornerAnchor):
                feature = replace(anchor, id=anchors[anchor.id],
                    entry_command_id=commands[anchor.entry_command_id],
                    exit_command_id=commands[anchor.exit_command_id],
                    owned_command_ids=tuple(commands[item] for item in anchor.owned_command_ids))
            else:
                feature = replace(anchor, id=anchors[anchor.id],
                    command_ids=tuple(commands[item] for item in anchor.command_ids))
                if isinstance(feature, JunctionAnchor):
                    feature = replace(feature, reference_junction_id=anchors.get(
                        feature.reference_junction_id, feature.reference_junction_id))
            features.append(feature)
        rebound.append(replace(path, path=replace(path.path, id=f'{region_id}.p{index}',
            commands=tuple(replace(command, id=commands[command.id]) for command in path.path.commands)),
            anchors=tuple(features)))
    return replace_component_paths(geometry, tuple(rebound))


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


def set_region_symmetry(
    drawing: Drawing, action: SetSymmetryAction,
) -> DrawingRegion:
    """Bind plan addresses/settings to the same symmetry function used by passes."""
    from .drawing_symmetry import Axis2D, set_symmetry

    target = drawing.region(action.target)
    return set_symmetry(target, region_path(target, _index(drawing)),
        axis=Axis2D(radians(action.axis.angle_degrees), *action.axis.point),
        residual=action.residual, side=action.side)

"""Temporary adapter from the Python refinement's records to canonical drawings.

The native/public contract is :class:`Drawing`. ``RegionSample`` remains the
Python engine's working record while it still owns raster masks and Skia
footprints. This module is the only sanctioned projection across that seam.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from types import MappingProxyType
import numpy as np

if TYPE_CHECKING:
    from .refinement.automatic_replay import RefreshDependentClonesOperation, RegionResolver

from .candidate import (
    FlatFill as LegacyFlatFill,
    LinearGradientFill as LegacyLinearGradientFill,
    RadialGradientFill as LegacyRadialGradientFill,
    RasterFill as LegacyRasterFill,
)
from .drawing import Canvas, Drawing, DrawingRegion
from .fit import Shape
from .drawing_fill import compose_affine
from .path_geometry import (
    AffineTransform,
    CirclePrimitive,
    CloneGeometry,
    EllipsePrimitive,
    PolygonPrimitive,
    PrimitiveGeometry,
    PathGeometry,
    PathGeometryWithCutouts,
    CompoundPathGeometry,
    RectanglePrimitive,
    RoundedRectanglePrimitive,
    geometry_from_svg_d,
    parse_svg_paths,
)
from .corner_diagnostics import rebind_prefit_corner_diagnostics
from . import corner_provenance
from .paint_fitting import canonical_fill as _fill
from .refinement.region_sample import RegionSample
from .types import CoverageField
from .skia_geometry import SkPath
from .refinement.symmetry_materialize import index_region_paths, resolve_source_path


@dataclass(frozen=True)
class WorkingLeafTarget:
    """Derived address and source evidence for one working leaf."""

    id: str
    root_id: str
    region: RegionSample
    source_regions: frozenset[str]
    region_path: tuple[int, ...]


@dataclass(frozen=True)
class BoundRegionInputs:
    """One operation's canonical inputs and separately retained observations."""

    drawing: Drawing
    working: Mapping[str, RegionSample]
    paths: Mapping[str, tuple[int, ...]]
    shapes: Mapping[str, Shape]

    def publish(self, selected: DrawingRegion, *, symbolic: bool = True,
                diagnostics: Mapping[str, object] | None = None,
                footprint: SkPath | None = None,
                shape_metadata: Mapping[str, object] | None = None) -> RegionSample:
        from .geometry_native import affine_tuple
        from .refinement.symmetry_materialize import relative_source_path

        target = self.working[selected.id]
        geometry = selected.geometry
        if not isinstance(geometry, CloneGeometry):
            return publish_selected_leaf(selected, target)
        if len(geometry.transforms) != 1:
            raise ValueError('automatic relationship publication requires one selected transform')
        source_id = geometry.source_region_id
        source = self.working[source_id]
        matrix = affine_tuple(geometry.transforms[0])
        if symbolic:
            params = {**(shape_metadata or {}), 'href_obj_id': source.id, 'transform': matrix}
            source_path, target_path = self.paths[source_id], self.paths[selected.id]
            if source_path != (*target_path[:-1], source.id) or 'href_region_path' in params:
                params['href_region_path'] = relative_source_path(target_path, source_path)
            shape = Shape('use', params)
        else:
            shape = transform_working_shape(self.shapes[source_id], matrix)
        return target.with_current(shape,
            footprint=footprint if footprint is not None else source.footprint.affine_transform(matrix),
            diagnostics=selected.diagnostics if diagnostics is None else diagnostics)


@dataclass(frozen=True)
class BoundPathInput:
    """Canonical command inputs before filled-component publication."""

    target: RegionSample
    paths: tuple[PathGeometry, ...]

    def publish(self, paths: Sequence[PathGeometry], *, preserve_subpaths: bool = True,
                path_text: str | None = None) -> RegionSample:
        from .path_topology import component_paths, replace_component_paths
        from .refinement.region_sample import to_polygon
        from .path_geometry import command_token

        paths = tuple(paths)
        if preserve_subpaths and tuple(path.path.id for path in paths) != tuple(path.path.id for path in self.paths):
            raise ValueError('path publication changed bound path membership')
        if path_text is not None:
            parsed = parse_svg_paths(path_text, region_id='publication',
                path_ids=tuple(path.path.id for path in paths))
            if any(tuple(command_token(command) for command in old.path.commands) !=
                   tuple(command_token(command) for command in new.path.commands)
                   for old, new in zip(paths, parsed, strict=True)):
                raise ValueError('path text differs from selected canonical commands')
        params = {**self.target.current.params,
                  'd': path_text if path_text is not None else ' '.join(path.to_svg_d() for path in paths)}
        if (geometry := params.get('canonical_path_geometry')) is not None:
            by_id = {path.path.id: path for path in paths}
            params['canonical_path_geometry'] = replace_component_paths(geometry,
                tuple(by_id[path.path.id] for path in component_paths(geometry))) if preserve_subpaths else paths[0]
        elif len(paths) == 1:
            params['canonical_path_geometry'] = paths[0]
        if not preserve_subpaths:
            params.pop('fill_rule', None)
        params['path_ids'] = tuple(path.path.id for path in paths)
        shape = Shape('path', params)
        return self.target.with_current(shape, footprint=to_polygon(shape))


def _project_cached_paths(geometry, d: str, *, region_id: str, path_ids=None) -> tuple[PathGeometry, ...]:
    """Validate numeric correspondence and recover source-ordered retained paths."""
    from .path_topology import component_paths
    from .path_geometry import command_token

    source = parse_svg_paths(d, region_id=region_id, path_ids=path_ids)
    retained = {path.path.id: path for path in component_paths(geometry)}
    paths = []
    for path in source:
        values = tuple(command_token(command) for command in path.path.commands)
        candidates = [retained[path.path.id]] if path.path.id in retained else list(retained.values())
        matches = []
        for candidate in candidates:
            exact = tuple(command_token(command) for command in candidate.path.commands)
            rounded, = parse_svg_paths(candidate.to_svg_d(), region_id='projection')
            display = tuple(command_token(command) for command in rounded.path.commands)
            if values in (exact, display):
                matches.append(candidate)
        if len(matches) != 1:
            raise ValueError('canonical path replay geometry does not match path data')
        selected = matches[0]
        paths.append(selected)
        del retained[selected.path.id]
    if retained:
        raise ValueError('canonical path replay geometry does not match path data')
    return tuple(paths)


def bind_path_input(target: RegionSample) -> BoundPathInput:
    """Select exact commands without forcing intermediate paths into regions."""
    if not target.is_leaf or target.current.kind != 'path':
        raise ValueError('path input binding requires a retained path leaf')
    params = target.current.params
    d, identity = str(params['d']), target.drawing_id or f'r{target.id}'
    geometry = params.get('canonical_path_geometry')
    paths = _project_cached_paths(geometry, d, region_id=identity, path_ids=params.get('path_ids')) \
        if geometry is not None else parse_svg_paths(d, region_id=identity, path_ids=params.get('path_ids'))
    return BoundPathInput(target, paths)


def resolved_input_shape(region: RegionSample) -> Shape:
    """Bind retained instance geometry for an operation that consumes its shape."""
    shape = region.current
    if shape is None:
        raise ValueError('operation input requires retained leaf geometry')
    return resolved_instance_shape(shape, region.footprint)


def resolved_instance_shape(shape: Shape, footprint: SkPath) -> Shape:
    """Use the retained instance outline when a selected operation consumes a clone."""
    if shape.kind == 'use' or 'canonical_clone_geometry' in shape.params:
        return Shape('path', {'d': footprint.to_svg_d()})
    return shape


def bind_region_inputs(
    regions: Mapping[str, RegionSample], *, paths: Mapping[str, tuple[int, ...]] | None = None,
) -> BoundRegionInputs:
    """Bind selected leaves without exposing private IDs as canonical handles.

    Symbolic dependencies inside this selection stay symbolic. Dependencies
    outside it use their retained resolved footprint, without rasterizing or
    recognition/refitting.
    """
    addresses = {key: tuple(paths[key]) if paths is not None else (region.id,)
                 for key, region in regions.items()}
    if len(set(addresses.values())) != len(addresses):
        raise ValueError('operation inputs refer to the same scoped region')
    prepared = []
    shapes = {}
    for number, (key, region) in enumerate(regions.items(), 1):
        if not region.is_leaf or region.current is None:
            raise ValueError('automatic input binding requires retained leaf geometry')
        shape = resolved_input_shape(region)
        shapes[key] = shape
        prepared.append(replace(region, id=number, drawing_id=key, current=shape))
    drawing = drawing_from_region_samples('automatic-inputs', width=1, height=1, regions=prepared)
    from .refinement.symmetry_materialize import UnboundCloneSourceError
    native = {addresses[key]: region for key, region in regions.items()}
    keys = {path: key for key, path in addresses.items()}
    selected = []
    for region in drawing.regions:
        original = regions[region.id].current
        if original.kind == 'use':
            try:
                source_path = resolve_source_path(native, addresses[region.id], original.params,
                    allow_external=True)
            except UnboundCloneSourceError:
                source_path = None  # Valid dependency outside this operation's input selection.
            if source_path is not None:
                region = replace(region, geometry=CloneGeometry(keys[source_path],
                    (AffineTransform(*original.params.get('transform', (1, 0, 0, 1, 0, 0))),)))
        selected.append(region)
    drawing = replace(drawing, regions=tuple(selected))
    return BoundRegionInputs(drawing, MappingProxyType(dict(regions)),
        MappingProxyType(addresses), MappingProxyType(shapes))


def working_leaf_targets(regions: Sequence[RegionSample]) -> dict[str, WorkingLeafTarget]:
    """Derive public target handles from the region tree; do not persist an index."""
    result: dict[str, WorkingLeafTarget] = {}

    def visit(region: RegionSample, root_label: str, legacy_label: str, path: tuple[int, ...]) -> None:
        if region.is_leaf:
            label = _public_region_id(region, root_label, legacy_label)
            if label in result:
                raise ValueError(f"duplicate drawing region id: {label}")
            # Target lineage belongs to the retained leaf, not to its root
            # container.  A root can be a semantic union while its children
            # remain independently traced, cloned, or split regions.
            sources = frozenset(region.source_regions or (root_label,))
            result[label] = WorkingLeafTarget(label, root_label, region, sources, path)
            return
        for child in region.children:
            visit(child, root_label, f"{legacy_label}-{child.id}", (*path, child.id))

    for region in regions:
        drawing_id = region.drawing_id
        if type(drawing_id) is not str:
            raise ValueError("drawing root is missing a stable drawing_id")
        visit(region, drawing_id, drawing_id, (region.id,))
    return result


def replace_working_region(
    regions: Sequence[RegionSample], target_region_id: str,
    replacement: Callable[[RegionSample], RegionSample],
) -> tuple[RegionSample, ...]:
    """Replace exactly one addressed working node without changing siblings."""
    matched = 0
    def visit(region: RegionSample) -> RegionSample:
        nonlocal matched
        if region.drawing_id == target_region_id:
            matched += 1
            return replacement(region)
        if region.is_leaf:
            return region
        children = tuple(visit(child) for child in region.children)
        return region if children == region.children else region.with_children(children)
    result = tuple(visit(region) for region in regions)
    if matched != 1:
        raise ValueError(f'drawing region target is absent or ambiguous: {target_region_id!r}')
    return result


def shape_from_region_geometry(region: DrawingRegion, metadata: Mapping[str, object]) -> Shape:
    """Project one concrete canonical leaf into the temporary Shape vocabulary."""
    geometry = region.geometry
    if isinstance(geometry, PrimitiveGeometry):
        shape = shape_from_primitive_geometry(geometry)
        return Shape(shape.kind, {**metadata, **shape.params})
    if not isinstance(geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        raise ValueError('selected leaf requires concrete primitive or path geometry')
    return Shape('path', {**metadata, 'd': geometry.to_svg_d(), 'canonical_path_geometry': geometry})


def shape_from_primitive_geometry(geometry: PrimitiveGeometry) -> Shape:
    """Project selected numeric primitive values; never invoke recognition."""
    primitive = geometry.primitive
    if isinstance(primitive, CirclePrimitive):
        return Shape("circle", {"cx": primitive.center[0], "cy": primitive.center[1], "r": primitive.radius})
    if isinstance(primitive, EllipsePrimitive):
        return Shape("ellipse", {"cx": primitive.center[0], "cy": primitive.center[1],
                                 "rx": primitive.radius_x, "ry": primitive.radius_y})
    if isinstance(primitive, RoundedRectanglePrimitive):
        return Shape("rect", {"x": primitive.origin[0], "y": primitive.origin[1],
                              "w": primitive.width, "h": primitive.height,
                              "rx": primitive.radius_x, "ry": primitive.radius_y})
    if isinstance(primitive, RectanglePrimitive):
        return Shape("rect", {"x": primitive.origin[0], "y": primitive.origin[1],
                              "w": primitive.width, "h": primitive.height})
    if isinstance(primitive, PolygonPrimitive):
        return Shape("polygon", {"points": tuple(primitive.vertices)})
    raise TypeError(f"unsupported typed primitive: {type(primitive).__name__}")


def publish_seam_shape(obj: RegionSample, shape: Shape, other_id: int, candidate, *, selected: str) -> RegionSample:
    from .refinement.region_sample import to_polygon

    return obj.with_current(shape, footprint=to_polygon(shape), diagnostics={'seams': {
        'accepted': True, 'paired_with': int(other_id), 'selected': selected,
        'axis': candidate.axis, 'gap_before': abs(candidate.a_value - candidate.b_value),
        'seam': candidate.seam}})


def replace_private_leaves(root: RegionSample, replacements: Mapping[int, RegionSample]) -> RegionSample:
    """Publish addressed leaf edits without replacing the retained exterior."""
    if root.is_leaf:
        return replacements.get(int(root.id), root)
    children = tuple(replace_private_leaves(child, replacements) for child in root.children)
    return replace(root, children=children)


def publish_collapsed_region(
    selected: DrawingRegion, target: RegionSample, *, footprint: SkPath,
) -> RegionSample:
    """Retain observations while publishing a selected single painted union."""
    if selected.is_branch:
        identities = {selected.id: target.id}
        def identify(region, number=0):
            for index, child in enumerate(region.children, 1):
                identities[child.id] = index
                identify(child)
        identify(selected)
        result = publish_generated_tree(selected, identities=identities,
            raster_shape=target.raster.shape, retained_rasters={selected.id: target.raster})
        return replace(result, raster=target.raster, footprint=footprint,
            source_label=target.source_label, color_hex=target.color_hex,
            source_regions=target.source_regions, coverage=target.coverage)
    leaves = target.leaves()
    shape = shape_from_region_geometry(selected, {'fill_rule': 'evenodd'})
    shape = Shape(shape.kind, {**shape.params, 'd': footprint.to_svg_d()})
    consumed = tuple(dict.fromkeys(leaf.id for leaf in leaves))
    diagnostics = {**selected.diagnostics, 'materialization': {
        **selected.diagnostics.get('materialization', {}),
        'consumed_region_ids': consumed,
        'retired_region_ids': tuple(identity for identity in consumed if identity != target.id),
        'source_region_remap': tuple((identity, target.id) for identity in consumed),
    }}
    return replace(target, current=shape, children=(), footprint=footprint,
        fill=leaves[0].fill, original=leaves[0].original, diagnostics=diagnostics,
        source_regions=tuple(sorted(set(target.source_regions).union(
            *(leaf.source_regions for leaf in leaves)))))


def publish_selected_leaf(
    selected: DrawingRegion, target: RegionSample, *,
    shape_metadata: Mapping[str, object] | None = None,
) -> RegionSample:
    """Publish concrete canonical leaf geometry, retaining working paint/evidence.

    This boundary does not fit geometry or assign public IDs. Paint, source
    observations and private numeric identity belong to the retained target.
    """
    from .refinement.region_sample import to_polygon
    if not selected.is_leaf or not target.is_leaf:
        raise ValueError('selected geometry publication requires leaf targets')
    shape = shape_from_region_geometry(selected, shape_metadata or {})
    return replace(target, current=shape, footprint=to_polygon(shape), diagnostics=selected.diagnostics)


def transform_working_shape(
    shape: Shape, matrix: tuple[float, float, float, float, float, float],
) -> Shape:
    """Project the shared numeric transform and keep retained path annotations current."""
    from .emit import apply_affine_point, shape_to_path_d, transform_path_d
    from .drawing_corner_actions import rebind_corner_annotations
    from .path_topology import component_paths, replace_component_paths
    if shape.kind == "circle":
        cx, cy = apply_affine_point(matrix, float(shape.params["cx"]), float(shape.params["cy"]))
        return Shape("circle", {"cx": cx, "cy": cy, "r": float(shape.params["r"])})
    if shape.kind != "path":
        return Shape("path", {"d": transform_path_d(shape_to_path_d(shape), matrix)})
    params = dict(shape.params)
    params["d"] = transform_path_d(str(shape.params.get("d", "")), matrix)
    canonical = params.get("canonical_path_geometry")
    if canonical is not None:
        originals = component_paths(canonical)
        parsed = parse_svg_paths(params["d"], region_id="transformed",
            path_ids=tuple(path.path.id for path in originals))
        # Projection retains the existing numeric rounding policy, but IDs and
        # anchors belong to the original typed commands, including sparse IDs.
        aligned = tuple(replace(path, path=replace(path.path, commands=tuple(
            replace(command, id=previous.id)
            for command, previous in zip(path.path.commands, original.path.commands, strict=True))))
            for path, original in zip(parsed, originals, strict=True))
        updated = rebind_corner_annotations(replace_component_paths(canonical, aligned), canonical)
        params.update(canonical_path_geometry=updated, d=updated.to_svg_d())
    return Shape("path", params)


def rebind_working_corner_provenance(
    before: Sequence[RegionSample],
    after: Sequence[RegionSample],
    *,
    tolerance: float,
) -> tuple[RegionSample, ...]:
    """Carry declared corners across a rewrite, scoped by full tree ancestry."""
    originals = index_region_paths(before)

    def visit(region: RegionSample, parent: tuple[int, ...] = ()) -> RegionSample:
        path = (*parent, region.id)
        if region.is_branch:
            children = tuple(visit(child, path) for child in region.children)
            # Equality ignores diagnostics; annotation-only child changes still
            # require publishing a rebuilt branch.
            return (
                region.with_children(children)
                if any(child is not original for child, original in zip(children, region.children, strict=True))
                else region
            )
        assert region.current is not None
        original = originals.get(path)
        if (
            original is None
            or original.current is None
            or original.current.kind != "path"
            or region.current.kind != "path"
        ):
            return region
        corners = original.diagnostics.get("corners")
        if (
            not isinstance(corners, Mapping)
            or not corners.get("spans")
            or corners.get("path") != str(original.current.params.get("d", ""))
        ):
            return region
        rebound = rebind_prefit_corner_diagnostics(
            str(region.current.params["d"]), corners, tolerance=tolerance,
        )
        return region.with_diagnostics({"corners": rebound})

    return tuple(visit(region) for region in after)


def restore_working_root_metadata(region: RegionSample, original: RegionSample | None) -> RegionSample:
    """Publish selected geometry with its original root-owned drawing identity."""
    if original is None or original.drawing_id is None:
        return region
    return replace(
        region,
        drawing_id=original.drawing_id,
        source_regions=original.source_regions,
    )


def working_region_evidence(regions: Sequence[RegionSample], *, baseline: Sequence[RegionSample] = ()) -> dict[str, dict[str, object]]:
    """Retain only boundary evidence needed to project canonical selections."""
    result = {}
    indexed = index_region_paths(baseline)
    prior = {address: indexed[path] for path, address in region_sample_address_index(baseline).items()}
    addresses = region_sample_address_index(regions)
    def visit(region: RegionSample, parent: tuple[int, ...] = ()) -> None:
        path = (*parent, region.id)
        address = addresses[path]
        coverage = region.coverage
        values = coverage.values if isinstance(coverage, CoverageField) else coverage
        previous = prior.get(address)
        borrow_raster = previous is not None and np.array_equal(previous.raster, region.raster)
        old_coverage = None if previous is None else previous.coverage
        old_values = old_coverage.values if isinstance(old_coverage, CoverageField) else old_coverage
        borrow_coverage = previous is not None and type(old_coverage) is type(coverage) and (
            values is None or (np.array_equal(values, old_values) and (
                not isinstance(coverage, CoverageField) or (coverage.origin == old_coverage.origin
                    and coverage.canvas_shape == old_coverage.canvas_shape))))
        result[address] = {
            'id': region.id, 'ancestry': path, 'raster_shape': region.raster.shape,
            'raster': None if borrow_raster else region.raster.tobytes(),
            'raster_source': address if borrow_raster else None,
            'source_label': region.source_label, 'color_hex': region.color_hex,
            'original_kind': None if region.original is None else region.original.kind,
            'original_params': {} if region.original is None else {
                key: value for key, value in region.original.params.items()
                if key not in {'canonical_path_geometry', 'canonical_clone_geometry'}
            },
            'coverage': None if values is None or borrow_coverage else values.tobytes(),
            'coverage_source': address if borrow_coverage else None,
            'coverage_shape': None if values is None else values.shape,
            'coverage_dtype': None if values is None else values.dtype.str,
            'coverage_origin': coverage.origin if isinstance(coverage, CoverageField) else None,
            'coverage_canvas': coverage.canvas_shape if isinstance(coverage, CoverageField) else None,
        }
        for child in region.children:
            visit(child, path)
    for root in regions:
        visit(root)
    return result


def working_shape_metadata(regions: Sequence[RegionSample], selected: Sequence[DrawingRegion]) -> dict[str, dict[str, object]]:
    """Retain kernel annotations, never a second copy of canonical geometry."""
    indexed = index_region_paths(regions)
    native = {address: indexed[path] for path, address in region_sample_address_index(regions).items()}
    metadata = {}
    for root in selected:
        for leaf in root.leaves():
            target = native[leaf.id]
            assert target.current is not None
            core = ({"d", "href_obj_id", "href_region_path", "transform", "canonical_path_geometry"}
                if isinstance(leaf.geometry, CloneGeometry) else shape_from_region_geometry(leaf, {}).params)
            metadata[leaf.id] = {key: value for key, value in target.current.params.items()
                if key not in core and key != "canonical_clone_geometry"}
    return metadata


def current_shape_for_region(
    region: DrawingRegion, by_id: Mapping[str, DrawingRegion],
    native_paths: Mapping[str, tuple[int, ...]], metadata: Mapping[str, object],
) -> Shape:
    """Project selected geometry once for numeric consumers and input receipts."""
    from .geometry_native import region_path, affine_tuple
    from .refinement.symmetry_materialize import relative_source_path
    geometry = region.geometry
    if not isinstance(geometry, CloneGeometry):
        return shape_from_region_geometry(region, metadata)
    if len(geometry.transforms) == 1:
        source_path = native_paths[geometry.source_region_id]
        return Shape('use', {**metadata, 'href_obj_id': source_path[-1],
            'href_region_path': relative_source_path(native_paths[region.id], source_path),
            'transform': affine_tuple(geometry.transforms[0])})
    return Shape('path', {**metadata, 'd': region_path(region, by_id).to_svg_d(),
        'canonical_clone_geometry': geometry})


def restore_selected_roots(
    selected: Sequence[DrawingRegion], root_order: Sequence[str], regions: Sequence[RegionSample], *,
    shape_metadata: Mapping[str, Mapping[str, object]], region_evidence: Mapping[str, Mapping[str, object]],
) -> tuple[RegionSample, ...]:
    """Project an already selected root delta without topology/fitting decisions."""
    from .geometry_native import region_path
    from .drawing_fill import serialize_fill
    from .skia_geometry import unary_union
    from .refinement.region_sample import to_polygon
    existing = {root.drawing_id: root for root in regions}
    changed = {root.id: root for root in selected}
    canonical = (drawing_from_region_samples('selected-input', width=1, height=1, regions=regions)
        if regions else None)
    previous_metadata = working_shape_metadata(regions, canonical.regions) if canonical is not None else {}
    all_roots = tuple(changed[identity] if identity in changed else canonical.region(identity)
        for identity in root_order)
    previous = index_region_paths(regions)
    addresses = region_sample_address_index(regions)
    paths_by_address = {address: path for path, address in addresses.items()}
    by_id = {}
    native_paths = {}
    def index(region: DrawingRegion, parent: tuple[int, ...] = ()) -> None:
        by_id[region.id] = region
        info = region_evidence.get(region.id)
        if info is not None:
            path = (*parent, int(info['id']))
        else:
            path = paths_by_address[region.id]
        native_paths[region.id] = path
        for child in region.children:
            index(child, path)
    for root in all_roots:
        index(root)
    prior = {address: previous[path] for address, path in paths_by_address.items()}
    def convert(region: DrawingRegion) -> RegionSample:
        info = region_evidence[region.id]
        baseline = prior.get(region.id)
        if (region.is_leaf and baseline is not None
                and info.get('raster_source') == region.id and info.get('coverage_source') == region.id):
            original = canonical.region(region.id)
            if (shape_metadata.get(region.id, {}) == previous_metadata.get(region.id, {})
                    and region == original and region.diagnostics == original.diagnostics
                    and (not isinstance(region.geometry, CloneGeometry)
                         or region_path(region, by_id).to_svg_d() == baseline.footprint.to_svg_d())):
                return baseline
            if (not isinstance(region.geometry, CloneGeometry)
                    and _fill(baseline.fill) == region.fill and baseline.z == region.z
                    and baseline.source_regions == region.source_region_ids):
                return publish_selected_leaf(region, baseline,
                    shape_metadata=shape_metadata.get(region.id, {}))
        if region.is_branch:
            children = tuple(convert(child) for child in region.children)
            shape = None
            footprint = unary_union([child.footprint for child in children])
        else:
            children = ()
            shape = current_shape_for_region(region, by_id, native_paths, shape_metadata.get(region.id, {}))
            footprint = to_polygon(shape) if info.get('footprint_codec') == 'svg' else region_path(region, by_id)
        original = None if info['original_kind'] is None else Shape(info['original_kind'], dict(info['original_params']))
        fill = (baseline.fill if baseline is not None and _fill(baseline.fill) == region.fill
            else None if region.fill is None else working_fill_from_spec(serialize_fill(region.fill)))
        coverage = prior[info['coverage_source']].coverage if info.get('coverage_source') is not None else None
        if info.get('coverage') is not None:
            coverage = np.frombuffer(info['coverage'], dtype=info['coverage_dtype']).reshape(info['coverage_shape'])
            if info['coverage_origin'] is not None:
                coverage = CoverageField(coverage, info['coverage_origin'], info['coverage_canvas'])
        raster = (prior[info['raster_source']].raster if info.get('raster_source') is not None
            else np.frombuffer(info['raster'], dtype=bool).reshape(info['raster_shape']))
        return RegionSample(int(info['id']), shape, fill,
            region.z, footprint=footprint, raster=raster,
            original=original, children=children, source_label=info['source_label'], color_hex=info['color_hex'],
            drawing_id=region.id, source_regions=region.source_region_ids,
            coverage=coverage, diagnostics=region.diagnostics)
    return tuple(convert(changed[identity]) if identity in changed else existing[identity] for identity in root_order)


def publish_region_update(selected: DrawingRegion, regions: Sequence[RegionSample], *,
    shape_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[RegionSample, ...]:
    """Project a canonical region update without repeating its geometry decision."""
    from .drawing_relationships import replace_region
    current = drawing_from_region_samples('relationship', width=1, height=1, regions=regions)
    return _publish_drawing_delta(current, replace_region(current, selected), regions, shape_metadata or {})


def publish_drawing_update(updated: Drawing, regions: Sequence[RegionSample]) -> tuple[RegionSample, ...]:
    """Publish canonical structural changes with their retained observations."""
    current = drawing_from_region_samples(updated.drawing_id, width=updated.canvas.width,
        height=updated.canvas.height, regions=regions)
    return _publish_drawing_delta(current, updated, regions, {})


def generated_tree_observations(
    selected: DrawingRegion, *, identities: Mapping[str, int], raster_shape: tuple[int, int],
    retained_rasters: Mapping[str, np.ndarray] | None = None,
    parent_ancestry: tuple[int, ...] = (),
):
    """Capture generated observations without allocating working regions."""
    from .refinement.gate import rasterize
    from .refinement.region_sample import to_polygon
    from .drawing_fill import serialize_fill

    observations, metadata = {}, {}
    def capture(region: DrawingRegion, parent=()) -> np.ndarray:
        ancestry = (*parent, identities[region.id])
        shape = None
        if region.is_branch:
            children = [capture(child, ancestry) for child in region.children]
            raster = np.logical_or.reduce(children) if children else np.zeros(raster_shape, bool)
        else:
            shape = shape_from_region_geometry(region, {})
            raster = rasterize(to_polygon(shape), raster_shape)
        if region.id in (retained_rasters or {}):
            raster = np.asarray(retained_rasters[region.id], dtype=bool)
        paint = None if region.fill is None else serialize_fill(region.fill)
        observations[region.id] = {'id': identities[region.id], 'raster_shape': raster.shape,
            'ancestry': ancestry,
            'raster': raster.tobytes(), 'raster_source': None, 'coverage_source': None,
            'coverage': None, 'original_kind': None if shape is None else shape.kind,
            'original_params': {} if shape is None else {key: value for key, value in shape.params.items()
                if key not in {'canonical_path_geometry', 'canonical_clone_geometry'}}, 'source_label': None,
            'color_hex': paint['color'] if paint is not None and paint['type'] == 'flat' else None,
            'coverage_shape': None, 'coverage_dtype': None, 'coverage_origin': None, 'coverage_canvas': None}
        if shape is not None:
            core = {'d', 'canonical_path_geometry'} if shape.kind == 'path' else set(shape.params)
            metadata[region.id] = {key: value for key, value in shape.params.items() if key not in core}
        return raster
    capture(selected, parent_ancestry)
    return observations, metadata


def publish_generated_tree(
    selected: DrawingRegion, *, identities: Mapping[str, int], raster_shape: tuple[int, int],
    retained_rasters: Mapping[str, np.ndarray] | None = None,
) -> RegionSample:
    """Project generated selections for remaining numerical proposal consumers."""
    observations, metadata = generated_tree_observations(selected, identities=identities,
        raster_shape=raster_shape, retained_rasters=retained_rasters)
    observations = {identity: {**info, 'footprint_codec': 'svg'} for identity, info in observations.items()}
    return restore_selected_roots((selected,), (selected.id,), (), shape_metadata=metadata,
                                  region_evidence=observations)[0]


def _publish_drawing_delta(
    current: Drawing, updated: Drawing, regions: Sequence[RegionSample],
    shape_metadata: Mapping[str, Mapping[str, object]],
) -> tuple[RegionSample, ...]:
    from .replay_state import ReplayState, publish_drawing_state
    state = publish_drawing_state(ReplayState(current,
        working_shape_metadata(regions, current.regions), working_region_evidence(regions)),
        updated, shape_metadata=shape_metadata)
    # The disposable boundary may borrow existing buffers to preserve working
    # identity; canonical state itself retains self-contained observations.
    borrowed = working_region_evidence(regions, baseline=regions)
    evidence = {identity: borrowed.get(identity, info)
        for identity, info in state.region_evidence.items()}
    original_roots = {root.id: root for root in current.regions}
    changed = tuple(root for root in updated.regions if original_roots.get(root.id) is not root)
    return restore_selected_roots(changed, tuple(root.id for root in updated.regions), regions,
        shape_metadata=state.shape_metadata, region_evidence=evidence)


def region_sample_address_index(
    regions: Sequence[RegionSample],
) -> dict[tuple[int, ...], str]:
    """Return the public region ID owned by every engine-tree ancestry."""
    addresses: dict[tuple[int, ...], str] = {}

    def index(
        region: RegionSample,
        root_id: str,
        legacy_id: str,
        path: tuple[int, ...],
    ) -> None:
        addresses[path] = _public_region_id(region, root_id, legacy_id)
        for child in region.children:
            index(child, root_id, f"{legacy_id}-{child.id}", (*path, child.id))

    for root in regions:
        if not isinstance(root.drawing_id, str):
            raise ValueError("Python-engine roots require stable drawing IDs")
        index(root, root.drawing_id, root.drawing_id, (root.id,))
    return addresses


def drawing_from_region_samples(
    drawing_id: str,
    *,
    width: float,
    height: float,
    regions: Sequence[RegionSample],
) -> Drawing:
    """Project Python-engine working regions into the canonical scene model."""
    region_addresses = region_sample_address_index(regions)
    reference_index = index_region_paths(regions)

    def clone_geometry(region: RegionSample, path: tuple[int, ...]) -> CloneGeometry:
        assert region.current is not None and region.current.kind == "use"
        source_path = resolve_source_path(reference_index, path, region.current.params)
        transform = AffineTransform(
            *(float(value) for value in region.current.params.get(
                "transform", (1, 0, 0, 1, 0, 0)
            ))
        )
        visited = {path}
        while source_path is not None:
            if source_path in visited:
                raise ValueError("clone source chain contains a cycle")
            visited.add(source_path)
            source = reference_index[source_path]
            if not source.is_leaf or source.current is None:
                raise ValueError("clone source requires retained concrete geometry")
            if source.current.kind != "use":
                return CloneGeometry(region_addresses[source_path], (transform,))
            inner = AffineTransform(
                *(float(value) for value in source.current.params.get(
                    "transform", (1, 0, 0, 1, 0, 0)
                ))
            )
            transform = compose_affine(transform, inner)
            source_path = resolve_source_path(
                reference_index, source_path, source.current.params
            )
        raise ValueError("clone source requires retained concrete geometry")

    def convert(region: RegionSample, root_id: str, legacy_id: str, path: tuple[int, ...]) -> DrawingRegion:
        region_id = region_addresses[path]
        diagnostics = dict(region.diagnostics)
        common = {
            "id": region_id,
            "fill": _fill(region.fill),
            "z": float(region.z),
            "source_region_ids": tuple(region.source_regions or (region_id,)),
            "diagnostics": diagnostics,
        }
        if region.is_branch:
            return DrawingRegion(
                **common,
                children=tuple(
                    convert(child, root_id, f"{legacy_id}-{child.id}", (*path, child.id))
                    for child in region.children
                ),
            )
        assert region.current is not None
        if 'canonical_clone_geometry' in region.current.params:
            geometry = region.current.params['canonical_clone_geometry']
        elif region.current.kind == "use":
            geometry = clone_geometry(region, path)
        else:
            geometry = _geometry(
                region.current.kind, region.current.params, region_id, {}
            )
        if (
            region.current.kind == "path"
            and "canonical_path_geometry" not in region.current.params
            and isinstance(geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry))
        ):
            geometry, diagnostics["corners"] = corner_provenance.ingress_prefit_corners(
                geometry,
                str(region.current.params["d"]),
                region_id,
                region.diagnostics.get("corners"),
                path_ids=region.current.params.get("path_ids"),
            )
        return DrawingRegion(
            **common,
            geometry=geometry,
        )

    return Drawing(
        drawing_id=drawing_id,
        canvas=Canvas(float(width), float(height)),
        regions=tuple(
            convert(root, root.drawing_id, root.drawing_id, (root.id,))
            for root in regions
            if isinstance(root.drawing_id, str)
        ),
    )


def _public_region_id(region: RegionSample, root_id: str, legacy_id: str) -> str:
    candidate = region.drawing_id
    if isinstance(candidate, str) and (
        candidate.startswith(f"{root_id}-") or (candidate == root_id and legacy_id == root_id)
    ):
        return candidate
    return legacy_id


def _geometry(kind: str, params: Mapping[str, object], region_id: str, public_ids: dict[int, str]):
    if kind == "path":
        canonical = params.get("canonical_path_geometry")
        if canonical is not None:
            if not isinstance(canonical, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
                raise TypeError(
                    "canonical path replay geometry must be PathGeometry or "
                    "PathGeometryWithCutouts or CompoundPathGeometry"
                )
            _project_cached_paths(canonical, str(params['d']), region_id=region_id,
                                  path_ids=params.get('path_ids'))
            return canonical
        fill_rule = "evenodd" if params.get("fill_rule") == "evenodd" else "nonzero"
        return geometry_from_svg_d(str(params["d"]), region_id=region_id, fill_rule=fill_rule, path_ids=params.get("path_ids"), defer_overlaps=True)
    if kind in {"circle", "ellipse", "rect", "polygon"}:
        from .primitive_fitting import primitive_geometry_from_shape
        return primitive_geometry_from_shape(Shape(kind, dict(params)))
    if kind == "use":
        source_id = public_ids[int(params["href_obj_id"])]
        values = tuple(float(value) for value in params.get("transform", (1, 0, 0, 1, 0, 0)))
        return CloneGeometry(source_id, (AffineTransform(*values),))
    raise ValueError(f"unsupported Python-engine shape kind: {kind!r}")


def working_fill_from_spec(spec: Mapping[str, object]):
    """Project canonical paint data at the working-record boundary only."""
    kind = spec['type']
    if kind == 'flat':
        return LegacyFlatFill(spec['color'])
    if kind == 'linear_gradient':
        start, end = spec['start'], spec['end']
        return LegacyLinearGradientFill({'x1': start[0], 'y1': start[1], 'x2': end[0], 'y2': end[1]},
            [(stop['offset'], stop['color']) for stop in spec['stops']])
    if kind == 'radial_gradient':
        center = spec['center']
        geometry = {'cx': center[0], 'cy': center[1], 'r': spec['radius']}
        focal = spec.get('focal_point')
        if focal is not None:
            geometry.update(fx=focal[0], fy=focal[1])
        return LegacyRadialGradientFill(geometry, [(stop['offset'], stop['color']) for stop in spec['stops']])
    origin = spec['origin']
    return LegacyRasterFill({'x': origin[0], 'y': origin[1], 'w': spec['width'], 'h': spec['height']}, spec['png_base64'])


def refresh_dependent_clones(
    action: RefreshDependentClonesOperation, resolver: RegionResolver,
) -> list[RegionSample]:
    """Refresh the recorded clones and publish their containing roots atomically."""
    from .refinement.automatic_replay import apply_clone_reference_update
    replacements_by_path: dict[tuple[int, ...], RegionSample] = {}
    for refresh in action.clones:
        clone = resolver.resolve(refresh.clone)
        source = resolver.resolve(refresh.source)
        clone_path = tuple(refresh.clone.ancestry)
        bindings = bind_region_inputs({'source': source, 'target': clone},
            paths={'source': tuple(refresh.source.ancestry), 'target': clone_path})
        replacements_by_path[clone_path] = apply_clone_reference_update(bindings,
            previous_source_id=refresh.previous_source_id,
            previous_source_path=refresh.previous_source_path,
        )

    def replace_paths(region: RegionSample, path: tuple[int, ...]) -> RegionSample:
        if path in replacements_by_path:
            return replacements_by_path[path]
        if region.is_leaf:
            return region
        children = tuple(replace_paths(child, (*path, int(child.id))) for child in region.children)
        return region.with_children(children) if children != region.children else region

    return [replace_paths(resolver.resolve(root), tuple(root.ancestry)) for root in action.roots]


def root_region_address(region: RegionSample) -> str:
    if type(region.drawing_id) is not str:
        raise ValueError("drawing root is missing a stable drawing_id")
    return region.drawing_id


def with_drawing_address(region: RegionSample, address: str) -> RegionSample:
    """Copy *region* with its public, version-scoped drawing address."""
    if region.drawing_id == address:
        return region
    if region.is_leaf:
        assert region.current is not None
        return RegionSample(
            id=region.id,
            current=region.current,
            fill=region.fill,
            z=region.z,
            footprint=region.footprint,
            raster=region.raster,
            original=region.original,
            source_label=region.source_label,
            color_hex=region.color_hex,
            drawing_id=address,
            source_regions=region.source_regions,
            coverage=region.coverage,
            diagnostics=region.diagnostics,
        )
    return RegionSample.branch(
        id=region.id,
        children=region.children,
        z=region.z,
        raster=region.raster,
        footprint=region.footprint,
        fill=region.fill,
        source_label=region.source_label,
        color_hex=region.color_hex,
        drawing_id=address,
        source_regions=region.source_regions,
        diagnostics=region.diagnostics,
        original=region.original,
    )


def normalize_drawing_addresses(regions: Sequence[RegionSample]) -> tuple[RegionSample, ...]:
    """Expose compact local child paths while preserving internal region IDs.

    Refinement-created child IDs are intentionally global (for example ``1001``)
    and can grow with every split.  They are implementation details.  The
    drawing protocol instead names a child by its one-based position in its
    retained parent: ``r1-1``, ``r1-2``, and so on.  Addresses are normalized
    only once a version is complete, so every plan still resolves against the
    exact version it names.
    """
    def visit(region: RegionSample, address: str) -> RegionSample:
        if region.is_branch:
            used = {child.drawing_id for child in region.children if child.drawing_id}
            children = []
            next_index = 1
            for child in region.children:
                candidate = child.drawing_id
                if not isinstance(candidate, str) or not candidate.startswith(f"{address}-"):
                    while f"{address}-{next_index}" in used:
                        next_index += 1
                    candidate = f"{address}-{next_index}"
                    used.add(candidate)
                children.append(visit(child, candidate))
            if any(a is not b for a, b in zip(children, region.children, strict=True)):
                region = region.with_children(children)
        return with_drawing_address(region, address)

    return tuple(visit(region, root_region_address(region)) for region in regions)

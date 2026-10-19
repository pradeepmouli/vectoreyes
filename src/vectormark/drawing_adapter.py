"""Temporary adapter from the Python optimizer's records to canonical drawings.

The native/public contract is :class:`Drawing`. ``VectorRegion`` remains the
Python engine's working record while it still owns raster masks and Skia
footprints. This module is the only sanctioned projection across that seam.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .optimizer.automatic_replay import RefreshDependentClonesOperation, RegionResolver

from .candidate import (
    FlatFill as LegacyFlatFill,
    LinearGradientFill as LegacyLinearGradientFill,
    RadialGradientFill as LegacyRadialGradientFill,
    RasterFill as LegacyRasterFill,
)
from .drawing import Canvas, Drawing, DrawingRegion
from .drawing_actions import SymmetryAxisInput
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
from .optimizer.corners import rebind_prefit_corner_diagnostics, source_ordered_corner_diagnostics
from . import corner_provenance
from .paint_fitting import canonical_fill as _fill
from .optimizer.vector_region import VectorRegion
from .types import CoverageField
from .skia_geometry import SkPath
from .optimizer.symmetry_materialize import index_region_paths, resolve_source_path


def replace_working_region(
    regions: Sequence[VectorRegion], target_region_id: str,
    replacement: Callable[[VectorRegion], VectorRegion],
) -> tuple[VectorRegion, ...]:
    """Replace exactly one addressed working node without changing siblings."""
    matched = 0
    def visit(region: VectorRegion) -> VectorRegion:
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


def primitive_geometry_from_shape(shape: Shape) -> PrimitiveGeometry:
    """Decode a working primitive at the canonical boundary, without fitting."""
    params = shape.params
    if shape.kind == "circle":
        return PrimitiveGeometry(CirclePrimitive((params["cx"], params["cy"]), params["r"]))
    if shape.kind == "ellipse":
        return PrimitiveGeometry(EllipsePrimitive(
            (params["cx"], params["cy"]), params["rx"], params["ry"]
        ))
    if shape.kind == "rect":
        origin = (params["x"], params["y"])
        if "rx" in params or "ry" in params:
            return PrimitiveGeometry(RoundedRectanglePrimitive(
                origin, params["w"], params["h"],
                params.get("rx", params.get("ry", 0.0)),
                params.get("ry", params.get("rx", 0.0)),
            ))
        return PrimitiveGeometry(RectanglePrimitive(origin, params["w"], params["h"]))
    if shape.kind == "polygon":
        return PrimitiveGeometry(PolygonPrimitive(tuple(tuple(point) for point in params["points"])))
    raise TypeError(f"automatic primitive intent requires a typed primitive, got {shape.kind!r}")


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


def publish_primitive_geometry(
    target: VectorRegion, geometry: PrimitiveGeometry, *,
    diagnostics: Mapping[str, object] | None = None,
) -> VectorRegion:
    """Publish already-selected geometry while retaining source and paint evidence."""
    from .optimizer.vector_region import to_polygon
    shape = shape_from_primitive_geometry(geometry)
    return target.with_current(shape, footprint=to_polygon(shape), diagnostics=diagnostics)


def publish_clone_assignment(
    source: VectorRegion, target: VectorRegion, *,
    matrix: tuple[float, float, float, float, float, float], symbolic: bool,
    family_size: int | None = None, transformed: SkPath | None = None,
) -> VectorRegion:
    """Publish a retained clone choice; selection and detection stay outside."""
    if source.current is None:
        raise ValueError("clone source has no geometry")
    diagnostics: dict[str, object] = {"accepted": True, "matched_source": int(source.id),
        "fill_preserved": type(target.fill).__name__, "transform": matrix}
    if family_size is not None:
        diagnostics["family_size"] = family_size
    shape = (Shape("use", {"href_obj_id": source.id, "transform": matrix}) if symbolic
             else transform_working_shape(source.current, matrix))
    return target.with_current(shape,
        footprint=transformed if transformed is not None else source.footprint.affine_transform(matrix),
        diagnostics={"clones": diagnostics})


def publish_mirror_pair(
    source: VectorRegion, target: VectorRegion, *, axis: SymmetryAxisInput,
    matrix: tuple[float, float, float, float, float, float],
) -> tuple[VectorRegion, VectorRegion]:
    """Publish a selected mirror without reinterpreting its axis or paint."""
    if source.current is None:
        raise ValueError("mirror source has no geometry")
    return source, target.with_current(transform_working_shape(source.current, matrix),
        footprint=source.footprint.affine_transform(matrix), diagnostics={"symmetry": {
            "accepted": True, "mode": "pair", "matched_source": int(source.id),
            "axis": axis.model_dump(mode="json"),
        }})


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


def restore_corner_annotations(
    regions: Sequence[VectorRegion], *, target_region_id: str,
    annotated: PathGeometry | PathGeometryWithCutouts | CompoundPathGeometry,
) -> tuple[VectorRegion, ...]:
    """Carry anchors across exact downstream topology without rewriting SVG values."""
    from .drawing_corner_actions import rebind_corner_annotations
    from .optimizer.vector_region import to_polygon

    def restore(target: VectorRegion) -> VectorRegion:
        if target.current is None or target.current.kind != "path":
            raise ValueError("corner downstream replay target is not a path")
        params = dict(target.current.params)
        parsed = geometry_from_svg_d(str(params.get("d", "")), region_id=target_region_id,
            fill_rule="evenodd" if params.get("fill_rule") == "evenodd" else "nonzero",
            path_ids=params.get("path_ids"), defer_overlaps=True)
        params["canonical_path_geometry"] = rebind_corner_annotations(parsed, annotated)
        shape = Shape("path", params)
        return target.with_current(shape, footprint=to_polygon(shape))

    return replace_working_region(regions, target_region_id, restore)


def apply_region_geometry(
    selected: DrawingRegion, regions: Sequence[VectorRegion], *,
    shape_metadata: Mapping[str, Mapping[str, object]] | None = None,
    ingress_prefit_corners: bool = False,
) -> tuple[VectorRegion, ...]:
    """Apply a canonical selection in the shared clone/topology transaction.

    Fresh numeric fits still need the existing pre-fit provenance translation;
    retained selections instead carry their exact canonical annotations.
    """
    from .optimizer.vector_region import to_polygon
    from .optimizer.symmetry_materialize import materialize_self_symmetry
    from .drawing_topology import normalize_retained_topology
    from .path_topology import component_paths
    metadata = shape_metadata or {}
    prepared = normalize_retained_topology(materialize_self_symmetry(regions))
    if selected.is_leaf:
        shape = shape_from_region_geometry(selected, metadata.get(selected.id, {}))
        if ingress_prefit_corners and shape.kind == 'path':
            paths = component_paths(selected.geometry)
            params = {key: value for key, value in shape.params.items() if key != 'canonical_path_geometry'}
            shape = Shape('path', {**params, 'path_ids': tuple(path.path.id for path in paths),
                                  'fill_rule': 'evenodd'})
    else:
        all_paths = tuple(path for leaf in selected.leaves() for path in component_paths(leaf.geometry))
        paths = {path.path.id: path for path in all_paths}
        topology = selected.diagnostics.get('topology', {})
        source_order = tuple(topology.get('source_path_ids', tuple(paths)))
        retired = set(topology.get('retired_path_ids', ()))
        if (len(paths) != len(all_paths) or len(set(source_order)) != len(source_order)
                or not set(paths).issubset(source_order)
                or not (set(source_order) - set(paths)).issubset(retired)
                or set(paths) & retired):
            raise ValueError('selected component path ownership is inconsistent')
        order = tuple(key for key in source_order if key in paths)
        shape = Shape('path', {'d': ' '.join(paths[key].to_svg_d() for key in order),
                               'path_ids': order, 'fill_rule': 'evenodd'})
    result = normalize_retained_topology(replace_working_region(prepared, selected.id,
        lambda region: region.with_current(shape, footprint=to_polygon(shape), diagnostics=selected.diagnostics)),
        component_path_orders={selected.id: source_order} if selected.is_branch else None)
    if selected.is_branch and not ingress_prefit_corners:
        for leaf in selected.leaves():
            shape = shape_from_region_geometry(leaf, metadata.get(leaf.id, {}))
            result = replace_working_region(result, leaf.id,
                lambda region, leaf=leaf, shape=shape: region.with_current(
                    shape, footprint=to_polygon(shape), diagnostics=leaf.diagnostics))
        result = replace_working_region(result, selected.id,
            lambda region: region.with_diagnostics(selected.diagnostics))
    return result


def working_region_evidence(regions: Sequence[VectorRegion], *, baseline: Sequence[VectorRegion] = ()) -> dict[str, dict[str, object]]:
    """Retain only boundary evidence needed to project canonical selections."""
    result = {}
    indexed = index_region_paths(baseline)
    prior = {address: indexed[path] for path, address in vector_region_address_index(baseline).items()}
    def visit(region: VectorRegion) -> None:
        coverage = region.coverage
        values = coverage.values if isinstance(coverage, CoverageField) else coverage
        previous = prior.get(region.drawing_id)
        borrow_raster = previous is not None and np.array_equal(previous.raster, region.raster)
        old_coverage = None if previous is None else previous.coverage
        old_values = old_coverage.values if isinstance(old_coverage, CoverageField) else old_coverage
        borrow_coverage = previous is not None and type(old_coverage) is type(coverage) and (
            values is None or (np.array_equal(values, old_values) and (
                not isinstance(coverage, CoverageField) or (coverage.origin == old_coverage.origin
                    and coverage.canvas_shape == old_coverage.canvas_shape))))
        result[region.drawing_id] = {
            'id': region.id, 'raster_shape': region.raster.shape,
            'raster': None if borrow_raster else region.raster.tobytes(),
            'raster_source': region.drawing_id if borrow_raster else None,
            'source_label': region.source_label, 'color_hex': region.color_hex,
            'original_kind': None if region.original is None else region.original.kind,
            'original_params': {} if region.original is None else region.original.params,
            'coverage': None if values is None or borrow_coverage else values.tobytes(),
            'coverage_source': region.drawing_id if borrow_coverage else None,
            'coverage_shape': None if values is None else values.shape,
            'coverage_dtype': None if values is None else values.dtype.str,
            'coverage_origin': coverage.origin if isinstance(coverage, CoverageField) else None,
            'coverage_canvas': coverage.canvas_shape if isinstance(coverage, CoverageField) else None,
        }
        for child in region.children:
            visit(child)
    for root in regions:
        visit(root)
    return result


def restore_selected_roots(
    selected: Sequence[DrawingRegion], root_order: Sequence[str], regions: Sequence[VectorRegion], *,
    shape_metadata: Mapping[str, Mapping[str, object]], region_evidence: Mapping[str, Mapping[str, object]],
) -> tuple[VectorRegion, ...]:
    """Project an already selected root delta without topology/fitting decisions."""
    from .geometry_native import region_path, affine_tuple
    from .optimizer.symmetry_materialize import relative_source_path
    from .drawing_fill import serialize_fill
    from .skia_geometry import unary_union
    existing = {root.drawing_id: root for root in regions}
    changed = {root.id: root for root in selected}
    canonical = drawing_from_vector_regions('selected-input', width=1, height=1, regions=regions)
    all_roots = tuple(canonical.region(identity) if identity not in changed else changed[identity]
        for identity in root_order)
    previous = index_region_paths(regions)
    addresses = vector_region_address_index(regions)
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
    def convert(region: DrawingRegion) -> VectorRegion:
        info = region_evidence[region.id]
        baseline = prior.get(region.id)
        if region.is_branch:
            children = tuple(convert(child) for child in region.children)
            shape = None
            footprint = unary_union([child.footprint for child in children])
        else:
            children = ()
            if isinstance(region.geometry, CloneGeometry):
                geometry = region.geometry
                if len(geometry.transforms) == 1:
                    source_path = native_paths[geometry.source_region_id]
                    shape = Shape('use', {**shape_metadata.get(region.id, {}), 'href_obj_id': source_path[-1],
                        'href_region_path': relative_source_path(native_paths[region.id], source_path),
                        'transform': affine_tuple(geometry.transforms[0])})
                else:
                    shape = Shape('path', {**shape_metadata.get(region.id, {}),
                        'd': region_path(region, by_id).to_svg_d(), 'canonical_clone_geometry': geometry})
            else:
                shape = shape_from_region_geometry(region, shape_metadata.get(region.id, {}))
            footprint = region_path(region, by_id)
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
        return VectorRegion(int(info['id']), shape, fill,
            region.z, footprint=footprint, raster=raster,
            original=original, children=children, source_label=info['source_label'], color_hex=info['color_hex'],
            drawing_id=region.id, source_regions=region.source_region_ids,
            coverage=coverage, diagnostics=region.diagnostics)
    return tuple(convert(changed[identity]) if identity in changed else existing[identity] for identity in root_order)


def publish_region_update(selected: DrawingRegion, regions: Sequence[VectorRegion], *,
    shape_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[VectorRegion, ...]:
    """Project a canonical region update without repeating its geometry decision."""
    from .drawing_relationships import replace_region
    current = drawing_from_vector_regions('relationship', width=1, height=1, regions=regions)
    updated = replace_region(current, selected)
    evidence = working_region_evidence(regions, baseline=regions)
    def inherit(region: DrawingRegion, parent: DrawingRegion | None = None) -> None:
        if region.id not in evidence:
            evidence[region.id] = {**evidence[parent.id], 'id': int(region.id.rsplit('-', 1)[-1])}
        for child in region.children:
            inherit(child, region)
    for root in updated.regions:
        inherit(root)
    changed = tuple(after for before, after in zip(current.regions, updated.regions) if before is not after)
    return restore_selected_roots(changed, tuple(root.id for root in updated.regions), regions,
        shape_metadata=shape_metadata or {}, region_evidence=evidence)


def vector_region_address_index(
    regions: Sequence[VectorRegion],
) -> dict[tuple[int, ...], str]:
    """Return the public region ID owned by every engine-tree ancestry."""
    addresses: dict[tuple[int, ...], str] = {}

    def index(
        region: VectorRegion,
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


def drawing_from_vector_regions(
    drawing_id: str,
    *,
    width: float,
    height: float,
    regions: Sequence[VectorRegion],
) -> Drawing:
    """Project Python-engine working regions into the canonical scene model."""
    region_addresses = vector_region_address_index(regions)
    reference_index = index_region_paths(regions)

    def clone_geometry(region: VectorRegion, path: tuple[int, ...]) -> CloneGeometry:
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

    def convert(region: VectorRegion, root_id: str, legacy_id: str, path: tuple[int, ...]) -> DrawingRegion:
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
        ):
            geometry, diagnostics["corners"] = _ingress_corners(
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


def _public_region_id(region: VectorRegion, root_id: str, legacy_id: str) -> str:
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
            if canonical.to_svg_d() != str(params["d"]):
                raise ValueError("canonical path replay geometry does not match path data")
            return canonical
        fill_rule = "evenodd" if params.get("fill_rule") == "evenodd" else "nonzero"
        return geometry_from_svg_d(str(params["d"]), region_id=region_id, fill_rule=fill_rule, path_ids=params.get("path_ids"), defer_overlaps=True)
    if kind in {"circle", "ellipse", "rect", "polygon"}:
        return primitive_geometry_from_shape(Shape(kind, dict(params)))
    if kind == "use":
        source_id = public_ids[int(params["href_obj_id"])]
        values = tuple(float(value) for value in params.get("transform", (1, 0, 0, 1, 0, 0)))
        return CloneGeometry(source_id, (AffineTransform(*values),))
    raise ValueError(f"unsupported Python-engine shape kind: {kind!r}")


def _ingress_corners(geometry, d: str, region_id: str, evidence: object, *, path_ids: Sequence[str] | None = None):
    """Attach only matching, pre-fit-declared corners to canonical subpaths."""
    if not isinstance(evidence, Mapping) or evidence.get("path") != d:
        diagnostics = dict(evidence) if isinstance(evidence, Mapping) else {"path": d, "spans": []}
        prior = diagnostics.get("skipped", ())
        diagnostics["skipped"] = [
            *(dict(item) for item in prior if isinstance(item, Mapping)),
            {"reason": "missing_or_stale_provenance"},
        ]
        return geometry, diagnostics

    evidence = source_ordered_corner_diagnostics(d, evidence)
    rebound = rebind_prefit_corner_diagnostics(d, evidence, tolerance=0.02)
    prior = evidence.get("skipped", ())
    rebound["skipped"] = [
        *(dict(item) for item in prior if isinstance(item, Mapping)),
        *(dict(item) for item in rebound.get("skipped", ()) if isinstance(item, Mapping)),
    ]
    source_paths = parse_svg_paths(d, region_id=region_id, path_ids=path_ids)
    from .path_topology import component_paths, replace_component_paths

    by_id = {path.path.id: path for path in component_paths(geometry)}
    paths = [by_id[path.path.id] for path in source_paths]
    bound_paths, skipped, _feature_ids = corner_provenance.bind_prefit_corner_anchors(
        dict(enumerate(paths)), rebound["spans"], overlap_policy="retain_first",
    )
    rebound["skipped"].extend(dict(item) for item in skipped)
    converted = [bound_paths[index] for index in range(len(paths))]
    if isinstance(geometry, PathGeometry):
        return converted[0], rebound
    by_id = {path.path.id: path for path in converted}
    return replace_component_paths(geometry, tuple(by_id[path.path.id] for path in component_paths(geometry))), rebound


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


def publish_clone_refresh(
    clone: VectorRegion, source: VectorRegion, *,
    clone_path: tuple[int, ...], source_path: tuple[int, ...],
    previous_source_id: int, previous_source_path: tuple[int, ...],
    refresh_footprint: bool = True,
) -> VectorRegion:
    """Publish an exact scoped clone rebinding without rediscovering its source."""
    from .optimizer.vector_region import to_polygon
    from .optimizer.symmetry_materialize import relative_source_path

    if clone.current is None or clone.current.kind != "use" or source.current is None:
        raise ValueError("dependent clone refresh requires bound clone and source geometry")
    params = {**clone.current.params, "href_obj_id": int(source.id)}
    if source_path != (*clone_path[:-1], int(source.id)) or "href_region_path" in params:
        params["href_region_path"] = relative_source_path(clone_path, source_path)
    diagnostics: dict[str, object] = {"clone_reference_remap": {
        "previous_source_region_id": previous_source_id,
        "source_region_id": int(source.id),
        "previous_source_region_path": previous_source_path,
        "source_region_path": source_path,
        "reason": "self_symmetry_materialization",
    }}
    for key, field in (("clone", "source_id"), ("clones", "matched_source")):
        relation = clone.diagnostics.get(key)
        if isinstance(relation, Mapping):
            diagnostics[key] = {**relation, field: int(source.id)}
    footprint = (to_polygon(transform_working_shape(source.current, tuple(clone.current.params["transform"])))
                 if refresh_footprint else clone.footprint)
    return clone.with_current(Shape("use", params), footprint=footprint, diagnostics=diagnostics)


def refresh_dependent_clones(
    action: RefreshDependentClonesOperation, resolver: RegionResolver,
) -> list[VectorRegion]:
    """Refresh the recorded clones and publish their containing roots atomically."""
    replacements_by_path: dict[tuple[int, ...], VectorRegion] = {}
    for refresh in action.clones:
        clone = resolver.resolve(refresh.clone)
        source = resolver.resolve(refresh.source)
        clone_path = tuple(refresh.clone.ancestry)
        replacements_by_path[clone_path] = publish_clone_refresh(
            clone, source, clone_path=clone_path, source_path=tuple(refresh.source.ancestry),
            previous_source_id=refresh.previous_source_id,
            previous_source_path=refresh.previous_source_path,
        )

    def replace_paths(region: VectorRegion, path: tuple[int, ...]) -> VectorRegion:
        if path in replacements_by_path:
            return replacements_by_path[path]
        if region.is_leaf:
            return region
        children = tuple(replace_paths(child, (*path, int(child.id))) for child in region.children)
        return region.with_children(children) if children != region.children else region

    return [replace_paths(resolver.resolve(root), tuple(root.ancestry)) for root in action.roots]


def root_region_address(region: VectorRegion) -> str:
    if type(region.drawing_id) is not str:
        raise ValueError("drawing root is missing a stable drawing_id")
    return region.drawing_id


def with_drawing_address(region: VectorRegion, address: str) -> VectorRegion:
    """Copy *region* with its public, version-scoped drawing address."""
    if region.drawing_id == address:
        return region
    if region.is_leaf:
        assert region.current is not None
        return VectorRegion(
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
    return VectorRegion.branch(
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


def normalize_drawing_addresses(regions: Sequence[VectorRegion]) -> tuple[VectorRegion, ...]:
    """Expose compact local child paths while preserving internal region IDs.

    Optimizer-created child IDs are intentionally global (for example ``1001``)
    and can grow with every split.  They are implementation details.  The
    drawing protocol instead names a child by its one-based position in its
    retained parent: ``r1-1``, ``r1-2``, and so on.  Addresses are normalized
    only once a version is complete, so every plan still resolves against the
    exact version it names.
    """
    def visit(region: VectorRegion, address: str) -> VectorRegion:
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

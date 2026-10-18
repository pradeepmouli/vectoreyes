"""Apply drawing plans directly to cached :class:`VectorRegion` roots.

The interactive drawing state is the optimizer's native region forest.  SVG and
reports are derived artifacts, never a second editable scene representation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage

from ._fitcurve import fit_cubic_once, fit_quadratic_once
from .candidate import Fill, FlatFill, LinearGradientFill, RadialGradientFill, RasterFill
from .components import decompose_components
from .contour import region_contours
from .drawing_actions import ActionSettings, SetZOrderAction
from .drawing_plan import DrawingPlan, PlanValidationError, validate_plan
from .drawing_adapter import apply_region_geometry, drawing_from_vector_regions
from .drawing_fill import serialize_fill
from .drawing_trace import RootPathEvidence, TraceResult, svg_path_commands, _polyline_trace_path
from .path_fit_evidence import AutomaticFitEvidence
from .path_geometry import CompoundPathGeometry, CornerAnchor, PathGeometry, PathGeometryWithCutouts, parse_svg_paths
from .path_topology import component_paths, replace_component_paths
from .emit import apply_affine_point, render_svg_doc, resolve_fill, resolve_use_shape, shape_to_svg
from .fill_fit import fit_fill
from .fit import Shape, _curved_run_d, _fmt, fit_path_stages
from .optimizer.vector_region import VectorRegion, _parse_subpaths, _sample_subpath, to_polygon
from .optimizer.corners import path_corner_diagnostics, rebind_prefit_corner_diagnostics
from .optimizer.gate import rasterize
from .optimizer.automatic_replay import (
    AutomaticActionOutcome,
    ReplayBatch,
    optimize_with_intent,
    shape_from_primitive_geometry,
)
from .optimizer.passes import corner_normalize_pass, clones_pass, primitives_pass, seams_pass, simplify_pass, split_compound_pass, symmetry_pass
from .optimizer.shape_transform import bake_shape_transform
from .optimizer.symmetry_materialize import index_region_paths, materialize_self_symmetry, relative_source_path, remap_symmetry_clones, resolve_source_path, union_self_symmetry
from .optimizer.topology import normalize_retained_topology
from .pipeline import Options, _segment_image
from .refinement_settings import RefinementPasses
from .output_settings import DrawingOutputSettings
from .color import extract_palette, quantize
from .skia_geometry import SkPath, affinity, unary_union
from .quality import polyline_boundary_distance
from .surface_merge import material_groups, merge_material_regions
from .segment import fill_small_compatible_holes
from .types import Region
from . import drawing_region_actions


DrawingRegions = tuple[VectorRegion, ...]
DecompositionObserver = Callable[[str, tuple[Region, ...], tuple[Fill, ...] | None], None]


@dataclass(frozen=True)
class RootRegionsResult:
    """The one retained forest and the trace enriched by its actual root fits."""

    regions: DrawingRegions
    trace: TraceResult


@dataclass(frozen=True)
class RecordedRefinement:
    """Transient region result plus exact accepted-action replay records."""

    regions: DrawingRegions
    batches: tuple[ReplayBatch, ...]
    outcomes: tuple[AutomaticActionOutcome, ...]


def ordered_path_operation_stages(
    operations: Sequence[object],
) -> tuple[tuple[str, tuple[object, ...]], ...]:
    """Partition path edits without changing cleanup position or repetition."""
    stages: list[tuple[str, tuple[object, ...]]] = []
    direct: list[object] = []
    for operation in operations:
        name = (
            operation.get("op")
            if isinstance(operation, Mapping)
            else getattr(operation, "op", None)
        )
        if name in {"simplify", "normalize_corners"}:
            if direct:
                stages.append(("direct", tuple(direct)))
                direct = []
            stages.append((str(name), (operation,)))
        else:
            direct.append(operation)
    if direct:
        stages.append(("direct", tuple(direct)))
    return tuple(stages)


@dataclass(frozen=True)
class _RootAssembly:
    """Transient fit records keyed internally until public addresses are final."""

    regions: DrawingRegions
    evidence: Mapping[int, tuple[AutomaticFitEvidence, ...]]


@dataclass(frozen=True)
class _SurfaceFit:
    shape: Shape
    corners: dict[str, object]
    evidence: tuple[AutomaticFitEvidence, ...]


def _bind_root_evidence(trace: TraceResult, assembled: _RootAssembly) -> RootRegionsResult:
    regions = _normalize_drawing_addresses(assembled.regions)
    evidence = []
    for root in regions:
        for leaf in root.leaves():
            assert leaf.drawing_id is not None and leaf.current is not None
            paths = tuple(d.strip() for d in re.findall(r"M[^M]*", leaf.current.params["d"]))
            fits = assembled.evidence[leaf.id]
            if len(paths) != len(fits) or any(not fit.matches_output(d) for d, fit in zip(paths, fits)):
                raise ValueError("root fitting evidence does not match the retained leaf subpaths")
            evidence.extend(
                RootPathEvidence(leaf.drawing_id, f"{leaf.drawing_id}.p{index}", fitted)
                for index, fitted in enumerate(fits)
            )
    normalized = normalize_retained_topology(regions, defer_overlaps=True)
    owners = {
        path_id: leaf.drawing_id for root in normalized for leaf in root.leaves()
        for path_id in leaf.current.params.get("path_ids", ())
    }
    return RootRegionsResult(normalized, replace(trace, root_path_evidence=tuple(
        replace(record, region_id=owners.get(record.path_id, record.region_id)) for record in evidence
    )))


@dataclass(frozen=True)
class RenderedDrawing:
    """A transient SVG/report projection of one drawing-version's regions."""

    svg: str
    report: Mapping[str, object]


@dataclass(frozen=True)
class _Target:
    id: str
    root_id: str
    region: VectorRegion
    source_regions: frozenset[str]
    region_path: tuple[int, ...]


def root_regions(
    trace: TraceResult,
    rgb: np.ndarray | None = None,
    *,
    material_rgb: np.ndarray | None = None,
    min_region_fraction: float = 0.02,
    allow_gradients: bool = True,
    preserve_material_labels: bool | None = None,
    on_decomposition_stage: DecompositionObserver | None = None,
) -> RootRegionsResult:
    """Create retained roots, optionally reducing one-shot surfaces into roots.

    The raw trace deliberately preserves every palette fragment so path-command
    provenance is available on demand.  Its regions are therefore not the
    agent-facing roots.  When pixels are available, create those roots from the
    same raw regions that the trace exposed.  Material masks are formed from
    source-colour boundaries before their geometry and fills are fitted.
    ``preserve_material_labels`` keeps a geometry guide's explicit palette
    labels separate instead of treating similar colors as mergeable material.
    """
    if rgb is None:
        return _bind_root_evidence(trace, _raw_root_regions(trace))
    if not 0.0 <= min_region_fraction < 1.0:
        raise ValueError("min_region_fraction must be at least 0 and less than 1")
    if rgb.shape[:2] != (trace.height, trace.width):
        raise ValueError("material-merge RGB dimensions must match the trace canvas")
    if material_rgb is None and trace.material_pretrace is not None:
        material_rgb = trace.material_pretrace.material_proxy
    if preserve_material_labels is None:
        preserve_material_labels = trace.material_pretrace is not None
    segmentation_rgb = rgb if material_rgb is None else np.asarray(material_rgb, dtype=np.uint8)
    if segmentation_rgb.shape != rgb.shape:
        raise ValueError("material-trace RGB dimensions must match the source RGB")
    # Raw trace paths remain the plan's stable provenance.  Geometry roots use
    # the regular, AA-aware segmentation contract so a low-frequency AA shade
    # cannot punch a false notch out of a retained surface.  Each root records
    # the raw trace regions it covers.
    geometry_options = Options(
        epsilon=trace.options.simplify_tolerance,
        max_error=trace.options.curve_tolerance,
        cubic_paths=False,
        aa_contours=trace.options.trace_level == "subpixel",
        max_sampling_colors=trace.options.max_sampling_colors,
        # The relative material threshold belongs after boundary slicing and
        # compatible-material merging.  Keeping it out of seed segmentation
        # prevents small source fragments from disappearing before they can
        # become part of their intended material surface.
        min_region_fraction=0.0,
        min_region_size=trace.options.min_region_size,
    )
    _, _, seed_regions = _segment_image(segmentation_rgb, geometry_options)
    if trace.geometry_regions:
        # Background classification can reject a wholly transparent material
        # (for example the canvas color inside a native-alpha cutout). It does
        # not clip retained materials or set their subpixel contour positions.
        foreground = np.logical_or.reduce([region.mask for region in trace.geometry_regions])
        seed_regions = [region for region in seed_regions if np.any(region.mask & foreground)]
    min_material_area = min_region_fraction * max(
        (region.area for region in seed_regions),
        default=0,
    )
    # Geometry is derived from the material polylines themselves. The separate
    # foreground trace is diagnostic only: it must neither replace a material's
    # subpixel exterior nor clip it back to a binary foreground contour.
    surfaces: list[tuple[Region, tuple[Region, ...]]] = []
    for component in decompose_components(seed_regions, (trace.height, trace.width)):
        labelled = list(component)
        groups = [(index,) for index in range(len(labelled))] if preserve_material_labels else material_groups(labelled, rgb)
        for group in groups:
            members = tuple(labelled[index] for index in group)
            representative = max(members, key=lambda member: (member.area, -member.label))
            mask = np.logical_or.reduce([member.mask for member in members])
            surface = Region(representative.label, mask, representative.color_hex)
            if surface.area >= min_material_area:
                surfaces.append((surface, members))
    if on_decomposition_stage is not None:
        on_decomposition_stage("material_union", tuple(surface for surface, _members in surfaces), None)
    filled_surfaces = tuple(
        (surface, members, fit_fill(surface.mask, rgb, flat_hex=surface.color_hex, allow_gradients=allow_gradients))
        for surface, members in surfaces
    )
    if on_decomposition_stage is not None:
        on_decomposition_stage(
            "apply_fills",
            tuple(surface for surface, _members, _fill in filled_surfaces),
            tuple(fill for _surface, _members, fill in filled_surfaces),
        )

    assembled: list[tuple[Region, np.ndarray, _SurfaceFit, Fill, tuple[str, ...], dict[str, object]]] = []
    for surface, members, source_fill in filled_surfaces:
        cleaned_mask, filled_hole_area = fill_small_compatible_holes(
            surface.mask,
            rgb,
            max_area=trace.options.max_hole_area,
        )
        cleaned_surface = Region(
            surface.label,
            cleaned_mask,
            surface.color_hex,
            coverage=surface.coverage,
        )
        contours = _material_union_contours(
            members,
            added_pixels=cleaned_mask & ~surface.mask if filled_hole_area else None,
        )
        fitted_surface = _fit_contours_with_corners(contours, trace)
        if fitted_surface is None:
            continue
        corners = fitted_surface.corners
        source_regions = tuple(
            region.id for region in trace.regions if np.any(cleaned_mask & region.mask)
        )
        if source_regions:
            diagnostics: dict[str, object] = {
                "surface_merge": {
                    "sources": source_regions,
                    "merged": len(source_regions) > 1,
                      "geometry_source": "material_polyline_union",
                },
                "corners": corners,
            }
            if filled_hole_area:
                diagnostics["hole_cleanup"] = {
                    "filled_area": filled_hole_area,
                    "max_hole_area": trace.options.max_hole_area,
                }
            fill = fit_fill(cleaned_mask, rgb, flat_hex=surface.color_hex, allow_gradients=allow_gradients) if filled_hole_area else source_fill
            assembled.append((cleaned_surface, cleaned_mask, fitted_surface, fill, source_regions, diagnostics))
    trace_order = {region.id: index for index, region in enumerate(trace.regions)}
    # Geometry seeds can gain or lose a one-pixel AA fringe.  Ordering by that
    # area would make otherwise stable drawing IDs swap between equivalent
    # traces, so anchor each root to its earliest raw-trace provenance instead.
    ordered = sorted(
        assembled,
        key=lambda item: (
            min(trace_order[source] for source in item[4]),
            -item[1].sum(),
            item[0].label,
            item[0].color_hex,
        ),
    )
    return _bind_root_evidence(trace, _RootAssembly(tuple(
        VectorRegion.from_shape(
            id=index,
            shape=fitted.shape,
            fill=fill,
            z=index - 1,
            raster=mask,
            source_label=surface.label,
            color_hex=surface.color_hex,
            drawing_id=f"r{index}",
            source_regions=source_regions,
            diagnostics=diagnostics,
        )
        for index, (surface, mask, fitted, fill, source_regions, diagnostics) in enumerate(ordered, start=1)
    ), {index: item[2].evidence for index, item in enumerate(ordered, start=1)}))


def _complete_material_masks(
    baseline: np.ndarray,
    masks: list[np.ndarray],
) -> list[np.ndarray]:
    """Assign AA/unlabelled baseline pixels to their nearest material surface."""
    if not masks:
        return []
    owner = np.full(baseline.shape, -1, dtype=np.int32)
    for index, mask in enumerate(masks):
        owner[mask & baseline] = index
    uncovered = baseline & (owner < 0)
    if uncovered.any() and (owner >= 0).any():
        _distance, indices = ndimage.distance_transform_edt(owner < 0, return_indices=True)
        nearest = owner[tuple(indices[:, uncovered])]
        owner[uncovered] = nearest
    return [owner == index for index in range(len(masks))]


def _geometry_root_regions(
    trace: TraceResult,
    rgb: np.ndarray,
    geometry_options: Options,
    *,
    seed_regions: list[Region],
    min_material_area: float,
    allow_gradients: bool = True,
    preserve_material_labels: bool = False,
    on_decomposition_stage: DecompositionObserver | None = None,
) -> _RootAssembly:
    """Seed the editable region forest from trace-owned exterior geometry.

    Material segmentation only partitions each trace-owned seed using source
    seams: soft steps stay within one material while sharp steps remain an
    interior boundary.  The seed is provenance, not a permanent clip: later
    fitting and refinement may improve the exterior path.
    """
    roots: list[VectorRegion] = []
    evidence: dict[int, tuple[AutomaticFitEvidence, ...]] = {}
    assembled_materials: list[Region] = []
    unioned_materials: list[Region] = []
    applied_fills: list[tuple[Region, Fill]] = []
    for index, geometry in enumerate(trace.geometry_regions, start=1):
        root_sources = tuple(
            region.id for region in trace.regions if np.any(geometry.mask & region.mask)
        )
        if not root_sources:
            continue
        baseline_shape = Shape(
            "path",
            {"d": geometry.trace_path.d, "fill_rule": geometry.trace_path.fill_rule},
        )
        clipped_seeds: list[Region] = []
        for seed in seed_regions:
            mask = seed.mask & geometry.mask
            if mask.any():
                clipped_seeds.append(Region(seed.label, mask, seed.color_hex, coverage=seed.coverage))
        assembled_materials.extend(clipped_seeds)
        merged_materials = (
            clipped_seeds
            if preserve_material_labels
            else merge_material_regions(clipped_seeds, rgb)
        )
        materials = [
            material for material in merged_materials
            if material.area >= min_material_area
        ]
        unioned_materials.extend(materials)
        filled_materials = tuple(
            (member, fit_fill(member.mask, rgb, flat_hex=member.color_hex, allow_gradients=allow_gradients))
            for member in materials
        )
        applied_fills.extend(filled_materials)
        children: list[VectorRegion] = []
        child_evidence: dict[int, tuple[AutomaticFitEvidence, ...]] = {}
        for child_index, (member, source_fill) in enumerate(filled_materials, start=1):
            # Material masks define geometry.  Do not assign their unlabelled
            # AA fringe to a nearest neighbour before fitting: doing so moves
            # sharp counters and exterior contours according to palette
            # ownership rather than source boundary evidence.  Tiny enclosed
            # islands are safe to repair locally because they cannot bridge a
            # different material surface.
            mask, filled_hole_area = fill_small_compatible_holes(
                member.mask,
                rgb,
                max_area=trace.options.max_hole_area,
            )
            surface = Region(member.label, mask, member.color_hex)
            fitted_surface = _surface_shape_with_corners(surface, trace)
            if fitted_surface is None:
                continue
            shape, corners = fitted_surface.shape, fitted_surface.corners
            child_evidence[index * 1000 + child_index] = fitted_surface.evidence
            source_regions = tuple(
                region.id for region in trace.regions if np.any(mask & region.mask)
            )
            if not source_regions:
                source_regions = root_sources
            children.append(
                VectorRegion.from_shape(
                    id=index * 1000 + child_index,
                    shape=shape,
                fill=fit_fill(mask, rgb, flat_hex=member.color_hex, allow_gradients=allow_gradients) if filled_hole_area else source_fill,
                    z=child_index - 1,
                    # Alpha/background geometry only establishes this initial
                    # trace.  Subsequent optimizer passes receive a mask from
                    # the fitted vector path, so they are free to improve it.
                    raster=rasterize(to_polygon(shape), (trace.height, trace.width)),
                    source_label=member.label,
                    color_hex=member.color_hex,
                    source_regions=source_regions,
                    diagnostics={
                        "material_surface": {
                            "root": geometry.id,
                            "edge_de": 0.06,
                            "boundary_preserving": True,
                            "filled_hole_area": filled_hole_area,
                        },
                        "corners": corners,
                    },
                )
            )
        if not children:
            continue
        baseline_diagnostics = {
            "corners": {
                **geometry.trace_path.corner_diagnostics,
                "path": str(baseline_shape.params["d"]),
            },
            "geometry_seed": {
                "trace_region": geometry.id,
                "background": dict(trace.background),
                "fill": "material_children_seeded",
                "path": baseline_shape.params["d"],
                "fill_rule": baseline_shape.params["fill_rule"],
            }
        }
        # A single interior material is the baseline itself.  Keep it as an
        # ordinary ``rN`` leaf so existing plans can address simple drawings
        # without acquiring an artificial child suffix.
        if len(children) == 1:
            child = children[0]
            roots.append(
                VectorRegion.from_shape(
                    id=index,
                    shape=baseline_shape,
                    fill=child.fill,
                    z=index - 1,
                    raster=rasterize(to_polygon(baseline_shape), (trace.height, trace.width)),
                    footprint=to_polygon(baseline_shape),
                    source_label=child.source_label,
                    color_hex=child.color_hex,
                    drawing_id=f"r{index}",
                    source_regions=root_sources,
                    diagnostics={**child.diagnostics, **baseline_diagnostics},
                )
            )
            evidence[index] = geometry.trace_path.path_evidence
            continue
        evidence.update(child_evidence)
        roots.append(
                VectorRegion.branch(
                id=index,
                children=children,
                z=index - 1,
                footprint=to_polygon(baseline_shape),
                source_label=None,
                color_hex=None,
                drawing_id=f"r{index}",
                source_regions=root_sources,
                diagnostics=baseline_diagnostics,
            )
        )
    if on_decomposition_stage is not None:
        on_decomposition_stage("boundary_assemble_material", tuple(assembled_materials), None)
        on_decomposition_stage("material_union", tuple(unioned_materials), None)
        on_decomposition_stage(
            "apply_fills",
            tuple(member for member, _fill in applied_fills),
            tuple(fill for _member, fill in applied_fills),
        )
    return _RootAssembly(tuple(roots), evidence)


def _raw_root_regions(trace: TraceResult) -> _RootAssembly:
    roots = []
    evidence = {}
    for index, region in enumerate(trace.regions):
        fitted = _fit_contours_with_corners(region.contours, trace)
        if fitted is None:
            raise ValueError("raw root fitting requires retained source contours")
        # Preserve the no-RGB entry's explicit fill-rule serialization.
        fitted.shape.params["fill_rule"] = "evenodd" if len(fitted.evidence) > 1 else "nonzero"
        evidence[index + 1] = fitted.evidence
        roots.append(VectorRegion.from_shape(
            id=index + 1,
            shape=fitted.shape,
            fill=FlatFill(region.color),
            z=index,
            raster=region.mask,
            source_label=region.source_label,
            color_hex=region.color,
            drawing_id=region.id,
            source_regions=(region.id,),
            diagnostics={"corners": fitted.corners},
        ))
    return _RootAssembly(tuple(roots), evidence)


def _with_drawing_address(region: VectorRegion, address: str) -> VectorRegion:
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


def _normalize_drawing_addresses(regions: Sequence[VectorRegion]) -> DrawingRegions:
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
        return _with_drawing_address(region, address)

    return tuple(visit(region, _root_label(region)) for region in regions)


def _material_union_contours(
    members: Sequence[Region], *, added_pixels: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Union coverage-aware material polylines before any curve fitting.

    The material fields are sampled once. PathOps unions their polylines without
    a raster round trip, retaining every exterior and hole. Existing compatible
    hole cleanup may add interior pixels, but never replaces the source contour.
    """
    paths: list[SkPath] = []
    contour_groups = [
        tuple(region_contours(member.mask, coverage=member.coverage))
        for member in members
    ]
    if added_pixels is not None and added_pixels.any():
        contour_groups.append(tuple(region_contours(added_pixels)))
    for material_contours in contour_groups:
        if not material_contours:
            continue
        polyline = _polyline_trace_path(material_contours, region_id="material")
        paths.append(SkPath.from_svg_d(polyline.d, fill_rule="evenodd"))
    merged = unary_union(paths)
    contours: list[np.ndarray] = []
    for component in merged.geoms:
        for ring in (component.exterior, *component.interiors):
            points = np.asarray(ring.coords, dtype=float)
            # PathOps may retain an explicit closing vertex as well as the
            # closure marker. Remove exact duplicates, not geometric detail:
            # every source LineCommand must have a nonzero-length edge.
            if len(points):
                points = points[np.r_[True, np.any(np.diff(points, axis=0) != 0.0, axis=1)]]
            if len(points) >= 4:
                contours.append(points)
    return tuple(contours)


def _surface_shape_with_corners(
    surface: Region,
    trace: TraceResult,
) -> _SurfaceFit | None:
    return _fit_contours_with_corners(region_contours(surface.mask, coverage=surface.coverage), trace)


def _fit_contours_with_corners(contours: Sequence[np.ndarray], trace: TraceResult) -> _SurfaceFit | None:
    contours = [contour for contour in contours if len(contour) >= 3]
    if not contours:
        return None
    stages = [
        fit_path_stages(
            contour,
            epsilon=trace.options.simplify_tolerance,
            max_error=trace.options.curve_tolerance,
            cubic=False,
            progressive=trace.options.fit_strategy != "quadratic",
            progressive_allow_lines=trace.options.fit_strategy == "progressive_allow_lines",
            support_weighted_corners=trace.options.support_weighted_corners,
            corner_rounding=trace.options.corner_rounding,
            canvas_extent=float(max(trace.width, trace.height)),
            # Root fitting follows the trace curve family, but it must not
            # run a second command-reduction search over a mask that has just
            # been material-merged.  This establishes the merged boundary.
            prefer_simple_curves=False,
        )
        for contour in contours
    ]
    paths = [stage.atomic_d for stage in stages]
    params: dict[str, object] = {"d": " ".join(paths)}
    if len(paths) > 1:
        params["fill_rule"] = "evenodd"
    diagnostics = tuple(
        path_corner_diagnostics(
            path,
            declared_corners=stage.corner_points,
            declared_corner_spans=stage.corner_spans,
        )
        for path, stage in zip(paths, stages, strict=True)
    )
    corners: dict[str, object] = (
        dict(diagnostics[0])
        if len(diagnostics) == 1
        else {"subpaths": diagnostics}
    )
    corners["path"] = str(params["d"])
    return _SurfaceFit(Shape("path", params), corners, tuple(stage.baseline_evidence for stage in stages))


def _surface_shape(surface: Region, trace: TraceResult) -> Shape | None:
    """Return only geometry where callers do not retain a drawing region."""
    fitted = _surface_shape_with_corners(surface, trace)
    return fitted.shape if fitted is not None else None


def render_drawing(trace: TraceResult, regions: Sequence[VectorRegion]) -> RenderedDrawing:
    """Serialize retained geometry and paint; execution owns all reconstruction."""
    def validate(region: VectorRegion) -> None:
        symmetry = region.diagnostics.get("symmetry")
        if isinstance(symmetry, Mapping) and (
            (region.is_branch and symmetry.get("mode") == "self")
            or symmetry.get("mode") == "self_mirror"
        ):
            raise ValueError("self-symmetry must be materialized before rendering")
        for child in region.children:
            validate(child)
    for region in regions:
        validate(region)
    targets = _targets(regions)
    ordered = sorted(targets.values(), key=lambda target: (target.region.z, target.region.id))
    reference_index = index_region_paths(regions)
    targets_by_path = {target.region_path: target for target in ordered}
    resolved_sources: dict[str, str] = {}
    defs: list[str] = []
    # A clone shares geometry, not paint.  An SVG ``<use fill=...>`` cannot
    # override a ``fill`` attribute on the referenced source element, so the
    # reusable geometry lives in an unpainted definition.  The source and each
    # clone are then independently painted instances with stable target IDs.
    clone_definition_ids: dict[str, str] = {}
    clone_sources: dict[str, _Target] = {}
    targets_by_id = {target.id: target for target in ordered}
    for target in ordered:
        current = target.region.current
        if current is None or current.kind != "use":
            continue
        source_path = resolve_source_path(reference_index, target.region_path, current.params)
        if source_path not in targets_by_path:
            raise ValueError("clone source is not a retained target")
        source_target_id = targets_by_path[source_path].id
        resolved_sources[target.id] = source_target_id
        clone_diagnostics = target.region.diagnostics.get("clones")
        is_clone = isinstance(clone_diagnostics, Mapping) or isinstance(target.region.diagnostics.get("clone"), Mapping)
        if not is_clone:
            continue
        source_target = targets_by_id[source_target_id]
        source_shape = source_target.region.current
        if source_shape is None or source_shape.kind == "use":
            raise ValueError("clone source must be retained concrete geometry")
        clone_sources[source_target_id] = source_target
        clone_definition_ids[source_target_id] = f"{source_target_id}-geometry"

    for source_target_id, source_target in clone_sources.items():
        source_shape = source_target.region.current
        assert source_shape is not None
        defs.append(shape_to_svg(source_shape, None, clone_definition_ids[source_target_id]))

    body: list[str] = []
    report_targets: list[dict[str, object]] = []
    for target in ordered:
        assert target.region.current is not None
        current = target.region.current
        id_map = {current.params["href_obj_id"]: resolved_sources[target.id]} if current.kind == "use" else {}
        shape = resolve_use_shape(current, id_map)
        fill = target.region.fill
        if fill is None:
            raise ValueError(f"drawing region {target.id!r} has no fill")
        resolved_fill = resolve_fill(fill, defs)
        if target.id in clone_definition_ids:
            shape = Shape(
                "use",
                {"href": clone_definition_ids[target.id], "transform": (1, 0, 0, 1, 0, 0)},
            )
        elif shape.kind == "use":
            clone_diagnostics = target.region.diagnostics.get("clones")
            is_clone = isinstance(clone_diagnostics, Mapping) or isinstance(target.region.diagnostics.get("clone"), Mapping)
            if is_clone:
                source_target_id = resolved_sources[target.id]
                definition_id = clone_definition_ids.get(source_target_id)
                if definition_id is not None:
                    params = dict(shape.params)
                    params["href"] = definition_id
                    shape = Shape("use", params)
        body.append(shape_to_svg(shape, resolved_fill, target.id))
        report_targets.append(
            {
                "id": target.id,
                "root_id": target.root_id,
                "source_regions": tuple(sorted(target.source_regions)),
                "geometry": shape.kind,
                "fill": type(fill).__name__,
                "z": target.region.z,
                "diagnostics": _public_diagnostics(target.region.diagnostics),
            }
        )
    return RenderedDrawing(
        render_svg_doc(trace.width, trace.height, body, defs),
        {"targets": tuple(report_targets), "groups": _drawing_groups(regions)},
    )


def drawing_summary(trace: TraceResult, regions: Sequence[VectorRegion]) -> dict[str, object]:
    """Return the compact, merged-root view agents receive after interactive trace."""
    targets = _targets(regions)
    ordered = sorted(targets.values(), key=lambda target: (target.region.z, target.region.id))
    summaries: list[dict[str, object]] = []
    for target in ordered:
        assert target.region.current is not None
        shape = target.region.current
        geometry: dict[str, object] = {"type": shape.kind}
        if shape.kind == "path":
            geometry["d"] = shape.params["d"]
            if shape.params.get("fill_rule"):
                geometry["fill_rule"] = shape.params["fill_rule"]
            commands = [
                {
                    "id": command.id,
                    "command": command.command,
                    "values": list(command.values),
                    "anchor_kind": command.anchor_kind,
                    "corner_id": command.corner_id,
                }
                for command in svg_path_commands(
                    str(shape.params["d"]),
                    target.id,
                    path_ids=shape.params.get("path_ids"),
                    corner_diagnostics=(
                        target.region.diagnostics.get("corners")
                        if isinstance(target.region.diagnostics.get("corners"), Mapping)
                        and target.region.diagnostics["corners"].get("path") == str(shape.params["d"])
                        else None
                    ),
                )
            ]
            geometry["commands"] = commands
        summaries.append(
            {
                "id": target.id,
                "root_id": target.root_id,
                "source_regions": tuple(sorted(target.source_regions)),
                "geometry": geometry,
                "fill": _public_fill(target.region.fill),
            }
        )
    return {"width": trace.width, "height": trace.height, "options": asdict(trace.options), "regions": summaries,
            "groups": _drawing_groups(regions)}


def labeled_drawing_svg(trace: TraceResult, regions: Sequence[VectorRegion]) -> str:
    """Render the public, plan-addressable region map for a drawing version."""
    from .drawing_render import render_labeled_drawing

    return render_labeled_drawing(drawing_from_vector_regions(
        "region-map", width=trace.width, height=trace.height, regions=regions,
    ))


def stitch_regions(trace: TraceResult, regions: Sequence[VectorRegion]) -> DrawingRegions:
    """Normalize shared root boundaries before retaining interactive ``v0``.

    Initial trace stitching may snap existing shared geometry, but it must not
    construct an underlap beneath a neighbouring region. More invasive seam
    reconstruction remains available to explicit and automatic refinement.
    """
    return stitch_regions_with_intent(trace, regions).regions


def stitch_regions_with_intent(
    trace: TraceResult, regions: Sequence[VectorRegion], *, enabled: bool = True
) -> RecordedRefinement:
    """Normalize initial seams and retain the accepted concrete constraints."""
    if not enabled:
        return RecordedRefinement(_normalize_drawing_addresses(tuple(regions)), (), ())
    recorded = _run_detection_recorded(
            regions,
            trace,
            "stitch",
            None,
            epsilon=trace.options.simplify_tolerance,
            max_error=trace.options.curve_tolerance,
            allow_underlap=False,
    )
    return replace(recorded, regions=_normalize_drawing_addresses(recorded.regions))


def auto_refine(
    trace: TraceResult,
    regions: Sequence[VectorRegion],
    *,
    rgb: np.ndarray | None = None,
    corner_normalize: bool = False,
    refinement_passes: RefinementPasses | None = None,
    on_pass: Callable[[str, DrawingRegions], None] | None = None,
) -> DrawingRegions:
    """Run the one-shot optimizer over already-traced drawing roots.

    ``trace_drawing(refine="auto")`` must not enter the optimizer's alternate
    raster-to-region path.  The trace artifact is the drawing's source of truth:
    it owns stable region and command IDs, and every auto-refined version must
    remain a descendant of those roots.
    """
    return auto_refine_with_intent(
        trace,
        regions,
        rgb=rgb,
        corner_normalize=corner_normalize,
        refinement_passes=refinement_passes,
        on_pass=on_pass,
    ).regions


def auto_refine_with_intent(
    trace: TraceResult,
    regions: Sequence[VectorRegion],
    *,
    rgb: np.ndarray | None = None,
    allow_gradients: bool = True,
    corner_normalize: bool = False,
    refinement_passes: RefinementPasses | None = None,
    on_pass: Callable[[str, DrawingRegions], None] | None = None,
) -> RecordedRefinement:
    """Run automatic refinement and retain its exact committed operations."""
    # Import locally: pipeline owns the pass ordering, while drawing_refine owns
    # the persistent VectorRegion forest used by MCP drawing versions.
    from .pipeline import _optimizer_passes

    options = Options(
        epsilon=trace.options.simplify_tolerance,
        max_error=trace.options.curve_tolerance,
        cubic_paths=False,
        aa_contours=trace.options.trace_level == "subpixel",
        max_sampling_colors=trace.options.max_sampling_colors,
        min_region_size=trace.options.min_region_size,
        optimizer=True,
        corner_normalize=corner_normalize,
        refinement_passes=refinement_passes or RefinementPasses(),
    )
    if rgb is not None and rgb.shape[:2] != (trace.height, trace.width):
        raise ValueError("fill RGB dimensions must match the trace canvas")
    regions = normalize_retained_topology(materialize_self_symmetry(regions))
    before = {region.id: region for region in regions}
    masks = {region.id: np.asarray(region.raster, dtype=bool) for region in regions}
    def _capture(pass_name: str, current: list[VectorRegion]) -> None:
        if on_pass is not None:
            on_pass(
                pass_name,
                normalize_retained_topology(_normalize_drawing_addresses(tuple(
                    _restore_root_metadata(region, before.get(region.id)) for region in current
                ))),
            )

    optimized = optimize_with_intent(
        list(regions), masks, _optimizer_passes(options), on_pass=_capture
    )
    result = normalize_retained_topology(_normalize_drawing_addresses(
        tuple(
            _restore_root_metadata(region, before.get(region.id))
            for region in optimized.objects
        )
    ))
    if rgb is None:
        return RecordedRefinement(result, optimized.batches, optimized.outcomes)
    # Automatic refinement is a geometry pipeline.  Re-fit every final leaf
    # from the source only after the full pass sequence has established its
    # footprint, rather than retaining palette-fragment fills merely because a
    # particular leaf's geometry key happened not to change.
    result = refill_all_regions_from_source(
        result,
        rgb,
        allow_gradients=allow_gradients,
        # Gradient pretrace has already converted quantization bands into
        # intentional material supports. Sibling leaves may therefore be
        # different gradients that merely share one exterior geometry root.
        share_root_gradients=trace.material_pretrace is None,
    )
    return RecordedRefinement(result, optimized.batches, optimized.outcomes)


def refine(
    trace: TraceResult,
    base: Sequence[VectorRegion],
    plan: DrawingPlan,
    *,
    rgb: np.ndarray | None = None,
    fit_sources: Mapping[str, np.ndarray] | None = None,
) -> DrawingRegions:
    """Execute *plan* as immutable transformations over native vector regions."""
    if rgb is not None and rgb.shape[:2] != (trace.height, trace.width):
        raise ValueError("fill RGB dimensions must match the trace canvas")
    regions = normalize_retained_topology(materialize_self_symmetry(base))
    drawing = drawing_from_vector_regions(
        plan.drawing_id, width=trace.width, height=trace.height, regions=regions,
    )
    validate_plan(plan, drawing)
    z_order: SetZOrderAction | None = None

    for index, action in enumerate(plan.ops):
        before = regions
        op = action.model_dump(exclude_none=True)
        name = action.op
        settings = _action_settings(trace, plan, op)
        epsilon, max_error = settings.epsilon, settings.max_error
        if index and name in {"fit_path", "fit_primitive", "set_fill"}:
            # Temporary adapter: later actions must see earlier plan edits,
            # while the first action reuses the already validated projection.
            drawing = drawing_from_vector_regions(
                plan.drawing_id, width=trace.width, height=trace.height, regions=regions,
            )
        if name == "merge":
            regions = _group_regions(regions, op)
        elif name == "split":
            try:
                regions = _split_region(regions, op)
            except ValueError as error:
                raise PlanValidationError(f"/ops/{index}/divider", str(error)) from error
        elif name in {"detect_primitives", "detect_symmetry", "detect_clones"}:
            regions = _run_detection(regions, trace, name, op.get("target"), epsilon=epsilon, max_error=max_error)
        elif name == "stitch":
            regions = _run_detection(
                regions,
                trace,
                name,
                op.get("targets"),
                epsilon=epsilon,
                max_error=max_error,
            )
        elif name == "set_symmetry":
            regions = _set_symmetry(regions, op)
        elif name == "assign_clone":
            regions = _clone(regions, op)
        elif name == "align":
            regions = _align(regions, op)
        elif name == "fit_path":
            targets = _targets(regions)
            target = targets[op["target"]]
            try:
                updated = drawing_region_actions.fit_region_path(
                    drawing.region(target.id), action, raster=target.region.raster, settings=settings,
                )
                regions = apply_region_geometry(updated, regions, ingress_prefit_corners=True)
            except ValueError as error:
                raise PlanValidationError(f"/ops/{index}/target", str(error)) from error
        elif name == "fit_primitive":
            targets = _targets(regions)
            target = targets[op["target"]]
            try:
                updated = drawing_region_actions.fit_region_primitive(
                    drawing.region(target.id), action,
                    raster=target.region.raster, settings=settings,
                )
                shape = shape_from_primitive_geometry(updated.geometry)
            except ValueError as error:
                raise PlanValidationError(f"/ops/{index}/target", str(error)) from error
            regions = _replace_leaf(
                regions,
                target.id,
                lambda region: region.with_current(
                    shape,
                    footprint=to_polygon(shape),
                    diagnostics={"geometry": updated.diagnostics["geometry"]},
                ),
            )
        elif name == "update_geometry":
            geometry_ops = op["ops"]
            target_id = op["target"]
            for stage_name, stage_ops in ordered_path_operation_stages(geometry_ops):
                if stage_name != "direct":
                    regions = _run_detection(
                        regions,
                        trace,
                        stage_name,
                        target_id,
                        epsilon=epsilon,
                        max_error=max_error,
                    )
                    continue
                targets = _targets(regions)
                target = targets[target_id]
                geometry = {"type": "path", "ops": stage_ops}
                try:
                    shape = _path_shape(
                        trace,
                        target,
                        geometry,
                        references=_segment_references(targets),
                        epsilon=epsilon,
                        max_error=max_error,
                        fit_sources=fit_sources,
                    )
                except ValueError as error:
                    raise PlanValidationError(f"/ops/{index}/ops", str(error)) from error
                regions = _replace_leaf(
                    regions,
                    target.id,
                    lambda region: region.with_current(
                        shape,
                        footprint=to_polygon(shape),
                        diagnostics={
                            "geometry": {"explicit": "path"},
                            "fit_acceptance": shape.params.get("fit_acceptance", {}),
                        },
                    ),
                )
        elif name == "set_fill":
            target_id = op["target"]
            updated = drawing_region_actions.set_region_fill(drawing.region(target_id), action)
            # Public fill assignments have no affine overrides.
            # This conversion remains only until working-region paint migrates.
            fill = _fill(serialize_fill(updated.fill))
            regions = _replace_leaf(regions, target_id, lambda region: _with_fill(region, fill))
        elif name == "set_z_order":
            z_order = action

        # Geometry is authoritative.  A changed leaf must not retain a fill
        # fitted to its earlier mask; clear it, then run the one shared fill
        # pass against its new vector footprint.  An explicit ``set_fill`` is
        # intentionally excluded because it is already the caller's final
        # fill decision for that operation.
        if rgb is not None and name not in {"set_fill", "set_z_order"}:
            regions = refill_changed_regions_from_source(before, regions, rgb)
        regions = normalize_retained_topology(regions)

    if z_order is not None:
        drawing = drawing_from_vector_regions(
            plan.drawing_id, width=trace.width, height=trace.height, regions=regions,
        )
        ordered = drawing_region_actions.set_drawing_z_order(drawing, z_order)
        for root in ordered.regions:
            for leaf in root.leaves():
                regions = _replace_leaf(regions, leaf.id, lambda region, z=leaf.z: _with_z(region, z))
    return normalize_retained_topology(_normalize_drawing_addresses(regions))


def _run_detection(
    regions: Sequence[VectorRegion],
    trace: TraceResult,
    operation: str,
    target_id: object | None,
    *,
    epsilon: float,
    max_error: float,
    allow_underlap: bool = True,
) -> DrawingRegions:
    """Run one existing optimizer pass over the cached region forest."""
    return run_detection_with_intent(
        regions,
        trace,
        operation,
        target_id,
        epsilon=epsilon,
        max_error=max_error,
        allow_underlap=allow_underlap,
    ).regions


def run_detection_with_intent(
    regions: Sequence[VectorRegion],
    trace: TraceResult,
    operation: str,
    target_id: object | None,
    *,
    epsilon: float,
    max_error: float,
    allow_underlap: bool = True,
) -> RecordedRefinement:
    """Run one producer and return only its typed accepted intent for replay."""
    regions = normalize_retained_topology(materialize_self_symmetry(regions))
    cubic = False
    if operation == "detect_primitives":
        def pass_fn(objects, masks):
            return primitives_pass(objects, masks, epsilon=epsilon)
    elif operation == "detect_symmetry":
        def pass_fn(objects, masks):
            return symmetry_pass(objects, masks, epsilon=epsilon, max_error=max_error, cubic=cubic)
    elif operation == "detect_clones":
        def pass_fn(objects, masks):
            return clones_pass(objects, masks, symbolic=True)
    elif operation == "split":
        pass_fn = split_compound_pass
    elif operation == "simplify":
        def pass_fn(objects, masks):
            return simplify_pass(
                objects,
                masks,
                epsilon=epsilon,
                max_error=max_error,
                cubic=cubic,
                max_boundary_error=max_error,
            )
    elif operation == "stitch":
        def pass_fn(objects, masks):
            return seams_pass(
                objects,
                masks,
                epsilon=epsilon,
                max_error=max_error,
                cubic=cubic,
                allow_underlap=allow_underlap,
            )
    elif operation == "normalize_corners":
        def pass_fn(objects, masks):
            return corner_normalize_pass(objects, masks, max_error=max_error, enabled=True)
    else:
        raise ValueError(f"unknown detection operation: {operation}")

    selected_ids = (
        (target_id,)
        if type(target_id) is str
        else tuple(target_id)
        if isinstance(target_id, Sequence) and not isinstance(target_id, (str, bytes))
        else ()
    )
    if selected_ids:
        selected_region_ids = {_targets(regions)[selected_id].region.id for selected_id in selected_ids}
        original_pass = pass_fn

        def pass_fn(objects, masks):  # type: ignore[no-redef]
            proposals = original_pass(objects, masks)
            if operation != "detect_clones":
                return [proposal for proposal in proposals if selected_region_ids.intersection(proposal.obj_ids)]
            return [
                proposal
                for proposal in proposals
                if selected_region_ids.intersection(proposal.obj_ids)
                or any(
                    replacement.diagnostics.get("clones", {}).get("matched_source") in selected_region_ids
                    for replacement in proposal.new_objects
                )
            ]

    before = {region.id: region for region in regions}
    masks = {region.id: np.asarray(region.raster, dtype=bool) for region in regions}
    optimized = optimize_with_intent(list(regions), masks, [pass_fn])
    restored = tuple(
        _restore_root_metadata(region, before.get(region.id))
        for region in optimized.objects
    )
    result = normalize_retained_topology(
        _rebind_corner_provenance(
            regions, restored, tolerance=max(1.0, epsilon * 2.0)
        )
    )
    return RecordedRefinement(result, optimized.batches, optimized.outcomes)


# Transitional private alias for internal callers retained until Task 17C removes
# the VectorRegion adapter boundary.
_run_detection_recorded = run_detection_with_intent


def _rebind_corner_provenance(
    before: Sequence[VectorRegion],
    after: Sequence[VectorRegion],
    *,
    tolerance: float,
) -> DrawingRegions:
    """Preserve trace-declared corners through a command-preserving rewrite."""
    originals = {leaf.id: leaf for root in before for leaf in root.leaves()}

    def visit(region: VectorRegion) -> VectorRegion:
        if region.is_branch:
            children = tuple(visit(child) for child in region.children)
            # VectorRegion equality deliberately ignores diagnostics.  Corner
            # provenance is a diagnostics-only update, so retain the rebuilt
            # branch whenever any child instance changed.
            return (
                region.with_children(children)
                if any(child is not original for child, original in zip(children, region.children, strict=True))
                else region
            )
        assert region.current is not None
        original = originals.get(region.id)
        if (
            original is None
            or original.current is None
            or original.current.kind != "path"
            or region.current.kind != "path"
        ):
            return region
        corners = original.diagnostics.get("corners")
        if not isinstance(corners, Mapping) or corners.get("path") != str(original.current.params.get("d", "")):
            return region
        rebound = rebind_prefit_corner_diagnostics(
            str(region.current.params["d"]),
            corners,
            tolerance=tolerance,
        )
        return region.with_diagnostics({"corners": rebound})

    return tuple(visit(region) for region in after)




def _restore_root_metadata(region: VectorRegion, original: VectorRegion | None) -> VectorRegion:
    """Optimizer passes preserve geometry IDs, while drawing IDs stay root-owned."""
    if original is None or original.drawing_id is None:
        return region
    if region.is_leaf:
        assert region.current is not None
        return VectorRegion(
            id=region.id,
            current=region.current,
            original=region.original,
            fill=region.fill,
            z=region.z,
            footprint=region.footprint,
            raster=region.raster,
            source_label=region.source_label,
            color_hex=region.color_hex,
            drawing_id=original.drawing_id,
            source_regions=original.source_regions,
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
        drawing_id=original.drawing_id,
        source_regions=original.source_regions,
        diagnostics=region.diagnostics,
        original=region.original,
    )


def _clone(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    targets = _targets(regions)
    source_info = targets[op["source"]]
    source = source_info.region
    target_info = targets[op["target"]]
    target = target_info.region
    assert source.current is not None and source.fill is not None and target.current is not None and target.fill is not None
    transforms = tuple(_affine_tuple(transform) for transform in op["transforms"])
    footprints = tuple(
        _transform_footprint(source.footprint, transform, target.footprint)
        for transform in transforms
    )
    if len(transforms) == 1:
        shape = Shape("use", {"href_obj_id": source.id, "transform": transforms[0],
                              "href_region_path": relative_source_path(target_info.region_path, source_info.region_path)})
        footprint = footprints[0]
    else:
        if not all(isinstance(footprint, SkPath) for footprint in footprints):
            raise ValueError("multi-transform clones require polygonal footprints")
        footprint = unary_union([footprint for footprint in footprints if isinstance(footprint, SkPath)])
        shape = Shape("path", {"d": footprint.to_svg_d()})
    return _replace_leaf(
        regions,
        op["target"],
        lambda region: region.with_current(
            shape,
            fill=target.fill,
            footprint=footprint,
            diagnostics={"clone": {"source_id": source.id, "transforms": transforms}},
        ),
    )


def _affine_tuple(spec: Mapping[str, object]) -> tuple[float, float, float, float, float, float]:
    return tuple(float(spec[key]) for key in ("a", "b", "c", "d", "tx", "ty"))  # type: ignore[return-value]


def _transform_footprint(
    footprint: object,
    transform: tuple[float, float, float, float, float, float],
    fallback: object,
) -> object:
    if not isinstance(footprint, SkPath):
        return fallback
    a, b, c, d, e, f = transform
    return affinity.affine_transform(footprint, [a, c, b, d, e, f])


def _set_symmetry(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    """Reconstruct one target from the best-fitting half around an explicit axis."""
    targets = _targets(regions)
    target_info = targets[op["target"]]
    target = target_info.region
    assert target.current is not None and target.fill is not None
    if not isinstance(target.footprint, SkPath):
        raise ValueError("set_symmetry requires a polygonal footprint")
    axis = op["axis"]
    point = axis["point"]
    theta = math.radians(float(axis["angle_degrees"]))
    cx, cy = float(point[0]), float(point[1])
    matrix = _reflection_matrix(theta, cx, cy)
    direction = (math.cos(theta), math.sin(theta))
    parts = _split_footprint_by_line(
        target.footprint,
        (cx, cy),
        (cx + direction[0], cy + direction[1]),
    )
    if len(parts) != 2:
        raise ValueError("symmetry axis must divide the target into two non-empty halves")
    candidates: list[tuple[float, SkPath]] = []
    for half in parts:
        reflected = _transform_footprint(half, matrix, half)
        if not isinstance(reflected, SkPath):
            continue
        reconstructed = unary_union([half, reflected])
        residual = target.footprint.symmetric_difference(reconstructed).area / max(target.footprint.area, 1.0)
        candidates.append((float(residual), reconstructed))
    if not candidates:
        raise ValueError("symmetry reconstruction produced no valid geometry")
    residual, footprint = min(candidates, key=lambda item: item[0])
    joined = union_self_symmetry(
        target, (footprint,),
        diagnostics={"mode": "explicit_self", "axis": dict(axis), "residual": residual},
    )
    replaced = _replace_leaf(regions, op["target"], lambda region: joined)
    return remap_symmetry_clones(replaced, {target_info.region_path: target_info.region_path}, before=regions)


def _reflection_matrix(theta: float, cx: float, cy: float) -> tuple[float, float, float, float, float, float]:
    ux, uy = math.cos(theta), math.sin(theta)
    a, b, c, d = 2 * ux * ux - 1, 2 * ux * uy, 2 * ux * uy, 2 * uy * uy - 1
    e = cx - (a * cx + c * cy)
    f = cy - (b * cx + d * cy)
    return tuple(0.0 if abs(value) < 1e-12 else float(value) for value in (a, b, c, d, e, f))


def _drawing_groups(regions: Sequence[VectorRegion]) -> tuple[dict[str, object], ...]:
    """Derive unpainted containers independently of the painted target list."""
    groups = []
    def visit(region, address):
        if region.is_leaf:
            return
        children = tuple(child.drawing_id or f"{address}-{child.id}" for child in region.children)
        groups.append({"id": address, "child_ids": children, "diagnostics": _public_diagnostics(region.diagnostics)})
        for child, child_id in zip(region.children, children, strict=True):
            visit(child, child_id)
    for region in regions:
        visit(region, _root_label(region))
    return tuple(groups)


def _targets(regions: Sequence[VectorRegion]) -> dict[str, _Target]:
    """Derive public target handles from the region tree; do not persist an index."""
    result: dict[str, _Target] = {}

    def visit(region: VectorRegion, root_label: str, legacy_label: str, path: tuple[int, ...]) -> None:
        if region.is_leaf:
            label = _leaf_address(root_label, region, legacy_label)
            if label in result:
                raise ValueError(f"duplicate drawing region id: {label}")
            # Target lineage belongs to the retained leaf, not to its root
            # container.  A root can be a semantic union while its children
            # remain independently traced, cloned, or split regions.
            sources = frozenset(region.source_regions or (root_label,))
            result[label] = _Target(label, root_label, region, sources, path)
            return
        for child in region.children:
            visit(child, root_label, f"{legacy_label}-{child.id}", (*path, child.id))

    for region in regions:
        drawing_id = region.drawing_id
        if type(drawing_id) is not str:
            raise ValueError("drawing root is missing a stable drawing_id")
        visit(region, drawing_id, drawing_id, (region.id,))
    return result


def _leaf_address(root_address: str, region: VectorRegion, legacy_address: str) -> str:
    """Return a compact retained leaf address, with legacy-tree fallback."""
    candidate = region.drawing_id
    if isinstance(candidate, str) and (
        candidate.startswith(f"{root_address}-")
        or (candidate == root_address and legacy_address == root_address)
    ):
        return candidate
    return legacy_address


def _replace_leaf(
    regions: Sequence[VectorRegion], target_id: str, transform: Callable[[VectorRegion], VectorRegion]
) -> DrawingRegions:
    """Replace one labeled leaf while retaining the rest of the forest by value."""
    changed = False

    def visit(region: VectorRegion, root_label: str, legacy_label: str) -> VectorRegion:
        nonlocal changed
        if region.is_leaf:
            label = _leaf_address(root_label, region, legacy_label)
            if label == target_id:
                changed = True
                return transform(region)
            return region
        children = tuple(
            visit(child, root_label, f"{legacy_label}-{child.id}")
            for child in region.children
        )
        return region.with_children(children) if children != region.children else region

    replaced = tuple(visit(region, _root_label(region), _root_label(region)) for region in regions)
    if not changed:
        raise KeyError(target_id)
    return replaced


def _group_regions(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    """Create a semantic target with a SkPath-unioned, seam-free outline."""
    targets = _targets(regions)
    member_ids = _expand_merge_member_ids(targets, tuple(op["regions"]))
    members = tuple(targets[region_id] for region_id in member_ids)
    member_regions = tuple(member.region for member in members)
    first = member_regions[0]
    assert first.current is not None and first.fill is not None
    raster = np.zeros_like(first.raster, dtype=bool)
    for member in member_regions:
        raster |= member.raster
    footprints = [member.footprint for member in member_regions]
    if not all(isinstance(footprint, SkPath) for footprint in footprints):
        raise ValueError("merge requires polygonal vector-region footprints")
    footprint = unary_union([footprint for footprint in footprints if isinstance(footprint, SkPath)])
    shape = Shape("path", {"d": footprint.to_svg_d()})
    merged = VectorRegion.from_shape(
        id=max(region.id for region in _all_regions(regions)) + 1,
        shape=shape,
        fill=first.fill,
        z=min(member.z for member in member_regions),
        raster=raster,
        footprint=footprint,
        color_hex=first.color_hex,
        drawing_id=op["id"],
        source_regions=tuple(sorted({source for member in members for source in member.source_regions})),
        diagnostics={"merge": {"unioned": True, "sources": member_ids}},
    )
    return _remove_leaves(regions, frozenset(member_ids)) + (merged,)


def _expand_merge_member_ids(targets: Mapping[str, _Target], requested_ids: Sequence[object]) -> tuple[str, ...]:
    """Resolve a branch/root reference to every leaf currently retained below it."""
    resolved: list[str] = []
    for requested in requested_ids:
        assert isinstance(requested, str)
        members = (requested,) if requested in targets else tuple(
            sorted(target_id for target_id in targets if target_id.startswith(f"{requested}-"))
        )
        if not members:
            raise KeyError(requested)
        for member in members:
            if member in resolved:
                raise ValueError(f"merge region {requested!r} resolves to an already selected child")
            resolved.append(member)
    return tuple(resolved)


def _split_region(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    """Replace one retained target with two exact vector clips across an explicit divider."""
    targets = _targets(regions)
    target_id = op["target"]
    assert isinstance(target_id, str)
    target = targets[target_id]
    source = target.region
    if not source.is_leaf or source.current is None or source.fill is None:
        raise ValueError("split requires a filled leaf target")
    if not isinstance(source.footprint, SkPath):
        raise ValueError("split requires a polygonal vector-region footprint")

    divider = op["divider"]
    assert isinstance(divider, Mapping)
    divider_type = divider["type"]
    if divider_type == "line":
        points = divider["points"]
        assert isinstance(points, Sequence)
        first = points[0]
        second = points[1]
        assert isinstance(first, Sequence) and isinstance(second, Sequence)
        parts = _split_footprint_by_line(
            source.footprint,
            (float(first[0]), float(first[1])),
            (float(second[0]), float(second[1])),
        )
    elif divider_type == "path":
        d = divider["d"]
        assert isinstance(d, str)
        parts = _split_footprint_by_path(source.footprint, d)
    else:  # guarded by plan validation; keeps this executor safe for direct callers.
        raise ValueError("divider type must be 'line' or 'path'")

    if len(parts) != 2 or any(part.is_empty or part.area <= 1e-6 for part in parts):
        raise ValueError("divider does not divide the target into two non-empty regions")

    next_id = max(region.id for region in _all_regions(regions)) + 1
    children: list[VectorRegion] = []
    for index, footprint in enumerate(parts):
        shape = Shape("path", {"d": footprint.to_svg_d()})
        children.append(
            VectorRegion.from_shape(
                id=source.id if index == 0 else next_id,
                shape=shape,
                fill=source.fill,
                z=source.z + index * 0.001,
                footprint=footprint,
                raster=rasterize(footprint, source.raster.shape),
                source_label=source.source_label,
                color_hex=source.color_hex,
                source_regions=source.source_regions,
                coverage=source.coverage,
                diagnostics={
                    "split": {
                        "source": target_id,
                        "divider": dict(divider),
                        "part": index + 1,
                    }
                },
            )
        )
    branch = VectorRegion.branch(
        id=source.id,
        children=tuple(children),
        z=source.z,
        raster=source.raster,
        footprint=source.footprint,
        fill=source.fill,
        source_label=source.source_label,
        color_hex=source.color_hex,
        drawing_id=source.drawing_id,
        source_regions=source.source_regions,
        diagnostics={"split": {"source": target_id, "divider": dict(divider), "children": 2}},
    )
    return _replace_leaf(regions, target_id, lambda _region: branch)


def _split_footprint_by_line(
    footprint: SkPath, first: tuple[float, float], second: tuple[float, float]
) -> tuple[SkPath, SkPath]:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("line divider points must be distinct")
    direction = (dx / length, dy / length)
    normal = (-direction[1], direction[0])
    min_x, min_y, max_x, max_y = footprint.bounds
    extent = max(math.hypot(max_x - min_x, max_y - min_y) * 4.0, 1.0)
    start = (first[0] - direction[0] * extent, first[1] - direction[1] * extent)
    end = (first[0] + direction[0] * extent, first[1] + direction[1] * extent)
    positive = SkPath([
        start,
        end,
        (end[0] + normal[0] * extent * 2.0, end[1] + normal[1] * extent * 2.0),
        (start[0] + normal[0] * extent * 2.0, start[1] + normal[1] * extent * 2.0),
    ])
    negative = SkPath([
        start,
        (start[0] - normal[0] * extent * 2.0, start[1] - normal[1] * extent * 2.0),
        (end[0] - normal[0] * extent * 2.0, end[1] - normal[1] * extent * 2.0),
        end,
    ])
    return footprint.intersection(positive), footprint.intersection(negative)


def _split_footprint_by_path(footprint: SkPath, d: str) -> tuple[SkPath, SkPath]:
    """Split one simple footprint using an open divider whose endpoints touch its boundary."""
    subpaths = _parse_subpaths(d)
    if len(subpaths) != 1 or not subpaths[0] or subpaths[0][0][0] != "M" or any(kind == "Z" for kind, _ in subpaths[0]):
        raise ValueError("path divider must contain exactly one open SVG subpath")
    sampled_divider = _sample_subpath(subpaths[0], samples=32)
    if len(sampled_divider) < 2:
        raise ValueError("path divider must contain a drawable segment")
    rings = footprint.linearized_subpaths
    if len(rings) != 1:
        raise ValueError("path divider currently requires a simple target without holes or disconnected islands")
    ring = rings[0]
    start, start_segment, start_t, start_distance = _project_to_ring(sampled_divider[0], ring)
    end, end_segment, end_t, end_distance = _project_to_ring(sampled_divider[-1], ring)
    min_x, min_y, max_x, max_y = footprint.bounds
    endpoint_tolerance = max(1.0, min(3.0, math.hypot(max_x - min_x, max_y - min_y) * 0.005))
    if max(start_distance, end_distance) > endpoint_tolerance:
        raise ValueError("path divider endpoints must lie on the target boundary")
    boundary = _forward_ring_arc(ring, end, end_segment, end_t, start, start_segment, start_t)
    divider_d = _snapped_open_divider_d(subpaths[0], start, end)
    boundary_d = " ".join(f"L{_fmt(x)} {_fmt(y)}" for x, y in boundary[1:])
    clip = SkPath.from_svg_d(f"{divider_d} {boundary_d} Z")
    first = footprint.intersection(clip)
    second = footprint.difference(first)
    return first, second


def _snapped_open_divider_d(
    tokens: Sequence[tuple[str, list[float]]], start: tuple[float, float], end: tuple[float, float]
) -> str:
    """Serialize one open divider while snapping only its boundary endpoints."""
    normalized = [(kind, list(values)) for kind, values in tokens]
    normalized[0][1][:2] = [start[0], start[1]]
    final_kind, final_values = normalized[-1]
    endpoint_offset = {"L": 0, "Q": 2, "C": 4, "A": 5}.get(final_kind)
    if endpoint_offset is None:
        raise ValueError("path divider must end with a drawable SVG command")
    final_values[endpoint_offset:endpoint_offset + 2] = [end[0], end[1]]
    parts: list[str] = []
    for kind, values in normalized:
        parts.append(kind + " ".join(_fmt(value) for value in values))
    return " ".join(parts)


def _project_to_ring(
    point: tuple[float, float], ring: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float], int, float, float]:
    px, py = point
    best: tuple[tuple[float, float], int, float, float] | None = None
    for index, start in enumerate(ring):
        end = ring[(index + 1) % len(ring)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq <= 1e-12 else max(0.0, min(1.0, ((px - start[0]) * dx + (py - start[1]) * dy) / length_sq))
        projected = (start[0] + t * dx, start[1] + t * dy)
        distance = math.hypot(px - projected[0], py - projected[1])
        if best is None or distance < best[3]:
            best = (projected, index, t, distance)
    assert best is not None
    return best


def _forward_ring_arc(
    ring: Sequence[tuple[float, float]],
    start: tuple[float, float],
    start_segment: int,
    start_t: float,
    end: tuple[float, float],
    end_segment: int,
    end_t: float,
) -> list[tuple[float, float]]:
    """Return the forward boundary arc, including projected start/end points."""
    if start_segment == end_segment and start_t <= end_t:
        return [start, end]
    points = [start]
    index = (start_segment + 1) % len(ring)
    while index != (end_segment + 1) % len(ring):
        points.append(ring[index])
        index = (index + 1) % len(ring)
    points.append(end)
    return points


def _remove_leaves(regions: Sequence[VectorRegion], removed_ids: frozenset[str]) -> DrawingRegions:
    """Remove selected labeled leaves, pruning only now-empty branch nodes."""

    def visit(region: VectorRegion, root_label: str, legacy_label: str) -> VectorRegion | None:
        if region.is_leaf:
            label = _leaf_address(root_label, region, legacy_label)
            return None if label in removed_ids else region
        children = tuple(
            child
            for child in (
                visit(child, root_label, f"{legacy_label}-{child.id}")
                for child in region.children
            )
            if child is not None
        )
        if not children:
            return None
        return region if children == region.children else region.with_children(children)

    return tuple(
        region
        for root in regions
        if (region := visit(root, _root_label(root), _root_label(root))) is not None
    )


def _all_regions(regions: Sequence[VectorRegion]) -> tuple[VectorRegion, ...]:
    all_regions: list[VectorRegion] = []
    for region in regions:
        all_regions.append(region)
        all_regions.extend(region.leaves()) if region.is_branch else None
    return tuple(all_regions)


def _root_label(region: VectorRegion) -> str:
    if type(region.drawing_id) is not str:
        raise ValueError("drawing root is missing a stable drawing_id")
    return region.drawing_id


def _with_fill(region: VectorRegion, fill: Fill) -> VectorRegion:
    assert region.current is not None
    return region.with_current(region.current, fill=fill, footprint=region.footprint)


def _geometry_key(region: VectorRegion) -> tuple[object, ...]:
    """Stable enough comparison for deciding whether a retained leaf moved."""
    assert region.current is not None
    footprint_d = region.footprint.to_svg_d() if isinstance(region.footprint, SkPath) else None
    return (region.current.kind, repr(region.current.params), footprint_d)


def _invalidate_changed_fills(
    before: Sequence[VectorRegion], current: Sequence[VectorRegion]
) -> DrawingRegions:
    """Clear fills on new or geometrically changed leaves, preserving all others."""
    prior = {leaf.id: _geometry_key(leaf) for root in before for leaf in root.leaves()}

    def visit(region: VectorRegion) -> VectorRegion:
        if region.is_leaf:
            assert region.current is not None
            if prior.get(region.id) == _geometry_key(region):
                return region
            return region.with_current(
                region.current,
                fill=None,
                footprint=region.footprint,
                diagnostics={"fill": {"status": "invalidated", "reason": "geometry_changed"}},
            )
        children = tuple(visit(child) for child in region.children)
        return region if children == region.children else region.with_children(children)

    return tuple(visit(region) for region in current)


def refill_changed_regions_from_source(
    before: Sequence[VectorRegion],
    current: Sequence[VectorRegion],
    rgb: np.ndarray,
    *, allow_gradients: bool = True,
) -> DrawingRegions:
    """Apply the shared fill resolver only to geometry changed since *before*."""
    return refill_regions(_invalidate_changed_fills(before, current), rgb, allow_gradients=allow_gradients)


def _sampled_flat_hex(mask: np.ndarray, rgb: np.ndarray) -> str:
    """Use the source's median color when gradient fitting selects a flat fill."""
    pixels = rgb[mask]
    if len(pixels) == 0:
        return "#000000"
    red, green, blue = np.rint(np.median(pixels, axis=0)).astype(np.uint8)
    return f"#{red:02X}{green:02X}{blue:02X}"


def _visible_leaf_masks(
    regions: Sequence[VectorRegion],
    canvas_shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Return each leaf's visible paint surface after later siblings occlude it.

    Drawing geometry frequently uses overlapping shapes to create a ring or a
    cutout.  The fill stage must sample the portion that remains visible in
    SVG paint order, not the complete construction shape underneath it.
    """
    ordered = [
        leaf
        for root in regions
        for leaf in root.leaves()
    ]
    ordered = [
        leaf
        for _index, leaf in sorted(
            enumerate(ordered), key=lambda item: (item[1].z, item[0])
        )
    ]
    occluded = np.zeros(canvas_shape, dtype=bool)
    masks: dict[int, np.ndarray] = {}
    for leaf in reversed(ordered):
        footprint = leaf.footprint if isinstance(leaf.footprint, SkPath) else to_polygon(leaf.current)
        full_mask = rasterize(footprint, canvas_shape)
        masks[leaf.id] = full_mask & ~occluded
        occluded |= full_mask
    return masks


def _fill_sample_mask(
    region: VectorRegion,
    visible_mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return pixels that are visible now and supported by the source material.

    ``footprint`` is the final geometric claim and can intentionally move beyond
    the original trace after primitive fitting, symmetry, or stitching.
    ``raster`` remains the leaf's traced material evidence.  Sampling their
    intersection prevents adjacent paint and the white-composited AA fringe
    from teaching a leaf the wrong fill.

    A deliberate plan can move a leaf far enough that the old material overlap
    is no longer useful.  Fall back to its visible footprint in that case so
    the fill pass still has meaningful source samples.
    """
    source_mask = np.asarray(region.raster, dtype=bool)
    if source_mask.shape != visible_mask.shape:
        return visible_mask, "visible_footprint"
    supported = visible_mask & source_mask
    visible_area = int(visible_mask.sum())
    minimum_support = max(3, int(np.ceil(visible_area * 0.05)))
    # Fill analysis should learn the material's interior, not the composited
    # white/background fringe around a traced outline.  Preserve the previous
    # supported-mask fallback for thin details that do not have a two-pixel
    # interior core.
    core = ndimage.binary_erosion(supported, iterations=2)
    if int(core.sum()) >= minimum_support:
        return core, "visible_source_material_core"
    if int(supported.sum()) >= minimum_support:
        return supported, "visible_source_material"
    return visible_mask, "visible_footprint"


def apply_output_fill_settings(
    regions: Sequence[VectorRegion], rgb: np.ndarray, settings: DrawingOutputSettings,
) -> DrawingRegions:
    """Constrain final paint without touching geometry or source evidence."""
    if settings.allow_gradients and settings.max_output_colors is None:
        return tuple(regions)
    palette = (
        extract_palette(rgb, max_sampling_colors=settings.max_output_colors, min_fraction=0.0)
        if settings.max_output_colors is not None else None
    )

    def flat_source_color(region: VectorRegion) -> str:
        # Select the representative paint from the finalized shape itself.
        # Its old segmentation raster may contain only a boundary fringe after
        # compound splitting; that fringe must not choose the flat output color.
        footprint = region.footprint if isinstance(region.footprint, SkPath) else to_polygon(region.current)
        sample = rasterize(footprint, rgb.shape[:2])
        core = ndimage.binary_erosion(sample, iterations=2)
        return _sampled_flat_hex(core if core.any() else sample, rgb)

    def color(value: str) -> str:
        if palette is None:
            return value
        sample = np.asarray([[[int(value[i:i + 2], 16) for i in (1, 3, 5)]]], dtype=np.uint8)
        red, green, blue = quantize(sample, palette)[0, 0]
        return f"#{red:02X}{green:02X}{blue:02X}"

    def visit(region: VectorRegion) -> VectorRegion:
        if not region.is_leaf:
            return region.with_children(tuple(visit(child) for child in region.children))
        assert region.current is not None
        fill = region.fill
        if not settings.allow_gradients and not isinstance(fill, FlatFill):
            fill = FlatFill(flat_source_color(region))
        if isinstance(fill, FlatFill):
            fill = FlatFill(color(fill.hex))
        elif isinstance(fill, (LinearGradientFill, RadialGradientFill)):
            stops = [(offset, color(value)) for offset, value in fill.stops]
            if len({value for _offset, value in stops}) == 1:
                fill = FlatFill(stops[0][1])
            else:
                fill = replace(fill, stops=stops)
        elif isinstance(fill, RasterFill) and palette is not None:
            # A raster paint cannot honor a finite vector palette. Resolve it
            # to the same source-derived representative used for flat output.
            fill = FlatFill(color(flat_source_color(region)))
        return region.with_current(region.current, fill=fill, footprint=region.footprint)

    return tuple(visit(region) for region in regions)


def refill_regions(regions: Sequence[VectorRegion], rgb: np.ndarray, *, allow_gradients: bool = True) -> DrawingRegions:
    """Run the shared fill stage for leaves invalidated by a geometry mutation.

    Callers deliberately leave an invalidated leaf with ``fill=None``.  This
    function is the sole point that turns its finalized vector footprint back
    into a source-derived FlatFill, LinearGradientFill, or RadialGradientFill.
    """
    canvas_shape = rgb.shape[:2]
    visible_masks = _visible_leaf_masks(regions, canvas_shape)

    def visit(region: VectorRegion) -> VectorRegion:
        if region.is_leaf:
            if region.fill is not None:
                return region
            assert region.current is not None
            footprint = region.footprint
            if not isinstance(footprint, SkPath):
                footprint = to_polygon(region.current)
            sample_mask, sample_domain = _fill_sample_mask(region, visible_masks[region.id])
            fill = fit_fill(sample_mask, rgb, flat_hex=_sampled_flat_hex(sample_mask, rgb), allow_gradients=allow_gradients)
            return region.with_current(
                region.current,
                fill=fill,
                footprint=footprint,
                diagnostics={"fill": {
                    "status": "refit",
                    "mask": sample_domain,
                    "sampled_pixels": int(sample_mask.sum()),
                }},
            )
        children = tuple(visit(child) for child in region.children)
        return region if children == region.children else region.with_children(children)

    return tuple(visit(region) for region in regions)


def refill_all_regions_from_source(
    regions: Sequence[VectorRegion],
    rgb: np.ndarray,
    *,
    share_root_gradients: bool = True,
    allow_gradients: bool = True,
) -> DrawingRegions:
    """Discard every existing fill and re-fit finalized geometry from *rgb*.

    Geometry-guide retracing deliberately uses a palette-labelled image to
    establish boundaries.  Its colors are temporary segmentation labels, never
    a candidate final paint source.
    """
    gradient_roots = {
        root.id
        for root in regions
        if any(isinstance(leaf.fill, (LinearGradientFill, RadialGradientFill)) for leaf in root.leaves())
    }

    def clear(region: VectorRegion) -> VectorRegion:
        if region.is_leaf:
            assert region.current is not None
            compound = region.diagnostics.get("compound")
            # ``split_compound`` materializes a canvas/cutout plate when a
            # nested subpath has no source paint owner.  That fill is semantic
            # render evidence, not a stale source-derived fit: clearing it
            # makes later refill sample an empty footprint and emit black.
            if (
                isinstance(compound, Mapping)
                and compound.get("fill_source_id") is None
                and region.fill is not None
            ):
                return region
            return region.with_current(
                region.current,
                fill=None,
                footprint=region.footprint,
                diagnostics={"fill": {"status": "invalidated", "reason": "geometry_guide"}},
            )
        children = tuple(clear(child) for child in region.children)
        return region.with_children(children)

    cleared = tuple(clear(region) for region in regions)
    # A ``self`` branch represents one material surface split only so the
    # optimizer can construct a reflected half.  Re-fit its paint once from
    # the finalized union and assign that paint to both implementation leaves.
    # This is deliberately distinct from ``root_self``: the latter retains
    # genuine material children and must continue to fit them independently.
    shared = _refill_self_symmetry_surfaces(cleared, rgb, allow_gradients=allow_gradients)
    refilled = refill_regions(shared, rgb, allow_gradients=allow_gradients)
    if not share_root_gradients or not allow_gradients:
        return refilled
    visible_masks = _visible_leaf_masks(refilled, rgb.shape[:2])
    return _refit_shared_root_gradients(refilled, rgb, gradient_roots, visible_masks)


def _refill_self_symmetry_surfaces(
    regions: Sequence[VectorRegion],
    rgb: np.ndarray,
    *, allow_gradients: bool = True,
) -> DrawingRegions:
    """Restore one source-derived fill to leaves created only for reflection."""
    visible_masks = _visible_leaf_masks(regions, rgb.shape[:2])

    def with_fill(region: VectorRegion, fill: Fill, sample_mask: np.ndarray, sample_domain: str) -> VectorRegion:
        if region.is_leaf:
            assert region.current is not None
            return region.with_current(
                region.current,
                fill=fill,
                footprint=region.footprint,
                diagnostics={
                    "fill": {
                        "status": "refit",
                        "mask": sample_domain,
                        "sampled_pixels": int(sample_mask.sum()),
                        "shared_self_symmetry": True,
                    }
                },
            )
        return region.with_children(tuple(with_fill(child, fill, sample_mask, sample_domain) for child in region.children))

    def visit(region: VectorRegion) -> VectorRegion:
        if region.is_leaf:
            return region
        children = tuple(visit(child) for child in region.children)
        updated = region if children == region.children else region.with_children(children)
        symmetry = updated.diagnostics.get("symmetry")
        leaves = updated.leaves()
        if (
            not isinstance(symmetry, Mapping)
            or symmetry.get("mode") != "self"
            or not leaves
            # ``self`` children are implementation halves of one traced root.
            # If provenance ever differs, they are independently addressable
            # source regions and must remain candidates for clone detection
            # (which deliberately preserves per-region fills).
            or any(leaf.source_regions != leaves[0].source_regions for leaf in leaves[1:])
            or any(leaf.fill is not None for leaf in leaves)
        ):
            return updated
        visible = np.zeros(rgb.shape[:2], dtype=bool)
        for leaf in leaves:
            visible |= visible_masks.get(leaf.id, np.zeros_like(visible))
        sample_mask, sample_domain = _fill_sample_mask(updated, visible)
        fill = fit_fill(sample_mask, rgb, flat_hex=_sampled_flat_hex(sample_mask, rgb), allow_gradients=allow_gradients)
        return with_fill(updated, fill, sample_mask, sample_domain)

    return tuple(visit(region) for region in regions)


def _visible_material_fraction(mask: np.ndarray, rgb: np.ndarray) -> float:
    """Return the fraction of a visible surface occupied by non-white source paint."""
    pixels = rgb[mask]
    if len(pixels) == 0:
        return 0.0
    return float(np.mean(np.any(pixels < 240, axis=1)))


def _refit_shared_root_gradients(
    regions: Sequence[VectorRegion],
    rgb: np.ndarray,
    gradient_roots: set[int],
    visible_masks: Mapping[int, np.ndarray],
) -> DrawingRegions:
    """Apply a shared source gradient across compatible finalized root leaves.

    Material quantization can split one continuous gradient into several
    palette-labelled leaves.  The leaves' geometry remains independent, but
    their union provides the sample domain needed to recover the original
    paint.  White cutouts and one-colour material groups are intentionally
    excluded.
    """
    canvas_shape = rgb.shape[:2]

    def refit_root(root: VectorRegion) -> VectorRegion:
        if root.id not in gradient_roots:
            return root
        visible_cutouts = [
            leaf
            for leaf in root.leaves()
            if visible_masks[leaf.id].any()
            and _visible_material_fraction(visible_masks[leaf.id], rgb) < 0.5
        ]
        eligible = [
            leaf
            for leaf in root.leaves()
            if _visible_material_fraction(visible_masks[leaf.id], rgb) >= 0.5
        ]
        if len(eligible) < 2:
            return root
        mask = np.zeros(canvas_shape, dtype=bool)
        for leaf in eligible:
            mask |= visible_masks[leaf.id]
        # Rasterization includes the outer path edge.  Those pixels can belong
        # to a white source background rather than the coloured material (in
        # particular for adjacent, separately traced gradient bands), so they
        # must not bias the gradient model itself.  The resulting fill is
        # nevertheless applied to each leaf's full finalized footprint below.
        material_mask = mask & np.any(rgb < 240, axis=2)
        if int(material_mask.sum()) < 3:
            return root
        fill = fit_fill(
            material_mask,
            rgb,
            flat_hex=_sampled_flat_hex(material_mask, rgb),
        )
        if not isinstance(fill, (LinearGradientFill, RadialGradientFill)):
            return root
        eligible_ids = {leaf.id for leaf in eligible}
        cutout_fills = {
            leaf.id: FlatFill(_sampled_flat_hex(visible_masks[leaf.id], rgb))
            for leaf in visible_cutouts
        }

        def visit(region: VectorRegion) -> VectorRegion:
            if region.is_leaf:
                if region.id not in eligible_ids and region.id not in cutout_fills:
                    return region
                assert region.current is not None
                is_cutout = region.id in cutout_fills
                return region.with_current(
                    region.current,
                    fill=cutout_fills.get(region.id, fill),
                    footprint=region.footprint,
                    diagnostics={"fill": {
                        "status": "visible_cutout" if is_cutout else "shared_root_gradient",
                        "root": root.drawing_id,
                    }},
                )
            children = tuple(visit(child) for child in region.children)
            return region if children == region.children else region.with_children(children)

        return visit(root)

    return tuple(refit_root(root) for root in regions)


def _with_z(region: VectorRegion, z: float) -> VectorRegion:
    assert region.current is not None
    return region.with_current(region.current, z=z, footprint=region.footprint)


def _align(regions: Sequence[VectorRegion], op: Mapping[str, object]) -> DrawingRegions:
    """Translate one retained region until its bounds center aligns with another."""
    targets = _targets(regions)
    target = targets[op["target"]]
    reference = targets[op["reference"]]
    if not isinstance(target.region.footprint, SkPath) or not isinstance(reference.region.footprint, SkPath):
        raise ValueError("align requires polygonal vector-region footprints")
    target_bounds = target.region.footprint.bounds
    reference_bounds = reference.region.footprint.bounds
    target_center = ((target_bounds[0] + target_bounds[2]) / 2, (target_bounds[1] + target_bounds[3]) / 2)
    reference_center = ((reference_bounds[0] + reference_bounds[2]) / 2, (reference_bounds[1] + reference_bounds[3]) / 2)
    axes = set(op["axes"])
    dx = reference_center[0] - target_center[0] if "x" in axes else 0.0
    dy = reference_center[1] - target_center[1] if "y" in axes else 0.0
    assert target.region.current is not None
    shape = bake_shape_transform(target.region.current, (1, 0, 0, 1, dx, dy))
    return _replace_leaf(
        regions,
        target.id,
        lambda region: region.with_current(
            shape,
            footprint=to_polygon(shape),
            diagnostics={"align": {"reference": reference.id, "axes": tuple(sorted(axes))}},
        ),
    )


def _path_shape(
    trace: TraceResult,
    target: _Target,
    geometry: Mapping[str, object],
    *,
    references: Mapping[str, tuple[object, np.ndarray]],
    epsilon: float,
    max_error: float,
    fit_sources: Mapping[str, np.ndarray] | None = None,
) -> Shape:
    assert target.region.current is not None and target.region.current.kind == "path"
    source_commands = svg_path_commands(str(target.region.current.params["d"]), target.id, path_ids=target.region.current.params.get("path_ids"))
    return _path_shape_from_segment_fits(
        target,
        source_commands,
        geometry,
        references=references,
        epsilon=epsilon,
        max_error=max_error,
        fit_sources=fit_sources,
    )


def _path_shape_from_segment_fits(
    target: _Target,
    commands: Sequence[object],
    geometry: Mapping[str, object],
    *,
    references: Mapping[str, tuple[object, np.ndarray]],
    epsilon: float,
    max_error: float,
    fit_sources: Mapping[str, np.ndarray] | None = None,
) -> Shape:
    """Apply command-addressed refits without dropping untouched path geometry."""
    fit_metrics: dict[str, object] = {}
    def source_for(command_id: str) -> np.ndarray | None:
        if fit_sources is None:
            return None
        if command_id not in fit_sources:
            raise ValueError(f"fit {command_id} has no exact original material-polyline span")
        return fit_sources[command_id]

    fits = {
        op["target"]: _fit_mode(op)
        for op in geometry["ops"]
        if op["op"] == "fit" and "target" in op
    }
    length_matches = {
        op["target"]: op["reference"]
        for op in geometry["ops"]
        if op["op"] == "match_length"
    }
    matches = {
        op["target"]: (op["reference"], _affine_tuple(op["transform"]))
        for op in geometry["ops"]
        if op["op"] == "match"
    }
    parallels = {
        op["target"]: (op["reference"], op.get("distance"))
        for op in geometry["ops"]
        if op["op"] == "set_parallel"
    }
    alignments = {
        op["target"]: (op["reference"], set(op["axes"]))
        for op in geometry["ops"]
        if op["op"] == "align"
    }
    snaps = {
        op["target"]: op["reference"]
        for op in geometry["ops"]
        if op["op"] == "snap"
    }
    removals = {op["target"] for op in geometry["ops"] if op["op"] == "remove"}
    values_by_command = {command.id: tuple(command.values) for command in commands}
    command_index = {command.id: index for index, command in enumerate(commands)}
    raw_starts: dict[str, tuple[float, float]] = {}
    raw_current: tuple[float, float] | None = None
    raw_subpath_start: tuple[float, float] | None = None
    for command in commands:
        if command.command == "M":
            raw_current = (float(command.values[0]), float(command.values[1]))
            raw_subpath_start = raw_current
            continue
        if raw_current is not None:
            raw_starts[command.id] = raw_current
        if command.command == "Z":
            raw_current = raw_subpath_start
        else:
            raw_current = (float(command.values[-2]), float(command.values[-1]))

    span_fits: dict[str, tuple[tuple[str, ...], tuple[float, float], str, bool]] = {}
    skipped_by_span: set[str] = set()
    for op in geometry["ops"]:
        if op["op"] != "fit" or "between" not in op:
            continue
        left_id, right_id = op["between"]
        left_index, right_index = command_index[left_id], command_index[right_id]
        interior = tuple(
            command.id
            for command in commands[left_index + 1:right_index + 1]
            if command.command not in {"M", "Z"}
        )
        # ``drawing_plan.validate_plan`` has already proved that this is a
        # non-empty, target-local, ordered span in one subpath.
        assert interior and right_id in raw_starts
        kind, recursive = _fit_mode(op)
        end = tuple(float(value) for value in values_by_command[right_id][-2:])
        span_fits[interior[0]] = (interior, end, kind, recursive)
        skipped_by_span.update(interior[1:])
    parallel_lines: set[str] = set()
    for index, command in enumerate(commands):
        if command.id not in parallels:
            continue
        reference_id, requested_distance = parallels[command.id]
        reference_command, reference_start = references[reference_id]
        reference_vector = np.asarray(reference_command.values[-2:], dtype=float) - reference_start
        reference_length = float(np.linalg.norm(reference_vector))
        if reference_length == 0.0:
            raise ValueError(f"cannot make a segment parallel to zero-length {reference_command.id!r}")
        unit = reference_vector / reference_length
        normal = np.array([-unit[1], unit[0]])
        previous = commands[index - 1]
        start = np.asarray(values_by_command[previous.id][-2:], dtype=float)
        values = values_by_command[command.id]
        samples = np.array([start, *[values[offset:offset + 2] for offset in range(0, len(values), 2)]], dtype=float)
        segment_length = float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
        if segment_length == 0.0:
            raise ValueError(f"cannot fit zero-length segment {command.id!r} as parallel")
        along = (samples - reference_start) @ unit
        distance = float(requested_distance) if requested_distance is not None else float(((samples - reference_start) @ normal).mean())
        fitted_start = reference_start + unit * (float(along.mean()) - segment_length / 2) + normal * distance
        fitted_end = reference_start + unit * (float(along.mean()) + segment_length / 2) + normal * distance
        previous_values = list(values_by_command[previous.id])
        previous_values[-2:] = [float(fitted_start[0]), float(fitted_start[1])]
        values_by_command[previous.id] = tuple(previous_values)
        values_by_command[command.id] = (float(fitted_end[0]), float(fitted_end[1]))
        parallel_lines.add(command.id)
    pieces: list[str] = []
    command_lineage: dict[str, str] = {}
    output_index = -1
    output_path_id = ""

    def append_piece(piece: str, source_id: str) -> None:
        nonlocal output_index, output_path_id
        if piece.startswith("M"):
            output_path_id = source_id.rsplit(".c", 1)[0]
            output_index = -1
        pieces.append(piece)
        output_index += sum(char in "MLQCA" for char in piece)
        command_lineage[source_id] = f"{output_path_id}.c{output_index}"

    current: tuple[float, float] | None = None
    subpath_start: tuple[float, float] | None = None
    for index, command in enumerate(commands):
        if command.command == "M":
            values = values_by_command[command.id]
            append_piece(f"M{_fmt(values[0])} {_fmt(values[1])}", command.id)
            current = (float(values[0]), float(values[1]))
            subpath_start = current
            continue
        if command.command == "Z":
            pieces.append("Z")
            current = subpath_start
            continue
        if command.id in skipped_by_span or command.id in removals:
            continue
        if current is None:
            raise ValueError("path segment appears before its move command")
        start = current
        if (span := span_fits.get(command.id)) is not None:
            interior, end, kind, recursive = span
            samples = _sample_subpath(
                [("M", list(start)), *[
                    (commands[command_index[interior_id]].command,
                     list(values_by_command[interior_id]))
                    for interior_id in interior
                ]],
                samples=32,
            )
            if not np.allclose(samples[-1], end):
                samples.append(end)
            sample_points = np.asarray(samples, dtype=float)
            source_points = source_for(interior[-1])
            if recursive:
                fitted = _recursive_fit_d(sample_points, kind, max_error=max_error, epsilon=epsilon)
                fitted = _validate_fit_source(start, fitted, source_points, max_error, f"{interior[0]}..{interior[-1]}", diagnostics=fit_metrics)
            else:
                fitted = _single_fit_d(
                    sample_points, kind, max_error=max_error,
                    address=f"{interior[0]}..{interior[-1]}", source_points=source_points, diagnostics=fit_metrics,
                )
            append_piece(fitted, interior[-1])
            current = end
            continue
        kind, recursive = fits.get(command.id, ("keep", False))
        output_command = "L" if command.id in parallel_lines else command.command
        values = values_by_command[command.id]
        if (match := matches.get(command.id)) is not None:
            reference_id, matrix = match
            reference_command, _ = references[reference_id]
            values = tuple(
                coordinate
                for point in zip(reference_command.values[::2], reference_command.values[1::2], strict=True)
                for coordinate in apply_affine_point(matrix, float(point[0]), float(point[1]))
            )
            output_command = reference_command.command
        if (reference_id := length_matches.get(command.id)) is not None:
            _, reference_start = references[reference_id]
            reference_command = references[reference_id][0]
            reference_vector = np.asarray(reference_command.values[-2:], dtype=float) - reference_start
            target_vector = np.asarray(values[-2:], dtype=float) - np.asarray(start, dtype=float)
            target_length = float(np.linalg.norm(target_vector))
            if target_length == 0.0:
                raise ValueError(f"cannot match length of zero-length segment {command.id!r}")
            scale = float(np.linalg.norm(reference_vector)) / target_length
            values = tuple(
                float(start[coordinate % 2]) + (value - start[coordinate % 2]) * scale
                for coordinate, value in enumerate(values)
            )
        if (alignment := alignments.get(command.id)) is not None:
            reference_id, axes = alignment
            reference_command, _ = references[reference_id]
            aligned = list(values)
            if "x" in axes:
                aligned[-2] = float(reference_command.values[-2])
            if "y" in axes:
                aligned[-1] = float(reference_command.values[-1])
            values = tuple(aligned)
        if (reference_id := snaps.get(command.id)) is not None:
            # Shared corners are the common case: move the target endpoint to
            # the reference endpoint on both axes, without translating the
            # region or exposing coordinate authoring to the caller.
            reference_command, _ = references[reference_id]
            snapped = list(values)
            snapped[-2:] = reference_command.values[-2:]
            values = tuple(snapped)
        # A retained command already in the requested family is an exact,
        # zero-residual one-command fit; numerical refitting can only lose it.
        exact_family = {"line": "L", "quadratic": "Q", "cubic": "C"}
        if kind == "keep" or (not recursive and output_command == exact_family.get(kind)):
            fitted = output_command + " ".join(_fmt(value) for value in values)
            if kind != "keep":
                fitted = _validate_fit_source(start, fitted, source_for(command.id), max_error, command.id, diagnostics=fit_metrics)
            append_piece(fitted, command.id)
        else:
            points = np.asarray(_sample_subpath(
                [("M", list(start)), (output_command, list(values))],
                samples=32,
            ), dtype=float)
            source_points = source_for(command.id)
            if recursive:
                fitted = _recursive_fit_d(points, kind, max_error=max_error, epsilon=epsilon)
                fitted = _validate_fit_source(start, fitted, source_points, max_error, command.id, diagnostics=fit_metrics)
            else:
                fitted = _single_fit_d(points, kind, max_error=max_error, address=command.id, source_points=source_points, diagnostics=fit_metrics)
            append_piece(fitted, command.id)
        current = (float(values[-2]), float(values[-1]))
    params: dict[str, object] = {"d": " ".join(pieces)}
    if fit_metrics:
        params["fit_acceptance"] = fit_metrics
    assert target.region.current is not None
    if (fill_rule := target.region.current.params.get("fill_rule")) is not None:
        params["fill_rule"] = fill_rule
    if "path_ids" in target.region.current.params:
        params["path_ids"] = target.region.current.params["path_ids"]
    immediate = dict(command_lineage)
    previous_lineage = target.region.current.params.get("path_command_lineage")
    input_d = str(target.region.current.params["d"])
    if isinstance(previous_lineage, Mapping):
        input_d = str(previous_lineage["input_d"])
        command_lineage = {
            old: command_lineage[intermediate]
            for old, intermediate in previous_lineage["commands"].items()
            if intermediate in command_lineage
        }
    params["path_command_lineage"] = {
        "input_d": input_d,
        "output_d": params["d"],
        "commands": command_lineage,
    }
    annotated = drawing_from_vector_regions(
        "path-refit", width=1.0, height=1.0, regions=(target.region,)
    ).regions[0].geometry
    if isinstance(annotated, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
        originals = component_paths(annotated)
        parsed = parse_svg_paths(str(params["d"]), region_id=target.id, path_ids=tuple(path.path.id for path in originals))
        # Only endpoint/owned-command bindings proven by the executor survive.
        # A refit that explicitly consumes a corner retires that declaration.
        rebound_paths = []
        for path, original in zip(parsed, originals, strict=True):
            ids = tuple(command.id for command in path.path.commands)
            rebound_anchors = []
            for anchor in original.anchors:
                if isinstance(anchor, CornerAnchor):
                    if not all(item in immediate for item in (anchor.entry_command_id, anchor.exit_command_id)):
                        continue
                    entry = immediate[anchor.entry_command_id]
                    exit = immediate[anchor.exit_command_id]
                    owned = []
                    index = ids.index(entry)
                    while index != ids.index(exit):
                        index = (index + 1) % len(ids)
                        if index != 0:
                            owned.append(ids[index])
                    rebound_anchors.append(replace(anchor,
                        entry_command_id=entry, exit_command_id=exit,
                        owned_command_ids=tuple(owned)))
                elif all(item in immediate for item in anchor.command_ids):
                    rebound_anchors.append(replace(anchor,
                        command_ids=tuple(immediate[item] for item in anchor.command_ids)))
            rebound_paths.append(replace(path, anchors=tuple(rebound_anchors)))
        rebound = tuple(rebound_paths)
        params["canonical_path_geometry"] = replace_component_paths(annotated, rebound)
    return Shape("path", params)


def _fit_mode(op: Mapping[str, object]) -> tuple[str, bool]:
    """Return the requested fit family and whether it may recursively split."""
    if "type" in op:
        return str(op["type"]), False
    return str(op["strategy"]), True


def _single_fit_d(
    points: np.ndarray, kind: str, *, max_error: float, address: str = "",
    source_points: np.ndarray | None = None,
    diagnostics: dict[str, object] | None = None,
) -> str:
    """Emit exactly one command or reject a forced fit that exceeds tolerance."""
    start, end = points[0], points[-1]

    def reject(residual: float, split: int) -> None:
        worst = points[split]
        raise ValueError(
            f"single {kind} fit exceeds max_error; "
            f"target={address or 'unspecified'}; "
            f"measured_error_px={float(np.sqrt(residual)):.6g}; "
            f"max_error_px={max_error:.6g}; "
            f"worst_sample={split}/{len(points) - 1}; "
            f"worst_point=({worst[0]:.6g}, {worst[1]:.6g}); "
            "metric=parameterized_sample_distance; "
            f"use strategy='{kind}' to allow recursive splitting or inspect the span boundaries"
        )

    if kind == "line":
        return _validate_fit_source(start, f"L{_fmt(end[0])} {_fmt(end[1])}", source_points, max_error, address, diagnostics=diagnostics)
    if kind == "quadratic":
        curve, residual, split = fit_quadratic_once(points, max_error)
        if source_points is None and residual > max_error * max_error:
            reject(residual, split)
        return _validate_fit_source(start, f"Q{_fmt(curve[1][0])} {_fmt(curve[1][1])} {_fmt(curve[2][0])} {_fmt(curve[2][1])}", source_points, max_error, address, diagnostics=diagnostics)
    if kind == "cubic":
        # ``type`` is an explicit geometric instruction: unlike automatic
        # cubic fitting, it deliberately permits an S-curve when that is what
        # best represents the requested retained command or span. Its controls
        # are unconstrained unless the plan explicitly adds tangent constraints.
        curve, residual, split = fit_cubic_once(
            points, max_error, guard_inflections=False, constrain_tangents=False
        )
        if source_points is None and residual > max_error * max_error:
            reject(residual, split)
        return _validate_fit_source(start, (
            f"C{_fmt(curve[1][0])} {_fmt(curve[1][1])} {_fmt(curve[2][0])} {_fmt(curve[2][1])} "
            f"{_fmt(curve[3][0])} {_fmt(curve[3][1])}"
        ), source_points, max_error, address, diagnostics=diagnostics)
    raise ValueError(f"unsupported one-command fit type {kind!r}")


def _validate_fit_source(
    start: Sequence[float], d: str, source_points: np.ndarray | None,
    max_error: float, address: str, *, diagnostics: dict[str, object] | None = None,
) -> str:
    """Check an emitted L/Q/C span against its exact original material span."""
    if source_points is None:
        return d
    # Subdivide Bézier control polygons until their chord error is bounded.
    # This also validates the rounded emitted coordinates, not hidden controls.
    flatten_error = min(0.01, max_error / 20.0)
    vertices = [np.asarray(start, dtype=float)]

    def flatten(control: np.ndarray, depth: int = 0) -> None:
        a, b = control[0], control[-1]
        chord = b - a
        denom = float(chord @ chord)
        t = np.clip(((control - a) @ chord) / denom, 0, 1) if denom else np.zeros(len(control))
        error = float(np.max(np.linalg.norm(control - (a + t[:, None] * chord), axis=1)))
        if error <= flatten_error:
            vertices.append(b)
            return
        if depth >= 24:
            raise ValueError("fit source validation could not bound curve linearization")
        levels = [control]
        while len(levels[-1]) > 1:
            levels.append((levels[-1][:-1] + levels[-1][1:]) / 2.0)
        flatten(np.asarray([level[0] for level in levels]), depth + 1)
        flatten(np.asarray([level[-1] for level in reversed(levels)]), depth + 1)

    for command in svg_path_commands(f"M{_fmt(start[0])} {_fmt(start[1])} {d}", "fit-check"):
        if command.command == "M":
            vertices[0] = np.asarray(command.values, dtype=float)
        elif command.command in {"L", "Q", "C"}:
            control = np.vstack([vertices[-1], np.asarray(command.values).reshape(-1, 2)])
            flatten(control)
        else:
            raise ValueError("source fit guard supports only open L/Q/C spans")
    original = [list(map(tuple, source_points))]
    candidate = [list(map(tuple, vertices))]
    measured = polyline_boundary_distance(
        original, candidate,
        closed=False, precision=flatten_error, direction="replacement_to_source",
    ) + flatten_error
    reverse = polyline_boundary_distance(
        original, candidate,
        closed=False, precision=flatten_error, direction="source_to_replacement",
    ) + flatten_error
    if diagnostics is not None:
        diagnostics[address] = {
            "reference": "original_material_polyline",
            "acceptance_direction": "curve_to_original",
            "curve_to_original_px": measured,
            "original_to_curve_px": reverse,
            "reverse_is_diagnostic_only": True,
            "max_error_px": max_error,
            "measurement_precision_px": 2 * flatten_error,
        }
    if not math.isfinite(measured) or measured > max_error:
        raise ValueError(
            f"fit exceeds max_error; target={address}; "
            f"measured_error_px={measured:.6g}; max_error_px={max_error:.6g}; "
            "reference=original_material_polyline; metric=curve_to_original_distance_upper_bound; "
            f"original_to_curve_diagnostic_px={reverse:.6g}; "
            f"measurement_precision_px={2 * flatten_error:.6g}; "
            "inspect source span boundaries or use recursive fitting"
        )
    return d


def _recursive_fit_d(points: np.ndarray, strategy: str, *, max_error: float, epsilon: float) -> str:
    """Emit one or more commands according to a residual-driven fit strategy."""
    return _curved_run_d(
        points,
        max_error,
        cubic=strategy == "cubic",
        line_epsilon=epsilon,
        progressive=strategy in {"progressive", "progressive_allow_lines"},
        progressive_allow_lines=strategy == "progressive_allow_lines",
    ).strip()


def _segment_references(targets: Mapping[str, _Target]) -> dict[str, tuple[object, np.ndarray]]:
    """Index retained command segments for cross-region path constraints."""
    references: dict[str, tuple[object, np.ndarray]] = {}
    for target in targets.values():
        assert target.region.current is not None
        if target.region.current.kind != "path":
            continue
        commands = svg_path_commands(str(target.region.current.params["d"]), target.id, path_ids=target.region.current.params.get("path_ids"))
        for index, command in enumerate(commands):
            if command.command in {"M", "Z"}:
                continue
            references[command.id] = (command, np.asarray(commands[index - 1].values[-2:], dtype=float))
    return references


def _action_settings(trace: TraceResult, plan: DrawingPlan, op: Mapping[str, object]) -> ActionSettings:
    """Resolve retained trace settings, plan settings, then one action override."""
    plan_settings = plan.settings
    action_override = op.get("settings", {})
    assert isinstance(action_override, Mapping)
    return ActionSettings(
        epsilon=float(
            action_override.get(
                "epsilon",
                plan_settings.epsilon
                if plan_settings.epsilon is not None
                else trace.options.simplify_tolerance,
            )
        ),
        max_error=float(
            action_override.get(
                "max_error",
                plan_settings.max_error
                if plan_settings.max_error is not None
                else trace.options.curve_tolerance,
            )
        ),
        fit_strategy=action_override.get(
            "fit_strategy",
            plan_settings.fit_strategy
            if plan_settings.fit_strategy is not None
            else trace.options.fit_strategy,
        ),
        purpose=action_override.get("purpose", plan_settings.purpose or "accuracy"),
        execution=action_override.get("execution", plan_settings.execution or "deterministic"),
    )


def resolved_action_settings(
    trace: TraceResult, plan: DrawingPlan, action: object
) -> ActionSettings:
    """Expose the one retained/plan/action settings resolution to recipe capture."""
    if not hasattr(action, "model_dump"):
        raise TypeError("drawing action must be a typed action contract")
    return _action_settings(trace, plan, action.model_dump(exclude_none=True))


def _fill(spec: Mapping[str, object]) -> Fill:
    """Project an assigned canonical paint payload into temporary working paint."""
    kind = spec["type"]
    if kind == "flat":
        return FlatFill(spec["color"])
    if kind == "linear_gradient":
        start, end = spec["start"], spec["end"]
        return LinearGradientFill(
            {"x1": start[0], "y1": start[1], "x2": end[0], "y2": end[1]},
            [(stop["offset"], stop["color"]) for stop in spec["stops"]],
        )
    if kind == "radial_gradient":
        center = spec["center"]
        geometry = {"cx": center[0], "cy": center[1], "r": spec["radius"]}
        focal = spec.get("focal_point")
        if focal is not None:
            geometry.update(fx=focal[0], fy=focal[1])
        return RadialGradientFill(
            geometry,
            [(stop["offset"], stop["color"]) for stop in spec["stops"]],
        )
    origin = spec["origin"]
    return RasterFill(
        {"x": origin[0], "y": origin[1], "w": spec["width"], "h": spec["height"]},
        spec["png_base64"],
    )


def _public_fill(fill: Fill | None) -> dict[str, object]:
    if isinstance(fill, FlatFill):
        return {"type": "flat", "color": fill.hex}
    if isinstance(fill, LinearGradientFill):
        return {"type": "linear_gradient", "geometry": dict(fill.geometry), "stops": list(fill.stops)}
    if isinstance(fill, RadialGradientFill):
        return {"type": "radial_gradient", "geometry": dict(fill.geometry), "stops": list(fill.stops)}
    if isinstance(fill, RasterFill):
        return {"type": "raster", "geometry": dict(fill.geometry)}
    return {"type": "none"}


def _public_diagnostics(value: object) -> object:
    """Make optimizer diagnostics safe to return in an MCP structured result."""
    if isinstance(value, Mapping):
        return {str(key): _public_diagnostics(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_public_diagnostics(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or type(value) in {bool, int, float, str}:
        return value
    return repr(value)

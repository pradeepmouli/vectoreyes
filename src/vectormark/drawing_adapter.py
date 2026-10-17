"""Temporary adapter from the Python optimizer's records to canonical drawings.

The native/public contract is :class:`Drawing`. ``VectorRegion`` remains the
Python engine's working record while it still owns raster masks and Skia
footprints. This module is the only sanctioned projection across that seam.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .candidate import (
    FlatFill as LegacyFlatFill,
    LinearGradientFill as LegacyLinearGradientFill,
    RadialGradientFill as LegacyRadialGradientFill,
    RasterFill as LegacyRasterFill,
)
from .drawing import Canvas, Drawing, DrawingRegion
from .drawing_fill import (
    FlatFill,
    GradientStop,
    LinearGradientFill,
    RadialGradientFill,
    RasterFill,
    compose_affine,
)
from .path_geometry import (
    AffineTransform,
    CirclePrimitive,
    CloneGeometry,
    CornerAnchor,
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
from .optimizer.vector_region import VectorRegion
from .optimizer.symmetry_materialize import index_region_paths, resolve_source_path


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
        if region.current.kind == "use":
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
    if kind == "circle":
        return PrimitiveGeometry(CirclePrimitive((params["cx"], params["cy"]), params["r"]))
    if kind == "ellipse":
        return PrimitiveGeometry(
            EllipsePrimitive((params["cx"], params["cy"]), params["rx"], params["ry"])
        )
    if kind == "rect":
        origin = (params["x"], params["y"])
        if "rx" in params or "ry" in params:
            return PrimitiveGeometry(
                RoundedRectanglePrimitive(
                    origin,
                    params["w"],
                    params["h"],
                    params.get("rx", params.get("ry", 0.0)),
                    params.get("ry", params.get("rx", 0.0)),
                )
            )
        return PrimitiveGeometry(RectanglePrimitive(origin, params["w"], params["h"]))
    if kind == "polygon":
        return PrimitiveGeometry(PolygonPrimitive(tuple(tuple(point) for point in params["points"])))
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
    anchors: list[list[CornerAnchor]] = [[] for _ in paths]
    owned_commands: list[set[str]] = [set() for _ in paths]
    for span in rebound["spans"]:
        subpath = int(span["subpath"])
        path_id = paths[subpath].path.id
        source_id = str(span["corner_id"])
        prefix = f"p{subpath}."
        local_id = source_id[len(prefix):] if source_id.startswith(prefix) else source_id
        command_id = lambda index: f"{path_id}.c{int(index)}"
        try:
            anchor = CornerAnchor(
                id=f"{path_id}.{local_id}",
                entry_command_id=command_id(span["entry_command"]),
                exit_command_id=command_id(span["exit_command"]),
                owned_command_ids=tuple(command_id(index) for index in span["commands"]),
                radius=span.get("radius"),
            )
            if owned_commands[subpath].intersection(anchor.owned_command_ids):
                rebound["skipped"].append({
                    "corner_id": source_id,
                    "reason": "overlapping_corner_ownership",
                })
                continue
            PathGeometry(paths[subpath].path, (*anchors[subpath], anchor))
            anchors[subpath].append(anchor)
            owned_commands[subpath].update(anchor.owned_command_ids)
        except (KeyError, TypeError, ValueError) as error:
            rebound["skipped"].append({
                "corner_id": source_id,
                "reason": "invalid_corner_anchor",
                "detail": str(error),
            })
    converted = [replace(path, anchors=tuple(anchor_list)) for path, anchor_list in zip(paths, anchors)]
    if isinstance(geometry, PathGeometry):
        return converted[0], rebound
    by_id = {path.path.id: path for path in converted}
    return replace_component_paths(geometry, tuple(by_id[path.path.id] for path in component_paths(geometry))), rebound


def _fill(fill: object):
    if isinstance(fill, LegacyFlatFill):
        return FlatFill(fill.hex)
    if isinstance(fill, LegacyLinearGradientFill):
        geometry = fill.geometry
        return LinearGradientFill(
            (geometry["x1"], geometry["y1"]),
            (geometry["x2"], geometry["y2"]),
            tuple(GradientStop(offset, color) for offset, color in fill.stops),
        )
    if isinstance(fill, LegacyRadialGradientFill):
        geometry = fill.geometry
        return RadialGradientFill(
            (geometry["cx"], geometry["cy"]),
            geometry["r"],
            tuple(GradientStop(offset, color) for offset, color in fill.stops),
            focal_point=(
                (geometry.get("fx", geometry["cx"]), geometry.get("fy", geometry["cy"]))
                if "fx" in geometry or "fy" in geometry else None
            ),
        )
    if isinstance(fill, LegacyRasterFill):
        geometry = fill.geometry
        return RasterFill(
            (geometry["x"], geometry["y"]),
            geometry["w"],
            geometry["h"],
            fill.png_b64,
        )
    return None

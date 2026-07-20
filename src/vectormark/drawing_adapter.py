"""Temporary adapter from the Python optimizer's records to canonical drawings.

The native/public contract is :class:`Drawing`. ``VectorRegion`` remains the
Python engine's working record while it still owns raster masks and Skia
footprints. This module is the only sanctioned projection across that seam.
"""

from __future__ import annotations

from collections.abc import Sequence

from .candidate import (
    FlatFill as LegacyFlatFill,
    LinearGradientFill as LegacyLinearGradientFill,
    RadialGradientFill as LegacyRadialGradientFill,
    RasterFill as LegacyRasterFill,
)
from .drawing import Canvas, Drawing, DrawingRegion
from .drawing_fill import FlatFill, GradientStop, LinearGradientFill, RadialGradientFill, RasterFill
from .path_geometry import (
    AffineTransform,
    CirclePrimitive,
    CloneGeometry,
    EllipsePrimitive,
    PolygonPrimitive,
    PrimitiveGeometry,
    RectanglePrimitive,
    RoundedRectanglePrimitive,
    geometry_from_svg_d,
)
from .optimizer.vector_region import VectorRegion


def drawing_from_vector_regions(
    drawing_id: str,
    *,
    width: float,
    height: float,
    regions: Sequence[VectorRegion],
) -> Drawing:
    """Project Python-engine working regions into the canonical scene model."""
    region_addresses: dict[int, str] = {}
    leaf_addresses: dict[int, str] = {}

    def index(region: VectorRegion, root_id: str, legacy_id: str) -> None:
        address = _public_region_id(region, root_id, legacy_id)
        region_addresses[id(region)] = address
        if region.is_leaf:
            if region.id in leaf_addresses and leaf_addresses[region.id] != address:
                raise ValueError("Python-engine leaves require unique internal IDs")
            leaf_addresses[region.id] = address
        for child in region.children:
            index(child, root_id, f"{legacy_id}-{child.id}")

    for root in regions:
        if not isinstance(root.drawing_id, str):
            raise ValueError("Python-engine roots require stable drawing IDs")
        index(root, root.drawing_id, root.drawing_id)

    def convert(region: VectorRegion, root_id: str, legacy_id: str) -> DrawingRegion:
        region_id = region_addresses[id(region)]
        common = {
            "id": region_id,
            "fill": _fill(region.fill),
            "z": float(region.z),
            "source_region_ids": tuple(region.source_regions or (region_id,)),
            "diagnostics": region.diagnostics,
        }
        if region.is_branch:
            return DrawingRegion(
                **common,
                children=tuple(
                    convert(child, root_id, f"{legacy_id}-{child.id}")
                    for child in region.children
                ),
            )
        assert region.current is not None
        return DrawingRegion(
            **common,
            geometry=_geometry(region.current.kind, region.current.params, region_id, leaf_addresses),
        )

    return Drawing(
        drawing_id=drawing_id,
        canvas=Canvas(float(width), float(height)),
        regions=tuple(
            convert(root, root.drawing_id, root.drawing_id)
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


def _geometry(kind: str, params: dict, region_id: str, public_ids: dict[int, str]):
    if kind == "path":
        fill_rule = "evenodd" if params.get("fill_rule") == "evenodd" else "nonzero"
        return geometry_from_svg_d(str(params["d"]), region_id=region_id, fill_rule=fill_rule)
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

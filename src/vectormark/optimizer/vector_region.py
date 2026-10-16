from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np

from ..candidate import Fill
from ..emit import shape_to_path_d
from ..fit import Shape
from ..skia_geometry import SkPath, unary_union
from ..types import CoverageField

_NUM = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PATH_TOKEN = re.compile(rf"[MLQCAZ]|{_NUM}")
_UNSET = object()


def _parse_subpaths(d: str) -> list[list[tuple[str, list[float]]]]:
    raw = _PATH_TOKEN.findall(d)
    if not raw:
        return []

    coord_count = {"M": 2, "L": 2, "Q": 4, "C": 6, "A": 7, "Z": 0}
    subpaths: list[list[tuple[str, list[float]]]] = []
    current: list[tuple[str, list[float]]] = []
    i = 0
    while i < len(raw):
        cmd = raw[i]
        if cmd not in coord_count:
            raise ValueError(f"expected path command, got {cmd!r}")
        i += 1
        count = coord_count[cmd]
        values = [float(token) for token in raw[i:i + count]]
        if len(values) != count:
            raise ValueError(f"incomplete SVG path command {cmd!r}")
        i += count
        if cmd == "M" and current:
            subpaths.append(current)
            current = []
        current.append((cmd, values))
    if current:
        subpaths.append(current)
    return subpaths


def _point_tuple(pt: np.ndarray) -> tuple[float, float]:
    return (float(pt[0]), float(pt[1]))


def _sample_arc(
    start: np.ndarray,
    values: list[float],
    samples: int,
) -> list[tuple[float, float]]:
    rx, ry, xrot_deg, large_arc, sweep, x2, y2 = values
    end = np.array([x2, y2], dtype=float)
    if np.allclose(start, end):
        return []

    rx = abs(rx)
    ry = abs(ry)
    if rx == 0.0 or ry == 0.0:
        return [_point_tuple(end)]

    phi = math.radians(xrot_deg % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    dx2 = (start[0] - end[0]) / 2.0
    dy2 = (start[1] - end[1]) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale

    rx2 = rx * rx
    ry2 = ry * ry
    x1p2 = x1p * x1p
    y1p2 = y1p * y1p

    denom = rx2 * y1p2 + ry2 * x1p2
    if denom == 0.0:
        return [_point_tuple(end)]
    numer = max(0.0, rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2)
    sign = -1.0 if bool(large_arc) == bool(sweep) else 1.0
    coef = sign * math.sqrt(numer / denom)
    cxp = coef * (rx * y1p) / ry
    cyp = coef * (-ry * x1p) / rx

    cx = cos_phi * cxp - sin_phi * cyp + (start[0] + end[0]) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (start[1] + end[1]) / 2.0

    ux = (x1p - cxp) / rx
    uy = (y1p - cyp) / ry
    vx = (-x1p - cxp) / rx
    vy = (-y1p - cyp) / ry

    theta1 = math.atan2(uy, ux)
    delta = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    if not sweep and delta > 0.0:
        delta -= 2.0 * math.pi
    elif sweep and delta < 0.0:
        delta += 2.0 * math.pi

    pts: list[tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, samples)[1:]:
        theta = theta1 + delta * float(t)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        x = cos_phi * rx * cos_theta - sin_phi * ry * sin_theta + cx
        y = sin_phi * rx * cos_theta + cos_phi * ry * sin_theta + cy
        pts.append((float(x), float(y)))
    return pts


def _ring_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + [points[0]], strict=False):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _sample_subpath(
    tokens: list[tuple[str, list[float]]],
    samples: int,
) -> list[tuple[float, float]]:
    cur = start = None
    pts: list[tuple[float, float]] = []
    for kind, values in tokens:
        if kind == "M":
            cur = np.array(values[:2], dtype=float)
            start = cur
            pts.append(_point_tuple(cur))
        elif kind == "L":
            cur = np.array(values[:2], dtype=float)
            pts.append(_point_tuple(cur))
        elif kind == "Q":
            assert cur is not None
            c = np.array(values[:2], dtype=float)
            p = np.array(values[2:4], dtype=float)
            for t in np.linspace(0, 1, samples)[1:]:
                pt = (1 - t) ** 2 * cur + 2 * (1 - t) * t * c + t**2 * p
                pts.append(_point_tuple(pt))
            cur = p
        elif kind == "C":
            assert cur is not None
            c1 = np.array(values[:2], dtype=float)
            c2 = np.array(values[2:4], dtype=float)
            p = np.array(values[4:6], dtype=float)
            for t in np.linspace(0, 1, samples)[1:]:
                pt = (
                    (1 - t) ** 3 * cur
                    + 3 * (1 - t) ** 2 * t * c1
                    + 3 * (1 - t) * t**2 * c2
                    + t**3 * p
                )
                pts.append(_point_tuple(pt))
            cur = p
        elif kind == "A":
            assert cur is not None
            pts.extend(_sample_arc(cur, values, samples))
            cur = np.array(values[5:7], dtype=float)
        elif kind == "Z":
            cur = start
    return pts


def flatten_points(shape: Shape, *, samples: int = 24) -> list[tuple[float, float]]:
    subpaths = _parse_subpaths(shape_to_path_d(shape))
    rings = [_sample_subpath(subpath, samples) for subpath in subpaths]
    rings = [ring for ring in rings if len(ring) >= 3]
    if not rings:
        return []
    return max(rings, key=_ring_area)


def to_polygon(shape: Shape, *, samples: int = 24) -> SkPath:
    """Convert a :class:`~vectormark.fit.Shape` to an :class:`SkPath` footprint.

    The *samples* parameter is kept for API compatibility but is no longer used
    (the Skia path represents curves exactly via the original SVG commands).
    """
    d = shape_to_path_d(shape)
    if not d:
        return SkPath()
    fill_rule = shape.params.get("fill_rule")
    return SkPath.from_svg_d(d, fill_rule=str(fill_rule) if fill_rule is not None else None)


@dataclass(frozen=True, init=False)
class VectorRegion:
    id: int
    raster: np.ndarray = field(compare=False)
    footprint: SkPath | object = field(compare=False)
    fill: Fill | None
    z: float
    original: Shape | None
    current: Shape | None
    children: tuple["VectorRegion", ...]
    source_label: int | None = None
    color_hex: str | None = None
    drawing_id: str | None = None
    source_regions: tuple[str, ...] = ()
    coverage: CoverageField | np.ndarray | None = field(default=None, compare=False)
    diagnostics: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __init__(
        self,
        id: int,
        current: Shape | None,
        fill: Fill | None,
        z: float = 0,
        footprint: SkPath | object | None = None,
        *,
        raster: np.ndarray | None = None,
        original: Shape | None = None,
        children: Sequence["VectorRegion"] = (),
        source_label: int | None = None,
        color_hex: str | None = None,
        drawing_id: str | None = None,
        source_regions: Sequence[str] = (),
        coverage: CoverageField | np.ndarray | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        child_tuple = tuple(children)
        if current is None and not child_tuple:
            raise TypeError("VectorRegion branch nodes require children")
        if current is not None and original is None:
            original = current
        if footprint is None:
            if current is not None:
                footprint = to_polygon(current)
            else:
                footprint = unary_union(
                    [child.footprint for child in child_tuple if isinstance(child.footprint, SkPath)]
                )
        if raster is None:
            raster = _union_child_rasters(child_tuple)

        object.__setattr__(self, "id", int(id))
        object.__setattr__(self, "raster", np.asarray(raster, dtype=bool).copy())
        object.__setattr__(self, "footprint", footprint)
        object.__setattr__(self, "fill", fill)
        object.__setattr__(self, "z", float(z))
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "children", child_tuple)
        object.__setattr__(self, "source_label", source_label)
        object.__setattr__(self, "color_hex", color_hex)
        object.__setattr__(self, "drawing_id", drawing_id)
        object.__setattr__(self, "source_regions", tuple(str(region) for region in source_regions))
        object.__setattr__(self, "coverage", coverage if isinstance(coverage, CoverageField) else None if coverage is None else np.asarray(coverage).copy())
        object.__setattr__(self, "diagnostics", dict(diagnostics or {}))

    @classmethod
    def from_shape(
        cls,
        *,
        id: int,
        shape: Shape,
        fill: Fill,
        z: float,
        raster: np.ndarray | None = None,
        footprint: SkPath | object | None = None,
        source_label: int | None = None,
        color_hex: str | None = None,
        drawing_id: str | None = None,
        source_regions: Sequence[str] = (),
        coverage: CoverageField | np.ndarray | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "VectorRegion":
        return cls(
            id=id,
            current=shape,
            fill=fill,
            z=z,
            raster=raster,
            footprint=footprint,
            original=shape,
            source_label=source_label,
            color_hex=color_hex,
            drawing_id=drawing_id,
            source_regions=source_regions,
            coverage=coverage,
            diagnostics=diagnostics,
        )

    @classmethod
    def branch(
        cls,
        *,
        id: int,
        children: Sequence["VectorRegion"],
        z: float = 0,
        raster: np.ndarray | None = None,
        footprint: SkPath | object | None = None,
        fill: Fill | None = None,
        source_label: int | None = None,
        color_hex: str | None = None,
        drawing_id: str | None = None,
        source_regions: Sequence[str] = (),
        diagnostics: Mapping[str, Any] | None = None,
        original: Shape | None = None,
    ) -> "VectorRegion":
        if not children:
            raise ValueError("branch VectorRegion requires at least one child")
        return cls(
            id=id,
            current=None,
            fill=fill,
            z=z,
            raster=raster,
            footprint=footprint,
            children=children,
            source_label=source_label,
            color_hex=color_hex,
            drawing_id=drawing_id,
            source_regions=source_regions,
            diagnostics=diagnostics,
            original=original,
        )

    @property
    def is_leaf(self) -> bool:
        return self.current is not None

    @property
    def is_branch(self) -> bool:
        return self.current is None

    def leaves(self) -> tuple["VectorRegion", ...]:
        if self.is_leaf:
            return (self,)
        leaves: list[VectorRegion] = []
        for child in self.children:
            leaves.extend(child.leaves())
        return tuple(leaves)

    def with_current(
        self,
        new_shape: Shape,
        *,
        footprint: SkPath | object | None = None,
        raster: np.ndarray | None = None,
        fill: Fill | None | object = _UNSET,
        z: float | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "VectorRegion":
        if not self.is_leaf:
            raise ValueError("branch VectorRegion does not have current geometry")
        merged_diagnostics = dict(self.diagnostics)
        if diagnostics:
            merged_diagnostics.update(diagnostics)
        return VectorRegion(
            id=self.id,
            current=new_shape,
            # ``None`` is a meaningful state for a geometry mutation: its old
            # fill has been invalidated and must pass through fill fitting again.
            # Omission retains the existing fill for optimizer-only updates.
            fill=self.fill if fill is _UNSET else fill,
            z=self.z if z is None else z,
            raster=self.raster if raster is None else raster,
            footprint=to_polygon(new_shape) if footprint is None else footprint,
            original=self.original,
            source_label=self.source_label,
            color_hex=self.color_hex,
            drawing_id=self.drawing_id,
            source_regions=self.source_regions,
            coverage=self.coverage,
            diagnostics=merged_diagnostics,
        )

    def with_diagnostics(
        self,
        diagnostics: Mapping[str, Any],
        *,
        raster: np.ndarray | None = None,
    ) -> "VectorRegion":
        merged_diagnostics = dict(self.diagnostics)
        merged_diagnostics.update(diagnostics)
        if self.is_leaf:
            assert self.current is not None
            return VectorRegion(
                id=self.id,
                current=self.current,
                fill=self.fill,
                z=self.z,
                raster=self.raster if raster is None else raster,
                footprint=self.footprint,
                original=self.original,
                source_label=self.source_label,
                color_hex=self.color_hex,
                drawing_id=self.drawing_id,
                source_regions=self.source_regions,
                coverage=self.coverage,
                diagnostics=merged_diagnostics,
            )
        return VectorRegion.branch(
            id=self.id,
            children=self.children,
            z=self.z,
            raster=self.raster if raster is None else raster,
            footprint=self.footprint,
            fill=self.fill,
            source_label=self.source_label,
            color_hex=self.color_hex,
            drawing_id=self.drawing_id,
            source_regions=self.source_regions,
            diagnostics=merged_diagnostics,
            original=self.original,
        )

    def with_children(
        self,
        children: Sequence["VectorRegion"],
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "VectorRegion":
        if not self.is_branch:
            raise ValueError("leaf VectorRegion does not have children")
        merged_diagnostics = dict(self.diagnostics)
        if diagnostics:
            merged_diagnostics.update(diagnostics)
        return VectorRegion.branch(
            id=self.id,
            children=children,
            z=self.z,
            raster=self.raster,
            footprint=self.footprint,
            fill=self.fill,
            source_label=self.source_label,
            color_hex=self.color_hex,
            drawing_id=self.drawing_id,
            source_regions=self.source_regions,
            diagnostics=merged_diagnostics,
            original=self.original,
        )


def _union_child_rasters(children: Sequence[VectorRegion]) -> np.ndarray:
    if not children:
        return np.zeros((0, 0), dtype=bool)
    raster = np.zeros_like(children[0].raster, dtype=bool)
    for child in children:
        if child.raster.shape == raster.shape:
            raster |= np.asarray(child.raster, dtype=bool)
    return raster


def leaves(regions: Sequence[VectorRegion]) -> list[VectorRegion]:
    out: list[VectorRegion] = []
    for region in regions:
        out.extend(region.leaves())
    return out

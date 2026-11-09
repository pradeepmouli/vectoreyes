"""Skia-backed replacement for shapely.geometry.{Polygon, MultiPolygon}.

Provides :class:`SkPath` — a drop-in for the Shapely Polygon / MultiPolygon
types used throughout the vectoreyes refinement — backed by ``skia-python`` path
operations (Simplify, Op, OpBuilder).  Geometric properties that Skia does not
expose directly (area, centroid, perimeter, OBB) are computed analytically on
the linearised path.

Public names that must match the Shapely interface consumed by vectoreyes:
    SkPath                      Polygon / MultiPolygon
    SkPath.area                 .area
    SkPath.length               .length  (perimeter)
    SkPath.bounds               .bounds  (minx, miny, maxx, maxy)
    SkPath.centroid             .centroid  (.x, .y)
    SkPath.is_empty             .is_empty
    SkPath.is_valid             .is_valid
    SkPath.exterior             .exterior  (.coords, .xy)
    SkPath.interiors            .interiors  (list of ring-like objects)
    SkPath.geoms                .geoms  (component polygons, MultiPolygon compat)
    SkPath.minimum_rotated_rectangle  .minimum_rotated_rectangle
    SkPath.buffer(0)            .buffer(0)  (normalise winding)
    SkPath.intersection(other)  .intersection
    SkPath.difference(other)    .difference
    SkPath.symmetric_difference(other)  .symmetric_difference
    unary_union(paths)          shapely.ops.unary_union
    affinity.affine_transform   shapely.affinity.affine_transform
    affinity.rotate             shapely.affinity.rotate
"""

from __future__ import annotations

from collections.abc import Iterable

import math
import re

import numpy as np
import skia

from ._boundary_distance import boundary_distance

# ---------------------------------------------------------------------------
# SVG path ↔ skia.Path conversion
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[MLQCAZ]|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_NUM_RE_WITH_A = re.compile(
    r"[MLQCAZA]|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)

# Verb codes returned by RawIter
_V_MOVE = 0
_V_LINE = 1
_V_QUAD = 2
_V_CONIC = 3
_V_CUBIC = 4
_V_CLOSE = 5
_V_DONE = 6
DEFAULT_CONIC_TO_QUADS_POW2 = 4


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def svg_d_to_skia(d: str, *, fill_rule: str | None = None) -> skia.Path:
    """Parse an SVG path *d* string into a :class:`skia.Path`.

    Supports M, L, Q, C, Z, and SVG elliptical-arc A commands.
    """
    path = skia.Path()
    if fill_rule == "evenodd":
        path.setFillType(skia.PathFillType.kEvenOdd)
    tokens = _NUM_RE_WITH_A.findall(d)
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        if cmd == "M":
            path.moveTo(float(tokens[i]), float(tokens[i + 1]))
            i += 2
        elif cmd == "L":
            path.lineTo(float(tokens[i]), float(tokens[i + 1]))
            i += 2
        elif cmd == "Q":
            path.quadTo(
                float(tokens[i]), float(tokens[i + 1]),
                float(tokens[i + 2]), float(tokens[i + 3]),
            )
            i += 4
        elif cmd == "C":
            path.cubicTo(
                float(tokens[i]), float(tokens[i + 1]),
                float(tokens[i + 2]), float(tokens[i + 3]),
                float(tokens[i + 4]), float(tokens[i + 5]),
            )
            i += 6
        elif cmd == "A":
            rx = float(tokens[i])
            ry = float(tokens[i + 1])
            x_rot = float(tokens[i + 2])
            large = bool(int(float(tokens[i + 3])))
            sweep = bool(int(float(tokens[i + 4])))
            x = float(tokens[i + 5])
            y = float(tokens[i + 6])
            arc_size = skia.Path.ArcSize.kLarge_ArcSize if large else skia.Path.ArcSize.kSmall_ArcSize
            direction = skia.PathDirection.kCW if sweep else skia.PathDirection.kCCW
            path.arcTo(rx, ry, x_rot, arc_size, direction, x, y)
            i += 7
        elif cmd == "Z":
            path.close()
    return path


def _path_api_svg_d(path: skia.Path) -> str | None:
    parse_path = getattr(skia, "ParsePath", None)
    to_svg_string = getattr(parse_path, "ToSVGString", None)
    if callable(to_svg_string):
        return str(to_svg_string(path))

    to_svg = getattr(path, "toSVGString", None)
    if callable(to_svg):
        raw = str(to_svg())
        match = re.search(r'\sd="([^"]+)"', raw)
        return match.group(1) if match else raw

    return None


def path_to_svg_string_faithful(path: skia.Path, pow2: int = DEFAULT_CONIC_TO_QUADS_POW2) -> str:
    """Serialise a :class:`skia.Path` to SVG path data.

    SVG has no rational quadratic command, so Skia conics are decomposed with
    ``Path.ConvertConicToQuads`` into ``2 ** pow2`` quadratic Bézier segments.
    """
    pow2 = int(np.clip(pow2, 1, 5))
    parts: list[str] = []
    iterator = skia.Path.Iter(path, False)
    while True:
        verb, pts = iterator.next()
        if verb == skia.Path.Verb.kDone_Verb:
            break
        if verb == skia.Path.Verb.kMove_Verb:
            parts.append(f"M{_fmt(pts[0].x())} {_fmt(pts[0].y())}")
        elif verb == skia.Path.Verb.kLine_Verb:
            parts.append(f"L{_fmt(pts[1].x())} {_fmt(pts[1].y())}")
        elif verb == skia.Path.Verb.kQuad_Verb:
            parts.append(
                f"Q{_fmt(pts[1].x())} {_fmt(pts[1].y())} "
                f"{_fmt(pts[2].x())} {_fmt(pts[2].y())}"
            )
        elif verb == skia.Path.Verb.kConic_Verb:
            quads = skia.Path.ConvertConicToQuads(
                pts[0],
                pts[1],
                pts[2],
                iterator.conicWeight(),
                pow2,
            )
            j = 0
            while j + 2 < len(quads):
                ctrl = quads[j + 1]
                end = quads[j + 2]
                parts.append(
                    f"Q{_fmt(ctrl.x())} {_fmt(ctrl.y())} "
                    f"{_fmt(end.x())} {_fmt(end.y())}"
                )
                j += 2
        elif verb == skia.Path.Verb.kCubic_Verb:
            parts.append(
                f"C{_fmt(pts[1].x())} {_fmt(pts[1].y())} "
                f"{_fmt(pts[2].x())} {_fmt(pts[2].y())} "
                f"{_fmt(pts[3].x())} {_fmt(pts[3].y())}"
            )
        elif verb == skia.Path.Verb.kClose_Verb:
            parts.append("Z")
    return " ".join(parts)


def skia_to_svg_d(path: skia.Path) -> str:
    """Serialise a :class:`skia.Path` back to an SVG *d* string (M/L/Q/C/Z)."""
    api_d = _path_api_svg_d(path)
    if api_d is not None:
        return api_d

    return path_to_svg_string_faithful(path)


def _skia_to_svg_d_raw(path: skia.Path) -> str:
    """Serialise via RawIter for low-level debugging."""
    parts: list[str] = []
    ri = skia.Path.RawIter(path)
    while True:
        verb, pts = ri.next()
        v = int(verb)
        if v == _V_DONE:
            break
        if v == _V_MOVE:
            parts.append(f"M{_fmt(pts[0].x())} {_fmt(pts[0].y())}")
        elif v == _V_LINE:
            parts.append(f"L{_fmt(pts[1].x())} {_fmt(pts[1].y())}")
        elif v == _V_QUAD:
            parts.append(
                f"Q{_fmt(pts[1].x())} {_fmt(pts[1].y())} "
                f"{_fmt(pts[2].x())} {_fmt(pts[2].y())}"
            )
        elif v == _V_CONIC:
            w = ri.conicWeight()
            p0 = pts[0]
            p1 = pts[1]
            p2 = pts[2]
            quads = skia.Path.ConvertConicToQuads(p0, p1, p2, w, 2)
            # quads is a flat list of Points: p0, c0, p1, c1, p2, ... (pairs for each quad)
            j = 0
            while j + 2 < len(quads):
                ctrl = quads[j + 1]
                end = quads[j + 2]
                parts.append(
                    f"Q{_fmt(ctrl.x())} {_fmt(ctrl.y())} "
                    f"{_fmt(end.x())} {_fmt(end.y())}"
                )
                j += 2
        elif v == _V_CUBIC:
            parts.append(
                f"C{_fmt(pts[1].x())} {_fmt(pts[1].y())} "
                f"{_fmt(pts[2].x())} {_fmt(pts[2].y())} "
                f"{_fmt(pts[3].x())} {_fmt(pts[3].y())}"
            )
        elif v == _V_CLOSE:
            parts.append("Z")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Path linearisation (sample curves into (x,y) point lists)
# ---------------------------------------------------------------------------

def _sample_quad(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    n: int,
) -> list[tuple[float, float]]:
    pts = []
    for i in range(1, n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        pts.append((float(x), float(y)))
    return pts


def _sample_cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    n: int,
) -> list[tuple[float, float]]:
    pts = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        x = mt ** 3 * p0[0] + 3 * mt ** 2 * t * p1[0] + 3 * mt * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = mt ** 3 * p0[1] + 3 * mt ** 2 * t * p1[1] + 3 * mt * t ** 2 * p2[1] + t ** 3 * p3[1]
        pts.append((float(x), float(y)))
    return pts


def _sample_conic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    w: float,
    n: int,
) -> list[tuple[float, float]]:
    """Sample a rational quadratic Bezier (conic) into *n* points."""
    pts = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1 - t
        denom = mt ** 2 + 2 * mt * t * w + t ** 2
        if abs(denom) < 1e-12:
            pts.append(p2)
            continue
        x = (mt ** 2 * p0[0] + 2 * mt * t * w * p1[0] + t ** 2 * p2[0]) / denom
        y = (mt ** 2 * p0[1] + 2 * mt * t * w * p1[1] + t ** 2 * p2[1]) / denom
        pts.append((float(x), float(y)))
    return pts


_CURVE_SAMPLES = 8


def _linearize_skia_path(
    path: skia.Path,
    curve_samples: int = _CURVE_SAMPLES,
) -> list[list[tuple[float, float]]]:
    """Return a list of subpaths; each subpath is a list of (x, y) tuples.

    Curves are sampled at *curve_samples* points each.  The returned rings do
    NOT repeat the first point at the end (use modular indexing for area etc.).
    """
    subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    start_pt: tuple[float, float] | None = None

    ri = skia.Path.RawIter(path)
    while True:
        verb, pts = ri.next()
        v = int(verb)
        if v == _V_DONE:
            break

        if v == _V_MOVE:
            if current:
                subpaths.append(current)
            p = (float(pts[0].x()), float(pts[0].y()))
            current = [p]
            start_pt = p

        elif v == _V_LINE:
            p = (float(pts[1].x()), float(pts[1].y()))
            current.append(p)

        elif v == _V_QUAD:
            p0 = (float(pts[0].x()), float(pts[0].y()))
            p1 = (float(pts[1].x()), float(pts[1].y()))
            p2 = (float(pts[2].x()), float(pts[2].y()))
            current.extend(_sample_quad(p0, p1, p2, curve_samples))

        elif v == _V_CONIC:
            w = ri.conicWeight()
            p0 = (float(pts[0].x()), float(pts[0].y()))
            p1 = (float(pts[1].x()), float(pts[1].y()))
            p2 = (float(pts[2].x()), float(pts[2].y()))
            current.extend(_sample_conic(p0, p1, p2, w, curve_samples))

        elif v == _V_CUBIC:
            p0 = (float(pts[0].x()), float(pts[0].y()))
            p1 = (float(pts[1].x()), float(pts[1].y()))
            p2 = (float(pts[2].x()), float(pts[2].y()))
            p3 = (float(pts[3].x()), float(pts[3].y()))
            current.extend(_sample_cubic(p0, p1, p2, p3, curve_samples))

        elif v == _V_CLOSE:
            # Closing line back to start_pt is implicit in every ring consumer.
            if current:
                subpaths.append(current)
            current = []
            start_pt = None

    if current:
        subpaths.append(current)

    return [sp for sp in subpaths if len(sp) >= 3]


# ---------------------------------------------------------------------------
# Geometric primitives on point rings
# ---------------------------------------------------------------------------

def _ring_signed_area(pts: list[tuple[float, float]]) -> float:
    """Signed shoelace area; positive = CCW in standard math coords (Y-up)."""
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _ring_area(pts: list[tuple[float, float]]) -> float:
    return abs(_ring_signed_area(pts))


def _ring_centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    area = _ring_signed_area(pts)
    if abs(area) < 1e-12:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))
    cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    denom = 6.0 * area
    return (float(cx / denom), float(cy / denom))


def _ring_perimeter(pts: list[tuple[float, float]]) -> float:
    total = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        total += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total


def _point_in_ring(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = point
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o: tuple, a: tuple, b: tuple) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _minimum_rotated_rectangle_pts(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Rotating-calipers minimum-area enclosing rectangle."""
    hull = _convex_hull(points)
    n = len(hull)
    if n < 2:
        return hull * 4 if hull else []

    min_area = float("inf")
    best: list[tuple[float, float]] = []

    for i in range(n):
        dx = hull[(i + 1) % n][0] - hull[i][0]
        dy = hull[(i + 1) % n][1] - hull[i][1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-9:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux

        projs = [px * ux + py * uy for px, py in hull]
        perps = [px * nx + py * ny for px, py in hull]
        min_p, max_p = min(projs), max(projs)
        min_n, max_n = min(perps), max(perps)

        area = (max_p - min_p) * (max_n - min_n)
        if area < min_area:
            min_area = area
            best = [
                (s * ux + t * nx, s * uy + t * ny)
                for s, t in [(min_p, min_n), (max_p, min_n), (max_p, max_n), (min_p, max_n)]
            ]

    return best


# ---------------------------------------------------------------------------
# Ring and centroid proxy types (Shapely-compatible interface)
# ---------------------------------------------------------------------------

class _Ring:
    """Shapely-compatible ring exposing ``.coords`` and ``.xy``."""

    __slots__ = ("_coords",)

    def __init__(self, coords: list[tuple[float, float]]) -> None:
        self._coords = coords

    @property
    def coords(self) -> list[tuple[float, float]]:
        return self._coords

    @property
    def xy(self) -> tuple[list[float], list[float]]:
        xs = [p[0] for p in self._coords]
        ys = [p[1] for p in self._coords]
        return xs, ys


class _Centroid:
    """Shapely-compatible centroid proxy with ``.x`` and ``.y``."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)


# ---------------------------------------------------------------------------
# SkPath — main class
# ---------------------------------------------------------------------------

class SkPath:
    """Skia-backed replacement for ``shapely.geometry.Polygon`` /
    ``shapely.geometry.MultiPolygon``.

    Can be constructed:

    * ``SkPath(shell, holes)``  — from point lists (like ``Polygon(coords, holes)``)
    * ``SkPath.from_skia(path)``  — from an existing :class:`skia.Path`
    * ``SkPath.from_svg_d(d)``  — from an SVG path *d* string
    """

    def __init__(
        self,
        shell: list[tuple[float, float]] | None = None,
        holes: list[list[tuple[float, float]]] | None = None,
    ) -> None:
        path = skia.Path()
        if shell is not None and len(shell) >= 2:
            path.moveTo(float(shell[0][0]), float(shell[0][1]))
            for x, y in shell[1:]:
                path.lineTo(float(x), float(y))
            path.close()
        if holes:
            for hole in holes:
                if len(hole) >= 2:
                    path.moveTo(float(hole[0][0]), float(hole[0][1]))
                    for x, y in hole[1:]:
                        path.lineTo(float(x), float(y))
                    path.close()
            path.setFillType(skia.PathFillType.kEvenOdd)
        self._path: skia.Path = path
        self._subpaths: list[list[tuple[float, float]]] | None = None

    @classmethod
    def from_skia(cls, path: skia.Path) -> "SkPath":
        obj = cls.__new__(cls)
        obj._path = path
        obj._subpaths = None
        return obj

    def to_skia(self) -> skia.Path:
        """Return a detached native copy without SVG coordinate rounding."""
        return skia.Path(self._path)

    @property
    def linearized_subpaths(self) -> list[list[tuple[float, float]]]:
        """Return the path as a list of linearised subpaths (list of vertex lists).

        Each subpath is a closed ring of ``(x, y)`` float tuples.  Curves are
        approximated by linear segments; straight-sided polygons are exact.
        """
        self._ensure_subpaths()
        return list(self._subpaths or [])

    @classmethod
    def make_circle(cls, cx: float, cy: float, r: float) -> "SkPath":
        """Create a circular path (analogue of ``Point(cx,cy).buffer(r)``)."""
        p = skia.Path()
        p.addCircle(float(cx), float(cy), float(r))
        return cls.from_skia(p)

    @classmethod
    def from_svg_d(cls, d: str, *, fill_rule: str | None = None) -> "SkPath":
        return cls.from_skia(svg_d_to_skia(d, fill_rule=fill_rule))

    # ------------------------------------------------------------------
    # Lazily-computed subpath cache
    # ------------------------------------------------------------------

    def _ensure_subpaths(self) -> None:
        if self._subpaths is None:
            self._subpaths = _linearize_skia_path(self._path)

    # ------------------------------------------------------------------
    # Core Shapely-compatible properties
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return self._path.isEmpty()

    @property
    def is_valid(self) -> bool:
        return not self.is_empty

    @property
    def area(self) -> float:
        if self.is_empty:
            return 0.0
        self._ensure_subpaths()
        assert self._subpaths is not None
        if not self._subpaths:
            return 0.0
        return self._net_area(self._subpaths)

    def _net_area(self, subpaths: list[list[tuple[float, float]]]) -> float:
        """Net area accounting for holes (even-odd containment)."""
        total = 0.0
        for index, ring in enumerate(subpaths):
            if not ring:
                continue
            depth = sum(
                _point_in_ring(ring[0], other)
                for other_index, other in enumerate(subpaths)
                if other_index != index
            )
            total += (-1.0 if depth % 2 else 1.0) * _ring_area(ring)
        return max(0.0, total)

    @property
    def length(self) -> float:
        """Total perimeter of all subpaths."""
        if self.is_empty:
            return 0.0
        self._ensure_subpaths()
        assert self._subpaths is not None
        return sum(_ring_perimeter(sp) for sp in self._subpaths)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy)."""
        if self.is_empty:
            return (0.0, 0.0, 0.0, 0.0)
        r = self._path.getBounds()
        return (float(r.left()), float(r.top()), float(r.right()), float(r.bottom()))

    @property
    def centroid(self) -> _Centroid:
        if self.is_empty:
            return _Centroid(0.0, 0.0)
        self._ensure_subpaths()
        assert self._subpaths is not None
        if not self._subpaths:
            return _Centroid(0.0, 0.0)
        weighted_x = weighted_y = signed_area = 0.0
        for index, ring in enumerate(self._subpaths):
            if not ring:
                continue
            depth = sum(
                _point_in_ring(ring[0], other)
                for other_index, other in enumerate(self._subpaths)
                if other_index != index
            )
            weight = (-1.0 if depth % 2 else 1.0) * _ring_area(ring)
            cx, cy = _ring_centroid(ring)
            weighted_x += weight * cx
            weighted_y += weight * cy
            signed_area += weight
        if abs(signed_area) < 1e-12:
            return _Centroid(0.0, 0.0)
        return _Centroid(weighted_x / signed_area, weighted_y / signed_area)

    @property
    def exterior(self) -> _Ring:
        """Exterior ring of the (largest) polygon component."""
        self._ensure_subpaths()
        assert self._subpaths is not None
        if not self._subpaths:
            return _Ring([])
        exterior = max(self._subpaths, key=_ring_area)
        # Shapely convention: first point repeated at end.
        coords = exterior + [exterior[0]] if exterior else exterior
        return _Ring(coords)

    @property
    def interiors(self) -> list[_Ring]:
        """Interior rings (holes) of the polygon."""
        self._ensure_subpaths()
        assert self._subpaths is not None
        if len(self._subpaths) < 2:
            return []
        ext_idx = max(range(len(self._subpaths)), key=lambda i: _ring_area(self._subpaths[i]))
        ext_ring = self._subpaths[ext_idx]
        result: list[_Ring] = []
        for i, sp in enumerate(self._subpaths):
            if i == ext_idx:
                continue
            c = _ring_centroid(sp)
            if _point_in_ring(c, ext_ring):
                coords = sp + [sp[0]] if sp else sp
                result.append(_Ring(coords))
        return result

    @property
    def geoms(self) -> list["SkPath"]:
        """Component polygons (MultiPolygon compatibility)."""
        self._ensure_subpaths()
        assert self._subpaths is not None
        if not self._subpaths:
            return []

        sorted_sp = sorted(self._subpaths, key=_ring_area, reverse=True)
        exteriors: list[int] = []
        holes_for: dict[int, list[int]] = {}

        for i, sp in enumerate(sorted_sp):
            c = _ring_centroid(sp)
            parent = next(
                (j for j in exteriors if _point_in_ring(c, sorted_sp[j])),
                None,
            )
            if parent is None:
                exteriors.append(i)
                holes_for[i] = []
            else:
                holes_for[parent].append(i)

        result: list[SkPath] = []
        for ext_i in exteriors:
            shell = sorted_sp[ext_i]
            hs = [sorted_sp[hi] for hi in holes_for.get(ext_i, [])]
            result.append(SkPath(shell=shell, holes=hs))
        return result if result else [self]

    @property
    def minimum_rotated_rectangle(self) -> "SkPath":
        """Minimum-area enclosing rectangle of all points."""
        if self.is_empty:
            return SkPath()
        self._ensure_subpaths()
        assert self._subpaths is not None
        all_pts: list[tuple[float, float]] = []
        for sp in self._subpaths:
            all_pts.extend(sp)
        if not all_pts:
            return SkPath()
        rect = _minimum_rotated_rectangle_pts(all_pts)
        return SkPath(shell=rect) if rect else SkPath()

    # ------------------------------------------------------------------
    # Boolean operations (backed by Skia PathOps)
    # ------------------------------------------------------------------

    def buffer(self, distance: float) -> "SkPath":
        """``buffer(0)`` normalises path winding using Simplify.

        Non-zero distances are not supported and return *self* unchanged.
        """
        if distance == 0:
            result = skia.Simplify(self._path)
            return SkPath.from_skia(result) if result is not None else SkPath()
        return self

    def outset(self, distance: float) -> "SkPath":
        """Return the path plus a round-joined exterior stroke of ``distance``.

        This is intentionally distinct from the Shapely-compatible ``buffer``
        shim above. Refinement construction passes use it to create a temporary
        overlap before a Skia boolean operation; callers should not treat it as
        a general-purpose offset-curve implementation.
        """
        if distance < 0:
            raise ValueError("SkPath.outset distance must be non-negative")
        if distance == 0 or self.is_empty:
            return self.buffer(0)
        paint = skia.Paint(
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=float(distance) * 2.0,
            StrokeJoin=skia.Paint.kRound_Join,
            StrokeCap=skia.Paint.kRound_Cap,
        )
        stroke = skia.Path()
        if not paint.getFillPath(self._path, stroke):
            return SkPath()
        source = skia.Simplify(self._path)
        normalized_stroke = skia.Simplify(stroke)
        builder = skia.OpBuilder()
        try:
            builder.add(source if source is not None else self._path, skia.kUnion_PathOp)
            builder.add(normalized_stroke if normalized_stroke is not None else stroke, skia.kUnion_PathOp)
            result = builder.resolve()
        except RuntimeError:
            return SkPath()
        if result is None:
            return SkPath()
        simplified = skia.Simplify(result)
        return SkPath.from_skia(simplified if simplified is not None else result)

    def filled_small_holes(self, max_area: float) -> "SkPath":
        """Return a copy with interior rings up to ``max_area`` filled."""
        if max_area < 0:
            raise ValueError("maximum hole area must be non-negative")
        holes: list[SkPath] = []
        for ring in self.interiors:
            hole = SkPath(shell=list(ring.coords))
            if hole.area <= max_area:
                holes.append(hole)
        if not holes:
            return self
        builder = skia.OpBuilder()
        try:
            builder.add(self._path, skia.kUnion_PathOp)
            for hole in holes:
                builder.add(hole._path, skia.kUnion_PathOp)
            result = builder.resolve()
        except RuntimeError:
            return self
        return SkPath.from_skia(result) if result is not None else self

    def intersection(self, other: "SkPath") -> "SkPath":
        result = skia.Op(self._path, other._path, skia.kIntersect_PathOp)
        return SkPath.from_skia(result) if result is not None else SkPath()

    def difference(self, other: "SkPath") -> "SkPath":
        result = skia.Op(self._path, other._path, skia.kDifference_PathOp)
        return SkPath.from_skia(result) if result is not None else SkPath()

    def symmetric_difference(self, other: "SkPath") -> "SkPath":
        result = skia.Op(self._path, other._path, skia.kXOR_PathOp)
        if result is None:
            return SkPath()
        # PathOps can represent an XOR as overlapping even-odd contours.  Its
        # fill is correct, but the compatibility ``area`` implementation
        # operates on contour containment and cannot measure that form.  A
        # Simplify resolves it into disjoint contours before callers use the
        # result as a residual or acceptance gate.
        simplified = skia.Simplify(result)
        return SkPath.from_skia(simplified if simplified is not None else result)

    def union(self, other: "SkPath") -> "SkPath":
        result = skia.Op(self._path, other._path, skia.kUnion_PathOp)
        return SkPath.from_skia(result) if result is not None else SkPath()

    def affine_transform(
        self,
        matrix: tuple[float, float, float, float, float, float],
    ) -> "SkPath":
        """Apply an SVG ``matrix(a,b,c,d,e,f)`` transform.

        SVG convention: ``x' = a·x + c·y + e``, ``y' = b·x + d·y + f``.
        """
        a, b, c, d, e, f = matrix
        # Skia setAffine([scaleX, skewY, skewX, scaleY, transX, transY]):
        #   x' = scaleX·x + skewX·y + transX  →  scaleX=a, skewX=c, transX=e
        #   y' = skewY·x  + scaleY·y + transY  →  skewY=b,  scaleY=d, transY=f
        m = skia.Matrix()
        m.setAffine([a, b, c, d, e, f])
        new_path = skia.Path(self._path)
        new_path.transform(m)
        return SkPath.from_skia(new_path)

    def to_svg_d(self) -> str:
        """Return the underlying Skia path as an SVG *d* string."""
        return skia_to_svg_d(self._path)

    @property
    def boundary(self) -> "SkPath":
        """The outline of the path (mirrors Shapely's ``.boundary`` for polygons)."""
        return self

    def distance(self, other: "SkPath") -> float:
        """Approximate minimum distance between the boundaries of *self* and *other*.

        Uses linearised vertices; exact for polygons and a close approximation
        for curved paths. Returns 0.0 if either path is empty.
        """
        self._ensure_subpaths()
        other._ensure_subpaths()
        return boundary_distance(self._subpaths, other._subpaths)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        return not self.is_empty

    def __repr__(self) -> str:
        return f"SkPath(area={self.area:.2f}, bounds={self.bounds})"

    def __getstate__(self) -> dict[str, object]:
        return {
            "d": skia_to_svg_d(self._path),
            "fill_type": int(self._path.getFillType()),
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self._path = svg_d_to_skia(str(state["d"]))
        self._path.setFillType(skia.PathFillType(int(state.get("fill_type", 0))))
        self._subpaths = None


# ---------------------------------------------------------------------------
# Module-level functions matching shapely.ops / shapely.affinity
# ---------------------------------------------------------------------------

def unary_union(paths: Iterable[SkPath]) -> SkPath:
    """Union all *paths* into one :class:`SkPath`."""
    non_empty = [p for p in paths if not p.is_empty]
    if not non_empty:
        return SkPath()
    if len(non_empty) == 1:
        # A union of one operand is an exact copy. Re-running OpBuilder on
        # its own even-odd result can drop counters on successive unions.
        # Keep detached ownership without changing the path or its fill type.
        return SkPath.from_skia(skia.Path(non_empty[0]._path))
    builder = skia.OpBuilder()
    for p in non_empty:
        builder.add(p._path, skia.kUnion_PathOp)
    result = builder.resolve()
    return SkPath.from_skia(result) if result is not None else SkPath()


class affinity:  # noqa: N801 — matching Shapely's module-level namespace
    """Shapely-compatible affine-transform helpers."""

    @staticmethod
    def affine_transform(
        flat: SkPath,
        matrix: list[float],
    ) -> SkPath:
        """Apply a transform in Shapely's ``[a, b, d, e, xoff, yoff]`` convention.

        Shapely: ``x' = a·x + b·y + xoff``, ``y' = d·x + e·y + yoff``.
        """
        a, b, d, e, xoff, yoff = matrix
        # Convert to SVG matrix (A,B,C,D,E,F): x'=A·x+C·y+E, y'=B·x+D·y+F
        # → A=a, C=b, E=xoff, B=d, D=e, F=yoff
        return flat.affine_transform((a, d, b, e, xoff, yoff))

    @staticmethod
    def rotate(
        flat: SkPath,
        angle: float,
        origin: tuple[float, float] | str = "center",
    ) -> SkPath:
        """Rotate *flat* by *angle* degrees around *origin*.

        *origin* may be a ``(x, y)`` tuple or the string ``"center"``
        (uses the centroid of *flat*).
        """
        if origin == "center":
            c = flat.centroid
            ox, oy = float(c.x), float(c.y)
        else:
            ox, oy = float(origin[0]), float(origin[1])
        theta = math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        # SVG matrix: x'=cos·x - sin·y + (1-cos)·ox+sin·oy
        #             y'=sin·x + cos·y + (1-cos)·oy-sin·ox
        a = cos_t
        b = sin_t
        c_val = -sin_t
        d = cos_t
        e = ox - cos_t * ox + sin_t * oy
        f = oy - sin_t * ox - cos_t * oy
        return flat.affine_transform((a, b, c_val, d, e, f))

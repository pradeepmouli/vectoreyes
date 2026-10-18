from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..._fitcurve import cbezier, cubic_inflects, fit_quadratic_beziers
from ...fit import Shape, _curve_controls_are_straight, _fmt
from ..automatic_replay import (
    ApplySeamConstraintsOperation,
    AxisSeamConstraint,
    EndpointAssignment,
    EndpointConstraintStage,
    PassInputRegionRef,
    PathEndpointRef,
    PathOpsUnderlapConstraint,
    SeamCommandIndexCorrespondence,
    SeamPathCorrespondence,
    VertexClusterConstraint,
    operation_intent,
)
from ..framework import Proposal
from ...drawing_adapter import transform_working_shape
from ..symmetry_materialize import index_region_paths
from ..vector_region import VectorRegion, _parse_subpaths, to_polygon
from .simplify import _simplified_path_shape


# Trace fitting can move the endpoints of an otherwise clearly shared edge a
# little over two pixels.  This is only an *admission* tolerance: accepted
# seams are still projected to one exact coordinate.
_SEAM_TOL = 3.0
_JUNCTION_TOL = 6.5
_MIN_SUBPATH_SPAN = 3.0
_PATHOPS_UNION_RESIDUAL_TOL = 0.02
_PATHOPS_MAX_INPUT_COMMANDS = 64


@dataclass(frozen=True)
class _SeamCandidate:
    axis: str
    a_side: str
    b_side: str
    a_value: float
    b_value: float
    seam: float
    span_min: float
    span_max: float


@dataclass(frozen=True)
class _LeafRef:
    owner: VectorRegion
    leaf: VectorRegion


@dataclass(frozen=True)
class _LeafUpdate:
    shape: Shape
    paired_with: int
    candidate: _SeamCandidate
    selected: str


@dataclass(frozen=True)
class _CoordinateRef:
    key: tuple[int, int]
    subpath_index: int
    token_index: int
    x_index: int
    y_index: int
    x: float
    y: float


@dataclass(frozen=True)
class _EndpointMatch:
    left: tuple[int, int, int, int, float, float]
    right: tuple[int, int, int, int, float, float]
    point: tuple[float, float]


def _path_id(shape: Shape) -> str:
    if shape.kind != "path":
        raise ValueError("seam endpoint binding requires a path")
    import hashlib

    payload = str(shape.params.get("d", "")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _command_id(
    subpath_index: int,
    token_index: int,
    command: str,
    values: list[float],
) -> str:
    import hashlib

    encoded = ":".join([
        str(subpath_index),
        str(token_index),
        command,
        *(float(value).hex() for value in values),
    ])
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _subpath_revision(subpath: list[tuple[str, list[float]]]) -> str:
    import hashlib

    return hashlib.sha256(_subpath_d(subpath).encode("utf-8")).hexdigest()


def seam_path_revision(d: str, subpath_index: int) -> str:
    """Return the executor's exact revision for one ordered path component."""
    paths = _parse_subpaths(d)
    if not 0 <= subpath_index < len(paths):
        raise ValueError("seam path revision subpath is out of range")
    return _subpath_revision(paths[subpath_index])


def _drawable_tokens(
    subpath: list[tuple[str, list[float]]],
) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (command, tuple(float(value).hex() for value in values))
        for command, values in subpath
        if command != "Z"
    ]


def _unchanged_command_pairs(
    before: list[tuple[str, tuple[str, ...]]],
    after: list[tuple[str, tuple[str, ...]]],
) -> tuple[SeamCommandIndexCorrespondence, ...]:
    """Return an exact LCS of byte-identical commands, never a geometric guess."""
    lengths = [[0] * (len(after) + 1) for _ in range(len(before) + 1)]
    for left in range(len(before) - 1, -1, -1):
        for right in range(len(after) - 1, -1, -1):
            lengths[left][right] = (
                1 + lengths[left + 1][right + 1]
                if before[left] == after[right]
                else max(lengths[left + 1][right], lengths[left][right + 1])
            )
    pairs: list[SeamCommandIndexCorrespondence] = []
    left = right = 0
    while left < len(before) and right < len(after):
        if before[left] == after[right]:
            pairs.append(SeamCommandIndexCorrespondence(left, right))
            left += 1
            right += 1
        elif lengths[left + 1][right] >= lengths[left][right + 1]:
            left += 1
        else:
            right += 1
    return tuple(pairs)


def _path_correspondence(
    target: PassInputRegionRef,
    before: Shape,
    after: Shape,
    *,
    positional_mutation: bool,
) -> tuple[SeamPathCorrespondence, ...]:
    before_paths = _parse_subpaths(str(before.params.get("d", "")))
    after_paths = _parse_subpaths(str(after.params.get("d", "")))
    records: list[SeamPathCorrespondence] = []
    for index, before_path in enumerate(before_paths):
        after_path = after_paths[index] if index < len(after_paths) else None
        before_tokens = _drawable_tokens(before_path)
        after_tokens = [] if after_path is None else _drawable_tokens(after_path)
        if (
            positional_mutation
            and len(before_tokens) == len(after_tokens)
            and [item[0] for item in before_tokens]
            == [item[0] for item in after_tokens]
        ):
            commands = tuple(
                SeamCommandIndexCorrespondence(command_index, command_index)
                for command_index in range(len(before_tokens))
            )
        else:
            commands = _unchanged_command_pairs(before_tokens, after_tokens)
        records.append(SeamPathCorrespondence(
            target=target,
            subpath_index=index,
            input_path_revision=_subpath_revision(before_path),
            output_path_revision=(
                _subpath_revision(after_path) if after_path is not None else ""
            ),
            commands=commands,
        ))
    return tuple(records)


def _endpoint_ref(
    region: PassInputRegionRef,
    shape: Shape,
    raw: tuple[int, int, int, int, float, float] | _CoordinateRef,
) -> PathEndpointRef:
    tokens = _parse_subpaths(str(shape.params.get("d", "")))
    if isinstance(raw, _CoordinateRef):
        subpath_index, token_index = raw.subpath_index, raw.token_index
    else:
        subpath_index, token_index = raw[0], raw[1]
    command, values = tokens[subpath_index][token_index]
    return PathEndpointRef(
        region=region,
        path_id=_path_id(shape),
        command_id=_command_id(subpath_index, token_index, command, values),
    )


def _endpoint_stage(
    left_ref: PassInputRegionRef,
    right_ref: PassInputRegionRef,
    left_shape: Shape,
    right_shape: Shape,
    matches: list[_EndpointMatch],
) -> EndpointConstraintStage | None:
    if not matches:
        return None
    return EndpointConstraintStage(tuple(
        EndpointAssignment(
            members=(
                _endpoint_ref(left_ref, left_shape, match.left),
                _endpoint_ref(right_ref, right_shape, match.right),
            ),
            point=match.point,
        )
        for match in matches
    ))


def _shape_bounds(obj: VectorRegion) -> tuple[float, float, float, float] | None:
    if obj.footprint is None or getattr(obj.footprint, "is_empty", False):
        return None
    minx, miny, maxx, maxy = obj.footprint.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _same_surface_fill_group(a: VectorRegion, b: VectorRegion) -> bool:
    """Whether automatic stitching may alter this pair's shared boundary.

    Trace/refine assigns a ``surface_fill`` group after geometry fitting.  Its
    members are palette fragments of one material; neighbouring members of a
    different group can instead be a deliberate counter or overlaid detail.
    Legacy/manual regions carry no group and retain the previous permissive
    behavior.
    """
    # A compound child represents a nested fill/cutout layer from one source
    # path.  Its boundary is containment, not an independently traced material
    # seam.  It must never be moved toward a sibling by generic seam repair.
    if isinstance(a.diagnostics.get("compound"), dict) or isinstance(b.diagnostics.get("compound"), dict):
        return False
    a_fill = a.diagnostics.get("surface_fill")
    b_fill = b.diagnostics.get("surface_fill")
    if not isinstance(a_fill, dict) or not isinstance(b_fill, dict):
        return True
    return a_fill.get("group") == b_fill.get("group")


def _regions_close(a: VectorRegion, b: VectorRegion, *, tol: float) -> bool:
    """Cheaply reject separated bounds before asking Skia for exact distance.

    ``SkPath.distance`` is intentionally exact, but it is disproportionately
    expensive for the large, detailed paths retained by an interactive trace.
    The Euclidean distance between two axis-aligned bounding boxes is a lower
    bound on the path distance, so a bound that exceeds ``tol`` can never
    contain a seam candidate.  Nearby bounds still use the exact geometry
    check below.
    """
    a_bounds = _shape_bounds(a)
    b_bounds = _shape_bounds(b)
    if a_bounds is None or b_bounds is None:
        return False
    ax0, ay0, ax1, ay1 = a_bounds
    bx0, by0, bx1, by1 = b_bounds
    dx = max(ax0 - bx1, bx0 - ax1, 0.0)
    dy = max(ay0 - by1, by0 - ay1, 0.0)
    if dx * dx + dy * dy > tol * tol:
        return False
    if (
        a.current is not None
        and b.current is not None
        and a.current.kind == "path"
        and b.current.kind == "path"
    ):
        # Endpoint proximity is the actionable criterion for a path seam:
        # subsequent rewriting can move only those coordinates.  Avoid an
        # exact path-to-path distance here; it is costly for detailed,
        # overlapping traced paths and contributes no additional signal.
        return _path_endpoints_within(a.current, b.current, tol=tol)
    try:
        return float(a.footprint.distance(b.footprint)) <= tol
    except Exception:
        return False


def _overlap(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float] | None:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi <= lo:
        return None
    return float(lo), float(hi)


def _candidate_for_pair(a: VectorRegion, b: VectorRegion, *, tol: float) -> _SeamCandidate | None:
    if a.current is None or b.current is None:
        return None
    if a.current.kind == "use" or b.current.kind == "use":
        return None
    if not _regions_close(a, b, tol=tol):
        return None

    a_bounds = _shape_bounds(a)
    b_bounds = _shape_bounds(b)
    if a_bounds is None or b_bounds is None:
        return None
    ax0, ay0, ax1, ay1 = a_bounds
    bx0, by0, bx1, by1 = b_bounds

    y_span = _overlap(ay0, ay1, by0, by1)
    if y_span is not None:
        if 0.0 < bx0 - ax1 <= tol:
            seam = (ax1 + bx0) / 2.0
            return _SeamCandidate("x", "right", "left", ax1, bx0, seam, *y_span)
        if 0.0 < ax0 - bx1 <= tol:
            seam = (bx1 + ax0) / 2.0
            return _SeamCandidate("x", "left", "right", ax0, bx1, seam, *y_span)

    x_span = _overlap(ax0, ax1, bx0, bx1)
    if x_span is not None:
        if 0.0 < by0 - ay1 <= tol:
            seam = (ay1 + by0) / 2.0
            return _SeamCandidate("y", "bottom", "top", ay1, by0, seam, *x_span)
        if 0.0 < ay0 - by1 <= tol:
            seam = (by1 + ay0) / 2.0
            return _SeamCandidate("y", "top", "bottom", ay0, by1, seam, *x_span)
    if a.current.kind != "path" or b.current.kind != "path":
        return None
    return _SeamCandidate("vertices", "", "", 0.0, 0.0, 0.0, 0.0, 0.0)


def _subpath_d(tokens: list[tuple[str, list[float]]]) -> str:
    parts: list[str] = []
    for command, values in tokens:
        if values:
            parts.append(f"{command}{' '.join(_fmt(value) for value in values)}")
        else:
            parts.append(command)
    return " ".join(parts)


def _rewrite_path_side(
    shape: Shape,
    *,
    axis: str,
    side_value: float,
    seam: float,
    span_min: float,
    span_max: float,
    tol: float,
) -> Shape | None:
    if shape.kind != "path":
        return None
    changed = False
    subpaths = _parse_subpaths(str(shape.params.get("d", "")))
    rewritten: list[list[tuple[str, list[float]]]] = []
    for subpath in subpaths:
        out_subpath: list[tuple[str, list[float]]] = []
        for command, values in subpath:
            new_values = list(values)
            coord_indexes: tuple[tuple[int, int], ...]
            if command in {"M", "L"}:
                coord_indexes = ((0, 1),)
            elif command == "Q":
                coord_indexes = ((2, 3),)
            elif command == "C":
                coord_indexes = ((4, 5),)
            elif command == "A":
                coord_indexes = ((5, 6),)
            else:
                coord_indexes = ()
            for x_idx, y_idx in coord_indexes:
                x = float(new_values[x_idx])
                y = float(new_values[y_idx])
                if axis == "x":
                    if abs(x - side_value) <= tol and span_min - tol <= y <= span_max + tol:
                        new_values[x_idx] = seam
                        changed = True
                elif abs(y - side_value) <= tol and span_min - tol <= x <= span_max + tol:
                    new_values[y_idx] = seam
                    changed = True
            out_subpath.append((command, new_values))
        rewritten.append(out_subpath)
    if not changed:
        return None

    params = dict(shape.params)
    params["d"] = " ".join(_subpath_d(subpath) for subpath in rewritten)
    return Shape("path", params)


def _rewrite_rect_side(shape: Shape, *, side: str, seam: float) -> Shape | None:
    if shape.kind != "rect":
        return None
    params = dict(shape.params)
    width_key = "w" if "w" in params else "width"
    height_key = "h" if "h" in params else "height"
    x = float(params["x"])
    y = float(params["y"])
    w = float(params[width_key])
    h = float(params[height_key])
    if side == "left":
        right = x + w
        w = right - seam
        x = seam
    elif side == "right":
        w = seam - x
    elif side == "top":
        bottom = y + h
        h = bottom - seam
        y = seam
    elif side == "bottom":
        h = seam - y
    else:
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    params.update({"x": x, "y": y, width_key: w, height_key: h})
    return Shape("rect", params)


def _rewrite_ellipse_side(shape: Shape, *, side: str, seam: float) -> Shape | None:
    if shape.kind not in {"circle", "ellipse"}:
        return None
    params = dict(shape.params)
    cx = float(params["cx"])
    cy = float(params["cy"])
    if shape.kind == "circle":
        r = float(params["r"])
        if side == "left":
            right = cx + r
            r = (right - seam) / 2.0
            cx = (right + seam) / 2.0
        elif side == "right":
            left = cx - r
            r = (seam - left) / 2.0
            cx = (left + seam) / 2.0
        elif side == "top":
            bottom = cy + r
            r = (bottom - seam) / 2.0
            cy = (bottom + seam) / 2.0
        elif side == "bottom":
            top = cy - r
            r = (seam - top) / 2.0
            cy = (top + seam) / 2.0
        else:
            return None
        if r <= 0.0:
            return None
        params.update({"cx": cx, "cy": cy, "r": r})
        return Shape("circle", params)

    rx = float(params["rx"])
    ry = float(params["ry"])
    if side == "left":
        right = cx + rx
        rx = (right - seam) / 2.0
        cx = (right + seam) / 2.0
    elif side == "right":
        left = cx - rx
        rx = (seam - left) / 2.0
        cx = (left + seam) / 2.0
    elif side == "top":
        bottom = cy + ry
        ry = (bottom - seam) / 2.0
        cy = (bottom + seam) / 2.0
    elif side == "bottom":
        top = cy - ry
        ry = (seam - top) / 2.0
        cy = (top + seam) / 2.0
    else:
        return None
    if rx <= 0.0 or ry <= 0.0:
        return None
    params.update({"cx": cx, "cy": cy, "rx": rx, "ry": ry})
    return Shape("ellipse", params)


def _rewrite_shape_side(
    shape: Shape,
    *,
    side: str,
    candidate: _SeamCandidate,
    side_value: float,
    tol: float,
) -> Shape | None:
    if shape.kind == "use":
        return None
    if candidate.axis == "vertices":
        return shape
    path = _rewrite_path_side(
        shape,
        axis=candidate.axis,
        side_value=side_value,
        seam=candidate.seam,
        span_min=candidate.span_min,
        span_max=candidate.span_max,
        tol=tol,
    )
    if path is not None:
        return path
    primitive = _rewrite_rect_side(shape, side=side, seam=candidate.seam)
    if primitive is not None:
        return primitive
    return _rewrite_ellipse_side(shape, side=side, seam=candidate.seam)


def _coordinate_indexes(command: str) -> tuple[tuple[int, int], ...]:
    if command in {"M", "L"}:
        return ((0, 1),)
    if command == "Q":
        return ((2, 3),)
    if command == "C":
        return ((4, 5),)
    if command == "A":
        return ((5, 6),)
    return ()


def _subpath_is_degenerate(subpath: list[tuple[str, list[float]]]) -> bool:
    """Ignore tiny AA-fringe loops when finding a material seam.

    These loops are trace provenance, not meaningful boundary segments.  If
    they participate in nearest-point matching, they can consume a match that
    belongs to the actual shared line of the containing path.
    """
    points = [
        (float(values[x_idx]), float(values[y_idx]))
        for command, values in subpath
        for x_idx, y_idx in _coordinate_indexes(command)
    ]
    if len(points) < 2:
        return True
    xs, ys = zip(*points, strict=True)
    return max(xs) - min(xs) < _MIN_SUBPATH_SPAN and max(ys) - min(ys) < _MIN_SUBPATH_SPAN


def _path_coordinate_refs(tokens: list[list[tuple[str, list[float]]]]):
    refs = []
    for subpath_index, subpath in enumerate(tokens):
        if _subpath_is_degenerate(subpath):
            continue
        for token_index, (command, values) in enumerate(subpath):
            for x_idx, y_idx in _coordinate_indexes(command):
                refs.append((subpath_index, token_index, x_idx, y_idx, float(values[x_idx]), float(values[y_idx])))
    return refs


def _path_endpoints_within(a: Shape, b: Shape, *, tol: float) -> bool:
    """Return whether two paths have endpoints within *tol* in expected O(n+m).

    This is a broad-phase filter for detailed raw trace paths.  A seam rewrite
    can only move path endpoints, so a pair with no nearby endpoints cannot
    produce a seam candidate.  The caller still asks Skia for exact distance
    once this filter admits a pair.
    """
    if tol < 0:
        return False
    cell_size = max(float(tol), 1e-9)
    a_refs = _path_coordinate_refs(_parse_subpaths(str(a.params.get("d", ""))))
    b_refs = _path_coordinate_refs(_parse_subpaths(str(b.params.get("d", ""))))
    if not a_refs or not b_refs:
        return False

    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for *_indices, x, y in b_refs:
        key = (math.floor(x / cell_size), math.floor(y / cell_size))
        buckets.setdefault(key, []).append((x, y))

    tol_sq = tol * tol
    for *_indices, x, y in a_refs:
        cell_x = math.floor(x / cell_size)
        cell_y = math.floor(y / cell_size)
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for other_x, other_y in buckets.get((cell_x + offset_x, cell_y + offset_y), []):
                    dx = x - other_x
                    dy = y - other_y
                    if dx * dx + dy * dy <= tol_sq:
                        return True
    return False


def _path_endpoint_refs(
    key: tuple[int, int],
    tokens: list[list[tuple[str, list[float]]]],
) -> list[_CoordinateRef]:
    refs: list[_CoordinateRef] = []
    for subpath_index, subpath in enumerate(tokens):
        if _subpath_is_degenerate(subpath):
            continue
        for token_index, (command, values) in enumerate(subpath):
            for x_idx, y_idx in _coordinate_indexes(command):
                refs.append(
                    _CoordinateRef(
                        key,
                        subpath_index,
                        token_index,
                        x_idx,
                        y_idx,
                        float(values[x_idx]),
                        float(values[y_idx]),
                    )
                )
    return refs


def _shape_from_tokens(shape: Shape, tokens: list[list[tuple[str, list[float]]]]) -> Shape:
    params = dict(shape.params)
    params["d"] = " ".join(_subpath_d(subpath) for subpath in tokens)
    return Shape("path", params)


def _cleanup_linelets(
    shape: Shape,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape:
    simplified = _simplified_path_shape(
        shape,
        epsilon=epsilon,
        max_error=max_error,
        samples=16,
        cubic=cubic,
        linelet_only=True,
    )
    return simplified or shape


def _cleanup_inflecting_cubics(shape: Shape, *, max_error: float, line_epsilon: float) -> Shape:
    if shape.kind != "path":
        return shape
    subpaths = _parse_subpaths(str(shape.params.get("d", "")))
    if not subpaths:
        return shape
    changed = False
    rewritten: list[list[tuple[str, list[float]]]] = []
    for subpath in subpaths:
        current: np.ndarray | None = None
        start: np.ndarray | None = None
        out_subpath: list[tuple[str, list[float]]] = []
        for command, values in subpath:
            if command == "M":
                current = np.array(values[:2], dtype=float)
                start = current.copy()
                out_subpath.append((command, list(values)))
                continue
            if command == "Z":
                current = start
                out_subpath.append((command, list(values)))
                continue
            if command == "C" and current is not None:
                ctrl = np.array(
                    [
                        current,
                        values[:2],
                        values[2:4],
                        values[4:6],
                    ],
                    dtype=float,
                )
                if cubic_inflects(ctrl):
                    samples = np.asarray([cbezier(ctrl, t) for t in np.linspace(0.0, 1.0, 9)], dtype=float)
                    for bezier in fit_quadratic_beziers(samples, max_error):
                        if _curve_controls_are_straight(bezier[0], [bezier[1]], bezier[2], line_epsilon):
                            out_subpath.append(("L", [float(bezier[2][0]), float(bezier[2][1])]))
                        else:
                            out_subpath.append(
                                (
                                    "Q",
                                    [
                                        float(bezier[1][0]),
                                        float(bezier[1][1]),
                                        float(bezier[2][0]),
                                        float(bezier[2][1]),
                                    ],
                                )
                            )
                    current = ctrl[3]
                    changed = True
                    continue
            out_subpath.append((command, list(values)))
            if command == "L":
                current = np.array(values[:2], dtype=float)
            elif command == "Q":
                current = np.array(values[2:4], dtype=float)
            elif command == "C":
                current = np.array(values[4:6], dtype=float)
            elif command == "A":
                current = np.array(values[5:7], dtype=float)
        rewritten.append(out_subpath)
    if not changed:
        return shape
    return _shape_from_tokens(shape, rewritten)


def _cleanup_mutated_path(
    shape: Shape,
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> Shape:
    cleaned = _cleanup_linelets(shape, epsilon=epsilon, max_error=max_error, cubic=cubic)
    return _cleanup_inflecting_cubics(cleaned, max_error=max_error, line_epsilon=epsilon)


def _true_up_path_vertices(
    a_shape: Shape,
    b_shape: Shape,
    *,
    tol: float,
    _matches: list[_EndpointMatch] | None = None,
) -> tuple[Shape, Shape, bool]:
    if a_shape.kind != "path" or b_shape.kind != "path":
        return a_shape, b_shape, False

    a_tokens = _parse_subpaths(str(a_shape.params.get("d", "")))
    b_tokens = _parse_subpaths(str(b_shape.params.get("d", "")))
    a_refs = _path_coordinate_refs(a_tokens)
    b_refs = _path_coordinate_refs(b_tokens)
    if not a_refs or not b_refs:
        return a_shape, b_shape, False

    cell_size = max(float(tol), 1e-9)
    buckets: dict[tuple[int, int], list[tuple]] = {}
    for ref in b_refs:
        cell = (math.floor(ref[4] / cell_size), math.floor(ref[5] / cell_size))
        buckets.setdefault(cell, []).append(ref)

    # Greedily select the nearest unmatched local counterpart.  The old global
    # pair sort was quadratic and only existed to establish the same one-to-one
    # vertex relation; local nearest matching is deterministic in path order
    # and bounds work by nearby vertices rather than all path commands.
    changed = False
    used_b: set[tuple[int, int, int, int]] = set()
    for a_ref in a_refs:
        cell_x = math.floor(a_ref[4] / cell_size)
        cell_y = math.floor(a_ref[5] / cell_size)
        best: tuple[float, tuple] | None = None
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                for b_ref in buckets.get((cell_x + dx_cell, cell_y + dy_cell), []):
                    b_key = tuple(b_ref[:4])
                    if b_key in used_b:
                        continue
                    dist = math.hypot(a_ref[4] - b_ref[4], a_ref[5] - b_ref[5])
                    if 0.0 < dist <= tol and (best is None or dist < best[0]):
                        best = (dist, b_ref)
        if best is None:
            continue
        _dist, b_ref = best
        b_key = tuple(b_ref[:4])
        midpoint_x = (a_ref[4] + b_ref[4]) / 2.0
        midpoint_y = (a_ref[5] + b_ref[5]) / 2.0
        a_tokens[a_ref[0]][a_ref[1]][1][a_ref[2]] = midpoint_x
        a_tokens[a_ref[0]][a_ref[1]][1][a_ref[3]] = midpoint_y
        b_tokens[b_ref[0]][b_ref[1]][1][b_ref[2]] = midpoint_x
        b_tokens[b_ref[0]][b_ref[1]][1][b_ref[3]] = midpoint_y
        if _matches is not None:
            _matches.append(_EndpointMatch(a_ref, b_ref, (midpoint_x, midpoint_y)))
        used_b.add(b_key)
        changed = True

    if not changed:
        return a_shape, b_shape, False
    return _shape_from_tokens(a_shape, a_tokens), _shape_from_tokens(b_shape, b_tokens), True


def _replacement(
    obj: VectorRegion,
    shape: Shape,
    other_id: int,
    candidate: _SeamCandidate,
    *,
    selected: str,
) -> VectorRegion:
    return obj.with_current(
        shape,
        footprint=to_polygon(shape),
        diagnostics={
            "seams": {
                "accepted": True,
                "paired_with": int(other_id),
                "selected": selected,
                "axis": candidate.axis,
                "gap_before": abs(candidate.a_value - candidate.b_value),
                "seam": candidate.seam,
            }
        },
    )


def _leaf_pairs_between(
    a_owner: VectorRegion,
    a_region: VectorRegion,
    b_owner: VectorRegion,
    b_region: VectorRegion,
    *,
    tol: float,
) -> list[tuple[_LeafRef, _LeafRef]]:
    if not _regions_close(a_region, b_region, tol=tol):
        return []
    if a_region.is_leaf and b_region.is_leaf:
        return [(_LeafRef(owner=a_owner, leaf=a_region), _LeafRef(owner=b_owner, leaf=b_region))]

    pairs: list[tuple[_LeafRef, _LeafRef]] = []
    if a_region.is_branch and b_region.is_branch:
        for a_child in a_region.children:
            for b_child in b_region.children:
                pairs.extend(_leaf_pairs_between(a_owner, a_child, b_owner, b_child, tol=tol))
        return pairs

    if a_region.is_branch:
        for a_child in a_region.children:
            pairs.extend(_leaf_pairs_between(a_owner, a_child, b_owner, b_region, tol=tol))
        return pairs

    for b_child in b_region.children:
        pairs.extend(_leaf_pairs_between(a_owner, a_region, b_owner, b_child, tol=tol))
    return pairs


def _leaf_pairs_within(
    owner: VectorRegion,
    region: VectorRegion,
    *,
    tol: float,
) -> list[tuple[_LeafRef, _LeafRef]]:
    if region.is_leaf:
        return []

    pairs: list[tuple[_LeafRef, _LeafRef]] = []
    children = list(region.children)
    for index, a_child in enumerate(children):
        for b_child in children[index + 1:]:
            pairs.extend(_leaf_pairs_between(owner, a_child, owner, b_child, tol=tol))
    for child in children:
        pairs.extend(_leaf_pairs_within(owner, child, tol=tol))
    return pairs


def _replace_leaf(root: VectorRegion, leaf_id: int, replacement: VectorRegion) -> VectorRegion:
    if root.is_leaf:
        return replacement if int(root.id) == int(leaf_id) else root
    children = [
        _replace_leaf(child, leaf_id, replacement)
        for child in root.children
    ]
    return root.with_children(children)


def _replace_leaves(root: VectorRegion, replacements_by_leaf_id: dict[int, VectorRegion]) -> VectorRegion:
    if root.is_leaf:
        return replacements_by_leaf_id.get(int(root.id), root)
    children = [
        _replace_leaves(child, replacements_by_leaf_id)
        for child in root.children
    ]
    # A branch's footprint is the retained exterior geometry root.  Its
    # children are material partitions, whose fitted paths can have tiny
    # independent residuals.  Recomputing the branch footprint from those
    # children turns an interior stitch into a spurious exterior shape change
    # and makes the optimizer reject it.  Preserve the container footprint and
    # raster; only its addressed children changed.
    return VectorRegion.branch(
        id=root.id,
        children=children,
        z=root.z,
        raster=root.raster,
        footprint=root.footprint,
        fill=root.fill,
        source_label=root.source_label,
        color_hex=root.color_hex,
        drawing_id=root.drawing_id,
        source_regions=root.source_regions,
        diagnostics=root.diagnostics,
    )


def _leaf_key(ref: _LeafRef) -> tuple[int, int]:
    return int(ref.owner.id), int(ref.leaf.id)


def _paint_order(ref: _LeafRef) -> tuple[float, float, int, int]:
    """Stable back-to-front ordering for a leaf participating in a seam."""
    return float(ref.owner.z), float(ref.leaf.z), int(ref.owner.id), int(ref.leaf.id)


def _shape_from_pathops(footprint) -> Shape:
    params: dict[str, object] = {"d": footprint.to_svg_d()}
    if len(footprint.linearized_subpaths) > 1:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


def _pathops_input_is_bounded(shape: Shape) -> bool:
    if shape.kind != "path":
        return False
    subpaths = _parse_subpaths(str(shape.params.get("d", "")))
    return bool(subpaths) and sum(len(subpath) for subpath in subpaths) <= _PATHOPS_MAX_INPUT_COMMANDS


def _pathops_shared_boundary_pair(
    a_ref: _LeafRef,
    b_ref: _LeafRef,
    a_shape: Shape,
    b_shape: Shape,
    *,
    tol: float,
) -> tuple[Shape, Shape, _SeamCandidate] | None:
    """Transfer a complex top boundary into the lower region using PathOps.

    The outsets are temporary construction geometry. Their intersection forms
    a narrow bridge across a fitted gap; subtracting the authoritative top
    footprint from that bridge rebuilds the bottom with the exact top seam.
    The repaired lower region is then clipped to its retained root footprint,
    so temporary underlap can never escape the drawing's exterior boundary.
    """
    if int(a_ref.owner.id) != int(b_ref.owner.id):
        return None
    if not _pathops_input_is_bounded(a_shape) or not _pathops_input_is_bounded(b_shape):
        return None
    if _paint_order(a_ref) <= _paint_order(b_ref):
        bottom_ref, top_ref = a_ref, b_ref
        bottom_shape, top_shape = a_shape, b_shape
        bottom_is_a = True
    else:
        bottom_ref, top_ref = b_ref, a_ref
        bottom_shape, top_shape = b_shape, a_shape
        bottom_is_a = False

    bottom = to_polygon(bottom_shape)
    top = to_polygon(top_shape)
    if bottom.is_empty or top.is_empty:
        return None
    boundary = bottom_ref.owner.footprint
    if boundary is None or getattr(boundary, "is_empty", False):
        return None
    gap_before = float(bottom.distance(top))
    try:
        bridge = bottom.outset(tol).intersection(top.outset(tol))
        if bridge.is_empty:
            return None
        clip_top = top.filled_small_holes(tol * tol)
        updated_bottom = (
            bottom.union(bridge)
            .difference(clip_top)
            .intersection(boundary)
            .buffer(0)
        )
    except RuntimeError:
        return None
    if updated_bottom.is_empty:
        return None

    before = bottom.union(top)
    after = updated_bottom.union(top)
    denominator = max(float(before.area), 1.0)
    residual = float(before.symmetric_difference(after).area / denominator)
    if residual > _PATHOPS_UNION_RESIDUAL_TOL:
        return None

    candidate = _SeamCandidate(
        "pathops",
        "under",
        "top",
        gap_before,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    updated_bottom_shape = _shape_from_pathops(updated_bottom)
    if bottom_is_a:
        return updated_bottom_shape, top_shape, candidate
    return top_shape, updated_bottom_shape, candidate


def _apply_leaf_pair(
    a_ref: _LeafRef,
    b_ref: _LeafRef,
    updates_by_leaf: dict[tuple[int, int], _LeafUpdate],
    owner_leaf_maps: dict[int, dict[int, VectorRegion]],
    intent_ref_by_key: dict[tuple[int, int], PassInputRegionRef],
    constraints: list[AxisSeamConstraint | PathOpsUnderlapConstraint],
    *,
    epsilon: float,
    max_error: float,
    cubic: bool,
    allow_underlap: bool,
) -> tuple[int, int] | None:
    a = a_ref.leaf
    if a.current is None:
        return None
    b = b_ref.leaf
    if b.current is None:
        return None

    def current_shape(ref: _LeafRef) -> Shape | None:
        key = _leaf_key(ref)
        if key in updates_by_leaf:
            return updates_by_leaf[key].shape
        leaf = ref.leaf
        if leaf.current is None:
            return None
        if leaf.current.kind != "use":
            return leaf.current
        source_id = int(leaf.current.params.get("href_obj_id", -1))
        source = owner_leaf_maps.get(int(ref.owner.id), {}).get(source_id)
        if source is None or source.current is None:
            return None
        source_key = (int(ref.owner.id), source_id)
        source_shape = updates_by_leaf[source_key].shape if source_key in updates_by_leaf else source.current
        return transform_working_shape(source_shape, leaf.current.params["transform"])

    a_current = current_shape(a_ref)
    b_current = current_shape(b_ref)
    if a_current is None or b_current is None:
        return None
    a_for_candidate = a.with_current(a_current, footprint=a.footprint)
    b_for_candidate = b.with_current(b_current, footprint=b.footprint)
    candidate = _candidate_for_pair(a_for_candidate, b_for_candidate, tol=_SEAM_TOL)
    if candidate is None:
        return None
    a_key = _leaf_key(a_ref)
    b_key = _leaf_key(b_ref)
    if allow_underlap and candidate.axis == "vertices":
        pathops = _pathops_shared_boundary_pair(
            a_ref,
            b_ref,
            a_current,
            b_current,
            tol=_SEAM_TOL,
        )
        if pathops is not None:
            a_shape, b_shape, candidate = pathops
            constraints.append(PathOpsUnderlapConstraint(
                left=intent_ref_by_key[a_key],
                right=intent_ref_by_key[b_key],
                bottom="left" if _paint_order(a_ref) <= _paint_order(b_ref) else "right",
                tolerance=_SEAM_TOL,
            ))
            updates_by_leaf[a_key] = _LeafUpdate(
                a_shape,
                int(b.id),
                candidate,
                "pathops_shared_boundary",
            )
            updates_by_leaf[b_key] = _LeafUpdate(
                b_shape,
                int(a.id),
                candidate,
                "pathops_authoritative",
            )
            return int(a_ref.owner.id), int(b_ref.owner.id)
    a_shape = _rewrite_shape_side(
        a_current,
        side=candidate.a_side,
        candidate=candidate,
        side_value=candidate.a_value,
        tol=_SEAM_TOL,
    )
    b_shape = _rewrite_shape_side(
        b_current,
        side=candidate.b_side,
        candidate=candidate,
        side_value=candidate.b_value,
        tol=_SEAM_TOL,
    )
    if a_shape is None or b_shape is None:
        return None
    before_cleanup_a = a_shape
    before_cleanup_b = b_shape
    before_matches: list[_EndpointMatch] = []
    a_shape, b_shape, vertices_changed = _true_up_path_vertices(
        a_shape, b_shape, tol=_SEAM_TOL, _matches=before_matches
    )
    before_stage = _endpoint_stage(
        intent_ref_by_key[a_key],
        intent_ref_by_key[b_key],
        before_cleanup_a,
        before_cleanup_b,
        before_matches,
    )
    # Simplification has its own pass before stitch in the automatic workflow.
    # Do retain the local inflecting-cubic repair caused by moving an endpoint,
    # but do not rerun general linelet/path simplification for every pair.
    cleaned_a = _cleanup_inflecting_cubics(a_shape, max_error=max_error, line_epsilon=epsilon)
    cleaned_b = _cleanup_inflecting_cubics(b_shape, max_error=max_error, line_epsilon=epsilon)
    if cleaned_a != a_shape or cleaned_b != b_shape:
        a_shape, b_shape = cleaned_a, cleaned_b
        after_cleanup_a = a_shape
        after_cleanup_b = b_shape
        after_matches: list[_EndpointMatch] = []
        a_shape, b_shape, cleanup_vertices_changed = _true_up_path_vertices(
            a_shape, b_shape, tol=_SEAM_TOL, _matches=after_matches
        )
        vertices_changed = vertices_changed or cleanup_vertices_changed
        after_stage = _endpoint_stage(
            intent_ref_by_key[a_key],
            intent_ref_by_key[b_key],
            after_cleanup_a,
            after_cleanup_b,
            after_matches,
        )
    else:
        after_stage = None
    selected = "vertex_midpoint" if vertices_changed or candidate.axis == "vertices" else "midpoint"
    constraints.append(AxisSeamConstraint(
        left=intent_ref_by_key[a_key],
        right=intent_ref_by_key[b_key],
        axis=candidate.axis,
        left_side=candidate.a_side,
        right_side=candidate.b_side,
        left_value=candidate.a_value,
        right_value=candidate.b_value,
        seam=candidate.seam,
        span_min=candidate.span_min,
        span_max=candidate.span_max,
        before_cleanup=before_stage,
        after_cleanup=after_stage,
        selected=selected,
    ))
    updates_by_leaf[a_key] = _LeafUpdate(a_shape, int(b.id), candidate, selected)
    updates_by_leaf[b_key] = _LeafUpdate(b_shape, int(a.id), candidate, selected)
    return int(a_ref.owner.id), int(b_ref.owner.id)


def _owner_components(edges: list[tuple[int, int]]) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    groups: dict[int, set[int]] = {}
    for value in parent:
        groups.setdefault(find(value), set()).add(value)
    return [groups[key] for key in sorted(groups)]


def _cluster_leaf_vertices(
    updates_by_leaf: dict[tuple[int, int], _LeafUpdate],
    owner_ids: set[int],
    intent_ref_by_key: dict[tuple[int, int], PassInputRegionRef],
    constraints: list[VertexClusterConstraint],
    clustered_refs: set[PassInputRegionRef],
    *,
    tol: float,
    epsilon: float,
    max_error: float,
    cubic: bool,
) -> None:
    keys = [
        key for key, update in updates_by_leaf.items()
        if (
            key[0] in owner_ids
            and update.shape.kind == "path"
            and update.candidate.axis != "pathops"
        )
    ]
    if len(keys) < 2:
        return

    tokens_by_key = {
        key: _parse_subpaths(str(updates_by_leaf[key].shape.params.get("d", "")))
        for key in keys
    }
    changed_keys: set[tuple[int, int]] = set()

    def apply_clusters(*, link_tol: float, min_keys: int, allow_same_key_links: bool) -> None:
        refs: list[_CoordinateRef] = []
        for key, tokens in tokens_by_key.items():
            refs.extend(_path_endpoint_refs(key, tokens))
        if len(refs) < 2:
            return

        parent = list(range(len(refs)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        # Only endpoints in neighboring ``link_tol`` cells can be joined.
        # The former all-pairs loop made a detailed traced surface quadratic in
        # its command count (and could exhaust memory before v0 was retained).
        # Looking only backward through the spatial buckets considers each pair
        # once while preserving the original strict distance predicate.
        buckets: dict[tuple[int, int], list[int]] = {}
        for i, a_ref in enumerate(refs):
            cell_x = math.floor(a_ref.x / link_tol)
            cell_y = math.floor(a_ref.y / link_tol)
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    for j in buckets.get((cell_x + dx_cell, cell_y + dy_cell), []):
                        b_ref = refs[j]
                        if not allow_same_key_links and a_ref.key == b_ref.key:
                            continue
                        dx = a_ref.x - b_ref.x
                        dy = a_ref.y - b_ref.y
                        if 0.0 < math.hypot(dx, dy) <= link_tol:
                            union(i, j)
            buckets.setdefault((cell_x, cell_y), []).append(i)

        groups: dict[int, list[_CoordinateRef]] = {}
        for index, ref in enumerate(refs):
            groups.setdefault(find(index), []).append(ref)

        shape_by_key = {
            key: _shape_from_tokens(updates_by_leaf[key].shape, tokens)
            for key, tokens in tokens_by_key.items()
        }
        assignments: list[EndpointAssignment] = []
        for group in groups.values():
            if len({ref.key for ref in group}) < min_keys:
                continue
            x = sum(ref.x for ref in group) / len(group)
            y = sum(ref.y for ref in group) / len(group)
            assignments.append(EndpointAssignment(
                members=tuple(
                    _endpoint_ref(intent_ref_by_key[ref.key], shape_by_key[ref.key], ref)
                    for ref in group
                ),
                point=(x, y),
            ))
            for ref in group:
                tokens = tokens_by_key[ref.key]
                values = tokens[ref.subpath_index][ref.token_index][1]
                values[ref.x_index] = x
                values[ref.y_index] = y
                changed_keys.add(ref.key)
        if assignments:
            constraints.append(VertexClusterConstraint(
                EndpointConstraintStage(tuple(assignments))
            ))

    apply_clusters(link_tol=tol, min_keys=2, allow_same_key_links=False)
    apply_clusters(link_tol=_JUNCTION_TOL, min_keys=3, allow_same_key_links=True)

    for key in changed_keys:
        update = updates_by_leaf[key]
        shape = _shape_from_tokens(update.shape, tokens_by_key[key])
        # Stitching owns only shared boundary coordinates.  Running the
        # linelet simplifier here can rewrite unrelated outer geometry, which
        # makes the framework reject the entire seam proposal before its exact
        # endpoint projection is applied.  Preserve the local inflecting-cubic
        # guard, but leave all general simplification to its dedicated pass.
        shape = _cleanup_inflecting_cubics(shape, max_error=max_error, line_epsilon=epsilon)
        updates_by_leaf[key] = _LeafUpdate(
            shape,
            update.paired_with,
            update.candidate,
            "vertex_cluster",
        )
        clustered_refs.add(intent_ref_by_key[key])


def _replacement_for_update(ref: _LeafRef, update: _LeafUpdate) -> VectorRegion:
    return _replacement(
        ref.leaf,
        update.shape,
        update.paired_with,
        update.candidate,
        selected=update.selected,
    )


def _apply_endpoint_stage(
    shapes: dict[tuple[int, ...], Shape],
    stage: EndpointConstraintStage | None,
) -> None:
    if stage is None:
        return
    tokens_by_path: dict[tuple[int, ...], list[list[tuple[str, list[float]]]]] = {}
    located: list[tuple[tuple[int, ...], int, int, tuple[float, float]]] = []
    for assignment in stage.assignments:
        if len(assignment.members) < 2:
            raise ValueError("seam endpoint assignment requires at least two members")
        for endpoint in assignment.members:
            key = tuple(endpoint.region.ancestry)
            shape = shapes.get(key)
            if shape is None or shape.kind != "path":
                raise ValueError("seam endpoint target is not an active path")
            if _path_id(shape) != endpoint.path_id:
                raise ValueError("seam endpoint path identity mismatch")
            tokens = tokens_by_path.setdefault(
                key, _parse_subpaths(str(shape.params.get("d", "")))
            )
            matches: list[tuple[int, int]] = []
            for subpath_index, subpath in enumerate(tokens):
                for token_index, (command, values) in enumerate(subpath):
                    if _command_id(subpath_index, token_index, command, values) == endpoint.command_id:
                        matches.append((subpath_index, token_index))
            if len(matches) != 1:
                raise ValueError("seam endpoint command identity is absent or ambiguous")
            located.append((*matches[0], key, assignment.point))
    for subpath_index, token_index, key, point in located:
        command, values = tokens_by_path[key][subpath_index][token_index]
        indexes = _coordinate_indexes(command)
        if len(indexes) != 1:
            raise ValueError("seam endpoint command has no unique exit coordinate")
        x_index, y_index = indexes[0]
        values[x_index], values[y_index] = point
    for key, tokens in tokens_by_path.items():
        shapes[key] = _shape_from_tokens(shapes[key], tokens)


def apply_seam_constraints(
    action: ApplySeamConstraintsOperation,
    resolver,
) -> list[VectorRegion]:
    """Execute recorded seam decisions without pair or cluster discovery."""
    shapes: dict[tuple[int, ...], Shape] = {}
    input_shapes: dict[tuple[int, ...], Shape] = {}
    metadata: dict[tuple[int, ...], tuple[int, _SeamCandidate, str]] = {}

    def current_shape(ref: PassInputRegionRef) -> Shape:
        key = tuple(ref.ancestry)
        if key in shapes:
            return shapes[key]
        region = resolver.resolve(ref)
        if region.current is None:
            raise ValueError("seam replay target has no geometry")
        if region.current.kind != "use":
            shapes[key] = region.current
            input_shapes[key] = region.current
            return region.current
        owner = resolver.resolve(PassInputRegionRef(
            ref.batch_id, ref.input_revision, (int(ref.ancestry[0]),)
        ))
        source_id = int(region.current.params.get("href_obj_id", -1))
        source_matches = [leaf for leaf in owner.leaves() if int(leaf.id) == source_id]
        if len(source_matches) != 1 or source_matches[0].current is None:
            raise ValueError("seam replay clone source is absent or ambiguous")
        shape = transform_working_shape(source_matches[0].current, region.current.params["transform"])
        shapes[key] = shape
        input_shapes[key] = shape
        return shape

    for constraint in action.pairs:
        left_key = tuple(constraint.left.ancestry)
        right_key = tuple(constraint.right.ancestry)
        left_region = resolver.resolve(constraint.left)
        right_region = resolver.resolve(constraint.right)
        left_shape = current_shape(constraint.left)
        right_shape = current_shape(constraint.right)
        if isinstance(constraint, PathOpsUnderlapConstraint):
            left_owner = resolver.resolve(PassInputRegionRef(
                constraint.left.batch_id,
                constraint.left.input_revision,
                (int(constraint.left.ancestry[0]),),
            ))
            right_owner = resolver.resolve(PassInputRegionRef(
                constraint.right.batch_id,
                constraint.right.input_revision,
                (int(constraint.right.ancestry[0]),),
            ))
            actual_bottom = (
                "left"
                if _paint_order(_LeafRef(left_owner, left_region))
                <= _paint_order(_LeafRef(right_owner, right_region))
                else "right"
            )
            if actual_bottom != constraint.bottom:
                raise ValueError("recorded seam underlap paint order no longer matches")
            result = _pathops_shared_boundary_pair(
                _LeafRef(left_owner, left_region),
                _LeafRef(right_owner, right_region),
                left_shape,
                right_shape,
                tol=constraint.tolerance,
            )
            if result is None:
                raise ValueError("recorded seam underlap no longer applies")
            left_shape, right_shape, candidate = result
            shapes[left_key], shapes[right_key] = left_shape, right_shape
            metadata[left_key] = (int(right_region.id), candidate, "pathops_shared_boundary" if actual_bottom == "left" else "pathops_authoritative")
            metadata[right_key] = (int(left_region.id), candidate, "pathops_authoritative" if actual_bottom == "left" else "pathops_shared_boundary")
            continue

        candidate = _SeamCandidate(
            constraint.axis,
            constraint.left_side,
            constraint.right_side,
            constraint.left_value,
            constraint.right_value,
            constraint.seam,
            constraint.span_min,
            constraint.span_max,
        )
        rewritten_left = _rewrite_shape_side(
            left_shape,
            side=candidate.a_side,
            candidate=candidate,
            side_value=candidate.a_value,
            tol=_SEAM_TOL,
        )
        rewritten_right = _rewrite_shape_side(
            right_shape,
            side=candidate.b_side,
            candidate=candidate,
            side_value=candidate.b_value,
            tol=_SEAM_TOL,
        )
        if rewritten_left is None or rewritten_right is None:
            raise ValueError("recorded axis seam no longer applies")
        shapes[left_key], shapes[right_key] = rewritten_left, rewritten_right
        _apply_endpoint_stage(shapes, constraint.before_cleanup)
        cleaned_left = _cleanup_inflecting_cubics(
            shapes[left_key], max_error=action.max_error, line_epsilon=action.epsilon
        )
        cleaned_right = _cleanup_inflecting_cubics(
            shapes[right_key], max_error=action.max_error, line_epsilon=action.epsilon
        )
        shapes[left_key], shapes[right_key] = cleaned_left, cleaned_right
        _apply_endpoint_stage(shapes, constraint.after_cleanup)
        metadata[left_key] = (int(right_region.id), candidate, constraint.selected)
        metadata[right_key] = (int(left_region.id), candidate, constraint.selected)

    for constraint in action.clusters:
        _apply_endpoint_stage(shapes, constraint.stage)
    for ref in action.clustered:
        key = tuple(ref.ancestry)
        shapes[key] = _cleanup_inflecting_cubics(
            shapes[key], max_error=action.max_error, line_epsilon=action.epsilon
        )
        paired_with, candidate, _selected = metadata[key]
        metadata[key] = (paired_with, candidate, "vertex_cluster")

    for correspondence in action.path_correspondence:
        key = tuple(correspondence.target.ancestry)
        before_paths = _parse_subpaths(
            str(input_shapes[key].params.get("d", ""))
        )
        after_paths = _parse_subpaths(str(shapes[key].params.get("d", "")))
        if correspondence.subpath_index >= len(before_paths):
            raise ValueError("seam input path correspondence is stale")
        before_path = before_paths[correspondence.subpath_index]
        if _subpath_revision(before_path) != correspondence.input_path_revision:
            raise ValueError("seam input path revision is stale")
        after_path = (
            after_paths[correspondence.subpath_index]
            if correspondence.subpath_index < len(after_paths)
            else None
        )
        actual_output_revision = (
            _subpath_revision(after_path) if after_path is not None else ""
        )
        if actual_output_revision != correspondence.output_path_revision:
            raise ValueError("seam output path revision does not match receipt")
        before_count = len(_drawable_tokens(before_path))
        after_count = (
            len(_drawable_tokens(after_path)) if after_path is not None else 0
        )
        if any(
            item.input_index >= before_count or item.output_index >= after_count
            for item in correspondence.commands
        ):
            raise ValueError("seam command correspondence is out of range")

    replacements_by_root: dict[int, dict[int, VectorRegion]] = {}
    for key, shape in shapes.items():
        if key not in metadata:
            continue
        region = resolver.resolve(PassInputRegionRef(
            action.roots[0].batch_id, action.roots[0].input_revision, key
        ))
        paired_with, candidate, selected = metadata[key]
        replacements_by_root.setdefault(int(key[0]), {})[int(region.id)] = _replacement(
            region, shape, paired_with, candidate, selected=selected
        )
    return [
        _replace_leaves(
            resolver.resolve(root), replacements_by_root.get(int(root.ancestry[0]), {})
        )
        for root in action.roots
    ]


def seams_pass(
    objects: list[VectorRegion],
    masks: dict[int, np.ndarray],
    *,
    epsilon: float = 1.0,
    max_error: float = 1.0,
    cubic: bool = False,
    post_symmetry: bool = False,
    allow_underlap: bool = True,
) -> list[Proposal]:
    owners = sorted(objects, key=lambda current: (float(current.z), int(current.id)))
    owner_by_id = {int(owner.id): owner for owner in owners}
    owner_leaf_maps = {
        int(owner.id): {int(leaf.id): leaf for leaf in owner.leaves()}
        for owner in owners
    }
    leaf_ref_by_key: dict[tuple[int, int], _LeafRef] = {}
    intent_ref_by_key = {
        (int(path[0]), int(region.id)): PassInputRegionRef("", "", tuple(path))
        for path, region in index_region_paths(objects).items()
        if region.is_leaf
    }
    updates_by_leaf: dict[tuple[int, int], _LeafUpdate] = {}
    owner_edges: list[tuple[int, int]] = []
    pair_constraints: list[AxisSeamConstraint | PathOpsUnderlapConstraint] = []

    def record_pair(a_ref: _LeafRef, b_ref: _LeafRef) -> None:
        if not _same_surface_fill_group(a_ref.leaf, b_ref.leaf):
            return
        leaf_ref_by_key.setdefault(_leaf_key(a_ref), a_ref)
        leaf_ref_by_key.setdefault(_leaf_key(b_ref), b_ref)
        edge = _apply_leaf_pair(
            a_ref,
            b_ref,
            updates_by_leaf,
            owner_leaf_maps,
            intent_ref_by_key,
            pair_constraints,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
            allow_underlap=allow_underlap,
        )
        if edge is not None:
            owner_edges.append(edge)

    if post_symmetry:
        # Before refinement, stitch can repair every retained root.  After
        # symmetry, only a self-symmetry branch gained a new coincident axis;
        # revisiting unrelated compound roots is redundant and expensive.
        for owner in owners:
            symmetry = owner.diagnostics.get("symmetry")
            if isinstance(symmetry, dict) and symmetry.get("mode") == "self":
                for a_ref, b_ref in _leaf_pairs_within(owner, owner, tol=_SEAM_TOL):
                    record_pair(a_ref, b_ref)
    else:
        for owner in owners:
            for a_ref, b_ref in _leaf_pairs_within(owner, owner, tol=_SEAM_TOL):
                record_pair(a_ref, b_ref)

        for owner_index, a_owner in enumerate(owners):
            for b_owner in owners[owner_index + 1:]:
                leaf_pairs = _leaf_pairs_between(a_owner, a_owner, b_owner, b_owner, tol=_SEAM_TOL)
                for a_ref, b_ref in leaf_pairs:
                    record_pair(a_ref, b_ref)

    proposals: list[Proposal] = []
    for owner_ids in _owner_components(owner_edges):
        cluster_constraints: list[VertexClusterConstraint] = []
        clustered_refs: set[PassInputRegionRef] = set()
        _cluster_leaf_vertices(
            updates_by_leaf,
            owner_ids,
            intent_ref_by_key,
            cluster_constraints,
            clustered_refs,
            tol=_SEAM_TOL,
            epsilon=epsilon,
            max_error=max_error,
            cubic=cubic,
        )
        replacements: list[VectorRegion] = []
        for owner_id in sorted(owner_ids):
            owner = owner_by_id[owner_id]
            leaf_replacements: dict[int, VectorRegion] = {}
            for key, update in updates_by_leaf.items():
                if key[0] != owner_id:
                    continue
                leaf_replacements[key[1]] = _replacement_for_update(leaf_ref_by_key[key], update)
            if not leaf_replacements:
                continue
            replacements.append(_replace_leaves(owner, leaf_replacements))
        if replacements:
            # Endpoint projection can fold a tiny closing curve back across
            # its own contour or move a cutout across its exterior. Validate
            # the complete proposal with the same retained-topology contract
            # used by replay before recording any seam constraints. Keep the
            # original geometry when the proposed adjustment is unsafe; the
            # normalized result is a predicate, not replacement geometry.
            from ..topology import normalize_retained_topology

            try:
                normalize_retained_topology(replacements, allow_external=True)
            except ValueError:
                continue
            root_refs = tuple(
                PassInputRegionRef("", "", (owner_id,))
                for owner_id in sorted(owner_ids)
            )
            component_pairs = tuple(
                constraint for constraint in pair_constraints
                if int(constraint.left.ancestry[0]) in owner_ids
            )
            pathops_rebuilt = {
                tuple(
                    constraint.left.ancestry
                    if constraint.bottom == "left"
                    else constraint.right.ancestry
                )
                for constraint in component_pairs
                if isinstance(constraint, PathOpsUnderlapConstraint)
            }
            path_correspondence = tuple(
                correspondence
                for key, update in sorted(updates_by_leaf.items())
                if key[0] in owner_ids
                and leaf_ref_by_key[key].leaf.current is not None
                and leaf_ref_by_key[key].leaf.current.kind == "path"
                and update.shape.kind == "path"
                for correspondence in _path_correspondence(
                    intent_ref_by_key[key],
                    leaf_ref_by_key[key].leaf.current,
                    update.shape,
                    positional_mutation=key not in pathops_rebuilt,
                )
            )
            action = ApplySeamConstraintsOperation(
                roots=root_refs,
                pairs=component_pairs,
                clusters=tuple(cluster_constraints),
                clustered=tuple(sorted(clustered_refs, key=lambda ref: ref.ancestry)),
                epsilon=float(epsilon),
                max_error=float(max_error),
                cubic=bool(cubic),
                path_correspondence=path_correspondence,
            )
            proposals.append(Proposal(
                tuple(sorted(owner_ids)),
                replacements,
                intent=operation_intent(action, root_refs, tuple(sorted(owner_ids))),
            ))
    return proposals

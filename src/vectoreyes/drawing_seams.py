"""Retained seam execution and shared endpoint/boundary operations.

Pair/cluster discovery stays in the refinement. Execution consumes recorded
constraints and validates exact path/command correspondence. Working-region
projection remains the existing engine adapter, not a second scene.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import math
from .fit import Shape, _fmt
from .emit import shape_to_path_d
from .skia_geometry import SkPath
from .refinement.automatic_replay import (
    ApplySeamConstraintsOperation,
    EndpointAssignment,
    EndpointConstraintStage,
    PassInputRegionRef,
    PathEndpointRef,
    PathOpsUnderlapConstraint,
    SeamCommandIndexCorrespondence,
    SeamPathCorrespondence,
)
from .path_sampling import _parse_subpaths

# Admission tolerance only; accepted endpoints are projected exactly.
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
class SeamInput:
    """Selected geometry and numeric observations required by seam replay."""
    shape: Shape
    id: int
    owner_id: int
    paint_order: tuple[float, float, int, int]
    boundary: SkPath


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

    encoded = ":".join(
        [
            str(subpath_index),
            str(token_index),
            command,
            *(float(value).hex() for value in values),
        ]
    )
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
        records.append(
            SeamPathCorrespondence(
                target=target,
                subpath_index=index,
                input_path_revision=_subpath_revision(before_path),
                output_path_revision=(
                    _subpath_revision(after_path) if after_path is not None else ""
                ),
                commands=commands,
            )
        )
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
    return EndpointConstraintStage(
        tuple(
            EndpointAssignment(
                members=(
                    _endpoint_ref(left_ref, left_shape, match.left),
                    _endpoint_ref(right_ref, right_shape, match.right),
                ),
                point=match.point,
            )
            for match in matches
        )
    )


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
                    if (
                        abs(x - side_value) <= tol
                        and span_min - tol <= y <= span_max + tol
                    ):
                        new_values[x_idx] = seam
                        changed = True
                elif (
                    abs(y - side_value) <= tol and span_min - tol <= x <= span_max + tol
                ):
                    new_values[y_idx] = seam
                    changed = True
            out_subpath.append((command, new_values))
        rewritten.append(out_subpath)
    if not changed:
        return None

    return _shape_from_tokens(shape, rewritten)


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
    return (
        max(xs) - min(xs) < _MIN_SUBPATH_SPAN and max(ys) - min(ys) < _MIN_SUBPATH_SPAN
    )


def _path_coordinate_refs(tokens: list[list[tuple[str, list[float]]]]):
    refs = []
    for subpath_index, subpath in enumerate(tokens):
        if _subpath_is_degenerate(subpath):
            continue
        for token_index, (command, values) in enumerate(subpath):
            for x_idx, y_idx in _coordinate_indexes(command):
                refs.append(
                    (
                        subpath_index,
                        token_index,
                        x_idx,
                        y_idx,
                        float(values[x_idx]),
                        float(values[y_idx]),
                    )
                )
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
                for other_x, other_y in buckets.get(
                    (cell_x + offset_x, cell_y + offset_y), []
                ):
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


def _shape_from_tokens(
    shape: Shape, tokens: list[list[tuple[str, list[float]]]]
) -> Shape:
    params = dict(shape.params)
    params["d"] = " ".join(_subpath_d(subpath) for subpath in tokens)
    if params["d"] != shape.params.get("d"):
        params.pop("canonical_path_geometry", None)
        params.pop("canonical_clone_geometry", None)
    return Shape("path", params)


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
    return (
        _shape_from_tokens(a_shape, a_tokens),
        _shape_from_tokens(b_shape, b_tokens),
        True,
    )




def _shape_from_pathops(footprint) -> Shape:
    params: dict[str, object] = {"d": footprint.to_svg_d()}
    if len(footprint.linearized_subpaths) > 1:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


def _pathops_input_is_bounded(shape: Shape) -> bool:
    if shape.kind != "path":
        return False
    subpaths = _parse_subpaths(str(shape.params.get("d", "")))
    return (
        bool(subpaths)
        and sum(len(subpath) for subpath in subpaths) <= _PATHOPS_MAX_INPUT_COMMANDS
    )


def _pathops_shared_boundary_pair(
    a_ref: SeamInput,
    b_ref: SeamInput,
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
    if a_ref.owner_id != b_ref.owner_id:
        return None
    if not _pathops_input_is_bounded(a_shape) or not _pathops_input_is_bounded(b_shape):
        return None
    if a_ref.paint_order <= b_ref.paint_order:
        bottom_ref, top_ref = a_ref, b_ref
        bottom_shape, top_shape = a_shape, b_shape
        bottom_is_a = True
    else:
        bottom_ref, top_ref = b_ref, a_ref
        bottom_shape, top_shape = b_shape, a_shape
        bottom_is_a = False

    bottom = SkPath.from_svg_d(shape_to_path_d(bottom_shape), fill_rule=bottom_shape.params.get('fill_rule'))
    top = SkPath.from_svg_d(shape_to_path_d(top_shape), fill_rule=top_shape.params.get('fill_rule'))
    if bottom.is_empty or top.is_empty:
        return None
    boundary = bottom_ref.boundary
    if boundary is None or getattr(boundary, "is_empty", False):
        return None
    gap_before = float(bottom.distance(top))
    try:
        bridge = bottom.outset(tol).intersection(top.outset(tol))
        if bridge.is_empty:
            return None
        clip_top = top.filled_small_holes(tol * tol)
        updated_bottom = (
            bottom.union(bridge).difference(clip_top).intersection(boundary).buffer(0)
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
                    if (
                        _command_id(subpath_index, token_index, command, values)
                        == endpoint.command_id
                    ):
                        matches.append((subpath_index, token_index))
            if len(matches) != 1:
                raise ValueError(
                    "seam endpoint command identity is absent or ambiguous"
                )
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


def execute_seam_constraints(
    action: ApplySeamConstraintsOperation,
    inputs: Mapping[tuple[int, ...], SeamInput],
) -> dict[tuple[int, ...], _LeafUpdate]:
    """Execute recorded seam decisions without pair or cluster discovery."""
    shapes: dict[tuple[int, ...], Shape] = {}
    input_shapes: dict[tuple[int, ...], Shape] = {}
    metadata: dict[tuple[int, ...], tuple[int, _SeamCandidate, str]] = {}

    def current_shape(ref: PassInputRegionRef) -> Shape:
        key = tuple(ref.ancestry)
        if key not in shapes:
            shapes[key] = inputs[key].shape
            input_shapes[key] = inputs[key].shape
        return shapes[key]

    for constraint in action.pairs:
        left_key = tuple(constraint.left.ancestry)
        right_key = tuple(constraint.right.ancestry)
        left_region = inputs[left_key]
        right_region = inputs[right_key]
        left_shape = current_shape(constraint.left)
        right_shape = current_shape(constraint.right)
        if isinstance(constraint, PathOpsUnderlapConstraint):
            actual_bottom = (
                "left"
                if left_region.paint_order
                <= right_region.paint_order
                else "right"
            )
            if actual_bottom != constraint.bottom:
                raise ValueError("recorded seam underlap paint order no longer matches")
            result = _pathops_shared_boundary_pair(
                left_region,
                right_region,
                left_shape,
                right_shape,
                tol=constraint.tolerance,
            )
            if result is None:
                raise ValueError("recorded seam underlap no longer applies")
            left_shape, right_shape, candidate = result
            shapes[left_key], shapes[right_key] = left_shape, right_shape
            metadata[left_key] = (
                int(right_region.id),
                candidate,
                "pathops_shared_boundary"
                if actual_bottom == "left"
                else "pathops_authoritative",
            )
            metadata[right_key] = (
                int(left_region.id),
                candidate,
                "pathops_authoritative"
                if actual_bottom == "left"
                else "pathops_shared_boundary",
            )
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
        _apply_endpoint_stage(shapes, constraint.after_cleanup)
        metadata[left_key] = (int(right_region.id), candidate, constraint.selected)
        metadata[right_key] = (int(left_region.id), candidate, constraint.selected)

    for constraint in action.clusters:
        _apply_endpoint_stage(shapes, constraint.stage)
    for ref in action.clustered:
        key = tuple(ref.ancestry)
        paired_with, candidate, _selected = metadata[key]
        metadata[key] = (paired_with, candidate, "vertex_cluster")

    for correspondence in action.path_correspondence:
        key = tuple(correspondence.target.ancestry)
        before_paths = _parse_subpaths(str(input_shapes[key].params.get("d", "")))
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
        after_count = len(_drawable_tokens(after_path)) if after_path is not None else 0
        if any(
            item.input_index >= before_count or item.output_index >= after_count
            for item in correspondence.commands
        ):
            raise ValueError("seam command correspondence is out of range")

    return {key: _LeafUpdate(shapes[key], paired_with, candidate, selected)
        for key, (paired_with, candidate, selected) in metadata.items()}

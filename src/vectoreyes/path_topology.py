"""Classify retained closed paths using filled-set predicates, never refitting."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import skia

from .path_geometry import CompoundPathGeometry, PathGeometry, PathGeometryWithCutouts, LineCommand, QuadraticCommand, CubicCommand
from .skia_geometry import SkPath


@dataclass(frozen=True)
class PathTopology:
    components: tuple[PathGeometry | PathGeometryWithCutouts | CompoundPathGeometry, ...]
    redundant_paths: tuple[PathGeometry, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "redundant_paths", tuple(self.redundant_paths))


def component_paths(geometry: PathGeometry | PathGeometryWithCutouts | CompoundPathGeometry) -> tuple[PathGeometry, ...]:
    if isinstance(geometry, CompoundPathGeometry):
        return geometry.paths
    return (geometry,) if isinstance(geometry, PathGeometry) else (geometry.outer, *geometry.cutouts)


def replace_component_paths(geometry, paths: Sequence[PathGeometry]):
    """Replace values without changing retained membership or fill semantics."""
    paths = tuple(paths)
    if isinstance(geometry, CompoundPathGeometry):
        return CompoundPathGeometry(paths, geometry.fill_rule, geometry.reason)
    if isinstance(geometry, PathGeometry):
        return paths[0]
    return PathGeometryWithCutouts(paths[0], paths[1:])


def _filled_path(geometry: PathGeometry) -> skia.Path:
    """Build disposable native predicates from values, not rounded SVG output."""
    path = skia.Path()
    path.moveTo(*geometry.path.start)
    for command in geometry.path.drawable_commands:
        if isinstance(command, LineCommand):
            path.lineTo(*command.end)
        elif isinstance(command, QuadraticCommand):
            path.quadTo(*command.control, *command.end)
        elif isinstance(command, CubicCommand):
            path.cubicTo(*command.control_1, *command.control_2, *command.end)
        else:
            path.arcTo(command.rx, command.ry, command.rotation_degrees,
                       skia.Path.kLarge_ArcSize if command.large_arc else skia.Path.kSmall_ArcSize,
                       skia.PathDirection.kCW if command.sweep else skia.PathDirection.kCCW, *command.end)
    path.close()
    return path


def classify_filled_paths(
    paths: Sequence[PathGeometry], *, fill_rule: Literal["nonzero", "evenodd"],
    defer_overlaps: bool = False,
) -> PathTopology:
    """Find actual exterior/cutout transitions without changing source commands.

    PathOps results are disposable predicates only. When requested, overlapping
    paths remain a compound with unresolved ownership for later geometry edits.
    Non-simple compound boundaries may also remain explicitly unresolved.
    Strict validation still fails atomically.
    """
    paths = tuple(paths)
    if not paths or fill_rule not in {"nonzero", "evenodd"}:
        raise ValueError("topology requires closed source paths and a valid fill rule")
    footprints = [_filled_path(path) for path in paths]

    def unresolved(reason: str, message: str) -> PathTopology:
        if defer_overlaps and len(paths) > 1:
            return PathTopology((CompoundPathGeometry(paths, fill_rule, reason),))
        raise ValueError(f"source topology requires retained boolean normalization: {message}")

    def op(a, b, operation):
        try:
            return skia.Op(a, b, operation)
        except RuntimeError as error:
            raise ValueError("source topology requires retained boolean normalization: PathOps failed") from error

    if len(paths) == 1:
        if fill_rule == "evenodd":
            parity = skia.Path(footprints[0])
            parity.setFillType(skia.PathFillType.kEvenOdd)
            if not op(footprints[0], parity, skia.PathOp.kXOR_PathOp).isEmpty():
                raise ValueError("source topology requires retained boolean normalization: single-path fill rules differ")
        return PathTopology(paths)

    for path, footprint in zip(paths, footprints, strict=True):
        simple = skia.Simplify(footprint)
        contours = sum(verb == skia.Path.kMove_Verb for verb, _points in simple)
        if simple.isEmpty() or contours != 1:
            return unresolved("non_simple_boundaries", f"degenerate or self-intersecting {path.path.id}")
        parity = skia.Path(footprint)
        parity.setFillType(skia.PathFillType.kEvenOdd)
        if not op(footprint, parity, skia.PathOp.kXOR_PathOp).isEmpty():
            return unresolved("non_simple_boundaries", f"self-intersecting {path.path.id}")

    containers: list[set[int]] = [set() for _ in paths]
    for i, first in enumerate(footprints):
        for j in range(i + 1, len(paths)):
            second = footprints[j]
            if op(first, second, skia.PathOp.kIntersect_PathOp).isEmpty():
                continue
            first_out = op(first, second, skia.PathOp.kDifference_PathOp).isEmpty()
            second_out = op(second, first, skia.PathOp.kDifference_PathOp).isEmpty()
            if first_out and second_out:
                if defer_overlaps:
                    return PathTopology((CompoundPathGeometry(paths, fill_rule, "coincident_boundaries"),))
                raise ValueError("source topology requires retained boolean normalization: coincident boundaries")
            if first_out:
                containers[i].add(j)
            elif second_out:
                containers[j].add(i)
            else:
                if defer_overlaps:
                    return PathTopology((CompoundPathGeometry(paths, fill_rule, "partially_overlapping_boundaries"),))
                raise ValueError("source topology requires retained boolean normalization: partially overlapping boundaries")
    parents = [max(owners, key=lambda owner: len(containers[owner])) if owners else None for owners in containers]
    whole = skia.Path()
    for footprint in footprints:
        whole.addPath(footprint)
    if fill_rule == "evenodd":
        whole.setFillType(skia.PathFillType.kEvenOdd)
    filled = []
    for i, footprint in enumerate(footprints):
        cell = footprint
        for child, parent in enumerate(parents):
            if parent == i:
                cell = op(cell, footprints[child], skia.PathOp.kDifference_PathOp)
        if cell.isEmpty():
            raise ValueError("source topology requires retained boolean normalization: cancelled boundary cell")
        inside = op(cell, whole, skia.PathOp.kIntersect_PathOp)
        if inside.isEmpty():
            if parents[i] is None:
                raise ValueError("source topology requires retained boolean normalization: inconsistent isolated exterior predicate")
            filled.append(False)
        else:
            outside = op(cell, whole, skia.PathOp.kDifference_PathOp)
            if outside.isEmpty() or _negligible_boolean_residue(outside, cell):
                filled.append(True)
            else:
                return unresolved("mixed_boundary_cells", "mixed boundary cell")
    exteriors = []
    holes: dict[int, list[PathGeometry]] = {}
    redundant = []
    for i, path in enumerate(paths):
        parent = parents[i]
        outside_filled = False if parent is None else filled[parent]
        if filled[i] == outside_filled:
            redundant.append(path)
        elif filled[i]:
            exteriors.append(i)
        else:
            owner = parent
            while owner is not None and (parents[owner] is not None and filled[parents[owner]]):
                owner = parents[owner]
            if owner is None:
                raise ValueError("subtractive path lacks an exterior owner")
            holes.setdefault(owner, []).append(path)
    return PathTopology(tuple(
        PathGeometryWithCutouts(paths[index], tuple(holes[index])) if holes.get(index) else paths[index]
        for index in exteriors
    ), tuple(redundant))


def _negligible_boolean_residue(residue: skia.Path, reference: skia.Path) -> bool:
    """Ignore only tiny, thin native Boolean predicate residue, never geometry.

    PathOps uses float32 coordinates. Repeated operations on the same curved
    boundary can leave a sliver instead of an exactly empty path. Both an area
    bound and a coordinate-scale thickness bound must hold. Source commands
    and their actual painted fill are retained unchanged.
    """
    bounds = residue.computeTightBounds()
    reference_bounds = reference.computeTightBounds()
    scale = max(1.0, *(abs(float(value)) for value in reference_bounds))
    if min(float(bounds.width()), float(bounds.height())) > scale * 2.0**-15:
        return False
    residue_area = SkPath.from_skia(residue).area
    reference_area = SkPath.from_skia(reference).area
    return residue_area <= max(1e-10, reference_area * 1e-8)

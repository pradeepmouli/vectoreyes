"""Version-aware semantic validation for canonical drawing plans.

Static shape validation lives in :mod:`vectormark.drawing_actions` and is the
same Pydantic contract exposed by FastMCP. This module validates only facts
that require the retained drawing version: region existence and evolution,
path-command ownership and order, and terminal-operation ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from .drawing import Drawing, DrawingRegion
from .drawing_actions import DrawingPlan
from .path_geometry import (
    CloneGeometry,
    MoveCommand,
    PathCommand,
    PathGeometry,
    PathGeometryWithCutouts,
    CompoundPathGeometry,
    PathGeometryWithSymmetry,
)


class PlanValidationError(ValueError):
    """A user-facing plan error located by an RFC 6901 JSON pointer."""

    def __init__(self, pointer: str, message: str) -> None:
        self.pointer = pointer
        self.message = message
        super().__init__(f"{pointer}: {message}")


_SCHEMA_VARIANT_TOKENS = frozenset({
    "align", "break", "detect_clones", "detect_primitives", "detect_symmetry",
    "fit", "fit_path", "flat", "linear_gradient", "match", "match_length", "merge",
    "mirror", "normalize_corners", "path", "radial_gradient", "raster",
    "remove", "reverse", "assign_clone", "set_fill", "update_geometry", "fit_primitive",
    "set_parallel", "set_symmetry", "set_z_order", "simplify", "snap",
    "split", "stitch", "line", "circle", "ellipse", "rect", "rounded_rect",
    "polygon",
})


def parse_plan(payload: Mapping[str, object]) -> DrawingPlan:
    """Parse the one canonical static plan contract."""
    try:
        return DrawingPlan.model_validate(payload)
    except ValidationError as error:
        issue = error.errors(include_url=False)[0]
        location = tuple(
            token for token in issue["loc"]
            if not (isinstance(token, str) and token in _SCHEMA_VARIANT_TOKENS)
        )
        pointer = "".join(f"/{_escape(token)}" for token in location)
        raise PlanValidationError(pointer, str(issue["msg"])) from None


def validate_plan(plan: DrawingPlan, drawing: Drawing) -> None:
    """Validate one canonical plan against one retained drawing snapshot."""
    if plan.drawing_id != drawing.drawing_id:
        raise PlanValidationError("/drawing_id", "does not identify the retained drawing")
    source_regions = _region_target_source_regions(drawing)
    commands = _target_commands(drawing)
    known_targets = set(source_regions)
    used_target_ids = set(known_targets)
    z_order_seen = False
    symmetry_targets: set[str] = set()
    symmetry_block = False

    for index, action in enumerate(plan.ops):
        pointer = f"/ops/{index}"
        op = action.model_dump(exclude_none=True)
        name = action.op
        if symmetry_block and name != "detect_symmetry":
            raise PlanValidationError(
                pointer,
                "detect_symmetry must be the final operation or part of a terminal targeted-symmetry block",
            )

        if name == "merge":
            group_id = str(op["id"])
            if group_id in used_target_ids:
                raise PlanValidationError(f"{pointer}/id", "duplicate operation id")
            selected = _expand_merge_regions(tuple(op["regions"]), known_targets, f"{pointer}/regions")
            group_sources = frozenset().union(*(source_regions[region_id] for region_id in selected))
            known_targets.difference_update(selected)
            known_targets.add(group_id)
            for region_id in selected:
                del source_regions[region_id]
            source_regions[group_id] = group_sources
            used_target_ids.add(group_id)
            continue

        if name == "split":
            _known_target(str(op["target"]), known_targets, f"{pointer}/target")
            if index != len(plan.ops) - 1:
                raise PlanValidationError(pointer, "split must be the final operation; refine child regions in a new version")
            continue

        if name in {"detect_primitives", "detect_clones"}:
            if "target" in op:
                _known_target(str(op["target"]), known_targets, f"{pointer}/target")
            continue

        if name == "detect_symmetry":
            target = op.get("target")
            if target is None:
                if index != len(plan.ops) - 1:
                    raise PlanValidationError(pointer, "global detect_symmetry must be the final operation")
                if symmetry_block:
                    raise PlanValidationError(pointer, "global detect_symmetry cannot follow targeted symmetry operations")
            else:
                target_id = str(target)
                _known_target(target_id, known_targets, f"{pointer}/target")
                if target_id in symmetry_targets:
                    raise PlanValidationError(f"{pointer}/target", "target already has a symmetry operation in this plan")
                symmetry_targets.add(target_id)
                symmetry_block = True
            continue

        if name == "stitch":
            for target_index, target_id in enumerate(op.get("targets", ())):
                _known_target(str(target_id), known_targets, f"{pointer}/targets/{target_index}")
            continue

        if name in {"fit_path", "fit_primitive"}:
            _known_target(str(op["target"]), known_targets, f"{pointer}/target")
            continue

        if name == "update_geometry":
            target_id = str(op["target"])
            _known_target(target_id, known_targets, f"{pointer}/target")
            _validate_path_ops(op["ops"], f"{pointer}/ops", commands, target_id)
            continue

        if name == "set_fill":
            _known_target(str(op["target"]), known_targets, f"{pointer}/target")
            continue

        if name == "assign_clone":
            source, target = str(op["source"]), str(op["target"])
            _known_target(source, known_targets, f"{pointer}/source")
            _known_target(target, known_targets, f"{pointer}/target")
            continue

        if name == "set_symmetry":
            _known_target(str(op["target"]), known_targets, f"{pointer}/target")
            continue

        if name == "align":
            _known_target(str(op["target"]), known_targets, f"{pointer}/target")
            _known_target(str(op["reference"]), known_targets, f"{pointer}/reference")
            continue

        if name == "set_z_order":
            if z_order_seen:
                raise PlanValidationError(pointer, "only one set_z_order operation is allowed")
            if index != len(plan.ops) - 1:
                raise PlanValidationError(pointer, "set_z_order must be the final operation")
            z_order_seen = True
            targets = tuple(str(target) for target in op["targets"])
            for target_index, target_id in enumerate(targets):
                _known_target(target_id, known_targets, f"{pointer}/targets/{target_index}")
            if set(targets) != known_targets:
                raise PlanValidationError(f"{pointer}/targets", "must list every target exactly once")
            continue

        raise AssertionError(f"unhandled canonical drawing action: {name}")


def _validate_path_ops(
    path_ops: Sequence[Mapping[str, object]],
    pointer: str,
    commands: Mapping[str, tuple[str, int, int, PathCommand]],
    target_id: str,
) -> None:
    segment_ids = {
        command_id for command_id, record in commands.items()
        if record[0] == target_id and not isinstance(record[3], MoveCommand)
    }
    all_segment_ids = {
        command_id for command_id, record in commands.items()
        if not isinstance(record[3], MoveCommand)
    }
    fitted: set[str] = set()
    removed: set[str] = set()
    broken: set[str] = set()

    for index, op in enumerate(path_ops):
        op_pointer = f"{pointer}/{index}"
        name = str(op["op"])
        if name == "fit":
            target = op.get("target")
            between = op.get("between")
            if target is not None:
                target_id_value = str(target)
                _known_command(target_id_value, segment_ids, f"{op_pointer}/target")
                fitted_targets = (target_id_value,)
            else:
                assert isinstance(between, Sequence) and len(between) == 2
                left, right = str(between[0]), str(between[1])
                if left not in commands or right not in commands:
                    raise PlanValidationError(f"{op_pointer}/between", "unknown retained boundary command")
                left_owner, left_subpath, left_index, _ = commands[left]
                right_owner, right_subpath, right_index, right_command = commands[right]
                if left_owner != target_id or right_owner != target_id:
                    raise PlanValidationError(f"{op_pointer}/between", "boundary commands must belong to the geometry target")
                if left_subpath != right_subpath or left_index >= right_index:
                    raise PlanValidationError(f"{op_pointer}/between", "boundaries must be ordered commands in one subpath")
                if isinstance(right_command, MoveCommand):
                    raise PlanValidationError(f"{op_pointer}/between", "right boundary cannot be a move command")
                fitted_targets = tuple(
                    command_id
                    for command_id, (owner, subpath, command_index, command) in commands.items()
                    if owner == target_id
                    and subpath == left_subpath
                    and left_index < command_index <= right_index
                    and not isinstance(command, MoveCommand)
                )
                if not fitted_targets:
                    raise PlanValidationError(f"{op_pointer}/between", "boundaries must enclose at least one retained command")
            if any(command_id in fitted or command_id in removed for command_id in fitted_targets):
                field = "target" if target is not None else "between"
                raise PlanValidationError(f"{op_pointer}/{field}", "command is already refitted or removed")
            fitted.update(fitted_targets)
            continue

        if name == "remove":
            target = str(op["target"])
            _known_command(target, segment_ids, f"{op_pointer}/target")
            if target in removed or target in fitted:
                raise PlanValidationError(f"{op_pointer}/target", "command is already refitted or removed")
            removed.add(target)
            continue

        if name in {"match", "set_parallel", "match_length", "align", "snap", "mirror"}:
            target, reference = str(op["target"]), str(op["reference"])
            _known_command(target, segment_ids, f"{op_pointer}/target")
            _known_command(reference, all_segment_ids, f"{op_pointer}/reference")
            continue

        if name in {"reverse", "break"}:
            target = str(op["target"])
            _known_command(target, segment_ids, f"{op_pointer}/target")
            if name == "break":
                if target not in fitted or target in broken:
                    raise PlanValidationError(f"{op_pointer}/target", "break requires an unbroken fitted command")
                broken.add(target)
            continue

        if name in {"simplify", "normalize_corners"}:
            continue

        raise AssertionError(f"unhandled canonical path action: {name}")


def _known_target(target_id: str, targets: set[str], pointer: str) -> None:
    if target_id not in targets:
        raise PlanValidationError(pointer, "unknown target")


def _known_command(command_id: str, commands: set[str], pointer: str) -> None:
    if command_id not in commands:
        raise PlanValidationError(pointer, "unknown retained command")


def _expand_merge_regions(requested: Sequence[str], known_targets: set[str], pointer: str) -> tuple[str, ...]:
    expanded: list[str] = []
    for index, region_id in enumerate(requested):
        members = (region_id,) if region_id in known_targets else tuple(
            sorted(target for target in known_targets if target.startswith(f"{region_id}-"))
        )
        if not members:
            raise PlanValidationError(f"{pointer}/{index}", "unknown region")
        for member in members:
            if member in expanded:
                raise PlanValidationError(f"{pointer}/{index}", "region is repeated")
            expanded.append(member)
    return tuple(expanded)


def _region_target_source_regions(drawing: Drawing) -> dict[str, frozenset[str]]:
    source_regions: dict[str, frozenset[str]] = {}

    def visit(region: DrawingRegion) -> None:
        if region.is_leaf:
            if region.id in source_regions:
                raise PlanValidationError("/ops", "base drawing contains duplicate target ids")
            source_regions[region.id] = frozenset(region.source_region_ids or (region.id,))
            return
        for child in region.children:
            visit(child)

    for region in drawing.regions:
        visit(region)
    return source_regions


def _target_commands(drawing: Drawing) -> dict[str, tuple[str, int, int, PathCommand]]:
    commands: dict[str, tuple[str, int, int, PathCommand]] = {}

    def index_path(region_id: str, path: PathGeometry, subpath: int) -> None:
        for command_index, command in enumerate(path.path.commands):
            if command.id in commands:
                raise PlanValidationError("/ops", "retained drawing contains duplicate path command ids")
            commands[command.id] = (region_id, subpath, command_index, command)

    def visit(region: DrawingRegion) -> None:
        if region.is_leaf:
            geometry = region.geometry
            if isinstance(geometry, PathGeometry):
                index_path(region.id, geometry, 0)
            elif isinstance(geometry, CompoundPathGeometry):
                for subpath, path in enumerate(geometry.paths):
                    index_path(region.id, path, subpath)
            elif isinstance(geometry, PathGeometryWithCutouts):
                index_path(region.id, geometry.outer, 0)
                for subpath, cutout in enumerate(geometry.cutouts, start=1):
                    index_path(region.id, cutout, subpath)
            elif isinstance(geometry, PathGeometryWithSymmetry):
                index_path(region.id, geometry.half, 0)
            elif isinstance(geometry, CloneGeometry):
                return
            return
        for child in region.children:
            visit(child)

    for region in drawing.regions:
        visit(region)
    return commands


def _escape(token: object) -> str:
    return str(token).replace("~", "~0").replace("/", "~1")

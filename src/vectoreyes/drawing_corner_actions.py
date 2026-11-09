"""Source-bound corner fitting and annotation publication on canonical regions."""

from dataclasses import replace
from .corner_recipe import CornerRecipe, CornerRecipeExecutor, CornerRecipeReplay
from .drawing import DrawingRegion
from .path_fit_evidence import AutomaticFitEvidence
from .path_geometry import (
    CornerAnchor,
    PathGeometry,
    PathGeometryWithCutouts,
    CompoundPathGeometry,
)
from .path_topology import component_paths, replace_component_paths


EditablePathGeometry = PathGeometry | PathGeometryWithCutouts | CompoundPathGeometry


def displayed_corner_id(
    feature_id: str, source_path_id: str, target_path_id: str
) -> str:
    prefix = source_path_id + "."
    return (
        f"{target_path_id}.{feature_id.removeprefix(prefix)}"
        if feature_id.startswith(prefix)
        else feature_id
    )


def bind_corner_fit_identity(
    geometry: PathGeometry, target_path_id: str, *, source_path_id: str
) -> PathGeometry:
    """Publish fitted values under the selected path's version-scoped addresses."""
    command_ids = {
        command.id: f"{target_path_id}.c{index}"
        for index, command in enumerate(geometry.path.commands)
    }
    commands = tuple(
        replace(command, id=command_ids[command.id])
        for command in geometry.path.commands
    )
    anchors = []
    for anchor in geometry.anchors:
        if not isinstance(anchor, CornerAnchor):
            raise TypeError("corner fitter emitted a non-corner anchor")
        anchors.append(
            replace(
                anchor,
                id=displayed_corner_id(anchor.id, source_path_id, target_path_id),
                entry_command_id=command_ids[anchor.entry_command_id],
                exit_command_id=command_ids[anchor.exit_command_id],
                owned_command_ids=tuple(
                    command_ids[item] for item in anchor.owned_command_ids
                ),
            )
        )
    return PathGeometry(
        replace(geometry.path, id=target_path_id, commands=commands), tuple(anchors)
    )


def corner_diagnostics(geometry: PathGeometry) -> dict[str, object]:
    positions = {
        command.id: index for index, command in enumerate(geometry.path.commands)
    }
    return {
        "path": geometry.to_svg_d(),
        "spans": [
            {
                "corner_id": anchor.id,
                "entry_command": positions[anchor.entry_command_id],
                "exit_command": positions[anchor.exit_command_id],
                "commands": [positions[item] for item in anchor.owned_command_ids],
                "radius": anchor.radius,
            }
            for anchor in geometry.anchors
            if isinstance(anchor, CornerAnchor)
        ],
    }


def apply_region_corner_recipe(
    region: DrawingRegion,
    *,
    path_id: str,
    recipe: CornerRecipe,
    evidence: AutomaticFitEvidence,
    executor: CornerRecipeExecutor | None = None,
) -> tuple[DrawingRegion, CornerRecipeReplay]:
    """Refit one source-bound path; preserve the other component values exactly."""
    canonical = region.geometry
    if not isinstance(
        canonical, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)
    ):
        raise ValueError("corner recipe target has no directly editable path geometry")
    paths = component_paths(canonical)
    matches = [index for index, path in enumerate(paths) if path.path.id == path_id]
    if len(matches) != 1:
        raise ValueError("corner recipe target path is stale")
    replay = (executor or CornerRecipeExecutor()).replay(recipe, evidence)
    fitted = bind_corner_fit_identity(
        replay.fit.geometry, path_id, source_path_id=evidence.source.path.id
    )
    updated = list(paths)
    updated[matches[0]] = fitted
    return replace(
        region,
        geometry=replace_component_paths(canonical, updated),
        diagnostics={**region.diagnostics, "corners": corner_diagnostics(fitted)},
    ), replay


def rebind_corner_annotations(
    parsed: EditablePathGeometry, annotated: EditablePathGeometry
) -> EditablePathGeometry:
    """Keep current geometry and transfer features only across exact topology."""
    current_paths, annotated_paths = component_paths(parsed), component_paths(annotated)
    if tuple(path.path.id for path in current_paths) != tuple(
        path.path.id for path in annotated_paths
    ):
        raise ValueError(
            "corner downstream replay changed path identity; no exact annotation rebinding is available"
        )
    rebound = []
    for current, original in zip(current_paths, annotated_paths, strict=True):
        if tuple(
            (command.id, type(command)) for command in current.path.commands
        ) != tuple((command.id, type(command)) for command in original.path.commands):
            raise ValueError(
                "corner downstream replay changed command topology; no exact annotation rebinding is available"
            )
        rebound.append(replace(current, anchors=original.anchors))
    return replace_component_paths(parsed, rebound)

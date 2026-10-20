"""Working executor boundaries; canonical state is authoritative between steps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..corner_recipe import CornerRecipeReplay
from ..drawing import Canvas
from ..drawing_adapter import (
    drawing_from_vector_regions, restore_selected_roots,
    working_region_evidence, working_shape_metadata,
)
from ..drawing_recipe import (
    _snapshot_canonical, GeometryEvidence, PaintEvidence, EffectiveDrawingRecipe,
    RecipeStep, ApplyCornerRecipeOperation,
)
from ..replay_state import ReplayState
from .vector_region import VectorRegion

if TYPE_CHECKING:
    from ..drawing_replay import _ProducedTarget
    from .automatic_replay import ReplayBatch


def capture_replay_state(regions: Sequence[VectorRegion], canvas: Canvas) -> ReplayState:
    drawing = drawing_from_vector_regions('recipe-state', width=canvas.width,
        height=canvas.height, regions=regions)
    return ReplayState(_snapshot_canonical(drawing),
        _snapshot_canonical(working_shape_metadata(regions, drawing.regions)),
        _snapshot_canonical(working_region_evidence(regions)))


def replay_numeric_batches(state: ReplayState, batches: tuple[ReplayBatch, ...], *,
                           finalize_topology: bool = True) -> ReplayState:
    """Retain numeric transaction/receipt semantics around shared engine operations."""
    from .automatic_replay import replay_batches
    from .topology_binding import normalize_working_topology
    regions = list(working_regions_from_state(state))
    masks = {int(root.id): root.raster for root in regions}
    selected = replay_batches(regions, batches, masks=masks)
    if finalize_topology:
        selected = normalize_working_topology(selected)
    return capture_replay_state(selected, state.drawing.canvas)


def working_regions_from_state(state: ReplayState) -> tuple[VectorRegion, ...]:
    drawing = _snapshot_canonical(state.drawing)
    return restore_selected_roots(drawing.regions, tuple(root.id for root in drawing.regions), (),
        shape_metadata=state.shape_metadata, region_evidence=state.region_evidence)


def replay_working_with_producers(
    evidence: GeometryEvidence, paint: PaintEvidence, recipe: EffectiveDrawingRecipe, *,
    rebound_steps: list[RecipeStep] | None = None, rebind_from_index: int = 0,
    finalize_output: bool = True,
) -> tuple[tuple[VectorRegion, ...], dict[str, _ProducedTarget],
           list[tuple[ApplyCornerRecipeOperation, CornerRecipeReplay]]]:
    from ..drawing_replay import _replay_drawing_with_producers
    state, producers, corners = _replay_drawing_with_producers(evidence, paint, recipe,
        rebound_steps=rebound_steps, rebind_from_index=rebind_from_index,
        finalize_output=finalize_output)
    return working_regions_from_state(state), producers, corners


def replay_working_regions(
    evidence: GeometryEvidence, paint: PaintEvidence, recipe: EffectiveDrawingRecipe, *,
    finalize_output: bool = True,
) -> tuple[VectorRegion, ...]:
    from ..drawing_replay import replay_drawing
    return working_regions_from_state(replay_drawing(evidence, paint, recipe,
        finalize_output=finalize_output))

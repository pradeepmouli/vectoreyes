from dataclasses import replace

import numpy as np
import pytest

from vectormark.corner_recipe import (
    AddCorner,
    CornerRecipeExecutor,
    RemoveCorner,
    ReplaceCorner,
)
from vectormark.drawing_actions import ActionSettings
from vectormark.drawing_engine import PathFitOptions, RasterImage
from vectormark.drawing_recipe import (
    ApplyCornerRecipeOperation,
    EffectiveDrawingRecipe,
    ResolveOriginalPaintOperation,
    VersionPathSelection,
    build_geometry_evidence,
    build_paint_evidence,
)
from vectormark.drawing_assembly import root_regions
from vectormark.drawing_replay import (
    extend_recipe_with_corner_edits,
    propose_effective_recipe,
    replay_regions,
)
from vectormark.drawing_state import DrawingStore, VersionCandidate
from vectormark.drawing_trace import PythonTraceEngine, TraceOptions
from vectormark.optimizer.automatic_replay import input_revision
from vectormark.path_geometry import CornerAnchor, PathGeometry, PathGeometryWithCutouts


def _source_store(*, cutout: bool = False):
    rgb = np.full((64, 64, 3), 255, dtype=np.uint8)
    rgb[8 if cutout else 16 : 56 if cutout else 49, 8 if cutout else 16 : 56 if cutout else 49] = (20, 100, 240)
    if cutout:
        rgb[20:44, 20:44] = 255
    trace = PythonTraceEngine().trace(rgb, TraceOptions())
    rooted = root_regions(trace, rgb)
    source = RasterImage(64, 64, "rgb8", rgb.tobytes())
    evidence = build_geometry_evidence(
        kind="source",
        trace=rooted.trace,
        geometry_source=source,
        trace_roots=rooted.regions,
    )
    paint = build_paint_evidence(source)
    settings = ActionSettings(epsilon=1.0, max_error=1.5, fit_strategy="quadratic")
    base_recipe = EffectiveDrawingRecipe(
        evidence_revision=evidence.revision,
        engine_revision=evidence.engine_revision,
        settings=settings,
    )
    drawing_id = "drw_corner_replay"
    execution = propose_effective_recipe(
        drawing_id=drawing_id, evidence=evidence, paint=paint, recipe=base_recipe
    )
    store = DrawingStore()
    (base,) = store.create(
        drawing_id,
        paint,
        (
            VersionCandidate(
                drawing=execution.drawing,
                evidence=evidence,
                submitted_plan=None,
                submitted_edits=(),
                effective_recipe=base_recipe,
                replay=execution.replay,
                baseline_candidate_index=0,
            ),
        ),
    )
    return store, base, paint, evidence


def _selection(version, evidence) -> VersionPathSelection:
    geometry = version.drawing.regions[0].geometry
    assert isinstance(geometry, PathGeometry)
    command = geometry.path.commands[0].id
    return VersionPathSelection(
        version.id,
        evidence.revision,
        version.effective_recipe.revision,
        (1,),
        geometry.path.id,
        command,
        command,
    )


def _candidate(execution, recipe, edits, *, baseline="v0") -> VersionCandidate:
    return VersionCandidate(
        drawing=execution.drawing,
        evidence=None,
        submitted_plan=None,
        submitted_edits=edits,
        effective_recipe=recipe,
        replay=execution.replay,
        baseline_version_id=baseline,
    )


def test_add_then_remove_replays_like_never_added_sibling_and_preserves_branches(
    monkeypatch,
):
    store, base, paint, evidence = _source_store()
    source = evidence.root_paths[0].fit.source
    source_command = source.path.commands[10].id
    add = AddCorner(CornerAnchor("agent.corner", source_command, source_command))
    added_recipe, added_edits = extend_recipe_with_corner_edits(
        evidence=evidence,
        paint=paint,
        base_recipe=base.effective_recipe,
        base_version_id=base.id,
        selection=_selection(base, evidence),
        edits=(add,),
    )
    added_execution = propose_effective_recipe(
        drawing_id=base.drawing.drawing_id,
        evidence=evidence,
        paint=paint,
        recipe=added_recipe,
    )
    (added,) = store.append(
        base.drawing.drawing_id,
        base.id,
        (_candidate(added_execution, added_recipe, added_edits),),
    )
    added_geometry = added.drawing.regions[0].geometry
    assert isinstance(added_geometry, PathGeometry)
    assert "agent.corner" in {anchor.id for anchor in added_geometry.anchors}

    removed_recipe, removed_edits = extend_recipe_with_corner_edits(
        evidence=evidence,
        paint=paint,
        base_recipe=added.effective_recipe,
        base_version_id=added.id,
        selection=_selection(added, evidence),
        edits=(RemoveCorner("agent.corner"),),
    )
    monkeypatch.setattr(
        "vectormark.optimizer.corners.path_corner_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("corner detector ran during retained recipe replay")
        ),
    )
    removed_execution = propose_effective_recipe(
        drawing_id=base.drawing.drawing_id,
        evidence=evidence,
        paint=paint,
        recipe=removed_recipe,
    )
    (removed,) = store.append(
        base.drawing.drawing_id,
        added.id,
        (_candidate(removed_execution, removed_recipe, removed_edits),),
    )

    root_path = evidence.root_paths[0]
    seed = CornerRecipeExecutor().create(
        root_path.fit,
        PathFitOptions(
            base.resolved_settings.epsilon,
            base.resolved_settings.max_error,
            base.resolved_settings.fit_strategy,
        ),
    )
    base_regions = replay_regions(evidence, paint, base.effective_recipe)
    never_added_recipe = replace(
        base.effective_recipe,
        steps=(
            *base.effective_recipe.steps,
            ApplyCornerRecipeOperation(
                seed,
                evidence.revision,
                (1,),
                "r1",
                root_path.path_id,
                input_revision(list(base_regions)),
            ),
            ResolveOriginalPaintOperation(
                scope="changed",
                share_root_gradients=evidence.trace.material_pretrace is None,
                input_revision=input_revision(list(base_regions)),
            ),
        ),
    )
    never_added_execution = propose_effective_recipe(
        drawing_id=base.drawing.drawing_id,
        evidence=evidence,
        paint=paint,
        recipe=never_added_recipe,
    )
    (sibling,) = store.append(
        base.drawing.drawing_id,
        base.id,
        (_candidate(never_added_execution, never_added_recipe, ()),),
    )

    assert removed.drawing == sibling.drawing
    assert store.get_version(base.drawing.drawing_id, base.id) == base
    assert store.get_version(base.drawing.drawing_id, added.id) == added
    assert added.id == "v0.0" and removed.id == "v0.0.0" and sibling.id == "v0.1"
    assert "canonical_path_geometry" not in str(removed.effective_recipe)


def test_replace_keeps_feature_identity_changes_source_binding_and_stale_selection_is_atomic():
    store, base, paint, evidence = _source_store()
    source = evidence.root_paths[0].fit.source
    original = source.anchors[0]
    replacement = CornerAnchor(
        original.id,
        source.path.commands[31].id,
        source.path.commands[31].id,
    )
    manifest = store.get_manifest(base.drawing.drawing_id)
    with pytest.raises(ValueError, match="recipe revision is stale"):
        extend_recipe_with_corner_edits(
            evidence=evidence,
            paint=paint,
            base_recipe=base.effective_recipe,
            base_version_id=base.id,
            selection=replace(_selection(base, evidence), recipe_revision="stale"),
            edits=(ReplaceCorner(original.id, replacement),),
        )
    assert store.get_manifest(base.drawing.drawing_id) == manifest

    recipe, edits = extend_recipe_with_corner_edits(
        evidence=evidence,
        paint=paint,
        base_recipe=base.effective_recipe,
        base_version_id=base.id,
        selection=_selection(base, evidence),
        edits=(ReplaceCorner(original.id, replacement),),
    )
    execution = propose_effective_recipe(
        drawing_id=base.drawing.drawing_id,
        evidence=evidence,
        paint=paint,
        recipe=recipe,
    )
    (child,) = store.append(
        base.drawing.drawing_id,
        base.id,
        (_candidate(execution, recipe, edits),),
    )

    assert child.id == "v0.0"
    assert edits[0].binding.source_entry_command_id == replacement.entry_command_id
    assert edits[0].binding.source_entry_command_id != original.entry_command_id
    geometry = child.drawing.regions[0].geometry
    assert isinstance(geometry, PathGeometry)
    revised = next(anchor for anchor in geometry.anchors if anchor.id == "r1.p0.corner0")
    assert revised.id == base.drawing.regions[0].geometry.anchors[0].id


def test_corner_selection_rejects_wrong_public_version_even_when_recipe_matches():
    _store, base, paint, evidence = _source_store()

    with pytest.raises(ValueError, match="version is stale"):
        extend_recipe_with_corner_edits(
            evidence=evidence,
            paint=paint,
            base_recipe=base.effective_recipe,
            base_version_id=base.id,
            selection=replace(_selection(base, evidence), version_id="v999"),
            edits=(RemoveCorner("missing"),),
        )


def test_corner_recipe_replaces_one_cutout_path_without_touching_its_outer_path():
    _store, base, paint, evidence = _source_store(cutout=True)
    geometry = base.drawing.region("r1").geometry
    assert isinstance(geometry, PathGeometryWithCutouts)
    cutout = geometry.cutouts[0]
    source = next(
        item.fit.source
        for item in evidence.root_paths
        if item.region_id == "r1" and item.path_id == cutout.path.id
    )
    source_command = source.path.commands[0].id
    displayed_command = cutout.path.commands[0].id
    selection = VersionPathSelection(
        base.id,
        evidence.revision,
        base.effective_recipe.revision,
        (1,),
        cutout.path.id,
        displayed_command,
        displayed_command,
    )

    recipe, _edits = extend_recipe_with_corner_edits(
        evidence=evidence,
        paint=paint,
        base_recipe=base.effective_recipe,
        base_version_id=base.id,
        selection=selection,
        edits=(AddCorner(CornerAnchor("cutout.agent", source_command, source_command)),),
    )
    replayed = propose_effective_recipe(
        drawing_id=base.drawing.drawing_id,
        evidence=evidence,
        paint=paint,
        recipe=recipe,
    ).drawing.region("r1").geometry

    assert isinstance(replayed, PathGeometryWithCutouts)
    assert replayed.outer == geometry.outer
    assert "cutout.agent" in {anchor.id for anchor in replayed.cutouts[0].anchors}

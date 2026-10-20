from __future__ import annotations

from dataclasses import replace

import numpy as np

from vectormark.drawing import Canvas
from vectormark.drawing_actions import ActionSettings
from vectormark.drawing_adapter import drawing_from_vector_regions
from vectormark.drawing_engine import RasterImage
from tests._working_evidence import capture_working_geometry_evidence
from vectormark.optimizer.assembly_binding import working_regions_from_evidence
from vectormark.drawing_recipe import (
    EffectiveDrawingRecipe,
    build_paint_evidence,
    )
from vectormark.drawing_replay import propose_effective_recipe
from vectormark.drawing_trace import TraceOptions, TraceResult
from vectormark.optimizer.automatic_replay import FitPrimitiveOperation
from vectormark.optimizer.framework import optimize_with_intent
from vectormark.optimizer.passes.primitives import primitives_pass
from vectormark.optimizer.trace import trace_regions
from vectormark.pipeline import Options


def _disk_image() -> np.ndarray:
    yy, xx = np.ogrid[:120, :120]
    mask = (yy - 60) ** 2 + (xx - 60) ** 2 <= 40**2
    image = np.full((120, 120, 3), 255, np.uint8)
    image[mask] = (200, 30, 30)
    return image


def test_effective_auto_recipe_replays_from_evidence_without_detector(monkeypatch):
    rgb = _disk_image()
    objects, masks = trace_regions(rgb, Options())
    roots = tuple(
        replace(
            root,
            drawing_id=f"r{root.id}",
            source_regions=(f"r{root.id}",),
        )
        for root in objects
    )
    masks = {int(root.id): np.asarray(root.raster, dtype=bool) for root in roots}
    source = RasterImage(120, 120, "rgb8", rgb.tobytes())
    trace = TraceResult(120, 120, TraceOptions(), (), "<svg/>")
    evidence = capture_working_geometry_evidence(
        kind="source",
        trace=trace,
        geometry_source=source,
        trace_roots=roots,
    )
    paint = build_paint_evidence(source)
    optimized = optimize_with_intent(list(working_regions_from_evidence(evidence)), masks, [primitives_pass])
    assert len(optimized.batches[0].operations) == 1
    assert isinstance(optimized.batches[0].operations[0].intent.action, FitPrimitiveOperation)
    recipe = EffectiveDrawingRecipe(
        evidence_revision=evidence.revision,
        engine_revision=evidence.engine_revision,
        settings=ActionSettings(
            epsilon=1.0, max_error=1.5, fit_strategy="quadratic"
        ),
        steps=optimized.batches,
    )
    expected = drawing_from_vector_regions(
        "drw_recipe",
        width=Canvas(120, 120).width,
        height=Canvas(120, 120).height,
        regions=optimized.objects,
    )

    monkeypatch.setattr(
        "vectormark.optimizer.passes.primitives.recognize_primitive_geometry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("semantic primitive detector ran during replay")
        ),
    )
    execution = propose_effective_recipe(
        drawing_id="drw_recipe", evidence=evidence, paint=paint, recipe=recipe
    )

    assert execution.drawing == expected
    assert execution.replay.recipe_revision == recipe.revision
    assert execution.replay.output_revision


def test_replay_fails_closed_for_unimplemented_retained_operations():
    rgb = _disk_image()
    objects, _masks = trace_regions(rgb, Options())
    roots = tuple(
        replace(root, drawing_id=f"r{root.id}", source_regions=(f"r{root.id}",))
        for root in objects
    )
    source = RasterImage(120, 120, "rgb8", rgb.tobytes())
    evidence = capture_working_geometry_evidence(
        kind="source",
        trace=TraceResult(120, 120, TraceOptions(), (), "<svg/>"),
        geometry_source=source,
        trace_roots=roots,
    )
    recipe = EffectiveDrawingRecipe(
        evidence_revision=evidence.revision,
        engine_revision=evidence.engine_revision,
        settings=ActionSettings(
            epsilon=1.0, max_error=1.5, fit_strategy="quadratic"
        ),
        steps=(object(),),  # type: ignore[arg-type]
    )

    import pytest

    with pytest.raises(TypeError, match="unsupported effective recipe step"):
        propose_effective_recipe(
            drawing_id="drw_recipe",
            evidence=evidence,
            paint=build_paint_evidence(source),
            recipe=recipe,
        )

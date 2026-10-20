"""Canonical evidence fixtures for tests of still-working optimizer consumers."""

from vectormark.drawing_adapter import (
    drawing_from_vector_regions, working_region_evidence, working_shape_metadata,
)
from vectormark.drawing_recipe import build_geometry_evidence
from vectormark.optimizer.topology_binding import normalize_working_topology


def capture_working_geometry_evidence(*, trace_roots, trace, **options):
    roots = normalize_working_topology(trace_roots)
    drawing = drawing_from_vector_regions('geometry-evidence', width=trace.width,
        height=trace.height, regions=roots)
    return build_geometry_evidence(trace=trace, drawing=drawing,
        shape_metadata=working_shape_metadata(roots, drawing.regions),
        region_evidence=working_region_evidence(roots), **options)


def retain_working_operation_outcome(operation, before, after, *, evidence, paint, upstream):
    """Capture old working fixtures through the canonical selected-result API."""
    from vectormark.optimizer.replay_binding import capture_replay_state, working_regions_from_state
    from vectormark.selected_outcomes import retain_selected_outcome
    operation, state = retain_selected_outcome(operation,
        capture_replay_state(before, paint.canvas), capture_replay_state(after, paint.canvas),
        evidence=evidence, paint=paint, upstream=upstream)
    return operation, working_regions_from_state(state)

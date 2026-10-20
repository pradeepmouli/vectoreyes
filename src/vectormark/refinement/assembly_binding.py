"""Disposable input codecs for refinement consumers of canonical root assembly."""

from dataclasses import dataclass
import numpy as np

from ..drawing_assembly import DecompositionObserver, assemble_drawing_roots
from ..drawing_recipe import GeometryEvidence, _snapshot_canonical
from ..drawing_trace import TraceResult
from .region_sample import RegionSample


@dataclass(frozen=True)
class WorkingRootResult:
    regions: tuple[RegionSample, ...]
    trace: TraceResult


def working_regions_from_evidence(evidence: GeometryEvidence) -> tuple[RegionSample, ...]:
    """Detach canonical values before projecting disposable refinement inputs."""
    from ..drawing_adapter import restore_selected_roots

    drawing = _snapshot_canonical(evidence.drawing)
    return restore_selected_roots(drawing.regions,
        tuple(root.id for root in drawing.regions), (),
        shape_metadata=evidence.shape_metadata, region_evidence=evidence.region_evidence)


def assemble_working_roots(
    trace: TraceResult, rgb: np.ndarray | None = None, *,
    material_rgb: np.ndarray | None = None, min_region_fraction: float = 0.02,
    allow_gradients: bool = True, preserve_material_labels: bool | None = None,
    on_decomposition_stage: DecompositionObserver | None = None,
) -> WorkingRootResult:
    """Materialize disposable inputs only for consumers still requiring them."""
    from ..drawing_adapter import restore_selected_roots

    result = assemble_drawing_roots(trace, rgb, material_rgb=material_rgb,
        min_region_fraction=min_region_fraction, allow_gradients=allow_gradients,
        preserve_material_labels=preserve_material_labels,
        on_decomposition_stage=on_decomposition_stage)
    regions = restore_selected_roots(result.drawing.regions,
        tuple(root.id for root in result.drawing.regions), (),
        shape_metadata=result.shape_metadata, region_evidence=result.region_evidence)
    return WorkingRootResult(regions, result.trace)

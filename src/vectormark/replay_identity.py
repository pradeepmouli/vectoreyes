"""Replay input identity: selected values and observations, not derived footprints."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

from .candidate import Fill
from .drawing import DrawingRegion
from .fit import Shape
from .replay_state import ReplayState, retained_region_raster
from .types import CoverageField

if TYPE_CHECKING:
    from .optimizer.automatic_replay import ReplayBatch


@dataclass(frozen=True)
class RegionInput:
    id: int
    current: Shape | None
    original: Shape | None
    fill: Fill | None
    z: float
    raster: np.ndarray
    color_hex: str | None
    children: tuple['RegionInput', ...]
    source_label: int | None
    drawing_id: str | None
    source_regions: tuple[str, ...]
    coverage: CoverageField | np.ndarray | None
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        from .optimizer.automatic_replay import _semantic_diagnostics
        object.__setattr__(self, 'diagnostics', _semantic_diagnostics(self.diagnostics))


def input_records_revision(records: Sequence[RegionInput], masks: Mapping[int, np.ndarray]) -> str:
    from .optimizer.automatic_replay import _stable
    ids = {record.id for record in records}
    relevant = {int(identity): np.asarray(mask, dtype=bool)
        for identity, mask in masks.items() if int(identity) in ids}
    payload = json.dumps(_stable({'contract': 'vectormark.replay.input.v2',
        'regions': tuple(records), 'masks': relevant}), sort_keys=True,
        separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def scoped_input_regions(state: ReplayState, scope: tuple[int, ...]) -> tuple[DrawingRegion, ...]:
    current = state.drawing.regions
    for identity in scope:
        matches = [region for region in current if int(state.region_evidence[region.id]['id']) == identity]
        if len(matches) != 1:
            raise ValueError(f'automatic replay scope is not unique: {scope!r}')
        current = matches[0].children
    return tuple(current)


def canonical_input_records(state: ReplayState, scope: tuple[int, ...] = ()) -> tuple[RegionInput, ...]:
    from .drawing import _walk_regions
    from .drawing_adapter import current_shape_for_region, working_fill_from_spec
    from .drawing_fill import serialize_fill

    indexed = {region.id: region for root in state.drawing.regions for region in _walk_regions(root)}
    paths = {identity: tuple(info['ancestry']) for identity, info in state.region_evidence.items()}

    def visit(region: DrawingRegion) -> RegionInput:
        info = state.region_evidence[region.id]
        original = None if info['original_kind'] is None else Shape(info['original_kind'], dict(info['original_params']))
        coverage = None
        if info.get('coverage') is not None:
            coverage = np.frombuffer(info['coverage'], dtype=info['coverage_dtype']).reshape(info['coverage_shape'])
            if info['coverage_origin'] is not None:
                coverage = CoverageField(coverage, info['coverage_origin'], info['coverage_canvas'])
        return RegionInput(int(info['id']),
            None if region.is_branch else current_shape_for_region(region, indexed, paths,
                state.shape_metadata.get(region.id, {})), original,
            None if region.fill is None else working_fill_from_spec(serialize_fill(region.fill)),
            float(region.z), retained_region_raster(state, region.id), info['color_hex'],
            tuple(visit(child) for child in region.children), info['source_label'], region.id,
            tuple(region.source_region_ids), coverage, region.diagnostics)
    return tuple(visit(region) for region in scoped_input_regions(state, scope))


def canonical_input_revision(state: ReplayState, scope: tuple[int, ...] = ()) -> str:
    records = canonical_input_records(state, scope)
    return input_records_revision(records, {record.id: record.raster for record in records if record.raster.size})


def validate_batch_revision(actual: str, batch: 'ReplayBatch') -> None:
    if actual != batch.input_revision:
        passes = ', '.join(dict.fromkeys(entry.pass_name for entry in batch.entries))
        raise ValueError(f'automatic replay input revision mismatch ({passes or "empty batch"}, scope={batch.scope})')


def bind_automatic_inputs(state: ReplayState, batch: 'ReplayBatch') -> dict[int, np.ndarray]:
    records = canonical_input_records(state, batch.scope)
    masks = {record.id: record.raster.copy() for record in records if record.raster.size}
    validate_batch_revision(input_records_revision(records, masks), batch)
    return masks

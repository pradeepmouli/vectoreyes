"""Representation-independent addressing for a single replay transaction."""

from typing import Callable, Generic, Mapping, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .optimizer.automatic_replay import NamedRegionOutput, RegionRef, ReplayBatch


T = TypeVar('T')


class ReplayInputResolver(Generic[T]):
    """Keep pass snapshots separate from results produced inside the pass."""

    def __init__(self, inputs: Mapping[tuple[int, ...], T], batch: 'ReplayBatch',
                 index_descendants: Callable[[T], Mapping[tuple[int, ...], T]]) -> None:
        self._batch = batch
        self._inputs = inputs
        self._index_descendants = index_descendants
        self._outputs: dict[tuple[str, str], T] = {}
        self._output_paths: dict[tuple[str, str], tuple[int, ...]] = {}

    def resolve(self, ref: 'RegionRef') -> T:
        from .optimizer.automatic_replay import PassInputRegionRef
        if isinstance(ref, PassInputRegionRef):
            if ref.batch_id != self._batch.batch_id or ref.input_revision != self._batch.input_revision:
                raise ValueError('automatic replay input binding does not belong to this batch revision')
            try:
                return self._inputs[tuple(ref.ancestry)]
            except KeyError as exc:
                raise ValueError(f'automatic replay input path is absent: {ref.ancestry!r}') from exc
        key = (ref.operation_id, ref.output_name)
        try:
            region = self._outputs[key]
        except KeyError as exc:
            raise ValueError(f'automatic replay output is absent or forward-referenced: {key!r}') from exc
        if not ref.descendant_ancestry:
            return region
        path = (self._output_paths[key][-1], *ref.descendant_ancestry)
        try:
            return self._index_descendants(region)[path]
        except KeyError as exc:
            raise ValueError(f'automatic replay output descendant is absent: {path!r}') from exc

    def register(self, operation_id: str, output: 'NamedRegionOutput', region: T) -> None:
        key = (operation_id, output.name)
        if key in self._outputs:
            raise ValueError(f'duplicate automatic replay named output: {key!r}')
        self._outputs[key] = region
        self._output_paths[key] = output.ancestry

    def path(self, ref: 'RegionRef') -> tuple[int, ...]:
        from .optimizer.automatic_replay import PassInputRegionRef
        self.resolve(ref)
        if isinstance(ref, PassInputRegionRef):
            return tuple(ref.ancestry)
        return (*self._output_paths[(ref.operation_id, ref.output_name)], *ref.descendant_ancestry)


def clone_report(source_id: int, fill_name: str, transform: tuple[float, ...],
                 family_size: int | None) -> dict[str, object]:
    report: dict[str, object] = {'accepted': True, 'matched_source': source_id,
        'fill_preserved': fill_name, 'transform': transform}
    if family_size is not None:
        report['family_size'] = family_size
    return report


def mirror_report(source_id: int, axis: tuple[float, float, float]) -> dict[str, object]:
    import math
    from .drawing_actions import SymmetryAxisInput
    selected = SymmetryAxisInput(point=(float(axis[1]), float(axis[2])),
        angle_degrees=math.degrees(float(axis[0])))
    return {'accepted': True, 'mode': 'pair', 'matched_source': source_id,
        'axis': selected.model_dump(mode='json')}


def union_masks(mask_by_id, identities):
    import numpy as np
    ordered = tuple(sorted(int(identity) for identity in identities))
    union = np.zeros_like(mask_by_id[ordered[0]], dtype=bool)
    for identity in ordered:
        union |= np.asarray(mask_by_id[identity], dtype=bool)
    return union


def assign_replacement_masks(replacement_ids, proposal_ids, *, original_ids,
                             pass_masks, current_masks, raster_for):
    """Assign raster ownership independently of the selected region representation."""
    import numpy as np
    union = union_masks(pass_masks, proposal_ids)
    preserved = {identity: np.asarray(pass_masks[identity], dtype=bool).copy()
                 for identity in proposal_ids}
    split = len(proposal_ids) == 1 and len(replacement_ids) > 1
    for identity in proposal_ids:
        current_masks.pop(identity, None)
    assigned, result = set(), {}
    for identity in replacement_ids:
        current_masks.pop(identity, None)
        if split:
            mask = raster_for(identity, union.shape)
        elif identity in original_ids and identity in proposal_ids and identity not in assigned:
            mask = preserved[identity]
            assigned.add(identity)
        else:
            mask = union.copy()
        result[identity] = current_masks[identity] = mask
    return result

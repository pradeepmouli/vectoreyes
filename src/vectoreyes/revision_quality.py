"""Region-aware shape assessment on canonical drawings, independent of paint.

Filled-union fidelity alone cannot detect a displaced internal material edge.
Explicit region correspondence keeps those boundaries accountable while allowing
declared merges and splits. The original reference and immediate parent retain
independent measurements and budgets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Literal, cast

from .drawing import Drawing, DrawingRegion, _walk_regions
from .geometry_native import region_path
from .quality import GeometryAssessment, GeometryPolicy, assess_geometry
from .skia_geometry import SkPath, unary_union


@dataclass(frozen=True)
class RegionMatch:
    before_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    reference_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ('before_ids', 'candidate_ids', 'reference_ids'):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class RegionQuality:
    before_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    step: GeometryAssessment
    reference: GeometryAssessment


@dataclass(frozen=True)
class QualityReport:
    geometry_accepted: bool
    regions: tuple[RegionQuality, ...]
    coverage: GeometryAssessment
    reasons: tuple[str, ...]
    status: Literal['geometry_passed', 'needs_refinement']
    fill_status: Literal['not_assessed'] = 'not_assessed'

    def diagnostics(self) -> dict[str, object]:
        """Emit JSON-safe metrics; absent finite measurements remain explicit."""
        def finite(value: object) -> object:
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, Mapping):
                return {str(key): finite(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite(item) for item in value]
            return value
        return cast(dict[str, object], finite({
            'geometry_accepted': self.geometry_accepted, 'status': self.status,
            'fill_status': self.fill_status, 'reasons': self.reasons,
            'coverage': self.coverage.diagnostics(),
            'regions': [{
                'before_ids': item.before_ids, 'candidate_ids': item.candidate_ids,
                'reference_ids': item.reference_ids, 'step': item.step.diagnostics(),
                'reference': item.reference.diagnostics(),
            } for item in self.regions],
        }))


def _index(drawing: Drawing) -> dict[str, DrawingRegion]:
    return {region.id: region for root in drawing.regions for region in _walk_regions(root)}


def _complete_correspondence(matches: Sequence[RegionMatch], field: str,
                             indexed: Mapping[str, DrawingRegion]) -> bool:
    seen: set[str] = set()
    for match in matches:
        identities = getattr(match, field)
        if not identities or len(set(identities)) != len(identities):
            return False
        for identity in identities:
            if identity not in indexed:
                return False
            leaves = {leaf.id for leaf in indexed[identity].leaves()}
            if seen.intersection(leaves):
                return False
            seen.update(leaves)
    return seen == {identity for identity, region in indexed.items() if region.is_leaf}


def assess_revision(
    reference: Drawing, before: Drawing, candidate: Drawing, *,
    policy: GeometryPolicy,
    reference_policy: GeometryPolicy | None = None,
    matches: Sequence[RegionMatch] | None = None,
) -> QualityReport:
    """Assess every matched region, total coverage and cumulative shape drift.

    With no explicit matches, leaves must retain their identities. Correspondence
    may instead name complete groups/branches for a declared merge or split. It
    must account for every painted leaf exactly once in all three drawings.
    Reference measurements are informational unless a reference budget is given;
    an inaccurate baseline must not silently prohibit an intentional improvement.
    """
    reference_index, before_index, candidate_index = map(_index, (reference, before, candidate))
    selected = tuple(matches) if matches is not None else tuple(
        RegionMatch((identity,), (identity,), (identity,))
        for identity, region in before_index.items() if region.is_leaf)
    selected = tuple(RegionMatch(item.before_ids, item.candidate_ids,
        item.reference_ids or item.before_ids) for item in selected)
    reasons: list[str] = []
    if not reference.canvas == before.canvas == candidate.canvas:
        reasons.append('canvas_registration_mismatch')
    correspondence_valid = all(_complete_correspondence(selected, field, indexed)
        for field, indexed in (('reference_ids', reference_index),
            ('before_ids', before_index), ('candidate_ids', candidate_index)))
    if not correspondence_valid:
        reasons.append('invalid_region_correspondence')

    def paths(indexed: Mapping[str, DrawingRegion]) -> dict[str, SkPath]:
        result = {}
        for identity, region in indexed.items():
            try:
                result[identity] = region_path(region, indexed)
            except (ValueError, TypeError, KeyError, ArithmeticError):
                reasons.append(f'invalid_geometry:{identity}')
                result[identity] = SkPath()
        return result

    reference_paths, before_paths, candidate_paths = map(paths, (reference_index, before_index, candidate_index))

    def union(identities: Sequence[str], values: Mapping[str, SkPath]) -> SkPath:
        return unary_union([values[identity] for identity in identities])

    coverage = assess_geometry(
        union(tuple(region.id for region in before.regions), before_paths),
        union(tuple(region.id for region in candidate.regions), candidate_paths), policy)
    regions = []
    if correspondence_valid:
        for match in selected:
            output = union(match.candidate_ids, candidate_paths)
            step = assess_geometry(union(match.before_ids, before_paths), output, policy)
            original = assess_geometry(union(match.reference_ids, reference_paths), output,
                reference_policy or GeometryPolicy(preserve_topology=False))
            regions.append(RegionQuality(match.before_ids, match.candidate_ids,
                match.reference_ids, step, original))
            for reason in step.reasons:
                reasons.append(f"step:{','.join(match.candidate_ids)}:{reason}")
            if reference_policy is not None:
                for reason in original.reasons:
                    reasons.append(f"reference:{','.join(match.candidate_ids)}:{reason}")
    reasons.extend(f'coverage:{reason}' for reason in coverage.reasons)
    unique_reasons = tuple(dict.fromkeys(reasons))
    return QualityReport(not unique_reasons, tuple(regions), coverage, unique_reasons,
        'needs_refinement' if unique_reasons else 'geometry_passed')

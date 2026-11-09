"""User-facing manual-selection policy (slice 4b). An agent/user restricts which
geometry strategies are considered for an element (pre-execution) and/or overrides
the auto-scored winner (post-evaluation), addressed per element by the stable `sN`
id the emit layer stamps. Imports nothing from the pipeline (kept dependency-free so
Options can hold it without a cycle).

Only *fitted* elements (regions, gradient footprints) are policy-addressable.
Reconstructed-occlusion and lens primitives advance the same `sN` index counter
(occlusion is emitted with a visible `id="sN"`; lens is emitted as a plain path with
no `id` at all), but neither is produced by the strategy selector — a policy keyed to
one of those indices is silently ignored (out of scope: manual selection steers fitted
geometry, not occlusion reconstruction). Labels are validated lazily, per element when
first consulted, so a typo in a `by_id` entry whose element is never emitted will not
raise."""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass

# Strategy provenance labels — one per fitter in generate_geometry_candidates.
PRIMITIVE = "primitive"          # recognize_primitive -> circle/rect/ellipse
TRAPEZOID = "trapezoid"          # rounded_trapezoid_fit
SYM_POLYGON = "sym_polygon"      # symmetric_polygon_fit
CAP = "cap"                      # half_ellipse_cap_fit
SYMMETRIC = "symmetric"          # symmetric_fit
POLYGON = "polygon"              # recognize_polygon
PATH = "path"                    # fit_path
HOLED_SYM = "holed_symmetric"    # multi-contour mirrored halves (even-odd)
HOLED_PATH = "holed_path"        # multi-contour per-contour fit (even-odd)

KNOWN_STRATEGIES = frozenset({
    PRIMITIVE, TRAPEZOID, SYM_POLYGON, CAP, SYMMETRIC,
    POLYGON, PATH, HOLED_SYM, HOLED_PATH,
})


@dataclass(frozen=True)
class ElementSelection:
    """One element's manual policy. `allow` (None = all) restricts which strategies
    are scored; `force` (None = auto) names the strategy whose candidate should win."""
    allow: frozenset[str] | None = None
    force: str | None = None


_EMPTY: "Mapping[str, ElementSelection]" = types.MappingProxyType({})


@dataclass(frozen=True)
class SelectionPolicy:
    """Per-element selection keyed by emit-time id (`s0`, `s1`, ...), with an optional
    `default` applied to elements that have no explicit entry."""
    by_id: Mapping[str, ElementSelection] = _EMPTY
    default: ElementSelection | None = None

    def __post_init__(self):
        # freeze the mapping so a frozen policy is truly immutable
        if not isinstance(self.by_id, types.MappingProxyType):
            object.__setattr__(self, "by_id", types.MappingProxyType(dict(self.by_id)))

    def for_id(self, eid: str) -> ElementSelection | None:
        return self.by_id.get(eid, self.default)


def validate_strategies(sel: ElementSelection) -> None:
    """Raise ValueError if `allow` or `force` names a strategy outside KNOWN_STRATEGIES
    (catches typos loudly instead of silently warning per element)."""
    labels = set(sel.allow or ())
    if sel.force is not None:
        labels.add(sel.force)
    unknown = labels - KNOWN_STRATEGIES
    if unknown:
        raise ValueError(
            f"unknown selection strategy {sorted(unknown)}; "
            f"known: {sorted(KNOWN_STRATEGIES)}"
        )

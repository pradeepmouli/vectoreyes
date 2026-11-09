"""Immutable scene values shared by tracing, refinement, and SVG emission.

``Drawing`` deliberately contains no version graph, source raster, temporary
footprint, or refinement buffers.  Those belong respectively to persistence,
trace provenance, and engine working state.  A drawing is the vector scene
that can be rendered, refined, and persisted as a version snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite

from .drawing_fill import Fill
from .background import BackgroundColor
from .immutable_value import freeze_value
from .path_geometry import Geometry
from .output_settings import DrawingOutputSettings


@dataclass(frozen=True)
class Canvas:
    """The finite drawing-coordinate extent used by all region geometry."""

    width: float
    height: float

    def __post_init__(self) -> None:
        if not all(isfinite(float(value)) and value > 0 for value in (self.width, self.height)):
            raise ValueError("canvas dimensions must be finite and positive")


@dataclass(frozen=True)
class DrawingRegion:
    """One addressable, painted vector region in a drawing.

    ``source_region_ids`` is the region-level provenance for geometry and fill
    decisions.  It intentionally does not appear on commands: a command can
    change or be regenerated while the region lineage remains meaningful.
    """

    id: str
    geometry: Geometry | None = None
    fill: Fill | None = None
    z: float = 0.0
    source_region_ids: tuple[str, ...] = ()
    children: tuple["DrawingRegion", ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)
    name: str | None = None
    cutout_targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from .region_names import validate_region_name
        validate_region_name(self.name)
        object.__setattr__(self, "cutout_targets", tuple(self.cutout_targets))
        if any(not target or target == self.id for target in self.cutout_targets):
            raise ValueError("cutout source and targets must differ and have nonempty IDs")
        if len(set(self.cutout_targets)) != len(self.cutout_targets):
            raise ValueError("cutout targets must be unique")
        object.__setattr__(self, "source_region_ids", tuple(self.source_region_ids))
        object.__setattr__(self, "children", tuple(self.children))
        if not self.id:
            raise ValueError("drawing region IDs must not be empty")
        if self.geometry is None and not self.children:
            raise ValueError("drawing-region branches require children")
        if self.geometry is not None and self.children:
            raise ValueError("drawing-region leaves cannot also have children")
        if len({child.id for child in self.children}) != len(self.children):
            raise ValueError("drawing-region child IDs must be unique")
        if not isfinite(float(self.z)):
            raise ValueError("drawing-region z must be finite")
        if any(not source_id for source_id in self.source_region_ids):
            raise ValueError("source region IDs must not be empty")
        if len(set(self.source_region_ids)) != len(self.source_region_ids):
            raise ValueError("source region IDs must be unique")
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))

    @property
    def is_leaf(self) -> bool:
        return self.geometry is not None

    @property
    def is_branch(self) -> bool:
        return self.geometry is None

    def leaves(self) -> tuple["DrawingRegion", ...]:
        """Return the renderable leaves below this region in paint-tree order."""
        if self.is_leaf:
            return (self,)
        return tuple(leaf for child in self.children for leaf in child.leaves())


@dataclass(frozen=True)
class Drawing:
    """A complete, renderable vector scene with stable drawing identity.

    Versions, parent links, and retention remain store concerns.  The drawing
    ID travels with every snapshot so plans and artifact references can name
    it without a parallel scene-to-store lookup.
    """

    drawing_id: str
    canvas: Canvas
    regions: tuple[DrawingRegion, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)
    background: BackgroundColor | None = None
    output_settings: DrawingOutputSettings = field(default_factory=DrawingOutputSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))
        if not self.drawing_id:
            raise ValueError("drawing IDs must not be empty")
        if not self.regions:
            raise ValueError("drawings require at least one region")
        all_regions = tuple(region for root in self.regions for region in _walk_regions(root))
        if len({region.id for region in all_regions}) != len(all_regions):
            raise ValueError("drawing region IDs must be unique")
        known = {region.id for region in all_regions}
        if any(target not in known for region in all_regions for target in region.cutout_targets):
            raise ValueError("cutout relationship references an absent region; reassign it before removing the target")
        _validate_cutouts(all_regions)
        object.__setattr__(self, "diagnostics", freeze_value(self.diagnostics))

    def region(self, region_id: str) -> DrawingRegion:
        """Return one addressable region or raise ``KeyError``."""
        for root in self.regions:
            for region in _walk_regions(root):
                if region.id == region_id:
                    return region
        raise KeyError(region_id)


def _validate_cutouts(regions: tuple[DrawingRegion, ...]) -> None:
    indexed = {region.id:region for region in regions}
    operands, targets = set(), set()
    for source in regions:
        if not source.cutout_targets:
            continue
        source_ids = {leaf.id for leaf in source.leaves()}
        target_ids = {leaf.id for identity in source.cutout_targets for leaf in indexed[identity].leaves()}
        if source_ids & target_ids:
            raise ValueError('a cutout cannot subtract from itself or an overlapping subtree')
        operands.update(source_ids)
        targets.update(target_ids)
    if operands & targets:
        raise ValueError('nested or cyclic cutout relationships require an explicit intermediate region')


def _walk_regions(region: DrawingRegion) -> tuple[DrawingRegion, ...]:
    """Yield a region followed by all descendant regions in stable tree order."""
    return (region, *(descendant for child in region.children for descendant in _walk_regions(child)))

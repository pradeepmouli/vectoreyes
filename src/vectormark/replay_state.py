"""Canonical geometry and immutable observations carried between recipe steps."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .drawing import Drawing, DrawingRegion, _walk_regions
from .immutable_value import freeze_value


@dataclass(frozen=True)
class ReplayState:
    drawing: Drawing
    shape_metadata: Mapping[str, Mapping[str, object]]
    region_evidence: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'shape_metadata', freeze_value(self.shape_metadata))
        object.__setattr__(self, 'region_evidence', freeze_value(self.region_evidence))


def retained_region_raster(state: ReplayState, identity: str) -> np.ndarray:
    """Read only the raster observation needed by a region fit, not a scene codec."""
    info = state.region_evidence.get(identity, {})
    if info.get('raster') is None or info.get('raster_shape') is None:
        raise ValueError(f'region {identity} has no retained raster observation')
    return np.frombuffer(info['raster'], dtype=bool).reshape(info['raster_shape'])


def publish_drawing_state(
    state: ReplayState, drawing: Drawing, *,
    shape_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> ReplayState:
    """Publish canonical geometry and inherit observations without projecting shapes.

    Geometry changes retire kernel annotations. New components inherit source
    observations but receive their own raster, while a merged root unions the
    retained source rasters. Existing observations remain immutable.
    """
    from .geometry_native import region_path
    from .optimizer.gate import rasterize
    from .path_topology import component_paths
    from .path_geometry import PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry

    prior = {region.id: region for root in state.drawing.regions for region in _walk_regions(root)}
    indexed = {region.id: region for root in drawing.regions for region in _walk_regions(root)}
    observations = {key: dict(value) for key, value in state.region_evidence.items()}
    metadata = {}
    for identity, region in indexed.items():
        if region.is_leaf:
            previous = prior.get(identity)
            metadata[identity] = {
                **(state.shape_metadata.get(identity, {}) if previous is not None
                    and previous.geometry == region.geometry else {}),
                **(shape_metadata or {}).get(identity, {}),
            }
            if identity not in prior and isinstance(region.geometry,
                    (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
                paths = component_paths(region.geometry)
                if paths:
                    metadata[identity].update(path_ids=tuple(path.path.id for path in paths),
                        fill_rule='evenodd' if len(paths) > 1 else 'nonzero')
    next_id = max((int(info['id']) for info in observations.values() if 'id' in info), default=0) + 1

    def inherit(region: DrawingRegion, parent: DrawingRegion | None = None) -> None:
        nonlocal next_id
        if region.id not in observations:
            if parent is None:
                sources = tuple(region.diagnostics.get('merge', {}).get('sources', ()))
                if not sources or any(identity not in observations for identity in sources):
                    raise ValueError('new canonical root requires retained source observations')
                info = dict(observations[sources[0]])
                raster = np.zeros(info['raster_shape'], dtype=bool)
                for identity in sources:
                    raster |= retained_region_raster(state, identity)
                observations[region.id] = {**info, 'id': next_id, 'ancestry': (next_id,),
                    'raster': raster.tobytes(), 'raster_source': None, 'source_label': None,
                    'original_kind': 'path', 'original_params': {'d': region_path(region, indexed).to_svg_d()},
                    'coverage': None, 'coverage_source': None, 'coverage_shape': None,
                    'coverage_dtype': None, 'coverage_origin': None, 'coverage_canvas': None}
                next_id += 1
            else:
                info = observations[parent.id]
                identity = int(region.id.rsplit('-', 1)[-1])
                observations[region.id] = {**info, 'id': identity, 'ancestry': (*info['ancestry'], identity)}
                if region.is_leaf:
                    raster = rasterize(region_path(region, indexed), info['raster_shape'])
                    observations[region.id].update(raster=raster.tobytes(), raster_source=None)
        for child in region.children:
            inherit(child, region)

    for root in drawing.regions:
        inherit(root)
    return ReplayState(drawing, metadata, {key: value for key, value in observations.items() if key in indexed})

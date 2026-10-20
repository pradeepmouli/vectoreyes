from __future__ import annotations

import numpy as np

from ..candidate import FlatFill
from ..components import decompose_components
from ..contour import region_contours
from ..fill_fit import fit_fill
from ..fit import Shape, fit_path
from ..pipeline import Options, _segment_image
from ..surface_merge import merge_surfaces
from ..types import Region
from .region_sample import RegionSample, to_polygon


def _region_path_contours(region: Region) -> list[np.ndarray]:
    return [c for c in region_contours(region.mask, coverage=region.coverage) if len(c) >= 3]


def _trace_shape_from_contours(contours: list[np.ndarray], opt: Options) -> Shape | None:
    if not contours:
        return None

    ds = [
        fit_path(
            contour,
            epsilon=opt.epsilon,
            max_error=opt.max_error,
            cubic=False,
        ).params["d"]
        for contour in contours
    ]
    params = {"d": " ".join(ds)}
    if len(ds) > 1:
        params["fill_rule"] = "evenodd"
    return Shape("path", params)


def _trace_shape(region: Region, opt: Options) -> Shape | None:
    return _trace_shape_from_contours(_region_path_contours(region), opt)


def trace_regions(arr: np.ndarray, opt: Options) -> tuple[list[RegionSample], dict[int, np.ndarray]]:
    w, h, regions = _segment_image(arr, opt)
    components = decompose_components(regions, (h, w))

    merged_regions: list[Region] = []
    fills_by_label = {}
    for comp in components:
        flat_filled = [(r, FlatFill(r.color_hex)) for r in comp]
        merged = merge_surfaces(flat_filled, arr)
        comp_regions = [r for r, _ in merged]
        merged_regions.extend(comp_regions)
        fills_by_label.update({
            r.label: fit_fill(r.mask, arr, flat_hex=r.color_hex)
            for r in comp_regions
        })

    ordered = sorted(
        merged_regions,
        key=lambda r: (-r.area, r.label, r.color_hex),
    )

    region_samples: list[RegionSample] = []
    true_masks: dict[int, np.ndarray] = {}
    for idx, region in enumerate(ordered):
        contours = _region_path_contours(region)
        shape = _trace_shape_from_contours(contours, opt)
        if shape is None:
            continue
        raster = region.mask.copy()
        region_sample = RegionSample.from_shape(
            id=idx,
            shape=shape,
            fill=fills_by_label.get(region.label, FlatFill(region.color_hex)),
            z=idx,
            raster=raster,
            footprint=to_polygon(shape),
            source_label=region.label,
            color_hex=region.color_hex,
            coverage=region.coverage,
            diagnostics={
                "trace": {
                    "source_label": int(region.label),
                    "color_hex": region.color_hex,
                    "area": int(region.area),
                    "shape": shape.kind,
                    "contours": len(contours),
                }
            },
        )
        region_samples.append(region_sample)
        true_masks[region_sample.id] = raster

    return region_samples, true_masks

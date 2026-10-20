"""Canonical source-paint fitting; visibility/grouping are orchestration concerns."""

from dataclasses import dataclass, replace
from collections.abc import Mapping

import numpy as np
from scipy import ndimage

from . import candidate
from .drawing import DrawingRegion
from .drawing_fill import Fill, FlatFill, GradientStop, LinearGradientFill, RadialGradientFill, RasterFill
from .fill_fit import fit_fill
from .immutable_value import freeze_value
from .output_settings import DrawingOutputSettings
from .color import quantize


@dataclass(frozen=True)
class PaintFitResult:
    fill: Fill
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'diagnostics', freeze_value(self.diagnostics))


def canonical_fill(fill: object) -> Fill | None:
    """Project the shared numeric fill fitter's values into canonical paint."""
    if isinstance(fill, candidate.FlatFill):
        return FlatFill(fill.hex)
    if isinstance(fill, candidate.LinearGradientFill):
        geometry = fill.geometry
        return LinearGradientFill((geometry['x1'], geometry['y1']), (geometry['x2'], geometry['y2']),
            tuple(GradientStop(offset, color) for offset, color in fill.stops))
    if isinstance(fill, candidate.RadialGradientFill):
        geometry = fill.geometry
        return RadialGradientFill((geometry['cx'], geometry['cy']), geometry['r'],
            tuple(GradientStop(offset, color) for offset, color in fill.stops),
            focal_point=(geometry.get('fx', geometry['cx']), geometry.get('fy', geometry['cy']))
            if 'fx' in geometry or 'fy' in geometry else None)
    if isinstance(fill, candidate.RasterFill):
        geometry = fill.geometry
        return RasterFill((geometry['x'], geometry['y']), geometry['w'], geometry['h'], fill.png_b64)
    return None


def sampled_flat_hex(mask: np.ndarray, rgb: np.ndarray) -> str:
    """Preserve the source median fallback used by the numeric fill fitter."""
    pixels = rgb[mask]
    if len(pixels) == 0:
        return '#000000'
    red, green, blue = np.rint(np.median(pixels, axis=0)).astype(np.uint8)
    return f'#{red:02X}{green:02X}{blue:02X}'


def source_sample_mask(source_mask: np.ndarray, visible_mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Sample visible source material, retaining the existing thin-detail fallback."""
    source_mask = np.asarray(source_mask, dtype=bool)
    if source_mask.shape != visible_mask.shape:
        return visible_mask, 'visible_footprint'
    supported = visible_mask & source_mask
    minimum_support = max(3, int(np.ceil(int(visible_mask.sum()) * .05)))
    core = ndimage.binary_erosion(supported, iterations=2)
    if int(core.sum()) >= minimum_support:
        return core, 'visible_source_material_core'
    if int(supported.sum()) >= minimum_support:
        return supported, 'visible_source_material'
    return visible_mask, 'visible_footprint'


def fit_source_paint(
    rgb: np.ndarray, *, source_mask: np.ndarray,
    visible_mask: np.ndarray, allow_gradients: bool = True,
) -> PaintFitResult:
    """Fit canonical paint from pixel evidence without interpreting geometry."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or visible_mask.shape != rgb.shape[:2]:
        raise ValueError('visible mask must match the RGB canvas')
    samples, domain = source_sample_mask(source_mask, np.asarray(visible_mask, dtype=bool))
    fill = canonical_fill(fit_fill(samples, rgb, flat_hex=sampled_flat_hex(samples, rgb), allow_gradients=allow_gradients))
    if fill is None:
        raise ValueError('fill fitter returned unsupported paint')
    return PaintFitResult(fill, {'status': 'refit', 'mask': domain, 'sampled_pixels': int(samples.sum())})


def fit_region_paint(
    region: DrawingRegion, rgb: np.ndarray, *, source_mask: np.ndarray,
    visible_mask: np.ndarray, allow_gradients: bool = True,
) -> DrawingRegion:
    """Fit one leaf's paint from retained source samples; never change geometry."""
    if not region.is_leaf:
        raise ValueError('paint fitting requires a drawing-region leaf')
    result = fit_source_paint(rgb, source_mask=source_mask, visible_mask=visible_mask, allow_gradients=allow_gradients)
    return replace(region, fill=result.fill, diagnostics={**region.diagnostics, 'fill': result.diagnostics})


def constrain_output_paint(
    fill: Fill | None, *, rgb: np.ndarray, sample_mask: np.ndarray,
    settings: DrawingOutputSettings, palette: np.ndarray | None = None,
) -> Fill | None:
    """Apply selected output limits to paint alone, never to geometry."""
    if settings.allow_gradients and palette is None:
        return fill

    def flat_source_color() -> str:
        if sample_mask.shape != rgb.shape[:2]:
            raise ValueError('paint sample mask must match the RGB canvas')
        core = ndimage.binary_erosion(sample_mask, iterations=2)
        return sampled_flat_hex(core if core.any() else sample_mask, rgb)

    def color(value: str) -> str:
        if palette is None:
            return value
        sample = np.asarray([[[int(value[i:i + 2], 16) for i in (1, 3, 5)]]], dtype=np.uint8)
        red, green, blue = quantize(sample, palette)[0, 0]
        return f'#{red:02X}{green:02X}{blue:02X}'

    if not settings.allow_gradients and not isinstance(fill, FlatFill):
        fill = FlatFill(flat_source_color())
    if isinstance(fill, FlatFill):
        return FlatFill(color(fill.color))
    if isinstance(fill, (LinearGradientFill, RadialGradientFill)):
        stops = tuple(GradientStop(stop.offset, color(stop.color)) for stop in fill.stops)
        if len({stop.color for stop in stops}) == 1:
            return FlatFill(stops[0].color)
        return replace(fill, stops=stops)
    if isinstance(fill, RasterFill) and palette is not None:
        return FlatFill(color(flat_source_color()))
    return fill

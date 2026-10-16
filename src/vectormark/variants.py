# src/vectormark/variants.py
"""Whole-mark variant matrix: idealize one raster across an epsilon × max_error
grid, for SVG export, a JSON manifest, and an annotated contact sheet."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .fit import _fmt
from .pipeline import AxisLine, IdealizeReport, Options, _flatten_on_white, idealize
from .score import SvgRendererUnavailable, _rasterize

DEFAULT_EPSILONS = (0.5, 1.5, 3.0)
DEFAULT_MAX_ERRORS = (0.5, 1.0, 2.5)


@dataclass(frozen=True)
class Variant:
    """One matrix cell: the (epsilon, max_error) used, the emitted SVG, the
    strategy report, and an error string if that cell failed to idealize."""

    epsilon: float
    max_error: float
    svg: str
    report: IdealizeReport
    error: str | None = None


def _as_rgb(image) -> np.ndarray:
    """Load any accepted image input to an (H, W, 3) uint8 array ONCE, so the grid
    does not re-read/re-flatten the source per cell."""
    if isinstance(image, str):
        with Image.open(image) as im:
            return _flatten_on_white(im)
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return _flatten_on_white(Image.fromarray(arr, "RGBA"))
    return arr


def _axis_tag(value: float) -> str:
    """Filename-safe axis token: 0.5 -> '0_5', 3.0 -> '3', 1.0 -> '1'."""
    return _fmt(value).replace(".", "_")


def variant_filename(v: Variant) -> str:
    """SVG filename for one matrix cell, e.g. 'variant-e0_5-m1.svg'."""
    return f"variant-e{_axis_tag(v.epsilon)}-m{_axis_tag(v.max_error)}.svg"


def write_variant_set(variants: list[Variant], out_dir, *, source: str) -> Path:
    """Write one SVG per successful cell plus manifest.json into out_dir (created if
    needed). A failed cell contributes a manifest entry with `error` and no `file`.
    Returns the out_dir path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    epsilons, max_errors = [], []
    for v in variants:
        if v.epsilon not in epsilons:
            epsilons.append(v.epsilon)
        if v.max_error not in max_errors:
            max_errors.append(v.max_error)

    entries = []
    for v in variants:
        entry = {"epsilon": v.epsilon, "max_error": v.max_error}
        if v.error is not None:
            entry["error"] = v.error
        else:
            fname = variant_filename(v)
            (out / fname).write_text(v.svg)
            entry.update(
                file=fname,
                svg_bytes=len(v.svg.encode()),
                strategies=dict(v.report.strategies),
                gradients=v.report.gradients,
                elements=v.report.elements,
            )
        entries.append(entry)

    manifest = {
        "source": source,
        "axes": {"epsilon": epsilons, "max_error": max_errors},
        "variants": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def generate_variants(
    image, *, epsilons=DEFAULT_EPSILONS, max_errors=DEFAULT_MAX_ERRORS,
    base: Options | None = None,
) -> list[Variant]:
    """idealize() the image once per (epsilon, max_error) cell, row-major (epsilon
    outer, max_error inner). `base` supplies the non-geometry knobs held constant
    across the grid (max_sampling_colors, no_symmetry, …); epsilon/max_error are overridden
    per cell. A cell that raises is captured as a failed Variant, never aborting
    the grid."""
    arr = _as_rgb(image)
    base = base or Options()
    out: list[Variant] = []
    for eps in epsilons:
        for me in max_errors:
            opt = replace(base, epsilon=eps, max_error=me)
            try:
                svg, report = idealize(arr, options=opt, report=True)
                out.append(Variant(eps, me, svg, report))
            except Exception as exc:                       # one bad cell must not kill the grid
                out.append(Variant(eps, me, "", IdealizeReport.empty(), error=str(exc)))
    return out


# ---------------------------------------------------------------------------
# Contact-sheet composer
# ---------------------------------------------------------------------------

_TILE = 220          # rendered variant size (px)
_PAD = 10            # gap between tiles
_LABEL_H = 26        # caption strip height under each tile
_AXIS_W = 80         # left gutter for epsilon row labels
_AXIS_H = 24         # top strip for max_error column labels


def _histogram_caption(report: IdealizeReport) -> str:
    """Compact strategy histogram, e.g. 'path:12 holed:3 prim:1'. ASCII only — the
    contact sheet draws with PIL's default bitmap font, which has no glyph for × / ε."""
    short = {"primitive": "prim", "sym_polygon": "sym_poly", "trapezoid": "trap",
             "symmetric": "sym", "polygon": "poly", "holed_symmetric": "holed_sym",
             "holed_path": "holed", "cap": "cap", "path": "path"}
    # dominant strategy first (count desc, then name)
    ordered = sorted(report.strategies.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = [f"{short.get(k, k)}:{n}" for k, n in ordered]
    if report.gradients:
        parts.append(f"grad:{report.gradients}")
    return "  ".join(parts) or "(none)"


_AXIS_STROKE = "#e000a0"   # magenta — distinct from typical logo palettes


def _axis_line_svg(a: AxisLine, stroke_w: float) -> str:
    dash = max(6.0, stroke_w * 4.0)
    return (f'<line x1="{a.x1:.1f}" y1="{a.y1:.1f}" x2="{a.x2:.1f}" y2="{a.y2:.1f}" '
            f'stroke="{_AXIS_STROKE}" stroke-width="{stroke_w:.2f}" '
            f'stroke-dasharray="{dash:.1f},{dash * 0.7:.1f}" stroke-opacity="0.85"/>')


def _inject_axes(svg: str, axes: tuple[AxisLine, ...]) -> str:
    """Insert one dashed <line> per axis (already in viewBox coords) just before the
    closing </svg>, scaling stroke width to the viewBox so it renders at any size."""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    extent = max(float(m.group(1)), float(m.group(2))) if m else 100.0
    stroke_w = max(1.0, extent / 250.0)
    fragment = "".join(_axis_line_svg(a, stroke_w) for a in axes)
    head, _, tail = svg.rpartition("</svg>")          # inject before the LAST </svg> only
    return f"{head}{fragment}</svg>{tail}" if head else svg + fragment


def _render_tile(v: Variant, *, draw_axes: bool = False) -> Image.Image:
    """One variant rendered into a _TILE×_TILE white tile, or a placeholder for a
    failed cell. Raises SvgRendererUnavailable if the renderer is missing."""
    tile = Image.new("RGB", (_TILE, _TILE), "white")
    if v.error is None and v.svg:
        svg = _inject_axes(v.svg, v.report.axes) if draw_axes and v.report.axes else v.svg
        arr = _rasterize(svg, _TILE, _TILE)          # may raise SvgRendererUnavailable
        tile = Image.fromarray(arr)
    else:
        ImageDraw.Draw(tile).text((8, _TILE // 2), "failed", fill=(180, 40, 40))
    return tile


def compose_contact_sheet(variants: list[Variant], *, epsilons, max_errors, draw_axes: bool = False) -> bytes | None:
    """Render the variants into an annotated grid PNG (epsilon rows × max_error
    columns), each tile captioned with its strategy histogram and the axes labelled.
    Returns PNG bytes, or None if the SVG renderer is unavailable."""
    by_cell = {(v.epsilon, v.max_error): v for v in variants}
    cell_w = _TILE + _PAD
    cell_h = _TILE + _LABEL_H + _PAD
    sheet_w = _AXIS_W + len(max_errors) * cell_w + _PAD
    sheet_h = _AXIS_H + len(epsilons) * cell_h + _PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for ci, me in enumerate(max_errors):
        x = _AXIS_W + ci * cell_w + _PAD
        draw.text((x, 6), f"max_error={_fmt(me)}", fill=(20, 30, 40))

    try:
        for ri, eps in enumerate(epsilons):
            y0 = _AXIS_H + ri * cell_h + _PAD
            draw.text((8, y0 + _TILE // 2), f"eps={_fmt(eps)}", fill=(20, 30, 40))
            for ci, me in enumerate(max_errors):
                v = by_cell[(eps, me)]
                tile = _render_tile(v, draw_axes=draw_axes)  # may raise SvgRendererUnavailable
                x = _AXIS_W + ci * cell_w + _PAD
                sheet.paste(tile, (x, y0))
                draw.text((x, y0 + _TILE + 4), _histogram_caption(v.report), fill=(60, 70, 80))
    except SvgRendererUnavailable:
        return None

    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()

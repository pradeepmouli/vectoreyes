#!/usr/bin/env python3
"""Render an MCP trace/refine workflow into a review panel.

This deliberately calls the same ``trace_drawing`` and artifact handlers that
an agent uses, rather than the one-shot CLI pipeline.  It makes it easy to
compare the source, retained v0 roots, auto-refined child version, and labeled
targets when changing drawing-state or refinement behavior.

Example:
    uv run python scripts/generate_mcp_drawing_panel.py corpus/input/vbird.png \\
        --output /tmp/vbird-mcp-panel.png
"""

from __future__ import annotations

from vectoreyes.drawing_adapter import drawing_from_region_samples as _export_drawing, normalize_drawing_addresses as _export_addresses
from vectoreyes.drawing_render import render_drawing as _canonical_render, render_labeled_drawing as _canonical_labels
import argparse
import base64
from collections.abc import Mapping
import html
import io
import json
from pathlib import Path
import re

import numpy as np
import resvg_py
from PIL import Image, ImageDraw, ImageFont

from vectoreyes._fitcurve import cubic_inflects
from vectoreyes.contour import region_contours
from vectoreyes.drawing_adapter import drawing_from_region_samples
from vectoreyes.drawing_render import render_path_map
from vectoreyes.drawing_refine import auto_refine, stitch_regions
from vectoreyes.refinement.assembly_binding import assemble_working_roots
from vectoreyes.emit import render_svg_doc, resolve_fill
from vectoreyes.fit import _fmt
from vectoreyes.mcp_image import DEFAULT_MAX_SIZE_PX
from vectoreyes.mcp_server import ImageRef, TraceDrawingOptions, _trace_result
from vectoreyes.corner_diagnostics import path_corner_diagnostics
from vectoreyes.path_sampling import _parse_subpaths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="PNG, JPEG, or WebP input")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG panel")
    parser.add_argument("--colors", default="16", help="Trace palette ceiling, or 'auto'")
    parser.add_argument(
        "--fit-strategy",
        choices=("quadratic", "progressive", "progressive_allow_lines"),
        default="quadratic",
        help="Contour fitting strategy used by trace_drawing",
    )
    parser.add_argument(
        "--trace-level",
        choices=("pixel", "subpixel"),
        default="pixel",
        help="Boundary extraction precision used before path fitting.",
    )
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE_PX, help="MCP preprocess maximum edge")
    parser.add_argument(
        "--debug-passes",
        action="store_true",
        help="Write SVG snapshots after trace and every auto-refinement pass, plus an HTML review index.",
    )
    parser.add_argument(
        "--corner-normalize",
        action="store_true",
        help="Canonicalize recognized rounded corners during automatic refinement.",
    )
    parser.add_argument(
        "--gradient-pretrace",
        action="store_true",
        help="Build a gradient-neutral material proxy before palette inference and tracing.",
    )
    parser.add_argument(
        "--support-weighted-corners",
        action="store_true",
        help="Preserve shallow turns when long supporting runs make their displacement significant.",
    )
    parser.add_argument(
        "--v0-only",
        action="store_true",
        help="Stop after initial stitching instead of running automatic refinement.",
    )
    return parser.parse_args()


def _render_svg(svg: str, width: int, height: int) -> Image.Image:
    png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height)
    rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    rendered = Image.new("RGB", rgba.size, "white")
    rendered.paste(rgba, mask=rgba.getchannel("A"))
    return rendered


def _tile(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    preview = image.convert("RGB")
    preview.thumbnail((360, 280), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (380, 320), "#F6F7F9")
    tile.paste(preview, ((380 - preview.width) // 2, 30 + (280 - preview.height) // 2))
    ImageDraw.Draw(tile).text((10, 8), label, fill="#111827", font=font)
    return tile


_PATH_D = re.compile(r'<path\b[^>]*\bd="([^"]*)"')
_PATH_ELEMENT = re.compile(r'<path\b(?P<attributes>[^>]*)\bd="(?P<d>[^"]*)"[^>]*>')
_ID_ATTRIBUTE = re.compile(r'\bid="([^"]+)"')
_SVG_ELEMENT = re.compile(r'<(path|circle|ellipse|rect|polygon|use)\b')
_SVG_COMMAND = re.compile(r"[MLQCAZ]")


def _path_inflections(d: str) -> int:
    """Count actual interior inflections in cubic commands within ``d``."""
    count = 0
    for subpath in _parse_subpaths(d):
        cursor: np.ndarray | None = None
        subpath_start: np.ndarray | None = None
        for command, values in subpath:
            if command == "M":
                cursor = np.asarray(values[:2], dtype=float)
                subpath_start = cursor.copy()
            elif command == "L":
                cursor = np.asarray(values[:2], dtype=float)
            elif command == "Q":
                cursor = np.asarray(values[2:4], dtype=float)
            elif command == "C":
                if cursor is not None:
                    cubic = np.asarray(
                        [cursor, values[:2], values[2:4], values[4:6]],
                        dtype=float,
                    )
                    count += len(cubic_inflects(cubic))
                cursor = np.asarray(values[4:6], dtype=float)
            elif command == "A":
                cursor = np.asarray(values[5:7], dtype=float)
            elif command == "Z" and subpath_start is not None:
                cursor = subpath_start.copy()
    return count


def _path_elements(svg: str) -> dict[str, str]:
    """Return emitted path data keyed by the same target IDs used in reports."""
    paths: dict[str, str] = {}
    for index, match in enumerate(_PATH_ELEMENT.finditer(svg)):
        attributes, d = match.group("attributes"), match.group("d")
        identifier = _ID_ATTRIBUTE.search(attributes)
        target_id = identifier.group(1) if identifier is not None else f"path_{index}"
        paths[target_id] = d
    return paths


def _path_diagnostics(paths: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """Return command/corner metrics keyed by emitted target ID."""
    diagnostics: dict[str, dict[str, int]] = {}
    for region_id, d in paths.items():
        commands = {command: _SVG_COMMAND.findall(d).count(command) for command in ("L", "Q", "C", "A")}
        corners = path_corner_diagnostics(d)
        diagnostics[region_id] = {
            "drawable_commands": sum(commands.values()),
            **commands,
            "corners": len(corners["spans"]),
            "corner_commands": int(corners["commands"]["corner"]),
            "free_commands": int(corners["commands"]["free"]),
            "inflections": _path_inflections(d),
        }
    return diagnostics


def _declared_corner_metrics(diagnostics: Mapping[str, object]) -> dict[str, int] | None:
    """Count retained pre-fit corners without rediscovering them from geometry."""
    subpaths = diagnostics.get("subpaths")
    if isinstance(subpaths, (list, tuple)):
        children = [
            metrics
            for child in subpaths
            if isinstance(child, Mapping)
            for metrics in [_declared_corner_metrics(child)]
            if metrics is not None
        ]
        if not children:
            return None
        return {
            key: sum(child[key] for child in children)
            for key in ("corners", "corner_commands", "free_commands", "corner_anchors")
        }

    spans = diagnostics.get("spans")
    commands = diagnostics.get("commands")
    anchors = diagnostics.get("anchor_counts")
    if not isinstance(spans, (list, tuple)) or not isinstance(commands, Mapping):
        return None
    return {
        "corners": len(spans),
        "corner_commands": int(commands.get("corner", 0)),
        "free_commands": int(commands.get("free", 0)),
        "corner_anchors": int(anchors.get("corner", 0)) if isinstance(anchors, Mapping) else 0,
    }


def _report_with_path_diagnostics(svg: str, report: dict[str, object] | None = None) -> dict[str, object]:
    """Return the standard MCP drawing-report shape enriched with path metrics.

    Real drawing snapshots retain their native MCP report; trace/decomposition
    snapshots synthesize the same target envelope so the panel never exposes a
    second diagnostics schema.
    """
    path_data = _path_elements(svg)
    path_metrics = _path_diagnostics(path_data)
    if report is None:
        return {
            "targets": [
                {
                    "id": target_id,
                    "root_id": target_id,
                    "source_regions": [],
                    "geometry": "path",
                    "fill": "unresolved",
                    "z": index,
                    "diagnostics": {"path": metrics},
                }
                for index, (target_id, metrics) in enumerate(path_metrics.items())
            ]
        }
    targets: list[dict[str, object]] = []
    for target in report.get("targets", ()):
        target_record = dict(target)
        diagnostics = dict(target_record.get("diagnostics", {}))
        target_id = str(target_record["id"])
        if target_id in path_metrics:
            metrics = path_metrics.pop(target_id)
            declared = diagnostics.get("corners")
            declared_metrics = _declared_corner_metrics(declared) if isinstance(declared, Mapping) else None
            if declared_metrics is not None:
                # Counts describe detected design intent, so they come from
                # retained pre-fit provenance. The emitted SVG is still the
                # authority for L/Q/C totals and actual cubic inflections.
                metrics.update(declared_metrics)
            diagnostics["path"] = metrics
        target_record["diagnostics"] = diagnostics
        targets.append(target_record)
    # SVG-only overlays can exist without a retained target. Keep them
    # addressable in the same report schema instead of dropping diagnostics.
    for target_id, metrics in path_metrics.items():
        targets.append({
            "id": target_id,
            "root_id": target_id,
            "source_regions": [],
            "geometry": "path",
            "fill": "unresolved",
            "z": len(targets),
            "diagnostics": {"path": metrics},
        })
    return {"targets": targets}


def _stage_parameters(
    name: str,
    trace,
    *,
    min_region_fraction: float,
    corner_normalize: bool,
) -> dict[str, object]:
    """Expose only the controls that govern the named diagnostic stage."""
    from vectoreyes.fit import FIT_REFERENCE_CANVAS_EXTENT, MIN_PROTECTED_POLYLINE_LINE_LENGTH

    options = trace.options
    trace_fit = {
        "simplify_tolerance": options.simplify_tolerance,
        "curve_tolerance": options.curve_tolerance,
        "fit_strategy": options.fit_strategy,
        "corner_rounding": options.corner_rounding,
        "geometric_support_scale": max(trace.width, trace.height) / FIT_REFERENCE_CANVAS_EXTENT,
        "protected_line_support_px": MIN_PROTECTED_POLYLINE_LINE_LENGTH * max(trace.width, trace.height) / FIT_REFERENCE_CANVAS_EXTENT,
    }
    if name == "source":
        return {"canvas": [trace.width, trace.height]}
    if name == "high_color_quantized":
        return {"max_sampling_colors": 64, "min_region_size": trace.options.min_region_size}
    if name == "shape_merged_supports":
        return {"merge_evidence": "noncredible shared boundary"}
    if name == "color_field_merged_supports":
        return {"merge_evidence": ["soft source-color seam", "single accepted gradient model"]}
    if name == "material_proxy":
        return {"gradient_replacement": "per-support perceptual average", "final_fill_source": "original RGB"}
    if name == "boundary_polyline_trace":
        return {
            "remove_background": options.remove_background,
            "background": trace.background,
            "min_region_size": options.min_region_size,
            "trace_level": options.trace_level,
        }
    if name == "boundary_path_fit":
        return trace_fit
    if name == "boundary_assemble_material":
        # This is a drawing-construction threshold, not a trace option.  Keep
        # the report honest about the value passed to ``assemble_working_roots`` below.
        return {
            "min_region_fraction": min_region_fraction,
            "filter_timing": "after material_union",
        }
    if name == "material_union":
        return {
            "merge_policy": [
                "noncredible shared boundary",
                "source-color continuity fallback",
            ],
            "source_seam_delta_e": 0.06,
        }
    if name == "apply_fills":
        return {"fill_source": "original RGB", "models": ["flat", "linear_gradient", "radial_gradient", "raster"]}
    if name == "root_fit":
        return trace_fit
    if name == "initial_stitch":
        return {"epsilon": options.simplify_tolerance, "max_error": options.curve_tolerance}
    return {
        "epsilon": options.simplify_tolerance,
        "max_error": options.curve_tolerance,
        "corner_normalize": corner_normalize,
    }


def _write_pass_index(
    output: Path,
    snapshots,
    trace,
    *,
    min_region_fraction: float,
    corner_normalize: bool,
) -> None:
    cards = "\n".join(
        f'''<article><h2>{html.escape(label)}</h2>
<a class="artifact" href="{html.escape(filename)}" title="Open {html.escape(filename)}"><object data="{html.escape(filename)}" type="image/svg+xml"></object></a>
<details><summary>parameters and report</summary><pre>{html.escape(json.dumps({"parameters": _stage_parameters(name, trace, min_region_fraction=min_region_fraction, corner_normalize=corner_normalize), "report": _report_with_path_diagnostics(svg, report)}, indent=2, sort_keys=True))}</pre></details>
<p><a href="{html.escape(filename)}">{html.escape(filename)}</a></p></article>'''
        for name, label, filename, svg, report in snapshots
    )
    output.write_text(f'''<!doctype html>
<title>vectorEyes MCP refinement pass review</title>
<style>body{{font-family:system-ui;margin:24px}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}article{{border:1px solid #ddd;padding:12px}}h2{{font-size:16px;margin:0 0 8px}}.artifact{{display:block}}object{{width:100%;aspect-ratio:1;background:#fff;pointer-events:none}}summary{{cursor:pointer;font-size:13px}}pre{{overflow:auto;background:#f6f7f9;padding:8px;font-size:12px}}</style>
<h1>vectorEyes MCP refinement pass review</h1><main>{cards}</main>''')


def _trace_stage_svg(trace, stage: str, *, geometry: bool = False) -> str:
    """Render material or geometry trace paths at one trace-time fit boundary."""
    body: list[str] = []
    regions = trace.geometry_regions if geometry else trace.regions
    for region in regions:
        if stage == "contours":
            d = " ".join(
                "M" + " L".join(f"{float(x):g} {float(y):g}" for x, y in contour) + " Z"
                for contour in region.contours
            )
        else:
            d = region.trace_path.fit_stages.get(stage, region.trace_path.d)
        body.append(
            f'<path id="{region.id}-{stage}" fill="{"#64748B" if geometry else region.color}" '
            f'fill-rule="{region.trace_path.fill_rule}" d="{d}"/>'
        )
    return render_svg_doc(trace.width, trace.height, body)


def _region_polyline_svg(trace, regions, fills=None) -> str:
    """Render decomposition regions before their boundaries are path-fitted."""
    body: list[str] = []
    defs: list[str] = []
    for index, region in enumerate(regions):
        d = " ".join(
            "M" + " L".join(f"{_fmt(float(x))} {_fmt(float(y))}" for x, y in contour) + " Z"
            for contour in region_contours(region.mask)
            if len(contour) >= 3
        )
        if not d:
            continue
        fill = region.color_hex if fills is None else resolve_fill(fills[index], defs)
        body.append(f'<path id="decompose-{index}" fill="{fill}" d="{d}"/>')
    return render_svg_doc(trace.width, trace.height, body, defs)


def _raster_svg(rgb: np.ndarray) -> str:
    """Embed a raster artifact in a portable SVG review frame."""
    png = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(png, format="PNG")
    encoded = base64.b64encode(png.getvalue()).decode()
    height, width = rgb.shape[:2]
    return render_svg_doc(
        width,
        height,
        [f'<image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>'],
    )


def _source_svg(rgb: np.ndarray) -> str:
    """Embed the exact MCP-preprocessed source as the first review artifact."""
    return _raster_svg(rgb)


def main() -> None:
    args = _parse_args()
    source = args.image.expanduser().resolve()
    raw = source.read_bytes()
    image = ImageRef(data_uri="data:image/png;base64," + base64.b64encode(raw).decode())
    max_sampling_colors = "auto" if args.colors == "auto" else int(args.colors)
    options = TraceDrawingOptions(
        refine="auto",
        max_sampling_colors=max_sampling_colors,
        fit_strategy=args.fit_strategy,
        trace_level=args.trace_level,
        corner_normalize=args.corner_normalize,
        gradient_pretrace=args.gradient_pretrace,
        support_weighted_corners=args.support_weighted_corners,
        preprocess={"max_size_px": args.max_size},
    )
    trace, rgb = _trace_result(image, options)
    material_pretrace = trace.material_pretrace
    # Keep the assembled, pre-stitch roots distinct from retained v0.  The
    # stage names deliberately describe drawing representations rather than
    # implementation helpers, so a reviewer can localize a regression without
    # needing to know the refinement internals.
    decomposition_snapshots: list[tuple[str, str, dict[str, object] | None]] = []

    def capture_decomposition_stage(name, regions, fills) -> None:
        decomposition_snapshots.append((name, _region_polyline_svg(trace, regions, fills), None))

    rooted = assemble_working_roots(
        trace,
        rgb,
        min_region_fraction=options.min_region_fraction,
        on_decomposition_stage=capture_decomposition_stage,
    )
    trace, raw_roots = rooted.trace, rooted.regions
    raw_rendered = _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(raw_roots)))
    v0 = stitch_regions(trace, raw_roots)
    v0_rendered = _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(v0)))
    pass_snapshots = [
        ("source", _source_svg(rgb), None),
    ]
    if material_pretrace is not None:
        pass_snapshots.extend([
            ("high_color_quantized", _raster_svg(material_pretrace.high_color_quantized), None),
            ("shape_merged_supports", _raster_svg(material_pretrace.shape_merged), None),
            ("color_field_merged_supports", _raster_svg(material_pretrace.color_field_merged), None),
            ("material_proxy", _raster_svg(material_pretrace.material_proxy), None),
        ])
    if trace.geometry_regions:
        pass_snapshots.extend([
            ("boundary_polyline_trace", _trace_stage_svg(trace, "contours", geometry=True), None),
            ("boundary_path_fit", _trace_stage_svg(trace, "baseline", geometry=True), None),
        ])
    pass_snapshots.extend(decomposition_snapshots)
    pass_snapshots.extend([
        ("root_fit", raw_rendered.svg, dict(raw_rendered.report)),
        ("initial_stitch", v0_rendered.svg, dict(v0_rendered.report)),
    ])

    def capture_auto_pass(name, regions) -> None:
        rendered = _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(regions)))
        pass_snapshots.append((name, rendered.svg, dict(rendered.report)))

    auto = v0 if args.v0_only else auto_refine(
        trace,
        v0,
        rgb=rgb,
        corner_normalize=options.corner_normalize,
        on_pass=capture_auto_pass,
    )
    v0_svg = _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(v0))).svg
    auto_svg = _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(auto))).svg
    labels_svg = _canonical_labels(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(auto)))
    width, height = trace.width, trace.height

    frames = (
        (Image.fromarray(np.asarray(rgb, dtype=np.uint8)), "source (MCP preprocess)"),
        (_render_svg(v0_svg, width, height), "MCP trace v0"),
        (_render_svg(auto_svg, width, height), "MCP v0" if args.v0_only else "MCP auto"),
        (_render_svg(labels_svg, width, height), "v0 labels" if args.v0_only else "auto labels"),
    )
    font = ImageFont.load_default()
    panel = Image.new("RGB", (760, 640), "white")
    for index, (image, label) in enumerate(frames):
        panel.paste(_tile(image, label, font), ((index % 2) * 380, (index // 2) * 320))
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    output.with_suffix(".v0.svg").write_text(v0_svg)
    if not args.v0_only:
        output.with_suffix(".auto.svg").write_text(auto_svg)
    output.with_suffix(".labels.svg").write_text(labels_svg)
    paths_svg = render_path_map(drawing_from_region_samples(
        "path-map", width=trace.width, height=trace.height, regions=auto,
    ))
    output.with_suffix(".paths.svg").write_text(paths_svg)
    if args.debug_passes:
        final = _render_svg(auto_svg, trace.width, trace.height)
        difference = np.abs(np.asarray(rgb, dtype=np.int16) - np.asarray(final, dtype=np.int16))
        pass_snapshots.extend([
            ("region_map", labels_svg, None),
            ("path_map", paths_svg, None),
            (
                "source_vector_diff",
                _raster_svg(np.clip(difference * 4, 0, 255).astype(np.uint8)),
                None,
            ),
        ])
        debug_dir = output.with_name(output.stem + "-passes")
        debug_dir.mkdir(parents=True, exist_ok=True)
        for stale_svg in debug_dir.glob("*.svg"):
            stale_svg.unlink()
        index_entries = []
        for index, (name, svg, report) in enumerate(pass_snapshots):
            filename = f"{index:02d}-{name.removesuffix('_pass')}.svg"
            (debug_dir / filename).write_text(svg)
            index_entries.append((name, f"{index:02d} {name.removesuffix('_pass')}", filename, svg, report))
        _write_pass_index(
            debug_dir / "index.html",
            index_entries,
            trace,
            min_region_fraction=options.min_region_fraction,
            corner_normalize=options.corner_normalize,
        )
        print(debug_dir / "index.html")
    print(f"{output} targets={len(_canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(auto))).report['targets'])}")


if __name__ == "__main__":
    main()

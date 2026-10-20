#!/usr/bin/env python3
"""Render source, trace, and every refinement pass into one review panel.

This development diagnostic deliberately applies the configured refinement passes
one at a time.  It makes regressions such as a bad symmetry reconstruction or
a seam introduced during final serialization visible at the exact pass where
they first appear.

Example:
    uv run python scripts/generate_refinement_pass_panel.py \
        tests/fixtures/daikonic/source.png \
        --output /tmp/daikonic-passes.png
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import resvg_py
from PIL import Image, ImageDraw, ImageFont

from vectormark.emit import render_svg_doc
from vectormark.mcp_image import DEFAULT_MAX_SIZE_PX, preprocess_image
from vectormark.refinement.trace import trace_regions
from vectormark.pipeline import Options, _render_refinement_body
from vectormark.drawing import Canvas
from vectormark.refinement.replay_binding import capture_replay_state, working_regions_from_state
from vectormark.refinement.topology_binding import normalize_working_topology
from vectormark.drawing_materialization import materialize_self_symmetry_state
from vectormark.drawing_refinement import refine_canonical_drawing


def _render_svg(svg: str, width: int, height: int) -> np.ndarray:
    """Render SVG on white, matching the MCP preview's visible appearance."""
    png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height)
    rendered = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    composited = Image.new("RGB", rendered.size, "white")
    composited.paste(rendered, mask=rendered.getchannel("A"))
    return np.asarray(composited, dtype=np.uint8)


def _render_objects(objects, width: int, height: int, options: Options) -> np.ndarray:
    body, defs = _render_refinement_body(
        objects,
        flatten=options.flatten,
        epsilon=options.epsilon,
        max_error=options.max_error,
        cubic=options.cubic_paths,
    )
    return _render_svg(render_svg_doc(width, height, body, defs), width, height)


def _tile(image: np.ndarray, label: str, font: ImageFont.ImageFont) -> Image.Image:
    preview = Image.fromarray(image).convert("RGB")
    preview.thumbnail((300, 200), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (320, 238), "#F6F7F9")
    result.paste(preview, ((320 - preview.width) // 2, 28 + (200 - preview.height) // 2))
    ImageDraw.Draw(result).text((10, 7), label, fill="#111827", font=font)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="PNG, JPEG, or WebP input image")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG panel")
    parser.add_argument("--columns", type=int, default=3, help="Panel column count")
    parser.add_argument("--colors", type=int, default=16, help="Maximum traced palette colors")
    parser.add_argument("--epsilon", type=float, default=1.5, help="Path-fit tolerance in pixels")
    parser.add_argument("--max-error", type=float, default=1.0, help="Bézier fit tolerance in pixels")
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE_PX, help="Preprocess longer-edge cap")
    parser.add_argument("--no-crop", action="store_true", help="Keep the source canvas margin")
    parser.add_argument("--no-symmetry", action="store_true", help="Omit the symmetry refinement pass")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.columns < 1:
        raise SystemExit("--columns must be at least 1")

    source = args.image.expanduser().resolve()
    rgb, meta = preprocess_image(
        source.read_bytes(),
        crop_to_content=not args.no_crop,
        max_size_px=args.max_size,
    )
    options = Options(
        refinement=True,
        epsilon=args.epsilon,
        max_error=args.max_error,
        max_sampling_colors=args.colors,
        no_symmetry=args.no_symmetry,
    )
    objects, masks = trace_regions(rgb, options)
    frames: list[tuple[str, np.ndarray]] = [
        ("source (preprocessed)", rgb),
        ("trace", _render_objects(objects, meta.width, meta.height, options)),
    ]

    state = materialize_self_symmetry_state(capture_replay_state(
        normalize_working_topology(objects), Canvas(meta.width, meta.height)))
    def capture(name, current):
        index = len(frames) - 1
        frames.append((f"{index:02d} {name.removesuffix('_pass')}",
            _render_objects(working_regions_from_state(current), meta.width, meta.height, options)))
    refine_canonical_drawing(state, epsilon=options.epsilon, max_error=options.max_error,
        cubic=options.cubic_paths, no_symmetry=options.no_symmetry,
        corner_normalize=options.corner_normalize, refinement_passes=options.refinement_passes,
        on_pass=capture)

    font = ImageFont.load_default()
    tiles = [_tile(image, label, font) for label, image in frames]
    rows = (len(tiles) + args.columns - 1) // args.columns
    panel = Image.new("RGB", (args.columns * 320, rows * 238), "white")
    for index, tile in enumerate(tiles):
        panel.paste(tile, ((index % args.columns) * 320, (index // args.columns) * 238))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    print(output)


if __name__ == "__main__":
    main()

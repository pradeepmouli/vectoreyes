"""MCP server for exposing vectormark logo idealization tools."""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from .mcp_image import (
    DEFAULT_COLORS,
    DEFAULT_MAX_SIZE_PX,
    ImageError,
    preprocess_image,
    preprocess_image_with_alpha,
    resolve_image,
    svg_output_facts,
)
from .pipeline import Options, _flatten_on_white, idealize
from .drawing_plan import PlanValidationError, parse_plan
from .drawing_actions import DrawingPlan
from .drawing_refine import (
    auto_refine,
    drawing_summary,
    labeled_drawing_svg,
    refine,
    render_drawing as render_drawing_regions,
    root_regions,
    refill_all_regions_from_source,
    stitch_regions,
)
from .drawing_state import DrawingNotFound, DrawingStore
from .drawing_trace import PythonTraceEngine, TraceOptions
from .color import infer_max_colors, white_canvas_mask

# Transport is chosen at startup. stdio = local, full-trust (your own machine, your
# files). Any HTTP transport is potentially network-reachable, so the filesystem tools
# (arbitrary-path READ via idealize_logo, WRITE via output_path) are withheld and only
# the byte-in/SVG-out tools are exposed. See docs/mcp.md.
_TRANSPORT = (os.environ.get("VECTORMARK_MCP_TRANSPORT") or "stdio").strip()
_LOCAL_TRUST = _TRANSPORT == "stdio"

WIDGET_URI = "ui://vectormark/logo-widget.html"
_DRAWING_WIDGET_META = {
    "ui": {"resourceUri": WIDGET_URI},
    "openai/outputTemplate": WIDGET_URI,
}
_DRAWINGS = DrawingStore()
_WIDGET_HTML_PATH = Path(__file__).parents[2] / "integrations" / "mcp-app" / "dist" / "mcp-app.html"
_WIDGET_BUILD_REQUIRED_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>vectormark</title>
</head>
<body>
  <main id="root">
    <h1>Logo idealizer</h1>
    <p>vectormark MCP app has not been built yet. Run <code>npm --prefix integrations/mcp-app run build</code>.</p>
  </main>
</body>
</html>
""".strip()


def _read_widget_html() -> str:
    if _WIDGET_HTML_PATH.is_file():
        return _WIDGET_HTML_PATH.read_text()
    return _WIDGET_BUILD_REQUIRED_HTML


@dataclass(frozen=True)
class IdealizeLogoResult:
    """Structured result returned by the MCP tool and testable helper."""

    image_path: str
    output_path: str | None
    width: int
    height: int
    svg_bytes: int
    svg: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def idealize_logo_file(
    image_path: str,
    *,
    output_path: str | None = None,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    colors: int = 16,
    flatten: bool = False,
    no_symmetry: bool = False,
) -> IdealizeLogoResult:
    """Idealize one raster logo file and optionally write the SVG to disk."""

    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input raster does not exist: {source}")

    options = Options(
        epsilon=epsilon,
        max_error=max_error,
        max_colors=colors,
        flatten=flatten,
        no_symmetry=no_symmetry,
    )
    svg = idealize(str(source), options=options)

    destination: Path | None = None
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(svg)

    with Image.open(source) as image:
        width, height = image.size

    return IdealizeLogoResult(
        image_path=str(source),
        output_path=str(destination) if destination is not None else None,
        width=width,
        height=height,
        svg_bytes=len(svg.encode()),
        svg=svg,
    )


def _decode_image_base64(image_base64: str) -> bytes:
    """Decode a base64 raster, tolerating a `data:image/...;base64,` URI prefix.
    Raises ValueError on invalid base64."""
    data = image_base64.strip()
    if data.startswith("data:"):
        _, _, data = data.partition(",")   # drop the data-URI media-type prefix
    try:
        return base64.b64decode(data, validate=True)   # binascii.Error is a ValueError
    except ValueError as exc:
        raise ValueError("image_base64 is not valid base64") from exc


def idealize_logo_bytes(
    image_bytes: bytes,
    *,
    output_path: str | None = None,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    colors: int = 16,
    flatten: bool = False,
    no_symmetry: bool = False,
) -> IdealizeLogoResult:
    """Idealize an in-memory raster (e.g. an image an agent just generated) into SVG,
    with no local file needed. Alpha is composited on white via the shared pipeline
    helper, matching file input."""

    arr = _flatten_on_white(Image.open(io.BytesIO(image_bytes)))
    height, width = arr.shape[:2]
    options = Options(
        epsilon=epsilon,
        max_error=max_error,
        max_colors=colors,
        flatten=flatten,
        no_symmetry=no_symmetry,
    )
    svg = idealize(arr, options=options)

    destination: Path | None = None
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(svg)

    return IdealizeLogoResult(
        image_path="(inline base64 image)",
        output_path=str(destination) if destination is not None else None,
        width=width,
        height=height,
        svg_bytes=len(svg.encode()),
        svg=svg,
    )


def _render_preview_png(svg: str, width: int, height: int, warnings: list[str]) -> bytes | None:
    """Best-effort: render the SVG to PNG via resvg-py; None (+ a warning) if unavailable."""
    try:
        import resvg_py
        return bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height))
    except Exception as exc:
        warnings.append(f"preview unavailable: {exc}")
        return None


def _render_review_panel_png(
    source_rgb: np.ndarray,
    trace,
    v0_regions,
    current_svg: str,
    labeled_svg: str,
    warnings: list[str],
    *,
    geometry_guide_rgb: np.ndarray | None = None,
) -> bytes | None:
    """Render the source, retained trace, current output, and target map for review."""
    try:
        import resvg_py

        def render_svg(svg: str) -> Image.Image:
            png = resvg_py.svg_to_bytes(svg_string=svg, width=trace.width, height=trace.height)
            rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
            return image

        def tile(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
            preview = image.convert("RGB")
            preview.thumbnail((360, 280), Image.Resampling.LANCZOS)
            result = Image.new("RGB", (380, 320), "#F6F7F9")
            result.paste(preview, ((380 - preview.width) // 2, 30 + (280 - preview.height) // 2))
            ImageDraw.Draw(result).text((10, 8), label, fill="#111827", font=font)
            return result

        source = Image.fromarray(np.asarray(source_rgb, dtype=np.uint8)).convert("RGB")
        current = render_svg(current_svg)
        difference = np.abs(np.asarray(source, dtype=np.int16) - np.asarray(current, dtype=np.int16))
        difference = Image.fromarray(np.clip(difference * 4, 0, 255).astype(np.uint8), "RGB")
        legend = Image.new("RGB", (380, 320), "#111827")
        legend_draw = ImageDraw.Draw(legend)
        legend_draw.multiline_text(
            (16, 16),
            "Difference map\\n\\nblack = matching pixels\\nbrighter = larger source-SVG mismatch\\n\\nDifferences are amplified 4x for review.",
            fill="white", font=ImageFont.load_default(), spacing=6,
        )
        v0_svg = render_drawing_regions(trace, v0_regions).svg
        if geometry_guide_rgb is None:
            frames = (
                (source, "source (MCP preprocess)"),
                (render_svg(v0_svg), "MCP trace v0"),
                (current, "current refined SVG"),
                (render_svg(labeled_svg), "addressable region map"),
                (difference, "absolute source-SVG diff (4x)"),
                (legend, "difference legend"),
            )
        else:
            guide = Image.fromarray(np.asarray(geometry_guide_rgb, dtype=np.uint8)).convert("RGB")
            frames = (
                (source, "original source"),
                (guide, "geometry guide (labels only)"),
                (render_svg(v0_svg), "guide trace"),
                (current, "original-filled final SVG"),
                (render_svg(labeled_svg), "addressable region map"),
                (difference, "absolute source-SVG diff (4x; black = match)"),
            )
        font = ImageFont.load_default()
        panel = Image.new("RGB", (1140, 640), "white")
        for index, (image, label) in enumerate(frames):
            panel.paste(tile(image, label, font), ((index % 3) * 380, (index // 3) * 320))
        output = io.BytesIO()
        panel.save(output, format="PNG")
        return output.getvalue()
    except Exception as exc:
        warnings.append(f"review panel unavailable: {exc}")
        return None


def idealize_logo_image(
    image: dict, options: dict | None, *, local_trust: bool
) -> tuple[dict, bytes | None]:
    """Resolve -> preprocess -> idealize a referenced image. Returns (structured_result,
    preview_png_bytes|None). Raises ImageError with a structured code on input failure."""
    options = options or {}
    pre = options.get("preprocess") or {}
    resolved = resolve_image(image, local_trust=local_trust)

    arr, meta = preprocess_image(
        resolved.bytes,
        crop_to_content=pre.get("crop_to_content", True),
        max_size_px=pre.get("max_size_px", DEFAULT_MAX_SIZE_PX),
        preserve_transparency=pre.get("preserve_transparency", True),
        quantize=pre.get("quantize", False),
    )

    opts = Options(
        epsilon=options.get("epsilon", 1.5),
        max_error=options.get("max_error", 1.0),
        max_colors=options.get("colors", DEFAULT_COLORS),
        min_region_size=options.get("min_region_size", 16),
        min_region_fraction=options.get("min_region_fraction", 0.02),
        flatten=options.get("flatten", False),
        no_symmetry=options.get("no_symmetry", False),
        optimizer=options.get("optimizer", False),
        corner_normalize=options.get("corner_normalize", False),
    )
    try:
        svg = idealize(arr, options=opts)
    except Exception as exc:
        raise ImageError(
            "VECTORMARK_FAILED", f"vectormark failed to process the image: {exc}"
        ) from exc

    with Image.open(io.BytesIO(resolved.bytes)) as src:
        ow, oh = src.size
    warnings: list[str] = []
    preview = _render_preview_png(svg, meta.width, meta.height, warnings)
    _svg_bytes = len(svg.encode())

    diagnostics = {
        "input": {
            "source_kind": resolved.source_kind,
            "mime_type": resolved.mime_type,
            "bytes": len(resolved.bytes),
            "sha256": resolved.sha256,
            "original_width": ow,
            "original_height": oh,
        },
        "processed": {
            "width": meta.width,
            "height": meta.height,
            "cropped": meta.cropped,
            "resized": meta.resized,
            "transparent": meta.transparent,
            "quantized": meta.quantized,
        },
        "vectormark": {
            "colors": opts.max_colors,
            "flatten": opts.flatten,
            "no_symmetry": opts.no_symmetry,
            "epsilon": opts.epsilon,
            "max_error": opts.max_error,
        },
        "output": {"svg_bytes": _svg_bytes, **svg_output_facts(svg)},
        "warnings": warnings,
    }
    result = {
        "svg": svg,
        "width": meta.width,
        "height": meta.height,
        "svg_bytes": _svg_bytes,
        "preview_available": preview is not None,
        "diagnostics": diagnostics,
    }
    return result, preview


mcp = FastMCP(
    "vectormark",
    instructions=(
        "Convert rendered raster artwork into clean, editable SVG. The final SVG must be a designer-quality reconstruction: "
        "at the target size its rendering should be visually indistinguishable from the supplied render, so the raster can be "
        "discarded and the vector drawing can become the source of truth. Treat the source-minus-SVG diff as a fidelity gate: "
        "remove every visible structural, geometric, alignment, proportion, or fill discrepancy. Residuals limited to normal SVG "
        "versus raster antialiasing, resampling, and continuous-gradient approximation are acceptable; merely being a close trace is not. "
        "There are three geometry modalities. Both trace_drawing refine modes are interactive: refine='auto' always returns pre-optimizer v0 plus automatic v1, "
        "whose final fills are re-fitted from the original source RGB; refine='none' returns only v0 and retains "
        "unrefined trace roots when you need to judge the raw decomposition before any automatic cleanup. Use retrace_drawing only when you can supply "
        "a better palette-labelled geometry guide; its guide colours are temporary labels and all final fills are still fitted from the original source. "
        "For trace options, use max_colors='auto' when palette complexity is unknown: it selects the smallest 8/16/32/64 palette whose perceptual improvement "
        "has flattened. Use a fixed integer when a known material count matters. fit_strategy='quadratic' is the stable default; 'progressive' protects "
        "pre-fit straight sides, then uses Q/C recursive fitting for the remaining spans; 'progressive_allow_lines' also permits new straight L segments during recursion. "
        "For either refine mode, inspect the returned five-panel review image before making plan decisions: it shows the preprocessed "
        "source, retained trace v0, current SVG (v1 when auto refinement is enabled), addressable region map, and amplified source-minus-SVG diff. "
        "Use it together with the "
        "labeled_svg and raw_trace artifacts, then make semantic "
        "judgments about every retained region (primitive, path geometry, symmetry, clones, fill, "
        "z-order, simplify, stitch, and corner normalization). Set corner_normalize=true when auto refinement should "
        "canonicalize recognised rounded corners to one endpoint-preserving quadratic; every result reports which "
        "anchors are corner members and separates corner-owned from free command counts. Submit refine_drawing plans with the drawing_id and an "
        "existing base_version; every plan creates a branch version. Preserve a path when no supported "
        "operation improves it. Use trace.regions[*].id as the authoritative, plan-addressable target IDs; "
        "path-local operations use command IDs exposed on each retained target geometry. A path fit operation may refit one retained "
        "command by its version-scoped command ID (for example r1.p1.c14), or fit a span between two retained boundary commands. Use fit.type to force exactly one "
        "line, quadratic, cubic, or keep command; an explicit type=cubic may inflect. Use fit.strategy for a recursive fit that may emit multiple commands: quadratic and cubic are inflection-safe; progressive selects Q/C, and progressive_allow_lines selects L/Q/C. Retain or remove existing spans and then use path-local simplify or normalize_corners to clean them. "
        "There is no arbitrary insert-segment operation and plans must not author exported SVG coordinates. Use snap to make a shared endpoint "
        "exactly coincide with another segment endpoint; use align only for intentional one-axis alignment. When refine='auto', "
        "automatic detection has already run: do not use detect_* as a default follow-up; use it only when the "
        "review panel identifies a specific missed relationship worth reconsidering. To create semantic material regions, "
        "merge a semantic root (a single root ID unions all of its children) and then split the merged target across an explicit divider. "
        "A line divider is infinite; an open path divider must start and end on the target boundary. A split is terminal and its new children "
        "are refined from the resulting child version."
    ),
)


class _DrawingOutputSchema(BaseModel):
    """Stable structured result fields returned by drawing-first tools."""

    model_config = ConfigDict(extra="forbid")


class DrawingArtifactsOutput(_DrawingOutputSchema):
    svg: str = Field(description="Rendered SVG for this drawing version.")
    preview: str = Field(description="Clean PNG preview for visual review.")
    review_panel: str = Field(description="Five-panel PNG: source, retained trace v0, current SVG, addressable region map, and amplified source-minus-SVG diff.")
    labeled_svg: str = Field(description="SVG region map. It labels the public, plan-addressable retained region IDs for this version.")
    raw_trace: str = Field(description="Raw trace provenance artifact.")
    plan: str = Field(description="JSON provenance for the operation that created this version.")
    versions: str = Field(description="JSON manifest of every live version and branch for this drawing.")


class DrawingTargetOutput(_DrawingOutputSchema):
    id: str
    root_id: str = Field(description="Public semantic root ID. Use this single ID in merge.regions to union all of this root's current children.")
    source_regions: list[str]
    geometry: str
    fill: str
    z: float
    diagnostics: dict[str, Any]


class DrawingReportOutput(_DrawingOutputSchema):
    targets: list[DrawingTargetOutput]


class TraceRegionOutput(_DrawingOutputSchema):
    id: str
    root_id: str
    source_regions: list[str]
    geometry: dict[str, Any]
    fill: dict[str, Any]


class TraceSummaryOutput(_DrawingOutputSchema):
    width: int
    height: int
    options: dict[str, Any]
    regions: list[TraceRegionOutput]


class TraceVersionOutput(_DrawingOutputSchema):
    """One immutable drawing version produced by ``trace_drawing``."""

    version: str = Field(description="Version ID for this output: v0 is always the pre-optimizer trace; v1 is the automatic refinement when requested.")
    parent_version: str | None = Field(description="Parent version, or null for the pre-optimizer v0 trace.")
    artifacts: DrawingArtifactsOutput
    report: DrawingReportOutput


class TraceDrawingOutput(_DrawingOutputSchema):
    """Trace envelope with one pre-optimizer output and optional automatic child."""

    drawing_id: str | None = None
    outputs: list[TraceVersionOutput] = Field(
        default_factory=list,
        description="Ordered immutable trace outputs: [v0] for refine='none', or [v0, v1] for refine='auto'.",
    )
    version: str | None = None
    trace: TraceSummaryOutput | None = None
    artifacts: DrawingArtifactsOutput | None = None
    report: DrawingReportOutput | None = None
    svg: str | None = None
    width: int | None = None
    height: int | None = None
    svg_bytes: int | None = None
    preview_available: bool | None = None
    diagnostics: dict[str, Any] | None = None
    trace_options_schema: dict[str, Any] | None = Field(
        None,
        description="Pydantic-generated schema for the trace options. The MCP app uses its enum values to render trace controls.",
    )


class RefinedDrawingOutput(_DrawingOutputSchema):
    drawing_id: str
    version: str
    parent_version: str
    artifacts: DrawingArtifactsOutput
    report: DrawingReportOutput


class DrawingArtifactOutput(_DrawingOutputSchema):
    """Artifact envelope; fields vary predictably by the typed MIME value."""

    mime_type: Literal["image/svg+xml", "application/json", "image/png"]
    svg: str | None = None
    trace: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    versions: list[dict[str, Any]] | None = None


class RenderedSvgOutput(_DrawingOutputSchema):
    image_path: str
    output_path: str | None
    width: int
    height: int
    svg_bytes: int
    svg: str


class ImageRef(BaseModel):
    """A reference to the source image. Provide exactly one source. ChatGPT/host file
    params fill `download_url` (+ `file_id`); callers may instead pass a `path`, https
    `url`, `data_uri`, or bare `base64`."""

    download_url: str | None = Field(
        None, description="Temporary URL to GET the file bytes (ChatGPT/host file param)."
    )
    file_id: str | None = Field(None, description="Host file identifier accompanying download_url.")
    mime_type: str | None = Field(None, description="Optional declared MIME type of the file.")
    file_name: str | None = Field(None, description="Optional original file name.")
    path: str | None = Field(
        None, description="Local filesystem path. Resolved only on the stdio/local-trust server; rejected over HTTP."
    )
    url: str | None = Field(None, description="Explicit https URL to fetch (SSRF-guarded).")
    data_uri: str | None = Field(None, description="A data:image/...;base64,... URI.")
    base64: str | None = Field(None, description="Bare base64-encoded image bytes (PNG/JPEG/WebP).")


class PreprocessOpts(BaseModel):
    """Server-side preprocessing applied before idealization."""

    crop_to_content: bool = Field(True, description="Trim transparent/near-white margins to the content bounding box.")
    max_size_px: int = Field(DEFAULT_MAX_SIZE_PX, description="Downscale (never upscale) so the longer side is at most this many px.")
    preserve_transparency: bool = Field(True, description="Keep alpha through cropping; composite on white at the end.")
    quantize: bool = Field(False, description="Pre-quantize the raster (usually harmful; vectormark extracts its own palette).")


class IdealizeOptions(BaseModel):
    """Idealization parameters. All optional with sensible defaults."""

    colors: int = Field(DEFAULT_COLORS, description="Max palette colors. A CEILING, not a target — flats stay flat; raise it to let gradients keep their bands.")
    flatten: bool = Field(False, description="Emit plain paths instead of native SVG primitives and <use> mirror.")
    no_symmetry: bool = Field(False, description="Disable symmetry detection.")
    optimizer: bool = Field(False, description="Run the existing geometry optimizer passes.")
    corner_normalize: bool = Field(False, description="When optimizer is enabled, reduce recognised rounded corners to one quadratic or a sharp point.")
    epsilon: float = Field(1.5, description="Primitive/polygon recognition tolerance in pixels.")
    max_error: float = Field(1.0, description="Bézier fit tolerance in pixels.")
    min_region_size: int = Field(16, ge=1, description="Absolute pixel-area floor before optimizer tracing.")
    min_region_fraction: float = Field(0.02, ge=0, lt=1, description="Relative retained-region floor before optimizer tracing.")
    preprocess: PreprocessOpts = Field(default_factory=PreprocessOpts, description="Server-side preprocessing options.")


class TraceDrawingOptions(BaseModel):
    refine: Literal["auto", "none"] = Field(
        "none", description="Automatic refinement at trace time: none retains trace roots as v0; auto creates an auto-refined child version from those same roots."
    )
    max_colors: int | Literal["auto"] = Field(
        16,
        description="Maximum quantized palette colors, or auto to select the smallest 8/16/32/64 palette before perceptual improvement flattens.",
    )
    min_region_size: int = Field(16, ge=1, description="Absolute pixel-area floor for raw trace regions.")
    max_hole_area: int = Field(128, ge=0, description="Maximum area of an enclosed, locally color-compatible root-mask hole to fill before path fitting; zero disables this cleanup.")
    min_region_fraction: float = Field(0.02, ge=0, lt=1, description="Relative area floor for retained trace roots; raw trace remains available on demand.")
    trace_level: Literal["pixel", "subpixel"] = Field(
        "pixel", description="Boundary trace precision. subpixel uses anti-alias coverage when available."
    )
    simplify_tolerance: float = Field(1.5, ge=0, description="Raw contour simplification tolerance in pixels.")
    curve_tolerance: float = Field(1.0, ge=0, description="Raw quadratic/cubic path-fit tolerance in pixels.")
    fit_strategy: Literal["quadratic", "progressive", "progressive_allow_lines"] = Field(
        "quadratic",
        description="Raw path fitting: quadratic preserves the standard quadratic fitter; progressive protects pre-fit straight sides then uses Q/C recursion; progressive_allow_lines also permits L during recursion.",
    )
    corner_normalize: bool = Field(
        False,
        description="Auto-refinement toggle: canonicalize recognised rounded corners to one quadratic while keeping their endpoints fixed.",
    )
    remove_background: Literal["auto", "off", "on"] = Field(
        "auto",
        description="Infer a border-connected canvas plate before geometry tracing. Transparent source pixels are composited for this analysis; alpha is never retained as a vector boundary.",
    )
    preprocess: PreprocessOpts = Field(default_factory=PreprocessOpts)


def _trace_options_schema() -> dict[str, Any]:
    """Expose the Pydantic input schema to the MCP app without duplicating enums.

    MCP already uses this model to generate the tool input schema. Returning its
    property definitions with a trace result lets the interactive widget render
    the same enum values when it offers a rerun.
    """
    return TraceDrawingOptions.model_json_schema()["properties"]


def _drawing_artifact_refs(drawing_id: str, version_id: str) -> dict[str, str]:
    """Return the complete, version-addressable artifact set for a live drawing."""
    base = f"drawing://{drawing_id}/{version_id}"
    return {
        "svg": base + ".svg",
        "preview": base + ".png",
        "review_panel": base + ".review.png",
        "labeled_svg": base + ".labels.svg",
        "raw_trace": base + ".trace.json",
        "plan": base + ".plan.json",
        "versions": f"drawing://{drawing_id}/versions.json",
    }


def _jsonable(value: object) -> Any:
    """Convert immutable drawing provenance into JSON-safe public data."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _version_manifest(drawing_id: str, drawing: Any) -> list[dict[str, Any]]:
    return [
        {
            "version": version.id,
            "parent_version": version.parent_id,
            "label": version.label,
            "artifacts": _drawing_artifact_refs(drawing_id, version.id),
        }
        for version in drawing.versions.values()
    ]


class GeometryGuideOptions(BaseModel):
    """Trace controls for a palette-labelled geometry guide."""

    max_colors: int = Field(16, ge=2, le=256, description="Maximum guide-label palette colors to retain.")
    min_region_size: int = Field(16, ge=1, description="Absolute guide-label component area floor.")
    max_hole_area: int = Field(128, ge=0, description="Maximum guide-mask hole cleanup area before path fitting.")
    trace_level: Literal["pixel", "subpixel"] = Field("pixel", description="Guide boundary trace precision.")
    simplify_tolerance: float = Field(1.5, ge=0, description="Guide contour simplification tolerance in pixels.")
    curve_tolerance: float = Field(1.0, ge=0, description="Guide path-fit residual tolerance in pixels.")
    fit_strategy: Literal["quadratic", "progressive", "progressive_allow_lines"] = Field(
        "quadratic", description="Guide path fitting strategy."
    )
    corner_normalize: bool = Field(
        False,
        description="Canonicalize recognised rounded corners to one endpoint-preserving quadratic during guide auto-refinement.",
    )


def _geometry_guide_trace(
    image: ImageRef, options: GeometryGuideOptions, *, expected_width: int, expected_height: int
):
    """Trace a palette-label map in the live drawing's coordinate system."""
    resolved = resolve_image(image.model_dump(exclude_none=True), local_trust=_LOCAL_TRUST)
    guide = Image.open(io.BytesIO(resolved.bytes)).convert("RGBA")
    if guide.size != (expected_width, expected_height):
        raise ValueError(
            f"geometry guide must be exactly {expected_width}x{expected_height}, the live drawing canvas; "
            f"received {guide.width}x{guide.height}"
        )
    rgb = _flatten_on_white(guide)
    trace = PythonTraceEngine().trace(
        rgb,
        TraceOptions(
            max_colors=options.max_colors,
            min_region_size=options.min_region_size,
            max_hole_area=options.max_hole_area,
            trace_level=options.trace_level,
            simplify_tolerance=options.simplify_tolerance,
            curve_tolerance=options.curve_tolerance,
            fit_strategy=options.fit_strategy,
            corner_normalize=options.corner_normalize,
            remove_background="auto",
            source_has_alpha=False,
        ),
        alpha=None,
    )
    return trace, rgb


def _trace_result(image: ImageRef, options: TraceDrawingOptions):
    resolved = resolve_image(image.model_dump(exclude_none=True), local_trust=_LOCAL_TRUST)
    pre = options.preprocess
    # Raw region tracing needs a border/background plate; cropping it away makes a
    # single-colour drawing indistinguishable from the background.
    rgb, meta, alpha = preprocess_image_with_alpha(
        resolved.bytes,
        crop_to_content=False,
        max_size_px=pre.max_size_px,
        preserve_transparency=pre.preserve_transparency,
        quantize=pre.quantize,
    )
    # Source alpha establishes the initial trace geometry only.  RGB remains
    # white-composited for material and fill analysis, and every later vector
    # pass is free to refine the fitted geometry; alpha is never retained as a
    # downstream clipping constraint.
    white_canvas = white_canvas_mask(rgb)
    effective_max_colors = (
        infer_max_colors(rgb, exclude_mask=white_canvas)
        if options.max_colors == "auto"
        else options.max_colors
    )
    trace = PythonTraceEngine().trace(rgb, TraceOptions(max_colors=effective_max_colors,
        requested_max_colors=options.max_colors,
        reserve_white_background=options.max_colors == "auto" and white_canvas is not None,
        min_region_size=options.min_region_size, max_hole_area=options.max_hole_area, trace_level=options.trace_level,
        simplify_tolerance=options.simplify_tolerance, curve_tolerance=options.curve_tolerance,
        fit_strategy=options.fit_strategy, corner_normalize=options.corner_normalize,
        remove_background=options.remove_background,
        source_has_alpha=meta.transparent), alpha=alpha)
    return trace, rgb


@mcp.tool(
    title="Trace drawing",
    description=(
        "Trace a drawing into labeled raw paths and return a five-panel visual decision surface: source, trace v0, "
        "current result, addressable region map, and amplified source-minus-SVG diff. Inspect that panel before choosing any refine_drawing plan. "
        "The final SVG must be a designer-quality replacement for the supplied render: eliminate visible design discrepancies, "
        "not merely achieve a close trace; renderer-level antialiasing and gradient residuals are the only acceptable diff. "
        "Both refine modes retain a live drawing: refine='auto' returns two immutable outputs, pre-optimizer v0 and automatic v1; "
        "refine='none' retains raw trace roots for diagnostic review. "
        "Use max_colors='auto' when palette complexity is unknown. fit_strategy='quadratic' is the stable default; "
        "'progressive' protects pre-fit straight sides then uses Q/C recursion, and 'progressive_allow_lines' also permits L/Q/C fitting during recursion. Set corner_normalize=true to canonicalize recognized rounded corners "
        "to one endpoint-preserving quadratic during automatic refinement; command diagnostics identify corner anchors and count only free commands separately. "
        "Interactive traces retain a live drawing for refine_drawing."
    ),
    meta={**_DRAWING_WIDGET_META, "openai/fileParams": ["image"]},
)
def trace_drawing(
    image: ImageRef, options: TraceDrawingOptions | None = None, ctx: Context | None = None
) -> Annotated[CallToolResult, TraceDrawingOutput]:
    options = options or TraceDrawingOptions()
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] tracing requires an MCP session context")
    try:
        trace, rgb = _trace_result(image, options)
    except ImageError as err:
        raise ToolError(f"[{err.error_code}] {err.message}") from err
    v0_regions = stitch_regions(trace, root_regions(trace, rgb, min_region_fraction=options.min_region_fraction))
    drawing = _DRAWINGS.create(ctx.session, trace, regions=v0_regions, source_rgb=rgb)
    v0_rendered = render_drawing_regions(trace, v0_regions)
    outputs = [{
        "version": "v0",
        "parent_version": None,
        "artifacts": _drawing_artifact_refs(drawing.id, "v0"),
        "report": dict(v0_rendered.report),
    }]
    regions = v0_regions
    version_id = "v0"
    if options.refine == "auto":
        regions = auto_refine(trace, v0_regions, rgb=rgb)
        child = _DRAWINGS.append(
            ctx.session,
            drawing.id,
            "v0",
            plan={
                "version": "vectormark.auto.v1",
                "base_version": "v0",
                "passes": [
                    "primitives", "occlusion", "split", "primitives", "clones", "straighten",
                    "linelet_simplify", "smooth", "symmetry", "stitch",
                ],
            },
            regions=regions,
            label="auto",
            version_id="v1",
        )
        version_id = child.id
        rendered_auto = render_drawing_regions(trace, regions)
        outputs.append({
            "version": child.id,
            "parent_version": "v0",
            "artifacts": _drawing_artifact_refs(drawing.id, child.id),
            "report": dict(rendered_auto.report),
        })
    rendered = render_drawing_regions(trace, regions)
    preview = _render_preview_png(rendered.svg, trace.width, trace.height, [])
    labels_svg = labeled_drawing_svg(trace, regions)
    assert drawing.versions["v0"].regions is not None
    review_panel = _render_review_panel_png(rgb, trace, drawing.versions["v0"].regions, rendered.svg, labels_svg, [])
    result = {"drawing_id": drawing.id, "outputs": outputs, "version": version_id, "trace": drawing_summary(trace, v0_regions),
        "artifacts": _drawing_artifact_refs(drawing.id, version_id), "report": dict(rendered.report)}
    result["trace_options_schema"] = _trace_options_schema()
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=json.dumps(result))]
    if review_panel is not None:
        content.append(ImageContent(type="image", data=base64.b64encode(review_panel).decode(), mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=result, isError=False)


@mcp.tool(
    title="Retrace drawing from geometry guide",
    description=(
        "Create a child version from a palette-labelled geometry guide whose pixel dimensions exactly match the live trace canvas. "
        "The guide must use a transparent background and flat filled colors for intended regions; its colors are temporary labels, not final paint. "
        "VectorMark traces and auto-refines only that geometry, then discards every guide fill and fits all final fills from the original retained raster. "
        "Use it when a semantic plan cannot express the intended geometry directly. Inspect the returned six-panel review before applying further semantic plans."
    ),
    meta={**_DRAWING_WIDGET_META, "openai/fileParams": ["geometry_guide"]},
)
def retrace_drawing(
    drawing_id: str,
    base_version: str,
    geometry_guide: ImageRef,
    options: GeometryGuideOptions | None = None,
    label: str | None = None,
    ctx: Context | None = None,
) -> Annotated[CallToolResult, RefinedDrawingOutput]:
    """Replace a version's geometry from an agent-proposed palette label map."""
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] geometry-guide retracing requires an MCP session context")
    options = options or GeometryGuideOptions()
    try:
        drawing, version = _DRAWINGS.get(ctx.session, drawing_id, base_version)
        assert drawing.source_rgb is not None
        guide_trace, guide_rgb = _geometry_guide_trace(
            geometry_guide, options, expected_width=drawing.trace.width, expected_height=drawing.trace.height
        )
        guide_roots = stitch_regions(
            guide_trace,
            root_regions(guide_trace, guide_rgb, preserve_material_labels=True),
        )
        geometry = auto_refine(guide_trace, guide_roots)
        regions = refill_all_regions_from_source(geometry, drawing.source_rgb)
        child = _DRAWINGS.append(
            ctx.session,
            drawing_id,
            base_version,
            plan={
                "version": "vectormark.geometry-guide.v1",
                "base_version": base_version,
                "guide": {"mode": "palette_labels", "fills": "original_source"},
            },
            regions=regions,
            label=label or "geometry guide",
            trace=guide_trace,
            geometry_guide_rgb=guide_rgb,
        )
    except (DrawingNotFound, ValueError, KeyError) as err:
        raise ToolError(f"[{getattr(err, 'error_code', 'GEOMETRY_GUIDE_INVALID')}] {err}") from err
    rendered = render_drawing_regions(guide_trace, regions)
    labels_svg = labeled_drawing_svg(guide_trace, regions)
    review_panel = _render_review_panel_png(
        drawing.source_rgb, guide_trace, guide_roots, rendered.svg, labels_svg, [], geometry_guide_rgb=guide_rgb
    )
    result = {
        "drawing_id": drawing_id,
        "version": child.id,
        "parent_version": base_version,
        "artifacts": _drawing_artifact_refs(drawing_id, child.id),
        "report": dict(rendered.report),
    }
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=json.dumps(result))]
    if review_panel is not None:
        content.append(ImageContent(type="image", data=base64.b64encode(review_panel).decode(), mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=result, isError=False)


@mcp.tool(
    title="Refine drawing",
    description=(
        "Apply an ordered, strongly typed semantic plan to a live traced drawing and create a child version. "
        "Inspect the trace_drawing review_panel first and use its source/trace/current/region-map/diff comparison to justify plan operations; "
        "labeled_svg identifies the exact retained targets for this base version. "
        "Each path target geometry supplies version-scoped command IDs. Use path op fit with target to refit one retained command, or with "
        "between=[left,right] to replace commands strictly between two retained boundaries. Path-local simplify and normalize_corners live inside "
        "set_geometry.assignment.ops; top-level stitch reconciles shared boundaries between regions. There is no arbitrary insert-segment op and "
        "do not author exported SVG coordinates. "
        "Make the final SVG a designer-quality replacement for the original render. Use the review diff to remove any visible structural, "
        "geometric, alignment, proportion, or fill mismatch; do not stop merely because the result is close. Normal differences caused by "
        "raster versus SVG antialiasing, resampling, or continuous-gradient approximation are acceptable. "
        "omitting an operation preserves the target. For a refine='auto' base version, do not submit detect_* operations "
        "as a routine follow-up: use them only to investigate a concrete missed relationship visible in the review panel. "
        "Use set_clone(source,target,transforms) for clone relationships and set_symmetry(target,axis) for explicit self-symmetry. "
        "For a material split, merge one semantic root to union all of its children, then submit a terminal split with an explicit divider; "
        "inspect and address its newly created children from the next version."
    ),
    meta=_DRAWING_WIDGET_META,
)
def refine_drawing(
    plan: DrawingPlan, ctx: Context | None = None
) -> Annotated[CallToolResult, RefinedDrawingOutput]:
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] refinement requires an MCP session context")
    try:
        parsed = plan if isinstance(plan, DrawingPlan) else parse_plan(plan)
        plan_payload = parsed.model_dump(exclude_none=True)
        drawing, version = _DRAWINGS.get(ctx.session, parsed.drawing_id, parsed.base_version)
        assert version.regions is not None
        trace = version.trace or drawing.trace
        base = version.regions
        assert drawing.source_rgb is not None
        regions = refine(trace, base, parsed, rgb=drawing.source_rgb)
        child = _DRAWINGS.append(
            ctx.session, parsed.drawing_id, parsed.base_version, plan=plan_payload, regions=regions, label=parsed.label,
            trace=trace,
        )
    except (PlanValidationError, DrawingNotFound, ValueError, KeyError) as err:
        raise ToolError(f"[{getattr(err, 'error_code', 'PLAN_INVALID')}] {err}") from err
    rendered = render_drawing_regions(trace, regions)
    preview = _render_preview_png(rendered.svg, trace.width, trace.height, [])
    labels_svg = labeled_drawing_svg(trace, regions)
    assert drawing.versions["v0"].regions is not None
    assert drawing.source_rgb is not None
    review_panel = _render_review_panel_png(
        drawing.source_rgb, trace, base, rendered.svg, labels_svg, [], geometry_guide_rgb=version.geometry_guide_rgb
    )
    result = {"drawing_id": parsed.drawing_id, "version": child.id, "parent_version": parsed.base_version,
        "artifacts": _drawing_artifact_refs(parsed.drawing_id, child.id), "report": dict(rendered.report)}
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=json.dumps(result))]
    if review_panel is not None:
        content.append(ImageContent(type="image", data=base64.b64encode(review_panel).decode(), mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=result, isError=False)


@mcp.tool(title="Get drawing artifact", description="Fetch a rendered file, plan provenance, or version manifest for a live drawing.", meta=_DRAWING_WIDGET_META)
def get_drawing_artifact(
    drawing_id: str, version: str = "v0", artifact: Literal["svg", "preview", "preview_png", "review_panel", "labeled_svg", "raw_trace", "plan", "versions"] = "svg",
    ctx: Context | None = None,
) -> Annotated[CallToolResult, DrawingArtifactOutput]:
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] artifact retrieval requires an MCP session context")
    try:
        drawing, stored = _DRAWINGS.get(ctx.session, drawing_id, version)
    except DrawingNotFound as err:
        raise ToolError("[DRAWING_NOT_FOUND] drawing or version is unavailable") from err
    assert stored.regions is not None
    regions = stored.regions
    trace = stored.trace or drawing.trace
    rendered = render_drawing_regions(trace, regions)
    if artifact == "labeled_svg":
        payload = labeled_drawing_svg(trace, regions)
        return CallToolResult(content=[TextContent(type="text", text=payload)], structuredContent={"mime_type": "image/svg+xml", "svg": payload})
    if artifact == "raw_trace":
        payload = trace.to_public_dict(include_region_map=True)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structuredContent={"mime_type": "application/json", "trace": payload},
        )
    if artifact == "plan":
        payload = {
            "drawing_id": drawing_id,
            "version": stored.id,
            "parent_version": stored.parent_id,
            "label": stored.label,
            "plan": _jsonable(stored.plan),
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structuredContent={"mime_type": "application/json", "plan": payload},
        )
    if artifact == "versions":
        payload = _version_manifest(drawing_id, drawing)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structuredContent={"mime_type": "application/json", "versions": payload},
        )
    if artifact == "svg":
        return CallToolResult(content=[TextContent(type="text", text=rendered.svg)], structuredContent={"mime_type": "image/svg+xml", "svg": rendered.svg})
    if artifact in {"preview", "preview_png"}:
        preview = _render_preview_png(rendered.svg, trace.width, trace.height, [])
        if preview is None:
            raise ToolError("[PREVIEW_UNAVAILABLE] PNG preview renderer is unavailable")
        content = ImageContent(type="image", data=base64.b64encode(preview).decode(), mimeType="image/png")
        return CallToolResult(content=[content], structuredContent={"mime_type": "image/png"})
    if artifact == "review_panel":
        assert drawing.versions["v0"].regions is not None
        assert drawing.source_rgb is not None
        panel = _render_review_panel_png(
            drawing.source_rgb, trace, regions, rendered.svg,
            labeled_drawing_svg(trace, regions), [], geometry_guide_rgb=stored.geometry_guide_rgb,
        )
        if panel is None:
            raise ToolError("[REVIEW_PANEL_UNAVAILABLE] review panel renderer is unavailable")
        content = ImageContent(type="image", data=base64.b64encode(panel).decode(), mimeType="image/png")
        return CallToolResult(content=[content], structuredContent={"mime_type": "image/png"})
    raise ToolError("[ARTIFACT_UNKNOWN] artifact must be svg, preview, preview_png, review_panel, labeled_svg, raw_trace, plan, or versions")


def idealize_logo(image: ImageRef, options: IdealizeOptions | None = None) -> CallToolResult:
    """File-first logo idealization. Returns structured content plus a best-effort image block."""
    image_dict = image.model_dump(exclude_none=True)
    options_dict = options.model_dump() if options is not None else None
    try:
        result, preview = idealize_logo_image(image_dict, options_dict, local_trust=_LOCAL_TRUST)
    except ImageError as err:
        raise ToolError(f"[{err.error_code}] {err.message}") from err
    content: list[TextContent | ImageContent] = [
        TextContent(type="text", text=json.dumps(result))
    ]
    if preview is not None:
        content.append(
            ImageContent(
                type="image",
                data=base64.b64encode(preview).decode(),
                mimeType="image/png",
            )
        )
    return CallToolResult(content=content, structuredContent=result, isError=False)


def idealize_logo_data(
    image_base64: str,
    output_path: str | None = None,
    epsilon: float = 1.5,
    max_error: float = 1.0,
    colors: int = 16,
    flatten: bool = False,
    no_symmetry: bool = False,
) -> dict[str, object]:
    """Convert a base64-encoded raster image into structured SVG."""

    # No host-filesystem writes unless local-trust (stdio): drop output_path otherwise.
    return idealize_logo_bytes(
        _decode_image_base64(image_base64),
        output_path=output_path if _LOCAL_TRUST else None,
        epsilon=epsilon,
        max_error=max_error,
        colors=colors,
        flatten=flatten,
        no_symmetry=no_symmetry,
    ).to_dict()


@mcp.tool(
    title="Render drawing",
    description="Render a drawing SVG in the ChatGPT/MCP Apps widget.",
    meta={
        **_DRAWING_WIDGET_META,
        "openai/toolInvocation/invoking": "Rendering SVG preview...",
        "openai/toolInvocation/invoked": "Rendered SVG preview.",
    },
)
def render_drawing(result: dict | None = None, image_path: str = "", svg: str = "",
                          width: int = 0, height: int = 0) -> RenderedSvgOutput:
    """Render an existing traced/refined drawing SVG in the vectormark app.

    Accepts a drawing-tool result (preferred) or the legacy flat SVG fields.
    """
    if result:
        svg = result.get("svg", svg)
        width = result.get("width", width)
        height = result.get("height", height)
        image_path = (result.get("diagnostics", {}).get("input", {}).get("source_kind")) or image_path
    return RenderedSvgOutput.model_validate(IdealizeLogoResult(
        image_path=image_path, output_path=None, width=width, height=height,
        svg_bytes=len(svg.encode()), svg=svg,   # svg_bytes re-derived, never trusted from caller
    ).to_dict()).model_dump()


@mcp.resource(
    WIDGET_URI,
    name="vectormark-logo-widget",
    title="vectormark drawing review",
    description="Inspect drawing artifacts and rerun a trace with typed trace settings.",
    mime_type="text/html;profile=mcp-app",
    meta={
        "ui": {
            "prefersBorder": True,
            "csp": {"connectDomains": [], "resourceDomains": []},
        },
        "openai/widgetDescription": "Interactive trace settings and drawing artifact review for a vectormark result.",
        "openai/widgetPrefersBorder": True,
        "openai/widgetCSP": {
            "connect_domains": [],
            "resource_domains": [],
        },
    },
)
def logo_widget() -> str:
    """Return the HTML component used by MCP Apps hosts."""

    return _read_widget_html()


def main() -> None:
    """Run the server. Transport from VECTORMARK_MCP_TRANSPORT (stdio | sse |
    streamable-http; default stdio). For HTTP transports, VECTORMARK_MCP_HOST /
    VECTORMARK_MCP_PORT set the bind address (default 127.0.0.1:8000; the MCP endpoint
    is served at /mcp)."""
    if _TRANSPORT != "stdio":
        host = os.environ.get("VECTORMARK_MCP_HOST")
        port = os.environ.get("VECTORMARK_MCP_PORT")
        if host:
            mcp.settings.host = host
        if port:
            mcp.settings.port = int(port)
    mcp.run(_TRANSPORT)


if __name__ == "__main__":
    main()

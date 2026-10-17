"""MCP server for exposing vectormark logo idealization tools."""

from __future__ import annotations

import base64
import io
import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.http import StreamableHTTPASGIApp
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from mcp.server.streamable_http_manager import RequestBodyLimitMiddleware
from mcp.types import CallToolResult, ImageContent, ResourceLink, TextContent
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic.json_schema import SkipJsonSchema

from .mcp_image import (
    DEFAULT_COLORS,
    DEFAULT_MAX_SIZE_PX,
    MAX_INPUT_BYTES,
    ImageError,
    preprocess_image,
    preprocess_image_with_alpha,
    resolve_image,
    svg_output_facts,
)
from .pipeline import Options, _flatten_on_white, idealize
from .refinement_settings import RefinementPasses
from .output_settings import DrawingOutputSettings
from .background import BackgroundColor
from .drawing_plan import PlanValidationError, parse_plan
from .drawing_actions import DrawingPlan
from .drawing_engine import RasterImage
from .python_drawing_engine import PythonDrawingEngine, PythonEngineEvidence
from .drawing_recipe import recipe_json_value
from .drawing_replay import raster_pixels
from .drawing_render import (
    drawing_summary as canonical_drawing_summary,
    render_drawing as render_canonical_drawing,
    render_labeled_drawing,
    render_path_map,
    render_command_map,
)
from .drawing_refine import (
    auto_refine,
    drawing_summary,
    labeled_drawing_svg,
    refine,
    render_drawing as render_drawing_regions,
    refill_all_regions_from_source,
    stitch_regions,
)
from .drawing_state import (
    DrawingManifest,
    DrawingNotFound,
    DrawingStore,
    VersionCandidate,
)
from .drawing_trace import TraceOptions
from .json_value import to_json_value

# Transport is chosen at startup. stdio = local, full-trust (your own machine, your
# files). Any HTTP transport is potentially network-reachable, so the filesystem tools
# (arbitrary-path READ via idealize_logo, WRITE via output_path) are withheld and only
# the byte-in/SVG-out tools are exposed. See docs/mcp.md.
_TRANSPORT = (os.environ.get("VECTORMARK_MCP_TRANSPORT") or "stdio").strip()
_LOCAL_TRUST = _TRANSPORT == "stdio"
# HTTP carries a base64 image plus its JSON envelope. Keep the transport limit
# above the image resolver's decoded-byte limit, not MCP's smaller default.
_MAX_MCP_REQUEST_BYTES = 4 * ((MAX_INPUT_BYTES + 2) // 3) + 1024 * 1024

WIDGET_URI = "ui://vectormark/logo-widget.html"
_DRAWING_WIDGET_META = {
    "ui": {"resourceUri": WIDGET_URI},
    "openai/outputTemplate": WIDGET_URI,
}
_DRAWINGS = DrawingStore()
_ENGINE = PythonDrawingEngine()
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
    packaged = files("vectormark").joinpath("assets", "mcp-app.html")
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    # Editable installs read the Vite output directly; wheels carry the same
    # bytes as package data and never depend on the checkout's directory layout.
    if _WIDGET_HTML_PATH.is_file():
        return _WIDGET_HTML_PATH.read_text(encoding="utf-8")
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
        max_sampling_colors=colors,
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
        max_sampling_colors=colors,
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
    baseline_svg: str,
    current_svg: str,
    labeled_svg: str,
    warnings: list[str],
    *,
    baseline_version: str,
    parent_svg: str | None = None,
    parent_version: str | None = None,
    requested_base_svg: str | None = None,
    requested_base_version: str | None = None,
    geometry_guide_rgb: np.ndarray | None = None,
    path_map_svg: str | None = None,
) -> bytes | None:
    """Render the source, retained trace, current output, and target map for review."""
    try:
        import resvg_py

        def render_svg(svg: str, *, native_size: bool = False) -> Image.Image:
            dimensions = {} if native_size else {"width": trace.width, "height": trace.height}
            png = resvg_py.svg_to_bytes(svg_string=svg, **dimensions)
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
        no_parent = Image.new("RGB", source.size, "#E5E7EB")
        ImageDraw.Draw(no_parent).multiline_text(
            (16, 16),
            "No retained parent\n(source baseline)",
            fill="#374151",
            font=ImageFont.load_default(),
            spacing=6,
        )
        baseline_label = f"selected baseline ({baseline_version})"
        parent_label = (
            f"actual parent ({parent_version})"
            if parent_version is not None
            else "actual parent: none"
        )
        if geometry_guide_rgb is None:
            frames = (
                (source, "source (MCP preprocess)"),
                (render_svg(baseline_svg), baseline_label),
                (
                    render_svg(parent_svg) if parent_svg is not None else no_parent,
                    parent_label,
                ),
                (current, "current refined SVG"),
                (render_svg(labeled_svg, native_size=True), "addressable region map"),
                (difference, "absolute source-SVG diff (4x)"),
            )
        else:
            guide = Image.fromarray(np.asarray(geometry_guide_rgb, dtype=np.uint8)).convert("RGB")
            requested_label = f"requested base ({requested_base_version})"
            if requested_base_version == parent_version:
                requested_label = (
                    f"requested base / actual parent ({requested_base_version})"
                )
            if baseline_version == parent_version:
                baseline_label = (
                    f"selected baseline / actual parent ({baseline_version})"
                )
            lineage_frames = [
                (
                    render_svg(requested_base_svg)
                    if requested_base_svg is not None
                    else no_parent,
                    requested_label,
                ),
                (render_svg(baseline_svg), baseline_label),
            ]
            if (
                parent_version not in {requested_base_version, baseline_version}
                and parent_svg is not None
            ):
                lineage_frames.append((render_svg(parent_svg), parent_label))
            frames = (
                (source, "original source"),
                (guide, "geometry guide (labels only)"),
                *lineage_frames,
                (current, "original-filled final SVG"),
                (difference, "absolute source-SVG diff (4x; black = match)"),
            )
        if path_map_svg is not None:
            frames = (*frames, (render_svg(path_map_svg, native_size=True), "addressable path segments"))
        font = ImageFont.load_default()
        rows = (len(frames) + 2) // 3
        panel = Image.new("RGB", (1140, rows * 320), "white")
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
        max_sampling_colors=options.get("colors", DEFAULT_COLORS),
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
            "colors": opts.max_sampling_colors,
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
    version=version("vectormark"),
    dereference_schemas=False,
    instructions=(
        "Read skill://designer-reconstruction/SKILL.md for the reconstruction workflow and worked refinement plans. "
        "For trace_drawing supply exactly one of image (ChatGPT file with download_url and file_id) or image_data (direct PNG path, URL, data_uri, or base64). "
        "For retrace_drawing use geometry_guide for a host file or geometry_guide_data for direct input; never put direct input in a file-param field. "
        "Convert rendered raster artwork into clean, editable SVG. The final SVG must be a designer-quality reconstruction: "
        "at the target size its rendering should be visually indistinguishable from the supplied render, so the raster can be "
        "discarded and the vector drawing can become the source of truth. Treat the source-minus-SVG diff as a fidelity gate: "
        "remove every visible structural, geometric, alignment, proportion, or fill discrepancy. Residuals limited to normal SVG "
        "versus raster antialiasing, resampling, and continuous-gradient approximation are acceptable; merely being a close trace is not. "
        "There are three geometry modalities. Both trace_drawing refine modes are interactive: refine='auto' always returns pre-optimizer v0 plus automatic v1, "
        "whose final fills are re-fitted from the original source RGB; refine='none' returns only v0 and retains "
        "unrefined trace roots when you need to judge the raw decomposition before any automatic cleanup. Use retrace_drawing only when you can supply "
        "a better palette-labelled geometry guide; its guide colours are temporary labels and all final fills are still fitted from the original source. "
        "For trace options, use max_sampling_colors='auto' when palette complexity is unknown: it selects the smallest 8/16/32/64 palette whose perceptual improvement "
        "has flattened. Use a fixed integer when a known material count matters. fit_strategy='quadratic' is the stable default; 'progressive' protects "
        "pre-fit straight sides, then uses Q/C recursive fitting for the remaining spans; 'progressive_allow_lines' also permits new straight L segments during recursion. "
        "For either refine mode, inspect the returned lineage review image before making plan decisions: it shows the preprocessed "
        "source, selected retained baseline, actual parent (or an explicit no-parent state), current SVG, addressable region map, and amplified source-minus-SVG diff. "
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
mcp.add_provider(SkillsDirectoryProvider(
    roots=Path(__file__).parent / "assets" / "skills",
    supporting_files="resources",
))


class _DrawingOutputSchema(BaseModel):
    """Stable structured result fields returned by drawing-first tools."""

    model_config = ConfigDict(extra="forbid")


class DrawingArtifactsOutput(_DrawingOutputSchema):
    svg: str = Field(description="Rendered SVG for this drawing version.")
    preview: str = Field(description="Clean PNG preview for visual review.")
    review_panel: str = Field(description="Lineage-aware PNG comparing source, selected baseline, actual parent, current SVG, addressable geometry context, and amplified source-minus-SVG diff.")
    labeled_svg: str = Field(description="SVG region map. It labels the public, plan-addressable retained region IDs for this version.")
    path_map: str = Field(description="SVG map of individual L/Q/C/A segments with colors and an exact command-ID side legend.")
    command_map: str = Field(description="SVG map of individual command endpoints for path operations.")
    raw_trace: str = Field(description="Raw trace provenance artifact.")
    plan: str = Field(description="JSON provenance for the operation that created this version.")
    versions: str = Field(description="JSON manifest of every live version and branch for this drawing.")


class PointOutput(RootModel[tuple[float, float]]):
    pass


class CornerBoundaryOutput(_DrawingOutputSchema):
    editable: bool


class MoveCommandOutput(_DrawingOutputSchema):
    id: str
    type: Literal["move"]
    end: PointOutput
    corner_boundary: CornerBoundaryOutput


class LineCommandOutput(MoveCommandOutput):
    type: Literal["line"]


class QuadraticCommandOutput(MoveCommandOutput):
    type: Literal["quadratic"]
    control: PointOutput


class CubicCommandOutput(MoveCommandOutput):
    type: Literal["cubic"]
    control_1: PointOutput
    control_2: PointOutput


class ArcCommandOutput(MoveCommandOutput):
    type: Literal["arc"]
    rx: float
    ry: float
    rotation_degrees: float
    large_arc: bool
    sweep: bool


PathCommandOutput = Annotated[
    MoveCommandOutput
    | LineCommandOutput
    | QuadraticCommandOutput
    | CubicCommandOutput
    | ArcCommandOutput,
    Field(discriminator="type"),
]


class CornerAnchorOutput(_DrawingOutputSchema):
    id: str
    type: Literal["corner"]
    entry_command_id: str
    exit_command_id: str
    owned_command_ids: list[str]
    radius: float | None


class InflectionAnchorOutput(_DrawingOutputSchema):
    id: str
    type: Literal["inflection"]
    command_ids: list[str]


class ExtremumAnchorOutput(InflectionAnchorOutput):
    type: Literal["extremum"]


class JunctionAnchorOutput(InflectionAnchorOutput):
    type: Literal["junction"]
    reference_junction_id: str


PathAnchorOutput = Annotated[
    CornerAnchorOutput
    | InflectionAnchorOutput
    | ExtremumAnchorOutput
    | JunctionAnchorOutput,
    Field(discriminator="type"),
]


class CornerFeatureOutput(_DrawingOutputSchema):
    feature_id: str = Field(description="Short displayed corner ID scoped to this drawing version and path; use it in corner plans, not a source provenance hash.")
    role: str
    command_id: str = Field(description="Fitted command ID in the same displayed path and drawing version.")


class CornerEditAvailabilityOutput(_DrawingOutputSchema):
    supported: bool
    reason: Literal["no_exact_retained_source_output_correspondence"] | None


class PathDetailOutput(_DrawingOutputSchema):
    id: str
    d: str
    commands: list[PathCommandOutput]
    anchors: list[PathAnchorOutput]
    corner_features: list[CornerFeatureOutput]
    corner_edit: CornerEditAvailabilityOutput


class PathGeometryOutput(PathDetailOutput):
    type: Literal["path"]


class CutoutPathGeometryOutput(_DrawingOutputSchema):
    type: Literal["path_with_cutouts"]
    d: str
    fill_rule: Literal["evenodd"]
    paths: list[PathDetailOutput]


class CompoundPathGeometryOutput(_DrawingOutputSchema):
    type: Literal["compound_path"]
    d: str
    fill_rule: Literal["nonzero", "evenodd"]
    topology_pending: Literal[True]
    reason: str
    paths: list[PathDetailOutput]


class SymmetryAxisOutput(_DrawingOutputSchema):
    point: PointOutput
    angle_degrees: float


class SymmetricPathGeometryOutput(_DrawingOutputSchema):
    type: Literal["symmetric_path"]
    half: PathDetailOutput
    axis: SymmetryAxisOutput


class AffineTransformOutput(_DrawingOutputSchema):
    a: float
    b: float
    c: float
    d: float
    tx: float
    ty: float


class CloneGeometryOutput(_DrawingOutputSchema):
    type: Literal["clone"]
    source_region_id: str
    transforms: list[AffineTransformOutput]


class CircleParametersOutput(_DrawingOutputSchema):
    center: PointOutput
    radius: float


class EllipseParametersOutput(_DrawingOutputSchema):
    center: PointOutput
    radius_x: float
    radius_y: float


class RectangleParametersOutput(_DrawingOutputSchema):
    origin: PointOutput
    width: float
    height: float


class RoundedRectangleParametersOutput(RectangleParametersOutput):
    radius_x: float
    radius_y: float


class PolygonParametersOutput(_DrawingOutputSchema):
    vertices: list[PointOutput]


class CircleGeometryOutput(_DrawingOutputSchema):
    type: Literal["circle"]
    parameters: CircleParametersOutput


class EllipseGeometryOutput(_DrawingOutputSchema):
    type: Literal["ellipse"]
    parameters: EllipseParametersOutput


class RectangleGeometryOutput(_DrawingOutputSchema):
    type: Literal["rectangle"]
    parameters: RectangleParametersOutput


class RoundedRectangleGeometryOutput(_DrawingOutputSchema):
    type: Literal["roundedrectangle"]
    parameters: RoundedRectangleParametersOutput


class PolygonGeometryOutput(_DrawingOutputSchema):
    type: Literal["polygon"]
    parameters: PolygonParametersOutput


CanonicalGeometryOutput = Annotated[
    PathGeometryOutput
    | CutoutPathGeometryOutput
    | CompoundPathGeometryOutput
    | SymmetricPathGeometryOutput
    | CloneGeometryOutput
    | CircleGeometryOutput
    | EllipseGeometryOutput
    | RectangleGeometryOutput
    | RoundedRectangleGeometryOutput
    | PolygonGeometryOutput,
    Field(discriminator="type"),
]


class DrawingTargetOutput(_DrawingOutputSchema):
    cutout: bool = Field(False, description="This region is exported as transparent negative space because its fill matches the retained background color.")
    id: str
    root_id: str = Field(description="Public semantic root ID. Use this single ID in merge.regions to union all of this root's current children.")
    source_regions: list[str]
    geometry: CanonicalGeometryOutput = Field(
        description="Canonical structured geometry, including version-scoped path and command addresses when applicable."
    )
    fill: str
    z: float
    diagnostics: dict[str, Any]


class DrawingGroupOutput(_DrawingOutputSchema):
    id: str
    child_ids: list[str]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class DrawingReportOutput(_DrawingOutputSchema):
    output_settings: DrawingOutputSettings = Field(default_factory=DrawingOutputSettings)
    export: dict[str, str | int] | None = None
    background: BackgroundColor | None = None
    targets: list[DrawingTargetOutput]
    groups: list[DrawingGroupOutput] = Field(default_factory=list)


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
    groups: list[DrawingGroupOutput] = Field(default_factory=list)


class TraceVersionOutput(_DrawingOutputSchema):
    """One immutable canonical drawing version returned by a drawing tool."""

    version: str = Field(description="Opaque version ID for this retained canonical Drawing.")
    parent_version: str | None = Field(description="Actual retained parent version, or null for the source baseline.")
    artifacts: DrawingArtifactsOutput
    report: DrawingReportOutput


class TraceDrawingOutput(_DrawingOutputSchema):
    """Trace envelope with one pre-optimizer output and optional automatic child."""

    drawing_id: str
    outputs: list[TraceVersionOutput] = Field(
        description="Ordered immutable trace outputs: [v0] for refine='none', or [v0, v1] for refine='auto'.",
    )
    version: str
    parent_version: str | None = Field(
        description="Actual parent of the selected current version."
    )
    requested_base_version: None = Field(
        description="Source tracing has no caller-selected retained base."
    )
    baseline_version: str = Field(
        description="Retained geometry baseline used by review and replay."
    )
    trace: TraceSummaryOutput
    trace_options: "TraceDrawingOptions" = Field(
        description="Complete validated settings used for this trace request, including defaults and preprocessing. Kept separate from resolved trace diagnostics such as the automatic palette size.",
    )
    artifacts: DrawingArtifactsOutput
    report: DrawingReportOutput
    trace_options_schema: dict[str, Any] = Field(
        description="Pydantic-generated schema for the trace options. The MCP app uses its enum values to render trace controls.",
    )


class RetracedDrawingOutput(_DrawingOutputSchema):
    drawing_id: str
    version: str
    parent_version: str
    requested_base_version: str = Field(
        description="Version explicitly selected by the caller."
    )
    baseline_version: str = Field(
        description="Retained geometry baseline used by review and replay."
    )
    outputs: list[TraceVersionOutput] = Field(
        description="Ordered versions published atomically by this call; guide retrace returns baseline then optimized child.",
    )
    artifacts: DrawingArtifactsOutput
    report: DrawingReportOutput


class RefinedDrawingOutput(_DrawingOutputSchema):
    drawing_id: str
    version: str
    parent_version: str
    requested_base_version: str = Field(
        description="Version explicitly selected by the caller."
    )
    baseline_version: str = Field(
        description="Retained geometry baseline used by review and replay."
    )
    outputs: list[TraceVersionOutput] = Field(
        description="Versions published atomically by this call; semantic refine returns one child."
    )
    artifacts: DrawingArtifactsOutput
    report: DrawingReportOutput


class SourcePolylinePointOutput(_DrawingOutputSchema):
    id: str
    end: tuple[float, float]


class SourcePolylineCornerOutput(_DrawingOutputSchema):
    id: str
    entry_command_id: str
    exit_command_id: str
    owned_command_ids: list[str]


class SourcePolylineOutput(_DrawingOutputSchema):
    region_id: str
    path_id: str
    source_path_id: str
    commands: list[SourcePolylinePointOutput]
    corners: list[SourcePolylineCornerOutput]


class DrawingArtifactOutput(_DrawingOutputSchema):
    """Artifact envelope; fields vary predictably by the typed MIME value."""

    mime_type: Literal["image/svg+xml", "application/json", "image/png"]
    svg: str | None = None
    trace: dict[str, Any] | None = None
    source_paths: list[SourcePolylineOutput] | None = None
    plan: dict[str, Any] | None = None
    versions: list[dict[str, Any]] | None = None


class RenderedSvgOutput(_DrawingOutputSchema):
    image_path: str
    output_path: str | None
    width: int
    height: int
    svg_bytes: int
    svg: str


def _validated_drawing_result(
    schema: type[BaseModel], result: dict[str, object], review_panel: bytes
) -> CallToolResult:
    """Validate the exact public envelope before its version becomes visible."""
    payload = schema.model_validate(result).model_dump(mode="json")
    return CallToolResult(
        content=[
            TextContent(type="text", text=json.dumps(payload)),
            ImageContent(
                type="image",
                data=base64.b64encode(review_panel).decode(),
                mimeType="image/png",
            ),
        ],
        structuredContent=payload,
        isError=False,
    )


class ImageRef(BaseModel):
    """A reference to a PNG source image. Other encoded formats are rejected.
    Provide exactly one source. ChatGPT/host file
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
    data_uri: str | None = Field(None, description="A data:image/png;base64,... URI.")
    base64: str | None = Field(None, description="Bare base64-encoded PNG bytes.")


class PlatformImageRef(BaseModel):
    """A ChatGPT-provided PNG file, never a local path or encoded image."""

    model_config = ConfigDict(extra="forbid")
    download_url: str = Field(min_length=1, description="Host-provided temporary file download URL.")
    file_id: str = Field(min_length=1, description="Host-provided file identifier.")
    mime_type: str = Field(default="image/png", description="Optional declared MIME type.")
    file_name: str = Field(default="", description="Optional original filename.")


class DirectImageRef(BaseModel):
    """Explicit non-file-param PNG ingress; provide exactly one source."""

    model_config = ConfigDict(extra="forbid")
    path: str | None = Field(None, description="Local path, accepted only by trusted stdio servers.")
    url: str | None = Field(None, description="HTTPS image URL, subject to SSRF validation.")
    data_uri: str | None = Field(None, description="data:image/png;base64,...")
    base64: str | None = Field(None, description="Bare base64-encoded PNG bytes.")


def _select_image_ref(
    image: PlatformImageRef | ImageRef | None,
    image_data: DirectImageRef | None,
) -> ImageRef:
    """Normalize either public ingress route into the shared image resolver."""
    if (image is None) == (image_data is None):
        raise ToolError("[INVALID_ARGUMENT] provide exactly one host file or direct image source")
    selected = image if image is not None else image_data
    payload = selected.model_dump(exclude_none=True)
    if image_data is not None and sum(bool(value) for value in payload.values()) != 1:
        raise ToolError("[INVALID_ARGUMENT] direct image input requires exactly one of path, url, data_uri, or base64")
    return ImageRef.model_validate(payload)


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


class TraceDrawingOptions(DrawingOutputSettings):
    corner_rounding: bool = Field(True, description="Protect detected rounded corner spans during v0 fitting. False keeps detected point boundaries instead; it does not disable corner detection or run corner normalization.")
    background_color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$", description="Explicit canvas/background color. Otherwise use boundary detection's inferred color. All matching final regions become transparent cutouts; remove_background=off disables this.")
    background_tolerance: float = Field(32.0, ge=0, le=442, allow_inf_nan=False, description="Shared Euclidean RGB matching tolerance for boundary background detection and final cutouts.")
    refinement_passes: RefinementPasses = Field(
        default_factory=RefinementPasses,
        description="Individual v1 automatic geometry passes. Omitted fields use engine defaults; ignored when refine=none. Order stays fixed. Final source-fill resolution is always performed.",
    )
    initial_stitch: bool = Field(True, description="Repair shared region boundaries in v0 before optional automatic refinement.")
    refine: Literal["auto", "none"] = Field(
        "none", description="Automatic refinement at trace time: none retains trace roots as v0; auto creates an auto-refined child version from those same roots."
    )
    max_sampling_colors: int | Literal["auto"] = Field(
        16,
        description="Maximum sampling palette for tracing/segmentation only; final fills are controlled separately. Auto infers a sampling palette.",
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
    support_weighted_corners: bool = Field(
        False,
        description=(
            "Preserve shallow rounded corners by accumulating adjacent same-direction turns and requiring a credible long exterior support edge. "
            "This augments pre-fit corner evidence without globally relaxing the corner angle threshold or cornering smooth closed curves."
        ),
    )
    gradient_pretrace: bool = Field(
        False,
        description=(
            "Build a high-color material analysis, merge non-credible seams and continuous color fields, then trace a flat-color proxy. "
            "The original raster remains authoritative for exterior geometry and final fills."
        ),
    )
    corner_normalize: bool = Field(
        False,
        description=(
            "v1 auto-refinement toggle: canonicalize recognised rounded corners to one quadratic while keeping their endpoints fixed. "
            "This never changes the pre-optimizer v0 trace."
        ),
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
    schema = TraceDrawingOptions.model_json_schema()
    properties = schema["properties"]
    properties["refinement_passes"] = {
        **properties["refinement_passes"],
        **schema["$defs"]["RefinementPasses"],
        "default": RefinementPasses().model_dump(),
    }
    properties["refinement_passes"].pop("$ref", None)
    return properties


def _drawing_artifact_refs(drawing_id: str, version_id: str) -> dict[str, str]:
    """Return the complete, version-addressable artifact set for a live drawing."""
    base = f"drawing://{drawing_id}/{version_id}"
    return {
        "svg": base + ".svg",
        "preview": base + ".png",
        "review_panel": base + ".review.png",
        "labeled_svg": base + ".labels.svg",
        "path_map": base + ".paths.svg",
        "command_map": base + ".commands.svg",
        "raw_trace": base + ".trace.json",
        "plan": base + ".plan.json",
        "versions": f"drawing://{drawing_id}/versions.json",
    }


def _version_manifest(drawing_id: str, manifest: DrawingManifest) -> list[dict[str, Any]]:
    return [
        {
            "version": version.id,
            "parent_version": version.parent_id,
            "label": version.label,
            "artifacts": _drawing_artifact_refs(drawing_id, version.id),
        }
        for version in manifest.versions
    ]


def _raster_image(rgb: np.ndarray) -> RasterImage:
    height, width = rgb.shape[:2]
    return RasterImage(width, height, "rgb8", np.ascontiguousarray(rgb).tobytes())


class GeometryGuideOptions(BaseModel):
    """Trace controls for a palette-labelled geometry guide."""
    corner_rounding: bool = Field(True, description="Protect detected rounded corner spans during guide fitting; false retains detected point boundaries without disabling detection.")
    refinement_passes: RefinementPasses = Field(default_factory=RefinementPasses, description="Individual automatic geometry passes; omitted fields retain engine defaults.")
    initial_stitch: bool = Field(True, description="Repair shared guide region boundaries before automatic refinement.")

    max_sampling_colors: int = Field(16, ge=2, le=256, description="Maximum guide-label palette colors to retain.")
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
    trace = _ENGINE.trace_polylines(
        rgb,
        TraceOptions(
            max_sampling_colors=options.max_sampling_colors,
            min_region_size=options.min_region_size,
            max_hole_area=options.max_hole_area,
            trace_level=options.trace_level,
            simplify_tolerance=options.simplify_tolerance,
            curve_tolerance=options.curve_tolerance,
            fit_strategy=options.fit_strategy,
            corner_rounding=options.corner_rounding,
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
    trace = _ENGINE.trace_polylines(
        rgb,
        TraceOptions(
            max_sampling_colors=16 if options.max_sampling_colors == "auto" else options.max_sampling_colors,
            requested_max_sampling_colors=options.max_sampling_colors,
            min_region_size=options.min_region_size,
            max_hole_area=options.max_hole_area,
            trace_level=options.trace_level,
            simplify_tolerance=options.simplify_tolerance,
            curve_tolerance=options.curve_tolerance,
            fit_strategy=options.fit_strategy,
            support_weighted_corners=options.support_weighted_corners,
            corner_rounding=options.corner_rounding,
            remove_background=options.remove_background,
            background_color=options.background_color,
            background_tolerance=options.background_tolerance,
            source_has_alpha=meta.transparent,
        ),
        alpha=alpha,
        gradient_pretrace=options.gradient_pretrace,
        geometry_rgb=rgb,
    )
    return trace, rgb


@mcp.tool(
    title="Trace drawing",
    output_schema=TraceDrawingOutput.model_json_schema(),
    description=(
      "Trace a drawing into labeled raw paths and return a lineage decision surface: source, selected baseline, "
      "actual parent (or an explicit no-parent tile), current result, addressable region and segment maps, and amplified source-minus-SVG diff. Inspect that panel before choosing any refine_drawing plan. "
        "The final SVG must be a designer-quality replacement for the supplied render: eliminate visible design discrepancies, "
        "not merely achieve a close trace; renderer-level antialiasing and gradient residuals are the only acceptable diff. "
        "Both refine modes retain a live drawing: refine='auto' returns two immutable outputs, pre-optimizer v0 and automatic v1; "
        "refine='none' retains raw trace roots for diagnostic review. "
        "Use max_sampling_colors='auto' when palette complexity is unknown. fit_strategy='quadratic' is the stable default; "
        "'progressive' protects pre-fit straight sides then uses Q/C recursion, and 'progressive_allow_lines' also permits L/Q/C fitting during recursion. "
        "Set corner_normalize=true to canonicalize recognized rounded corners to one endpoint-preserving quadratic in v1 automatic refinement; v0 is unchanged. "
        "Set support_weighted_corners=true when a shallow rounded corner is distributed across adjacent trace vertices; "
        "the detector aggregates that turn only when it meets a credible long support edge. "
        "Set gradient_pretrace=true for gradient-heavy artwork: it traces a simplified material proxy while retaining the original raster for geometry and fills. "
        "Command diagnostics identify corner anchors and count only free commands separately. "
        "Interactive traces retain a live drawing for refine_drawing."
        " Supply exactly one of image (host file with required download_url and file_id) or image_data (one direct path, url, data_uri, or base64 PNG source)."
    ),
    meta={**_DRAWING_WIDGET_META, "openai/fileParams": ["image"]},
)
def trace_drawing(
    image: PlatformImageRef | SkipJsonSchema[None] = None,
    options: TraceDrawingOptions | None = None,
    ctx: Context | None = None,
    *,
    image_data: DirectImageRef | None = None,
) -> CallToolResult:
    options = options or TraceDrawingOptions()
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] tracing requires an MCP session context")
    drawing_id = _DRAWINGS.new_drawing_id()
    try:
        trace, rgb = _trace_result(_select_image_ref(image, image_data), options)
    except ImageError as err:
        raise ToolError(f"[{err.error_code}] {err.message}") from err
    baseline_result = _ENGINE.fit_trace(drawing_id, trace, _raster_image(rgb),
        min_region_fraction=options.min_region_fraction, initial_stitch=options.initial_stitch,
        output_settings=DrawingOutputSettings(**{name: getattr(options, name) for name in DrawingOutputSettings.model_fields}))
    state = baseline_result.evidence
    evidence, paint, v0_recipe = state.geometry, state.paint, state.recipe
    trace = evidence.trace
    candidates = [
        VersionCandidate(
            drawing=baseline_result.drawing,
            evidence=evidence,
            submitted_plan=None,
            submitted_edits=(),
            effective_recipe=v0_recipe,
            replay=state.replay,
            label="source baseline",
            baseline_candidate_index=0,
        )
    ]
    if options.refine == "auto":
        automatic = _ENGINE.refine_automatically(
            baseline_result.drawing, state,
            corner_normalize=options.corner_normalize,
            refinement_passes=options.refinement_passes,
        )
        candidates.append(
            VersionCandidate(
                drawing=automatic.drawing,
                evidence=None,
                submitted_plan=None,
                submitted_edits=(),
                effective_recipe=automatic.evidence.recipe,
                replay=automatic.evidence.replay,
                label="auto",
                baseline_candidate_index=0,
            )
        )

    def prepare(published):
        selected = published[-1]
        baseline = published[0]
        parent = next(
            (item for item in published if item.id == selected.parent_id), None
        )
        rendered = render_canonical_drawing(
            selected.drawing, selected.replay.path_correspondence
        )
        if _render_preview_png(rendered.svg, trace.width, trace.height, []) is None:
            raise ValueError("preview rendering failed before publication")
        labels_svg = render_labeled_drawing(selected.drawing)
        baseline_svg = render_canonical_drawing(baseline.drawing).svg
        review_panel = _render_review_panel_png(
            rgb,
            trace,
            baseline_svg,
            rendered.svg,
            labels_svg,
            [],
            baseline_version=baseline.id,
            path_map_svg=render_path_map(selected.drawing),
            parent_svg=(
                render_canonical_drawing(parent.drawing).svg
                if parent is not None
                else None
            ),
            parent_version=parent.id if parent is not None else None,
        )
        if review_panel is None:
            raise ValueError("review panel rendering failed before publication")
        outputs = [
            {
                "version": item.id,
                "parent_version": item.parent_id,
                "artifacts": _drawing_artifact_refs(drawing_id, item.id),
                "report": canonical_drawing_summary(
                    item.drawing, item.replay.path_correspondence
                ),
            }
            for item in published
        ]
        result = {
            "drawing_id": drawing_id,
            "outputs": outputs,
            "version": selected.id,
            "parent_version": selected.parent_id,
            "requested_base_version": None,
            "baseline_version": selected.baseline_version_id,
            "trace": to_json_value(baseline_result.diagnostics),
            "trace_options": options.model_dump(mode="json"),
            "artifacts": _drawing_artifact_refs(drawing_id, selected.id),
            "report": dict(rendered.report),
            "trace_options_schema": _trace_options_schema(),
        }
        return _validated_drawing_result(TraceDrawingOutput, result, review_panel)

    _published, response = _DRAWINGS.create_prepared(
        drawing_id, paint, tuple(candidates), prepare=prepare
    )
    return response


@mcp.tool(
    title="Retrace drawing from geometry guide",
    output_schema=RetracedDrawingOutput.model_json_schema(),
    description=(
        "Create a child version from a palette-labelled geometry guide whose pixel dimensions exactly match the live trace canvas. "
        "The guide must use a transparent background and flat filled colors for intended regions; its colors are temporary labels, not final paint. "
        "VectorMark traces and auto-refines only that geometry, then discards every guide fill and fits all final fills from the original retained raster. "
        "Use it when a semantic plan cannot express the intended geometry directly. Inspect the returned lineage review, which distinguishes the requested base from the retained guide baseline and actual parent, before applying further semantic plans."
        " Supply exactly one of geometry_guide (host file with required download_url and file_id) or geometry_guide_data (one direct path, url, data_uri, or base64 PNG source)."
    ),
    meta={**_DRAWING_WIDGET_META, "openai/fileParams": ["geometry_guide"]},
)
def retrace_drawing(
    drawing_id: str,
    base_version: str,
    geometry_guide: PlatformImageRef | SkipJsonSchema[None] = None,
    options: GeometryGuideOptions | None = None,
    label: str | None = None,
    ctx: Context | None = None,
    *,
    geometry_guide_data: DirectImageRef | None = None,
) -> CallToolResult:
    """Replace a version's geometry from an agent-proposed palette label map."""
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] geometry-guide retracing requires an MCP session context")
    options = options or GeometryGuideOptions()
    try:
        base, paint, _base_evidence = _DRAWINGS.load_replay(
            drawing_id, base_version
        )
        guide_trace, guide_rgb = _geometry_guide_trace(
            _select_image_ref(geometry_guide, geometry_guide_data),
            options,
            expected_width=int(paint.canvas.width),
            expected_height=int(paint.canvas.height),
        )
        baseline_execution = _ENGINE.fit_trace(drawing_id, guide_trace, _raster_image(guide_rgb),
            kind="geometry-guide", paint=paint, initial_stitch=options.initial_stitch)
        state = baseline_execution.evidence
        guide_evidence, baseline_recipe = state.geometry, state.recipe
        guide_trace = guide_evidence.trace
        optimized_execution = _ENGINE.refine_automatically(
            baseline_execution.drawing, state,
            corner_normalize=options.corner_normalize,
            refinement_passes=options.refinement_passes,
        )
        optimized_recipe = optimized_execution.evidence.recipe
        candidates = (
            VersionCandidate(
                drawing=baseline_execution.drawing,
                evidence=guide_evidence,
                submitted_plan=None,
                submitted_edits=(),
                effective_recipe=baseline_recipe,
                replay=state.replay,
                label="geometry guide baseline",
                baseline_candidate_index=0,
            ),
            VersionCandidate(
                drawing=optimized_execution.drawing,
                evidence=None,
                submitted_plan=None,
                submitted_edits=(),
                effective_recipe=optimized_recipe,
                replay=optimized_execution.evidence.replay,
                label=label or "geometry guide auto",
                baseline_candidate_index=0,
            ),
        )

        def prepare(published):
            baseline, child = published
            rendered = render_canonical_drawing(
                child.drawing, child.replay.path_correspondence
            )
            if _render_preview_png(
                rendered.svg, guide_trace.width, guide_trace.height, []
            ) is None:
                raise ValueError("preview rendering failed before publication")
            labels_svg = render_labeled_drawing(child.drawing)
            baseline_svg = render_canonical_drawing(baseline.drawing).svg
            requested_base_svg = render_canonical_drawing(base.drawing).svg
            review_panel = _render_review_panel_png(
                raster_pixels(paint.source),
                guide_trace,
                baseline_svg,
                rendered.svg,
                labels_svg,
                [],
                baseline_version=baseline.id,
                parent_svg=baseline_svg,
                parent_version=baseline.id,
                requested_base_svg=requested_base_svg,
                requested_base_version=base.id,
                geometry_guide_rgb=guide_rgb,
                path_map_svg=render_path_map(child.drawing),
            )
            if review_panel is None:
                raise ValueError("review panel rendering failed before publication")
            outputs = [
                {
                    "version": item.id,
                    "parent_version": item.parent_id,
                    "artifacts": _drawing_artifact_refs(drawing_id, item.id),
                    "report": canonical_drawing_summary(
                        item.drawing, item.replay.path_correspondence
                    ),
                }
                for item in published
            ]
            result = {
                "drawing_id": drawing_id,
                "version": child.id,
                "parent_version": child.parent_id,
                "requested_base_version": base.id,
                "baseline_version": child.baseline_version_id,
                "outputs": outputs,
                "artifacts": _drawing_artifact_refs(drawing_id, child.id),
                "report": dict(rendered.report),
            }
            return _validated_drawing_result(
                RetracedDrawingOutput, result, review_panel
            )

        _published, response = _DRAWINGS.append_prepared(
            drawing_id,
            base_version,
            candidates,
            prepare=prepare,
        )
    except (DrawingNotFound, ValueError, KeyError) as err:
        raise ToolError(f"[{getattr(err, 'error_code', 'GEOMETRY_GUIDE_INVALID')}] {err}") from err
    return response


@mcp.tool(
    title="Refine drawing",
    output_schema=RefinedDrawingOutput.model_json_schema(),
    description=(
        "Apply an ordered, strongly typed semantic plan to a live traced drawing and create a child version. "
        "Inspect the lineage-aware review_panel first and use its source/baseline/actual-parent/current/region-map/segment-map/diff comparison to justify plan operations; "
        "labeled_svg identifies the exact retained targets for this base version. "
        "Corner plans use only displayed feature IDs and fitted command IDs (for example r1.p0.corner2 and r1.p0.c4); source hashes are internal and not accepted as plan targets. "
        "Each path target geometry supplies version-scoped command IDs. Use path op fit with target to refit one retained command, or with "
        "between=[A,B] to fit from the end of A to the end of B, replacing commands after A through B inclusive; A is unchanged. Path-local simplify and normalize_corners live inside "
        "update_geometry.ops. Use top-level fit_path to refit a region's complete path and top-level fit_primitive to replace a region with a fitted primitive. "
        "Plan settings inherit from the retained trace; each drawing action may provide a settings override, and nested update_geometry operations inherit it. "
        "Top-level stitch reconciles shared boundaries between regions. There is no arbitrary insert-segment op and "
        "do not author exported SVG coordinates. "
        "Make the final SVG a designer-quality replacement for the original render. Use the review diff to remove any visible structural, "
        "geometric, alignment, proportion, or fill mismatch; do not stop merely because the result is close. Normal differences caused by "
        "raster versus SVG antialiasing, resampling, or continuous-gradient approximation are acceptable. "
        "omitting an operation preserves the target. For a refine='auto' base version, do not submit detect_* operations "
        "as a routine follow-up: use them only to investigate a concrete missed relationship visible in the review panel. "
        "Use assign_clone(source,target,transforms) for clone relationships and set_symmetry(target,axis) for explicit self-symmetry. "
        "For a material split, merge one semantic root to union all of its children, then submit a terminal split with an explicit divider; "
        "inspect and address its newly created children from the next version."
    ),
    meta=_DRAWING_WIDGET_META,
)
def refine_drawing(
    plan: DrawingPlan, ctx: Context | None = None
) -> CallToolResult:
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] refinement requires an MCP session context")
    try:
        parsed = plan if isinstance(plan, DrawingPlan) else parse_plan(plan)
        version, paint, evidence = _DRAWINGS.load_replay(
            parsed.drawing_id, parsed.base_version
        )
        execution = _ENGINE.apply_plan(version.drawing,
            PythonEngineEvidence(evidence, paint, version.effective_recipe, version.replay), parsed)
        recipe = execution.evidence.recipe
        baseline = _DRAWINGS.get_version(
            parsed.drawing_id, version.baseline_version_id
        )
        candidate = VersionCandidate(
            drawing=execution.drawing,
            evidence=None,
            submitted_plan=parsed,
            submitted_edits=execution.submitted_edits,
            effective_recipe=recipe,
            replay=execution.evidence.replay,
            label=parsed.label,
            baseline_version_id=version.baseline_version_id,
        )

        def prepare(published):
            (child,) = published
            rendered = render_canonical_drawing(
                child.drawing, child.replay.path_correspondence
            )
            if _render_preview_png(
                rendered.svg,
                int(child.drawing.canvas.width),
                int(child.drawing.canvas.height),
                [],
            ) is None:
                raise ValueError("preview rendering failed before publication")
            labels_svg = render_labeled_drawing(child.drawing)
            baseline_svg = render_canonical_drawing(baseline.drawing).svg
            parent_svg = render_canonical_drawing(version.drawing).svg
            review_panel = _render_review_panel_png(
                raster_pixels(paint.source),
                evidence.trace,
                baseline_svg,
                rendered.svg,
                labels_svg,
                [],
                baseline_version=baseline.id,
                parent_svg=parent_svg,
                parent_version=version.id,
                requested_base_svg=parent_svg,
                requested_base_version=parsed.base_version,
                path_map_svg=render_path_map(child.drawing),
                geometry_guide_rgb=(
                    raster_pixels(evidence.geometry_source)
                    if evidence.kind == "geometry-guide"
                    else None
                ),
            )
            if review_panel is None:
                raise ValueError("review panel rendering failed before publication")
            result = {
                "drawing_id": parsed.drawing_id,
                "version": child.id,
                "parent_version": child.parent_id,
                "requested_base_version": parsed.base_version,
                "baseline_version": child.baseline_version_id,
                "outputs": [
                    {
                        "version": child.id,
                        "parent_version": child.parent_id,
                        "artifacts": _drawing_artifact_refs(
                            parsed.drawing_id, child.id
                        ),
                        "report": canonical_drawing_summary(
                            child.drawing, child.replay.path_correspondence
                        ),
                    }
                ],
                "artifacts": _drawing_artifact_refs(parsed.drawing_id, child.id),
                "report": dict(rendered.report),
            }
            return _validated_drawing_result(
                RefinedDrawingOutput, result, review_panel
            )

        _published, response = _DRAWINGS.append_prepared(
            parsed.drawing_id,
            parsed.base_version,
            (candidate,),
            prepare=prepare,
        )
    except (PlanValidationError, DrawingNotFound, ValueError, KeyError) as err:
        raise ToolError(f"[{getattr(err, 'error_code', 'PLAN_INVALID')}] {err}") from err
    return response


@mcp.tool(title="Get drawing artifact", output_schema=DrawingArtifactOutput.model_json_schema(), description="Retrieve artifact contents for a live drawing without opening the review UI. For artifact='svg', structuredContent.svg contains the complete SVG text and mime_type is image/svg+xml; save that text verbatim to an .svg file and provide a downloadable file link when requested. A drawing:// reference is an artifact identifier, not an HTTP download URL. JSON artifacts return structured data; PNG artifacts return image content. labeled_svg colors regions with an exact-ID legend. path_map colors individual L/Q/C/A segments and includes separate segment and retained-anchor legends: point corners, rounded corner spans with entry/exit markers, and stored inflection/extremum/junction anchors. Anchor metadata includes exact source anchor IDs and referenced command IDs; no features are redetected. Use IDs, not hue guesses. command_map overlays version-scoped command IDs at their endpoints, including moves. Filter path_map and command_map with region_ids and command_types before selecting commands for refinement plans. Primitives and clones have no local path commands to label.")
def get_drawing_artifact(
    drawing_id: str, version: str = "v0", artifact: Literal["svg", "preview", "preview_png", "review_panel", "labeled_svg", "path_map", "command_map", "raw_trace", "plan", "versions"] = "svg",
    ctx: Context | None = None,
    region_ids: list[str] | None = None,
    command_types: list[Literal["move", "line", "quadratic", "cubic", "arc"]] | None = None,
) -> CallToolResult:
    if ctx is None:
        raise ToolError("[DRAWING_CONTEXT_REQUIRED] artifact retrieval requires an MCP session context")
    try:
        if artifact == "versions":
            manifest = _DRAWINGS.get_manifest(drawing_id)
            payload = _version_manifest(drawing_id, manifest)
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(payload))],
                structuredContent={"mime_type": "application/json", "versions": payload},
            )
        stored = _DRAWINGS.get_version(drawing_id, version)
        if artifact == "plan":
            payload = {
                "drawing_id": drawing_id,
                "version": stored.id,
                "parent_version": stored.parent_id,
                "baseline_version": stored.baseline_version_id,
                "label": stored.label,
                "plan": recipe_json_value(stored.submitted_plan),
                "submitted_plan": recipe_json_value(stored.submitted_plan),
                "submitted_edits": recipe_json_value(stored.submitted_edits),
                "effective_recipe": recipe_json_value(stored.effective_recipe),
                "replay": recipe_json_value(stored.replay),
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(payload))],
                structuredContent={"mime_type": "application/json", "plan": payload},
            )
        if artifact == "raw_trace":
            _selected, _paint, evidence = _DRAWINGS.load_replay(drawing_id, version)
            payload = evidence.trace.to_public_dict(include_region_map=True)
            source_paths = [
                {
                    "region_id": record.region_id,
                    "path_id": record.path_id,
                    "source_path_id": record.fit.source.path.id,
                    "commands": [
                        {"id": command.id, "end": list(command.end)}
                        for command in record.fit.source.path.commands
                    ],
                    "corners": [
                        {
                            "id": anchor.id,
                            "entry_command_id": anchor.entry_command_id,
                            "exit_command_id": anchor.exit_command_id,
                            "owned_command_ids": list(anchor.owned_command_ids),
                        }
                        for anchor in record.fit.source.anchors
                    ],
                }
                for record in evidence.root_paths
            ]
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"trace": payload, "source_paths": source_paths}))],
                structuredContent={"mime_type": "application/json", "trace": payload, "source_paths": source_paths},
            )
    except DrawingNotFound as err:
        raise ToolError("[DRAWING_NOT_FOUND] drawing or version is unavailable") from err
    if artifact == "command_map":
        try:
            payload = render_command_map(
                stored.drawing, stored.replay.path_correspondence,
                region_ids=tuple(region_ids or ()), command_types=tuple(command_types or ()),
            )
        except ValueError as err:
            raise ToolError(f"[INVALID_ARGUMENT] {err}") from err
        preview = _render_preview_png(payload, int(stored.drawing.canvas.width), int(stored.drawing.canvas.height), [])
        content = [TextContent(type="text", text=payload)]
        if preview is not None:
            content.append(ImageContent(type="image", data=base64.b64encode(preview).decode(), mimeType="image/png"))
        return CallToolResult(
            content=content,
            structuredContent={"mime_type": "image/svg+xml", "svg": payload},
        )
    if artifact in {"labeled_svg", "path_map"}:
        try:
            payload = render_path_map(stored.drawing, region_ids=tuple(region_ids or ()), command_types=tuple(command_types or ())) if artifact == "path_map" else render_labeled_drawing(stored.drawing)
        except ValueError as err:
            raise ToolError(f"[INVALID_ARGUMENT] {err}") from err
        return CallToolResult(content=[TextContent(type="text", text=payload)], structuredContent={"mime_type": "image/svg+xml", "svg": payload})
    rendered = render_canonical_drawing(
        stored.drawing, stored.replay.path_correspondence
    )
    if artifact == "svg":
        return CallToolResult(
            content=[
                ResourceLink(
                    type="resource_link",
                    uri=f"drawing://{drawing_id}/{stored.id}.svg",
                    name=f"{drawing_id}-{stored.id}.svg",
                    mimeType="image/svg+xml",
                    size=len(rendered.svg.encode("utf-8")),
                    description="Exact SVG file; read this MCP resource to retrieve its contents. Not an HTTP URL.",
                ),
                TextContent(type="text", text=rendered.svg),
            ],
            structuredContent={"mime_type": "image/svg+xml", "svg": rendered.svg},
        )
    if artifact in {"preview", "preview_png"}:
        preview = _render_preview_png(
            rendered.svg,
            int(stored.drawing.canvas.width),
            int(stored.drawing.canvas.height),
            [],
        )
        if preview is None:
            raise ToolError("[PREVIEW_UNAVAILABLE] PNG preview renderer is unavailable")
        content = ImageContent(type="image", data=base64.b64encode(preview).decode(), mimeType="image/png")
        return CallToolResult(content=[content], structuredContent={"mime_type": "image/png"})
    if artifact == "review_panel":
        _selected, paint, evidence = _DRAWINGS.load_replay(drawing_id, version)
        baseline = _DRAWINGS.get_version(drawing_id, stored.baseline_version_id)
        parent = (
            _DRAWINGS.get_version(drawing_id, stored.parent_id)
            if stored.parent_id is not None
            else None
        )
        if stored.submitted_plan is not None:
            requested_base_version = stored.submitted_plan.base_version
        elif evidence.kind == "geometry-guide":
            requested_base_version = (
                stored.parent_id
                if stored.id == baseline.id
                else baseline.parent_id
            )
        else:
            requested_base_version = None
        requested_base = (
            _DRAWINGS.get_version(drawing_id, requested_base_version)
            if requested_base_version is not None
            else None
        )
        panel = _render_review_panel_png(
            raster_pixels(paint.source),
            evidence.trace,
            render_canonical_drawing(baseline.drawing).svg,
            rendered.svg,
            render_labeled_drawing(stored.drawing),
            [],
            baseline_version=baseline.id,
            parent_svg=(
                render_canonical_drawing(parent.drawing).svg
                if parent is not None
                else None
            ),
            parent_version=parent.id if parent is not None else None,
            path_map_svg=render_path_map(stored.drawing),
            requested_base_svg=(
                render_canonical_drawing(requested_base.drawing).svg
                if requested_base is not None
                else None
            ),
            requested_base_version=(
                requested_base.id if requested_base is not None else None
            ),
            geometry_guide_rgb=(
                raster_pixels(evidence.geometry_source)
                if evidence.kind == "geometry-guide"
                else None
            ),
        )
        if panel is None:
            raise ToolError("[REVIEW_PANEL_UNAVAILABLE] review panel renderer is unavailable")
        content = ImageContent(type="image", data=base64.b64encode(panel).decode(), mimeType="image/png")
        return CallToolResult(content=[content], structuredContent={"mime_type": "image/png"})
    raise ToolError("[ARTIFACT_UNKNOWN] artifact must be svg, preview, preview_png, review_panel, labeled_svg, raw_trace, plan, or versions")


@mcp.resource("drawing://{drawing_id}/{version}.svg", mime_type="image/svg+xml")
def drawing_svg_resource(drawing_id: str, version: str, ctx: Context) -> str:
    """Read the same canonical SVG exposed by artifact retrieval, without UI."""
    result = get_drawing_artifact(drawing_id, version, "svg", ctx)
    return result.structured_content["svg"]


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


def _http_app():
    """Preserve the SDK body limit until FastMCP exposes it in its factory."""
    # Review panels and command maps can exceed the SDK client's 1 MiB SSE
    # event cap. JSON POST responses preserve the complete tool envelope
    # without depending on a client-specific EventSource limit override.
    app = mcp.http_app(transport="streamable-http", json_response=True)
    lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def configured_lifespan(app):
        async with lifespan(app) as state:
            endpoints = [
                route.app for route in app.routes
                if isinstance(getattr(route, "app", None), StreamableHTTPASGIApp)
            ]
            if len(endpoints) != 1:
                raise RuntimeError("FastMCP HTTP endpoint layout changed; review body-limit adapter")
            manager = endpoints[0].session_manager
            if manager is None or not isinstance(manager.asgi_app, RequestBodyLimitMiddleware):
                raise RuntimeError("FastMCP body-limit middleware changed; review adapter")
            # Both layers must agree: the manager passes this value to new
            # transports, while its ASGI middleware rejects oversized requests.
            manager.max_request_body_size = _MAX_MCP_REQUEST_BYTES
            manager.asgi_app.max_body_size = _MAX_MCP_REQUEST_BYTES
            yield state

    app.router.lifespan_context = configured_lifespan
    return app


def main() -> None:
    """Run the server. Transport from VECTORMARK_MCP_TRANSPORT (stdio | sse |
    streamable-http; default stdio). For HTTP transports, VECTORMARK_MCP_HOST /
    VECTORMARK_MCP_PORT set the bind address (default 127.0.0.1:8000; the MCP endpoint
    is served at /mcp)."""
    if _TRANSPORT == "stdio":
        mcp.run("stdio")
    elif _TRANSPORT in {"streamable-http", "http"}:
        import uvicorn

        uvicorn.run(
            _http_app(),
            host=os.environ.get("VECTORMARK_MCP_HOST") or "127.0.0.1",
            port=int(os.environ.get("VECTORMARK_MCP_PORT") or "8000"),
        )
    else:
        mcp.run(
            _TRANSPORT,
            host=os.environ.get("VECTORMARK_MCP_HOST") or "127.0.0.1",
            port=int(os.environ.get("VECTORMARK_MCP_PORT") or "8000"),
        )


if __name__ == "__main__":
    main()

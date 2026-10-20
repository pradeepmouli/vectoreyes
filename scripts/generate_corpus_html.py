"""Generate an inspectable HTML gallery for the verification SVG corpus.

Run:
    PYTHONPATH=src ./.venv/bin/python scripts/generate_corpus_html.py
"""

from __future__ import annotations

from dataclasses import asdict as _trace_asdict
from vectormark.drawing_render import trace_summary as _canonical_trace
from vectormark.drawing_adapter import drawing_from_vector_regions as _export_drawing, normalize_drawing_addresses as _export_addresses
from vectormark.drawing_render import render_drawing as _canonical_render, render_labeled_drawing as _canonical_labels
import argparse
import base64
import copyreg
import dataclasses
import hashlib
import html
import io
import json
import pickle
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from PIL import Image

from vectormark import Options, idealize
from vectormark.drawing_refine import auto_refine, stitch_regions
from vectormark.optimizer.assembly_binding import assemble_working_roots
from vectormark.mcp_image import DEFAULT_MAX_SIZE_PX
from vectormark.mcp_server import ImageRef, TraceDrawingOptions, _trace_result
from vectormark.pipeline import _flatten_on_white

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CORPUS_INPUT = REPO_ROOT / "corpus" / "input"
DEFAULT_CORPUS_OUTPUT = Path("corpus")
DEFAULT_CORPUS_MANIFEST = REPO_ROOT / "corpus" / "sources.json"
DEFAULT_CORPUS_CACHE = REPO_ROOT / "corpus" / "cache"
TRACE_CACHE_VERSION = 3
DRAWING_TRACE_CACHE_VERSION = 36
TRACE_OPTION_FIELDS = (
    "epsilon",
    "max_error",
    "aa_contours",
    "max_sampling_colors",
    "min_region_fraction",
    "working_max_dim",
)
LEGACY_CORPUS_INPUT = REPO_ROOT / "scratch" / "real-logos"
DEFAULT_CORPUS_EPSILON = 1.0
DEFAULT_CORPUS_MAX_ERROR = 1.0


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    mode: str
    image_factory: Callable[[], np.ndarray]
    options: Options
    source_bytes_factory: Callable[[], bytes] | None = None


def default_entries() -> list[CorpusEntry]:
    """Verification corpus entries rendered by the gallery.

    Prefer the local ignored URL manifest, then the local ignored drop-in corpus
    directory, then the legacy untracked real-logo corpus used for acceptance
    inspection. For each logo, render the current default pipeline and the
    optimizer pipeline.
    Fall back to a tiny synthetic set when the local corpus is unavailable.
    """
    entries = source_manifest_entries(DEFAULT_CORPUS_MANIFEST, DEFAULT_CORPUS_CACHE)
    if entries:
        return entries

    for corpus in (DEFAULT_CORPUS_INPUT, LEGACY_CORPUS_INPUT):
        entries = corpus_entries(corpus)
        if entries:
            return entries

    from tests.optimizer.test_integration import _asymmetric_cloud_image, _disk_image

    entries = [
        CorpusEntry("disk", "current", _disk_image, Options()),
        CorpusEntry("disk", "optimizer", _disk_image, Options(optimizer=True)),
        CorpusEntry("asymmetric_cloud", "current", _asymmetric_cloud_image, Options()),
        CorpusEntry(
            "asymmetric_cloud",
            "optimizer",
            _asymmetric_cloud_image,
            Options(optimizer=True),
        )
    ]
    return entries


def _image_factory(path: Path) -> Callable[[], np.ndarray]:
    def _load() -> np.ndarray:
        with Image.open(path) as image:
            return _flatten_on_white(image)

    return _load


def _source_bytes_factory(path: Path) -> Callable[[], bytes]:
    return path.read_bytes


def _cache_filename(name: str, url: str, filename: str | None = None) -> str:
    if filename:
        suffix = Path(filename).suffix
    else:
        suffix = Path(urllib.parse.urlparse(url).path).suffix
    safe_name = name.replace("/", "_")
    return f"{safe_name}{suffix or '.png'}"


def _download_source(url: str, destination: Path, *, refresh: bool = False) -> Path:
    if destination.exists() and not refresh:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "vectormark-corpus/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        tmp.write_bytes(response.read())
    tmp.replace(destination)
    return destination


def _downloaded_image_factory(
    name: str,
    url: str,
    cache_dir: Path,
    *,
    filename: str | None = None,
    refresh: bool = False,
) -> Callable[[], np.ndarray]:
    cache_path = cache_dir / _cache_filename(name, url, filename)

    def _load() -> np.ndarray:
        return _image_factory(_download_source(url, cache_path, refresh=refresh))()

    return _load


def _downloaded_source_bytes_factory(
    name: str,
    url: str,
    cache_dir: Path,
    *,
    filename: str | None = None,
    refresh: bool = False,
) -> Callable[[], bytes]:
    cache_path = cache_dir / _cache_filename(name, url, filename)

    def _load() -> bytes:
        return _download_source(url, cache_path, refresh=refresh).read_bytes()

    return _load


def _source_items(manifest: Any) -> list[dict[str, str]]:
    raw_entries = manifest.get("entries", manifest) if isinstance(manifest, dict) else manifest
    if isinstance(raw_entries, dict):
        return [{"name": str(name), "url": str(url)} for name, url in raw_entries.items()]
    if not isinstance(raw_entries, list):
        raise ValueError("source manifest must be a list, a name-to-url object, or an object with entries")

    items: list[dict[str, str]] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("source manifest entries must be objects")
        if "name" not in raw or "url" not in raw:
            raise ValueError("source manifest entries require name and url")
        item = {"name": str(raw["name"]), "url": str(raw["url"])}
        if "filename" in raw:
            item["filename"] = str(raw["filename"])
        items.append(item)
    return items


def source_manifest_entries(
    manifest_path: Path,
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> list[CorpusEntry]:
    """Build paired entries from a local URL manifest, downloading images on demand."""
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    entries: list[CorpusEntry] = []
    for item in sorted(_source_items(manifest), key=lambda source: source["name"]):
        factory = _downloaded_image_factory(
            item["name"],
            item["url"],
            cache_dir,
            filename=item.get("filename"),
            refresh=refresh,
        )
        source_bytes = _downloaded_source_bytes_factory(
            item["name"], item["url"], cache_dir, filename=item.get("filename"), refresh=refresh,
        )
        entries.append(CorpusEntry(item["name"], "current", factory, Options(), source_bytes))
        entries.append(CorpusEntry(item["name"], "optimizer", factory, Options(optimizer=True), source_bytes))
    return entries


def corpus_entries(corpus: Path) -> list[CorpusEntry]:
    """Build paired current/optimizer entries from PNG files in a corpus directory."""
    skip_prefixes = (
        "cap",
        "cmp_",
        "contact_sheet",
        "corpus",
        "cubic_demo",
        "lever_",
        "node",
        "out_",
        "poc_",
        "pre",
        "rebuild",
        "s40",
        "settir_cmp",
        "settir_spike",
        "threeway",
        "vbird_",
        "zoom",
    )
    if not corpus.exists():
        return []
    pngs = [
        path for path in sorted(corpus.glob("*.png"))
        if not path.name.startswith(".") and not path.stem.startswith(skip_prefixes)
    ]
    if not any(path.stem == "vbird" for path in pngs):
        vbird = corpus / "vbird.png"
        if vbird.exists():
            pngs.append(vbird)
            pngs.sort()

    entries: list[CorpusEntry] = []
    for path in pngs:
        factory = _image_factory(path)
        source_bytes = _source_bytes_factory(path)
        entries.append(CorpusEntry(path.stem, "current", factory, Options(), source_bytes))
        entries.append(CorpusEntry(path.stem, "optimizer", factory, Options(optimizer=True), source_bytes))
    return entries


def _diagnostics_payload(report: Any) -> dict[str, Any]:
    if isinstance(report, dict) and "trace" in report and "report" in report:
        return {
            "workflow": "trace_drawing_auto",
            "drawing_id": report["drawing_id"],
            "version": report["version"],
            "trace": report["trace"],
            "report": report["report"],
        }
    diagnostics = report.diagnostics.to_dict() if report.diagnostics is not None else None
    return {
        "strategies": dict(report.strategies),
        "gradients": report.gradients,
        "elements": report.elements,
        "axes": [dataclasses.asdict(axis) for axis in report.axes],
        "symmetry": report.symmetry,
        "diagnostics": diagnostics,
    }


def _diagnostics_json(report: Any) -> str:
    payload = _diagnostics_payload(report)
    return json.dumps(payload, indent=2, sort_keys=True)


def _options_json(options: Any) -> str:
    if dataclasses.is_dataclass(options):
        payload = dataclasses.asdict(options)
    elif isinstance(options, dict):
        payload = options
    else:
        payload = options.model_dump(mode="json")
    return json.dumps(payload, indent=2, sort_keys=True, default=repr)


def _trace_options(options: Options) -> dict[str, object]:
    return {field: getattr(options, field) for field in TRACE_OPTION_FIELDS}


def _trace_cache_key(entry: CorpusEntry, image: np.ndarray, options: Options) -> str:
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    payload = {
        "version": TRACE_CACHE_VERSION,
        "entry": entry.name,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "trace_options": _trace_options(options),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, default=repr).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()[:20]


def _trace_cache_path(cache_dir: Path, entry: CorpusEntry, image: np.ndarray, options: Options) -> Path:
    safe_name = entry.name.replace("/", "_")
    return cache_dir / f"trace-{safe_name}-{_trace_cache_key(entry, image, options)}.pkl"


def _load_cached_trace(path: Path, *, trace_options: dict[str, object]):
    if not path.exists():
        return None
    payload = pickle.loads(path.read_bytes())
    if (
        payload.get("version") == TRACE_CACHE_VERSION
        and payload.get("trace_options") == trace_options
    ):
        return payload
    return None


def _load_or_create_trace(cache_path: Path, image: np.ndarray, options: Options):
    from vectormark.optimizer.trace import trace_regions

    trace_options = _trace_options(options)
    payload = _load_cached_trace(cache_path, trace_options=trace_options)
    if payload is not None:
        return payload["objects"], payload["masks"]

    objects, masks = trace_regions(image, options)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_bytes(
        pickle.dumps(
            {
                "version": TRACE_CACHE_VERSION,
                "trace_options": trace_options,
                "objects": objects,
                "masks": masks,
            }
        )
    )
    tmp.replace(cache_path)
    return objects, masks


def _idealize_optimizer_with_trace_cache(
    image: np.ndarray,
    *,
    options: Options,
    entry: CorpusEntry,
    cache_dir: Path,
):
    from vectormark.emit import render_svg_doc
    from vectormark.optimizer.framework import optimize
    from vectormark.pipeline import (
        _condition_input,
        _optimizer_passes,
        _optimizer_report,
        _prefer_optimizer_svg,
        _render_optimizer_body,
        _set_svg_output_size,
    )

    orig_h, orig_w = image.shape[:2]
    working = _condition_input(np.asarray(image, dtype=np.uint8), options.working_max_dim)
    h0, w0 = working.shape[:2]
    cache_path = _trace_cache_path(cache_dir, entry, working, options)
    objects, masks = _load_or_create_trace(
        cache_path,
        working,
        options,
    )
    optimized = optimize(objects, masks, _optimizer_passes(options))
    trace_body, trace_defs = _render_optimizer_body(
        objects,
        flatten=options.flatten,
        epsilon=options.epsilon,
        max_error=options.max_error,
        cubic=options.cubic_paths,
    )
    trace_svg = render_svg_doc(w0, h0, trace_body, trace_defs)
    optimized_body, optimized_defs = _render_optimizer_body(
        optimized,
        flatten=options.flatten,
        epsilon=options.epsilon,
        max_error=options.max_error,
        cubic=options.cubic_paths,
    )
    optimized_svg = render_svg_doc(w0, h0, optimized_body, optimized_defs)
    fallback_reason = None
    if not _prefer_optimizer_svg(trace_svg, optimized_svg):
        fallback_reason = "optimized output has more path segments than trace baseline"
    svg = optimized_svg
    report = _optimizer_report(optimized, options, fallback_reason=fallback_reason)
    if (working.shape[1], working.shape[0]) != (orig_w, orig_h):
        svg = _set_svg_output_size(svg, orig_w, orig_h)
    return svg, report


def _entry_source_bytes(entry: CorpusEntry, image: np.ndarray) -> bytes:
    """Use original bytes whenever possible so MCP can retain alpha provenance."""
    if entry.source_bytes_factory is not None:
        return entry.source_bytes_factory()
    encoded = io.BytesIO()
    Image.fromarray(image).save(encoded, format="PNG")
    return encoded.getvalue()


def _drawing_trace_options(
    options: Options,
    *,
    max_sampling_colors: int | Literal["auto"],
    fit_strategy: Literal["quadratic", "progressive", "progressive_allow_lines"],
    trace_level: Literal["pixel", "subpixel"] | None = None,
    gradient_pretrace: bool = False,
    support_weighted_corners: bool = False,
) -> TraceDrawingOptions:
    return TraceDrawingOptions(
        refine="auto",
        max_sampling_colors=max_sampling_colors,
        min_region_size=options.min_region_size,
        min_region_fraction=options.min_region_fraction,
        trace_level=trace_level or ("subpixel" if options.aa_contours else "pixel"),
        simplify_tolerance=options.epsilon,
        curve_tolerance=options.max_error,
        fit_strategy=fit_strategy,
        gradient_pretrace=gradient_pretrace,
        support_weighted_corners=support_weighted_corners,
        corner_normalize=options.corner_normalize,
        preprocess={"max_size_px": options.working_max_dim or DEFAULT_MAX_SIZE_PX},
    )


def _drawing_trace_cache_path(
    cache_dir: Path,
    entry: CorpusEntry,
    source: bytes,
    options: TraceDrawingOptions,
) -> Path:
    digest = hashlib.sha256()
    digest.update(source)
    trace_inputs = options.model_dump(mode="json")
    # These controls create descendants from retained v0; neither changes the
    # trace artifact or its stitched roots.
    trace_inputs.pop("refine", None)
    trace_inputs.pop("corner_normalize", None)
    digest.update(json.dumps(trace_inputs, sort_keys=True).encode("utf-8"))
    safe_name = entry.name.replace("/", "_")
    return cache_dir / f"drawing-trace-{safe_name}-{digest.hexdigest()[:20]}.pkl"


def _trace_drawing_auto_with_cache(
    source: bytes,
    *,
    entry: CorpusEntry,
    options: Options,
    cache_dir: Path,
    max_sampling_colors: int | Literal["auto"],
    fit_strategy: Literal["quadratic", "progressive", "progressive_allow_lines"],
    trace_level: Literal["pixel", "subpixel"] | None = None,
    gradient_pretrace: bool = False,
    support_weighted_corners: bool = False,
    debug_dir: Path | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    trace_options = _drawing_trace_options(
        options,
        max_sampling_colors=max_sampling_colors,
        fit_strategy=fit_strategy,
        trace_level=trace_level,
        gradient_pretrace=gradient_pretrace,
        support_weighted_corners=support_weighted_corners,
    )
    cache_path = _drawing_trace_cache_path(cache_dir, entry, source, trace_options)
    if cache_path.exists():
        cached = pickle.loads(cache_path.read_bytes())
        if cached.get("version") == DRAWING_TRACE_CACHE_VERSION:
            trace = cached["trace"]
            rgb = cached["rgb"]
            roots = cached["roots"]
        else:
            trace = rgb = roots = None
    else:
        trace = rgb = roots = None

    if trace is None or rgb is None or roots is None:
        trace, rgb = _trace_result(
            ImageRef(data_uri="data:image/png;base64," + base64.b64encode(source).decode("ascii")),
            trace_options,
        )
        rooted = assemble_working_roots(trace, rgb, min_region_fraction=trace_options.min_region_fraction)
        trace = rooted.trace
        roots = stitch_regions(trace, rooted.regions)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        cache_buffer = io.BytesIO()
        cache_pickler = pickle.Pickler(cache_buffer)
        cache_pickler.dispatch_table = {
            **copyreg.dispatch_table,
            MappingProxyType: lambda value: (dict, (dict(value),)),
        }
        cache_pickler.dump({
            "version": DRAWING_TRACE_CACHE_VERSION,
            "trace": trace,
            "rgb": rgb,
            "roots": roots,
        })
        tmp.write_bytes(cache_buffer.getvalue())
        tmp.replace(cache_path)

    # Cache only tracing and retained v0 geometry.  Auto refinement, final fill
    # fitting, SVG emission, and diagnostics are intentionally rerun so an
    # optimizer or renderer change appears in the next corpus invocation.
    pass_snapshots: list[tuple[str, str, dict[str, Any]]] = []
    if debug_dir is not None:
        pass_snapshots.append(("trace_v0", _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(roots))).svg, dict(_canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(roots))).report)))

    def capture_auto_pass(name: str, regions) -> None:
        if debug_dir is not None:
            rendered_pass = _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(regions)))
            pass_snapshots.append((name, rendered_pass.svg, dict(rendered_pass.report)))

    regions = auto_refine(
        trace,
        roots,
        rgb=rgb,
        corner_normalize=trace_options.corner_normalize,
        on_pass=capture_auto_pass if debug_dir is not None else None,
    )
    rendered = _canonical_render(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(regions)))
    svg = rendered.svg
    if debug_dir is not None:
        pass_snapshots.append(("final_refill", svg, dict(rendered.report)))
    report = {
        "trace": _canonical_trace(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(regions)), _trace_asdict((trace).options)),
        "report": rendered.report,
        "drawing_id": f"corpus-{entry.name}",
        "version": "v0.auto",
    }
    public_options = {
        **trace_options.model_dump(mode="json"),
        "effective_max_sampling_colors": trace.options.max_sampling_colors,
    }
    if debug_dir is not None:
        _write_trace_auto_debug_report(
            debug_dir,
            source_rgb=rgb,
            trace=trace,
            final_svg=svg,
            final_regions=regions,
            pass_snapshots=pass_snapshots,
            trace_report=report,
            public_options=public_options,
        )
    return svg, report, public_options


def _render_svg_rgb(svg: str, *, width: int, height: int) -> Image.Image:
    """Render an SVG on white, matching the MCP review-panel comparison."""
    import resvg_py

    png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height)
    rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    result = Image.new("RGB", rgba.size, "white")
    result.paste(rgba, mask=rgba.getchannel("A"))
    return result


def _write_trace_auto_debug_report(
    debug_dir: Path,
    *,
    source_rgb: np.ndarray,
    trace: Any,
    final_svg: str,
    final_regions: Any,
    pass_snapshots: list[tuple[str, str, dict[str, Any]]],
    trace_report: dict[str, Any],
    public_options: dict[str, Any],
) -> None:
    """Write a self-contained trace-auto pass report without retracing.

    The first artifacts retain v0 and every optimizer pass.  The final two
    artifacts mirror the MCP review panel's agent-facing evidence: an
    addressable region map and a source-to-final SVG difference map.
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    for stale in debug_dir.iterdir():
        if stale.is_file():
            stale.unlink()

    entries: list[tuple[str, str, str, dict[str, Any] | None]] = []
    if trace.material_pretrace is not None:
        for name, raster in (
            ("pretrace-high-color-quantized", trace.material_pretrace.high_color_quantized),
            ("pretrace-shape-merged", trace.material_pretrace.shape_merged),
            ("pretrace-color-field-merged", trace.material_pretrace.color_field_merged),
            ("pretrace-material-proxy", trace.material_pretrace.material_proxy),
        ):
            filename = f"{name}.png"
            Image.fromarray(np.asarray(raster, dtype=np.uint8), "RGB").save(debug_dir / filename)
            entries.append((name.replace("-", " "), filename, "png", None))
    for index, (name, svg, pass_report) in enumerate(pass_snapshots):
        filename = f"{index:02d}-{name.removesuffix('_pass')}.svg"
        (debug_dir / filename).write_text(svg)
        entries.append((f"{index:02d} {name.removesuffix('_pass')}", filename, "svg", pass_report))

    region_map_name = "region-map.svg"
    (debug_dir / region_map_name).write_text(_canonical_labels(_export_drawing("export", width=(trace).width, height=(trace).height, regions=_export_addresses(final_regions))))
    entries.append(("addressable region map", region_map_name, "svg", None))

    source = Image.fromarray(np.asarray(source_rgb, dtype=np.uint8)).convert("RGB")
    final = _render_svg_rgb(final_svg, width=trace.width, height=trace.height)
    difference = np.abs(np.asarray(source, dtype=np.int16) - np.asarray(final, dtype=np.int16))
    diff_name = "source-vector-diff.png"
    Image.fromarray(np.clip(difference * 4, 0, 255).astype(np.uint8), "RGB").save(debug_dir / diff_name)
    entries.append(("source-to-vector diff (4x; black = match)", diff_name, "png", None))

    report_name = "trace-report.json"
    (debug_dir / report_name).write_text(json.dumps({"options": public_options, "result": trace_report}, indent=2, sort_keys=True))
    cards = "\n".join(
        (
            f'''<article><h2>{html.escape(label)}</h2>
<a class="artifact" href="{html.escape(filename)}" title="Open {html.escape(filename)}">
<object data="{html.escape(filename)}" type="image/svg+xml"></object></a>
<p><a href="{html.escape(filename)}">{html.escape(filename)}</a></p></article>'''
            if kind == "svg" else
            f'''<article><h2>{html.escape(label)}</h2>
<a class="artifact" href="{html.escape(filename)}"><img src="{html.escape(filename)}" alt="{html.escape(label)}"></a></article>'''
        )
        for label, filename, kind, _pass_report in entries
    )
    details = "\n".join(
        f'''<details><summary>{html.escape(label)} diagnostics</summary><pre>{html.escape(json.dumps(pass_report, indent=2, sort_keys=True))}</pre></details>'''
        for label, _filename, _kind, pass_report in entries
        if pass_report is not None
    )
    (debug_dir / "index.html").write_text(f'''<!doctype html>
<meta charset="utf-8"><title>VectorMark trace-auto debug</title>
<style>body{{font:14px/1.45 system-ui;margin:24px}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}article{{border:1px solid #ddd;padding:12px}}h2{{font-size:16px;margin:0 0 8px}}.artifact{{display:block}}object,img{{display:block;width:100%;aspect-ratio:1;background:#fff;border:0}}.artifact object{{pointer-events:none}}details{{margin-top:12px}}pre{{overflow:auto;background:#f6f7f9;padding:8px;font-size:12px}}</style>
<h1>VectorMark trace-auto debug</h1><p><a href="{html.escape(report_name)}">trace report and effective options</a></p><main>{cards}</main>{details}''')


def _safe_filename(entry: CorpusEntry) -> str:
    return f"{entry.mode}-{entry.name}.svg".replace("/", "_")


def _input_filename(entry: CorpusEntry) -> str:
    return f"{entry.mode}-{entry.name}.png".replace("/", "_")


def _entry_id(entry: CorpusEntry) -> str:
    return f"{entry.mode}-{entry.name}".replace("/", "_")


def _diagnostics_filename(entry: CorpusEntry) -> str:
    return f"{_entry_id(entry)}.json"


def _diagnostics_summary_filename(entry: CorpusEntry) -> str:
    return f"{_entry_id(entry)}.html"


def _options_filename(entry: CorpusEntry) -> str:
    return f"{_entry_id(entry)}.json"


def _raw_svg_filename(entry: CorpusEntry) -> str:
    return f"{_entry_id(entry)}.svg.txt"


def _gallery_options(
    options: Options,
    working_max_dim: int | None,
    *,
    cubic_paths: bool = False,
    epsilon: float = DEFAULT_CORPUS_EPSILON,
    max_error: float = DEFAULT_CORPUS_MAX_ERROR,
) -> Options:
    updates: dict[str, object] = {}
    if options.epsilon != epsilon:
        updates["epsilon"] = epsilon
    if options.max_error != max_error:
        updates["max_error"] = max_error
    if working_max_dim is not None and options.working_max_dim is None:
        updates["working_max_dim"] = working_max_dim
    if cubic_paths:
        updates["cubic_paths"] = True
    if not updates:
        return options
    return dataclasses.replace(options, **updates)


def _sorted_entries(entries: Iterable[CorpusEntry]) -> list[CorpusEntry]:
    mode_order = {"current": 0, "optimizer": 1, "trace_auto": 2}
    return sorted(entries, key=lambda item: (item.name, mode_order.get(item.mode, 99)))


def _entries_for_workflow(entries: Iterable[CorpusEntry], workflow: str) -> list[CorpusEntry]:
    if workflow == "oneshot":
        return _sorted_entries(entries)
    if workflow != "trace_auto":
        raise ValueError(f"unsupported corpus workflow: {workflow!r}")
    chosen: dict[str, CorpusEntry] = {}
    for entry in _sorted_entries(entries):
        # The old corpus has a current/optimizer pair per source.  Drawing-auto
        # has one stateful trace per source, so retain the current entry's trace
        # settings and its original-byte factory exactly once.
        if entry.name not in chosen or entry.mode == "current":
            chosen[entry.name] = dataclasses.replace(entry, mode="trace_auto", options=Options())
    return _sorted_entries(chosen.values())


def _manifest_payload(entries: list[CorpusEntry], *, debug_reports: bool = False) -> dict[str, Any]:
    return {
        "version": 4,
        "debug_reports": debug_reports,
        "entries": [
            {
                "id": _entry_id(entry),
                "name": entry.name,
                "mode": entry.mode,
            }
            for entry in entries
        ],
    }


def _fmt_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_scalar(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _summary_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "<p class=\"empty\">none</p>"
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(_fmt_scalar(value))}</td>" for value in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _diagnostics_summary_html(report: Any) -> str:
    payload = _diagnostics_payload(report)
    if payload.get("workflow") == "trace_drawing_auto":
        trace = payload["trace"]
        drawing_report = payload["report"]
        target_rows = [
            [
                target.get("id"),
                target.get("geometry"),
                target.get("fill"),
                target.get("z"),
                ", ".join(target.get("source_regions") or ()),
                json.dumps(target.get("diagnostics") or {}, sort_keys=True),
            ]
            for target in drawing_report.get("targets", [])
        ]
        geometry_rows = [
            [
                region.get("id"),
                region.get("geometry", {}).get("type"),
                region.get("fill", {}).get("type"),
                ", ".join(region.get("source_regions") or ()),
            ]
            for region in trace.get("regions", [])
        ]
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>
body {{ margin: 0; color: #161616; font: 12px/1.45 system-ui, sans-serif; }}
h3 {{ margin: 12px 0 6px; font-size: 13px; }} h3:first-child {{ margin-top: 0; }}
table {{ width: 100%; border-collapse: collapse; }} th,td {{ padding: 4px 6px; border: 1px solid #e2e2dc; text-align: left; vertical-align: top; overflow-wrap: anywhere; }} th {{ background: #f2f2ee; }}
</style></head><body>
  <h3>Trace drawing / {html.escape(str(payload["version"]))}</h3>
  {_summary_table(["id", "geometry", "fill", "source regions"], geometry_rows)}
  <h3>Auto-refined targets</h3>
  {_summary_table(["id", "geometry", "fill", "z", "source regions", "diagnostics"], target_rows)}
</body></html>"""
    diagnostics = payload.get("diagnostics") or {}
    stats = diagnostics.get("stats") or {}
    optimizer_regions = diagnostics.get("optimizer_regions") or []
    optimizer_objects = diagnostics.get("optimizer_objects") or []
    legacy_regions = diagnostics.get("regions") or []

    stat_rows = [[key, value] for key, value in sorted(stats.items())]
    strategy_rows = [[key, value] for key, value in sorted((payload.get("strategies") or {}).items())]
    region_rows = [
        [
            region.get("id"),
            region.get("z"),
            region.get("kind"),
            region.get("children"),
            json.dumps(region.get("diagnostics") or {}, sort_keys=True),
        ]
        for region in optimizer_regions
    ]
    object_rows = [
        [
            obj.get("id"),
            obj.get("z"),
            obj.get("shape"),
            obj.get("fill"),
            obj.get("bounds"),
            obj.get("area"),
        ]
        for obj in optimizer_objects
    ]
    legacy_region_rows = [
        [
            region.get("id"),
            region.get("area"),
            region.get("color_hex"),
            (region.get("strategies") or {}).get("geom"),
            region.get("symmetry"),
        ]
        for region in legacy_regions
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0;
      color: #161616;
      font: 12px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h3 {{ margin: 12px 0 6px; font-size: 13px; }}
    h3:first-child {{ margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      padding: 4px 6px;
      border: 1px solid #e2e2dc;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #f2f2ee; font-weight: 600; }}
    td {{ overflow-wrap: anywhere; }}
    .empty {{ margin: 0; color: #666; }}
  </style>
</head>
<body>
  <h3>Stats</h3>
  {_summary_table(["metric", "value"], stat_rows)}
  <h3>Strategies</h3>
  {_summary_table(["strategy", "count"], strategy_rows)}
  <h3>Optimizer Regions</h3>
  {_summary_table(["id", "z", "kind", "children", "diagnostics"], region_rows)}
  <h3>Optimizer Objects</h3>
  {_summary_table(["id", "z", "shape", "fill", "bounds", "area"], object_rows)}
  <h3>Trace Regions</h3>
  {_summary_table(["id", "area", "color", "shape", "symmetry"], legacy_region_rows)}
</body>
</html>
"""


def _matches_filter(entry: CorpusEntry, selectors: set[str]) -> bool:
    if not selectors:
        return True
    entry_id = _entry_id(entry)
    return bool({
        entry.name,
        entry.mode,
        f"{entry.mode}/{entry.name}",
        entry_id,
        f"{entry.mode}-{entry.name}",
    } & selectors)


def _write_index_if_needed(
    output: Path,
    entries: list[CorpusEntry],
    *,
    force: bool = False,
    debug_reports: bool = False,
) -> None:
    manifest_path = output / "corpus-manifest.json"
    manifest = _manifest_payload(entries, debug_reports=debug_reports)
    current_manifest = None
    if manifest_path.exists():
        current_manifest = json.loads(manifest_path.read_text())
    if not force and current_manifest == manifest and (output / "index.html").exists():
        return

    cards = [_entry_card(entry, debug_reports=debug_reports) for entry in entries]
    (output / "index.html").write_text(_html_document(cards))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _entry_card(entry: CorpusEntry, *, debug_reports: bool = False) -> str:
    svg_name = _safe_filename(entry)
    input_name = _input_filename(entry)
    raw_name = _raw_svg_filename(entry)
    diagnostics_name = _diagnostics_filename(entry)
    diagnostics_summary_name = _diagnostics_summary_filename(entry)
    options_name = _options_filename(entry)

    rel_svg = Path("svg") / svg_name
    rel_input = Path("rendered-input") / input_name
    rel_raw = Path("raw") / raw_name
    rel_diagnostics = Path("diagnostics") / diagnostics_name
    rel_diagnostics_summary = Path("diagnostic-summary") / diagnostics_summary_name
    rel_options = Path("options") / options_name
    debug_link = (
        f'<a href="debug/{html.escape(_entry_id(entry))}-passes/index.html">debug report</a>'
        if debug_reports else
        ""
    )
    return f"""
    <article class="entry" id="{html.escape(_entry_id(entry))}">
      <header>
        <h2>{html.escape(entry.mode)} / {html.escape(entry.name)}</h2>
        <a href="{html.escape(str(rel_svg))}">{html.escape(str(rel_svg))}</a>
        {debug_link}
      </header>
      <section class="renders">
        <figure>
          <figcaption>Input PNG</figcaption>
          <img src="{html.escape(str(rel_input))}" alt="{html.escape(entry.name)} input">
        </figure>
        <figure>
          <figcaption>Generated SVG</figcaption>
          <object data="{html.escape(str(rel_svg))}" type="image/svg+xml"></object>
        </figure>
      </section>
      <details>
        <summary>SVG</summary>
        <iframe class="code-frame" src="{html.escape(str(rel_raw))}" title="{html.escape(entry.name)} SVG source"></iframe>
      </details>
      <details>
        <summary>Diagnostics</summary>
        <iframe class="summary-frame" src="{html.escape(str(rel_diagnostics_summary))}" title="{html.escape(entry.name)} diagnostics summary"></iframe>
        <iframe class="code-frame" src="{html.escape(str(rel_diagnostics))}" title="{html.escape(entry.name)} diagnostics"></iframe>
      </details>
      <details>
        <summary>Options</summary>
        <iframe class="code-frame" src="{html.escape(str(rel_options))}" title="{html.escape(entry.name)} options"></iframe>
      </details>
    </article>"""


def generate_corpus_html(
    output: Path,
    entries: Iterable[CorpusEntry] | None = None,
    *,
    working_max_dim: int | None = None,
    cubic_paths: bool = False,
    epsilon: float = DEFAULT_CORPUS_EPSILON,
    max_error: float = DEFAULT_CORPUS_MAX_ERROR,
    only: Iterable[str] | None = None,
    rebuild_index: bool = False,
    workflow: str = "oneshot",
    trace_max_sampling_colors: int | Literal["auto"] = 16,
    trace_fit_strategy: Literal["quadratic", "progressive", "progressive_allow_lines"] = "quadratic",
    trace_level: Literal["pixel", "subpixel"] | None = None,
    trace_gradient_pretrace: bool = False,
    trace_support_weighted_corners: bool = False,
    debug_reports: bool = False,
) -> Path:
    output = Path(output)
    svg_dir = output / "svg"
    input_dir = output / "rendered-input"
    raw_dir = output / "raw"
    diagnostics_dir = output / "diagnostics"
    diagnostics_summary_dir = output / "diagnostic-summary"
    options_dir = output / "options"
    trace_cache_dir = output / "cache"
    debug_dir = output / "debug"
    svg_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_summary_dir.mkdir(parents=True, exist_ok=True)
    options_dir.mkdir(parents=True, exist_ok=True)
    trace_cache_dir.mkdir(parents=True, exist_ok=True)
    if debug_reports:
        debug_dir.mkdir(parents=True, exist_ok=True)

    all_entries = _entries_for_workflow(entries or default_entries(), workflow)
    _write_index_if_needed(output, all_entries, force=rebuild_index, debug_reports=debug_reports)

    selectors = set(only or ())
    for entry in [entry for entry in all_entries if _matches_filter(entry, selectors)]:
        image = np.asarray(entry.image_factory(), dtype=np.uint8)
        input_name = _input_filename(entry)
        input_path = input_dir / input_name
        Image.fromarray(image).save(input_path)

        options = _gallery_options(
            entry.options,
            working_max_dim,
            cubic_paths=cubic_paths,
            epsilon=epsilon,
            max_error=max_error,
        )
        print(f"rendering {entry.mode}/{entry.name}", file=sys.stderr)
        if workflow == "trace_auto":
            source = _entry_source_bytes(entry, image)
            svg, report, rendered_options = _trace_drawing_auto_with_cache(
                source,
                entry=entry,
                options=options,
                cache_dir=trace_cache_dir,
                max_sampling_colors=trace_max_sampling_colors,
                fit_strategy=trace_fit_strategy,
                trace_level=trace_level,
                gradient_pretrace=trace_gradient_pretrace,
                support_weighted_corners=trace_support_weighted_corners,
                debug_dir=debug_dir / f"{_entry_id(entry)}-passes" if debug_reports else None,
            )
        elif options.optimizer:
            svg, report = _idealize_optimizer_with_trace_cache(
                image,
                options=options,
                entry=entry,
                cache_dir=trace_cache_dir,
            )
        else:
            svg, report = idealize(image, options=options, report=True)
        svg_name = _safe_filename(entry)
        svg_path = svg_dir / svg_name
        svg_path.write_text(svg)
        (raw_dir / _raw_svg_filename(entry)).write_text(svg)
        (diagnostics_dir / _diagnostics_filename(entry)).write_text(_diagnostics_json(report))
        (diagnostics_summary_dir / _diagnostics_summary_filename(entry)).write_text(_diagnostics_summary_html(report))
        (options_dir / _options_filename(entry)).write_text(
            _options_json(rendered_options if workflow == "trace_auto" else options)
        )
    return output / "index.html"


def _html_document(cards: list[str]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>vectormark SVG corpus</title>
  <style>
    body {{
      margin: 24px;
      background: #f7f7f5;
      color: #161616;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    h1 {{ margin: 0 0 16px; font-size: 24px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .entry {{
      border: 1px solid #d6d6d2;
      border-radius: 6px;
      background: white;
      padding: 12px;
    }}
    .entry header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }}
    h2 {{ margin: 0; font-size: 14px; }}
    a {{ color: #2457a7; }}
    .renders {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    figure {{ margin: 0; }}
    figcaption {{
      margin-bottom: 4px;
      color: #555;
      font-size: 12px;
      font-weight: 600;
    }}
    object,
    img {{
      display: block;
      width: 100%;
      height: 240px;
      border: 1px solid #e7e7e2;
      background: white;
      object-fit: contain;
    }}
    details {{ margin-top: 10px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    pre,
    .code-frame,
    .summary-frame {{
      max-height: 360px;
      overflow: auto;
      padding: 10px;
      border: 1px solid #e5e5df;
      border-radius: 4px;
      background: #fbfbf8;
      font-size: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .code-frame,
    .summary-frame {{
      display: block;
      box-sizing: border-box;
      width: 100%;
      height: 360px;
      white-space: normal;
    }}
    .summary-frame {{
      height: 260px;
      margin-bottom: 8px;
      background: white;
    }}
  </style>
</head>
<body>
  <h1>vectormark SVG corpus</h1>
  <main class="grid">{''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the vectormark SVG corpus gallery.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CORPUS_OUTPUT)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="JSON manifest of corpus image URLs. Defaults to corpus/sources.json when present.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CORPUS_CACHE,
        help="Directory for downloaded source images.",
    )
    parser.add_argument(
        "--refresh-sources",
        action="store_true",
        help="Redownload source-manifest images even when cached.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of corpus PNGs. Defaults to corpus/input, then scratch/real-logos.",
    )
    parser.add_argument(
        "--working-max-dim",
        type=int,
        default=None,
        help="Opt-in downscale long side for faster gallery generation.",
    )
    parser.add_argument(
        "--cubic-paths",
        action="store_true",
        help="Opt in to cubic Bézier fitting for curved path runs.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_CORPUS_EPSILON,
        help=f"Corpus geometry tolerance in pixels. Defaults to {DEFAULT_CORPUS_EPSILON}.",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=DEFAULT_CORPUS_MAX_ERROR,
        help=f"Corpus Bezier fit tolerance in pixels. Defaults to {DEFAULT_CORPUS_MAX_ERROR}.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Render only matching entries. Accepts name, mode, mode/name, or mode-name. May be repeated.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rewrite index.html even when the corpus manifest is unchanged.",
    )
    parser.add_argument(
        "--workflow",
        choices=("trace-auto", "oneshot"),
        default="trace-auto",
        help="trace-auto uses the MCP trace_drawing(refine='auto') workflow; oneshot preserves the legacy current/optimizer gallery.",
    )
    parser.add_argument(
        "--trace-max-sampling-colors",
        default="16",
        help="Trace-auto palette ceiling: an integer >=2 or auto. Defaults to 16.",
    )
    parser.add_argument(
        "--trace-fit-strategy",
        choices=("quadratic", "progressive", "progressive_allow_lines"),
        default="quadratic",
        help="Trace-auto path fitting strategy. Defaults to quadratic.",
    )
    parser.add_argument(
        "--trace-level",
        choices=("pixel", "subpixel"),
        default=None,
        help="Trace-auto boundary level. Defaults to each entry's legacy aa_contours setting.",
    )
    parser.add_argument(
        "--trace-gradient-pretrace",
        action="store_true",
        help="Trace a gradient-neutral material proxy while retaining the original raster for geometry and fills.",
    )
    parser.add_argument(
        "--trace-support-weighted-corners",
        action="store_true",
        help="Preserve shallow corners whose long supporting runs make flattening visibly inaccurate.",
    )
    parser.add_argument(
        "--debug-reports",
        action="store_true",
        help="Write linked per-image trace-auto pass reports under output/debug without retracing.",
    )
    args = parser.parse_args()
    if args.source_manifest is not None:
        entries = source_manifest_entries(
            args.source_manifest,
            args.cache_dir,
            refresh=args.refresh_sources,
        )
    elif args.input_dir is not None:
        entries = corpus_entries(args.input_dir)
    else:
        entries = None
    trace_max_sampling_colors: int | Literal["auto"]
    if args.trace_max_sampling_colors == "auto":
        trace_max_sampling_colors = "auto"
    else:
        try:
            trace_max_sampling_colors = int(args.trace_max_sampling_colors)
        except ValueError as error:
            parser.error("--trace-max-sampling-colors must be auto or an integer >= 2")
            raise AssertionError("parser.error exits") from error
        if trace_max_sampling_colors < 2:
            parser.error("--trace-max-sampling-colors must be auto or an integer >= 2")
    index = generate_corpus_html(
        args.output,
        entries,
        working_max_dim=args.working_max_dim,
        cubic_paths=args.cubic_paths,
        epsilon=args.epsilon,
        max_error=args.max_error,
        only=args.only,
        rebuild_index=args.rebuild_index,
        workflow=args.workflow.replace("-", "_"),
        trace_max_sampling_colors=trace_max_sampling_colors,
        trace_fit_strategy=args.trace_fit_strategy,
        trace_level=args.trace_level,
        trace_gradient_pretrace=args.trace_gradient_pretrace,
        trace_support_weighted_corners=args.trace_support_weighted_corners,
        debug_reports=args.debug_reports,
    )
    print(index)


if __name__ == "__main__":
    main()

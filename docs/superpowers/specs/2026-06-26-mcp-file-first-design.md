# File-First Image Handoff for the vectoreyes MCP Server — Design

**Status:** approved (brainstorming)
**Date:** 2026-06-26
**Area:** `src/vectoreyes/mcp_server.py` + new `src/vectoreyes/mcp_image.py`; `docs/mcp.md`.

## Goal

Replace the base64-first MCP API with a **file/artifact-first** one optimized for
ChatGPT-generated images, user uploads, and future design-app handoffs. The server
accepts an image *reference*, resolves it to bytes server-side, preprocesses it, runs
the vectoreyes pipeline, and returns SVG + diagnostics + an optional preview. The
assistant does no base64 encoding, downsampling, or quantization.

## Why file-first (now viable)

ChatGPT's Apps SDK supports **file parameters**: a tool marks an input with
`_meta["openai/fileParams"]`, and ChatGPT passes a file *reference* object
`{download_url, file_id, mime_type?, file_name?}` (download_url + file_id required). The
server fetches bytes with a plain GET on `download_url` (temporary, no auth, valid only
during the tool call). This removes the reason base64 was the default: ChatGPT-generated
and uploaded images can now arrive as references, avoiding huge JSON payloads, forced
downsampling, and colour-fidelity loss. (Known caveat: the Apps SDK has reported mobile
file-handling issues; web works.)

## Public tool surface (after)

- `idealize_logo({ image, options? })` — **primary**, file-first.
- `render_idealized_logo({ result })` — render an existing `idealize_logo` result.
- `idealize_logo_data({ image_base64, ... })` — **deprecated internal fallback**, kept
  for hosts that cannot provide a file reference; not advertised as primary.

## `idealize_logo`

### Input

`image` is ONE object that is a superset of every source, marked as a file param so
ChatGPT fills the reference half while CLI/local/design-app hosts fill a string field.
All fields optional:

```
image: {
  download_url?: str, file_id?: str, mime_type?: str, file_name?: str,  # ChatGPT file ref
  path?: str,        # local filesystem path  (LOCAL-TRUST / stdio only)
  url?: str,         # explicit https URL     (HTTP: SSRF-guarded)
  data_uri?: str,    # data:image/...;base64,...
  base64?: str,      # bare base64
}
options?: {
  colors?: int = 16, flatten?: bool = false, no_symmetry?: bool = false,
  epsilon?: float = 1.5, max_error?: float = 1.0,
  preprocess?: {
    crop_to_content?: bool = true, max_size_px?: int = 1024,
    preserve_transparency?: bool = true, quantize?: bool = false,
  }
}
```
Tool `_meta`: `{ "openai/fileParams": ["image"], "ui": {...}, "openai/outputTemplate": WIDGET_URI, ... }`.

### Output

The tool returns **structured content** plus, best-effort, a **sibling image block**
(MCP can't embed binary inside JSON, so the preview is its own content item, not a JSON
field):

```
structured: { svg: str, width: int, height: int, svg_bytes: int,
              preview_available: bool, diagnostics: {...} }
+ (optional) ImageContent(PNG)   # the rendered preview, as a separate content block
```
The preview is best-effort: when `resvg-py` (the scoring extra) is present, the SVG is
rendered to PNG and appended as an MCP `ImageContent` block (works for remote/ChatGPT and
lets the model see the result), and `preview_available` is true; otherwise it is omitted,
`preview_available` is false, and a `warnings[]` note explains why. The widget renders the
SVG itself regardless.

## `resolve_image` (new, in `mcp_image.py`)

```
resolve_image(image: dict, *, local_trust: bool) -> ResolvedImage
ResolvedImage = { bytes, mime_type: "image/png"|"image/jpeg"|"image/webp",
                  filename?, sha256, source_kind: "platform_file"|"local_path"|"url"|"data_uri"|"base64" }
```

Resolution order (first present field wins):
1. `download_url` → GET it (→ `platform_file`). Temporary, no auth.
2. `path` → read the file (→ `local_path`). **Only when `local_trust`** (stdio); over
   HTTP a `path` is rejected with `IMAGE_UNRESOLVABLE` (arbitrary host read is unsafe).
3. `url` → GET it (→ `url`). Over HTTP, **SSRF-guarded**: https only, reject
   loopback/private/link-local/metadata IPs, enforce `MAX_INPUT_BYTES`, require an
   image mime.
4. `data_uri` → strip the prefix, decode (→ `data_uri`).
5. `base64` → decode (→ `base64`).
6. none → `IMAGE_UNRESOLVABLE`.

After fetch: sniff the real mime from the bytes (Pillow), reject non PNG/JPEG/WebP with
`UNSUPPORTED_IMAGE_TYPE`; reject `> MAX_INPUT_BYTES` with `IMAGE_TOO_LARGE`; reject
dimensions `> MAX_DIMENSION_PX` with `IMAGE_TOO_LARGE`. `download_url`/`url` fetches share
those byte/size guards.

## `preprocess_image` (new, in `mcp_image.py`)

```
preprocess_image(bytes, opts) -> (np.ndarray, ProcessedMeta)
```
Pillow-based, fidelity-preserving:
- `preserve_transparency`: keep alpha through cropping; composite on white only at the
  end (reusing the pipeline's `_flatten_on_white`) so the pipeline gets RGB.
- `crop_to_content`: trim fully-transparent margins (alpha) or near-white margins (no
  alpha) to the content bounding box.
- `max_size_px`: downscale (high-quality `LANCZOS`) only if the larger side exceeds it;
  never upscale.
- `quantize`: only if requested (default off) — vectoreyes does its own palette
  extraction, so pre-quantizing is usually harmful.
`ProcessedMeta` records `{width, height, cropped, resized, transparent, quantized}`.

## Execution

In-process Python API — no shell-out, no temp SVG roundtrip:
```python
from vectoreyes.pipeline import idealize, Options
svg = idealize(processed_array, options=Options(max_colors=colors, flatten=..., ...))
```
The CLI flags exist too, but the in-process call is the same pipeline, faster, and avoids
a subprocess. (The spec's optional `/tmp/vectoreyes/<id>/` temp layout is not needed for
the in-process path; it would only matter for a future shell-out worker.)

## Diagnostics

Returned as structured JSON:
```
{ input:   { source_kind, mime_type, bytes, sha256, original_width, original_height },
  processed:{ width, height, cropped, resized, transparent, quantized },
  vectoreyes:{ colors, flatten, no_symmetry, epsilon, max_error },
  output:  { svg_bytes, element_count, has_defs, has_paths, has_primitives, has_symmetry },
  warnings: [] }
```
`output.*` derived by scanning the emitted SVG (`<path`, `<rect`/`<circle`/… primitives,
`<use` symmetry, `<defs`). `warnings` collects soft issues (e.g. preview unavailable,
upscaled-not, mime sniff differed from declared).

## Structured errors

Raised as MCP tool errors carrying `error_code`:
`IMAGE_UNRESOLVABLE`, `UNSUPPORTED_IMAGE_TYPE`, `IMAGE_TOO_LARGE`, `VECTOREYES_FAILED`
(with `stderr`/message), plus `URL_NOT_ALLOWED` for an SSRF-blocked `url`.

## Limits (module constants)

`MAX_INPUT_BYTES = 20*1024*1024`, `MAX_DIMENSION_PX = 4096`,
`DEFAULT_MAX_SIZE_PX = 1024`, `DEFAULT_COLORS = 16`.

## `render_idealized_logo({ result })`

Accept the whole `idealize_logo` result object (svg/width/height) instead of separate
positional fields; re-derive `svg_bytes` from the svg (never trust a caller count). Keep
the widget-bridge behaviour. (Back-compat: also accept the old flat shape if present, so
existing callers don't break.)

## Security posture (preserved + extended)

The existing local-trust gate (`_LOCAL_TRUST = transport == "stdio"`) still governs the
filesystem surface:
- `path` resolution and any `output_path` writes are **stdio-only**; withheld/ignored over
  HTTP, exactly as today's file tool.
- `url` resolution over HTTP is SSRF-guarded (above).
- `download_url`/`data_uri`/`base64` are remote-safe.
So the ChatGPT remote flow (`image.download_url` → bytes → SVG) is fully safe, and no new
arbitrary host read/write is introduced over the network.

## File structure

- **New** `src/vectoreyes/mcp_image.py` — `resolve_image`, `preprocess_image`, the limits,
  the error types, and the diagnostics builder. Pure, no MCP/FastMCP imports → unit-testable
  without a server.
- **Modify** `src/vectoreyes/mcp_server.py` — the `idealize_logo` tool (file-first, fileParams
  meta), `render_idealized_logo({result})`, demote `idealize_logo_data` to a deprecated
  fallback. Keep the widget resource + transport/`main` unchanged.
- **Modify** `docs/mcp.md` — file-first flow as primary; base64 documented as fallback.

## Testing

- `mcp_image.py` units (`tests/test_mcp_image.py`): resolve each source kind (data_uri,
  base64, local path, a stubbed download_url/url via a local fixture/monkeypatched fetch);
  `path` rejected when `local_trust=False`; SSRF guard rejects loopback/private URLs;
  unsupported type / too-large / too-many-pixels raise the right `error_code`. Preprocess:
  transparent crop, near-white crop, downscale-only-when-bigger, quantize off by default,
  `ProcessedMeta` correctness.
- `mcp_server.py` updates (`tests/test_mcp_server.py`): `idealize_logo` with a `data_uri`
  image returns svg + diagnostics; `image.path` works under stdio and is rejected under
  HTTP-mode; `render_idealized_logo({result})` round-trips; `idealize_logo_data` still
  works (fallback). Full suite green.
- Acceptance: a real corpus PNG through `idealize_logo` (via `path` locally, and via a
  `data_uri` built from the bytes) yields the same SVG as the CLI on that file.

## Risks

- **`download_url` fetch** needs a network call inside the tool; bounded by a timeout and
  `MAX_INPUT_BYTES`; failures surface as `IMAGE_UNRESOLVABLE`. Unit tests stub it (no live
  ChatGPT URL available offline).
- **fileParams schema** must match OpenAI's documented shape exactly or ChatGPT won't fill
  it; the object is a superset, so a stray extra field must not break ChatGPT's filling —
  keep the file-ref fields first-class and the rest optional.
- **SSRF** is the main new network surface; default-deny private ranges and require https.

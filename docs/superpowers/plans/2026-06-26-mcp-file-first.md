# File-First MCP Image Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the vectoreyes MCP server to a file/artifact-first `idealize_logo({image, options})` that resolves an image reference to bytes server-side, preprocesses it, runs the in-process pipeline, and returns SVG + diagnostics + a best-effort preview.

**Architecture:** A new pure `src/vectoreyes/mcp_image.py` (resolve_image / preprocess_image / svg_output_facts / errors / limits — no MCP imports, unit-testable) does the heavy lifting. `mcp_server.py` adds the file-first `idealize_logo` tool (marked `openai/fileParams`), updates `render_idealized_logo` to take a full result, and demotes the base64 tool to a deprecated fallback.

**Tech Stack:** Python 3.12+, FastMCP (`mcp>=1.13`), Pillow, numpy, httpx (transitive via mcp), pytest.

## Global Constraints

- Python ≥ 3.12; pure-Python. DRY/YAGNI/TDD.
- `mcp_image.py` is pure (no `mcp`/FastMCP imports) so it unit-tests without a server.
- Limits (module constants, exact values): `MAX_INPUT_BYTES = 20*1024*1024`, `MAX_DIMENSION_PX = 4096`, `DEFAULT_MAX_SIZE_PX = 1024`, `DEFAULT_COLORS = 16`.
- Allowed mimes: `image/png`, `image/jpeg`, `image/webp`.
- Error codes (exact): `IMAGE_UNRESOLVABLE`, `UNSUPPORTED_IMAGE_TYPE`, `IMAGE_TOO_LARGE`, `URL_NOT_ALLOWED`, `VECTOREYES_FAILED`.
- Security: `path` resolution is **stdio/local-trust only** (rejected when `local_trust=False`); `url` over HTTP is SSRF-guarded (https-only, reject loopback/private/link-local/metadata IPs); `download_url`/`data_uri`/`base64` are remote-safe. Preserve the existing `_LOCAL_TRUST` gate.
- Preprocess defaults: `crop_to_content=True`, `max_size_px=1024`, `preserve_transparency=True`, `quantize=False`.
- Reuse `vectoreyes.pipeline.idealize` / `Options` / `_flatten_on_white`; do NOT shell out.
- Commit trailer exactly `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`, no other trailer.

---

### Task 1: `mcp_image.resolve_image` — reference → bytes

**Files:**
- Create: `src/vectoreyes/mcp_image.py`
- Test: `tests/test_mcp_image.py`

**Interfaces:**
- Produces:
  - constants `MAX_INPUT_BYTES`, `MAX_DIMENSION_PX`, `DEFAULT_MAX_SIZE_PX`, `DEFAULT_COLORS`, `ALLOWED_MIME`.
  - `class ImageError(Exception)` with `.error_code: str`, `.message: str`, `.to_dict()`.
  - `@dataclass ResolvedImage` `{bytes, mime_type, sha256, source_kind, filename}`.
  - `resolve_image(image: dict, *, local_trust: bool, fetch=_default_fetch) -> ResolvedImage`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_image.py`:

```python
import base64
import io
import numpy as np
import pytest
from PIL import Image
from vectoreyes.mcp_image import resolve_image, ImageError, MAX_INPUT_BYTES


def _png_bytes(size=(32, 32), color=(20, 120, 200)):
    buf = io.BytesIO(); Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_resolve_base64_and_data_uri():
    raw = _png_bytes()
    b64 = base64.b64encode(raw).decode()
    r = resolve_image({"base64": b64}, local_trust=False)
    assert r.source_kind == "base64" and r.mime_type == "image/png" and r.bytes == raw
    r2 = resolve_image({"data_uri": f"data:image/png;base64,{b64}"}, local_trust=False)
    assert r2.source_kind == "data_uri" and r2.bytes == raw and len(r2.sha256) == 64


def test_resolve_local_path_requires_local_trust(tmp_path):
    p = tmp_path / "m.png"; p.write_bytes(_png_bytes())
    r = resolve_image({"path": str(p)}, local_trust=True)
    assert r.source_kind == "local_path" and r.bytes == p.read_bytes()
    with pytest.raises(ImageError) as exc:
        resolve_image({"path": str(p)}, local_trust=False)
    assert exc.value.error_code == "IMAGE_UNRESOLVABLE"


def test_resolve_download_url_via_injected_fetch():
    raw = _png_bytes()
    r = resolve_image({"download_url": "https://host/file"}, local_trust=False,
                      fetch=lambda url, **kw: raw)
    assert r.source_kind == "platform_file" and r.bytes == raw


def test_resolve_url_ssrf_blocks_loopback():
    with pytest.raises(ImageError) as exc:
        resolve_image({"url": "http://127.0.0.1/x.png"}, local_trust=False,
                      fetch=lambda url, **kw: _png_bytes())
    assert exc.value.error_code == "URL_NOT_ALLOWED"


def test_resolve_rejects_unsupported_type():
    gif = io.BytesIO(); Image.new("RGB", (8, 8)).save(gif, format="GIF")
    with pytest.raises(ImageError) as exc:
        resolve_image({"base64": base64.b64encode(gif.getvalue()).decode()}, local_trust=False)
    assert exc.value.error_code == "UNSUPPORTED_IMAGE_TYPE"


def test_resolve_rejects_too_large():
    big = b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_INPUT_BYTES + 1)
    with pytest.raises(ImageError) as exc:
        resolve_image({"base64": base64.b64encode(big).decode()}, local_trust=False)
    assert exc.value.error_code == "IMAGE_TOO_LARGE"


def test_resolve_none_is_unresolvable():
    with pytest.raises(ImageError) as exc:
        resolve_image({}, local_trust=True)
    assert exc.value.error_code == "IMAGE_UNRESOLVABLE"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_mcp_image.py -k resolve -v`
Expected: FAIL — `vectoreyes.mcp_image` does not exist.

- [ ] **Step 3: Implement `mcp_image.py` (resolver half)**

Create `src/vectoreyes/mcp_image.py`:

```python
# SPDX-License-Identifier: MIT
"""Pure helpers for the file-first MCP image handoff: resolve an image reference to
validated bytes, preprocess it, and summarize the emitted SVG. No MCP imports."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_DIMENSION_PX = 4096
DEFAULT_MAX_SIZE_PX = 1024
DEFAULT_COLORS = 16
ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
_PIL_FORMAT_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_FETCH_TIMEOUT = 15.0


class ImageError(Exception):
    """A structured, code-carrying error for the MCP image tools."""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"error_code": self.error_code, "message": self.message}


@dataclass(frozen=True)
class ResolvedImage:
    bytes: bytes
    mime_type: str
    sha256: str
    source_kind: str
    filename: str | None = None


def _default_fetch(url: str, *, max_bytes: int = MAX_INPUT_BYTES, timeout: float = _FETCH_TIMEOUT) -> bytes:
    """GET `url` and return up to max_bytes; raises ImageError(IMAGE_UNRESOLVABLE) on failure."""
    import httpx
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.content
    except Exception as exc:  # network / HTTP / decode
        raise ImageError("IMAGE_UNRESOLVABLE", f"could not fetch image url: {exc}") from exc
    if len(data) > max_bytes:
        raise ImageError("IMAGE_TOO_LARGE", "fetched image exceeds the size limit")
    return data


def _assert_url_safe(url: str) -> None:
    """SSRF guard for a caller-supplied `url`: https only, no loopback/private/link-local/
    metadata hosts."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ImageError("URL_NOT_ALLOWED", "url must be an absolute https URL")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ImageError("URL_NOT_ALLOWED", f"could not resolve url host: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ImageError("URL_NOT_ALLOWED", "url resolves to a non-public address")


def _decode_b64(data: str) -> bytes:
    data = data.strip()
    if data.startswith("data:"):
        _, _, data = data.partition(",")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("IMAGE_UNRESOLVABLE", "image is not valid base64") from exc


def _validate(data: bytes) -> str:
    """Size + dimension + mime validation. Returns the sniffed mime."""
    if len(data) > MAX_INPUT_BYTES:
        raise ImageError("IMAGE_TOO_LARGE", "input exceeds the configured size limit")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except Exception as exc:
        raise ImageError("UNSUPPORTED_IMAGE_TYPE", "could not decode image; expected PNG, JPEG, or WebP") from exc
    mime = _PIL_FORMAT_MIME.get(fmt or "")
    if mime not in ALLOWED_MIME:
        raise ImageError("UNSUPPORTED_IMAGE_TYPE", "expected PNG, JPEG, or WebP")
    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
        raise ImageError("IMAGE_TOO_LARGE", f"image dimension exceeds {MAX_DIMENSION_PX}px")
    return mime


def resolve_image(image: dict, *, local_trust: bool, fetch=_default_fetch) -> ResolvedImage:
    """Resolve an image reference object to validated bytes. Resolution order:
    download_url (platform_file) -> path (local_trust only) -> url (SSRF-guarded) ->
    data_uri -> base64. Raises ImageError with a structured code on any failure."""
    image = image or {}
    if image.get("download_url"):
        data, kind, name = fetch(image["download_url"]), "platform_file", image.get("file_name")
    elif image.get("path"):
        if not local_trust:
            raise ImageError("IMAGE_UNRESOLVABLE", "local paths are not accepted over a network transport")
        p = Path(image["path"]).expanduser()
        if not p.is_file():
            raise ImageError("IMAGE_UNRESOLVABLE", f"image path does not exist: {p}")
        data, kind, name = p.read_bytes(), "local_path", p.name
    elif image.get("url"):
        _assert_url_safe(image["url"])
        data, kind, name = fetch(image["url"]), "url", None
    elif image.get("data_uri"):
        data, kind, name = _decode_b64(image["data_uri"]), "data_uri", None
    elif image.get("base64"):
        data, kind, name = _decode_b64(image["base64"]), "base64", None
    else:
        raise ImageError("IMAGE_UNRESOLVABLE", "no resolvable image reference was provided")
    mime = _validate(data)
    return ResolvedImage(bytes=data, mime_type=mime, sha256=hashlib.sha256(data).hexdigest(),
                         source_kind=kind, filename=name)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_mcp_image.py -k resolve -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/mcp_image.py tests/test_mcp_image.py
git commit -m "feat(mcp): resolve_image — file-ref -> validated bytes with security gating

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: `preprocess_image` + `svg_output_facts`

**Files:**
- Modify: `src/vectoreyes/mcp_image.py`
- Test: `tests/test_mcp_image.py`

**Interfaces:**
- Consumes: `vectoreyes.pipeline._flatten_on_white`, `MAX_*`/`DEFAULT_*` constants.
- Produces:
  - `@dataclass ProcessedMeta` `{width, height, cropped, resized, transparent, quantized}`.
  - `preprocess_image(data: bytes, *, crop_to_content=True, max_size_px=DEFAULT_MAX_SIZE_PX, preserve_transparency=True, quantize=False) -> tuple[np.ndarray, ProcessedMeta]` — returns an (H,W,3) uint8 RGB array (composited on white) ready for `idealize`.
  - `svg_output_facts(svg: str) -> dict` `{element_count, has_defs, has_paths, has_primitives, has_symmetry}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp_image.py`:

```python
def _rgba(size, color, alpha):
    im = Image.new("RGBA", size, (*color, alpha))
    return im


def test_preprocess_crops_transparent_margin_and_keeps_size():
    from vectoreyes.mcp_image import preprocess_image
    im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))           # all transparent
    im.paste((200, 30, 30, 255), (30, 30, 70, 70))            # a 40x40 opaque block
    buf = io.BytesIO(); im.save(buf, format="PNG")
    arr, meta = preprocess_image(buf.getvalue())
    assert meta.transparent is True and meta.cropped is True
    assert meta.width == 40 and meta.height == 40              # cropped to content
    assert arr.shape == (40, 40, 3) and arr.dtype == np.uint8


def test_preprocess_downscales_only_when_larger():
    from vectoreyes.mcp_image import preprocess_image
    big = io.BytesIO(); Image.new("RGB", (2000, 1000), (10, 20, 30)).save(big, format="PNG")
    arr, meta = preprocess_image(big.getvalue(), crop_to_content=False, max_size_px=1024)
    assert meta.resized is True and max(meta.width, meta.height) == 1024
    small = io.BytesIO(); Image.new("RGB", (300, 200), (10, 20, 30)).save(small, format="PNG")
    arr2, meta2 = preprocess_image(small.getvalue(), crop_to_content=False, max_size_px=1024)
    assert meta2.resized is False and (meta2.width, meta2.height) == (300, 200)


def test_svg_output_facts():
    from vectoreyes.mcp_image import svg_output_facts
    svg = '<svg><defs><linearGradient/></defs><rect/><path/><use href="#s0"/></svg>'
    f = svg_output_facts(svg)
    assert f["has_defs"] and f["has_paths"] and f["has_primitives"] and f["has_symmetry"]
    assert f["element_count"] == 3                              # rect + path + use
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_mcp_image.py -k "preprocess or svg_output" -v`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement preprocess + facts**

Append to `src/vectoreyes/mcp_image.py` (add `import re`, `import numpy as np`, and `from .pipeline import _flatten_on_white` at the top with the other imports):

```python
@dataclass(frozen=True)
class ProcessedMeta:
    width: int
    height: int
    cropped: bool
    resized: bool
    transparent: bool
    quantized: bool


def _content_bbox(im: Image.Image):
    """Bounding box of non-transparent (if alpha) or non-near-white (if opaque) content."""
    if "A" in im.getbands():
        bbox = im.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    else:
        gray = im.convert("L").point(lambda v: 0 if v >= 250 else 255)   # near-white -> background
        bbox = gray.getbbox()
    return bbox


def preprocess_image(data: bytes, *, crop_to_content: bool = True,
                     max_size_px: int = DEFAULT_MAX_SIZE_PX, preserve_transparency: bool = True,
                     quantize: bool = False) -> tuple["np.ndarray", ProcessedMeta]:
    """Fidelity-preserving preprocess: optionally crop to content, downscale only if larger
    than max_size_px, keep alpha until the final white composite. Returns an (H,W,3) uint8
    RGB array ready for idealize, plus the processing metadata."""
    im = Image.open(io.BytesIO(data))
    im.load()
    transparent = "A" in im.getbands() and im.getchannel("A").getextrema()[0] < 255

    cropped = False
    if crop_to_content:
        bbox = _content_bbox(im)
        if bbox and bbox != (0, 0, im.width, im.height):
            im = im.crop(bbox)
            cropped = True

    resized = False
    if max(im.size) > max_size_px:
        scale = max_size_px / max(im.size)
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        resized = True

    quantized = False
    if quantize:
        im = im.convert("RGBA").quantize(colors=256).convert("RGBA")
        quantized = True

    arr = _flatten_on_white(im) if (preserve_transparency or transparent) else np.asarray(im.convert("RGB"), np.uint8)
    h, w = arr.shape[:2]
    return arr, ProcessedMeta(width=w, height=h, cropped=cropped, resized=resized,
                              transparent=bool(transparent), quantized=quantized)


_PRIMITIVE_RE = re.compile(r"<(rect|circle|ellipse|polygon|line)\b")


def svg_output_facts(svg: str) -> dict:
    """Structural facts about an emitted SVG for diagnostics."""
    element_count = sum(svg.count(f"<{tag}") for tag in
                        ("path", "rect", "circle", "ellipse", "polygon", "line", "use", "image"))
    return {
        "element_count": element_count,
        "has_defs": "<defs" in svg,
        "has_paths": "<path" in svg,
        "has_primitives": bool(_PRIMITIVE_RE.search(svg)),
        "has_symmetry": "<use" in svg,
    }
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_mcp_image.py -k "preprocess or svg_output" -v`
Expected: PASS. (If `_content_bbox`'s near-white threshold trims a legitimately white-content mark in a later corpus check, that is a Task 4 tuning note — keep the alpha path exact.)

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/mcp_image.py tests/test_mcp_image.py
git commit -m "feat(mcp): preprocess_image (crop/resize/keep-alpha) + svg_output_facts

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: `idealize_logo` tool (file-first) + diagnostics

**Files:**
- Modify: `src/vectoreyes/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `resolve_image`, `preprocess_image`, `svg_output_facts`, `ImageError`, constants (Tasks 1-2); `idealize`, `Options` (pipeline).
- Produces:
  - `idealize_logo_image(image: dict, options: dict | None, *, local_trust: bool) -> tuple[dict, bytes | None]` — pure helper: resolve → preprocess → idealize → assemble `(structured_result, preview_png_bytes_or_None)`. Raises `ImageError`.
  - `idealize_logo` MCP tool marked `_meta["openai/fileParams"] = ["image"]`, returning the structured result + an `ImageContent` preview block when available.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`:

```python
import base64, io
import numpy as np
from PIL import Image as _Img


def _png_b64(size=(48, 48), color=(30, 100, 220)):
    buf = io.BytesIO(); _Img.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_idealize_logo_image_from_data_uri():
    from vectoreyes.mcp_server import idealize_logo_image
    result, preview = idealize_logo_image(
        {"data_uri": f"data:image/png;base64,{_png_b64()}"}, {"colors": 8}, local_trust=False)
    assert result["svg"].startswith("<svg ") and result["svg_bytes"] == len(result["svg"].encode())
    d = result["diagnostics"]
    assert d["input"]["source_kind"] == "data_uri" and d["input"]["mime_type"] == "image/png"
    assert d["vectoreyes"]["colors"] == 8 and "element_count" in d["output"]
    assert result["preview_available"] in (True, False)        # best-effort
    assert preview is None or isinstance(preview, (bytes, bytearray))


def test_idealize_logo_image_path_blocked_without_local_trust(tmp_path):
    from vectoreyes.mcp_server import idealize_logo_image
    from vectoreyes.mcp_image import ImageError
    p = tmp_path / "m.png"; _Img.new("RGB", (16, 16), "white").save(p)
    # local_trust=True works; False rejects
    r, _ = idealize_logo_image({"path": str(p)}, None, local_trust=True)
    assert r["svg"].startswith("<svg ")
    try:
        idealize_logo_image({"path": str(p)}, None, local_trust=False)
    except ImageError as e:
        assert e.error_code == "IMAGE_UNRESOLVABLE"
    else:
        raise AssertionError("path must be rejected without local trust")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -k idealize_logo_image -v`
Expected: FAIL — `idealize_logo_image` not defined.

- [ ] **Step 3: Implement the helper + tool**

In `src/vectoreyes/mcp_server.py`, add imports near the top:

```python
from .mcp_image import (
    DEFAULT_COLORS, DEFAULT_MAX_SIZE_PX, ImageError, preprocess_image,
    resolve_image, svg_output_facts,
)
```

Add the pure helper (above the `mcp = FastMCP(...)` block):

```python
def idealize_logo_image(image: dict, options: dict | None, *, local_trust: bool) -> tuple[dict, bytes | None]:
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
        flatten=options.get("flatten", False),
        no_symmetry=options.get("no_symmetry", False),
    )
    try:
        svg = idealize(arr, options=opts)
    except Exception as exc:
        raise ImageError("VECTOREYES_FAILED", f"vectoreyes failed to process the image: {exc}") from exc

    with Image.open(io.BytesIO(resolved.bytes)) as src:
        ow, oh = src.size
    warnings: list[str] = []
    preview = _render_preview_png(svg, meta.width, meta.height, warnings)

    diagnostics = {
        "input": {"source_kind": resolved.source_kind, "mime_type": resolved.mime_type,
                  "bytes": len(resolved.bytes), "sha256": resolved.sha256,
                  "original_width": ow, "original_height": oh},
        "processed": {"width": meta.width, "height": meta.height, "cropped": meta.cropped,
                      "resized": meta.resized, "transparent": meta.transparent, "quantized": meta.quantized},
        "vectoreyes": {"colors": opts.max_colors, "flatten": opts.flatten,
                       "no_symmetry": opts.no_symmetry, "epsilon": opts.epsilon, "max_error": opts.max_error},
        "output": {"svg_bytes": len(svg.encode()), **svg_output_facts(svg)},
        "warnings": warnings,
    }
    result = {"svg": svg, "width": meta.width, "height": meta.height,
              "svg_bytes": len(svg.encode()), "preview_available": preview is not None,
              "diagnostics": diagnostics}
    return result, preview


def _render_preview_png(svg: str, width: int, height: int, warnings: list[str]) -> bytes | None:
    """Best-effort: render the SVG to PNG via resvg-py; None (+ a warning) if unavailable."""
    try:
        import resvg_py
        return bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height))
    except Exception as exc:
        warnings.append(f"preview unavailable: {exc}")
        return None
```

Register the tool (after the existing tool definitions, replacing nothing — this is the new primary). FastMCP infers the input schema from the type hints; `image`/`options` are dicts so the schema is permissive, and the `openai/fileParams` meta marks `image` as a file field:

```python
@mcp.tool(
    title="Idealize logo",
    description=(
        "Idealize a raster logo into clean editable SVG. Pass the image by reference: a "
        "ChatGPT/host file (download_url+file_id), a local path, an https url, a data: URI, "
        "or base64. The server resolves and preprocesses it; no client-side base64 needed."
    ),
    meta={
        "openai/fileParams": ["image"],
        "ui": {"resourceUri": WIDGET_URI},
        "openai/outputTemplate": WIDGET_URI,
        "openai/toolInvocation/invoking": "Idealizing logo...",
        "openai/toolInvocation/invoked": "Idealized logo.",
    },
)
def idealize_logo(image: dict, options: dict | None = None) -> list:
    """File-first logo idealization. Returns the structured result plus a preview image."""
    from mcp.server.fastmcp.utilities.types import Image as MCPImage  # PNG content helper
    result, preview = idealize_logo_image(image, options, local_trust=_LOCAL_TRUST)
    contents: list = [result]
    if preview is not None:
        contents.append(MCPImage(data=preview, format="png"))
    return contents
```

> Note for the implementer: FastMCP returns a list of content items; a dict becomes structured/JSON content and `MCPImage(...)` becomes an `ImageContent` block. If `mcp.server.fastmcp.utilities.types.Image` is not importable in the installed `mcp` version, build the block directly: `from mcp.types import ImageContent; ImageContent(type="image", data=base64.b64encode(preview).decode(), mimeType="image/png")` and return `[result, that]`. Verify which import the installed `mcp>=1.13` provides and use it; keep the dict-first ordering so structured content is preserved.

- [ ] **Step 4: Run to verify they pass, then the focused server suite**

Run: `uv run pytest tests/test_mcp_server.py -k "idealize_logo_image" -v`
Expected: PASS.

Run: `uv run pytest tests/test_mcp_server.py -q`
Expected: PASS (existing tests unaffected — the base64/file helpers are untouched).

- [ ] **Step 5: Commit**

```bash
git add src/vectoreyes/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): file-first idealize_logo tool with diagnostics + preview

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: render tool result-shape, base64 demotion, docs, E2E

**Files:**
- Modify: `src/vectoreyes/mcp_server.py` (`render_idealized_logo`, `idealize_logo_data` description)
- Modify: `docs/mcp.md`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: the `idealize_logo` tool (Task 3).
- Produces: `render_idealized_logo(result: dict | None = None, **legacy)` accepting the full `idealize_logo` result; `idealize_logo_data` re-described as a deprecated fallback.

- [ ] **Step 1: Write the failing E2E + render test**

Add to `tests/test_mcp_server.py`:

```python
def test_render_idealized_logo_accepts_full_result():
    from vectoreyes.mcp_server import render_idealized_logo
    res = {"svg": "<svg >x</svg>", "width": 10, "height": 12}
    out = render_idealized_logo(result=res)
    assert out["svg"] == res["svg"] and out["width"] == 10 and out["height"] == 12
    assert out["svg_bytes"] == len(res["svg"].encode())        # re-derived, not trusted


def test_stdio_server_exposes_file_first_idealize_logo_with_fileparams():
    async def go():
        params = StdioServerParameters(command="uv", args=["run", "vectoreyes-mcp"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                assert "idealize_logo" in tools
                meta = tools["idealize_logo"].meta or {}
                assert meta.get("openai/fileParams") == ["image"]
                # call it with a data_uri image
                import base64, io
                from PIL import Image
                b = io.BytesIO(); Image.new("RGB", (32, 32), (200, 30, 30)).save(b, format="PNG")
                uri = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
                r = await session.call_tool("idealize_logo", {"image": {"data_uri": uri}})
                sc = r.structuredContent or {}
                assert "<svg" in (sc.get("svg") or "") and sc.get("diagnostics")
    asyncio.run(go())
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -k "render_idealized_logo_accepts_full or file_first" -v`
Expected: FAIL — `render_idealized_logo` doesn't accept `result=`; fileParams meta absent on the old tool set until Task 3+4 land. (If run after Task 3, the fileParams test passes; the render test still fails until Step 3 here.)

- [ ] **Step 3: Update `render_idealized_logo` + demote `idealize_logo_data`**

Replace the `render_idealized_logo` definition (currently takes `image_path, svg, width, height, output_path`) with a result-first version that keeps back-compat:

```python
@mcp.tool(
    title="Render idealized logo",
    description="Render an idealize_logo result in the ChatGPT/MCP Apps widget. Pass the whole result object.",
    meta={
        "ui": {"resourceUri": WIDGET_URI},
        "openai/outputTemplate": WIDGET_URI,
        "openai/toolInvocation/invoking": "Rendering SVG preview...",
        "openai/toolInvocation/invoked": "Rendered SVG preview.",
    },
)
def render_idealized_logo(result: dict | None = None, image_path: str = "", svg: str = "",
                          width: int = 0, height: int = 0) -> dict[str, object]:
    """Render an existing idealized SVG result in the vectoreyes app. Accepts the full
    `idealize_logo` result (preferred) or the legacy flat fields."""
    if result:
        svg = result.get("svg", svg)
        width = result.get("width", width)
        height = result.get("height", height)
        image_path = (result.get("diagnostics", {}).get("input", {}).get("source_kind")) or image_path
    return IdealizeLogoResult(
        image_path=image_path, output_path=None, width=width, height=height,
        svg_bytes=len(svg.encode()), svg=svg,   # svg_bytes re-derived, never trusted from caller
    ).to_dict()
```

Update the `idealize_logo_data` tool description to mark it a deprecated fallback (edit its `description=` string):

```python
    description=(
        "DEPRECATED fallback. Prefer `idealize_logo` with an image reference. Idealize a "
        "base64-encoded raster (bare base64 or a data:image/...;base64,... URI) into SVG, "
        "for hosts that cannot pass a file reference."
    ),
```

- [ ] **Step 4: Run the tests, then the full suite**

Run: `uv run pytest tests/test_mcp_server.py -q`
Expected: PASS (incl. the E2E fileParams + render tests).

Run: `uv run pytest -q`
Expected: PASS — full suite (paste the verbatim summary line).

- [ ] **Step 5: Update `docs/mcp.md`**

In `docs/mcp.md`, make the file-first flow primary: in the tool list, describe `idealize_logo({image, options})` as the primary file-first tool (image is a reference: ChatGPT file param `download_url`/`file_id`, or `path`/`url`/`data_uri`/`base64`); note `idealize_logo_data` is a deprecated fallback; and add the explicit line:

> For ChatGPT-generated or uploaded images, pass the image/file reference directly (the tool is marked `openai/fileParams`). Do not base64-encode unless using a fallback host that cannot provide file references.

Keep the existing security and transport sections. Commit.

- [ ] **Step 6: Commit**

```bash
git add src/vectoreyes/mcp_server.py tests/test_mcp_server.py docs/mcp.md
git commit -m "feat(mcp): result-shaped render tool, demote base64 tool, file-first docs

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final Review

After all tasks: a quick manual E2E (`/tmp` harness like the one used during integration testing — drive `idealize_logo` over stdio with a `data_uri` image and confirm svg+diagnostics+preview), then dispatch a whole-branch review, then superpowers:finishing-a-development-branch to open the PR. PR body ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

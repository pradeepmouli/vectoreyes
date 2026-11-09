# SPDX-License-Identifier: MIT
"""Pure helpers for the file-first MCP image handoff: resolve an image reference to
validated bytes, preprocess it, and summarize the emitted SVG. No MCP imports."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import ipaddress
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urljoin

import numpy as np
from PIL import Image

from .pipeline import _flatten_on_white

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_DIMENSION_PX = 4096
DEFAULT_MAX_SIZE_PX = 1254
DEFAULT_COLORS = 16
ALLOWED_MIME = {"image/png"}
_PIL_FORMAT_MIME = {"PNG": "image/png"}
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


def _assert_url_safe(url: str) -> str:
    """SSRF guard for a caller-supplied `url`: https only, no loopback/private/link-local/
    metadata hosts. Resolves DNS once and returns the first validated IP string so callers
    can pin the connection to that address (closing the TOCTOU rebind window)."""
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
    return infos[0][4][0]  # validated IP for DNS pinning


def _default_fetch(url: str, *, max_bytes: int = MAX_INPUT_BYTES, timeout: float = _FETCH_TIMEOUT) -> bytes:
    """GET `url` and return up to max_bytes; raises ImageError on failure.

    Security hardening:
    - ``follow_redirects=False`` + manual loop: every Location header is re-validated
      via ``_assert_url_safe`` before following, with a cap of 5 hops.  A redirect to a
      private IP or non-https scheme raises ``ImageError("URL_NOT_ALLOWED", ...)``.
      (Closes SSRF redirect bypass — HIGH-1.)
    - DNS pinning: ``_assert_url_safe`` resolves the hostname once and returns the
      validated IP.  The TCP connection is made to that IP so httpx never re-resolves
      the hostname.  The original hostname is forwarded via the ``Host`` header and the
      ``sni_hostname`` request extension so the TLS handshake and certificate verification
      use the hostname, not the IP.  (Closes TOCTOU rebind — HIGH-3.)
    - Body is streamed and aborted as soon as it exceeds ``max_bytes``, capping memory
      usage against hostile oversized responses.
    """
    import httpx

    _MAX_REDIRECTS = 5
    current_url = url

    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            for hop in range(_MAX_REDIRECTS + 1):
                if hop == _MAX_REDIRECTS:
                    raise ImageError("URL_NOT_ALLOWED", "too many redirects")

                # Validate and pin DNS in one step: resolve once, connect to that IP.
                pinned_ip = _assert_url_safe(current_url)
                parsed = urlparse(current_url)
                hostname = parsed.hostname
                port = parsed.port or 443
                path_qs = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")

                # Build the request URL from the pinned IP so httpx never re-resolves.
                # IPv6 literals must be enclosed in brackets.
                ip_obj = ipaddress.ip_address(pinned_ip)
                ip_literal = f"[{pinned_ip}]" if ip_obj.version == 6 else pinned_ip
                pinned_url = f"https://{ip_literal}:{port}{path_qs}"

                request = httpx.Request(
                    "GET",
                    pinned_url,
                    headers={"Host": hostname},
                    # sni_hostname is forwarded to httpcore so TLS SNI and cert
                    # verification use the original hostname, not the IP address.
                    extensions={"sni_hostname": hostname.encode()},
                )
                resp = client.send(request, stream=True)

                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location", "")
                    resp.close()
                    if not location:
                        raise ImageError("IMAGE_UNRESOLVABLE", "redirect with no Location header")
                    # Resolve relative Location references to an absolute URL, then
                    # re-validate via _assert_url_safe on the next hop (HIGH-1).
                    if not location.startswith(("http://", "https://")):
                        location = urljoin(current_url, location)
                    current_url = location
                    continue

                chunks: list[bytes] = []
                size = 0
                try:
                    resp.raise_for_status()
                    for chunk in resp.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise ImageError("IMAGE_TOO_LARGE", "fetched image exceeds the size limit")
                        chunks.append(chunk)
                finally:
                    resp.close()
                return b"".join(chunks)
    except ImageError:
        raise
    except Exception as exc:
        raise ImageError("IMAGE_UNRESOLVABLE", f"could not fetch image url: {exc}") from exc


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
        raise ImageError("UNSUPPORTED_IMAGE_TYPE", "Could not decode PNG input. Export the image as PNG and retry.") from exc
    mime = _PIL_FORMAT_MIME.get(fmt or "")
    if mime not in ALLOWED_MIME:
        raise ImageError("UNSUPPORTED_IMAGE_TYPE", "Only PNG input is supported. Export the image as PNG and retry.")
    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
        raise ImageError("IMAGE_TOO_LARGE", f"image dimension exceeds {MAX_DIMENSION_PX}px")
    return mime


def resolve_image(image: dict, *, local_trust: bool, fetch=_default_fetch) -> ResolvedImage:
    """Resolve an image reference object to validated bytes. Resolution order:
    download_url (platform_file) -> path (local_trust only) -> url (SSRF-guarded) ->
    data_uri -> base64. Raises ImageError with a structured code on any failure."""
    image = image or {}
    if image.get("download_url"):
        # HIGH-2: apply the same SSRF guard to download_url that the url branch uses.
        # download_url is caller-supplied with no signature; treat it as untrusted.
        _assert_url_safe(image["download_url"])
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


# ---------------------------------------------------------------------------
# Task-2: preprocess_image + svg_output_facts
# ---------------------------------------------------------------------------

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


def preprocess_image_with_alpha(data: bytes, *, crop_to_content: bool = True,
                                max_size_px: int = DEFAULT_MAX_SIZE_PX,
                                preserve_transparency: bool = True,
                                quantize: bool = False) -> tuple[np.ndarray, ProcessedMeta, np.ndarray | None]:
    """Preprocess RGB plus aligned source alpha coverage when present.

    Geometry tracing can use the alpha plane as its exterior baseline, while
    material and fill analysis keeps using the white-composited RGB values.
    Both products undergo exactly the same crop/resize/quantize operations.
    """
    im = Image.open(io.BytesIO(data))
    im.load()
    # Palette PNGs can carry transparency in a tRNS chunk instead of exposing
    # an ``A`` band.  Normalize before inspecting alpha so indexed and RGBA
    # sources follow the same geometry-trace contract.
    im = im.convert("RGBA")
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

    alpha = np.asarray(im.getchannel("A"), dtype=np.uint8) if transparent else None
    arr = _flatten_on_white(im) if (preserve_transparency or transparent) else np.asarray(im.convert("RGB"), np.uint8)
    h, w = arr.shape[:2]
    return arr, ProcessedMeta(width=w, height=h, cropped=cropped, resized=resized,
                              transparent=bool(transparent), quantized=quantized), alpha


def preprocess_image(data: bytes, *, crop_to_content: bool = True,
                     max_size_px: int = DEFAULT_MAX_SIZE_PX, preserve_transparency: bool = True,
                     quantize: bool = False) -> tuple[np.ndarray, ProcessedMeta]:
    """Preprocess an image for RGB consumers.

    Use :func:`preprocess_image_with_alpha` when geometry must retain source
    transparency too.
    """
    arr, meta, _alpha = preprocess_image_with_alpha(
        data,
        crop_to_content=crop_to_content,
        max_size_px=max_size_px,
        preserve_transparency=preserve_transparency,
        quantize=quantize,
    )
    return arr, meta


_PRIMITIVE_RE = re.compile(r"<(rect|circle|ellipse|polygon|line)\b")
_ELEMENT_RE = re.compile(r"<(path|rect|circle|ellipse|polygon|line|use|image)\b")


def svg_output_facts(svg: str) -> dict:
    """Structural facts about an emitted SVG for diagnostics."""
    element_count = len(_ELEMENT_RE.findall(svg))
    return {
        "element_count": element_count,
        "has_defs": "<defs" in svg,
        "has_paths": "<path" in svg,
        "has_primitives": bool(_PRIMITIVE_RE.search(svg)),
        "has_symmetry": "<use" in svg,
    }

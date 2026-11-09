import base64
import io
import socket
import numpy as np
import pytest
from PIL import Image
from vectoreyes.mcp_image import resolve_image, ImageError, MAX_INPUT_BYTES


def _png_bytes(size=(32, 32), color=(20, 120, 200)):
    buf = io.BytesIO(); Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _public_addrinfo():
    """A getaddrinfo return value for a genuinely public routable IP (1.1.1.1).
    TEST-NET-3 (203.0.113.x) is classified is_private=True in Python 3.11+, so use
    an address that ipaddress considers globally routable."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 0))]


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


def test_resolve_download_url_via_injected_fetch(monkeypatch):
    import vectoreyes.mcp_image as mcp_mod
    raw = _png_bytes()
    # download_url now goes through the SSRF guard; monkeypatch DNS to a public IP.
    monkeypatch.setattr(mcp_mod.socket, "getaddrinfo", lambda *a, **kw: _public_addrinfo())
    r = resolve_image({"download_url": "https://host/file"}, local_trust=False,
                      fetch=lambda url, **kw: raw)
    assert r.source_kind == "platform_file" and r.bytes == raw


def test_resolve_download_url_ssrf_blocked(monkeypatch):
    """download_url whose host resolves to a private IP must be rejected (HIGH-2)."""
    import vectoreyes.mcp_image as mcp_mod
    monkeypatch.setattr(
        mcp_mod.socket, "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )
    with pytest.raises(ImageError) as exc:
        resolve_image({"download_url": "https://evil.example.com/img.png"}, local_trust=False,
                      fetch=lambda url, **kw: _png_bytes())
    assert exc.value.error_code == "URL_NOT_ALLOWED"


def test_resolve_url_ssrf_blocks_non_https():
    """Scheme guard: non-https URLs are rejected before DNS is even consulted."""
    with pytest.raises(ImageError) as exc:
        resolve_image({"url": "http://127.0.0.1/x.png"}, local_trust=False,
                      fetch=lambda url, **kw: _png_bytes())
    assert exc.value.error_code == "URL_NOT_ALLOWED"


def test_resolve_url_ssrf_blocks_private_ip_via_dns(monkeypatch):
    """IP guard: an https URL whose host DNS-resolves to a private IP is rejected."""
    import vectoreyes.mcp_image as mcp_mod
    monkeypatch.setattr(
        mcp_mod.socket, "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    )
    with pytest.raises(ImageError) as exc:
        resolve_image({"url": "https://evil.example.com/img.png"}, local_trust=False,
                      fetch=lambda url, **kw: _png_bytes())
    assert exc.value.error_code == "URL_NOT_ALLOWED"


def test_default_fetch_pins_dns_and_sets_sni(monkeypatch):
    """_default_fetch must connect to the pinned IP with Host + sni_hostname set (HIGH-3)."""
    import httpx
    import vectoreyes.mcp_image as mcp_mod

    captured: list = []
    raw = _png_bytes()

    monkeypatch.setattr(mcp_mod.socket, "getaddrinfo", lambda *a, **kw: _public_addrinfo())

    class _MockResp:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self): pass
        def iter_bytes(self): yield raw
        def close(self): pass

    class _MockClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

        def send(self, request, *, stream=False):
            captured.append(request)
            return _MockResp()

    monkeypatch.setattr(httpx, "Client", _MockClient)

    from vectoreyes.mcp_image import _default_fetch
    _default_fetch("https://example.com/img.png")

    assert len(captured) == 1
    req = captured[0]
    assert "1.1.1.1" in str(req.url), f"expected pinned IP in URL, got {req.url}"
    assert req.headers.get("host") == "example.com", f"expected Host: example.com, got {req.headers.get('host')}"
    assert req.extensions.get("sni_hostname") == b"example.com", (
        f"expected sni_hostname=b'example.com', got {req.extensions.get('sni_hostname')}"
    )


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


# ---------------------------------------------------------------------------
# Task-2 tests: preprocess_image + svg_output_facts
# ---------------------------------------------------------------------------

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


def test_preprocess_with_alpha_keeps_aligned_source_coverage():
    from vectoreyes.mcp_image import preprocess_image_with_alpha
    im = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
    im.paste((200, 30, 30, 255), (3, 4, 9, 10))
    buf = io.BytesIO(); im.save(buf, format="PNG")

    arr, meta, alpha = preprocess_image_with_alpha(buf.getvalue(), crop_to_content=False)

    assert meta.transparent is True
    assert arr.shape == alpha.shape + (3,)
    assert alpha[4, 3] == 255 and alpha[0, 0] == 0


def test_preprocess_with_alpha_recognizes_palette_png_transparency():
    from vectoreyes.mcp_image import preprocess_image_with_alpha
    rgba = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
    rgba.paste((0, 126, 229, 255), (3, 3, 9, 9))
    indexed = rgba.convert("P", palette=Image.Palette.ADAPTIVE)
    buf = io.BytesIO()
    indexed.save(buf, format="PNG", transparency=0)

    _rgb, meta, alpha = preprocess_image_with_alpha(buf.getvalue(), crop_to_content=False)

    assert meta.transparent is True
    assert alpha is not None
    assert alpha[0, 0] == 0 and alpha[4, 4] == 255


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

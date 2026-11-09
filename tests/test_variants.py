import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

import vectoreyes.variants as V
from vectoreyes.cli import main as cli_main
from vectoreyes.pipeline import AxisLine, IdealizeReport
from vectoreyes.score import SvgRendererUnavailable
from vectoreyes.variants import (
    DEFAULT_EPSILONS, DEFAULT_MAX_ERRORS, Variant, _axis_line_svg, _inject_axes,
    generate_variants, write_variant_set,
)


def _mark(n=80):
    im = Image.new("RGB", (n, n), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((8, 8, n - 8, n - 8), fill=(30, 100, 220))
    d.rectangle((30, 30, 50, 70), fill=(220, 40, 40))
    return np.asarray(im, dtype=np.uint8)


def test_generate_variants_grid_shape_and_order():
    eps, mes = (0.5, 3.0), (0.5, 2.5)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    assert len(variants) == 4
    # row-major: epsilon outer, max_error inner
    assert [(v.epsilon, v.max_error) for v in variants] == [
        (0.5, 0.5), (0.5, 2.5), (3.0, 0.5), (3.0, 2.5),
    ]
    for v in variants:
        assert isinstance(v, Variant)
        assert v.svg.startswith("<svg ")
        assert v.error is None
        assert v.report.elements >= 1


def test_generate_variants_defaults_are_3x3():
    variants = generate_variants(_mark())
    assert len(variants) == len(DEFAULT_EPSILONS) * len(DEFAULT_MAX_ERRORS) == 9


def test_generate_variants_params_take_effect():
    variants = generate_variants(_mark(), epsilons=(0.3, 6.0), max_errors=(0.5,))
    # a very loose epsilon must change the geometry vs a very tight one
    assert variants[0].svg != variants[1].svg


def test_generate_variants_captures_failed_cell(monkeypatch):
    # a cell whose idealize raises must be captured as a failed Variant (error set,
    # empty svg) without aborting the grid.
    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr("vectoreyes.variants.idealize", boom)
    variants = generate_variants(_mark(), epsilons=(0.5, 3.0), max_errors=(1.0,))
    assert len(variants) == 2
    assert all(v.error == "kaboom" and v.svg == "" for v in variants)
    assert all(isinstance(v.report, IdealizeReport) for v in variants)


def test_write_variant_set_writes_svgs_and_manifest(tmp_path):
    variants = generate_variants(_mark(), epsilons=(0.5, 3.0), max_errors=(1.0,))
    write_variant_set(variants, tmp_path, source="mark.png")

    assert (tmp_path / "variant-e0_5-m1.svg").read_text().startswith("<svg ")
    assert (tmp_path / "variant-e3-m1.svg").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["source"] == "mark.png"
    assert manifest["axes"] == {"epsilon": [0.5, 3.0], "max_error": [1.0]}
    assert len(manifest["variants"]) == 2
    first = manifest["variants"][0]
    assert first["epsilon"] == 0.5 and first["max_error"] == 1.0
    assert first["file"] == "variant-e0_5-m1.svg"
    assert first["svg_bytes"] > 0
    assert isinstance(first["strategies"], dict) and first["elements"] >= 1


def _renderer_available():
    try:
        import resvg_py  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not _renderer_available(), reason="needs resvg-py")
def test_contact_sheet_renders_grid_png():
    eps, mes = (0.5, 3.0), (0.5, 2.5)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    png = V.compose_contact_sheet(variants, epsilons=eps, max_errors=mes)
    assert isinstance(png, (bytes, bytearray)) and len(png) > 0
    img = Image.open(io.BytesIO(bytes(png)))
    assert img.width > 0 and img.height > 0


def test_contact_sheet_none_without_renderer(monkeypatch):
    def boom(*a, **k):
        raise SvgRendererUnavailable("no renderer")
    monkeypatch.setattr(V, "_rasterize", boom)
    eps, mes = (0.5,), (0.5,)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    assert V.compose_contact_sheet(variants, epsilons=eps, max_errors=mes) is None


def test_cli_variants_writes_set(tmp_path):
    src = tmp_path / "mark.png"
    Image.fromarray(_mark()).save(src)
    out = tmp_path / "looks"
    rc = cli_main([str(src), "--variants", "--out-dir", str(out),
                   "--epsilons", "0.5,3", "--max-errors", "1"])
    assert rc == 0
    assert (out / "manifest.json").exists()
    assert (out / "variant-e0_5-m1.svg").exists()
    assert (out / "variant-e3-m1.svg").exists()


def test_cli_variants_rejects_bad_axis(tmp_path):
    src = tmp_path / "mark.png"
    Image.fromarray(_mark()).save(src)
    with pytest.raises(SystemExit):
        cli_main([str(src), "--variants", "--epsilons", "abc"])


def test_axis_line_svg_emits_line():
    frag = _axis_line_svg(AxisLine(10.0, 0.0, 10.0, 40.0), 1.5)
    assert frag.startswith("<line ") and 'x1="10' in frag and "stroke" in frag


def test_inject_axes_inserts_before_closing_svg():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width="40" height="40"><circle/></svg>'
    out = _inject_axes(svg, (AxisLine(20.0, 0.0, 20.0, 40.0),))
    assert "<line " in out
    assert out.rstrip().endswith("</svg>")
    assert out.index("<line ") < out.index("</svg>")


@pytest.mark.skipif(not _renderer_available(), reason="needs resvg-py")
def test_contact_sheet_with_axes_renders():
    eps, mes = (0.5, 3.0), (1.0,)
    variants = generate_variants(_mark(), epsilons=eps, max_errors=mes)
    png = V.compose_contact_sheet(variants, epsilons=eps, max_errors=mes, draw_axes=True)
    assert isinstance(png, (bytes, bytearray)) and len(png) > 0


def test_cli_variants_axes_writes_contact_sheet(tmp_path):
    src = tmp_path / "mark.png"
    Image.fromarray(_mark()).save(src)
    out = tmp_path / "looks"
    rc = cli_main([str(src), "--variants", "--axes", "--out-dir", str(out),
                   "--epsilons", "0.5,3", "--max-errors", "1"])
    assert rc == 0
    assert (out / "manifest.json").exists()
    if _renderer_available():
        assert (out / "contact-sheet.png").exists()


def test_cli_axes_without_variants_is_noted(tmp_path, capsys):
    src = tmp_path / "mark.png"
    Image.fromarray(_mark()).save(src)
    out = tmp_path / "out.svg"
    rc = cli_main([str(src), "--axes", "-o", str(out)])
    assert rc == 0 and out.exists()
    assert "axes" in capsys.readouterr().err.lower()

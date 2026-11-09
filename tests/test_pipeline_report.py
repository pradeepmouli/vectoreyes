import numpy as np
from PIL import Image, ImageDraw

from tests._render import render_svg
from vectoreyes.pipeline import AxisLine, IdealizeReport, Options, idealize


def _disc(n=64):
    im = Image.new("RGB", (n, n), "white")
    ImageDraw.Draw(im).ellipse((10, 10, n - 10, n - 10), fill=(30, 100, 220))
    return np.asarray(im, dtype=np.uint8)


def test_idealize_default_returns_bare_string():
    out = idealize(_disc())
    assert isinstance(out, str) and out.startswith("<svg ")


def test_idealize_report_returns_strategy_counts():
    svg, report = idealize(_disc(), report=True)
    assert isinstance(svg, str) and svg.startswith("<svg ")
    assert isinstance(report, IdealizeReport)
    # a clean disc is one region recognised as a single primitive (circle)
    assert report.strategies == {"primitive": 1}
    assert report.elements == 1
    assert report.gradients == 0


def test_report_empty_for_blank_image():
    blank = np.full((40, 40, 3), 255, dtype=np.uint8)
    svg, report = idealize(blank, report=True)
    assert report.elements == 0 and dict(report.strategies) == {}


def _arch_svg(deg):
    # rectangle + semicircle dome: exactly one mirror axis (vertical at deg=0)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240">'
            f'<g transform="rotate({deg} 120 120)">'
            f'<rect x="75" y="120" width="90" height="80" fill="#36c"/>'
            f'<path d="M75 120 A45 45 0 0 1 165 120 Z" fill="#36c"/></g></svg>')


def test_report_axes_upright_is_vertical():
    svg, report = idealize(_disc(), report=True)
    assert len(report.axes) == 1
    a = report.axes[0]
    assert isinstance(a, AxisLine)
    assert abs(a.x1 - a.x2) < 0.01          # vertical segment
    assert abs(a.x1 - 32) < 2.0             # disc centre (64/2)
    assert a.y2 > a.y1                       # spans a real vertical extent


def test_report_axes_tilted_is_non_vertical():
    arr = render_svg(_arch_svg(45), 240, 240)
    svg, report = idealize(arr, report=True)
    assert "rotate(" in svg                  # took the rectify path
    assert len(report.axes) >= 1
    assert any(abs(a.x1 - a.x2) > 1.0 for a in report.axes)   # at least one tilted line


def test_report_axes_empty_without_symmetry():
    svg, report = idealize(_disc(), options=Options(no_symmetry=True), report=True)
    assert report.axes == ()

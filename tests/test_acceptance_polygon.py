"""Polygon occlusion reconstruction, end-to-end through idealize.

A reconstructed convex polygon emits as a <polygon> whose vertices match the true
(un-occluded) corners. The per-region fallback instead emits a path or a polygon
distorted by the occluder's bite, so a recovered-vertex check distinguishes a real
reconstruction from a fallback. SSIM is a faithful-render sanity floor."""

import re

import numpy as np

from vectoreyes import Options, idealize
from tests._render import disk as _disk, paint as _paint, render_svg, ssim


def _diamond(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (np.abs(xx - cx) + np.abs(yy - cy)) <= r


def _polygon_points(svg):
    """Every <polygon> element's vertices, as a list of (x, y) lists."""
    out = []
    for pstr in re.findall(r'<polygon[^>]*points="([^"]*)"', svg):
        pts = [tuple(float(v) for v in pair.split(",")) for pair in pstr.split()]
        out.append(pts)
    return out


def _corners_match(poly_pts, truth, tol=3.0):
    """True if every truth corner is within `tol` of some recovered vertex."""
    if not poly_pts:
        return False
    return all(min(np.hypot(px - tx, py - ty) for px, py in poly_pts) <= tol
               for tx, ty in truth)


_BLUE, _RED = (51, 102, 204), (204, 51, 51)


def test_two_overlapping_diamonds_reconstruct():
    # Centers 56 px apart (instead of 60) so the blue diamond's bitten corner
    # gives solidity ≈ 0.918 < 0.92, making has_bite() trigger reconstruction.
    h, w = 160, 220
    a = _diamond(80, 80, 46, h, w)
    b = _diamond(136, 80, 46, h, w)
    img = _paint([(a, _BLUE), (b, _RED)], h, w)        # B painted over A
    svg = idealize(img, options=Options())
    polys = _polygon_points(svg)
    assert len(polys) == 2                              # two clean convex polygons
    a_truth = [(80, 34), (126, 80), (80, 126), (34, 80)]
    b_truth = [(136, 34), (182, 80), (136, 126), (90, 80)]
    assert any(_corners_match(p, a_truth) for p in polys)
    assert any(_corners_match(p, b_truth) for p in polys)
    assert ssim(render_svg(svg, w, h), img) >= 0.95


def test_diamond_occluded_by_disk_reconstructs():
    # Disk off the horizontal centerline (cy=60) on purpose: at (135,80) the combined
    # diamond+disk silhouette has a near-90 deg tilt symmetry, so the pipeline runs
    # _idealize_rectified and wraps the SVG body in a <g transform=rotate(...)> — that
    # rotation moves every pixel coord, so the (un-rotated) vertex check would fail.
    # The cx shift (129) also ensures the bite crosses the has_bite solidity threshold.
    h, w = 160, 200
    diamond = _diamond(80, 80, 46, h, w)
    disk = _disk(129, 60, 38, h, w)
    img = _paint([(diamond, _BLUE), (disk, _RED)], h, w)   # disk painted over diamond
    svg = idealize(img, options=Options())
    polys = _polygon_points(svg)
    assert len(polys) == 1                                 # the diamond, recovered
    assert svg.count("<circle") == 1                       # ...and the disk as a circle
    assert _corners_match(polys[0], [(80, 34), (126, 80), (80, 126), (34, 80)])
    assert ssim(render_svg(svg, w, h), img) >= 0.95

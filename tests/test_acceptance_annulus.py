"""Annulus occlusion reconstruction, end-to-end through idealize.

Reconstruction is confirmed by the *clean-annulus fingerprint*: a reconstructed
annulus is emitted as two exact concentric circles — one path with eight cubic
quarter-arcs and no quadratics. The per-region fallback fit instead contains free
quadratics (`Q`), so the fingerprint distinguishes a real reconstruction from a
fallback. SSIM is only a faithful-render sanity floor (thin concentric rings are
inherently harsh on SSIM, so the floor is loose for the busier multi-ring scenes)."""

import re

import numpy as np

from vectoreyes import Options, idealize
from tests._render import disk as _disk, paint as _paint, render_svg, ssim


def _ring(cx, cy, r_out, r_in, h, w):
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 <= r_out ** 2) & (d2 >= r_in ** 2)


def _annulus_paths(svg):
    """Count reconstructed-annulus paths: two exact circles == 8 cubic arcs, two
    move commands, and no free quadratics (a fallback fit would carry `Q`)."""
    return sum(1 for d in re.findall(r'\sd="([^"]*)"', svg)
               if d.count("C") == 8 and d.count("M") == 2 and "Q" not in d)


_BLUE, _YELLOW, _GREEN, _RED = (51, 102, 204), (240, 200, 20), (20, 160, 60), (204, 51, 51)


def test_ring_plus_disk_reconstructs_and_renders():
    h, w = 160, 200
    img = _paint([(_ring(70, 70, 45, 25, h, w), _BLUE),
                  (_disk(135, 70, 38, h, w), _RED)], h, w)
    svg = idealize(img, options=Options())
    assert _annulus_paths(svg) == 1        # the ring recovered as a clean annulus
    assert svg.count("<circle") == 1       # ...and the disk as a circle
    assert ssim(render_svg(svg, w, h), img) >= 0.95


def test_two_overlapping_rings_reconstruct():
    h, w = 160, 220
    img = _paint([(_ring(72, 80, 42, 24, h, w), _BLUE),
                  (_ring(148, 80, 42, 24, h, w), _YELLOW)], h, w)
    svg = idealize(img, options=Options())
    assert _annulus_paths(svg) == 2
    assert ssim(render_svg(svg, w, h), img) >= 0.95


def test_three_ring_consistent_stack_reconstructs():
    # painter's order left-to-right gives a clean acyclic z-order -> three annuli.
    # Three thin concentric bands are SSIM-harsh, so the fingerprint (==3) is the real
    # reconstruction proof; SSIM is a looser sanity floor here.
    h, w = 160, 220
    img = _paint([(_ring(46, 80, 38, 22, h, w), _BLUE),
                  (_ring(110, 80, 38, 22, h, w), _YELLOW),
                  (_ring(174, 80, 38, 22, h, w), _GREEN)], h, w)
    svg = idealize(img, options=Options())
    assert _annulus_paths(svg) == 3
    assert ssim(render_svg(svg, w, h), img) >= 0.93


def test_flattened_annulus_keeps_its_hole():
    # --flatten bakes the annulus to a path; its two same-winding circles only cut a
    # hole under even-odd fill, so the flattened path must carry fill-rule="evenodd".
    h, w = 160, 200
    img = _paint([(_ring(70, 70, 45, 25, h, w), _BLUE),
                  (_disk(135, 70, 38, h, w), _RED)], h, w)
    svg = idealize(img, options=Options(flatten=True))
    out = render_svg(svg, w, h)
    assert tuple(int(v) for v in out[70, 70]) != _BLUE   # centre is the hole, not filled

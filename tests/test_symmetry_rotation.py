"""Any-axis (rectify-to-vertical) symmetry detection."""

from __future__ import annotations

import numpy as np

from vectoreyes.pipeline import Options, idealize
from vectoreyes.symmetry import _axis_mismatch, detect_symmetry_rotation

from tests._render import render_svg, ssim


def _render(svg: str, n: int = 240) -> np.ndarray:
    return render_svg(svg, n, n)


def _silhouette(svg: str, n: int = 240) -> np.ndarray:
    return np.any(_render(svg, n) < 250, axis=2)


def _arch(deg: float) -> str:
    # rectangle + semicircle dome: exactly one mirror axis (vertical when deg=0)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240">'
        f'<g transform="rotate({deg} 120 120)">'
        f'<rect x="75" y="120" width="90" height="80" fill="#36c"/>'
        f'<path d="M75 120 A45 45 0 0 1 165 120 Z" fill="#36c"/></g></svg>'
    )


def _flag(deg: float) -> str:
    # lopsided pentagon: no mirror axis at any rotation
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240">'
        f'<g transform="rotate({deg} 120 120)">'
        f'<path d="M60 60 L180 80 L150 130 L180 170 L60 180 Z" fill="#c33"/></g></svg>'
    )


def _offshape_frac(sil: np.ndarray, rho: float) -> float:
    """Resampling-free check that `rho` names a real symmetry axis: reflect the
    foreground across the line at angle (rho + 90)° through the centroid and
    measure the fraction landing off the shape."""
    from scipy import ndimage as ndi

    ys, xs = np.nonzero(sil)
    cx, cy = float(xs.mean()), float(ys.mean())
    theta = np.radians(-rho - 90.0)             # invert rho = -(φ + 90)
    dist = ndi.distance_transform_edt(~sil)
    return _axis_mismatch((xs.astype(float), ys.astype(float)), cx, cy, theta, dist)


def test_recovers_tilt_of_symmetric_shape():
    for deg in (30.0, -40.0, 45.0, 60.0):
        sil = _silhouette(_arch(deg))
        rho = detect_symmetry_rotation(sil)
        assert rho is not None, f"missed symmetry at {deg}°"
        # the recovered axis must actually be a mirror line of the shape
        assert _offshape_frac(sil, rho) <= 0.10


def test_upright_symmetric_recovered_too():
    sil = _silhouette(_arch(0.0))
    rho = detect_symmetry_rotation(sil)
    assert rho is not None
    assert _offshape_frac(sil, rho) <= 0.10


def test_rejects_asymmetric_shape():
    for deg in (0.0, 25.0, -25.0):
        assert detect_symmetry_rotation(_silhouette(_flag(deg))) is None


def test_rejects_degenerate_tiny_input():
    assert detect_symmetry_rotation(np.zeros((10, 10), dtype=bool)) is None


# --- acceptance: full idealize round-trip ---

def test_idealize_rectifies_tilted_mark():
    for deg in (30.0, 45.0, -40.0):
        arr = _render(_arch(deg))
        svg = idealize(arr)
        assert "rotate(" in svg, f"no rectification at {deg}°"
        # the wrapped output must still reproduce the input raster
        assert ssim(render_svg(svg, 240, 240), arr) >= 0.95


def test_idealize_leaves_upright_mark_unrotated():
    # an already-vertical mark goes through the existing vertical path: no wrapper
    svg = idealize(_render(_arch(0.0)))
    assert "rotate(" not in svg


def test_no_symmetry_disables_rectification():
    svg = idealize(_render(_arch(45.0)), options=Options(no_symmetry=True))
    assert "rotate(" not in svg


def test_flatten_bakes_rotation_into_coordinates():
    # flatten discards the <use> symmetry, so the rectifying rotation is vestigial:
    # it must be baked into the path coordinates, leaving no transform at all.
    arr = _render(_arch(45.0))
    svg = idealize(arr, options=Options(flatten=True))
    assert "transform" not in svg and "rotate(" not in svg
    assert ssim(render_svg(svg, 240, 240), arr) >= 0.95


def test_rectified_wrapper_floats_are_rounded():
    # the non-flatten wrapper keeps the rotation, but its angle must be tidy
    import re

    svg = idealize(_render(_arch(45.0)))
    angle = re.search(r"rotate\(([-\d.]+)\)", svg)
    assert angle is not None
    frac = angle.group(1).split(".")[1] if "." in angle.group(1) else ""
    assert len(frac) <= 3, f"un-rounded angle {angle.group(1)}"

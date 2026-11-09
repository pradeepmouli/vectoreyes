import numpy as np

from vectoreyes.contour import region_contours
from vectoreyes.emit import path_svg, render_svg_doc
from vectoreyes.refine import symmetric_fit
from tests._render import render_svg, ssim


def _fold_ssim(img, axis_x):
    w = img.shape[1]
    k = min(axis_x, w - axis_x)
    return float(ssim(img[:, axis_x - k:axis_x][:, ::-1], img[:, axis_x:axis_x + k]))


def test_symmetric_fit_dome_is_exactly_symmetric():
    H, W = 70, 80
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 2025) <= 1) & (yy <= 55)  # half-ellipse
    cs = region_contours(dome)
    sh = symmetric_fit(cs[0], 40.0, epsilon=1.5, max_error=1.0)
    assert sh is not None and sh.kind == "path"
    img = render_svg(render_svg_doc(W, H, [path_svg(sh.params["d"], "#062336")]), W, H)
    assert _fold_ssim(img, 40) >= 0.999          # exactly symmetric about the axis


def _tapered_band(H=60, W=120):
    m = np.zeros((H, W), bool)
    for y in range(12, 48):
        hw = int(46 - (y - 12) * 0.2)     # straight taper
        m[y, 60 - hw:60 + hw] = True
    return m


def test_rounded_trapezoid_fit_symmetric_and_rejects_curves():
    from vectoreyes.refine import rounded_trapezoid_fit
    c = region_contours(_tapered_band())[0]
    sh = rounded_trapezoid_fit(c, 60.0, radius=8.0, max_error=1.5)
    assert sh is not None and sh.kind == "path"
    img = render_svg(render_svg_doc(120, 60, [path_svg(sh.params["d"], "#000")]), 120, 60)
    assert _fold_ssim(img, 60) >= 0.999            # exactly symmetric

    # a curved half-ellipse is NOT a trapezoid -> None (falls through to symmetric_fit)
    H, W = 70, 80
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 2025) <= 1) & (yy <= 55)
    assert rounded_trapezoid_fit(region_contours(dome)[0], 40.0, radius=8.0, max_error=1.0) is None


def test_half_ellipse_cap_fit_is_convex_and_symmetric():
    from vectoreyes.refine import half_ellipse_cap_fit
    H, W = 70, 80
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 2025) <= 1) & (yy <= 55)
    sh = half_ellipse_cap_fit(region_contours(dome)[0], 40.0, max_error=1.0)
    assert sh is not None and sh.kind == "path"
    d = sh.params["d"]
    # a flat-based half-ellipse: exactly two convex kappa quarter-arcs, no quadratic wobble
    assert d.count("C") == 2 and "Q" not in d
    img = render_svg(render_svg_doc(W, H, [path_svg(d, "#000")]), W, H)
    assert _fold_ssim(img, 40) >= 0.999            # exactly symmetric


def test_half_ellipse_cap_fit_rounds_base_corners_when_asked():
    from vectoreyes.refine import half_ellipse_cap_fit
    H, W = 80, 80
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 2025) <= 1) & (yy <= 55)
    c = region_contours(dome)[0]
    sharp = half_ellipse_cap_fit(c, 40.0, max_error=1.0)
    rounded = half_ellipse_cap_fit(c, 40.0, corner_radius=10.0, max_error=1.0)
    assert sharp.params["d"].count("C") == 2          # dome only, sharp base
    assert rounded.params["d"].count("C") == 4        # dome + two base fillets
    img = render_svg(render_svg_doc(W, H, [path_svg(rounded.params["d"], "#000")]), W, H)
    assert _fold_ssim(img, 40) >= 0.999               # still exactly symmetric


def test_symmetric_fit_rounds_sharp_corner_with_shared_radius():
    """A flat-topped, curved-sided 'cone' gets its top corners filleted by the
    shared radius (a cubic fillet appears), and stays exactly symmetric."""
    from vectoreyes.refine import symmetric_fit
    H, W = 90, 120
    m = np.zeros((H, W), bool)
    for y in range(15, 78):                            # flat top, sides curving to a point
        t = (y - 15) / 63
        hw = int(40 * (1 - t * t))
        m[y, 60 - hw:60 + hw + 1] = True
    c = region_contours(m)[0]
    sharp = symmetric_fit(c, 60.0, epsilon=1.5, max_error=1.0)
    rounded = symmetric_fit(c, 60.0, corner_radius=10.0, epsilon=1.5, max_error=1.0)
    assert rounded.params["d"].count("C") > sharp.params["d"].count("C")  # fillet(s) added
    img = render_svg(render_svg_doc(W, H, [path_svg(rounded.params["d"], "#000")]), W, H)
    assert _fold_ssim(img, 60) >= 0.999


def test_half_ellipse_cap_fit_rejects_pointed_tip():
    """A teardrop/point has no wide flat base -> not a cap; falls through to None."""
    from vectoreyes.refine import half_ellipse_cap_fit
    H, W = 80, 80
    m = np.zeros((H, W), bool)
    for y in range(10, 70):                         # triangle narrowing to a point at top
        hw = int((y - 10) * 0.5)
        m[y, 40 - hw:40 + hw + 1] = True
    assert half_ellipse_cap_fit(region_contours(m)[0], 40.0, max_error=1.0) is None


def test_pipeline_output_is_inflection_free():
    """End-to-end: a symmetric mark idealizes with only convex arcs — every free
    curved run is a quadratic (Q); the only cubics (C) are parametric quarter-arcs."""
    from vectoreyes.pipeline import idealize, Options
    H, W = 90, 120
    img = np.full((H, W, 3), 255, np.uint8)
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 60) ** 2 / 1600 + (yy - 70) ** 2 / 3600) <= 1) & (yy <= 70)
    img[dome] = (20, 40, 60)
    svg = idealize(img, options=Options())
    import re
    for d in re.findall(r'd="([^"]+)"', svg):
        # every free curved run is a quadratic; the only cubics are parametric
        # quarter-arcs, which always come in even counts (a cap is two) — so no
        # lone free cubic that could carry an inflection
        assert "Q" in d or d.count("C") % 2 == 0


def test_symmetric_polygon_fit_keeps_diamond_sharp():
    """A straddling diamond stays a straight-edged, sharp-cornered, exactly
    symmetric polygon — not curve-fit into a petal."""
    from vectoreyes.refine import symmetric_polygon_fit
    H, W = 120, 100
    yy, xx = np.ogrid[:H, :W]
    diamond = (np.abs(xx - 50) / 30 + np.abs(yy - 60) / 50) <= 1
    sh = symmetric_polygon_fit(region_contours(diamond)[0], 50.0, epsilon=1.5)
    assert sh is not None and sh.kind == "path"
    assert "C" not in sh.params["d"] and "Q" not in sh.params["d"]   # straight edges only
    img = render_svg(render_svg_doc(W, H, [path_svg(sh.params["d"], "#000")]), W, H)
    assert _fold_ssim(img, 50) >= 0.999                              # exactly symmetric


def test_symmetric_polygon_fit_rejects_curved_region():
    """A dome is curved, not polygonal -> None, so it falls through to the curve fits."""
    from vectoreyes.refine import symmetric_polygon_fit
    H, W = 70, 80
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 2025) <= 1) & (yy <= 55)
    assert symmetric_polygon_fit(region_contours(dome)[0], 40.0, epsilon=1.5) is None


def test_symmetric_fit_beats_raw_path_symmetry():
    from vectoreyes.fit import fit_path
    H, W = 70, 80
    yy, xx = np.ogrid[:H, :W]
    dome = (((xx - 40) ** 2 / 900 + (yy - 55) ** 2 / 2025) <= 1) & (yy <= 55)
    c = region_contours(dome)[0]
    sym_shape = symmetric_fit(c, 40.0, epsilon=1.5, max_error=1.0)
    assert sym_shape is not None
    sym = render_svg(render_svg_doc(W, H, [path_svg(sym_shape.params["d"], "#000")]), W, H)
    raw = render_svg(render_svg_doc(W, H, [path_svg(fit_path(c, epsilon=1.5, max_error=1.0).params["d"], "#000")]), W, H)
    assert _fold_ssim(sym, 40) > _fold_ssim(raw, 40)

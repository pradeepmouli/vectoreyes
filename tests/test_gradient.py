# SPDX-License-Identifier: MIT
import numpy as np


def _OKLAB(img_uint8):
    from vectoreyes.color import srgb_to_oklab
    return srgb_to_oklab(img_uint8.reshape(-1, 3) / 255.0)


def test_mean_delta_e_zero_for_identical_and_positive_for_different():
    from vectoreyes.color import mean_delta_e
    a = np.full((4, 4, 3), 120, np.uint8)
    assert mean_delta_e(a, a) == 0.0
    b = a.copy()
    b[:, :, 0] = 200          # shift red channel
    assert mean_delta_e(a, b) > 0.02


def test_linear_gradient_def_emits_stops_and_coords():
    from vectoreyes.emit import linear_gradient_def
    d = linear_gradient_def("g0", 10, 20, 110, 20, [(0.0, "#ff0000"), (1.0, "#0000ff")])
    assert d.startswith("<linearGradient") and 'id="g0"' in d
    assert 'gradientUnits="userSpaceOnUse"' in d
    assert 'x1="10"' in d and 'x2="110"' in d
    assert d.count("<stop") == 2
    assert 'offset="0"' in d and 'stop-color="#ff0000"' in d
    assert 'offset="1"' in d and 'stop-color="#0000ff"' in d


def test_radial_gradient_def_emits_center_radius():
    from vectoreyes.emit import radial_gradient_def
    d = radial_gradient_def("g1", 50, 60, 40, [(0.0, "#ffffff"), (1.0, "#000000")])
    assert d.startswith("<radialGradient") and 'id="g1"' in d
    assert 'cx="50"' in d and 'cy="60"' in d and 'r="40"' in d
    assert d.count("<stop") == 2


def test_render_svg_doc_wraps_defs():
    from vectoreyes.emit import render_svg_doc
    out = render_svg_doc(100, 100, ['<rect/>'], defs=['<linearGradient id="g0"></linearGradient>'])
    assert "<defs>" in out and "</defs>" in out
    assert out.index("<defs>") < out.index("<rect/>")     # defs before body
    out2 = render_svg_doc(100, 100, ['<rect/>'])
    assert "<defs>" not in out2                            # no defs block when none given


def _linear_gradient_image(h, w, p0, p1, stops_rgb):
    """Render a ground-truth linear gradient (for fitting against)."""
    yy, xx = np.mgrid[:h, :w]
    d = np.array(p1, float) - np.array(p0, float)
    L = np.dot(d, d)
    t = (((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb])
    cols = np.array([c for _, c in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    return img.round().astype(np.uint8)


def test_fit_linear_recovers_axis_and_endpoints():
    from vectoreyes.gradient import _fit_linear
    h, w = 80, 120
    p0, p1 = (10, 40), (110, 40)                       # horizontal axis
    img = _linear_gradient_image(h, w, p0, p1, [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    ys, xs = np.mgrid[:h, :w]
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    oklab = _OKLAB(img)
    model = _fit_linear(pts, oklab, img.reshape(-1, 3))
    assert model is not None and model["kind"] == "linear"
    g = model["geometry"]
    # axis is ~horizontal: endpoints span x, ~constant y
    assert abs(g["y1"] - g["y2"]) < 3.0
    # span tolerance is loose: the test gradient fills the whole 120px canvas, so the
    # fitted axis spans all pixels (~119px) not the 100px p0->p1 range; the DIRECTION
    # assertion above (|y1-y2|) is the real check. (In the pipeline the footprint mask
    # bounds the span.)
    assert abs(abs(g["x2"] - g["x1"]) - 100) < 25
    assert len(model["stops"]) >= 2


def test_reduce_stops_drops_redundant_midpoints():
    from vectoreyes.gradient import _reduce_stops
    # an sRGB gray ramp: some interior stops fall within max_delta_e of the
    # piecewise-linear (OKLab) reconstruction from their neighbors, so they get dropped
    stops = [(0.0, "#000000"), (0.25, "#404040"), (0.5, "#808080"),
             (0.75, "#bfbfbf"), (1.0, "#ffffff")]
    reduced = _reduce_stops(stops, max_delta_e=0.02)
    assert len(reduced) < len(stops) and reduced[0][0] == 0.0 and reduced[-1][0] == 1.0


def _radial_gradient_image(h, w, c, r, stops_rgb):
    yy, xx = np.mgrid[:h, :w]
    t = (np.hypot(xx - c[0], yy - c[1]) / r).clip(0, 1)
    offs = np.array([o for o, _ in stops_rgb]); cols = np.array([col for _, col in stops_rgb], float)
    img = np.empty((h, w, 3))
    for ch in range(3):
        img[:, :, ch] = np.interp(t, offs, cols[:, ch])
    return img.round().astype(np.uint8)


def test_fit_radial_recovers_center():
    from vectoreyes.gradient import _fit_radial
    h, w = 120, 120
    c, r = (60, 60), 50
    img = _radial_gradient_image(h, w, c, r, [(0.0, (125, 211, 252)), (1.0, (29, 78, 216))])
    ys, xs = np.mgrid[:h, :w]
    # restrict to the disc so background doesn't dominate the fit
    inside = np.hypot(xs - c[0], ys - c[1]) <= r
    pts = np.column_stack([xs[inside], ys[inside]]).astype(float)
    oklab = _OKLAB(img[inside])
    model = _fit_radial(pts, oklab, img[inside].reshape(-1, 3))
    assert model is not None and model["kind"] == "radial"
    g = model["geometry"]
    assert abs(g["cx"] - 60) < 6 and abs(g["cy"] - 60) < 6
    assert g["r"] > 30


def test_fit_gradient_accepts_linear_rejects_flat():
    from vectoreyes.gradient import fit_gradient
    h, w = 80, 120
    img = _linear_gradient_image(h, w, (10, 40), (110, 40),
                                 [(0.0, (37, 99, 235)), (1.0, (219, 39, 119))])
    mask = np.ones((h, w), bool)
    model = fit_gradient(mask, img)
    assert model is not None and model["kind"] == "linear"
    flat = np.full((h, w, 3), (37, 99, 235), np.uint8)
    assert fit_gradient(mask, flat) is None             # flat -> no gradient


def test_dominant_blob_fraction():
    from vectoreyes.gradient import _dominant_blob_fraction
    m = np.zeros((20, 40), bool)
    m[2:18, 2:18] = True                       # one 16x16 blob, rest empty
    assert _dominant_blob_fraction(m) == 1.0
    m[2:18, 22:38] = True                      # add a second, equal, disconnected blob
    assert abs(_dominant_blob_fraction(m) - 0.5) < 1e-9
    assert _dominant_blob_fraction(np.zeros((5, 5), bool)) == 0.0   # empty -> 0


def test_fit_gradient_rejects_near_flat_region():
    # a near-flat red region with only a faint ~2-level tint (like a real flat logo's
    # antialiasing/compression noise): its fitted stops barely travel, so it must NOT be
    # emitted as a near-constant gradient (regression for Pinterest/Vimeo over-detection).
    from vectoreyes.gradient import fit_gradient
    h, w = 60, 60
    img = np.empty((h, w, 3))
    t = np.linspace(0.0, 1.0, w)
    for ch, (a, b) in enumerate(((189, 191), (8, 10), (28, 30))):   # #BD081C +/- ~2 levels
        img[:, :, ch] = a + t * (b - a)
    img = img.round().astype(np.uint8)
    assert fit_gradient(np.ones((h, w), bool), img) is None         # span below _MIN_STOP_SPAN


def test_fit_gradient_accepts_traveling_gradient():
    from vectoreyes.gradient import fit_gradient
    h, w = 60, 120
    img = np.empty((h, w, 3))
    t = np.linspace(0.0, 1.0, w)
    for ch, (a, b) in enumerate(((37, 219), (99, 39), (235, 119))):  # blue -> magenta
        img[:, :, ch] = a + t * (b - a)
    img = img.round().astype(np.uint8)
    model = fit_gradient(np.ones((h, w), bool), img)
    assert model is not None and model["kind"] == "linear"          # real travel: still fires


def test_best_parametric_searched_beats_heuristic_on_offcenter_radial():
    # a radial gradient whose centre is in a corner (where the principal-axis-extreme
    # heuristic lands poorly); the searched fit must find a low-mean-ΔE radial model.
    from vectoreyes.gradient import _best_parametric
    h, w = 80, 80
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(xx - 5, yy - 5) / np.hypot(w, h)        # centre near (5,5) corner
    img = np.empty((h, w, 3))
    for ch, (a, b) in enumerate(((230, 30), (120, 60), (40, 210))):
        img[:, :, ch] = (a + r * (b - a))
    img = img.round().astype(np.uint8)
    out = _best_parametric(np.ones((h, w), bool), img)
    assert out is not None
    model, mean_de, median_de = out
    assert model["kind"] in ("radial", "linear")
    assert mean_de < 0.05 and median_de < 0.05          # a real gradient fits tightly


def test_best_parametric_returns_none_for_flat():
    from vectoreyes.gradient import _best_parametric
    img = np.full((40, 40, 3), (50, 100, 150), np.uint8)
    assert _best_parametric(np.ones((40, 40), bool), img) is None   # span below minimum


def _diagonal_2d_field(h, w):
    """A separable 2-D field (horizontal hue x vertical luminance) that NO single
    linear/radial gradient fits: hue runs left->right, brightness runs top->bottom."""
    yy, xx = np.mgrid[:h, :w]
    tx = xx / (w - 1)
    ty = yy / (h - 1)
    img = np.empty((h, w, 3))
    img[:, :, 0] = 30 + tx * 200                       # R climbs with x
    img[:, :, 1] = 20 + ty * 200                       # G climbs with y
    img[:, :, 2] = 200 - tx * 160                      # B falls with x
    return img.round().astype(np.uint8)


def test_fit_stretch_returns_raster_model_under_target():
    from vectoreyes.gradient import _fit_stretch, _STRETCH_TARGET
    img = _diagonal_2d_field(96, 96)
    model = _fit_stretch(np.ones((96, 96), bool), img)
    assert model is not None and model["kind"] == "raster"
    g = model["geometry"]
    assert (g["x"], g["y"], g["w"], g["h"]) == (0.0, 0.0, 96.0, 96.0)
    assert isinstance(model["png_b64"], str) and len(model["png_b64"]) > 0


def test_fit_stretch_none_for_degenerate_bbox():
    from vectoreyes.gradient import _fit_stretch
    m = np.zeros((40, 40), bool)
    m[10, 5:9] = True                                  # 1px tall footprint
    assert _fit_stretch(m, np.zeros((40, 40, 3), np.uint8)) is None


def test_stop_span_registers_cyclic_travel():
    # a cyclic stop set (returns to its starting colour) must register travel, not ~0.
    from vectoreyes.gradient import _stop_span, _MIN_STOP_SPAN
    cyclic = [(0.0, "#ff0000"), (0.5, "#0000ff"), (1.0, "#ff0000")]   # red -> blue -> red
    assert _stop_span(cyclic) > _MIN_STOP_SPAN
    # a monotone ramp is unchanged (max-from-first == last-from-first)
    mono = [(0.0, "#ff0000"), (0.5, "#7f007f"), (1.0, "#0000ff")]
    # this triple is ~collinear in OKLab, so max-from-first == last-from-first; for a
    # curved/cyclic stop path the new span is >= the old, never less (so the gate only
    # admits MORE fields, never fewer).
    assert abs(_stop_span(mono) - _stop_span([mono[0], mono[-1]])) < 1e-9


def test_fit_stretch_masks_outside_pixels():
    # a smooth footprint whose bbox also contains a contrasting OUTSIDE-mask patch (a hole):
    # the bright hole colour must NOT bleed into the downsampled fill (it is replaced by the
    # footprint mean before downsampling).
    from vectoreyes.gradient import _fit_stretch
    import base64, io
    from PIL import Image
    h, w = 64, 64
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = np.clip(30 + xx * 3, 0, 255)        # smooth blue->red-ish horizontal field
    img[:, :, 2] = np.clip(200 - xx * 3, 0, 255)
    mask = np.ones((h, w), bool); mask[24:40, 24:40] = False   # a hole
    img[~mask] = (0, 255, 0)                            # bright green in the hole (outside-mask)
    model = _fit_stretch(mask, img)
    assert model is not None and model["kind"] == "raster"
    small = np.asarray(Image.open(io.BytesIO(base64.b64decode(model["png_b64"]))).convert("RGB"))
    # the green hole colour must be absent from the downsampled fill (it was masked out)
    assert float(np.linalg.norm(small.reshape(-1, 3).astype(float) - np.array([0, 255, 0]), axis=1).min()) > 60.0

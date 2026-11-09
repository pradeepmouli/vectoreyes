import numpy as np
from PIL import Image
from vectoreyes.candidate import FlatFill, LinearGradientFill
from vectoreyes.fit import Shape
from vectoreyes.refinement.region_sample import RegionSample
from vectoreyes.pipeline import Options, _render_refinement_body, _segment_image, idealize
from tests._render import render_svg, ssim


def test_segment_image_honors_absolute_min_region_size():
    image = np.full((12, 12, 3), 255, dtype=np.uint8)
    image[3:5, 3:5] = (20, 80, 220)  # four-pixel component

    _width, _height, retained = _segment_image(
        image,
        Options(max_sampling_colors=2, min_region_size=4, min_region_fraction=0),
    )
    _width, _height, filtered = _segment_image(
        image,
        Options(max_sampling_colors=2, min_region_size=5, min_region_fraction=0),
    )

    assert len(retained) == 1
    assert filtered == []


def _path_points(d):
    import re

    toks = re.findall(r"[MLCQZ]|-?\d*\.?\d+", d)
    counts = {"M": 2, "L": 2, "C": 6, "Q": 4, "Z": 0}
    pts, i = [], 0
    while i < len(toks):
        cmd = toks[i]
        i += 1
        n = counts[cmd]
        nums = [float(t) for t in toks[i:i + n]]
        i += n
        pts += [(nums[k], nums[k + 1]) for k in range(0, n, 2)]
    return pts


def test_symmetric_holed_region_emits_exactly_symmetric():
    # a centered annulus (filled disk with a centered hole) is a self-symmetric
    # holed straddler; its idealized geometry should be EXACTLY bilaterally
    # symmetric (built from mirrored halves), not independently-fit contours.
    import re

    h = w = 120
    yy, xx = np.ogrid[:h, :w]
    r2 = (xx - 59.5) ** 2 + (yy - 59.5) ** 2
    mask = (r2 <= 46 ** 2) & (r2 >= 22 ** 2)
    img = np.full((h, w, 3), 255, np.uint8)
    img[mask] = (20, 40, 80)
    svg = idealize(img, options=Options())
    assert "evenodd" in svg                                   # the hole survives

    # every control point's mirror about the axis is also a control point —
    # render-independent proof the geometry is exactly symmetric
    d = re.search(r'\sd="([^"]*)"', svg).group(1)   # \s avoids matching id="s0"
    pts = _path_points(d)
    ax = sum(x for x, _ in pts) / len(pts)
    for x, y in pts:
        assert any(abs(mx - (2 * ax - x)) < 0.6 and abs(my - y) < 0.6 for mx, my in pts)

    # and it still renders recognizably to the source (a thin ring is harsh on
    # SSIM, so this is a sanity floor, not a fidelity assertion)
    assert ssim(render_svg(svg, w, h), img) >= 0.90


def _evenodd_source(obj_id: int, *, fill: str = "#111111", z: int = 0) -> RegionSample:
    return RegionSample(
        obj_id,
        Shape(
            "path",
            {
                "d": "M0 0 L20 0 L20 20 L0 20 Z M5 5 L15 5 L15 15 L5 15 Z",
                "fill_rule": "evenodd",
            },
        ),
        FlatFill(fill),
        z,
    )


def test_refinement_body_preserves_evenodd_when_flattening_use():
    source = _evenodd_source(10)
    clone = RegionSample(
        20,
        Shape("use", {"href_obj_id": 10, "transform": (1.0, 0.0, 0.0, 1.0, 24.0, 0.0)}),
        FlatFill("#111111"),
        0,
        footprint=source.footprint,
    )

    body, _defs = _render_refinement_body([clone, source], flatten=True)

    assert body[0].count('fill-rule="evenodd"') == 1
    assert body[1].count('fill-rule="evenodd"') == 1


def test_refinement_body_preserves_evenodd_when_inlining_recolored_use():
    source = _evenodd_source(10, fill="#111111")
    clone = RegionSample(
        20,
        Shape(
            "use",
            {
                "href_obj_id": 10,
                "transform": (1.0, 0.0, 0.0, 1.0, 24.0, 0.0),
                "fill": "#222222",
            },
        ),
        FlatFill("#222222"),
        0,
        footprint=source.footprint,
    )

    body, _defs = _render_refinement_body([clone, source])

    assert body[0].startswith('<path id="s10"')
    assert 'fill="#111111"' in body[0]
    assert body[1].startswith('<path fill="#222222"')
    assert 'fill-rule="evenodd"' in body[1]


def test_refinement_body_inlines_gradient_use_to_preserve_global_paint_space():
    fill = LinearGradientFill(
        {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 40.0},
        [(0.0, "#ff0000"), (1.0, "#0000ff")],
    )
    source = RegionSample(
        10,
        Shape("path", {"d": "M0 20 L20 20 L20 40 L0 40 Z"}),
        fill,
        0,
    )
    mirror = RegionSample(
        20,
        Shape("use", {"href_obj_id": 10, "transform": (1.0, 0.0, 0.0, -1.0, 0.0, 40.0)}),
        fill,
        0.1,
        footprint=source.footprint,
    )

    body, defs = _render_refinement_body([source, mirror])
    svg_body = "".join(body)

    assert len(defs) == 1
    assert "<use" not in svg_body
    assert svg_body.count("<path") == 2
    assert svg_body.count('fill="url(#g0)"') == 2


def test_refinement_body_inlines_same_fill_use_to_keep_output_concrete():
    fill = FlatFill("#111111")
    source = RegionSample(
        10,
        Shape("path", {"d": "M0 0 L20 0 L20 20 L0 20 Z"}),
        fill,
        0,
    )
    clone = RegionSample(
        20,
        Shape("use", {"href_obj_id": 10, "transform": (1.0, 0.0, 0.0, 1.0, 24.0, 0.0)}),
        fill,
        0.1,
        footprint=source.footprint,
    )

    body, _defs = _render_refinement_body([source, clone])

    assert body[0].startswith('<path id="s10"')
    assert body[1].startswith('<path fill="#111111"')
    assert "<use" not in "".join(body)


def test_refinement_body_renders_by_z_then_id():
    low_source = RegionSample(
        10,
        Shape("rect", {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}),
        FlatFill("#111111"),
        0,
    )
    low_use = RegionSample(
        99,
        Shape("use", {"href_obj_id": 10, "transform": (1.0, 0.0, 0.0, 1.0, 12.0, 0.0)}),
        FlatFill("#111111"),
        0,
        footprint=low_source.footprint,
    )
    high_cover = RegionSample(
        2,
        Shape("rect", {"x": 0.0, "y": 0.0, "w": 30.0, "h": 30.0}),
        FlatFill("#333333"),
        1,
    )

    body, _defs = _render_refinement_body([high_cover, low_use, low_source])

    assert 'fill="#111111"' in body[0]
    assert body[1].startswith('<path')
    assert 'fill="#111111"' in body[1]
    assert 'fill="#333333"' in body[2]


def test_area_filter_is_scale_independent():
    # a big block + a small-but-intentional block; padding the canvas with more
    # background must NOT drop the small block (the old canvas-fraction filter did,
    # because its threshold grew with the canvas — it should track the mark).
    # same colour for both (so palette quantization keeps it regardless of how
    # rare the small block becomes); disconnected, so they are two components.
    def scene(pad):
        a = np.full((90 + pad, 90 + pad, 3), 255, np.uint8)
        a[10:60, 10:60] = (0, 0, 0)       # 2500-px block
        a[10:20, 70:80] = (0, 0, 0)       # 100-px block (4% of the largest)
        return a

    tight = _segment_image(scene(0), Options())[2]
    padded = _segment_image(scene(400), Options())[2]   # 4× more background
    assert len(tight) == 2
    assert len(padded) == 2, "small region dropped only because the canvas grew"


def _two_band_logo(path):
    img = np.full((60, 80, 3), 255, np.uint8)
    img[8:26, 12:68] = (6, 35, 54)      # navy rect
    img[34:52, 20:60] = (61, 168, 157)  # teal rect
    Image.fromarray(img).save(path)


def test_idealize_emits_two_rects(tmp_path):
    p = tmp_path / "logo.png"
    _two_band_logo(p)
    svg = idealize(str(p))
    assert svg.count("<rect") == 2
    assert "#062336" in svg and "#3DA89D" in svg
    assert 'viewBox="0 0 80 60"' in svg


def test_solid_color_image_does_not_crash(tmp_path):
    from PIL import Image
    Image.fromarray(np.full((24, 24, 3), (40, 40, 40), np.uint8)).save(tmp_path / "solid.png")
    svg = idealize(str(tmp_path / "solid.png"))
    assert svg.startswith("<svg") and svg.strip().endswith("</svg>")

def test_gradient_image_does_not_crash():
    grad = np.zeros((40, 40, 3), np.uint8)
    for x in range(40):
        grad[:, x] = (x * 6, 100, 255 - x * 6)
    svg = idealize(grad)
    assert svg.startswith("<svg")
    assert "<linearGradient" in svg or "<radialGradient" in svg


def test_region_with_hole_uses_evenodd_and_renders_hole():
    from tests._render import render_svg
    img = np.full((60, 60, 3), 255, np.uint8)
    img[10:50, 10:50] = (6, 35, 54)      # navy square
    img[24:36, 24:36] = (255, 255, 255)  # white counter (hole)
    svg = idealize(img)
    assert 'fill-rule="evenodd"' in svg
    out = render_svg(svg, 60, 60)
    assert out[30, 30].min() > 200       # hole renders as background (light)
    assert int(out[15, 30].sum()) < 200  # frame renders navy (dark)


def test_flatten_emits_only_paths():
    img = np.full((60, 80, 3), 255, np.uint8)
    img[8:26, 12:68] = (6, 35, 54)
    img[34:52, 20:60] = (61, 168, 157)
    flat = idealize(img, options=Options(flatten=True))
    assert "<path" in flat
    for elem in ("<rect", "<ellipse", "<circle", "<polygon", "<use"):
        assert elem not in flat


def _rounded_band_img(H=90, W=120, cx=60, halfw=34, top=18, bot=72, rad=14):
    """A vertical band with genuinely rounded (antialiased) corners — an axis-symmetric
    rounded rectangle. Its corners are filleted in the SOURCE, so a corner_radius that
    matches them yields a more faithful fit than a sharp polygon. (The scored selector
    keeps a *sharp* source sharp regardless of corner_radius — it only rounds when the
    rounding is actually more faithful — so this fixture must itself be rounded for
    corner_radius to drive the output.)"""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dx = np.clip(np.abs(xx - cx) - (halfw - rad), 0, None)
    dy = np.clip(np.abs(yy - (top + bot) / 2) - ((bot - top) / 2 - rad), 0, None)
    cov = np.clip(0.5 - (np.hypot(dx, dy) - rad), 0, 1)          # antialiased coverage
    navy = np.array([6, 35, 54])
    return np.round(255 * (1 - cov)[..., None] + navy * cov[..., None]).astype(np.uint8)


def test_corner_radius_override_changes_output():
    img = _rounded_band_img()
    sharp = idealize(img, options=Options(corner_radius=0.0))
    rounded = idealize(img, options=Options(corner_radius=10.0))
    assert "<rect" not in rounded                  # rounded band -> rounded-trapezoid path
    # corner_radius drives which fitters produce a (winning) candidate: at cr=10
    # rounded_trapezoid_fit yields a fillet that matches this rounded source and is
    # selected; at cr=0 it does not, so a sharp fit wins. (The selector keeps a *sharp*
    # source sharp regardless of corner_radius — hence this fixture must be rounded.)
    assert sharp != rounded                        # corner_radius drives the selected geometry


def test_selection_none_is_byte_identical_to_default():
    # parity: supplying selection=None must not change output at all
    img = _rounded_band_img()
    a = idealize(img, options=Options())
    b = idealize(img, options=Options(selection=None))
    assert a == b


def test_force_symmetric_on_s0_emits_path_not_circle():
    from vectoreyes.selection import SelectionPolicy, ElementSelection
    h = w = 100
    img = np.full((h, w, 3), 255, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    img[(xx - 50) ** 2 + (yy - 50) ** 2 <= 32 ** 2] = (30, 100, 235)
    auto = idealize(img, options=Options())
    assert "<circle" in auto                                  # auto picks a circle
    # force="symmetric": the circle is on the symmetry axis so only "primitive" and
    # "symmetric" strategies are generated (PATH is excluded for symmetry-preserving shapes).
    # "symmetric" produces a <path>, proving sN addressing reached the emit layer.
    policy = SelectionPolicy(by_id={"s0": ElementSelection(force="symmetric")})
    forced = idealize(img, options=Options(selection=policy))
    assert "<circle" not in forced and "<path" in forced     # sN addressing reached emit


def test_default_policy_restricts_all_elements():
    from vectoreyes.selection import SelectionPolicy, ElementSelection
    h = w = 100
    img = np.full((h, w, 3), 255, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    img[(xx - 50) ** 2 + (yy - 50) ** 2 <= 32 ** 2] = (30, 100, 235)
    # force="symmetric": same reasoning as test_force_symmetric_on_s0_emits_path_not_circle
    policy = SelectionPolicy(default=ElementSelection(force="symmetric"))
    out = idealize(img, options=Options(selection=policy))
    assert "<circle" not in out and "<path" in out           # default reached the un-keyed element


def _gradient_bar(img, r0, r1, c0, c1, c_lo, c_hi):
    # horizontal left->right gradient from c_lo to c_hi inside the box
    width = c1 - c0
    for i, x in enumerate(range(c0, c1)):
        t = i / max(1, width - 1)
        img[r0:r1, x] = tuple(int(round(c_lo[k] + t * (c_hi[k] - c_lo[k]))) for k in range(3))


def test_two_separated_gradient_blobs_emit_two_gradients():
    # two gradient bars separated by a wide vertical gutter: each is a single dominant
    # blob within its own component, so both pass the single-blob gate (this is the
    # multi-blob unblock — together they would fail _dominant_blob_fraction).
    h, w = 60, 200
    img = np.full((h, w, 3), 255, np.uint8)
    _gradient_bar(img, 15, 45, 10, 70, (10, 20, 200), (200, 40, 20))      # left bar  cols 10..70
    _gradient_bar(img, 15, 45, 130, 190, (20, 200, 30), (10, 40, 220))    # right bar cols 130..190
    # gutter cols 70..130 = 60 wide; block extent 10..190 = 180; threshold 0.3*180=54 -> splits
    svg = idealize(img)
    assert svg.count("<linearGradient") + svg.count("<radialGradient") >= 2


def test_two_component_ids_are_continuous_no_collision():
    # two separated solid blobs -> two components -> ids s0, s1 with no gap/collision
    import re
    h, w = 60, 160
    img = np.full((h, w, 3), 255, np.uint8)
    img[15:45, 10:50] = (10, 30, 90)       # left square  cols 10..50
    img[15:45, 110:150] = (90, 30, 10)     # right square cols 110..150 (gutter 50..110 = 60)
    svg = idealize(img)
    ids = re.findall(r'id="(s\d+)"', svg)
    assert ids == ["s0", "s1"]             # global, continuous across the component boundary


def test_per_component_vertical_symmetry_emits_mirror_use():
    # LEFT component: a mirror PAIR of squares about col 30 (cols 10-25 and 35-50).
    # RIGHT component: a single offset square (cols 120-150). The WHOLE silhouette has
    # no vertical mirror, so the pre-slice-5 mark-wide axis is None and the left pair is
    # NOT deduped (no <use>). Per component, the left component's local axis (col 30)
    # makes the two squares a mirror pair -> one element + a <use> mirror.
    from vectoreyes.symmetry import detect_axis
    h, w = 70, 170
    img = np.full((h, w, 3), 255, np.uint8)
    img[20:50, 10:25] = (20, 40, 80)       # left-of-pair
    img[20:50, 35:50] = (20, 40, 80)       # right-of-pair (mirror about col 30)
    img[20:50, 120:150] = (20, 40, 80)     # lone square right (gutter 50..120 = 70 wide)
    silhouette = np.any(img != 255, axis=2)
    assert detect_axis(silhouette) is None              # premise: no global vertical axis
    svg = idealize(img)
    assert "<use" in svg                                # the pair dedups only per-component


def test_multicomponent_selection_addresses_global_sn():
    # two separated squares -> components s0 (left), s1 (right). A policy keyed to the
    # GLOBAL id "s1" must reach the SECOND component. With the per-component-local eid
    # bug, for_id was queried with "s0","s0" and this policy was silently ignored.
    import pytest
    from vectoreyes.selection import SelectionPolicy, ElementSelection
    h, w = 60, 160
    img = np.full((h, w, 3), 255, np.uint8)
    img[15:45, 10:50] = (10, 30, 90)        # left square  -> s0
    img[15:45, 110:150] = (90, 30, 10)      # right square -> s1
    # force a known-but-unavailable strategy on s1: the second element's policy IS
    # consulted (warns naming s1) only if global sN addressing reaches build_candidates.
    policy = SelectionPolicy(by_id={"s1": ElementSelection(force="holed_symmetric")})
    with pytest.warns(UserWarning, match="s1"):
        idealize(img, options=Options(selection=policy))


def test_conditioning_default_off_large_input_full_res():
    """With working_max_dim=None (new default), a >768 image is vectorized at full resolution."""
    arr = np.full((200, 1000, 3), 255, dtype=np.uint8)
    arr[50:150, 100:900] = (30, 100, 220)               # blue rect on white
    svg = idealize(arr)                                  # Options() → working_max_dim=None
    assert 'viewBox="0 0 1000 200"' in svg, "large input must use full-res viewBox by default"


def test_conditioning_applies_when_explicitly_set():
    """With working_max_dim explicitly set, a >threshold image is downscaled for vectorization."""
    arr = np.full((200, 1000, 3), 255, dtype=np.uint8)
    arr[50:150, 100:900] = (30, 100, 220)
    svg = idealize(arr, options=Options(working_max_dim=512))
    # width/height rewritten to original; viewBox reflects conditioned resolution
    assert 'width="1000" height="200"' in svg
    assert 'viewBox="0 0 1000 200"' not in svg, "conditioned input must NOT use original viewBox"


def test_rgba_file_input_composites_on_white(tmp_path):
    # A semi-transparent block: composited on WHITE, 50%-alpha red becomes pink. PIL's
    # convert("RGB") instead DROPS alpha (keeps the stored RGB), yielding over-saturated
    # full red (#FF0000). idealize() of the file path must equal idealizing the manually
    # on-white-composited array — proving it composites on white, not drop-alpha.
    from PIL import Image
    h = w = 60
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[10:50, 10:50] = (255, 0, 0, 128)           # 50%-alpha red on transparent ground
    Image.fromarray(rgba, "RGBA").save(tmp_path / "b.png")
    on_white = np.asarray(
        Image.alpha_composite(Image.new("RGBA", (w, h), (255, 255, 255, 255)),
                              Image.fromarray(rgba, "RGBA")).convert("RGB"),
        np.uint8,
    )
    assert idealize(str(tmp_path / "b.png")) == idealize(on_white)
    # the emitted fill is the pink composite (G,B raised), never #FF0000 — the
    # over-saturated result of dropping alpha instead of compositing it on white
    assert "#FF0000" not in idealize(str(tmp_path / "b.png"))

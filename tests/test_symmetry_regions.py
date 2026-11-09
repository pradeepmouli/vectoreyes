import numpy as np
from PIL import Image
from vectoreyes.symmetry import Axis2D, _perimeter, reflection_off_count
from vectoreyes.symmetry import region_is_self_symmetric, regions_mirror_pair, K_BAND
from vectoreyes.symmetry import propose_axes
from vectoreyes.symmetry import cluster_axes, AxisProposal
from vectoreyes.symmetry import detect_symmetry_groups, sym_off_ratio, _threshold_to_off_ratio
from vectoreyes.types import Region
from vectoreyes.pipeline import idealize, Options
from scipy import ndimage as ndi

def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r

def test_perimeter_of_disk_is_boundary_band():
    m = _disk(60, 60, 30, 30, 20)
    p = _perimeter(m)
    assert 100 < p < 180   # ~2*pi*r, a one-pixel ring

def test_reflection_off_count_zero_for_symmetric_axis():
    m = _disk(60, 60, 30, 30, 20)           # disk: symmetric about every axis through center
    fg = np.nonzero(m)                       # (rows, cols) == (ys, xs)
    fg_xy = (fg[1], fg[0])                   # _axis_mismatch wants (xs, ys)
    dist = ndi.distance_transform_edt(~m)
    axis = Axis2D(theta=0.0, cx=30.0, cy=30.0)        # horizontal line through center
    assert reflection_off_count(fg_xy, axis, dist) == 0

def test_reflection_off_count_large_for_wrong_axis():
    # an L-shaped (asymmetric) region: reflection lands well off-shape
    m = np.zeros((60, 60), bool); m[10:50, 10:20] = True; m[40:50, 10:45] = True
    fg = np.nonzero(m); fg_xy = (fg[1], fg[0])
    dist = ndi.distance_transform_edt(~m)
    cy, cx = [c.mean() for c in fg[::-1]][::-1]   # centroid (cy, cx)
    axis = Axis2D(theta=np.pi / 2, cx=float(cx), cy=float(cy))  # vertical through centroid
    assert reflection_off_count(fg_xy, axis, dist) > _perimeter(m)


def test_reflection_off_count_treats_out_of_bounds_reflections_as_off_shape():
    fg_xy = (np.array([0]), np.array([1]))
    dist = np.zeros((3, 3), dtype=float)
    axis = Axis2D(theta=np.pi / 2, cx=2.0, cy=1.0)

    assert reflection_off_count(fg_xy, axis, dist) == 1


def _region(mask, label=1):
    return Region(label=label, mask=mask, color_hex="#000000")


def test_disk_is_self_symmetric_about_any_central_axis():
    m = _disk(80, 80, 40, 40, 25)
    for theta in (0.0, np.pi / 4, np.pi / 2):
        assert region_is_self_symmetric(_region(m), Axis2D(theta, 40.0, 40.0))


def test_asymmetric_region_is_not_self_symmetric():
    m = np.zeros((80, 80), bool); m[20:60, 20:30] = True; m[50:60, 20:55] = True  # L
    fg = np.nonzero(m); cy, cx = float(fg[0].mean()), float(fg[1].mean())
    assert not region_is_self_symmetric(_region(m), Axis2D(np.pi / 2, cx, cy))


def test_mirror_pair_detected_and_size_guarded():
    left = np.zeros((80, 80), bool); left[30:50, 15:25] = True
    right = np.zeros((80, 80), bool); right[30:50, 55:65] = True   # mirror of left about x=40
    axis = Axis2D(np.pi / 2, 40.0, 40.0)                            # vertical through x=40
    assert regions_mirror_pair(_region(left, 1), _region(right, 2), axis)
    big = np.zeros((80, 80), bool); big[20:60, 50:70] = True        # bigger -> not a pair
    assert not regions_mirror_pair(_region(left, 1), _region(big, 3), axis)


def test_mirror_pair_requires_matching_color():
    left = np.zeros((80, 80), bool); left[30:50, 15:25] = True
    right = np.zeros((80, 80), bool); right[30:50, 55:65] = True
    axis = Axis2D(np.pi / 2, 40.0, 40.0)

    assert not regions_mirror_pair(
        Region(label=1, mask=left, color_hex="#000000"),
        Region(label=2, mask=right, color_hex="#ffffff"),
        axis,
    )


def test_iou_threshold_mapping_preserves_default_and_scales_monotonically():
    default = _threshold_to_off_ratio(0.96, 0.96)

    assert default == K_BAND
    assert _threshold_to_off_ratio(0.98, 0.96) < default
    assert _threshold_to_off_ratio(0.90, 0.96) > default


def test_propose_axes_finds_vertical_for_centered_symmetric_region():
    m = _disk(100, 120, 50, 60, 30)                    # centered disk
    props = propose_axes([_region(m)])
    # a disk proposes many self-axes through its center; at least one ~vertical
    assert any(abs(p.theta - np.pi / 2) < 0.3 and abs(p.cx - 60) < 2 for p in props)


def test_propose_axes_finds_pair_bisector():
    left = np.zeros((80, 120), bool); left[30:50, 20:35] = True
    right = np.zeros((80, 120), bool); right[30:50, 85:100] = True   # mirror about x=60
    props = propose_axes([_region(left, 1), _region(right, 2)])
    assert any(abs(p.theta - np.pi / 2) < 0.2 and abs(p.cx - 60) < 3 for p in props)


def test_cluster_merges_near_duplicates_and_ranks_by_weight():
    props = [
        AxisProposal(np.pi/2, 60, 50, 100), AxisProposal(np.pi/2 + 0.02, 61, 50, 100),  # vertical, heavy
        AxisProposal(0.0, 60, 50, 30),                                                   # horizontal, light
    ]
    axes = cluster_axes(props)
    assert len(axes) == 2
    (a0, w0), (a1, w1) = axes
    assert w0 > w1 and abs(a0.theta - np.pi/2) < 0.05    # heaviest first == vertical


def test_radish_plus_text_isolates_symmetric_subset():
    # 3 centered symmetric bands (radish) + 1 off-center asymmetric blob (text)
    bands = []
    for i, y in enumerate((20, 45, 70)):
        m = np.zeros((140, 120), bool); m[y:y+18, 30:90] = True; bands.append(_region(m, i + 1))
    text = np.zeros((140, 120), bool); text[110:130, 10:40] = True  # off-axis, asymmetric placement
    groups = detect_symmetry_groups(bands + [_region(text, 9)])
    sym = [g for g in groups if g.axes]
    assert sym, "a symmetric group must be found"
    g = sym[0]
    assert abs(g.axes[0].theta - np.pi/2) < 0.1 and abs(g.axes[0].cx - 60) < 3
    assert len(g.straddlers) == 3
    claimed = {r.label for r in g.straddlers}
    assert 9 not in claimed   # the text region is NOT pulled into the symmetric group


def test_two_independent_symmetric_figures_two_groups():
    d1 = _disk(80, 200, 40, 40, 22); d2 = _disk(80, 200, 40, 160, 22)
    groups = [g for g in detect_symmetry_groups([_region(d1, 1), _region(d2, 2)]) if g.axes]
    # each disk is self-symmetric about its own centre -> at least two distinct axes/groups
    assert sum(len(g.straddlers) for g in groups) == 2


def test_determinism_repeated_runs_identical():
    m = _disk(80, 80, 40, 40, 25)
    a = detect_symmetry_groups([_region(m)])
    b = detect_symmetry_groups([_region(m)])
    assert [(g.axes, [r.label for r in g.straddlers]) for g in a] == \
           [(g.axes, [r.label for r in g.straddlers]) for g in b]


# --- Task 6: end-to-end daikonic integration ---

def _sym_iou_of_largest_body(svg, axis_x):
    from tests._symhelp import sym_iou_largest
    return sym_iou_largest(svg, axis_x)


def test_daikonic_body_is_exactly_symmetric_end_to_end():
    arr = np.asarray(Image.open("tests/fixtures/daikonic/source.png").convert("RGBA"))
    bg = Image.new("RGB", (arr.shape[1], arr.shape[0]), (255, 255, 255))
    im = Image.fromarray(arr); bg.paste(im, mask=im.split()[3])
    svg, rep = idealize(np.asarray(bg.convert("RGB"), np.uint8), options=Options(), report=True)
    assert rep.axes, "an axis must now be detected for the radish"
    assert _sym_iou_of_largest_body(svg, rep.axes[0].x1) >= 0.999


# --- Task 6b: K_BAND floor regression ---

def test_k_band_floor_rejects_near_symmetric_and_accepts_rasterization_perfect():
    """K_BAND=0.3 must pass off/peri=0.0 (rasterization-perfect shapes) and reject
    off/peri>0.4 (the near-symmetric-but-not class that includes icloud's cloud body
    at ~0.78 and pokeball's outer circle at ~0.58)."""
    h, w = 60, 80

    # Rasterization-perfect: centred rectangle → off/peri = 0.0
    rect = np.zeros((h, w), bool)
    rect[10:50, 20:60] = True   # 40 px wide, vertical axis at x=40
    axis_rect = Axis2D(np.pi / 2, 40.0, 30.0)
    ratio_perfect = sym_off_ratio(rect, axis_rect)
    assert ratio_perfect == 0.0, f"perfect rect must have off/peri=0.0, got {ratio_perfect}"
    assert ratio_perfect <= K_BAND
    assert region_is_self_symmetric(_region(rect), axis_rect)

    # Near-symmetric-but-not: L-shape (off/peri >> K_BAND, typically ~2.7)
    lshape = np.zeros((h, w), bool)
    lshape[10:50, 5:35] = True   # vertical bar on left
    lshape[30:50, 35:65] = True  # bottom-right extension
    fg_ys, fg_xs = np.nonzero(lshape)
    axis_l = Axis2D(np.pi / 2, float(fg_xs.mean()), float(fg_ys.mean()))
    ratio_l = sym_off_ratio(lshape, axis_l)
    assert ratio_l > K_BAND, f"L-shape off/peri={ratio_l:.3f} must exceed K_BAND={K_BAND}"
    assert ratio_l >= 0.4, f"expected L-shape off/peri >= 0.4, got {ratio_l:.3f}"
    assert not region_is_self_symmetric(_region(lshape), axis_l)

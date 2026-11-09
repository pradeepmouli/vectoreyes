import numpy as np
from vectoreyes.types import Axis, Region
from vectoreyes.symmetry import _iou, _reflect_cols, classify_regions, detect_axis


def _leaning_triangle(apex_x, H=44, W=40, base_l=6, base_r=33, base_y=38, apex_y=5):
    """A pointed triangle with its base centred on x=19.5 but its apex shifted to
    `apex_x` — i.e. a tip whose extreme point lies off the symmetry axis."""
    m = np.zeros((H, W), bool)
    for y in range(apex_y, base_y + 1):
        t = (y - apex_y) / (base_y - apex_y)
        xl = int(round(apex_x + (base_l - apex_x) * t))
        xr = int(round(apex_x + (base_r - apex_x) * t))
        m[y, xl:xr + 1] = True
    return m


def _sym_masks():
    H, W = 40, 40
    dome = np.zeros((H, W), bool)
    yy, xx = np.ogrid[:H, :W]
    dome[((xx - 20) ** 2 / 100 + (yy - 20) ** 2 / 64) <= 1] = True  # centered ellipse
    left = np.zeros((H, W), bool); left[5:10, 6:12] = True
    right = np.zeros((H, W), bool); right[5:10, 29:35] = True       # mirror of left[6:12) about x=20
    return [Region(1, dome, "#062336"), Region(2, left, "#062336"), Region(3, right, "#062336")]


def test_detect_axis_finds_center():
    regions = _sym_masks()
    union = np.any([r.mask for r in regions], axis=0)
    axis = detect_axis(union)
    assert axis is not None
    assert abs(axis.x - 19.5) < 1.0  # image center for W=40 is 19.5


def test_classify_straddle_vs_pair():
    regions = _sym_masks()
    axis = detect_axis(np.any([r.mask for r in regions], axis=0))
    straddlers, pairs, loners = classify_regions(regions, axis)
    assert len(straddlers) == 1 and straddlers[0].label == 1       # the dome
    assert len(pairs) == 1                                          # left+right as one pair
    assert loners == []                                            # nothing asymmetric+unpaired
    canon, _mirror = pairs[0]
    assert {canon.label, _mirror.label} == {2, 3}


def test_classify_isolates_lone_asymmetric_region():
    # an asymmetric region with no mirror partner must NOT be treated as a
    # self-symmetric straddler (else _fit_region would force-mirror it).
    regions = _sym_masks()
    asym = np.zeros((40, 40), bool)
    asym[6:10, 6:9] = True
    asym[8:10, 6:16] = True                                        # an L — not self-symmetric
    regions.append(Region(4, asym, "#ff0000"))                    # unique colour ⇒ no partner
    axis = detect_axis(np.any([r.mask for r in regions], axis=0))
    straddlers, pairs, loners = classify_regions(regions, axis)
    assert [r.label for r in loners] == [4]
    assert 4 not in [r.label for r in straddlers]


def test_classify_demotes_pointed_offaxis_region_to_loner():
    # A leaning, pointed region overlaps its own reflection enough to clear a loose
    # IoU gate, but its apex sits well off the axis: half-mirroring it about the axis
    # would split that single apex into a fork (two prongs + a central notch). It must
    # be classified as a loner (fit as-is), NOT a self-symmetric straddler.
    axis = Axis(x=19.5)
    tri = _leaning_triangle(apex_x=24)
    # in the band the old 0.5 gate wrongly admitted, yet with no true vertical axis
    assert 0.5 <= _iou(tri, _reflect_cols(tri, axis.x)) < 0.90
    assert detect_axis(tri) is None
    straddlers, pairs, loners = classify_regions([Region(1, tri, "#ff0000")], axis)
    assert [r.label for r in loners] == [1]
    assert straddlers == [] and pairs == []


def test_pairs_reject_coincidental_non_twins():
    # two same-colour blobs that are only COARSELY mirror-similar about x=19.5:
    # reflect(left) overlaps right enough to clear the OLD 0.6 gate, but is well
    # under the 0.90 (= 1 - SYM_TOL) symmetry bar. They are NOT mirror twins (think
    # the "B" and "R" of a wordmark), so they must fall through to loners — never a
    # `<use>`-mirror pair (which would substitute one with the mirror of the other).
    axis = Axis(x=19.5)
    left = np.zeros((40, 40), bool); left[8:20, 6:14] = True
    right = np.zeros((40, 40), bool); right[10:22, 26:34] = True   # shifted off the exact mirror
    iou = _iou(_reflect_cols(left, axis.x), right)
    assert 0.6 <= iou < 0.90
    straddlers, pairs, loners = classify_regions(
        [Region(1, left, "#abcabc"), Region(2, right, "#abcabc")], axis)
    assert pairs == []
    assert {r.label for r in loners} == {1, 2}


def test_pairs_keep_genuine_mirror_twins():
    # an exact mirror twin (IoU ~1.0) must still pair and `<use>`-mirror.
    axis = Axis(x=19.5)
    left = np.zeros((40, 40), bool); left[8:20, 6:14] = True
    right = _reflect_cols(left, axis.x)
    straddlers, pairs, loners = classify_regions(
        [Region(1, left, "#abcabc"), Region(2, right, "#abcabc")], axis)
    assert len(pairs) == 1 and loners == []


def test_no_symmetry_returns_none():
    # a centered square IS bilaterally symmetric — knock out one corner so the
    # shape has no vertical axis at all
    asym = np.zeros((30, 30), bool); asym[2:8, 2:8] = True
    asym[2:4, 2:4] = False
    assert detect_axis(asym) is None


def test_axis_reachable_for_fractional_centroid():
    # symmetric about x=19.5 but with an off-grid centroid
    m = np.zeros((30, 40), bool)
    m[5:25, 12:28] = True          # symmetric block about 19.5
    m[5:8, 12:14] = True           # nudge centroid off .5 grid (still ~symmetric enough)
    ax = detect_axis(m)
    assert ax is not None and abs(ax.x - 19.5) <= 0.5

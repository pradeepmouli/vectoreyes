import numpy as np
import pytest
from unittest.mock import patch
from vectoreyes.surface_merge import seam_is_soft, gradients_continuous, merge_surfaces
from vectoreyes.fill_fit import FillFitResult, fit_fill as _ff
from vectoreyes.types import Region
from vectoreyes.candidate import FlatFill, LinearGradientFill, RadialGradientFill

_AnyGradient = (LinearGradientFill, RadialGradientFill)


@pytest.mark.parametrize("curved", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_closed_hard_seam_is_not_palette_noise(curved, reverse):
    from vectoreyes.material_pretrace import _noncredible_edges
    from vectoreyes.surface_merge import material_groups, seam_is_noncredible

    yy, xx = np.mgrid[:60, :60]
    outer = (xx >= 10) & (xx < 50) & (yy >= 10) & (yy < 50)
    inner = ((xx - 30)**2 + (yy - 30)**2 < 10**2) if curved else (
        (xx >= 24) & (xx < 36) & (yy >= 24) & (yy < 36)
    )
    regions = [Region(1, outer & ~inner, "#062336"), Region(2, inner, "#FFFFFF")]
    if reverse:
        regions.reverse()
    rgb = np.full((60, 60, 3), 255, dtype=np.uint8)
    rgb[outer & ~inner] = (6, 35, 54)

    assert not seam_is_noncredible(regions[0].mask, regions[1].mask)
    assert _noncredible_edges(regions) == set()
    assert len(material_groups(regions, rgb)) == 2


def test_closed_soft_seam_can_still_merge_by_source_color():
    from vectoreyes.surface_merge import material_groups

    yy, xx = np.mgrid[:60, :60]
    inner = (xx - 30)**2 + (yy - 30)**2 < 10**2
    regions = [Region(1, ~inner, "#062336"), Region(2, inner, "#072437")]
    rgb = np.full((60, 60, 3), (6, 35, 54), dtype=np.uint8)
    rgb[inner] = (7, 36, 55)
    assert material_groups(regions, rgb) == [(0, 1)]


def test_open_tortuous_seam_still_merges_by_shape():
    from vectoreyes.material_pretrace import _noncredible_edges
    from vectoreyes.surface_merge import seam_is_noncredible

    yy, xx = np.mgrid[:80, :100]
    boundary = 50 + 18 * np.sin(yy * np.pi / 10)
    left = xx < boundary
    regions = [Region(1, left, "#062336"), Region(2, ~left, "#FFFFFF")]
    assert seam_is_noncredible(left, ~left)
    assert _noncredible_edges(regions) == {(0, 1)}


def _ramp(h, w, c0, c1):
    rgb = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        t = x / (w - 1)
        rgb[:, x] = [round(c0[i] + t * (c1[i] - c0[i])) for i in range(3)]
    return rgb


def _full_ramp_split():
    # one blue->red ramp, split into left and right halves as two masks over one canvas
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    return rgb, a, b


def test_within_ramp_seam_is_soft():
    rgb, a, b = _full_ramp_split()
    assert seam_is_soft(a, b, rgb)                       # one continuous ramp sliced in two


def test_hard_color_step_is_not_soft():
    # left half ramps to red; right half is a uniform blue patch (a "dot") -> sharp step
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    rgb[:, W // 2:] = (20, 60, 210)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    assert not seam_is_soft(a, b, rgb)


# NOTE: a same-color-at-seam feature is not distinguishable by seam_is_soft — when a
# feature's color matches the ramp at the shared border there is genuinely no step, so
# seam_is_soft correctly reads soft. The same-color protection lives at the merge level
# (the union-fits-gradient guard in Task 3), tested in test_surface_merge merge tests.


def test_non_adjacent_masks_are_not_soft():
    rgb, a, _b = _full_ramp_split()
    far = np.zeros_like(a); far[:5, 70:75] = True         # disjoint from a (a is left half)
    assert not seam_is_soft(a, far, rgb)


# ── B: gradient-continuity path (both regions wide enough to fit a gradient) ──

def test_two_gradient_halves_are_continuous():
    # each half of a wide ramp spans enough colour to fit a gradient -> compare at the seam
    rgb, a, b = _full_ramp_split()
    fa, fb = _ff(a, rgb, flat_hex="#000000").fill, _ff(b, rgb, flat_hex="#000000").fill
    assert gradients_continuous(fa, a, fb, b)


def test_gradient_meets_flat_is_not_continuous():
    # one side ramps, the other is a uniform patch (flat) -> B does not apply (False)
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    rgb[:, W // 2:] = (20, 60, 210)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    fa, fb = _ff(a, rgb, flat_hex="#000000").fill, _ff(b, rgb, flat_hex="#143CD2").fill
    assert not gradients_continuous(fa, a, fb, b)


def test_non_adjacent_gradients_not_continuous():
    # two non-touching gradient blobs have no shared seam -> False
    rgb, a, _ = _full_ramp_split()
    far = np.zeros_like(a); far[:, 70:] = True       # disjoint from a (left half)
    fa = _ff(a, rgb, flat_hex="#000000").fill
    fb = _ff(far, rgb, flat_hex="#000000").fill
    assert not gradients_continuous(fa, a, fb, far)


# ── Task 3: merge_surfaces ────────────────────────────────────────────────────

def _region(label, mask, hex_):
    return Region(label=label, mask=mask, color_hex=hex_)


def test_ramp_halves_merge_into_one_gradient():
    # wide halves each fit a gradient -> B path
    # NOTE: the brief says "a horizontal ramp may fit RADIAL — that's fine; the tests
    # assert 'is a gradient'"; we check _AnyGradient accordingly.
    rgb, a, b = _full_ramp_split()
    filled = [(_region(0, a, "#1428C8"), _ff(a, rgb, flat_hex="#1428C8").fill),
              (_region(1, b, "#DC2814"), _ff(b, rgb, flat_hex="#DC2814").fill)]
    out = merge_surfaces(filled, rgb)
    assert len(out) == 1
    region, fill = out[0]
    assert isinstance(fill, _AnyGradient)
    assert region.mask.sum() == (a | b).sum()           # union silhouette, clean


def test_narrow_bands_collapse_to_one_gradient():
    # narrow strips of one smooth ramp collapse to a single gradient: a strip too narrow
    # to fit a gradient devolves to flat and merges via the A path; a wide-enough one via
    # B. Either way the outcome is one gradient.
    H, W = 40, 96
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    filled = []
    for k in range(0, W, 8):
        m = np.zeros((H, W), bool); m[:, k:k + 8] = True
        filled.append((_region(k, m, "#808080"), _ff(m, rgb, flat_hex="#808080").fill))
    out = merge_surfaces(filled, rgb)
    assert len(out) == 1 and isinstance(out[0][1], _AnyGradient)


def test_flat_dot_on_ramp_stays_separate():
    # the "dot" has a HARD border against the ramp -> hard seam -> never merged,
    # even though its blue is in the wing's color family.
    H, W = 40, 80
    rgb = _ramp(H, W, (20, 40, 200), (220, 40, 20))
    rgb[:, W // 2:] = (20, 60, 210)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    filled = [(_region(0, a, "#1428C8"), _ff(a, rgb, flat_hex="#1428C8").fill),
              (_region(1, b, "#143CD2"), _ff(b, rgb, flat_hex="#143CD2").fill)]
    out = merge_surfaces(filled, rgb)
    assert len(out) == 2                                 # NOT merged


def test_two_distinct_flats_do_not_merge():
    # two solid colors meeting at a soft-ish AA edge: union is NOT a gradient -> no merge.
    H, W = 40, 80
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, : W // 2] = (200, 40, 40); rgb[:, W // 2:] = (40, 160, 60)
    a = np.zeros((H, W), bool); a[:, : W // 2] = True
    b = np.zeros((H, W), bool); b[:, W // 2:] = True
    filled = [(_region(0, a, "#C82828"), _ff(a, rgb, flat_hex="#C82828").fill),
              (_region(1, b, "#28A03C"), _ff(b, rgb, flat_hex="#28A03C").fill)]
    assert len(merge_surfaces(filled, rgb)) == 2


def test_seam_cache_evicts_stale_entries_after_merge():
    """Regression: after A absorbs X (rep=A), the stale (A.label, C.label)=False
    cache entry must be evicted. Without eviction a later pass sees the stale False
    and silently skips the valid A_new+C merge even though A's expanded mask IS
    adjacent to C.

    Layout (H=40, W=48):
      A [label=0, cols  0-23, area=960] | X [label=2, cols 24-31, area=320] | C [label=1, cols 32-47, area=640]
      A and C are not adjacent; A and X are adjacent; X and C are adjacent.

    Sort (descending area): [A(960), C(640), X(320)] → scan (A,C) before (A,X).
    Pass 1:
      (A,C): seam_is_soft=False (not adjacent) → cached (0,1)=False  ← the stale entry
      (A,X): seam_is_soft=True, fit_fill(A|X)=gradient → MERGE; rep=A; label 0 reused
             with mask cols 0-31, now adjacent to C.
             Correct fix: evict cache entries for labels {0,2} → (0,1) removed.
    Pass 2:
      (A_new, C): key (0,1). With fix: not in cache → fresh seam_is_soft=True → MERGE.
                             Without fix: stale False → no merge (bug).
    """
    H, W = 40, 48
    # Strong blue→red ramp so union regions have clearly detectable gradients.
    rgb = _ramp(H, W, (0, 0, 255), (255, 0, 0))

    mask_a = np.zeros((H, W), bool); mask_a[:, :24] = True    # label 0, area 960
    mask_c = np.zeros((H, W), bool); mask_c[:, 32:48] = True  # label 1, area 640
    mask_x = np.zeros((H, W), bool); mask_x[:, 24:32] = True  # label 2, area 320

    # Seed all three as flat so path A (seam_is_soft cache) is exercised throughout.
    filled = [
        (_region(0, mask_a, "#0000ff"), FlatFill(hex="#0000ff")),
        (_region(1, mask_c, "#ff0000"), FlatFill(hex="#ff0000")),
        (_region(2, mask_x, "#808080"), FlatFill(hex="#808080")),
    ]

    # fit_fill: return gradient iff the union spans both A's territory (cols <24) and
    # the right side (cols ≥24), so A, X, and C individually stay flat while A|X and
    # A|X|C are promoted to gradient.  Real gradient detection runs on qualifying masks.
    def _controlled_fit_fill(mask, rgb_, *, flat_hex):
        if mask[:, :24].any() and mask[:, 24:].any():
            return _ff(mask, rgb_, flat_hex=flat_hex)
        return FillFitResult(FlatFill(hex=flat_hex), 'flat_evidence', 'controlled_flat', int(mask.sum()))

    with patch("vectoreyes.surface_merge.fit_fill", _controlled_fit_fill):
        out = merge_surfaces(filled, rgb)

    assert len(out) == 1, (
        f"expected 1 merged region, got {len(out)}; "
        "likely cause: stale seam_cache entry (A.label, C.label)=False was not evicted "
        "after the A|X merge, blocking the valid A_new+C merge in pass 2"
    )
    assert isinstance(out[0][1], _AnyGradient)

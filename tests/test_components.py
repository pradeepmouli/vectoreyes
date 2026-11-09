import numpy as np

from vectoreyes.types import Region
from vectoreyes.components import decompose_components


def _rect_region(rid, r0, r1, c0, c1, h, w, hexc="#222222"):
    m = np.zeros((h, w), bool)
    m[r0:r1, c0:c1] = True
    return Region(rid, m, hexc)


def _disk_region(rid, cy, cx, rad, h, w, hexc="#222222"):
    yy, xx = np.ogrid[:h, :w]
    return Region(rid, (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2, hexc)


def test_single_region_is_one_component():
    h = w = 80
    regs = [_disk_region(1, 40, 40, 20, h, w)]
    comps = decompose_components(regs, (h, w))
    assert len(comps) == 1 and comps[0] == regs


def test_two_widely_separated_blobs_split_left_to_right():
    h, w = 60, 140
    left = _disk_region(1, 30, 25, 18, h, w)        # cols ~7..43
    right = _rect_region(2, 12, 48, 95, 130, h, w)  # cols 95..130
    comps = decompose_components([left, right], (h, w))   # gutter cols ~43..95 (~52 wide)
    assert len(comps) == 2
    assert comps[0] == [left] and comps[1] == [right]     # reading order L->R


def test_vertical_stack_splits_top_to_bottom():
    h, w = 140, 60
    top = _disk_region(1, 25, 30, 18, h, w)         # rows ~7..43
    bot = _rect_region(2, 95, 130, 12, 48, h, w)    # rows 95..130
    comps = decompose_components([top, bot], (h, w))      # gutter rows ~43..95 (~52 tall)
    assert len(comps) == 2
    assert comps[0] == [top] and comps[1] == [bot]        # reading order top->bottom


def test_borderline_narrow_gap_stays_one_component():
    # two-band-logo geometry: bands at rows 8-26 and 34-52 -> ~8px gap on a 44px block
    # (~18% < 30% threshold) must NOT split.
    h, w = 60, 80
    top = _rect_region(1, 8, 26, 12, 68, h, w)
    bot = _rect_region(2, 34, 52, 20, 60, h, w)
    comps = decompose_components([top, bot], (h, w))
    assert len(comps) == 1
    assert set(id(r) for r in comps[0]) == {id(top), id(bot)}


def test_nested_icon_over_two_word_row():
    # icon on top (rows 5..30), then a word row (rows 65..95) split by a vertical gutter.
    # Top-level horizontal gutter rows 30..65 = 35 (>= 0.3*90=27) -> cut icon vs words;
    # the bottom block then has a vertical gutter cols 40..80 = 40 (>= 0.3*100=30).
    h, w = 110, 120
    icon = _rect_region(1, 5, 30, 45, 75, h, w)        # top block
    word_l = _rect_region(2, 65, 95, 10, 40, h, w)     # bottom-left
    word_r = _rect_region(3, 65, 95, 80, 110, h, w)    # bottom-right
    comps = decompose_components([icon, word_l, word_r], (h, w))
    assert len(comps) == 3
    assert comps[0] == [icon]                          # horizontal cut first: icon on top
    assert comps[1] == [word_l] and comps[2] == [word_r]  # then vertical: L, R


def test_partition_is_clean_no_loss_no_duplication():
    h, w = 60, 140
    a = _disk_region(1, 30, 25, 18, h, w)
    b = _rect_region(2, 12, 48, 95, 130, h, w)
    comps = decompose_components([a, b], (h, w))
    flat = [r for c in comps for r in c]
    assert len(flat) == 2
    assert {id(r) for r in flat} == {id(a), id(b)}     # every input exactly once

"""Labelled synthetic corpus for the scorer: each case's correct winner is known
by construction. The headline gate is false-accept == 0 — especially the
flat-square-stays-flat over-accept regression."""
import numpy as np
import pytest

from vectoreyes.candidate import LinearGradientFill, RadialGradientFill
from vectoreyes.types import Region
from vectoreyes.score import rank_candidates
from tests._candidates import generate_candidates


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _flat_circle():
    h = w = 100
    src = np.full((h, w, 3), 255, np.uint8)
    mask = _disk(50, 50, 32, h, w)
    src[mask] = (30, 100, 235)
    return src, Region(1, mask, "#1e64eb"), "primitive_flat"


def _flat_square():
    h = w = 100
    src = np.full((h, w, 3), 255, np.uint8)
    mask = np.zeros((h, w), bool); mask[25:75, 25:75] = True
    src[mask] = (200, 30, 30)
    return src, Region(1, mask, "#c81e1e"), "primitive_flat"


def _linear_gradient_rect():
    h = w = 120
    src = np.full((h, w, 3), 255, np.uint8)
    mask = np.zeros((h, w), bool); mask[30:90, 20:100] = True
    c0 = np.array([37, 99, 235]); c1 = np.array([219, 39, 119])
    xs = np.arange(20, 100)
    for i, x in enumerate(xs):
        t = i / (len(xs) - 1)
        src[30:90, x] = np.round(c0 * (1 - t) + c1 * t).astype(np.uint8)
    return src, Region(1, mask, "#7b51ab"), "gradient"


def _radial_gradient_disc():
    h = w = 120
    src = np.full((h, w, 3), 255, np.uint8)
    cx, cy, r = 60, 60, 45
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= r
    c0 = np.array([125, 211, 252]); c1 = np.array([29, 78, 216])
    t = np.clip(dist / r, 0, 1)[..., None]
    grad = np.round(c0 * (1 - t) + c1 * t).astype(np.uint8)
    src[mask] = grad[mask]
    return src, Region(1, mask, "#52a5e4"), "gradient"


CASES = [_flat_circle(), _flat_square(), _linear_gradient_rect(), _radial_gradient_disc()]


def _winner_label(cand):
    if isinstance(cand.fill, (LinearGradientFill, RadialGradientFill)):
        return "gradient"
    return "primitive_flat" if cand.geometry.kind in ("circle", "ellipse", "rect", "polygon") else "path_flat"


@pytest.mark.parametrize("src,region,expected", CASES)
def test_scorer_picks_expected_winner(src, region, expected):
    cands = generate_candidates(region, src)
    ranked = rank_candidates(cands, src, region)
    assert ranked, "no candidates generated"
    got = _winner_label(ranked[0][0])
    assert got == expected, f"expected {expected}, got {got}; breakdown={ranked[0][1]}"


def test_false_accept_rate_is_zero():
    """No case may rank a more-complex/wrong candidate above the correct one."""
    misses = []
    for src, region, expected in CASES:
        ranked = rank_candidates(generate_candidates(region, src), src, region)
        if not ranked or _winner_label(ranked[0][0]) != expected:
            misses.append((expected, ranked[0][1] if ranked else None))
    assert not misses, f"false accepts: {misses}"

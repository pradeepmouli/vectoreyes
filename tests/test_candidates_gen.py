import numpy as np

from vectoreyes.candidate import FlatFill, LinearGradientFill, RadialGradientFill
from tests._candidates import generate_candidates


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def test_generate_includes_primitive_and_path_for_a_disk():
    from vectoreyes.types import Region
    h = w = 80
    src = np.full((h, w, 3), 255, np.uint8)
    mask = _disk(40, 40, 25, h, w)
    src[mask] = (30, 100, 235)
    region = Region(1, mask, "#1e64eb")
    cands = generate_candidates(region, src)
    kinds = {c.geometry.kind for c in cands}
    assert "circle" in kinds
    assert "path" in kinds
    assert all(isinstance(c.fill, FlatFill) for c in cands)

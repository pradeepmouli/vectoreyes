# SPDX-License-Identifier: MIT
import numpy as np
from skimage.metrics import structural_similarity
from vectoreyes.pipeline import idealize, Options
from tests._render import render_svg


def _synthetic_mastercard(H=240, W=360, gap=58, r=104):
    """Two overlapping disks (distinct colours) with a third colour in the overlap."""
    yy, xx = np.ogrid[:H, :W]
    cx, cy = W // 2, H // 2
    da = (xx - (cx - gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    db = (xx - (cx + gap)) ** 2 + (yy - cy) ** 2 <= r ** 2
    img = np.full((H, W, 3), 255, np.uint8)
    img[da] = (235, 0, 27)         # red
    img[db] = (247, 158, 27)       # yellow
    img[da & db] = (255, 95, 0)    # orange overlap
    return img, cx


def test_synthetic_mastercard_reconstructs_to_circles_plus_lens():
    img, cx = _synthetic_mastercard()
    H, W = img.shape[:2]
    svg = idealize(img, options=Options())
    assert svg.count("<circle") == 2                      # two real disks recovered
    assert "<path" in svg                                 # the lens
    out = render_svg(svg, W, H)
    k = min(cx, W - cx)
    # Symmetry is checked on the binary silhouette (coverage) — the two disks have
    # distinct colours by design, so per-channel SSIM would always be low regardless
    # of geometric correctness.  The silhouette must fold ≥ 0.99 about the axis.
    left_cov = (out[:, cx - k:cx] < 250).any(axis=2).astype(float)[:, ::-1]
    right_cov = (out[:, cx:cx + k] < 250).any(axis=2).astype(float)
    assert float(structural_similarity(left_cov, right_cov, data_range=1.0)) >= 0.99   # symmetric
    assert float(structural_similarity(out.astype(float), img.astype(float), channel_axis=2, data_range=255)) >= 0.95


def test_daikonic_unchanged_by_occlusion_pass():
    """The bite trigger / gate must NOT fire false occlusion on a non-overlapping mark."""
    from pathlib import Path
    from PIL import Image
    fix = Path(__file__).parent / "fixtures" / "daikonic" / "source.png"
    icon = np.asarray(Image.open(fix).convert("RGB"))[:392]
    svg = idealize(icon, options=Options())
    assert "<use" in svg                                  # leaves still mirror
    assert svg.count("<circle") == 0                      # nothing wrongly completed to a disk

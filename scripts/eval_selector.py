"""Before/after eval for the selector harness (slice 4a). Measures idealize output
fidelity (render-ΔE, SSIM) and element counts per image, over a few synthetic
shapes plus the untracked real-logo corpus. Dev tool — NOT a CI test.

Run on master (baseline), then on feat/selector, and compare the printed tables.
Run: .venv/bin/python scripts/eval_selector.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vectoreyes import Options, idealize
from vectoreyes.color import mean_delta_e
from tests._render import render_svg, ssim

CORPUS = _REPO / "scratch" / "real-logos"


def _disk(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def _synthetic():
    out = []
    h = w = 120
    circ = np.full((h, w, 3), 255, np.uint8); circ[_disk(60, 60, 40, h, w)] = (30, 100, 235)
    out.append(("syn_circle", circ))
    sq = np.full((h, w, 3), 255, np.uint8); sq[30:90, 30:90] = (200, 30, 30)
    out.append(("syn_square", sq))
    return out


def _row(name, img):
    h, w = img.shape[:2]
    svg = idealize(img, options=Options())
    out = render_svg(svg, w, h)
    de = mean_delta_e(img, out)
    ss = ssim(img, out)
    print(f"  {name:28s} ΔE={de:.4f} SSIM={ss:.4f} "
          f"path={svg.count('<path')} circle={svg.count('<circle')} "
          f"rect={svg.count('<rect')} use={svg.count('<use')}")
    return de


def main() -> int:
    print("=== synthetic ===")
    des = [_row(n, img) for n, img in _synthetic()]
    if CORPUS.exists():
        print("=== real-logos (untracked) ===")
        for png in sorted(CORPUS.glob("*.png")):
            arr = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
            des.append(_row(png.name, arr))
    else:
        print(f"(no corpus at {CORPUS} — synthetic only)")
    print(f"\nmean render-ΔE over {len(des)} images: {sum(des) / len(des):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

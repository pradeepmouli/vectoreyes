"""Manual scorer eval over the untracked real-logo corpus (scratch/real-logos/).
Dev tool — NOT a CI test (brand assets are uncommitted). Prints the scorer's
ranked candidates per region for visual inspection.

Run: .venv/bin/python scripts/score_real_logos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so the uninstalled `tests` package is importable.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from PIL import Image

from vectoreyes.pipeline import _segment_image, Options
from vectoreyes.score import rank_candidates
from tests._candidates import generate_candidates

CORPUS = Path(__file__).resolve().parent.parent / "scratch" / "real-logos"


def main() -> int:
    if not CORPUS.exists():
        print(f"corpus not found: {CORPUS} (brand assets are local/untracked) — nothing to do")
        return 0
    pngs = sorted(CORPUS.glob("*.png"))
    if not pngs:
        print(f"no PNGs in {CORPUS}")
        return 0
    for png in pngs:
        arr = np.asarray(Image.open(png).convert("RGB"), dtype=np.uint8)
        w, h, regions = _segment_image(arr, Options())
        print(f"\n=== {png.name} ({len(regions)} regions) ===")
        for region in regions:
            cands = generate_candidates(region, arr)
            if not cands:
                continue
            ranked = rank_candidates(cands, arr, region)
            top, bd = ranked[0]
            fill = type(top.fill).__name__
            print(f"  region {region.label} area={region.area}: "
                  f"winner={top.geometry.kind}/{fill} "
                  f"ΔE={bd.delta_e:.4f} parsimony={bd.parsimony:.0f} "
                  f"(of {len(cands)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

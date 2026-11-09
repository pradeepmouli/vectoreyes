"""TDD: structured diagnostics schema on IdealizeReport.

Tests that report.diagnostics.to_dict() satisfies the agreed schema shape
on two representative inputs: a single disc and the daikonic fixture.
"""

import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vectoreyes.pipeline import Options, idealize


def _disc(n: int = 64) -> np.ndarray:
    im = Image.new("RGB", (n, n), "white")
    ImageDraw.Draw(im).ellipse((10, 10, n - 10, n - 10), fill=(30, 100, 220))
    return np.asarray(im, dtype=np.uint8)


def _daikonic() -> np.ndarray:
    src = Path(__file__).parent / "fixtures" / "daikonic" / "source.png"
    im = Image.open(src)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    return np.asarray(im.convert("RGB"), np.uint8)


def test_diagnostics_schema_shape_disk():
    """report.diagnostics.to_dict() has the agreed schema for a single disc."""
    svg, report = idealize(_disc(), report=True)
    assert report.diagnostics is not None

    d = report.diagnostics.to_dict()

    # Top-level keys
    assert set(d.keys()) >= {"options", "stats", "regions", "axes"}

    # options: every Options field is present
    opt_fields = {f.name for f in dataclasses.fields(Options)}
    assert opt_fields.issubset(set(d["options"].keys()))

    # stats
    s = d["stats"]
    assert s["regions"] == 1
    assert s["components"] == 1
    assert s["elements"] == 1
    assert s["gradients"] == 0
    assert isinstance(s["axes"], int)

    # regions list
    assert len(d["regions"]) == 1
    reg = d["regions"][0]
    assert set(reg.keys()) >= {"id", "area", "color_hex", "bbox", "symmetry", "strategies", "options"}
    assert isinstance(reg["area"], int) and reg["area"] > 0
    bbox = reg["bbox"]
    assert isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox)
    assert isinstance(reg["color_hex"], str) and reg["color_hex"].startswith("#")

    # per-region options mirrors global Options
    assert opt_fields.issubset(set(reg["options"].keys()))

    # symmetry sub-dict
    sym = reg["symmetry"]
    assert set(sym.keys()) >= {"role", "axis", "off_ratio", "partner"}
    assert sym["role"] in ("straddler", "pair", "loner")
    assert isinstance(sym["off_ratio"], float)

    # strategies: geom has exactly one chosen=True entry, and it is "primitive"
    geom = reg["strategies"]["geom"]
    assert len(geom) >= 1
    chosen = [k for k, v in geom.items() if v.get("chosen")]
    assert len(chosen) == 1, f"expected one chosen geom, got {chosen}"
    assert "primitive" in chosen, f"disc expected primitive, got {chosen}"

    # fill strategies present
    assert "fill" in reg["strategies"]
    fill_strats = reg["strategies"]["fill"]
    assert any(v.get("chosen") for v in fill_strats.values())


def test_diagnostics_daikonic_has_straddler():
    """Daikonic: ≥1 region with role==straddler and off_ratio/axis populated."""
    svg, report = idealize(_daikonic(), report=True)
    d = report.diagnostics.to_dict()

    roles = [r["symmetry"]["role"] for r in d["regions"]]
    assert "straddler" in roles, f"no straddler found: roles={roles}"

    # Every straddler/pair has a non-None axis dict with theta/cx/cy keys
    for reg in d["regions"]:
        role = reg["symmetry"]["role"]
        if role in ("straddler", "pair"):
            ax = reg["symmetry"]["axis"]
            assert ax is not None, f"region {reg['id']} role={role} has null axis"
            assert set(ax.keys()) >= {"theta", "cx", "cy"}
            assert isinstance(reg["symmetry"]["off_ratio"], float)

    # stats sanity
    s = d["stats"]
    assert s["regions"] >= 1
    assert s["components"] >= 1

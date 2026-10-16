#!/usr/bin/env python3
"""Compare trace-auto corpus SVGs against the existing current/optimizer files."""

from __future__ import annotations

import argparse
import html
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import resvg_py
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _source_rgb(path: Path) -> np.ndarray:
    rgba = Image.open(path).convert("RGBA")
    background = Image.new("RGB", rgba.size, "white")
    background.paste(rgba, mask=rgba.getchannel("A"))
    return np.asarray(background, dtype=np.float32)


def _render_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    png = resvg_py.svg_to_bytes(svg_string=path.read_text(), width=size[0], height=size[1])
    rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    background = Image.new("RGB", rgba.size, "white")
    background.paste(rgba, mask=rgba.getchannel("A"))
    return np.asarray(background, dtype=np.float32)


def _mae(source: np.ndarray, svg: Path) -> float:
    rendered = _render_rgb(svg, (source.shape[1], source.shape[0]))
    return float(np.abs(rendered - source).mean())


def _trace_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    report = payload.get("report", {})
    targets = report.get("targets", [])
    geometry = Counter(target.get("geometry", "unknown") for target in targets)
    diagnostics = [target.get("diagnostics") or {} for target in targets]
    return {
        "targets": len(targets),
        "geometry": dict(sorted(geometry.items())),
        "symmetry": sum("symmetry" in item for item in diagnostics),
        "clones": sum("clones" in item or "clone" in item for item in diagnostics),
        "stitched": sum("stitch" in item for item in diagnostics),
    }


def _flags(name: str, variants: dict[str, dict[str, float]], trace: dict[str, Any]) -> list[str]:
    prior = min(variants["current"]["mae"], variants["optimizer"]["mae"])
    flags: list[str] = []
    if variants["trace_auto"]["mae"] > prior * 1.15:
        flags.append("fidelity regression vs best existing")
    if name == "apple_music" and not trace["clones"]:
        flags.append("inspect connected-note clone decomposition")
    if name == "vbird":
        if trace["geometry"].get("circle", 0) < 3:
            flags.append("orange endpoint circle remains part of compound region")
        if not trace["clones"]:
            flags.append("inspect connected-shape clone decomposition")
    return flags


def _row(corpus: Path, source_path: Path) -> dict[str, Any] | None:
    name = source_path.stem
    svg_dir = corpus / "svg"
    diagnostics = corpus / "diagnostics" / f"trace_auto-{name}.json"
    paths = {
        mode: svg_dir / f"{mode}-{name}.svg"
        for mode in ("current", "optimizer", "trace_auto")
    }
    if not diagnostics.exists() or any(not path.exists() for path in paths.values()):
        return None
    source = _source_rgb(source_path)
    variants = {
        mode: {"mae": round(_mae(source, path), 3), "bytes": path.stat().st_size}
        for mode, path in paths.items()
    }
    trace = _trace_summary(diagnostics)
    return {
        "name": name,
        "source_size": [source.shape[1], source.shape[0]],
        "variants": variants,
        "trace": trace,
        "flags": _flags(name, variants, trace),
    }


def _html(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        name = html.escape(row["name"])
        variants = row["variants"]
        panels = ['<figure><figcaption>source</figcaption><img src="input/%s.png"></figure>' % name]
        for mode in ("current", "optimizer", "trace_auto"):
            label = mode.replace("_", " ")
            panels.append(
                '<figure><figcaption>%s · MAE %s</figcaption><object data="svg/%s-%s.svg" type="image/svg+xml"></object></figure>'
                % (label, variants[mode]["mae"], mode, name)
            )
        trace = row["trace"]
        flags = "<br>".join(html.escape(flag) for flag in row["flags"]) or "—"
        cards.append(
            '<article id="%s"><h2>%s</h2><div class="renders">%s</div>'
            '<p><b>trace-auto:</b> %s targets; %s; symmetry=%s; clones=%s; stitched=%s.<br>'
            '<b>review:</b> %s</p></article>'
            % (
                name,
                name,
                "".join(panels),
                trace["targets"],
                html.escape(str(trace["geometry"])),
                trace["symmetry"],
                trace["clones"],
                trace["stitched"],
                flags,
            )
        )
    return """<!doctype html><meta charset=\"utf-8\"><title>Trace-auto corpus comparison</title>
<style>body{font:14px system-ui;margin:20px;background:#fafafa}article{background:#fff;padding:14px;margin:16px 0;border:1px solid #ddd}.renders{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}figure{margin:0}figcaption{font-weight:600;margin-bottom:5px}img,object{width:100%;height:220px;background:repeating-conic-gradient(#eee 0 25%,#fff 0 50%) 50%/16px 16px;border:1px solid #eee}</style>
<h1>Trace-auto vs existing corpus outputs</h1><p>MAE is rendered RGB error against the corpus source after compositing transparency over white. Flags are triage prompts, not automatic verdicts.</p>""" + "".join(cards)


def main() -> None:
    args = _parse_args()
    corpus = args.corpus.resolve()
    html_path = args.html or corpus / "trace_auto_comparison.html"
    json_path = args.json or corpus / "trace_auto_comparison.json"
    rows = [row for source in sorted((corpus / "input").glob("*.png")) if (row := _row(corpus, source))]
    html_path.write_text(_html(rows))
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(html_path)
    print(json_path)


if __name__ == "__main__":
    main()

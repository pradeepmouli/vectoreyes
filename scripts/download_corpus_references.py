"""Collect reviewable SVG references using the existing corpus byte downloader.

Run with PYTHONPATH=src python scripts/download_corpus_references.py.
Never overwrites an existing SVG or its review record. The URL manifest is
separate from corpus/sources.json, which controls raster tracing inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_corpus_html import _download_source


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_svg(path: Path) -> ET.Element:
    """Require self-contained vector artwork, not executable or raster content."""
    raw = path.read_text(encoding="utf-8-sig")
    if "<!ENTITY" in raw.upper():
        raise ValueError("SVG contains entity declarations")
    root = ET.fromstring(raw)
    if root.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("Downloaded file is not an SVG")
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] in {"script", "image", "foreignObject"}:
            raise ValueError("SVG contains active or embedded raster content")
        for key, value in node.attrib.items():
            local = key.rsplit("}", 1)[-1]
            if local.lower().startswith("on"):
                raise ValueError("SVG contains an event handler")
            if local == "href" and not value.startswith("#"):
                raise ValueError("SVG references an external asset")
    if re.search(r"@import", raw, re.I):
        raise ValueError("SVG imports external styles")
    for value in re.findall(r"url\(([^)]+)\)", raw, flags=re.I):
        if not value.strip(" \t\r\n\"'").startswith("#"):
            raise ValueError("SVG references an external URL")
    return root


def acquire(item: dict, directory: Path, cache: Path) -> str:
    name = item["name"]
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("Unsafe corpus entry name")
    svg, record = directory / f"{name}.svg", directory / f"{name}.json"
    if svg.exists() or record.exists():
        if not (svg.exists() and record.exists()):
            raise ValueError("Incomplete reference; refusing to overwrite")
        if digest(svg) != json.loads(record.read_text())["sha256"]:
            raise ValueError("Existing reference hash mismatch")
        validate_svg(svg)
        return "preserved"
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "imageinfo",
        "iiprop": "url|extmetadata|timestamp|sha1", "titles": item["title"], "redirects": 1,
    })
    url = "https://commons.wikimedia.org/w/api.php?" + query
    # Include title in the cache key so changing a candidate cannot reuse another file.
    key = hashlib.sha256(item["title"].encode()).hexdigest()[:16]
    metadata = json.loads(_download_source(url, cache / f"{key}.json").read_text())
    pages = list(metadata["query"]["pages"].values())
    if len(pages) != 1 or not pages[0].get("imageinfo"):
        raise ValueError("Commons title has no downloadable file")
    info = pages[0]["imageinfo"][0]
    download = info["url"]
    parsed = urllib.parse.urlsplit(download)
    if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org":
        raise ValueError("Unexpected download host")
    candidate = _download_source(download, cache / f"{key}.svg")
    root = validate_svg(candidate)
    raster = ROOT / "corpus" / "input" / f"{name}.png"
    if not raster.is_file():
        raise ValueError("Missing associated corpus raster")
    values = info.get("extmetadata", {})
    provenance = {
        "kind": "downloaded_vector_candidate",
        "page": info["descriptionurl"], "title": pages[0]["title"],
        "download": download, "retrieved_on": datetime.now(timezone.utc).date().isoformat(),
        "upstream_timestamp": info.get("timestamp"), "upstream_sha1": info.get("sha1"),
        "modified": False,
        "metadata_attribution": "The following credit/license fields are supplied by Wikimedia Commons; original vector authorship is not independently verified.",
    }
    for key in ("Artist", "Credit", "LicenseShortName", "LicenseUrl", "UsageTerms", "Restrictions"):
        provenance[key] = values.get(key, {}).get("value")
    result = {
        "entry": name, "svg": svg.name, "revision": 1, "sha256": digest(candidate),
        "provenance": provenance,
        "review": {"status": "draft", "reviewer": None},
        "canvas": {"viewBox": [float(x) for x in root.get("viewBox", "").replace(",", " ").split()],
                   "width": root.get("width"), "height": root.get("height")},
        "raster": {"path": f"../input/{name}.png", "sha256": digest(raster)},
        "registration": {"status": "unverified"},
        "notes": ["Candidate only: compare the exact variant, colors, framing and geometry before approval."],
    }
    # Exclusive creates prevent an accidental re-run from replacing approved work.
    with svg.open("xb") as output:
        output.write(candidate.read_bytes())
    with record.open("x") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    return "downloaded draft"


def write_review(directory: Path, manifest: dict, errors: dict[str, str]) -> None:
    cards = []
    for record in sorted(directory.glob("*.json")):
        data = json.loads(record.read_text())
        if "entry" not in data:
            continue
        name = data["entry"]
        if not (directory / data["svg"]).is_file():
            continue
        page = data["provenance"].get("page")
        source = f'<a href="{html.escape(page, quote=True)}">Source page</a> · ' if page else ""
        cards.append(f'<section><h2>{html.escape(name)} — {html.escape(data["review"]["status"])}</h2>'
                     f'<div class="pair"><figure><a href="../input/{name}.png"><img src="../input/{name}.png"></a><figcaption>Corpus raster</figcaption></figure>'
                     f'<figure><a href="{name}.svg"><img src="{name}.svg"></a><figcaption>Reference candidate</figcaption></figure></div>'
                     f'<p>{source}<a href="{name}.json">Provenance / review</a></p></section>')
    unresolved = "".join(f'<li>{html.escape(x["name"])}: {html.escape(x["reason"])}</li>' for x in manifest.get("unresolved", []))
    failures = "".join(f'<li>{html.escape(k)}: {html.escape(v)}</li>' for k, v in errors.items())
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Corpus reference review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#eee;color:#222}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;height:260px;object-fit:contain;background:repeating-conic-gradient(#ddd 0% 25%,#fff 0% 50%) 0/20px 20px}figcaption{padding:8px}a{color:#174ba0}</style>
<h1>Corpus reference review</h1><p>Drafts are not approved targets. Compare source variant, framing, color and geometry. Images link to the actual files. Registration and automated scoring are not yet verified.</p>'''
    (directory / "index.html").write_text(document + "".join(cards) + '<h2>Unresolved / custom artwork</h2><ul>' + unresolved + '</ul><h2>Download errors</h2><ul>' + failures + '</ul></html>')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "corpus/reference/sources.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    directory = args.manifest.parent
    cache = ROOT / "corpus/cache/reference-downloads"
    errors = {}
    for item in manifest["entries"]:
        try:
            try:
                status = acquire(item, directory, cache)
            except urllib.error.HTTPError as error:
                if error.code != 429:
                    raise
                # Do not hammer a rate-limited source. Honor numeric Retry-After;
                # leave longer/date-based waits for a later invocation.
                delay = error.headers.get("Retry-After", "30")
                if not delay.isdigit() or int(delay) > 60:
                    raise
                print(f'Rate limited; waiting {delay}s before one retry', flush=True)
                time.sleep(max(1, int(delay)))
                status = acquire(item, directory, cache)
            print(f'{item["name"]}: {status}', flush=True)
            if status != "preserved":
                time.sleep(3)
        except Exception as error:
            errors[item["name"]] = str(error)
            print(f'{item["name"]}: ERROR {error}', flush=True)
    write_review(directory, manifest, errors)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

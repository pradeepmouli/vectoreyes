"""Shared non-mutating SVG/PNG views, reusing canonical exported map elements."""
from __future__ import annotations

import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .bundle import atomic_json
from .models import ViewSpec

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)


def safe_svg(svg: str) -> ET.Element:
    if "<!DOCTYPE" in svg.upper() or "<!ENTITY" in svg.upper():
        raise ValueError("SVG entities are not allowed in review artifacts")
    root = ET.fromstring(svg)
    allowed = {"svg", "g", "defs", "path", "rect", "circle", "ellipse", "polygon", "polyline", "line", "use", "mask", "clipPath", "linearGradient", "radialGradient", "stop", "pattern", "image", "text", "tspan", "title", "desc"}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] not in allowed:
            raise ValueError("unsupported executable or external SVG element")
        for key, value in element.attrib.items():
            local = key.rsplit("}", 1)[-1].lower()
            if local.startswith("on") or local == "style":
                raise ValueError("inline script/style is not allowed in review SVG")
            if local == "href" and not (value.startswith("#") or value.startswith("data:image/png;base64,") or value.startswith("data:image/jpeg;base64,")):
                raise ValueError("external SVG references are not allowed")
            if re.search(r"url\(\s*['\"]?(?!#)[^)]", value, re.I):
                raise ValueError("external SVG paint references are not allowed")
    return root


def neutral_svg(svg: str, *, outline: bool = False) -> str:
    root = safe_svg(svg)

    def visit(element: ET.Element, definition: bool = False) -> None:
        definition = definition or element.tag.rsplit("}", 1)[-1] in {"defs", "mask", "clipPath"}
        if not definition:
            for opacity in ("opacity", "fill-opacity", "stroke-opacity"):
                if opacity in element.attrib:
                    element.set(opacity, "1")
            if element.get("fill") not in {None, "none"}:
                element.set("fill", "#c4cec8")
                if outline:
                    element.set("stroke", "#657b6d")
                    element.set("stroke-width", "1")
            if element.get("stroke") not in {None, "none"}:
                element.set("stroke", "#51645a")
        for child in element:
            visit(child, definition)
    visit(root)
    return ET.tostring(root, encoding="unicode")


def region_viewport(inspection: dict[str, Any], region_ids: list[str], padding: float = 18) -> tuple[float, float, float, float]:
    from ..skia_geometry import SkPath
    bounds = []
    for region in inspection["regions"]:
        if region["id"] in region_ids:
            for path in region["paths"]:
                if path.get("d"):
                    bounds.append(SkPath.from_svg_d(path["d"]).bounds)
    if not bounds:
        raise ValueError("no exact local path bounds; use an explicit shared viewport")
    x, y = min(b[0] for b in bounds)-padding, min(b[1] for b in bounds)-padding
    right, bottom = max(b[2] for b in bounds)+padding, max(b[3] for b in bounds)+padding
    return x, y, right-x, bottom-y


def export_view(svg: str, spec: ViewSpec, output: Path, *, path_map: str | None = None, inspection: dict[str, Any] | None = None) -> dict[str, Any]:
    import resvg_py
    root = safe_svg(neutral_svg(svg, outline=True) if spec.track == "shape" else svg)
    viewport = spec.viewport or tuple(float(x) for x in root.get("viewBox", "0 0 1254 1254").split())
    root.set("viewBox", " ".join(map(str, viewport)))
    root.set("width", str(spec.width))
    root.set("height", str(round(spec.width * viewport[3]/viewport[2])))
    if path_map and spec.track == "shape":
        for element in safe_svg(path_map).iter():
            command = element.get("data-command-id")
            feature = element.get("data-anchor-id") or element.get("data-anchor-ref")
            identity = command or feature
            if not identity or (spec.region_ids and not any(identity.startswith(r+".") for r in spec.region_ids)):
                continue
            if (command and "commands" in spec.layers) or (feature and "features" in spec.layers):
                root.append(copy.deepcopy(element))
    if inspection and spec.track == "shape":
        for region in inspection["regions"]:
            if spec.region_ids and region["id"] not in spec.region_ids:
                continue
            if "paths" in spec.layers:
                for path in region["paths"]:
                    if path.get("d"):
                        ET.SubElement(root, f"{{{NS}}}path", {"d": path["d"], "fill": "none", "stroke": "#46685a", "stroke-width": "2", "data-path-id": path["id"]})
            if "regions" in spec.layers:
                try:
                    box = region_viewport(inspection, [region["id"]], 0)
                except ValueError:
                    continue
                text = ET.SubElement(root, f"{{{NS}}}text", {"x": str(box[0]+box[2]/2), "y": str(box[1]+box[3]/2), "font-size": "18", "fill": "#173c2e", "stroke": "white", "stroke-width": "3", "paint-order": "stroke", "data-region-id": region["id"]})
                text.text = region["id"]
        if "source" in spec.layers:
            for path in inspection["source_paths"]:
                if spec.region_ids and path["region_id"] not in spec.region_ids:
                    continue
                points = " ".join(",".join(map(str, c["end"])) for c in path["commands"])
                ET.SubElement(root, f"{{{NS}}}polyline", {"points": points, "fill": "none", "stroke": "#8b526d", "stroke-width": "1", "data-source-path-id": path["source_path_id"], "data-source-region-id": path["region_id"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = ET.tostring(root, encoding="unicode")
    output.with_suffix(".svg").write_text(encoded)
    import io
    from PIL import Image
    rgba = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=encoded)))).convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    white.alpha_composite(rgba)
    white.convert("RGB").save(output.with_suffix(".png"))
    metadata = {**spec.model_dump(mode="json"), "viewport": list(viewport), "inspection": inspection,
                "source_correspondence": "retained_only" if inspection else "unmapped"}
    atomic_json(output.with_suffix(".json"), metadata)
    return metadata

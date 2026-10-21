"""Offline report writer. Review metadata is not a second drawing store."""
from __future__ import annotations

import base64
import json
import io
from importlib.resources import files
from pathlib import Path
from typing import Any

from .bundle import RunBundle
from .models import ViewSpec
from .views import export_view, safe_svg
from .inspection import _fields, build_inspection, compare_inspections


def parameter_snapshot(bundle: RunBundle, version: str) -> dict[str, Any]:
    """Expose retained provenance, never re-resolve historical settings today."""
    def read(relative: str) -> dict[str, Any]:
        path = bundle.artifact_path(relative)
        return json.loads(path.read_text()) if path.exists() else {}

    response_path = f"versions/{version}/response.json"
    response = read(response_path)
    baseline_path = f"versions/{bundle.load().baseline_version}/response.json"
    baseline = read(baseline_path)
    trace_source = response if response.get("trace_options") is not None else baseline
    trace_path = response_path if response.get("trace_options") is not None else baseline_path
    retained = _fields(read(f"versions/{version}/plan.json").get("plan", {}))
    recipe = retained.get("effective_recipe") or {}
    actions = []
    path_fits = []
    for step in recipe.get("steps", []):
        if "action" in step:
            actions.append({"id": step.get("id"), "op": step["action"].get("op"),
                            "target": step["action"].get("target"), "settings": step.get("settings"),
                            "action": step["action"], "status": step.get("status")})
        if "recipe" in step:
            path_fits.append({"target_region_id": step.get("target_region_id"),
                              "target_path_id": step.get("target_path_id"),
                              "options": step["recipe"].get("options")})
    return {"version": version,
            "trace_options": {"value": trace_source.get("trace_options"), "artifact": trace_path},
            "observed_trace_options": trace_source.get("trace", {}).get("options"),
            "effective_fit_settings": recipe.get("settings"),
            "effective_actions": actions, "source_path_fit_options": path_fits,
            "submitted_plan": retained.get("submitted_plan"),
            "output_settings": response.get("report", {}).get("output_settings"),
            "recipe_artifact": f"versions/{version}/plan.json",
            "availability": "Only captured fields are shown; null means unavailable, not a default."}


def write_report(bundle: RunBundle) -> Path:
    run = bundle.load()
    versions: dict[str, Any] = {}
    trials = [bundle.load_trial(identity).model_dump(mode="json") for identity in run.trial_ids]
    for trial in trials:
        # Exact requests stay in the trial file. Do not duplicate the source's
        # entire base64 payload in a visible request preview.
        image_data = trial["request"].get("image_data")
        if isinstance(image_data, dict) and "base64" in image_data:
            trial["request"] = {**trial["request"], "image_data": {"saved_source": "original.png", "sha256": run.source_sha256,
                                "exact_request": f"trials/{trial['id']}.json"}}
    for version in dict.fromkeys(t["candidate_version"] for t in trials if t["candidate_version"]):
        prefix = f"versions/{version}"
        svg_file = bundle.artifact_path(f"{prefix}/svg.svg")
        response_file = bundle.artifact_path(f"{prefix}/response.json")
        if not svg_file.exists() or not response_file.exists():
            continue
        response = json.loads(response_file.read_text())
        def captured(kind):
            path = bundle.artifact_path(f"{prefix}/{kind}.json")
            return json.loads(path.read_text()) if path.exists() else {}
        # Recompute derived views from captured engine output, not a potentially
        # stale derived inspection. Never rewrite the immutable capture files.
        svg = svg_file.read_text()
        inspection = build_inspection(response, captured('raw_trace'), captured('plan'))
        safe_svg(svg)
        mapping = bundle.artifact_path(f"{prefix}/path_map.svg")
        out = bundle.artifact_path(f"{prefix}/inspector")
        export_view(svg, ViewSpec(layers=["regions", "paths", "commands", "features", "source"]), out,
                    path_map=mapping.read_text() if mapping.exists() else None, inspection=inspection)
        import xml.etree.ElementTree as ET
        versions[version] = {"inspection": inspection, "svg": ET.tostring(safe_svg(svg), encoding="unicode"),
                               "inspector_svg": out.with_suffix(".svg").read_text(), "file": f"{prefix}/svg.svg",
                                "parameters": parameter_snapshot(bundle, version),
                                "parent_version": response.get('parent_version')}
        # Use the exact current geometry's coverage for both source and paint
        # inspection. This is a comparison support mask, not a new segmentation.
        if bundle.artifact_path("source.png").exists():
            import resvg_py
            from PIL import Image
            from .views import neutral_svg
            mask = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=neutral_svg(svg))))).convert("RGBA")
            with Image.open(bundle.artifact_path("source.png")) as image:
                reference = image.convert("RGBA")
            if reference.size == mask.size:
                reference.putalpha(mask.getchannel("A"))
                buffer = io.BytesIO(); reference.save(buffer, format="PNG")
                bundle.write_artifact(f"{prefix}/fill-reference.png", buffer.getvalue())
                versions[version]["fill_reference"] = "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()
                versions[version]["fill_support"] = "current_geometry_coverage"
            else:
                versions[version]["fill_support"] = "unavailable_source_canvas_mismatch"
    for value in versions.values():
        value['comparisons'] = {name: compare_inspections(versions[reference]['inspection'], value['inspection'])
            if reference in versions else None
            for name, reference in (('baseline', run.baseline_version), ('parent', value['parent_version']))}
    source = bundle.artifact_path("source.png")
    data = {"run": run.model_dump(mode="json"), "trials": trials, "versions": versions,
            "source": "data:image/png;base64,"+base64.b64encode(source.read_bytes()).decode() if source.exists() else None}
    assets = files("vectormark").joinpath("assets/evaluation")
    template = assets.joinpath("report.html").read_text()
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = template.replace("<!--STYLE-->", "<style>"+assets.joinpath("report.css").read_text()+"</style>")
    html = html.replace("<!--DATA-->", '<script id="bundle-data" type="application/json">'+encoded+'</script>')
    html = html.replace("<!--SCRIPT-->", "<script>"+assets.joinpath("report.js").read_text()+"</script>")
    target = bundle.artifact_path("index.html")
    target.write_text(html)
    return target

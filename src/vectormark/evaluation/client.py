"""Headless orchestration over the public MCP surface, without automatic retries."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .bundle import RunBundle, atomic_json
from .inspection import build_inspection
from .models import EvaluationRun, TrialRecord, ViewSpec
from .views import export_view


class ToolClient(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any: ...


def error_kind(message: str) -> str:
    if "DRAWING_NOT_FOUND" in message:
        return "expired_handle"
    if "exceeds max_error" in message:
        return "fit_guard"
    if "revision" in message or "replay" in message:
        return "replay_provenance"
    if "PLAN_INVALID" in message or "validation" in message or "INVALID_ARGUMENT" in message:
        return "schema_or_address"
    return "transport_or_internal"


def payload(result: Any) -> dict[str, Any]:
    if getattr(result, "is_error", False):
        raise ValueError(str(result.content))
    value = getattr(result, "structured_content", None) or getattr(result, "data", None)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        raise ValueError("tool returned no structured artifact data")
    return value


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()


async def capture_version(bundle: RunBundle, client: ToolClient, response: dict[str, Any]) -> dict[str, str]:
    drawing, version = response["drawing_id"], response["version"]
    prefix = f"versions/{version}"
    bundle.write_artifact(f"{prefix}/response.json", _json_bytes(response))
    errors: dict[str, str] = {}
    artifacts: dict[str, Any] = {}
    hashes: dict[str, object] = {}
    for kind in ("svg", "labeled_svg", "path_map", "command_map", "raw_trace", "plan", "preview", "review_panel"):
        try:
            result = await client.call_tool("get_drawing_artifact", {"drawing_id": drawing, "version": version, "artifact": kind})
            value = payload(result)
            artifacts[kind] = value
            hashes[kind+"_json"] = bundle.write_artifact(f"{prefix}/{kind}.json", _json_bytes(value))
            if isinstance(value.get("svg"), str):
                hashes[kind] = bundle.write_artifact(f"{prefix}/{kind}.svg", value["svg"].encode())
            for content in result.content:
                if content.type == "image":
                    hashes[kind] = bundle.write_artifact(f"{prefix}/{kind}.png", base64.b64decode(content.data))
        except Exception as exc:
            errors[kind] = str(exc)
    snapshot = build_inspection(response, artifacts.get("raw_trace", {}), artifacts.get("plan", {}))
    bundle.write_artifact(f"{prefix}/inspection.json", _json_bytes(snapshot))
    if "svg" in artifacts:
        for name, spec in (("shape", ViewSpec(layers=[])), ("annotated", ViewSpec(layers=["regions", "commands", "features"])), ("fill", ViewSpec(track="fill", layers=[]))):
            try:
                export_view(artifacts["svg"]["svg"], spec, bundle.artifact_path(f"{prefix}/{name}"),
                            path_map=artifacts.get("path_map", {}).get("svg"), inspection=snapshot)
            except Exception as exc:
                errors[name] = str(exc)
    hashes["inspection"] = {"path": f"{prefix}/inspection.json", "sha256": hashlib.sha256(bundle.artifact_path(f"{prefix}/inspection.json").read_bytes()).hexdigest()}
    atomic_json(bundle.artifact_path(f"{prefix}/capture.json"), {"artifacts": hashes, "errors": errors})
    return errors


async def refine_trial(bundle: RunBundle, client: ToolClient, plan: dict[str, Any], hypothesis: str) -> TrialRecord:
    run = bundle.load()
    if plan.get("drawing_id") != run.drawing_id:
        raise ValueError("plan drawing_id does not match this run")
    trial = TrialRecord(id=f"t{len(run.trial_ids)+1:04d}", base_version=plan.get("base_version"), hypothesis=hypothesis, request=plan)
    bundle.save_trial(trial)  # Persist before the first potentially mutating call.
    started = time.perf_counter()
    try:
        result = payload(await client.call_tool("refine_drawing", {"plan": plan}))
    except Exception as exc:
        message = str(exc)
        kind = error_kind(message)
        trial.execution = "unknown" if kind == "transport_or_internal" else "failed"
        trial.error, trial.error_kind, trial.disposition = message, kind, "blocked"
    else:
        trial.execution, trial.candidate_version = "succeeded", result["version"]
        bundle.save_trial(trial)  # Publication is known even if later capture fails.
        try:
            trial.artifact_errors = await capture_version(bundle, client, result)
        except Exception as exc:
            trial.artifact_errors = {"capture": str(exc)}
    trial.elapsed_seconds = time.perf_counter()-started
    bundle.save_trial(trial)
    return trial


async def trace_run(root: Path, client: ToolClient, image: Path, options: dict[str, Any], title: str) -> RunBundle:
    from ..mcp_image import preprocess_image
    source = image.read_bytes()
    with Image.open(io.BytesIO(source)) as opened:
        if opened.format != "PNG":
            raise ValueError("evaluation inputs must be PNG")
    run = EvaluationRun(source_sha256=hashlib.sha256(source).hexdigest(), drawing_id="", baseline_version="v0", title=title, options=options)
    bundle = RunBundle.create(root, run)
    bundle.write_artifact("original.png", source)
    rgb, _meta = preprocess_image(source, **options.get("preprocess", {}))
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG")
    bundle.write_artifact("source.png", buf.getvalue())
    request = {"image_data": {"base64": base64.b64encode(source).decode()}, "options": options}
    trial = TrialRecord(id="t0000", base_version=None, hypothesis="Retained baseline, before plan edits", request=request)
    bundle.save_trial(trial)
    started = time.perf_counter()
    try:
        if hasattr(client, "list_tools"):
            schemas = {tool.name: tool.input_schema for tool in await client.list_tools()}
            bundle.write_artifact("tool-schemas.json", _json_bytes(schemas))
        result = payload(await client.call_tool("trace_drawing", request))
    except Exception as exc:
        trial.error, trial.error_kind, trial.disposition = str(exc), error_kind(str(exc)), "blocked"
        trial.execution = "unknown" if trial.error_kind == "transport_or_internal" else "failed"
    else:
        run = bundle.load()
        run.drawing_id, run.baseline_version = result["drawing_id"], "v0"
        run.build = {"server_identity": "not_exposed", "schema_sha256": hashlib.sha256(bundle.artifact_path("tool-schemas.json").read_bytes()).hexdigest() if bundle.artifact_path("tool-schemas.json").exists() else None}
        run.options = result.get("trace_options", options)
        bundle.save(run)
        bundle.write_artifact("trace-response.json", _json_bytes(result))
        for index, output in enumerate(result["outputs"]):
            receipt = trial if index == 0 else TrialRecord(
                id=f"t{index:04d}", base_version=output.get("parent_version"),
                hypothesis=output.get("label", "Trace output"),
                request={"trace_output_receipt": {"request_trial": trial.id,
                    "version": output["version"], "label": output.get("label")}})
            receipt.execution, receipt.candidate_version = "succeeded", output["version"]
            response = {**result, **output}
            try:
                receipt.artifact_errors = await capture_version(bundle, client, response)
            except Exception as exc:
                receipt.artifact_errors = {"capture": str(exc)}
            bundle.save_trial(receipt)
    trial.elapsed_seconds = time.perf_counter()-started
    bundle.save_trial(trial)
    return bundle

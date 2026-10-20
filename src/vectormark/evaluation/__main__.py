"""Run with ``python -m vectormark.evaluation --help``."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from .bundle import RunBundle, atomic_json
from .client import capture_version, refine_trial, trace_run
from .inspection import build_inspection
from .models import EvaluationRun, ReviewVerdict, TrialRecord, ViewSpec
from .report import write_report
from .views import export_view, region_viewport


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def import_snapshot(root: Path, snapshot: Path, source: Path, title: str) -> RunBundle:
    """Import recorded evidence, not a resumable engine session or new execution."""
    response = read_json(snapshot/"response.json")
    request = read_json(snapshot/"request.json") if (snapshot/"request.json").exists() else {}
    original = source.read_bytes()
    run = EvaluationRun(source_sha256=hashlib.sha256(original).hexdigest(), drawing_id=response["drawing_id"],
                        baseline_version=response["version"], title=title,
                        options=request.get("options", {}), build={"origin": "imported_recorded_evidence", "server_identity": "not_recorded"})
    bundle = RunBundle.create(root, run)
    bundle.write_artifact("original.png", original)
    from ..mcp_image import preprocess_image
    rgb, _ = preprocess_image(original, **run.options.get("preprocess", {}))
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG")
    bundle.write_artifact("source.png", buf.getvalue())
    prefix = f"versions/{run.baseline_version}"
    for path in snapshot.iterdir():
        if path.is_file() and path.suffix in {".json", ".svg", ".png"}:
            bundle.write_artifact(f"{prefix}/{path.name}", path.read_bytes())
    raw = read_json(snapshot/"raw_trace.json") if (snapshot/"raw_trace.json").exists() else {}
    provenance = read_json(snapshot/"plan.json") if (snapshot/"plan.json").exists() else {}
    bundle.write_artifact(f"{prefix}/inspection.json", json.dumps(build_inspection(response, raw, provenance)).encode())
    trial = TrialRecord(id="t0000", base_version=None, hypothesis="Imported recorded baseline; not a new tool execution",
                        request=request, execution="succeeded", candidate_version=run.baseline_version)
    bundle.save_trial(trial)
    return bundle


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Plan-driven SVG evaluation, separate shape and fill reviews")
    commands = cli.add_subparsers(dest="command", required=True)
    trace = commands.add_parser("trace"); trace.add_argument("source", type=Path); trace.add_argument("run", type=Path)
    trace.add_argument("--options", type=Path); trace.add_argument("--title", default="Vector artwork")
    trace.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    imported = commands.add_parser("import"); imported.add_argument("snapshot", type=Path); imported.add_argument("source", type=Path); imported.add_argument("run", type=Path); imported.add_argument("--title", default="Vector artwork")
    refine = commands.add_parser("refine"); refine.add_argument("run", type=Path); refine.add_argument("plan", type=Path); refine.add_argument("--hypothesis", required=True); refine.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    capture = commands.add_parser("capture"); capture.add_argument("run", type=Path); capture.add_argument("response", type=Path); capture.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    reconcile = commands.add_parser("reconcile"); reconcile.add_argument("run", type=Path); reconcile.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    for name in ("report", "accept", "reject", "review", "view"):
        p = commands.add_parser(name); p.add_argument("run", type=Path)
        if name in {"accept", "reject", "review"}:p.add_argument("trial")
        if name == "reject":p.add_argument("--reason", required=True)
        if name == "review":
            p.add_argument("--track", choices=["shape", "fill"], required=True);p.add_argument("--verdict", choices=["pass", "fail", "not_evaluated"], required=True)
            p.add_argument("--reason", required=True);p.add_argument("--reviewer", required=True);p.add_argument("--evidence", action="append", default=[])
        if name == "view":
            p.add_argument("--version", required=True);p.add_argument("--region", action="append", default=[]);p.add_argument("--layers", nargs="*", default=["regions", "commands", "features"])
            p.add_argument("--track", choices=["shape", "fill"], default="shape");p.add_argument("--viewport", nargs=4, type=float);p.add_argument("--name", default="selected-view")
    return cli


async def network(args: argparse.Namespace) -> RunBundle:
    from fastmcp import Client
    async with Client(args.endpoint, timeout=300) as client:
        if args.command == "trace":
            options = read_json(args.options) if args.options else {"refine": "none", "fit_strategy": "progressive", "trace_level": "subpixel"}
            if options.get("refine", "none") != "none":
                raise ValueError("start evaluation with refine='none'; evaluate automatic refinement in a separate explicit trial")
            options["refine"] = "none"
            return await trace_run(args.run, client, args.source, options, args.title)
        bundle = RunBundle(args.run)
        if args.command == "refine":
            trial = await refine_trial(bundle, client, read_json(args.plan), args.hypothesis)
            print(trial.model_dump_json(indent=2))
        elif args.command == "capture":
            response = read_json(args.response)
            if response.get("drawing_id") != bundle.load().drawing_id:
                raise ValueError("response does not belong to this run")
            errors = await capture_version(bundle, client, response)
            for identity in bundle.load().trial_ids:
                trial = bundle.load_trial(identity)
                if trial.candidate_version == response["version"]:
                    trial.artifact_errors = errors;bundle.save_trial(trial)
        else:
            result = await client.call_tool("get_drawing_artifact", {"drawing_id": bundle.load().drawing_id, "artifact": "versions"})
            from .client import payload
            atomic_json(bundle.artifact_path("reconciliation.json"), payload(result))
            print("Version manifest saved; no unknown request was retried or assigned a candidate.")
        return bundle


def main() -> None:
    args = parser().parse_args()
    if args.command in {"trace", "refine", "capture", "reconcile"}:
        bundle = asyncio.run(network(args))
    elif args.command == "import":
        bundle = import_snapshot(args.run, args.snapshot, args.source, args.title)
    else:
        bundle = RunBundle(args.run)
        if args.command == "review":
            trial = bundle.load_trial(args.trial)
            if trial.execution != "succeeded":
                raise ValueError("unpublished/unknown trials cannot receive visual verdicts")
            setattr(trial, args.track, ReviewVerdict(status=args.verdict, reason=args.reason, reviewer=args.reviewer, evidence=args.evidence))
            trial.review_history.append({"track": args.track, "verdict": getattr(trial, args.track).model_dump(mode="json")})
            bundle.save_trial(trial)
        elif args.command == "accept":bundle.promote(args.trial)
        elif args.command == "reject":
            from .models import now
            trial = bundle.load_trial(args.trial)
            run = bundle.load()
            if run.accepted_version == trial.candidate_version:
                run.accepted_version = None
            run.decisions.append({"action": "reject", "trial_id": trial.id, "reason": args.reason, "recorded_at": now()})
            bundle.save(run)
            trial.disposition="rejected";trial.reason=args.reason;bundle.save_trial(trial)
        elif args.command == "view":
            prefix=f"versions/{args.version}"
            inspection=read_json(bundle.artifact_path(f"{prefix}/inspection.json"))
            box=tuple(args.viewport) if args.viewport else region_viewport(inspection,args.region) if args.region else None
            spec=ViewSpec(track=args.track,region_ids=args.region,layers=args.layers,viewport=box)
            mapping=bundle.artifact_path(f"{prefix}/path_map.svg")
            export_view(bundle.artifact_path(f"{prefix}/svg.svg").read_text(),spec,bundle.artifact_path(f"views/{args.name}"),path_map=mapping.read_text() if mapping.exists() else None,inspection=inspection)
    print(write_report(bundle))


if __name__ == "__main__":
    main()

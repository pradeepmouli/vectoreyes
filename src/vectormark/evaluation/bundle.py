"""Durable run files. Drawing IDs remain live handles, not session backups."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .models import EvaluationRun, TrialRecord


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunBundle:
    def __init__(self, root: Path):
        self.root = root.resolve()

    @classmethod
    def create(cls, root: Path, run: EvaluationRun) -> RunBundle:
        root.mkdir(parents=True, exist_ok=False)
        result = cls(root)
        result.save(run)
        return result

    def artifact_path(self, relative: str) -> Path:
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
            raise ValueError("artifact paths must stay inside the run bundle")
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError("artifact paths must stay inside the run bundle")
        return target

    def load(self) -> EvaluationRun:
        return EvaluationRun.model_validate_json(self.artifact_path("run.json").read_text())

    def save(self, run: EvaluationRun) -> None:
        atomic_json(self.artifact_path("run.json"), run.model_dump(mode="json"))

    def load_trial(self, trial_id: str) -> TrialRecord:
        return TrialRecord.model_validate_json(self.artifact_path(f"trials/{trial_id}.json").read_text())

    def save_trial(self, trial: TrialRecord) -> None:
        target = self.artifact_path(f"trials/{trial.id}.json")
        run = self.load()
        if target.exists():
            original = self.load_trial(trial.id)
            if (original.request, original.base_version, original.hypothesis) != (trial.request, trial.base_version, trial.hypothesis):
                raise ValueError("a saved trial request cannot be rewritten; create a new trial")
            if original.execution != "pending":
                receipt = ("execution", "candidate_version", "error", "error_kind")
                if any(getattr(original, key) != getattr(trial, key) for key in receipt):
                    raise ValueError("a known execution receipt cannot be rewritten")
                if original.elapsed_seconds is not None and original.elapsed_seconds != trial.elapsed_seconds:
                    raise ValueError("a known execution duration cannot be rewritten")
            if run.accepted_version and run.accepted_version == original.candidate_version:
                if trial.execution != "succeeded" or trial.shape.status != "pass" or trial.fill.status != "pass" or trial.disposition != "accepted" or trial.artifact_errors:
                    raise ValueError("explicitly reject the accepted version before revoking its review")
        atomic_json(target, trial.model_dump(mode="json"))
        if trial.id not in run.trial_ids:
            run.trial_ids.append(trial.id)
            self.save(run)

    def write_artifact(self, relative: str, contents: bytes) -> dict[str, object]:
        target = self.artifact_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != contents:
            raise ValueError(f"immutable artifact already exists: {relative}")
        target.write_bytes(contents)
        return {"path": relative, "sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}

    def promote(self, trial_id: str) -> None:
        run, trial = self.load(), self.load_trial(trial_id)
        if trial_id not in run.trial_ids:
            raise ValueError("trial does not belong to this run")
        if not self.artifact_path(f"versions/{trial.candidate_version}/svg.svg").exists():
            raise ValueError("accepted SVG must be saved locally")
        prefix = f"versions/{trial.candidate_version}/"
        capture = self.artifact_path(prefix+"capture.json")
        if not capture.exists():
            raise ValueError("acceptance requires a complete capture receipt")
        receipt = json.loads(capture.read_text())
        if receipt.get("errors") or not receipt.get("artifacts"):
            raise ValueError("capture is incomplete")
        required = {"svg.svg", "labeled_svg.svg", "path_map.svg", "raw_trace.json", "plan.json", "preview.png", "review_panel.png"}
        verified = set()
        for artifact in receipt["artifacts"].values():
            path = self.artifact_path(artifact["path"])
            if not artifact["path"].startswith(prefix) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("captured artifact identity mismatch")
            verified.add(path.name)
        if not required.issubset(verified):
            raise ValueError("acceptance requires all source, map and review artifacts")
        snapshot = json.loads(self.artifact_path(prefix+"inspection.json").read_text())
        if snapshot.get("drawing_id") != run.drawing_id or snapshot.get("version") != trial.candidate_version:
            raise ValueError("inspection snapshot belongs to a different drawing/version")
        for verdict in (trial.shape, trial.fill):
            if not verdict.evidence or any(not self.artifact_path(ref).is_file() for ref in verdict.evidence):
                raise ValueError("both reviews must reference saved evidence")
        run.promote(trial)
        from .models import now
        run.decisions.append({"action": "accept", "trial_id": trial.id, "version": trial.candidate_version, "recorded_at": now()})
        self.save_trial(trial)
        self.save(run)

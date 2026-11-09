"""Strict review metadata, independent of the canonical drawing model."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ReviewVerdict(Record):
    status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    reason: str = ""
    reviewer: str = ""
    evidence: list[str] = Field(default_factory=list)
    recorded_at: str = Field(default_factory=now)

    @model_validator(mode="after")
    def justified(self) -> ReviewVerdict:
        if self.status != "not_evaluated" and (not self.reason.strip() or not self.reviewer.strip()):
            raise ValueError("a review verdict requires a reason and reviewer")
        return self


class TrialRecord(Record):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    base_version: str | None
    hypothesis: str = Field(min_length=1)
    request: dict[str, JsonValue]
    execution: Literal["pending", "succeeded", "failed", "unknown"] = "pending"
    candidate_version: str | None = None
    shape: ReviewVerdict = Field(default_factory=ReviewVerdict)
    fill: ReviewVerdict = Field(default_factory=ReviewVerdict)
    disposition: Literal["proposed", "accepted", "rejected", "blocked"] = "proposed"
    reason: str = ""
    error: str | None = None
    error_kind: str | None = None
    artifact_errors: dict[str, str] = Field(default_factory=dict)
    review_history: list[dict[str, JsonValue]] = Field(default_factory=list)
    elapsed_seconds: float | None = None
    created_at: str = Field(default_factory=now)


class ViewSpec(Record):
    track: Literal["shape", "fill"] = "shape"
    region_ids: list[str] = Field(default_factory=list)
    layers: list[Literal["regions", "paths", "commands", "features", "source"]] = Field(default_factory=lambda: ["regions"])
    # Drawing coordinates, shared by every pane. None means the whole canvas.
    viewport: tuple[float, float, float, float] | None = None
    width: int = Field(default=900, ge=64, le=4096)

    @model_validator(mode="after")
    def valid_viewport(self) -> ViewSpec:
        if self.viewport is not None:
            import math
            if not all(math.isfinite(x) for x in self.viewport) or self.viewport[2] <= 0 or self.viewport[3] <= 0:
                raise ValueError("viewport requires finite coordinates and positive dimensions")
        return self


class EvaluationRun(Record):
    schema_version: Literal["vectoreyes.eval.v1"] = "vectoreyes.eval.v1"
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    drawing_id: str
    baseline_version: str
    title: str = "Vector artwork"
    accepted_version: str | None = None
    trial_ids: list[str] = Field(default_factory=list)
    options: dict[str, JsonValue] = Field(default_factory=dict)
    build: dict[str, JsonValue] = Field(default_factory=dict)
    decisions: list[dict[str, JsonValue]] = Field(default_factory=list)
    created_at: str = Field(default_factory=now)

    def promote(self, trial: TrialRecord) -> None:
        if trial.execution != "succeeded" or not trial.candidate_version:
            raise ValueError("only a published successful trial can be accepted")
        if trial.artifact_errors:
            raise ValueError("capture missing review artifacts before acceptance")
        if trial.shape.status != "pass" or trial.fill.status != "pass":
            raise ValueError("independent shape and fill passes are required")
        self.accepted_version = trial.candidate_version
        trial.disposition = "accepted"

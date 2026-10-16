"""Canonical immutable drawing-version storage and private replay evidence."""

from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Callable, TypeVar

from .drawing import Drawing
from .drawing_actions import ActionSettings, DrawingPlan
from .drawing_recipe import (
    EffectiveDrawingRecipe,
    GeometryEvidence,
    PaintEvidence,
    ReplayRecord,
    ResolvedCornerEdit,
    snapshot_geometry_evidence,
    snapshot_paint_evidence,
    validate_replay_record,
)


class DrawingNotFound(Exception):
    """Raised when a live drawing ID or one of its versions is unavailable."""

    error_code = "DRAWING_NOT_FOUND"


@dataclass(frozen=True)
class DrawingVersion:
    """One published canonical value and its exact source-replay receipt."""

    id: str
    parent_id: str | None
    drawing: Drawing
    evidence_revision: str
    submitted_plan: DrawingPlan | None
    submitted_edits: tuple[ResolvedCornerEdit, ...]
    effective_recipe: EffectiveDrawingRecipe
    replay: ReplayRecord
    label: str | None = None
    baseline_version_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "submitted_edits", tuple(self.submitted_edits))

    @property
    def resolved_settings(self) -> ActionSettings:
        """Derive settings from the authoritative effective recipe."""
        return self.effective_recipe.settings


@dataclass(frozen=True)
class VersionMetadata:
    id: str
    parent_id: str | None
    label: str | None
    baseline_version_id: str
    evidence_revision: str
    recipe_revision: str
    output_revision: str


@dataclass(frozen=True)
class DrawingManifest:
    drawing_id: str
    versions: tuple[VersionMetadata, ...]


@dataclass(frozen=True)
class VersionCandidate:
    """Fully executed value awaiting one atomic publication transaction."""

    drawing: Drawing
    evidence: GeometryEvidence | None
    submitted_plan: DrawingPlan | None
    submitted_edits: tuple[ResolvedCornerEdit, ...]
    effective_recipe: EffectiveDrawingRecipe
    replay: ReplayRecord
    label: str | None = None
    baseline_version_id: str | None = None
    baseline_candidate_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "submitted_edits", tuple(self.submitted_edits))
        if (self.baseline_version_id is None) == (
            self.baseline_candidate_index is None
        ):
            raise ValueError(
                "candidate must select exactly one existing or transaction baseline"
            )


@dataclass
class _StoredDrawing:
    paint: PaintEvidence
    evidence: dict[str, GeometryEvidence]
    versions: dict[str, DrawingVersion]
    child_counts: dict[str, int]
    last_access: float


_VERSION_ID = re.compile(r"v\d+(?:\.\d+)*$")
_Prepared = TypeVar("_Prepared")


def _metadata(version: DrawingVersion) -> VersionMetadata:
    return VersionMetadata(
        id=version.id,
        parent_id=version.parent_id,
        label=version.label,
        baseline_version_id=version.baseline_version_id,
        evidence_revision=version.evidence_revision,
        recipe_revision=version.effective_recipe.revision,
        output_revision=version.replay.output_revision,
    )


def _validate_label(label: str | None) -> None:
    if type(label) not in (str, type(None)):
        raise TypeError("label must be exactly str or None")


class DrawingStore:
    """Retain canonical drawings behind a cross-transport opaque capability."""

    def __init__(
        self,
        *,
        idle_ttl_seconds: float = 1800,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._idle_ttl_seconds = idle_ttl_seconds
        self._now = now
        self._drawings: dict[str, _StoredDrawing] = {}
        self._lock = RLock()

    def new_drawing_id(self) -> str:
        """Mint an identity without creating live or abandoned store state."""
        return f"drw_{secrets.token_urlsafe(18)}"

    def create(
        self,
        drawing_id: str,
        paint: PaintEvidence,
        candidates: tuple[VersionCandidate, ...],
    ) -> tuple[DrawingVersion, ...]:
        """Validate and publish an entire source chain, or publish nothing."""
        versions, _ = self.create_prepared(
            drawing_id, paint, candidates, prepare=lambda _versions: None
        )
        return versions

    def create_prepared(
        self,
        drawing_id: str,
        paint: PaintEvidence,
        candidates: tuple[VersionCandidate, ...],
        *,
        prepare: Callable[[tuple[DrawingVersion, ...]], _Prepared],
    ) -> tuple[tuple[DrawingVersion, ...], _Prepared]:
        """Prepare exact proposed IDs under lock, then atomically publish."""
        if not drawing_id.startswith("drw_"):
            raise ValueError("drawing identity must be an opaque drw_ capability")
        if not candidates:
            raise ValueError("drawing creation requires at least one candidate")
        with self._lock:
            now = self._now()
            self._evict_expired(now)
            if drawing_id in self._drawings:
                raise ValueError("duplicate drawing identity")
            paint = snapshot_paint_evidence(paint)
            version_ids = tuple(f"v{index}" for index in range(len(candidates)))
            parent_ids = (None, *version_ids[:-1])
            versions, evidence = self._validate_chain(
                drawing_id=drawing_id,
                candidates=candidates,
                version_ids=version_ids,
                parent_ids=parent_ids,
                existing_versions={},
                existing_evidence={},
                paint=paint,
            )
            prepared = prepare(versions)
            self._drawings[drawing_id] = _StoredDrawing(
                paint=paint,
                evidence=evidence,
                versions={version.id: version for version in versions},
                child_counts={version.id: 0 for version in versions},
                last_access=now,
            )
            return versions, prepared

    def append(
        self,
        drawing_id: str,
        base_version: str,
        candidates: tuple[VersionCandidate, ...],
    ) -> tuple[DrawingVersion, ...]:
        """Publish a validated linear child chain without partial allocation."""
        versions, _ = self.append_prepared(
            drawing_id, base_version, candidates, prepare=lambda _versions: None
        )
        return versions

    def append_prepared(
        self,
        drawing_id: str,
        base_version: str,
        candidates: tuple[VersionCandidate, ...],
        *,
        prepare: Callable[[tuple[DrawingVersion, ...]], _Prepared],
    ) -> tuple[tuple[DrawingVersion, ...], _Prepared]:
        """Prepare one exact child chain under lock, then publish all or none."""
        if not candidates:
            raise ValueError("append requires at least one candidate")
        with self._lock:
            stored = self._get_live(drawing_id)
            if base_version not in stored.versions:
                raise DrawingNotFound(f"unknown drawing version: {base_version}")
            child_index = stored.child_counts[base_version]
            first_id = f"{base_version}.{child_index}"
            version_ids = (first_id,) + tuple(
                f"{first_id}{'.0' * index}"
                for index in range(1, len(candidates))
            )
            parent_ids = (base_version, *version_ids[:-1])
            if any(version_id in stored.versions for version_id in version_ids):
                raise ValueError("version identity collision")
            versions, evidence = self._validate_chain(
                drawing_id=drawing_id,
                candidates=candidates,
                version_ids=version_ids,
                parent_ids=parent_ids,
                existing_versions=stored.versions,
                existing_evidence=stored.evidence,
                paint=stored.paint,
            )
            prepared = prepare(versions)
            # All validation and evidence detachment completed before counters
            # or manifests become observable.
            stored.child_counts[base_version] = child_index + 1
            for version in versions:
                stored.versions[version.id] = version
                stored.child_counts[version.id] = 0
            for version in versions[:-1]:
                stored.child_counts[version.id] = 1
            stored.evidence = evidence
            return versions, prepared

    def get_version(self, drawing_id: str, version_id: str) -> DrawingVersion:
        with self._lock:
            stored = self._get_live(drawing_id)
            try:
                return stored.versions[version_id]
            except KeyError:
                raise DrawingNotFound(
                    f"unknown drawing version: {version_id}"
                ) from None

    def get_manifest(self, drawing_id: str) -> DrawingManifest:
        with self._lock:
            stored = self._get_live(drawing_id)
            return DrawingManifest(
                drawing_id=drawing_id,
                versions=tuple(_metadata(version) for version in stored.versions.values()),
            )

    def load_replay(
        self, drawing_id: str, version_id: str
    ) -> tuple[DrawingVersion, PaintEvidence, GeometryEvidence]:
        """Load one selected value and detached private evidence for execution."""
        with self._lock:
            stored = self._get_live(drawing_id)
            try:
                version = stored.versions[version_id]
                evidence = stored.evidence[version.evidence_revision]
            except KeyError:
                raise DrawingNotFound(
                    f"unknown drawing version: {version_id}"
                ) from None
            return version, stored.paint, snapshot_geometry_evidence(evidence)

    def _validate_chain(
        self,
        *,
        drawing_id: str,
        candidates: tuple[VersionCandidate, ...],
        version_ids: tuple[str, ...],
        parent_ids: tuple[str | None, ...],
        existing_versions: dict[str, DrawingVersion],
        existing_evidence: dict[str, GeometryEvidence],
        paint: PaintEvidence,
    ) -> tuple[tuple[DrawingVersion, ...], dict[str, GeometryEvidence]]:
        evidence = dict(existing_evidence)
        versions: list[DrawingVersion] = []
        for index, (candidate, version_id, parent_id) in enumerate(
            zip(candidates, version_ids, parent_ids, strict=True)
        ):
            _validate_label(candidate.label)
            if candidate.drawing.drawing_id != drawing_id:
                raise ValueError("candidate drawing identity does not match store identity")
            if candidate.evidence is not None:
                snapshot = snapshot_geometry_evidence(candidate.evidence)
                prior = evidence.get(snapshot.revision)
                if prior is not None and prior != snapshot:
                    raise ValueError("evidence revision collision")
                evidence[snapshot.revision] = snapshot
            try:
                selected_evidence = evidence[candidate.effective_recipe.evidence_revision]
            except KeyError as exc:
                raise ValueError("candidate references unknown geometry evidence") from exc
            recipe = candidate.effective_recipe
            replay = candidate.replay
            validate_replay_record(
                drawing=candidate.drawing, recipe=recipe, replay=replay,
                evidence=selected_evidence, paint=paint,
            )
            if candidate.baseline_candidate_index is not None:
                baseline_index = candidate.baseline_candidate_index
                if not 0 <= baseline_index <= index:
                    raise ValueError("transaction baseline must be an earlier candidate")
                baseline_id = version_ids[baseline_index]
            else:
                assert candidate.baseline_version_id is not None
                if candidate.baseline_version_id not in existing_versions:
                    raise ValueError("candidate references unknown baseline version")
                baseline_id = candidate.baseline_version_id
            versions.append(
                DrawingVersion(
                    id=version_id,
                    parent_id=parent_id,
                    drawing=candidate.drawing,
                    evidence_revision=selected_evidence.revision,
                    submitted_plan=candidate.submitted_plan,
                    submitted_edits=candidate.submitted_edits,
                    effective_recipe=recipe,
                    replay=replay,
                    label=candidate.label,
                    baseline_version_id=baseline_id,
                )
            )
        return tuple(versions), evidence

    def _get_live(self, drawing_id: str) -> _StoredDrawing:
        now = self._now()
        self._evict_expired(now)
        try:
            stored = self._drawings[drawing_id]
        except KeyError:
            raise DrawingNotFound(f"unknown drawing: {drawing_id}") from None
        stored.last_access = now
        return stored

    def _evict_expired(self, now: float) -> None:
        expired = [
            drawing_id
            for drawing_id, stored in self._drawings.items()
            if now - stored.last_access >= self._idle_ttl_seconds
        ]
        for drawing_id in expired:
            del self._drawings[drawing_id]


def validate_version_id(value: str) -> str:
    """Validate an externally supplied version-selection coordinate."""
    if not _VERSION_ID.fullmatch(value):
        raise ValueError("invalid drawing version")
    return value

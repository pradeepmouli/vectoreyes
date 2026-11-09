"""Internal effective corner recipes over immutable automatic-fit evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Literal, Protocol, TypeAlias

from .drawing_engine import FitPathRequest, FitPathResult, PathFitOptions
from .json_value import to_json_value
from .path_fit_evidence import (
    AutomaticFitEvidence,
    FeatureBindingDiagnostic,
    FitBoundaryCorrespondence,
    FitEndpointCorrespondence,
    FitPathCorrespondence,
    FitSpanCorrespondence,
)
from .path_geometry import (
    ArcCommand,
    ClosedPath,
    CornerAnchor,
    CubicCommand,
    LineCommand,
    MoveCommand,
    PathGeometry,
    QuadraticCommand,
)
from .python_path_fitter import (
    validate_explicit_fit_source,
    validate_path_fit_options,
)


CORNER_RECIPE_ENGINE_REVISION = "vectoreyes.corner-recipe.v1/python-fit.v1"
CornerDeclarationOrigin: TypeAlias = Literal["detected", "agent"]


def _copy_anchor(anchor: CornerAnchor) -> CornerAnchor:
    if not isinstance(anchor, CornerAnchor):
        raise TypeError("corner recipes require CornerAnchor declarations")
    return CornerAnchor(
        anchor.id,
        anchor.entry_command_id,
        anchor.exit_command_id,
        tuple(anchor.owned_command_ids),
        None if anchor.radius is None else float(anchor.radius),
    )


def _copy_diagnostic(diagnostic: FeatureBindingDiagnostic) -> FeatureBindingDiagnostic:
    return FeatureBindingDiagnostic(_copy_anchor(diagnostic.declaration), str(diagnostic.reason))


def _anchor_payload(anchor: CornerAnchor) -> tuple[object, ...]:
    return (
        anchor.id,
        anchor.entry_command_id,
        anchor.exit_command_id,
        anchor.owned_command_ids,
        anchor.radius,
    )


def _anchor_report(anchor: CornerAnchor) -> dict[str, object]:
    return {
        "id": anchor.id,
        "entry_command_id": anchor.entry_command_id,
        "exit_command_id": anchor.exit_command_id,
        "owned_command_ids": list(anchor.owned_command_ids),
        "radius": anchor.radius,
    }


@dataclass(frozen=True)
class RecipeCornerDeclaration:
    """One active corner declaration and its current authorship origin."""

    anchor: CornerAnchor
    origin: CornerDeclarationOrigin

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor", _copy_anchor(self.anchor))
        if self.origin not in {"detected", "agent"}:
            raise ValueError(f"unsupported corner declaration origin: {self.origin!r}")


@dataclass(frozen=True)
class CornerRecipe:
    """One effective path recipe bound to retained source evidence."""

    source_evidence_revision: str
    source_path_id: str
    options: PathFitOptions
    declarations: tuple[RecipeCornerDeclaration, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_evidence_revision:
            raise ValueError("corner recipe evidence revision must not be empty")
        if not self.source_path_id:
            raise ValueError("corner recipe source path ID must not be empty")
        normalized_options = PathFitOptions(
            float(self.options.epsilon),
            float(self.options.max_error),
            self.options.strategy,
        )
        normalized_declarations = tuple(sorted(
            (
                RecipeCornerDeclaration(declaration.anchor, declaration.origin)
                for declaration in self.declarations
            ),
            key=lambda declaration: declaration.anchor.id,
        ))
        feature_ids = tuple(declaration.anchor.id for declaration in normalized_declarations)
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("corner recipe feature IDs must be unique")
        object.__setattr__(self, "options", normalized_options)
        object.__setattr__(self, "declarations", normalized_declarations)

    @property
    def revision(self) -> str:
        """Content identity for this effective recipe and its engine contract."""
        payload = (
            CORNER_RECIPE_ENGINE_REVISION,
            self.source_evidence_revision,
            self.source_path_id,
            (self.options.epsilon, self.options.max_error, self.options.strategy),
            tuple(
                (declaration.origin, *_anchor_payload(declaration.anchor))
                for declaration in self.declarations
            ),
        )
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AddCorner:
    declaration: CornerAnchor

    def __post_init__(self) -> None:
        object.__setattr__(self, "declaration", _copy_anchor(self.declaration))


@dataclass(frozen=True)
class ReplaceCorner:
    feature_id: str
    declaration: CornerAnchor

    def __post_init__(self) -> None:
        if not self.feature_id:
            raise ValueError("replace corner feature ID must not be empty")
        object.__setattr__(self, "declaration", _copy_anchor(self.declaration))


@dataclass(frozen=True)
class RemoveCorner:
    feature_id: str

    def __post_init__(self) -> None:
        if not self.feature_id:
            raise ValueError("remove corner feature ID must not be empty")


CornerRecipeEdit: TypeAlias = AddCorner | ReplaceCorner | RemoveCorner


@dataclass(frozen=True)
class CornerDeclarationOutcome:
    """Observable binding state for one active declaration in a new fit."""

    feature_id: str
    origin: CornerDeclarationOrigin
    applied: bool
    boundaries: tuple[FitBoundaryCorrespondence, ...] = ()
    spans: tuple[FitSpanCorrespondence, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "spans", tuple(self.spans))
        if self.applied == (self.reason is not None):
            raise ValueError(
                "applied declaration outcomes require no reason; unbound outcomes require one"
            )


def _command_payload(command: object) -> tuple[object, ...]:
    if isinstance(command, MoveCommand):
        return ("M", command.end)
    if isinstance(command, LineCommand):
        return ("L", command.end)
    if isinstance(command, QuadraticCommand):
        return ("Q", command.control, command.end)
    if isinstance(command, CubicCommand):
        return ("C", command.control_1, command.control_2, command.end)
    if isinstance(command, ArcCommand):
        return (
            "A",
            command.rx,
            command.ry,
            command.rotation_degrees,
            command.large_arc,
            command.sweep,
            command.end,
        )
    raise TypeError(f"unsupported fitted command type: {type(command).__name__}")


def _output_revision(recipe_revision: str, fit: FitPathResult) -> str:
    output_position = {
        command.id: index
        for index, command in enumerate(fit.geometry.path.commands)
    }
    correspondence = (
        tuple(
            (
                boundary.feature_id,
                boundary.role,
                boundary.source_command_id,
                output_position[boundary.output_command_id],
            )
            for boundary in fit.correspondence.boundaries
        ),
        tuple(
            (
                span.kind,
                span.source_command_ids,
                tuple(output_position[command_id] for command_id in span.output_command_ids),
                span.feature_id,
            )
            for span in fit.correspondence.spans
        ),
    )
    payload = (
        CORNER_RECIPE_ENGINE_REVISION,
        recipe_revision,
        tuple(_command_payload(command) for command in fit.geometry.path.commands),
        correspondence,
    )
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CornerRecipeReplay:
    """One newly fitted geometry plus recipe and detector observation outcomes."""

    recipe: CornerRecipe
    fit: FitPathResult
    declaration_outcomes: tuple[CornerDeclarationOutcome, ...]
    source_diagnostics: tuple[FeatureBindingDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "declaration_outcomes", tuple(self.declaration_outcomes))
        object.__setattr__(self, "source_diagnostics", tuple(
            _copy_diagnostic(diagnostic) for diagnostic in self.source_diagnostics
        ))
        expected_path_id = f"r{self.output_revision}.p0"
        if self.fit.geometry.path.id != expected_path_id:
            raise ValueError("corner recipe replay output identity does not match fitted path")
        expected_command_ids = tuple(
            f"{expected_path_id}.c{index}"
            for index in range(len(self.fit.geometry.path.commands))
        )
        actual_command_ids = tuple(
            command.id for command in self.fit.geometry.path.commands
        )
        if actual_command_ids != expected_command_ids:
            raise ValueError("corner recipe replay command IDs do not match output identity")

    @property
    def output_revision(self) -> str:
        """Identity of the exact fit and actual source/output correspondence."""
        return _output_revision(self.recipe.revision, self.fit)

    def to_report(self) -> dict[str, object]:
        """Return a detached JSON-native summary without replacing typed fit data."""
        return {
            "recipe_revision": self.recipe.revision,
            "output_revision": self.output_revision,
            "source_evidence_revision": self.recipe.source_evidence_revision,
            "source_path_id": self.recipe.source_path_id,
            "output_path_id": self.fit.geometry.path.id,
            "fit_diagnostics": to_json_value(self.fit.diagnostics),
            "declaration_outcomes": [
                {
                    "feature_id": outcome.feature_id,
                    "origin": outcome.origin,
                    "applied": outcome.applied,
                    "reason": outcome.reason,
                    "boundaries": [
                        {
                            "role": boundary.role,
                            "source_command_id": boundary.source_command_id,
                            "output_command_id": boundary.output_command_id,
                        }
                        for boundary in outcome.boundaries
                    ],
                    "spans": [
                        {
                            "kind": span.kind,
                            "source_command_ids": list(span.source_command_ids),
                            "output_command_ids": list(span.output_command_ids),
                        }
                        for span in outcome.spans
                    ],
                }
                for outcome in self.declaration_outcomes
            ],
            "source_diagnostics": [
                {
                    "declaration": _anchor_report(diagnostic.declaration),
                    "reason": diagnostic.reason,
                }
                for diagnostic in self.source_diagnostics
            ],
        }


class PathFitCapability(Protocol):
    def fit_path(self, request: FitPathRequest) -> FitPathResult:
        """Fit one validated explicit path request."""


def _output_command_ids(evidence: AutomaticFitEvidence) -> frozenset[str]:
    return frozenset((
        evidence.output_path_id,
        *(boundary.output_command_id for boundary in evidence.correspondence.boundaries),
        *(
            command_id
            for span in evidence.correspondence.spans
            for command_id in span.output_command_ids
        ),
    ))


def _validate_anchor_references(anchor: CornerAnchor, evidence: AutomaticFitEvidence) -> None:
    source_ids = {command.id for command in evidence.source.path.commands}
    references = set(anchor.command_ids)
    unknown = references - source_ids
    if not unknown:
        return
    output_references = unknown & _output_command_ids(evidence)
    if output_references:
        offending = sorted(output_references)[0]
        raise ValueError(
            f"output command reference {offending!r} is not accepted; use a source command ID"
        )
    generated_references = sorted(
        reference
        for reference in unknown
        if isinstance(reference, str) and reference.startswith("r") and ".p0.c" in reference
    )
    if generated_references:
        raise ValueError(
            f"generated output command reference {generated_references[0]!r} is not accepted; "
            "use a source command ID"
        )
    offending = sorted(map(str, unknown))[0]
    raise ValueError(f"unknown source command reference {offending!r}")


def _validate_recipe_binding(recipe: CornerRecipe, evidence: AutomaticFitEvidence) -> None:
    if recipe.source_evidence_revision != evidence.revision:
        raise ValueError(
            "corner recipe evidence revision does not match supplied evidence revision"
        )
    if recipe.source_path_id != evidence.source.path.id:
        raise ValueError("corner recipe source path ID does not match supplied evidence")


def _recipe_source(recipe: CornerRecipe, evidence: AutomaticFitEvidence) -> PathGeometry:
    _validate_recipe_binding(recipe, evidence)
    anchors = tuple(declaration.anchor for declaration in recipe.declarations)
    for anchor in anchors:
        _validate_anchor_references(anchor, evidence)
    source = PathGeometry(evidence.source.path, anchors)
    validate_path_fit_options(recipe.options)
    validate_explicit_fit_source(source)
    return source


def _remap_fit_output(raw: FitPathResult, output_revision: str) -> FitPathResult:
    path_id = f"r{output_revision}.p0"
    command_id_map = {
        command.id: f"{path_id}.c{index}"
        for index, command in enumerate(raw.geometry.path.commands)
    }

    def mapped(command_id: str) -> str:
        try:
            return command_id_map[command_id]
        except KeyError as error:
            raise ValueError(
                f"fitter correspondence references unknown output command {command_id!r}"
            ) from error

    commands = tuple(
        replace(command, id=command_id_map[command.id])
        for command in raw.geometry.path.commands
    )
    anchors = tuple(
        CornerAnchor(
            anchor.id,
            mapped(anchor.entry_command_id),
            mapped(anchor.exit_command_id),
            tuple(mapped(command_id) for command_id in anchor.owned_command_ids),
            anchor.radius,
        )
        for anchor in raw.geometry.anchors
        if isinstance(anchor, CornerAnchor)
    )
    if len(anchors) != len(raw.geometry.anchors):
        raise ValueError("explicit fitter returned an unsupported non-corner anchor")
    geometry = PathGeometry(ClosedPath(path_id, commands), anchors)
    correspondence = FitPathCorrespondence(
        tuple(
            FitBoundaryCorrespondence(
                boundary.feature_id,
                boundary.role,
                boundary.source_command_id,
                mapped(boundary.output_command_id),
            )
            for boundary in raw.correspondence.boundaries
        ),
        tuple(
            FitSpanCorrespondence(
                span.kind,
                tuple(span.source_command_ids),
                tuple(mapped(command_id) for command_id in span.output_command_ids),
                span.feature_id,
            )
            for span in raw.correspondence.spans
        ),
        tuple(
            FitEndpointCorrespondence(item.source_command_id, mapped(item.output_command_id))
            for item in raw.correspondence.endpoints
        ),
    )
    return FitPathResult(geometry, correspondence, raw.diagnostics)


def _declaration_outcomes(
    recipe: CornerRecipe,
    fit: FitPathResult,
) -> tuple[CornerDeclarationOutcome, ...]:
    outcomes = []
    for declaration in recipe.declarations:
        anchor = declaration.anchor
        boundaries = tuple(
            boundary
            for boundary in fit.correspondence.boundaries
            if boundary.feature_id == anchor.id
        )
        spans = tuple(
            span
            for span in fit.correspondence.spans
            if span.feature_id == anchor.id
        )
        if anchor.entry_command_id == anchor.exit_command_id:
            applied = (
                len(boundaries) == 1
                and boundaries[0].role == "point"
                and not spans
            )
        else:
            applied = (
                len(boundaries) == 2
                and {boundary.role for boundary in boundaries} == {"entry", "exit"}
                and len(spans) == 1
                and spans[0].kind == "corner"
            )
        outcomes.append(CornerDeclarationOutcome(
            anchor.id,
            declaration.origin,
            applied,
            boundaries,
            spans,
            None if applied else "fitter did not return complete declaration correspondence",
        ))
    return tuple(outcomes)


class CornerRecipeExecutor:
    """Create, transactionally revise, and replay one effective path recipe."""

    def __init__(self, path_fitter: PathFitCapability | None = None) -> None:
        if path_fitter is None:
            from .python_drawing_engine import PythonDrawingEngine
            path_fitter = PythonDrawingEngine()
        self._path_fitter = path_fitter

    def create(
        self,
        evidence: AutomaticFitEvidence,
        options: PathFitOptions,
    ) -> CornerRecipe:
        """Seed detected active declarations exactly once from source-valid evidence."""
        declarations = tuple(
            RecipeCornerDeclaration(anchor, "detected")
            for anchor in evidence.source.anchors
            if isinstance(anchor, CornerAnchor)
        )
        if len(declarations) != len(evidence.source.anchors):
            raise ValueError("automatic fit evidence contains non-corner source declarations")
        recipe = CornerRecipe(evidence.revision, evidence.source.path.id, options, declarations)
        _recipe_source(recipe, evidence)
        return recipe

    def revise(
        self,
        recipe: CornerRecipe,
        evidence: AutomaticFitEvidence,
        edits: Sequence[CornerRecipeEdit],
    ) -> CornerRecipe:
        """Apply typed edits transactionally, returning no partial recipe on failure."""
        _recipe_source(recipe, evidence)
        revised = recipe
        for index, edit in enumerate(tuple(edits)):
            declarations = {
                declaration.anchor.id: declaration
                for declaration in revised.declarations
            }
            if isinstance(edit, AddCorner):
                kind = "add"
                feature_id = edit.declaration.id
                if feature_id in declarations:
                    raise ValueError(
                        f"corner recipe edit {index} (add {feature_id!r}) rejected: "
                        "feature already exists"
                    )
                candidate_anchor = edit.declaration
                declarations[feature_id] = RecipeCornerDeclaration(candidate_anchor, "agent")
            elif isinstance(edit, ReplaceCorner):
                kind = "replace"
                feature_id = edit.feature_id
                if feature_id not in declarations:
                    raise ValueError(
                        f"corner recipe edit {index} (replace {feature_id!r}) rejected: "
                        "unknown feature reference"
                    )
                if edit.declaration.id != feature_id:
                    raise ValueError(
                        f"corner recipe edit {index} (replace {feature_id!r}) rejected: "
                        "replacement identity "
                        f"{edit.declaration.id!r} is inconsistent"
                    )
                candidate_anchor = edit.declaration
                declarations[feature_id] = RecipeCornerDeclaration(candidate_anchor, "agent")
            elif isinstance(edit, RemoveCorner):
                kind = "remove"
                feature_id = edit.feature_id
                if feature_id not in declarations:
                    raise ValueError(
                        f"corner recipe edit {index} (remove {feature_id!r}) rejected: "
                        "unknown feature reference"
                    )
                del declarations[feature_id]
                candidate_anchor = None
            else:
                raise TypeError(
                    f"corner recipe edit {index} has unsupported type {type(edit).__name__}"
                )

            try:
                if candidate_anchor is not None:
                    _validate_anchor_references(candidate_anchor, evidence)
                candidate = CornerRecipe(
                    revised.source_evidence_revision,
                    revised.source_path_id,
                    revised.options,
                    tuple(declarations.values()),
                )
                _recipe_source(candidate, evidence)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"corner recipe edit {index} ({kind} {feature_id!r}) rejected: {error}"
                ) from error
            revised = candidate
        return revised

    def replay(
        self,
        recipe: CornerRecipe,
        evidence: AutomaticFitEvidence,
    ) -> CornerRecipeReplay:
        """Fit retained source commands once under the effective active declarations."""
        source = _recipe_source(recipe, evidence)
        # Declarations partition the retained original polyline; they do not
        # delete its samples. In particular a rounded-to-point replacement
        # must not turn the old transition into an unsupported fitting gap.
        raw = self._path_fitter.fit_path(FitPathRequest(source, recipe.options))
        output_revision = _output_revision(recipe.revision, raw)
        fit = _remap_fit_output(raw, output_revision)
        return CornerRecipeReplay(
            recipe,
            fit,
            _declaration_outcomes(recipe, fit),
            tuple(evidence.diagnostics),
        )

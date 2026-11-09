import json
from dataclasses import replace
from hashlib import sha256

import pytest

import vectoreyes.fit as fit
from vectoreyes.corner_ownership import directed_corner_ownership
from vectoreyes.corner_recipe import (
    AddCorner,
    CornerRecipeExecutor,
    RemoveCorner,
    ReplaceCorner,
)
from vectoreyes.drawing_engine import FitPathResult, PathFitOptions
from vectoreyes.path_fit_evidence import (
    AutomaticFitEvidence,
    FeatureBindingDiagnostic,
    FitPathCorrespondence,
    FitSpanCorrespondence,
)
from vectoreyes.path_geometry import (
    ClosedPath,
    CornerAnchor,
    LineCommand,
    MoveCommand,
    PathGeometry,
)
from vectoreyes.python_path_fitter import PythonPathFitter


def _source(points, anchors=(), *, path_id="evidence.source.p0"):
    commands = [MoveCommand(f"{path_id}.c0", tuple(points[0]))]
    commands.extend(
        LineCommand(f"{path_id}.c{index}", tuple(point))
        for index, point in enumerate(points[1:], 1)
    )
    commands.append(LineCommand(f"{path_id}.c{len(points)}", tuple(points[0])))
    return PathGeometry(ClosedPath(path_id, tuple(commands)), tuple(anchors))


def _evidence(points, anchors=(), *, revision="evidence-r1", path_id="evidence.source.p0",
              diagnostics=()):
    source = _source(points, anchors, path_id=path_id)
    output_id = f"{revision}.automatic.p0"
    source_ids = tuple(command.id for command in source.path.drawable_commands)
    correspondence = FitPathCorrespondence((), (
        FitSpanCorrespondence(
            "side",
            source_ids,
            (f"{output_id}.c1",),
        ),
    ))
    return AutomaticFitEvidence(
        revision,
        source,
        output_id,
        sha256(b"automatic-output").hexdigest(),
        correspondence,
        tuple(diagnostics),
    )


def _anchor(source, feature_id, entry_index, exit_index, radius=None):
    command_ids = tuple(command.id for command in source.path.commands)
    entry = command_ids[entry_index]
    exit = command_ids[exit_index]
    return CornerAnchor(
        feature_id,
        entry,
        exit,
        directed_corner_ownership(
            command_ids,
            entry,
            exit,
            explicit_closing_drawable=True,
        ),
        radius,
    )


def _options(max_error=1.5):
    return PathFitOptions(epsilon=0.25, max_error=max_error, strategy="quadratic")


def test_add_replay_remove_replay_equals_recipe_that_never_contained_corner():
    points = [(0, 0), (8, 0), (12, 1), (15, 5), (13, 10), (5, 13), (0, 9)]
    evidence = _evidence(points)
    executor = CornerRecipeExecutor()
    original = executor.create(evidence, _options())
    sibling = executor.create(evidence, _options())
    source_before = evidence.source
    added = executor.revise(original, evidence, [
        AddCorner(_anchor(evidence.source, "agent-apex", 2, 2)),
    ])

    added_replay = executor.replay(added, evidence)
    removed = executor.revise(added, evidence, [RemoveCorner("agent-apex")])
    removed_replay = executor.replay(removed, evidence)
    never_added_replay = executor.replay(original, evidence)

    assert {outcome.feature_id for outcome in added_replay.declaration_outcomes} == {"agent-apex"}
    assert removed == original
    assert removed_replay == never_added_replay
    assert evidence.source == source_before
    assert sibling == original
    assert sibling.declarations == ()


def test_replace_rounded_declaration_with_wrapped_span_retains_identity_only():
    points = [(2, 0), (5, 1), (7, 4), (7, 10), (-5, 10), (-7, 4), (-5, 1)]
    bare = _source(points)
    initial = _anchor(bare, "rounded", 1, 3, radius=2.0)
    evidence = _evidence(points, (initial,))
    replacement = _anchor(evidence.source, "rounded", 5, 1, radius=3.0)
    executor = CornerRecipeExecutor()
    recipe = executor.create(evidence, _options())

    revised = executor.revise(recipe, evidence, [ReplaceCorner("rounded", replacement)])
    replay = executor.replay(revised, evidence)

    assert revised.declarations[0].anchor == replacement
    assert revised.declarations[0].origin == "agent"
    assert replay.fit.geometry.anchors[0].id == "rounded"
    corner_spans = [span for span in replay.fit.correspondence.spans if span.kind == "corner"]
    assert len(corner_spans) == 1
    assert corner_spans[0].source_command_ids == replacement.owned_command_ids
    assert initial.owned_command_ids != replacement.owned_command_ids


def test_empty_declarations_are_authoritative_and_use_actual_explicit_fitter(monkeypatch):
    points = [(0, 0), (4, -1), (9, 2), (10, 8), (5, 11), (-1, 7)]
    evidence = _evidence(points)
    monkeypatch.setattr(fit, "corner_indices", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("recipe replay must not invoke corner detection")
    ))
    recipe = CornerRecipeExecutor().create(evidence, _options())

    replay = CornerRecipeExecutor().replay(recipe, evidence)

    assert replay.fit.geometry.anchors == ()
    assert replay.fit.correspondence.boundaries == ()
    assert replay.declaration_outcomes == ()
    assert replay.fit.diagnostics["semantic_corner_count"] == 0


def test_replay_calls_the_injected_explicit_fitter_exactly_once():
    class CountingFitter:
        def __init__(self):
            self.calls = []

        def fit_path(self, request):
            self.calls.append(request)
            return PythonPathFitter().fit_path(request)

    points = [(0, 0), (10, 0), (12, 4), (10, 9), (0, 9), (-2, 4)]
    evidence = _evidence(points)
    fitter = CountingFitter()
    executor = CornerRecipeExecutor(fitter)
    recipe = executor.create(evidence, _options())

    replay = executor.replay(recipe, evidence)

    assert len(fitter.calls) == 1
    assert fitter.calls[0].source.path is evidence.source.path
    assert replay.fit.correspondence.spans


@pytest.mark.parametrize(
    ("stale", "message"),
    [
        (lambda evidence: replace(evidence, revision="stale-revision"), "evidence revision"),
        (
            lambda evidence: replace(
                evidence,
                source=_source(
                    [(0, 0), (8, 0), (9, 5), (4, 9), (-1, 5)],
                    path_id="different.source.p0",
                ),
            ),
            "source path ID",
        ),
    ],
)
def test_stale_evidence_binding_rejects_revision_and_replay(stale, message):
    points = [(0, 0), (8, 0), (9, 5), (4, 9), (-1, 5)]
    evidence = _evidence(points)
    executor = CornerRecipeExecutor()
    recipe = executor.create(evidence, _options())

    with pytest.raises(ValueError, match=message):
        executor.revise(recipe, stale(evidence), [])
    with pytest.raises(ValueError, match=message):
        executor.replay(recipe, stale(evidence))


def test_typed_edit_identity_and_reference_errors_are_atomic_and_precise():
    points = [(0, 0), (10, 0), (12, 4), (10, 9), (0, 9), (-2, 4)]
    evidence = _evidence(points)
    executor = CornerRecipeExecutor()
    recipe = executor.create(evidence, _options())
    valid = _anchor(evidence.source, "new", 2, 2)
    output_command_id = evidence.correspondence.spans[0].output_command_ids[0]

    cases = [
        (RemoveCorner("missing"), r"edit 0.*remove.*missing"),
        (ReplaceCorner("missing", replace(valid, id="missing")), r"edit 0.*replace.*missing"),
        (
            AddCorner(CornerAnchor("output-ref", output_command_id, output_command_id)),
            r"edit 0.*output command.*source command",
        ),
        (
            AddCorner(CornerAnchor("unknown", "invented.c1", "invented.c1")),
            r"edit 0.*unknown source command.*invented.c1",
        ),
    ]
    for edit, message in cases:
        with pytest.raises(ValueError, match=message):
            executor.revise(recipe, evidence, [edit])
        assert recipe.declarations == ()

    added = executor.revise(recipe, evidence, [AddCorner(valid)])
    with pytest.raises(ValueError, match=r"edit 0.*replace.*identity"):
        executor.revise(added, evidence, [ReplaceCorner("new", replace(valid, id="other"))])
    with pytest.raises(ValueError, match=r"edit 0.*add.*new.*already exists"):
        executor.revise(added, evidence, [AddCorner(valid)])
    assert added.declarations[0].anchor == valid


def test_invalid_ownership_overlap_and_point_inside_rounded_reject_before_replay():
    points = [(0, 0), (4, -1), (8, 0), (10, 4), (8, 8), (4, 9), (0, 8), (-2, 4)]
    evidence = _evidence(points)
    executor = CornerRecipeExecutor()
    recipe = executor.create(evidence, _options())
    rounded = _anchor(evidence.source, "rounded", 1, 4)
    bad_ownership = replace(
        rounded,
        id="bad-ownership",
        owned_command_ids=(rounded.owned_command_ids[0],),
    )

    with pytest.raises(ValueError, match=r"edit 0.*ownership"):
        executor.revise(recipe, evidence, [AddCorner(bad_ownership)])

    with_rounded = executor.revise(recipe, evidence, [AddCorner(rounded)])
    overlapping = _anchor(evidence.source, "overlap", 2, 5)
    with pytest.raises(
        ValueError,
        match=r"edit 0.*ambiguous.*(?:overlap.*rounded|rounded.*overlap)",
    ):
        executor.revise(with_rounded, evidence, [AddCorner(overlapping)])

    inside = _anchor(evidence.source, "inside", 3, 3)
    with pytest.raises(ValueError, match=r"edit 0.*inside.*rounded corner span"):
        executor.revise(with_rounded, evidence, [AddCorner(inside)])
    assert [item.anchor.id for item in with_rounded.declarations] == ["rounded"]


def test_exact_source_endpoint_and_actual_correspondence_survive_identity_remap():
    points = [
        (0.1234567890123, 0.25),
        (8.75, -0.125),
        (12.9876543210987, 4.875),
        (9.5, 10.25),
        (0.5, 9.75),
    ]
    evidence = _evidence(points)
    executor = CornerRecipeExecutor()
    declaration = _anchor(evidence.source, "precise", 2, 2)
    recipe = executor.revise(
        executor.create(evidence, _options()),
        evidence,
        [AddCorner(declaration)],
    )

    first = executor.replay(recipe, evidence)
    second = executor.replay(recipe, evidence)
    boundary = next(
        item
        for item in first.fit.correspondence.boundaries
        if item.feature_id == "precise"
    )

    assert first == second
    assert first.output_revision == second.output_revision
    assert first.fit.geometry.endpoint(boundary.output_command_id) == points[2]
    assert boundary.source_command_id == declaration.entry_command_id
    assert boundary.output_command_id != boundary.source_command_id
    assert first.fit.geometry.path.id.startswith(f"r{first.output_revision}.")
    assert all(
        command_id.startswith(f"r{first.output_revision}.")
        for span in first.fit.correspondence.spans
        for command_id in span.output_command_ids
    )

    changed_settings = executor.create(evidence, _options(max_error=2.0))
    changed = executor.replay(changed_settings, evidence)
    assert changed.recipe.revision != recipe.revision
    assert changed.output_revision != first.output_revision
    assert changed.fit.geometry.path.id != first.fit.geometry.path.id

    generated_reference = first.fit.geometry.path.commands[1].id
    with pytest.raises(ValueError, match=r"generated output command.*source command"):
        executor.revise(recipe, evidence, [
            AddCorner(CornerAnchor("generated-ref", generated_reference, generated_reference)),
        ])


def test_declaration_origin_is_reported_even_when_fitter_cannot_bind_it():
    class UnbindingFitter:
        def fit_path(self, request):
            raw = PythonPathFitter().fit_path(request)
            return FitPathResult(
                raw.geometry,
                FitPathCorrespondence((), raw.correspondence.spans),
                raw.diagnostics,
            )

    points = [(0, 0), (8, 0), (10, 5), (5, 9), (0, 7)]
    bare = _source(points)
    detected = _anchor(bare, "detected", 2, 2)
    evidence = _evidence(points, (detected,))
    executor = CornerRecipeExecutor(UnbindingFitter())
    recipe = executor.create(evidence, _options())

    replay = executor.replay(recipe, evidence)

    assert replay.declaration_outcomes[0].feature_id == "detected"
    assert replay.declaration_outcomes[0].origin == "detected"
    assert replay.declaration_outcomes[0].applied is False
    assert replay.declaration_outcomes[0].reason


def test_invalid_observations_remain_lossless_non_active_and_report_is_json_safe():
    points = [(0, 0), (8, 0), (10, 5), (5, 9), (0, 7)]
    invalid = CornerAnchor("detected-invalid", "missing.c1", "missing.c1")
    diagnostic = FeatureBindingDiagnostic(invalid, "ambiguous_source_declaration: unknown command")
    evidence = _evidence(points, diagnostics=(diagnostic,))
    executor = CornerRecipeExecutor()

    recipe = executor.create(evidence, _options())
    replay = executor.replay(recipe, evidence)
    report = replay.to_report()

    assert recipe.declarations == ()
    assert replay.fit.geometry.anchors == ()
    assert replay.source_diagnostics == (diagnostic,)
    assert report["source_diagnostics"][0]["declaration"]["id"] == "detected-invalid"
    assert report["source_diagnostics"][0]["reason"] == diagnostic.reason
    json.dumps(report, allow_nan=False)


def test_recipe_and_edit_owned_sequences_are_detached_from_mutable_inputs():
    points = [(0, 0), (8, 0), (10, 5), (5, 9), (0, 7)]
    bare = _source(points)
    owned = list(directed_corner_ownership(
        tuple(command.id for command in bare.path.commands),
        bare.path.commands[1].id,
        bare.path.commands[3].id,
        explicit_closing_drawable=True,
    ))
    anchor = CornerAnchor("rounded", bare.path.commands[1].id, bare.path.commands[3].id, owned)
    evidence = _evidence(points)
    edits = [AddCorner(anchor)]
    executor = CornerRecipeExecutor()
    original = executor.create(evidence, _options())

    revised = executor.revise(original, evidence, edits)
    owned.clear()
    edits.clear()

    assert revised.declarations[0].anchor.owned_command_ids
    assert revised.declarations[0].anchor is not anchor
    assert original.declarations == ()
    assert executor.replay(revised, evidence).fit.geometry.anchors


def test_caller_owned_source_command_list_cannot_change_evidence_or_replay():
    points = [(0, 0), (8, 0), (10, 5), (5, 9), (0, 7)]
    path_id = "mutable.source.p0"
    commands = [MoveCommand(f"{path_id}.c0", points[0])]
    commands.extend(
        LineCommand(f"{path_id}.c{index}", point)
        for index, point in enumerate(points[1:], 1)
    )
    commands.append(LineCommand(f"{path_id}.c{len(points)}", points[0]))
    source = PathGeometry(ClosedPath(path_id, commands))
    output_id = "mutable.automatic.p0"
    evidence = AutomaticFitEvidence(
        "mutable-evidence-r1",
        source,
        output_id,
        sha256(b"automatic-output").hexdigest(),
        FitPathCorrespondence((), (
            FitSpanCorrespondence(
                "side",
                tuple(command.id for command in commands[1:]),
                (f"{output_id}.c1",),
            ),
        )),
    )
    executor = CornerRecipeExecutor()
    recipe = executor.create(evidence, _options())
    sibling = executor.create(evidence, _options())
    first = executor.replay(recipe, evidence)

    commands[2] = LineCommand(f"{path_id}.c2", (12, 6))
    second = executor.replay(recipe, evidence)

    assert evidence.source.path.commands is not commands
    assert evidence.source.path.commands[2].end == points[2]
    assert recipe == sibling
    assert first == second


def test_replay_nested_diagnostics_are_frozen_both_ways_and_report_is_detached():
    diagnostics = {"nested": {"values": [1]}}

    class NestedDiagnosticFitter:
        def fit_path(self, request):
            raw = PythonPathFitter().fit_path(request)
            return FitPathResult(raw.geometry, raw.correspondence, diagnostics)

    points = [(0, 0), (8, 0), (10, 5), (5, 9), (0, 7)]
    evidence = _evidence(points)
    executor = CornerRecipeExecutor(NestedDiagnosticFitter())
    replay = executor.replay(executor.create(evidence, _options()), evidence)
    report = replay.to_report()
    json.dumps(report, allow_nan=False)

    diagnostics["nested"]["values"].append(2)
    assert replay.fit.diagnostics["nested"]["values"] == (1,)
    with pytest.raises(AttributeError):
        replay.fit.diagnostics["nested"]["values"].append(3)
    with pytest.raises(TypeError):
        replay.fit.diagnostics["nested"]["added"] = True

    report["fit_diagnostics"]["nested"]["values"].append(4)
    report["fit_diagnostics"]["added"] = True
    assert replay.fit.diagnostics["nested"]["values"] == (1,)
    assert "added" not in replay.fit.diagnostics["nested"]
    assert replay.to_report()["fit_diagnostics"] == {"nested": {"values": [1]}}
    assert diagnostics == {"nested": {"values": [1, 2]}}

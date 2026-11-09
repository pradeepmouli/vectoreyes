"""Feature-discovery isolation and exact pre-refactor fitting receipts."""

from hashlib import sha256

import numpy as np
import pytest

import vectoreyes.fit as fit


def contours():
    t = np.arange(33, dtype=float)[:, None] / 32.0

    def line(a, b):
        return (1 - t) * np.asarray(a) + t * np.asarray(b)

    def quadratic(a, b, c):
        return (1 - t) ** 2 * np.asarray(a) + 2 * (1 - t) * t * np.asarray(b) + t ** 2 * np.asarray(c)

    rounded = [
        line((8, 0), (112, 0)), quadratic((112, 0), (120, 0), (120, 8)),
        line((120, 8), (120, 72)), quadratic((120, 72), (120, 80), (112, 80)),
        line((112, 80), (8, 80)), quadratic((8, 80), (0, 80), (0, 72)),
        line((0, 72), (0, 8)), quadratic((0, 8), (0, 0), (8, 0)),
    ]
    vertices = [(0, 0), (96, 0), (96, 2), (112, 2), (112, 64), (64, 66), (0, 64)]
    serif = [line(a, b) for a, b in zip(vertices, vertices[1:] + vertices[:1])]
    return {
        name: np.vstack([*(segment[:-1] for segment in segments), segments[-1][-1:]])
        for name, segments in (("rounded", rounded), ("serif", serif))
    }


PRESETS = (
    dict(),
    dict(progressive=True, progressive_allow_lines=False),
    dict(progressive=True, progressive_allow_lines=True, support_weighted_corners=True, canvas_extent=1254),
    dict(progressive=True, corner_rounding=False, canvas_extent=1254),
    dict(cubic=True),
)

# Original geometry baselines, rehashed for the renamed evidence namespace.
# Verified that substituting only the prior namespace reproduces all ten old digests.
BASELINES = {
    ("rounded", 0): "d328f4b0933a964e6a1c6f53a9b6b315706aebb3aca7707623b5024e120c4b37",
    ("serif", 0): "7bffc41ca073e4f609b66d373b3ad5143a4f9dda4d79772e889977f6d47162a7",
    ("rounded", 1): "b3ff424d4e8fa55e9761f78a7b8785248f35f318c94ee5899da529a6de02cfd2",
    ("serif", 1): "cd874e36e4a1c5cf3fb6a6ab997e6a2628effd93464abe50c84f887cd29e9385",
    ("rounded", 2): "9c07d58c998dd1f997706e18e2a44bc6d57427e1ad8dddd5838a87815418fe39",
    ("serif", 2): "a5cac1b11bb6c9804fa260168b7e4a753e4e3646912821d697b431bdbe42321b",
    ("rounded", 3): "c4bd939ce5fe883be1f019b3b9dac78ddafef3636c91873723857b0f53e10870",
    ("serif", 3): "1bcad1886dcef9bb05c2867ea8437691585d089251c15047d15ad618b429b796",
    ("rounded", 4): "6426b483b65f4ff6e743431e438ce920adacb6482bb86f69250945250a3f3a53",
    ("serif", 4): "66a95752d6e3684d55238c56fdc5f2f2dae1b92977f5cb55f2605030d1f5a68a",
}


@pytest.mark.parametrize("name", ("rounded", "serif"))
@pytest.mark.parametrize("preset", range(len(PRESETS)))
def test_fit_output_and_source_receipt_remain_identical(name, preset):
    result = fit._fit_path_d_with_evidence(contours()[name], epsilon=0.5, max_error=1.5, **PRESETS[preset])
    digest = sha256(repr((result.d, result.corner_points, result.corner_spans, result.evidence.revision)).encode()).hexdigest()
    assert digest == BASELINES.get((name, preset)), ((name, preset), digest)


def test_feature_discovery_is_immutable_and_does_not_fit(monkeypatch):
    assert hasattr(fit, "_detect_path_features"), "automatic feature discovery must be independently callable"

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("feature discovery must not fit path commands")

    monkeypatch.setattr(fit, "_fit_path_d_from_cuts", unexpected_fit)
    contour = contours()["rounded"]
    detected = fit._detect_path_features(contour, epsilon=0.5, progressive=True)
    assert isinstance(detected.ring, tuple)
    assert isinstance(detected.cut_indices, tuple)
    assert isinstance(detected.point_indices, tuple)
    assert isinstance(detected.corner_ranges, tuple)
    original = detected.ring
    contour[:] = -1
    assert detected.ring == original
    assert all(0 <= index < len(original) for index in detected.cut_indices)


def test_automatic_fit_consumes_one_discovery_result(monkeypatch):
    assert hasattr(fit, "_detect_path_features"), "automatic fit must orchestrate the discovery operation"
    discover = fit._detect_path_features
    calls = []

    def observe(*args, **kwargs):
        result = discover(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(fit, "_detect_path_features", observe)
    fitted = fit._fit_path_d_with_evidence(contours()["serif"], epsilon=0.5, max_error=1.5, progressive=True)
    assert len(calls) == 1
    assert fitted.corner_points == tuple(calls[0].ring[index] for index in calls[0].point_indices)


def source_geometry():
    from vectoreyes.path_geometry import ClosedPath, LineCommand, MoveCommand, PathGeometry
    points = contours()["rounded"]
    return PathGeometry(ClosedPath("r1.p0", (
        MoveCommand("r1.p0.c0", tuple(points[0])),
        *(LineCommand(f"r1.p0.c{index}", tuple(point)) for index, point in enumerate(points[1:], 1)),
    )))


def test_engine_discovery_returns_original_commands_with_typed_features(monkeypatch):
    import vectoreyes.drawing_engine as contracts
    from vectoreyes.drawing_trace import TraceOptions
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    assert hasattr(contracts, "DetectPathFeaturesRequest"), "engine must expose pre-fit detection"
    source = source_geometry()

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("engine discovery must not fit path geometry")

    monkeypatch.setattr(fit, "_fit_path_d_from_cuts", unexpected_fit)
    result = PythonDrawingEngine().detect_path_features(contracts.DetectPathFeaturesRequest(
        source, TraceOptions(simplify_tolerance=0.5, fit_strategy="progressive"), 1254,
        forced_corner_command_ids=(source.path.commands[0].id,)))
    assert result.geometry.path == source.path
    assert result.geometry.anchors
    assert result.geometry.anchors[0].entry_command_id == source.path.commands[-1].id
    assert source.anchors == ()
    assert all(command_id in {command.id for command in source.path.commands}
        for command_id in result.fit_boundary_command_ids)


def test_engine_detection_rejects_existing_declarations_and_foreign_forced_ids():
    from dataclasses import replace
    import vectoreyes.drawing_engine as contracts
    from vectoreyes.drawing_trace import TraceOptions
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    from vectoreyes.path_geometry import CornerAnchor
    assert hasattr(contracts, "DetectPathFeaturesRequest"), "detection must validate canonical source addresses"
    source = source_geometry()
    engine = PythonDrawingEngine()
    request = contracts.DetectPathFeaturesRequest(source, TraceOptions(), 1254)
    with pytest.raises(ValueError, match="existing declarations"):
        engine.detect_path_features(replace(request, source=replace(source,
            anchors=(CornerAnchor("explicit", source.path.commands[1].id, source.path.commands[1].id),))))
    with pytest.raises(ValueError, match="forced.*source"):
        engine.detect_path_features(replace(request, forced_corner_command_ids=("other.c1",)))


def test_forced_corner_cannot_silently_bind_a_different_coincident_command():
    from dataclasses import replace
    from vectoreyes.drawing_engine import DetectPathFeaturesRequest
    from vectoreyes.drawing_trace import TraceOptions
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    source = source_geometry()
    commands = list(source.path.commands)
    commands[128] = replace(commands[128], end=commands[1].end)
    source = replace(source, path=replace(source.path, commands=tuple(commands)))
    result = PythonDrawingEngine().detect_path_features(DetectPathFeaturesRequest(source, TraceOptions(), 1254,
        forced_corner_command_ids=(commands[128].id,)))
    assert any(anchor.entry_command_id == commands[128].id and anchor.exit_command_id == commands[128].id
        for anchor in result.geometry.anchors)


@pytest.mark.parametrize("use_discovered_cuts", (False, True))
def test_discovered_fit_boundaries_are_authoritative_for_pure_fitting(monkeypatch, use_discovered_cuts):
    import vectoreyes.python_path_fitter as fitter
    from vectoreyes.drawing_engine import DetectPathFeaturesRequest, FitPathRequest, PathFitOptions
    from vectoreyes.drawing_trace import TraceOptions
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    engine = PythonDrawingEngine()
    features = engine.detect_path_features(DetectPathFeaturesRequest(source_geometry(),
        TraceOptions(simplify_tolerance=0.5, fit_strategy="progressive"), 1254))
    assert "fit_boundary_command_ids" in FitPathRequest.__dataclass_fields__, "fit must accept discovered boundaries"

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("supplied boundaries must not trigger another line discovery pass")

    monkeypatch.setattr(fitter, "_protected_line_spans", unexpected_detection)
    boundaries = features.fit_boundary_command_ids if use_discovered_cuts else ()
    request = FitPathRequest(features.geometry, PathFitOptions(0.5, 1.5, "progressive"),
        fit_boundary_command_ids=boundaries)
    result = engine.fit_path(request)
    endpoints = {item.source_command_id for item in result.correspondence.endpoint_bindings}
    assert set(boundaries).issubset(endpoints)
    if not use_discovered_cuts:
        assert len(result.correspondence.spans) == 2


@pytest.mark.parametrize("excluded,boundary", (("r1.p0.c32", "r1.p0.c32"), ("r1.p0.c0", "r1.p0.c256")))
def test_explicit_fit_boundaries_reject_foreign_or_excluded_source_commands(excluded, boundary):
    from vectoreyes.drawing_engine import FitPathRequest, PathFitOptions
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    assert "fit_boundary_command_ids" in FitPathRequest.__dataclass_fields__, "fit boundaries need source validation"
    source = source_geometry()
    options = PathFitOptions(0.5, 1.5, "progressive")
    engine = PythonDrawingEngine()
    with pytest.raises(ValueError, match="boundar.*source"):
        engine.fit_path(FitPathRequest(source, options, fit_boundary_command_ids=("other.c1",)))
    with pytest.raises(ValueError, match="boundar.*excluded|excluded.*boundar"):
        engine.fit_path(FitPathRequest(source, options, excluded_command_ids=(excluded,),
            fit_boundary_command_ids=(boundary,)))


def test_forced_corner_survives_recovery_candidate_cleanup():
    from vectoreyes.drawing_engine import DetectPathFeaturesRequest
    from vectoreyes.drawing_trace import TraceOptions
    from vectoreyes.path_geometry import ClosedPath, LineCommand, MoveCommand, PathGeometry
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    vertices = np.asarray([[92, 0], [50, 29], [39, 67], [0, 68], [-30, 53], [-60, 35],
        [-71, 0], [-70, -40], [-43, -75], [0, -51], [31, -54], [69, -40]], dtype=float)
    t = np.arange(16)[:, None] / 16
    ring = np.vstack([(1 - t) * a + t * b for a, b in zip(vertices, np.roll(vertices, -1, axis=0))])
    contour = np.vstack([ring, ring[:1]])
    source = PathGeometry(ClosedPath("p0", (MoveCommand("c0", tuple(contour[0])),
        *(LineCommand(f"c{index}", tuple(point)) for index, point in enumerate(contour[1:], 1)))))
    features = PythonDrawingEngine().detect_path_features(DetectPathFeaturesRequest(source,
        TraceOptions(fit_strategy="progressive"), 1254, ("c160",)))
    assert any(anchor.entry_command_id == anchor.exit_command_id == "c160" for anchor in features.geometry.anchors)
    legacy = fit._detect_path_features(contour, epsilon=1.5, progressive=True,
        canvas_extent=1254, forced_corners=ring[[160]])
    assert 160 in legacy.point_indices


def test_removing_detected_corner_does_not_retain_its_cut_implicitly(monkeypatch):
    from dataclasses import replace
    from vectoreyes.drawing_engine import DetectPathFeaturesRequest, FitPathRequest, PathFitOptions
    from vectoreyes.drawing_trace import TraceOptions
    from vectoreyes.python_drawing_engine import PythonDrawingEngine
    engine = PythonDrawingEngine()
    source = source_geometry()
    features = engine.detect_path_features(DetectPathFeaturesRequest(source,
        TraceOptions(simplify_tolerance=0.5, fit_strategy="progressive"), 1254, ("r1.p0.c240",)))
    assert any(anchor.entry_command_id == "r1.p0.c240" for anchor in features.geometry.anchors)
    assert "r1.p0.c240" not in features.fit_boundary_command_ids
    unprotected = replace(features.geometry, anchors=tuple(anchor for anchor in features.geometry.anchors
        if anchor.entry_command_id != "r1.p0.c240"))
    original = fit._fit_path_d_from_cuts
    observed = []

    def capture(ring, cuts, **kwargs):
        observed.extend(cuts)
        return original(ring, cuts, **kwargs)

    monkeypatch.setattr("vectoreyes.python_path_fitter._fit_path_d_from_cuts", capture)
    engine.fit_path(FitPathRequest(unprotected, PathFitOptions(0.5, 1.5, "progressive"),
        fit_boundary_command_ids=features.fit_boundary_command_ids))
    assert 240 not in observed

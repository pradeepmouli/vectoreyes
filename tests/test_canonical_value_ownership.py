from operator import setitem
from types import MappingProxyType

import pytest

from vectormark.drawing import Canvas, Drawing, DrawingRegion
from vectormark.drawing_engine import (
    DrawingActionResult,
    PathActionResult,
    TraceDrawingResult,
)
from vectormark.drawing_fill import (
    GradientStop,
    LinearGradientFill,
    RadialGradientFill,
    RasterFill,
    deserialize_fill,
    serialize_fill,
    transform_fill,
)
from vectormark.json_value import to_json_value
from vectormark.path_geometry import (
    AffineTransform,
    CloneGeometry,
    ClosedPath,
    InflectionAnchor,
    LineCommand,
    MoveCommand,
    PathGeometry,
    PathGeometryWithCutouts,
    PolygonPrimitive,
)


def _geometry(path_id: str, x: float = 0.0) -> PathGeometry:
    commands = [
        MoveCommand(f"{path_id}_m", (x, 0.0)),
        LineCommand(f"{path_id}_l1", (x + 2.0, 0.0)),
        LineCommand(f"{path_id}_l2", (x + 2.0, 2.0)),
        LineCommand(f"{path_id}_l3", (x, 0.0)),
    ]
    anchors = [InflectionAnchor(f"{path_id}_a", [f"{path_id}_l1"])]
    return PathGeometry(ClosedPath(path_id, commands), anchors)


def _drawing(diagnostics=None) -> Drawing:
    return Drawing(
        "drawing",
        Canvas(10.0, 10.0),
        [
            DrawingRegion(
                "root",
                children=[
                    DrawingRegion(
                        "leaf",
                        geometry=_geometry("leaf_path"),
                        source_region_ids=["source"],
                    )
                ],
            )
        ],
        diagnostics={} if diagnostics is None else diagnostics,
    )


def test_path_and_fill_sequences_are_owned_without_changing_values_or_svg():
    outer = _geometry("outer")
    cutout = _geometry("cutout", 0.5)
    cutouts = [cutout]
    compound = PathGeometryWithCutouts(outer, cutouts)
    expected_svg = f"{outer.to_svg_d()} {cutout.to_svg_d()}"

    transforms = [AffineTransform(tx=3.0), AffineTransform(ty=4.0)]
    clone = CloneGeometry("source", transforms)
    vertices = [(0, 0), (2, 0), (0, 2)]
    polygon = PolygonPrimitive(vertices)
    stops = [GradientStop(0.0, "#FF0000"), GradientStop(1.0, "#0000FF")]
    linear = LinearGradientFill((0, 0), (10, 0), stops)
    radial = RadialGradientFill((5, 5), 5.0, stops, (4, 4))

    assert cutouts == [cutout]
    assert transforms == [AffineTransform(tx=3.0), AffineTransform(ty=4.0)]
    assert vertices == [(0, 0), (2, 0), (0, 2)]
    assert stops == [GradientStop(0.0, "#ff0000"), GradientStop(1.0, "#0000ff")]
    cutouts.clear()
    transforms.reverse()
    vertices.clear()
    stops.clear()

    assert isinstance(outer.path.commands, tuple)
    assert isinstance(outer.anchors, tuple)
    assert compound.cutouts == (cutout,)
    assert compound.to_svg_d() == expected_svg
    assert clone.transforms == (AffineTransform(tx=3.0), AffineTransform(ty=4.0))
    assert polygon.vertices == ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))
    assert linear.stops == radial.stops == (
        GradientStop(0.0, "#ff0000"),
        GradientStop(1.0, "#0000ff"),
    )
    for owned in (
        outer.path.commands,
        outer.anchors,
        compound.cutouts,
        clone.transforms,
        polygon.vertices,
        linear.stops,
        radial.stops,
    ):
        with pytest.raises(TypeError):
            setitem(owned, 0, owned[0])


def test_affine_gradient_paint_round_trips_and_preserves_local_parameter():
    stops = (GradientStop(0.0, "#ff0000"), GradientStop(1.0, "#0000ff"))
    transform = AffineTransform(a=1.5, b=0.25, c=0.75, d=2.0, tx=11.0, ty=-7.0)
    linear = transform_fill(
        LinearGradientFill((1.0, 2.0), (9.0, 5.0), stops), transform
    )
    radial = transform_fill(
        RadialGradientFill((4.0, 6.0), 3.0, stops, (5.0, 6.5)), transform
    )
    raster = transform_fill(RasterFill((0.0, 0.0), 4.0, 3.0, "AAAA"), transform)

    assert linear.transform == radial.transform == transform
    assert deserialize_fill(serialize_fill(linear)) == linear
    assert deserialize_fill(serialize_fill(radial)) == radial
    assert deserialize_fill(serialize_fill(raster)) == raster

    q = (6.0, 4.0)
    p = (
        transform.a * q[0] + transform.c * q[1] + transform.tx,
        transform.b * q[0] + transform.d * q[1] + transform.ty,
    )
    determinant = transform.a * transform.d - transform.b * transform.c
    local = (
        (transform.d * (p[0] - transform.tx) - transform.c * (p[1] - transform.ty))
        / determinant,
        (-transform.b * (p[0] - transform.tx) + transform.a * (p[1] - transform.ty))
        / determinant,
    )
    direction = (
        linear.end[0] - linear.start[0],
        linear.end[1] - linear.start[1],
    )
    expected_t = (
        (q[0] - linear.start[0]) * direction[0]
        + (q[1] - linear.start[1]) * direction[1]
    ) / (direction[0] ** 2 + direction[1] ** 2)
    actual_t = (
        (local[0] - linear.start[0]) * direction[0]
        + (local[1] - linear.start[1]) * direction[1]
    ) / (direction[0] ** 2 + direction[1] ** 2)
    assert actual_t == pytest.approx(expected_t)

    expected_radius_t = (
        (q[0] - radial.center[0]) ** 2 + (q[1] - radial.center[1]) ** 2
    ) ** 0.5 / radial.radius
    actual_radius_t = (
        (local[0] - radial.center[0]) ** 2
        + (local[1] - radial.center[1]) ** 2
    ) ** 0.5 / radial.radius
    assert actual_radius_t == pytest.approx(expected_radius_t)


def test_drawing_sequences_are_owned_before_tree_and_uniqueness_validation():
    sources = ["source-a", "source-b"]
    leaf = DrawingRegion(
        "leaf",
        geometry=_geometry("leaf"),
        source_region_ids=sources,
    )
    children = [leaf]
    root = DrawingRegion("root", children=children)
    roots = [root]
    drawing = Drawing("drawing", Canvas(10.0, 10.0), roots)

    assert sources == ["source-a", "source-b"]
    assert children == [leaf]
    assert roots == [root]
    children.clear()
    sources.clear()
    roots.clear()

    assert leaf.source_region_ids == ("source-a", "source-b")
    assert root.children == (leaf,)
    assert drawing.regions == (root,)
    assert drawing.region("leaf") is leaf
    assert tuple(region.id for region in drawing.regions) == ("root",)
    for owned in (leaf.source_region_ids, root.children, drawing.regions):
        with pytest.raises(TypeError):
            setitem(owned, 0, owned[0])

    duplicate_sources = ["same", "same"]
    with pytest.raises(ValueError, match="source region IDs must be unique"):
        DrawingRegion(
            "invalid",
            geometry=_geometry("invalid"),
            source_region_ids=duplicate_sources,
        )
    assert duplicate_sources == ["same", "same"]


def test_canonical_diagnostics_are_deeply_detached_frozen_and_json_safe():
    sources = []
    owners = []
    for make_owner in (
        lambda diagnostics: DrawingRegion(
            "region",
            geometry=_geometry("region"),
            diagnostics=diagnostics,
        ),
        lambda diagnostics: _drawing(diagnostics),
        lambda diagnostics: TraceDrawingResult(_drawing(), object(), diagnostics),
        lambda diagnostics: PathActionResult(_geometry("action"), ["action_l1"], diagnostics),
        lambda diagnostics: DrawingActionResult(
            _drawing(),
            object(),
            ["root"],
            ["created"],
            ["retired"],
            diagnostics,
        ),
    ):
        source = {"nested": {"items": [1, 2]}}
        owner = make_owner(source)
        assert source == {"nested": {"items": [1, 2]}}
        sources.append(source)
        owners.append(owner)

    for source, owner in zip(sources, owners, strict=True):
        source["nested"]["items"].append(3)
        assert isinstance(owner.diagnostics, MappingProxyType)
        assert owner.diagnostics["nested"]["items"] == (1, 2)
        assert to_json_value(owner.diagnostics) == {"nested": {"items": [1, 2]}}
        with pytest.raises(TypeError):
            owner.diagnostics["nested"]["added"] = True
        with pytest.raises(TypeError):
            setitem(owner.diagnostics["nested"]["items"], 0, 9)


def test_engine_result_id_sequences_are_owned_and_evidence_remains_opaque():
    path_ids = ["path_l1", "path_l2"]
    path_result = PathActionResult(_geometry("path"), path_ids)
    affected = ["root", "leaf"]
    created = ["created-a", "created-b"]
    retired = ["retired-a", "retired-b"]
    evidence = {"mutable": []}
    trace_evidence = {"mutable": []}
    trace_result = TraceDrawingResult(_drawing(), trace_evidence)
    drawing_result = DrawingActionResult(
        _drawing(), evidence, affected, created, retired
    )

    path_ids.clear()
    affected.reverse()
    created.clear()
    retired.clear()
    evidence["mutable"].append("retained by evidence contract")
    trace_evidence["mutable"].append("also retained")

    assert path_result.affected_command_ids == ("path_l1", "path_l2")
    assert drawing_result.affected_region_ids == ("root", "leaf")
    assert drawing_result.created_region_ids == ("created-a", "created-b")
    assert drawing_result.retired_region_ids == ("retired-a", "retired-b")
    assert drawing_result.evidence is evidence
    assert drawing_result.evidence == {"mutable": ["retained by evidence contract"]}
    assert trace_result.evidence is trace_evidence
    assert trace_result.evidence == {"mutable": ["also retained"]}
    for owned in (
        path_result.affected_command_ids,
        drawing_result.affected_region_ids,
        drawing_result.created_region_ids,
        drawing_result.retired_region_ids,
    ):
        with pytest.raises(TypeError):
            setitem(owned, 0, owned[0])


def test_invalid_owned_metadata_rejects_precisely_without_mutating_sources():
    cyclic = []
    cyclic.append(cyclic)
    cycle_diagnostics = {"cycle": cyclic}
    with pytest.raises(TypeError, match="cyclic container values are unsupported"):
        DrawingRegion(
            "region",
            geometry=_geometry("cycle"),
            diagnostics=cycle_diagnostics,
        )
    assert cycle_diagnostics == {"cycle": cyclic}
    assert cyclic[0] is cyclic

    unsupported = object()
    unsupported_diagnostics = {"unsupported": unsupported}
    with pytest.raises(TypeError, match="unsupported value type: object"):
        TraceDrawingResult(_drawing(), object(), unsupported_diagnostics)
    assert unsupported_diagnostics == {"unsupported": unsupported}

    unsorted_stops = [GradientStop(1.0, "#ffffff"), GradientStop(0.0, "#000000")]
    with pytest.raises(ValueError, match="gradient stops must be sorted by offset"):
        LinearGradientFill((0, 0), (1, 0), unsorted_stops)
    assert unsorted_stops == [
        GradientStop(1.0, "#ffffff"),
        GradientStop(0.0, "#000000"),
    ]

import copy
from dataclasses import asdict

import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.drawing_adapter import drawing_from_region_samples
from vectormark.fit import Shape
from vectormark.corner_diagnostics import path_corner_diagnostics
from vectormark.refinement.region_sample import RegionSample
from vectormark.path_geometry import (
    CornerAnchor,
    ExtremumAnchor,
    InflectionAnchor,
    JunctionAnchor,
    PathGeometry,
    PathGeometryWithCutouts,
    geometry_from_svg_d,
)


def _path(d: str, region_id: str = "r") -> PathGeometry:
    geometry = geometry_from_svg_d(d, region_id=region_id)
    assert isinstance(geometry, PathGeometry)
    return geometry


def test_corner_anchor_derived_membership_and_dataclass_round_trip():
    anchor = CornerAnchor("r.p0.rounded", "r.p0.c1", "r.p0.c3", ["r.p0.c2", "r.p0.c3"], None)
    assert anchor.command_ids == ("r.p0.c1", "r.p0.c2", "r.p0.c3")
    restored = CornerAnchor(**asdict(anchor))
    assert restored == anchor
    assert restored.command_ids == anchor.command_ids
    assert restored.kind == "corner"


def test_path_geometry_validates_point_rounded_radius_and_exact_ownership():
    geometry = _path("M0 0 L10 0 Q12 0 12 2 L0 2 L0 0 Z")
    point = CornerAnchor("point", "r.p0.c0", "r.p0.c0", radius=None)
    rounded = CornerAnchor("round", "r.p0.c1", "r.p0.c2", ("r.p0.c2",), 2.0)
    PathGeometry(geometry.path, (point, rounded))
    with pytest.raises(ValueError, match="point corners"):
        CornerAnchor("bad", "r.p0.c1", "r.p0.c1", ("r.p0.c2",))
    with pytest.raises(ValueError, match="rounded corners"):
        CornerAnchor("bad", "r.p0.c1", "r.p0.c2")
    with pytest.raises(ValueError, match="non-negative"):
        CornerAnchor("bad", "r.p0.c1", "r.p0.c2", ("r.p0.c2",), -1)
    with pytest.raises(ValueError, match="exactly match"):
        PathGeometry(geometry.path, (CornerAnchor("bad", "r.p0.c1", "r.p0.c3", ("r.p0.c3",)),))


def test_directed_wrap_requires_explicit_closing_drawable_and_owns_it():
    explicit = _path("M0 0 L10 0 L10 10 L0 10 L0 0 Z")
    wrapped = CornerAnchor(
        "wrap", "r.p0.c3", "r.p0.c1", ("r.p0.c4", "r.p0.c1")
    )
    PathGeometry(explicit.path, (wrapped,))
    implicit = _path("M0 0 L10 0 L10 10 L0 10 Z")
    with pytest.raises(ValueError, match="implicit closing"):
        PathGeometry(
            implicit.path,
            (CornerAnchor("wrap", "r.p0.c3", "r.p0.c1", ("r.p0.c1",)),),
        )


def test_shared_boundary_is_allowed_but_overlapping_ownership_is_ambiguous():
    geometry = _path("M0 0 Q2 0 3 1 Q4 2 4 4 Q4 6 5 7 Q6 8 8 8 L0 8 L0 0 Z")
    first = CornerAnchor("first", "r.p0.c0", "r.p0.c2", ("r.p0.c1", "r.p0.c2"))
    second = CornerAnchor("second", "r.p0.c2", "r.p0.c4", ("r.p0.c3", "r.p0.c4"))
    PathGeometry(geometry.path, (first, second))
    overlapping = CornerAnchor("overlap", "r.p0.c1", "r.p0.c3", ("r.p0.c2", "r.p0.c3"))
    with pytest.raises(ValueError, match="ownership is ambiguous"):
        PathGeometry(geometry.path, (first, overlapping))


def test_non_corner_anchor_variants_keep_their_contracts():
    command = ("r.p0.c1",)
    assert InflectionAnchor("i", command).command_ids == command
    assert ExtremumAnchor("e", list(command)).command_ids == command
    assert JunctionAnchor("j", command, "other").reference_junction_id == "other"


def _region(d: str, *, drawing_id: str, diagnostics: dict, region_id: int = 1) -> RegionSample:
    return RegionSample.from_shape(
        id=region_id,
        shape=Shape("path", {"d": d, "fill_rule": "evenodd"}),
        fill=FlatFill("#112233"),
        z=0,
        raster=np.ones((2, 2), dtype=bool),
        drawing_id=drawing_id,
        source_regions=(drawing_id,),
        diagnostics=diagnostics,
    )


def test_adapter_ingresses_prefit_corners_on_outer_and_cutout_without_mutation():
    d = "M0 0 L10 0 Q12 0 12 2 L0 2 L0 0 Z M3 .5 L7 .5 Q8 .5 8 1.5 L3 1.5 L3 .5 Z"
    corners = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (12, 2)),))
    region = _region(d, drawing_id="root-child", diagnostics={"corners": corners})
    snapshot = copy.deepcopy(region)
    drawing = drawing_from_region_samples("drawing", width=20, height=20, regions=(region,))
    geometry = drawing.regions[0].geometry
    assert isinstance(geometry, PathGeometryWithCutouts)
    assert geometry.outer.to_svg_d() == geometry_from_svg_d(d, region_id="root-child", fill_rule="evenodd").outer.to_svg_d()
    assert geometry.outer.anchors[0].command_ids == (
        "root-child.p0.c1", "root-child.p0.c2"
    )
    assert geometry.cutouts[0].anchors == ()
    assert region == snapshot
    assert drawing.regions[0].fill == drawing_from_region_samples(
        "drawing", width=20, height=20, regions=(region,)
    ).regions[0].fill


def test_adapter_scopes_cutout_corner_and_preserves_stale_skip():
    d = "M0 0 L10 0 L10 10 L0 10 L0 0 Z M2 2 L8 2 Q9 2 9 3 L2 3 L2 2 Z"
    corners = path_corner_diagnostics(d, declared_corner_spans=(((8, 2), (9, 3)),))
    corners["spans"] = [span for span in corners["spans"] if span["subpath"] == 1]
    region = _region(d, drawing_id="r", diagnostics={"corners": corners})
    geometry = drawing_from_region_samples("drawing", width=10, height=10, regions=(region,)).regions[0].geometry
    assert isinstance(geometry, PathGeometryWithCutouts)
    assert geometry.outer.anchors == ()
    assert geometry.cutouts[0].anchors[0].id == "r.p1.rounded_corner0"

    stale = _region(d, drawing_id="stale", diagnostics={"corners": {**corners, "path": "M0 0 L1 1 Z", "skipped": [{"reason": "prior"}]}})
    output = drawing_from_region_samples("drawing", width=10, height=10, regions=(stale,)).regions[0]
    assert isinstance(output.geometry, PathGeometryWithCutouts)
    assert all(not path.anchors for path in (output.geometry.outer, *output.geometry.cutouts))
    assert [item["reason"] for item in output.diagnostics["corners"]["skipped"]] == [
        "prior", "missing_or_stale_provenance"
    ]
    assert stale.diagnostics["corners"]["skipped"] == [{"reason": "prior"}]


def test_adapter_uses_child_public_region_id_for_corner_addresses():
    d = "M0 0 L10 0 Q12 0 12 2 L0 2 L0 0 Z"
    child = _region(
        d,
        drawing_id="root-2",
        diagnostics={"corners": path_corner_diagnostics(
            d, declared_corner_spans=(((10, 0), (12, 2)),)
        )},
        region_id=2,
    )
    root = RegionSample.branch(id=1, children=(child,), drawing_id="root")
    drawing = drawing_from_region_samples("drawing", width=20, height=20, regions=(root,))
    geometry = drawing.regions[0].children[0].geometry
    assert isinstance(geometry, PathGeometry)
    assert geometry.anchors[0].command_ids == ("root-2.p0.c1", "root-2.p0.c2")


def test_adapter_skips_overlapping_span_but_keeps_valid_corner():
    d = "M0 0 Q2 0 3 1 Q4 2 4 4 Q4 6 5 7 L0 8 L0 0 Z"
    corners = path_corner_diagnostics(
        d,
        declared_corner_spans=(((0, 0), (4, 4)), ((3, 1), (5, 7))),
    )
    region = _region(d, drawing_id="r", diagnostics={"corners": corners})
    output = drawing_from_region_samples("drawing", width=10, height=10, regions=(region,)).regions[0]
    assert isinstance(output.geometry, PathGeometry)
    assert [anchor.id for anchor in output.geometry.anchors] == ["r.p0.rounded_corner0"]
    assert output.diagnostics["corners"]["skipped"][-1]["reason"] == "overlapping_corner_ownership"


@pytest.mark.parametrize(
    "malformed",
    [
        {"source": "prefit"},
        {"source": "prefit", "corner_id": "bad", "subpath": "0", "kind": "point", "point": (0, 0)},
        {"source": "prefit", "corner_id": "bad", "subpath": 0, "kind": "point", "point": (0,)},
        {"source": "prefit", "corner_id": "bad", "subpath": 0, "kind": "curve", "start_point": (0, 0)},
        {"source": "prefit", "corner_id": "bad", "subpath": 0, "kind": "unknown", "point": (0, 0)},
    ],
)
def test_adapter_skips_each_malformed_prefit_declaration_without_mutating_source(malformed):
    d = "M0 0 L10 0 Q12 0 12 2 L0 2 L0 0 Z"
    valid = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (12, 2)),))["spans"][0]
    evidence = {"path": d, "spans": [malformed, valid]}
    region = _region(d, drawing_id="r", diagnostics={"corners": evidence})
    snapshot = copy.deepcopy(region.diagnostics)

    output = drawing_from_region_samples("drawing", width=20, height=20, regions=(region,)).regions[0]

    assert isinstance(output.geometry, PathGeometry)
    assert [anchor.id for anchor in output.geometry.anchors] == ["r.p0.rounded_corner0"]
    assert len(output.diagnostics["corners"]["skipped"]) == 1
    assert output.diagnostics["corners"]["skipped"][0]["reason"] == "invalid_corner_provenance"
    assert region.diagnostics == snapshot

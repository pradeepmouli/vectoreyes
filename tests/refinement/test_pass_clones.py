import numpy as np

from vectoreyes.skia_geometry import SkPath, affinity
from vectoreyes.candidate import FlatFill, LinearGradientFill
from vectoreyes.emit import refinement_objects_to_svg
from vectoreyes.fit import Shape
from vectoreyes.refinement.framework import select_refinements
from vectoreyes.refinement.region_sample import RegionSample
from vectoreyes.refinement.passes.clones import clones_pass


def _mask_for_polygon(poly: SkPath, shape_hw: tuple[int, int] = (96, 96)) -> np.ndarray:
    from vectoreyes.refinement.gate import rasterize

    return rasterize(poly, shape_hw)


def _square(
    obj_id: int,
    *,
    size: float = 12.0,
    center: tuple[float, float] = (20.0, 20.0),
    angle_deg: float = 0.0,
    fill: object = FlatFill("#112233"),
) -> RegionSample:
    cx, cy = center
    half = size / 2.0
    poly = SkPath(
        shell=[
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
    )
    if angle_deg:
        poly = affinity.rotate(poly, angle_deg, origin=(cx, cy))

    return RegionSample(
        id=obj_id,
        current=Shape(
            "path",
            {"d": f"M{cx - half} {cy - half} L{cx + half} {cy - half} L{cx + half} {cy + half} L{cx - half} {cy + half} Z"},
        ),
        fill=fill,
        z=0,
        footprint=poly,
    )


def _circle(obj_id: int, *, center=(20.0, 20.0), radius=6.0, fill="#778899") -> RegionSample:
    poly = SkPath.make_circle(center[0], center[1], radius)
    return RegionSample(
        id=obj_id,
        current=Shape("circle", {"cx": center[0], "cy": center[1], "r": radius}),
        fill=FlatFill(fill),
        z=0,
        footprint=poly,
    )


def test_clones_pass_copies_translated_square_geometry_with_same_flat_fill():
    canonical = _square(1, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(2, center=(52.0, 40.0), fill=FlatFill("#112233"))
    objects = [clone, canonical]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = select_refinements(objects, masks, [clones_pass])

    assert [obj.id for obj in out] == [1, 2]
    assert out[0].current.kind == "path"
    assert out[1].current.kind == "use"
    assert out[1].current.params["href_obj_id"] == 1
    assert out[1].diagnostics["clones"]["matched_source"] == 1


def test_clones_pass_matches_rotated_congruent_square():
    canonical = _square(1, center=(22.0, 22.0), fill=FlatFill("#102030"))
    rotated = _square(2, center=(60.0, 48.0), angle_deg=30.0, fill=FlatFill("#102030"))
    objects = [canonical, rotated]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = select_refinements(objects, masks, [clones_pass])

    assert [obj.id for obj in out] == [1, 2]
    clone_obj = out[1]
    assert clone_obj.current.kind == "use"
    assert clone_obj.diagnostics["clones"]["matched_source"] == 1


def test_clones_pass_skips_non_congruent_shapes():
    square = _square(1, center=(20.0, 20.0))
    circle = _circle(2, center=(50.0, 20.0))
    objects = [square, circle]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    proposals = clones_pass(objects, masks)
    out = select_refinements(objects, masks, [clones_pass])

    assert proposals == []
    assert [obj.current.kind for obj in out] == ["path", "circle"]


def test_clones_pass_reuses_geometry_for_recolored_clone_proposals():
    canonical = _square(1, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(2, center=(52.0, 40.0), fill=FlatFill("#abcdef"))
    objects = [canonical, clone]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = select_refinements(objects, masks, [clones_pass])
    assert [obj.current.kind for obj in out] == ["path", "use"]
    assert out[1].fill == FlatFill("#abcdef")
    assert out[1].diagnostics["clones"]["matched_source"] == 1


def test_clones_pass_checks_nearby_descriptor_buckets_before_rejecting(monkeypatch):
    canonical = _square(1, size=10.0, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(2, size=10.03, center=(52.0, 40.0), fill=FlatFill("#abcdef"))
    objects = [canonical, clone]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    monkeypatch.setattr(
        "vectoreyes.clone_detection._best_transform",
        lambda _canonical, target, **_kwargs: ((1.0, 0.0, 0.0, 1.0, 34.0, 22.0), target),
    )
    proposals = clones_pass(objects, masks)

    assert [proposal.obj_ids for proposal in proposals] == [(2,)]


def test_clones_pass_does_not_use_raster_mask_as_acceptance_gate():
    canonical = _square(1, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(2, center=(52.0, 40.0), fill=FlatFill("#abcdef"))
    objects = [canonical, clone]
    masks = {
        canonical.id: _mask_for_polygon(canonical.footprint),
        clone.id: np.zeros((96, 96), dtype=bool),
    }

    proposals = clones_pass(objects, masks)

    assert [proposal.obj_ids for proposal in proposals] == [(2,)]


def test_clones_pass_skips_self_symmetric_regions():
    canonical = _square(1, center=(18.0, 18.0), fill=FlatFill("#112233")).with_diagnostics(
        {"symmetry": {"accepted": True, "mode": "self"}}
    )
    clone = _square(2, center=(52.0, 40.0), fill=FlatFill("#abcdef"))
    objects = [canonical, clone]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    assert clones_pass(objects, masks) == []


def test_clones_pass_reuses_geometry_for_gradient_filled_clone_proposals():
    canonical = _square(1, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(
        2,
        center=(52.0, 40.0),
        fill=LinearGradientFill(
            {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0},
            [(0.0, "#000000"), (1.0, "#ffffff")],
        ),
    )
    objects = [canonical, clone]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    assert [proposal.obj_ids for proposal in clones_pass(objects, masks)] == [(2,)]
    out = select_refinements(objects, masks, [clones_pass])
    assert [obj.current.kind for obj in out] == ["path", "use"]
    assert out[1].fill == clone.fill


def test_clones_pass_does_not_chain_targets_as_canonicals(monkeypatch):
    canonical = _square(1, center=(18.0, 18.0), fill=FlatFill("#112233"))
    middle = _square(2, center=(42.0, 18.0), fill=FlatFill("#112233"))
    target = _square(3, center=(66.0, 18.0), fill=FlatFill("#112233"))

    def _nontransitive_match(canonical_flat, target_flat, **_kwargs):
        pair = (round(canonical_flat.centroid.x), round(target_flat.centroid.x))
        if pair in {(18, 42), (42, 66)}:
            return (1.0, 0.0, 0.0, 1.0, target_flat.centroid.x - canonical_flat.centroid.x, 0.0), target_flat
        return None

    monkeypatch.setattr("vectoreyes.clone_detection._best_transform", _nontransitive_match)
    objects = [canonical, middle, target]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    proposals = clones_pass(objects, masks)

    assert [proposal.obj_ids for proposal in proposals] == [(2,)]


def test_refinement_object_svg_resolves_clone_href_to_emitted_id():
    canonical = _square(10, center=(18.0, 18.0), fill=FlatFill("#112233"))
    clone = _square(20, center=(52.0, 40.0), fill=FlatFill("#112233"))
    objects = [clone, canonical]
    masks = {obj.id: _mask_for_polygon(obj.footprint) for obj in objects}

    out = select_refinements(objects, masks, [clones_pass])
    body = refinement_objects_to_svg(out)

    assert body[0].startswith('<path id="s0"')
    assert body[1].startswith('<use id="s1"')
    assert 'href="#s0"' in body[1]

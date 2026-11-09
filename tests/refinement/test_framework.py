import numpy as np
from typing import cast

from vectoreyes.candidate import FlatFill
from vectoreyes.fit import Shape
from vectoreyes.refinement.automatic_replay import ProposalIntent
from vectoreyes.refinement.framework import Proposal as _Proposal, select_refinements
from vectoreyes.refinement.gate import rasterize
from vectoreyes.refinement.region_sample import RegionSample


def Proposal(*args, **kwargs):
    """Build a framework-only fixture that deliberately has no replay capture."""
    return _Proposal(*args, **kwargs, intent=cast(ProposalIntent, None))


def _rect_obj(i, w, h, *, x=0.0, y=0.0):
    return RegionSample(
        i,
        Shape("rect", {"x": float(x), "y": float(y), "w": float(w), "h": float(h)}),
        FlatFill("#000"),
        0,
    )


def _path_square_obj(i, *, x=0.0, y=0.0, size=10.0):
    x2 = x + size
    y2 = y + size
    d = f"M{x} {y} L{x2} {y} L{x2} {y2} L{x} {y2} Z"
    return RegionSample(i, Shape("path", {"d": d}), FlatFill("#000"), 0)


def test_framework_rejects_proposals_that_diverge_from_the_trace():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True

    good = lambda os, ms: [Proposal((1,), [os[0]])]
    out = select_refinements(objs, masks, [good])
    assert len(out) == 1

    bad = lambda os, ms: [Proposal((1,), [_rect_obj(1, 2, 2)])]
    out2 = select_refinements(objs, masks, [bad])
    assert abs(out2[0].current.params["w"] - 10) < 1e-9
    assert out2[0] is not objs[0]
    assessment = out2[0].diagnostics["<lambda>"]["geometry_assessment"]
    assert assessment["accepted"] is False
    assert assessment["reasons"] == ["relative_area_error_exceeded"]
    assert objs[0].diagnostics == {}


def test_framework_explicit_zero_budget_rejects_nonidentical_policy_proposal():
    obj = _rect_obj(1, 10, 10)
    masks = {1: rasterize(obj.footprint, (20, 20))}

    def almost_same(os, ms):
        from vectoreyes.quality import GeometryPolicy
        return [Proposal((1,), [_rect_obj(1, 9.9, 10)], GeometryPolicy(max_boundary_error_px=1.0))]

    out = select_refinements([obj], masks, [almost_same], budget=0.0)
    assert out[0].current.params["w"] == 10.0
    assert "relative_area_error_exceeded" in out[0].diagnostics["almost_same"]["geometry_assessment"]["reasons"]


def test_framework_uses_the_trace_not_the_raster_as_the_quality_baseline():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}

    def primitives_pass(_os, _ms):
        return [Proposal((1,), [_rect_obj(9, 10, 10)])]

    out = select_refinements(objs, masks, [primitives_pass])

    assert [obj.id for obj in out] == [9]


def test_framework_orders_multi_id_proposals_and_unions_masks_for_new_ids():
    objs = [_rect_obj(1, 10, 10), _rect_obj(2, 6, 6)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:16] = True

    def first_pass(os, ms):
        return [
            Proposal((2,), [_rect_obj(2, 6, 6)]),
            Proposal((1, 2), [_rect_obj(9, 10, 10)]),
        ]

    def second_pass(os, ms):
        assert [obj.id for obj in os] == [9]
        assert np.array_equal(ms[9], masks[1] | masks[2])
        return []

    out = select_refinements(objs, masks, [first_pass, second_pass])
    assert [obj.id for obj in out] == [9]


def test_framework_rejects_multi_id_new_id_that_drops_trace_coverage():
    objs = [
        _rect_obj(1, 10, 10, x=0, y=0),
        _rect_obj(2, 10, 10, x=10, y=0),
    ]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True

    def bad_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(9, 2, 2, x=0, y=0)])]

    out = select_refinements(objs, masks, [bad_pass])

    assert [obj.id for obj in out] == [1, 2]


def test_framework_rejects_multi_id_proposal_that_drops_consumed_trace_coverage():
    objs = [
        _rect_obj(1, 10, 10, x=0, y=0),
        _rect_obj(2, 10, 10, x=10, y=0),
    ]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True

    def drop_second_object_pass(os, ms):
        return [Proposal((1, 2), [os[0]])]

    out = select_refinements(objs, masks, [drop_second_object_pass])

    assert [obj.id for obj in out] == [1, 2]


def test_framework_accepts_multi_id_new_id_with_union_coverage():
    objs = [
        _rect_obj(1, 10, 10, x=0, y=0),
        _rect_obj(2, 10, 10, x=10, y=0),
    ]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True
    union_mask = masks[1] | masks[2]

    def merge_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(9, 20, 10, x=0, y=0)])]

    def verify_union_mask(os, ms):
        assert [obj.id for obj in os] == [9]
        assert np.array_equal(ms[9], union_mask)
        return []

    out = select_refinements(objs, masks, [merge_pass, verify_union_mask])

    assert [obj.id for obj in out] == [9]
    assert out[0].current.params["w"] == 20.0


def test_framework_accepts_single_id_split_and_assigns_replacement_masks():
    obj = _rect_obj(1, 20, 10, x=0, y=0)
    masks = {1: rasterize(obj.footprint, (20, 30))}

    def split_pass(os, ms):
        return [Proposal((1,), [_rect_obj(1, 10, 10, x=0, y=0), _rect_obj(9, 10, 10, x=10, y=0)])]

    def verify_split_masks(os, ms):
        assert [current.id for current in os] == [1, 9]
        expected_left = rasterize(os[0].footprint, (20, 30))
        expected_right = rasterize(os[1].footprint, (20, 30))
        assert np.array_equal(ms[1], expected_left)
        assert np.array_equal(ms[9], expected_right)
        assert int(ms[1].sum()) < int(masks[1].sum())
        assert int(ms[9].sum()) < int(masks[1].sum())
        return []

    out = select_refinements([obj], masks, [split_pass, verify_split_masks])

    assert [current.id for current in out] == [1, 9]


def test_framework_updates_region_raster_and_pass_diagnostics():
    obj = _rect_obj(1, 20, 10, x=0, y=0)
    masks = {1: rasterize(obj.footprint, (20, 30))}

    def split_for_diagnostics(os, ms):
        return [Proposal((1,), [_rect_obj(1, 10, 10, x=0, y=0), _rect_obj(9, 10, 10, x=10, y=0)])]

    out = select_refinements([obj], masks, [split_for_diagnostics])
    by_id = {region.id: region for region in out}

    assert np.array_equal(by_id[1].raster, rasterize(by_id[1].footprint, (20, 30)))
    assert np.array_equal(by_id[9].raster, rasterize(by_id[9].footprint, (20, 30)))
    assert by_id[1].diagnostics["split_for_diagnostics"]["accepted"] is True
    assert by_id[1].diagnostics["split_for_diagnostics"]["proposal_ids"] == [1]
    assert by_id[9].diagnostics["split_for_diagnostics"]["proposal_ids"] == [1]


def test_framework_runs_passes_inside_branch_children():
    path_child = _path_square_obj(1, size=10.0)
    child_mask = rasterize(path_child.footprint, (20, 20))
    path_child = RegionSample(
        id=path_child.id,
        current=path_child.current,
        fill=path_child.fill,
        z=path_child.z,
        raster=child_mask,
        footprint=path_child.footprint,
    )
    branch = RegionSample.branch(id=9, children=[path_child])

    def promote_child(os, ms):
        if any(not obj.is_leaf for obj in os):
            return []
        assert [obj.id for obj in os] == [1]
        assert 1 in ms
        replacement = os[0].with_current(
            Shape("rect", {"x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0}),
            footprint=os[0].footprint,
        )
        return [Proposal((1,), [replacement])]

    out = select_refinements([branch], {9: branch.raster}, [promote_child])

    assert len(out) == 1
    assert out[0].is_branch
    assert out[0].diagnostics["promote_child"]["children_optimized"] is True
    child = out[0].children[0]
    assert child.current.kind == "rect"
    assert child.diagnostics["promote_child"]["accepted"] is True
    assert np.array_equal(out[0].raster, child.raster)


def test_framework_rejects_replacement_id_aliasing_live_object():
    objs = [
        _rect_obj(1, 10, 10, x=0, y=0),
        _rect_obj(2, 10, 10, x=10, y=0),
        _rect_obj(3, 5, 5, x=0, y=10),
    ]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
        3: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True
    masks[3][10:15, 0:5] = True
    original_masks = {obj_id: mask.copy() for obj_id, mask in masks.items()}

    def bad_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(3, 20, 10, x=0, y=0)])]

    def verify_unchanged(os, ms):
        assert [obj.id for obj in os] == [1, 2, 3]
        for obj_id in (1, 2, 3):
            assert np.array_equal(ms[obj_id], original_masks[obj_id])
        return []

    out = select_refinements(objs, masks, [bad_pass, verify_unchanged])

    assert [obj.id for obj in out] == [1, 2, 3]
    for obj_id in (1, 2, 3):
        assert np.array_equal(masks[obj_id], original_masks[obj_id])


def test_framework_sorts_replacements_before_later_passes():
    objs = [_rect_obj(1, 5, 5), _rect_obj(2, 5, 5, x=5)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:5, 0:5] = True
    masks[2][0:5, 5:10] = True

    def reverse_order_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(8, 9, 4), _rect_obj(7, 9, 4)])]

    def verify_sorted(os, ms):
        assert [obj.id for obj in os] == [7, 8]
        return []

    out = select_refinements(objs, masks, [reverse_order_pass, verify_sorted], budget=2.0)

    assert [obj.id for obj in out] == [7, 8]


def test_framework_aggregate_gate_is_independent_of_replacement_order():
    objs = [
        _rect_obj(1, 10, 10, x=0, y=0),
        _rect_obj(2, 10, 10, x=10, y=0),
    ]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True

    def forward_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(8, 10, 10, x=10), _rect_obj(7, 10, 10)])]

    def reverse_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(7, 10, 10), _rect_obj(8, 10, 10, x=10)])]

    forward = select_refinements(objs, masks, [forward_pass])
    reverse = select_refinements(objs, masks, [reverse_pass])

    assert [obj.id for obj in forward] == [obj.id for obj in reverse]


def test_framework_multi_id_matching_replacement_uses_own_mask_not_union():
    objs = [_rect_obj(1, 10, 10), _rect_obj(2, 10, 10, x=10)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True

    def replace_one_consumed_id(os, ms):
        return [Proposal((1, 2), [_path_square_obj(1, size=9.0), os[1]])]

    out = select_refinements(objs, masks, [replace_one_consumed_id], budget=0.25)

    assert [obj.id for obj in out] == [1, 2]
    assert out[0].current.kind == "path"


def test_framework_tiebreaks_same_consumed_ids_by_replacement_values():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True

    def conflicting_pass(os, ms):
        return [
            Proposal((1,), [_rect_obj(9, 10, 10)]),
            Proposal((1,), [_rect_obj(8, 10, 10)]),
        ]

    out = select_refinements(objs, masks, [conflicting_pass])

    assert [obj.id for obj in out] == [8]


def test_framework_rejects_duplicate_replacement_ids_without_mutation():
    objs = [_rect_obj(1, 10, 10), _rect_obj(2, 10, 10, x=10)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True
    original_masks = {obj_id: mask.copy() for obj_id, mask in masks.items()}

    def duplicate_id_pass(os, ms):
        return [Proposal((1, 2), [_rect_obj(9, 20, 10), _rect_obj(9, 20, 10)])]

    def verify_unchanged(os, ms):
        assert [obj.id for obj in os] == [1, 2]
        for obj_id in (1, 2):
            assert np.array_equal(ms[obj_id], original_masks[obj_id])
        return []

    out = select_refinements(objs, masks, [duplicate_id_pass, verify_unchanged])

    assert [obj.id for obj in out] == [1, 2]


def test_framework_rejects_empty_replacements_without_mutation():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True
    original_mask = masks[1].copy()

    def empty_pass(os, ms):
        return [Proposal((1,), [])]

    def verify_unchanged(os, ms):
        assert [obj.id for obj in os] == [1]
        assert np.array_equal(ms[1], original_mask)
        return []

    out = select_refinements(objs, masks, [empty_pass, verify_unchanged])

    assert [obj.id for obj in out] == [1]


def test_framework_canonicalizes_input_order_for_later_passes():
    objs = [_rect_obj(2, 5, 5, x=5), _rect_obj(1, 5, 5)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:5, 0:5] = True
    masks[2][0:5, 5:10] = True

    def verify_canonical(os, ms):
        assert [obj.id for obj in os] == [1, 2]
        return []

    out = select_refinements(objs, masks, [verify_canonical])

    assert [obj.id for obj in out] == [1, 2]


def test_framework_rejects_duplicate_consumed_ids_without_mutation():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True
    original_mask = masks[1].copy()

    def duplicate_consumed_pass(os, ms):
        return [Proposal((1, 1), [_rect_obj(9, 10, 10)])]

    def verify_unchanged(os, ms):
        assert [obj.id for obj in os] == [1]
        assert np.array_equal(ms[1], original_mask)
        return []

    out = select_refinements(objs, masks, [duplicate_consumed_pass, verify_unchanged])

    assert [obj.id for obj in out] == [1]


def test_framework_rejects_same_pass_consumption_of_created_id():
    objs = [_rect_obj(1, 10, 10)]
    masks = {1: np.zeros((20, 20), bool)}
    masks[1][0:10, 0:10] = True

    def chaining_pass(os, ms):
        return [
            Proposal((1,), [_rect_obj(9, 10, 10)]),
            Proposal((9,), [_rect_obj(10, 10, 10)]),
        ]

    def verify_only_first_proposal_applied(os, ms):
        assert [obj.id for obj in os] == [9]
        assert 9 in ms
        assert 10 not in ms
        return []

    out = select_refinements(objs, masks, [chaining_pass, verify_only_first_proposal_applied])

    assert [obj.id for obj in out] == [9]


def test_framework_rejects_same_pass_reuse_of_created_replacement_id():
    objs = [_rect_obj(1, 10, 10), _rect_obj(2, 10, 10, x=10)]
    masks = {
        1: np.zeros((20, 20), bool),
        2: np.zeros((20, 20), bool),
    }
    masks[1][0:10, 0:10] = True
    masks[2][0:10, 10:20] = True

    def duplicate_created_id_pass(os, ms):
        return [
            Proposal((1,), [_rect_obj(9, 10, 10)]),
            Proposal((2,), [_rect_obj(9, 10, 10, x=10)]),
        ]

    out = select_refinements(objs, masks, [duplicate_created_id_pass])

    assert [obj.id for obj in out] == [2, 9]
    assert len({obj.id for obj in out}) == len(out)

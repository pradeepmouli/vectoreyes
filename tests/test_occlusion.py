# SPDX-License-Identifier: MIT
import numpy as np
from vectormark.types import Region
from vectormark.occlusion import ScenePrimitive, region_adjacency, has_bite


def _region(label, mask, color="#000000"):
    return Region(label, mask.astype(bool), color)


def test_scene_primitive_holds_geometry():
    p = ScenePrimitive(kind="circle", params={"cx": 1.0, "cy": 2.0, "r": 3.0}, color_hex="#FF0000", z=0)
    assert p.kind == "circle" and p.params["r"] == 3.0 and p.color_hex == "#FF0000" and p.z == 0


def test_region_adjacency_touching_vs_separate():
    a = np.zeros((20, 30), bool); a[5:15, 4:14] = True
    b = np.zeros((20, 30), bool); b[5:15, 14:24] = True   # shares the x=14 seam with a
    c = np.zeros((20, 30), bool); c[5:15, 26:29] = True   # gap from b
    adj = region_adjacency([_region(1, a), _region(2, b), _region(3, c)])
    assert adj[1] == {2} and adj[2] == {1} and adj[3] == set()


def test_has_bite_crescent_vs_convex():
    H = W = 80
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - 35) ** 2 + (yy - 40) ** 2 <= 28 ** 2
    occ = (xx - 60) ** 2 + (yy - 40) ** 2 <= 28 ** 2
    crescent = disk & ~occ
    assert has_bite(crescent) is True
    assert has_bite(disk) is False                      # convex
    rect = np.zeros((H, W), bool); rect[10:70, 10:40] = True
    assert has_bite(rect) is False


from vectormark.occlusion import label_boundary, complete_primitive


def test_label_boundary_marks_the_bite_as_seam():
    H = W = 90
    yy, xx = np.ogrid[:H, :W]
    disk = (xx - 38) ** 2 + (yy - 45) ** 2 <= 30 ** 2
    occ = (xx - 66) ** 2 + (yy - 45) ** 2 <= 30 ** 2
    crescent = disk & ~occ
    other = occ & ~disk                                  # the occluding region's visible part
    contour, seam = label_boundary(_region(1, crescent), [_region(2, other)])
    assert contour.shape[1] == 2 and seam.dtype == bool and len(seam) == len(contour)
    seam_x = contour[seam][:, 0].mean()
    own_x = contour[~seam][:, 0].mean()
    assert seam_x > own_x                                # the bite is on the +x side
    assert seam.any() and (~seam).any()


def test_complete_primitive_recovers_full_disk_from_crescent():
    H = W = 120
    yy, xx = np.ogrid[:H, :W]
    cx0, cy0, r0 = 45.0, 60.0, 34.0
    disk = (xx - cx0) ** 2 + (yy - cy0) ** 2 <= r0 ** 2
    occ = (xx - 78) ** 2 + (yy - 60) ** 2 <= 34 ** 2
    crescent = disk & ~occ
    other = occ & ~disk
    contour, seam = label_boundary(_region(1, crescent), [_region(2, other)])
    prim = complete_primitive(contour, seam, max_residual=1.5, min_arc_deg=120.0)
    assert prim is not None and prim["kind"] == "circle"
    assert abs(prim["params"]["cx"] - cx0) < 2.0
    assert abs(prim["params"]["cy"] - cy0) < 2.0
    assert abs(prim["params"]["r"] - r0) < 2.0


def test_complete_primitive_rejects_when_own_arc_too_short():
    contour = np.array([[0, 0], [1, 0], [2, 1], [2, 2], [1, 3], [0, 3]], float)
    seam = np.ones(len(contour), bool); seam[0] = False
    assert complete_primitive(contour, seam, max_residual=1.5, min_arc_deg=120.0) is None


import re
from vectormark.occlusion import intersection_lens_d


def test_intersection_lens_path_two_arcs_between_crossing_points():
    a = {"cx": 0.0, "cy": 0.0, "r": 6.0}
    b = {"cx": 8.0, "cy": 0.0, "r": 6.0}
    d = intersection_lens_d(a, b)
    assert d is not None
    assert d.count("A") == 2 and d.startswith("M") and d.rstrip().endswith("Z")
    nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", d)]
    assert any(abs(n - 4.0) < 0.5 for n in nums)          # crossing points at x=4 (midpoint)


def test_intersection_lens_none_when_disjoint():
    a = {"cx": 0.0, "cy": 0.0, "r": 3.0}
    b = {"cx": 20.0, "cy": 0.0, "r": 3.0}
    assert intersection_lens_d(a, b) is None


from vectormark.occlusion import primitive_mask, stack_agreement


def test_primitive_mask_circle():
    m = primitive_mask({"kind": "circle", "params": {"cx": 10.0, "cy": 10.0, "r": 5.0}}, 20, 20)
    assert m[10, 10] and not m[10, 17] and m.dtype == bool


def test_stack_agreement_high_for_true_reconstruction():
    H = W = 100
    yy, xx = np.ogrid[:H, :W]
    da = (xx - 38) ** 2 + (yy - 50) ** 2 <= 30 ** 2
    db = (xx - 62) ** 2 + (yy - 50) ** 2 <= 30 ** 2
    red = da & ~db; yellow = db & ~da; lens = da & db
    regions = [_region(1, red, "#FF0000"), _region(2, yellow, "#FFFF00"), _region(3, lens, "#FFA500")]
    prims = [
        {"kind": "circle", "params": {"cx": 38.0, "cy": 50.0, "r": 30.0}, "color": "#FF0000", "z": 0},
        {"kind": "circle", "params": {"cx": 62.0, "cy": 50.0, "r": 30.0}, "color": "#FFFF00", "z": 1},
    ]
    lens_shape = {"mask_color": "#FFA500", "lens_of": (0, 1)}
    assert stack_agreement(prims, lens_shape, regions, H, W) > 0.97


def test_stack_agreement_low_for_wrong_reconstruction():
    H = W = 100
    yy, xx = np.ogrid[:H, :W]
    da = (xx - 38) ** 2 + (yy - 50) ** 2 <= 30 ** 2
    regions = [_region(1, da, "#FF0000")]
    prims = [{"kind": "circle", "params": {"cx": 38.0, "cy": 50.0, "r": 45.0}, "color": "#FF0000", "z": 0}]
    assert stack_agreement(prims, None, regions, H, W) < 0.9   # oversized disk disagrees


from vectormark.types import Axis
from vectormark.fit import Shape
from vectormark.occlusion import reconstruct_scene


def _two_disk_mark(H=140, W=200, gap=24, r=44):
    yy, xx = np.ogrid[:H, :W]
    cx = W // 2
    da = (xx - (cx - gap)) ** 2 + (yy - H // 2) ** 2 <= r ** 2
    db = (xx - (cx + gap)) ** 2 + (yy - H // 2) ** 2 <= r ** 2
    return da, db, cx


def test_reconstruct_scene_two_disks_with_lens():
    H, W = 140, 200
    da, db, cx = _two_disk_mark(H, W)
    regions = [
        _region(1, da & ~db, "#FF0000"),
        _region(2, db & ~da, "#FFFF00"),
        _region(3, da & db, "#FFA500"),
    ]
    reconstructed, remaining = reconstruct_scene(regions, Axis(x=float(cx)), (H, W))
    prims = [e for e in reconstructed if isinstance(e, ScenePrimitive)]
    lenses = [e for e in reconstructed if isinstance(e, Shape)]
    assert len(prims) == 2 and len(lenses) == 1
    assert abs(prims[0].params["r"] - prims[1].params["r"]) < 1.0
    assert abs((prims[0].params["cx"] + prims[1].params["cx"]) / 2 - cx) < 1.0
    assert remaining == []


def test_reconstruct_scene_passes_through_non_occluded():
    H = W = 80
    yy, xx = np.ogrid[:H, :W]
    band = np.zeros((H, W), bool); band[20:40, 10:70] = True     # convex, no bite
    regions = [_region(1, band, "#062336")]
    reconstructed, remaining = reconstruct_scene(regions, Axis(x=40.0), (H, W))
    assert reconstructed == [] and [r.label for r in remaining] == [1]


def test_reconstruct_scene_does_not_drop_members_of_larger_component():
    """A connected component larger than the canonical (2 crescents + 1 lens) must
    decline reconstruction WITHOUT dropping any region (regression: the old code
    consumed the whole transitive group, losing the extra member)."""
    H, W = 140, 220
    da, db, cx = _two_disk_mark(H, W)
    red = da & ~db
    extra = np.zeros((H, W), bool)
    extra[66:74, 30:42] = True          # small convex region abutting the red disk's left edge
    regions = [
        _region(1, red, "#FF0000"),
        _region(2, db & ~da, "#FFFF00"),
        _region(3, da & db, "#FFA500"),
        _region(4, extra & ~red, "#00FF00"),
    ]
    reconstructed, remaining = reconstruct_scene(regions, Axis(x=float(cx)), (H, W))
    assert reconstructed == []                              # non-canonical group declined
    assert {r.label for r in remaining} == {1, 2, 3, 4}     # nothing dropped


# --- Task 2: label_boundary per-contour ---
from vectormark.occlusion import label_boundary as _lb_contour_index  # noqa: E402


def _ring_region(label, cx, cy, r_out, r_in, h=120, w=120, color="#3366cc"):
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (d2 <= r_out ** 2) & (d2 >= r_in ** 2)
    return Region(label, mask, color)


def test_label_boundary_reads_inner_contour():
    ring = _ring_region(1, 60, 60, 40, 22)
    yy, xx = np.ogrid[:120, :120]
    occ = Region(2, ((xx - 85) ** 2 + (yy - 75) ** 2) <= 30 ** 2, "#cc3333")
    outer, outer_seam = _lb_contour_index(ring, [occ], contour_index=0)
    inner, inner_seam = _lb_contour_index(ring, [occ], contour_index=1)
    assert len(outer) > 0 and len(inner) > 0
    assert outer_seam.any()        # the occluder reaches the outer rim
    assert inner_seam.any()        # ...and the inner rim, only readable via contour_index=1


# --- Task 3: _fit_circle ---
from vectormark.occlusion import _fit_circle  # noqa: E402


def test_fit_circle_recovers_full_circle():
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    contour = np.column_stack([50 + 30 * np.cos(th), 60 + 30 * np.sin(th)])
    seam = np.zeros(len(contour), bool)
    fit = _fit_circle(contour, seam, max_residual=1.6, min_arc_deg=110.0)
    assert fit is not None
    assert abs(fit["cx"] - 50) < 1 and abs(fit["cy"] - 60) < 1 and abs(fit["r"] - 30) < 1


def test_fit_circle_rejects_short_arc():
    th = np.linspace(0, 0.3, 40)            # ~17 deg, far below min_arc_deg
    contour = np.column_stack([50 + 30 * np.cos(th), 60 + 30 * np.sin(th)])
    seam = np.zeros(len(contour), bool)
    assert _fit_circle(contour, seam, max_residual=1.6, min_arc_deg=110.0) is None


# --- Task 4: primitive_mask annulus ---
def test_primitive_mask_annulus():
    prim = {"kind": "annulus", "params": {"cx": 60, "cy": 60, "r_outer": 40, "r_inner": 22}}
    m = primitive_mask(prim, 120, 120)
    assert not m[60, 60]            # hole
    assert m[60, 60 - 31]          # on the band (31 px out, between 22 and 40)
    assert not m[60, 60 - 50]      # outside the outer radius


# --- Task 5: complete_annulus ---
from vectormark.occlusion import complete_annulus  # noqa: E402


def _occluded_ring(h=160, w=200):
    # a ring whose right OUTER rim is clipped by a big disk sitting mostly outside it:
    # the disk stops short of the inner radius (so the hole stays enclosed and the ring
    # keeps its outer+hole contours) yet borders enough background to complete as a
    # circle itself.
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - 70) ** 2 + (yy - 70) ** 2
    ring = (d2 <= 45 ** 2) & (d2 >= 25 ** 2)
    occ = ((xx - 135) ** 2 + (yy - 70) ** 2) <= 38 ** 2
    return Region(1, ring & ~occ, "#3366cc"), Region(2, occ, "#cc3333")


def test_complete_annulus_recovers_ring():
    ring, occ = _occluded_ring()
    prim = complete_annulus(ring, [occ], max_residual=1.6, min_arc_deg=110.0, concentric_tol=2.0)
    assert prim is not None and prim["kind"] == "annulus"
    p = prim["params"]
    assert abs(p["cx"] - 70) < 2 and abs(p["cy"] - 70) < 2
    assert abs(p["r_outer"] - 45) < 2 and abs(p["r_inner"] - 25) < 2


def test_complete_annulus_rejects_solid_disk():
    yy, xx = np.ogrid[:120, :120]
    disk = Region(1, ((xx - 60) ** 2 + (yy - 60) ** 2) <= 40 ** 2, "#3366cc")
    assert complete_annulus(disk, [], max_residual=1.6, min_arc_deg=110.0, concentric_tol=2.0) is None


def test_cavity_containment_is_geometric_and_keeps_crossing_occluder():
    from vectormark.occlusion import _cavity_contained_labels

    h = w = 140
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - 60) ** 2 + (yy - 70) ** 2
    ring = Region(1, (d2 <= 42 ** 2) & (d2 >= 24 ** 2), "#3366cc")
    colored_art = Region(2, (xx - 60) ** 2 + (yy - 70) ** 2 <= 12 ** 2, "#11aa55")
    crossing = Region(3, (xx - 93) ** 2 + (yy - 70) ** 2 <= 25 ** 2, "#cc3333")

    assert _cavity_contained_labels(ring, [colored_art, crossing]) == {colored_art.label}


def test_gate_rejects_same_color_candidate_overpainting_retained_artwork():
    h = w = 120
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - 60) ** 2 + (yy - 60) ** 2
    ring = Region(1, (d2 <= 40 ** 2) & (d2 >= 20 ** 2), "#3366cc")
    retained = Region(2, d2 < 20 ** 2, "#3366cc")
    annulus = {"kind": "annulus", "params": {"cx": 60, "cy": 60, "r_outer": 40,
                                                "r_inner": 20}, "color": "#3366cc", "z": 0}
    thin_intrusion = {"kind": "annulus", "params": {"cx": 60, "cy": 60,
                                                       "r_outer": 40, "r_inner": 19.5},
                      "color": "#3366cc", "z": 0}
    filled = {"kind": "circle", "params": {"cx": 60, "cy": 60, "r": 40},
              "color": "#3366cc", "z": 0}

    assert stack_agreement([annulus], None, [ring, retained], h, w,
                           retained_regions=[retained]) > 0.99
    assert stack_agreement([thin_intrusion], None, [ring, retained], h, w,
                           retained_regions=[retained]) == 0.0
    assert stack_agreement([filled], None, [ring, retained], h, w,
                           retained_regions=[retained]) < 0.96


def test_reconstruct_ring_carries_colored_cavity_artwork_by_source_identity():
    from vectormark.fit import Shape

    h, w = 160, 200
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - 70) ** 2 + (yy - 70) ** 2
    full_ring = (d2 <= 45 ** 2) & (d2 >= 25 ** 2)
    occluder_mask = (xx - 135) ** 2 + (yy - 70) ** 2 <= 38 ** 2
    ring = Region(1, full_ring & ~occluder_mask, "#3366cc")
    occluder = Region(2, occluder_mask, "#cc3333")
    artwork = Region(3, d2 < 25 ** 2, "#22aa66")

    reconstructed, remaining = reconstruct_scene([ring, occluder, artwork], None, (h, w))
    retained = [item for item in reconstructed
                if isinstance(item, Shape) and item.params.get("source_label") == artwork.label]

    assert len(retained) == 1
    assert retained[0].params["color_hex"] == artwork.color_hex
    assert sorted(item.kind for item in reconstructed if isinstance(item, ScenePrimitive)) == [
        "annulus", "circle"
    ]
    assert remaining == []


# --- Task 6: pairwise over/under + topological order ---
from vectormark.occlusion import _pair_constraint, _topo_order  # noqa: E402


def test_pair_constraint_from_overlap_ownership():
    h = w = 140
    yy, xx = np.ogrid[:h, :w]
    ring_full = ((xx - 70) ** 2 + (yy - 70) ** 2 <= 45 ** 2) & ((xx - 70) ** 2 + (yy - 70) ** 2 >= 25 ** 2)
    disk_full = (xx - 105) ** 2 + (yy - 70) ** 2 <= 28 ** 2
    ring_vis = Region(1, ring_full & ~disk_full, "#3366cc")   # ring loses the overlap
    disk_vis = Region(2, disk_full, "#cc3333")                # disk keeps it (on top)
    ring_prim = {"kind": "annulus", "params": {"cx": 70, "cy": 70, "r_outer": 45, "r_inner": 25}}
    disk_prim = {"kind": "circle", "params": {"cx": 105, "cy": 70, "r": 28}}
    assert _pair_constraint(ring_prim, ring_vis, disk_prim, disk_vis, h, w) == "j_over_i"


def test_topo_order_linear_and_cycle():
    assert _topo_order(3, [(0, 1), (1, 2)]) == [0, 1, 2]      # 0 under 1 under 2
    assert _topo_order(3, [(0, 1), (1, 2), (2, 0)]) is None   # cycle -> decline


# --- Task 7: N-shape reconstruct_scene ---
def test_reconstruct_ring_plus_disk():
    ring, occ = _occluded_ring()
    reconstructed, remaining = reconstruct_scene([ring, occ], None, (160, 200))
    kinds = sorted(p.kind for p in reconstructed if hasattr(p, "kind"))
    assert kinds == ["annulus", "circle"]
    assert remaining == []                       # both consumed


def test_reconstruct_declines_weave():
    # three mutually-interlocked rings whose overlap ownership is cyclic. Construct it
    # exactly: each ring keeps everything EXCEPT where the NEXT ring overlaps it
    # (A loses to C, B loses to A, C loses to B) -> A-over-B, B-over-C, C-over-A.
    h = w = 180
    yy, xx = np.ogrid[:h, :w]
    def ring(cx, cy):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return (d2 <= 40 ** 2) & (d2 >= 24 ** 2)
    a, b, c = ring(75, 75), ring(110, 75), ring(92, 110)
    regions = [Region(1, a & ~c, "#1111ee"),
               Region(2, b & ~a, "#eeee11"),
               Region(3, c & ~b, "#11aa11")]
    reconstructed, remaining = reconstruct_scene(regions, None, (h, w))
    assert reconstructed == []                    # cyclic -> declined
    assert len(remaining) == 3


# --- Task 1 (polygon completer): geometry helpers ---


def test_fit_line_recovers_horizontal_line():
    from vectormark.occlusion import _fit_line
    pts = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0], [6.0, 5.0]])
    a, b, c, resid = _fit_line(pts)
    # line is y == 5  ->  normal is (0, 1), offset c == 5
    assert resid < 1e-6
    assert abs(a * 3.0 + b * 5.0 - c) < 1e-6      # a point on the line satisfies ax+by=c
    assert abs(a * 3.0 + b * 9.0 - c) > 3.0       # a point 4px off has large signed distance


def test_fit_line_reports_residual_for_noisy_points():
    from vectormark.occlusion import _fit_line
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 3.0], [6.0, 0.0]])  # one 3px outlier
    _, _, _, resid = _fit_line(pts)
    assert resid > 1.0


def test_line_intersect_crossing_lines():
    from vectormark.occlusion import _fit_line, _line_intersect
    horiz = _fit_line(np.array([[0.0, 4.0], [8.0, 4.0]]))
    vert = _fit_line(np.array([[5.0, 0.0], [5.0, 9.0]]))
    p = _line_intersect(horiz, vert)
    assert p is not None
    assert abs(p[0] - 5.0) < 1e-6 and abs(p[1] - 4.0) < 1e-6


def test_line_intersect_parallel_returns_none():
    from vectormark.occlusion import _fit_line, _line_intersect
    l1 = _fit_line(np.array([[0.0, 0.0], [8.0, 0.0]]))
    l2 = _fit_line(np.array([[0.0, 3.0], [8.0, 3.0]]))
    assert _line_intersect(l1, l2) is None


def test_is_convex_accepts_square_rejects_arrow():
    from vectormark.occlusion import _is_convex
    square = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    assert _is_convex(square)
    concave = [(0.0, 0.0), (4.0, 0.0), (2.0, 2.0), (4.0, 4.0), (0.0, 4.0)]  # arrowhead notch
    assert not _is_convex(concave)


def test_own_runs_single_contiguous_arc():
    from vectormark.occlusion import _own_runs
    contour = np.array([[float(i), 0.0] for i in range(10)])
    seam = np.array([False, False, False, True, True, True, False, False, False, False])
    runs = _own_runs(contour, seam)
    assert len(runs) == 1                       # the seam splits the cyclic loop into one own arc
    assert len(runs[0]) == 7                     # indices 6,7,8,9,0,1,2 wrap across 0


def test_own_runs_no_seam_returns_whole_contour():
    from vectormark.occlusion import _own_runs
    contour = np.array([[float(i), 0.0] for i in range(5)])
    seam = np.zeros(5, bool)
    runs = _own_runs(contour, seam)
    assert len(runs) == 1 and len(runs[0]) == 5


def test_own_runs_all_seam_returns_empty():
    from vectormark.occlusion import _own_runs
    contour = np.array([[float(i), 0.0] for i in range(5)])
    runs = _own_runs(contour, np.ones(5, bool))
    assert runs == []


def test_own_runs_two_disjoint_arcs():
    from vectormark.occlusion import _own_runs
    contour = np.array([[float(i), 0.0] for i in range(10)])
    seam = np.array([True, False, False, True, True, False, False, False, True, True])
    runs = _own_runs(contour, seam)
    assert len(runs) == 2
    lengths = sorted(len(r) for r in runs)
    assert lengths == [2, 3]   # runs [1,2] and [5,6,7]


def _diamond_mask(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (np.abs(xx - cx) + np.abs(yy - cy)) <= r          # L1 ball == axis-aligned diamond


def _disk_mask(cx, cy, r, h, w):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def test_complete_polygon_recovers_occluded_diamond():
    from vectormark.occlusion import complete_polygon, _MAX_RESIDUAL, _MAX_VERTICES
    h, w = 160, 220
    cx, cy, r = 80, 80, 46
    diamond = _diamond_mask(cx, cy, r, h, w) & ~_disk_mask(126, 80, 30, h, w)  # right corner bitten
    occluder = _disk_mask(126, 80, 30, h, w)
    region = Region(label=1, mask=diamond, color_hex="#3366cc")
    other = Region(label=2, mask=occluder, color_hex="#cc3333")
    prim = complete_polygon(region, [other], max_residual=_MAX_RESIDUAL, max_vertices=_MAX_VERTICES)
    assert prim is not None and prim["kind"] == "polygon"
    pts = prim["params"]["points"]
    assert len(pts) == 4
    # the four recovered corners are each within tolerance of a true diamond corner
    truth = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    for tx, ty in truth:
        assert min(np.hypot(px - tx, py - ty) for px, py in pts) <= 2.5


def test_complete_polygon_rejects_curved_disk_fragment():
    from vectormark.occlusion import complete_polygon, _MAX_RESIDUAL, _MAX_VERTICES
    h, w = 160, 200
    disk = _disk_mask(90, 80, 50, h, w) & ~_disk_mask(150, 80, 28, h, w)   # big disk, small bite
    occluder = _disk_mask(150, 80, 28, h, w)
    region = Region(label=1, mask=disk, color_hex="#3366cc")
    other = Region(label=2, mask=occluder, color_hex="#cc3333")
    # a curved boundary RDP-splits into more than _MAX_VERTICES straight edges -> rejected
    assert complete_polygon(region, [other], max_residual=_MAX_RESIDUAL, max_vertices=_MAX_VERTICES) is None


def test_complete_polygon_recovers_diamond_with_two_corners_bitten():
    # A diamond clipped by TWO separate occluders (top and bottom corners), with all
    # four edges still partly visible -> the own boundary is TWO runs. complete_polygon
    # must collect edge lines across BOTH runs in cyclic order to recover all four
    # corners (including the two hidden ones). Regression for keeping only the longest
    # run, which would drop one arc's edges and decline.
    from vectormark.occlusion import complete_polygon, _MAX_RESIDUAL, _MAX_VERTICES
    h, w = 180, 180
    cx, cy, r = 90, 90, 50
    top = _disk_mask(90, 40, 16, h, w)       # hides top corner (90,40)
    bot = _disk_mask(90, 140, 16, h, w)      # hides bottom corner (90,140)
    diamond = _diamond_mask(cx, cy, r, h, w) & ~top & ~bot
    region = Region(label=1, mask=diamond, color_hex="#3366cc")
    others = [Region(label=2, mask=top, color_hex="#cc3333"),
              Region(label=3, mask=bot, color_hex="#cc3333")]
    prim = complete_polygon(region, others, max_residual=_MAX_RESIDUAL, max_vertices=_MAX_VERTICES)
    assert prim is not None and prim["kind"] == "polygon"
    pts = prim["params"]["points"]
    assert len(pts) == 4
    truth = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    for tx, ty in truth:
        assert min(np.hypot(px - tx, py - ty) for px, py in pts) <= 2.5


def test_complete_polygon_recovers_diamond_with_midedge_bite():
    # An occluder biting the MIDDLE of one edge (touching neither corner) splits that
    # edge into two collinear fitted segments (one each side of the bite). They land at
    # the start and end of the single own run, so the cyclic wrap pairs them — two
    # collinear lines whose intersection is None. complete_polygon must COALESCE the
    # collinear (incl. wraparound) pair before intersecting, else it declines a valid
    # "every edge partly visible" case. (Codex P2: split collinear edge runs.)
    from vectormark.occlusion import complete_polygon, _MAX_RESIDUAL, _MAX_VERTICES
    h, w = 180, 180
    cx, cy, r = 90, 90, 50
    bite = _disk_mask(115, 65, 14, h, w)     # midpoint of edge T(90,40)-R(140,90); reaches no corner
    diamond = _diamond_mask(cx, cy, r, h, w) & ~bite
    region = Region(label=1, mask=diamond, color_hex="#3366cc")
    other = Region(label=2, mask=bite, color_hex="#cc3333")
    prim = complete_polygon(region, [other], max_residual=_MAX_RESIDUAL, max_vertices=_MAX_VERTICES)
    assert prim is not None and prim["kind"] == "polygon"
    pts = prim["params"]["points"]
    assert len(pts) == 4
    truth = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    for tx, ty in truth:
        assert min(np.hypot(px - tx, py - ty) for px, py in pts) <= 2.5


# --- Task 4 (polygon branch): primitive_mask polygon case ---
def test_primitive_mask_polygon_membership():
    from vectormark.occlusion import primitive_mask
    # an asymmetric rectangle: x in [2,7] (width 5), y in [2,12] (height 10).
    # Asymmetric extents make the (x,y)->(row,col) swap load-bearing — a reversed
    # swap would fail these assertions.
    prim = {"kind": "polygon", "params": {"points": [(2.0, 2.0), (7.0, 2.0), (7.0, 12.0), (2.0, 12.0)]}}
    mask = primitive_mask(prim, 15, 10)
    assert mask[6, 4]        # row=6 (y=6 in [2,12]), col=4 (x=4 in [2,7]) -> inside
    assert not mask[1, 4]    # row=1 (y=1 < 2) -> outside
    assert not mask[0, 0]    # grid corner -> outside
    assert mask.dtype == bool


# --- Task 5: _complete_member dispatches polygonal fragments ---
def test_complete_member_dispatches_diamond_to_polygon():
    from vectormark.occlusion import _complete_member
    h, w = 160, 220
    diamond = _diamond_mask(80, 80, 46, h, w) & ~_disk_mask(126, 80, 30, h, w)
    occluder = _disk_mask(126, 80, 30, h, w)
    region = Region(label=1, mask=diamond, color_hex="#3366cc")
    other = Region(label=2, mask=occluder, color_hex="#cc3333")
    prim = _complete_member(region, [other])
    assert prim is not None and prim["kind"] == "polygon"


# --- Task 7: gate declines polygon with whole edge hidden ---
def test_reconstruct_scene_declines_polygon_with_whole_edge_hidden():
    """Safety-contract test: when a WHOLE edge of a polygon is hidden, complete_polygon
    recovers a wrong corner; the consistency gate (stack_agreement < _GATE_AGREEMENT)
    must detect the disagreement and decline the group, leaving the region in remaining.

    Fixture: a regular pentagon (5 vertices) whose top-right edge (corner[0]→corner[1])
    is completely covered by a disk occluder. The disk creates a concave bite, so
    has_bite is True and a group forms. The two visible edges flanking the seam are
    non-parallel, so complete_polygon does return *some* polygon — but it is wrong
    (the recovered corner replaces the two hidden corners with a single intersection
    point outside the true shape). The gate rejects this and leaves label 1 in remaining.
    """
    from skimage.draw import polygon2mask
    from vectormark.occlusion import _complete_member

    h, w = 200, 240
    cx, cy, r_pent = 100, 100, 65
    # Five vertices of a regular pentagon: top first, then clockwise in image coords (y-down).
    angles = [np.pi / 2 - 2 * np.pi * k / 5 for k in range(5)]
    pts_xy = [(cx + r_pent * np.cos(a), cy - r_pent * np.sin(a)) for a in angles]
    # polygon2mask wants (row, col) == (y, x) for each vertex.
    pentagon = polygon2mask((h, w), np.array([(y, x) for x, y in pts_xy])).astype(bool)

    # Disk occluder centred on the midpoint of the top-right edge (corner 0 → corner 1),
    # with radius large enough to fully cover both endpoints of that edge.
    c0, c1 = np.array(pts_xy[0]), np.array(pts_xy[1])
    mid_xy = (c0 + c1) / 2
    occ_r = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])) / 2 + 10
    occ_mask = _disk_mask(mid_xy[0], mid_xy[1], occ_r, h, w)

    region = Region(label=1, mask=pentagon & ~occ_mask, color_hex="#3366cc")
    occ = Region(label=2, mask=occ_mask, color_hex="#cc3333")

    # Non-vacuous guard: completion DOES produce a (wrong) convex-polygon candidate
    # (the two seam-flanking pentagon edges are non-parallel, unlike a diamond's),
    # so the decline below is the consistency gate rejecting it
    # (stack_agreement ~= 0.9455 < _GATE_AGREEMENT = 0.96), NOT a silent skip from
    # has_bite=False or _complete_member returning None.
    assert _complete_member(region, [occ]) is not None
    reconstructed, remaining = reconstruct_scene([region, occ], None, (h, w))
    assert 1 in {r.label for r in remaining}                       # declined -> fallback
    assert not any(getattr(e, "kind", None) == "polygon" for e in reconstructed)  # nothing emitted

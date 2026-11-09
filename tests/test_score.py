from vectoreyes.candidate import (
    Candidate, FlatFill, LinearGradientFill, RadialGradientFill,
)
from vectoreyes.fit import Shape
from vectoreyes.score import parsimony_cost


def _flat(shape):
    return Candidate(shape, FlatFill("#123456"), "region")


def test_parsimony_primitive_cheaper_than_path():
    circle = _flat(Shape("circle", {"cx": 5, "cy": 5, "r": 4}))
    path = _flat(Shape("path", {"d": "M0 0 C1 1 2 2 3 3 C4 4 5 5 6 6 C7 7 8 8 9 9 Z"}))
    assert parsimony_cost(circle) < parsimony_cost(path)


def test_parsimony_flat_cheaper_than_gradient_same_geometry():
    geom = Shape("rect", {"x": 0, "y": 0, "w": 10, "h": 10})
    flat = Candidate(geom, FlatFill("#000000"), "region")
    grad = Candidate(geom, LinearGradientFill({"x1": 0, "y1": 0, "x2": 10, "y2": 0},
                                              [(0.0, "#000"), (1.0, "#fff")]), "gradient")
    assert parsimony_cost(flat) < parsimony_cost(grad)


def test_parsimony_polygon_scales_with_vertices():
    tri = _flat(Shape("polygon", {"points": [(0, 0), (1, 0), (0, 1)]}))
    hexa = _flat(Shape("polygon", {"points": [(0, 0), (1, 0), (2, 1), (1, 2), (0, 2), (-1, 1)]}))
    assert parsimony_cost(hexa) > parsimony_cost(tri)


import numpy as np
import pytest
from vectoreyes.types import Region
from vectoreyes.score import render_delta_e


def _disk_region(cx, cy, r, h, w, color_hex):
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    return Region(1, mask, color_hex)


def test_render_delta_e_near_zero_on_matching_flat():
    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(30, 30, 18, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    cand = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}),
                     FlatFill("#1e64eb"), "region")
    assert render_delta_e(cand, src, region) < 0.03


def test_render_delta_e_large_on_wrong_color():
    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(30, 30, 18, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    wrong = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}),
                      FlatFill("#ff0000"), "region")
    assert render_delta_e(wrong, src, region) > 0.2


from vectoreyes.score import structural_priors


def _square_region(h, w, color_hex):
    mask = np.zeros((h, w), bool)
    mask[10:50, 10:50] = True
    return Region(1, mask, color_hex)


def test_priors_reject_degenerate_gradient_stop_span():
    region = _square_region(60, 60, "#202020")
    grad = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     LinearGradientFill({"x1": 10, "y1": 10, "x2": 50, "y2": 10},
                                        [(0.0, "#202020"), (1.0, "#212121")]), "gradient")
    ok, reason = structural_priors(grad, region)
    assert ok is False and "stop-span" in reason


def test_priors_pass_flat_and_primitive():
    region = _square_region(60, 60, "#202020")
    flat = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     FlatFill("#202020"), "region")
    ok, reason = structural_priors(flat, region)
    assert ok is True and reason is None


from vectoreyes.score import ScoreBreakdown, SvgRendererUnavailable, rank_candidates


def test_rank_prefers_parsimony_among_fidelity_qualifiers():
    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(30, 30, 18, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    circle = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}),
                       FlatFill("#1e64eb"), "region")
    path_d = "M48 30 C48 40 40 48 30 48 C20 48 12 40 12 30 C12 20 20 12 30 12 C40 12 48 20 48 30 Z"
    path = Candidate(Shape("path", {"d": path_d}), FlatFill("#1e64eb"), "region")
    ranked = rank_candidates([path, circle], src, region, fidelity_tol=0.06)
    assert ranked[0][0] is circle
    assert all(isinstance(b, ScoreBreakdown) for _, b in ranked)
    assert ranked[0][1].qualified is True


def test_rank_preserves_candidate_order_without_svg_renderer(monkeypatch):
    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    first = Candidate(Shape("path", {"d": "M12 30 L30 12 L48 30 L30 48 Z"}),
                      FlatFill("#1e64eb"), "region")
    second = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}),
                       FlatFill("#1e64eb"), "region")

    def unavailable(*args, **kwargs):
        raise SvgRendererUnavailable("missing renderer")

    monkeypatch.setattr("vectoreyes.score.render_delta_e", unavailable)

    ranked = rank_candidates([first, second], src, region)

    assert [candidate for candidate, _ in ranked] == [first, second]
    assert all(breakdown.qualified is False for _, breakdown in ranked)


def test_rank_warns_when_svg_renderer_unavailable(monkeypatch):
    import vectoreyes.score as score
    monkeypatch.setattr(score, "_renderer_warned", False)   # reset the once-per-process latch

    h = w = 60
    src = np.full((h, w, 3), 255, np.uint8)
    region = _disk_region(30, 30, 18, h, w, "#1e64eb")
    cand = Candidate(Shape("circle", {"cx": 30, "cy": 30, "r": 18}), FlatFill("#1e64eb"), "region")
    monkeypatch.setattr("vectoreyes.score.render_delta_e",
                        lambda *a, **k: (_ for _ in ()).throw(SvgRendererUnavailable("x")))

    with pytest.warns(RuntimeWarning, match="resvg-py"):    # silent downgrade is surfaced
        rank_candidates([cand], src, region)


def test_rank_disqualifies_degenerate_gradient_keeps_flat():
    region = _square_region(60, 60, "#202020")
    src = np.full((60, 60, 3), 255, np.uint8)
    src[region.mask] = (32, 32, 32)
    flat = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     FlatFill("#202020"), "region")
    grad = Candidate(Shape("rect", {"x": 10, "y": 10, "w": 40, "h": 40}),
                     LinearGradientFill({"x1": 10, "y1": 10, "x2": 50, "y2": 10},
                                        [(0.0, "#202020"), (1.0, "#212121")]), "gradient")
    ranked = rank_candidates([grad, flat], src, region)
    assert ranked[0][0] is flat
    grad_bd = next(b for c, b in ranked if c is grad)
    assert grad_bd.priors_ok is False and grad_bd.qualified is False


def test_render_delta_e_bbox_matches_full_canvas():
    h = w = 120
    src = np.full((h, w, 3), 255, np.uint8)
    src[_disk_region(60, 60, 24, h, w, "#1e64eb").mask] = (30, 100, 235)
    region = _disk_region(60, 60, 24, h, w, "#1e64eb")
    cand = Candidate(Shape("circle", {"cx": 60, "cy": 60, "r": 24}),
                     FlatFill("#1e64eb"), "region")
    full = render_delta_e(cand, src, region)
    ys, xs = np.where(region.mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    cropped = render_delta_e(cand, src, region, bbox=bbox)
    assert abs(full - cropped) < 1e-6

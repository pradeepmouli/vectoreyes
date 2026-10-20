from dataclasses import replace

from vectormark.path_geometry import CornerAnchor, parse_svg_paths


D = ("M247 417.5 Q241.23 417.72 237.5 413 L237.5 405 L242 400.5 "
     "Q256.31 399.74 253.5 413 L247 417.5 Z")


def test_simplification_accepts_canonical_paths_without_working_shapes():
    from vectormark.drawing_simplification import simplify_paths

    paths = parse_svg_paths(D, region_id="r1")
    changed = simplify_paths(paths, epsilon=1, max_error=1, samples=16,
                             linelet_only=True, max_boundary_error=1)

    assert changed is not None
    assert changed[0].path.id == "r1.p0"
    assert len(changed[0].path.commands) == 5
    assert changed[0].path.start == (247, 417.5)
    assert paths[0].to_svg_d() == D


def test_simplification_cannot_erase_an_explicit_point_corner():
    from vectormark.drawing_simplification import simplify_paths

    (path,) = parse_svg_paths(D, region_id="r1")
    corner = CornerAnchor("point", "r1.p0.c1", "r1.p0.c1", radius=0)
    path = replace(path, anchors=(corner,))
    changed = simplify_paths((path,), epsilon=1, max_error=1, samples=16,
                             linelet_only=True, max_boundary_error=1)

    # The ordinary candidate removes this endpoint; an optimizer must not
    # silently remove the declaration to gain a lower command count.
    assert changed is None
    assert path.anchors == (corner,)


def test_simplification_reuses_retained_endpoint_ids_for_surviving_corner():
    from vectormark.drawing_simplification import simplify_paths

    (path,) = parse_svg_paths(D, region_id="r1")
    corner = CornerAnchor("point", "r1.p0.c3", "r1.p0.c3", radius=0)
    path = replace(path, anchors=(corner,))
    changed = simplify_paths((path,), epsilon=1, max_error=1, samples=16,
                             linelet_only=True, max_boundary_error=1)

    assert changed is not None
    assert changed[0].anchors == (corner,)
    assert changed[0].endpoint("r1.p0.c3") == (242, 400.5)


def test_optimizer_publication_keeps_selected_ids_without_an_input_cache():
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    from vectormark.optimizer.passes.simplify import simplify_pass
    from vectormark.optimizer.vector_region import VectorRegion

    target = VectorRegion(1, Shape('path', {'d': D}), FlatFill('#123456'))
    (proposal,) = simplify_pass([target], {}, epsilon=1, max_error=1, linelet_only=True,
                               max_boundary_error=1)
    geometry = proposal.new_objects[0].current.params['canonical_path_geometry']

    assert 'r1.p0.c1' not in {command.id for command in geometry.path.commands}
    assert geometry.endpoint('r1.p0.c3') == (242, 400.5)
    assert proposal.new_objects[0].fill is target.fill

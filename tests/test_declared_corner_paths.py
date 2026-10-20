from dataclasses import replace

from vectormark.path_geometry import CornerAnchor, InflectionAnchor, PathGeometryWithCutouts, parse_svg_paths


D = "M0 0 L10 0 Q12 0 13 1 Q14 2 14 4 L0 12 Z"


def test_declared_corner_normalization_retains_actual_command_identity():
    from vectormark.corner_diagnostics import path_corner_diagnostics
    from vectormark.corner_normalization import normalize_declared_paths

    (path,) = parse_svg_paths(D, region_id="r1")
    commands = tuple(replace(command, id=f"r1.p0.c{index * 3}")
                     for index, command in enumerate(path.path.commands))
    path = replace(path, path=replace(path.path, commands=commands))
    declaration = path_corner_diagnostics(D, declared_corner_spans=(((10, 0), (14, 4)),))

    (changed,), report = normalize_declared_paths((path,), declaration=declaration, max_error=1,
                                                source_text=D)

    assert len(changed.path.commands) == len(path.path.commands) - 1
    assert changed.path.commands[1] is commands[1]
    assert changed.path.commands[2].id == "r1.p0.c9"
    assert changed.path.commands[2].end == (14, 4)
    assert changed.anchors[0].owned_command_ids == ("r1.p0.c9",)
    assert report["normalized"] == ["p0.rounded_corner0"]


def test_authoritative_empty_corner_declaration_keeps_other_features():
    from vectormark.corner_diagnostics import path_corner_diagnostics
    from vectormark.corner_normalization import normalize_declared_paths

    (path,) = parse_svg_paths(D, region_id="r1")
    other = InflectionAnchor("other", ("r1.p0.c4",))
    path = replace(path, anchors=(CornerAnchor("old", "r1.p0.c1", "r1.p0.c3",
        owned_command_ids=("r1.p0.c2", "r1.p0.c3")), other))
    declaration = path_corner_diagnostics(D)

    (changed,), report = normalize_declared_paths((path,), declaration=declaration, max_error=1,
                                                source_text=D)

    assert changed.path is path.path
    assert changed.anchors == (other,)
    assert report["normalized"] == []


def test_corner_publication_keeps_hole_first_source_order_distinct_from_filled_roles():
    from vectormark.candidate import FlatFill
    from vectormark.corner_diagnostics import path_corner_diagnostics
    from vectormark.drawing_adapter import bind_path_input
    from vectormark.fit import Shape
    from vectormark.refinement.automatic_replay import (
        NormalizeCornersOperation, PassInputRegionRef, apply_corner_normalization_choice,
    )
    from vectormark.refinement.region_sample import RegionSample

    d = 'M3 3 L4 3 L4 4 L3 4 Z ' + D
    hole, outer = parse_svg_paths(d, region_id='r1')
    geometry = PathGeometryWithCutouts(outer, (hole,))
    shape = Shape('path', {'d': d, 'fill_rule': 'evenodd', 'path_ids': ('r1.p0', 'r1.p1'),
                           'canonical_path_geometry': geometry})
    target = RegionSample(1, shape, FlatFill('#123456'))
    declaration = path_corner_diagnostics(d, declared_corner_spans=(((10, 0), (14, 4)),))
    action = NormalizeCornersOperation(PassInputRegionRef('', '', (1,)), 1, declaration)

    (changed,) = apply_corner_normalization_choice(action, bind_path_input(target))
    selected = changed.current.params['canonical_path_geometry']

    assert selected.outer.path.id == 'r1.p1'
    assert len(selected.outer.path.commands) == len(outer.path.commands) - 1
    assert selected.cutouts[0].path == hole.path
    assert changed.diagnostics['corners']['normalized'] == ['p1.rounded_corner0']

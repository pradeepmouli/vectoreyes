from dataclasses import replace

from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
from vectoreyes.drawing_actions import ActionSettings, UpdateGeometryAction
from vectoreyes.path_geometry import parse_svg_paths
from vectoreyes.path_operations import apply_region_path_operations


def test_sparse_command_ids_cannot_be_shadowed_by_synthetic_closure():
    (path,) = parse_svg_paths("M0 0 L10 0 L0 10 Z", region_id="r1")
    commands = tuple(replace(command, id=f"r1.p0.c{index}") for command, index in
                     zip(path.path.commands, (0, 3, 5), strict=True))
    path = replace(path, path=replace(path.path, commands=commands))
    drawing = Drawing("drawing", Canvas(20, 20), (DrawingRegion("r1", path),))
    action = UpdateGeometryAction.model_validate({"op": "update_geometry", "target": "r1",
        "ops": [{"op": "fit", "target": "r1.p0.c3", "type": "keep"}]})

    result, _ = apply_region_path_operations(drawing, action,
        settings=ActionSettings(epsilon=1, max_error=1, fit_strategy='quadratic'))

    assert result.geometry.to_svg_d() == "M0 0 L10 0 L0 10 Z"
    assert path.endpoint("r1.p0.c3") == (10, 0)

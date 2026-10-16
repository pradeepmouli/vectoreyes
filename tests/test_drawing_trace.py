from dataclasses import replace
import json

import numpy as np
import pytest

from vectormark.drawing_trace import PythonTraceEngine, TraceOptions, svg_path_commands


def _annulus_image() -> np.ndarray:
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:80, :80]
    ring = (12**2 <= (xx - 40) ** 2 + (yy - 40) ** 2) & ((xx - 40) ** 2 + (yy - 40) ** 2 <= 28**2)
    image[ring] = (0, 110, 240)
    return image


def _two_region_image() -> np.ndarray:
    image = np.full((80, 80, 3), 255, dtype=np.uint8)
    image[12:38, 10:36] = (0, 110, 240)
    image[42:70, 44:72] = (255, 100, 0)
    return image


def test_trace_keeps_small_region_using_absolute_size_only():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[10:90, 10:90] = (0, 110, 240)
    image[4:8, 4:8] = (255, 100, 0)

    result = PythonTraceEngine().trace(image, TraceOptions(min_region_size=16))

    assert [region.id for region in result.regions] == ["r1", "r2"]


def test_subpixel_trace_keeps_absolute_floor_region_below_palette_fraction():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[4:8, 4:8] = (255, 100, 0)  # 16 px / 10,000 = 0.16% < 0.2% palette floor

    result = PythonTraceEngine().trace(
        image,
        TraceOptions(min_region_size=16, trace_level="subpixel"),
    )

    small_region = next(region for region in result.regions if region.color == "#FF6400")
    assert small_region.effective_trace_level == "subpixel"


def test_trace_commands_are_scoped_by_region_and_subpath():
    region = PythonTraceEngine().trace(_annulus_image(), TraceOptions()).regions[0]

    assert region.trace_path.commands[0].id == "r1.p0.c0"
    assert any(command.id.startswith("r1.p1.") for command in region.trace_path.commands)


def test_svg_path_commands_preserves_arc_commands():
    commands = svg_path_commands("M10 20 A5 6 0 0 1 20 30 Z", "r1")

    assert [(command.command, command.values) for command in commands] == [
        ("M", (10.0, 20.0)),
        ("A", (5.0, 6.0, 0.0, 0.0, 1.0, 20.0, 30.0)),
        ("Z", ()),
    ]


def test_region_map_labels_every_region():
    result = PythonTraceEngine().trace(_two_region_image(), TraceOptions())

    assert all(f">{region.id}<" in result.region_map_svg for region in result.regions)


def test_trace_result_public_dict_excludes_pixel_arrays():
    result = PythonTraceEngine().trace(_two_region_image(), TraceOptions())

    public = result.to_public_dict()

    assert public["width"] == 80
    assert public["regions"][0]["trace_path"]["commands"][0]["id"] == "r1.p0.c0"
    assert "mask" not in public["regions"][0]
    assert "region_map_svg" not in public
    assert result.to_public_dict(include_region_map=True)["region_map_svg"] == result.region_map_svg


def test_trace_result_public_dict_detaches_nested_frozen_background_as_json_native_data():
    result = PythonTraceEngine().trace(_two_region_image(), TraceOptions())
    nested = {"samples": [{"rgb": [1, 2, 3]}]}
    result = replace(result, background={"mode": "test", "metadata": nested})

    public = result.to_public_dict()
    json.dumps(public)
    public["geometry"]["background"]["metadata"]["samples"][0]["rgb"][0] = 99

    assert nested["samples"][0]["rgb"] == [1, 2, 3]
    assert result.background["metadata"]["samples"][0]["rgb"] == [1, 2, 3]


def test_trace_result_region_arrays_are_read_only():
    region = PythonTraceEngine().trace(_two_region_image(), TraceOptions()).regions[0]

    with pytest.raises(ValueError):
        region.mask[0, 0] = not region.mask[0, 0]
    with pytest.raises(ValueError):
        region.contours[0][0, 0] = 0


def test_opaque_source_can_emit_an_authoritative_background_removed_geometry_trace():
    image = np.full((48, 48, 3), (254, 254, 254), dtype=np.uint8)
    image[12:36, 12:36] = (10, 120, 240)

    result = PythonTraceEngine().trace(
        image,
        TraceOptions(source_has_alpha=False, remove_background="auto"),
    )

    assert result.background["mode"] == "inferred_alpha"
    assert result.background["applied"] is True
    assert len(result.geometry_regions) == 1
    assert result.geometry_regions[0].trace_path.d.startswith("M")
    assert result.to_public_dict()["geometry"]["regions"][0]["id"] == "g1"


def test_inferred_geometry_trace_discards_tiny_background_pinholes_before_path_fit():
    image = np.full((64, 64, 3), (254, 254, 254), dtype=np.uint8)
    image[12:52, 12:52] = (10, 120, 240)
    image[28, 28] = (254, 254, 254)

    result = PythonTraceEngine().trace(
        image,
        TraceOptions(source_has_alpha=False, remove_background="auto", max_hole_area=4),
    )

    region = result.geometry_regions[0]
    assert region.mask[28, 28]
    assert region.trace_path.d.count("M") == 1


def test_inferred_geometry_trace_drops_diagonally_attached_noise_island():
    image = np.full((64, 64, 3), (254, 254, 254), dtype=np.uint8)
    image[12:36, 12:36] = (10, 120, 240)
    image[36, 36] = (10, 120, 240)  # diagonal-only contact with the square

    result = PythonTraceEngine().trace(
        image,
        TraceOptions(source_has_alpha=False, remove_background="auto"),
    )

    assert len(result.geometry_regions) == 1
    assert result.geometry_regions[0].trace_path.d.count("M") == 1


def test_native_alpha_can_emit_an_authoritative_geometry_trace():
    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    alpha = np.zeros((48, 48), dtype=np.uint8)
    alpha[12:36, 12:36] = 255

    result = PythonTraceEngine().trace(
        image,
        TraceOptions(source_has_alpha=True, remove_background="auto"),
        alpha=alpha,
    )

    assert result.background == {"mode": "native_alpha", "applied": True, "threshold": 8}
    assert len(result.geometry_regions) == 1
    assert result.geometry_regions[0].mask.sum() == 24 * 24


def test_native_alpha_splits_substantial_surfaces_joined_only_by_a_one_pixel_pinch():
    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    alpha = np.zeros((48, 48), dtype=np.uint8)
    alpha[8:24, 8:24] = 255
    alpha[24:40, 24:40] = 255
    alpha[24, 23] = 255  # one 4-connected bridge between the two squares

    result = PythonTraceEngine().trace(
        image,
        TraceOptions(source_has_alpha=True, remove_background="auto"),
        alpha=alpha,
    )

    assert len(result.geometry_regions) == 2
    assert sum(int(region.mask.sum()) for region in result.geometry_regions) == int((alpha > 8).sum())

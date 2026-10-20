from __future__ import annotations

import asyncio
from typing import get_args, get_type_hints
import base64
import io
import os
import socket
import subprocess
import sys
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest
import vectormark.mcp_server as mcp_server_module
from mcp import Client, ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from fastmcp.exceptions import ToolError
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

from vectormark.mcp_server import (
    WIDGET_URI,
    ImageRef,
    GeometryGuideOptions,
    TraceDrawingOptions,
    get_drawing_artifact,
    mcp,
    refine_drawing,
    retrace_drawing,
    render_drawing,
    trace_drawing,
)


def _png_data_uri() -> str:
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((16, 16, 48, 48), fill=(20, 100, 240))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def _cutout_png_data_uri() -> str:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 55, 55), fill=(20, 100, 240))
    draw.rectangle((20, 20, 43, 43), fill="white")
    data = io.BytesIO()
    image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def _transparent_cutout_png_data_uri() -> str:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 55, 55), fill=(20, 100, 240, 255))
    draw.rectangle((20, 20, 43, 43), fill=(0, 0, 0, 0))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def _transparent_png_data_uri() -> str:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(image).rounded_rectangle((12, 12, 52, 52), radius=10, fill=(200, 55, 171, 255))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def _geometry_guide_data_uri() -> str:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(image).rounded_rectangle((16, 16, 48, 48), radius=6, fill=(255, 0, 255, 255))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def _small_geometry_guide_data_uri() -> str:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ImageDraw.Draw(image).rectangle((8, 8, 24, 24), fill=(255, 0, 255, 255))
    data = io.BytesIO()
    image.save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def test_trace_drawing_uses_alpha_for_initial_geometry_and_composited_rgb_for_fill():
    trace, rgb = mcp_server_module._trace_result(
        ImageRef(data_uri=_transparent_png_data_uri()), TraceDrawingOptions()
    )

    assert trace.background["mode"] == "native_alpha"
    assert trace.background["applied"] is True
    assert trace.geometry_regions
    assert tuple(rgb[0, 0]) == (255, 255, 255)


def test_unrefined_trace_refine_and_artifact_share_one_live_region_forest():
    """The public MCP helpers preserve v0 roots and refine child versions."""
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(), ctx)
    trace = traced.structured_content
    assert trace is not None
    drawing_id = trace["drawing_id"]
    assert trace["version"] == "v0"
    assert trace["artifacts"]["review_panel"].endswith(".review.png")
    panel_content = next(item for item in traced.content if item.type == "image")
    assert Image.open(io.BytesIO(base64.b64decode(panel_content.data))).size == (1140, 960)
    panel_artifact = get_drawing_artifact(drawing_id, "v0", "review_panel", ctx)
    panel = panel_artifact.content[0]
    assert panel.type == "image"
    assert Image.open(io.BytesIO(base64.b64decode(panel.data))).size == (1140, 960)
    assert trace["report"]["targets"]
    assert "region_map_svg" not in trace["trace"]
    assert "geometry" in trace["trace"]["regions"][0]
    raw_trace = get_drawing_artifact(drawing_id, "v0", "raw_trace", ctx).structured_content
    assert raw_trace is not None
    assert "trace_path" in raw_trace["trace"]["regions"][0]

    refined = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": drawing_id,
            "base_version": "v0",
            "ops": [{"op": "fit_primitive", "target": "r1", "type": "circle"}],
        },
        ctx,
    )
    result = refined.structured_content
    assert result is not None and result["version"] == "v0.0"
    assert result["report"]["targets"][0]["geometry"]["type"] == "circle"

    artifact = get_drawing_artifact(drawing_id, "v0.0", "svg", ctx).structured_content
    assert artifact is not None and '<circle id="r1"' in artifact["svg"]


def test_mcp_corner_actions_follow_reported_path_addresses_and_replay_without_detection(
    monkeypatch,
):
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(
        ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(refine="none"), ctx
    ).structured_content
    assert traced is not None
    target = traced["report"]["targets"][0]
    path = target["geometry"]
    assert path["corner_edit"] == {"supported": True, "reason": None}
    boundary = next(
        command["id"]
        for command in path["commands"]
        if command["corner_boundary"]["editable"]
    )

    monkeypatch.setattr(
        "vectormark.corner_diagnostics.path_corner_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("corner detector ran after retained evidence capture")
        ),
    )
    added = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": "v0",
            "ops": [{
                "op": "add_corner",
                "target": target["id"],
                "path_id": path["id"],
                "feature_id": "agent.corner",
                "entry_command_id": boundary,
                "exit_command_id": boundary,
            }],
        },
        ctx,
    ).structured_content
    assert added is not None and added["version"] == "v0.0"
    added_target = added["report"]["targets"][0]
    added_path = added_target["geometry"]
    assert "agent.corner" in {anchor["id"] for anchor in added_path["anchors"]}
    assert added_target["diagnostics"]["corners"]["path"] == added_path["d"]
    assert "agent.corner" in {
        span["corner_id"]
        for span in added_target["diagnostics"]["corners"]["spans"]
    }

    replacement_boundary = next(
        command["id"]
        for command in added_path["commands"]
        if command["corner_boundary"]["editable"] and command["id"] != boundary
    )
    replaced = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": added["version"],
            "ops": [{
                "op": "replace_corner",
                "target": target["id"],
                "path_id": added_path["id"],
                "feature_id": "agent.corner",
                "entry_command_id": replacement_boundary,
                "exit_command_id": replacement_boundary,
            }],
        },
        ctx,
    ).structured_content
    assert replaced is not None and replaced["version"] == "v0.0.0"
    replaced_target = replaced["report"]["targets"][0]
    replaced_path = replaced_target["geometry"]
    replaced_anchor = next(
        anchor for anchor in replaced_path["anchors"] if anchor["id"] == "agent.corner"
    )
    assert replaced_anchor["entry_command_id"] != boundary
    assert replaced_target["diagnostics"]["corners"]["path"] == replaced_path["d"]

    removed = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": replaced["version"],
            "ops": [{
                "op": "remove_corner",
                "target": target["id"],
                "path_id": replaced_path["id"],
                "feature_id": "agent.corner",
            }],
        },
        ctx,
    ).structured_content
    assert removed is not None and removed["version"] == "v0.0.0.0"
    removed_target = removed["report"]["targets"][0]
    removed_path = removed_target["geometry"]
    assert "agent.corner" not in {anchor["id"] for anchor in removed_path["anchors"]}
    assert removed_target["diagnostics"]["corners"]["path"] == removed_path["d"]

    retained = path["corner_features"][0]
    never_added = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": "v0",
            "ops": [{
                "op": "replace_corner",
                "target": target["id"],
                "path_id": path["id"],
                "feature_id": retained["feature_id"],
                "entry_command_id": retained["command_id"],
                "exit_command_id": retained["command_id"],
            }],
        },
        ctx,
    ).structured_content
    assert never_added is not None and never_added["version"] == "v0.1"
    removed_svg = get_drawing_artifact(
        traced["drawing_id"], removed["version"], "svg", ctx
    ).structured_content["svg"]
    never_added_svg = get_drawing_artifact(
        traced["drawing_id"], never_added["version"], "svg", ctx
    ).structured_content["svg"]
    assert removed_svg == never_added_svg
    never_added_version = mcp_server_module._DRAWINGS.get_version(
        traced["drawing_id"], never_added["version"]
    )
    assert "agent.corner" not in str(never_added_version.effective_recipe)

    normalized = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": removed["version"],
            "ops": [{
                "op": "update_geometry",
                "target": target["id"],
                "ops": [{"op": "normalize_corners"}],
            }],
        },
        ctx,
    ).structured_content
    assert normalized is not None
    assert normalized["report"]["targets"][0]["diagnostics"]["corners"]


def test_mcp_corner_actions_address_one_cutout_subpath_without_rewriting_outer(
    monkeypatch,
):
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(
        ImageRef(data_uri=_cutout_png_data_uri()),
        TraceDrawingOptions(refine="none"),
        ctx,
    ).structured_content
    assert traced is not None
    target = next(
        item
        for item in traced["report"]["targets"]
        if "paths" in item["geometry"]
    )
    outer, cutout = target["geometry"]["paths"]
    boundary = next(
        command["id"]
        for command in cutout["commands"]
        if command["corner_boundary"]["editable"]
    )
    monkeypatch.setattr(
        "vectormark.corner_diagnostics.path_corner_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("corner detector ran after retained evidence capture")
        ),
    )

    added = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": "v0",
            "ops": [{
                "op": "add_corner",
                "target": target["id"],
                "path_id": cutout["id"],
                "feature_id": "cutout.agent",
                "entry_command_id": boundary,
                "exit_command_id": boundary,
            }],
        },
        ctx,
    ).structured_content
    assert added is not None
    added_paths = next(
        item for item in added["report"]["targets"] if item["id"] == target["id"]
    )["geometry"]["paths"]
    assert added_paths[0]["d"] == outer["d"]
    assert "cutout.agent" in {anchor["id"] for anchor in added_paths[1]["anchors"]}

    removed = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": added["version"],
            "ops": [{
                "op": "remove_corner",
                "target": target["id"],
                "path_id": added_paths[1]["id"],
                "feature_id": "cutout.agent",
            }],
        },
        ctx,
    ).structured_content
    assert removed is not None
    removed_paths = next(
        item for item in removed["report"]["targets"] if item["id"] == target["id"]
    )["geometry"]["paths"]
    assert removed_paths[0]["d"] == outer["d"]
    assert "cutout.agent" not in {
        anchor["id"] for anchor in removed_paths[1]["anchors"]
    }


def test_mcp_corner_actions_refit_one_cutout_span_without_detection(monkeypatch):
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(
        ImageRef(data_uri=_transparent_cutout_png_data_uri()),
        TraceDrawingOptions(refine="none"),
        ctx,
    ).structured_content
    assert traced is not None
    target = next(
        item
        for item in traced["report"]["targets"]
        if "paths" in item["geometry"]
    )
    outer, cutout = target["geometry"]["paths"]
    features = {
        item["command_id"]: item["feature_id"]
        for item in cutout["corner_features"]
    }
    monkeypatch.setattr(
        "vectormark.corner_diagnostics.path_corner_diagnostics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("corner detector ran after retained evidence capture")
        ),
    )

    rounded = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": "v0",
            "ops": [
                {
                    "op": "remove_corner",
                    "target": target["id"],
                    "path_id": cutout["id"],
                    "feature_id": features[f"{cutout['id']}.c2"],
                },
                {
                    "op": "add_corner",
                    "target": target["id"],
                    "path_id": cutout["id"],
                    "feature_id": "cutout.rounded",
                    "entry_command_id": f"{cutout['id']}.c1",
                    "exit_command_id": f"{cutout['id']}.c3",
                    "radius": 3.0,
                },
            ],
        },
        ctx,
    ).structured_content

    assert rounded is not None
    rounded_paths = next(
        item
        for item in rounded["report"]["targets"]
        if item["id"] == target["id"]
    )["geometry"]["paths"]
    assert rounded_paths[0]["d"] == outer["d"]
    assert rounded_paths[1]["id"] == cutout["id"]
    assert rounded_paths[1]["d"] != cutout["d"]
    assert [
        command["type"] for command in rounded_paths[1]["commands"]
    ].count("quadratic") == 2
    declaration = next(
        anchor
        for anchor in rounded_paths[1]["anchors"]
        if anchor["id"] == "cutout.rounded"
    )
    assert declaration["radius"] == 3.0


def test_mcp_split_accepts_an_explicit_infinite_line_divider():
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(refine="none"), ctx).structured_content
    assert traced is not None

    split = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": "v0",
            "ops": [
                {"op": "split", "target": "r1", "divider": {"type": "line", "points": [[32, 0], [32, 64]]}},
            ],
        },
        ctx,
    ).structured_content

    assert split is not None
    assert [target["id"] for target in split["report"]["targets"]] == ["r1-1", "r1-2"]


def test_trace_options_name_the_absence_of_automatic_refinement_none():
    from pydantic import ValidationError

    assert TraceDrawingOptions(refine="none").refine == "none"
    with pytest.raises(ValidationError):
        TraceDrawingOptions(refine="interactive")


def test_unrefined_trace_stitches_retained_roots_before_creating_v0(monkeypatch):
    observed: list[int] = []
    import vectormark.python_drawing_engine as engine_module
    original_stitch = engine_module.stitch_regions_with_intent

    def observe(trace, regions, *, enabled=True):
        observed.append(len(regions))
        return original_stitch(trace, regions, enabled=enabled)

    monkeypatch.setattr(engine_module, "stitch_regions_with_intent", observe)
    ctx = SimpleNamespace(session=object())

    result = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(), ctx).structured_content

    assert result is not None
    assert observed == [len(result["trace"]["regions"])]


def test_mcp_detect_symmetry_exposes_an_axis_for_a_follow_up_set_symmetry_plan():
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(), ctx).structured_content
    assert traced is not None
    drawing_id = traced["drawing_id"]

    detected = refine_drawing(
        {
            "version": "vectormark.plan.v1", "drawing_id": drawing_id, "base_version": "v0",
            "ops": [{"op": "detect_symmetry", "target": "r1", "settings": {"epsilon": 0.5}}],
        },
        ctx,
    ).structured_content
    assert detected is not None
    assert [target["id"] for target in detected["report"]["targets"]] == ["r1"]
    source = detected["report"]["targets"][0]
    assert source["diagnostics"]["symmetry"]["mode"] == "self"
    assert source["diagnostics"]["symmetry"]["materialized"] is True
    detected_axis = source["diagnostics"]["symmetry"]["axis"]
    assert set(detected_axis) == {"point", "angle_degrees"}
    detected_svg = get_drawing_artifact(drawing_id, detected["version"], "svg", ctx).structured_content["svg"]
    _assert_full_square_target(detected_svg)

    set_result = refine_drawing(
        {
            "version": "vectormark.plan.v1", "drawing_id": drawing_id, "base_version": detected["version"],
            "ops": [{"op": "set_symmetry", "target": source["id"],
                      "axis": detected_axis}],
        },
        ctx,
    ).structured_content
    assert set_result is not None
    updated = next(item for item in set_result["report"]["targets"]
                   if item["diagnostics"].get("symmetry", {}).get("mode") == "explicit_self")
    assert updated["root_id"] == "r1"
    assert updated["diagnostics"]["symmetry"]["axis"] == detected_axis
    assert updated["diagnostics"]["symmetry"]["materialized"] is True
    _assert_full_square_target(get_drawing_artifact(drawing_id, set_result["version"], "svg", ctx).structured_content["svg"])
    assert get_drawing_artifact(drawing_id, detected["version"], "svg", ctx).structured_content["svg"] == detected_svg
    with pytest.raises(ToolError, match="PLAN_INVALID"):
        refine_drawing({
            "version": "vectormark.plan.v1", "drawing_id": drawing_id, "base_version": detected["version"],
            "ops": [{"op": "set_symmetry", "target": "r1-2", "axis": detected_axis}],
        }, ctx)


def _assert_full_square_target(svg: str) -> None:
    """A joined test-square target retains all four sides, not one mirror half."""
    from vectormark.fit import Shape
    from vectormark.refinement.region_sample import to_polygon

    target = next(element for element in ElementTree.fromstring(svg).iter() if element.get("id") == "r1")
    assert target.tag.endswith("path")
    footprint = to_polygon(Shape("path", {"d": target.attrib["d"], "fill_rule": target.get("fill-rule", "nonzero")}))
    minx, miny, maxx, maxy = footprint.bounds
    assert minx < 20 and miny < 20 and maxx > 44 and maxy > 44
    assert footprint.area > 900


def test_trace_auto_refines_the_retained_trace_into_a_live_child_version():
    ctx = SimpleNamespace(session=object())
    result = trace_drawing(
        ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(refine="auto", max_sampling_colors=4), ctx
    ).structured_content

    assert result is not None
    assert result["drawing_id"]
    assert result["version"] == "v1"
    assert result["parent_version"] == "v0"
    assert result["baseline_version"] == "v0"
    assert [output["version"] for output in result["outputs"]] == ["v0", "v1"]
    assert result["artifacts"]["svg"].endswith("/v1.svg")


def test_source_render_failure_does_not_publish_a_live_drawing(monkeypatch):
    before = set(mcp_server_module._DRAWINGS._drawings)
    monkeypatch.setattr(
        mcp_server_module,
        "render_canonical_drawing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("response render failed")
        ),
    )

    with pytest.raises(RuntimeError, match="response render failed"):
        trace_drawing(
            ImageRef(data_uri=_png_data_uri()),
            TraceDrawingOptions(refine="auto", max_sampling_colors=4),
            SimpleNamespace(session=object()),
        )

    assert set(mcp_server_module._DRAWINGS._drawings) == before


def test_guide_and_refine_render_failures_do_not_publish_or_consume_ids(monkeypatch):
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(
        ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(refine="none"), ctx
    ).structured_content
    assert traced is not None
    drawing_id = traced["drawing_id"]
    original_render = mcp_server_module.render_canonical_drawing

    monkeypatch.setattr(
        mcp_server_module,
        "render_canonical_drawing",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("candidate render failed")
        ),
    )
    with pytest.raises(ToolError, match="candidate render failed"):
        retrace_drawing(
            drawing_id,
            "v0",
            ImageRef(data_uri=_geometry_guide_data_uri()),
            GeometryGuideOptions(max_sampling_colors=4),
            ctx=ctx,
        )
    assert [
        item.id for item in mcp_server_module._DRAWINGS.get_manifest(drawing_id).versions
    ] == ["v0"]

    with pytest.raises(ToolError, match="candidate render failed"):
        refine_drawing(
            {
                "version": "vectormark.plan.v1",
                "drawing_id": drawing_id,
                "base_version": "v0",
                "ops": [
                    {
                        "op": "set_fill",
                        "target": "r1",
                        "fill": {"type": "flat", "color": "#123456"},
                    }
                ],
            },
            ctx,
        )
    assert [
        item.id for item in mcp_server_module._DRAWINGS.get_manifest(drawing_id).versions
    ] == ["v0"]

    monkeypatch.setattr(
        mcp_server_module, "render_canonical_drawing", original_render
    )
    guided = retrace_drawing(
        drawing_id,
        "v0",
        ImageRef(data_uri=_geometry_guide_data_uri()),
        GeometryGuideOptions(max_sampling_colors=4),
        ctx=ctx,
    ).structured_content
    assert guided is not None
    assert [item["version"] for item in guided["outputs"]] == [
        "v0.0",
        "v0.0.0",
    ]


def test_output_schema_validation_failures_are_atomic_for_all_mutations(monkeypatch):
    def reject(_cls, *_args, **_kwargs):
        raise ValueError("response schema rejected")

    ctx = SimpleNamespace(session=object())
    before = set(mcp_server_module._DRAWINGS._drawings)
    with monkeypatch.context() as patch:
        patch.setattr(
            mcp_server_module.TraceDrawingOutput,
            "model_validate",
            classmethod(reject),
        )
        with pytest.raises(ValueError, match="response schema rejected"):
            trace_drawing(
                ImageRef(data_uri=_png_data_uri()),
                TraceDrawingOptions(refine="auto", max_sampling_colors=4),
                ctx,
            )
    assert set(mcp_server_module._DRAWINGS._drawings) == before

    traced = trace_drawing(
        ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(refine="none"), ctx
    ).structured_content
    assert traced is not None
    drawing_id = traced["drawing_id"]
    expected_manifest = mcp_server_module._DRAWINGS.get_manifest(drawing_id)

    with monkeypatch.context() as patch:
        patch.setattr(
            mcp_server_module.RefinedDrawingOutput,
            "model_validate",
            classmethod(reject),
        )
        with pytest.raises(ToolError, match="response schema rejected"):
            refine_drawing(
                {
                    "version": "vectormark.plan.v1",
                    "drawing_id": drawing_id,
                    "base_version": "v0",
                    "ops": [{
                        "op": "set_fill",
                        "target": "r1",
                        "fill": {"type": "flat", "color": "#123456"},
                    }],
                },
                ctx,
            )
    assert mcp_server_module._DRAWINGS.get_manifest(drawing_id) == expected_manifest

    with monkeypatch.context() as patch:
        patch.setattr(
            mcp_server_module.RetracedDrawingOutput,
            "model_validate",
            classmethod(reject),
        )
        with pytest.raises(ToolError, match="response schema rejected"):
            retrace_drawing(
                drawing_id,
                "v0",
                ImageRef(data_uri=_geometry_guide_data_uri()),
                GeometryGuideOptions(max_sampling_colors=4),
                ctx=ctx,
            )
    assert mcp_server_module._DRAWINGS.get_manifest(drawing_id) == expected_manifest


def test_parentless_review_panel_does_not_render_the_baseline_as_parent():
    def svg(color: str) -> str:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8" '
            f'viewBox="0 0 8 8"><rect width="8" height="8" fill="{color}"/></svg>'
        )

    panel = mcp_server_module._render_review_panel_png(
        np.full((8, 8, 3), 255, np.uint8),
        SimpleNamespace(width=8, height=8),
        svg("#ff0000"),
        svg("#0000ff"),
        svg("#00ff00"),
        [],
        baseline_version="v0",
        parent_svg=None,
        parent_version=None,
    )
    assert panel is not None
    image = Image.open(io.BytesIO(panel)).convert("RGB")
    assert image.getpixel((950, 170)) == (229, 231, 235)


def test_geometry_guide_retraces_geometry_but_refills_every_leaf_from_the_original_source():
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(refine="none"), ctx).structured_content
    assert traced is not None

    guided_result = retrace_drawing(
        traced["drawing_id"],
        "v0",
        ImageRef(data_uri=_geometry_guide_data_uri()),
        GeometryGuideOptions(max_sampling_colors=4),
        ctx=ctx,
    )
    guided = guided_result.structured_content

    assert guided is not None
    assert guided["version"] == "v0.0.0"
    assert guided["requested_base_version"] == "v0"
    assert guided["baseline_version"] == "v0.0"
    assert guided["parent_version"] == "v0.0"
    assert [item["version"] for item in guided["outputs"]] == ["v0.0", "v0.0.0"]
    panel = guided_result.content
    assert Image.open(io.BytesIO(base64.b64decode(next(item for item in panel if item.type == "image").data))).size == (1140, 960)

    guide_trace = get_drawing_artifact(traced["drawing_id"], guided["version"], "raw_trace", ctx).structured_content
    svg = get_drawing_artifact(traced["drawing_id"], guided["version"], "svg", ctx).structured_content
    assert guide_trace is not None and "#ff00ff" in str(guide_trace["trace"]).lower()
    assert svg is not None and "#ff00ff" not in svg["svg"].lower()
    base, paint, base_evidence = mcp_server_module._DRAWINGS.load_replay(
        traced["drawing_id"], "v0"
    )
    child, child_paint, child_evidence = mcp_server_module._DRAWINGS.load_replay(
        traced["drawing_id"], guided["version"]
    )
    assert base_evidence.root_paths
    assert child_evidence.root_paths
    assert child_evidence.root_paths != base_evidence.root_paths
    assert child_paint == paint
    assert all(record.stage == "root-pre-stitch" for record in child_evidence.root_paths)
    assert "root_path_evidence" not in guide_trace["trace"]
    # Guide geometry remains available on descendants; the original raster is
    # still the only retained paint source.
    descendant = refine_drawing(
        {
            "version": "vectormark.plan.v1",
            "drawing_id": traced["drawing_id"],
            "base_version": child.id,
            "ops": [{"op": "fit_primitive", "target": "r1", "type": "rounded_rect"}],
        },
        ctx,
    ).structured_content
    assert descendant is not None
    _version, _paint, descendant_evidence = mcp_server_module._DRAWINGS.load_replay(
        traced["drawing_id"], descendant["version"]
    )
    assert descendant_evidence.revision == child_evidence.revision
    source_rgb = mcp_server_module.raster_pixels(paint.source)
    assert tuple(source_rgb[32, 32]) == (20, 100, 240)


def test_geometry_guide_requires_the_live_trace_canvas_without_silent_resampling():
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(refine="none"), ctx).structured_content
    assert traced is not None

    with pytest.raises(ToolError, match="geometry guide must be exactly 64x64"):
        retrace_drawing(
            traced["drawing_id"],
            "v0",
            ImageRef(data_uri=_small_geometry_guide_data_uri()),
            GeometryGuideOptions(max_sampling_colors=4),
            ctx=ctx,
        )


def test_trace_auto_threads_trace_controls_into_python_trace_and_auto_refinement(monkeypatch):
    captured = {}

    from vectormark.python_drawing_engine import PythonDrawingEngine
    import vectormark.drawing_refinement as refinement_module
    original_engine = PythonDrawingEngine.refine_automatically
    original_refine = refinement_module.refine_canonical_drawing

    def capture_engine(self, drawing, evidence, **kwargs):
        captured['trace'] = evidence.geometry.trace
        captured['canvas'] = evidence.paint.canvas
        return original_engine(self, drawing, evidence, **kwargs)

    def capture_refine(state, **kwargs):
        captured['state'] = state
        captured['settings'] = kwargs
        return original_refine(state, **kwargs)

    monkeypatch.setattr(PythonDrawingEngine, 'refine_automatically', capture_engine)
    monkeypatch.setattr(refinement_module, 'refine_canonical_drawing', capture_refine)
    trace_drawing(
        ImageRef(data_uri=_png_data_uri()),
        TraceDrawingOptions(
            refine="auto",
            max_sampling_colors=9,
            min_region_size=37,
            max_hole_area=91,
            min_region_fraction=0.125,
            simplify_tolerance=2.25,
            curve_tolerance=0.75,
        ),
        SimpleNamespace(session=object()),
    )

    options = captured["trace"].options
    assert options.max_sampling_colors == 9
    assert options.min_region_size == 37
    assert options.max_hole_area == 91
    assert options.simplify_tolerance == 2.25
    assert options.curve_tolerance == 0.75
    assert (captured['canvas'].height, captured['canvas'].width) == (captured['trace'].height, captured['trace'].width)
    assert captured['settings']['epsilon'] == 2.25
    assert captured['settings']['max_error'] == 0.75
    assert captured['state'].drawing.regions


def test_idealize_logo_threads_region_thresholds_into_pipeline_options(monkeypatch):
    captured = {}

    def capture(_rgb, *, options):
        captured["options"] = options
        return '<svg xmlns="http://www.w3.org/2000/svg"/>'

    monkeypatch.setattr(mcp_server_module, "idealize", capture)
    mcp_server_module.idealize_logo_image(
        {"data_uri": _png_data_uri()},
        {"min_region_size": 37, "min_region_fraction": 0.125},
        local_trust=True,
    )

    options = captured["options"]
    assert options.min_region_size == 37
    assert options.min_region_fraction == 0.125


def test_render_drawing_accepts_widget_result_shape():
    result = {"svg": "<svg>x</svg>", "width": 10, "height": 12}

    rendered = render_drawing(result=result)

    assert rendered["svg"] == result["svg"]
    assert rendered["svg_bytes"] == len(result["svg"].encode())


def test_trace_drawing_surfaces_image_error_codes():
    async def call():
        return await mcp.call_tool(
            "trace_drawing",
            {"image_data": {"data_uri": "data:image/png;base64,AA=="}, "options": {"refine": "auto"}},
        )

    try:
        asyncio.run(call())
        raise AssertionError("Expected trace_drawing to reject invalid image data")
    except ToolError as exc:
        assert "UNSUPPORTED_IMAGE_TYPE" in str(exc)


def test_refine_drawing_converts_an_executor_error_into_a_plan_error(monkeypatch):
    ctx = SimpleNamespace(session=object())
    traced = trace_drawing(ImageRef(data_uri=_png_data_uri()), TraceDrawingOptions(), ctx).structured_content
    assert traced is not None

    def fail(*_args, **_kwargs):
        raise ValueError("executor rejected this valid-looking plan")

    monkeypatch.setattr("vectormark.drawing_region_actions.fit_region_primitive", fail)
    with pytest.raises(ToolError, match="PLAN_INVALID.*executor rejected"):
        refine_drawing(
            {
                "version": "vectormark.plan.v1",
                "drawing_id": traced["drawing_id"],
                "base_version": "v0",
                "ops": [{"op": "fit_primitive", "target": "r1", "type": "circle"}],
            },
            ctx,
        )


def test_stdio_server_exposes_only_the_drawing_first_surface():
    async def list_tools_and_run_drawing_workflow():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "vectormark.mcp_server"],
            env={**os.environ, "VECTORMARK_MCP_TRANSPORT": "stdio"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                resources = await session.list_resources()
                widget = await session.read_resource(WIDGET_URI)
                auto = await session.call_tool(
                    "trace_drawing",
                    {"image_data": {"data_uri": _png_data_uri()}, "options": {"refine": "auto", "max_sampling_colors": 4}},
                )
                traced = await session.call_tool(
                    "trace_drawing",
                    {"image_data": {"data_uri": _png_data_uri()}, "options": {"refine": "none", "max_sampling_colors": 4}},
                )
                drawing_id = traced.structured_content["drawing_id"]
                refined = await session.call_tool(
                    "refine_drawing",
                    {
                        "plan": {
                            "version": "vectormark.plan.v1",
                            "drawing_id": drawing_id,
                            "base_version": "v0",
                            "ops": [{"op": "fit_primitive", "target": "r1", "type": "circle"}],
                        }
                    },
                )
                artifact = await session.call_tool(
                    "get_drawing_artifact",
                    {"drawing_id": drawing_id, "version": refined.structured_content["version"], "artifact": "svg"},
                )
                return initialized, tools, [str(resource.uri) for resource in resources.resources], widget.contents[0].text, auto, refined, artifact

    initialized, tools, resources, widget, auto, refined, artifact = asyncio.run(list_tools_and_run_drawing_workflow())

    assert set(tools) == {"trace_drawing", "retrace_drawing", "refine_drawing", "get_drawing_artifact", "render_drawing"}
    assert "refine='none'" in (initialized.instructions or "")
    assert tools["trace_drawing"].meta["openai/fileParams"] == ["image"]
    assert tools["retrace_drawing"].meta["openai/fileParams"] == ["geometry_guide"]
    refine_schema = tools["refine_drawing"].input_schema
    plan_schema = refine_schema["$defs"]["DrawingPlan"]
    assert "oneOf" in plan_schema["properties"]["ops"]["items"]
    assert "fit_primitive" in str(refine_schema)
    trace_output_schema = tools["trace_drawing"].output_schema
    target_geometry_schema = trace_output_schema["$defs"]["DrawingTargetOutput"]["properties"]["geometry"]
    assert target_geometry_schema["discriminator"]["propertyName"] == "type"
    assert "additionalProperties" not in target_geometry_schema
    assert set(trace_output_schema["required"]) >= {
        "drawing_id",
        "outputs",
        "version",
        "parent_version",
        "requested_base_version",
        "baseline_version",
        "trace",
        "artifacts",
        "report",
    }
    assert "svg" not in trace_output_schema["properties"]
    refine_output_schema = tools["refine_drawing"].output_schema
    assert set(refine_output_schema["required"]) >= {
        "drawing_id",
        "outputs",
        "version",
        "parent_version",
        "requested_base_version",
        "baseline_version",
        "artifacts",
        "report",
    }
    path_fit_schema = refine_schema["$defs"]["FitPathAction"]["properties"]
    assert path_fit_schema["between"]["anyOf"][0]["minItems"] == 2
    assert path_fit_schema["between"]["anyOf"][0]["maxItems"] == 2
    assert "match_length" in str(refine_schema)
    assert "quadratic" in str(refine_schema)
    for tool_name in ("trace_drawing", "retrace_drawing", "refine_drawing", "get_drawing_artifact", "render_drawing"):
        assert tools[tool_name].output_schema is not None
    assert WIDGET_URI in resources
    assert "Trace & refine" in widget
    assert "Requested base" in widget
    assert "Actual parent" in widget
    assert "Baseline" in widget
    assert auto.structured_content is not None
    assert auto.structured_content["drawing_id"]
    assert auto.structured_content["version"] == "v1"
    assert [output["version"] for output in auto.structured_content["outputs"]] == ["v0", "v1"]
    assert refined.structured_content["version"] == "v0.0"
    assert refined.structured_content["requested_base_version"] == "v0"
    assert refined.structured_content["parent_version"] == "v0"
    assert refined.structured_content["baseline_version"] == "v0"
    assert '<circle id="r1"' in artifact.structured_content["svg"]


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_http_artifacts_and_branches_survive_transport_reconnect(mode, tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {
        **os.environ,
        "VECTORMARK_MCP_TRANSPORT": "streamable-http",
        "VECTORMARK_MCP_HOST": "127.0.0.1",
        "VECTORMARK_MCP_PORT": str(port),
    }

    async def workflow(process):
        for _ in range(200):
            if process.poll() is not None:
                pytest.fail("HTTP server exited before accepting connections")
            try:
                _reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
                break
            except OSError:
                await asyncio.sleep(0.05)
        else:
            pytest.fail("HTTP server did not start within 10 seconds")

        url = f"http://127.0.0.1:{port}/mcp"
        data_uri = _png_data_uri()
        if mode == "auto":
            # Exercise an accepted image above MCP 2's default 4 MiB HTTP
            # limit without making the actual geometry expensive to trace.
            metadata = PngInfo()
            metadata.add_text("transport-size-check", "x" * (4 * 1024 * 1024))
            encoded = io.BytesIO()
            with Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))) as image:
                image.save(encoded, format="PNG", pnginfo=metadata)
            data_uri = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()
        async with Client(url, mode=mode, read_timeout_seconds=30) as client:
            tools = (await client.list_tools()).tools
            assert len(tools) == 5
            assert all(tool.output_schema is not None for tool in tools)
            traced = await client.call_tool("trace_drawing", {
                "image_data": {"data_uri": data_uri},
                "options": {"refine": "auto", "max_sampling_colors": 4},
            })
            assert not traced.is_error
            assert [output["version"] for output in traced.structured_content["outputs"]] == ["v0", "v1"]
            targets_by_version = {
                output["version"]: [target["id"] for target in output["report"]["targets"]]
                for output in traced.structured_content["outputs"]
            }
            assert targets_by_version["v0"] == ["r1"]
            assert targets_by_version["v1"] == ["r1"]
            auto_target = traced.structured_content["outputs"][1]["report"]["targets"][0]
            assert auto_target["diagnostics"]["symmetry"]["materialized"] is True
            assert set(auto_target["diagnostics"]["symmetry"]["axis"]) == {"point", "angle_degrees"}
            assert any(content.type == "image" for content in traced.content)
            wire = traced.model_dump(mode="json", by_alias=True)
            assert "structuredContent" in wire and "structured_content" not in wire
            assert next(content for content in wire["content"] if content["type"] == "image")["mimeType"] == "image/png"
            drawing_id = traced.structured_content["drawing_id"]
            base_svgs = {}
            for version in ("v0", "v1"):
                artifact = await client.call_tool("get_drawing_artifact", {
                    "drawing_id": drawing_id, "version": version, "artifact": "svg",
                })
                assert not artifact.is_error
                base_svgs[version] = artifact.structured_content["svg"]
            _assert_full_square_target(base_svgs["v1"])

        # The live drawing belongs to the drawing store, not the HTTP session.
        async with Client(url, mode=mode, read_timeout_seconds=30) as client:
            for base_version in ("v0", "v1"):
                target_ids = targets_by_version[base_version]
                if base_version == "v0":
                    ops = [{"op": "fit_primitive", "target": target_ids[0], "type": "circle"}]
                else:
                    ops = [
                        {
                            "op": "set_fill",
                            "target": target_id,
                            "fill": {"type": "flat", "color": "#1464F0"},
                        }
                        for target_id in target_ids
                    ]
                refined = await client.call_tool("refine_drawing", {"plan": {
                    "version": "vectormark.plan.v1",
                    "drawing_id": drawing_id,
                    "base_version": base_version,
                    "ops": ops,
                }})
                assert not refined.is_error
                assert refined.structured_content["version"] == f"{base_version}.0"
                assert refined.structured_content["report"]["targets"]
                for artifact in ("svg", "labeled_svg", "plan", "preview", "versions"):
                    result = await client.call_tool("get_drawing_artifact", {
                        "drawing_id": drawing_id,
                        "version": refined.structured_content["version"],
                        "artifact": artifact,
                    })
                    assert not result.is_error
                    if artifact == "svg":
                        if base_version == "v0":
                            assert '<circle id="r1"' in result.structured_content["svg"]
                        else:
                            _assert_full_square_target(result.structured_content["svg"])
                    elif artifact == "labeled_svg":
                        labeled_svg = result.structured_content["svg"]
                        assert all(f'id="map-{target_id}"' in labeled_svg for target_id in target_ids)
                    elif artifact == "plan":
                        assert result.structured_content["plan"]["plan"]["base_version"] == base_version
                    elif artifact == "preview":
                        assert any(content.type == "image" for content in result.content)
                    elif artifact == "versions":
                        assert result.structured_content["versions"]
                unchanged = await client.call_tool("get_drawing_artifact", {
                    "drawing_id": drawing_id, "version": base_version, "artifact": "svg",
                })
                assert not unchanged.is_error
                assert unchanged.structured_content["svg"] == base_svgs[base_version]
            stale = await client.call_tool("refine_drawing", {"plan": {
                "version": "vectormark.plan.v1", "drawing_id": drawing_id, "base_version": "v1",
                "ops": [{"op": "set_fill", "target": "r1-2", "fill": {"type": "flat", "color": "#1464F0"}}],
            }})
            assert stale.is_error
            assert "PLAN_INVALID" in "".join(item.text for item in stale.content if item.type == "text")

    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "vectormark.mcp_server"],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            asyncio.run(workflow(process))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_drawing_artifact_accepts_the_preview_alias_returned_by_trace_and_refine():
    from vectormark.mcp_server import get_drawing_artifact

    artifact_type = get_type_hints(get_drawing_artifact)["artifact"]

    assert "preview" in get_args(artifact_type)
    assert "review_panel" in get_args(artifact_type)

import numpy as np
import pytest

from vectormark.candidate import FlatFill
from vectormark.drawing_trace import PythonTraceEngine, TraceOptions
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion


@pytest.mark.parametrize("with_rgb", [False, True])
def test_engine_root_assembly_retains_actual_fit_evidence_and_stage_order(with_rgb):
    from vectormark.drawing_assembly import root_regions

    yy, xx = np.mgrid[:40, :40]
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 14**2
    rgb = np.full((40, 40, 3), 255, np.uint8)
    rgb[mask] = (180, 30, 40)
    trace = PythonTraceEngine().trace(
        rgb,
        TraceOptions(min_region_size=1, source_has_alpha=True),
        alpha=mask.astype(np.uint8) * 255,
    )
    stages = []
    result = root_regions(
        trace,
        rgb if with_rgb else None,
        min_region_fraction=0,
        allow_gradients=False,
        on_decomposition_stage=lambda name, regions, fills: stages.append(name),
    )
    assert len(result.regions) == 1
    assert result.regions[0].drawing_id == "r1"
    assert stages == (["material_union", "apply_fills"] if with_rgb else [])
    assert trace.root_path_evidence == ()
    (record,) = result.trace.root_path_evidence
    assert record.region_id == "r1" and record.path_id == "r1.p0"
    assert record.matches_output(
        result.regions[0].current.params["d"],
        stage=record.stage,
        revision=record.fit.revision,
    )


def test_shared_address_publication_keeps_existing_child_slots_and_source_geometry():
    from vectormark.drawing_adapter import normalize_drawing_addresses

    shape = Shape("rect", dict(x=0, y=0, w=4, h=4))
    first = VectorRegion(1001, shape, FlatFill("#123456"))
    second = VectorRegion(1002, shape, FlatFill("#abcdef"), drawing_id="r1-3")
    root = VectorRegion.branch(id=1, children=(first, second), drawing_id="r1")
    (result,) = normalize_drawing_addresses((root,))
    assert [child.drawing_id for child in result.children] == ["r1-1", "r1-3"]
    assert [child.id for child in result.children] == [1001, 1002]
    assert all(child.current is shape for child in result.children)
    assert root.children[0].drawing_id is None
    assert normalize_drawing_addresses((result,))[0] is result


def test_shared_address_publication_rejects_missing_root_address():
    from vectormark.drawing_adapter import normalize_drawing_addresses

    root = VectorRegion(1, Shape("rect", dict(x=0, y=0, w=4, h=4)), FlatFill("#123456"))
    with pytest.raises(ValueError, match="missing a stable drawing_id"):
        normalize_drawing_addresses((root,))

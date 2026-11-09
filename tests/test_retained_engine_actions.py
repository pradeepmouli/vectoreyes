import math

import pytest
import numpy as np

from vectoreyes.candidate import FlatFill
from vectoreyes.fit import Shape
from vectoreyes.refinement.automatic_replay import (
    ApplySeamConstraintsOperation,
    AxisSeamConstraint,
    PassInputRegionRef,
)
from vectoreyes.refinement.region_sample import RegionSample
from vectoreyes.skia_geometry import SkPath


def test_engine_applies_recorded_self_symmetry_without_selecting_another_fit():
    from vectoreyes.drawing_symmetry import Axis2D
    from vectoreyes.refinement.automatic_replay import apply_self_symmetry_choice

    angles = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    footprint = SkPath(
        shell=[(48 + 20 * math.cos(t), 48 + 30 * math.sin(t)) for t in angles]
    )
    points = list(footprint.exterior.coords)
    source = "M" + " L".join(f"{x} {y}" for x, y in points) + " Z"
    target = RegionSample(
        1,
        Shape("path", {"d": source}),
        FlatFill("#123456"),
        drawing_id="r1",
        footprint=footprint,
    )
    result = apply_self_symmetry_choice(
        [target],
        target,
        (),
        axis=Axis2D(math.pi / 2, 48, 48),
        residual=0,
        side="positive",
    )
    assert len(result) == 1 and result[0].id == 1
    assert result[0].fill is target.fill and result[0].original is target.original
    assert 1800 < result[0].footprint.area < 2000
    assert result[0].diagnostics["symmetry"]["materialized"] is True
    assert result[0].diagnostics["materialization"]["retired_region_ids"] == ()
    assert target.current.params["d"] == source


def test_engine_rejects_unknown_recorded_symmetry_fitter_without_mutation():
    from vectoreyes.drawing_symmetry import Axis2D
    from vectoreyes.refinement.automatic_replay import apply_self_symmetry_choice

    target = RegionSample(
        1, Shape("path", {"d": "M0 0 L10 0 L10 10 L0 10 Z"}), FlatFill("#123456")
    )
    with pytest.raises(ValueError, match="symmetry side"):
        apply_self_symmetry_choice(
            [target],
            target,
            (),
            axis=Axis2D(math.pi / 2, 5, 5),
            residual=0,
            side="not-a-side",
        )
    assert target.diagnostics == {}


def test_engine_applies_recorded_axis_seam_to_both_regions_and_preserves_paint():
    from vectoreyes.refinement.automatic_replay import apply_seam_constraints

    first = RegionSample(
        1, Shape("rect", {"x": 0, "y": 0, "w": 10, "h": 10}), FlatFill("#123456")
    )
    second = RegionSample(
        2, Shape("rect", {"x": 10.4, "y": 0, "w": 10, "h": 10}), FlatFill("#abcdef")
    )
    left, right = PassInputRegionRef("", "", (1,)), PassInputRegionRef("", "", (2,))
    action = ApplySeamConstraintsOperation(
        (left, right),
        (
            AxisSeamConstraint(
                left,
                right,
                "x",
                "right",
                "left",
                10,
                10.4,
                10.2,
                0,
                10,
                None,
                None,
                "midpoint",
            ),
        ),
        (),
        (),
        0.1,
        0.1,
        False,
        (),
    )

    class Resolver:
        def resolve(self, ref):
            return {1: first, 2: second}[ref.ancestry[0]]

    result = apply_seam_constraints(action, Resolver())
    assert result[0].footprint.bounds == pytest.approx((0, 0, 10.2, 10))
    assert result[1].footprint.bounds == pytest.approx((10.2, 0, 20.4, 10))
    assert result[0].fill is first.fill and result[1].fill is second.fill
    assert (
        result[0].original is first.original and result[1].original is second.original
    )
    assert first.current.params["w"] == 10 and second.current.params["x"] == 10.4

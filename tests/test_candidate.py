from vectoreyes.candidate import (
    Candidate, FlatFill, LinearGradientFill, RadialGradientFill,
)
from vectoreyes.fit import Shape
from vectoreyes.types import Axis


def test_flat_fill_holds_hex():
    assert FlatFill("#ff0000").hex == "#ff0000"


def test_gradient_fills_hold_geometry_and_stops():
    lin = LinearGradientFill({"x1": 0, "y1": 0, "x2": 1, "y2": 0}, [(0.0, "#000")])
    rad = RadialGradientFill({"cx": 1, "cy": 1, "r": 2}, [(1.0, "#fff")])
    assert lin.geometry["x2"] == 1 and lin.stops[0][1] == "#000"
    assert rad.geometry["r"] == 2 and rad.stops[0][0] == 1.0


def test_candidate_defaults_mirror_none():
    c = Candidate(Shape("circle", {"cx": 1, "cy": 1, "r": 1}), FlatFill("#000"), "region")
    assert c.mirror is None and c.source == "region"


def test_candidate_carries_mirror_axis():
    c = Candidate(Shape("path", {"d": "M0 0"}), FlatFill("#000"), "region", mirror=Axis(5.0))
    assert c.mirror.x == 5.0

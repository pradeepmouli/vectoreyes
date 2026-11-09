import pytest

from vectoreyes.candidate import FlatFill
from vectoreyes.drawing_adapter import bind_path_input
from vectoreyes.fit import Shape
from vectoreyes.refinement.region_sample import RegionSample
from vectoreyes.path_geometry import parse_svg_paths


def test_path_projection_rejects_stale_numeric_geometry_despite_format_normalization():
    (geometry,) = parse_svg_paths('M0 0 Q2 4 4 0 L0 0 Z', region_id='r1')
    target = RegionSample(1, Shape('path', {'d': 'M0.0 0.0 Q2.0 4.0 4.0 0.0 L0.0 0.0 Z',
        'canonical_path_geometry': geometry}), FlatFill('#123456'))

    assert bind_path_input(target).paths == (geometry,)
    stale = target.with_current(Shape('path', {**target.current.params,
        'd': 'M0 0 Q2 4.2 4 0 L0 0 Z'}))
    with pytest.raises(ValueError, match='does not match path data'):
        bind_path_input(stale)

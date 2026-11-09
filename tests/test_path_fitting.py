"""Whole-region numeric fitting retains ordered, typed contour geometry."""

import numpy as np
import pytest

from vectoreyes.contour import region_contours
from vectoreyes.drawing_actions import ActionSettings
from vectoreyes.fit import fit_path
from vectoreyes.path_geometry import PathGeometry, PathGeometryWithCutouts, parse_svg_paths
from vectoreyes.path_topology import classify_filled_paths


@pytest.mark.parametrize('strategy', ['quadratic', 'progressive', 'progressive_allow_lines'])
@pytest.mark.parametrize('cutout', [False, True])
def test_raster_path_kernel_preserves_fitter_geometry_and_contour_order(strategy, cutout):
    from vectoreyes.path_fitting import fit_raster_paths
    raster = np.zeros((20, 24), dtype=bool)
    raster[2:12, 3:13] = True
    raster[15:18, 18:22] = True
    if cutout:
        raster[5:8, 6:9] = False
    original = raster.copy()
    settings = ActionSettings(epsilon=0.1, max_error=0.2, fit_strategy=strategy)
    expected_d = ' '.join(str(fit_path(contour, epsilon=settings.epsilon,
        max_error=settings.max_error, cubic=False, progressive=strategy != 'quadratic',
        progressive_allow_lines=strategy == 'progressive_allow_lines',
        prefer_simple_curves=False).params['d'])
        for contour in region_contours(raster) if len(contour) >= 3)
    result = fit_raster_paths(raster, region_id='r2-3', settings=settings)
    assert result == parse_svg_paths(expected_d, region_id='r2-3')
    assert all(isinstance(path, PathGeometry) for path in result)
    assert [path.path.id for path in result] == [f'r2-3.p{i}' for i in range(len(result))]
    topology = classify_filled_paths(result, fill_rule='evenodd')
    assert len(topology.components) == 2
    assert any(isinstance(component, PathGeometryWithCutouts)
               for component in topology.components) == cutout
    np.testing.assert_array_equal(raster, original)


@pytest.mark.parametrize('raster', [np.zeros((0, 0), dtype=bool), np.zeros((8, 8), dtype=bool)])
def test_raster_path_kernel_rejects_empty_evidence(raster):
    from vectoreyes.path_fitting import fit_raster_paths
    with pytest.raises(ValueError, match='empty drawing region'):
        fit_raster_paths(raster, region_id='r1',
            settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))


@pytest.mark.parametrize('raster', [np.ones(8), np.ones((2, 3, 4))])
def test_raster_path_kernel_rejects_nonplanar_evidence(raster):
    from vectoreyes.path_fitting import fit_raster_paths
    with pytest.raises(ValueError, match='two-dimensional'):
        fit_raster_paths(raster, region_id='r1',
            settings=ActionSettings(epsilon=1, max_error=2, fit_strategy='progressive'))

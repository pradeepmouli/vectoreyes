from dataclasses import replace

import numpy as np
import pytest

from tests.test_canonical_automatic_replay import primitive_case
from vectormark.optimizer.automatic_replay import input_revision
from vectormark.optimizer.replay_binding import working_regions_from_state
from vectormark.replay_identity import canonical_input_revision


def test_input_identity_ignores_derived_footprint_encoding():
    from vectormark.skia_geometry import SkPath
    state, _, _ = primitive_case()
    region, = working_regions_from_state(state)
    # A footprint is disposable derived data, not the selected path itself.
    changed_cache = replace(region, footprint=SkPath(shell=[(0, 0), (1, 0), (0, 1)]))
    assert input_revision([region]) == input_revision([changed_cache])


def test_canonical_input_identity_needs_no_working_region_projection(monkeypatch):
    state, _, _ = primitive_case()
    expected = input_revision(list(working_regions_from_state(state)))
    def forbidden(*args, **kwargs):
        raise AssertionError('receipt materialized working regions')
    monkeypatch.setattr('vectormark.optimizer.replay_binding.working_regions_from_state', forbidden)
    assert canonical_input_revision(state) == expected


@pytest.mark.parametrize('field', ['raster', 'paint', 'geometry', 'original', 'diagnostics'])
def test_input_identity_still_binds_semantic_input_changes(field):
    from vectormark.candidate import FlatFill
    from vectormark.fit import Shape
    state, _, _ = primitive_case()
    region, = working_regions_from_state(state)
    if field == 'raster':
        raster = np.logical_not(region.raster)
        changed = replace(region, raster=raster)
    elif field == 'paint':
        changed = replace(region, fill=FlatFill('#ff00ff'))
    elif field == 'geometry':
        changed = replace(region, current=Shape('rect', {'x': 1., 'y': 1., 'w': 3., 'h': 4.}))
    elif field == 'original':
        changed = replace(region, original=Shape('rect', {'x': 2., 'y': 1., 'w': 3., 'h': 4.}))
    else:
        changed = region.with_diagnostics({'semantic': {'axis': 17}})
    assert input_revision([region]) != input_revision([changed])

"""Report entry points retain fitted overlaps instead of rejecting trace output."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_retained_topology_ingress import _region
from vectormark.refinement_settings import RefinementPasses


@pytest.mark.parametrize('entry', ['diagnostic', 'pipeline', 'cached'])
def test_trace_reports_retain_overlapping_subpaths(entry, monkeypatch, tmp_path):
    from vectormark.pipeline import Options, _idealize_refinement
    from vectormark.drawing_refine import auto_refine_with_intent

    # Fitting can push two formerly separate material fragments across each
    # other. This is retained input, not an authorized Boolean geometry edit.
    d = 'M2 2 L12 2 L12 12 L2 12 Z M11 11 L15 11 L15 15 L11 15 Z'
    source = _region(d, fill_rule='nonzero')
    masks = {source.id: source.raster}
    rgb = np.full((24, 48, 3), 255, np.uint8)
    passes = RefinementPasses(**{name: False for name in RefinementPasses.model_fields})
    options = Options(refinement=True, refinement_passes=passes)
    monkeypatch.setattr('vectormark.refinement.trace.trace_regions',
        lambda *args, **kwargs: ([source], masks))

    if entry == 'diagnostic':
        trace = SimpleNamespace(width=48, height=24, material_pretrace=None,
            options=SimpleNamespace(simplify_tolerance=1., curve_tolerance=1.))
        result = auto_refine_with_intent(trace, [source], refinement_passes=passes)
        assert result.regions[0].diagnostics['topology']['pending']
        assert result.regions[0].current.params['d'] == d
    elif entry == 'pipeline':
        svg, report = _idealize_refinement(rgb, options, 48, 24)
        assert '<svg' in svg and report.elements == 1
    else:
        from scripts import generate_corpus_html as corpus
        monkeypatch.setattr(corpus, '_load_or_create_trace', lambda *args: ([source], masks))
        svg, report = corpus._idealize_refinement_with_trace_cache(rgb, options=options,
            entry=SimpleNamespace(name='overlap'), cache_dir=tmp_path)
        assert '<svg' in svg and report.elements == 1
    assert source.current.params['d'] == d

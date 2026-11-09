import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vectoreyes.mcp_server import TraceDrawingOptions, _trace_options_schema


def test_selectable_presets_match_the_documented_bundles():
    text=Path('src/vectoreyes/assets/skills/designer-reconstruction/references/trace-presets.md').read_text()
    import re
    bundles=[json.loads(block) for block in re.findall(r'```json\n(.*?)\n```',text,re.S)]
    names=('black-and-white','flat-logo','gradient-artwork','conservative')
    assert len(bundles)==len(names)
    for name,bundle in zip(names,bundles,strict=True):
        resolved=TraceDrawingOptions(preset=name).model_dump(mode='json')
        assert all(resolved[key]==value for key,value in bundle.items())
        assert resolved['preset']==name


def test_explicit_options_override_presets_and_defaults_do_not_change():
    options=TraceDrawingOptions(preset='black-and-white',min_region_size=9,allow_gradients=True)
    assert options.min_region_size==9 and options.allow_gradients is True
    assert options.max_output_colors==2 and options.max_sampling_colors==2
    assert TraceDrawingOptions().fit_strategy=='quadratic'
    with pytest.raises(ValidationError):
        TraceDrawingOptions(preset='not-a-preset')


def test_preset_schema_exposes_registry_for_selectable_controls():
    presets=_trace_options_schema()['preset']['x-vectoreyes-presets']
    assert set(presets)=={'black-and-white','flat-logo','gradient-artwork','conservative'}
    assert presets['gradient-artwork']['options']['gradient_pretrace'] is True


def test_effective_preset_options_roundtrip_without_resetting_overrides():
    original=TraceDrawingOptions(preset='gradient-artwork',refine='none',curve_tolerance=1.5)
    assert TraceDrawingOptions.model_validate(original.model_dump())==original

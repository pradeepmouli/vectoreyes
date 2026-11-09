"""Trace starting points shared by the MCP schema and selectable controls."""
from copy import deepcopy
from typing import Literal

TracePreset = Literal['black-and-white', 'flat-logo', 'gradient-artwork', 'conservative']

TRACE_PRESETS: dict[str, dict[str, object]] = {
    'black-and-white': {'label':'Black-and-white / cutting', 'options':{
        'refine':'none', 'max_sampling_colors':2, 'max_output_colors':2,
        'allow_gradients':False, 'background_cutouts':'subtract', 'remove_background':'auto',
        'trace_level':'subpixel', 'fit_strategy':'progressive_allow_lines',
        'min_region_size':1, 'min_region_fraction':0, 'max_hole_area':0}},
    'flat-logo': {'label':'Flat-color geometric logo', 'options':{
        'refine':'auto', 'max_sampling_colors':'auto', 'max_output_colors':None,
        'allow_gradients':False, 'background_cutouts':'subtract',
        'trace_level':'subpixel', 'fit_strategy':'progressive_allow_lines'}},
    'gradient-artwork': {'label':'Gradient logo / illustration', 'options':{
        'refine':'auto', 'max_sampling_colors':'auto', 'max_output_colors':None,
        'allow_gradients':True, 'background_cutouts':'subtract',
        'trace_level':'subpixel', 'fit_strategy':'progressive', 'gradient_pretrace':True}},
    'conservative': {'label':'Conservative / diagnostic', 'options':{
        'refine':'none', 'max_sampling_colors':'auto', 'trace_level':'subpixel',
        'fit_strategy':'quadratic', 'gradient_pretrace':False}},
}


def apply_trace_preset(value: object) -> object:
    """Apply a bundle before model defaults; every explicit option wins."""
    if not isinstance(value, dict) or value.get('preset') is None:
        return value
    preset = value['preset']
    if not isinstance(preset, str) or preset not in TRACE_PRESETS:
        raise ValueError(f'unknown trace preset: {preset!r}')
    return {**deepcopy(TRACE_PRESETS[preset]['options']), **value}

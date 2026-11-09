"""Semantic region addresses resolve once; retained recipes use stable IDs."""
from dataclasses import replace
from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel

from .drawing import Drawing, _walk_regions
from .drawing_actions import AssignCutoutAction, RegionAddress, RegionNameReference, SetRegionNameAction
from .drawing_relationships import replace_region


_Action = TypeVar('_Action', bound=BaseModel)


def _named_region(name: str, names: Mapping[str, str | None]) -> str:
    matches = [identity for identity, value in names.items() if value == name]
    if not matches:
        raise ValueError(f"region name {name!r} not found in this version")
    if len(matches) != 1:
        raise ValueError(f"ambiguous region name {name!r}; matching IDs: {', '.join(matches)}")
    return matches[0]


def resolve_region_address(address: RegionAddress | Mapping[str, str], drawing: Drawing) -> str:
    if isinstance(address, str):
        drawing.region(address)
        return address
    name = RegionNameReference.model_validate(address).name
    return _named_region(name, {r.id:r.name for root in drawing.regions for r in _walk_regions(root)})


def resolve_action_addresses(action: _Action, drawing: Drawing, *,
        names: Mapping[str, str | None] | None = None) -> _Action:
    """Resolve typed name references recursively, never names inside arbitrary strings."""
    def resolve(value: object) -> object:
        if isinstance(value, RegionNameReference):
            return (_named_region(value.name, names) if names is not None else resolve_region_address(value, drawing))
        if isinstance(value, BaseModel):
            return {field: resolve(getattr(value, field)) for field in type(value).model_fields}
        if isinstance(value, tuple):
            return tuple(resolve(item) for item in value)
        return value
    return type(action).model_validate(resolve(action))


def set_region_name(drawing: Drawing, action: SetRegionNameAction) -> Drawing:
    action = resolve_action_addresses(action, drawing)
    return replace_region(drawing, replace(drawing.region(action.target), name=action.name))


def assign_cutout(drawing: Drawing, action: AssignCutoutAction) -> Drawing:
    action = resolve_action_addresses(action, drawing)
    source = drawing.region(action.source)
    if len(set(action.targets)) != len(action.targets):
        raise ValueError('cutout targets must be unique')
    return replace_region(drawing, replace(source, cutout_targets=tuple(action.targets)))

"""Shared ownership conversion for immutable diagnostic and snapshot values."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


_IMMUTABLE_SCALAR_TYPES = (type(None), bool, int, float, complex, str, bytes)


def freeze_value(value: object, ancestors: set[int] | None = None) -> object:
    """Detach supported containers and reject values without a safe frozen form."""
    snapshot = getattr(value, "__drawing_state_snapshot__", None)
    if callable(snapshot):
        return snapshot()
    if isinstance(value, np.generic):
        return value.item()
    if type(value) in _IMMUTABLE_SCALAR_TYPES:
        return value

    if not isinstance(value, Mapping | list | tuple | set | frozenset):
        raise TypeError(f"unsupported value type: {type(value).__qualname__}")

    ancestors = set() if ancestors is None else ancestors
    value_id = id(value)
    if value_id in ancestors:
        raise TypeError("cyclic container values are unsupported")
    ancestors.add(value_id)
    try:
        if isinstance(value, Mapping):
            return MappingProxyType({
                freeze_value(key, ancestors): freeze_value(item, ancestors)
                for key, item in value.items()
            })
        if isinstance(value, list | tuple):
            return tuple(freeze_value(item, ancestors) for item in value)
        return frozenset(freeze_value(item, ancestors) for item in value)
    finally:
        ancestors.remove(value_id)

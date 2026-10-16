"""Small conversions at public JSON boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def to_json_value(value: object) -> Any:
    """Return detached JSON-native containers for supported immutable evidence."""
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

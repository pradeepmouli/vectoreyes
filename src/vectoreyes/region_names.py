"""Shared naming contract for retained regions and public name addresses."""
import re
from typing import Annotated

from pydantic import StringConstraints

REGION_NAME_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
RegionName = Annotated[str, StringConstraints(pattern=REGION_NAME_PATTERN, max_length=128)]


def validate_region_name(name: str | None) -> None:
    if name is not None and (not isinstance(name, str) or len(name) > 128
            or re.fullmatch(REGION_NAME_PATTERN, name) is None):
        raise ValueError("region names must be lowercase hyphenated words (for example orange-band), at most128 characters")

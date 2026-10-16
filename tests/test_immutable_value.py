from types import MappingProxyType

import numpy as np
import pytest

from vectormark.immutable_value import freeze_value


class SnapshotValue:
    def __drawing_state_snapshot__(self):
        return ("snapshot",)


def test_freeze_value_preserves_snapshot_scalars_and_container_ownership():
    source = {
        "items": [np.int64(3), {"value"}],
        "snapshot": SnapshotValue(),
    }

    frozen = freeze_value(source)
    source["items"].append(4)

    assert isinstance(frozen, MappingProxyType)
    assert frozen == {
        "items": (3, frozenset({"value"})),
        "snapshot": ("snapshot",),
    }
    assert type(frozen["items"][0]) is int
    with pytest.raises(TypeError):
        frozen["added"] = True


def test_freeze_value_preserves_cycle_and_unsupported_type_errors():
    cyclic = []
    cyclic.append(cyclic)

    with pytest.raises(TypeError, match="cyclic container values are unsupported"):
        freeze_value(cyclic)
    with pytest.raises(TypeError, match="unsupported value type: object"):
        freeze_value(object())

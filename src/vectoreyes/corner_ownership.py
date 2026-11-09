"""Representation-independent ownership for directed closed-path corners."""

from __future__ import annotations

from collections.abc import Sequence


def directed_corner_ownership(
    command_ids: Sequence[str],
    entry_command_id: str,
    exit_command_id: str,
    *,
    explicit_closing_drawable: bool,
) -> tuple[str, ...]:
    """Return drawable commands after ``entry`` through ``exit`` in path order.

    ``command_ids`` contains the leading move followed by drawable commands.
    Crossing the serialization origin is addressable only when the final
    drawable explicitly returns to the move endpoint.
    """
    if len(command_ids) < 2 or len(set(command_ids)) != len(command_ids):
        raise ValueError("corner ownership requires unique move and drawable command IDs")
    try:
        entry = command_ids.index(entry_command_id)
        exit = command_ids.index(exit_command_id)
    except ValueError as error:
        raise ValueError("corner boundary references an unknown command") from error
    if entry == exit:
        return ()
    wraps = exit <= entry
    if wraps and not explicit_closing_drawable:
        raise ValueError("corner span crosses an unaddressable implicit closing edge")
    drawable = tuple(command_ids[1:])
    entry_position = -1 if entry == 0 else entry - 1
    exit_position = len(drawable) - 1 if exit == 0 else exit - 1
    if exit_position > entry_position:
        return drawable[entry_position + 1:exit_position + 1]
    return drawable[entry_position + 1:] + drawable[:exit_position + 1]

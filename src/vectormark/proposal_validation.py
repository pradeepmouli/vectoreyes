"""Identity rules shared by automatic proposal selection representations."""

from collections.abc import Collection, Sequence


def proposal_rejection_reason(
    proposal_ids: Sequence[int], replacement_ids: Sequence[int] | None, *,
    input_ids: Collection[int], mask_ids: Collection[int],
    consumed_ids: Collection[int], current_ids: Collection[int],
) -> str | None:
    """Validate a proposal without mutating or reserving its input identities.

    ``None`` replacements performs input-only preflight before numeric work.
    An empty sequence is a prepared, but invalid, empty replacement.
    """
    if len(set(proposal_ids)) != len(proposal_ids):
        return 'duplicate_proposal_id'
    if not proposal_ids:
        return 'empty_proposal'
    if replacement_ids is not None:
        if not replacement_ids:
            return 'empty_replacement'
        if len(set(replacement_ids)) != len(replacement_ids):
            return 'duplicate_replacement_id'
    if any(identity in consumed_ids for identity in proposal_ids):
        return 'overlapping_accepted_proposal'
    if any(identity not in input_ids or identity not in mask_ids for identity in proposal_ids):
        return 'missing_input_or_mask'
    if replacement_ids is not None:
        if any(identity in input_ids and identity not in proposal_ids for identity in replacement_ids):
            return 'replacement_collides_with_input'
        if any(identity in current_ids and identity not in proposal_ids for identity in replacement_ids):
            return 'replacement_collides_with_current'
    return None

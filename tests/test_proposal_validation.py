import pytest

from vectoreyes.proposal_validation import proposal_rejection_reason


@pytest.mark.parametrize('proposed,replacements,consumed,current,reason', [
    ((1, 1), (1,), (), (1, 2), 'duplicate_proposal_id'),
    ((), (1,), (), (1, 2), 'empty_proposal'),
    ((1,), (), (), (1, 2), 'empty_replacement'),
    ((1,), (1, 1), (), (1, 2), 'duplicate_replacement_id'),
    ((1,), (1,), (1,), (1, 2), 'overlapping_accepted_proposal'),
    ((3,), (3,), (), (1, 2), 'missing_input_or_mask'),
    ((1,), (2,), (), (1, 2), 'replacement_collides_with_input'),
    ((1,), (3,), (), (1, 2, 3), 'replacement_collides_with_current'),
    ((1,), (3,), (), (1, 2), None),
    ((1,), None, (), (1, 2), None),
])
def test_shared_proposal_rules(proposed, replacements, consumed, current, reason):
    assert proposal_rejection_reason(proposed, replacements, input_ids=(1, 2),
        mask_ids=(1, 2), consumed_ids=consumed, current_ids=current) == reason


def test_preflight_requires_source_observation_even_when_geometry_exists():
    assert proposal_rejection_reason((1,), None, input_ids=(1,), mask_ids=(),
        consumed_ids=(), current_ids=(1,)) == 'missing_input_or_mask'

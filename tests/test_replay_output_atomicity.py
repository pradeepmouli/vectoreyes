import pytest

from vectoreyes.refinement.automatic_replay import NamedRegionOutput, OperationOutputRegionRef, ReplayBatch
from vectoreyes.replay_inputs import ReplayInputResolver


def test_failed_output_transaction_does_not_leave_earlier_outputs_registered():
    resolver = ReplayInputResolver({}, ReplayBatch('batch', 'revision', (), ()), lambda item: {})
    resolver.register('producer', NamedRegionOutput('existing', (1,)), 'old')
    with pytest.raises(ValueError, match='duplicate automatic replay named output'):
        resolver.register_many('producer', (
            (NamedRegionOutput('new', (2,)), 'new'),
            (NamedRegionOutput('existing', (3,)), 'replacement'),
        ))
    assert resolver.resolve(OperationOutputRegionRef('producer', 'existing')) == 'old'
    with pytest.raises(ValueError, match='forward-referenced'):
        resolver.resolve(OperationOutputRegionRef('producer', 'new'))

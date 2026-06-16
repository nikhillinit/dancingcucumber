import pytest

from advisor.backtest.splits import purged_splits, inner_blocks


def test_train_strictly_precedes_test_with_embargo():
    splits = purged_splits(n=1000, folds=5, embargo=5)
    assert len(splits) == 4                      # folds-1 evaluable test folds
    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx) - 5     # embargo gap enforced
        assert set(train_idx).isdisjoint(test_idx)    # no overlap


def test_no_test_index_appears_in_any_train():
    splits = purged_splits(n=800, folds=4, embargo=10)
    for _, test_idx in splits:
        for other_train, _ in splits:
            future = [i for i in other_train if i > min(test_idx)]
            assert all(i < min(test_idx) - 10 for i in future) or future == []


def test_inner_blocks_partitions_train_for_rule_b():
    blocks = inner_blocks(train_len=300, n_blocks=2)
    assert len(blocks) == 2
    assert blocks[0][-1] < blocks[1][0]          # ordered, disjoint


def test_degenerate_inputs_return_empty():
    assert purged_splits(n=20, folds=5, embargo=5) == []   # too small -> no folds

from __future__ import annotations

import numpy as np


def purged_splits(n: int, folds: int = 5, embargo: int = 5) -> list[tuple[list[int], list[int]]]:
    """Expanding-window purged walk-forward. Train = everything strictly before
    (test_start - embargo); test = the fold block. Returns [] if too small."""
    if folds < 2 or embargo < 0 or n < folds * 40:
        return []
    fold_size = n // folds
    out: list[tuple[list[int], list[int]]] = []
    used_test_idx: set[int] = set()
    for fold in range(1, folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < folds - 1 else n
        train_end = test_start - embargo
        if train_end <= 0 or test_start >= test_end:
            continue
        train_idx = [i for i in range(0, train_end) if i not in used_test_idx]
        test_idx = list(range(test_start, test_end))
        assert max(train_idx) < min(test_idx) - embargo
        out.append((train_idx, test_idx))
        used_test_idx.update(test_idx)
    return out


def inner_blocks(train_len: int, n_blocks: int = 2) -> list[list[int]]:
    """Disjoint ordered sub-blocks of a train fold (Rule B 'appears in >=2 blocks')."""
    if train_len < n_blocks * 20:
        return []
    edges = np.linspace(0, train_len, n_blocks + 1).astype(int)
    return [list(range(edges[i], edges[i + 1])) for i in range(n_blocks)]

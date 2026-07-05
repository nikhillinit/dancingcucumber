from __future__ import annotations

import pytest

from advisor.backtest.select import minimax_select, rank_minimax, RobustPick


def test_steady_beats_lucky_spike():
    # A has a great mean but a terrible worst fold; B is steadier with higher min.
    cands = {"A": [0.9, 0.9, -0.5], "B": [0.3, 0.35, 0.32]}
    pick = minimax_select(cands)
    assert pick.candidate_key == "B"
    assert pick.min_score == pytest.approx(0.30)


def test_tie_on_min_broken_by_mean():
    cands = {"A": [0.2, 0.5], "B": [0.2, 0.9]}
    pick = minimax_select(cands)
    assert pick.candidate_key == "B"
    assert pick.min_score == pytest.approx(0.2)
    assert pick.mean_score == pytest.approx(0.55)


def test_tie_on_min_and_mean_broken_by_key():
    cands = {"b": [0.4, 0.6], "a": [0.4, 0.6]}
    pick = minimax_select(cands)
    assert pick.candidate_key == "a"  # deterministic by key


def test_rank_full_order_and_fields():
    cands = {"A": [0.9, 0.9, -0.5], "B": [0.3, 0.35, 0.32], "C": [0.1, 0.1, 0.1]}
    ranked = rank_minimax(cands)
    assert [p.candidate_key for p in ranked] == ["B", "C", "A"]
    b = ranked[0]
    assert isinstance(b, RobustPick)
    assert b.per_fold == (0.3, 0.35, 0.32)
    assert b.min_score == pytest.approx(0.30)
    assert b.mean_score == pytest.approx((0.3 + 0.35 + 0.32) / 3)


def test_empty_mapping_raises():
    with pytest.raises(ValueError):
        minimax_select({})
    with pytest.raises(ValueError):
        rank_minimax({})


def test_empty_fold_list_raises():
    with pytest.raises(ValueError):
        minimax_select({"A": [0.1, 0.2], "B": []})

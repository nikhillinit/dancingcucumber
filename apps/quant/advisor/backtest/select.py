# apps/quant/advisor/backtest/select.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence


@dataclass(frozen=True)
class RobustPick:
    candidate_key: Hashable
    min_score: float
    mean_score: float
    per_fold: tuple[float, ...]


def _summarize(key: Hashable, folds: Sequence[float]) -> RobustPick:
    vals = tuple(float(v) for v in folds)
    if not vals:
        raise ValueError(f"candidate {key!r} has empty fold list")
    return RobustPick(key, min(vals), sum(vals) / len(vals), vals)


def _order_key(p: RobustPick) -> tuple:
    # most robust first: highest min, then highest mean, then key for determinism
    return (-p.min_score, -p.mean_score, repr(p.candidate_key))


def rank_minimax(candidates: Mapping[Hashable, Sequence[float]]) -> list[RobustPick]:
    """Full ranking, best (most robust) first, by (min, then mean, then key).
    Raises ValueError on empty mapping or any empty fold list."""
    if not candidates:
        raise ValueError("candidates mapping is empty")
    picks = [_summarize(k, v) for k, v in candidates.items()]
    return sorted(picks, key=_order_key)


def minimax_select(candidates: Mapping[Hashable, Sequence[float]]) -> RobustPick:
    """Pick the candidate whose MINIMUM per-fold score is largest. Tie-break by
    higher mean per-fold score, then by key for determinism. Raises ValueError on
    empty mapping or any empty fold list."""
    return rank_minimax(candidates)[0]

from __future__ import annotations

# Deterministic, auditable sentiment lexicon. Intentionally NOT a model: an LLM scorer
# would reintroduce pretraining look-ahead, which the floor exists to exclude.
POSITIVE = {
    "beat", "beats", "surge", "surges", "upgrade", "upgraded", "growth", "record",
    "strong", "raises", "raised", "outperform", "profit", "profits", "gains", "wins",
    "approval", "approved", "soars", "rally", "tops",
}
NEGATIVE = {
    "miss", "misses", "cut", "cuts", "downgrade", "downgraded", "probe", "lawsuit",
    "weak", "loss", "losses", "falls", "plunge", "plunges", "recall", "fraud",
    "warning", "warns", "halts", "slumps", "bankruptcy",
}


def lexicon_score(headline: str) -> float:
    """Bag-of-words sentiment in [-1, 1]. 0.0 when no lexicon word is present."""
    tokens = [t.strip(".,!?:;\"'()[]").lower() for t in headline.split()]
    pos = sum(1 for t in tokens if t in POSITIVE)
    neg = sum(1 for t in tokens if t in NEGATIVE)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)

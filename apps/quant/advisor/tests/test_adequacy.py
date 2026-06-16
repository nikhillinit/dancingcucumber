import numpy as np
import pandas as pd

from advisor.backtest.adequacy import is_adequate


def test_genuinely_continuous_scores_pass():
    scores = pd.DataFrame(np.random.default_rng(0).uniform(0, 1, (200, 25)))
    assert is_adequate(scores) is True


def test_near_binary_scores_fail():
    # almost all 0 or 1 -> few distinct levels, tiny IQR among positives
    scores = pd.DataFrame(np.random.default_rng(0).integers(0, 2, (200, 25)).astype(float))
    assert is_adequate(scores) is False

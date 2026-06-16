from __future__ import annotations

import numpy as np
import pandas as pd


def is_adequate(train_scores: pd.DataFrame,
                min_distinct: int = 10, min_iqr: float = 0.15) -> bool:
    """A family's train scores are 'genuinely continuous' if positive scores have
    >= min_distinct levels OR the median per-date IQR of positive scores >= min_iqr."""
    vals = pd.DataFrame(train_scores).to_numpy().ravel()
    pos = vals[vals > 0]
    if len(pos) == 0:
        return False
    if len(np.unique(np.round(pos, 6))) >= min_distinct:
        return True
    iqrs = []
    for _, row in pd.DataFrame(train_scores).iterrows():
        rp = row[row > 0].to_numpy()
        if len(rp) >= 4:
            iqrs.append(np.percentile(rp, 75) - np.percentile(rp, 25))
    return bool(iqrs) and float(np.median(iqrs)) >= min_iqr

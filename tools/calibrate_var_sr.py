# tools/calibrate_var_sr.py -- compute declared_var_sr from the pre-registered trial book.
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "apps/quant")
from advisor.backtest.pipeline import run_dev_sweep            # noqa: E402
from advisor.backtest.prereg import DEFAULT_CONFIG             # noqa: E402
from advisor.backtest.validation import per_obs_sharpe, var_sr_trials  # noqa: E402

FIXTURE = Path("apps/quant/advisor/tests/fixtures/floor_prices.csv")

# The pre-registered candidate order (PREREG.md "Candidate order", smallest-first).
TRIAL_BOOK = {
    "A": ("trend",),
    "B": ("momentum",),
    "C": ("momentum", "trend"),
    "D": ("momentum", "trend", "mean_reversion"),
    "E": ("momentum", "trend", "mean_reversion", "breakout"),
}


def main() -> int:
    panel = pd.read_csv(FIXTURE, index_col=0, parse_dates=True)
    sharpes = {}
    for name, fams in TRIAL_BOOK.items():
        sweep = run_dev_sweep(panel, fams, DEFAULT_CONFIG)
        sharpes[name] = per_obs_sharpe(sweep.ensemble_test_returns)
    vals = list(sharpes.values())
    measured = var_sr_trials(vals)
    declared = math.ceil(measured * 1e4) / 1e4          # round UP to next 1e-4 -> errs strict
    print(f"trial_count = {len(vals)}")
    print(f"trial_sharpes (per-obs) = {sharpes}")
    print(f"var_sr_trials (measured) = {measured:.6e}")
    print(f"declared_var_sr (rounded up) = {declared:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

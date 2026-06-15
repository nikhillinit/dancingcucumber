import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "apps/quant")
from advisor.backtest.data_floor import floor_metrics  # noqa: E402

FIXTURE = Path("apps/quant/advisor/tests/fixtures/floor_prices.csv")
MARGIN = 0.0  # pre-registered v1 floor margin over SPY (spec section 15)


def main(argv: list[str]) -> int:
    enforce = "--enforce" in argv
    if not FIXTURE.exists():
        print(f"floor: fixture missing at {FIXTURE} -- commit real price history first", flush=True)
        return 1  # a broken mechanism always fails, in any mode
    panel = pd.read_csv(FIXTURE, index_col=0, parse_dates=True)
    m = floor_metrics(panel, benchmark="SPY", margin=MARGIN)
    print("floor metrics: " + json.dumps(m), flush=True)
    if m["passes"]:
        print("floor: PASSED", flush=True)
        return 0
    print(
        f"floor: NOT CLEARED -- ensemble Sharpe {m['ensemble']:.2f} vs SPY {m['spy']:.2f} "
        f"and best single family {m['best_family']:.2f}. v1 equal-weight is not "
        f"production-ready; needs v2 calibration (spec section 8).",
        flush=True,
    )
    return 1 if enforce else 0  # block release; do NOT block dev commits


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

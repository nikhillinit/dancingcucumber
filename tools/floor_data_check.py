import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "apps/quant")
from advisor.backtest.data_floor import floor_metrics  # noqa: E402
from advisor.backtest.prereg import DEFAULT_CONFIG  # noqa: E402

FIXTURE = Path("apps/quant/advisor/tests/fixtures/floor_prices.csv")


def _print_verdict(m: dict) -> None:
    print("floor metrics: " + json.dumps(m), flush=True)
    verdict = m["verdict"]
    if verdict == "PASSED":
        print(
            f"floor: PASSED -- ensemble Sharpe {m['ensemble']:.2f}; "
            f"SPY Sharpe {m['spy']:.2f}; best family Sharpe {m['best_family']:.2f}.",
            flush=True,
        )
    elif verdict == "INCONCLUSIVE":
        print(
            f"floor: INCONCLUSIVE -- dev gate cleared, but holdout LCBs did not clear "
            f"both section 7 bars. ensemble Sharpe {m['ensemble']:.2f}; "
            f"SPY Sharpe {m['spy']:.2f}; best family Sharpe {m['best_family']:.2f}.",
            flush=True,
        )
    elif verdict == "UNSUPPORTED":
        print(
            f"floor: UNSUPPORTED -- universe classified as {m['universe']}. "
            f"ensemble Sharpe {m['ensemble']:.2f}; SPY Sharpe {m['spy']:.2f}; "
            f"best family Sharpe {m['best_family']:.2f}.",
            flush=True,
        )
    else:
        reasons = "; ".join(m["dev"]["reasons"]) or "dev gate did not pass"
        print(
            f"floor: DEV_FAILED -- {reasons}. ensemble Sharpe {m['ensemble']:.2f}; "
            f"SPY Sharpe {m['spy']:.2f}; best family Sharpe {m['best_family']:.2f}.",
            flush=True,
        )


def main(argv: list[str]) -> int:
    enforce = "--enforce" in argv
    if not FIXTURE.exists():
        print(f"floor: fixture missing at {FIXTURE} -- commit real price history first", flush=True)
        return 1 if enforce else 0
    panel = pd.read_csv(FIXTURE, index_col=0, parse_dates=True)
    m = floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=None)
    _print_verdict(m)
    return 0 if not enforce or m["verdict"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

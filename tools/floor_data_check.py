import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "apps/quant")
from advisor.backtest.data_floor import floor_metrics  # noqa: E402
from advisor.backtest.prereg import DEFAULT_CONFIG, config_hash  # noqa: E402
from advisor.backtest.walk_forward import disclosure_header  # noqa: E402

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
    v = m.get("validation")
    if v:
        flag = "PASS" if v["passes"] else "FAIL"
        print(
            f"floor: validation (report-only) -- DSR {v['dsr']:.2f} vs bar "
            f"{v['dsr_pass_bar']:.2f} [{flag}]; N_used {v['n_used']}; "
            f"MinBTL_exceeded {v['minbtl_exceeded']}. This guard can only confirm "
            f"the floor, never unlock the holdout or authorize sizing.",
            flush=True,
        )
    d = m.get("diagnostics")
    if d:
        c = d["concentration"]
        rf = d.get("robust_family")
        rf_s = (f"; minimax-robust family '{rf['family']}' "
                f"(min fold Sharpe {rf['min_fold_sharpe']:.2f})") if rf else ""
        print(
            f"floor: risk diagnostics (report-only) -- ensemble Sortino "
            f"{d['ensemble_sortino']:.2f}, maxDD {d['ensemble_max_drawdown'] * 100:.1f}%; "
            f"SPY Sortino {d['spy_sortino']:.2f}, maxDD {d['spy_max_drawdown'] * 100:.1f}% "
            f"(both dev-OOS){rf_s}.",
            flush=True,
        )
        if "min_invested_breadth" in c:
            cflag = "PASS" if c["passes"] else "FAIL"
            t = c["thresholds"]
            print(
                f"floor: book concentration (report-only) -- breadth "
                f"{c['min_invested_breadth']}..{c['median_invested_breadth']:.0f} names, "
                f"max single {c['max_single_name'] * 100:.0f}%, "
                f"top-{c['k']} {c['max_top_k'] * 100:.0f}% "
                f"[{cflag} vs {t['min_breadth']}/"
                f"{t['max_single_name'] * 100:.0f}%/{t['max_top_k'] * 100:.0f}%]. "
                f"Report-only: cannot unlock the holdout or authorize sizing.",
                flush=True,
            )


def main(argv: list[str]) -> int:
    enforce = "--enforce" in argv
    if not FIXTURE.exists():
        print(f"floor: fixture missing at {FIXTURE} -- commit real price history first", flush=True)
        return 1 if enforce else 0
    panel = pd.read_csv(FIXTURE, index_col=0, parse_dates=True)
    # Holdout integrity (debate finding #1): the held-out tail is unlocked ONLY by
    # the content hash of the actual (config + fixture) -- never an arbitrary string.
    # Report/enforce default to prereg_hash=None so the holdout stays blinded during
    # dev; the operator (Task 14) runs --holdout ONCE and confirms the printed hash
    # equals the one committed in PREREG.md before trusting the verdict.
    prereg_hash = None
    if "--holdout" in argv:
        prereg_hash = config_hash(DEFAULT_CONFIG, FIXTURE)
        print(f"floor: holdout unlocked with config+fixture hash {prereg_hash}", flush=True)
    m = floor_metrics(panel, DEFAULT_CONFIG, prereg_hash=prereg_hash)
    _print_verdict(m)
    print(disclosure_header(), flush=True)  # necessary-not-sufficient + survivorship + regime caveats
    return 0 if not enforce or m["verdict"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

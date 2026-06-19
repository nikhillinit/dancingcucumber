"""WS4 — Reading-B fundamental candidate DEV-gate run (holdout BLINDED).

Calls fundamental_candidate_metrics with prereg_hash=None so the reserved tail is
never read. Decision is driven off res["dev"]["passed"], NOT res["verdict"] (which
reads DEV_FAILED by construction whenever the holdout is blinded). Report-only.
Run: PYTHONPATH=apps/quant python ai-logs/hermes/run_ws4_dev.py
"""
import json

import pandas as pd

from advisor.data.edgar_xbrl_fixture import load_fixture
from advisor.research.fundamental_value import build_fundamental_panel
from advisor.research.candidate_floor import fundamental_candidate_metrics
from advisor.research.candidate_prereg_fundamental import (
    DEFAULT_FUNDAMENTAL_CANDIDATE as CFG,
)

PANEL_PATH = "apps/quant/advisor/tests/fixtures/floor_prices.csv"
FIXTURE_PATH = "apps/quant/advisor/tests/fixtures/edgar_xbrl_fundamentals.csv"

panel = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)
records = load_fixture(FIXTURE_PATH)
panel_funda = build_fundamental_panel(records, panel, warmup=CFG.warmup)  # assets default = non-SPY

res = fundamental_candidate_metrics(panel, panel_funda, CFG, prereg_hash=None)  # HOLDOUT BLINDED

print(json.dumps(res, indent=2, default=str))
print("=== WS4 DECISION INPUTS ===")
print("dev.passed       :", res["dev"]["passed"])
print("dev.reasons      :", res["dev"]["reasons"])
print("verdict (blinded):", res["verdict"], "(reads DEV_FAILED by construction when holdout blinded)")
print("holdout          :", res["holdout"], "(MUST be null with prereg_hash=None)")
print("ensemble         :", res["ensemble"])
print("best_family      :", res["best_family"])
print("spy (dev-window) :", res["spy"])
print("power.power_limited:", res["power"]["power_limited"])

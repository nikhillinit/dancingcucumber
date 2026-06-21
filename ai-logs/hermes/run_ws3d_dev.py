"""WS3D — Reading-C lazy_prices candidate DEV-gate run (holdout BLINDED).

Calls lazy_prices_candidate_metrics with prereg_hash=None so the reserved tail is never
read. Decision is driven off res["dev"]["passed"], NOT res["verdict"] (which reads
DEV_FAILED by construction whenever the holdout is blinded). Also reports the two
report-only diagnostics (dev_lazy_momentum_corr, dev_cross_sectional_dispersion) and the
three pinned Reading-C hashes. Report-only.
Run: PYTHONPATH=apps/quant python ai-logs/hermes/run_ws3d_dev.py
"""
import json

import pandas as pd

from advisor.data.edgar_xbrl_fixture import load_fixture
from advisor.research.lazy_prices import (
    build_lazy_prices_panel, dev_lazy_momentum_corr, dev_cross_sectional_dispersion,
)
from advisor.research.candidate_floor import lazy_prices_candidate_metrics
from advisor.research.candidate_prereg_lazy_prices import (
    DEFAULT_LAZY_PRICES_CANDIDATE as CFG,
    lazy_prices_candidate_hash, lazy_prices_candidate_run_hash,
)
from advisor.research.candidate_validation_prereg_lazy_prices import (
    DEFAULT_LAZY_PRICES_CANDIDATE_VALIDATION as VCFG,
    lazy_prices_candidate_validation_hash,
)

PANEL_PATH = "apps/quant/advisor/tests/fixtures/floor_prices.csv"
FIXTURE_PATH = "apps/quant/advisor/tests/fixtures/lazy_prices_similarity.csv"

panel = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)
records = load_fixture(FIXTURE_PATH)
panel_lp = build_lazy_prices_panel(records, panel, warmup=CFG.warmup)  # assets default = non-SPY

res = lazy_prices_candidate_metrics(panel, panel_lp, CFG, prereg_hash=None)  # HOLDOUT BLINDED
corr = dev_lazy_momentum_corr(panel, panel_lp, warmup=CFG.warmup, holdout_frac=0.2)
disp = dev_cross_sectional_dispersion(panel, panel_lp, warmup=CFG.warmup, holdout_frac=0.2)

print(json.dumps(res, indent=2, default=str))
print("=== WS3D DECISION INPUTS ===")
print("dev.passed         :", res["dev"]["passed"])
print("dev.reasons        :", res["dev"]["reasons"])
print("dev.fold_deltas    :", res["dev"]["fold_deltas"])
print("verdict (blinded)  :", res["verdict"], "(reads DEV_FAILED by construction when holdout blinded)")
print("holdout            :", res["holdout"], "(MUST be null with prereg_hash=None)")
print("weights            :", res["weights"])
print("ensemble           :", res["ensemble"])
print("best_family        :", res["best_family"])
print("spy (dev-window)   :", res["spy"])
print("power.power_limited :", res["power"]["power_limited"])
print("power.folds         :", res["power"]["folds"])
print("validation          :", json.dumps(res["validation"], default=str))
print("dev_lazy_momentum_corr        :", corr)
print("dev_cross_sectional_dispersion:", disp)
print("=== PINNED HASHES ===")
print("lazy_prices_candidate_hash            :", lazy_prices_candidate_hash(CFG))
print("lazy_prices_candidate_validation_hash :", lazy_prices_candidate_validation_hash(VCFG))
print("lazy_prices_candidate_run_hash        :", lazy_prices_candidate_run_hash(CFG, FIXTURE_PATH))

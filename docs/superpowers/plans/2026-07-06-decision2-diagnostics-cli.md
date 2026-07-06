# Plan: Decision 2 — Operator portfolio diagnostics CLI

Date: 2026-07-06
Branch: exec/decision2-diagnostics-cli
Status: FROZEN before implementation (this commit precedes all code)
Sources: ~/.gstack/projects/AIHedgeFund/nikhi-main-design-20260706-082720.md (design),
nikhi-main-eng-review-test-plan-20260706-082720.md (test plan). Eng-reviewed +
outside-voice-hardened (12 findings incl. 1 P1; 3 decisions flipped on re-ask; 0
unresolved). Operator formally picked Decision 2 on 2026-07-06. Reviewed decisions are
pinned below and are NOT to be relitigated during implementation.

## KILL CRITERION (frozen — evaluate before reading any usage data)

Evaluate +60 days from merge (if merged 2026-07-06, evaluation date = 2026-09-04):
**if the operator ran it fewer than 8 times OR it never informed an actual decision,
close the lane and record the negative** like any other. Run count is self-reported
(shell history); v1 deliberately has no usage telemetry — the report-only guarantee
means no writes beyond requested output.

## Problem

Eight research negatives; floor is DEV_FAILED and stays that way. The one value
surface not gated on a signal claim is the report-only diagnostics stack
(`backtest/stats.py`, `backtest/concentration.py`). Missing: a way to run it on the
operator's real positions from local files — deterministic, explained, zero signal
claim, zero contact with the research gates.

## Pinned decisions (verbatim; reviewed — do not relitigate)

- **qty-only positions CSV** (ticker, qty); weights-only input REJECTED with clear
  error (a static-weights book is constant-mix, a different strategy whose risk
  numbers do not describe the held portfolio); shorts (negative qty) rejected;
  optional CASH row (qty=dollars, price≡1.0), else report labeled "invested sleeve
  only".
- **prices.csv MUST be split+dividend-adjusted** (total-return basis); hard error on
  any single-name single-day move >40% (phantom-split guard, tested via split
  fixture); sanctioned production path = existing `advisor/data/price_fetch.py`
  (keyless yfinance, auto_adjust) as a SEPARATE documented step; the CLI itself makes
  NO network calls.
- **Missing data: common-trading-days INTERSECTION window** (no imputation ever),
  dropped-date count reported; partial-history ticker (leading NaNs) = hard error
  naming it; unknown ticker in positions but not prices = hard error; median date-gap
  must be 1-4 calendar days else hard error (252 annualization is hardcoded in
  stats.py:7,15,24).
- **Metrics via read-only imports** of `backtest.stats` + `backtest.concentration`
  (parity-tested); concentration = NUMBERS ONLY (never surface
  `passes_concentration` thresholds); banner extends
  `walk_forward.disclosure_header` with report-only + floor-DEV_FAILED + ETF caveat
  + basis statement, present in text AND JSON.
- **Bootstrap LCB**: module constants seed=42, block=21, draws=1000 recorded in
  report metadata; advisory line when n<250 ("window under one year; LCB unstable");
  wording = realized-history resample, no forward claim.
- **Sentinels never print 0.00**: Sortino "n/a — no downside observed"; LCB "n/a —
  window < block"; book_sharpe "n/a — zero variance".
- **Determinism**: byte-identical reruns on Windows — pin newline='\n', UTF-8,
  explicit float format in text and JSON.
- **Deliberate DRY exception**: pct_change one-liner duplicated in portfolio.py
  (commented intentional) rather than editing `backtest/**`.
- **Invocation reality**: v1 runs as `python -m advisor.diagnostics` with
  `apps/quant` on PYTHONPATH; packaging out of scope.

## Interface contract (pinned BEFORE parallel dispatch; both lanes code to this)

New package `apps/quant/advisor/diagnostics/` (NOT under `backtest/**`).

```python
# apps/quant/advisor/diagnostics/portfolio.py  (Lane A)
@dataclass(frozen=True)
class LoadedPortfolio:
    returns: pd.Series          # daily pct_change of total equity curve, dropna'd
    weights_book: pd.DataFrame  # rows=intersection dates, cols=non-CASH tickers
    n_obs: int                  # len(returns)
    dropped_dates: int          # union-minus-intersection date count within [start, end]
    tickers: list[str]          # non-CASH tickers, sorted
    cash_dollars: float | None  # None when no CASH row
    start: date                 # first common observation date
    end: date                   # last common observation date

def load_portfolio(positions_path, prices_path) -> LoadedPortfolio: ...

class DiagnosticsInputError(ValueError): ...  # all guard failures; message names ticker
```

Semantics (pinned):

- positions.csv: header `ticker,qty` (case-insensitive). A `weight`/`weights` column
  or missing `qty` column → weights-rejection error (by column name, no magnitude
  heuristics). qty ≤ 0 → hard error (negative named as shorts-rejected). Duplicate
  ticker rows → hard error. `CASH` (case-insensitive) = optional cash row, qty is
  dollars at price ≡ 1.0.
- prices.csv: floor_prices-shaped — `Date` first column (ISO dates), one column per
  ticker, values = split+dividend-adjusted closes (the price_fetch.py output shape).
- Equity curve: qty × adjusted close per intersection date, summed; cash (if
  present) adds a constant `cash_dollars` to total value. `returns` =
  `equity.pct_change().dropna()` (the deliberate DRY exception, commented).
- weights_book: per-date (qty × px) / total book value (total INCLUDES cash when
  present, so invested weights sum < 1 with cash; without cash, denominator =
  invested value and the report is labeled "invested sleeve only").
- Guard order: parse errors → weights-rejection → non-positive qty → unknown ticker
  → partial history (ticker whose data does not span the intersection window) →
  empty window → frequency (median calendar-day gap of consecutive dates must be in
  [1, 4]) → jump guard (any single-name single-day abs move > `MAX_DAILY_MOVE =
  0.40` on adjusted prices → phantom-split hard error naming ticker + date).

```python
# apps/quant/advisor/diagnostics/report.py  (Lane B)
BOOTSTRAP_SEED = 42
BOOTSTRAP_BLOCK = 21
BOOTSTRAP_DRAWS = 1000

def build_report(lp: LoadedPortfolio) -> dict: ...   # strict-JSON-serializable
def render_text(report: dict) -> str: ...            # '\n' newlines only
def render_json(report: dict) -> str: ...            # json.dumps(..., indent=2, sort_keys=True) + '\n'
```

- Metrics: `book_sharpe`, `sortino`, `downside_deviation`, `max_drawdown`,
  `block_bootstrap_lcb` from `advisor.backtest.stats`; concentration numbers from
  `advisor.backtest.concentration.concentration_report` (never
  `passes_concentration`).
- Sentinels: report layer re-derives the degeneracy conditions from `lp.returns`
  (NOT by comparing outputs to 0.0): zero variance → book_sharpe sentinel; no
  downside observations → Sortino sentinel; n_obs < BOOTSTRAP_BLOCK → LCB sentinel.
  Sentinel text exactly as pinned above; JSON carries `null` + a `"note"` string.
- Banner: `disclosure_header()` from `advisor.backtest.walk_forward` PLUS
  diagnostics lines: report-only (no signal/direction/sizing), floor is DEV_FAILED
  (advisor has no validated alpha), ETF caveat (a one-ETF book prints
  max_single_name = 1.0 by construction), basis statement (inputs assumed
  split+dividend-adjusted total-return prices). Present in text AND in JSON
  (`"disclosures"` array).
- Floats: text ratios `{:.4f}`, dollar values `{:,.2f}`; JSON floats rounded to 6
  decimal places. Bootstrap metadata block (seed/block/draws/n_obs) always present.
- n_obs < 250 → advisory line "window under one year; LCB unstable" (text + JSON).

```python
# apps/quant/advisor/diagnostics/__main__.py  (Lane B)
# python -m advisor.diagnostics --positions X.csv --prices Y.csv [--json] [--out PATH]
```

- Exit 0 on success; exit 2 on DiagnosticsInputError / missing file, message on
  stderr. `--out` writes with `open(..., "w", encoding="utf-8", newline="\n")`;
  default output to stdout. No other writes, no network.

## Lane split (disjoint files; parallel Hermes dispatch)

Lane A — loader + guards + oracle:
- `apps/quant/advisor/diagnostics/__init__.py`
- `apps/quant/advisor/diagnostics/portfolio.py`
- `apps/quant/advisor/tests/test_diag_portfolio.py` (13 paths)
- `apps/quant/advisor/tests/fixtures/diag_positions.csv`
- `apps/quant/advisor/tests/fixtures/diag_prices_adjusted.csv` (3-ticker
  hand-computed oracle, in-window split ON ADJUSTED BASIS = no jump)
- `apps/quant/advisor/tests/fixtures/diag_prices_unadjusted_split.csv` (same split
  left raw → trips the 40% jump guard)

Lane B — report + CLI + report-only proof (does NOT create `__init__.py`; codes
against the contract above; builds test CSVs in tmp_path, owns no fixture files):
- `apps/quant/advisor/diagnostics/report.py`
- `apps/quant/advisor/diagnostics/__main__.py`
- `apps/quant/advisor/tests/test_diag_report.py` (9 paths)
- `apps/quant/advisor/tests/test_diag_cli.py` (3 paths)
- `apps/quant/advisor/tests/test_diag_report_only.py` (2 paths)

## Acceptance tests (~27 paths, 4 files)

- `test_diag_portfolio.py` — qty parsing; weights-input rejected; shorts rejected;
  CASH row; malformed/empty CSV; unknown ticker error; intersection window +
  dropped-count; partial-history error; frequency error (weekly data); jump-guard
  error (split fixture); single name; equity-curve oracle (hand-computed 3-ticker
  fixture with in-window split on adjusted prices); duplicate-ticker error.
- `test_diag_report.py` — metric parity vs direct stats/concentration calls; all
  three sentinel branches; banner (text + JSON); basis statement; bootstrap metadata
  + min-n advisory; determinism byte-identical (Windows newline/encoding pinned);
  strict-JSON parse.
- `test_diag_cli.py` — subprocess E2E: happy path exit 0; missing file exit != 0
  with clear message; --json.
- `test_diag_report_only.py` — importing the diagnostics package pulls no network
  providers (no `advisor.data.*`, no `yfinance`/`requests` newly in sys.modules); no
  writes beyond requested output.

## Postflight (standing rails)

- Full suite green: `node tools/run-pytest.mjs apps/quant/advisor/tests` (361 + ~27)
- `node tools/run-floor.mjs --enforce` exit 1 (floor stays DEV_FAILED)
- Verdict pins 0.7323/0.7562/0.8277 byte-identical
- ZERO diffs under `apps/quant/advisor/backtest/**` (and pipeline/run.py, prereg
  surfaces, floor fixtures)
- Secret scan clean; `.claude/settings.local.json` never committed
- PR to main; operator merges (self-merge classifier-blocked)

## NOT in scope (v1)

Weights/constant-mix input mode (TODO); short or leveraged books (TODO gated on
Decision 5); broker-export ingestion (TODO); live/keyed data in the CLI; any signal,
direction, score, or sizing output; any change under `backtest/**`, `pipeline/run.py`
(P2 tripwire coupling untouched), prereg surfaces, or gates; packaging/distribution.

## Success criteria

Operator runs it on his real positions and gets a correct, explained, deterministic
risk report on a stated total-return basis. Suite grows ~27 tests. Research record
and all rails unchanged. Lane survives its own +60-day kill criterion or closes
honestly.

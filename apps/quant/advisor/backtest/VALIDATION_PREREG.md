# Validation Gate Pre-Registration

Separate immutable surface from PREREG.md (which records the floor config hash
`1ad2ed4a...`). This file pre-registers the deflation gate parameters.

- validation_hash(DEFAULT_VALIDATION) = 5f5254d3025a0ab3e4de3151825e77d1aee813cf69d802250b82a8811cb4ca8b
- DSR pass bar: 0.95 (PSR at the multiple-testing-deflated benchmark).
- Sharpe units: PER-OBSERVATION (mean/std). `book_sharpe` (annualized x sqrt(252)) is NOT used in the gate.
- Declared multiple-testing N: 45 = the MinBTL budget ceiling. Conservative by
  construction (larger N -> stricter), chosen over fragile reconstruction of the
  exact families x weights x constructions (C/D/E) x versions (v1/v2) trial count.
- Effective-N method: PCA participation ratio over candidate family OOS return
  series. GUARDRAIL: n_used = max(declared_N, ceil(effective_N)); effective-N may
  never lower the penalty below the declared integer without a new pre-registration.
  Dormant-as-a-count for the current floor (family set <= 5 << 45) BY DESIGN -- the
  PCA machinery is built/tested for forward use when a candidate set exceeds the
  declared meta-count. (The family Sharpes still feed the var_sr dispersion channel.)
- V[SR_hat] for SR0 = declared_var_sr, a PRE-REGISTERED CONSTANT (NOT the single-strategy
  sampling variance, which biases the gate lenient; NOT a live estimate off 2 series, which
  is noise). It is the gate's dominant leniency knob: SR0 ∝ sqrt(var_sr), higher = stricter.
  Calibrated value: 1.000000e-04 (declared_var_sr = 1e-4). Method: sample variance (ddof=1)
  of the per-obs Sharpes of every trial actually run (each family standalone + each C/D/E
  construction) on the dev portion of the fixture. Recalibration requires a new pre-registration.

  Calibration record (output of `python tools/calibrate_var_sr.py`, 2026-06-16):
  ```
  trial_count = 5
  trial_sharpes (per-obs) = {'A': 0.05214009785475365, 'B': 0.03893832858540658, 'C': 0.046130084850232234, 'D': 0.03746837072145384, 'E': 0.03730743671155317}
  var_sr_trials (measured) = 4.275533e-05
  declared_var_sr (rounded up) = 1.000000e-04
  ```
  Caveat: the estimate is from a 5-trial book (A-E), a small sample for a variance, and is
  frozen by this pre-registration. Rounding UP to the next 1e-4 errs strict (SR0 ∝ sqrt(var_sr)).
  The measured 4.28e-05 rounds up to 1e-4; the interim 4e-4 was an uncalibrated overestimate and
  has been replaced. Recalibration (e.g. a larger trial book) requires a new pre-registration.
- HLZ t-stat hurdle: 3.0 (applies only where a factor/signal selection claim is made).
- PBO/CSCV: deferred (audit-only, medium-confidence, synthetic-only; not built in this slice).
- Gross-vs-net: the floor already nets costs (cost_per_turn=0.0005, turnover_cap=0.20);
  gross-vs-net rejection deferred to a future high-turnover candidate.
- Posture: report-only. Cannot flip DEV_FAILED->PASSED, unlock the holdout, or authorize sizing.

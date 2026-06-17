# Lane B Candidate-Search Plan — Debate Findings & Verdicts

**Artifact under test:** `docs/superpowers/plans/2026-06-16-lane-b-candidate-search.md` (commit `9145249`, `main`, not pushed).
**Method:** Hermes `--workflow debate --live` over `ai-logs/hermes/lane-b-stress-test.md` (8 seeded attack surfaces, SETTLED rails fenced off). Run `hermes-2026-06-17T05-08-36-461Z`.
**Who carried it:** the **codex** leg produced a complete, self-de-duped, severity-ranked review (7 findings); the kimi/claude downstream legs failed (kimi cp1252 + `windows sandbox: spawn setup refresh`, the known Windows fragility — `hermes-dispatch-windows`). The wrapper exited 1 *after* `npm run check` passed **160/160** — a cosmetic ceremony failure, not a content failure (plan4 precedent). Abstentions/sandbox errors triaged as non-findings, not refutations.
**Corroboration:** every finding was independently re-derived and **verified against the actual frozen code** (`pipeline.py`, `data_floor.py`, `validation.py`, `validation_prereg.py`, `splits.py`, `continuous_signals.py`, `dev_gate.py`, `tools/floor_data_check.py`, `FLOOR_RESULT.md`). No finding contradicted the implementation; all 7 are accepted (one with a refinement). The promotion-transfer surface (seed #6) is **not** a finding — Task 9 already handles it correctly (recorded below).

Triage rule honored: a finding is accepted only if it survives a check against the real code; the verification (with file:line) is recorded per finding.

---

## Severity-ranked, de-duped findings

| # | Sev | Finding | Plan hook | Verdict |
|---|-----|---------|-----------|---------|
| F1 | **Critical** | Candidate DSR prereg fields are dead — `validation_report` ignores `CandidatePreReg` | rail #4, Tasks 7–8 | ACCEPT (refined) |
| F2 | **Critical** | Shared holdout reuse uncontrolled; candidate hash omits fixture bytes; no global touch ledger | rail #5, Tasks 8–9 | ACCEPT |
| F3 | **High** | Golden replication too weak — aggregate `abs=0.01`, doesn't exercise the value path | Task 4 | ACCEPT (merges seed #1+#8) |
| F4 | **High** | Orthogonality measured on raw, not the post-transform conviction surface that enters the blend | Tasks 5–6 | ACCEPT |
| F5 | **Medium** | Build order doesn't fail-fast — expensive mirror (Tasks 3–4) built before the cheap kill-gate (Task 6) | Phase B0/B1 order | ACCEPT (merges seed #3+#7 build half) |
| F6 | **Medium** | "Clean negative" overclaims power — `>=10` positive points is too weak a sufficiency bar | design note, Tasks 4/8 | ACCEPT (seed #7 power half) |
| F7 | **Low** | Fixture-loader placeholders point at a non-existent module + wrong path | Task 4 Step 1, Tasks 6/8 | ACCEPT |
| — | n/a | Promotion transfer (seed #6) | Task 9 | NOT A FINDING — already handled |

---

## F1 — CRITICAL: Candidate DSR pre-registration fields are dead

**Codex claim:** the plan freezes `CandidatePreReg.declared_trials_N`, but Task 7's `candidate_metrics` calls `validation_report(..., DEFAULT_VALIDATION)`, which reads `vcfg.declared_trials_N` / `vcfg.declared_var_sr` — so candidate N/var_sr changes are ignored. Rail #4's "each additional family-set increments `declared_trials_N` and re-runs DSR at the higher N" is therefore unimplementable.

**Verification (frozen code):**
- `validation.py:117-130` — `validation_report(candidate_returns, family_returns, vcfg, tstat=None)` reads `var_sr = vcfg.declared_var_sr` (line 123) and `n_used = n_for_dsr(family_returns, vcfg.declared_trials_N)` (line 126). It NEVER sees `CandidatePreReg`.
- Plan Task 7 Step 3 (`candidate_floor.py`) passes `DEFAULT_VALIDATION` (plan lines 766-771), i.e. the FLOOR's `declared_trials_N=45`, `declared_var_sr=1e-4` (`validation_prereg.py:14-25`).
- `CandidatePreReg.declared_trials_N=45` (plan line 120) is read by NOTHING in the mirror → **dead field**. Incrementing it (rail #4, Task 8 secondary run) changes the candidate hash but does nothing to DSR. **Confirmed unimplementable.**
- `var_sr=1e-4` was calibrated from the floor's A–E trial Sharpes (`validation-gate-floor-internals`); a value+momentum book has a different return distribution → reuse is unjustified-by-default.

**Refinement (verified nuance, against codex's fix):** codex also suggested "make candidate pass status require DSR." That would DEVIATE from floor-faithfulness: the floor's validation is **report-only** — `data_floor.py:74` sets `"passes": verdict == "PASSED"` and the `"validation"` key never mutates the verdict (`validation.py:121` "Report-only: never mutates verdict"). To mirror the floor honestly, the bench must ALSO keep `passes` independent of DSR; DSR-confirmation stays the **operator promotion-readiness gate** (Task 8 Step 3 / Task 9), exactly as the floor treats it. So: wire candidate N/var_sr through; do NOT fold DSR into the machine `passes`.

**Fix applied:** add an immutable `CandidateValidationPreReg` (own hash surface) whose `declared_trials_N` / `declared_var_sr` ARE what `candidate_metrics` passes to `validation_report`; recalibrate `declared_var_sr` from the candidate's own per-obs trial Sharpes (or explicitly justify reusing 1e-4 as conservative-and-stricter); a unit test asserting a change to candidate N changes `validation["n_used"]`; rail #4 re-points "increment N" at the new surface; validation stays report-only.

---

## F2 — CRITICAL: Shared holdout reuse is not globally controlled

**Codex claim:** the candidate bench reuses the floor's reserved tail under a separate prereg with no global touch ledger; `candidate_metrics(..., prereg_hash=...)` unlocks the holdout on ANY non-None hash; `candidate_hash()` excludes fixture bytes.

**Verification (frozen code) — confirmed, and the floor already learned this exact lesson:**
- `FLOOR_RESULT.md:32` — "The holdout … remains untouched and is reserved; do NOT run it without a dev-passing candidate." The tail IS the floor's reserved holdout.
- `floor_data_check.py:63-71` — explicit prior hardening ("Holdout integrity (debate finding #1): the held-out tail is unlocked ONLY by the content hash of the actual (config + fixture) — never an arbitrary string"). The floor unlocks via `config_hash(DEFAULT_CONFIG, FIXTURE)` which **includes fixture bytes** (`prereg.py` config_hash SHA-256s asdict(cfg)+fixture bytes, per `validation-gate-floor-internals`).
- Plan `candidate_hash()` (lines 145-150): "SHA-256 over the canonical config JSON (no fixture bytes…)". → the candidate is **strictly weaker** than the floor on the precise dimension the floor already hardened: it cannot detect a fixture swap, and `prereg_hash is not None` (plan line 745, mirroring `data_floor.py:34`) unlocks the holdout on any truthy hash.
- No ledger anywhere records that Reading A, a future Reading B, and the floor's own future `--holdout` run all draw from ONE shared reserved tail → uncontrolled multiple-testing on the holdout.

**Fix applied:** (1) candidate run-hash over config **+ fixture bytes** (mirror `config_hash(cfg, FIXTURE)` discipline) — new rail; (2) a shared **holdout-touch ledger** (`research/HOLDOUT_LEDGER.md`) appended-to on every reserved-tail evaluation (floor or any candidate), recording date/hash/verdict, so each look is counted; (3) explicit statement (Task 9 + rail #5) that ANY side-bench holdout touch burns the shared tail and **promotion requires a fresh holdout**, not the already-peeked tail.

---

## F3 — HIGH: Golden replication is too weak (merges seeds #1 + #8)

**Codex claim:** asserting aggregate 0.732/0.828 at `abs=0.01` can miss element-level mirror drift, conflates intentional floor changes with mirror bugs, and never exercises the value empty-`pos` transform path.

**Verification (frozen code):**
- Plan Task 4 (lines 486-495) asserts only `book_sharpe(...) == approx(0.732/0.828, abs=0.01)` on `("momentum","trend")`. `abs=0.01` on a 0.732 Sharpe is ~1.4% relative — a one-line mirror divergence can hide under it.
- The golden families are NaN-short (momentum `shift(126)`, trend `rolling(200)`); the value path (270-row NaN prefix → possibly tiny/empty `pos` in `fit_percentile_transform`, `continuous_signals.py:26-31`) is NEVER exercised by the replication.
- A hardcoded 0.732 also can't tell a legit floor change from a mirror bug.

**Verified caution (added):** an element-wise equality check on the *dev sweep* is holdout-safe (`run_dev_sweep` only touches `dev`, `pipeline.py:46-49`), but an equality check that calls `run_holdout_ext` vs `run_holdout` WOULD touch the real reserved tail — so the holdout-mirror equality must run on **synthetic** data only (ties to F2).

**Fix applied:** replace/augment Task 4 with element-wise equality — `run_dev_sweep_ext(panel, fams, raw_fn=raw_metric)` vs frozen `run_dev_sweep(panel, fams)` asserting equal `fold_deltas`, `ensemble_test_returns`, `best_family_test_returns`, `chosen_weights` (true per-change drift guard); holdout-mirror equality on synthetic only; a targeted empty-`pos` value-transform test (degenerate fit → flat, no crash).

---

## F4 — HIGH: Orthogonality is measured on the wrong surface

**Codex claim:** construction uses transformed long-flat conviction scores (negatives/NaN → flat); raw pooled Pearson across heterogeneous assets can disagree with what actually enters the blend. Momentum is diagnostic-only despite being the blend partner.

**Verification (frozen code):**
- `pipeline.py:67` blends `chosen[f] * scores[f]` where `scores` are `apply_transform` outputs (`continuous_signals.py:34-45`): empirical-CDF percentile of POSITIVE raw, everything ≤0 → 0. So the blend's diversification lives in the **transformed** scores, not the raw.
- Plan Task 5 (`orthogonality.py`, lines 587-611) correlates **raw** `candidate_raw` series via `np.corrcoef` (Pearson), pooled across assets. Two signals that are both flat (≤0→0) on the same days can have low raw corr but high post-transform co-flatness, or vice versa → the gate can mis-measure the diversification §7.2 rewards.
- The plan itself concedes "§7.2 ultimately adjudicates the blend" (line 641) yet calls orthogonality "the pivot."

**Fix applied:** keep raw Pearson as a pre-registered **diagnostic**, but additionally gate/report the **post-transform fold-level** correlation (reuse `fit_percentile_transform`/`apply_transform`, dev-train-fit per fold, holdout excluded); add **Spearman/rank** corr as a scale-robust cross-check; write an explicit `momentum` decision rule (diagnostic threshold that, if breached, demotes the gate to "coarse pre-filter — §7.2 is the real test" rather than a clean PASS).

---

## F5 — MEDIUM: Build order should fail-fast (merges seed #3 + #7-build)

**Codex claim:** Task 6 only needs `CandidatePreReg`, `candidate_raw`, `purged_splits`; the plan admits the value horizon sits near `long_momentum` and may die cheaply — so building the whole mirror + golden first is avoidable.

**Verification (frozen code):**
- Plan Task 5/6 import only `splits.purged_splits` + `candidate_signals.candidate_raw` + `candidate_prereg` — they do NOT depend on `candidate_pipeline` (Task 3) or `candidate_floor` (Task 7). Confirmed: the kill-gate is reachable after Tasks 1+2 alone.
- The plan's own Design-decision note (lines 29-31) says value@270 "sits near `long_momentum`, so the orthogonality kill-gate becomes the genuine pivot" and the cheap kill is "the *intended* … outcome" (line 643). So the expensive B2 infra is, by the plan's own thesis, the less-likely-to-be-reached path.

**Fix applied:** reorder the build to **Task 1 → Task 2 → Task 5 → Task 6**; only if orthogonality PASSES, build Tasks 3/4/7/8. Keep Task 4 (golden) **mandatory before any candidate holdout evaluation** (i.e., before Task 8). Update the Execution-handoff section to match.

---

## F6 — MEDIUM: "Clean negative" overclaims power

**Codex claim:** with `purged_splits`, fold-1 train length ≈ `fold_size - embargo`; `value_lookback=270` leaves a small live training tail; the `>=10` positive-raw guard isn't enough to prove stable percentile fitting under the dev gate's median/70%/LCB/lift bars.

**Verification (frozen code):**
- `splits.py:11-25` — `fold_size = n // folds`; fold-1 `train_end = test_start - embargo = fold_size - embargo`. On dev≈1654, folds=5 → fold_size=330, fold-1 train = rows `[0, 325)`.
- `value` is NaN for its first `value_lookback=270` rows → fold-1 has only ~55 candidate rows, of which positive ≈ half ≈ ~25 feeding `fit_percentile_transform` (`continuous_signals.py:26-31`). Plan Task 4 guard asserts only `>=10` (line 512).
- `dev_gate.py:18-41` requires median Δ>0 AND ≥70% folds positive AND 90% bootstrap LCB>0 AND total lift ≥0.05 — a fit on ~10–25 points is too noisy to clear these reliably, so a `DEV_FAILED` can be a power artifact, not a signal verdict. The plan's Task 8 caveat (line 821) calls 270 "a genuine signal verdict, not a rigged one" — that overclaims given the thin fit.

**Fix applied:** add a pre-run **power/sufficiency report** (per-fold positive-raw count, nonzero-transformed coverage fraction, effective observations) emitted alongside the verdict; if coverage is below a pre-registered threshold, the verdict is labeled **power-limited / inconclusive**, NOT "Reading A exhausted." Soften the Task 8 caveat accordingly.

---

## F7 — LOW: Fixture-loader placeholders point at a non-existent module + wrong path

**Codex claim:** the real fixture loader is script-local in `tools/floor_data_check.py`; there is no `advisor.backtest.<floor_fixture_module>.<load_panel>` to import; placeholder commands aren't executable.

**Verification (frozen code):**
- Glob: the fixture is at `apps/quant/advisor/tests/fixtures/floor_prices.csv` — NOT the plan's stated `apps/quant/advisor/backtest/fixtures/floor_prices.csv` (plan line 466). **Path is wrong.**
- `floor_data_check.py:12,62` — `FIXTURE = Path("apps/quant/advisor/tests/fixtures/floor_prices.csv")`; loaded by `pd.read_csv(FIXTURE, index_col=0, parse_dates=True)`. The loader is **script-local**, not an importable module. The plan's `from advisor.backtest.<floor_fixture_module> import <load_panel>` (lines 484, 504) resolves to nothing; the inline `-c` snippets (Tasks 6/8, lines 636/814) import the same non-existent module. Blocks Tasks 4/6/8 as written.

**Fix applied:** replace every `<floor_fixture_module>`/`<load_panel>` placeholder with the concrete loader `pd.read_csv(Path("apps/quant/advisor/tests/fixtures/floor_prices.csv"), index_col=0, parse_dates=True)` (or a one-line shared `research`/test helper `load_floor_panel()` wrapping it), and correct the fixture path in the Task 4 file list. Resolves the Self-review "placeholder scan" honestly.

---

## NOT A FINDING — Promotion transfer (seed #6)

Codex explicitly declined to file this, and the code agrees: Task 9 (plan lines 834-840) states a PASSED bench result does **not** promote into the frozen floor without (1) a new plan, (2) a new floor pre-registration (re-hash, new `PREREG.md`), and (3) operator sign-off; `run-floor.mjs --enforce` keeps running momentum/trend on `DEFAULT_CONFIG` and stays exit 1 regardless of the bench verdict. The only sharpening (folded into F2's fix) is that re-pre-registration must use a **fresh holdout**, since the side bench burns the shared reserved tail.

# Skill-Weight Live Seam Activation — Implementation Spec

**Status:** REVIEWED, READY TO IMPLEMENT. CEO review (HOLD SCOPE) + eng review both CLEAN 2026-07-06 at main `58fcef6`; Codex adversarial pass ran twice (8 + 2 findings, all adjudicated). All decisions below are operator-locked — do NOT re-litigate them; implement.

**Goal:** `ensemble_vote()` honors `FamilySignal.skill_weight` (`effective = confidence * skill_weight` for both signed score and total weight). Return contract `(Direction, float)` unchanged. `weight == 0 or score == 0 → (NEUTRAL, 50.0)` unchanged. Behavior is provably identical while all weights are 1.0 (all five producers in `apps/quant/advisor/analysis/*.py` use the default; confidence formula `min(100, 50 + |score|/weight*50)` is scale-invariant under uniform weights).

**Hard constraints:** No schema changes. No backtest/floor/holdout changes (`node tools/run-floor.mjs --enforce` must still exit 1). `test_rails.py:42-46` (backtest never imports the allocator) must keep passing. Preserve unrelated dirt in `.claude/settings.local.json`.

## Sequencing note (do step 1 first)

`.claude/hooks/guard-frozen-floor.mjs:13` currently hard-blocks (exit 2) Edit/Write to `allocator.py`. The hook file itself is NOT frozen. Land step 1 before touching the allocator, or every seam edit bounces.

## Changes

1. **Guard** — `.claude/hooks/guard-frozen-floor.mjs`
   - Remove `/advisor/portfolio/allocator.py` from the FROZEN regex (line 13). Keep `floor_prices.csv`, `PREREG.md`, `UNIVERSE_RULE.md` frozen.
   - Rewrite the lines 1-2 comment: it must name exactly what remains frozen and no longer claim to protect "the live ensemble_vote seam". Permanent unfreeze — no restore commit.

2. **Live seam** — `apps/quant/advisor/portfolio/allocator.py`
   - In `ensemble_vote()` (lines 16-30): per signal, `w = s.confidence * s.skill_weight`; BULLISH adds `w` to score, BEARISH subtracts `w`; `weight += w` unconditionally (NEUTRAL-direction signals keep diluting the denominator, same as the oracle). Neutral branch and confidence formula unchanged in shape.
   - Rewrite the docstring (currently "Equal-weight vote across families" — false post-fix): state the weighting AND the pact: must remain equivalent to `parity.skill_weighted_vote`; the parity test suite enforces agreement. (Terminology: "independent mirror", NOT "byte-equivalence".)

3. **Policy tripwire** — `apps/quant/advisor/pipeline/run.py`
   - After bundle construction (line 29), before `ensemble_vote`:
     `if len({s.skill_weight for s in bundle.signals}) > 1: raise ValueError(...)`.
   - Message must say non-uniform skill weights are rejected until a validated calibration exists, citing `docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md` §8.
   - Rationale (locked): spec §8 forbids weighting on unmeasured skill; this covers every live vote entry point (`cli.py:57` routes `--families all` → `run_all` → `run_pipeline`; the `--families value` path never votes). Uniform weights pass (any uniform scale is provably identical math). Empty set (empty bundle) passes. Plain raise — no friendly CLI wrapper.

4. **Oracle** — `apps/quant/advisor/portfolio/parity.py`
   - Do NOT import it from allocator (circular: `parity.py:3` imports `ensemble_vote`). Keep `skill_weighted_vote` as the independent mirror.
   - Update both docstrings: "documented live gap" → parity/audit helper for the live weighted seam; `vote_parity`'s "guaranteed to agree only when uniform" is false post-fix — post-fix the seams must ALWAYS agree; any divergence is a regression signal. Add one durable policy sentence: non-uniform weights require validated skill estimates; none exist as of 2026-07.

5. **Tests** — `apps/quant/advisor/tests/test_allocator_parity.py` (+ `test_pipeline.py`, `test_schemas.py` where noted)
   - Remove the `xfail` from `test_live_seam_honors_skill_weight` (retires the suite's tracked "+1 x").
   - Rework `test_nonuniform_flip_setup_is_a_real_divergence` (lines 74-81): it currently asserts live BULLISH and `direction_match is False` — both false post-fix. New form: assert the flip fixture would be bullish under raw equal voting via an inline signed-confidence sum (do NOT resurrect a raw helper), is BEARISH through the live seam, and `vote_parity` now matches.
   - Flip-bundle allocation assertion: `allocate(flip, price=100.0, position_limit_dollars=25_000.0)` → action `"sell"`, quantity > 0.
   - Parametrized equality `ensemble_vote(b) == skill_weighted_vote(b)` across ALL fixture bundles (UNIFORM_BUNDLES + flip + mute). This is the oracle pact's enforcement.
   - Mute case (spec precision — locked): the zero-weight signal MUST be directly constructed (`FamilySignal(..., skill_weight=0.0)`); `FamilySignal.neutral()` has NO skill_weight param (always 1.0). Mixed bundle = one live signal (w=1) + one directly-constructed w=0 signal + one separate `FamilySignal.neutral()` member. Assert: w=0 signal fully excluded (numerator AND denominator — this is the behavioral delta vs old dilution); neutral() member dilutes denominator only; live == oracle.
   - All-zero weights → `(NEUTRAL, 50.0)` at the live seam.
   - Schema rejection: negative and NaN `skill_weight` raise ValidationError (`ge=0`; NaN fails the ge comparison).
   - Tripwire tests (test_pipeline.py): `pytest.raises(ValueError)` via `run_pipeline` with a coro returning a non-default-weight signal; uniform bundle passes end-to-end.
   - Pre-existing guard branches in `allocate()` (file is in-diff, so cover them): `price<=0` → hold; `position_limit_dollars<=0` → hold; conviction rounding to 0 shares → hold.

6. **Docs** — `apps/quant/advisor/backtest/FLOOR_RESULT.md`
   - Line 12 says the live seam "still ignores `skill_weight`" — false post-merge and NOT pinned by `test_docs_truth.py`. Fix with a dated APPEND-ONLY addendum under the Decision section (do not rewrite the 2026-06-16 record text): "Update 2026-07: the live seam now honors `skill_weight`; `run_pipeline` rejects non-uniform weights until spec-§8 calibration exists." File is not hook-frozen.

7. **Include** the untracked `TODOS.md` (repo root, created by review) in the PR — P2 weight observability (must land whenever the tripwire is relaxed), P3 skill_weight upper bound (inf admissible today; schema change deliberately out of this PR's scope).

## Acceptance

```powershell
python -m pytest apps/quant/advisor/tests/test_allocator_parity.py -q       # zero xfail remaining
python -m pytest apps/quant/advisor/tests/test_allocator.py apps/quant/advisor/tests/test_pipeline.py apps/quant/advisor/tests/test_cli_all.py -q
python -m pytest apps/quant/advisor/tests/test_rails.py -q
python -m pytest apps/quant/advisor/tests -q                                 # full advisor suite green
node tools/run-floor.mjs --enforce                                           # MUST still exit 1
```

Manual: after step 1, Edit on `allocator.py` is no longer hook-blocked while `PREREG.md`/`floor_prices.csv`/`UNIVERSE_RULE.md` still are. CLI fixtures (`--families all`) byte-identical (producers all default 1.0).

Companion QA artifact: `~/.gstack/projects/AIHedgeFund/nikhi-main-eng-review-test-plan-20260706.md`.

## Review record

CEO review: HOLD SCOPE, 0 critical gaps, guard/approach/mode operator-decided. Eng review: clean, tripwire placement verified, coverage 100% of new/modified paths. Codex pass 1 (8 findings): tripwire + guard-comment + mute-semantics + doc-consistency accepted; shared-module suggestion REJECTED (independent oracle kept). Codex pass 2 (2 findings): mute-test spec precision + FLOOR_RESULT addendum, both accepted. Full record: project memory `skill-weight-seam-plan-2026-07-06`.

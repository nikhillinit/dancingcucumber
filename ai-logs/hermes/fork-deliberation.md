# Hermes deliberation task — the final fork: STOP vs QuantConnect-overlay GO (DELIBERATION ONLY)

## Hard constraints (read first)
- Reasoning/deliberation only. Do NOT edit source code. Do NOT commit. Do NOT run `npm` or `node`.
- Your ONLY file write: create `ai-logs/hermes/runs/hermes-fork-deliberation-response.md`. Nothing else.

## Your role
You are the **independent Hermes voice**, second round. A Step-0 data-source capability matrix has now
been RUN. Claude leans STOP. Your job: adversarially weigh **STOP vs the QuantConnect-overlay GO** on
the *concrete* matrix result, challenge Claude's lean-STOP, and give your own verdict. Disagree if the
evidence warrants. Background + Claude's prior synthesis:
- `docs/superpowers/notes/2026-06-23-phase1-direction-roleplay-debate-synthesis.md` (read the Step-0
  result section + the amended recommendation).

## The matrix result (the new facts)
Goal: ONE pre-registered market-neutral (beta+dollar-neutral) long-short backtest on a delisting-aware,
point-in-time, small/mid-cap US universe — to test whether a broader cross-section carries residual
alpha, after 5+ prior DEV_FAILED results. Decisive requirement (b): **delisting RETURNS embedded in the
total-return stream** (terminal acquisition/bankruptcy return), not mere delisted-symbol retention.

Matrix verdict (8 vendors, cited):
- **No affordable source NATIVELY supplies verified (b).** Only **CRSP** has a true delisting-return
  field (`DLRET`) — and CRSP is **access-locked** for a solo non-academic ($5k–70k/yr, WRDS
  academic-only).
- **QuantConnect** = ~$0 cloud / ~$24/mo: survivorship-bias-free, small/mid-cap, PIT *universe*
  selection, corp-actions — BUT delisting handled by **last-price auto-liquidation, not `DLRET`**
  (captures mergers ≈ deal price; UNDERSTATES bankruptcy/performance terminal losses), and its
  Morningstar PIT fundamentals are vendor-caveated as having had PIT issues.
- **Sharadar** (low-100s/yr) and **Norgate** (~$630/yr): same PARTIAL (b) gap (acquisitions captured,
  bankruptcy stub not embedded); Sharadar best PIT value fundamentals (S&P500-only membership, self-build
  small/mid); Norgate best off-the-shelf PIT small/mid universe.
- EODHD / Polygon / Tiingo / Intrinio: FAIL (no PIT membership and/or UNCLEAR delisting returns).
- **Cost is NOT the binding constraint** (QC ~free). The binding constraint is (b) fidelity + PIT-fundamentals quality.

The asymmetry that matters: the missing terminal loss hits the **short leg** and **contrarian longs
that go to zero** hardest — exactly the survivorship axis this experiment cares about.

## The two branches
- **STOP (Claude's lean):** the pre-committed kill-biased rule fires — no affordable source clears
  native+verified (b) → close the residual-alpha lane, write the negative. Bounded, honest, consistent
  with 5+ DEV_FAILED.
- **GO (eyes-open A′):** QuantConnect-free (or Sharadar) + a **self-built Shumway delisting overlay**
  (~−30% performance / −100% bankruptcy, keyed on the delist-reason field), gated by a MANDATORY
  **source-integrity diagnostic** (after the overlay, do delisted losers populate the bottom/cheap
  deciles? else kill at ~$0). ~$0 dollars, ~3 dev-days, disclosed fidelity gap; validity then rests on
  the overlay being correct.

## Attack surfaces (engage each — agree or break, with reasons)
1. **Is STOP premature given QC is ~free?** The pre-committed rule said "(b) native + verified." Is
   "native" too strict when the Shumway overlay is a STANDARD, documented, auditable correction — i.e.
   is Claude hiding behind a rule to avoid a near-free shot?
2. **Circularity.** Claude calls the self-built overlay "circular" (trust my own repair of the defect).
   Does the source-integrity diagnostic — an INDEPENDENT check that delisted losers land in the bottom
   deciles — actually break that circularity, or not? Be precise about what the diagnostic can and
   cannot validate.
3. **EV of a near-free shot.** With ~$0 dollars + a hard pre-alpha kill gate, is GO strictly dominant
   over STOP (you learn more, lose little)? Or does the ~3 dev-days + the risk of an UNINTERPRETABLE
   "pass" (overlay-dependent) make STOP the better expected outcome?
4. **Does the overlay actually resolve the confound for a market-neutral book?** The residual freedom in
   the overlay (which names, what haircut, OTC drift) sits exactly on the short/contrarian axis. Can a
   "pass" be trusted, or is the overlay's uncertainty co-located with the very thing being measured?
5. **Cheaper decisive variant Claude under-weighted:** run ONLY the source-integrity diagnostic on
   QC-free data FIRST (does the small/mid universe even carry enough delisted losers to move the
   result?) and treat THAT as the kill/continue gate — deferring the overlay + alpha build until the
   data is shown to carry the signal. Is this diagnostic-before-overlay sub-rung the right first move?
6. **Anything Claude is structurally blind to.**

## Output (write ONLY ai-logs/hermes/runs/hermes-fork-deliberation-response.md)
1. **Verdict in one line:** STOP / GO / GO-WITH-AMENDMENTS, and the single move you'd make.
2. **Per-attack-surface (1-6):** your position with reasons (2-5 sentences each).
3. **Strongest point against your own view.**
4. **Recommended next action**, concrete, with a pre-committed kill rule.

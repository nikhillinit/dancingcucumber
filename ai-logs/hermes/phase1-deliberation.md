# Hermes deliberation task — Phase-1 direction (DELIBERATION ONLY, no code)

## Hard constraints (read first)
- This is a **reasoning/deliberation** task. Do **NOT** edit any source code. Do **NOT** commit.
- Do **NOT** run `npm` or `node`. You do not need to run anything.
- Your ONLY file write: create `ai-logs/hermes/runs/hermes-deliberation-response.md` with your written
  deliberation (structure specified at the bottom). Touch nothing else.

## Your role
You are the **independent Hermes voice** in a two-voice deliberation. Claude has already produced a
synthesis (its position is quoted below). Your job is NOT to agree — it is to **adversarially
pressure-test Claude's position and then give your own best recommendation** on the next research
move. The operator will read Claude's voice and your voice side by side and decide. Be concrete,
technical, and willing to disagree. If Claude is right, say *why* sharply; if Claude is wrong, break it.

## Background (full detail in these repo files — read them)
- `docs/superpowers/notes/2026-06-23-broad-universe-residual-screen-result.md` — the screen result.
- `docs/superpowers/notes/2026-06-23-phase1-direction-roleplay-debate-synthesis.md` — Claude's full
  roleplay→debate→synthesis (the position you are challenging).

Compressed facts:
- Project goal: find a price/fundamental/text signal candidate that passes a strict dev gate
  (deflation/DSR + blinded holdout + multiple-testing MinBTL budget). To date **five+ DEV_FAILED**
  results (advisor-v1, Plan 4, WS4 fundamental_value, WS3D lazy_prices Reading-C, and the 30-name
  floor itself). The floor failure is a **blend/additivity** failure (ensemble dilutes the best
  family), not "no family has idiosyncratic alpha."
- A keyless residual screen (info ratio = book_sharpe of stream minus beta*SPY) on a broad
  survivor panel (461 current-S&P-500 names backfilled to 2015) returned GREEN — but the verdict is
  **INCONCLUSIVE**, for three stacked reasons:
  1. The `max-family-IR > 0` rule is **non-discriminating**: it also fires on the DEV_FAILED floor
     (trend has +0.41 residual IR even on 30 mega-caps), so GREEN cannot license "proceed."
  2. The only informative quantity — per-family floor→broad delta — is **survivorship-confounded in
     a direction-specific way**: value goes worst (-0.41) → best (+0.42), the fingerprint of
     survivorship inflating contrarian/reversal signals (missing names = cheap stocks that fell and
     delisted). trend is universe-invariant (+0.41→+0.41); momentum less exposed.
  3. Every broad number is an upper bound (twice-filtered toward large-cap survivors).
- Cheap-test appendix: 40 information-less random long-flat signals through the same screen on the
  survivor panel earn null mean **+0.105** (p95 +0.170) — the panel pays ANY long-flat book a
  positive residual. Real top families clear p95 (value z=7.5, trend z=7.3) → NOT pure noise, but
  clearing a random-null is a **weaker** bar than the deflation+holdout that already rejected trend's
  identical 0.828/+0.41. Keyless ladder is **exhausted**; it cannot reach contrarian-survivorship,
  deflation, holdout, or the unobservable short leg.

## Claude's position (the thing you must challenge)
> Run a cheap, in-lane, keyless **data-availability spike** FIRST (scope the cheapest delisting-aware,
> point-in-time, small/mid-cap-reaching source: Norgate / Sharadar SF1+SEP / Tiingo / Polygon;
> CRSP/WRDS too dear). Pre-commit BOTH branches and a hard stop rule BEFORE the spike runs:
> - **Branch A** (affordable delisting-aware PIT sample exists, <= operator threshold): take **ONE**
>   pre-registered **market-neutral (beta + dollar-neutral) long-short** shot, small/mid-cap-inclusive,
>   on its **own immutable prereg surface** (mirror FundamentalCandidatePreReg; never touch
>   PreRegConfig), blinded holdout + ledger, its own random-null floor AND DSR bar, judged on
>   residual/absolute Sharpe (not "beat SPY"). Pass = hypothesis, still not a tradable edge.
> - **Branch B** (no affordable source): **STOP**, write the negative, close the residual-alpha lane.
> - **Stop rule** (frozen with the design): one shot fails its null+DSR+holdout -> lane CLOSED, no
>   universe-iterating.
> Explicitly refuse: long-short on the keyless survivor panel; folding SESTM news in now; building the
> automated Phase-1 infra before the one manual shot; treating a null-clear as an edge.

## Attack surfaces (engage each — agree or break, with reasons)
1. **EV after five negatives.** Is "one more shot, but cheap and stop-ruled" genuinely EV-positive,
   or is it sunk-cost rationalization dressed as discipline? What posterior on a real edge would
   justify the spend? Is STOP-now the more honest call?
2. **Is the spike the right first move,** or a procrastination ritual? Would you instead commit to a
   design immediately, or kill immediately?
3. **Reachability.** Is a genuinely PIT, delisting-aware, small/mid-cap long-short backtest actually
   buildable by a solo dev at low-hundreds cost, or is Claude hand-waving a hard data-engineering
   problem (point-in-time membership, delisting returns, borrow/short constraints)?
4. **Right instrument?** Claude pivots to market-neutral long-short. But that lane is motivated by the
   LEAST trustworthy numbers (survivorship-inflated contrarian IRs) and is MORE exposed to missing
   delisted names. Is long-short the right bet, or does it just move the same confound to a costlier
   stage? Is there a CHEAPER discriminating test Claude missed?
5. **Sequencing vs the news lane.** SESTM news is the one truly orthogonal signal (not a price
   re-slice). Should the news lane go FIRST instead of the long-short price lane, despite needing a
   fixture + keys?
6. **Anything Claude is structurally blind to** — a wrong framing, an unstated assumption, a cheaper
   path to a decision.

## Output (write ONLY this file: ai-logs/hermes/runs/hermes-deliberation-response.md)
1. **Verdict in one line:** AGREE / AGREE-WITH-AMENDMENTS / DISAGREE, and the single move you'd make.
2. **Per-attack-surface (1-6):** your position with reasons (2-5 sentences each).
3. **Strongest point against your own view** (steelman the other side).
4. **Your recommended next action**, concrete enough to execute, with a pre-committed stop/kill rule.

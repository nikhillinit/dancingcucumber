Both runs converged; advisor signed off with framing fixes I've applied. Here's the combined report.

---

# Combined synthesis: can the advisor get a *new, validated* edge — and how would you know?

The two follow-ups answer different halves of one question, and read together they point the same direction as your floor already did. **First, the honest bottom line**, then the evidence.

## The sober read (lead with this)

The proven lane is closed (family-reweighting → `DEV_FAILED`). The **best-evidenced *new* lane — SESTM news sentiment — turns out to be gross-only for the universe you actually trade**, and the validation rigor you'd add is a **gate that will keep confirming most candidates are false**, not an advance that clears anything. For a **30-name, large-cap, price-only book, the achievable net edge looks thin.** What's actually actionable is narrow: build the honest gate now, and there's exactly one cheap signal lane left to test through it — and it isn't a free lunch either.

This is consistent with how this project already behaves (accept the negative, block the release). I'm not dressing it as a 1-2-3 success path, because the evidence doesn't support one.

---

## Part A — The orthogonal-signal verdict: SESTM gets **demoted** (own the pivot)

I featured SESTM as the headline orthogonal lane twice, and the earlier advisor pass called it "highest-evidence." **The net-of-cost dive changes that ranking: SESTM drops from #1 to research-only/conditional.** Here's why, all 3-0 confirmed against primary sources:

| Finding                                            | Evidence                                                                                                                                                                                                                                                          | Conf.      |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| Headline is **gross + microcap-concentrated**      | EW Sharpe **4.29** @ 94.7% daily turnover vs value-weighted **1.33** @ 91.6%; small-stock news response 52bps vs **11bps** large-cap → alpha lives in small/illiquid (Ke-Kelly-Xiu, NBER 26186)                                                                   | High (3-0) |
| **Net survives only by engineering turnover down** | Authors' own 10bps/day analysis: net Sharpe **peaks 2.30** only after cutting turnover 0.95→0.46; **1.17** at γ=0.1. Naive 10bps on the 94%-turnover original → **~0**                                                                                            | High (3-0) |
| High-turnover is the **cost-fragile** category     | Frazzini-Israel-Moskowitz (~$1T live trades): short-term reversal — the turnover analog — is the **only** standard anomaly that dies on cost ($9.5B break-even vs $52-103B for mom/value/size). Novy-Marx-Velikov: <50% monthly turnover survives, higher doesn't | High (3-0) |
| Post-pub decay is **worst in large/liquid names**  | McLean-Pontiff (JF 2016): 26-58% decay, greatest exactly in the large, liquid, cheap-to-arbitrage stocks → undercuts the "liquid large-cap variant preserves it" hope                                                                                             | High (3-0) |

**Reinforced by the abstention-killed Chen-Velikov "Anomaly Zoo" claims** (casualties of the API outage, not refutations — peer-reviewed and consistent): value-weighting + low-turnover "banding" leaves only **4-13 bps/month** net post-publication; the modal EW long-short collapses 66bps gross → **−3bps net** post-pub. Same conclusion from a second source.

**Verdict:** for your large-cap price-only universe, **SESTM is effectively gross-only** — no confirmed evidence of a net, post-decay, large-cap-only tradable slice. The residual edge is real but microcap-concentrated.

## Part B — The validation gate: highest-confidence, but it **raises the bar, it doesn't clear it**

Critical framing: this is the lowest-risk, highest-fit addition — *and* it's a **guard, not an advance**. Applied to your current candidates it can only confirm `DEV_FAILED` *harder* (the floor failed on the undeflated relative bar; deflation makes the bar higher). Its value is **forward**: it's the gate your future signal hunt must survive. Confirmed bolt-ons for `backtest/`:

- **N-tracking is the precondition** (3-0) — dev/holdout alone is provably inadequate; it ignores how many configs you tried, and walk-forward OOS is *not* a multiple-testing defense (~20 WF iterations find a false 5%-significant strategy).
- **Deflated Sharpe** (3-0): DSR = PSR(SR₀), with SR₀ = √V[SR̂ₙ]·((1−γ)Φ⁻¹[1−1/N]+γΦ⁻¹[1−1/(Ne)]). Every input (N, V[SR̂ₙ], T, skew, kurt) is *already produced by your dev sweep*. My worked example proves the teeth: a **2.5 Sharpe, N=100, skew −3, kurt 10, T=1250 → DSR ≈ 0.90 → FAILS** the 0.95 bar.
- **MinBTL** (3-0): ≤~45 independent configs on 5yr of data before the gate is statistically meaningless — a pre-registered throughput budget.
- **Harvey-Liu-Zhu t > 3.0** (2-0) for any factor/signal selection; **purging/embargo** (h≈0.01T, 3-0) — the highest-confidence, IID-independent leakage guard, worth adopting even without full CPCV.
- **PBO via CSCV** (3-0): use to *audit* the selection process, **never as the objective**; its IID-block assumption is fragile for overlapping labels. **CPCV > walk-forward** is medium-confidence (synthetic-only).

---

## What this means, sequenced (honest, not a success ladder)

1. **Build the honest gate now** (DSR + MinBTL + t>3.0 + purging/embargo). Signal-agnostic, hardens what you already have. **Caveat it correctly: this makes finding a winner *harder*, not easier** — it's insurance against the next false positive, not a path to a pass.
2. **One cheap signal lane left: timely-price value+momentum** (no new data — uses families you already have; from the first deep-dive). But apply the same scrutiny I used to demote SESTM: **value (HML) is spanned/redundant in the FF5 model with a post-2017 drawdown**, and decay applies. Momentum survives on *capacity* (turnover isn't its killer, unlike STR), so the blend is the cheapest *live* lane — **but it must clear the same deflated bar, and it is not evidenced-to-work, just cheap to test.**
3. **SESTM → research-only/conditional.** If ever pursued, *only* as a pre-registered minimum-viable test: weekly rebalance, value-weighted, large-cap-only, **free text first** (EDGAR 8-K / Alpha Vantage news) + Loughran-McDonald dictionary, measured **net** with explicit turnover — escalate to the supervised pipeline + paid feed *only if* the cheap variant clears a pre-registered net floor. Don't build the expensive version on the strength of a gross Sharpe.

## Leads & forks (one-liners)
- **Microcap fork:** capacity is *not* binding at $1M-$500M — the book is small enough to reach the microcaps where SESTM's edge actually lives. But that's a **universe change your project hasn't signaled wanting** (you trade large-cap). Noting it exists; not featuring it.
- **Kirtac-Germano (arXiv:2412.19245)** LLM-sentiment was refuted as "gross-only" (it *does* report net) — but it's **leakage-suspect for the exact reason the first run established**: LLM-on-financial-text results collapse ~50% past the training cutoff ("Profit Mirage," arXiv:2510.07920). A lead, deflated honestly, not a validated result.

## Reliability note
Both runs hit the same severe API-500 storm in their verify phase. I triaged every kill: **`0-3` = genuine adversarial refute** (LLM "gross-only", the small-cost inference, STR realized-drag specifics — correctly dropped); **`0-0`/`1-0` = abstention casualty** of the outage, not disproven. The abstention casualties I rescued (Chen-Velikov, the full DSR formula) all *reinforce* the confirmed findings — so the verdict is, if anything, more robust than the raw confirmed-counts suggest.

**Sources (primary):** Ke-Kelly-Xiu [NBER 26186](https://www.nber.org/papers/w26186); Frazzini-Israel-Moskowitz *Trading Costs of Asset Pricing Anomalies* (RFS 2018); Novy-Marx-Velikov ([SSRN 2535173](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2535173)); McLean-Pontiff ([JF 2016](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)); Chen-Velikov *Anomaly Zoo*; Bailey-López de Prado *Deflated Sharpe* ([SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)) & *PBO* ([SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)) & *False Strategy Theorem* ([SSRN 3221798](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3221798)); Harvey-Liu-Zhu ([RFS 2016](https://academic.oup.com/rfs/article/29/1/5/1843824)); Arian-Norouzi-Seco (CPCV, KBS 2024); *Profit Mirage* ([arXiv:2510.07920](https://arxiv.org/pdf/2510.07920)).

Resolved conclusions are saved to project memory. The single most defensible next action is **#1 — build the deflation gate** — precisely because it's the one move that's valuable regardless of which signal you chase next.
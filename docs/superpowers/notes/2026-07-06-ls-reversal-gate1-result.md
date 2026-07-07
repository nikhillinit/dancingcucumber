# L/S Reversal Gate-1 Kill Screen — One-Shot Result (2026-07-06)

**Verdict: CLOSED** — lane closed forever (write-once lock), negative recorded.

Executed exactly once per the §7.5 ceremony, post-merge of PR #27 (main `bb9f951`),
under explicit operator authorization for this session. This was the single sanctioned
real-data execution; the CLI is now self-locked (`LS_REVERSAL_RESULT.json` verdict
CLOSED refuses all reruns).

## Governing artifacts (frozen)

- Plan: [2026-07-06-decision5-ls-reversal-gate1.md](../plans/2026-07-06-decision5-ls-reversal-gate1.md)
  (§3 kill rule, §7.5 ceremony)
- Prereg: [LS_REVERSAL_PREREG.md](../../../apps/quant/advisor/research/LS_REVERSAL_PREREG.md)
  (pinned methodology hash)

## Verbatim stdout

```
methodology_hash 3784b0998303861f8392f308324d789c032bbd64c0f56f20a3ddaaa3c2394c6b
run_hash e2e5045cf5432c5fec91bfc98d5e23f7b482fb27c2c29e2923c87e4ab9b01d93
value | precost -0.4062 | beta 0.6368 | postcost_rev 0.1787
fundamental_value | precost -0.3200 | beta 0.9177 | postcost_rev 0.1922
lazy_prices | precost -0.3968 | beta 0.9474 | postcost_rev 0.2389
VERDICT | CLOSED (tau_ls=0.2, rule 2-of-3)
```

Full log: `ai-logs/ls-gate1-oneshot-stdout.txt` (committed with this note).

## Hashes

| Hash | Value | Check |
| --- | --- | --- |
| methodology_hash | `3784b0998303861f8392f308324d789c032bbd64c0f56f20a3ddaaa3c2394c6b` | == pinned in LS_REVERSAL_PREREG.md ✓ |
| run_hash | `e2e5045cf5432c5fec91bfc98d5e23f7b482fb27c2c29e2923c87e4ab9b01d93` | binds config + panel + both fixture bytes |

## Per-family results

| Family | Published pre-cost IR | Reproduced pre-cost IR | Tripwire (±0.02) | Beta | Post-cost reversed IR | ≥ τ_LS 0.20 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| value | −0.41 | −0.4062 | ✓ (Δ 0.0038) | 0.6368 | 0.1787 | ✗ |
| fundamental_value | −0.32 | −0.3200 | ✓ (Δ 0.0000) | 0.9177 | 0.1922 | ✗ |
| lazy_prices | −0.40 | −0.3968 | ✓ (Δ 0.0032) | 0.9474 | 0.2389 | ✓ |

Survivors: 1 of 3 (lazy_prices only). Frozen kill rule requires ≥2 of 3 families with
post-cost reversed IR ≥ τ_LS 0.20 → **CLOSED**.

## Interpretation (frozen language)

CLOSED: lane closed forever (write-once lock), negative recorded; next ruling in queue
= insider Form-4 lane (queued 2nd). No reruns or threshold tweaks are proposed or
permitted.

The reproduction tripwire passed cleanly (all three pre-cost IRs within ±0.02 of
published), so this is a genuine cost verdict, not wiring drift: realistic short costs
(2× transaction re-charge + 50 bps borrow, zero rebate) ate the in-sample reversal
effect below the frozen futility bar in 2 of 3 families.

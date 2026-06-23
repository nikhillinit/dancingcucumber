# Hermes fork deliberation response

## 1. Verdict in one line

GO-WITH-AMENDMENTS: take only the diagnostic-before-overlay rung on QuantConnect-free data; if QC cannot prove enough classified delisted losers on the cross-sectional tails before any alpha metric is read, STOP and write the negative.

## 2. Per-attack-surface

### 1. Is STOP premature given QC is ~free?

Pure STOP is slightly premature because the cheapest next move is not the overlay or the alpha test; it is a source-integrity preflight that can fail fast at roughly zero dollars. But Claude is right that "native + verified" was not an arbitrary luxury: the market-neutral long-short question is maximally sensitive to missing terminal losses. A self-built Shumway overlay is not equivalent to CRSP DLRET, so the rule should not be weakened directly into GO. The amendment is a narrower rung: QC earns only a data-validity diagnostic, not an overlay-dependent alpha shot.

### 2. Circularity

The diagnostic breaks only part of the circularity. It can validate support and directionality: delisted losers exist in the chosen universe, have usable classifications, and concentrate where survivorship bias says they should, such as bottom or cheap deciles. It cannot validate the terminal-return magnitudes, the haircut mapping, OTC drift, or whether the exact missing loss is -30%, -60%, or -100%. Therefore a passing diagnostic makes the overlay auditable enough to specify and preregister; it does not make a later overlay-dependent alpha pass decisive on its own.

### 3. EV of a near-free shot

GO is not strictly dominant over STOP if GO means building the overlay and running the alpha test, because a positive result could be uninterpretable and consume the same three dev-days the lane was trying to protect. GO is positive EV only if the first move is a hard pre-alpha kill gate with no optimization and no performance readout. That diagnostic buys information STOP lacks: whether QC actually carries the delisted-loser mass needed for the experiment to be meaningful. If it fails, STOP becomes stronger than Claude's current STOP because it rests on direct source support failure, not just the native-DLRET rule.

### 4. Does the overlay resolve the market-neutral confound?

Not fully. The uncertain part of the overlay sits exactly on the short leg and contrarian-long tail, so the repair is co-located with the measured effect rather than orthogonal to it. A pass after the overlay could still be a statement about haircut assumptions, delist-reason mapping, or missing OTC liquidation paths. The only defensible use is as a separately preregistered sensitivity framework: results must survive disclosed haircut bands and must be labeled hypothesis-generating unless a CRSP-like source later confirms the terminal-return stream.

### 5. Diagnostic-before-overlay sub-rung

Yes, this is the right first move and Claude under-weighted it. Run no alpha and build no overlay until QC proves the raw ingredients exist: enough delisting events, usable reason classifications, and economically material concentration in bottom or cheap deciles. This isolates the question "can this source even expose the survivorship axis?" from "can our overlay price it correctly?" If the answer is no, the lane dies cleanly before any self-built correction can rationalize continuation.

### 6. Anything Claude is structurally blind to

Claude frames the fork as rigor versus one cheap shot, but the missing middle is a non-alpha data falsification rung. It is possible to challenge STOP without endorsing an overlay-dependent market-neutral result. Claude also underplays operator-risk: once an overlay is built, sunk-cost pressure will push toward interpreting whatever comes out. The diagnostic artifact should therefore be the only next deliverable, with the default written conclusion set to STOP unless its precommitted support thresholds clear.

## 3. Strongest point against my own view

The diagnostic-before-overlay rung can become a loophole that delays an already warranted negative conclusion. Even if QC shows delisted losers in the right tails, the later alpha test still depends on self-assigned terminal losses, and that is the exact field the matrix found no affordable vendor can verify natively. Claude's STOP is cleaner, more consistent with the precommitted rule, and less exposed to motivated interpretation after five-plus DEV_FAILED results.

## 4. Recommended next action

Run a one-session QC-free source-integrity diagnostic only. Pre-register the diagnostic before running it, and forbid alpha metrics, Sharpe, portfolio PnL, or family comparisons in the artifact.

Required diagnostic checks:

- Export the small/mid-cap eligible universe with delisting events over the intended dev period.
- Classify delistings into acquisition, bankruptcy/performance, and unknown using only QC-available event/reason fields or documented metadata.
- Compute delisting-event counts by pre-delist rank bucket for the candidate cross-sectional tails, especially bottom and cheap deciles, before reading any alpha returns.
- Report coverage gaps: unknown reason rate, symbols lacking event metadata, and names that cannot be tied back to pre-delist tradable-universe membership.
- Kill unless the source shows enough usable adverse delistings to move the test: at least 50 classified bankruptcy/performance or severe-loss delistings in the dev sample, at least 90% of delisting events mappable to a terminal class, and at least 2x concentration of adverse delistings in the bottom/cheap tails versus the top/expensive tails.

Pre-committed kill rule: if any required diagnostic check fails, STOP and close the residual-alpha / market-neutral lane with a negative note: no affordable native-DLRET source exists, and QC did not prove enough auditable delisted-loser support to justify a self-built overlay. If all checks pass, the next action is not an alpha run; it is a separate preregistration of the Shumway-overlay specification, sensitivity bands, null floor, DSR, blinded holdout, and no-iteration stop rule for exactly one market-neutral shot.

# Hermes Port (Plan 0 of 3) — Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Bootstrap exception:** This plan makes Hermes able to dispatch against `C:\dev\AIHedgeFund`. It therefore **cannot itself be dispatched via Hermes** (chicken-and-egg) — execute it **directly**. Every *subsequent* plan (1 and 2) dispatches its edits through the Hermes set up here, honoring the workspace contract.

**Goal:** Vendor the Hermes orchestrator into AIHedgeFund as a self-contained `orchestrate.mjs` with AIHedgeFund-specific routing, governance, Python gates, and a hardened `advisor-gate` — so `npm run hermes:production -- --task "..."` routes and gates work in this repo.

**Architecture:** Copy Updog's `orchestrate.js` verbatim (ROOT auto-resolves to `dirname` of the script, so a copy targets AIHedgeFund). Change only two strings (repo name; financial gate name `calc-gate`→`advisor-gate`). Add AIHedgeFund config (`model-routing.json`, `DEV_BRAIN.md`, `SOUL.md`), npm gate scripts backed by a resilient Python test runner that no-ops cleanly until the advisor package exists. **Never edit Updog's working copy.**

**Tech Stack:** Node.js (ESM `.mjs`), Python/pytest gates, CLI-binary model backends (claude/codex/kimi).

**Scope note:** Plan 0 of 3. After this, Plan 1 (foundation slice) runs through Hermes; Plan 3 extends `advisor-gate` to run the real purged walk-forward floor.

---

### Task 1: Vendor `orchestrate.mjs` with the two required edits

**Files:**
- Create: `orchestrate.mjs` (copied from `C:\dev\Updog_restore\orchestrate.js`)

- [ ] **Step 1: Copy the orchestrator (as ESM `.mjs`)**

Run (PowerShell):
```powershell
Copy-Item "C:\dev\Updog_restore\orchestrate.js" "C:\dev\AIHedgeFund\orchestrate.mjs"
```
(`.mjs` because the root `package.json` has no `"type": "module"`; orchestrate uses ESM `import`.)

- [ ] **Step 2: Re-point the hardcoded repo name**

In `orchestrate.mjs`, find (in `buildPrompt`):
```javascript
    'You are operating inside Updog_restore.',
```
Replace with:
```javascript
    'You are operating inside AIHedgeFund.',
```

- [ ] **Step 3: Rename the financial gate `calc-gate` → `advisor-gate`**

In `orchestrate.mjs`, find (in `assertFinancialGate`):
```javascript
  if (plan.gate !== 'npm run calc-gate') {
    throw new Error(
      `Financial gate proof failed: production-financial plan must resolve gate to "npm run calc-gate", got "${plan.gate}".`
    );
```
Replace both occurrences of `npm run calc-gate` with `npm run advisor-gate`:
```javascript
  if (plan.gate !== 'npm run advisor-gate') {
    throw new Error(
      `Financial gate proof failed: production-financial plan must resolve gate to "npm run advisor-gate", got "${plan.gate}".`
    );
```
(Optional cosmetic: the `printHelp` text also mentions `npm run calc-gate`; update it for consistency if you like — not functional.)

- [ ] **Step 4: Verify it parses and prints help**

Run: `node orchestrate.mjs --help`
Expected: usage text prints, exit 0 (no `SyntaxError`, no `ERR_REQUIRE_ESM`).

- [ ] **Step 5: Commit**

```
git add orchestrate.mjs
git commit -m "feat(hermes): vendor orchestrator as orchestrate.mjs targeting AIHedgeFund"
```

---

### Task 2: Resilient Python gate runner

A tiny Node wrapper so npm gate scripts (`check`, `lint`, `advisor-gate`) pass cleanly before the advisor package/tests exist (skip if target path absent; treat pytest "no tests collected" exit 5 as success), and fail correctly on real test failures.

**Files:**
- Create: `tools/run-pytest.mjs`

- [ ] **Step 1: Create the runner**

`tools/run-pytest.mjs`:
```javascript
#!/usr/bin/env node
// Resilient pytest gate. Skips (exit 0) if the target test path does not exist
// yet (bootstrap), maps pytest "no tests collected" (5) to success, and passes
// through any real failure code.
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2);
const target = args[0];

if (target && !existsSync(target)) {
  console.log(`gate: "${target}" not present yet — skipping (bootstrap).`);
  process.exit(0);
}

const result = spawnSync('python', ['-m', 'pytest', ...args], {
  stdio: 'inherit',
  shell: process.platform === 'win32',
});

let code = result.status ?? 1;
if (code === 5) code = 0; // "no tests collected" is acceptable during bootstrap
process.exit(code);
```

- [ ] **Step 2: Verify it skips a missing path**

Run: `node tools/run-pytest.mjs apps/quant/advisor/tests`
Expected: prints `gate: "apps/quant/advisor/tests" not present yet — skipping (bootstrap).`, exit 0.

- [ ] **Step 3: Commit**

```
git add tools/run-pytest.mjs
git commit -m "feat(hermes): resilient pytest gate runner"
```

---

### Task 3: AIHedgeFund routing config

Adapts Updog's `model-routing.json` to advisor-domain specialists. Financial-risk tasks (backtest/risk-sizing/valuation) auto-escalate to `production-financial` → `npm run advisor-gate` + reviewer + audit + human approval.

**Files:**
- Create: `.claude/hermes/model-routing.json`

- [ ] **Step 1: Create the routing config**

`.claude/hermes/model-routing.json`:
```json
{
  "version": 2,
  "defaults": { "research": "claude", "production": "codex", "distribution": "claude" },
  "ownership": {
    "research": {
      "owner": "claude", "reviewer": "kimi", "role": "leader-coordinator",
      "artifact": "implementation brief", "gate": "npm run doctor:quick"
    },
    "production": {
      "owner": "codex", "reviewer": "claude", "role": "worker-executor",
      "artifact": "diff plus tests", "gate": "npm run check"
    },
    "production-financial": {
      "owner": "codex", "reviewer": "claude", "audit": "kimi", "role": "worker-executor",
      "specialistRequired": true, "artifact": "diff plus backtest-disclosure notes",
      "gate": "npm run advisor-gate", "humanApproval": true
    },
    "distribution": {
      "owner": "claude", "repairOwner": "codex", "role": "release-manager",
      "artifact": "PR-ready summary", "gate": "npm run lint"
    }
  },
  "debate": { "comparators": ["claude", "codex", "kimi"], "synthesis": "claude" },
  "longContextModel": "kimi",
  "longContextTriggers": ["full repo audit", "repo-wide", "system-wide", "architecture scan", "cross-module trace"],
  "manualFlags": { "--claude": "claude", "--codex": "codex", "--kimi": "kimi", "--gemini": "gemini", "--agy": "agy" },
  "specialists": {
    "backtest-integrity": {
      "keywords": [
        { "phrase": "backtest", "weight": 4 },
        { "phrase": "walk-forward", "weight": 4 },
        { "phrase": "look-ahead", "weight": 4 },
        { "phrase": "survivorship", "weight": 4 },
        { "phrase": "point-in-time", "weight": 3 },
        { "phrase": "purged cv", "weight": 3 },
        { "phrase": "sharpe", "weight": 3 }
      ],
      "risk": "financial"
    },
    "risk-sizing": {
      "keywords": [
        { "phrase": "position sizing", "weight": 4 },
        { "phrase": "position limit", "weight": 4 },
        { "phrase": "risk limit", "weight": 4 },
        { "phrase": "kill switch", "weight": 4 },
        { "phrase": "drawdown", "weight": 3 },
        { "phrase": "volatility-adjusted", "weight": 3 },
        { "phrase": "correlation-adjusted", "weight": 3 }
      ],
      "risk": "financial"
    },
    "valuation-math": {
      "keywords": [
        { "phrase": "dcf", "weight": 4 },
        { "phrase": "intrinsic value", "weight": 4 },
        { "phrase": "owner earnings", "weight": 4 },
        { "phrase": "discount rate", "weight": 3 },
        { "phrase": "residual income", "weight": 3 },
        { "phrase": "ev/ebitda", "weight": 3 }
      ],
      "risk": "financial"
    },
    "signal-quality": {
      "keywords": [
        { "phrase": "signal family", "weight": 4 },
        { "phrase": "ensemble", "weight": 3 },
        { "phrase": "calibration", "weight": 3 },
        { "phrase": "skill-weight", "weight": 3 },
        { "phrase": "information coefficient", "weight": 3 }
      ],
      "risk": "operational"
    },
    "test-repair": {
      "keywords": [
        { "phrase": "failing test", "weight": 4 },
        { "phrase": "flaky test", "weight": 4 },
        { "phrase": "pytest failure", "weight": 3 },
        { "phrase": "test regression", "weight": 3 }
      ],
      "risk": "quality"
    },
    "code-reviewer": {
      "keywords": [
        { "phrase": "code review", "weight": 4 },
        { "phrase": "risk scan", "weight": 3 },
        { "phrase": "maintainability check", "weight": 3 }
      ],
      "risk": "quality"
    }
  },
  "scoring": {
    "minScoreToAssign": 3,
    "tieBreaker": "highest-risk-wins",
    "riskOrder": ["financial", "operational", "quality"]
  },
  "gates": {
    "research": "npm run doctor:quick",
    "production": "npm run check",
    "production-financial": "npm run advisor-gate",
    "distribution": "npm run lint"
  },
  "commands": {
    "claude": { "binEnv": "CLAUDE_CODE_BIN", "defaultBin": "claude", "args": ["-p"] },
    "codex": { "binEnv": "CODEX_BIN", "defaultBin": "codex", "args": ["exec", "--sandbox", "danger-full-access"] },
    "kimi": { "binEnv": "KIMI_CODE_BIN", "defaultBin": "kimi-cli", "args": ["--print", "--input-format", "text", "--final-message-only"] },
    "gemini": { "binEnv": "GEMINI_BIN", "defaultBin": "gemini", "args": ["--output-format", "text", "--yolo", "--prompt="] },
    "agy": { "binEnv": "AGY_BIN", "defaultBin": "agy", "args": [] }
  }
}
```

- [ ] **Step 2: Verify it is valid JSON**

Run: `node -e "JSON.parse(require('fs').readFileSync('.claude/hermes/model-routing.json','utf8')); console.log('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```
git add .claude/hermes/model-routing.json
git commit -m "feat(hermes): AIHedgeFund routing config (advisor financial specialists)"
```

---

### Task 4: Governance files (`DEV_BRAIN.md` required, `SOUL.md` optional)

`DEV_BRAIN.md` is injected into every Hermes prompt — use it to encode the spec's load-bearing constraints so every dispatched task inherits them.

**Files:**
- Create: `DEV_BRAIN.md`
- Create: `.claude/hermes/SOUL.md`

- [ ] **Step 1: Create `DEV_BRAIN.md`**

`DEV_BRAIN.md`:
```markdown
# DEV_BRAIN — AIHedgeFund advisor

This repo builds a deterministic-first AI investment advisor. Design spec:
`docs/superpowers/specs/2026-06-14-ai-advisor-architecture-design.md`. Active plans:
`docs/superpowers/plans/`.

## Non-negotiable constraints (every task inherits these)
1. **The LLM never sizes positions or sets risk limits.** Every number is computed in
   deterministic Python; the LLM only converts unstructured inputs to typed facts and explains.
2. **Trust boundary = the Pydantic `SignalBundle` seam.** Deterministic families produce signals;
   the deterministic risk manager + allocator own position size. "Graph proposes, rails dispose."
3. **Honest backtest discipline.** Price/volume walk-forward only is defensible. Fundamentals from
   yfinance are restated, not as-reported — disclose (never fabricate), lag ~90 days, snapshot forward.
   The 4-line disclosure header is mandatory on every backtest report.
4. **The `advisor-gate` is sacred.** Never weaken it to make a change pass. v1 floor: purged
   walk-forward OOS Sharpe net of costs beats SPY by the pre-registered margin AND beats the best
   single family across >=2 regimes.
5. **No fabricated data.** Missing source => signal `unavailable` and excluded, never a default value.

## Working rules
- Produce the smallest safe diff. TDD: failing test first.
- Search for existing implementations before adding new abstractions.
- Touching backtest/risk/valuation = financial risk => routes to production-financial + advisor-gate.
```

- [ ] **Step 2: Create `.claude/hermes/SOUL.md`**

`.claude/hermes/SOUL.md`:
```markdown
# Hermes SOUL — AIHedgeFund

You are a disciplined build agent for a financial advisor codebase. You value correctness over
cleverness and never let narrative override deterministic math. You refuse to weaken gates or
fabricate data to make work "pass." When a financial-risk area is touched, you confirm advisor-gate
coverage. You leave the repo cleaner than you found it and report risks honestly in your handoff.
```

- [ ] **Step 3: Commit**

```
git add DEV_BRAIN.md .claude/hermes/SOUL.md
git commit -m "feat(hermes): governance files encoding advisor constraints"
```

---

### Task 5: Hermes + gate npm scripts

Adds the `hermes:*` entry points and the gate scripts they call. Gates target the advisor test path explicitly (ignoring the repo's messy root-level scripts) and use the resilient runner from Task 2.

**Files:**
- Modify: `package.json` (root) — add to the `"scripts"` block

- [ ] **Step 1: Add the scripts**

In `package.json`, the `"scripts"` object currently ends at `"db:push": "drizzle-kit push"`. Add these keys (insert a comma after `db:push` and append):
```json
    "db:push": "drizzle-kit push",
    "hermes": "node orchestrate.mjs",
    "hermes:dry": "node orchestrate.mjs --dry-run",
    "hermes:route": "node orchestrate.mjs --json",
    "hermes:research": "node orchestrate.mjs --phase research",
    "hermes:production": "node orchestrate.mjs --phase production",
    "hermes:distribution": "node orchestrate.mjs --phase distribution",
    "hermes:doctor": "node orchestrate.mjs doctor",
    "doctor:quick": "node orchestrate.mjs doctor",
    "check": "node tools/run-pytest.mjs apps/quant/advisor/tests",
    "lint": "node tools/run-pytest.mjs apps/quant/advisor/tests",
    "advisor-gate": "node tools/run-pytest.mjs apps/quant/advisor/tests"
```
(`lint` and `advisor-gate` proxy to the advisor test suite for now; Plan 3 extends `advisor-gate` to also run the purged walk-forward floor check. `doctor:quick` reuses the Hermes provider report — always exits 0.)

- [ ] **Step 2: Verify the scripts parse**

Run: `npm run hermes:doctor`
Expected: a "Hermes CLI doctor" provider table prints (claude/codex/kimi/gemini/agy each `found` or `missing`), exit 0. Missing model CLIs are fine — Plan 1 only needs whichever models you have installed.

- [ ] **Step 3: Commit**

```
git add package.json
git commit -m "feat(hermes): hermes + Python gate npm scripts"
```

---

### Task 6: End-to-end verification (routing + gate, no model execution)

Proves Hermes routes an advisor task correctly and the gate no-ops cleanly pre-Plan-1.

**Files:** none (verification only)

- [ ] **Step 1: Financial routing dry-run**

Run:
```
npm run hermes:dry -- --phase production --task "fix look-ahead bias in walk-forward backtest position sizing"
```
Expected (in the printed `=== ROUTING PLAN ===` JSON): `"specialist"` is `backtest-integrity` or `risk-sizing`, `"risk": "financial"`, and `"gate": "npm run advisor-gate"`. The `=== PROMPT ===` section begins `You are operating inside AIHedgeFund.` and includes the `--- DEV_BRAIN ---` block.

- [ ] **Step 2: Non-financial routing dry-run**

Run:
```
npm run hermes:dry -- --phase production --task "add a docstring to the CLI help text"
```
Expected: `"risk": "standard"` (no specialist match) and `"gate": "npm run check"`.

- [ ] **Step 3: Gate self-test (skips cleanly, advisor package absent)**

Run: `npm run check`
Expected: `gate: "apps/quant/advisor/tests" not present yet — skipping (bootstrap).`, exit 0.

Run: `npm run advisor-gate`
Expected: same skip message, exit 0.

- [ ] **Step 4: Confirm a financial production plan asserts the renamed gate**

Run:
```
npm run hermes:route -- --phase production --task "implement dcf intrinsic value owner earnings"
```
Expected: JSON shows `"risk": "financial"` and `"gate": "npm run advisor-gate"` (proving the `assertFinancialGate` rename is internally consistent — a financial plan that resolved any other gate would throw on live execution).

- [ ] **Step 5: Commit a short port note (optional) and finish**

```
git add -A
git commit -m "docs(hermes): bootstrap port verified (routing + gate)" --allow-empty
```

---

## How Plan 1 changes after this

Once Plan 0 is merged, **Plan 1 executes via Hermes**. Each Plan 1 task becomes a dispatch, e.g.:
```
npm run hermes:production -- --task "Implement Task 3 of docs/superpowers/plans/2026-06-14-ai-advisor-foundation-slice.md: the frozen Pydantic SignalBundle seam. Follow the plan's TDD steps exactly."
```
- Plan 1 Task 1 (cleanup) and Task 2 (package skeleton) touch no advisor tests yet, so their preflight/postflight gates **skip cleanly** via the resilient runner. From Task 3 on, gates run the real suite.
- Tasks whose description hits financial keywords (valuation, backtest, risk) route to `production-financial` and must pass `npm run advisor-gate` (which, until Plan 3, equals the advisor test suite). That is the intended discipline.

## Self-Review

**1. Spec coverage (§9 Hermes):** vendor + de-hardcode (Task 1) ✅; Python gates (Tasks 2, 5) ✅; portfolio/risk routing config (Task 3) ✅; `advisor-gate` named + asserted (Tasks 1, 3, 5) ✅; governance encoding the spec constraints (Task 4) ✅; verification (Task 6) ✅. Deferred to Plan 3: `advisor-gate` running the *real* purged walk-forward floor (here it proxies to the test suite) — stated explicitly.

**2. Placeholder scan:** No TBD/TODO/"handle errors". Every file's full content is shown; `orchestrate.mjs` is an exact copy + two exact string edits. ✅

**3. Type/string consistency:** `npm run advisor-gate` is identical across the orchestrate edit (Task 1), routing `gates` + `production-financial` ownership (Task 3), and the npm script (Task 5). `apps/quant/advisor/tests` is identical across the runner usage in every gate script. `orchestrate.mjs` filename is consistent across all `hermes:*` scripts. ✅

**Known caveat:** `production` model owner is `codex`; if the Codex CLI is not installed, live Plan 1 production dispatches fail at model execution (not gate). Use `--model claude` to override per task, or set `CODEX_BIN`. `npm run hermes:doctor` (Task 5 Step 2) shows what is available before you start.

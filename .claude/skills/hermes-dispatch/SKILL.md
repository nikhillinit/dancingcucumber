---
name: hermes-dispatch
description: Dispatch a heavy task to the Hermes/Codex orchestrator using the correct Windows node invocation (npm swallows --task).
disable-model-invocation: true
---

# hermes-dispatch

Operator cheat-sheet for offloading heavy work to the Hermes -> Codex orchestrator on this Windows repo. Router lives at the global path `C:/Users/nikhi/.hermes/router/orchestrate.mjs`.

## 1. Preflight (always do this first)

Confirm routing before spending a real dispatch.

```powershell
npm run hermes:dry      # dry-run: shows what would dispatch, does nothing
npm run hermes:doctor   # doctor: checks router health / quotas / windows
```

## 2. npm wrappers (no task payload)

These wrap the global router but CANNOT carry a `--task`.

```powershell
npm run hermes:production   # node .../orchestrate.mjs --project C:/dev/AIHedgeFund --phase production
npm run hermes:research     # ... --phase research
npm run hermes:dry          # ... --dry-run
npm run hermes:doctor       # ... doctor
```

## 3. GOTCHA: `npm run` swallows `--task`

`npm run hermes:production -- --task "..."` does NOT work — npm eats the argument. To pass a task, call node DIRECTLY:

```powershell
node C:/Users/nikhi/.hermes/router/orchestrate.mjs --project C:/dev/AIHedgeFund --phase production --task "<short task here>"
```

## 4. GOTCHA: long task strings + the slash guard — use a file pointer

A large `--task` string overflows the Windows command-line length limit, and a PowerShell guard blocks task strings containing slashes. Slashes are NOT the blocker on their own — length and the guard are. FIX: write the full task text to a file, then dispatch a SHORT task that just points at it.

```powershell
# 1. Write the full task text to a file (scratchpad or ai-logs)
Set-Content -Path C:/dev/AIHedgeFund/ai-logs/hermes/task.txt -Value $taskText -Encoding utf8

# 2. Dispatch a short file-pointer task
node C:/Users/nikhi/.hermes/router/orchestrate.mjs --project C:/dev/AIHedgeFund --phase production --task "Execute the plan in ai-logs/hermes/task.txt"
```

Keep the inline `--task` short and slash-free; put paths, plans, and detail in the referenced file.

## 5. Prefer SOLO dispatch

Codex runs in a workspace-write sandbox with NO npm/node available inside it — do not ask Codex to run npm/node commands. Prefer SOLO dispatch: Kimi has been unreliable, and although the cp1252 issue was fixed in router 1.44.0, still prefer solo unless explicitly told otherwise.

## 6. Verify bulk git ops yourself

After any bulk git operation you asked Codex to perform (e.g. `git rm -r <dir>`), VERIFY Codex's real git state — it cherry-picks/skips bulk deletes. Do bulk directory deletes DIRECTLY instead of delegating them.

```powershell
git status
git log --oneline -5
```

## Quick reference

| Need | Command |
| --- | --- |
| Preflight routing | `npm run hermes:dry` / `npm run hermes:doctor` |
| Dispatch with task | `node C:/Users/nikhi/.hermes/router/orchestrate.mjs --project C:/dev/AIHedgeFund --phase production --task "..."` |
| Big/complex task | write to file, dispatch short file-pointer task |
| Research phase | swap `--phase production` -> `--phase research` |
| After Codex bulk git | `git status` + `git log --oneline` to confirm |

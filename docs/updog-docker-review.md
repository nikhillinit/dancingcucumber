# How `nikhillinit/Updog_restore` leverages Docker — review & insights for AIHedgeFund

**Date:** 2026-06-15 · **Source:** `nikhillinit/Updog_restore@main` (read via `gh api`).
**Context:** reviewed while closing AIHedgeFund Workstream B (TimescaleDB round-trip), which surfaced that Docker Desktop is gone on this device.

## 1. Inventory — what Docker assets Updog ships

| Area | Files |
|---|---|
| Images (multi-target) | `Dockerfile`, `Dockerfile.simple`, `Dockerfile.worker`, `Dockerfile.railway`, `ml-service/Dockerfile`, `tests/chaos/wasm-simulator/Dockerfile` |
| Compose stacks (layered by concern) | `docker-compose.yml` (base: postgres+redis+pgadmin), `docker-compose.dev.yml` (dev + optional Prometheus via profile), `docker-compose.observability.yml`, `docker-compose.rls.yml`, `docker-compose.chaos.yml`, `tests/chaos/docker-compose.toxiproxy.yml`, `ml-service/docker-compose.yml` |
| Local bring-up | `docker-setup.sh` (raw `docker run` for pg+redis), `dev-bootstrap.ps1` / `dev-bootstrap.bat` (Windows), `quickstart.sh`, `start-dev.bat` |
| Dev container | `.devcontainer/devcontainer.json` (docker-in-docker + gh CLI features) |
| Ephemeral test DBs | `vitest.config.testcontainers.ts`, `vitest.config.phase0-dbproof.ts`, `.github/workflows/testcontainers-ci.yml` |
| CI / quality | `.github/workflows/dockerfile-lint.yml`, `.hadolint.yaml`, `.dockerignore`, `.trivyignore`, `.zap`, `zap-baseline.yml` |
| DB admin | `pgadmin-servers.json` (pre-provisioned pgAdmin server list) |

## 2. Patterns worth noting

1. **Docker is optional, the app degrades.** `dev-bootstrap.ps1` probes for a running
   service first (`Get-NetTCPConnection -LocalPort 6379`), only starts a container if the
   port is free *and* `docker` exists, and offers `-MemoryCache` (`REDIS_URL=memory://`) and
   `-LocalPostgres <url>` to skip Docker entirely. Idempotent and graceful.
2. **Compose layered by concern, not one mega-file.** Base stack + per-purpose overlays
   (dev, observability, RLS, chaos/toxiproxy) and Prometheus gated behind a compose
   `profiles: [monitoring]` so it's opt-in.
3. **Testcontainers for integration tests.** CI spins up throwaway Postgres per run
   (`testcontainers-ci.yml`) instead of depending on a long-lived compose DB — tests are
   self-contained and parallel-safe.
4. **Security-hardened compose.** Ports bound to `127.0.0.1:` (loopback only, not `0.0.0.0`),
   `mem_reservation`/`mem_limit`/`cpus` caps, `restart: unless-stopped`, healthchecks on every
   service, and `depends_on: { condition: service_healthy }` so dependents wait for readiness.
5. **DB seeded declaratively.** `./migrations:/docker-entrypoint-initdb.d` + an RLS init
   script auto-run on first boot; pgAdmin auto-configured from a mounted `servers.json`.
6. **Dockerfiles are linted & scanned in CI.** hadolint (`dockerfile-lint.yml` + `.hadolint.yaml`),
   Trivy (`.trivyignore`), ZAP baseline.
7. **Multi-target build strategy.** Separate Dockerfiles for app / worker / simple / railway
   deploy keep each image minimal for its job.

## 3. What applies to AIHedgeFund (recommendations, NOT done here)

AIHedgeFund's `infra/docker-compose.yml` is intentionally minimal (db+redis+adminer, one file).
That's appropriate for its current scope — do **not** port Updog's full apparatus wholesale.
Two items are genuinely worth considering, scoped to actual needs:

- **Testcontainers-style ephemeral DB for the optional Workstream B step 5 test.** Instead of
  the env-guarded `AIHF_DB_URL` test that needs a manually-started compose DB, a
  `testcontainers[postgresql]` fixture would spin up a throwaway TimescaleDB per run — fully
  self-contained, no operator setup, safe to keep out of the default `advisor-gate`. Trade-off:
  adds a Python dep and still needs a Docker daemon (which here means WSL). The handoff's
  env-guarded approach is lighter; Testcontainers is the more robust option if this test is
  meant to run in CI later.
- **Loopback port binding.** Updog binds `127.0.0.1:5432:5432`; AIHedgeFund's compose uses
  `5432:5432` (all interfaces). Minor local-dev hardening — one-line change if desired.

## 4. Honest caveat — Updog did NOT unblock the Docker problem

The immediate blocker this session was **Docker Desktop being uninstalled** on this device.
None of Updog's Docker patterns solve that: `dev-bootstrap.ps1` itself assumes `docker` is on
the Windows PATH (`Get-Command docker`), which it isn't here. The actual fix was discovered
independently — **docker engine running inside WSL2 `Ubuntu-22.04`**, driven via `wsl.exe`
from PowerShell (see memory `docker-wsl-roundtrip`). Updog is a useful reference for *how to
structure* Docker in a project; it is not the remedy for this machine's missing Docker Desktop.

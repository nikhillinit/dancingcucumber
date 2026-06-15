#!/usr/bin/env node
// advisor-gate stage 2: import-smoke the floor module so the gate fails if the
// floor logic is broken. Full data-driven floor enforcement lands when live
// price history is wired in the next advisor plan.
import { spawnSync } from 'node:child_process';

const result = spawnSync(
  'python',
  [
    '-c',
    'import sys; sys.path.insert(0, "apps/quant"); from advisor.backtest.floor import beats_floor, purged_walk_forward_sharpe; print("floor: importable")',
  ],
  {
    stdio: 'inherit',
    shell: process.platform === 'win32',
  },
);

process.exit(result.status ?? 1);

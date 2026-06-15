#!/usr/bin/env node
// advisor-gate stage 2: data-driven floor. Backtests the deployed long-flat price-only
// ensemble vs SPY on the committed fixture and prints the real verdict every run.
// Report mode (default) exits 0 so dev commits are not blocked; --enforce exits non-zero
// on a floor miss (used by advisor-release-gate, gating PRODUCTION RELEASE only).
import { spawnSync } from 'node:child_process';

const args = ['tools/floor_data_check.py'];
if (process.argv.includes('--enforce')) args.push('--enforce');
const r = spawnSync('python', args, {
  stdio: 'inherit',
  shell: process.platform === 'win32',
});
process.exit(r.status ?? 1);

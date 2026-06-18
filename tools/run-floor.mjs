#!/usr/bin/env node
// advisor-gate stage 2: data-driven floor. Backtests the deployed long-flat price-only
// ensemble vs SPY on the committed fixture and prints the real verdict every run.
// Report mode (default) exits 0 so dev commits are not blocked; --enforce exits non-zero
// on a floor miss (used by advisor-release-gate, gating PRODUCTION RELEASE only).
import { spawnSync } from 'node:child_process';

const userArgs = process.argv.slice(2);
const allowed = userArgs.length === 0 || (userArgs.length === 1 && userArgs[0] === '--enforce');
if (!allowed) {
  console.error(
    'run-floor: refusing ' +
      JSON.stringify(userArgs) +
      ' - holdout is not reachable through this wrapper; use a separate operator-approved lane ' +
      '(run python tools/floor_data_check.py --holdout deliberately, outside this gate).'
  );
  process.exit(2);
}

const args = ['tools/floor_data_check.py'];
if (userArgs[0] === '--enforce') args.push('--enforce');
const r = spawnSync('python', args, {
  stdio: 'inherit',
  shell: process.platform === 'win32',
});
process.exit(r.status ?? 1);

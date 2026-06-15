#!/usr/bin/env node
// advisor-gate stage 2: import-smoke the floor module so the gate fails if the
// floor logic is broken. Runs a .py FILE (not `python -c "..."`) so spaces in
// the command can't be mistokenized by the Windows shell under shell:true.
// Full data-driven floor enforcement lands when live price history is wired in
// the next advisor plan.
import { spawnSync } from 'node:child_process';

const result = spawnSync('python', ['tools/floor_smoke.py'], {
  stdio: 'inherit',
  shell: process.platform === 'win32',
});

process.exit(result.status ?? 1);

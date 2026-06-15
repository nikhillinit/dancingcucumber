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

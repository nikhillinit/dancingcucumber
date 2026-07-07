import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { join } from 'node:path';
import { describe, test } from 'node:test';

import * as router from './orchestrate.mjs';

const {
  buildPrompt,
  chooseModel,
  createLiveRunStep,
  createRoutingPlan,
  evaluateReadiness,
  executeModelCapture,
  executeModelCommand,
  executeWorkflow,
  generateRunId,
  getGateRunPlan,
  isProductionFinancial,
  main,
  parseApprovalSignal,
  parseArgs,
  resolveEffectivePhase,
  resolveOwnership,
  shouldRunPostflightGate,
  scoreSpecialist,
  writeRunLedger,
} = router;

function routingFixture() {
  return {
    defaults: { research: 'claude', production: 'codex', distribution: 'claude' },
    ownership: {
      research: {
        owner: 'claude',
        reviewer: 'kimi',
        role: 'leader-coordinator',
        artifact: 'implementation brief',
      },
      production: {
        owner: 'codex',
        reviewer: 'claude',
        role: 'worker-executor',
        artifact: 'diff plus tests',
      },
      'production-financial': {
        owner: 'codex',
        reviewer: 'claude',
        role: 'worker-executor',
        specialistRequired: true,
        artifact: 'diff plus backtest-disclosure notes',
        humanApproval: true,
      },
      distribution: {
        owner: 'claude',
        repairOwner: 'codex',
        role: 'release-manager',
        artifact: 'PR-ready summary',
      },
    },
    debate: { comparators: ['claude', 'codex', 'kimi'], synthesis: 'claude' },
    longContextModel: 'kimi',
    longContextTriggers: ['full repo audit', 'repo-wide'],
    specialists: {
      'backtest-integrity': {
        keywords: [
          { phrase: 'backtest', weight: 4 },
          { phrase: 'walk-forward', weight: 4 },
          { phrase: 'sharpe', weight: 3 },
        ],
        risk: 'financial',
      },
      'test-repair': {
        keywords: [
          { phrase: 'failing test', weight: 4 },
          { phrase: 'flaky test', weight: 4 },
          { phrase: 'test regression', weight: 3 },
        ],
        risk: 'quality',
      },
      'ops-review': {
        keywords: [{ phrase: 'deployment', weight: 3 }],
        risk: 'operational',
      },
    },
    scoring: {
      minScoreToAssign: 3,
      riskOrder: ['financial', 'operational', 'quality'],
    },
    gates: {
      research: 'npm run doctor:quick',
      production: 'npm run check',
      'production-financial': 'npm run advisor-gate',
      distribution: 'npm run lint',
    },
    commands: {
      claude: { binEnv: 'CLAUDE_CODE_BIN', defaultBin: 'claude', args: ['-p'] },
      codex: {
        binEnv: 'CODEX_BIN',
        defaultBin: 'codex',
        args: ['exec', '--sandbox', 'workspace-write'],
      },
    },
  };
}

// Manual clock for tests and fakes. It never installs real timers or open handles.
function createFakeClock(start = '2026-07-07T19:04:07.684Z') {
  let currentMs = typeof start === 'number' ? start : new Date(start).getTime();
  const scheduled = [];

  function runDueJobs() {
    scheduled.sort((left, right) => left.at - right.at);
    while (scheduled.length > 0 && scheduled[0].at <= currentMs) {
      const job = scheduled.shift();
      if (!job.cancelled) {
        job.callback();
      }
      scheduled.sort((left, right) => left.at - right.at);
    }
  }

  return {
    now() {
      return new Date(currentMs);
    },
    nowMs() {
      return currentMs;
    },
    schedule(delayMs, callback) {
      const job = {
        at: currentMs + Number(delayMs || 0),
        callback,
        cancelled: false,
      };
      scheduled.push(job);
      return {
        cancel() {
          job.cancelled = true;
        },
      };
    },
    advance(ms) {
      currentMs += Number(ms || 0);
      runDueJobs();
    },
    pending() {
      return scheduled.filter((job) => !job.cancelled).length;
    },
  };
}

function toError(value, fallbackMessage = 'scripted error') {
  if (value instanceof Error) return value;
  const error = new Error(typeof value === 'string' ? value : fallbackMessage);
  if (value && typeof value === 'object') {
    Object.assign(error, value);
  }
  return error;
}

function createFakeChild({ pid, script = {}, clock }) {
  const child = new EventEmitter();
  const stdout = new EventEmitter();
  const stderr = new EventEmitter();
  const stdin = new EventEmitter();
  const stdinWrites = [];
  let stdinEnded = false;
  let flushed = false;
  let settled = false;

  function settle(callback) {
    if (settled) return;
    settled = true;
    callback();
  }

  function emitOutput() {
    for (const chunk of script.stdout ?? []) {
      stdout.emit('data', Buffer.from(String(chunk)));
    }
    for (const chunk of script.stderr ?? []) {
      stderr.emit('data', Buffer.from(String(chunk)));
    }
  }

  stdin.write = (chunk) => {
    stdinWrites.push(String(chunk));
    if (script.stdinErrorOnWrite) {
      const error = toError(script.stdinErrorOnWrite, 'stdin write failed');
      stdin.emit('error', error);
      return false;
    }
    return true;
  };
  stdin.end = () => {
    stdinEnded = true;
    if (script.stdinErrorOnEnd) {
      stdin.emit('error', toError(script.stdinErrorOnEnd, 'stdin end failed'));
    }
  };

  child.pid = pid;
  child.stdin = stdin;
  child.stdout = stdout;
  child.stderr = stderr;
  child.stdinWrites = stdinWrites;
  child.stdinEnded = () => stdinEnded;
  child.flushed = () => flushed;
  child.settled = () => settled;
  child.close = (code = script.code ?? 0) => {
    settle(() => child.emit('close', code, null));
  };
  child.fail = (error = script.error) => {
    settle(() => child.emit('error', toError(error, 'child process failed')));
  };
  child.flush = () => {
    if (flushed) return;
    flushed = true;
    emitOutput();

    if (script.error) {
      child.fail(script.error);
      return;
    }

    if (script.hang === true) return;

    if (script.hangForMs !== undefined || script.closeAfterMs !== undefined) {
      const delay = script.hangForMs ?? script.closeAfterMs;
      clock.schedule(delay, () => child.close(script.code ?? 0));
      return;
    }

    child.close(script.code ?? 0);
  };

  return child;
}

// Scripted spawn fake for executeModelCapture and main() dependency seams.
// Tests explicitly flush scripted children so no real timers or processes exist.
function createScriptedSpawn(scripts = [], { clock = createFakeClock(), startPid = 4000 } = {}) {
  const calls = [];
  const children = [];
  let nextPid = startPid;

  function spawnImpl(bin, args = [], options = {}) {
    const script = scripts[calls.length] ?? {};
    calls.push({ bin, args, options, script });
    if (script.spawnError) {
      throw toError(script.spawnError, 'spawn failed');
    }

    const child = createFakeChild({
      pid: script.pid ?? nextPid++,
      script,
      clock,
    });
    children.push(child);
    return child;
  }

  spawnImpl.calls = calls;
  spawnImpl.children = children;
  spawnImpl.flushNext = () => {
    const child = children.find((candidate) => !candidate.flushed());
    if (child) child.flush();
    return child;
  };
  spawnImpl.flushAll = () => {
    for (const child of children) {
      child.flush();
    }
  };

  return spawnImpl;
}

function createRecordingFs() {
  const calls = [];
  const files = new Map();
  const dirs = [];

  return {
    calls,
    files,
    dirs,
    mkdirSync(path, options) {
      calls.push({ method: 'mkdirSync', path, options });
      dirs.push({ path, options });
    },
    writeFileSync(path, contents) {
      calls.push({ method: 'writeFileSync', path, contents });
      files.set(path, String(contents));
    },
  };
}

function createRecordingKill() {
  const calls = [];
  const kill = (pid, signal = 'SIGTERM') => {
    calls.push({ pid, signal });
  };
  kill.calls = calls;
  return kill;
}

function captureIo() {
  let stdout = '';
  let stderr = '';

  return {
    io: {
      stdout: {
        write(chunk) {
          stdout += String(chunk);
        },
      },
      stderr: {
        write(chunk) {
          stderr += String(chunk);
        },
      },
    },
    output() {
      return { stdout, stderr };
    },
  };
}

function gateRunner(status = 0, calls = []) {
  return (bin, args, options) => {
    calls.push({ bin, args, options });
    return { status };
  };
}

function recordCommandExists(missingBins = []) {
  const missing = new Set(missingBins);
  const calls = [];
  const commandExists = (bin) => {
    calls.push(bin);
    return !missing.has(bin);
  };
  commandExists.calls = calls;
  return commandExists;
}

describe('Hermes router public surface', () => {
  test('exports the functions consumed by the router test foundation', () => {
    assert.deepEqual(
      [
        'Orchestrator',
        'assertFinancialGate',
        'buildDoctorReport',
        'buildPrompt',
        'chooseModel',
        'createLiveRunStep',
        'createWorkflowPlan',
        'createRoutingPlan',
        'evaluateReadiness',
        'executeModelCapture',
        'executeModelCommand',
        'executeWorkflow',
        'generateRunId',
        'getGateRunPlan',
        'isCliEntryPoint',
        'isProductionFinancial',
        'main',
        'parseApprovalSignal',
        'parseArgs',
        'recommendWorkflow',
        'resolveEffectivePhase',
        'resolveGate',
        'resolveOwnership',
        'runGate',
        'shouldRunPostflightGate',
        'scoreSpecialist',
        'writeRunLedger',
      ].sort(),
      Object.keys(router).sort()
    );
  });
});

describe('argument parsing and routing decisions', () => {
  test('parseArgs handles defaults, routing flags, and trim behavior', () => {
    assert.deepEqual(parseArgs([]), {
      phase: 'research',
      task: '',
      dryRun: false,
      json: false,
      help: false,
      skipPreflightGate: false,
      gateSkipReason: null,
      workflow: 'auto',
      workflowProvided: false,
      live: false,
      manualModel: null,
      legacyCommand: null,
    });

    assert.deepEqual(
      parseArgs([
        '--json',
        '--phase',
        'production',
        '--task',
        '  repair failing test  ',
        '--workflow',
        'pair',
        '--model',
        'codex',
        '--skip-preflight-gate',
        '--skip-reason',
        'local fixture only',
        '--live',
      ]),
      {
        phase: 'production',
        task: 'repair failing test',
        dryRun: false,
        json: true,
        help: false,
        skipPreflightGate: true,
        gateSkipReason: 'local fixture only',
        workflow: 'pair',
        workflowProvided: true,
        live: true,
        manualModel: 'codex',
        legacyCommand: null,
      }
    );

    assert.equal(parseArgs(['doctor']).legacyCommand, 'doctor');
    assert.equal(parseArgs(['--claude']).manualModel, 'claude');
  });

  test('parseArgs rejects unsafe or unknown controls', () => {
    assert.throws(
      () => parseArgs(['--skip-gates']),
      /Use --skip-preflight-gate with --skip-reason/
    );
    assert.throws(
      () => parseArgs(['--skip-preflight-gate']),
      /--skip-preflight-gate requires --skip-reason/
    );
    assert.throws(() => parseArgs(['--workflow', 'swarm']), /Unknown workflow/);
    assert.throws(() => parseArgs(['--model', 'llama']), /Unknown model/);
  });

  test('chooseModel honors manual and long-context routing before phase defaults', () => {
    const routing = routingFixture();

    assert.equal(chooseModel('small fix', 'production', routing), 'codex');
    assert.equal(chooseModel('repo-wide architecture scan', 'production', routing), 'kimi');
    assert.equal(chooseModel('small fix', 'production', routing, 'claude'), 'claude');
    assert.equal(chooseModel('unknown phase', 'maintenance', routing), 'claude');
  });

  test('scoreSpecialist selects weighted matches and uses risk order as a tiebreaker', () => {
    const routing = routingFixture();

    const financial = scoreSpecialist(
      'backtest walk-forward sharpe review',
      routing.specialists,
      routing.scoring
    );
    assert.equal(financial.name, 'backtest-integrity');
    assert.equal(financial.score, 11);
    assert.equal(financial.risk, 'financial');
    assert.deepEqual(financial.matched, ['backtest', 'walk-forward', 'sharpe']);

    const testRepair = scoreSpecialist(
      'repair a failing test regression',
      routing.specialists,
      routing.scoring
    );
    assert.equal(testRepair.name, 'test-repair');
    assert.equal(testRepair.score, 7);
    assert.equal(testRepair.confidence, 0.64);

    const tied = scoreSpecialist(
      'walk-forward deployment',
      {
        financial: { keywords: [{ phrase: 'walk-forward', weight: 3 }], risk: 'financial' },
        operational: { keywords: [{ phrase: 'deployment', weight: 3 }], risk: 'operational' },
      },
      routing.scoring
    );
    assert.equal(tied.name, 'financial');

    assert.equal(scoreSpecialist('plain docs update', routing.specialists, routing.scoring), null);
  });

  test('resolveEffectivePhase and resolveOwnership promote financial production work only', () => {
    const routing = routingFixture();
    const financialSpecialist = { name: 'backtest-integrity', risk: 'financial' };
    const qualitySpecialist = { name: 'test-repair', risk: 'quality' };

    assert.equal(resolveEffectivePhase('production', financialSpecialist), 'production-financial');
    assert.equal(resolveEffectivePhase('production', qualitySpecialist), 'production');
    assert.equal(resolveEffectivePhase('research', financialSpecialist), 'research');

    assert.deepEqual(resolveOwnership('production', financialSpecialist, routing.ownership), {
      effectivePhase: 'production-financial',
      owner: 'codex',
      reviewer: 'claude',
      role: 'worker-executor',
      specialistRequired: true,
      artifact: 'diff plus backtest-disclosure notes',
      humanApproval: true,
    });
    assert.equal(resolveOwnership('unknown', null, routing.ownership), null);
  });

  test('createRoutingPlan builds standard production and financial workflow plans', () => {
    const routing = routingFixture();
    const standard = createRoutingPlan({
      phase: 'production',
      task: 'add router tests',
      routing,
    });

    assert.equal(standard.model, 'codex');
    assert.equal(standard.risk, 'standard');
    assert.equal(standard.specialist, null);
    assert.equal(standard.gate, 'npm run check');
    assert.equal(standard.ownership.owner, 'codex');
    assert.equal(standard.ownership.artifact, 'diff plus tests');

    const financial = createRoutingPlan({
      phase: 'production',
      task: 'backtest walk-forward sharpe leakage review',
      routing,
      requestedWorkflow: 'auto',
      skipPreflightGate: true,
      gateSkipReason: 'unit-test fixture',
    });
    assert.equal(financial.specialist, 'backtest-integrity');
    assert.equal(financial.risk, 'financial');
    assert.equal(financial.gate, 'npm run advisor-gate');
    assert.equal(financial.ownership.effectivePhase, 'production-financial');
    assert.equal(financial.ownership.humanApproval, true);
    assert.equal(financial.workflow.selected, 'pair');
    assert.deepEqual(
      financial.workflow.steps.map((step) => step.role),
      ['owner', 'specialist', 'reviewer', 'gate']
    );
    assert.deepEqual(financial.gateSkip, { preflight: true, reason: 'unit-test fixture' });
  });
});

describe('prompt, approval, gate, and readiness helpers', () => {
  test('buildPrompt includes routing metadata, governance text, and run id', () => {
    const plan = createRoutingPlan({
      phase: 'production',
      task: 'repair failing test regression',
      routing: routingFixture(),
    });
    const prompt = buildPrompt({
      plan,
      brain: 'DEV_BRAIN fixture',
      soul: 'SOUL fixture',
      runId: 'hermes-2026-07-07T19-04-07-684Z',
    });

    assert.match(prompt, /You are operating inside AIHedgeFund\./);
    assert.match(prompt, /PHASE: production/);
    assert.match(prompt, /MODEL ROLE: codex/);
    assert.match(prompt, /OWNER: codex; reviewer: claude; role: worker-executor/);
    assert.match(prompt, /SPECIALIST: test-repair \(risk: quality; confidence: 64%\)/);
    assert.match(prompt, /REQUIRED GATE: npm run check/);
    assert.match(prompt, /RUN ID: hermes-2026-07-07T19-04-07-684Z/);
    assert.match(prompt, /--- DEV_BRAIN ---\nDEV_BRAIN fixture\n--- END DEV_BRAIN ---/);
    assert.match(prompt, /--- HERMES_SOUL ---\nSOUL fixture\n--- END HERMES_SOUL ---/);
    assert.match(prompt, /TASK: repair failing test regression/);
    assert.match(prompt, /Use \.claude\/DISCOVERY-MAP\.md and \.claude\/AGENT-DIRECTORY\.md/);
    assert.doesNotMatch(prompt, /handoff\.schema\.json/);
  });

  test('parseApprovalSignal accepts tolerant approvals and fails closed on rejection or ambiguity', () => {
    assert.equal(parseApprovalSignal('looks good\n**APPROVED**'), true);
    assert.equal(parseApprovalSignal('APPROVED.'), true);
    assert.equal(parseApprovalSignal('approved\nCHANGES REQUESTED: missing test'), false);
    assert.equal(parseApprovalSignal('no clear verdict'), false);
    assert.equal(parseApprovalSignal(''), false);
  });

  test('gate helpers model preflight, postflight, and financial-risk policy', () => {
    const standard = { phase: 'production', risk: 'standard', gate: 'npm run check' };
    const financial = { phase: 'production', risk: 'financial', gate: 'npm run advisor-gate' };
    const noGate = { phase: 'research', risk: 'standard', gate: null };

    assert.deepEqual(getGateRunPlan(standard), { preflight: true, postflight: true });
    assert.deepEqual(getGateRunPlan(standard, { skipPreflightGate: true }), {
      preflight: false,
      postflight: true,
    });
    assert.deepEqual(getGateRunPlan(noGate), { preflight: false, postflight: false });
    assert.equal(isProductionFinancial(financial), true);
    assert.equal(isProductionFinancial({ ...financial, phase: 'research' }), false);
    assert.equal(shouldRunPostflightGate(standard, 0, { postflight: true }), true);
    assert.equal(shouldRunPostflightGate(standard, 1, { postflight: true }), false);
    assert.equal(shouldRunPostflightGate(financial, 1, { postflight: true }), true);
    assert.equal(shouldRunPostflightGate(financial, 0, { postflight: false }), false);
  });

  test('evaluateReadiness reports invalid financial gates and execution failures', () => {
    assert.deepEqual(evaluateReadiness(null), {
      ready: false,
      reason: 'evaluateReadiness requires a plan object.',
    });
    assert.deepEqual(
      evaluateReadiness({ phase: 'production', risk: 'financial', gate: 'npm run check' }),
      {
        ready: false,
        reason:
          'Financial gate proof failed: production-financial plan must resolve gate to "npm run advisor-gate", got "npm run check".',
      }
    );
    assert.deepEqual(
      evaluateReadiness(
        { phase: 'production', risk: 'financial', gate: 'npm run advisor-gate' },
        { exitCode: 2 }
      ),
      { ready: false, reason: 'workflow exited with code 2' }
    );
    assert.deepEqual(
      evaluateReadiness(
        { phase: 'production', risk: 'standard', gate: 'npm run check' },
        { approved: false }
      ),
      { ready: false, reason: 'reviewer did not approve the artifact' }
    );
    assert.deepEqual(evaluateReadiness({ phase: 'production', risk: 'standard' }), {
      ready: true,
      reason: null,
    });
  });

  test('generateRunId is deterministic for an injected Date', () => {
    assert.equal(
      generateRunId(new Date('2026-07-07T19:04:07.684Z')),
      'hermes-2026-07-07T19-04-07-684Z'
    );
  });
});

describe('test fakes and DI seams', () => {
  test('writeRunLedger uses injected fs and records JSON without touching disk', () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const record = {
      runId: 'hermes-2026-07-07T19-04-07-684Z',
      exitCode: 0,
      plan: { phase: 'production' },
    };

    const file = writeRunLedger(record, { root, fs, envSecrets: [] });

    assert.equal(file, join(root, 'ai-logs', 'hermes', 'runs', `${record.runId}.json`));
    assert.deepEqual(fs.calls[0], {
      method: 'mkdirSync',
      path: join(root, 'ai-logs', 'hermes', 'runs'),
      options: { recursive: true },
    });
    assert.equal(fs.calls[1].method, 'writeFileSync');
    assert.equal(fs.calls[1].path, file);
    assert.deepEqual(JSON.parse(fs.files.get(file)), record);
    assert.equal(fs.files.get(file).endsWith('\n'), true);
  });

  test('writeRunLedger masks token, key-value, and injected env secret output bodies', () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const record = {
      runId: 'hermes-2026-07-07T19-04-07-684Z',
      workflow: 'pair',
      steps: [
        {
          role: 'owner',
          model: 'codex',
          attempt: 0,
          code: 0,
          output:
            'tokens sk-1234567890abcdef1234567890abcdef ghp_1234567890abcdef1234567890abcdef1234 AKIA1234567890ABCDEF',
        },
        {
          role: 'reviewer',
          model: 'claude',
          attempt: 0,
          code: 0,
          approved: true,
          output: 'API_KEY=raw-key-value\nplain env-secret-value-123',
        },
      ],
      gate: { command: 'npm run check', status: 0, skipped: false },
      exitCode: 0,
    };

    const file = writeRunLedger(record, {
      root,
      fs,
      envSecrets: ['env-secret-value-123'],
    });
    const written = JSON.parse(fs.files.get(file));

    assert.equal(
      written.steps[0].output,
      'tokens [SCRUBBED:token] [SCRUBBED:token] [SCRUBBED:token]'
    );
    assert.equal(written.steps[0].scrubCount, 3);
    assert.equal(
      written.steps[1].output,
      'API_KEY=[SCRUBBED:keyvalue]\nplain [SCRUBBED:envvalue]'
    );
    assert.equal(written.steps[1].scrubCount, 2);
    assert.equal(written.steps[1].role, 'reviewer');
    assert.equal(written.gate.command, 'npm run check');
  });

  test('writeRunLedger fail-closes one throwing output body without dropping metadata', () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const record = {
      runId: 'hermes-2026-07-07T19-04-07-684Z',
      steps: [
        {
          role: 'owner',
          model: 'codex',
          attempt: 1,
          code: 7,
          output: 'secret-bearing output',
        },
      ],
      availability: [{ provider: 'codex', bin: 'codex', found: true }],
      gate: { command: 'npm run check', status: 7, skipped: false },
      exitCode: 7,
    };

    const file = writeRunLedger(record, {
      root,
      fs,
      envSecrets: [],
      scrubber() {
        throw new Error('scrubber failed');
      },
    });
    const written = JSON.parse(fs.files.get(file));

    assert.equal(written.steps[0].output, '[SCRUB-ERROR: output withheld]');
    assert.equal(written.steps[0].scrubCount, 0);
    assert.equal(written.steps[0].scrubError, true);
    assert.equal(written.steps[0].role, 'owner');
    assert.equal(written.steps[0].model, 'codex');
    assert.equal(written.steps[0].code, 7);
    assert.deepEqual(written.availability, [{ provider: 'codex', bin: 'codex', found: true }]);
    assert.deepEqual(written.gate, { command: 'npm run check', status: 7, skipped: false });
  });

  test('large output and embedded token-like substrings scrub without overmatching', () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const embedded = 'prefixsk-1234567890abcdef1234567890abcdefsuffix';
    const record = {
      runId: 'hermes-2026-07-07T19-04-07-684Z',
      steps: [
        {
          role: 'owner',
          model: 'codex',
          attempt: 0,
          code: 0,
          output: `${'x'.repeat(100000)}\n${embedded}\nTOKEN: large-secret-value-123`,
        },
      ],
    };

    const file = writeRunLedger(record, { root, fs, envSecrets: [] });
    const written = JSON.parse(fs.files.get(file));

    assert.match(written.steps[0].output, new RegExp(embedded));
    assert.match(written.steps[0].output, /TOKEN: \[SCRUBBED:keyvalue\]/);
    assert.equal(written.steps[0].scrubCount, 1);
  });

  test('scripted spawn captures prompt writes and stdout chunks for executeModelCapture', async () => {
    const clock = createFakeClock();
    const spawn = createScriptedSpawn(
      [{ stdout: ['first ', 'second'], stderr: ['ignored'], code: 0, pid: 4321 }],
      { clock }
    );
    const routing = routingFixture();
    const env = { CODEX_BIN: process.execPath, KEEP: 'yes', OPENAI_API_KEY: 'secret' };

    const pending = executeModelCapture('codex', 'prompt body', routing, env, { spawn });
    const child = spawn.flushNext();
    const result = await pending;

    assert.deepEqual(result, { code: 0, output: 'first second' });
    assert.equal(spawn.calls.length, 1);
    assert.equal(spawn.calls[0].bin, process.execPath);
    assert.deepEqual(spawn.calls[0].args, ['exec', '--sandbox', 'workspace-write']);
    assert.notEqual(spawn.calls[0].options.env, env);
    assert.deepEqual(spawn.calls[0].options.env, {
      CODEX_BIN: process.execPath,
      KEEP: 'yes',
    });
    assert.equal(env.OPENAI_API_KEY, 'secret');
    assert.equal(child.pid, 4321);
    assert.deepEqual(child.stdinWrites, ['prompt body']);
    assert.equal(child.stdinEnded(), true);
  });

  test('scripted spawn can delay close through the injected clock only', async () => {
    const clock = createFakeClock();
    const spawn = createScriptedSpawn([{ stdout: ['delayed'], code: 5, hangForMs: 250 }], {
      clock,
    });

    const pending = executeModelCapture(
      'codex',
      'prompt',
      routingFixture(),
      { CODEX_BIN: process.execPath },
      { spawn }
    );
    const child = spawn.flushNext();
    assert.equal(child.settled(), false);
    assert.equal(clock.pending(), 1);

    clock.advance(249);
    assert.equal(child.settled(), false);
    clock.advance(1);

    assert.deepEqual(await pending, { code: 5, output: 'delayed' });
    assert.equal(clock.pending(), 0);
  });

  test('executeModelCapture rejects child error events from the spawn seam', async () => {
    const spawn = createScriptedSpawn([
      { error: Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' }) },
      { error: Object.assign(new Error('spawn ENOENT'), { code: 'ENOENT' }) },
    ]);

    const pending = executeModelCapture(
      'codex',
      'prompt',
      routingFixture(),
      { CODEX_BIN: process.execPath },
      { spawn }
    );
    spawn.flushNext();
    await Promise.resolve();
    spawn.flushNext();

    await assert.rejects(pending, /spawn ENOENT/);
    assert.equal(spawn.calls.length, 2);
  });

  test('executeModelCapture rejects synchronous spawn failures from the spawn seam', async () => {
    const spawn = createScriptedSpawn([
      { spawnError: Object.assign(new Error('spawn failed'), { code: 'ENOENT' }) },
      { spawnError: Object.assign(new Error('spawn failed'), { code: 'ENOENT' }) },
    ]);

    await assert.rejects(
      executeModelCapture(
        'codex',
        'prompt',
        routingFixture(),
        { CODEX_BIN: process.execPath },
        { spawn }
      ),
      /spawn failed/
    );
    assert.equal(spawn.calls.length, 2);
  });

  test('recording kill fake stores pid and signal for later timeout assertions', () => {
    const kill = createRecordingKill();

    kill(4321, 'SIGKILL');

    assert.deepEqual(kill.calls, [{ pid: 4321, signal: 'SIGKILL' }]);
  });
});

describe('main dependency seams', () => {
  test('main prints JSON routing without executing gates, models, or ledgers', async () => {
    const captured = captureIo();
    const code = await main(
      ['--json', '--phase', 'production', '--task', 'add router tests'],
      {},
      captured.io,
      {
        routing: routingFixture(),
        brain: 'DEV_BRAIN fixture',
        soul: 'SOUL fixture',
        executeModel() {
          throw new Error('model execution should not run for --json');
        },
        gateRunner() {
          throw new Error('gate should not run for --json');
        },
        writeRunLedger() {
          throw new Error('ledger should not be written for --json');
        },
        clock: () => new Date('2026-07-07T19:04:07.684Z'),
      }
    );
    const { stdout, stderr } = captured.output();
    const plan = JSON.parse(stdout);

    assert.equal(code, 0);
    assert.equal(stderr, '');
    assert.equal(plan.phase, 'production');
    assert.equal(plan.task, 'add router tests');
    assert.equal(plan.model, 'codex');
    assert.equal(plan.gate, 'npm run check');
    assert.equal(plan.ownership.owner, 'codex');
  });

  test('main aborts on preflight gate failure and records the ledger through deps', async () => {
    const captured = captureIo();
    const gateCalls = [];
    const ledgers = [];
    const code = await main(
      ['--phase', 'production', '--task', 'add router tests'],
      {},
      captured.io,
      {
        routing: routingFixture(),
        brain: 'DEV_BRAIN fixture',
        soul: 'SOUL fixture',
        gateRunner: gateRunner(7, gateCalls),
        commandExists: recordCommandExists(),
        executeModel() {
          throw new Error('model execution should abort after preflight failure');
        },
        writeRunLedger(record) {
          ledgers.push(record);
        },
        clock: () => new Date('2026-07-07T19:04:07.684Z'),
      }
    );

    assert.equal(code, 7);
    assert.equal(captured.output().stdout, '');
    assert.equal(captured.output().stderr, '');
    assert.equal(gateCalls.length, 1);
    assert.equal(gateCalls[0].bin, 'npm');
    assert.deepEqual(gateCalls[0].args, ['run', 'check']);
    assert.equal(ledgers.length, 1);
    assert.equal(ledgers[0].runId, 'hermes-2026-07-07T19-04-07-684Z');
    assert.deepEqual(ledgers[0].preflight, {
      command: 'npm run check',
      status: 7,
      skipped: false,
    });
    assert.equal(ledgers[0].model, null);
    assert.equal(ledgers[0].exitCode, 7);
  });

  test('main can skip only preflight and still run model plus postflight gate', async () => {
    const captured = captureIo();
    const gateCalls = [];
    const ledgers = [];
    const modelPrompts = [];
    const commandExists = recordCommandExists(['missing-codex']);
    const code = await main(
      [
        '--phase',
        'production',
        '--task',
        'add router tests',
        '--skip-preflight-gate',
        '--skip-reason',
        'test fixture',
      ],
      { CODEX_BIN: 'missing-codex' },
      captured.io,
      {
        routing: routingFixture(),
        brain: 'DEV_BRAIN fixture',
        soul: 'SOUL fixture',
        gateRunner: gateRunner(0, gateCalls),
        commandExists,
        executeModel(model, prompt) {
          modelPrompts.push({ model, prompt });
          return 0;
        },
        writeRunLedger(record) {
          ledgers.push(record);
        },
        clock: () => new Date('2026-07-07T19:04:07.684Z'),
      }
    );

    assert.equal(code, 0);
    assert.match(captured.output().stderr, /WARNING: skipping preflight gate "npm run check"/);
    assert.equal(gateCalls.length, 1);
    assert.equal(modelPrompts.length, 1);
    assert.equal(modelPrompts[0].model, 'codex');
    assert.match(modelPrompts[0].prompt, /TASK: add router tests/);
    assert.equal(ledgers.length, 1);
    assert.deepEqual(ledgers[0].availability, [
      { provider: 'claude', bin: 'claude', found: true },
      { provider: 'codex', bin: 'missing-codex', found: false },
    ]);
    assert.deepEqual(commandExists.calls, ['claude', 'missing-codex']);
    assert.deepEqual(ledgers[0].preflight, {
      command: 'npm run check',
      status: null,
      skipped: true,
      reason: 'test fixture',
    });
    assert.deepEqual(ledgers[0].model, { name: 'codex', exitCode: 0 });
    assert.deepEqual(ledgers[0].postflight, { command: 'npm run check', status: 0 });
  });

  test('main records provider availability for live workflow ledgers', async () => {
    const captured = captureIo();
    const gateCalls = [];
    const ledgers = [];
    const stepCalls = [];
    const commandExists = recordCommandExists(['missing-codex']);
    const code = await main(
      ['--workflow', 'pair', '--live', '--phase', 'production', '--task', 'add router tests'],
      { CODEX_BIN: 'missing-codex' },
      captured.io,
      {
        routing: routingFixture(),
        brain: 'DEV_BRAIN fixture',
        soul: 'SOUL fixture',
        gateRunner: gateRunner(0, gateCalls),
        commandExists,
        async runStep({ step }) {
          stepCalls.push(step.role);
          return { code: 0, approved: true, output: `${step.role} output` };
        },
        writeRunLedger(record) {
          ledgers.push(record);
        },
        clock: () => new Date('2026-07-07T19:04:07.684Z'),
      }
    );

    assert.equal(code, 0);
    assert.equal(captured.output().stderr, '');
    assert.deepEqual(stepCalls, ['owner', 'reviewer']);
    assert.equal(gateCalls.length, 2);
    assert.equal(ledgers.length, 1);
    assert.deepEqual(ledgers[0].availability, [
      { provider: 'claude', bin: 'claude', found: true },
      { provider: 'codex', bin: 'missing-codex', found: false },
    ]);
    assert.deepEqual(commandExists.calls, ['claude', 'missing-codex']);
    assert.equal(ledgers[0].exitCode, 0);
  });

  test('createLiveRunStep scrubs prior output before composing the next prompt', async () => {
    const prompts = [];
    const runStep = createLiveRunStep({
      routing: routingFixture(),
      basePrompt: 'BASE PROMPT',
      envSecrets: ['env-secret-value-123'],
      executor(_model, prompt) {
        prompts.push(prompt);
        return { code: 0, output: 'looks good\nAPPROVED' };
      },
    });

    const result = await runStep({
      step: { role: 'reviewer', model: 'claude', action: 'review diff' },
      input: 'prior sk-1234567890abcdef1234567890abcdef and env-secret-value-123',
    });

    assert.equal(result.approved, true);
    assert.equal(prompts.length, 1);
    assert.match(prompts[0], /PRIOR OUTPUT:\nprior \[SCRUBBED:token\] and \[SCRUBBED:envvalue\]/);
    assert.doesNotMatch(prompts[0], /sk-1234567890abcdef1234567890abcdef/);
    assert.doesNotMatch(prompts[0], /env-secret-value-123/);
  });

  test('executeWorkflow sends scrubbed step outputs to injected ledger writers', async () => {
    const ledgers = [];
    const plan = {
      phase: 'production',
      risk: 'standard',
      gate: null,
      workflow: {
        selected: 'pair',
        steps: [
          { role: 'owner', model: 'codex', action: 'execute' },
          { role: 'reviewer', model: 'claude', action: 'review' },
        ],
      },
    };

    const record = await executeWorkflow(plan, {
      envSecrets: ['env-secret-value-123'],
      async runStep({ step }) {
        if (step.role === 'owner') {
          return { code: 0, output: 'owner leaked env-secret-value-123' };
        }
        return { code: 0, approved: true, output: 'reviewer approved' };
      },
      writeRunLedger(ledger) {
        ledgers.push(ledger);
      },
      clock: () => new Date('2026-07-07T19:04:07.684Z'),
    });

    assert.equal(record.steps[0].output, 'owner leaked env-secret-value-123');
    assert.equal(ledgers.length, 1);
    assert.equal(ledgers[0].steps[0].output, 'owner leaked [SCRUBBED:envvalue]');
    assert.equal(ledgers[0].steps[0].scrubCount, 1);
    assert.equal(ledgers[0].steps[1].scrubCount, 0);
  });
});

describe('future timeout, retry, and environment contracts', () => {
  function routingWithTimeout(timeoutMs) {
    const routing = routingFixture();
    routing.commands.codex = { ...routing.commands.codex, timeoutMs };
    return routing;
  }

  test('per-command timeout kills the child and classifies failure as timeout', async () => {
    const clock = createFakeClock();
    const killTree = createRecordingKill();
    const spawn = createScriptedSpawn([{ hang: true, pid: 4321 }], { clock });

    const pending = executeModelCommand(
      'codex',
      'prompt',
      routingWithTimeout(50),
      { CODEX_BIN: process.execPath },
      { spawn, killTree, clock, captureOutput: true }
    );
    spawn.flushNext();
    clock.advance(49);
    assert.equal(killTree.calls.length, 0);
    clock.advance(1);

    const result = await pending;

    assert.equal(result.code, 124);
    assert.equal(result.failure, 'timeout');
    assert.equal(result.output, '');
    assert.equal(result.stderr, '');
    assert.deepEqual(killTree.calls, [{ pid: 4321, signal: 'SIGTERM' }]);
    assert.equal(spawn.calls.length, 1);
  });

  test('Windows tree-kill seam receives the timed-out child pid', async () => {
    const clock = createFakeClock();
    const killTree = createRecordingKill();
    const spawn = createScriptedSpawn([{ hang: true, pid: 2468 }], { clock });

    const pending = executeModelCommand(
      'codex',
      'prompt',
      routingWithTimeout(10),
      { CODEX_BIN: process.execPath },
      { spawn, killTree, clock, captureOutput: true }
    );
    spawn.flushNext();
    clock.advance(10);

    const result = await pending;

    assert.equal(result.failure, 'timeout');
    assert.deepEqual(killTree.calls, [{ pid: 2468, signal: 'SIGTERM' }]);
  });

  test('env sanitization strips API key and credential blocklist without mutating caller env', async () => {
    const clock = createFakeClock();
    const spawn = createScriptedSpawn([{ stdout: ['ok'], code: 0 }], { clock });
    const env = {
      CODEX_BIN: process.execPath,
      OPENAI_API_KEY: 'openai-secret',
      ANTHROPIC_API_KEY: 'anthropic-secret',
      GOOGLE_APPLICATION_CREDENTIALS: 'google-creds.json',
      GH_TOKEN: 'gh-token',
      GITHUB_TOKEN: 'github-token',
      KEEP_ME: 'keep',
    };

    const pending = executeModelCommand(
      'codex',
      'prompt',
      routingFixture(),
      env,
      { spawn, clock, captureOutput: true }
    );
    spawn.flushNext();
    const result = await pending;
    const childEnv = spawn.calls[0].options.env;

    assert.equal(result.failure, null);
    assert.notEqual(childEnv, env);
    assert.equal('OPENAI_API_KEY' in childEnv, false);
    assert.equal('ANTHROPIC_API_KEY' in childEnv, false);
    assert.equal('GOOGLE_APPLICATION_CREDENTIALS' in childEnv, false);
    assert.equal(childEnv.GH_TOKEN, 'gh-token');
    assert.equal(childEnv.GITHUB_TOKEN, 'github-token');
    assert.equal(childEnv.KEEP_ME, 'keep');
    assert.equal(env.OPENAI_API_KEY, 'openai-secret');
    assert.equal(env.ANTHROPIC_API_KEY, 'anthropic-secret');
    assert.equal(env.GOOGLE_APPLICATION_CREDENTIALS, 'google-creds.json');
  });

  test('failure classification enum covers spawn_error, timeout, nonzero_exit, empty_output, rate_limited', async () => {
    async function runClosed(script, options = {}) {
      const clock = createFakeClock();
      const spawn = createScriptedSpawn([script], { clock });
      const pending = executeModelCommand(
        'codex',
        'prompt',
        routingWithTimeout(options.timeoutMs ?? 100),
        { CODEX_BIN: process.execPath },
        {
          spawn,
          killTree: options.killTree ?? createRecordingKill(),
          clock,
          captureOutput: true,
          rateLimitSignatures: options.rateLimitSignatures,
        }
      );
      spawn.flushNext();
      if (options.advanceMs !== undefined) {
        clock.advance(options.advanceMs);
      }
      return pending;
    }

    const spawnErrorSpawn = createScriptedSpawn([
      { spawnError: Object.assign(new Error('spawn failed'), { code: 'ENOENT' }) },
      { spawnError: Object.assign(new Error('spawn failed'), { code: 'ENOENT' }) },
    ]);
    const spawnError = await executeModelCommand(
      'codex',
      'prompt',
      routingFixture(),
      { CODEX_BIN: process.execPath },
      { spawn: spawnErrorSpawn, captureOutput: true }
    );
    const timeout = await runClosed({ hang: true }, { advanceMs: 100, timeoutMs: 100 });
    const nonzero = await runClosed({ stdout: ['partial output'], code: 7 });
    const empty = await runClosed({ stdout: ['   \n\t'], code: 0 });
    const rateLimited = await runClosed(
      { stdout: [''], stderr: ['provider says retry later'], code: 7 },
      { rateLimitSignatures: ['retry later'] }
    );

    assert.deepEqual(
      [
        spawnError.failure,
        timeout.failure,
        nonzero.failure,
        empty.failure,
        rateLimited.failure,
      ],
      ['spawn_error', 'timeout', 'nonzero_exit', 'empty_output', 'rate_limited']
    );
    assert.equal(spawnErrorSpawn.calls.length, 2);
    assert.equal(rateLimited.code, 7);
  });

  test('settle-once prevents duplicate side effects when timeout-kill is followed by close', async () => {
    const clock = createFakeClock();
    const killTree = createRecordingKill();
    const spawn = createScriptedSpawn([{ hang: true, pid: 9876 }], { clock });
    let settlements = 0;

    const pending = executeModelCommand(
      'codex',
      'prompt',
      routingWithTimeout(25),
      { CODEX_BIN: process.execPath },
      { spawn, killTree, clock, captureOutput: true }
    ).then((result) => {
      settlements += 1;
      return result;
    });
    const child = spawn.flushNext();
    clock.advance(25);
    child.close(0);

    const result = await pending;

    assert.equal(result.failure, 'timeout');
    assert.equal(result.code, 124);
    assert.equal(settlements, 1);
    assert.deepEqual(killTree.calls, [{ pid: 9876, signal: 'SIGTERM' }]);
    assert.equal(clock.pending(), 0);
  });

  test('bounded retry retries spawn_error exactly once and never retries timeout', async () => {
    const retryClock = createFakeClock();
    const retrySpawn = createScriptedSpawn(
      [
        { spawnError: Object.assign(new Error('spawn failed'), { code: 'ENOENT' }) },
        { stdout: ['ok'], code: 0, pid: 5555 },
      ],
      { clock: retryClock }
    );

    const retried = executeModelCommand(
      'codex',
      'prompt',
      routingFixture(),
      { CODEX_BIN: process.execPath },
      { spawn: retrySpawn, clock: retryClock, captureOutput: true }
    );
    await Promise.resolve();
    await Promise.resolve();
    retrySpawn.flushNext();

    assert.deepEqual(await retried, {
      code: 0,
      output: 'ok',
      stderr: '',
      failure: null,
      stdinError: null,
    });
    assert.equal(retrySpawn.calls.length, 2);

    const timeoutClock = createFakeClock();
    const timeoutKill = createRecordingKill();
    const timeoutSpawn = createScriptedSpawn([{ hang: true, pid: 6666 }], {
      clock: timeoutClock,
    });
    const timedOut = executeModelCommand(
      'codex',
      'prompt',
      routingWithTimeout(30),
      { CODEX_BIN: process.execPath },
      { spawn: timeoutSpawn, killTree: timeoutKill, clock: timeoutClock, captureOutput: true }
    );
    timeoutSpawn.flushNext();
    timeoutClock.advance(30);

    assert.equal((await timedOut).failure, 'timeout');
    assert.equal(timeoutSpawn.calls.length, 1);
    assert.deepEqual(timeoutKill.calls, [{ pid: 6666, signal: 'SIGTERM' }]);
  });

  test('large prompt to fast-exiting child does not throw uncaught EPIPE or ECONNRESET', async () => {
    const observed = [];
    const onUncaught = (error) => observed.push(error);
    const onUnhandled = (error) => observed.push(error);
    process.on('uncaughtException', onUncaught);
    process.on('unhandledRejection', onUnhandled);

    try {
      for (const errorCode of ['EPIPE', 'ECONNRESET']) {
        const clock = createFakeClock();
        const spawn = createScriptedSpawn(
          [
            {
              stdinErrorOnWrite: Object.assign(new Error(errorCode), { code: errorCode }),
              code: 1,
            },
          ],
          { clock }
        );
        const pending = executeModelCommand(
          'codex',
          'x'.repeat(1024 * 1024),
          routingFixture(),
          { CODEX_BIN: process.execPath },
          { spawn, clock, captureOutput: true }
        );
        spawn.flushNext();
        const result = await pending;

        assert.equal(result.failure, 'nonzero_exit');
        assert.equal(result.stdinError.code, errorCode);
      }
      assert.deepEqual(observed, []);
    } finally {
      process.off('uncaughtException', onUncaught);
      process.off('unhandledRejection', onUnhandled);
    }
  });
});

import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { join } from 'node:path';
import { describe, test } from 'node:test';

import * as router from './orchestrate.mjs';

const {
  appendRunEvent,
  buildPrompt,
  chooseModel,
  createLiveRunStep,
  createWorkflowPlan,
  createRoutingPlan,
  evaluateReadiness,
  executeModelCapture,
  executeModelCommand,
  executeWorkflow,
  generateRunId,
  getGateRunPlan,
  isReviewLaneEligible,
  isProductionFinancial,
  main,
  parseApprovalSignal,
  parseArgs,
  resolveSandboxArgs,
  resolveEffectivePhase,
  resolveOwnership,
  shouldRunPostflightGate,
  scoreSpecialist,
  validateDebateRunRecordShape,
  writeDebateRunRecord,
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
        readOnlyArgs: ['exec', '--sandbox', 'read-only'],
      },
    },
  };
}

function routingWithLadder({ enabled = true, rungs = ['claude', 'codex', 'kimi', 'agy'] } = {}) {
  const routing = routingFixture();
  routing.ladder = { enabled, rungs };
  routing.commands = {
    ...routing.commands,
    kimi: {
      binEnv: 'KIMI_CODE_BIN',
      defaultBin: 'kimi-cli',
      args: ['--print', '--input-format', 'text', '--final-message-only'],
    },
    agy: { binEnv: 'AGY_BIN', defaultBin: 'agy', args: [] },
    gemini: {
      binEnv: 'GEMINI_BIN',
      defaultBin: 'gemini',
      args: ['--output-format', 'text', '--yolo', '--prompt='],
      readOnlyArgs: ['--output-format', 'text', '--prompt='],
    },
  };
  return routing;
}

function routingWithRateLimits({ enabled = true, ladderEnabled = false, defaults = {} } = {}) {
  const routing = routingWithLadder({ enabled: ladderEnabled, rungs: ['claude', 'codex', 'kimi', 'agy'] });
  routing.rateLimits = {
    enabled,
    signatures: {
      claude: { source: 'usage limit|rate limit|overloaded', flags: 'i' },
      codex: { source: 'rate.?limit|429|usage cap|quota', flags: 'i' },
      agy: { source: 'RESOURCE_EXHAUSTED|quota|429', flags: 'i' },
      kimi: { source: 'rate.?limit|quota|429', flags: 'i' },
    },
    defaults: {
      fallbackMinutes: 60,
      providers: {
        claude: 60,
        kimi: 60,
        agy: 60,
        codex: 360,
        ...(defaults.providers || {}),
      },
      ...defaults,
    },
  };
  return routing;
}

function providerStateFile(root) {
  return join(root, 'ai-logs', 'hermes', 'provider-state.json');
}

function readProviderState(fs, root) {
  return JSON.parse(fs.files.get(providerStateFile(root)));
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
    appendFileSync(path, contents) {
      calls.push({ method: 'appendFileSync', path, contents });
      files.set(path, `${files.get(path) || ''}${String(contents)}`);
    },
    renameSync(from, to) {
      calls.push({ method: 'renameSync', from, to });
      if (!files.has(from)) {
        const error = new Error(`ENOENT: no such file or directory, rename '${from}' -> '${to}'`);
        error.code = 'ENOENT';
        throw error;
      }
      files.set(to, files.get(from));
      files.delete(from);
    },
    existsSync(path) {
      calls.push({ method: 'existsSync', path });
      return files.has(path);
    },
    readFileSync(path) {
      calls.push({ method: 'readFileSync', path });
      if (!files.has(path)) {
        const error = new Error(`ENOENT: no such file or directory, open '${path}'`);
        error.code = 'ENOENT';
        throw error;
      }
      return files.get(path);
    },
    readdirSync(path) {
      calls.push({ method: 'readdirSync', path });
      const prefix = path.endsWith('\\') || path.endsWith('/') ? path : `${path}\\`;
      return [...files.keys()]
        .filter((file) => file.startsWith(prefix))
        .map((file) => file.slice(prefix.length))
        .filter((name) => name && !name.includes('\\') && !name.includes('/'));
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

function parseJsonl(contents) {
  return String(contents || '')
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
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
        'appendRunEvent',
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
        'isReviewLaneEligible',
        'main',
        'parseApprovalSignal',
        'parseArgs',
        'recommendWorkflow',
        'resolveSandboxArgs',
        'resolveEffectivePhase',
        'resolveGate',
        'resolveOwnership',
        'runGate',
        'shouldRunPostflightGate',
        'scoreSpecialist',
        'validateDebateRunRecordShape',
        'writeDebateRunRecord',
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
      allowFallback: false,
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
        '--allow-fallback',
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
        allowFallback: true,
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

  test('createWorkflowPlan inserts debate rebuttal between comparators and synthesis', () => {
    const workflow = createWorkflowPlan({
      requestedWorkflow: 'debate',
      phase: 'production',
      model: 'codex',
      ownership: { effectivePhase: 'production', artifact: 'diff plus tests' },
      gate: null,
      debate: { comparators: ['claude', 'codex'], synthesis: 'claude' },
    });

    assert.deepEqual(
      workflow.steps.map((step) => step.role),
      ['comparator', 'comparator', 'rebuttal', 'synthesis']
    );
    assert.deepEqual(
      workflow.steps.map((step) => step.model),
      ['claude', 'codex', 'claude', 'claude']
    );
    assert.equal(
      workflow.steps[2].action,
      'verify and rebut comparator critiques against ground truth'
    );
  });

  test('resolveSandboxArgs makes non-owner codex lanes read-only and fails unknown roles closed', () => {
    const routing = routingWithLadder();

    for (const role of [
      'reviewer',
      'comparator',
      'synthesis',
      'specialist',
      'audit',
      'blind',
      'redteam',
      'rebuttal',
    ]) {
      const args = resolveSandboxArgs(routing, 'codex', role);
      assert.deepEqual(args, ['exec', '--sandbox', 'read-only']);
      assert.equal(args.includes('workspace-write'), false);
    }

    assert.deepEqual(resolveSandboxArgs(routing, 'codex', 'unexpected-role'), [
      'exec',
      '--sandbox',
      'read-only',
    ]);
  });

  test('resolveSandboxArgs preserves owner write flags and read-only-by-construction args', () => {
    const routing = routingWithLadder();

    assert.deepEqual(resolveSandboxArgs(routing, 'codex', 'owner'), [
      'exec',
      '--sandbox',
      'workspace-write',
    ]);
    assert.deepEqual(resolveSandboxArgs(routing, 'gemini', 'owner'), [
      '--output-format',
      'text',
      '--yolo',
      '--prompt=',
    ]);
    assert.deepEqual(resolveSandboxArgs(routing, 'gemini', 'reviewer'), [
      '--output-format',
      'text',
      '--prompt=',
    ]);
    assert.deepEqual(resolveSandboxArgs(routing, 'claude', 'reviewer'), ['-p']);
    assert.deepEqual(resolveSandboxArgs(routing, 'kimi', 'reviewer'), [
      '--print',
      '--input-format',
      'text',
      '--final-message-only',
    ]);
    assert.deepEqual(resolveSandboxArgs(routing, 'agy', 'reviewer'), []);
  });

  test('isReviewLaneEligible excludes unproven or auth-dead models from deliberation lanes', () => {
    assert.equal(isReviewLaneEligible('agy'), false);
    assert.equal(isReviewLaneEligible('gemini'), false);
    assert.equal(isReviewLaneEligible('claude'), true);
    assert.equal(isReviewLaneEligible('codex'), true);
    assert.equal(isReviewLaneEligible('kimi'), true);
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

  test('plan-build ladder skips a missing first rung and annotates the selected step', () => {
    const routing = routingWithLadder({ rungs: ['claude', 'codex'] });
    const commandExists = recordCommandExists(['claude']);

    const plan = createRoutingPlan({
      phase: 'research',
      task: 'plain docs update',
      routing,
      requestedWorkflow: 'solo',
      commandExists,
    });

    const owner = plan.workflow.steps.find((step) => step.role === 'owner');
    assert.equal(plan.requestedModel, 'claude');
    assert.equal(plan.selectedModel, 'codex');
    assert.equal(owner.requestedModel, 'claude');
    assert.equal(owner.selectedModel, 'codex');
    assert.equal(owner.model, 'codex');
    assert.equal(owner.degradeReason, 'claude unavailable (command missing)');
    assert.deepEqual(owner.providerDiagnostics, [
      { provider: 'claude', status: 'skipped', detail: 'claude unavailable (command missing)' },
    ]);
  });

  test('plan-build ladder records all-rungs failure through the main failure ledger', async () => {
    const captured = captureIo();
    const ledgers = [];
    const routing = routingWithLadder({ rungs: ['claude', 'codex'] });
    const code = await main(
      ['--workflow', 'solo', '--live', '--phase', 'research', '--task', 'plain docs update'],
      {},
      captured.io,
      {
        routing,
        brain: 'DEV_BRAIN fixture',
        soul: 'SOUL fixture',
        commandExists: recordCommandExists(['claude', 'codex']),
        gateRunner() {
          throw new Error('preflight gate should not run without a provider');
        },
        writeRunLedger(record) {
          ledgers.push(record);
        },
        appendRunEvent: null,
        clock: () => new Date('2026-07-07T19:04:07.684Z'),
      }
    );

    assert.equal(code, 1);
    assert.equal(ledgers.length, 1);
    assert.equal(ledgers[0].error, 'no available provider for role solo; tried: [claude, codex]');
    assert.deepEqual(ledgers[0].failureLedger[0].tried, ['claude', 'codex']);
    assert.deepEqual(
      ledgers[0].providerDiagnostics.map((diagnostic) => diagnostic.status),
      ['skipped', 'skipped']
    );
  });

  test('plan-build ladder skips a cooling provider through the injected seam', () => {
    const routing = routingWithLadder({ rungs: ['claude', 'codex'] });

    const plan = createRoutingPlan({
      phase: 'research',
      task: 'plain docs update',
      routing,
      requestedWorkflow: 'solo',
      commandExists: recordCommandExists(),
      isCooling: (provider) => provider === 'claude',
    });

    const owner = plan.workflow.steps.find((step) => step.role === 'owner');
    assert.equal(owner.model, 'codex');
    assert.equal(owner.degradeReason, 'claude cooling');
    assert.deepEqual(owner.providerDiagnostics, [
      { provider: 'claude', status: 'skipped', detail: 'claude cooling' },
    ]);
  });

  test('financial-risk lanes refuse plan-build degradation outside production', () => {
    const routing = routingWithLadder({ rungs: ['claude', 'codex'] });
    const commandExists = recordCommandExists(['claude']);

    assert.throws(
      () =>
        createRoutingPlan({
          phase: 'research',
          task: 'backtest walk-forward sharpe review',
          routing,
          requestedWorkflow: 'solo',
          commandExists,
        }),
      /refused to degrade under a financial-risk classification/
    );
    assert.deepEqual(commandExists.calls, ['claude']);
  });

  test('manual model pins do not degrade unless allow-fallback is explicit', () => {
    const pinnedRouting = routingWithLadder({ rungs: ['claude', 'codex', 'kimi'] });
    const pinnedCommandExists = recordCommandExists(['codex']);

    assert.throws(
      () =>
        createRoutingPlan({
          phase: 'research',
          task: 'plain docs update',
          routing: pinnedRouting,
          manualModel: 'codex',
          requestedWorkflow: 'solo',
          commandExists: pinnedCommandExists,
        }),
      /manual provider codex unavailable \(command missing\); fallback disabled/
    );
    assert.deepEqual(pinnedCommandExists.calls, ['codex']);

    const fallbackCommandExists = recordCommandExists(['codex']);
    const fallbackPlan = createRoutingPlan({
      phase: 'research',
      task: 'plain docs update',
      routing: routingWithLadder({ rungs: ['claude', 'codex', 'kimi'] }),
      manualModel: 'codex',
      allowFallback: true,
      requestedWorkflow: 'solo',
      commandExists: fallbackCommandExists,
    });

    assert.equal(fallbackPlan.model, 'kimi');
    assert.equal(fallbackPlan.requestedModel, 'codex');
    assert.equal(fallbackPlan.degradeReason, 'codex unavailable (command missing)');
  });

  test('ladder disabled leaves routing plan shape unchanged', () => {
    const routing = routingWithLadder({ enabled: false, rungs: ['claude', 'codex'] });

    const plan = createRoutingPlan({
      phase: 'production',
      task: 'add router tests',
      routing,
      requestedWorkflow: 'pair',
      commandExists: recordCommandExists(['codex']),
    });

    assert.equal(plan.model, 'codex');
    assert.equal('requestedModel' in plan, false);
    assert.equal('providerDiagnostics' in plan.workflow.steps[0], false);
    assert.deepEqual(
      plan.workflow.steps.map((step) => step.model),
      ['codex', 'claude', null]
    );
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

  test('debate workflow emits a structural debate-run record with rebuttal lane', async () => {
    const captured = captureIo();
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const clock = createFakeClock();
    const workflow = createWorkflowPlan({
      requestedWorkflow: 'debate',
      phase: 'production',
      model: 'codex',
      ownership: { effectivePhase: 'production', artifact: 'diff plus tests' },
      gate: null,
      debate: { comparators: ['claude', 'codex'], synthesis: 'claude' },
    });
    const calls = [];
    let debateResult = null;

    const record = await executeWorkflow(
      { phase: 'production', risk: 'standard', gate: null, workflow },
      {
        runId: 'hermes-2026-07-07T19-04-07-684Z',
        async runStep({ step, input }) {
          clock.advance(1);
          calls.push({ role: step.role, input });
          if (step.role === 'comparator') {
            return { code: 0, output: `${step.model} critique` };
          }
          if (step.role === 'rebuttal') {
            assert.deepEqual(input, ['claude critique', 'codex critique']);
            return { code: 0, output: 'rebuttal ranked outcome' };
          }
          if (step.role === 'synthesis') {
            assert.deepEqual(input, [
              'claude critique',
              'codex critique',
              'REBUTTAL:\nrebuttal ranked outcome',
            ]);
            return { code: 0, output: 'FINAL VERDICT: APPROVE-WITH-CHANGES' };
          }
          throw new Error(`unexpected step ${step.role}`);
        },
        writeRunLedger(ledger) {
          return writeRunLedger(ledger, { root, fs, envSecrets: [] });
        },
        writeDebateRunRecord(ledger, options) {
          debateResult = writeDebateRunRecord(ledger, {
            ...options,
            root,
            fs,
            clock: () => clock.now(),
            io: captured.io,
          });
          return debateResult;
        },
        appendRunEvent: null,
        root,
        clock: () => clock.now(),
        io: captured.io,
      }
    );

    assert.equal(record.exitCode, 0);
    assert.deepEqual(
      calls.map((call) => call.role),
      ['comparator', 'comparator', 'rebuttal', 'synthesis']
    );
    const hermesFile = join(
      root,
      'ai-logs',
      'hermes',
      'runs',
      'hermes-2026-07-07T19-04-07-684Z.json'
    );
    const debateFile = [...fs.files.keys()].find((file) => file.includes('debate-run-'));
    assert.equal(fs.files.has(hermesFile), true);
    assert.equal(Boolean(debateFile), true);
    assert.equal(debateResult.file, debateFile);

    const debateRun = JSON.parse(fs.files.get(debateFile));
    const validation = validateDebateRunRecordShape(debateRun);
    assert.equal(validation.valid, true);
    assert.deepEqual(validation.errors, []);
    assert.deepEqual(
      debateRun.lanes.map((lane) => lane.lane),
      ['claude', 'codex', 'rebuttal']
    );
    assert.equal(debateRun.workflow, 'hermes-debate/v1');
    assert.equal(debateRun.finalVerdict, 'APPROVE-WITH-CHANGES');
    assert.equal(debateRun.synthesisFile, null);
    assert.deepEqual(debateRun.deviationsFromSpec[0], {
      what: 'router runs model-based comparators, not distinct redteam/blind lanes',
      why: 'router transport maps N comparator models onto the spec lane set',
    });
    assert.equal(debateResult.validation.errors.length, 0);
  });

  test('debate-run record captures failed rebuttal lane diagnostics and deviations', async () => {
    const captured = captureIo();
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const clock = createFakeClock();
    const workflow = createWorkflowPlan({
      requestedWorkflow: 'debate',
      phase: 'production',
      model: 'codex',
      ownership: { effectivePhase: 'production', artifact: 'diff plus tests' },
      gate: null,
      debate: { comparators: ['claude'], rebuttal: 'codex', synthesis: 'claude' },
    });

    await executeWorkflow(
      { phase: 'production', risk: 'standard', gate: null, workflow },
      {
        runId: 'hermes-2026-07-07T19-04-07-684Z',
        async runStep({ step }) {
          clock.advance(1);
          if (step.role === 'rebuttal') {
            return { code: 9, output: 'rebuttal failed' };
          }
          return {
            code: 0,
            output: step.role === 'synthesis' ? 'FINAL VERDICT: APPROVE' : 'comparator ok',
          };
        },
        writeRunLedger(ledger) {
          return writeRunLedger(ledger, { root, fs, envSecrets: [] });
        },
        writeDebateRunRecord(ledger, options) {
          return writeDebateRunRecord(ledger, {
            ...options,
            root,
            fs,
            clock: () => clock.now(),
            io: captured.io,
          });
        },
        appendRunEvent: null,
        root,
        clock: () => clock.now(),
        io: captured.io,
      }
    );

    const debateFile = [...fs.files.keys()].find((file) => file.includes('debate-run-'));
    const debateRun = JSON.parse(fs.files.get(debateFile));
    const rebuttalLane = debateRun.lanes.find((lane) => lane.lane === 'rebuttal');
    assert.equal(rebuttalLane.status, 'failed');
    assert.equal(
      debateRun.deviationsFromSpec.some((entry) => entry.what === 'rebuttal lane failed'),
      true
    );
    assert.deepEqual(debateRun.providerDiagnostics, [
      {
        provider: 'codex',
        status: 'nonzero_exit',
        detail: 'rebuttal lane exited with code 9',
      },
    ]);
  });

  test('debate-run record is still written when synthesis is skipped', async () => {
    const captured = captureIo();
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const clock = createFakeClock();
    const workflow = {
      selected: 'debate',
      steps: [
        { role: 'comparator', model: 'claude', action: 'compare' },
        {
          role: 'rebuttal',
          model: 'claude',
          action: 'verify and rebut comparator critiques against ground truth',
        },
      ],
    };

    await executeWorkflow(
      { phase: 'production', risk: 'standard', gate: null, workflow },
      {
        runId: 'hermes-2026-07-07T19-04-07-684Z',
        async runStep({ step }) {
          clock.advance(1);
          return { code: 0, output: `${step.role} output` };
        },
        writeRunLedger(ledger) {
          return writeRunLedger(ledger, { root, fs, envSecrets: [] });
        },
        writeDebateRunRecord(ledger, options) {
          return writeDebateRunRecord(ledger, {
            ...options,
            root,
            fs,
            clock: () => clock.now(),
            io: captured.io,
          });
        },
        appendRunEvent: null,
        root,
        clock: () => clock.now(),
        io: captured.io,
      }
    );

    const debateFile = [...fs.files.keys()].find((file) => file.includes('debate-run-'));
    assert.equal(Boolean(debateFile), true);
    const debateRun = JSON.parse(fs.files.get(debateFile));
    assert.equal(debateRun.synthesisFile, null);
    assert.equal(debateRun.finalVerdict, null);
    assert.equal(
      debateRun.deviationsFromSpec.some((entry) => entry.what === 'synthesis step skipped'),
      true
    );
  });

  test('non-debate workflow does not emit a debate-run record', async () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    let debateWrites = 0;
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

    await executeWorkflow(plan, {
      async runStep({ step }) {
        return { code: 0, approved: step.role === 'reviewer', output: `${step.role} output` };
      },
      writeRunLedger(ledger) {
        return writeRunLedger(ledger, { root, fs, envSecrets: [] });
      },
      writeDebateRunRecord() {
        debateWrites += 1;
      },
      appendRunEvent: null,
      root,
      clock: () => new Date('2026-07-07T19:04:07.684Z'),
    });

    assert.equal(debateWrites, 0);
    assert.equal([...fs.files.keys()].some((file) => file.includes('debate-run-')), false);
  });

  test('appendRunEvent appends scrubbed JSONL through the injected fs', () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const runId = 'hermes-2026-07-07T19-04-07-684Z';

    const file = appendRunEvent(
      runId,
      {
        type: 'dispatch_error',
        role: 'owner',
        model: 'codex',
        failure: 'nonzero_exit',
        excerpt: 'API_KEY=raw-key-value',
      },
      {
        root,
        fs,
        clock: () => new Date('2026-07-07T19:04:08.000Z'),
        envSecrets: [],
      }
    );

    assert.equal(file, join(root, 'ai-logs', 'hermes', 'runs', `${runId}.events.jsonl`));
    assert.deepEqual(fs.calls[0], {
      method: 'mkdirSync',
      path: join(root, 'ai-logs', 'hermes', 'runs'),
      options: { recursive: true },
    });
    const rows = parseJsonl(fs.files.get(file));
    assert.deepEqual(rows, [
      {
        ts: '2026-07-07T19:04:08.000Z',
        type: 'dispatch_error',
        role: 'owner',
        model: 'codex',
        failure: 'nonzero_exit',
        excerpt: 'API_KEY=[SCRUBBED:keyvalue]',
      },
    ]);
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

  test('createLiveRunStep applies read-only codex args for reviewer spawn', async () => {
    const clock = createFakeClock();
    const spawn = createScriptedSpawn([{ stdout: ['APPROVED'], code: 0 }], { clock });
    const routing = routingWithLadder();
    const runStep = createLiveRunStep({
      routing,
      env: { CODEX_BIN: process.execPath },
      modelClock: clock,
      executor(model, prompt, effectiveRouting, env, seams) {
        return executeModelCapture(model, prompt, effectiveRouting, env, {
          ...seams,
          spawn,
        });
      },
    });

    const pending = runStep({
      step: { role: 'reviewer', action: 'review diff', model: 'codex' },
      input: 'diff body',
    });
    spawn.flushNext();
    const result = await pending;

    assert.equal(result.approved, true);
    assert.deepEqual(spawn.calls[0].args, ['exec', '--sandbox', 'read-only']);
    assert.deepEqual(routing.commands.codex.args, ['exec', '--sandbox', 'workspace-write']);
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
        appendRunEvent: null,
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
        appendRunEvent: null,
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
        appendRunEvent: null,
        writeRunCheckpoint: null,
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
    const fs = createRecordingFs();
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
      appendRunEvent: null,
      checkpointFs: fs,
      root: join('C:\\', 'tmp', 'hermes-test-root'),
      clock: () => new Date('2026-07-07T19:04:07.684Z'),
    });

    assert.equal(record.steps[0].output, 'owner leaked env-secret-value-123');
    assert.equal(ledgers.length, 1);
    assert.equal(ledgers[0].steps[0].output, 'owner leaked [SCRUBBED:envvalue]');
    assert.equal(ledgers[0].steps[0].scrubCount, 1);
    assert.equal(ledgers[0].steps[1].scrubCount, 0);
  });

  test('dispatch fallback advances a non-write review step and emits a fallback event', async () => {
    const ledgers = [];
    const events = [];
    const calls = [];
    const routing = routingWithLadder({ rungs: ['claude', 'agy'] });
    const plan = {
      phase: 'research',
      risk: 'standard',
      gate: null,
      ladder: routing.ladder,
      workflow: {
        selected: 'pair',
        steps: [
          {
            role: 'reviewer',
            model: 'claude',
            requestedModel: 'claude',
            selectedModel: 'claude',
            action: 'review',
            providerDiagnostics: [],
          },
        ],
      },
    };

    const record = await executeWorkflow(plan, {
      routing,
      commandExists: recordCommandExists(),
      async runStep({ step }) {
        calls.push(step.model);
        if (step.model === 'claude') {
          return { code: 7, failure: 'nonzero_exit', output: 'provider failed' };
        }
        return { code: 0, approved: true, output: 'approved\nAPPROVED' };
      },
      writeRunLedger(ledger) {
        ledgers.push(ledger);
      },
      appendRunEvent(_runId, event) {
        events.push(event);
      },
      checkpointFs: createRecordingFs(),
      root: join('C:\\', 'tmp', 'hermes-test-root'),
      clock: () => new Date('2026-07-07T19:04:07.684Z'),
    });

    assert.deepEqual(calls, ['claude', 'agy']);
    assert.equal(record.exitCode, 0);
    assert.equal(record.steps[0].model, 'agy');
    assert.equal(record.steps[0].weakenedControl, true);
    assert.equal(record.steps[0].deviation, 'review-lane provider fallback weakened independence control');
    assert.deepEqual(record.steps[0].providerDiagnostics, [
      { provider: 'claude', status: 'failed', detail: 'claude failed (nonzero_exit)' },
      { provider: 'agy', status: 'fallback', detail: 'advanced from claude after nonzero_exit' },
    ]);
    assert.deepEqual(
      events.filter((event) => event.type === 'fallback'),
      [
        {
          type: 'fallback',
          role: 'reviewer',
          from: 'claude',
          to: 'agy',
          failure: 'nonzero_exit',
        },
      ]
    );
    assert.equal(ledgers.length, 1);
    assert.equal(ledgers[0].steps[0].providerDiagnostics[0].provider, 'claude');
  });

  test('write-capable provider failure does not dispatch the next rung', async () => {
    const events = [];
    const calls = [];
    const routing = routingWithLadder({ rungs: ['codex', 'agy'] });
    const plan = {
      phase: 'production',
      risk: 'standard',
      gate: null,
      ladder: routing.ladder,
      workflow: {
        selected: 'solo',
        steps: [
          {
            role: 'owner',
            model: 'codex',
            requestedModel: 'codex',
            selectedModel: 'codex',
            action: 'execute',
            providerDiagnostics: [],
          },
        ],
      },
    };

    const record = await executeWorkflow(plan, {
      routing,
      commandExists: recordCommandExists(),
      async runStep({ step }) {
        calls.push(step.model);
        return { code: 9, failure: 'nonzero_exit', output: 'partial write may have happened' };
      },
      appendRunEvent(_runId, event) {
        events.push(event);
      },
      checkpointFs: createRecordingFs(),
      root: join('C:\\', 'tmp', 'hermes-test-root'),
      clock: () => new Date('2026-07-07T19:04:07.684Z'),
    });

    assert.deepEqual(calls, ['codex']);
    assert.equal(record.exitCode, 9);
    assert.equal(record.steps[0].model, 'codex');
    assert.deepEqual(
      record.steps[0].providerDiagnostics.map((diagnostic) => diagnostic.status),
      ['failed', 'blocked']
    );
    assert.match(record.steps[0].providerDiagnostics[1].detail, /write-capable provider codex/);
    assert.deepEqual(events.filter((event) => event.type === 'fallback'), []);
    assert.equal(record.failureLedger[0].role, 'owner');
  });

  test('manual pin blocks dispatch fallback by default but allow-fallback advances', async () => {
    const routing = routingWithLadder({ rungs: ['claude', 'agy'] });
    const pinnedPlan = {
      phase: 'research',
      risk: 'standard',
      gate: null,
      ladder: routing.ladder,
      manualPinned: true,
      allowFallback: false,
      workflow: {
        selected: 'solo',
        steps: [
          {
            role: 'owner',
            model: 'claude',
            requestedModel: 'claude',
            selectedModel: 'claude',
            manualPinned: true,
            allowFallback: false,
            action: 'execute',
            providerDiagnostics: [],
          },
        ],
      },
    };
    const pinnedCalls = [];
    const pinned = await executeWorkflow(pinnedPlan, {
      routing,
      commandExists: recordCommandExists(),
      async runStep({ step }) {
        pinnedCalls.push(step.model);
        return { code: 4, failure: 'nonzero_exit', output: 'failed' };
      },
      appendRunEvent: null,
      checkpointFs: createRecordingFs(),
      root: join('C:\\', 'tmp', 'hermes-test-root'),
      clock: () => new Date('2026-07-07T19:04:07.684Z'),
    });

    assert.deepEqual(pinnedCalls, ['claude']);
    assert.equal(pinned.exitCode, 4);
    assert.match(pinned.steps[0].providerDiagnostics[1].detail, /fallback disabled/);

    const fallbackCalls = [];
    const fallback = await executeWorkflow(
      {
        ...pinnedPlan,
        allowFallback: true,
        workflow: {
          ...pinnedPlan.workflow,
          steps: [
            {
              ...pinnedPlan.workflow.steps[0],
              allowFallback: true,
            },
          ],
        },
      },
      {
        routing,
        commandExists: recordCommandExists(),
        async runStep({ step }) {
          fallbackCalls.push(step.model);
          if (step.model === 'claude') {
            return { code: 4, failure: 'nonzero_exit', output: 'failed' };
          }
          return { code: 0, output: 'ok' };
        },
        appendRunEvent: null,
        checkpointFs: createRecordingFs(),
        root: join('C:\\', 'tmp', 'hermes-test-root'),
        clock: () => new Date('2026-07-07T19:04:07.684Z'),
      }
    );

    assert.deepEqual(fallbackCalls, ['claude', 'agy']);
    assert.equal(fallback.exitCode, 0);
    assert.equal(fallback.steps[0].model, 'agy');
  });

  test('financial-risk dispatch refuses fallback outside production', async () => {
    const calls = [];
    const routing = routingWithLadder({ rungs: ['claude', 'agy'] });
    const record = await executeWorkflow(
      {
        phase: 'research',
        risk: 'financial',
        gate: null,
        ladder: routing.ladder,
        workflow: {
          selected: 'pair',
          steps: [
            {
              role: 'specialist',
              model: 'claude',
              requestedModel: 'claude',
              selectedModel: 'claude',
              action: 'review financial risk',
              providerDiagnostics: [],
            },
          ],
        },
      },
      {
        routing,
        commandExists: recordCommandExists(),
        async runStep({ step }) {
          calls.push(step.model);
          return { code: 8, failure: 'nonzero_exit', output: 'failed' };
        },
        appendRunEvent: null,
        checkpointFs: createRecordingFs(),
        root: join('C:\\', 'tmp', 'hermes-test-root'),
        clock: () => new Date('2026-07-07T19:04:07.684Z'),
      }
    );

    assert.deepEqual(calls, ['claude']);
    assert.equal(record.exitCode, 8);
    assert.match(
      record.steps[0].providerDiagnostics[1].detail,
      /refused to degrade under a financial-risk classification/
    );
    assert.equal(record.failureLedger[0].role, 'specialist');
  });

  test('live workflow appends compact step and gate events without output bodies', async () => {
    const captured = captureIo();
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const clock = createFakeClock();
    const runId = 'hermes-2026-07-07T19-04-07-684Z';
    const eventFile = join(root, 'ai-logs', 'hermes', 'runs', `${runId}.events.jsonl`);

    const code = await main(
      ['--workflow', 'pair', '--live', '--phase', 'production', '--task', 'add router tests'],
      {},
      captured.io,
      {
        routing: routingFixture(),
        brain: 'DEV_BRAIN fixture',
        soul: 'SOUL fixture',
        gateRunner: gateRunner(0),
        commandExists: recordCommandExists(),
        async runStep({ step }) {
          clock.advance(7);
          return { code: 0, approved: step.role === 'reviewer', output: `${step.role} body` };
        },
        writeRunLedger() {},
        appendRunEvent(run, event, options) {
          return appendRunEvent(run, event, { ...options, root, fs });
        },
        checkpointFs: fs,
        root,
        clock: () => clock.now(),
      }
    );

    assert.equal(code, 0);
    assert.equal(captured.output().stderr, '');
    const rows = parseJsonl(fs.files.get(eventFile));
    assert.deepEqual(
      rows.map((row) => row.type),
      ['gate_result', 'step_start', 'step_end', 'step_start', 'step_end', 'gate_result']
    );
    assert.equal(rows.find((row) => row.type === 'step_end').durationMs, 7);
    assert.equal(JSON.stringify(rows).includes('owner body'), false);
    assert.equal(JSON.stringify(rows).includes('reviewer body'), false);
  });

  test('appendRunEvent failure warns once and does not change workflow exitCode', async () => {
    const captured = captureIo();
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
      async runStep({ step }) {
        return { code: 0, approved: step.role === 'reviewer', output: `${step.role} output` };
      },
      writeRunLedger(ledger) {
        ledgers.push(ledger);
      },
      appendRunEvent() {
        throw new Error('events locked');
      },
      checkpointFs: createRecordingFs(),
      root: join('C:\\', 'tmp', 'hermes-test-root'),
      io: captured.io,
      clock: () => new Date('2026-07-07T19:04:07.684Z'),
    });

    assert.equal(record.exitCode, 0);
    assert.equal(ledgers.length, 1);
    const warning = captured.output().stderr.match(/failed to write run event log/g) || [];
    assert.equal(warning.length, 1);
  });

  test('malformed config errors include the filename and still write a failure ledger', async () => {
    const captured = captureIo();
    const ledgers = [];
    const fs = createRecordingFs();
    const configPath = join('C:\\', 'tmp', 'bad-routing.json');
    fs.files.set(configPath, '{ bad json');

    await assert.rejects(
      main(
        ['--phase', 'production', '--task', 'add router tests'],
        { HERMES_MODEL_ROUTING_FILE: configPath },
        captured.io,
        {
          fs,
          writeRunLedger(record) {
            ledgers.push(record);
          },
          clock: () => new Date('2026-07-07T19:04:07.684Z'),
        }
      ),
      (error) => error instanceof SyntaxError && error.message.includes(configPath)
    );

    assert.equal(ledgers.length, 1);
    assert.equal(ledgers[0].runId, 'hermes-2026-07-07T19-04-07-684Z');
    assert.equal(ledgers[0].phase, 'production');
    assert.equal(ledgers[0].exitCode, 1);
    assert.match(ledgers[0].error, /Invalid JSON in/);
    assert.match(ledgers[0].error, /bad-routing\.json/);
  });

  test('interrupted workflow flushes an interrupted checkpoint without approval bodies', async () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const runId = 'hermes-2026-07-07T19-04-07-684Z';
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

    await assert.rejects(
      executeWorkflow(plan, {
        runId,
        async runStep({ step }) {
          if (step.role === 'owner') {
            return { code: 0, output: 'owner body APPROVED' };
          }
          throw new Error('router died mid-run');
        },
        writeRunLedger() {
          throw new Error('final ledger should not be written');
        },
        appendRunEvent: null,
        checkpointFs: fs,
        root,
        clock: () => new Date('2026-07-07T19:04:07.684Z'),
      }),
      /router died mid-run/
    );

    const file = join(root, 'ai-logs', 'hermes', 'runs', `${runId}.json`);
    const persisted = fs.files.get(file);
    const checkpoint = JSON.parse(persisted);
    assert.equal(checkpoint.status, 'interrupted');
    assert.equal(persisted.includes('APPROVED'), false);
    assert.equal(persisted.includes('"approved": true'), false);
  });

  test('in-progress checkpoints exclude output bodies while the final ledger includes them', async () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const runId = 'hermes-2026-07-07T19-04-07-684Z';
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

    await executeWorkflow(plan, {
      runId,
      async runStep({ step }) {
        if (step.role === 'owner') {
          return { code: 0, output: 'owner full body' };
        }
        return { code: 0, approved: true, output: 'reviewer full body' };
      },
      writeRunLedger(record) {
        return writeRunLedger(record, { root, fs, envSecrets: [] });
      },
      appendRunEvent: null,
      checkpointFs: fs,
      root,
      clock: () => new Date('2026-07-07T19:04:07.684Z'),
    });

    const checkpointWrites = fs.calls.filter(
      (call) => call.method === 'writeFileSync' && String(call.contents).includes('"in-progress"')
    );
    assert.equal(checkpointWrites.length >= 2, true);
    for (const write of checkpointWrites) {
      assert.equal(String(write.contents).includes('owner full body'), false);
      assert.equal(String(write.contents).includes('reviewer full body'), false);
      assert.equal(String(write.contents).includes('"output"'), false);
    }

    const finalFile = join(root, 'ai-logs', 'hermes', 'runs', `${runId}.json`);
    const finalLedger = JSON.parse(fs.files.get(finalFile));
    assert.equal(finalLedger.steps[0].output, 'owner full body');
    assert.equal(finalLedger.steps[1].output, 'reviewer full body');
  });

  test('doctor reports stale in-progress run ledgers older than 24h', async () => {
    const captured = captureIo();
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-test-root');
    const staleRunId = 'hermes-2026-07-06T18-00-00-000Z';
    fs.files.set(
      join(root, 'ai-logs', 'hermes', 'runs', `${staleRunId}.json`),
      JSON.stringify({
        runId: staleRunId,
        status: 'in-progress',
        updatedAt: '2026-07-06T18:00:00.000Z',
      })
    );

    const code = await main(['doctor'], {}, captured.io, {
      routing: routingFixture(),
      commandExists: recordCommandExists(),
      fs,
      root,
      clock: () => new Date('2026-07-07T19:30:00.000Z'),
    });

    assert.equal(code, 0);
    assert.match(captured.output().stdout, /Stale in-progress runs/);
    assert.match(captured.output().stdout, new RegExp(staleRunId));
    assert.match(captured.output().stdout, /25h/);
  });
});

describe('rate-limit signature registry and cooldown store', () => {
  async function runRateLimitedCommand({
    provider = 'claude',
    routing = routingWithRateLimits(),
    stderr = 'usage limit reached',
    fs = createRecordingFs(),
    root = join('C:\\', 'tmp', 'hermes-cooldown-test'),
    clock = createFakeClock('2026-07-07T19:00:00.000Z'),
    events = [],
  } = {}) {
    const spawn = createScriptedSpawn([{ stderr: [stderr], code: 7 }], { clock });
    const env = {
      CLAUDE_CODE_BIN: process.execPath,
      CODEX_BIN: process.execPath,
      KIMI_CODE_BIN: process.execPath,
      AGY_BIN: process.execPath,
    };
    const pending = executeModelCommand(provider, 'prompt', routing, env, {
      spawn,
      clock,
      captureOutput: true,
      providerStateFs: fs,
      root,
      runId: 'hermes-test-run',
      appendRunEvent(_runId, event) {
        events.push(event);
      },
    });
    spawn.flushNext();
    return pending;
  }

  test('matched provider signature classifies as rate_limited, sets default cooldown, and emits event', async () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-cooldown-default');
    const events = [];

    const result = await runRateLimitedCommand({ fs, root, events });
    const state = readProviderState(fs, root);

    assert.equal(result.failure, 'rate_limited');
    assert.equal(state.providers.claude.coolingUntil, '2026-07-07T20:00:00.000Z');
    assert.equal(state.providers.claude.reason, 'rate_limited');
    assert.deepEqual(
      events.filter((event) => event.type === 'cooldown_set'),
      [{ type: 'cooldown_set', provider: 'claude', coolingUntil: '2026-07-07T20:00:00.000Z' }]
    );
  });

  test('provider-state cooling skips the ladder before expiry and releases at the boundary', async () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-cooling-boundary');
    fs.files.set(
      providerStateFile(root),
      `${JSON.stringify({
        providers: {
          claude: {
            coolingUntil: '2026-07-07T20:00:00.000Z',
            reason: 'rate_limited',
            triggeredAt: '2026-07-07T19:00:00.000Z',
          },
        },
      })}\n`
    );
    const routing = routingWithRateLimits({ ladderEnabled: true });

    async function planAt(now) {
      const captured = captureIo();
      const code = await main(
        ['--json', '--phase', 'research', '--task', 'plain docs update'],
        {},
        captured.io,
        {
          routing,
          brain: 'brain',
          soul: 'soul',
          fs,
          root,
          commandExists: recordCommandExists(),
          clock: () => new Date(now),
        }
      );
      assert.equal(code, 0);
      return JSON.parse(captured.output().stdout);
    }

    assert.equal((await planAt('2026-07-07T19:59:59.999Z')).model, 'codex');
    assert.equal((await planAt('2026-07-07T20:00:00.000Z')).model, 'claude');
  });

  test('corrupt provider-state fails open, warns once, sidelines the file, and doctor continues', async () => {
    const captured = captureIo();
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-corrupt-state');
    fs.files.set(providerStateFile(root), '{not-json');

    const code = await main(['doctor'], {}, captured.io, {
      routing: routingFixture(),
      commandExists: recordCommandExists(),
      fs,
      root,
      clock: () => new Date('2026-07-07T19:00:00.000Z'),
    });

    const output = captured.output();
    assert.equal(code, 0);
    assert.match(output.stdout, /Hermes CLI doctor/);
    assert.equal((output.stderr.match(/provider cooldown state was corrupt/g) || []).length, 1);
    assert.equal(fs.files.has(providerStateFile(root)), false);
    assert.equal(fs.files.has(`${providerStateFile(root)}.corrupt`), true);
  });

  test('cooldown writes re-read and keep a longer on-disk coolingUntil', async () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-merge-state');
    fs.files.set(
      providerStateFile(root),
      `${JSON.stringify({
        providers: {
          claude: {
            coolingUntil: '2026-07-08T00:00:00.000Z',
            reason: 'rate_limited',
            triggeredAt: '2026-07-07T18:00:00.000Z',
          },
        },
      })}\n`
    );

    const result = await runRateLimitedCommand({ fs, root });
    const state = readProviderState(fs, root);

    assert.equal(result.failure, 'rate_limited');
    assert.equal(state.providers.claude.coolingUntil, '2026-07-08T00:00:00.000Z');
  });

  test('repeat triggers double the duration and cap at 24 hours', async () => {
    const activeFs = createRecordingFs();
    const activeRoot = join('C:\\', 'tmp', 'hermes-repeat-active');
    activeFs.files.set(
      providerStateFile(activeRoot),
      `${JSON.stringify({
        providers: {
          claude: {
            coolingUntil: '2026-07-07T19:30:00.000Z',
            reason: 'rate_limited',
            triggeredAt: '2026-07-07T18:30:00.000Z',
          },
        },
      })}\n`
    );

    await runRateLimitedCommand({ fs: activeFs, root: activeRoot });
    assert.equal(
      readProviderState(activeFs, activeRoot).providers.claude.coolingUntil,
      '2026-07-07T21:00:00.000Z'
    );

    const cappedFs = createRecordingFs();
    const cappedRoot = join('C:\\', 'tmp', 'hermes-repeat-capped');
    cappedFs.files.set(
      providerStateFile(cappedRoot),
      `${JSON.stringify({
        providers: {
          claude: {
            coolingUntil: '2026-07-07T19:30:00.000Z',
            reason: 'rate_limited',
            triggeredAt: '2026-07-07T18:30:00.000Z',
          },
        },
      })}\n`
    );
    const routing = routingWithRateLimits({
      defaults: { providers: { claude: 800 } },
    });

    await runRateLimitedCommand({ routing, fs: cappedFs, root: cappedRoot });
    assert.equal(
      readProviderState(cappedFs, cappedRoot).providers.claude.coolingUntil,
      '2026-07-08T19:00:00.000Z'
    );
  });

  test('explicit reset time in the message is used instead of the provider default', async () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-explicit-reset');

    await runRateLimitedCommand({
      fs,
      root,
      stderr: 'usage limit reached; resets at 2026-07-07T22:30:00.000Z',
    });

    assert.equal(
      readProviderState(fs, root).providers.claude.coolingUntil,
      '2026-07-07T22:30:00.000Z'
    );
  });

  test('doctor renders WINDOW for cooling providers and dash for healthy providers', async () => {
    const captured = captureIo();
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-doctor-window');
    fs.files.set(
      providerStateFile(root),
      `${JSON.stringify({
        providers: {
          claude: {
            coolingUntil: '2026-07-07T20:00:00.000Z',
            reason: 'rate_limited',
            triggeredAt: '2026-07-07T19:00:00.000Z',
          },
        },
      })}\n`
    );

    const code = await main(['doctor'], {}, captured.io, {
      routing: routingWithRateLimits(),
      commandExists: recordCommandExists(),
      fs,
      root,
      clock: () => new Date('2026-07-07T19:30:00.000Z'),
    });

    const stdout = captured.output().stdout;
    assert.equal(code, 0);
    assert.match(stdout, /WINDOW/);
    assert.match(stdout, /claude\s+claude\s+default\s+found\s+2026-07-07T20:00:00.000Z/);
    assert.match(stdout, /codex\s+codex\s+default\s+found\s+-/);
  });

  test('rateLimits disabled leaves classification and ladder selection unchanged', async () => {
    const fs = createRecordingFs();
    const root = join('C:\\', 'tmp', 'hermes-disabled-rate-limits');
    const result = await runRateLimitedCommand({
      routing: routingWithRateLimits({ enabled: false }),
      fs,
      root,
      stderr: 'quota 429',
    });

    assert.equal(result.failure, 'nonzero_exit');
    assert.equal(fs.files.has(providerStateFile(root)), false);

    fs.files.set(
      providerStateFile(root),
      `${JSON.stringify({
        providers: {
          claude: {
            coolingUntil: '2026-07-07T20:00:00.000Z',
            reason: 'rate_limited',
            triggeredAt: '2026-07-07T19:00:00.000Z',
          },
        },
      })}\n`
    );
    const captured = captureIo();
    const code = await main(
      ['--json', '--phase', 'research', '--task', 'plain docs update'],
      {},
      captured.io,
      {
        routing: routingWithRateLimits({ enabled: false, ladderEnabled: false }),
        brain: 'brain',
        soul: 'soul',
        fs,
        root,
        commandExists: recordCommandExists(),
        clock: () => new Date('2026-07-07T19:30:00.000Z'),
      }
    );

    assert.equal(code, 0);
    assert.equal(JSON.parse(captured.output().stdout).model, 'claude');
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

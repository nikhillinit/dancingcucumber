#!/usr/bin/env node

import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from 'node:fs';
import { spawn, spawnSync } from 'node:child_process';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const ROOT = dirname(__filename);
const LEGACY_COMMANDS = new Set(['bootstrap', 'smoke', 'enable-algorithms', 'doctor']);
const DOCTOR_PROVIDERS = ['claude', 'codex', 'kimi', 'gemini', 'agy'];
const WORKFLOW_MODES = new Set(['auto', 'solo', 'pair', 'chain', 'debate', 'review']);
const MODEL_OVERRIDES = new Set(['claude', 'codex', 'kimi', 'gemini', 'agy']);
const WORKFLOW_DEFERRED_BEHAVIOR = [
  'model execution',
  'artifact handoff',
  'review platform automation',
];
const DEFAULT_MODEL_TIMEOUT_MS = 600000;
const STALE_RUN_MS = 24 * 60 * 60 * 1000;
const DEFAULT_PROVIDER_COOLDOWN_MINUTES = 60;
const COOLDOWN_REPEAT_WINDOW_MS = 10 * 60 * 1000;
const COOLDOWN_MAX_MS = 24 * 60 * 60 * 1000;
const MODEL_FAILURE = Object.freeze({
  SPAWN_ERROR: 'spawn_error',
  TIMEOUT: 'timeout',
  NONZERO_EXIT: 'nonzero_exit',
  EMPTY_OUTPUT: 'empty_output',
  RATE_LIMITED: 'rate_limited',
});
const MODEL_ENV_BLOCKLIST = [
  'ANTHROPIC_API_KEY',
  'OPENAI_API_KEY',
  'GEMINI_API_KEY',
  'GOOGLE_API_KEY',
  'GOOGLE_APPLICATION_CREDENTIALS',
  'OPENROUTER_API_KEY',
  'PERPLEXITY_API_KEY',
  'QWEN_API_KEY',
  'DASHSCOPE_API_KEY',
  'MOONSHOT_API_KEY',
  'KIMI_API_KEY',
  'XAI_API_KEY',
  'GROK_API_KEY',
  'NOUS_API_KEY',
  'HF_TOKEN',
  'MISTRAL_API_KEY',
  'COHERE_API_KEY',
];
const DEFAULT_RATE_LIMIT_SIGNATURES = [];
const SCRUB_ERROR_TEXT = '[SCRUB-ERROR: output withheld]';
const SCRUB_MARKER = Object.freeze({
  token: '[SCRUBBED:token]',
  keyvalue: '[SCRUBBED:keyvalue]',
  envvalue: '[SCRUBBED:envvalue]',
});
const RUN_EVENT_TYPES = Object.freeze({
  STEP_START: 'step_start',
  STEP_END: 'step_end',
  GATE_RESULT: 'gate_result',
  DISPATCH_ERROR: 'dispatch_error',
  TIMEOUT_KILL: 'timeout_kill',
  RETRY_ATTEMPT: 'retry_attempt',
  FALLBACK: 'fallback',
  COOLDOWN_SET: 'cooldown_set',
  SCRUB_FAILURE: 'scrub_failure',
  LEDGER_FLUSH_FAILURE: 'ledger_flush_failure',
});
const MIN_ENV_SECRET_LENGTH = 8;
const OUTPUT_BODY_FIELDS = new Set(['output', 'stdout', 'stderr', 'body', 'text']);
const KEY_VALUE_PATTERN =
  /(^|[^A-Za-z0-9_.-])([A-Za-z0-9_.-]{1,80})(\s*[:=]\s*)([^\r\n]+)/g;
const CREDENTIAL_KEY_PATTERN =
  /(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|ACCESS[_-]?KEY|PRIVATE[_-]?KEY|CLIENT[_-]?SECRET|AUTH[_-]?TOKEN|SESSION[_-]?TOKEN|WEBHOOK[_-]?SECRET|SIGNING[_-]?KEY)/i;
const TOKEN_PATTERNS = [
  /(^|[^A-Za-z0-9_])(sk-[A-Za-z0-9_-]{16,})/g,
  /(^|[^A-Za-z0-9_])(gh[opsur]_[A-Za-z0-9_]{20,})/g,
  /(^|[^A-Za-z0-9_])(github_pat_[A-Za-z0-9_]{20,})/g,
  /(^|[^A-Za-z0-9_])(AKIA[0-9A-Z]{16})/g,
  /(^|[^A-Za-z0-9_])(xox[baprs]-[A-Za-z0-9-]{10,})/g,
];
const BEARER_TOKEN_PATTERN = /\b((?:Bearer|Token)\s+)([A-Za-z0-9._~+/=-]{32,})/gi;

const DEFAULT_DEBATE = {
  comparators: ['claude', 'codex', 'kimi'],
  rebuttal: 'claude',
  synthesis: 'claude',
};
const REVIEW_FALLBACK_ROLES = new Set([
  'reviewer',
  'comparator',
  'rebuttal',
  'synthesis',
  'redteam',
  'blind',
]);
const WRITE_CAPABLE_STEP_ROLES = new Set(['owner']);
const REVIEW_LANE_ELIGIBLE_MODELS = new Set(['claude', 'codex', 'kimi']);

class ProviderResolutionError extends Error {
  constructor(message, { role, providerDiagnostics = [], failureLedger = [], tried = [] } = {}) {
    super(message);
    this.name = 'ProviderResolutionError';
    this.providerResolution = true;
    this.role = role;
    this.providerDiagnostics = providerDiagnostics;
    this.failureLedger = failureLedger;
    this.tried = tried;
  }
}

function normalizeEnvSecrets(values = []) {
  const unique = new Set();
  for (const value of values || []) {
    const text = String(value ?? '').trim();
    if (text.length >= MIN_ENV_SECRET_LENGTH) {
      unique.add(text);
    }
  }
  return [...unique].sort((left, right) => right.length - left.length);
}

function stripEnvQuotes(value) {
  const text = String(value || '').trim();
  if (text.length >= 2) {
    const first = text[0];
    const last = text[text.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return text.slice(1, -1);
    }
  }
  return text;
}

function parseEnvSecretValues(contents) {
  const values = [];
  for (const rawLine of String(contents || '').split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const withoutExport = line.startsWith('export ') ? line.slice('export '.length).trim() : line;
    const separator = withoutExport.indexOf('=');
    if (separator <= 0) continue;

    const value = stripEnvQuotes(withoutExport.slice(separator + 1));
    values.push(value);
  }
  return normalizeEnvSecrets(values);
}

function loadEnvSecretValues({
  envPath = join(ROOT, '.env'),
  fs = { existsSync, readFileSync },
} = {}) {
  try {
    if (!fs.existsSync(envPath)) return [];
    return parseEnvSecretValues(fs.readFileSync(envPath, 'utf8'));
  } catch {
    return [];
  }
}

function resolveEnvSecrets({ envSecrets, envLoader, envPath } = {}) {
  if (Array.isArray(envSecrets)) return normalizeEnvSecrets(envSecrets);
  if (typeof envLoader === 'function') {
    try {
      return normalizeEnvSecrets(envLoader({ envPath }) || []);
    } catch {
      return [];
    }
  }
  return loadEnvSecretValues({ envPath });
}

function scrubSecrets(text, { envSecrets = [] } = {}) {
  let scrubbedText = String(text ?? '');
  let count = 0;

  scrubbedText = scrubbedText.replace(KEY_VALUE_PATTERN, (match, prefix, key, separator, value) => {
    if (!CREDENTIAL_KEY_PATTERN.test(key) || String(value).trim() === '') {
      return match;
    }
    count += 1;
    return `${prefix}${key}${separator}${SCRUB_MARKER.keyvalue}`;
  });

  for (const pattern of TOKEN_PATTERNS) {
    scrubbedText = scrubbedText.replace(pattern, (_match, prefix) => {
      count += 1;
      return `${prefix}${SCRUB_MARKER.token}`;
    });
  }

  scrubbedText = scrubbedText.replace(BEARER_TOKEN_PATTERN, (_match, prefix) => {
    count += 1;
    return `${prefix}${SCRUB_MARKER.token}`;
  });

  for (const secret of normalizeEnvSecrets(envSecrets)) {
    const parts = scrubbedText.split(secret);
    if (parts.length === 1) continue;
    count += parts.length - 1;
    scrubbedText = parts.join(SCRUB_MARKER.envvalue);
  }

  return { text: scrubbedText, count };
}

function scrubOutputBody(text, { scrubber = scrubSecrets, envSecrets = [] } = {}) {
  try {
    const result = scrubber(String(text ?? ''), { envSecrets });
    return {
      text: String(result?.text ?? ''),
      count: Number.isInteger(result?.count) ? result.count : 0,
      error: false,
    };
  } catch {
    return { text: SCRUB_ERROR_TEXT, count: 0, error: true };
  }
}

function scrubOutputFields(container, { scrubber, envSecrets }) {
  let count = 0;
  let error = false;
  if (!container || typeof container !== 'object') {
    return { count, error };
  }

  for (const [key, value] of Object.entries(container)) {
    if (OUTPUT_BODY_FIELDS.has(key) && typeof value === 'string') {
      const scrubbed = scrubOutputBody(value, { scrubber, envSecrets });
      container[key] = scrubbed.text;
      count += scrubbed.count;
      error = error || scrubbed.error;
    } else if (value && typeof value === 'object') {
      const nested = scrubOutputFields(value, { scrubber, envSecrets });
      count += nested.count;
      error = error || nested.error;
    }
  }

  return { count, error };
}

function prepareRunLedgerRecord(record, options = {}) {
  const prepared = JSON.parse(JSON.stringify(record));
  const envSecrets = resolveEnvSecrets(options);
  const scrubber = options.scrubber || scrubSecrets;

  if (Array.isArray(prepared.steps)) {
    for (const step of prepared.steps) {
      if (!step || typeof step !== 'object') continue;
      const priorCount = Number.isInteger(step.scrubCount) ? step.scrubCount : 0;
      const scrubbed = scrubOutputFields(step, { scrubber, envSecrets });
      step.scrubCount = priorCount + scrubbed.count;
      if (scrubbed.error || step.scrubError) {
        step.scrubError = true;
      }
    }
  }

  for (const field of ['model', 'preflight', 'postflight', 'gate']) {
    if (prepared[field] && typeof prepared[field] === 'object') {
      scrubOutputFields(prepared[field], { scrubber, envSecrets });
    }
  }

  return prepared;
}

function getRunsDir(root = ROOT) {
  return join(root, 'ai-logs', 'hermes', 'runs');
}

function getRunLedgerPath(runId, { root = ROOT } = {}) {
  return join(getRunsDir(root), `${runId}.json`);
}

function getRunEventsPath(runId, { root = ROOT } = {}) {
  return join(getRunsDir(root), `${runId}.events.jsonl`);
}

function getDebateRunRecordPath(dateStamp, { root = ROOT } = {}) {
  return join(getRunsDir(root), `debate-run-${dateStamp}.json`);
}

function coerceDate(value) {
  const date = value instanceof Date ? value : new Date(value);
  return Number.isFinite(date.getTime()) ? date : null;
}

function dateFromHermesRunId(runId) {
  const match = String(runId || '').match(
    /^hermes-(\d{4}-\d{2}-\d{2}T)(\d{2})-(\d{2})-(\d{2})-(\d{3}Z)$/
  );
  if (!match) return null;
  return coerceDate(`${match[1]}${match[2]}:${match[3]}:${match[4]}.${match[5]}`);
}

function safeDateStamp(date) {
  return date.toISOString().replace(/[:.]/g, '-');
}

function projectNameFromRoot(root = ROOT) {
  return String(basename(root) || 'project').replace(/[^A-Za-z0-9._-]/g, '-');
}

function debateTimestamp({ record, clock }) {
  if (typeof clock === 'function') {
    const date = coerceDate(clock());
    if (date) return date;
  }
  return dateFromHermesRunId(record?.runId) || new Date();
}

function modelFamilyFor(model) {
  const value = String(model || '').toLowerCase();
  if (value.includes('claude')) return 'claude';
  if (value.includes('codex') || value.includes('openai') || /^gpt[-_]/.test(value)) {
    return 'openai';
  }
  if (value.includes('kimi')) return 'moonshot';
  if (value.includes('gemini') || value.includes('agy')) return 'gemini';
  if (value.includes('ollama')) return 'local';
  if (value.includes('qwen')) return 'qwen';
  return 'unknown';
}

function debateLaneStatus(step) {
  return (step?.code ?? 0) === 0 ? 'ok' : 'failed';
}

function debateFailureDetail(step, lane) {
  return `${lane} lane exited with code ${step?.code ?? 1}`;
}

function extractDebateVerdict(step) {
  const output = String(step?.output || '');
  if (/\bAPPROVE[-\s]?WITH[-\s]?CHANGES\b/i.test(output) || /CHANGES REQUESTED/i.test(output)) {
    return 'APPROVE-WITH-CHANGES';
  }
  if (/\bREJECT(?:ED)?\b/i.test(output)) {
    return 'REJECT';
  }
  if (/\bAPPROVE(?:D)?\b/i.test(output)) {
    return 'APPROVE';
  }
  if (step?.approved === true) return 'APPROVE';
  if (step?.approved === false) return 'APPROVE-WITH-CHANGES';
  return 'n/a';
}

function buildDebateRunRecord(record, { root = ROOT, clock } = {}) {
  const completedDate = debateTimestamp({ record, clock });
  const completedAt = record.completedAt || completedDate.toISOString();
  const startedAt =
    record.startedAt || dateFromHermesRunId(record.runId)?.toISOString() || completedAt;
  const project = record.project || projectNameFromRoot(root);
  const steps = Array.isArray(record.steps) ? record.steps : [];
  const comparatorSteps = steps.filter((step) => step?.role === 'comparator');
  const rebuttalSteps = steps.filter((step) => step?.role === 'rebuttal');
  const synthesisStep = steps.find((step) => step?.role === 'synthesis') || null;
  const lanes = [];
  const providerDiagnostics = [];
  const deviationsFromSpec = [
    {
      what: 'router runs model-based comparators, not distinct redteam/blind lanes',
      why: 'router transport maps N comparator models onto the spec lane set',
    },
  ];

  const addLane = (step, lane) => {
    const status = debateLaneStatus(step);
    const requested = step.requestedModel || null;
    const selected = step.selectedModel || step.model || null;
    const entry = {
      lane,
      modelFamily: modelFamilyFor(step.model),
      model: step.model || null,
      transport: 'router',
      threadId: step.threadId || null,
      rounds: 1,
      outputFile: step.outputFile || null,
      status,
      verdict: 'n/a',
    };
    lanes.push(entry);

    if (status === 'failed') {
      const detail = debateFailureDetail(step, lane);
      deviationsFromSpec.push({ what: `${lane} lane failed`, why: detail });
      providerDiagnostics.push({
        provider: step.model || lane,
        status: step.failure || 'nonzero_exit',
        detail,
      });
    }

    if (requested && selected && requested !== selected) {
      deviationsFromSpec.push({
        what: `${lane} lane substituted ${requested} with ${selected}`,
        why: 'router fallback selected a different provider',
      });
    }
  };

  // The router uses model-based comparator lanes; this is the explicit v1
  // impedance match to the role-named debate spec lanes.
  for (const step of comparatorSteps) {
    addLane(step, step.model || 'unknown');
  }
  for (const step of rebuttalSteps) {
    addLane(step, 'rebuttal');
  }

  const synthesisSkipped = comparatorSteps.length > 0 && !synthesisStep;
  if (synthesisSkipped) {
    deviationsFromSpec.push({
      what: 'synthesis step skipped',
      why: 'debate workflow completed comparator/rebuttal lanes without a synthesis step record',
    });
  }

  return {
    runId: `debate-${project}-${completedDate.toISOString()}`,
    workflow: 'hermes-debate/v1',
    project,
    artifactReviewed: record.artifactReviewed || null,
    startedAt,
    completedAt,
    lanes,
    providerDiagnostics,
    deviationsFromSpec,
    synthesisFile: synthesisStep?.outputFile || null,
    finalVerdict: synthesisSkipped ? null : synthesisStep ? extractDebateVerdict(synthesisStep) : null,
  };
}

function validateDebateRunRecordShape(record) {
  const errors = [];
  const warnings = [];
  const requireString = (field) => {
    if (typeof record?.[field] !== 'string' || record[field].length === 0) {
      errors.push(`${field} must be a non-empty string`);
    }
  };

  for (const field of ['runId', 'workflow', 'project', 'startedAt', 'completedAt']) {
    requireString(field);
  }
  if (!Array.isArray(record?.lanes)) errors.push('lanes must be an array');
  if (!Array.isArray(record?.providerDiagnostics)) {
    errors.push('providerDiagnostics must be an array');
  }
  if (!Array.isArray(record?.deviationsFromSpec)) {
    errors.push('deviationsFromSpec must be an array');
  }
  if (record?.finalVerdict !== null && typeof record?.finalVerdict !== 'string') {
    errors.push('finalVerdict must be a string or null');
  }

  const laneNames = new Set(['claude', 'redteam', 'blind', 'rebuttal']);
  const statuses = new Set(['ok', 'failed', 'fallback']);
  const verdicts = new Set(['APPROVE', 'APPROVE-WITH-CHANGES', 'REJECT', 'n/a']);
  const finalVerdicts = new Set(['APPROVE', 'APPROVE-WITH-CHANGES', 'REJECT']);

  if (Array.isArray(record?.lanes)) {
    record.lanes.forEach((lane, index) => {
      if (typeof lane?.lane !== 'string' || lane.lane.length === 0) {
        errors.push(`lanes[${index}].lane must be a non-empty string`);
      } else if (!laneNames.has(lane.lane)) {
        warnings.push(`lanes[${index}].lane is outside the role-name enum`);
      }
      if (typeof lane?.status !== 'string' || lane.status.length === 0) {
        errors.push(`lanes[${index}].status must be present`);
      } else if (!statuses.has(lane.status)) {
        warnings.push(`lanes[${index}].status is outside the status enum`);
      }
      if (typeof lane?.verdict !== 'string' || lane.verdict.length === 0) {
        errors.push(`lanes[${index}].verdict must be present`);
      } else if (!verdicts.has(lane.verdict)) {
        warnings.push(`lanes[${index}].verdict is outside the verdict enum`);
      }
    });
  }

  if (typeof record?.finalVerdict === 'string' && !finalVerdicts.has(record.finalVerdict)) {
    warnings.push('finalVerdict is outside the final verdict enum');
  }

  return { valid: errors.length === 0, errors, warnings };
}

function removeOutputBodyFields(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => removeOutputBodyFields(entry));
  }
  if (!value || typeof value !== 'object') {
    return value;
  }

  const next = {};
  for (const [key, entry] of Object.entries(value)) {
    if (OUTPUT_BODY_FIELDS.has(key)) continue;
    next[key] = removeOutputBodyFields(entry);
  }
  return next;
}

function prepareMetadataRecord(record, options = {}) {
  const envSecrets = resolveEnvSecrets(options);
  const scrubber = options.scrubber || scrubSecrets;
  let scrubError = false;

  const scrubValue = (value) => {
    if (typeof value === 'string') {
      const scrubbed = scrubOutputBody(value, { scrubber, envSecrets });
      scrubError = scrubError || scrubbed.error;
      return scrubbed.text;
    }
    if (Array.isArray(value)) {
      return value.map((entry) => scrubValue(entry));
    }
    if (value && typeof value === 'object') {
      const next = {};
      for (const [key, entry] of Object.entries(value)) {
        next[key] = scrubValue(entry);
      }
      return next;
    }
    return value;
  };

  return {
    record: scrubValue(removeOutputBodyFields(record)),
    scrubError,
  };
}

function hasScrubError(value) {
  if (!value || typeof value !== 'object') return false;
  if (value.scrubError === true) return true;
  if (Array.isArray(value)) return value.some((entry) => hasScrubError(entry));
  return Object.values(value).some((entry) => hasScrubError(entry));
}

function compactStepRecord(step) {
  return {
    role: step.role,
    model: step.model,
    attempt: step.attempt,
    code: step.code,
    approved: step.approved,
    durationMs: step.durationMs,
  };
}

function formatAgeMs(ageMs) {
  const hours = Math.floor(ageMs / (60 * 60 * 1000));
  return `${hours}h`;
}

function warnLedgerFailure(io, message) {
  io.stderr.write(`[hermes] WARNING: failed to write run ledger: ${message}\n`);
}

function warnEventFailure(io, message) {
  io.stderr.write(`[hermes] WARNING: failed to write run event log: ${message}\n`);
}

function warnDebateRunRecordValidation(io, validation) {
  const count = validation.errors.length + validation.warnings.length;
  io.stderr.write(
    `[hermes] WARNING: debate run record validation reported ${count} issue(s); writing anyway.\n`
  );
}

function warnDebateRunRecordFailure(io, message) {
  io.stderr.write(`[hermes] WARNING: failed to write debate run record: ${message}\n`);
}

function writePreparedLedgerBestEffort({
  ledgerWriter,
  record,
  scrubOptions,
  io = process,
  emitEvent = null,
}) {
  if (!ledgerWriter) return;
  try {
    const prepared = prepareRunLedgerRecord(record, scrubOptions);
    if (hasScrubError(prepared) && emitEvent) {
      emitEvent({ type: RUN_EVENT_TYPES.SCRUB_FAILURE, where: 'run_ledger' });
    }
    ledgerWriter(prepared, { envSecrets: [] });
  } catch (error) {
    warnLedgerFailure(io, error.message);
    if (emitEvent) {
      emitEvent({ type: RUN_EVENT_TYPES.LEDGER_FLUSH_FAILURE, message: error.message });
    }
  }
}

function writeDebateRunRecordBestEffort({
  debateRunWriter,
  record,
  options,
  io = process,
  emitEvent = null,
}) {
  if (!debateRunWriter || record?.workflow !== 'debate') return;
  try {
    debateRunWriter(record, options);
  } catch (error) {
    warnDebateRunRecordFailure(io, error.message);
    if (emitEvent) {
      emitEvent({ type: RUN_EVENT_TYPES.LEDGER_FLUSH_FAILURE, message: error.message });
    }
  }
}

function normalizeEntrypointPath(value) {
  return String(value || '')
    .replace(/^file:\/\//, '')
    .replace(/\\/g, '/')
    .replace(/^\/([A-Za-z]:\/)/, '$1')
    .toLowerCase();
}

function isCliEntryPoint(metaUrl, argv = process.argv) {
  if (!argv[1]) return false;

  let modulePath = metaUrl;
  try {
    modulePath = fileURLToPath(metaUrl);
  } catch {
    modulePath = metaUrl;
  }

  return normalizeEntrypointPath(modulePath) === normalizeEntrypointPath(argv[1]);
}

function parseArgs(argv = []) {
  const options = {
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
    project: null,
    legacyCommand: null,
  };

  if (argv[0] && LEGACY_COMMANDS.has(argv[0])) {
    options.legacyCommand = argv[0];
    return options;
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else if (arg === '--dry-run') {
      options.dryRun = true;
    } else if (arg === '--json') {
      options.json = true;
    } else if (arg === '--live') {
      options.live = true;
    } else if (arg === '--allow-fallback') {
      options.allowFallback = true;
    } else if (arg === '--skip-gates') {
      throw new Error('Use --skip-preflight-gate with --skip-reason instead of --skip-gates.');
    } else if (arg === '--skip-preflight-gate') {
      options.skipPreflightGate = true;
    } else if (arg === '--skip-reason') {
      options.gateSkipReason = argv[index + 1] || '';
      index += 1;
    } else if (arg === '--project' || arg === '--cwd') {
      options.project = resolve(argv[index + 1] || '');
      index += 1;
    } else if (arg === '--workflow') {
      const workflow = argv[index + 1] || '';
      if (!WORKFLOW_MODES.has(workflow)) {
        throw new Error(
          `Unknown workflow "${workflow}". Expected one of: ${[...WORKFLOW_MODES].join(', ')}.`
        );
      }
      options.workflow = workflow;
      options.workflowProvided = true;
      index += 1;
    } else if (arg === '--model') {
      const model = argv[index + 1] || '';
      if (!MODEL_OVERRIDES.has(model)) {
        throw new Error(
          `Unknown model "${model}". Expected one of: ${[...MODEL_OVERRIDES].join(', ')}.`
        );
      }
      options.manualModel = model;
      index += 1;
    } else if (arg === '--phase') {
      options.phase = argv[index + 1] || options.phase;
      index += 1;
    } else if (arg === '--task') {
      options.task = argv[index + 1] || '';
      index += 1;
    } else if (arg === '--claude') {
      options.manualModel = 'claude';
    } else if (arg === '--codex') {
      options.manualModel = 'codex';
    } else if (arg === '--kimi') {
      options.manualModel = 'kimi';
    } else if (arg === '--gemini') {
      options.manualModel = 'gemini';
    } else if (arg === '--agy') {
      options.manualModel = 'agy';
    }
  }

  options.task = options.task.trim();
  options.gateSkipReason = options.gateSkipReason?.trim() || null;

  if (options.skipPreflightGate && !options.gateSkipReason) {
    throw new Error('--skip-preflight-gate requires --skip-reason <reason>.');
  }

  return options;
}

function loadJSON(filePath, { fs = { existsSync, readFileSync } } = {}) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing JSON file: ${filePath}`);
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new SyntaxError(`Invalid JSON in ${filePath}: ${error.message}`);
    }
    throw error;
  }
}

function loadText(filePath, { optional = false, fs = { existsSync, readFileSync } } = {}) {
  if (!fs.existsSync(filePath)) {
    if (optional) return '';
    throw new Error(`Missing text file: ${filePath}`);
  }
  return fs.readFileSync(filePath, 'utf8');
}

function clockDate(clock = () => new Date()) {
  if (typeof clock?.now === 'function') return clock.now();
  if (typeof clock === 'function') return clock();
  return new Date();
}

function providerStatePath(root = ROOT) {
  return join(root, 'ai-logs', 'hermes', 'provider-state.json');
}

function normalizeProviderState(record) {
  const providers = {};
  for (const [provider, entry] of Object.entries(record?.providers || {})) {
    if (!entry || typeof entry !== 'object') continue;
    const untilMs = new Date(entry.coolingUntil).getTime();
    if (!Number.isFinite(untilMs)) continue;
    providers[provider] = {
      coolingUntil: new Date(untilMs).toISOString(),
      reason: String(entry.reason || 'rate_limited'),
      triggeredAt: entry.triggeredAt || null,
    };
  }
  return { providers };
}

function warnProviderState(io, message) {
  io.stderr.write(`[hermes] WARNING: ${message}\n`);
}

function createProviderCooldownStore({
  root = ROOT,
  fs = { existsSync, readFileSync, writeFileSync, renameSync, mkdirSync },
  clock = () => new Date(),
  io = process,
} = {}) {
  const file = providerStatePath(root);
  let warnedCorrupt = false;
  let warnedWrite = false;

  const readState = () => {
    if (!fs.existsSync(file)) return { providers: {} };
    try {
      return normalizeProviderState(JSON.parse(fs.readFileSync(file, 'utf8')));
    } catch (error) {
      if (!warnedCorrupt) {
        warnedCorrupt = true;
        warnProviderState(
          io,
          `provider cooldown state was corrupt; sidelining ${file} and starting empty: ${error.message}`
        );
      }
      try {
        fs.renameSync(file, `${file}.corrupt`);
      } catch {
        // Advisory state must fail open even when the corrupt file cannot be moved.
      }
      return { providers: {} };
    }
  };

  const maxState = (left, right) => {
    const merged = normalizeProviderState(left);
    for (const [provider, entry] of Object.entries(normalizeProviderState(right).providers)) {
      const prior = merged.providers[provider];
      if (!prior || new Date(entry.coolingUntil).getTime() > new Date(prior.coolingUntil).getTime()) {
        merged.providers[provider] = entry;
      }
    }
    return merged;
  };

  const writeState = (nextState) => {
    // Re-read immediately before writing so a parallel run cannot erase a longer advisory cooldown.
    const merged = maxState(readState(), nextState);
    const tempFile = `${file}.${process.pid}.${clockDate(clock).getTime()}.tmp`;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        fs.mkdirSync(dirname(file), { recursive: true });
        fs.writeFileSync(tempFile, `${JSON.stringify(merged, null, 2)}\n`);
        fs.renameSync(tempFile, file);
        return merged;
      } catch (error) {
        if ((error?.code === 'EPERM' || error?.code === 'EBUSY') && attempt === 0) {
          continue;
        }
        if (!warnedWrite) {
          warnedWrite = true;
          warnProviderState(io, `failed to persist provider cooldown state; skipping: ${error.message}`);
        }
        return nextState;
      }
    }
    return nextState;
  };

  const coolingUntil = (provider) => readState().providers?.[provider]?.coolingUntil || null;

  return {
    coolingUntil,
    isCooling(provider) {
      const until = coolingUntil(provider);
      return until ? new Date(until).getTime() > clockDate(clock).getTime() : false;
    },
    window(provider) {
      return this.isCooling(provider) ? coolingUntil(provider) : '-';
    },
    setCooldown(provider, { routing = {}, message = '' } = {}) {
      const now = clockDate(clock);
      const nowMs = now.getTime();
      const state = readState();
      const prior = state.providers?.[provider] || null;
      const priorMs = prior ? new Date(prior.coolingUntil).getTime() : NaN;
      const defaultMs = cooldownDefaultMinutes(provider, routing.rateLimits?.defaults) * 60 * 1000;
      const explicitUntilMs = parseExplicitCooldownUntil(message, nowMs);
      let durationMs = explicitUntilMs ? explicitUntilMs - nowMs : defaultMs;
      if (!Number.isFinite(durationMs) || durationMs <= 0) {
        durationMs = defaultMs;
      }
      const repeat =
        Number.isFinite(priorMs) &&
        (priorMs > nowMs || nowMs <= priorMs + COOLDOWN_REPEAT_WINDOW_MS);
      if (repeat) {
        // Repeat means active cooling or a trigger within ten minutes after expiry.
        durationMs *= 2;
      }
      durationMs = Math.min(durationMs, COOLDOWN_MAX_MS);
      const coolingUntilValue = new Date(nowMs + durationMs).toISOString();
      state.providers[provider] = {
        coolingUntil: coolingUntilValue,
        reason: 'rate_limited',
        triggeredAt: now.toISOString(),
      };
      const persisted = writeState(state);
      return persisted.providers?.[provider] || state.providers[provider];
    },
  };
}

function cooldownDefaultMinutes(provider, defaults = {}) {
  const providerDefaults = defaults.providers || defaults.providerMinutes || {};
  const value =
    providerDefaults[provider] ??
    defaults[provider] ??
    defaults[`${provider}Minutes`] ??
    defaults.defaultMinutes ??
    defaults.fallbackMinutes ??
    DEFAULT_PROVIDER_COOLDOWN_MINUTES;
  const minutes = Number(value);
  return Number.isFinite(minutes) && minutes > 0 ? minutes : DEFAULT_PROVIDER_COOLDOWN_MINUTES;
}

function parseExplicitCooldownUntil(message, nowMs) {
  const text = String(message || '');
  const relative = text.match(
    /(?:retry\s*after|try\s*again\s*in)\s*:?\s*(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)?/i
  );
  if (relative) {
    const amount = Number(relative[1]);
    const unit = String(relative[2] || 'seconds').toLowerCase();
    const scale = unit.startsWith('h') ? 60 * 60 * 1000 : unit.startsWith('m') ? 60 * 1000 : 1000;
    return nowMs + amount * scale;
  }

  const absolute = text.match(
    /(?:retry\s*after|resets?\s*(?:at|on)?|reset\s*time)\s*:?\s*([^\r\n;]+)/i
  );
  if (!absolute) return null;
  const parsed = Date.parse(absolute[1].trim());
  return Number.isFinite(parsed) && parsed > nowMs ? parsed : null;
}

function compileRateLimitSignatures(rateLimits = {}) {
  if (rateLimits?.enabled !== true) return {};
  const compiled = {};
  for (const [provider, entry] of Object.entries(rateLimits.signatures || {})) {
    const items = Array.isArray(entry) ? entry : [entry];
    compiled[provider] = items
      .map((item) => {
        const source = typeof item === 'string' ? item : item?.source;
        if (!source) return null;
        return new RegExp(source, typeof item === 'string' ? 'i' : item.flags || '');
      })
      .filter(Boolean);
  }
  return compiled;
}

function resolveRateLimitSignatures({ configured, injected, provider }) {
  const source = injected ?? configured;
  if (Array.isArray(source)) return source;
  return source?.[provider] || [];
}

function diagnosticTextForResult(result) {
  return [result?.output, result?.stderr, result?.error?.message, result?.error?.code]
    .filter(Boolean)
    .join('\n');
}

function scanUnclassifiedFailures({
  root = ROOT,
  fs = { readdirSync, readFileSync },
} = {}) {
  const dir = getRunsDir(root);
  let names = [];
  try {
    names = fs.readdirSync(dir);
  } catch {
    return 0;
  }

  let count = 0;
  for (const name of names) {
    if (!String(name).endsWith('.json') || String(name).endsWith('.events.jsonl')) continue;
    try {
      const record = JSON.parse(fs.readFileSync(join(dir, name), 'utf8'));
      for (const ledger of record.failureLedger || []) {
        for (const diagnostic of ledger.providerDiagnostics || []) {
          if (
            diagnostic?.status === 'failed' &&
            /\bfailed \((?!rate_limited\))/i.test(String(diagnostic.detail || ''))
          ) {
            count += 1;
          }
        }
      }
    } catch {
      // Doctor best-effort evidence should not mask provider readiness.
    }
  }
  return count;
}

function scoreSpecialist(task, specialists = {}, scoring = {}) {
  const input = task.toLowerCase();
  const minScore = scoring.minScoreToAssign ?? 3;
  const riskOrder = scoring.riskOrder ?? ['financial', 'operational', 'quality'];
  const candidates = [];

  for (const [name, config] of Object.entries(specialists)) {
    let score = 0;
    let maxScore = 0;
    const matched = [];

    for (const keyword of config.keywords || []) {
      const phrase = String(
        typeof keyword === 'string' ? keyword : keyword.phrase || ''
      ).toLowerCase();
      const weight = typeof keyword === 'string' ? 1 : keyword.weight || 1;

      maxScore += weight;
      if (phrase && input.includes(phrase)) {
        score += weight;
        matched.push(phrase);
      }
    }

    if (score >= minScore) {
      candidates.push({
        name,
        score,
        maxScore,
        confidence: maxScore > 0 ? Math.round((score / maxScore) * 100) / 100 : 0,
        matched,
        risk: config.risk || 'quality',
      });
    }
  }

  if (candidates.length === 0) return null;

  candidates.sort((left, right) => {
    if (right.score !== left.score) return right.score - left.score;

    const leftRisk = riskOrder.indexOf(left.risk);
    const rightRisk = riskOrder.indexOf(right.risk);
    return (leftRisk === -1 ? 99 : leftRisk) - (rightRisk === -1 ? 99 : rightRisk);
  });

  const [selected] = candidates;
  return { ...selected, candidates };
}

function chooseModel(task, phase, routing, manualModel = null) {
  if (manualModel) return manualModel;

  const input = task.toLowerCase();
  for (const trigger of routing.longContextTriggers || []) {
    if (input.includes(String(trigger).toLowerCase())) {
      return routing.longContextModel || 'kimi';
    }
  }

  return routing.defaults?.[phase] || 'claude';
}

function ladderEnabled(routing = {}) {
  return routing?.ladder?.enabled === true;
}

function configuredLadderRungs(routing = {}) {
  return Array.isArray(routing?.ladder?.rungs) ? routing.ladder.rungs.map(String) : [];
}

function commandBinForProvider(provider, routing = {}, env = process.env) {
  const commandConfig = routing.commands?.[provider] || null;
  if (!commandConfig) return { commandConfig: null, bin: null };
  const bin = env?.[commandConfig.binEnv] || commandConfig.defaultBin || provider;
  return { commandConfig, bin };
}

function providerStatus({
  provider,
  routing,
  env = process.env,
  commandExists: checkCommandExists = commandExists,
  isCooling = () => false,
}) {
  const { commandConfig, bin } = commandBinForProvider(provider, routing, env);
  if (!commandConfig || !bin || !checkCommandExists(bin)) {
    return { available: false, reason: 'command missing', bin, commandConfig };
  }
  if (isCooling(provider)) {
    return { available: false, reason: 'cooling', bin, commandConfig };
  }
  return { available: true, reason: null, bin, commandConfig };
}

function noAvailableProviderMessage(role, tried) {
  return `no available provider for role ${role}; tried: [${tried.join(', ')}]`;
}

function financialNoDegradeMessage(role, provider, reason) {
  return `provider ${provider} ${reason}; role ${role} refused to degrade under a financial-risk classification`;
}

function createProviderFailureLedger({ role, message, tried, providerDiagnostics }) {
  return [
    {
      role,
      message,
      tried,
      providerDiagnostics,
    },
  ];
}

function resolveLadderRungsFrom({
  requestedModel,
  rungs,
  manualPinned = false,
  allowFallback = false,
}) {
  if (manualPinned && allowFallback) {
    const requestedIndex = rungs.indexOf(requestedModel);
    if (requestedIndex >= 0) {
      return rungs.slice(requestedIndex);
    }
    return [requestedModel, ...rungs];
  }
  return rungs;
}

function resolveLadderProvider({
  role,
  requestedModel,
  routing,
  env = process.env,
  commandExists: checkCommandExists = commandExists,
  isCooling = () => false,
  risk = 'standard',
  manualPinned = false,
  allowFallback = false,
}) {
  if (!ladderEnabled(routing)) {
    return { enabled: false, selectedModel: requestedModel };
  }

  const rungs = configuredLadderRungs(routing);
  const candidateRungs = resolveLadderRungsFrom({
    requestedModel,
    rungs,
    manualPinned,
    allowFallback,
  });
  const noDegrade = risk === 'financial' || (manualPinned && !allowFallback);

  if (noDegrade) {
    const status = providerStatus({
      provider: requestedModel,
      routing,
      env,
      commandExists: checkCommandExists,
      isCooling,
    });
    if (status.available) {
      return {
        enabled: true,
        requestedModel,
        selectedModel: requestedModel,
        degradeReason: null,
        providerDiagnostics: [],
      };
    }

    const detail =
      risk === 'financial'
        ? financialNoDegradeMessage(role, requestedModel, status.reason)
        : `manual provider ${requestedModel} unavailable (${status.reason}); fallback disabled`;
    const providerDiagnostics = [{ provider: requestedModel, status: 'failed', detail }];
    throw new ProviderResolutionError(detail, {
      role,
      providerDiagnostics,
      failureLedger: createProviderFailureLedger({
        role,
        message: detail,
        tried: [requestedModel],
        providerDiagnostics,
      }),
      tried: [requestedModel],
    });
  }

  const providerDiagnostics = [];
  for (const provider of candidateRungs) {
    const status = providerStatus({
      provider,
      routing,
      env,
      commandExists: checkCommandExists,
      isCooling,
    });
    if (status.available) {
      const requestedDiagnostic = providerDiagnostics.find(
        (diagnostic) => diagnostic.provider === requestedModel
      );
      return {
        enabled: true,
        requestedModel,
        selectedModel: provider,
        degradeReason:
          provider === requestedModel
            ? null
            : requestedDiagnostic?.detail || `${requestedModel} degraded to ${provider} by ladder`,
        providerDiagnostics,
      };
    }
    const detail =
      status.reason === 'cooling'
        ? `${provider} cooling`
        : `${provider} unavailable (${status.reason})`;
    providerDiagnostics.push({ provider, status: 'skipped', detail });
  }

  const message = noAvailableProviderMessage(role, candidateRungs);
  throw new ProviderResolutionError(message, {
    role,
    providerDiagnostics,
    failureLedger: createProviderFailureLedger({
      role,
      message,
      tried: candidateRungs,
      providerDiagnostics,
    }),
    tried: candidateRungs,
  });
}

function applyLadderToStep(step, options) {
  if (!step.model || !ladderEnabled(options.routing)) return step;

  const requestedModel = step.model;
  const resolved = resolveLadderProvider({
    ...options,
    role: step.role,
    requestedModel,
  });
  const degradedReviewLane = resolved.degradeReason && REVIEW_FALLBACK_ROLES.has(step.role);
  return {
    ...step,
    requestedModel,
    selectedModel: resolved.selectedModel,
    model: resolved.selectedModel,
    degradeReason: resolved.degradeReason,
    providerDiagnostics: resolved.providerDiagnostics,
    manualPinned: options.manualPinned,
    allowFallback: options.allowFallback,
    ...(degradedReviewLane
      ? {
          weakenedControl: true,
          deviation: 'review-lane provider fallback weakened independence control',
        }
      : {}),
  };
}

function applyLadderToPlan(plan, options) {
  if (!ladderEnabled(options.routing)) return plan;

  const ladder = {
    enabled: true,
    rungs: configuredLadderRungs(options.routing),
  };
  const next = {
    ...plan,
    ladder,
    allowFallback: options.allowFallback,
    manualPinned: options.manualPinned,
  };

  const resolvedPlanModel = resolveLadderProvider({
    ...options,
    role: 'solo',
    requestedModel: plan.model,
  });
  next.requestedModel = plan.model;
  next.selectedModel = resolvedPlanModel.selectedModel;
  next.model = resolvedPlanModel.selectedModel;
  next.degradeReason = resolvedPlanModel.degradeReason;
  next.providerDiagnostics = resolvedPlanModel.providerDiagnostics;

  if (next.workflow?.steps) {
    next.workflow = {
      ...next.workflow,
      steps: next.workflow.steps.map((step) => applyLadderToStep(step, options)),
    };
  }

  return next;
}

function resolveGate(phase, specialist, gates = {}) {
  if (specialist?.risk === 'financial' && phase === 'production') {
    return gates['production-financial'] || gates.production || null;
  }
  return gates[phase] || null;
}

function resolveEffectivePhase(phase, specialist) {
  if (phase === 'production' && specialist?.risk === 'financial') {
    return 'production-financial';
  }
  return phase;
}

function resolveOwnership(phase, specialist, ownership = {}) {
  const effective = resolveEffectivePhase(phase, specialist);
  const entry = ownership[effective] || ownership[phase];
  if (!entry) return null;
  return { effectivePhase: effective, ...entry };
}

function recommendWorkflow({ phase, risk = 'standard' }) {
  if (risk === 'financial') return 'pair';
  if (phase === 'distribution') return 'review';
  if (phase === 'research') return 'chain';
  return 'solo';
}

function createWorkflowPlan({
  requestedWorkflow = 'auto',
  phase,
  model,
  specialist,
  gate,
  ownership = null,
  risk = 'standard',
  debate = null,
}) {
  if (!WORKFLOW_MODES.has(requestedWorkflow)) {
    throw new Error(
      `Unknown workflow "${requestedWorkflow}". Expected one of: ${[...WORKFLOW_MODES].join(', ')}.`
    );
  }

  const selected =
    requestedWorkflow === 'auto' ? recommendWorkflow({ phase, risk }) : requestedWorkflow;
  const effectivePhase = ownership?.effectivePhase || phase;
  const ownerModel = model || ownership?.owner;
  const artifact = ownership?.artifact || 'artifact';
  const steps = [];

  // 'review' mode reviews an existing artifact, so it never re-runs the owner lane;
  // because distribution ownership has no reviewer, distribution 'review' is
  // gate-only by design (the lint gate is the review). 'debate' replaces the single
  // owner lane with N comparators plus a synthesis step (added below).
  if (selected !== 'review' && selected !== 'debate' && ownerModel) {
    const role = ownership?.role ? ` ${ownership.role}` : '';
    steps.push({
      role: 'owner',
      model: ownerModel,
      action: `execute ${effectivePhase}${role} lane`,
    });
  }

  if ((selected === 'chain' || (selected === 'pair' && risk === 'financial')) && specialist) {
    // Specialists are roles, not models. Run the specialist review on a real
    // configured model (reviewer > audit > owner) and carry the specialist's
    // identity in the action so it adopts that persona.
    const specialistName = specialist.name || specialist;
    steps.push({
      role: 'specialist',
      model: ownership?.reviewer || ownership?.audit || ownerModel,
      action: `as the ${specialistName} specialist, review financial risk before completion`,
    });
  }

  if (
    (selected === 'pair' || selected === 'chain' || selected === 'review') &&
    ownership?.reviewer
  ) {
    steps.push({
      role: 'reviewer',
      model: ownership.reviewer,
      action: `review ${artifact}`,
    });
  }

  // Debate carries no audit step by design; synthesis is the accountable artifact.
  if ((selected === 'chain' || selected === 'pair') && ownership?.audit) {
    steps.push({
      role: 'audit',
      model: ownership.audit,
      action:
        risk === 'financial' ? 'audit financial readiness evidence' : 'audit workflow evidence',
    });
  }

  if (selected === 'debate') {
    const debateConfig = debate || DEFAULT_DEBATE;
    const comparatorModels =
      Array.isArray(debateConfig.comparators) && debateConfig.comparators.length > 0
        ? debateConfig.comparators
        : DEFAULT_DEBATE.comparators;
    const synthesisModel = debateConfig.synthesis || DEFAULT_DEBATE.synthesis;
    const rebuttalModel = debateConfig.rebuttal || synthesisModel;

    for (const comparatorModel of comparatorModels) {
      steps.push({
        role: 'comparator',
        model: comparatorModel,
        action: `compare ${effectivePhase} options`,
      });
    }

    steps.push({
      role: 'rebuttal',
      model: rebuttalModel,
      action: 'verify and rebut comparator critiques against ground truth',
    });

    steps.push({
      role: 'synthesis',
      model: synthesisModel,
      action: `synthesize ${effectivePhase} options into ${artifact}`,
    });
  }

  if (gate) {
    steps.push({
      role: 'gate',
      model: null,
      action: `run ${gate}`,
    });
  }

  return {
    requested: requestedWorkflow,
    selected,
    planningOnly: true,
    gate,
    deferred: WORKFLOW_DEFERRED_BEHAVIOR,
    steps,
  };
}

function createRoutingPlan({
  phase,
  task,
  routing,
  manualModel = null,
  allowFallback = false,
  requestedWorkflow = null,
  skipPreflightGate = false,
  gateSkipReason = null,
  env = process.env,
  commandExists: checkCommandExists = commandExists,
  isCooling = () => false,
}) {
  const specialist = scoreSpecialist(task, routing.specialists || {}, routing.scoring || {});
  const model = chooseModel(task, phase, routing, manualModel);
  const gate = resolveGate(phase, specialist, routing.gates || {});
  const ownership = resolveOwnership(phase, specialist, routing.ownership || {});
  const risk = specialist?.risk || 'standard';

  const plan = {
    phase,
    task,
    model,
    specialist: specialist?.name || null,
    risk,
    score: specialist?.score || 0,
    confidence: specialist?.confidence ?? 0,
    candidates: specialist?.candidates || [],
    gate,
  };

  if (requestedWorkflow) {
    plan.workflow = createWorkflowPlan({
      requestedWorkflow,
      phase,
      model,
      specialist,
      gate,
      ownership,
      risk,
      debate: routing.debate || null,
    });
  }

  if (ownership) {
    plan.ownership = ownership;
  }

  if (skipPreflightGate) {
    plan.gateSkip = {
      preflight: true,
      reason: gateSkipReason || 'unspecified',
    };
  }

  return applyLadderToPlan(plan, {
    routing,
    env,
    commandExists: checkCommandExists,
    isCooling,
    risk,
    manualPinned: Boolean(manualModel),
    allowFallback,
  });
}

function buildPrompt({ plan, brain, soul = '', runId = null }) {
  const lines = [
    'You are operating inside AIHedgeFund.',
    '',
    `PHASE: ${plan.phase}`,
    `MODEL ROLE: ${plan.model}`,
  ];

  if (plan.ownership) {
    const own = plan.ownership;
    const ownerLine = [
      `OWNER: ${own.owner}`,
      own.reviewer ? `reviewer: ${own.reviewer}` : null,
      own.role ? `role: ${own.role}` : null,
      own.artifact ? `artifact: ${own.artifact}` : null,
      own.humanApproval ? 'human approval required' : null,
    ]
      .filter(Boolean)
      .join('; ');
    lines.push(ownerLine);
  }

  if (plan.specialist) {
    const confidencePct = Math.round((plan.confidence ?? 0) * 100);
    lines.push(
      `SPECIALIST: ${plan.specialist} (risk: ${plan.risk}; confidence: ${confidencePct}%)`
    );
  }

  if (plan.gate) {
    lines.push(`REQUIRED GATE: ${plan.gate}`);
  }

  if (runId) {
    lines.push(`RUN ID: ${runId}`);
  }

  lines.push('', '--- DEV_BRAIN ---', brain.trim(), '--- END DEV_BRAIN ---');

  if (soul.trim()) {
    lines.push('', '--- HERMES_SOUL ---', soul.trim(), '--- END HERMES_SOUL ---');
  }

  lines.push(
    '',
    `TASK: ${plan.task}`,
    '',
    'Instructions:',
    '1. Read your governance file first when filesystem access is available.',
    '2. Search for existing implementations before proposing new code.',
    '3. Use .claude/DISCOVERY-MAP.md and .claude/AGENT-DIRECTORY.md for routing.',
    '4. Prefer existing specialists before inventing new abstractions.',
    '5. Produce the smallest safe diff.',
    '6. If financial logic is touched, confirm calc-gate coverage.',
    '7. Return: Summary, Affected Files, Changes or Plan, Verification, Risks.',
    '8. End with a compact Handoff block listing:',
    '   Run ID, Phase, Owner, Reviewer, Task, Protected areas, Files touched,',
    '   Commands run, Gate status, Decision needed, Next action.'
  );

  return lines.join('\n');
}

function commandExists(bin) {
  if (!bin) return false;
  // Absolute or relative path: check the filesystem directly. where.exe/which
  // only resolve bare names from PATH and error on absolute paths.
  if (/[\\/]/.test(bin)) {
    if (existsSync(bin)) return true;
    if (process.platform === 'win32' && !/\.[a-zA-Z0-9]+$/.test(bin)) {
      return ['.exe', '.cmd', '.bat'].some((ext) => existsSync(bin + ext));
    }
    return false;
  }
  const checker = process.platform === 'win32' ? 'where.exe' : 'which';
  const result = spawnSync(checker, [bin], { stdio: 'ignore' });
  return result.status === 0;
}

// Write capability is derived from the existing CLI args only; no new config
// semantics are introduced for W4.
function isWriteCapableCommand(commandConfig) {
  return (commandConfig?.args || []).some((arg) => arg === 'workspace-write' || arg === '--yolo');
}

function resolveSandboxArgs(routing, model, role) {
  const commandConfig = routing?.commands?.[model] || null;
  const normalArgs = Array.isArray(commandConfig?.args) ? commandConfig.args : [];
  if (WRITE_CAPABLE_STEP_ROLES.has(String(role || ''))) {
    return [...normalArgs];
  }

  if (Array.isArray(commandConfig?.readOnlyArgs)) {
    return [...commandConfig.readOnlyArgs];
  }

  // Claude and Kimi are text-only CLI lanes here; other providers without a
  // readOnlyArgs profile keep argv unchanged and must be governed separately.
  return [...normalArgs];
}

function isReviewLaneEligible(model) {
  const provider = String(model || '').toLowerCase();
  // Review-lane selection must consult this data before provider fallback:
  // agy has stdin delivery but unproven sandbox posture, while gemini is
  // auth-dead and its empty --prompt= config relies on stdin being honored.
  return REVIEW_LANE_ELIGIBLE_MODELS.has(provider);
}

function routingWithSandboxArgsForStep(routing, step) {
  const commandConfig = routing?.commands?.[step?.model] || null;
  if (!commandConfig) return routing;

  return {
    ...routing,
    commands: {
      ...(routing.commands || {}),
      [step.model]: {
        ...commandConfig,
        args: resolveSandboxArgs(routing, step.model, step.role),
      },
    },
  };
}

function findDoctorCommandConfig(routing, provider) {
  const commands = routing.commands || {};
  if (commands[provider]) return commands[provider];
  return Object.values(commands).find((config) => config?.defaultBin === provider) || null;
}

function buildDoctorReport({
  routing,
  env = process.env,
  providers = DOCTOR_PROVIDERS,
  commandExists: checkCommandExists = commandExists,
  cooldownStore = null,
}) {
  return providers.map((provider) => {
    const commandConfig = findDoctorCommandConfig(routing, provider);
    const envName = commandConfig?.binEnv;
    const envBin = envName ? env[envName] : null;
    const bin = envBin || commandConfig?.defaultBin || provider;
    const source = envBin ? `env:${envName}` : 'default';

    return {
      provider,
      bin,
      source,
      found: checkCommandExists(bin),
      window: cooldownStore?.window(provider) || '-',
    };
  });
}

function buildProviderAvailability({
  routing,
  env = process.env,
  commandExists: checkCommandExists = commandExists,
}) {
  return buildDoctorReport({
    routing: routing || {},
    env,
    providers: Object.keys(routing?.commands || {}),
    commandExists: checkCommandExists,
  }).map(({ provider, bin, found }) => ({ provider, bin, found }));
}

function scanStaleRunLedgers({
  root = ROOT,
  fs = { readdirSync, readFileSync },
  clock = () => new Date(),
  staleMs = STALE_RUN_MS,
} = {}) {
  const dir = getRunsDir(root);
  let names = [];
  try {
    names = fs.readdirSync(dir);
  } catch {
    return [];
  }

  const nowMs = clock().getTime();
  const stale = [];
  for (const name of names) {
    if (!String(name).endsWith('.json') || String(name).endsWith('.events.jsonl')) continue;
    const file = join(dir, name);
    try {
      const record = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (record.status !== 'in-progress') continue;
      const stamp = record.updatedAt || record.startedAt;
      const updatedMs = new Date(stamp).getTime();
      if (!Number.isFinite(updatedMs)) continue;
      const ageMs = nowMs - updatedMs;
      if (ageMs > staleMs) {
        stale.push({
          runId: record.runId || String(name).replace(/\.json$/, ''),
          ageMs,
          age: formatAgeMs(ageMs),
        });
      }
    } catch {
      // Doctor should keep reporting provider readiness even if one ledger is malformed.
    }
  }

  stale.sort((left, right) => right.ageMs - left.ageMs);
  return stale;
}

function formatDoctorReport(report) {
  const rows = [
    ['Provider', 'Binary', 'Source', 'Status', 'WINDOW'],
    ...report.map(({ provider, bin, source, found, window = '-' }) => [
      provider,
      bin,
      source,
      found ? 'found' : 'missing',
      window || '-',
    ]),
  ];
  const widths = rows[0].map((_, index) =>
    Math.max(...rows.map((row) => String(row[index]).length))
  );
  const formatRow = (row) =>
    row.map((value, index) => String(value).padEnd(widths[index])).join('  ');
  const divider = widths.map((width) => '-'.repeat(width)).join('  ');

  return [formatRow(rows[0]), divider, ...rows.slice(1).map(formatRow)].join('\n');
}

function printDoctorReport(
  report,
  stdout = process.stdout,
  { staleRuns = [], unclassifiedFailures = null } = {}
) {
  stdout.write('Hermes CLI doctor\n');
  stdout.write(`${formatDoctorReport(report)}\n`);
  if (Number.isInteger(unclassifiedFailures)) {
    stdout.write(`\nunclassified failures: ${unclassifiedFailures}\n`);
  }
  if (staleRuns.length > 0) {
    stdout.write('\nStale in-progress runs (>24h)\n');
    for (const run of staleRuns) {
      stdout.write(`- ${run.runId} age ${run.age}\n`);
    }
  }
}

function createChildEnv(env = process.env) {
  const childEnv = { ...env };
  for (const key of MODEL_ENV_BLOCKLIST) {
    delete childEnv[key];
  }
  return childEnv;
}

const realClock = {
  schedule(delayMs, callback) {
    const id = setTimeout(callback, delayMs);
    return {
      cancel() {
        clearTimeout(id);
      },
    };
  },
};

function killProcessTree(pid, child, stderr = process.stderr) {
  if (!pid) return;

  if (process.platform === 'win32') {
    const result = spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], {
      stdio: 'ignore',
    });
    if (result.error || result.status !== 0) {
      const detail = result.error?.message || `exit ${result.status}`;
      stderr.write(`[hermes] WARNING: failed to kill process tree for pid ${pid}: ${detail}\n`);
    }
    return;
  }

  if (child && typeof child.kill === 'function') {
    child.kill();
  }
}

function matchesRateLimitSignature(text, signatures = DEFAULT_RATE_LIMIT_SIGNATURES) {
  const haystack = String(text || '');
  if (!haystack || !Array.isArray(signatures)) return false;

  return signatures.some((signature) => {
    if (!signature) return false;
    if (signature instanceof RegExp) return signature.test(haystack);
    if (typeof signature === 'function') return Boolean(signature(haystack));
    return haystack.includes(String(signature));
  });
}

function classifyModelFailure({
  model,
  code,
  output,
  stderr,
  error,
  timedOut = false,
  captureOutput = false,
  rateLimitSignatures = DEFAULT_RATE_LIMIT_SIGNATURES,
}) {
  const diagnosticText = [output, stderr, error?.message, error?.code].filter(Boolean).join('\n');
  const providerSignatures = resolveRateLimitSignatures({
    configured: rateLimitSignatures,
    provider: model,
  });

  if (timedOut) return MODEL_FAILURE.TIMEOUT;
  if (matchesRateLimitSignature(diagnosticText, providerSignatures)) {
    return MODEL_FAILURE.RATE_LIMITED;
  }
  if (error) return MODEL_FAILURE.SPAWN_ERROR;
  if (code !== 0) return MODEL_FAILURE.NONZERO_EXIT;
  if (captureOutput && String(output || '').trim() === '') {
    return MODEL_FAILURE.EMPTY_OUTPUT;
  }
  return null;
}

function createModelResult({
  code,
  output,
  stderr,
  failure,
  stdinError = null,
  error = null,
}) {
  const result = {
    code,
    output,
    stderr,
    failure,
    stdinError,
  };
  if (error) {
    result.error = error;
  }
  return result;
}

function runModelAttempt({
  model,
  bin,
  args,
  prompt,
  env,
  timeoutMs,
  spawnImpl,
  killTree,
  clock,
  captureOutput,
  cwd = ROOT,
  rateLimitSignatures,
  emitEvent = null,
}) {
  const childEnv = createChildEnv(env);
  const stdio = captureOutput ? ['pipe', 'pipe', 'pipe'] : ['pipe', 'inherit', 'inherit'];
  let child = null;
  let timeoutHandle = null;
  let output = '';
  let stderr = '';
  let stdinError = null;
  let settled = false;

  return new Promise((resolvePromise) => {
    function settle(partial) {
      if (settled) return;
      settled = true;
      if (timeoutHandle) {
        timeoutHandle.cancel();
      }
      resolvePromise(
        createModelResult({
          output,
          stderr,
          stdinError,
          ...partial,
        })
      );
    }

    function handleStdinError(error) {
      stdinError = error;
      if (error?.code === 'EPIPE' || error?.code === 'ECONNRESET') {
        return;
      }
      settle({
        code: 1,
        failure: classifyModelFailure({
          model,
          code: 1,
          output,
          stderr,
          error,
          captureOutput,
          rateLimitSignatures,
        }),
        error,
      });
    }

    try {
      child = spawnImpl(bin, args, {
        stdio,
        shell: process.platform === 'win32',
        env: childEnv,
        // cwd selects the child edit target; governance paths stay ROOT-relative.
        cwd,
      });
    } catch (error) {
      settle({
        code: 1,
        failure: classifyModelFailure({
          model,
          code: 1,
          output,
          stderr,
          error,
          captureOutput,
          rateLimitSignatures,
        }),
        error,
      });
      return;
    }

    if (captureOutput) {
      child.stdout?.on('data', (chunk) => {
        output += chunk.toString();
      });
      child.stderr?.on('data', (chunk) => {
        stderr += chunk.toString();
      });
    }

    child.stdin?.on('error', handleStdinError);
    child.on('error', (error) => {
      settle({
        code: 1,
        failure: classifyModelFailure({
          model,
          code: 1,
          output,
          stderr,
          error,
          captureOutput,
          rateLimitSignatures,
        }),
        error,
      });
    });
    child.on('close', (code) => {
      const exitCode = code || 0;
      settle({
        code: exitCode,
        failure: classifyModelFailure({
          model,
          code: exitCode,
          output,
          stderr,
          captureOutput,
          rateLimitSignatures,
        }),
      });
    });

    timeoutHandle = clock.schedule(timeoutMs, () => {
      try {
        if (killTree) {
          killTree(child.pid);
        } else {
          killProcessTree(child.pid, child);
        }
      } catch (error) {
        process.stderr.write(
          `[hermes] WARNING: failed to kill process tree for pid ${child.pid}: ${error.message}\n`
        );
      }
      if (emitEvent) {
        emitEvent({
          type: RUN_EVENT_TYPES.TIMEOUT_KILL,
          model,
          pid: child.pid,
          timeoutMs,
        });
      }
      settle({
        code: 124,
        failure: MODEL_FAILURE.TIMEOUT,
      });
    });

    try {
      child.stdin?.write(prompt);
      child.stdin?.end();
    } catch (error) {
      handleStdinError(error);
    }
  });
}

async function executeModelCommand(
  model,
  prompt,
  routing,
  env = process.env,
  {
    spawn: spawnImpl = spawn,
    killTree = null,
    clock = realClock,
    rateLimitSignatures,
    providerCooldownStore = null,
    providerStateFs = { existsSync, readFileSync, writeFileSync, renameSync, mkdirSync },
    captureOutput = false,
    cwd = ROOT,
    appendRunEvent: appendEvent = null,
    runId = null,
    role = null,
    root = ROOT,
    eventFs,
    eventClock = typeof clock.now === 'function' ? () => clock.now() : () => new Date(),
    envSecrets,
    envPath = join(root, '.env'),
    envLoader,
    scrubber = scrubSecrets,
    io = process,
    commandExists: checkCommandExists = commandExists,
  } = {}
) {
  const commandConfig = routing.commands?.[model];
  if (!commandConfig) {
    throw new Error(`No command config for model: ${model}`);
  }

  const bin = env[commandConfig.binEnv] || commandConfig.defaultBin;
  if (!checkCommandExists(bin)) {
    throw new Error(
      `Command not found for model "${model}": ${bin}. Set ${commandConfig.binEnv} or install the CLI.`
    );
  }

  const timeoutMs = Number(commandConfig.timeoutMs ?? DEFAULT_MODEL_TIMEOUT_MS);
  const configuredRateLimitSignatures = compileRateLimitSignatures(routing.rateLimits || {});
  const effectiveRateLimitSignatures =
    rateLimitSignatures === undefined ? configuredRateLimitSignatures : rateLimitSignatures;
  const cooldownStore =
    providerCooldownStore ||
    (routing.rateLimits?.enabled === true
      ? createProviderCooldownStore({ root, fs: providerStateFs, clock, io })
      : null);
  let result = null;
  const emitEvent = createRunEventEmitter({
    runId,
    appendEvent,
    root,
    fs: eventFs,
    clock: eventClock,
    envSecrets,
    envPath,
    envLoader,
    scrubber,
    io,
  });

  // Pipeline: sanitize env -> spawn -> write prompt -> classify -> retry once.
  for (let attempt = 0; attempt < 2; attempt += 1) {
    result = await runModelAttempt({
      model,
      bin,
      args: commandConfig.args || [],
      prompt,
      env,
      timeoutMs,
      spawnImpl,
      killTree,
      clock,
      captureOutput,
      cwd,
      rateLimitSignatures: effectiveRateLimitSignatures,
      emitEvent,
    });
    if (result.failure) {
      const excerpt = String(result.stderr || result.error?.message || result.output || '').slice(0, 500);
      emitEvent({
        type: RUN_EVENT_TYPES.DISPATCH_ERROR,
        role,
        model,
        failure: result.failure,
        code: result.code,
        excerpt,
      });
      if (result.failure === MODEL_FAILURE.RATE_LIMITED && cooldownStore) {
        const cooldown = cooldownStore.setCooldown(model, {
          routing,
          message: diagnosticTextForResult(result),
        });
        emitEvent({
          type: RUN_EVENT_TYPES.COOLDOWN_SET,
          provider: model,
          coolingUntil: cooldown.coolingUntil,
        });
      }
    }
    if (result.failure !== MODEL_FAILURE.SPAWN_ERROR) {
      return result;
    }
    if (attempt === 0) {
      emitEvent({
        type: RUN_EVENT_TYPES.RETRY_ATTEMPT,
        role,
        model,
        failure: result.failure,
        attempt: attempt + 1,
      });
    }
  }

  return result;
}

function executeModel(model, prompt, routing, env = process.env, { cwd = ROOT, ...seams } = {}) {
  return executeModelCommand(model, prompt, routing, env, {
    ...seams,
    cwd,
    captureOutput: false,
  }).then((result) => result.code);
}

async function executeSoloModelWithFallback({
  plan,
  prompt,
  routing,
  env = process.env,
  seams = {},
  commandExists: checkCommandExists = commandExists,
  isCooling = () => false,
  io = process,
}) {
  const providerDiagnostics = [...(plan.providerDiagnostics || [])];
  const attemptedProviders = new Set();
  const emitEvent = createRunEventEmitter({
    runId: seams.runId,
    appendEvent: seams.appendRunEvent,
    root: seams.root || ROOT,
    fs: seams.eventFs,
    clock: seams.eventClock || (() => new Date()),
    envSecrets: seams.envSecrets,
    envPath: seams.envPath,
    envLoader: seams.envLoader,
    scrubber: seams.scrubber,
    io,
  });
  let currentModel = plan.model;

  while (true) {
    attemptedProviders.add(currentModel);
    const result = await executeModelCommand(currentModel, prompt, routing, env, {
      ...seams,
      commandExists: checkCommandExists,
      captureOutput: false,
    });
    if (!result.failure) {
      return { code: result.code, model: currentModel, providerDiagnostics };
    }

    providerDiagnostics.push({
      provider: currentModel,
      status: 'failed',
      detail: `${currentModel} failed (${result.failure})`,
    });

    if (plan.risk === 'financial') {
      const detail = financialNoDegradeMessage('solo', currentModel, `failed (${result.failure})`);
      providerDiagnostics.push({ provider: currentModel, status: 'blocked', detail });
      return {
        code: result.code || 1,
        model: currentModel,
        providerDiagnostics,
        failureLedger: createProviderFailureLedger({
          role: 'solo',
          message: detail,
          tried: [...attemptedProviders],
          providerDiagnostics,
        }),
      };
    }

    if (plan.manualPinned && !plan.allowFallback) {
      const detail = `manual provider ${currentModel} failed (${result.failure}); fallback disabled`;
      providerDiagnostics.push({ provider: currentModel, status: 'blocked', detail });
      return {
        code: result.code || 1,
        model: currentModel,
        providerDiagnostics,
        failureLedger: createProviderFailureLedger({
          role: 'solo',
          message: detail,
          tried: [...attemptedProviders],
          providerDiagnostics,
        }),
      };
    }

    if (isWriteCapableCommand(routing.commands?.[currentModel])) {
      const detail = `write-capable provider ${currentModel} failed (${result.failure}); fallback disabled`;
      providerDiagnostics.push({ provider: currentModel, status: 'blocked', detail });
      return {
        code: result.code || 1,
        model: currentModel,
        providerDiagnostics,
        failureLedger: createProviderFailureLedger({
          role: 'solo',
          message: detail,
          tried: [...attemptedProviders],
          providerDiagnostics,
        }),
      };
    }

    const next = findNextDispatchProvider({
      currentModel,
      attemptedProviders,
      routing,
      env,
      commandExists: checkCommandExists,
      isCooling,
    });
    providerDiagnostics.push(...next.skippedDiagnostics);
    if (!next.provider) {
      const message = noAvailableProviderMessage('solo', next.tried);
      return {
        code: result.code || 1,
        model: currentModel,
        providerDiagnostics,
        failureLedger: createProviderFailureLedger({
          role: 'solo',
          message,
          tried: next.tried,
          providerDiagnostics,
        }),
      };
    }

    emitEvent({
      type: RUN_EVENT_TYPES.FALLBACK,
      role: 'solo',
      from: currentModel,
      to: next.provider,
      failure: result.failure,
    });
    providerDiagnostics.push({
      provider: next.provider,
      status: 'fallback',
      detail: `advanced from ${currentModel} after ${result.failure}`,
    });
    currentModel = next.provider;
  }
}

// Sibling of executeModel that PIPES stdout so a step's output can be captured
// and fed back into executeWorkflow. executeModel keeps inheriting stdout so the
// non-workflow path keeps inheriting stdout.
async function executeModelCapture(
  model,
  prompt,
  routing,
  env = process.env,
  { spawn: spawnImpl = spawn, cwd = ROOT, ...seams } = {}
) {
  const result = await executeModelCommand(model, prompt, routing, env, {
    ...seams,
    spawn: spawnImpl,
    cwd,
    captureOutput: true,
  });
  if (result.failure === MODEL_FAILURE.SPAWN_ERROR && result.error) {
    throw result.error;
  }
  const captured = { code: result.code, output: result.output };
  if (result.failure) {
    Object.defineProperty(captured, 'failure', {
      value: result.failure,
      enumerable: false,
    });
  }
  return captured;
}

const APPROVAL_SENTINEL = 'APPROVED';
const REJECTION_SENTINEL = 'CHANGES REQUESTED';

// Fail-closed but format-tolerant. Each line is normalized to letters-only
// uppercase so markdown/punctuation around the verdict still parses
// (**APPROVED**, "APPROVED.", "## APPROVED"). Rejection wins; an absent or
// ambiguous response is treated as not approved.
function parseApprovalSignal(output) {
  const rejectToken = REJECTION_SENTINEL.replace(/[^a-zA-Z]/g, '').toUpperCase();
  const approveToken = APPROVAL_SENTINEL.replace(/[^a-zA-Z]/g, '').toUpperCase();
  const norm = String(output || '')
    .split('\n')
    .map((line) => line.replace(/[^a-zA-Z]/g, '').toUpperCase());
  if (norm.some((line) => line.startsWith(rejectToken))) {
    return false;
  }
  return norm.some((line) => line === approveToken);
}

function formatStepInput(input) {
  if (Array.isArray(input)) {
    return input.map((entry, index) => `--- INPUT ${index + 1} ---\n${entry ?? ''}`).join('\n\n');
  }
  return input == null ? '' : String(input);
}

// Live step runner injected into executeWorkflow. Composes a per-step prompt from
// the base task prompt plus role/action context, prior output, and specialist
// notes, runs the step's model with stdout captured, and (for reviewer steps)
// derives a fail-closed approval verdict. Prompt composition is intentionally a
// minimal v1; refine as real-model output patterns are observed.
function createLiveRunStep({
  routing,
  basePrompt = '',
  env = process.env,
  executor = executeModelCapture,
  cwd = ROOT,
  appendRunEvent: appendEvent = null,
  runId = null,
  root = ROOT,
  eventFs,
  modelClock = realClock,
  eventClock = () => new Date(),
  io = process,
  envSecrets,
  envPath = join(root, '.env'),
  envLoader,
  scrubber = scrubSecrets,
  commandExists: checkCommandExists = commandExists,
  providerCooldownStore = null,
  providerStateFs = { existsSync, readFileSync, writeFileSync, renameSync, mkdirSync },
} = {}) {
  const promptEnvSecrets = resolveEnvSecrets({ envSecrets, envLoader, envPath });

  return async function liveRunStep({ step, input, notes }) {
    const sections = [
      basePrompt,
      '',
      '--- WORKFLOW STEP ---',
      `ROLE: ${step.role}`,
      `ACTION: ${step.action}`,
    ];
    if (notes) {
      sections.push('', 'SPECIALIST NOTES:', String(notes));
    }
    const formattedInput = formatStepInput(input);
    if (formattedInput) {
      const scrubbedInput = scrubOutputBody(formattedInput, {
        scrubber,
        envSecrets: promptEnvSecrets,
      }).text;
      sections.push('', 'PRIOR OUTPUT:', scrubbedInput);
    }
    if (step.role === 'reviewer') {
      sections.push(
        '',
        'VERDICT REQUIRED. The LAST line of your reply must be exactly one of these two',
        'tokens, alone on its own line, with no markdown, quotes, or other characters:',
        `  ${APPROVAL_SENTINEL}`,
        `  ${REJECTION_SENTINEL}`,
        `Use ${APPROVAL_SENTINEL} only if the diff is correct and ready to ship. Otherwise use`,
        `${REJECTION_SENTINEL} and list the required changes on the lines above the verdict.`,
        'An absent or ambiguous verdict is treated as changes requested.'
      );
    }

    const stepRouting = routingWithSandboxArgsForStep(routing, step);
    const { code, output } = await executor(step.model, sections.join('\n'), stepRouting, env, {
      appendRunEvent: appendEvent,
      runId,
      role: step.role,
      root,
      eventFs,
      clock: modelClock,
      eventClock,
      io,
      cwd,
      envSecrets,
      envPath,
      envLoader,
      scrubber,
      commandExists: checkCommandExists,
      providerCooldownStore,
      providerStateFs,
    });
    const result = { code, output };
    if (step.role === 'reviewer') {
      result.approved = parseApprovalSignal(output);
    }
    return result;
  };
}

function runGate(
  gate,
  { runner = spawnSync, env = process.env, stdio = 'inherit', throwOnFailure = true } = {}
) {
  const command = String(gate || '').trim();
  if (!command) {
    return { command: null, skipped: true, status: 0 };
  }

  const [bin, ...args] = command.split(/\s+/);
  const result = runner(bin, args, {
    env,
    shell: process.platform === 'win32',
    stdio,
  });

  if (result.error) {
    throw result.error;
  }

  const status = result.status ?? 0;
  if (status !== 0 && throwOnFailure) {
    throw new Error(`Gate failed (${command}) with exit code ${status}`);
  }

  return { command, skipped: false, status };
}

function generateRunId(now = new Date()) {
  const iso = now.toISOString().replace(/[:.]/g, '-');
  return `hermes-${iso}`;
}

function writeRunLedger(
  record,
  {
    root = ROOT,
    fs = { mkdirSync, writeFileSync },
    envSecrets,
    envPath = join(root, '.env'),
    envLoader,
    scrubber = scrubSecrets,
  } = {}
) {
  const dir = getRunsDir(root);
  fs.mkdirSync(dir, { recursive: true });
  const file = getRunLedgerPath(record.runId, { root });
  const prepared = prepareRunLedgerRecord(record, { envSecrets, envPath, envLoader, scrubber });
  fs.writeFileSync(file, `${JSON.stringify(prepared, null, 2)}\n`);
  return file;
}

function writeDebateRunRecord(
  record,
  { root = ROOT, fs = { mkdirSync, writeFileSync }, clock, io = process } = {}
) {
  const completedDate = debateTimestamp({ record, clock });
  const debateRecord = buildDebateRunRecord(record, { root, clock: () => completedDate });
  const validation = validateDebateRunRecordShape(debateRecord);
  const dateStamp = safeDateStamp(completedDate);
  const dir = getRunsDir(root);
  fs.mkdirSync(dir, { recursive: true });
  const file = getDebateRunRecordPath(dateStamp, { root });
  fs.writeFileSync(file, `${JSON.stringify(debateRecord, null, 2)}\n`);
  if (validation.errors.length > 0 || validation.warnings.length > 0) {
    warnDebateRunRecordValidation(io, validation);
  }
  return {
    file,
    validation,
  };
}

function appendRunEvent(
  runId,
  event,
  {
    root = ROOT,
    fs = { mkdirSync, appendFileSync },
    clock = () => new Date(),
    envSecrets,
    envPath = join(root, '.env'),
    envLoader,
    scrubber = scrubSecrets,
  } = {}
) {
  const dir = getRunsDir(root);
  fs.mkdirSync(dir, { recursive: true });
  const file = getRunEventsPath(runId, { root });
  const payload = {
    ts: clock().toISOString(),
    type: event?.type,
    ...(event || {}),
  };
  const prepared = prepareMetadataRecord(payload, { envSecrets, envPath, envLoader, scrubber });
  if (typeof prepared.record.excerpt === 'string' && prepared.record.excerpt.length > 500) {
    prepared.record.excerpt = prepared.record.excerpt.slice(0, 500);
  }
  fs.appendFileSync(file, `${JSON.stringify(prepared.record)}\n`);
  return file;
}

function writeRunCheckpoint(
  record,
  {
    root = ROOT,
    fs = { mkdirSync, writeFileSync, renameSync },
    envSecrets,
    envPath = join(root, '.env'),
    envLoader,
    scrubber = scrubSecrets,
  } = {}
) {
  const dir = getRunsDir(root);
  fs.mkdirSync(dir, { recursive: true });
  const file = getRunLedgerPath(record.runId, { root });
  const tempFile = `${file}.tmp`;
  const prepared = prepareMetadataRecord(record, { envSecrets, envPath, envLoader, scrubber });
  fs.writeFileSync(tempFile, `${JSON.stringify(prepared.record, null, 2)}\n`);
  try {
    fs.renameSync(tempFile, file);
  } catch (error) {
    if (error?.code === 'EPERM' || error?.code === 'EBUSY') {
      fs.renameSync(tempFile, file);
      return file;
    }
    throw error;
  }
  return file;
}

function createRunEventEmitter({
  runId,
  appendEvent,
  root = ROOT,
  fs,
  clock = () => new Date(),
  envSecrets,
  envPath = join(root, '.env'),
  envLoader,
  scrubber = scrubSecrets,
  io = process,
}) {
  if (!appendEvent || !runId) {
    return () => {};
  }

  let warned = false;
  return (event) => {
    try {
      appendEvent(runId, event, { root, fs, clock, envSecrets, envPath, envLoader, scrubber });
    } catch (error) {
      if (!warned) {
        warned = true;
        warnEventFailure(io, error.message);
      }
    }
  };
}

function getGateRunPlan(plan, { skipPreflightGate = false } = {}) {
  const hasGate = Boolean(plan.gate);
  return {
    preflight: hasGate && !skipPreflightGate,
    postflight: hasGate,
  };
}

function isProductionFinancial(plan) {
  return plan.phase === 'production' && plan.risk === 'financial';
}

function assertFinancialGate(plan) {
  if (!isProductionFinancial(plan)) {
    return;
  }
  if (plan.gate !== 'npm run advisor-gate') {
    throw new Error(
      `Financial gate proof failed: production-financial plan must resolve gate to "npm run advisor-gate", got "${plan.gate}".`
    );
  }
}

// Structured readiness boundary. assertFinancialGate stays the throwing core; this
// wraps it so a CLI or report surface can present a not-ready outcome without
// crashing. When an execution result is supplied, a nonzero exit or an unapproved
// artifact also blocks readiness.
function evaluateReadiness(plan, result = null) {
  if (!plan || typeof plan !== 'object') {
    return { ready: false, reason: 'evaluateReadiness requires a plan object.' };
  }

  try {
    assertFinancialGate(plan);
  } catch (error) {
    return { ready: false, reason: error.message };
  }

  if (result) {
    if (Number.isInteger(result.exitCode) && result.exitCode !== 0) {
      return { ready: false, reason: `workflow exited with code ${result.exitCode}` };
    }
    if (result.approved === false) {
      return { ready: false, reason: 'reviewer did not approve the artifact' };
    }
  }

  return { ready: true, reason: null };
}

function shouldRunPostflightGate(plan, code, gates) {
  return gates.postflight && (code === 0 || isProductionFinancial(plan));
}

function defaultRunStep() {
  throw new Error(
    'executeWorkflow requires deps.runStep until live model wiring lands; pass an injected step runner.'
  );
}

function classifyStepResult(result) {
  if (result?.failure) return result.failure;
  const code = result?.code ?? 0;
  return code === 0 ? null : MODEL_FAILURE.NONZERO_EXIT;
}

function formatProviderSkipDetail(provider, reason) {
  return reason === 'cooling' ? `${provider} cooling` : `${provider} unavailable (${reason})`;
}

function findNextDispatchProvider({
  currentModel,
  attemptedProviders,
  routing,
  env = process.env,
  commandExists: checkCommandExists = commandExists,
  isCooling = () => false,
}) {
  const rungs = configuredLadderRungs(routing);
  const start = rungs.indexOf(currentModel);
  const candidates = start >= 0 ? rungs.slice(start + 1) : rungs.filter((rung) => rung !== currentModel);
  const skippedDiagnostics = [];

  for (const provider of candidates) {
    if (attemptedProviders.has(provider)) continue;
    const status = providerStatus({
      provider,
      routing,
      env,
      commandExists: checkCommandExists,
      isCooling,
    });
    if (status.available) {
      return {
        provider,
        skippedDiagnostics,
        tried: [currentModel, ...candidates],
      };
    }
    skippedDiagnostics.push({
      provider,
      status: 'skipped',
      detail: formatProviderSkipDetail(provider, status.reason),
    });
  }

  return {
    provider: null,
    skippedDiagnostics,
    tried: [currentModel, ...candidates],
  };
}

function failedStepResult(result, message, failure) {
  return {
    ...result,
    code: result?.code && result.code !== 0 ? result.code : 1,
    output: result?.output || message,
    failure,
  };
}

async function runStepWithProviderFallback({
  step,
  input,
  notes,
  plan,
  attempt,
  runId,
  cwd = ROOT,
  runStep,
  routing,
  env = process.env,
  commandExists: checkCommandExists = commandExists,
  isCooling = () => false,
  emitEvent = () => {},
}) {
  if (!ladderEnabled(routing) || !step.model) {
    const result = await runStep({ step, input, notes, plan, attempt, runId, cwd });
    return {
      result,
      step,
      providerDiagnostics: step.providerDiagnostics || [],
      weakenedControl: false,
      failureLedger: [],
    };
  }

  const providerDiagnostics = [...(step.providerDiagnostics || [])];
  const attemptedProviders = new Set();
  let currentStep = { ...step };
  let weakenedControl = false;

  while (true) {
    attemptedProviders.add(currentStep.model);
    const result = await runStep({ step: currentStep, input, notes, plan, attempt, runId, cwd });
    const failure = classifyStepResult(result);
    if (!failure) {
      return {
        result,
        step: currentStep,
        providerDiagnostics,
        weakenedControl,
        failureLedger: [],
      };
    }

    const baseDetail = `${currentStep.model} failed (${failure})`;
    providerDiagnostics.push({
      provider: currentStep.model,
      status: 'failed',
      detail: baseDetail,
    });

    if (plan.risk === 'financial') {
      const detail = financialNoDegradeMessage(step.role, currentStep.model, `failed (${failure})`);
      providerDiagnostics.push({ provider: currentStep.model, status: 'blocked', detail });
      return {
        result: failedStepResult(result, detail, failure),
        step: currentStep,
        providerDiagnostics,
        weakenedControl,
        failureLedger: createProviderFailureLedger({
          role: step.role,
          message: detail,
          tried: [...attemptedProviders],
          providerDiagnostics,
        }),
      };
    }

    if (currentStep.manualPinned && !currentStep.allowFallback) {
      const detail = `manual provider ${currentStep.model} failed (${failure}); fallback disabled`;
      providerDiagnostics.push({ provider: currentStep.model, status: 'blocked', detail });
      return {
        result: failedStepResult(result, detail, failure),
        step: currentStep,
        providerDiagnostics,
        weakenedControl,
        failureLedger: createProviderFailureLedger({
          role: step.role,
          message: detail,
          tried: [...attemptedProviders],
          providerDiagnostics,
        }),
      };
    }

    const commandConfig = routing.commands?.[currentStep.model] || null;
    if (isWriteCapableCommand(commandConfig)) {
      const detail = `write-capable provider ${currentStep.model} failed (${failure}); fallback disabled`;
      providerDiagnostics.push({ provider: currentStep.model, status: 'blocked', detail });
      return {
        result: failedStepResult(result, detail, failure),
        step: currentStep,
        providerDiagnostics,
        weakenedControl,
        failureLedger: createProviderFailureLedger({
          role: step.role,
          message: detail,
          tried: [...attemptedProviders],
          providerDiagnostics,
        }),
      };
    }

    const next = findNextDispatchProvider({
      currentModel: currentStep.model,
      attemptedProviders,
      routing,
      env,
      commandExists: checkCommandExists,
      isCooling,
    });
    providerDiagnostics.push(...next.skippedDiagnostics);
    if (!next.provider) {
      const message = noAvailableProviderMessage(step.role, next.tried);
      return {
        result: failedStepResult(result, message, failure),
        step: currentStep,
        providerDiagnostics,
        weakenedControl,
        failureLedger: createProviderFailureLedger({
          role: step.role,
          message,
          tried: next.tried,
          providerDiagnostics,
        }),
      };
    }

    emitEvent({
      type: RUN_EVENT_TYPES.FALLBACK,
      role: step.role,
      from: currentStep.model,
      to: next.provider,
      failure,
    });
    providerDiagnostics.push({
      provider: next.provider,
      status: 'fallback',
      detail: `advanced from ${currentStep.model} after ${failure}`,
    });
    weakenedControl = weakenedControl || REVIEW_FALLBACK_ROLES.has(step.role);
    currentStep = {
      ...currentStep,
      model: next.provider,
      selectedModel: next.provider,
      degradeReason: `${currentStep.model} failed (${failure})`,
    };
  }
}

async function executeWorkflow(plan, deps = {}) {
  const workflow = plan.workflow;
  if (!workflow || !Array.isArray(workflow.steps)) {
    throw new Error('executeWorkflow requires a plan with a workflow.steps array.');
  }

  const maxRepairs = Number.isInteger(deps.maxRepairs) ? deps.maxRepairs : 2;
  const runStep = deps.runStep || defaultRunStep;
  const gateRunner = deps.gateRunner || spawnSync;
  const assertGate = deps.assertFinancialGate || assertFinancialGate;
  const ledgerWriter = deps.writeRunLedger === undefined ? null : deps.writeRunLedger;
  const debateRunWriter =
    deps.writeDebateRunRecord === undefined ? null : deps.writeDebateRunRecord;
  const clock = deps.clock || (() => new Date());
  const runId = deps.runId || generateRunId(clock());
  const root = deps.root || ROOT;
  const cwd = deps.cwd || ROOT;
  const routing = deps.routing || { ladder: plan.ladder || { enabled: false }, commands: {} };
  const env = deps.env || process.env;
  const checkCommandExists = deps.commandExists || commandExists;
  const isCooling = deps.isCooling || (() => false);
  const io = deps.io || process;
  const ledgerScrubOptions = {
    envSecrets: deps.envSecrets,
    envPath: deps.envPath || join(root, '.env'),
    envLoader: deps.envLoader,
    scrubber: deps.scrubber,
  };
  const emitEvent = createRunEventEmitter({
    runId,
    appendEvent: deps.appendRunEvent === undefined ? null : deps.appendRunEvent,
    root,
    fs: deps.eventFs,
    clock,
    envSecrets: deps.envSecrets,
    envPath: deps.envPath || join(root, '.env'),
    envLoader: deps.envLoader,
    scrubber: deps.scrubber,
    io,
  });
  const checkpointWriter =
    deps.writeRunCheckpoint === undefined
      ? deps.checkpointFs
        ? writeRunCheckpoint
        : null
      : deps.writeRunCheckpoint;
  const checkpointOptions = {
    root,
    fs: deps.checkpointFs || { mkdirSync, writeFileSync, renameSync },
    ...ledgerScrubOptions,
  };
  const availability =
    deps.availability ||
    buildProviderAvailability({
      routing,
      env,
      commandExists: checkCommandExists,
    });

  const stepByRole = (role) => workflow.steps.find((step) => step.role === role) || null;
  const ownerStep = stepByRole('owner');
  const specialistStep = stepByRole('specialist');
  const reviewerStep = stepByRole('reviewer');
  const auditStep = stepByRole('audit');
  const comparatorSteps = workflow.steps.filter((step) => step.role === 'comparator');
  const rebuttalStep = stepByRole('rebuttal');
  const synthesisStep = stepByRole('synthesis');

  let specialistNotes = null;
  const records = [];
  const failureLedger = [];
  // First nonzero exit from ANY model step (owner, specialist, reviewer, audit,
  // comparator, synthesis). A crashed CLI must not be reported as success just
  // because the postflight gate passes.
  let stepFailureCode = 0;
  const modelStepTotal = workflow.steps.filter((step) => step.model).length;
  let cursor = { index: 0, total: modelStepTotal, role: null };
  let finalized = false;
  let gate = { command: plan.gate || null, skipped: !plan.gate, status: 0 };
  const buildCheckpointRecord = (status, checkpointCursor = cursor) => ({
    runId,
    status,
    updatedAt: clock().toISOString(),
    workflow: workflow.selected,
    phase: plan.phase,
    risk: plan.risk,
    repairs,
    steps: records.map(compactStepRecord),
    availability,
    gate: {
      command: gate.command ?? null,
      status: gate.status ?? null,
      skipped: gate.skipped ?? false,
    },
    cursor: checkpointCursor,
  });
  const flushCheckpoint = (status, checkpointCursor = cursor) => {
    if (!checkpointWriter) return;
    try {
      checkpointWriter(buildCheckpointRecord(status, checkpointCursor), checkpointOptions);
    } catch (error) {
      warnLedgerFailure(io, error.message);
      emitEvent({ type: RUN_EVENT_TYPES.LEDGER_FLUSH_FAILURE, message: error.message });
    }
  };
  const runRecorded = async (step, input, attempt) => {
    const startedAt = clock();
    emitEvent({ type: RUN_EVENT_TYPES.STEP_START, role: step.role, model: step.model });
    const executed = await runStepWithProviderFallback({
      step,
      input,
      notes: specialistNotes,
      plan,
      attempt,
      runId,
      cwd,
      runStep,
      routing,
      env,
      commandExists: checkCommandExists,
      isCooling,
      emitEvent,
    });
    const result = executed.result;
    const finalStep = executed.step;
    if (executed.failureLedger.length > 0) {
      failureLedger.push(...executed.failureLedger);
    }
    const endedAt = clock();
    const durationMs = Math.max(0, endedAt.getTime() - startedAt.getTime());
    const code = result.code ?? 0;
    if (stepFailureCode === 0 && code !== 0) {
      stepFailureCode = code;
    }
    const record = {
      role: step.role,
      model: finalStep.model,
      attempt,
      code,
      approved: result.approved ?? null,
      durationMs,
      output: result.output ?? '',
    };
    if (step.requestedModel || executed.providerDiagnostics.length > 0) {
      record.requestedModel = step.requestedModel || step.model;
      record.selectedModel = finalStep.selectedModel || finalStep.model;
      record.degradeReason = finalStep.degradeReason ?? step.degradeReason ?? null;
      record.providerDiagnostics = executed.providerDiagnostics;
    }
    if (executed.weakenedControl || finalStep.weakenedControl || step.weakenedControl) {
      record.weakenedControl = true;
      record.deviation =
        finalStep.deviation ||
        step.deviation ||
        'review-lane provider fallback weakened independence control';
    }
    records.push(record);
    emitEvent({
      type: RUN_EVENT_TYPES.STEP_END,
      role: step.role,
      model: finalStep.model,
      code,
      durationMs,
    });
    // Checkpoints intentionally carry only compact step metadata, never bodies.
    cursor = { index: records.length, total: modelStepTotal, role: step.role };
    flushCheckpoint('in-progress', cursor);
    return result;
  };

  let artifact = null;
  let approved = true;
  let repairs = 0;

  try {
    if (comparatorSteps.length > 0) {
      const comparatorOutputs = [];
      for (const comparatorStep of comparatorSteps) {
        const comparator = await runRecorded(comparatorStep, null, 0);
        comparatorOutputs.push(comparator.output ?? '');
      }
      let synthesisInput = comparatorOutputs;
      if (rebuttalStep) {
        const rebuttal = await runRecorded(rebuttalStep, comparatorOutputs, 0);
        synthesisInput = [...comparatorOutputs, `REBUTTAL:\n${rebuttal.output ?? ''}`];
      }
      if (synthesisStep) {
        const synthesis = await runRecorded(synthesisStep, synthesisInput, 0);
        artifact = synthesis.output ?? '';
      }
    } else {
      if (ownerStep) {
        const owner = await runRecorded(ownerStep, null, 0);
        artifact = owner.output ?? '';
      }

      if (specialistStep) {
        const specialist = await runRecorded(specialistStep, artifact, 0);
        specialistNotes = specialist.output ?? null;
      }

      if (reviewerStep) {
        let review = await runRecorded(reviewerStep, artifact, 0);
        approved = Boolean(review.approved);
        while (!approved && repairs < maxRepairs && ownerStep) {
          repairs += 1;
          const repair = await runRecorded(ownerStep, review.output ?? '', repairs);
          artifact = repair.output ?? artifact;
          review = await runRecorded(reviewerStep, artifact, repairs);
          approved = Boolean(review.approved);
        }
      }

      if (auditStep) {
        await runRecorded(auditStep, artifact, repairs);
      }
    }

    if (plan.gate) {
      if (isProductionFinancial(plan)) {
        assertGate(plan);
      }
      gate = runGate(plan.gate, { runner: gateRunner, throwOnFailure: false });
      emitEvent({
        type: RUN_EVENT_TYPES.GATE_RESULT,
        command: gate.command,
        status: gate.status ?? 0,
      });
    }

    let exitCode = 0;
    if (gate.status && gate.status !== 0) {
      exitCode = gate.status;
    } else if (stepFailureCode !== 0) {
      exitCode = stepFailureCode;
    } else if (reviewerStep && !approved) {
      exitCode = 1;
    }

    const record = {
      runId,
      workflow: workflow.selected,
      phase: plan.phase,
      risk: plan.risk,
      approved,
      repairs,
      steps: records,
      availability,
      gate: {
        command: gate.command ?? null,
        status: gate.status ?? 0,
        skipped: gate.skipped ?? false,
      },
      exitCode,
    };
    if (failureLedger.length > 0) {
      record.failureLedger = failureLedger;
    }

    writePreparedLedgerBestEffort({
      ledgerWriter,
      record,
      scrubOptions: ledgerScrubOptions,
      io,
      emitEvent,
    });
    writeDebateRunRecordBestEffort({
      debateRunWriter,
      record,
      options: {
        root,
        fs: deps.debateRunFs || { mkdirSync, writeFileSync },
        clock,
        io,
      },
      io,
      emitEvent,
    });
    finalized = true;
    return record;
  } finally {
    if (!finalized) {
      // This catches ordinary router interruptions; a hard process kill is reconciled by doctor.
      flushCheckpoint('interrupted', cursor);
    }
  }
}

function printHelp(stdout = process.stdout) {
  stdout.write(`Usage:
  node orchestrate.js --phase <research|production|distribution> --task "<description>"
  node orchestrate.js --json --phase production --task "fix xirr calculation"
  node orchestrate.js --dry-run --phase research --task "trace reserve engine flow"
  node orchestrate.js --dry-run --workflow pair --model codex --phase production --task "implement feature"
  node orchestrate.js --project <path> --phase production --task "edit another repo"
  node orchestrate.js --phase production --task "repair calc gate" --skip-preflight-gate --skip-reason "<reason>"

Phases:
  research      Default Claude planning lane; gate: npm run doctor:quick.
  production    Default Codex implementation lane; gate: npm run check.
  distribution  Default Claude handoff lane; gate: npm run lint.
  Financial production tasks are promoted internally to production-financial; gate: npm run calc-gate.

Model overrides:
  --claude | --codex | --kimi
  --model <claude|codex|kimi>
  --allow-fallback  Permit a manually pinned provider to use the configured fallback ladder.
  --project, --cwd <path>
                  Set the child CLI working directory. Governance remains router-local.

Output:
  --dry-run       Print the routing plan and prompt without model execution.
  --json          Print routing plan JSON only.
  --help, -h      Show this help.

Workflow planning:
  --workflow <auto|solo|pair|chain|debate|review>
                 Add a planning-only workflow recommendation to dry-run output.
  --live         Execute the planned workflow live (spawns real model CLIs).
                 Without --live, --workflow stays planning-only.

Gate controls:
  --skip-preflight-gate --skip-reason "<reason>"
                 Skip only the preflight gate; postflight gates still run.
                 Legacy --skip-gates is rejected.

Legacy commands:
  bootstrap | smoke | enable-algorithms | doctor
`);
}

class Orchestrator {
  constructor({ root = ROOT, routing = null, brain = '', soul = '' } = {}) {
    this.root = root;
    this.routing = routing;
    this.brain = brain;
    this.soul = soul;
  }

  plan({
    phase = 'research',
    task,
    manualModel = null,
    requestedWorkflow = 'auto',
    routing = this.routing,
  }) {
    if (!routing) throw new Error('Routing config is required to build a Hermes plan');
    return createRoutingPlan({ phase, task, routing, manualModel, requestedWorkflow });
  }

  execute({
    phase = 'research',
    task,
    manualModel = null,
    routing = this.routing,
    env = process.env,
  }) {
    const plan = this.plan({ phase, task, manualModel, routing });
    const prompt = buildPrompt({ plan, brain: this.brain, soul: this.soul });
    return executeModel(plan.model, prompt, routing, env);
  }

  async bootstrap() {
    const dirs = ['client/src/core/reserves', 'client/src/core/pacing', 'tests/fixtures'];

    for (const dir of dirs) {
      if (!existsSync(dir)) {
        mkdirSync(dir, { recursive: true });
        console.log(`[legacy] Created ${dir}`);
      }
    }

    console.log('[legacy] Bootstrap sequence completed');
  }

  async runSmokeTests(fetchImpl = globalThis.fetch) {
    const tests = [
      {
        name: 'ReserveEngine API',
        url: 'http://localhost:5000/api/reserves/1',
        validator: (data) =>
          Array.isArray(data) &&
          data.length > 0 &&
          data[0].allocation !== undefined &&
          data[0].confidence !== undefined,
      },
      {
        name: 'PacingEngine API',
        url: 'http://localhost:5000/api/pacing/summary',
        validator: (data) =>
          Array.isArray(data) &&
          data.length > 0 &&
          data[0].quarter !== undefined &&
          data[0].deployment !== undefined,
      },
    ];

    for (const test of tests) {
      try {
        const response = await fetchImpl(test.url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        const status = test.validator(data) ? 'PASS' : 'FAIL invalid response structure';
        console.log(`[legacy] ${test.name}: ${status}`);
      } catch (error) {
        console.log(`[legacy] ${test.name}: FAIL ${error.message}`);
      }
    }
  }

  enableAlgorithms() {
    process.env.ALG_RESERVE = 'true';
    process.env.ALG_PACING = 'true';
    console.log('[legacy] ALG_RESERVE=true');
    console.log('[legacy] ALG_PACING=true');
  }
}

async function main(argv = process.argv.slice(2), env = process.env, io = process, deps = {}) {
  const options = parseArgs(argv);
  const targetCwd = options.project || ROOT;
  const runModel = deps.executeModel || executeModel;
  const gateRunner = deps.gateRunner || spawnSync;
  const ledgerWriter = deps.writeRunLedger === undefined ? writeRunLedger : deps.writeRunLedger;
  const debateRunWriter =
    deps.writeDebateRunRecord === undefined ? writeDebateRunRecord : deps.writeDebateRunRecord;
  const clock = deps.clock || (() => new Date());
  const root = deps.root || ROOT;
  const fsReader = deps.fs || { existsSync, readFileSync, readdirSync };
  const providerStateFs =
    deps.providerStateFs ||
    deps.fs || { existsSync, readFileSync, writeFileSync, renameSync, mkdirSync };
  const providerCooldownStore =
    deps.providerCooldownStore ||
    createProviderCooldownStore({ root, fs: providerStateFs, clock, io });
  const appendEvent = deps.appendRunEvent === undefined ? appendRunEvent : deps.appendRunEvent;
  const checkCommandExists = deps.commandExists || commandExists;
  const isCooling = deps.isCooling || ((provider) => providerCooldownStore.isCooling(provider));
  const ledgerScrubOptions = {
    envSecrets: deps.envSecrets,
    envPath: deps.envPath || join(root, '.env'),
    envLoader: deps.envLoader,
    scrubber: deps.scrubber,
  };

  if (options.help) {
    printHelp(io.stdout);
    return 0;
  }

  if (options.legacyCommand) {
    if (options.legacyCommand === 'doctor') {
      const runId = generateRunId(clock());
      const routingPath =
        env.HERMES_MODEL_ROUTING_FILE || join(root, '.claude', 'hermes', 'model-routing.json');
      let routing;
      try {
        routing = deps.routing || loadJSON(routingPath, { fs: fsReader });
      } catch (error) {
        writePreparedLedgerBestEffort({
          ledgerWriter,
          record: {
            runId,
            phase: options.phase,
            error: error.message,
            exitCode: 1,
            completedAt: clock().toISOString(),
          },
          scrubOptions: ledgerScrubOptions,
          io,
        });
        throw error;
      }
      const report = buildDoctorReport({
        routing,
        env,
        providers: DOCTOR_PROVIDERS,
        commandExists: checkCommandExists,
        cooldownStore: providerCooldownStore,
      });
      const staleRuns = scanStaleRunLedgers({
        root,
        fs: fsReader,
        clock,
      });
      const unclassifiedFailures = scanUnclassifiedFailures({ root, fs: fsReader });
      printDoctorReport(report, io.stdout, { staleRuns, unclassifiedFailures });
      return 0;
    }

    const orchestrator = new Orchestrator();
    if (options.legacyCommand === 'bootstrap') await orchestrator.bootstrap();
    if (options.legacyCommand === 'smoke') await orchestrator.runSmokeTests();
    if (options.legacyCommand === 'enable-algorithms') orchestrator.enableAlgorithms();
    return 0;
  }

  if (!options.task) {
    throw new Error('--task is required. Use --help for usage.');
  }

  const liveExecution = options.live || env.HERMES_LIVE === '1' || env.HERMES_LIVE === 'true';

  if (options.workflowProvided && !options.dryRun && !options.json && !liveExecution) {
    throw new Error('--workflow is planning-only; use --dry-run or --json. Add --live to execute.');
  }

  const routingPath =
    env.HERMES_MODEL_ROUTING_FILE || join(root, '.claude', 'hermes', 'model-routing.json');
  const brainPath = env.HERMES_DEV_BRAIN_FILE || join(root, 'DEV_BRAIN.md');
  const soulPath = env.HERMES_SOUL_FILE || join(root, '.claude', 'hermes', 'SOUL.md');
  const runId = generateRunId(clock());
  const startedAt = clock().toISOString();
  let routing;
  let brain;
  let soul;
  try {
    routing = deps.routing || loadJSON(routingPath, { fs: fsReader });
    brain = deps.brain ?? loadText(brainPath, { fs: fsReader });
    soul = deps.soul ?? loadText(soulPath, { optional: true, fs: fsReader });
  } catch (error) {
    writePreparedLedgerBestEffort({
      ledgerWriter,
      record: {
        runId,
        phase: options.phase,
        error: error.message,
        exitCode: 1,
        completedAt: clock().toISOString(),
      },
      scrubOptions: ledgerScrubOptions,
      io,
    });
    throw error;
  }
  let plan;
  try {
    plan = createRoutingPlan({
      phase: options.phase,
      task: options.task,
      routing,
      manualModel: options.manualModel,
      allowFallback: options.allowFallback,
      requestedWorkflow: options.workflowProvided ? options.workflow : null,
      skipPreflightGate: options.skipPreflightGate,
      gateSkipReason: options.gateSkipReason,
      env,
      commandExists: checkCommandExists,
      isCooling,
    });
    plan.targetProject = targetCwd;
  } catch (error) {
    if (!error.providerResolution) {
      throw error;
    }
    writePreparedLedgerBestEffort({
      ledgerWriter,
      record: {
        runId,
        startedAt,
        phase: options.phase,
        task: options.task,
        error: error.message,
        providerDiagnostics: error.providerDiagnostics,
        failureLedger: error.failureLedger,
        exitCode: 1,
        completedAt: clock().toISOString(),
      },
      scrubOptions: ledgerScrubOptions,
      io,
    });
    return 1;
  }
  const prompt = buildPrompt({ plan, brain, soul, runId });
  const emitEvent = createRunEventEmitter({
    runId,
    appendEvent,
    root,
    fs: deps.eventFs,
    clock,
    envSecrets: deps.envSecrets,
    envPath: deps.envPath || join(root, '.env'),
    envLoader: deps.envLoader,
    scrubber: deps.scrubber,
    io,
  });

  if (options.json) {
    io.stdout.write(`${JSON.stringify(plan, null, 2)}\n`);
    return 0;
  }

  if (options.dryRun) {
    io.stdout.write('=== ROUTING PLAN ===\n');
    io.stdout.write(`${JSON.stringify(plan, null, 2)}\n`);
    io.stdout.write('\n=== PROMPT ===\n');
    io.stdout.write(`${prompt}\n`);
    return 0;
  }

  const availability = buildProviderAvailability({
    routing,
    env,
    commandExists: checkCommandExists,
  });

  if (options.workflowProvided && liveExecution && plan.workflow) {
    // Preflight gate parity with the non-workflow path: a failing gate (e.g.
    // npm run check) must abort BEFORE spawning the owner/reviewer CLIs, unless
    // explicitly skipped. executeWorkflow only runs the gate postflight.
    const gates = getGateRunPlan(plan, options);
    if (gates.preflight) {
      const preflight = runGate(plan.gate, {
        env,
        runner: gateRunner,
        throwOnFailure: false,
      });
      emitEvent({
        type: RUN_EVENT_TYPES.GATE_RESULT,
        command: plan.gate,
        status: preflight.status ?? 0,
      });
      if (preflight.status !== 0) {
        io.stderr.write(
          `[hermes] preflight gate "${plan.gate}" failed with exit code ${preflight.status}; aborting live workflow before model execution.\n`
        );
        writePreparedLedgerBestEffort({
          ledgerWriter,
          record: {
            runId,
            startedAt,
            plan,
            preflight: { command: plan.gate, status: preflight.status, skipped: false },
            model: null,
            postflight: null,
            availability,
            exitCode: preflight.status,
            completedAt: clock().toISOString(),
          },
          scrubOptions: ledgerScrubOptions,
          io,
          emitEvent,
        });
        return preflight.status;
      }
    } else if (plan.gate && options.skipPreflightGate) {
      io.stderr.write(
        `[hermes] WARNING: skipping preflight gate "${plan.gate}"; reason: ${options.gateSkipReason}\n`
      );
    }

    const runStep =
      deps.runStep ||
      createLiveRunStep({
        routing,
        basePrompt: prompt,
        env,
        envSecrets: deps.envSecrets,
        envPath: deps.envPath,
        envLoader: deps.envLoader,
        scrubber: deps.scrubber,
        appendRunEvent: appendEvent,
        runId,
        root,
        eventFs: deps.eventFs,
        eventClock: clock,
        io,
        commandExists: checkCommandExists,
        providerCooldownStore,
        providerStateFs,
        cwd: targetCwd,
      });
    const record = await executeWorkflow(plan, {
      runStep,
      gateRunner,
      writeRunLedger: ledgerWriter,
      writeDebateRunRecord: debateRunWriter,
      writeRunCheckpoint:
        deps.writeRunCheckpoint === undefined ? writeRunCheckpoint : deps.writeRunCheckpoint,
      checkpointFs: deps.checkpointFs,
      appendRunEvent: appendEvent,
      eventFs: deps.eventFs,
      clock,
      runId,
      root,
      io,
      availability,
      routing,
      env,
      cwd: targetCwd,
      commandExists: checkCommandExists,
      isCooling,
      envSecrets: deps.envSecrets,
      envPath: deps.envPath,
      envLoader: deps.envLoader,
      scrubber: deps.scrubber,
      providerCooldownStore,
      providerStateFs,
    });
    return record.exitCode;
  }

  const ledger = {
    runId,
    startedAt,
    plan,
    preflight: null,
    model: null,
    postflight: null,
    availability,
    exitCode: null,
  };

  const finalizeLedger = (exitCode) => {
    ledger.exitCode = exitCode;
    ledger.completedAt = clock().toISOString();
    writePreparedLedgerBestEffort({
      ledgerWriter,
      record: ledger,
      scrubOptions: ledgerScrubOptions,
      io,
      emitEvent,
    });
  };

  const gates = getGateRunPlan(plan, options);

  if (gates.preflight) {
    const preflight = runGate(plan.gate, {
      env,
      runner: gateRunner,
      throwOnFailure: false,
    });
    emitEvent({
      type: RUN_EVENT_TYPES.GATE_RESULT,
      command: plan.gate,
      status: preflight.status ?? 0,
    });
    ledger.preflight = { command: plan.gate, status: preflight.status, skipped: false };
    if (preflight.status !== 0) {
      finalizeLedger(preflight.status);
      return preflight.status;
    }
  } else if (plan.gate && options.skipPreflightGate) {
    ledger.preflight = {
      command: plan.gate,
      status: null,
      skipped: true,
      reason: options.gateSkipReason,
    };
    io.stderr.write(
      `[hermes] WARNING: skipping preflight gate "${plan.gate}"; reason: ${options.gateSkipReason}\n`
    );
  }

  let modelResult;
  if (ladderEnabled(routing)) {
    modelResult = await executeSoloModelWithFallback({
      plan,
      prompt,
      routing,
      env,
      commandExists: checkCommandExists,
      isCooling,
      io,
      seams: {
        appendRunEvent: appendEvent,
        runId,
        role: 'solo',
        root,
        eventFs: deps.eventFs,
        eventClock: clock,
        envSecrets: deps.envSecrets,
        envPath: deps.envPath || join(root, '.env'),
        envLoader: deps.envLoader,
        scrubber: deps.scrubber,
        providerCooldownStore,
        providerStateFs,
        cwd: targetCwd,
      },
    });
  } else {
    const code = await runModel(plan.model, prompt, routing, env, {
      appendRunEvent: appendEvent,
      runId,
      role: 'solo',
      root,
      eventFs: deps.eventFs,
      eventClock: clock,
      envSecrets: deps.envSecrets,
      envPath: deps.envPath || join(root, '.env'),
      envLoader: deps.envLoader,
      scrubber: deps.scrubber,
      io,
      providerCooldownStore,
      providerStateFs,
      cwd: targetCwd,
    });
    modelResult = { code, model: plan.model };
  }
  const code = modelResult.code;
  ledger.model = { name: modelResult.model, exitCode: code };
  if (modelResult.providerDiagnostics?.length > 0) {
    ledger.model.providerDiagnostics = modelResult.providerDiagnostics;
  }
  if (modelResult.failureLedger?.length > 0) {
    ledger.failureLedger = modelResult.failureLedger;
  }

  if (shouldRunPostflightGate(plan, code, gates)) {
    const postflight = runGate(plan.gate, {
      env,
      runner: gateRunner,
      throwOnFailure: false,
    });
    emitEvent({
      type: RUN_EVENT_TYPES.GATE_RESULT,
      command: plan.gate,
      status: postflight.status ?? 0,
    });
    ledger.postflight = { command: plan.gate, status: postflight.status };
    if (postflight.status !== 0) {
      if (code !== 0) {
        io.stderr.write(
          `[hermes] WARNING: model exited ${code} and postflight gate exited ${postflight.status}\n`
        );
      }
      finalizeLedger(postflight.status);
      return postflight.status;
    }
  }

  finalizeLedger(code);
  return code;
}

if (isCliEntryPoint(import.meta.url, process.argv)) {
  main(process.argv.slice(2))
    .then((code) => {
      process.exitCode = code;
    })
    .catch((error) => {
      process.stderr.write(`[hermes] ${error.message}\n`);
      process.exitCode = 1;
    });
}

export {
  Orchestrator,
  appendRunEvent,
  assertFinancialGate,
  buildDoctorReport,
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
  isCliEntryPoint,
  isProductionFinancial,
  main,
  parseApprovalSignal,
  parseArgs,
  recommendWorkflow,
  resolveSandboxArgs,
  resolveEffectivePhase,
  resolveGate,
  resolveOwnership,
  isReviewLaneEligible,
  runGate,
  shouldRunPostflightGate,
  scoreSpecialist,
  validateDebateRunRecordShape,
  writeDebateRunRecord,
  writeRunLedger,
};

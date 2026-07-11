// PreToolUse: block Edit/Write/MultiEdit that (a) write a real dotenv file, or
// (b) commit a hardcoded secret (API-key-shaped literal). This repo has a
// documented history of hardcoded FRED / AlphaVantage keys in scripts.
import path from "node:path";

let raw = "";
process.stdin.on("data", c => (raw += c));
process.stdin.on("end", () => {
  try {
    const { tool_input = {}, cwd } = JSON.parse(raw || "{}");

    // (a) real dotenv file? (.env, .env.local, foo.env) -- allow *.example/.sample/.template
    if (tool_input.file_path) {
      const fp = tool_input.file_path.replace(/\\/g, "/");
      const base = fp.split("/").pop() || fp;
      const isDotenv = /(^|[\\/])\.env(\.|$)/.test(fp);
      const isTemplate = /\.(example|sample|template)$/i.test(base);
      if (isDotenv && !isTemplate) {
        process.stderr.write(
          `BLOCKED: ${tool_input.file_path} looks like a real dotenv file. ` +
          `Write .env.example (placeholders only) and load real values from the environment.\n`);
        process.exitCode = 2;                               // 2 = block; stderr -> Claude
        return;
      }
    }

    // (b) hardcoded secret in the write payload (Write/Edit/MultiEdit shapes)
    const chunks = [];
    if (typeof tool_input.content === "string") chunks.push(tool_input.content);          // Write
    if (typeof tool_input.new_string === "string") chunks.push(tool_input.new_string);    // Edit
    if (Array.isArray(tool_input.edits)) {                                                // MultiEdit
      for (const e of tool_input.edits) {
        if (e && typeof e.new_string === "string") chunks.push(e.new_string);
      }
    }
    const hit = chunks.map(scanSecret).find(Boolean);
    if (hit) {
      process.stderr.write(
        `BLOCKED: hardcoded secret detected (${hit}). ` +
        `Do NOT commit API keys/tokens. Read them from the environment ` +
        `(process.env.* / os.environ / getenv) and keep real values in an untracked .env.\n`);
      process.exitCode = 2;                                 // 2 = block; stderr -> Claude
    }
  } catch (e) {
    process.stderr.write(`guard-secrets: NOT enforcing (${e.message})\n`);  // fail open, loudly
  }
});

// Return a short reason string if the text contains a hardcoded secret, else "".
function scanSecret(text) {
  if (!text) return "";

  // Placeholders / env reads are always fine -- if a matched line looks like one, skip it.
  const PLACEHOLDER =
    /<|>|\$\{|YOUR_|xxxx|changeme|example|dummy|test|env\(|process\.env|os\.environ|getenv/i;

  const lines = text.split(/\r?\n/);

  for (const line of lines) {
    if (PLACEHOLDER.test(line)) continue;

    // OpenAI-style key anywhere on the line.
    if (/sk-[A-Za-z0-9]{20,}/.test(line)) return "OpenAI-style sk- key";

    // Named-key assignment to a quoted literal >= 16 chars.
    // Matches VAR = "literal" / VAR: 'literal' across JS/py/env/yaml/json shapes.
    const m = line.match(
      /([A-Za-z0-9_]*(?:API[_-]?KEY|APIKEY|KEY|TOKEN|SECRET|PASSWORD)[A-Za-z0-9_]*)\s*[:=]\s*(["'`])([^"'`]+)\2/i
    );
    if (m) {
      const name = m[1];
      const val = m[3];
      // known-vendor names catch even short/edge literals, but still need a real literal
      const VENDOR = /FRED|ALPHA[_-]?VANTAGE|ALPHAVANTAGE|OPENAI|FINNHUB|NEWSAPI/i;
      if (val.length >= 16) return `key-shaped literal assigned to ${name}`;
      if (VENDOR.test(name) && val.length >= 8) return `vendor key literal assigned to ${name}`;
    }

    // Long hex / base64 blob assigned to a secret-named variable.
    const b = line.match(
      /([A-Za-z0-9_]*(?:API[_-]?KEY|APIKEY|KEY|TOKEN|SECRET|PASSWORD)[A-Za-z0-9_]*)\s*[:=]\s*(["'`]?)([A-Za-z0-9+/=]{32,})\2/i
    );
    if (b) return `long hex/base64 blob assigned to ${b[1]}`;
  }
  return "";
}

// PreToolUse: block Edit/Write/MultiEdit that OVERWRITE frozen floor_prices.csv,
// PREREG.md, or UNIVERSE_RULE.md. Creating a new file is allowed (Task 0/14 write these).
import fs from "node:fs";
import path from "node:path";

let raw = "";
process.stdin.on("data", c => (raw += c));
process.stdin.on("end", () => {
  try {
    const { tool_input = {}, cwd } = JSON.parse(raw || "{}");
    if (!tool_input.file_path) return;                    // exitCode defaults to 0 -> allow
    const resolved = path.resolve(cwd || process.cwd(), tool_input.file_path);
    const FROZEN = /(\/fixtures\/floor_prices\.csv|\/PREREG\.md|\/UNIVERSE_RULE\.md)$/;
    if (FROZEN.test(resolved.replace(/\\/g, "/")) && fs.existsSync(resolved)) {
      process.stderr.write(`BLOCKED: ${tool_input.file_path} is frozen (Plan 4 floor-only). ` +
        `Remove the guard in .claude/settings.json to override.\n`);
      process.exitCode = 2;                               // 2 = block; stderr -> Claude
    }
  } catch (e) {
    process.stderr.write(`guard-frozen-floor: NOT enforcing (${e.message})\n`);  // fail open, loudly
  }
});

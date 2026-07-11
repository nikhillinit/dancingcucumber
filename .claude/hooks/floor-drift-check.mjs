// PostToolUse (Edit/Write/MultiEdit): NON-BLOCKING drift warning. ALWAYS exit 0.
// If the edited file is under apps/quant/ or is a frozen pin, run a FAST git check
// (status --porcelain + diff --name-only) and warn -- LOUDLY -- if any frozen pin
// looks modified. Never runs the backtest (too slow for a per-edit hook).
import path from "node:path";
import { spawnSync } from "node:child_process";

// Frozen pins (path suffixes, forward-slash normalized).
const PINS = [/fixtures\/floor_prices\.csv$/, /PREREG\.md$/, /UNIVERSE_RULE\.md$/];

let raw = "";
process.stdin.on("data", c => (raw += c));
process.stdin.on("end", () => {
  try {
    const { tool_input = {}, cwd } = JSON.parse(raw || "{}");
    const fp = (tool_input.file_path || "").replace(/\\/g, "/");
    if (!fp) return;                                        // nothing to check -> exit 0

    const isQuant = /(^|\/)apps\/quant\//.test(fp);
    const isPin = PINS.some(re => re.test(fp));
    if (!isQuant && !isPin) return;                         // unrelated to quant engine -> exit 0

    const root = cwd || process.cwd();
    const opts = { cwd: root, encoding: "utf8", shell: process.platform === "win32" };
    const status = spawnSync("git", ["status", "--porcelain"], opts);
    const diff = spawnSync("git", ["diff", "--name-only"], opts);
    const gitText =
      `${status.stdout || ""}\n${status.stderr || ""}\n${diff.stdout || ""}`.replace(/\\/g, "/");

    const drifted = PINS.some(re =>
      gitText.split(/\r?\n/).some(line => re.test(line.trim()))
    );

    if (drifted) {
      process.stderr.write(
        "\n=== FLOOR DRIFT WARNING ==========================================\n" +
        "A FROZEN PIN (fixtures/floor_prices.csv, PREREG.md, or UNIVERSE_RULE.md)\n" +
        "appears MODIFIED in the working tree. The frozen floor may have drifted.\n" +
        "  -> Run:  npm run advisor-gate      (verify the floor is intact)\n" +
        "Release readiness = npm run advisor-release-gate\n" +
        "(NOTE: floor --enforce exiting 1 is the EXPECTED pinned behavior, not a bug.)\n" +
        "==================================================================\n"
      );
    }
  } catch (e) {
    process.stderr.write(`floor-drift-check: skipped (${e.message})\n`);  // never block
  }
  // exitCode stays 0 unconditionally.
});

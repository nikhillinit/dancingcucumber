"""Conformance floor for research-governance artifacts (spec 2026-07-05).

Machine-checks artifact discipline ONLY (see LANE_LIFECYCLE.md "Not machine-checked"):
  1. anchors on future preregs in the canonical home
  2. prereg location enforcement (repo-wide)
  3. canonical freeze citations, git-verified (rev-parse <commit>:<path> == blob SHA)
  4. PROGRAM_RECORD row hash chain + tip footer (append-only)
  5. template self-check
Helpers are module-level and unit-tested against synthetic inputs (negative tests).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
DOCS = REPO_ROOT / "docs" / "superpowers"
HARNESS = DOCS / "harness"
CANONICAL_PREREG_HOME = DOCS / "plans"

ANCHOR_PATTERNS = {
    "decision rule": re.compile(r"decision rule", re.IGNORECASE),
    "budget cap": re.compile(r"budget", re.IGNORECASE),
    "STOP condition": re.compile(r"\bSTOP\b"),
}

# Predate the anchor phrasing; verifiably lack it. Never grows.
GRANDFATHERED_PREREGS = {
    "2026-06-23-qc-edgar-diagnostic-v2-prereg.md",
    "2026-06-23-qc-source-integrity-diagnostic-prereg.md",
}

# Frozen surfaces governed by their own hash pins. Never grows without an
# operator decision recorded in the closeout that adds the entry.
LEGACY_PREREG_PATHS = {
    "apps/quant/advisor/backtest/PREREG.md",
    "apps/quant/advisor/backtest/VALIDATION_PREREG.md",
    "apps/quant/advisor/research/READING_B_PREREG.md",
    "apps/quant/advisor/research/READING_C_PREREG.md",
    "apps/quant/advisor/research/CANDIDATE_PREREG.md",
    "apps/quant/advisor/research/LS_REVERSAL_PREREG.md",
}

EXCLUDED_DIRS = {
    ".git", "node_modules", "stefan-jansen-ml", ".omx", ".claude", ".remember",
    ".venv", "venv",
}

CITATION_RE = re.compile(
    r"^Frozen criteria: (?P<path>\S+) \| Freeze commit: (?P<commit>[0-9a-f]{40})"
    r" \| blob SHA: (?P<blob>[0-9a-f]{40})\s*$"
)

# The SESTM Phase-0 matrix predates the one-line canonical format (two-line
# citation); its freeze claim is pinned here and verified like any other.
PINNED_LEGACY_CITATION = (
    "docs/superpowers/plans/2026-07-04-sestm-phase0-prereg.md",
    "d420af72518096a332ac85ee70e4fd104db7e972",
    "a42cc3dd643477724bcb8a8ab34c2eea0464198d",
)

TIP_RE = re.compile(r"^Chain tip: (?P<tip>[0-9a-f]{12}) over (?P<count>\d+) rows\s*$")


def missing_anchors(text: str) -> list[str]:
    return [name for name, pat in ANCHOR_PATTERNS.items() if not pat.search(text)]


def find_misplaced_preregs(root: Path) -> list[Path]:
    # Allowed homes derive from ROOT (not module constants) so the tmp_path
    # negative/positive tests exercise the identical code path used live.
    canonical = (root / "docs" / "superpowers" / "plans").resolve()
    templates = (root / "docs" / "superpowers" / "harness" / "templates").resolve()
    hits = []
    for p in root.rglob("*.md"):
        if any(part in EXCLUDED_DIRS for part in p.parts):
            continue
        if "prereg" not in p.name.lower():
            continue
        if p.parent.resolve() in (canonical, templates):
            continue
        if p.relative_to(root).as_posix() in LEGACY_PREREG_PATHS:
            continue
        hits.append(p)
    return hits


def citation_candidates(text: str) -> list[tuple[int, str]]:
    # Skip fenced code blocks and indented literals: docs (including plans and
    # the templates' instructional footers) may QUOTE the citation format; only
    # column-0 unfenced lines are real freeze claims. Known limit: an unbalanced
    # fence in a doc suppresses scanning for the file remainder.
    out, fenced = [], False
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced or line.startswith(("    ", "\t")):
            continue
        if line.startswith("Frozen criteria:") and "Freeze commit:" in line:
            out.append((i, line))
    return out


def git_blob_sha(commit: str, path: str) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", f"{commit}:{path}"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"freeze citation unverifiable: git rev-parse {commit}:{path} -> "
            f"{proc.stderr.strip()} (unreachable commit/path is REAL corruption, "
            "never skipped)"
        )
    return proc.stdout.strip()


def record_table(text: str) -> tuple[list[list[str]], str | None, int | None, int]:
    """Header-anchored parse: only rows of THE schema table count (a future second
    table or a stray pipe in prose must not join the chain). Returns
    (rows, tip, count, header_count)."""
    rows, tip, count, headers, in_table = [], None, None, 0, False
    for line in text.splitlines():
        if line.startswith("| #"):
            headers += 1
            in_table = True
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and set(cells[0]) <= {"-", " ", ":"}:
                continue
            rows.append(cells)
            continue
        in_table = False
        m = TIP_RE.match(line)
        if m:
            tip, count = m["tip"], int(m["count"])
    return rows, tip, count, headers


def expected_chain(rows: list[list[str]]) -> list[str]:
    chain, out = "GENESIS", []
    for cells in rows:
        row_text = "|".join(cells[:-1])
        chain = hashlib.sha256(f"{chain}|{row_text}".encode()).hexdigest()[:12]
        out.append(chain)
    return out


# ---------- live checks against the real tree ----------

def test_grandfather_files_exist():
    for name in GRANDFATHERED_PREREGS:
        assert (CANONICAL_PREREG_HOME / name).is_file(), f"grandfather pin rotted: {name}"
    assert (CANONICAL_PREREG_HOME / "2026-07-04-sestm-phase0-prereg.md").is_file()


def test_future_preregs_carry_anchors():
    for p in sorted(CANONICAL_PREREG_HOME.glob("*-prereg.md")):
        if p.name in GRANDFATHERED_PREREGS:
            continue
        missing = missing_anchors(p.read_text(encoding="utf-8"))
        assert not missing, f"{p.name} missing anchors: {missing}"


def test_no_misplaced_preregs():
    hits = find_misplaced_preregs(REPO_ROOT)
    assert not hits, f"prereg outside canonical home + legacy set: {[str(h) for h in hits]}"


def test_canonical_citations_verify():
    for sub in ("notes", "plans"):
        for p in sorted((DOCS / sub).glob("*.md")):
            text = p.read_text(encoding="utf-8")
            for lineno, line in citation_candidates(text):
                m = CITATION_RE.match(line)
                assert m, (
                    f"{p.name}:{lineno} looks like a freeze citation but does not "
                    f"parse canonically — a lane must not LOOK frozen while "
                    f"unverifiable: {line!r}"
                )
                actual = git_blob_sha(m["commit"], m["path"])
                assert actual == m["blob"], (
                    f"{p.name}:{lineno} freeze claim FALSE: blob at commit is "
                    f"{actual}, cited {m['blob']}"
                )


def test_pinned_sestm_freeze_verifies():
    path, commit, blob = PINNED_LEGACY_CITATION
    assert git_blob_sha(commit, path) == blob


def test_program_record_chain():
    text = (HARNESS / "PROGRAM_RECORD.md").read_text(encoding="utf-8")
    rows, tip, count, headers = record_table(text)
    assert headers == 1, "PROGRAM_RECORD must contain exactly ONE table"
    assert len(rows) >= 8, "seed rows missing"
    for i, cells in enumerate(rows, 1):
        assert len(cells) == 6, (
            f"row {i} has {len(cells)} cells; schema needs 6 — forgot the chain cell?"
        )
    exp = expected_chain(rows)
    for i, (cells, want) in enumerate(zip(rows, exp), 1):
        assert cells[-1] == want, (
            f"PROGRAM_RECORD row {i} chain mismatch: has {cells[-1]!r}, expected "
            f"{want!r}. If appending a new row, paste this expected hash. If NOT "
            "appending, a row was edited/reordered/deleted — restore it."
        )
    assert tip is not None and count is not None, "chain-tip footer missing"
    assert count == len(rows) and tip == exp[-1], (
        f"chain tip says {tip!r}/{count} rows but table has {exp[-1]!r}/{len(rows)}: "
        "either tail rows were deleted, or you appended without updating the footer "
        f"— expected footer: 'Chain tip: {exp[-1]} over {len(rows)} rows'"
    )


def test_templates_self_check():
    t = HARNESS / "templates"
    prereg = (t / "prereg-template.md").read_text(encoding="utf-8")
    assert not missing_anchors(prereg)
    for name in ("source-matrix-template.md", "closeout-template.md"):
        body = (t / name).read_text(encoding="utf-8")
        assert "Frozen criteria: <" in body, f"{name} lost the canonical citation line"


# ---------- negative self-tests (prove the checks detect violations) ----------

def _chained(n):
    rows = [[str(i), f"lane{i}", "STOP", "n", "ptr"] for i in range(1, n + 1)]
    hashes = expected_chain([r + [""] for r in rows])
    return [r + [h] for r, h in zip(rows, hashes)]


def test_neg_missing_anchor_detected():
    assert missing_anchors("decision rule and budget only") == ["STOP condition"]


def test_neg_malformed_citation_detected():
    bad = "Frozen criteria: some/path | Freeze commit: deadbeef | blob SHA: cafe"
    (lineno, line), = citation_candidates(bad)
    assert CITATION_RE.match(line) is None


def test_neg_fenced_citation_ignored():
    quoted = "```\nFrozen criteria: x | Freeze commit: y | blob SHA: z\n```\n" \
             "    Frozen criteria: a | Freeze commit: b | blob SHA: c"
    assert citation_candidates(quoted) == []


def test_neg_unreachable_commit_fails():
    with pytest.raises(AssertionError, match="unverifiable"):
        git_blob_sha("0" * 40, "README.md")


def test_neg_tampered_record_row_detected():
    good = _chained(2)
    good[0][1] = "edited-lane"  # tamper
    assert expected_chain(good)[0] != good[0][-1]


def test_neg_reordered_rows_detected():
    good = _chained(3)
    swapped = [good[1], good[0], good[2]]
    exp = expected_chain(swapped)
    assert any(r[-1] != w for r, w in zip(swapped, exp))


def test_neg_interior_deletion_detected():
    good = _chained(3)
    cut = [good[0], good[2]]  # delete middle row
    assert cut[1][-1] != expected_chain(cut)[1]


def test_neg_tail_deletion_detected_by_tip():
    good = _chained(3)
    tip_before = good[-1][-1]
    cut = good[:2]  # delete last row; footer still pins tip_before
    assert expected_chain(cut)[-1] != tip_before


def test_neg_misplaced_prereg_detected(tmp_path):
    nest = tmp_path / "apps" / "sub"
    nest.mkdir(parents=True)
    (nest / "sneaky-prereg.md").write_text("x", encoding="utf-8")
    (tmp_path / "docs" / "superpowers" / "plans").mkdir(parents=True)
    hits = find_misplaced_preregs(tmp_path)
    assert [h.name for h in hits] == ["sneaky-prereg.md"]


def test_neg_allowed_homes_not_flagged(tmp_path):
    plans = tmp_path / "docs" / "superpowers" / "plans"
    templates = tmp_path / "docs" / "superpowers" / "harness" / "templates"
    plans.mkdir(parents=True)
    templates.mkdir(parents=True)
    (plans / "2099-01-01-lane-prereg.md").write_text("x", encoding="utf-8")
    (templates / "prereg-template.md").write_text("x", encoding="utf-8")
    assert find_misplaced_preregs(tmp_path) == []

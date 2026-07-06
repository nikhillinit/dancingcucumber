from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from advisor.diagnostics.portfolio import DiagnosticsInputError, load_portfolio
from advisor.diagnostics.report import build_report, render_json, render_text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m advisor.diagnostics")
    parser.add_argument("--positions", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out")
    return parser


def _require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise DiagnosticsInputError(f"{label} file not found: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        _require_file(args.positions, "positions")
        _require_file(args.prices, "prices")
        report = build_report(load_portfolio(args.positions, args.prices))
        rendered = render_json(report) if args.json else render_text(report)
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(rendered)
        else:
            sys.stdout.write(rendered)
        return 0
    except (DiagnosticsInputError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

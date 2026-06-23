from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path


EDGAR_CLASSES = ("bankruptcy", "acquisition", "performance", "unknown")
ADVERSE_CLASSES = {"bankruptcy", "performance"}
FORM25_FORMS = {"25", "25-NSE"}
OUTPUT_FIELDS = ("cik", "company", "delist_date", "n_form25", "edgar_class", "adverse")

BASE_DIR = Path(__file__).resolve().parent
DELISTINGS_CSV = BASE_DIR / "edgar_delistings_2015_2023.csv"
FILINGS_CSV = BASE_DIR / "edgar_b2_filings.csv"
OUTPUT_CSV = BASE_DIR / "edgar_delisting_reasons.csv"


@dataclass(frozen=True)
class DelistingEvent:
    cik: str
    company: str
    delist_date: date
    n_form25: int


def parse_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def item_tokens(items: str) -> set[str]:
    return {token.strip() for token in items.split(",") if token.strip()}


def is_within_days(left: date, right: date, days: int) -> bool:
    return abs((left - right).days) <= days


def has_8k_item(filings: list[dict[str, str]], delist_date: date, item: str) -> bool:
    for filing in filings:
        if filing.get("form", "").strip().upper() != "8-K":
            continue
        filing_date = parse_date(filing.get("filing_date", ""))
        if is_within_days(filing_date, delist_date, 90) and item in item_tokens(filing.get("items", "")):
            return True
    return False


def has_prior_defm14a(filings: list[dict[str, str]], delist_date: date) -> bool:
    for filing in filings:
        if filing.get("form", "").strip().upper() != "DEFM14A":
            continue
        filing_date = parse_date(filing.get("filing_date", ""))
        days_before = (delist_date - filing_date).days
        if 0 <= days_before <= 365:
            return True
    return False


def classify_event(delist_date: date, filings: list[dict[str, str]]) -> str:
    if has_8k_item(filings, delist_date, "1.03"):
        return "bankruptcy"
    if has_8k_item(filings, delist_date, "2.01") or has_prior_defm14a(filings, delist_date):
        return "acquisition"
    if has_8k_item(filings, delist_date, "3.01"):
        return "performance"
    return "unknown"


def normalize_cik(cik: str) -> str:
    stripped = cik.strip()
    return stripped.lstrip("0") or "0"


def cik_sort_key(cik: str) -> tuple[int, int | str]:
    normalized = normalize_cik(cik)
    try:
        return (0, int(normalized))
    except ValueError:
        return (1, normalized)


def build_delisting_events(rows: list[dict[str, str]]) -> list[DelistingEvent]:
    by_cik: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        form = row.get("form", "").strip().upper()
        if form in FORM25_FORMS:
            by_cik[normalize_cik(row.get("cik", ""))].append(row)

    events: list[DelistingEvent] = []
    for cik in sorted(by_cik, key=cik_sort_key):
        cik_rows = sorted(by_cik[cik], key=lambda row: parse_date(row["date"]))
        cluster: list[dict[str, str]] = []
        previous_date: date | None = None
        for row in cik_rows:
            row_date = parse_date(row["date"])
            if previous_date is not None and (row_date - previous_date).days > 30:
                events.append(event_from_cluster(cluster))
                cluster = []
            cluster.append(row)
            previous_date = row_date
        if cluster:
            events.append(event_from_cluster(cluster))
    return events


def event_from_cluster(cluster: list[dict[str, str]]) -> DelistingEvent:
    ordered = sorted(cluster, key=lambda row: parse_date(row["date"]))
    first = ordered[0]
    return DelistingEvent(
        cik=first.get("cik", "").strip(),
        company=first.get("company", "").strip(),
        delist_date=parse_date(first["date"]),
        n_form25=len(ordered),
    )


def classify_delisting_events(
    events: list[DelistingEvent],
    filings: list[dict[str, str]],
) -> list[dict[str, str]]:
    filings_by_cik: dict[str, list[dict[str, str]]] = defaultdict(list)
    for filing in filings:
        filings_by_cik[normalize_cik(filing.get("cik", ""))].append(filing)

    rows: list[dict[str, str]] = []
    for event in events:
        edgar_class = classify_event(event.delist_date, filings_by_cik.get(normalize_cik(event.cik), []))
        rows.append(
            {
                "cik": event.cik,
                "company": event.company,
                "delist_date": event.delist_date.isoformat(),
                "n_form25": str(event.n_form25),
                "edgar_class": edgar_class,
                "adverse": str(edgar_class in ADVERSE_CLASSES).lower(),
            }
        )
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8-sig", b"", 0, 1, f"could not decode {path}")


def write_output(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_distribution(rows: list[dict[str, str]]) -> None:
    total = len(rows)
    counts = Counter(row["edgar_class"] for row in rows)
    adverse = sum(1 for row in rows if row["adverse"] == "true")

    print(f"total events: {total}")
    for edgar_class in EDGAR_CLASSES:
        pct = (counts[edgar_class] / total * 100) if total else 0.0
        print(f"{edgar_class}: {counts[edgar_class]} ({pct:.2f}%)")
    print(f"total adverse: {adverse}")


def main() -> None:
    delisting_rows = read_csv_rows(DELISTINGS_CSV)
    filing_rows = read_csv_rows(FILINGS_CSV)
    events = build_delisting_events(delisting_rows)
    output_rows = classify_delisting_events(events, filing_rows)
    write_output(output_rows, OUTPUT_CSV)
    print_distribution(output_rows)


if __name__ == "__main__":
    main()

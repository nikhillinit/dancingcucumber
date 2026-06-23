"""
QC B3 LEAN diagnostic: eligible-universe delisting rows for the EDGAR join.

NON-ALPHA guardrail:
- This file does not trade.
- It does not compute or output portfolio metrics.
- The only price-derived value is a trailing 12-1 month price-change label used
  to assign cross-sectional momentum deciles for delisting bookkeeping.

Operator workflow on QuantConnect free tier:
1. Research notebook:
   - Paste this file or import it into a QuantBook notebook.
   - Run:
       qb = QuantBook()
       run_b3_quantbook_research(qb)
   - This writes membership snapshots to ObjectStore.
2. Backtest project:
   - Paste/run the QCB3DelistingDiagnostic algorithm below.
   - It tries map-file delisting dates first when the API is reachable.
   - If map-file enumeration is unavailable, it keeps candidate securities
     subscribed and captures Delisting(Type=Delisted) events from Slice data.
   - It writes qc_eligible_delistings.csv to ObjectStore and logs the CSV.

Known QC API uncertainty:
- Calls marked "# UNVERIFIED" are based on prior API notes in the task handoff
  and are guarded with fallbacks where possible.
"""

from AlgorithmImports import *  # noqa: F401,F403

from collections import defaultdict
from datetime import datetime, timedelta
import csv
import io
import json
import math


START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2023, 12, 31)

MIN_MEDIAN_DOLLAR_VOLUME = 1_000_000
LIQUIDITY_LOOKBACK_TRADING_DAYS = 21
ELIGIBLE_LOOKBACK_DAYS = 366

COMMON_STOCK_SECURITY_TYPE = "ST00000001"

MEMBERSHIP_OBJECTSTORE_KEY = "qc_b3_membership_snapshots.json"
FINAL_CSV_OBJECTSTORE_KEY = "qc_eligible_delistings.csv"

FINAL_CSV_COLUMNS = [
    "symbol",
    "company_name",
    "last_eligible_date",
    "delist_date",
    "momentum_decile",
    "value_bucket",
    "negative_book_flag",
    "market_cap",
]


def _iso_date(value):
    if value is None:
        return ""
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)[:10]


def _parse_iso_date(value):
    if not value:
        return None
    return datetime.strptime(str(value)[:10], "%Y-%m-%d")


def _date_key(value):
    if value is None:
        return None
    if hasattr(value, "date"):
        return value.date().isoformat()
    text = str(value)
    if len(text) >= 10:
        return text[:10]
    return text


def _as_float(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _symbol_key(symbol):
    """Stable string key across notebook JSON and QCAlgorithm joins."""
    if symbol is None:
        return ""
    if hasattr(symbol, "Value"):
        return str(symbol.Value)
    return str(symbol)


def _get_attr(obj, dotted_path, default=None):
    current = obj
    for part in dotted_path.split("."):
        if current is None:
            return default
        current = getattr(current, part, default)
    return current


def _security_type_code(fine):
    sec_ref = _get_attr(fine, "SecurityReference")
    value = getattr(sec_ref, "SecurityType", None)
    if value is None:
        return ""
    return str(value)


def _is_us_common_share(fine):
    sec_ref = _get_attr(fine, "SecurityReference")
    return (
        bool(getattr(sec_ref, "IsPrimaryShare", False))
        and not bool(getattr(sec_ref, "IsDepositaryReceipt", True))
        and _security_type_code(fine) == COMMON_STOCK_SECURITY_TYPE
    )


def _company_name(fine):
    for path in (
        "CompanyReference.LegalName",
        "CompanyReference.ShortName",
        "CompanyReference.StandardName",
        "CompanyProfile.CompanyName",
    ):
        value = _get_attr(fine, path)
        if value:
            return str(value).replace("\n", " ").strip()
    return ""


def _book_value_per_share(fine):
    return _as_float(_get_attr(fine, "ValuationRatios.BookValuePerShare"))


def _market_cap(fine):
    return _as_float(getattr(fine, "MarketCap", None))


def _market_cap_percentiles(candidates):
    """Return symbol-key -> percentile in the cross-section, low cap to high cap."""
    ordered = sorted(
        (
            (_symbol_key(getattr(item["fine"], "Symbol", item["symbol"])), item["market_cap"])
            for item in candidates
            if item.get("market_cap") is not None and item.get("market_cap") > 0
        ),
        key=lambda pair: pair[1],
    )
    total = len(ordered)
    if total == 0:
        return {}
    if total == 1:
        return {ordered[0][0]: 100.0}
    return {
        key: 100.0 * rank_index / (total - 1)
        for rank_index, (key, _) in enumerate(ordered)
    }


def _deciles_from_values(rows, value_key, output_key, ascending=True):
    """
    Assign 1..10 cross-sectional buckets over rows with finite values.

    For momentum, ascending=True means decile 1 is the weakest price-change label
    and decile 10 is the strongest. For value, decile 1 is lowest book-to-price
    and decile 10 is highest book-to-price. Negative book is handled separately.
    """
    valid = [row for row in rows if _as_float(row.get(value_key)) is not None]
    valid.sort(key=lambda row: row[value_key], reverse=not ascending)
    count = len(valid)
    if count == 0:
        return
    for rank_index, row in enumerate(valid):
        if count == 1:
            decile = 10
        else:
            decile = 1 + int((rank_index * 10) / count)
            decile = max(1, min(10, decile))
        row[output_key] = decile


def _month_starts(start, end):
    current = datetime(start.year, start.month, 1)
    while current <= end:
        yield current
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)


def _history_rows(history_result):
    """
    Normalize common QuantBook history shapes into row-like objects.

    QC pandas history is the usual shape in Research; LEAN enumerables may also
    appear. This helper intentionally stays defensive because this file must be
    pasted into QC, not run in this local sandbox.
    """
    if history_result is None:
        return []

    if hasattr(history_result, "iterrows"):
        rows = []
        for index, row in history_result.iterrows():
            rows.append((index, row))
        return rows

    try:
        return list(history_result)
    except TypeError:
        return []


def _row_field(row, *names):
    for name in names:
        if isinstance(row, dict) and name in row:
            return row[name]
        if hasattr(row, "__getitem__"):
            try:
                return row[name]
            except Exception:
                pass
        if hasattr(row, name):
            return getattr(row, name)
    return None


def _daily_dollar_volume_median(qb, symbol, as_of):
    """
    Compute 21-day median dollar volume from daily price/volume history.

    The task notes CoarseFundamental exposes a daily DollarVolume value, but the
    research side is more portable if we compute close * volume from TradeBars.
    This is still a liquidity label, not an alpha metric.
    """
    start = as_of - timedelta(days=45)
    try:
        history = qb.History(symbol, start, as_of, Resolution.Daily)  # UNVERIFIED
    except Exception:
        try:
            history = qb.history(symbol, start, as_of, Resolution.Daily)  # UNVERIFIED
        except Exception:
            return None

    dollar_volumes = []
    for item in _history_rows(history):
        row = item[1] if isinstance(item, tuple) and len(item) == 2 else item
        close = _as_float(_row_field(row, "close", "Close", "price", "Price"))
        volume = _as_float(_row_field(row, "volume", "Volume"))
        dollar_volume = _as_float(_row_field(row, "dollarvolume", "DollarVolume"))
        if dollar_volume is None and close is not None and volume is not None:
            dollar_volume = close * volume
        if dollar_volume is not None and dollar_volume > 0:
            dollar_volumes.append(dollar_volume)

    if len(dollar_volumes) < LIQUIDITY_LOOKBACK_TRADING_DAYS:
        return None
    recent = dollar_volumes[-LIQUIDITY_LOOKBACK_TRADING_DAYS:]
    recent.sort()
    middle = len(recent) // 2
    if len(recent) % 2 == 1:
        return recent[middle]
    return 0.5 * (recent[middle - 1] + recent[middle])


def _price_on_or_before(qb, symbol, target_date, max_lookback_days=12):
    start = target_date - timedelta(days=max_lookback_days)
    try:
        history = qb.History(symbol, start, target_date, Resolution.Daily)  # UNVERIFIED
    except Exception:
        try:
            history = qb.history(symbol, start, target_date, Resolution.Daily)  # UNVERIFIED
        except Exception:
            return None

    last_close = None
    for item in _history_rows(history):
        row = item[1] if isinstance(item, tuple) and len(item) == 2 else item
        close = _as_float(_row_field(row, "close", "Close", "price", "Price"))
        if close is not None and close > 0:
            last_close = close
    return last_close


def _price_change_12_1_label(qb, symbol, as_of):
    """
    Price-only 12-1 month label for cross-sectional bucketing.

    This is only used to assign momentum_decile as required by the prereg.
    """
    end_price = _price_on_or_before(qb, symbol, as_of - timedelta(days=30))
    start_price = _price_on_or_before(qb, symbol, as_of - timedelta(days=365))
    if end_price is None or start_price is None or start_price <= 0:
        return None
    return (end_price / start_price) - 1.0


def _fine_items_from_universe_history(universe_history):
    """
    Yield (date, fine) pairs from QuantBook universe_history output.

    The task handoff says qb.universe_history(qb.fundamental, start, end,
    flatten=True) returns historical Fundamental snapshots. The exact Research
    return shape varies by QC version, so this handles DataFrame-like and
    enumerable-like results.
    """
    if universe_history is None:
        return

    if hasattr(universe_history, "iterrows"):
        for index, row in universe_history.iterrows():
            fine = _row_field(row, "fundamental", "Fundamental", "fine", "Fine")
            if fine is None:
                fine = row
            date = None
            if isinstance(index, tuple):
                for part in index:
                    if hasattr(part, "date"):
                        date = part
                        break
            elif hasattr(index, "date"):
                date = index
            date = date or _row_field(row, "time", "Time", "date", "Date")
            yield date, fine
        return

    for item in universe_history:
        date = getattr(item, "EndTime", None) or getattr(item, "Time", None)
        fine = getattr(item, "Fundamental", None) or item
        yield date, fine


def _select_monthly_membership(qb, selection_date):
    """
    Build one point-in-time eligible cross-section and assign rank labels.

    Filters:
    - US common shares only.
    - 21-day median dollar volume >= $1,000,000.
    - market-cap percentile 20..90 after common/liquidity filters.
    """
    start = selection_date
    end = selection_date + timedelta(days=7)
    try:
        history = qb.universe_history(qb.fundamental, start, end, flatten=True)  # UNVERIFIED
    except Exception:
        history = qb.UniverseHistory(qb.Fundamental, start, end, True)  # UNVERIFIED

    grouped_by_date = defaultdict(list)
    for snapshot_date, fine in _fine_items_from_universe_history(history):
        key = _date_key(snapshot_date)
        if key:
            grouped_by_date[key].append(fine)

    if not grouped_by_date:
        return []

    actual_selection_key = sorted(grouped_by_date.keys())[0]
    actual_selection_date = _parse_iso_date(actual_selection_key)

    prelim = []
    for fine in grouped_by_date[actual_selection_key]:
        symbol = getattr(fine, "Symbol", None)
        if symbol is None or not _is_us_common_share(fine):
            continue
        market_cap = _market_cap(fine)
        if market_cap is None or market_cap <= 0:
            continue
        median_dollar_volume = _daily_dollar_volume_median(qb, symbol, actual_selection_date)
        if (
            median_dollar_volume is None
            or median_dollar_volume < MIN_MEDIAN_DOLLAR_VOLUME
        ):
            continue
        prelim.append(
            {
                "symbol": symbol,
                "fine": fine,
                "market_cap": market_cap,
                "median_dollar_volume": median_dollar_volume,
            }
        )

    percentiles = _market_cap_percentiles(prelim)
    rows = []
    for item in prelim:
        fine = item["fine"]
        symbol = item["symbol"]
        key = _symbol_key(symbol)
        cap_pct = percentiles.get(key)
        if cap_pct is None or cap_pct < 20.0 or cap_pct > 90.0:
            continue

        current_price = _price_on_or_before(qb, symbol, actual_selection_date)
        bvps = _book_value_per_share(fine)
        negative_book = bool(bvps is not None and bvps < 0)
        book_to_price = None
        if (
            bvps is not None
            and bvps >= 0
            and current_price is not None
            and current_price > 0
        ):
            book_to_price = bvps / current_price

        rows.append(
            {
                "symbol": key,
                "company_name": _company_name(fine),
                "date": actual_selection_key,
                "market_cap": item["market_cap"],
                "median_dollar_volume": item["median_dollar_volume"],
                "price_change_12_1": _price_change_12_1_label(qb, symbol, actual_selection_date),
                "book_value_per_share": bvps,
                "negative_book_flag": negative_book,
                "book_to_price": book_to_price,
                "value_bucket": (
                    "negative_book" if negative_book else "missing_book_or_price"
                ),
            }
        )

    _deciles_from_values(rows, "price_change_12_1", "momentum_decile", ascending=True)

    positive_book_rows = [
        row
        for row in rows
        if not row["negative_book_flag"] and _as_float(row.get("book_to_price")) is not None
    ]
    _deciles_from_values(positive_book_rows, "book_to_price", "value_bucket", ascending=True)

    # Drop helper-only liquidity/sort-axis values. Retain the prereg-required
    # 12-1 price label and BVPS in the membership store; the final delisting CSV
    # emits only decile/bucket labels, not raw sort values.
    for row in rows:
        row.pop("median_dollar_volume", None)
        row.pop("book_to_price", None)
        if "momentum_decile" not in row:
            row["momentum_decile"] = ""
        row["value_bucket"] = str(row["value_bucket"])
    return rows


def run_b3_quantbook_research(qb, start=START_DATE, end=END_DATE):
    """
    Research-notebook half.

    Builds monthly eligible membership/rank snapshots and writes them to
    ObjectStore as JSON. Chunking is by year, matching the free-tier memory
    constraint in the handoff.
    """
    all_rows = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, datetime(year, 1, 1))
        year_end = min(end, datetime(year, 12, 31))
        year_rows = []
        for selection_date in _month_starts(year_start, year_end):
            if selection_date < start or selection_date > end:
                continue
            year_rows.extend(_select_monthly_membership(qb, selection_date))

        year_key = "qc_b3_membership_snapshots_{0}.json".format(year)
        payload = json.dumps(year_rows, sort_keys=True)
        qb.ObjectStore.Save(year_key, payload)  # UNVERIFIED
        all_rows.extend(year_rows)

    combined = json.dumps(all_rows, sort_keys=True)
    qb.ObjectStore.Save(MEMBERSHIP_OBJECTSTORE_KEY, combined)  # UNVERIFIED
    return all_rows


def _latest_eligible_rows_by_symbol(membership_rows):
    by_symbol = defaultdict(list)
    for row in membership_rows:
        symbol = row.get("symbol")
        date = _parse_iso_date(row.get("date") or row.get("last_eligible_date"))
        if symbol and date is not None:
            by_symbol[symbol].append((date, row))
    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda pair: pair[0])
    return by_symbol


def _last_eligible_before_delist(index, symbol, delist_date):
    rows = index.get(symbol, [])
    best = None
    best_date = None
    for row_date, row in rows:
        if row_date <= delist_date and (delist_date - row_date).days <= ELIGIBLE_LOOKBACK_DAYS:
            best = row
            best_date = row_date
    return best_date, best


def _final_csv_rows(membership_rows, delisting_dates_by_symbol):
    index = _latest_eligible_rows_by_symbol(membership_rows)
    output = []
    for symbol, delist_date in sorted(delisting_dates_by_symbol.items()):
        if isinstance(delist_date, str):
            delist_date = _parse_iso_date(delist_date)
        if delist_date is None:
            continue
        last_date, row = _last_eligible_before_delist(index, symbol, delist_date)
        if row is None:
            continue
        output.append(
            {
                "symbol": symbol,
                "company_name": row.get("company_name", ""),
                "last_eligible_date": _iso_date(last_date),
                "delist_date": _iso_date(delist_date),
                "momentum_decile": row.get("momentum_decile", ""),
                "value_bucket": row.get("value_bucket", ""),
                "negative_book_flag": str(bool(row.get("negative_book_flag", False))).lower(),
                "market_cap": row.get("market_cap", ""),
            }
        )
    return output


def _rows_to_csv(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FINAL_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in FINAL_CSV_COLUMNS})
    return output.getvalue()


class QCB3DelistingDiagnostic(QCAlgorithm):
    """
    Backtest half: collect delisting dates and emit the final non-alpha CSV.

    No orders are submitted. Securities are subscribed only so delisting events
    can fire when map-file enumeration is unavailable from the algorithm.
    """

    def Initialize(self):
        self.SetStartDate(START_DATE.year, START_DATE.month, START_DATE.day)
        self.SetEndDate(END_DATE.year, END_DATE.month, END_DATE.day)
        self.SetCash(100000)

        self._membership_rows = self._load_membership_rows()
        self._candidate_symbols = sorted({row["symbol"] for row in self._membership_rows})
        self._delisting_dates = {}
        self._emitted = False

        map_file_dates = self._try_map_file_delisting_dates(self._candidate_symbols)
        self._delisting_dates.update(map_file_dates)

        # Fallback path: keep candidates subscribed so Slice.Delistings can fire.
        # This may undercount versus map files, but it is the best available path
        # if map-file enumeration is not reachable in the QCAlgorithm sandbox.
        for ticker in self._candidate_symbols:
            if ticker in self._delisting_dates:
                continue
            try:
                self.AddEquity(ticker, Resolution.Daily)  # UNVERIFIED for delisted tickers
            except Exception as exc:
                self.Debug("B3 subscribe skipped for {0}: {1}".format(ticker, exc))

    def OnData(self, data):
        delistings = getattr(data, "Delistings", None)
        if delistings is not None:
            try:
                iterable = delistings.items()
            except Exception:
                iterable = []
            for symbol, delisting in iterable:
                delisting_type = getattr(delisting, "Type", None)
                if delisting_type == DelistingType.Delisted:
                    self._delisting_dates.setdefault(_symbol_key(symbol), self.Time)

        if not self._emitted and self.Time.date() >= END_DATE.date():
            self._emit_final_csv()
            self._emitted = True

    def OnEndOfAlgorithm(self):
        if not self._emitted:
            self._emit_final_csv()
            self._emitted = True

    def _load_membership_rows(self):
        if not self.ObjectStore.ContainsKey(MEMBERSHIP_OBJECTSTORE_KEY):  # UNVERIFIED
            raise ValueError(
                "Missing ObjectStore key {0}. Run run_b3_quantbook_research(qb) first.".format(
                    MEMBERSHIP_OBJECTSTORE_KEY
                )
            )
        payload = self.ObjectStore.Read(MEMBERSHIP_OBJECTSTORE_KEY)  # UNVERIFIED
        rows = json.loads(payload)
        # Keep only rows with the fields needed for the final join.
        cleaned = []
        for row in rows:
            symbol = row.get("symbol")
            date = row.get("date")
            if not symbol or not date:
                continue
            cleaned.append(row)
        return cleaned

    def _try_map_file_delisting_dates(self, tickers):
        """
        Prefer map-file dates, but do not fail the diagnostic if QC exposes no
        supported map-file API in Algorithm mode.

        QC docs/examples for direct map-file enumeration vary by environment, so
        this is intentionally isolated and marked UNVERIFIED.
        """
        delisting_dates = {}
        try:
            from QuantConnect.Data.Auxiliary import MapFileResolver  # UNVERIFIED
        except Exception:
            return delisting_dates

        try:
            market = Market.USA
            resolver = MapFileResolver(self.MapFileProvider, market)  # UNVERIFIED
        except Exception as exc:
            self.Debug("B3 map-file resolver unavailable: {0}".format(exc))
            return delisting_dates

        for ticker in tickers:
            try:
                map_file = resolver.ResolveMapFile(ticker, SecurityType.Equity)  # UNVERIFIED
                rows = getattr(map_file, "Data", None) or []
                terminal_date = None
                for map_row in rows:
                    # In LEAN map files, the final mapping row is the best proxy
                    # for delisting/end-of-mapping date. Reason is not present.
                    date_value = getattr(map_row, "Date", None)
                    if date_value is not None:
                        terminal_date = date_value
                if terminal_date is not None:
                    dt = terminal_date
                    if not hasattr(dt, "date"):
                        dt = datetime.strptime(str(dt)[:8], "%Y%m%d")
                    if START_DATE <= dt <= END_DATE:
                        delisting_dates[ticker] = dt
            except Exception:
                continue
        return delisting_dates

    def _emit_final_csv(self):
        rows = _final_csv_rows(self._membership_rows, self._delisting_dates)
        csv_payload = _rows_to_csv(rows)
        self.ObjectStore.Save(FINAL_CSV_OBJECTSTORE_KEY, csv_payload)  # UNVERIFIED

        # Log rows for operator export if ObjectStore download is inconvenient.
        # This is still non-alpha: only identifiers, eligibility dates, ranks,
        # value buckets, negative-book flags, and delisting dates are emitted.
        for line in csv_payload.splitlines():
            self.Log(line)

from __future__ import annotations

from datetime import date

from advisor.data.edgar_xbrl_fixture import EdgarXbrlRecord


def select_asof(
    records: list[EdgarXbrlRecord],
    asset: str,
    concept: str,
    as_of: date,
) -> EdgarXbrlRecord | None:
    """Point-in-time selection honoring the WS3A named rules (spec lines 74-83):

    - strict-lag != PIT / no lookahead: a record is usable only once its canonical
      `available_asof` has arrived (available_asof <= as_of). available_asof already
      folds in the conservative +90d reporting lag, the filing date, and the accepted
      datetime, so this is the honest as-of gate.
    - amendments-are-separate: an amendment/restatement is its own record with its own
      (later) available_asof. Before that date the ORIGINAL is returned; a later
      restatement is never backfilled to an earlier as_of.

    Returns the latest-available record for (asset, concept), or None when nothing for
    that key is knowable yet (missing -> None, never fabricated).
    """
    eligible = [
        r for r in records
        if r.asset == asset and r.concept == concept and r.available_asof <= as_of
    ]
    if not eligible:
        return None
    # latest available wins; deterministic accession tie-break for same-day availability
    return max(eligible, key=lambda r: (r.available_asof, r.accession))

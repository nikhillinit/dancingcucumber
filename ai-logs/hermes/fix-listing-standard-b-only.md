# Hermes task — tighten source_integrity listing-standard signal to Rule 12d2-2(b) only

## Hard constraints
- Edit ONLY the two files below. Do NOT commit. Do NOT run npm or node.
- Verify with: `python -m pytest apps/quant/advisor/tests/test_source_integrity.py` — all must pass; report the count.

## Why
The frozen v2 prereg (`docs/superpowers/plans/2026-06-23-qc-edgar-diagnostic-v2-prereg.md`) §4.4c defines
the performance branch as "8-K Item 3.01 OR the Form 25 cites a **listing-standard rule paragraph**." The
listing-standard paragraph is **Rule 12d2-2(b)** only (exchange removal for failing listing standards).
**(c)** is voluntary issuer delisting and **(d)** is procedural — neither is a performance/distress event.
The current `_listing_standard_signal` over-broadly matches (b), (c), AND (d), which would mis-tag
voluntary/procedural delistings as adverse if the diagnostic is ever run. Restrict to (b).

## Change 1 — `apps/quant/advisor/source_integrity/edgar.py`
In `_listing_standard_signal`, REMOVE the `"12d2-2(c)"` and `"12d2-2(d)"` tokens. KEEP `"continued
listing"`, `"listing standard"`, `"12d2-2(b)"`. Final function must read exactly:

```python
def _listing_standard_signal(cited_rule: str) -> bool:
    text = cited_rule.lower()
    return any(
        token in text
        for token in (
            "continued listing",
            "listing standard",
            "12d2-2(b)",
        )
    )
```

## Change 2 — `apps/quant/advisor/tests/test_source_integrity.py`
Add a test (mirroring the existing `cited_rule="continued listing standard under 12d2-2(b)"` →
`ReasonClass.PERFORMANCE` test) asserting that, with NO qualifying 8-K event (events=[]):
- a delisting whose `cited_rule="voluntary withdrawal under 12d2-2(c)"` classifies as
  `ReasonClass.UNKNOWN` (NOT performance), and
- a delisting whose `cited_rule="removed under 12d2-2(d)"` classifies as `ReasonClass.UNKNOWN`.
Keep the existing (b)→PERFORMANCE assertion intact (do not remove it).

## Verify
`python -m pytest apps/quant/advisor/tests/test_source_integrity.py` → all pass; report count.

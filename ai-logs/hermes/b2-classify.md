# Hermes task B2 — EDGAR delisting-reason classifier (deterministic + pytest)

## Hard constraints
- Do NOT run `npm` or `node`. Do NOT commit. You CAN run `python -m pytest` on your own test file
  and run the classifier on the local CSVs (no network needed — inputs are already on disk).
- Write exactly TWO files: `ai-logs/hermes/runs/edgar_b2_classify.py` (module + `main()`),
  `ai-logs/hermes/runs/test_edgar_b2_classify.py` (pytest). Touch nothing else.
- Then RUN: `python -m pytest ai-logs/hermes/runs/test_edgar_b2_classify.py` (must pass), AND run
  `python ai-logs/hermes/runs/edgar_b2_classify.py` to produce the output CSV + print the class
  distribution. Report the printed counts in your final message.

## Goal
Classify each EDGAR delisting event by reason, implementing the FROZEN prereg
`docs/superpowers/plans/2026-06-23-qc-edgar-diagnostic-v2-prereg.md` §4.4c (read it). Output one row
per delisting event → `ai-logs/hermes/runs/edgar_delisting_reasons.csv`.

## Inputs (already on disk; latin-1/utf-8 safe)
- `ai-logs/hermes/runs/edgar_delistings_2015_2023.csv` — the Form-25 delisting set.
  Columns: `cik, company, form, date, accession_path` (form ∈ {25, 25-NSE}; date = YYYY-MM-DD).
- `ai-logs/hermes/runs/edgar_b2_filings.csv` — per-issuer relevant filings.
  Columns: `cik, company, form, filing_date, report_date, items, accession`
  (`items` = comma-joined 8-K item numbers like `"1.03,3.01"`; form ∈ {25,25-NSE,8-K,15*,DEFM14A,PREM14A,DEFA14A}).

## Procedure
1. **Build delisting events from the Form-25 set.** Group rows by `cik`. Within a CIK, collapse Form-25
   / 25-NSE filings whose dates are within **30 days** of each other into ONE delisting event; the
   event `delist_date` = the **earliest** date in that cluster. (25 and 25-NSE usually both fire for one
   delisting.) Keep `n_form25` = how many Form-25 filings backed the event. Most CIKs = 1 event.
2. **Classify each event** using that CIK's filings from `edgar_b2_filings.csv`, precedence
   **bankruptcy > acquisition > performance > unknown**:
   - **bankruptcy** (ADVERSE): any 8-K with item `1.03` whose `filing_date` is within **±90 days** of
     `delist_date`.
   - else **acquisition** (non-adverse): any 8-K with item `2.01` within ±90 days, OR any `DEFM14A`
     within the **prior 365 days** (delist_date−365 .. delist_date).
   - else **performance** (ADVERSE): any 8-K with item `3.01` within ±90 days.
   - else **unknown**.
   - Item matching: split `items` on comma, strip; an item "1.03" matches exactly (do NOT substring-match
     "1.03" inside "11.03" — compare token equality).
3. **adverse** = class ∈ {bankruptcy, performance}.

## Output CSV `edgar_delisting_reasons.csv`
Columns: `cik, company, delist_date, n_form25, edgar_class, adverse` (adverse = true/false lowercase).
Print to stdout: total events, and the count + % by `edgar_class` (bankruptcy/acquisition/performance/
unknown), and total adverse.

## Tests (pytest, inline fixtures — no file I/O in tests; test the pure functions)
Cover at least: precedence (1.03 beats 2.01 beats 3.01); the ±90-day window excludes an 8-K 100 days
away; DEFM14A-in-prior-365 → acquisition; exact item-token matching ("11.03" does NOT count as "1.03");
Form-25 clustering within 30 days collapses to one event with the earliest date; unknown when no
qualifying filing. Keep the classifier logic in small pure functions so tests don't need the CSVs.

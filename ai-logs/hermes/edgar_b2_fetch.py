"""B2 fetch — per-issuer EDGAR filings for delisting-reason classification.
Direct keyless fetch (the network step Codex can't do). For each distinct CIK in the B1 set,
pull submissions JSON and keep the rows relevant to delisting classification:
  Form 25 / 25-NSE (the delisting notice), 8-K (with its `items`), 15* (deregistration), merger proxies.
Delisted firms stop filing -> their delisting-era filings sit in 'recent', so 'recent' suffices.
Output: ai-logs/hermes/runs/edgar_b2_filings.csv  (cik,company,form,filing_date,report_date,items,accession)
This CSV is the INPUT to the B2 classifier (dispatched to Hermes).
"""
import urllib.request
import json
import csv
import time

UA = "AIHedgeFund-research nikhil@narralytics.ai"
B1 = "ai-logs/hermes/runs/edgar_delistings_2015_2023.csv"
OUT = "ai-logs/hermes/runs/edgar_b2_filings.csv"
KEEP = {"25", "25-NSE", "8-K", "15", "15-12B", "15-12G", "15F-12B", "15F-12G",
        "DEFM14A", "PREM14A", "DEFA14A"}

ciks = []
seen = set()
with open(B1, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        c = row["cik"].strip()
        if c and c not in seen:
            seen.add(c); ciks.append(c)
print(f"distinct CIKs to fetch: {len(ciks)}")

def submissions(cik):
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

rows = []
fail = 0
for n, cik in enumerate(ciks, 1):
    try:
        d = submissions(cik)
    except Exception:
        fail += 1
        continue
    name = d.get("name", "")
    rec = d.get("filings", {}).get("recent", {})
    forms = rec.get("form", [])
    fdates = rec.get("filingDate", [])
    rdates = rec.get("reportDate", [])
    items = rec.get("items", [])
    accs = rec.get("accessionNumber", [])
    for i, form in enumerate(forms):
        if form in KEEP:
            it = items[i] if i < len(items) else ""
            if form == "8-K" and not it:
                continue  # 8-K with no items = nothing to classify on
            rows.append((cik, name, form, fdates[i] if i < len(fdates) else "",
                         rdates[i] if i < len(rdates) else "", it,
                         accs[i] if i < len(accs) else ""))
    if n % 500 == 0:
        print(f"  {n}/{len(ciks)} fetched, rows={len(rows)}, fails={fail}")
    time.sleep(0.12)  # ~8 req/s, under SEC's 10/s

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cik", "company", "form", "filing_date", "report_date", "items", "accession"])
    w.writerows(rows)
print(f"\nDONE: {len(rows)} filing rows from {len(ciks)-fail} CIKs ({fail} fetch failures) -> {OUT}")

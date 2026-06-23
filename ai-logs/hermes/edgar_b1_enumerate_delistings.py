"""B1 — EDGAR delisting enumeration (v2 diagnostic backbone).
Direct keyless network fetch (in-lane). Pulls master.idx (pipe-delimited) for every
quarter 2015Q1..2023Q4, keeps Form Type in {25, 25-NSE}, writes one CSV.
SEC: descriptive User-Agent required; <=10 req/s (36 sequential reqs is fine).
Output: ai-logs/hermes/runs/edgar_delistings_2015_2023.csv  (cik,company,form,date,accession_path)
"""
import urllib.request
import csv
import os

UA = "AIHedgeFund-research nikhil@narralytics.ai"
OUT = "ai-logs/hermes/runs/edgar_delistings_2015_2023.csv"
FORMS = {"25", "25-NSE"}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
rows = []
per_qtr = {}
for year in range(2015, 2024):
    for q in range(1, 5):
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read().decode("latin-1")
        except Exception as e:  # noqa
            print(f"{year}Q{q}: FETCH ERROR {e}")
            continue
        cnt = 0
        for ln in data.splitlines():
            parts = ln.split("|")
            if len(parts) != 5:
                continue
            cik, company, form, date, path = parts
            if form.strip() in FORMS:
                rows.append((cik.strip(), company.strip(), form.strip(), date.strip(), path.strip()))
                cnt += 1
        per_qtr[f"{year}Q{q}"] = cnt
        print(f"{year}Q{q}: {cnt} delisting filings")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cik", "company", "form", "date", "accession_path"])
    w.writerows(rows)

n25 = sum(1 for r in rows if r[2] == "25")
n25nse = sum(1 for r in rows if r[2] == "25-NSE")
print(f"\nTOTAL 2015-2023: {len(rows)} delisting filings  (Form-25={n25}, Form-25-NSE={n25nse})")
print(f"distinct CIKs: {len({r[0] for r in rows})}")
print(f"written: {OUT}")

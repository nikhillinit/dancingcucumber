"""Probe: enumerate Form 25 / 25-NSE filings from EDGAR full-index for ONE quarter.
Direct keyless network fetch (data-gathering, in-lane). Validates the EDGAR-first
backbone for the v2 diagnostic before scaling to all of 2015-2023.
SEC requires a descriptive User-Agent. Rate limit <=10 req/s (we do 1).
"""
import urllib.request
import io

UA = "AIHedgeFund-research nikhil@narralytics.ai"
URL = "https://www.sec.gov/Archives/edgar/full-index/2017/QTR1/form.idx"

req = urllib.request.Request(URL, headers={"User-Agent": UA})
with urllib.request.urlopen(req, timeout=60) as r:
    raw = r.read().decode("latin-1")

lines = raw.splitlines()
# form.idx is fixed-width, sorted by Form Type. Header then rows:
# Form Type | Company Name | CIK | Date Filed | File Name
form25 = []
for ln in lines:
    ft = ln[:12].strip()
    if ft in ("25", "25-NSE"):
        company = ln[12:74].strip()
        cik = ln[74:86].strip()
        date = ln[86:98].strip()
        path = ln[98:].strip()
        form25.append((ft, company, cik, date, path))

n25 = sum(1 for x in form25 if x[0] == "25")
n25nse = sum(1 for x in form25 if x[0] == "25-NSE")
print(f"2017 QTR1: total index lines={len(lines)}  Form-25={n25}  Form-25-NSE={n25nse}")
print("--- first 8 Form-25 rows (form | company | cik | date | path) ---")
for x in form25[:8]:
    print(f"{x[0]:7} | {x[1][:38]:38} | {x[2]:>10} | {x[3]} | {x[4]}")

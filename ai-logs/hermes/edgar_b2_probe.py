"""B2 probe: confirm the EDGAR submissions JSON carries 8-K `items` (and form/date).
If present, B2 = one fetch per issuer (read items directly) instead of parsing each 8-K body.
Direct keyless fetch. Probes a couple of CIKs from the B1 set.
"""
import urllib.request
import json

UA = "AIHedgeFund-research nikhil@narralytics.ai"

def submissions(cik):
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

for cik in (320193, 1494448):  # Apple (many 8-Ks); Emergent Capital (a 2017 Form-25 filer)
    d = submissions(cik)
    rec = d["filings"]["recent"]
    print(f"\nCIK {cik}  name={d.get('name')}")
    print("  recent keys:", list(rec.keys()))
    print("  has 'items' array:", "items" in rec, "  has additional 'files':", bool(d['filings'].get('files')))
    # show a few 8-K rows with their items
    shown = 0
    for i, form in enumerate(rec.get("form", [])):
        if form == "8-K" and rec.get("items", [""])[i]:
            print(f"    8-K {rec['filingDate'][i]}  items={rec['items'][i]!r}")
            shown += 1
            if shown >= 4:
                break
    # show whether 25 / 15 / DEFM14A appear
    forms_present = sorted({f for f in rec.get("form", []) if f in ("25", "25-NSE", "15", "15-12B", "15-12G", "DEFM14A", "8-K")})
    print("  relevant forms present in 'recent':", forms_present)

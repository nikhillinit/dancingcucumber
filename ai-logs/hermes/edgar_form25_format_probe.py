"""Probe the Form 25 document format to find the Rule 12d2-2 paragraph.
Within-freeze completion of prereg §4.4c ("8-K 3.01 OR Form 25 cites a listing-standard
rule paragraph"). We need to confirm the cited paragraph (esp. 12d2-2(b) = exchange/
listing-standard removal -> performance) is machine-parseable. Direct keyless fetch.
"""
import urllib.request
import csv
import re

UA = "AIHedgeFund-research nikhil@narralytics.ai"

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("latin-1")

# pick a few Form 25 / 25-NSE accessions from B1
b1 = list(csv.DictReader(open("ai-logs/hermes/runs/edgar_delistings_2015_2023.csv", encoding="utf-8")))
# Harman (acquired, CIK 800459) + first two distinct others
picks = []
seen = set()
for row in b1:
    if row["cik"] in ("800459",) or (len(picks) < 4 and row["cik"] not in seen):
        picks.append(row); seen.add(row["cik"])
    if len(picks) >= 4:
        break

for row in picks:
    url = "https://www.sec.gov/Archives/" + row["accession_path"]
    print(f"\n===== {row['form']} | {row['company'][:40]} | CIK {row['cik']} | {row['date']} =====")
    print("URL:", url)
    try:
        txt = fetch(url)
    except Exception as e:
        print("  fetch error:", e); continue
    # Look for the structured XML rule block + any 12d2-2 paragraph mentions
    for tag in ("CrossReference", "rule12d2_2", "12d2-2", "Rule12d2", "<ruleProvision",
                "ParagraphLetter", "<rule", "paragraph"):
        idx = txt.lower().find(tag.lower())
        if idx != -1:
            print(f"  [{tag}] @{idx}: ...{txt[idx:idx+160].strip()[:160]}...")
    # crude: show any (a)/(b)/(c) paragraph-letter context near '12d2'
    for m in re.finditer(r"12d2[-_ ]?2", txt):
        s = max(0, m.start() - 40); e = min(len(txt), m.end() + 120)
        snippet = re.sub(r"\s+", " ", txt[s:e])
        print("   ~12d2-2 ctx:", snippet[:170])
        break

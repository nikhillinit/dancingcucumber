"""Within-freeze completion of prereg §4.4c: recover performance delistings whose
Form 25 cites Rule 12d2-2(b) (exchange/listing-standard removal). MEASUREMENT to
estimate the EDGAR-side classification CEILING on the operating-company proxy.
Maps ONLY 12d2-2(b) -> performance (advisor guidance); (a)/(c)/(d) stay unknown.
Direct keyless fetch over the bounded unknown-8-K-filer set.
"""
import urllib.request
import csv
import re
import collections
import time

UA = "AIHedgeFund-research nikhil@narralytics.ai"

reasons = list(csv.DictReader(open("ai-logs/hermes/runs/edgar_delisting_reasons.csv", encoding="utf-8")))
b1 = list(csv.DictReader(open("ai-logs/hermes/runs/edgar_delistings_2015_2023.csv", encoding="utf-8")))

# 8-K filer set (operating-company proxy)
filings_forms = collections.defaultdict(set)
for x in csv.DictReader(open("ai-logs/hermes/runs/edgar_b2_filings.csv", encoding="utf-8")):
    filings_forms[x["cik"]].add(x["form"])
def has8k(cik):
    return "8-K" in filings_forms.get(cik, set())

# Form 25 accessions by CIK (prefer 25-NSE = exchange-filed, carries the (b) listing-standard cite)
acc_by_cik = collections.defaultdict(list)
for r in b1:
    acc_by_cik[r["cik"]].append((r["form"], r["date"], r["accession_path"]))
for c in acc_by_cik:
    acc_by_cik[c].sort(key=lambda t: (t[0] != "25-NSE", t[1]))  # 25-NSE first

unknown_8k = [x for x in reasons if x["edgar_class"] == "unknown" and has8k(x["cik"])]
op_total = sum(1 for x in reasons if has8k(x["cik"]))
op_mappable_before = op_total - sum(1 for x in reasons if x["edgar_class"] == "unknown" and has8k(x["cik"]))
print(f"operating-company-proxy events: {op_total}  mappable BEFORE: {op_mappable_before} ({100*op_mappable_before/op_total:.1f}%)")
print(f"unknown-8K-filer events to probe: {len(unknown_8k)}")

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("latin-1")

RULE = re.compile(r"12d2[-_]?2\(([a-d])\)")
para_counts = collections.Counter()
recovered_perf = 0
fail = 0
for n, ev in enumerate(unknown_8k, 1):
    accs = acc_by_cik.get(ev["cik"], [])
    if not accs:
        para_counts["no_form25_row"] += 1
        continue
    url = "https://www.sec.gov/Archives/" + accs[0][2]
    try:
        txt = fetch(url)
    except Exception:
        fail += 1
        continue
    letters = set(RULE.findall(txt))
    if not letters:
        para_counts["no_paragraph"] += 1
    for L in letters:
        para_counts[L] += 1
    if "b" in letters:           # 12d2-2(b) = listing-standard / exchange removal -> performance
        recovered_perf += 1
    if n % 300 == 0:
        print(f"  {n}/{len(unknown_8k)} probed, recovered(b)={recovered_perf}, fails={fail}")
    time.sleep(0.1)

mappable_after = op_mappable_before + recovered_perf
print(f"\nparagraph distribution among unknown-8K-filers: {dict(para_counts)}")
print(f"recovered as performance via 12d2-2(b): {recovered_perf}")
print(f"\nCEILING on operating-company proxy:")
print(f"  BEFORE: {100*op_mappable_before/op_total:.1f}%")
print(f"  AFTER (in-spec (b) recovery): {100*mappable_after/op_total:.1f}%   ({mappable_after}/{op_total})")
print(f"  remaining unknown-8K-filers: {len(unknown_8k)-recovered_perf-fail}  fails={fail}")
print("\nNOTE: this is the CLASSIFICATION ceiling. Combined mappability = this x P(QC<->EDGAR join match) <= this.")

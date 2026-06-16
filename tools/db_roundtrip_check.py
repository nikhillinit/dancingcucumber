"""Operator round-trip check for the SignalBundle TimescaleDB checkpoint.

Not part of the pytest gate. Run from the repo root against a live DB:
    python tools/db_roundtrip_check.py
Expect: "saved 2 rows; table count = 2" (idempotent — rerun yields the same).
"""
import sys
sys.path.insert(0, "apps/quant")
from datetime import date

import psycopg2

from advisor.persistence.checkpoint import SCHEMA_SQL, save_bundle
from advisor.schemas import Direction, FamilySignal, SignalBundle

DSN = "postgresql://aihf:aihf@localhost:5432/aihf"

conn = psycopg2.connect(DSN)
with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")  # image preloads it
    cur.execute(SCHEMA_SQL)
conn.commit()

bundle = SignalBundle(
    ticker="AAPL",
    as_of=date(2024, 5, 1),
    signals=[
        FamilySignal(family="trend", direction=Direction.BULLISH, confidence=70.0, as_of=date(2024, 5, 1)),
        FamilySignal(family="macro", direction=Direction.BEARISH, confidence=60.0, as_of=date(2024, 5, 1)),
    ],
)
n = save_bundle(conn, bundle)
with conn.cursor() as cur:
    cur.execute("SELECT count(*) FROM signal_bundle")
    total = cur.fetchone()[0]
print(f"saved {n} rows; table count = {total}")
conn.close()

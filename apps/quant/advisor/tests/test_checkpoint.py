from datetime import date

from advisor.persistence.checkpoint import SCHEMA_SQL, save_bundle
from advisor.schemas import Direction, FamilySignal, SignalBundle


class _FakeCursor:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def executemany(self, sql, rows):
        self.calls.append((sql, list(rows)))


class _FakeConn:
    def __init__(self):
        self.cur = _FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True


def _bundle():
    sigs = [
        FamilySignal(family="trend", direction=Direction.BULLISH, confidence=70.0, as_of=date(2024, 5, 1)),
        FamilySignal(family="macro", direction=Direction.BEARISH, confidence=60.0, as_of=date(2024, 5, 1)),
    ]
    return SignalBundle(ticker="AAPL", as_of=date(2024, 5, 1), signals=sigs)


def test_schema_declares_hypertable():
    assert "create_hypertable" in SCHEMA_SQL
    assert "signal_bundle" in SCHEMA_SQL


def test_save_bundle_is_parameterized_and_commits():
    conn = _FakeConn()
    n = save_bundle(conn, _bundle())
    assert n == 2
    assert conn.committed is True
    sql, rows = conn.cur.calls[0]
    assert "%s" in sql                 # parameterized -- no string interpolation
    assert "AAPL" not in sql           # ticker must be a bound param, not in the SQL text
    assert len(rows) == 2
    assert rows[0][1] == "AAPL"        # ticker bound as a value


def test_save_bundle_uses_upsert():
    conn = _FakeConn()
    save_bundle(conn, _bundle())
    sql, _ = conn.cur.calls[0]
    assert "ON CONFLICT (as_of, ticker, family) DO UPDATE SET" in sql

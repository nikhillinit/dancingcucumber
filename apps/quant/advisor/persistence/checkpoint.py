from __future__ import annotations

from advisor.schemas import SignalBundle

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_bundle (
    as_of        date              NOT NULL,
    ticker       text              NOT NULL,
    family       text              NOT NULL,
    direction    text              NOT NULL,
    confidence   double precision  NOT NULL,
    skill_weight double precision  NOT NULL,
    reasoning    text              NOT NULL DEFAULT '',
    PRIMARY KEY (as_of, ticker, family)
);
SELECT create_hypertable('signal_bundle', 'as_of', if_not_exists => TRUE);
"""

_INSERT = (
    "INSERT INTO signal_bundle "
    "(as_of, ticker, family, direction, confidence, skill_weight, reasoning) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (as_of, ticker, family) DO UPDATE SET "
    "direction = EXCLUDED.direction, confidence = EXCLUDED.confidence, "
    "skill_weight = EXCLUDED.skill_weight, reasoning = EXCLUDED.reasoning"
)


def save_bundle(conn, bundle: SignalBundle) -> int:
    """Persist one row per FamilySignal. Parameterized -- never interpolate values into SQL."""
    rows = [
        (bundle.as_of, bundle.ticker, s.family, s.direction.value,
         s.confidence, s.skill_weight, s.reasoning)
        for s in bundle.signals
    ]
    with conn.cursor() as cur:
        cur.executemany(_INSERT, rows)
    conn.commit()
    return len(rows)

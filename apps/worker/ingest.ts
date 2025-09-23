import { Queue, Worker } from "bullmq";
import IORedis from "ioredis";
import pg from "pg";
import { Env } from "../../packages/shared/env";
import { getBarsFromAlpaca } from "../api/alpacaAdapter";
import { getWatermark, setWatermark } from "./watermark";
import { startMetricsServer, ingestionLag, queueWaiting, jobsProcessed } from "./metricsServer";

const connection = new IORedis(Env.REDIS_URL);
const pool = new pg.Pool({ connectionString: Env.DATABASE_URL });

export const jobs = new Queue("aihf", {
  connection,
  defaultJobOptions: {
    removeOnComplete: true,
    removeOnFail: 200,
    attempts: 5,
    backoff: { type: "exponential", delay: 2000 }
  }
});

startMetricsServer();

// Schedule SPY 1m ingestion on weekdays (RTH)
(async () => {
  await jobs.add(
    "ingest-bars",
    { vendor: "alpaca", symbol: "SPY", interval: "1Min" },
    { repeat: { pattern: "*/1 9-16 * * 1-5", tz: "America/New_York" } }
  );
})();

// Helper: is bars_5m a continuous aggregate?
async function isContinuousBars5m(client: pg.PoolClient) {
  const q = `
    SELECT 1
      FROM timescaledb_information.continuous_aggregates
     WHERE view_name = 'bars_5m'
     LIMIT 1`;
  try {
    const { rows } = await client.query(q);
    return rows.length > 0;
  } catch {
    return false;
  }
}

// Periodic refresh for fallback MV (if cont. agg not present)
await jobs.add(
  "refresh-bars-5m",
  {},
  { repeat: { pattern: "*/1 * * * *" } } // every minute
);

new Worker("aihf", async job => {
  if (job.name === "refresh-bars-5m") {
    const client = await pool.connect();
    try {
      if (!(await isContinuousBars5m(client))) {
        await client.query("REFRESH MATERIALIZED VIEW CONCURRENTLY bars_5m");
        await client.query("ANALYZE bars_5m");
      }
    } finally {
      client.release();
    }
    const waiting = await jobs.getWaitingCount();
    queueWaiting.set(waiting);
    return { refreshed: true };
  }

  if (job.name !== "ingest-bars") return;
  const { vendor, symbol, interval } = job.data as { vendor: string; symbol: string; interval: "1Min"|"5Min"|"15Min"|"1Day" };

  // Resolve instrument_id & vendor symbol (fallback to raw symbol)
  const { rows: [inst] } = await pool.query(
    `SELECT i.instrument_id, COALESCE(m.vendor_symbol, $1) AS vendor_symbol
       FROM instruments i
       LEFT JOIN instrument_vendor_map m ON m.instrument_id = i.instrument_id AND m.vendor = $2
      WHERE i.symbol = $1`,
    [symbol, vendor]
  );
  if (!inst) throw new Error(`Unknown instrument ${symbol} (${vendor})`);

  const source = `${vendor}_${interval}`;
  const last = await getWatermark(inst.instrument_id, source);

  // Pull bars from vendor starting at watermark
  const bars = await getBarsFromAlpaca({
    symbol, timeframe: interval as any,
    start: last ? new Date(last).toISOString() : undefined
  } as any);

  if (!bars.length) {
    jobsProcessed.inc();
    return { symbol, interval, inserted: 0 };
  }

  // Idempotent upsert into bars_1m or bars_1d
  const table = interval === "1Day" ? "bars_1d" : "bars_1m";
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    for (const b of bars) {
      await client.query(
        `INSERT INTO ${table} (ts, instrument_id, open, high, low, close, volume, vendor)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
         ON CONFLICT (instrument_id, ts)
         DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                       close=EXCLUDED.close, volume=EXCLUDED.volume, vendor=EXCLUDED.vendor`,
        [b.ts, inst.instrument_id, b.open, b.high, b.low, b.close, b.volume ?? null, vendor]
      );
    }
    const lastTs = bars[bars.length - 1].ts as string;
    await setWatermark(inst.instrument_id, source, lastTs);
    await client.query("COMMIT");
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }

  // Update metrics: ingestion lag = now - lastTs
  const lagSec = Math.max(0, (Date.now() - new Date(bars[bars.length - 1].ts as string).getTime()) / 1000);
  ingestionLag.set({ symbol, interval, vendor }, lagSec);
  const waiting = await jobs.getWaitingCount();
  queueWaiting.set(waiting);
  jobsProcessed.inc();
  return { symbol, interval, inserted: bars.length };
}, {
  connection,
  concurrency: 4,
  limiter: { max: 8, duration: 1000 } // vendor rate protection
});

console.log("Ingestion worker up");
import Fastify from "fastify";
import { BarsReqZ, BacktestReqZ } from "../../packages/shared/types";
import { getBarsFromAlpaca } from "./alpacaAdapter";
import { enforceRisk } from "./mw/risk";
import { Env } from "../../packages/shared/env";
import { mountMetrics } from "./metrics";
import { runAndPersistBacktest } from "./backtest";

const app = Fastify({ logger: true });

app.get("/health", async () => ({ ok: true }));
mountMetrics(app);

app.get("/data/bars", async (req, rep) => {
  const parsed = BarsReqZ.safeParse(req.query);
  if (!parsed.success) return rep.code(400).send(parsed.error.format());
  const bars = await getBarsFromAlpaca(parsed.data);
  return { symbol: parsed.data.symbol, bars };
});

// Example guarded order route (DB trigger will also enforce)
app.post("/orders", { preHandler: [enforceRisk] }, async () => {
  return { ok: true, note: "order accepted (stub)" };
});

// Backtest orchestrator: calls Quant, persists to DB
app.post("/backtest/ma", async (req, rep) => {
  const parsed = BacktestReqZ.safeParse(req.body);
  if (!parsed.success) return rep.code(400).send(parsed.error.format());
  const { id, result } = await runAndPersistBacktest(parsed.data);
  return { id, result };
});

app.listen({ port: Number(Env.PORT), host: "0.0.0.0" })
  .then(() => app.log.info(`API started on ${Env.PORT}`))
  .catch(err => { app.log.error(err); process.exit(1); });
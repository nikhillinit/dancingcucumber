import Fastify from "fastify";
import client from "prom-client";
import { Env } from "../../packages/shared/env";

const app = Fastify({ logger: false });

const register = new client.Registry();
client.collectDefaultMetrics({ register });

export const ingestionLag = new client.Gauge({
  name: "ingestion_lag_seconds",
  help: "Ingestion lag per symbol/interval",
  labelNames: ["symbol","interval","vendor"],
  registers: [register],
});

export const queueWaiting = new client.Gauge({
  name: "queue_waiting_jobs",
  help: "Number of waiting jobs in the ingestion queue",
  registers: [register],
});

export const jobsProcessed = new client.Counter({
  name: "jobs_processed_total",
  help: "Total ingestion jobs processed",
  registers: [register],
});

export function startMetricsServer() {
  app.get("/metrics", async (_req, rep) => {
    rep.type("text/plain").send(await register.metrics());
  });
  app.listen({ port: Number(Env.WORKER_METRICS_PORT), host: "0.0.0.0" })
    .then(() => console.log(`[metrics] worker metrics on ${Env.WORKER_METRICS_PORT}`))
    .catch(err => { console.error(err); process.exit(1); });
}

export { register };
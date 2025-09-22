import { FastifyInstance } from "fastify";
import client from "prom-client";

const register = new client.Registry();
client.collectDefaultMetrics({ register });

export const vendorRequests = new client.Counter({
  name: "vendor_requests_total",
  help: "Total vendor API requests",
  registers: [register],
});
export const vendorErrors = new client.Counter({
  name: "vendor_errors_total",
  help: "Total vendor API errors",
  registers: [register],
});
export const vendorLatency = new client.Histogram({
  name: "vendor_request_duration_seconds",
  help: "Vendor request latency seconds",
  buckets: [0.1, 0.3, 0.5, 1, 2, 5, 10],
  registers: [register],
});

export function mountMetrics(app: FastifyInstance) {
  app.get("/metrics", async (_req, rep) => rep.type("text/plain").send(await register.metrics()));
}
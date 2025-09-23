import Bottleneck from "bottleneck";
import { vendorLatency, vendorRequests, vendorErrors } from "./metrics";

const limiter = new Bottleneck({
  minTime: 150,         // ~6-7 req/sec
  maxConcurrent: 1
});

export async function withRetry<T>(fn: () => Promise<T>, retries = 4): Promise<T> {
  let attempt = 0;
  let delay = 0.3;
  // eslint-disable-next-line no-constant-condition
  while (true) {
    vendorRequests.inc();
    const end = vendorLatency.startTimer();
    try {
      const res = await limiter.schedule(() => fn());
      end();
      return res;
    } catch (e: any) {
      end();
      vendorErrors.inc();
      if (attempt++ >= retries) throw e;
      await new Promise(r => setTimeout(r, delay * 1000));
      delay = Math.min(delay * 2, 5);
    }
  }
}
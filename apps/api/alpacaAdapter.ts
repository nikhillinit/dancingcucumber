import fetch from "cross-fetch";
import { Env } from "../../packages/shared/env";
import { BarsReq, Bar } from "../../packages/shared/types";
import { withRetry } from "./vendorClient";

const tfMap: Record<string,string> = { "1Min":"1Min", "5Min":"5Min", "15Min":"15Min", "1Day":"1Day" };

export async function getBarsFromAlpaca(req: BarsReq): Promise<Bar[]> {
  const tf = tfMap[req.timeframe];
  const url = new URL(`${Env.ALPACA_DATA_BASE_URL}/v2/stocks/bars`);
  url.searchParams.set("symbols", req.symbol);
  url.searchParams.set("timeframe", tf);
  if (req.start) url.searchParams.set("start", req.start);
  if (req.end) url.searchParams.set("end", req.end);
  url.searchParams.set("limit", "1000");

  const exec = async () => {
    const resp = await fetch(url.toString(), {
      headers: {
        "APCA-API-KEY-ID": Env.ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": Env.ALPACA_SECRET_KEY,
      }
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Alpaca error ${resp.status}: ${text}`);
    }
    const json: any = await resp.json();
    const rows = (json.bars?.[req.symbol] ?? []);
    return rows.map((r: any) => ({
      ts: r.t, symbol: req.symbol,
      open: r.o, high: r.h, low: r.l, close: r.c, volume: r.v
    })) as Bar[];
  };
  return withRetry(exec);
}
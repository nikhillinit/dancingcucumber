import { z } from "zod";

export const SymbolZ = z.string().min(1).max(12);
export const TimeframeZ = z.enum(["1Min","5Min","15Min","1Day"]);

export const BarZ = z.object({
  ts: z.union([z.string(), z.date()]),
  symbol: SymbolZ,
  open: z.number(),
  high: z.number(),
  low: z.number(),
  close: z.number(),
  volume: z.number().optional(),
});
export type Bar = z.infer<typeof BarZ>;

export const BarsReqZ = z.object({
  symbol: SymbolZ,
  timeframe: TimeframeZ.default("1Day"),
  start: z.string().optional(),
  end: z.string().optional(),
});
export type BarsReq = z.infer<typeof BarsReqZ>;

export const BacktestReqZ = z.object({
  symbol: SymbolZ,
  timeframe: TimeframeZ.default("1Day"),
  start: z.string().optional(),
  end: z.string().optional(),
  fast: z.coerce.number().default(10),
  slow: z.coerce.number().default(30),
  init_cash: z.coerce.number().default(100000),
  fees_bps: z.coerce.number().default(2),
  slip_bps: z.coerce.number().default(5),
});
export type BacktestReq = z.infer<typeof BacktestReqZ>;
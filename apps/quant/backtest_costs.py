import pandas as pd
import numpy as np
import vectorbt as vbt

def _downsample_curve(series: pd.Series, max_points: int = 250):
    n = len(series)
    if n <= max_points:
        return [(idx.isoformat(), float(val)) for idx, val in series.items()]
    step = int(np.ceil(n / max_points))
    ds = series.iloc[::step]
    if ds.index[-1] != series.index[-1]:
        ds = pd.concat([ds, series.iloc[[-1]]])
    return [(idx.isoformat(), float(val)) for idx, val in ds.items()]

def run_ma_with_costs(price: pd.Series, fast: int, slow: int, init_cash: float,
                      fees_bps: float = 2, slip_bps: float = 5):
    fast_sma = price.rolling(fast).mean()
    slow_sma = price.rolling(slow).mean()
    entries = (fast_sma > slow_sma) & (fast_sma.shift(1) <= slow_sma.shift(1))
    exits   = (fast_sma < slow_sma) & (fast_sma.shift(1) >= slow_sma.shift(1))
    fees = fees_bps / 10_000.0
    slippage = slip_bps / 10_000.0

    pf = vbt.Portfolio.from_signals(
        price, entries, exits,
        init_cash=init_cash, fees=fees, slippage=slippage
    )
    trades = pf.trades
    turnover = float(trades.size() / max(len(price), 1))
    equity_curve = pf.value()
    equity_ds = _downsample_curve(equity_curve)
    return {
        "total_return": float(pf.total_return()),
        "sharpe": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "turnover": turnover,
        "trades": int(len(trades.records)),
        "equity_curve": equity_ds
    }
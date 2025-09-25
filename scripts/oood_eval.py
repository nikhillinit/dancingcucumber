"""
Open-to-Open Daily Evaluation Framework
=======================================
Validates our AI Hedge Fund's alpha generation with proper time alignment
Zero commissions for Fidelity, optional slippage modeling
"""

import argparse
import os
import glob
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False
    print("Warning: QuantStats not installed. Tear sheets will be unavailable.")

ANNUAL_DAYS = 252

def download_open_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download opening prices for tickers"""
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")

    df = yf.download(
        tickers, start=start, end=end, interval="1d",
        auto_adjust=False, progress=False, threads=True
    )

    if len(tickers) == 1:
        opens = df[['Open']].copy()
        opens.columns = tickers
    else:
        if isinstance(df.columns, pd.MultiIndex):
            opens = df['Open'].copy()
        else:
            opens = df[['Open']].copy()

    opens = opens.dropna(how='all')
    return opens

def make_o2o_returns(opens: pd.DataFrame) -> pd.DataFrame:
    """Calculate open-to-open returns"""
    # Return from today's open to tomorrow's open
    return opens.shift(-1) / opens - 1.0

def realized_vol(returns: pd.Series) -> float:
    """Annualized volatility"""
    return returns.std(ddof=1) * np.sqrt(ANNUAL_DAYS)

def risk_match(target_ret: pd.Series, ref_ret: pd.Series) -> pd.Series:
    """Scale reference returns to match target volatility"""
    vol_t = realized_vol(target_ret)
    vol_r = realized_vol(ref_ret)
    if vol_r == 0 or np.isnan(vol_r):
        return ref_ret
    scale = vol_t / vol_r
    return ref_ret * scale

def max_drawdown(equity: pd.Series) -> tuple:
    """Calculate maximum drawdown"""
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min(), dd

def calculate_metrics(returns: pd.Series, name: str = "strategy") -> Dict:
    """Calculate comprehensive performance metrics"""
    s = returns.dropna()
    if s.empty:
        return {f"{name}_CAGR": np.nan}

    # Basic metrics
    cagr = (1 + s).prod() ** (ANNUAL_DAYS / len(s)) - 1
    vol = realized_vol(s)
    mean_return = s.mean() * ANNUAL_DAYS

    # Risk-adjusted metrics
    sharpe = mean_return / vol if vol > 0 else np.nan

    # Downside metrics
    downside = s[s < 0].std(ddof=1) * np.sqrt(ANNUAL_DAYS)
    sortino = mean_return / downside if downside > 0 else np.nan

    # Drawdown metrics
    eq = (1 + s).cumprod()
    mdd, dd_series = max_drawdown(eq)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    # Win rate
    win_rate = (s > 0).mean()

    # Best/worst periods
    best_day = s.max()
    worst_day = s.min()

    return {
        f"{name}_CAGR": cagr,
        f"{name}_Vol": vol,
        f"{name}_Sharpe": sharpe,
        f"{name}_Sortino": sortino,
        f"{name}_MaxDD": mdd,
        f"{name}_Calmar": calmar,
        f"{name}_WinRate": win_rate,
        f"{name}_BestDay": best_day,
        f"{name}_WorstDay": worst_day
    }

def information_ratio(strat: pd.Series, bench: pd.Series) -> float:
    """Calculate Information Ratio"""
    diff = (strat - bench).dropna()
    tracking_error = diff.std(ddof=1) * np.sqrt(ANNUAL_DAYS)
    excess_return = diff.mean() * ANNUAL_DAYS
    return excess_return / tracking_error if tracking_error > 0 else np.nan

def read_our_signals(signals_dir: str) -> Dict:
    """Read our AI system's recommendation CSVs"""
    pattern = os.path.join(signals_dir, "intelligence_recommendations_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        # Try alternative pattern
        pattern = os.path.join(signals_dir, "*.csv")
        files = sorted(glob.glob(pattern))

    print(f"Found {len(files)} signal files in {signals_dir}")

    daily_weights = {}

    for filepath in files:
        try:
            # Extract date from filename
            basename = os.path.basename(filepath)
            date_str = basename.replace('intelligence_recommendations_', '').replace('.csv', '')

            # Try different date formats
            for fmt in ['%Y%m%d', '%Y-%m-%d', '%Y_%m_%d']:
                try:
                    date = dt.datetime.strptime(date_str[-8:], fmt).date()
                    break
                except:
                    continue

            # Read CSV
            df = pd.read_csv(filepath)

            # Handle different column names
            if 'Symbol' in df.columns:
                symbol_col = 'Symbol'
            else:
                symbol_col = df.columns[0]

            if 'Weight' in df.columns:
                weight_col = 'Weight'
            elif 'Position_Size_Percent' in df.columns:
                weight_col = 'Position_Size_Percent'
            else:
                weight_col = df.columns[1]

            # Convert to weights dictionary
            weights = {}
            for _, row in df.iterrows():
                symbol = row[symbol_col]
                weight = float(row[weight_col])

                # Handle percentage vs decimal
                if weight > 1:
                    weight = weight / 100

                weights[symbol] = weight

            # Normalize weights to sum to 1
            total = sum(abs(w) for w in weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}

            daily_weights[date] = weights

        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")
            continue

    return daily_weights

def build_weight_matrix(dates: pd.DatetimeIndex, tickers: List[str],
                        daily_weights: Dict, rebalance_band: float = 0.0025) -> pd.DataFrame:
    """Build weight matrix with optional rebalancing band"""
    W = pd.DataFrame(index=dates, columns=tickers, dtype=float).fillna(0.0)

    prev_weights = pd.Series(0.0, index=tickers)

    for date in dates:
        # Check if we have weights for this date
        weights_dict = daily_weights.get(date.date(), None)

        if weights_dict is None:
            # Hold previous weights
            W.loc[date] = prev_weights
            continue

        # Build new weights
        curr_weights = pd.Series(0.0, index=tickers)
        for ticker in tickers:
            if ticker in weights_dict:
                curr_weights[ticker] = weights_dict[ticker]

        # Apply rebalancing band (don't trade if change < band)
        if rebalance_band > 0:
            changes = (curr_weights - prev_weights).abs()
            small_changes = changes < rebalance_band
            curr_weights[small_changes] = prev_weights[small_changes]

        # Renormalize
        if curr_weights.abs().sum() > 0:
            curr_weights = curr_weights / curr_weights.abs().sum()

        W.loc[date] = curr_weights
        prev_weights = curr_weights.copy()

    return W.fillna(method='ffill').fillna(0.0)

def validate_alpha_claims(args):
    """Main validation function"""

    print("\n" + "="*70)
    print("AI HEDGE FUND ALPHA VALIDATION")
    print("="*70)
    print(f"Period: {args.start} to {args.end}")
    print(f"Signals directory: {args.weights_dir}")
    print(f"Commission: 0% (Fidelity)")
    print(f"Slippage: {args.slippage_bps} bps")
    print("="*70)

    # Get universe from args or use default
    tickers = args.tickers.split(",")

    # Download data
    all_tickers = tickers + ["SPY", "QQQ", "IWM"]  # Include benchmarks
    opens = download_open_prices(all_tickers, args.start, args.end)

    # Calculate open-to-open returns
    o2o_returns = make_o2o_returns(opens)

    # Read our AI signals
    daily_weights_dict = read_our_signals(args.weights_dir)

    if not daily_weights_dict:
        print("\nERROR: No signal files found. Please generate signals first.")
        print("Run: python validate_alpha_claims.py --generate")
        return

    # Build weight matrix
    W = build_weight_matrix(opens.index, tickers, daily_weights_dict,
                           rebalance_band=args.band_bps/10000)

    # Calculate strategy returns
    strat_returns = (W * o2o_returns[tickers]).sum(axis=1)

    # Apply slippage if specified
    if args.slippage_bps > 0:
        turnover = W.diff().abs().sum(axis=1).fillna(0.0)
        slippage = turnover * (args.slippage_bps / 10000.0)
        strat_returns = strat_returns - slippage

    # Benchmark returns
    spy_returns = o2o_returns["SPY"]
    qqq_returns = o2o_returns["QQQ"]

    # Risk-matched SPY
    spy_risk_matched = risk_match(strat_returns, spy_returns)

    # Calculate all metrics
    metrics = {}
    metrics.update(calculate_metrics(strat_returns, "Strategy"))
    metrics.update(calculate_metrics(spy_returns, "SPY"))
    metrics.update(calculate_metrics(spy_risk_matched, "SPY_RiskMatched"))
    metrics.update(calculate_metrics(qqq_returns, "QQQ"))

    # Information ratios
    metrics["IR_vs_SPY"] = information_ratio(strat_returns, spy_returns)
    metrics["IR_vs_SPY_RiskMatched"] = information_ratio(strat_returns, spy_risk_matched)

    # Calculate alpha (excess return over SPY)
    alpha = metrics["Strategy_CAGR"] - metrics["SPY_CAGR"]
    risk_adj_alpha = metrics["Strategy_CAGR"] - metrics["SPY_RiskMatched_CAGR"]

    # Display results
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    print("\n>>> ABSOLUTE PERFORMANCE")
    print("-"*50)
    print(f"Strategy CAGR:           {metrics['Strategy_CAGR']:>8.2%}")
    print(f"Strategy Volatility:     {metrics['Strategy_Vol']:>8.2%}")
    print(f"Strategy Sharpe:         {metrics['Strategy_Sharpe']:>8.2f}")
    print(f"Strategy Sortino:        {metrics['Strategy_Sortino']:>8.2f}")
    print(f"Strategy Max Drawdown:   {metrics['Strategy_MaxDD']:>8.2%}")
    print(f"Strategy Calmar:         {metrics['Strategy_Calmar']:>8.2f}")
    print(f"Strategy Win Rate:       {metrics['Strategy_WinRate']:>8.2%}")

    print("\n>>> ALPHA GENERATION")
    print("-"*50)
    print(f"Raw Alpha (vs SPY):      {alpha:>8.2%}")
    print(f"Risk-Adj Alpha:          {risk_adj_alpha:>8.2%}")
    print(f"Information Ratio:       {metrics['IR_vs_SPY']:>8.2f}")
    print(f"IR (Risk-Matched):       {metrics['IR_vs_SPY_RiskMatched']:>8.2f}")

    print("\n>>> BENCHMARK COMPARISON")
    print("-"*50)
    print(f"SPY CAGR:                {metrics['SPY_CAGR']:>8.2%}")
    print(f"SPY Sharpe:              {metrics['SPY_Sharpe']:>8.2f}")
    print(f"QQQ CAGR:                {metrics['QQQ_CAGR']:>8.2%}")

    print("\n>>> ALPHA VALIDATION")
    print("-"*50)

    # Determine if claims are validated
    target_alpha = 0.50  # 50% minimum target

    if alpha >= target_alpha:
        print(f"✅ ALPHA CLAIM VALIDATED: {alpha:.1%} >= {target_alpha:.1%} target")
    elif alpha >= 0.30:
        print(f"⚠️  PARTIAL VALIDATION: {alpha:.1%} (Good but below 50% target)")
    else:
        print(f"❌ NEEDS OPTIMIZATION: {alpha:.1%} (Below expectations)")

    if metrics['IR_vs_SPY_RiskMatched'] >= 2.0:
        print(f"✅ RISK-ADJUSTED VALIDATION: IR {metrics['IR_vs_SPY_RiskMatched']:.2f} >= 2.0")
    else:
        print(f"⚠️  RISK-ADJUSTED: IR {metrics['IR_vs_SPY_RiskMatched']:.2f} < 2.0 target")

    # Generate tear sheet if requested
    if args.tear and HAS_QUANTSTATS:
        print("\nGenerating QuantStats tear sheet...")
        qs.extend_pandas()
        output_file = f"alpha_validation_{dt.date.today()}.html"
        qs.reports.html(strat_returns, benchmark=spy_risk_matched,
                       output=output_file, title="AI Hedge Fund Alpha Validation")
        print(f"Saved tear sheet to: {output_file}")

    return metrics, alpha

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate AI Hedge Fund Alpha Claims")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=dt.date.today().isoformat(), help="End date")
    parser.add_argument("--tickers", default="AAPL,GOOGL,MSFT,AMZN,TSLA,NVDA,META,JPM,BAC,WMT,JNJ,V",
                       help="Comma-separated tickers")
    parser.add_argument("--weights_dir", default="signals", help="Directory with signal CSVs")
    parser.add_argument("--band_bps", type=int, default=25, help="Rebalancing band in bps")
    parser.add_argument("--slippage_bps", type=int, default=0, help="Slippage in bps")
    parser.add_argument("--tear", action="store_true", help="Generate QuantStats tear sheet")

    args = parser.parse_args()

    # Run validation
    metrics, alpha = validate_alpha_claims(args)
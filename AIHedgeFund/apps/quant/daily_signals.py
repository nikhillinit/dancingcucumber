"""
Daily Trade Recommendation System
Generates trade signals before market open
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradeSignal:
    """Daily trade recommendation"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-1
    target_allocation: float  # % of portfolio
    entry_price: float
    stop_loss: float
    take_profit: float
    rationale: List[str]
    risk_score: float  # 1-10
    expected_return: float


class DailySignalGenerator:
    """Generate daily trade recommendations using multiple strategies"""

    def __init__(self,
                 symbols: List[str],
                 lookback_days: int = 252,
                 portfolio_value: float = 100000):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.portfolio_value = portfolio_value
        self.data = {}
        self.signals = []

    def fetch_market_data(self) -> pd.DataFrame:
        """Fetch historical data for all symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if not df.empty:
                    self.data[symbol] = df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        return self.data

    def calculate_technical_signals(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators and signals"""
        signals = {}

        # RSI
        df['rsi'] = ta.rsi(df['Close'], length=14)
        signals['rsi_oversold'] = df['rsi'].iloc[-1] < 30
        signals['rsi_overbought'] = df['rsi'].iloc[-1] > 70

        # MACD
        macd = ta.macd(df['Close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        signals['macd_bullish'] = df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]

        # Bollinger Bands
        bb = ta.bbands(df['Close'])
        df['bb_lower'] = bb['BBL_20_2.0']
        df['bb_upper'] = bb['BBU_20_2.0']
        signals['near_bb_lower'] = df['Close'].iloc[-1] < df['bb_lower'].iloc[-1] * 1.02
        signals['near_bb_upper'] = df['Close'].iloc[-1] > df['bb_upper'].iloc[-1] * 0.98

        # Moving Averages
        df['sma_20'] = ta.sma(df['Close'], 20)
        df['sma_50'] = ta.sma(df['Close'], 50)
        df['sma_200'] = ta.sma(df['Close'], 200)
        signals['above_sma200'] = df['Close'].iloc[-1] > df['sma_200'].iloc[-1]
        signals['golden_cross'] = (df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1] and
                                   df['sma_50'].iloc[-2] <= df['sma_200'].iloc[-2])

        # Volume
        df['volume_sma'] = ta.sma(df['Volume'], 20)
        signals['high_volume'] = df['Volume'].iloc[-1] > df['volume_sma'].iloc[-1] * 1.5

        # ATR for volatility
        df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        signals['volatility'] = df['atr'].iloc[-1] / df['Close'].iloc[-1]

        return signals

    def ml_prediction(self, df: pd.DataFrame) -> Tuple[float, float]:
        """ML model to predict next day direction and confidence"""
        if len(df) < 50:
            return 0, 0.5

        # Feature engineering
        df['returns'] = df['Close'].pct_change()
        df['rsi'] = ta.rsi(df['Close'], 14)
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['close_open_ratio'] = (df['Close'] - df['Open']) / df['Open']

        # Create target (1 if next day is up, 0 if down)
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)

        # Prepare features
        features = ['returns', 'rsi', 'volume_ratio', 'high_low_ratio', 'close_open_ratio']
        df_clean = df[features + ['target']].dropna()

        if len(df_clean) < 30:
            return 0, 0.5

        X = df_clean[features].iloc[:-1]
        y = df_clean['target'].iloc[:-1]
        X_pred = df_clean[features].iloc[-1:]

        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_pred_scaled = scaler.transform(X_pred)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        # Predict
        prediction = model.predict(X_pred_scaled)[0]
        confidence = model.predict_proba(X_pred_scaled)[0].max()

        return prediction, confidence

    def portfolio_optimization(self) -> Dict[str, float]:
        """Optimize portfolio allocation using Modern Portfolio Theory"""
        if len(self.data) < 2:
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

        # Prepare price data
        prices = pd.DataFrame()
        for symbol, df in self.data.items():
            if len(df) > 30:
                prices[symbol] = df['Close']

        if prices.empty:
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

        # Calculate expected returns and risk
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        # Optimize
        try:
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe(risk_free_rate=0.02)
            weights = ef.clean_weights()

            # Discrete allocation
            da = DiscreteAllocation(weights, prices.iloc[-1], total_portfolio_value=self.portfolio_value)
            allocation, leftover = da.greedy_portfolio()

            # Convert to percentages
            total_value = sum(allocation[s] * prices[s].iloc[-1] for s in allocation)
            percentages = {s: (allocation.get(s, 0) * prices[s].iloc[-1] / self.portfolio_value)
                          for s in self.symbols if s in prices.columns}

            return percentages
        except:
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}

    def calculate_risk_metrics(self, df: pd.DataFrame) -> float:
        """Calculate risk score for a symbol (1-10)"""
        risk_factors = []

        # Volatility
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        risk_factors.append(min(volatility * 20, 10))  # Scale to 1-10

        # Drawdown
        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative / cumulative.cummax() - 1).min()
        risk_factors.append(min(abs(drawdown) * 20, 10))

        # Beta (if SPY data available)
        try:
            spy = yf.download('SPY', period='6mo')['Close'].pct_change()
            if len(spy) > 0:
                correlation = returns.corr(spy)
                beta = correlation * (returns.std() / spy.std())
                risk_factors.append(min(abs(beta) * 3, 10))
        except:
            pass

        return np.mean(risk_factors) if risk_factors else 5.0

    def generate_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Generate trade signal for a single symbol"""
        if symbol not in self.data:
            return None

        df = self.data[symbol]
        if len(df) < 30:
            return None

        # Get signals
        tech_signals = self.calculate_technical_signals(df)
        ml_direction, ml_confidence = self.ml_prediction(df)
        risk_score = self.calculate_risk_metrics(df)

        # Decision logic
        current_price = df['Close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02
        rationale = []
        score = 0

        # Technical signals scoring
        if tech_signals['rsi_oversold'] and not tech_signals['rsi_overbought']:
            score += 2
            rationale.append("RSI oversold")
        if tech_signals['macd_bullish']:
            score += 1
            rationale.append("MACD bullish crossover")
        if tech_signals['near_bb_lower']:
            score += 1
            rationale.append("Near Bollinger Band lower")
        if tech_signals['above_sma200']:
            score += 1
            rationale.append("Above 200-day SMA (uptrend)")
        if tech_signals['golden_cross']:
            score += 3
            rationale.append("Golden cross detected")
        if tech_signals['high_volume']:
            score += 1
            rationale.append("High volume interest")

        # ML signal
        if ml_direction == 1 and ml_confidence > 0.6:
            score += 2
            rationale.append(f"ML bullish ({ml_confidence:.0%} confidence)")

        # Determine action
        if score >= 4:
            action = 'BUY'
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
        elif score <= -2:
            action = 'SELL'
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
        else:
            action = 'HOLD'
            stop_loss = current_price - atr
            take_profit = current_price + atr

        # Calculate expected return
        if action == 'BUY':
            expected_return = ((take_profit - current_price) / current_price) * 0.6  # 60% win probability
        elif action == 'SELL':
            expected_return = ((current_price - take_profit) / current_price) * 0.6
        else:
            expected_return = 0

        return TradeSignal(
            symbol=symbol,
            action=action,
            confidence=min(score / 10, 1.0),
            target_allocation=0,  # Will be set by portfolio optimization
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rationale=rationale,
            risk_score=risk_score,
            expected_return=expected_return
        )

    def generate_daily_recommendations(self) -> List[TradeSignal]:
        """Generate complete daily trading recommendations"""
        print(f"Fetching market data for {len(self.symbols)} symbols...")
        self.fetch_market_data()

        print("Generating individual signals...")
        signals = []
        for symbol in self.symbols:
            signal = self.generate_signal(symbol)
            if signal:
                signals.append(signal)

        # Filter to top opportunities
        buy_signals = [s for s in signals if s.action == 'BUY']
        buy_signals.sort(key=lambda x: (x.confidence, -x.risk_score), reverse=True)
        top_signals = buy_signals[:10]  # Top 10 opportunities

        if top_signals:
            # Optimize portfolio allocation
            print("Optimizing portfolio allocation...")
            allocations = self.portfolio_optimization()

            # Update allocations in signals
            for signal in top_signals:
                signal.target_allocation = allocations.get(signal.symbol, 0)

        self.signals = top_signals
        return top_signals

    def format_recommendations(self) -> str:
        """Format recommendations for display"""
        if not self.signals:
            return "No strong trade signals for today."

        output = []
        output.append("=" * 80)
        output.append(f"DAILY TRADE RECOMMENDATIONS - {datetime.now().strftime('%Y-%m-%d')}")
        output.append(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        output.append("=" * 80)

        for i, signal in enumerate(self.signals, 1):
            output.append(f"\n{i}. {signal.symbol}")
            output.append("-" * 40)
            output.append(f"Action: {signal.action}")
            output.append(f"Confidence: {signal.confidence:.1%}")
            output.append(f"Entry Price: ${signal.entry_price:.2f}")
            output.append(f"Stop Loss: ${signal.stop_loss:.2f} ({(signal.stop_loss/signal.entry_price - 1)*100:.1f}%)")
            output.append(f"Take Profit: ${signal.take_profit:.2f} ({(signal.take_profit/signal.entry_price - 1)*100:+.1f}%)")
            output.append(f"Risk Score: {signal.risk_score:.1f}/10")
            output.append(f"Expected Return: {signal.expected_return:.1%}")

            if signal.target_allocation > 0:
                position_size = self.portfolio_value * signal.target_allocation
                shares = int(position_size / signal.entry_price)
                output.append(f"Suggested Position: {shares} shares (${position_size:,.0f} / {signal.target_allocation:.1%} of portfolio)")

            output.append(f"Rationale: {', '.join(signal.rationale)}")

        output.append("\n" + "=" * 80)
        output.append("RISK DISCLAIMER: These are algorithmic suggestions only. Always do your own research.")

        return "\n".join(output)


# Preset watchlists
WATCHLISTS = {
    'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    'sp500_leaders': ['SPY', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG'],
    'high_volatility': ['TSLA', 'NVDA', 'AMD', 'COIN', 'RIVN', 'PLTR', 'SOFI', 'RIOT', 'MARA'],
    'dividend_aristocrats': ['JNJ', 'KO', 'PEP', 'PG', 'MMM', 'CL', 'CVX', 'XOM', 'MCD', 'WMT'],
    'ark_favorites': ['TSLA', 'ROKU', 'SQ', 'TDOC', 'COIN', 'PATH', 'DKNG', 'U', 'TWLO', 'SHOP']
}


def generate_morning_brief(watchlist: str = 'sp500_leaders', portfolio_value: float = 100000):
    """Generate complete morning trading brief"""
    symbols = WATCHLISTS.get(watchlist, WATCHLISTS['sp500_leaders'])
    generator = DailySignalGenerator(symbols, portfolio_value=portfolio_value)
    generator.generate_daily_recommendations()
    return generator.format_recommendations()


if __name__ == "__main__":
    # Example usage
    brief = generate_morning_brief('mega_tech', 100000)
    print(brief)
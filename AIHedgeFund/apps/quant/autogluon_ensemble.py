"""
AutoGluon Ensemble for Financial Forecasting
===========================================
Automatic machine learning with ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import yfinance as yf
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

logger = logging.getLogger(__name__)


@dataclass
class AutoGluonConfig:
    """Configuration for AutoGluon ensemble"""
    prediction_length: int = 24
    eval_metric: str = "MAPE"  # Mean Absolute Percentage Error
    preset: str = "best_quality"  # Options: 'fast', 'medium', 'high_quality', 'best_quality'
    time_limit: int = 600  # Training time limit in seconds
    freq: str = "H"  # Hourly frequency
    quantiles: List[float] = None

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]


class AutoGluonEnsemble:
    """AutoGluon ensemble predictor for time series"""

    def __init__(self, config: Optional[AutoGluonConfig] = None):
        self.config = config or AutoGluonConfig()
        self.predictor = None
        self.feature_importance = None

    def prepare_time_series_data(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        item_id_col: Optional[str] = None
    ) -> TimeSeriesDataFrame:
        """Prepare data in AutoGluon format"""

        # Create TimeSeriesDataFrame
        if item_id_col:
            ts_df = TimeSeriesDataFrame.from_data_frame(
                df,
                id_column=item_id_col,
                timestamp_column="timestamp"
            )
        else:
            # Single time series
            df["item_id"] = "stock"
            ts_df = TimeSeriesDataFrame.from_data_frame(
                df,
                id_column="item_id",
                timestamp_column="timestamp"
            )

        return ts_df

    def train(
        self,
        train_data: TimeSeriesDataFrame,
        validation_data: Optional[TimeSeriesDataFrame] = None
    ):
        """Train AutoGluon ensemble"""

        logger.info(f"Training AutoGluon ensemble with preset: {self.config.preset}")

        # Initialize predictor
        self.predictor = TimeSeriesPredictor(
            prediction_length=self.config.prediction_length,
            freq=self.config.freq,
            eval_metric=self.config.eval_metric,
            quantile_levels=self.config.quantiles
        )

        # Fit the model
        self.predictor.fit(
            train_data,
            presets=self.config.preset,
            time_limit=self.config.time_limit,
            tuning_data=validation_data
        )

        # Get leaderboard
        leaderboard = self.predictor.leaderboard()
        logger.info(f"Model leaderboard:\n{leaderboard}")

        return self.predictor

    def predict(
        self,
        data: TimeSeriesDataFrame,
        quantiles: bool = True
    ) -> pd.DataFrame:
        """Generate predictions with confidence intervals"""

        if self.predictor is None:
            raise ValueError("Model not trained yet")

        # Make predictions
        predictions = self.predictor.predict(data, quantile_levels=self.config.quantiles if quantiles else None)

        return predictions

    def evaluate(self, test_data: TimeSeriesDataFrame) -> Dict:
        """Evaluate model performance"""

        if self.predictor is None:
            raise ValueError("Model not trained yet")

        # Get predictions
        predictions = self.predict(test_data[:-self.config.prediction_length])

        # Calculate metrics
        metrics = self.predictor.evaluate(test_data)

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from best model"""

        if self.predictor is None:
            raise ValueError("Model not trained yet")

        # Get feature importance if available
        try:
            importance = self.predictor.feature_importance()
            return importance
        except:
            logger.warning("Feature importance not available for current models")
            return pd.DataFrame()


class MultiStockEnsemble:
    """Ensemble predictor for multiple stocks"""

    def __init__(self, config: Optional[AutoGluonConfig] = None):
        self.config = config or AutoGluonConfig()
        self.ensemble = AutoGluonEnsemble(config)

    def prepare_multi_stock_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> TimeSeriesDataFrame:
        """Prepare data for multiple stocks"""

        all_data = []

        for symbol in symbols:
            # Download data
            stock_data = yf.download(symbol, start=start_date, end=end_date)

            # Prepare dataframe
            df = pd.DataFrame({
                'timestamp': stock_data.index,
                'item_id': symbol,
                'close': stock_data['Close'].values,
                'volume': stock_data['Volume'].values,
                'high': stock_data['High'].values,
                'low': stock_data['Low'].values,
                'open': stock_data['Open'].values
            })

            # Add technical indicators
            df = self._add_features(df)

            all_data.append(df)

        # Combine all stocks
        combined_df = pd.concat(all_data, ignore_index=True)

        # Convert to TimeSeriesDataFrame
        ts_df = TimeSeriesDataFrame.from_data_frame(
            combined_df,
            id_column="item_id",
            timestamp_column="timestamp"
        )

        return ts_df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features"""

        # Returns
        df['returns'] = df['close'].pct_change()

        # Moving averages
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()

        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Price range
        df['price_range'] = (df['high'] - df['low']) / df['close']

        return df.dropna()

    def train_ensemble(
        self,
        symbols: List[str],
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str
    ):
        """Train ensemble on multiple stocks"""

        # Prepare training data
        train_data = self.prepare_multi_stock_data(symbols, train_start, train_end)

        # Prepare validation data
        val_data = self.prepare_multi_stock_data(symbols, val_start, val_end)

        # Train ensemble
        self.ensemble.train(train_data, val_data)

        return self.ensemble

    def predict_stocks(
        self,
        symbols: List[str],
        current_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Generate predictions for multiple stocks"""

        predictions = {}

        # Get recent data for context
        end_date = current_date
        start_date = (pd.Timestamp(current_date) - timedelta(days=100)).strftime('%Y-%m-%d')

        # Prepare data
        recent_data = self.prepare_multi_stock_data(symbols, start_date, end_date)

        # Generate predictions
        all_predictions = self.ensemble.predict(recent_data)

        # Split by symbol
        for symbol in symbols:
            symbol_preds = all_predictions[all_predictions.index.get_level_values('item_id') == symbol]
            predictions[symbol] = symbol_preds

        return predictions


class AutoGluonTradingSignals:
    """Generate trading signals from AutoGluon predictions"""

    def __init__(self):
        self.ensemble = MultiStockEnsemble()
        self.signal_threshold = 0.02  # 2% threshold for signals

    def generate_signals(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """Generate trading signals from predictions"""

        signals = {}

        for symbol, preds in predictions.items():
            # Get mean prediction and confidence intervals
            mean_pred = preds['mean']
            lower_bound = preds['0.1']
            upper_bound = preds['0.9']

            # Calculate expected return
            current_price = preds.iloc[0]['close'] if 'close' in preds.columns else 100
            expected_return = (mean_pred.iloc[-1] - current_price) / current_price

            # Generate signal based on expected return and confidence
            signal = self._determine_signal(
                expected_return,
                lower_bound.iloc[-1],
                upper_bound.iloc[-1],
                current_price
            )

            signals[symbol] = signal

        return signals

    def _determine_signal(
        self,
        expected_return: float,
        lower_bound: float,
        upper_bound: float,
        current_price: float
    ) -> Dict:
        """Determine trading signal"""

        # Calculate confidence based on prediction interval
        confidence_interval = (upper_bound - lower_bound) / current_price
        confidence = max(0, 1 - confidence_interval)

        # Determine action
        if expected_return > self.signal_threshold and lower_bound > current_price:
            action = "buy"
            strength = min(expected_return / self.signal_threshold, 2.0)
        elif expected_return < -self.signal_threshold and upper_bound < current_price:
            action = "sell"
            strength = min(abs(expected_return) / self.signal_threshold, 2.0)
        else:
            action = "hold"
            strength = 0.0

        return {
            'action': action,
            'expected_return': expected_return,
            'confidence': confidence,
            'strength': strength,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'reasoning': self._generate_reasoning(action, expected_return, confidence)
        }

    def _generate_reasoning(
        self,
        action: str,
        expected_return: float,
        confidence: float
    ) -> str:
        """Generate reasoning for signal"""

        if action == "buy":
            return f"Strong positive return expected ({expected_return:.2%}) with {confidence:.1%} confidence"
        elif action == "sell":
            return f"Negative return expected ({expected_return:.2%}) with {confidence:.1%} confidence"
        else:
            return f"Uncertain outlook ({expected_return:.2%} return), maintaining position"


class AutoGluonBacktester:
    """Backtesting for AutoGluon predictions"""

    def __init__(self):
        self.signal_generator = AutoGluonTradingSignals()

    def backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100000
    ) -> pd.DataFrame:
        """Run backtest on historical data"""

        results = []
        capital = initial_capital
        positions = {}

        # Generate dates for backtesting
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')

            # Get predictions for this date
            predictions = self._get_predictions_for_date(symbols, date_str)

            # Generate signals
            signals = self.signal_generator.generate_signals(predictions)

            # Execute trades
            for symbol, signal in signals.items():
                if signal['action'] == 'buy' and symbol not in positions:
                    # Buy signal
                    position_size = capital * 0.1 * signal['strength']
                    positions[symbol] = {
                        'size': position_size,
                        'entry_price': predictions[symbol].iloc[0]['close'],
                        'entry_date': date
                    }
                    capital -= position_size

                elif signal['action'] == 'sell' and symbol in positions:
                    # Sell signal
                    position = positions[symbol]
                    exit_price = predictions[symbol].iloc[0]['close']
                    pnl = (exit_price - position['entry_price']) / position['entry_price'] * position['size']
                    capital += position['size'] + pnl

                    results.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'pnl': pnl,
                        'return': pnl / position['size']
                    })

                    del positions[symbol]

        # Close remaining positions
        for symbol, position in positions.items():
            # Assume exit at last known price
            capital += position['size']

        # Calculate metrics
        backtest_results = pd.DataFrame(results)

        if not backtest_results.empty:
            total_return = (capital - initial_capital) / initial_capital
            win_rate = (backtest_results['pnl'] > 0).mean()
            avg_return = backtest_results['return'].mean()

            summary = {
                'total_return': total_return,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'num_trades': len(backtest_results),
                'final_capital': capital
            }
        else:
            summary = {
                'total_return': 0,
                'win_rate': 0,
                'avg_return': 0,
                'num_trades': 0,
                'final_capital': capital
            }

        return backtest_results, summary

    def _get_predictions_for_date(
        self,
        symbols: List[str],
        date: str
    ) -> Dict[str, pd.DataFrame]:
        """Get predictions for specific date (mock implementation)"""

        # In real implementation, this would get actual predictions
        predictions = {}

        for symbol in symbols:
            # Mock predictions
            predictions[symbol] = pd.DataFrame({
                'mean': [100, 101, 102],
                '0.1': [98, 99, 100],
                '0.9': [102, 103, 104],
                'close': [100]
            })

        return predictions


# Integration with existing system
class AutoGluonIntegration:
    """Integrate AutoGluon with AI Hedge Fund system"""

    def __init__(self):
        self.ensemble = AutoGluonEnsemble()
        self.multi_stock = MultiStockEnsemble()
        self.signal_generator = AutoGluonTradingSignals()

    def enhance_ai_decisions(
        self,
        ai_decisions: List[Dict],
        symbols: List[str]
    ) -> List[Dict]:
        """Enhance AI persona decisions with AutoGluon predictions"""

        # Get AutoGluon predictions
        predictions = self.multi_stock.predict_stocks(
            symbols,
            datetime.now().strftime('%Y-%m-%d')
        )

        # Generate signals
        ag_signals = self.signal_generator.generate_signals(predictions)

        # Enhance each AI decision
        enhanced_decisions = []

        for decision in ai_decisions:
            symbol = decision.get('symbol')

            if symbol in ag_signals:
                ag_signal = ag_signals[symbol]

                # Combine signals
                if ag_signal['confidence'] > 0.7:
                    # High confidence AutoGluon signal
                    if ag_signal['action'] == decision['action']:
                        # Agreement strengthens decision
                        decision['confidence'] *= 1.2
                        decision['reasoning'] += f". AutoGluon confirms: {ag_signal['reasoning']}"
                    else:
                        # Disagreement requires reconciliation
                        decision['autogluon_disagreement'] = True
                        decision['autogluon_signal'] = ag_signal

                decision['autogluon_analysis'] = {
                    'expected_return': ag_signal['expected_return'],
                    'confidence': ag_signal['confidence'],
                    'prediction_interval': [ag_signal['lower_bound'], ag_signal['upper_bound']]
                }

            enhanced_decisions.append(decision)

        return enhanced_decisions
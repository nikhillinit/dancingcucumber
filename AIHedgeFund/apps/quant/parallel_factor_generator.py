"""
Parallel Factor Generation System
=================================
High-performance technical and fundamental factor generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from joblib import Parallel, delayed
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import ray
import pandas_ta as ta
import talib
from numba import jit, vectorize, float64, int64
import asyncio
from functools import lru_cache
import polars as pl
import dask.dataframe as dd
from cachetools import TTLCache

logger = logging.getLogger(__name__)


@dataclass
class Factor:
    """Represents a calculated factor"""
    name: str
    values: np.ndarray
    timestamp: datetime
    calculation_time: float
    metadata: Dict[str, Any]


class ParallelFactorGenerator:
    """
    High-performance factor generation using parallel processing
    Generates 100+ technical and statistical factors
    """

    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or mp.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)
        self.cache = TTLCache(maxsize=1000, ttl=300)

        # Initialize Ray for distributed computing
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=self.n_workers)

        # Define factor groups
        self.factor_groups = {
            'price': self._get_price_factors(),
            'volume': self._get_volume_factors(),
            'momentum': self._get_momentum_factors(),
            'volatility': self._get_volatility_factors(),
            'pattern': self._get_pattern_factors(),
            'statistical': self._get_statistical_factors(),
        }

        logger.info(f"Initialized ParallelFactorGenerator with {self.n_workers} workers")

    def _get_price_factors(self) -> List[Callable]:
        """Define price-based factors"""
        return [
            self._price_to_ma,
            self._price_position,
            self._price_efficiency,
            self._price_channels,
            self._price_transforms,
        ]

    def _get_volume_factors(self) -> List[Callable]:
        """Define volume-based factors"""
        return [
            self._volume_ratio,
            self._volume_trend,
            self._vwap_deviation,
            self._money_flow,
            self._accumulation_distribution,
        ]

    def _get_momentum_factors(self) -> List[Callable]:
        """Define momentum factors"""
        return [
            self._rsi_variations,
            self._macd_features,
            self._stochastic_features,
            self._momentum_oscillators,
            self._rate_of_change,
        ]

    def _get_volatility_factors(self) -> List[Callable]:
        """Define volatility factors"""
        return [
            self._historical_volatility,
            self._garch_volatility,
            self._atr_features,
            self._bollinger_features,
            self._keltner_features,
        ]

    def _get_pattern_factors(self) -> List[Callable]:
        """Define pattern recognition factors"""
        return [
            self._candlestick_patterns,
            self._support_resistance,
            self._trend_strength,
            self._fractal_dimension,
            self._hurst_exponent,
        ]

    def _get_statistical_factors(self) -> List[Callable]:
        """Define statistical factors"""
        return [
            self._autocorrelation,
            self._entropy_features,
            self._distribution_features,
            self._regression_features,
            self._cointegration_features,
        ]

    async def generate_all_factors(
        self,
        data: pd.DataFrame,
        parallel_mode: str = 'ray'
    ) -> Dict[str, Factor]:
        """
        Generate all factors in parallel
        parallel_mode: 'ray', 'joblib', 'dask', or 'thread'
        """
        import time
        start_time = time.time()

        if parallel_mode == 'ray':
            factors = await self._generate_factors_ray(data)
        elif parallel_mode == 'joblib':
            factors = self._generate_factors_joblib(data)
        elif parallel_mode == 'dask':
            factors = self._generate_factors_dask(data)
        else:
            factors = await self._generate_factors_thread(data)

        total_time = time.time() - start_time
        logger.info(f"Generated {len(factors)} factors in {total_time:.2f}s using {parallel_mode}")

        return factors

    async def _generate_factors_ray(self, data: pd.DataFrame) -> Dict[str, Factor]:
        """Generate factors using Ray distributed computing"""
        # Convert to Ray dataset
        ray_data = ray.put(data)

        # Create Ray tasks
        tasks = []
        for group_name, factor_funcs in self.factor_groups.items():
            for func in factor_funcs:
                task = generate_factor_ray.remote(ray_data, func, group_name)
                tasks.append(task)

        # Execute all tasks in parallel
        results = ray.get(tasks)

        # Organize results
        factors = {}
        for result in results:
            if result:
                factors[result.name] = result

        return factors

    def _generate_factors_joblib(self, data: pd.DataFrame) -> Dict[str, Factor]:
        """Generate factors using joblib parallel processing"""
        all_tasks = []
        for group_name, factor_funcs in self.factor_groups.items():
            for func in factor_funcs:
                all_tasks.append((func, data, group_name))

        # Execute in parallel
        results = Parallel(n_jobs=self.n_workers)(
            delayed(self._compute_factor)(func, data, group)
            for func, data, group in all_tasks
        )

        # Organize results
        factors = {r.name: r for r in results if r}
        return factors

    def _generate_factors_dask(self, data: pd.DataFrame) -> Dict[str, Factor]:
        """Generate factors using Dask"""
        # Convert to Dask DataFrame
        ddf = dd.from_pandas(data, npartitions=self.n_workers)

        factors = {}
        for group_name, factor_funcs in self.factor_groups.items():
            for func in factor_funcs:
                result = func(ddf.compute())
                if result:
                    factors[result.name] = result

        return factors

    async def _generate_factors_thread(self, data: pd.DataFrame) -> Dict[str, Factor]:
        """Generate factors using thread pool"""
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=self.n_workers)

        tasks = []
        for group_name, factor_funcs in self.factor_groups.items():
            for func in factor_funcs:
                task = loop.run_in_executor(
                    executor,
                    self._compute_factor,
                    func, data, group_name
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks)
        factors = {r.name: r for r in results if r}

        executor.shutdown()
        return factors

    def _compute_factor(self, func: Callable, data: pd.DataFrame, group: str) -> Factor:
        """Compute a single factor"""
        import time
        start_time = time.time()

        try:
            result = func(data)
            if result is not None:
                return Factor(
                    name=f"{group}_{func.__name__}",
                    values=result,
                    timestamp=datetime.now(),
                    calculation_time=time.time() - start_time,
                    metadata={'group': group}
                )
        except Exception as e:
            logger.error(f"Error computing factor {func.__name__}: {e}")

        return None

    # Price-based factors
    def _price_to_ma(self, data: pd.DataFrame) -> np.ndarray:
        """Price relative to moving averages"""
        close = data['close'].values
        features = []

        for period in [5, 10, 20, 50, 100, 200]:
            if len(close) >= period:
                ma = self._fast_sma(close, period)
                features.append(close[-1] / ma[-1] - 1)
            else:
                features.append(0)

        return np.array(features)

    @staticmethod
    @jit(nopython=True)
    def _fast_sma(values: np.ndarray, period: int) -> np.ndarray:
        """Fast SMA calculation using Numba"""
        result = np.empty_like(values)
        result[:period-1] = np.nan

        for i in range(period-1, len(values)):
            result[i] = np.mean(values[i-period+1:i+1])

        return result

    def _price_position(self, data: pd.DataFrame) -> np.ndarray:
        """Price position within recent range"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        features = []
        for period in [5, 10, 20, 50]:
            if len(close) >= period:
                highest = np.max(high[-period:])
                lowest = np.min(low[-period:])
                position = (close[-1] - lowest) / (highest - lowest + 1e-10)
                features.append(position)
            else:
                features.append(0.5)

        return np.array(features)

    def _price_efficiency(self, data: pd.DataFrame) -> np.ndarray:
        """Kaufman's Efficiency Ratio"""
        close = data['close'].values
        if len(close) < 20:
            return np.array([0])

        direction = abs(close[-1] - close[-20])
        volatility = np.sum(np.abs(np.diff(close[-20:])))

        if volatility > 0:
            efficiency = direction / volatility
        else:
            efficiency = 0

        return np.array([efficiency])

    def _price_channels(self, data: pd.DataFrame) -> np.ndarray:
        """Donchian and Keltner channel positions"""
        if len(data) < 20:
            return np.zeros(4)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Donchian channels
        upper_dc = np.max(high[-20:])
        lower_dc = np.min(low[-20:])
        dc_position = (close[-1] - lower_dc) / (upper_dc - lower_dc + 1e-10)

        # Keltner channels
        ma = np.mean(close[-20:])
        atr = talib.ATR(high, low, close, timeperiod=20)[-1]
        upper_kc = ma + 2 * atr
        lower_kc = ma - 2 * atr
        kc_position = (close[-1] - lower_kc) / (upper_kc - lower_kc + 1e-10)

        return np.array([dc_position, kc_position, upper_dc/close[-1], lower_dc/close[-1]])

    def _price_transforms(self, data: pd.DataFrame) -> np.ndarray:
        """Price transformations"""
        close = data['close'].values

        if len(close) < 2:
            return np.zeros(4)

        # Log returns
        log_return = np.log(close[-1] / close[-2])

        # Normalized price
        mean = np.mean(close)
        std = np.std(close)
        z_score = (close[-1] - mean) / (std + 1e-10)

        # Price acceleration
        if len(close) >= 3:
            acceleration = (close[-1] - close[-2]) - (close[-2] - close[-3])
        else:
            acceleration = 0

        # Detrended price
        if len(close) >= 20:
            trend = np.polyfit(range(20), close[-20:], 1)[0]
            detrended = close[-1] - (mean + trend * 19)
        else:
            detrended = 0

        return np.array([log_return, z_score, acceleration, detrended])

    # Volume-based factors
    def _volume_ratio(self, data: pd.DataFrame) -> np.ndarray:
        """Volume ratios and trends"""
        volume = data['volume'].values

        if len(volume) < 20:
            return np.zeros(3)

        current_vol = volume[-1]
        avg_vol_5 = np.mean(volume[-5:])
        avg_vol_20 = np.mean(volume[-20:])

        vol_ratio_5 = current_vol / (avg_vol_5 + 1e-10)
        vol_ratio_20 = current_vol / (avg_vol_20 + 1e-10)
        vol_trend = avg_vol_5 / (avg_vol_20 + 1e-10)

        return np.array([vol_ratio_5, vol_ratio_20, vol_trend])

    def _volume_trend(self, data: pd.DataFrame) -> np.ndarray:
        """Volume trend indicators"""
        volume = data['volume'].values
        close = data['close'].values

        if len(volume) < 20:
            return np.zeros(2)

        # On-Balance Volume
        obv = talib.OBV(close, volume)
        obv_ma = np.mean(obv[-20:])
        obv_signal = obv[-1] / (obv_ma + 1e-10) - 1

        # Volume-Price Trend
        vpt = np.cumsum((close[1:] - close[:-1]) / close[:-1] * volume[1:])
        if len(vpt) >= 19:
            vpt_signal = vpt[-1] / (np.mean(vpt[-19:]) + 1e-10) - 1
        else:
            vpt_signal = 0

        return np.array([obv_signal, vpt_signal])

    def _vwap_deviation(self, data: pd.DataFrame) -> np.ndarray:
        """VWAP and price deviation"""
        if len(data) < 2:
            return np.zeros(2)

        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = np.sum(typical_price * data['volume']) / np.sum(data['volume'])
        vwap_dev = data['close'].iloc[-1] / vwap - 1

        # Intraday VWAP bands
        vwap_std = np.std(typical_price)
        upper_band = vwap + 2 * vwap_std
        band_position = (data['close'].iloc[-1] - vwap) / (2 * vwap_std + 1e-10)

        return np.array([vwap_dev, band_position])

    def _money_flow(self, data: pd.DataFrame) -> np.ndarray:
        """Money flow indicators"""
        if len(data) < 14:
            return np.zeros(2)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # Money Flow Index
        mfi = talib.MFI(high, low, close, volume, timeperiod=14)[-1]

        # Chaikin Money Flow
        cmf = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)[-1]

        return np.array([mfi/100, cmf])

    def _accumulation_distribution(self, data: pd.DataFrame) -> np.ndarray:
        """Accumulation/Distribution indicators"""
        if len(data) < 20:
            return np.zeros(2)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # A/D Line
        ad = talib.AD(high, low, close, volume)
        ad_ma = np.mean(ad[-20:])
        ad_signal = ad[-1] / (ad_ma + 1e-10) - 1

        # Williams A/D
        wad = np.cumsum((close - np.minimum(low, np.roll(close, 1))) * volume)
        if len(wad) >= 20:
            wad_signal = wad[-1] / (np.mean(wad[-20:]) + 1e-10) - 1
        else:
            wad_signal = 0

        return np.array([ad_signal, wad_signal])

    # Momentum factors
    def _rsi_variations(self, data: pd.DataFrame) -> np.ndarray:
        """RSI and variations"""
        if len(data) < 20:
            return np.zeros(4)

        close = data['close'].values

        rsi_14 = talib.RSI(close, timeperiod=14)[-1]
        rsi_7 = talib.RSI(close, timeperiod=7)[-1]
        rsi_21 = talib.RSI(close, timeperiod=21)[-1]

        # Stochastic RSI
        stoch_rsi = talib.STOCHRSI(close, timeperiod=14)[0][-1]

        return np.array([rsi_14/100, rsi_7/100, rsi_21/100, stoch_rsi/100])

    def _macd_features(self, data: pd.DataFrame) -> np.ndarray:
        """MACD-based features"""
        if len(data) < 26:
            return np.zeros(3)

        close = data['close'].values
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        # Normalize by price
        macd_norm = macd[-1] / close[-1]
        signal_norm = signal[-1] / close[-1]
        hist_norm = hist[-1] / close[-1]

        return np.array([macd_norm, signal_norm, hist_norm])

    def _stochastic_features(self, data: pd.DataFrame) -> np.ndarray:
        """Stochastic oscillator features"""
        if len(data) < 14:
            return np.zeros(3)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)

        # Williams %R
        willr = talib.WILLR(high, low, close, timeperiod=14)[-1]

        return np.array([slowk[-1]/100, slowd[-1]/100, (willr + 100)/100])

    def _momentum_oscillators(self, data: pd.DataFrame) -> np.ndarray:
        """Various momentum oscillators"""
        if len(data) < 20:
            return np.zeros(4)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values

        # Ultimate Oscillator
        ultosc = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)[-1]

        # Commodity Channel Index
        cci = talib.CCI(high, low, close, timeperiod=14)[-1]

        # Momentum
        mom = talib.MOM(close, timeperiod=10)[-1]

        # Rate of Change
        roc = talib.ROC(close, timeperiod=10)[-1]

        return np.array([ultosc/100, cci/200, mom/close[-1], roc/100])

    def _rate_of_change(self, data: pd.DataFrame) -> np.ndarray:
        """Rate of change features"""
        if len(data) < 20:
            return np.zeros(3)

        close = data['close'].values

        roc_5 = (close[-1] / close[-5] - 1) if len(close) >= 5 else 0
        roc_10 = (close[-1] / close[-10] - 1) if len(close) >= 10 else 0
        roc_20 = (close[-1] / close[-20] - 1) if len(close) >= 20 else 0

        return np.array([roc_5, roc_10, roc_20])

    # Volatility factors
    def _historical_volatility(self, data: pd.DataFrame) -> np.ndarray:
        """Historical volatility measures"""
        if len(data) < 20:
            return np.zeros(3)

        close = data['close'].values
        returns = np.diff(close) / close[:-1]

        # Different period volatilities
        vol_5 = np.std(returns[-5:]) * np.sqrt(252) if len(returns) >= 5 else 0
        vol_10 = np.std(returns[-10:]) * np.sqrt(252) if len(returns) >= 10 else 0
        vol_20 = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0

        return np.array([vol_5, vol_10, vol_20])

    def _garch_volatility(self, data: pd.DataFrame) -> np.ndarray:
        """GARCH volatility (simplified)"""
        if len(data) < 30:
            return np.zeros(2)

        returns = data['close'].pct_change().dropna()

        # Simple EWMA volatility as GARCH proxy
        ewm_vol = returns.ewm(span=10).std().iloc[-1]

        # Volatility ratio
        short_vol = returns.iloc[-5:].std()
        long_vol = returns.iloc[-20:].std()
        vol_ratio = short_vol / (long_vol + 1e-10)

        return np.array([ewm_vol, vol_ratio])

    def _atr_features(self, data: pd.DataFrame) -> np.ndarray:
        """ATR-based features"""
        if len(data) < 14:
            return np.zeros(3)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        atr_pct = atr / close[-1]

        # ATR bands
        upper_atr = close[-1] + 2 * atr
        lower_atr = close[-1] - 2 * atr
        atr_position = (close[-1] - lower_atr) / (upper_atr - lower_atr)

        # Normalized ATR
        natr = talib.NATR(high, low, close, timeperiod=14)[-1]

        return np.array([atr_pct, atr_position, natr/100])

    def _bollinger_features(self, data: pd.DataFrame) -> np.ndarray:
        """Bollinger Bands features"""
        if len(data) < 20:
            return np.zeros(4)

        close = data['close'].values

        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

        bb_position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1] + 1e-10)
        bb_width = (upper[-1] - lower[-1]) / middle[-1]
        price_to_upper = close[-1] / upper[-1]
        price_to_lower = close[-1] / lower[-1]

        return np.array([bb_position, bb_width, price_to_upper, price_to_lower])

    def _keltner_features(self, data: pd.DataFrame) -> np.ndarray:
        """Keltner Channel features"""
        if len(data) < 20:
            return np.zeros(3)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # EMA and ATR for Keltner
        ema = talib.EMA(close, timeperiod=20)[-1]
        atr = talib.ATR(high, low, close, timeperiod=20)[-1]

        upper_kc = ema + 2 * atr
        lower_kc = ema - 2 * atr

        kc_position = (close[-1] - lower_kc) / (upper_kc - lower_kc + 1e-10)
        kc_width = (upper_kc - lower_kc) / ema
        price_to_ema = close[-1] / ema

        return np.array([kc_position, kc_width, price_to_ema])

    # Pattern factors
    def _candlestick_patterns(self, data: pd.DataFrame) -> np.ndarray:
        """Candlestick pattern recognition"""
        if len(data) < 10:
            return np.zeros(5)

        open_ = data['open'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Selected important patterns
        hammer = talib.CDLHAMMER(open_, high, low, close)[-1]
        doji = talib.CDLDOJI(open_, high, low, close)[-1]
        engulfing = talib.CDLENGULFING(open_, high, low, close)[-1]
        harami = talib.CDLHARAMI(open_, high, low, close)[-1]
        marubozu = talib.CDLMARUBOZU(open_, high, low, close)[-1]

        # Normalize to -1, 0, 1
        patterns = np.array([hammer, doji, engulfing, harami, marubozu])
        return patterns / 100  # TA-Lib returns 0, ±100

    def _support_resistance(self, data: pd.DataFrame) -> np.ndarray:
        """Support and resistance levels"""
        if len(data) < 50:
            return np.zeros(4)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # Find local peaks and troughs
        peaks = []
        troughs = []

        for i in range(2, len(high) - 2):
            if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
                peaks.append(high[i])
            if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
                troughs.append(low[i])

        if peaks and troughs:
            resistance = np.mean(peaks[-3:]) if len(peaks) >= 3 else peaks[-1]
            support = np.mean(troughs[-3:]) if len(troughs) >= 3 else troughs[-1]

            distance_to_resistance = (resistance - close[-1]) / close[-1]
            distance_to_support = (close[-1] - support) / close[-1]
            sr_range = (resistance - support) / close[-1]
            sr_position = (close[-1] - support) / (resistance - support + 1e-10)
        else:
            distance_to_resistance = distance_to_support = sr_range = sr_position = 0

        return np.array([distance_to_resistance, distance_to_support, sr_range, sr_position])

    def _trend_strength(self, data: pd.DataFrame) -> np.ndarray:
        """Trend strength indicators"""
        if len(data) < 20:
            return np.zeros(3)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # ADX for trend strength
        adx = talib.ADX(high, low, close, timeperiod=14)[-1]

        # Linear regression slope
        x = np.arange(len(close[-20:]))
        slope, _ = np.polyfit(x, close[-20:], 1)
        normalized_slope = slope / close[-1]

        # Aroon for trend
        aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
        aroon_osc = (aroon_up[-1] - aroon_down[-1]) / 100

        return np.array([adx/100, normalized_slope, aroon_osc])

    def _fractal_dimension(self, data: pd.DataFrame) -> np.ndarray:
        """Fractal dimension (simplified)"""
        if len(data) < 30:
            return np.array([1.5])

        close = data['close'].values[-30:]

        # Simplified box-counting dimension
        n = len(close)
        max_price = np.max(close)
        min_price = np.min(close)

        # Normalize to unit square
        normalized = (close - min_price) / (max_price - min_price + 1e-10)

        # Count boxes at different scales
        boxes_2 = len(np.unique(np.floor(normalized * 2)))
        boxes_4 = len(np.unique(np.floor(normalized * 4)))

        if boxes_2 > 0 and boxes_4 > 0:
            fractal_dim = np.log(boxes_4 / boxes_2) / np.log(2)
        else:
            fractal_dim = 1.5

        return np.array([fractal_dim])

    def _hurst_exponent(self, data: pd.DataFrame) -> np.ndarray:
        """Hurst exponent for persistence"""
        if len(data) < 100:
            return np.array([0.5])

        close = data['close'].values[-100:]
        returns = np.diff(close) / close[:-1]

        # Simplified R/S analysis
        n = len(returns)
        mean_return = np.mean(returns)
        deviations = returns - mean_return
        cumulative_deviations = np.cumsum(deviations)

        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
        S = np.std(returns)

        if S > 0:
            RS = R / S
            hurst = np.log(RS) / np.log(n) if RS > 0 else 0.5
        else:
            hurst = 0.5

        return np.array([hurst])

    # Statistical factors
    def _autocorrelation(self, data: pd.DataFrame) -> np.ndarray:
        """Autocorrelation features"""
        if len(data) < 30:
            return np.zeros(3)

        returns = data['close'].pct_change().dropna().values

        # Different lag autocorrelations
        ac_1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        ac_5 = np.corrcoef(returns[:-5], returns[5:])[0, 1] if len(returns) > 5 else 0
        ac_10 = np.corrcoef(returns[:-10], returns[10:])[0, 1] if len(returns) > 10 else 0

        return np.array([ac_1, ac_5, ac_10])

    def _entropy_features(self, data: pd.DataFrame) -> np.ndarray:
        """Entropy-based features"""
        if len(data) < 20:
            return np.zeros(2)

        returns = data['close'].pct_change().dropna().values

        # Shannon entropy
        hist, _ = np.histogram(returns, bins=10)
        hist = hist / np.sum(hist)
        shannon_entropy = -np.sum(hist * np.log(hist + 1e-10))

        # Approximate entropy (simplified)
        def approx_entropy(data, m=2, r=0.2):
            N = len(data)
            patterns = np.array([data[i:i+m] for i in range(N-m+1)])
            C = []
            for pattern in patterns:
                template_matches = np.sum(np.max(np.abs(patterns - pattern), axis=1) <= r)
                C.append(template_matches / (N - m + 1))
            phi = np.mean(np.log(C))
            return phi

        ap_entropy = approx_entropy(returns[-20:])

        return np.array([shannon_entropy, ap_entropy])

    def _distribution_features(self, data: pd.DataFrame) -> np.ndarray:
        """Distribution characteristics"""
        if len(data) < 30:
            return np.zeros(4)

        returns = data['close'].pct_change().dropna().values

        # Statistical moments
        mean = np.mean(returns)
        std = np.std(returns)
        skew = np.mean(((returns - mean) / (std + 1e-10)) ** 3)
        kurtosis = np.mean(((returns - mean) / (std + 1e-10)) ** 4) - 3

        return np.array([mean, std, skew, kurtosis])

    def _regression_features(self, data: pd.DataFrame) -> np.ndarray:
        """Regression-based features"""
        if len(data) < 20:
            return np.zeros(3)

        close = data['close'].values[-20:]
        x = np.arange(len(close))

        # Linear regression
        slope, intercept = np.polyfit(x, close, 1)
        predicted = slope * x + intercept
        r_squared = 1 - np.sum((close - predicted) ** 2) / np.sum((close - np.mean(close)) ** 2)

        # Polynomial regression
        poly_coef = np.polyfit(x, close, 2)
        poly_predicted = np.polyval(poly_coef, x)
        poly_r_squared = 1 - np.sum((close - poly_predicted) ** 2) / np.sum((close - np.mean(close)) ** 2)

        # Residual standard error
        residuals = close - predicted
        rse = np.std(residuals) / close[-1]

        return np.array([r_squared, poly_r_squared, rse])

    def _cointegration_features(self, data: pd.DataFrame) -> np.ndarray:
        """Cointegration with market (simplified)"""
        if len(data) < 50:
            return np.array([0])

        # Simplified - would normally test against market index
        close = data['close'].values
        volume = data['volume'].values

        # Test stationarity of price-volume spread
        if len(close) == len(volume):
            spread = close - (volume / np.mean(volume)) * np.mean(close)
            adf_stat = np.std(spread) / np.mean(spread) if np.mean(spread) != 0 else 0
        else:
            adf_stat = 0

        return np.array([adf_stat])


@ray.remote
def generate_factor_ray(data, func, group_name):
    """Ray remote function for factor generation"""
    try:
        import time
        start_time = time.time()
        result = func(ray.get(data))
        if result is not None:
            return Factor(
                name=f"{group_name}_{func.__name__}",
                values=result,
                timestamp=datetime.now(),
                calculation_time=time.time() - start_time,
                metadata={'group': group_name}
            )
    except Exception as e:
        logger.error(f"Error in Ray factor generation: {e}")
    return None


# Example usage
async def main():
    """Example usage of parallel factor generator"""
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.random(len(dates)) * 100 + 100,
        'high': np.random.random(len(dates)) * 100 + 105,
        'low': np.random.random(len(dates)) * 100 + 95,
        'close': np.random.random(len(dates)) * 100 + 100,
        'volume': np.random.random(len(dates)) * 1000000
    }, index=dates)

    # Initialize generator
    generator = ParallelFactorGenerator(n_workers=8)

    # Generate factors using different parallel modes
    for mode in ['ray', 'joblib', 'thread']:
        print(f"\nGenerating factors using {mode}...")
        factors = await generator.generate_all_factors(data, parallel_mode=mode)
        print(f"Generated {len(factors)} factors")

        # Show sample factors
        for i, (name, factor) in enumerate(list(factors.items())[:5]):
            print(f"  {name}: shape={factor.values.shape}, time={factor.calculation_time:.3f}s")


if __name__ == "__main__":
    asyncio.run(main()
    ray.shutdown()
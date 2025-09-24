"""
Intraday Pattern Recognition System with Multi-Agent Processing
===============================================================
Advanced pattern detection for daily portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
import logging
import ray
import asyncio
from collections import deque
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class IntradayPattern:
    """Intraday pattern detection result"""
    pattern_type: str  # opening_momentum, lunch_reversal, closing_rally, etc.
    symbol: str
    confidence: float
    expected_move: float
    optimal_entry: datetime
    optimal_exit: datetime
    volume_profile: Dict[str, float]
    risk_score: float
    timestamp: datetime


@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    vwap: float
    volume_nodes: List[Tuple[float, float]]  # (price, volume)
    poc: float  # Point of control
    value_area_high: float
    value_area_low: float
    liquidity_score: float
    optimal_windows: List[Tuple[time, time]]


@dataclass
class MarketMicrostructure:
    """Market microstructure metrics"""
    bid_ask_spread: float
    order_flow_imbalance: float
    tick_direction: int
    volume_rate: float
    price_impact: float
    liquidity_depth: float


@ray.remote
class OpeningPatternAgent:
    """Agent for detecting opening market patterns"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.opening_window = (time(9, 30), time(10, 30))
        self.pattern_history = deque(maxlen=100)

    async def detect_opening_patterns(
        self,
        data: pd.DataFrame,
        premarket_data: Optional[pd.DataFrame] = None
    ) -> List[IntradayPattern]:
        """Detect opening bell patterns"""
        patterns = []

        # Extract opening hour data
        opening_data = self._filter_time_window(data, self.opening_window)

        if len(opening_data) < 10:
            return patterns

        # 1. Gap analysis
        if premarket_data is not None and len(premarket_data) > 0:
            gap_pattern = self._analyze_gap(premarket_data, opening_data)
            if gap_pattern:
                patterns.append(gap_pattern)

        # 2. Opening momentum
        momentum_pattern = self._detect_opening_momentum(opening_data)
        if momentum_pattern:
            patterns.append(momentum_pattern)

        # 3. Opening reversal
        reversal_pattern = self._detect_opening_reversal(opening_data)
        if reversal_pattern:
            patterns.append(reversal_pattern)

        # 4. Opening range breakout
        breakout_pattern = self._detect_opening_range_breakout(opening_data)
        if breakout_pattern:
            patterns.append(breakout_pattern)

        return patterns

    def _filter_time_window(
        self,
        data: pd.DataFrame,
        window: Tuple[time, time]
    ) -> pd.DataFrame:
        """Filter data for specific time window"""
        if data.empty:
            return pd.DataFrame()

        # Extract time from index
        data_copy = data.copy()
        data_copy['time'] = pd.to_datetime(data.index).time

        # Filter by time window
        mask = (data_copy['time'] >= window[0]) & (data_copy['time'] <= window[1])
        return data[mask]

    def _analyze_gap(
        self,
        premarket: pd.DataFrame,
        opening: pd.DataFrame
    ) -> Optional[IntradayPattern]:
        """Analyze gap between premarket and opening"""
        if premarket.empty or opening.empty:
            return None

        premarket_close = premarket['close'].iloc[-1]
        opening_price = opening['open'].iloc[0] if 'open' in opening else opening['close'].iloc[0]

        gap_pct = (opening_price - premarket_close) / premarket_close

        if abs(gap_pct) > 0.005:  # 0.5% gap threshold
            # Analyze gap fill probability
            gap_filled = False
            if gap_pct > 0:  # Gap up
                gap_filled = opening['low'].min() <= premarket_close
            else:  # Gap down
                gap_filled = opening['high'].max() >= premarket_close

            confidence = min(0.9, abs(gap_pct) * 20)  # Scale confidence with gap size

            return IntradayPattern(
                pattern_type='gap_play',
                symbol='UNKNOWN',
                confidence=confidence,
                expected_move=-gap_pct * 0.5 if not gap_filled else 0,  # Expect partial fill
                optimal_entry=opening.index[0],
                optimal_exit=opening.index[0] + timedelta(minutes=30),
                volume_profile={'opening': opening['volume'].sum()},
                risk_score=abs(gap_pct),
                timestamp=datetime.now()
            )

        return None

    def _detect_opening_momentum(self, data: pd.DataFrame) -> Optional[IntradayPattern]:
        """Detect opening momentum continuation"""
        if len(data) < 5:
            return None

        # Calculate momentum in first 15 minutes
        first_15min = data.iloc[:15] if len(data) >= 15 else data

        if 'close' in first_15min.columns:
            returns = first_15min['close'].pct_change().dropna()

            if len(returns) > 0:
                momentum = returns.mean()
                consistency = sum(returns > 0) / len(returns) if momentum > 0 else sum(returns < 0) / len(returns)

                if abs(momentum) > 0.001 and consistency > 0.7:  # Strong directional move
                    return IntradayPattern(
                        pattern_type='opening_momentum',
                        symbol='UNKNOWN',
                        confidence=consistency,
                        expected_move=momentum * 3,  # Expect continuation
                        optimal_entry=data.index[15] if len(data) > 15 else data.index[-1],
                        optimal_exit=data.index[0] + timedelta(minutes=45),
                        volume_profile={'momentum_volume': first_15min['volume'].sum()},
                        risk_score=1 - consistency,
                        timestamp=datetime.now()
                    )

        return None

    def _detect_opening_reversal(self, data: pd.DataFrame) -> Optional[IntradayPattern]:
        """Detect opening reversal pattern"""
        if len(data) < 20:
            return None

        # Check for reversal after initial move
        first_10min = data.iloc[:10]
        next_10min = data.iloc[10:20]

        if 'close' in data.columns:
            initial_move = (first_10min['close'].iloc[-1] - first_10min['close'].iloc[0]) / first_10min['close'].iloc[0]
            reversal_move = (next_10min['close'].iloc[-1] - next_10min['close'].iloc[0]) / next_10min['close'].iloc[0]

            # Check for reversal
            if initial_move * reversal_move < 0 and abs(reversal_move) > abs(initial_move) * 0.5:
                confidence = min(0.8, abs(reversal_move) / abs(initial_move))

                return IntradayPattern(
                    pattern_type='opening_reversal',
                    symbol='UNKNOWN',
                    confidence=confidence,
                    expected_move=reversal_move,
                    optimal_entry=data.index[20],
                    optimal_exit=data.index[0] + timedelta(minutes=60),
                    volume_profile={'reversal_volume': next_10min['volume'].sum()},
                    risk_score=1 - confidence,
                    timestamp=datetime.now()
                )

        return None

    def _detect_opening_range_breakout(self, data: pd.DataFrame) -> Optional[IntradayPattern]:
        """Detect opening range breakout"""
        if len(data) < 30:
            return None

        # Define opening range (first 30 minutes)
        opening_range = data.iloc[:30]

        if 'high' in opening_range.columns and 'low' in opening_range.columns:
            range_high = opening_range['high'].max()
            range_low = opening_range['low'].min()
            range_size = (range_high - range_low) / range_low

            # Check for breakout after opening range
            if len(data) > 30:
                post_range = data.iloc[30:]

                # Check for breakout
                breakout_up = post_range['close'].max() > range_high * 1.002
                breakout_down = post_range['close'].min() < range_low * 0.998

                if breakout_up or breakout_down:
                    direction = 1 if breakout_up else -1
                    confidence = min(0.75, range_size * 10)  # Tighter range = higher confidence

                    return IntradayPattern(
                        pattern_type='opening_range_breakout',
                        symbol='UNKNOWN',
                        confidence=confidence,
                        expected_move=direction * range_size,
                        optimal_entry=data.index[30],
                        optimal_exit=data.index[0] + timedelta(hours=2),
                        volume_profile={'breakout_volume': post_range['volume'].iloc[0] if len(post_range) > 0 else 0},
                        risk_score=range_size,
                        timestamp=datetime.now()
                    )

        return None


@ray.remote
class LunchPatternAgent:
    """Agent for detecting lunch hour patterns"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.lunch_window = (time(11, 30), time(13, 30))

    async def detect_lunch_patterns(
        self,
        data: pd.DataFrame,
        morning_patterns: List[IntradayPattern]
    ) -> List[IntradayPattern]:
        """Detect lunch hour patterns"""
        patterns = []

        # Extract lunch hour data
        lunch_data = self._filter_time_window(data, self.lunch_window)

        if len(lunch_data) < 10:
            return patterns

        # 1. Lunch lull pattern
        lull_pattern = self._detect_lunch_lull(lunch_data, data)
        if lull_pattern:
            patterns.append(lull_pattern)

        # 2. Lunch reversal
        reversal_pattern = self._detect_lunch_reversal(lunch_data, morning_patterns)
        if reversal_pattern:
            patterns.append(reversal_pattern)

        # 3. Consolidation breakout
        consolidation_pattern = self._detect_consolidation(lunch_data)
        if consolidation_pattern:
            patterns.append(consolidation_pattern)

        return patterns

    def _filter_time_window(
        self,
        data: pd.DataFrame,
        window: Tuple[time, time]
    ) -> pd.DataFrame:
        """Filter data for specific time window"""
        if data.empty:
            return pd.DataFrame()

        data_copy = data.copy()
        data_copy['time'] = pd.to_datetime(data.index).time
        mask = (data_copy['time'] >= window[0]) & (data_copy['time'] <= window[1])
        return data[mask]

    def _detect_lunch_lull(
        self,
        lunch_data: pd.DataFrame,
        full_data: pd.DataFrame
    ) -> Optional[IntradayPattern]:
        """Detect low volume lunch period"""
        if lunch_data.empty or full_data.empty:
            return None

        # Compare lunch volume to morning volume
        morning_data = self._filter_time_window(full_data, (time(9, 30), time(11, 30)))

        if not morning_data.empty and 'volume' in lunch_data.columns:
            lunch_avg_volume = lunch_data['volume'].mean()
            morning_avg_volume = morning_data['volume'].mean()

            if morning_avg_volume > 0:
                volume_ratio = lunch_avg_volume / morning_avg_volume

                if volume_ratio < 0.6:  # Significant volume drop
                    # Low volume = mean reversion opportunity
                    price_range = lunch_data['high'].max() - lunch_data['low'].min()
                    avg_price = lunch_data['close'].mean()

                    return IntradayPattern(
                        pattern_type='lunch_lull',
                        symbol='UNKNOWN',
                        confidence=0.7 * (1 - volume_ratio),
                        expected_move=0,  # Expect ranging
                        optimal_entry=lunch_data.index[len(lunch_data)//2],
                        optimal_exit=lunch_data.index[-1] + timedelta(minutes=30),
                        volume_profile={'lunch_volume_ratio': volume_ratio},
                        risk_score=price_range / avg_price if avg_price > 0 else 0,
                        timestamp=datetime.now()
                    )

        return None

    def _detect_lunch_reversal(
        self,
        lunch_data: pd.DataFrame,
        morning_patterns: List[IntradayPattern]
    ) -> Optional[IntradayPattern]:
        """Detect lunch hour reversal of morning trend"""
        if lunch_data.empty or not morning_patterns:
            return None

        # Get morning momentum if exists
        morning_momentum = None
        for pattern in morning_patterns:
            if pattern.pattern_type == 'opening_momentum':
                morning_momentum = pattern.expected_move
                break

        if morning_momentum and 'close' in lunch_data.columns:
            lunch_returns = lunch_data['close'].pct_change().dropna()

            if len(lunch_returns) > 0:
                lunch_direction = lunch_returns.mean()

                # Check for reversal
                if morning_momentum * lunch_direction < 0:  # Opposite signs
                    confidence = min(0.75, abs(lunch_direction) / abs(morning_momentum))

                    return IntradayPattern(
                        pattern_type='lunch_reversal',
                        symbol='UNKNOWN',
                        confidence=confidence,
                        expected_move=-morning_momentum * 0.5,  # Expect partial reversal
                        optimal_entry=lunch_data.index[0],
                        optimal_exit=lunch_data.index[-1] + timedelta(hours=1),
                        volume_profile={'reversal_strength': abs(lunch_direction)},
                        risk_score=1 - confidence,
                        timestamp=datetime.now()
                    )

        return None

    def _detect_consolidation(self, lunch_data: pd.DataFrame) -> Optional[IntradayPattern]:
        """Detect consolidation pattern during lunch"""
        if len(lunch_data) < 20 or 'close' not in lunch_data.columns:
            return None

        prices = lunch_data['close'].values
        returns = lunch_data['close'].pct_change().dropna()

        # Check for low volatility consolidation
        if len(returns) > 0:
            volatility = returns.std()
            price_range = (prices.max() - prices.min()) / prices.mean()

            if volatility < 0.002 and price_range < 0.01:  # Tight consolidation
                # Consolidation suggests breakout incoming
                return IntradayPattern(
                    pattern_type='lunch_consolidation',
                    symbol='UNKNOWN',
                    confidence=0.65,
                    expected_move=price_range * 2,  # Expect breakout
                    optimal_entry=lunch_data.index[-1],
                    optimal_exit=lunch_data.index[-1] + timedelta(hours=1.5),
                    volume_profile={'consolidation_vol': lunch_data['volume'].mean()},
                    risk_score=volatility * 100,
                    timestamp=datetime.now()
                )

        return None


@ray.remote
class ClosingPatternAgent:
    """Agent for detecting closing hour patterns"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.closing_window = (time(15, 0), time(16, 0))
        self.power_hour = (time(15, 0), time(16, 0))

    async def detect_closing_patterns(
        self,
        data: pd.DataFrame,
        day_patterns: List[IntradayPattern]
    ) -> List[IntradayPattern]:
        """Detect closing bell patterns"""
        patterns = []

        # Extract closing hour data
        closing_data = self._filter_time_window(data, self.closing_window)

        if len(closing_data) < 10:
            return patterns

        # 1. Power hour rally/selloff
        power_hour_pattern = self._detect_power_hour(closing_data, data)
        if power_hour_pattern:
            patterns.append(power_hour_pattern)

        # 2. MOC imbalance
        moc_pattern = self._detect_moc_imbalance(closing_data)
        if moc_pattern:
            patterns.append(moc_pattern)

        # 3. End-of-day reversal
        eod_reversal = self._detect_eod_reversal(closing_data, data)
        if eod_reversal:
            patterns.append(eod_reversal)

        # 4. Window dressing
        window_dressing = self._detect_window_dressing(closing_data, data)
        if window_dressing:
            patterns.append(window_dressing)

        return patterns

    def _filter_time_window(
        self,
        data: pd.DataFrame,
        window: Tuple[time, time]
    ) -> pd.DataFrame:
        """Filter data for specific time window"""
        if data.empty:
            return pd.DataFrame()

        data_copy = data.copy()
        data_copy['time'] = pd.to_datetime(data.index).time
        mask = (data_copy['time'] >= window[0]) & (data_copy['time'] <= window[1])
        return data[mask]

    def _detect_power_hour(
        self,
        closing_data: pd.DataFrame,
        full_data: pd.DataFrame
    ) -> Optional[IntradayPattern]:
        """Detect power hour momentum"""
        if closing_data.empty or full_data.empty:
            return None

        # Compare closing hour to rest of day
        pre_close = self._filter_time_window(full_data, (time(9, 30), time(15, 0)))

        if not pre_close.empty and 'close' in closing_data.columns:
            closing_momentum = closing_data['close'].pct_change().mean()
            day_momentum = pre_close['close'].pct_change().mean()

            # Check for acceleration
            if abs(closing_momentum) > abs(day_momentum) * 2:
                volume_surge = closing_data['volume'].mean() / pre_close['volume'].mean() if 'volume' in closing_data.columns else 1

                confidence = min(0.8, volume_surge * 0.4)

                return IntradayPattern(
                    pattern_type='power_hour',
                    symbol='UNKNOWN',
                    confidence=confidence,
                    expected_move=closing_momentum * 2,
                    optimal_entry=closing_data.index[0],
                    optimal_exit=closing_data.index[-1],
                    volume_profile={'volume_surge': volume_surge},
                    risk_score=1 / volume_surge if volume_surge > 0 else 1,
                    timestamp=datetime.now()
                )

        return None

    def _detect_moc_imbalance(self, closing_data: pd.DataFrame) -> Optional[IntradayPattern]:
        """Detect market-on-close imbalance"""
        if len(closing_data) < 30 or 'volume' not in closing_data.columns:
            return None

        # Check last 10 minutes volume
        last_10min = closing_data.iloc[-10:]
        earlier = closing_data.iloc[:-10]

        if len(last_10min) > 0 and len(earlier) > 0:
            late_volume = last_10min['volume'].sum()
            early_volume = earlier['volume'].sum()

            if early_volume > 0:
                volume_ratio = late_volume / early_volume

                if volume_ratio > 1.5:  # Significant late surge
                    # Analyze price action
                    price_change = (last_10min['close'].iloc[-1] - last_10min['close'].iloc[0]) / last_10min['close'].iloc[0]

                    return IntradayPattern(
                        pattern_type='moc_imbalance',
                        symbol='UNKNOWN',
                        confidence=min(0.75, volume_ratio * 0.3),
                        expected_move=price_change,
                        optimal_entry=last_10min.index[0],
                        optimal_exit=closing_data.index[-1],
                        volume_profile={'moc_volume_ratio': volume_ratio},
                        risk_score=abs(price_change),
                        timestamp=datetime.now()
                    )

        return None

    def _detect_eod_reversal(
        self,
        closing_data: pd.DataFrame,
        full_data: pd.DataFrame
    ) -> Optional[IntradayPattern]:
        """Detect end-of-day reversal"""
        if closing_data.empty or full_data.empty:
            return None

        # Get day's trend
        day_return = (full_data['close'].iloc[-1] - full_data['close'].iloc[0]) / full_data['close'].iloc[0]

        # Check last 30 minutes
        if len(closing_data) >= 30:
            last_30min = closing_data.iloc[-30:]
            last_30_return = (last_30min['close'].iloc[-1] - last_30min['close'].iloc[0]) / last_30min['close'].iloc[0]

            # Check for reversal
            if day_return * last_30_return < 0 and abs(last_30_return) > abs(day_return) * 0.2:
                confidence = min(0.7, abs(last_30_return) / abs(day_return))

                return IntradayPattern(
                    pattern_type='eod_reversal',
                    symbol='UNKNOWN',
                    confidence=confidence,
                    expected_move=last_30_return,
                    optimal_entry=last_30min.index[0],
                    optimal_exit=closing_data.index[-1],
                    volume_profile={'reversal_magnitude': abs(last_30_return)},
                    risk_score=1 - confidence,
                    timestamp=datetime.now()
                )

        return None

    def _detect_window_dressing(
        self,
        closing_data: pd.DataFrame,
        full_data: pd.DataFrame
    ) -> Optional[IntradayPattern]:
        """Detect institutional window dressing"""
        if closing_data.empty or 'volume' not in closing_data.columns:
            return None

        # Check if it's end of month/quarter
        current_date = closing_data.index[-1]
        is_month_end = current_date.day >= 28
        is_quarter_end = current_date.month in [3, 6, 9, 12] and is_month_end

        if is_month_end or is_quarter_end:
            # Check for unusual closing activity
            closing_volume = closing_data['volume'].sum()
            day_avg_volume = full_data['volume'].mean() * len(closing_data)

            if closing_volume > day_avg_volume * 1.5:
                # Likely window dressing
                price_impact = closing_data['close'].pct_change().sum()

                confidence = 0.8 if is_quarter_end else 0.6

                return IntradayPattern(
                    pattern_type='window_dressing',
                    symbol='UNKNOWN',
                    confidence=confidence,
                    expected_move=-price_impact * 0.3,  # Expect some reversal next day
                    optimal_entry=closing_data.index[-5] if len(closing_data) > 5 else closing_data.index[0],
                    optimal_exit=closing_data.index[-1],
                    volume_profile={'window_dressing_volume': closing_volume / day_avg_volume},
                    risk_score=abs(price_impact),
                    timestamp=datetime.now()
                )

        return None


@ray.remote
class VolumeProfileAgent:
    """Agent for volume profile analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def analyze_volume_profile(
        self,
        data: pd.DataFrame,
        price_bins: int = 50
    ) -> VolumeProfile:
        """Analyze volume profile and identify key levels"""

        if data.empty or 'close' not in data.columns or 'volume' not in data.columns:
            return self._empty_profile()

        # Calculate VWAP
        vwap = self._calculate_vwap(data)

        # Create volume profile
        volume_nodes = self._create_volume_profile(data, price_bins)

        # Find point of control (highest volume price)
        poc = self._find_poc(volume_nodes)

        # Calculate value area (70% of volume)
        value_area = self._calculate_value_area(volume_nodes)

        # Identify optimal trading windows
        optimal_windows = self._identify_optimal_windows(data)

        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(data)

        return VolumeProfile(
            vwap=vwap,
            volume_nodes=volume_nodes,
            poc=poc,
            value_area_high=value_area[1],
            value_area_low=value_area[0],
            liquidity_score=liquidity_score,
            optimal_windows=optimal_windows
        )

    def _empty_profile(self) -> VolumeProfile:
        """Return empty volume profile"""
        return VolumeProfile(
            vwap=0,
            volume_nodes=[],
            poc=0,
            value_area_high=0,
            value_area_low=0,
            liquidity_score=0,
            optimal_windows=[]
        )

    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """Calculate volume-weighted average price"""
        if 'close' in data.columns and 'volume' in data.columns:
            total_volume = data['volume'].sum()
            if total_volume > 0:
                return (data['close'] * data['volume']).sum() / total_volume
        return 0

    def _create_volume_profile(
        self,
        data: pd.DataFrame,
        bins: int
    ) -> List[Tuple[float, float]]:
        """Create volume profile histogram"""
        price_min = data['low'].min() if 'low' in data.columns else data['close'].min()
        price_max = data['high'].max() if 'high' in data.columns else data['close'].max()

        price_bins = np.linspace(price_min, price_max, bins)
        volume_profile = []

        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            bin_center = (bin_low + bin_high) / 2

            # Sum volume for prices in this bin
            mask = (data['close'] >= bin_low) & (data['close'] < bin_high)
            bin_volume = data.loc[mask, 'volume'].sum()

            volume_profile.append((bin_center, bin_volume))

        return volume_profile

    def _find_poc(self, volume_nodes: List[Tuple[float, float]]) -> float:
        """Find point of control (highest volume price)"""
        if not volume_nodes:
            return 0

        max_volume_node = max(volume_nodes, key=lambda x: x[1])
        return max_volume_node[0]

    def _calculate_value_area(
        self,
        volume_nodes: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Calculate value area (70% of volume)"""
        if not volume_nodes:
            return (0, 0)

        total_volume = sum(v for _, v in volume_nodes)
        target_volume = total_volume * 0.7

        # Sort by volume
        sorted_nodes = sorted(volume_nodes, key=lambda x: x[1], reverse=True)

        accumulated_volume = 0
        value_prices = []

        for price, volume in sorted_nodes:
            accumulated_volume += volume
            value_prices.append(price)

            if accumulated_volume >= target_volume:
                break

        if value_prices:
            return (min(value_prices), max(value_prices))
        return (0, 0)

    def _identify_optimal_windows(self, data: pd.DataFrame) -> List[Tuple[time, time]]:
        """Identify optimal trading windows based on volume"""
        optimal_windows = []

        if 'volume' not in data.columns:
            return optimal_windows

        # Group by hour
        data_copy = data.copy()
        data_copy['hour'] = pd.to_datetime(data.index).hour

        hourly_volume = data_copy.groupby('hour')['volume'].mean()

        # Find high volume hours
        volume_threshold = hourly_volume.quantile(0.7)

        high_volume_hours = hourly_volume[hourly_volume > volume_threshold].index.tolist()

        # Convert to time windows
        for hour in high_volume_hours:
            start = time(hour, 0)
            end = time(hour, 59)
            optimal_windows.append((start, end))

        return optimal_windows

    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """Calculate overall liquidity score"""
        if 'volume' not in data.columns:
            return 0

        # Factors: volume, spread, consistency
        avg_volume = data['volume'].mean()
        volume_std = data['volume'].std()

        # Volume consistency (lower CV = more consistent = better)
        if avg_volume > 0:
            cv = volume_std / avg_volume
            consistency_score = max(0, 1 - cv)
        else:
            consistency_score = 0

        # Normalize volume (simplified)
        volume_score = min(1, avg_volume / 1000000)  # Normalize to 1M shares

        # Combine scores
        liquidity_score = (volume_score + consistency_score) / 2

        return liquidity_score


class IntradayPatternOrchestrator:
    """Orchestrate intraday pattern recognition with multi-agent processing"""

    def __init__(self, n_agents: int = 4):
        ray.init(ignore_reinit_error=True)

        # Initialize specialized agents
        self.opening_agent = OpeningPatternAgent.remote("opening")
        self.lunch_agent = LunchPatternAgent.remote("lunch")
        self.closing_agent = ClosingPatternAgent.remote("closing")
        self.volume_agent = VolumeProfileAgent.remote("volume")

        # Pattern storage
        self.detected_patterns = deque(maxlen=1000)
        self.volume_profiles = {}

    async def analyze_intraday_patterns(
        self,
        data: Dict[str, pd.DataFrame],
        premarket_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive intraday pattern analysis"""

        analysis_tasks = []
        all_patterns = {}
        all_profiles = {}

        for symbol, df in data.items():
            # Opening patterns
            premarket = premarket_data.get(symbol) if premarket_data else None
            opening_task = self.opening_agent.detect_opening_patterns.remote(df, premarket)
            analysis_tasks.append((symbol, 'opening', opening_task))

            # Volume profile
            volume_task = self.volume_agent.analyze_volume_profile.remote(df)
            analysis_tasks.append((symbol, 'volume', volume_task))

        # Gather initial results
        opening_patterns = {}

        for symbol, pattern_type, task in analysis_tasks:
            result = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            )

            if pattern_type == 'opening':
                opening_patterns[symbol] = result
                if symbol not in all_patterns:
                    all_patterns[symbol] = []
                all_patterns[symbol].extend(result)
            elif pattern_type == 'volume':
                all_profiles[symbol] = result

        # Lunch patterns (depend on morning)
        lunch_tasks = []
        for symbol, df in data.items():
            morning = opening_patterns.get(symbol, [])
            lunch_task = self.lunch_agent.detect_lunch_patterns.remote(df, morning)
            lunch_tasks.append((symbol, lunch_task))

        # Gather lunch results
        for symbol, task in lunch_tasks:
            result = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            )
            if symbol not in all_patterns:
                all_patterns[symbol] = []
            all_patterns[symbol].extend(result)

        # Closing patterns (depend on day patterns)
        closing_tasks = []
        for symbol, df in data.items():
            day_patterns = all_patterns.get(symbol, [])
            closing_task = self.closing_agent.detect_closing_patterns.remote(df, day_patterns)
            closing_tasks.append((symbol, closing_task))

        # Gather closing results
        for symbol, task in closing_tasks:
            result = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            )
            if symbol not in all_patterns:
                all_patterns[symbol] = []
            all_patterns[symbol].extend(result)

        # Store results
        for symbol in all_patterns:
            self.detected_patterns.extend(all_patterns[symbol])
        self.volume_profiles.update(all_profiles)

        # Generate summary
        summary = self._generate_pattern_summary(all_patterns, all_profiles)

        return {
            'patterns': all_patterns,
            'volume_profiles': all_profiles,
            'summary': summary,
            'timestamp': datetime.now()
        }

    def _generate_pattern_summary(
        self,
        patterns: Dict[str, List[IntradayPattern]],
        profiles: Dict[str, VolumeProfile]
    ) -> Dict[str, Any]:
        """Generate summary of detected patterns"""

        summary = {
            'total_patterns': sum(len(p) for p in patterns.values()),
            'symbols_analyzed': len(patterns),
            'pattern_types': {},
            'avg_confidence': 0,
            'high_confidence_patterns': [],
            'optimal_entry_times': [],
            'avg_liquidity': 0
        }

        # Count pattern types
        all_patterns_flat = [p for symbol_patterns in patterns.values() for p in symbol_patterns]

        for pattern in all_patterns_flat:
            if pattern.pattern_type not in summary['pattern_types']:
                summary['pattern_types'][pattern.pattern_type] = 0
            summary['pattern_types'][pattern.pattern_type] += 1

        # Average confidence
        if all_patterns_flat:
            summary['avg_confidence'] = np.mean([p.confidence for p in all_patterns_flat])

            # High confidence patterns (>0.7)
            summary['high_confidence_patterns'] = [
                p for p in all_patterns_flat if p.confidence > 0.7
            ]

            # Optimal entry times
            entry_times = [p.optimal_entry.time() if hasattr(p.optimal_entry, 'time') else p.optimal_entry
                          for p in all_patterns_flat]
            summary['optimal_entry_times'] = list(set(entry_times))[:5]

        # Average liquidity
        if profiles:
            summary['avg_liquidity'] = np.mean([p.liquidity_score for p in profiles.values()])

        return summary

    def get_trading_recommendations(
        self,
        current_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get trading recommendations based on detected patterns"""

        recommendations = []
        current_time_only = current_time.time()

        for pattern in self.detected_patterns:
            # Check if pattern is active
            if hasattr(pattern.optimal_entry, 'time'):
                entry_time = pattern.optimal_entry.time()
            else:
                entry_time = pattern.optimal_entry

            if hasattr(pattern.optimal_exit, 'time'):
                exit_time = pattern.optimal_exit.time()
            else:
                exit_time = pattern.optimal_exit

            # Simple time comparison (would need date handling in production)
            if pattern.confidence > 0.6:
                recommendation = {
                    'symbol': pattern.symbol,
                    'pattern': pattern.pattern_type,
                    'action': 'long' if pattern.expected_move > 0 else 'short',
                    'confidence': pattern.confidence,
                    'expected_return': pattern.expected_move,
                    'risk_score': pattern.risk_score,
                    'timing': {
                        'entry': entry_time,
                        'exit': exit_time
                    }
                }
                recommendations.append(recommendation)

        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)

        return recommendations[:10]  # Top 10 recommendations

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of intraday pattern recognition"""
    orchestrator = IntradayPatternOrchestrator()

    # Generate sample intraday data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    market_data = {}
    premarket_data = {}

    for symbol in symbols:
        # Create synthetic intraday data
        times = pd.date_range(start='2024-01-15 09:30:00',
                              end='2024-01-15 16:00:00',
                              freq='1min')

        # Simulate intraday patterns
        n = len(times)
        base_price = 100

        # Add opening momentum
        opening_move = np.random.choice([1, -1]) * np.random.uniform(0.005, 0.02)

        prices = np.zeros(n)
        volumes = np.zeros(n)

        for i, t in enumerate(times):
            hour = t.hour
            minute = t.minute

            # Opening hour pattern
            if hour == 9 and minute >= 30:
                prices[i] = base_price * (1 + opening_move * (i / 60))
                volumes[i] = np.random.gamma(3, 100000)
            # Lunch lull
            elif 11 <= hour <= 13:
                prices[i] = prices[i-1] + np.random.normal(0, 0.1)
                volumes[i] = np.random.gamma(2, 50000)
            # Power hour
            elif hour >= 15:
                prices[i] = prices[i-1] + np.random.normal(0.01, 0.2)
                volumes[i] = np.random.gamma(4, 150000)
            else:
                prices[i] = prices[i-1] + np.random.normal(0, 0.15)
                volumes[i] = np.random.gamma(2.5, 75000)

        df = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.1, n),
            'high': prices + np.abs(np.random.normal(0, 0.2, n)),
            'low': prices - np.abs(np.random.normal(0, 0.2, n)),
            'close': prices,
            'volume': volumes
        }, index=times)

        market_data[symbol] = df

        # Premarket data
        premarket_times = pd.date_range(start='2024-01-15 08:00:00',
                                       end='2024-01-15 09:29:00',
                                       freq='1min')
        premarket_prices = base_price + np.cumsum(np.random.normal(0, 0.05, len(premarket_times)))

        premarket_data[symbol] = pd.DataFrame({
            'close': premarket_prices,
            'volume': np.random.gamma(1.5, 10000, len(premarket_times))
        }, index=premarket_times)

    # Analyze patterns
    print("Analyzing intraday patterns...")
    results = await orchestrator.analyze_intraday_patterns(market_data, premarket_data)

    # Display results
    print(f"\nPattern Analysis Summary:")
    summary = results['summary']
    print(f"  Total Patterns Detected: {summary['total_patterns']}")
    print(f"  Average Confidence: {summary['avg_confidence']:.1%}")
    print(f"  Average Liquidity Score: {summary['avg_liquidity']:.2f}")

    print(f"\nPattern Distribution:")
    for pattern_type, count in summary['pattern_types'].items():
        print(f"  {pattern_type}: {count}")

    print(f"\nHigh Confidence Patterns: {len(summary['high_confidence_patterns'])}")
    for pattern in summary['high_confidence_patterns'][:5]:
        print(f"  {pattern.pattern_type}: {pattern.confidence:.1%} confidence")

    # Get recommendations
    current_time = datetime(2024, 1, 15, 10, 30)
    recommendations = orchestrator.get_trading_recommendations(current_time)

    print(f"\nTop Trading Recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. {rec['symbol']}: {rec['action']} based on {rec['pattern']}")
        print(f"     Confidence: {rec['confidence']:.1%}, Expected: {rec['expected_return']:.2%}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
"""
AI Investor Persona System for AIHedgeFund
==========================================

A sophisticated system that creates AI-powered investor personas based on legendary
investors, analyzes market conditions, conducts debates, and reaches consensus
decisions for trading recommendations.

Author: AI Hedge Fund Team
Version: 1.0.0
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from abc import ABC, abstractmethod
import openai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_investor_personas.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class InvestmentAction(Enum):
    """Investment recommendation actions"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RiskTolerance(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    VERY_AGGRESSIVE = "VERY_AGGRESSIVE"


class InvestmentStyle(Enum):
    """Investment philosophy styles"""
    VALUE = "VALUE"
    GROWTH = "GROWTH"
    MOMENTUM = "MOMENTUM"
    CONTRARIAN = "CONTRARIAN"
    QUANTITATIVE = "QUANTITATIVE"
    MACRO = "MACRO"
    DISRUPTIVE = "DISRUPTIVE"


@dataclass
class PersonaCharacteristics:
    """Characteristics that define an investor persona"""
    name: str
    description: str
    investment_style: InvestmentStyle
    risk_tolerance: RiskTolerance
    time_horizon: str  # "short", "medium", "long"
    key_principles: List[str]
    preferred_sectors: List[str]
    avoided_sectors: List[str]
    decision_factors: List[str]
    communication_style: str
    famous_quotes: List[str]
    historical_returns: float  # Annual return percentage
    max_position_size: float  # Maximum % of portfolio in single position
    diversification_preference: str


@dataclass
class MarketAnalysis:
    """Market analysis data structure"""
    symbol: str
    current_price: float
    price_change: float
    price_change_percent: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    technical_indicators: Dict[str, float]
    fundamentals: Dict[str, Any]
    news_sentiment: Dict[str, Any]
    analyst_ratings: Dict[str, Any]


@dataclass
class PersonaAnalysis:
    """Analysis result from a single persona"""
    persona_name: str
    symbol: str
    recommendation: InvestmentAction
    confidence_score: float  # 0-100
    target_price: Optional[float]
    stop_loss: Optional[float]
    time_horizon: str
    reasoning: str
    key_factors: List[str]
    risk_assessment: Dict[str, Any]
    position_size_recommendation: float  # % of portfolio
    timestamp: datetime


@dataclass
class DebateMessage:
    """Message in a debate between personas"""
    persona_name: str
    message: str
    timestamp: datetime
    message_type: str  # "opening", "argument", "counter_argument", "consensus"
    supporting_data: Dict[str, Any]


@dataclass
class ConsensusResult:
    """Final consensus decision from all personas"""
    symbol: str
    final_recommendation: InvestmentAction
    consensus_score: float  # 0-100 (100 = unanimous agreement)
    average_confidence: float
    recommended_position_size: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    key_supporting_arguments: List[str]
    key_concerns: List[str]
    dissenting_opinions: List[str]
    market_conditions_assessment: str
    execution_timeline: str
    risk_level: str
    expected_return: float
    debate_summary: str
    timestamp: datetime


class NewsService:
    """Service for fetching and analyzing news"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

    def get_stock_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch recent news for a stock symbol"""
        try:
            if not self.api_key:
                # Fallback to mock news data
                return self._get_mock_news(symbol)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            url = f"{self.base_url}/everything"
            params = {
                'q': f'"{symbol}" OR "{symbol} stock"',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'apiKey': self.api_key,
                'language': 'en',
                'pageSize': 20
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('articles', [])

        except Exception as e:
            logger.warning(f"Failed to fetch news for {symbol}: {e}")
            return self._get_mock_news(symbol)

    def _get_mock_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate mock news data for testing"""
        return [
            {
                'title': f'{symbol} Reports Strong Quarterly Earnings',
                'description': f'{symbol} exceeded analyst expectations with strong revenue growth.',
                'publishedAt': datetime.now().isoformat(),
                'source': {'name': 'Financial Times'},
                'sentiment': 'positive'
            },
            {
                'title': f'Market Volatility Affects {symbol} Performance',
                'description': f'Recent market conditions have impacted {symbol} trading patterns.',
                'publishedAt': (datetime.now() - timedelta(days=1)).isoformat(),
                'source': {'name': 'Bloomberg'},
                'sentiment': 'neutral'
            }
        ]

    def analyze_sentiment(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall sentiment from news articles"""
        if not news_articles:
            return {'overall_sentiment': 'neutral', 'confidence': 0.5, 'article_count': 0}

        # Simple sentiment analysis (can be enhanced with NLP)
        positive_keywords = ['growth', 'earnings', 'profit', 'revenue', 'expansion', 'strong', 'beat', 'exceed']
        negative_keywords = ['loss', 'decline', 'fall', 'weak', 'miss', 'disappointing', 'concern', 'risk']

        positive_count = 0
        negative_count = 0

        for article in news_articles:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            positive_count += sum(1 for word in positive_keywords if word in text)
            negative_count += sum(1 for word in negative_keywords if word in text)

        if positive_count > negative_count * 1.2:
            sentiment = 'bullish'
        elif negative_count > positive_count * 1.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'

        confidence = min(abs(positive_count - negative_count) / max(positive_count + negative_count, 1), 1.0)

        return {
            'overall_sentiment': sentiment,
            'confidence': confidence,
            'positive_signals': positive_count,
            'negative_signals': negative_count,
            'article_count': len(news_articles)
        }


class MarketDataService:
    """Service for fetching market data and technical indicators"""

    def __init__(self):
        self.cache = {}
        self.cache_expiry = timedelta(minutes=15)

    def get_market_data(self, symbol: str) -> Optional[MarketAnalysis]:
        """Fetch comprehensive market data for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self.cache:
                cache_time, data = self.cache[cache_key]
                if datetime.now() - cache_time < self.cache_expiry:
                    return data

            ticker = yf.Ticker(symbol)

            # Get basic info
            info = ticker.info
            hist = ticker.history(period="6mo")

            if hist.empty:
                logger.warning(f"No historical data available for {symbol}")
                return None

            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            price_change = current_price - prev_close
            price_change_percent = (price_change / prev_close) * 100 if prev_close != 0 else 0

            # Technical indicators
            technical_indicators = self._calculate_technical_indicators(hist)

            # Fundamental data
            fundamentals = self._extract_fundamentals(info)

            analysis = MarketAnalysis(
                symbol=symbol,
                current_price=current_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                volume=int(hist['Volume'].iloc[-1]),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('forwardPE'),
                pb_ratio=info.get('priceToBook'),
                dividend_yield=info.get('dividendYield'),
                beta=info.get('beta'),
                technical_indicators=technical_indicators,
                fundamentals=fundamentals,
                news_sentiment={},
                analyst_ratings={}
            )

            # Cache the result
            self.cache[cache_key] = (datetime.now(), analysis)

            return analysis

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            return None

    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators from price data"""
        indicators = {}

        try:
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']

            # Moving averages
            indicators['sma_20'] = close.rolling(20).mean().iloc[-1]
            indicators['sma_50'] = close.rolling(50).mean().iloc[-1]
            indicators['sma_200'] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]

            # MACD
            exp1 = close.ewm(span=12).mean()
            exp2 = close.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_histogram'] = (macd - signal).iloc[-1]

            # Bollinger Bands
            bb_period = 20
            bb_std = close.rolling(bb_period).std()
            bb_middle = close.rolling(bb_period).mean()
            indicators['bb_upper'] = (bb_middle + 2 * bb_std).iloc[-1]
            indicators['bb_lower'] = (bb_middle - 2 * bb_std).iloc[-1]
            indicators['bb_position'] = (close.iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])

            # Volume indicators
            indicators['volume_sma_20'] = volume.rolling(20).mean().iloc[-1]
            indicators['volume_ratio'] = volume.iloc[-1] / indicators['volume_sma_20']

            # Volatility
            returns = close.pct_change()
            indicators['volatility_20d'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

            # Support and Resistance
            recent_high = high.tail(20).max()
            recent_low = low.tail(20).min()
            indicators['resistance_level'] = recent_high
            indicators['support_level'] = recent_low
            indicators['price_position'] = (close.iloc[-1] - recent_low) / (recent_high - recent_low)

        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {e}")

        return indicators

    def _extract_fundamentals(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fundamental data from ticker info"""
        fundamentals = {
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'revenue': info.get('totalRevenue'),
            'profit_margin': info.get('profitMargins'),
            'operating_margin': info.get('operatingMargins'),
            'return_on_equity': info.get('returnOnEquity'),
            'return_on_assets': info.get('returnOnAssets'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'book_value': info.get('bookValue'),
            'cash_per_share': info.get('totalCashPerShare'),
            'free_cash_flow': info.get('freeCashflow'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_sales': info.get('priceToSalesTrailing12Months'),
            'enterprise_to_revenue': info.get('enterpriseToRevenue'),
            'enterprise_to_ebitda': info.get('enterpriseToEbitda')
        }

        return {k: v for k, v in fundamentals.items() if v is not None}


class InvestorPersona(ABC):
    """Abstract base class for investor personas"""

    def __init__(self, characteristics: PersonaCharacteristics):
        self.characteristics = characteristics
        self.name = characteristics.name
        self.analysis_history = []

    @abstractmethod
    def analyze_investment(self, market_analysis: MarketAnalysis, news_sentiment: Dict[str, Any]) -> PersonaAnalysis:
        """Analyze an investment opportunity"""
        pass

    @abstractmethod
    def generate_debate_argument(self, analysis: PersonaAnalysis, opposing_views: List[PersonaAnalysis]) -> str:
        """Generate an argument for the debate"""
        pass

    def _calculate_confidence_score(self, factors: Dict[str, float]) -> float:
        """Calculate confidence score based on various factors"""
        weights = {
            'technical_alignment': 0.3,
            'fundamental_strength': 0.4,
            'news_sentiment': 0.2,
            'risk_reward_ratio': 0.1
        }

        score = sum(weights.get(factor, 0) * value for factor, value in factors.items())
        return max(0, min(100, score))


class WarrenBuffettPersona(InvestorPersona):
    """Warren Buffett investment persona - Value investing with focus on moats"""

    def __init__(self):
        characteristics = PersonaCharacteristics(
            name="Warren Buffett",
            description="The Oracle of Omaha - Value investing legend focused on businesses with strong moats",
            investment_style=InvestmentStyle.VALUE,
            risk_tolerance=RiskTolerance.CONSERVATIVE,
            time_horizon="long",
            key_principles=[
                "Buy wonderful businesses at fair prices",
                "Look for strong economic moats",
                "Invest in what you understand",
                "Think like a business owner",
                "Price is what you pay, value is what you get"
            ],
            preferred_sectors=["Consumer Goods", "Financial Services", "Insurance", "Utilities"],
            avoided_sectors=["Technology (traditionally)", "Biotechnology", "Mining"],
            decision_factors=[
                "Competitive advantages",
                "Predictable cash flows",
                "Strong management",
                "Reasonable valuation",
                "Long-term prospects"
            ],
            communication_style="Folksy wisdom with simple analogies",
            famous_quotes=[
                "Be fearful when others are greedy and greedy when others are fearful",
                "Time is the friend of the wonderful business",
                "Risk comes from not knowing what you're doing"
            ],
            historical_returns=20.1,  # Annual return
            max_position_size=0.4,  # 40% max position
            diversification_preference="Concentrated portfolio of high-conviction ideas"
        )
        super().__init__(characteristics)

    def analyze_investment(self, market_analysis: MarketAnalysis, news_sentiment: Dict[str, Any]) -> PersonaAnalysis:
        """Analyze investment through Buffett's value investing lens"""
        symbol = market_analysis.symbol
        fundamentals = market_analysis.fundamentals

        # Value factors
        value_score = 0
        reasoning_points = []

        # PE Ratio analysis
        pe_ratio = market_analysis.pe_ratio
        if pe_ratio and pe_ratio < 15:
            value_score += 20
            reasoning_points.append(f"Attractive P/E ratio of {pe_ratio:.1f}")
        elif pe_ratio and pe_ratio > 25:
            value_score -= 10
            reasoning_points.append(f"High P/E ratio of {pe_ratio:.1f} suggests overvaluation")

        # ROE analysis
        roe = fundamentals.get('return_on_equity')
        if roe and roe > 0.15:
            value_score += 25
            reasoning_points.append(f"Strong ROE of {roe*100:.1f}% indicates efficient management")
        elif roe and roe < 0.1:
            value_score -= 15
            reasoning_points.append(f"Low ROE of {roe*100:.1f}% raises concerns about profitability")

        # Debt analysis
        debt_to_equity = fundamentals.get('debt_to_equity')
        if debt_to_equity and debt_to_equity < 0.3:
            value_score += 15
            reasoning_points.append("Conservative debt levels provide financial stability")
        elif debt_to_equity and debt_to_equity > 1.0:
            value_score -= 20
            reasoning_points.append("High debt levels increase financial risk")

        # Current ratio
        current_ratio = fundamentals.get('current_ratio')
        if current_ratio and current_ratio > 1.5:
            value_score += 10
            reasoning_points.append("Strong liquidity position")

        # Revenue growth
        revenue_growth = fundamentals.get('revenue_growth')
        if revenue_growth and revenue_growth > 0.05:
            value_score += 15
            reasoning_points.append(f"Consistent revenue growth of {revenue_growth*100:.1f}%")

        # Technical factors (less important for Buffett)
        tech_indicators = market_analysis.technical_indicators
        if tech_indicators.get('sma_200') and market_analysis.current_price > tech_indicators['sma_200']:
            value_score += 5
            reasoning_points.append("Price above 200-day moving average shows long-term uptrend")

        # Determine recommendation
        if value_score >= 60:
            recommendation = InvestmentAction.BUY
        elif value_score >= 40:
            recommendation = InvestmentAction.HOLD
        else:
            recommendation = InvestmentAction.SELL

        confidence_factors = {
            'fundamental_strength': min(100, value_score),
            'technical_alignment': 20,  # Buffett cares less about technicals
            'news_sentiment': news_sentiment.get('confidence', 50) * 100,
            'risk_reward_ratio': 70 if value_score > 50 else 30
        }

        confidence_score = self._calculate_confidence_score(confidence_factors)

        # Calculate target price (simple P/E expansion model)
        target_price = None
        if pe_ratio and pe_ratio < 20:
            # Assume fair value P/E of 18
            earnings_per_share = market_analysis.current_price / pe_ratio if pe_ratio > 0 else None
            if earnings_per_share:
                target_price = earnings_per_share * 18

        reasoning = f"Value analysis reveals: {'. '.join(reasoning_points)}. " + \
                   f"This {'aligns' if value_score > 50 else 'conflicts'} with my investment principles of buying quality businesses at reasonable prices."

        return PersonaAnalysis(
            persona_name=self.name,
            symbol=symbol,
            recommendation=recommendation,
            confidence_score=confidence_score,
            target_price=target_price,
            stop_loss=market_analysis.current_price * 0.85 if recommendation in [InvestmentAction.BUY] else None,
            time_horizon="3-5 years",
            reasoning=reasoning,
            key_factors=reasoning_points,
            risk_assessment={
                'debt_risk': 'Low' if debt_to_equity and debt_to_equity < 0.5 else 'High',
                'valuation_risk': 'Low' if pe_ratio and pe_ratio < 20 else 'High',
                'business_quality': 'High' if roe and roe > 0.15 else 'Medium'
            },
            position_size_recommendation=min(0.2, max(0.05, value_score / 500)),
            timestamp=datetime.now()
        )

    def generate_debate_argument(self, analysis: PersonaAnalysis, opposing_views: List[PersonaAnalysis]) -> str:
        """Generate Buffett-style debate argument"""
        arguments = [
            f"When I look at {analysis.symbol}, I see the fundamentals that matter for long-term wealth creation.",
            f"My analysis shows {analysis.recommendation.value} with {analysis.confidence_score:.0f}% confidence.",
        ]

        if analysis.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY]:
            arguments.append("This company demonstrates the competitive advantages and predictable cash flows I look for.")
            arguments.append("Remember, time is the friend of the wonderful business, and short-term volatility creates opportunity.")

        # Address opposing views
        growth_opposition = [view for view in opposing_views if 'growth' in view.reasoning.lower()]
        if growth_opposition:
            arguments.append("While growth is exciting, I prefer paying a reasonable price for predictable returns over speculating on uncertain futures.")

        return " ".join(arguments)


class RayDalioPersona(InvestorPersona):
    """Ray Dalio investment persona - All Weather portfolio and macro analysis"""

    def __init__(self):
        characteristics = PersonaCharacteristics(
            name="Ray Dalio",
            description="Bridgewater founder - Macro investor focused on economic cycles and risk parity",
            investment_style=InvestmentStyle.MACRO,
            risk_tolerance=RiskTolerance.MODERATE,
            time_horizon="medium",
            key_principles=[
                "Diversification across asset classes and geographies",
                "Understanding economic cycles and debt cycles",
                "Risk parity approach",
                "Principled thinking and radical transparency",
                "All Weather portfolio construction"
            ],
            preferred_sectors=["All sectors with focus on balance"],
            avoided_sectors=["None - believes in diversification"],
            decision_factors=[
                "Economic cycle position",
                "Correlation with existing holdings",
                "Risk-adjusted returns",
                "Macro environment",
                "Currency considerations"
            ],
            communication_style="Systematic and principles-based",
            famous_quotes=[
                "He who lives by the crystal ball will eat shattered glass",
                "The biggest mistake investors make is to believe that what happened in the recent past is likely to persist",
                "Diversification is the Holy Grail of investing"
            ],
            historical_returns=12.8,
            max_position_size=0.15,  # More diversified approach
            diversification_preference="Broad diversification across asset classes"
        )
        super().__init__(characteristics)

    def analyze_investment(self, market_analysis: MarketAnalysis, news_sentiment: Dict[str, Any]) -> PersonaAnalysis:
        """Analyze through macro and risk parity lens"""
        symbol = market_analysis.symbol

        # Macro analysis
        macro_score = 0
        reasoning_points = []

        # Beta analysis for systematic risk
        beta = market_analysis.beta
        if beta and 0.8 <= beta <= 1.2:
            macro_score += 15
            reasoning_points.append(f"Beta of {beta:.2f} provides balanced market exposure")
        elif beta and beta > 1.5:
            macro_score -= 10
            reasoning_points.append(f"High beta of {beta:.2f} increases systematic risk")

        # Volatility analysis
        volatility = market_analysis.technical_indicators.get('volatility_20d')
        if volatility and volatility < 0.25:
            macro_score += 10
            reasoning_points.append("Low volatility supports risk parity principles")
        elif volatility and volatility > 0.4:
            macro_score -= 15
            reasoning_points.append("High volatility requires position size adjustment")

        # Sector diversification consideration
        sector = market_analysis.fundamentals.get('sector')
        if sector in ['Utilities', 'Consumer Staples', 'Healthcare']:
            macro_score += 10
            reasoning_points.append(f"{sector} sector provides defensive characteristics")
        elif sector in ['Technology', 'Communication Services']:
            macro_score += 5
            reasoning_points.append(f"{sector} sector offers growth potential with increased volatility")

        # Technical trend analysis
        tech_indicators = market_analysis.technical_indicators
        sma_20 = tech_indicators.get('sma_20')
        sma_50 = tech_indicators.get('sma_50')
        if sma_20 and sma_50 and sma_20 > sma_50:
            macro_score += 10
            reasoning_points.append("Short-term momentum supports medium-term position")

        # Risk-adjusted metrics
        if market_analysis.fundamentals.get('return_on_equity', 0) > 0.12:
            macro_score += 15
            reasoning_points.append("Strong risk-adjusted returns indicated by ROE")

        # News sentiment integration
        sentiment_score = news_sentiment.get('confidence', 0.5) * 100
        if news_sentiment.get('overall_sentiment') == 'bullish':
            macro_score += sentiment_score * 0.1
            reasoning_points.append("Positive sentiment supports near-term momentum")
        elif news_sentiment.get('overall_sentiment') == 'bearish':
            macro_score -= sentiment_score * 0.1
            reasoning_points.append("Negative sentiment creates potential contrarian opportunity")

        # Determine recommendation based on risk-adjusted analysis
        if macro_score >= 50:
            recommendation = InvestmentAction.BUY
        elif macro_score >= 25:
            recommendation = InvestmentAction.HOLD
        elif macro_score >= -25:
            recommendation = InvestmentAction.HOLD
        else:
            recommendation = InvestmentAction.SELL

        confidence_factors = {
            'fundamental_strength': 60,
            'technical_alignment': min(100, max(0, macro_score)),
            'news_sentiment': sentiment_score,
            'risk_reward_ratio': 80 - (volatility * 100 if volatility else 20)
        }

        confidence_score = self._calculate_confidence_score(confidence_factors)

        reasoning = f"Macro analysis considering economic cycles and risk factors: {'. '.join(reasoning_points)}. " + \
                   f"Position sizing should reflect correlation with existing portfolio holdings."

        return PersonaAnalysis(
            persona_name=self.name,
            symbol=symbol,
            recommendation=recommendation,
            confidence_score=confidence_score,
            target_price=None,  # Dalio focuses more on allocation than specific targets
            stop_loss=None,  # Uses portfolio-level risk management
            time_horizon="1-2 years",
            reasoning=reasoning,
            key_factors=reasoning_points,
            risk_assessment={
                'systematic_risk': f"Beta: {beta:.2f}" if beta else "Unknown",
                'volatility_risk': f"{volatility*100:.1f}% annualized" if volatility else "Unknown",
                'correlation_risk': "Requires portfolio analysis"
            },
            position_size_recommendation=min(0.1, max(0.02, macro_score / 1000)),  # More conservative sizing
            timestamp=datetime.now()
        )

    def generate_debate_argument(self, analysis: PersonaAnalysis, opposing_views: List[PersonaAnalysis]) -> str:
        """Generate Dalio-style systematic argument"""
        arguments = [
            f"My analysis of {analysis.symbol} focuses on how it fits within a diversified, all-weather portfolio approach.",
            f"The systematic risk factors and correlation characteristics suggest {analysis.recommendation.value}."
        ]

        if analysis.recommendation == InvestmentAction.HOLD:
            arguments.append("In uncertain environments, maintaining balanced exposure while monitoring risk factors is prudent.")

        # Address concentration risk from other personas
        concentrated_views = [view for view in opposing_views if view.position_size_recommendation > 0.15]
        if concentrated_views:
            arguments.append("While I respect high-conviction approaches, diversification remains the only free lunch in investing.")

        return " ".join(arguments)


class CathieWoodPersona(InvestorPersona):
    """Cathie Wood investment persona - Disruptive innovation focus"""

    def __init__(self):
        characteristics = PersonaCharacteristics(
            name="Cathie Wood",
            description="ARK Invest CEO - Focus on disruptive innovation and exponential growth",
            investment_style=InvestmentStyle.DISRUPTIVE,
            risk_tolerance=RiskTolerance.VERY_AGGRESSIVE,
            time_horizon="long",
            key_principles=[
                "Invest in disruptive innovation",
                "Focus on exponential growth technologies",
                "Long-term horizon for breakthrough technologies",
                "High conviction, concentrated positions",
                "Research-driven investment process"
            ],
            preferred_sectors=["Technology", "Healthcare", "Energy Storage", "Artificial Intelligence", "Genomics"],
            avoided_sectors=["Traditional Energy", "Old Economy", "Value Traps"],
            decision_factors=[
                "Disruptive potential",
                "Total addressable market",
                "Innovation moat",
                "Management vision",
                "Technology adoption curve"
            ],
            communication_style="Visionary and technology-focused",
            famous_quotes=[
                "Innovation is the key to growth",
                "We're investing in the companies that will define the future",
                "Disruption is not going to slow down"
            ],
            historical_returns=15.5,  # Highly variable, high risk/reward
            max_position_size=0.25,
            diversification_preference="Concentrated in disruptive themes"
        )
        super().__init__(characteristics)

    def analyze_investment(self, market_analysis: MarketAnalysis, news_sentiment: Dict[str, Any]) -> PersonaAnalysis:
        """Analyze through disruptive innovation lens"""
        symbol = market_analysis.symbol
        fundamentals = market_analysis.fundamentals

        # Innovation and disruption score
        innovation_score = 0
        reasoning_points = []

        # Sector preference
        sector = fundamentals.get('sector')
        if sector in ['Technology', 'Healthcare', 'Communication Services']:
            innovation_score += 30
            reasoning_points.append(f"{sector} sector aligns with disruptive innovation themes")
        elif sector in ['Energy', 'Industrials']:
            innovation_score += 15
            reasoning_points.append(f"{sector} sector has potential for technological disruption")
        elif sector in ['Utilities', 'Real Estate']:
            innovation_score -= 20
            reasoning_points.append(f"{sector} sector limited disruption potential")

        # Growth metrics
        revenue_growth = fundamentals.get('revenue_growth')
        if revenue_growth and revenue_growth > 0.2:
            innovation_score += 25
            reasoning_points.append(f"Exceptional revenue growth of {revenue_growth*100:.1f}% indicates market disruption")
        elif revenue_growth and revenue_growth > 0.1:
            innovation_score += 15
            reasoning_points.append(f"Strong revenue growth of {revenue_growth*100:.1f}% shows innovation traction")
        elif revenue_growth and revenue_growth < 0:
            innovation_score -= 15
            reasoning_points.append("Declining revenue raises concerns about competitive position")

        # R&D intensity (approximate from margins)
        operating_margin = fundamentals.get('operating_margin')
        if operating_margin and operating_margin < 0.1:
            innovation_score += 10
            reasoning_points.append("Lower margins may indicate heavy R&D investment in innovation")

        # Market opportunity (enterprise value as proxy)
        enterprise_value = fundamentals.get('enterprise_value')
        market_cap = fundamentals.get('market_cap')
        if enterprise_value and market_cap and enterprise_value > market_cap * 1.1:
            innovation_score += 10
            reasoning_points.append("Enterprise value premium suggests growth investment focus")

        # Technical momentum (important for growth stocks)
        tech_indicators = market_analysis.technical_indicators
        rsi = tech_indicators.get('rsi')
        if rsi and 30 <= rsi <= 70:
            innovation_score += 15
            reasoning_points.append("Balanced RSI supports sustainable momentum")
        elif rsi and rsi < 30:
            innovation_score += 20
            reasoning_points.append("Oversold conditions create opportunity in quality disruptors")

        # News sentiment particularly important for innovation stocks
        sentiment = news_sentiment.get('overall_sentiment')
        if sentiment == 'bullish':
            innovation_score += 20
            reasoning_points.append("Positive sentiment supports innovation narrative")
        elif sentiment == 'bearish':
            innovation_score -= 10
            reasoning_points.append("Negative sentiment creates contrarian opportunity")

        # Volatility acceptance (innovation stocks are volatile)
        volatility = tech_indicators.get('volatility_20d')
        if volatility and volatility > 0.3:
            reasoning_points.append("High volatility expected for disruptive innovation investments")

        # Determine recommendation
        if innovation_score >= 70:
            recommendation = InvestmentAction.STRONG_BUY
        elif innovation_score >= 50:
            recommendation = InvestmentAction.BUY
        elif innovation_score >= 20:
            recommendation = InvestmentAction.HOLD
        else:
            recommendation = InvestmentAction.SELL

        confidence_factors = {
            'fundamental_strength': min(100, max(0, innovation_score)),
            'technical_alignment': 70 if rsi and 40 <= rsi <= 60 else 40,
            'news_sentiment': news_sentiment.get('confidence', 50) * 100,
            'risk_reward_ratio': 90 if innovation_score > 60 else 50
        }

        confidence_score = self._calculate_confidence_score(confidence_factors)

        # Aggressive target price for disruptive companies
        target_price = None
        if recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY]:
            # Assume significant upside for successful disruptors
            target_price = market_analysis.current_price * 1.5

        reasoning = f"Disruptive innovation analysis: {'. '.join(reasoning_points)}. " + \
                   f"This company {'demonstrates' if innovation_score > 50 else 'lacks'} the characteristics of transformative technology investments."

        return PersonaAnalysis(
            persona_name=self.name,
            symbol=symbol,
            recommendation=recommendation,
            confidence_score=confidence_score,
            target_price=target_price,
            stop_loss=market_analysis.current_price * 0.75 if recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY] else None,
            time_horizon="3-7 years",
            reasoning=reasoning,
            key_factors=reasoning_points,
            risk_assessment={
                'disruption_risk': 'High - inherent in innovation investing',
                'execution_risk': 'High - technology development uncertain',
                'market_acceptance_risk': 'Medium to High',
                'competitive_risk': 'High - fast-moving markets'
            },
            position_size_recommendation=min(0.15, max(0.05, innovation_score / 600)),
            timestamp=datetime.now()
        )

    def generate_debate_argument(self, analysis: PersonaAnalysis, opposing_views: List[PersonaAnalysis]) -> str:
        """Generate innovation-focused debate argument"""
        arguments = [
            f"Looking at {analysis.symbol}, I see the potential for exponential returns through disruptive innovation.",
            f"My recommendation is {analysis.recommendation.value} based on the transformative potential I identify."
        ]

        if analysis.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY]:
            arguments.append("While traditional valuation metrics may seem high, disruptive companies create entirely new markets.")
            arguments.append("The risk of missing exponential growth far outweighs the risk of temporary volatility.")

        # Address value investor concerns
        value_opposition = [view for view in opposing_views if 'value' in view.reasoning.lower() or view.persona_name == "Warren Buffett"]
        if value_opposition:
            arguments.append("Traditional value metrics often miss the exponential potential of breakthrough technologies.")

        return " ".join(arguments)


class GeorgeSorosPersona(InvestorPersona):
    """George Soros investment persona - Reflexivity theory and macro bets"""

    def __init__(self):
        characteristics = PersonaCharacteristics(
            name="George Soros",
            description="The Man Who Broke the Bank of England - Reflexivity theory and large macro bets",
            investment_style=InvestmentStyle.MACRO,
            risk_tolerance=RiskTolerance.VERY_AGGRESSIVE,
            time_horizon="short",
            key_principles=[
                "Markets are always biased in one direction or another",
                "Reflexivity - market perceptions influence fundamentals",
                "Take large positions when conviction is high",
                "Markets are fallible and can be exploited",
                "Bet big when the odds are in your favor"
            ],
            preferred_sectors=["Currency markets", "Government bonds", "Index futures", "Commodities"],
            avoided_sectors=["Small cap stocks", "Individual stock picking"],
            decision_factors=[
                "Macro economic imbalances",
                "Market sentiment extremes",
                "Government policy changes",
                "Currency relationships",
                "Reflexive feedback loops"
            ],
            communication_style="Philosophical and contrarian",
            famous_quotes=[
                "The markets are always biased in one direction or another",
                "It's not whether you're right or wrong, but how much money you make when you're right",
                "I rely a great deal on animal instincts"
            ],
            historical_returns=30.5,  # Quantum Fund returns
            max_position_size=0.5,  # Very concentrated when confident
            diversification_preference="Concentrated macro bets"
        )
        super().__init__(characteristics)

    def analyze_investment(self, market_analysis: MarketAnalysis, news_sentiment: Dict[str, Any]) -> PersonaAnalysis:
        """Analyze through reflexivity and macro lens"""
        symbol = market_analysis.symbol

        # Macro reflexivity score
        reflexivity_score = 0
        reasoning_points = []

        # Market sentiment vs fundamentals divergence
        sentiment = news_sentiment.get('overall_sentiment', 'neutral')
        sentiment_confidence = news_sentiment.get('confidence', 0.5)

        pe_ratio = market_analysis.pe_ratio
        if pe_ratio:
            if sentiment == 'bullish' and pe_ratio > 25:
                reflexivity_score += 25
                reasoning_points.append("Market euphoria creating overvaluation - reflexive bubble forming")
            elif sentiment == 'bearish' and pe_ratio < 15:
                reflexivity_score += 20
                reasoning_points.append("Market pessimism creating undervaluation opportunity")

        # Technical momentum divergences
        tech_indicators = market_analysis.technical_indicators
        rsi = tech_indicators.get('rsi')
        if rsi:
            if rsi > 80:
                reflexivity_score += 15
                reasoning_points.append("Extreme overbought conditions signal reflexive excess")
            elif rsi < 20:
                reflexivity_score += 20
                reasoning_points.append("Extreme oversold creating contrarian opportunity")

        # Volume analysis for conviction
        volume_ratio = tech_indicators.get('volume_ratio')
        if volume_ratio and volume_ratio > 2:
            reflexivity_score += 15
            reasoning_points.append("High volume confirms strong market conviction")

        # Beta for systematic risk exposure
        beta = market_analysis.beta
        if beta and beta > 1.5:
            reflexivity_score += 10
            reasoning_points.append("High beta amplifies macro movements")

        # News sentiment momentum
        if sentiment_confidence > 0.7:
            if sentiment == 'bullish':
                reflexivity_score += 10
                reasoning_points.append("Strong positive sentiment supports momentum")
            else:
                reflexivity_score -= 5
                reasoning_points.append("Strong negative sentiment suggests caution")

        # Determine recommendation
        if reflexivity_score >= 60:
            recommendation = InvestmentAction.STRONG_BUY
        elif reflexivity_score >= 40:
            recommendation = InvestmentAction.BUY
        elif reflexivity_score >= -20:
            recommendation = InvestmentAction.HOLD
        elif reflexivity_score >= -40:
            recommendation = InvestmentAction.SELL
        else:
            recommendation = InvestmentAction.STRONG_SELL

        confidence_factors = {
            'fundamental_strength': 40,  # Less focus on fundamentals
            'technical_alignment': min(100, max(0, reflexivity_score + 20)),
            'news_sentiment': sentiment_confidence * 100,
            'risk_reward_ratio': 80 if abs(reflexivity_score) > 40 else 50
        }

        confidence_score = self._calculate_confidence_score(confidence_factors)

        reasoning = f"Reflexivity analysis reveals: {'. '.join(reasoning_points)}. " + \
                   f"Market bias and feedback loops {'support' if reflexivity_score > 30 else 'oppose'} this position."

        return PersonaAnalysis(
            persona_name=self.name,
            symbol=symbol,
            recommendation=recommendation,
            confidence_score=confidence_score,
            target_price=None,  # Soros focuses on momentum, not specific targets
            stop_loss=market_analysis.current_price * 0.9 if recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY] else None,
            time_horizon="3-6 months",
            reasoning=reasoning,
            key_factors=reasoning_points,
            risk_assessment={
                'reflexivity_risk': 'High - market perceptions can change rapidly',
                'macro_risk': 'High - sensitive to macro developments',
                'liquidity_risk': 'Medium - requires ability to exit quickly'
            },
            position_size_recommendation=min(0.3, max(0.05, abs(reflexivity_score) / 200)),
            timestamp=datetime.now()
        )

    def generate_debate_argument(self, analysis: PersonaAnalysis, opposing_views: List[PersonaAnalysis]) -> str:
        """Generate Soros-style reflexivity argument"""
        arguments = [
            f"The market's perception of {analysis.symbol} is creating a reflexive dynamic that I recommend we exploit.",
            f"My analysis suggests {analysis.recommendation.value} based on the bias currently embedded in market pricing."
        ]

        if analysis.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY]:
            arguments.append("When market sentiment and fundamentals align, reflexive feedback can create substantial momentum.")

        # Address value investor concerns
        value_views = [view for view in opposing_views if 'warren_buffett' in view.persona_name.lower()]
        if value_views:
            arguments.append("While fundamentals matter long-term, markets can remain irrational far longer than we can remain solvent.")

        return " ".join(arguments)


class JimSimonsPersona(InvestorPersona):
    """Jim Simons investment persona - Quantitative and statistical analysis"""

    def __init__(self):
        characteristics = PersonaCharacteristics(
            name="Jim Simons",
            description="Renaissance Technologies founder - Quantitative analysis and statistical arbitrage",
            investment_style=InvestmentStyle.QUANTITATIVE,
            risk_tolerance=RiskTolerance.MODERATE,
            time_horizon="short",
            key_principles=[
                "Data-driven decision making",
                "Statistical significance in patterns",
                "Risk management through diversification",
                "Mathematical models over intuition",
                "Systematic approach to market inefficiencies"
            ],
            preferred_sectors=["All sectors - model agnostic"],
            avoided_sectors=["None - purely quantitative approach"],
            decision_factors=[
                "Statistical significance",
                "Sharpe ratio optimization",
                "Correlation analysis",
                "Mean reversion patterns",
                "Momentum factors"
            ],
            communication_style="Mathematical and systematic",
            famous_quotes=[
                "The fundamental laws of mathematics and statistics are immutable",
                "We look for statistical inefficiencies",
                "Past performance is not indicative of future results, except when it is"
            ],
            historical_returns=66.1,  # Medallion Fund (gross)
            max_position_size=0.05,  # Highly diversified approach
            diversification_preference="Maximum diversification across uncorrelated strategies"
        )
        super().__init__(characteristics)

    def analyze_investment(self, market_analysis: MarketAnalysis, news_sentiment: Dict[str, Any]) -> PersonaAnalysis:
        """Analyze through quantitative statistical lens"""
        symbol = market_analysis.symbol

        # Statistical analysis score
        quant_score = 0
        reasoning_points = []

        # Technical indicators statistical analysis
        tech_indicators = market_analysis.technical_indicators

        # RSI mean reversion
        rsi = tech_indicators.get('rsi')
        if rsi:
            if rsi < 30:
                quant_score += 15
                reasoning_points.append(f"RSI {rsi:.1f} indicates oversold mean reversion opportunity")
            elif rsi > 70:
                quant_score -= 10
                reasoning_points.append(f"RSI {rsi:.1f} suggests overbought conditions")

        # Bollinger Band position
        bb_position = tech_indicators.get('bb_position')
        if bb_position is not None:
            if bb_position < 0.2:
                quant_score += 12
                reasoning_points.append("Price near lower Bollinger Band suggests statistical oversold")
            elif bb_position > 0.8:
                quant_score -= 8
                reasoning_points.append("Price near upper Bollinger Band indicates statistical overbought")

        # MACD momentum
        macd = tech_indicators.get('macd')
        macd_signal = tech_indicators.get('macd_signal')
        if macd and macd_signal:
            if macd > macd_signal:
                quant_score += 8
                reasoning_points.append("MACD bullish crossover indicates positive momentum")
            else:
                quant_score -= 5
                reasoning_points.append("MACD bearish signal suggests negative momentum")

        # Volume confirmation
        volume_ratio = tech_indicators.get('volume_ratio')
        if volume_ratio:
            if volume_ratio > 1.5:
                quant_score += 5
                reasoning_points.append("Above-average volume confirms price movement validity")

        # Volatility analysis
        volatility = tech_indicators.get('volatility_20d')
        if volatility:
            if volatility > 0.3:
                quant_score -= 5
                reasoning_points.append("High volatility increases risk-adjusted return penalty")
            elif volatility < 0.15:
                quant_score += 5
                reasoning_points.append("Low volatility improves risk-adjusted attractiveness")

        # Statistical significance of price movements
        price_change_percent = abs(market_analysis.price_change_percent)
        if price_change_percent > 5:
            quant_score += 8
            reasoning_points.append("Large price movement suggests statistical significance")

        # News sentiment as a factor
        sentiment_confidence = news_sentiment.get('confidence', 0.5)
        if sentiment_confidence > 0.7:
            if news_sentiment.get('overall_sentiment') == 'bullish':
                quant_score += 6
                reasoning_points.append("High-confidence positive sentiment factor")
            else:
                quant_score -= 4
                reasoning_points.append("High-confidence negative sentiment factor")

        # Determine recommendation based on statistical significance
        if quant_score >= 30:
            recommendation = InvestmentAction.BUY
        elif quant_score >= 15:
            recommendation = InvestmentAction.HOLD
        elif quant_score >= -15:
            recommendation = InvestmentAction.HOLD
        else:
            recommendation = InvestmentAction.SELL

        confidence_factors = {
            'fundamental_strength': 50,  # Moderate focus on fundamentals
            'technical_alignment': min(100, max(0, quant_score + 30)),
            'news_sentiment': sentiment_confidence * 80,
            'risk_reward_ratio': 70 - (volatility * 100 if volatility else 20)
        }

        confidence_score = self._calculate_confidence_score(confidence_factors)

        reasoning = f"Quantitative analysis: {'. '.join(reasoning_points)}. " + \
                   f"Statistical models {'support' if quant_score > 15 else 'do not support'} this position."

        return PersonaAnalysis(
            persona_name=self.name,
            symbol=symbol,
            recommendation=recommendation,
            confidence_score=confidence_score,
            target_price=None,  # Quantitative models focus on probabilities
            stop_loss=None,  # Uses portfolio-level risk management
            time_horizon="1-3 months",
            reasoning=reasoning,
            key_factors=reasoning_points,
            risk_assessment={
                'statistical_significance': f"Score: {quant_score}",
                'model_confidence': f"{confidence_score:.0f}%",
                'volatility_risk': f"{volatility*100:.1f}% annualized" if volatility else "Unknown"
            },
            position_size_recommendation=min(0.08, max(0.02, abs(quant_score) / 400)),  # Small, diversified positions
            timestamp=datetime.now()
        )

    def generate_debate_argument(self, analysis: PersonaAnalysis, opposing_views: List[PersonaAnalysis]) -> str:
        """Generate quantitative debate argument"""
        arguments = [
            f"My statistical models indicate {analysis.recommendation.value} for {analysis.symbol}.",
            f"The quantitative factors show {analysis.confidence_score:.0f}% confidence in this assessment."
        ]

        # Address qualitative arguments
        qualitative_views = [view for view in opposing_views if 'warren_buffett' in view.persona_name.lower()]
        if qualitative_views:
            arguments.append("While qualitative factors are important, mathematical models provide objective analysis free from cognitive bias.")

        return " ".join(arguments)


class PaulTudorJonesPersona(InvestorPersona):
    """Paul Tudor Jones investment persona - Technical analysis and risk management"""

    def __init__(self):
        characteristics = PersonaCharacteristics(
            name="Paul Tudor Jones",
            description="Tudor Investment Corporation founder - Technical analysis and superior risk management",
            investment_style=InvestmentStyle.MOMENTUM,
            risk_tolerance=RiskTolerance.AGGRESSIVE,
            time_horizon="short",
            key_principles=[
                "Technical analysis drives timing decisions",
                "Risk management is paramount",
                "Never average down on a losing position",
                "Adapt quickly to changing market conditions",
                "Focus on asymmetric risk/reward opportunities"
            ],
            preferred_sectors=["All sectors", "Commodities", "Currencies", "Interest Rate Products"],
            avoided_sectors=["None - trades all liquid markets"],
            decision_factors=[
                "Technical chart patterns",
                "Risk/reward ratios",
                "Market momentum",
                "Support and resistance levels",
                "Volume confirmation"
            ],
            communication_style="Risk-focused and tactical",
            famous_quotes=[
                "Where you want to be is always in control, never wishing, always trading",
                "The secret to being successful from a trading perspective is to have an indefatigable desire to succeed",
                "I believe the very best money is made at the market turns"
            ],
            historical_returns=19.5,  # Tudor Fund average
            max_position_size=0.2,
            diversification_preference="Tactical allocation based on technical setups"
        )
        super().__init__(characteristics)

    def analyze_investment(self, market_analysis: MarketAnalysis, news_sentiment: Dict[str, Any]) -> PersonaAnalysis:
        """Analyze through technical and risk management lens"""
        symbol = market_analysis.symbol

        # Technical momentum score
        technical_score = 0
        reasoning_points = []

        tech_indicators = market_analysis.technical_indicators
        current_price = market_analysis.current_price

        # Moving average analysis
        sma_20 = tech_indicators.get('sma_20')
        sma_50 = tech_indicators.get('sma_50')
        sma_200 = tech_indicators.get('sma_200')

        if sma_20 and sma_50:
            if sma_20 > sma_50:
                technical_score += 15
                reasoning_points.append("Short-term momentum positive (20 > 50 SMA)")
            else:
                technical_score -= 10
                reasoning_points.append("Short-term momentum negative (20 < 50 SMA)")

        if sma_200 and current_price > sma_200:
            technical_score += 10
            reasoning_points.append("Long-term uptrend intact (price > 200 SMA)")
        elif sma_200 and current_price < sma_200:
            technical_score -= 15
            reasoning_points.append("Long-term downtrend (price < 200 SMA)")

        # Support and resistance
        support_level = tech_indicators.get('support_level')
        resistance_level = tech_indicators.get('resistance_level')
        if support_level and resistance_level:
            distance_to_support = (current_price - support_level) / current_price
            distance_to_resistance = (resistance_level - current_price) / current_price

            if distance_to_support < 0.02:  # Within 2% of support
                technical_score += 20
                reasoning_points.append("Price near key support level - good risk/reward")
            elif distance_to_resistance < 0.02:  # Within 2% of resistance
                technical_score -= 15
                reasoning_points.append("Price near resistance - limited upside")

        # RSI for momentum
        rsi = tech_indicators.get('rsi')
        if rsi:
            if 40 <= rsi <= 60:
                technical_score += 10
                reasoning_points.append("RSI in neutral zone - sustainable momentum possible")
            elif rsi > 80:
                technical_score -= 20
                reasoning_points.append("RSI extremely overbought - reversal risk high")
            elif rsi < 20:
                technical_score += 15
                reasoning_points.append("RSI extremely oversold - reversal opportunity")

        # Volume confirmation
        volume_ratio = tech_indicators.get('volume_ratio')
        if volume_ratio and volume_ratio > 1.5:
            technical_score += 12
            reasoning_points.append("Strong volume confirms price movement")
        elif volume_ratio and volume_ratio < 0.7:
            technical_score -= 8
            reasoning_points.append("Weak volume questions sustainability")

        # Volatility for position sizing
        volatility = tech_indicators.get('volatility_20d')
        if volatility and volatility > 0.4:
            reasoning_points.append("High volatility requires reduced position size")
        elif volatility and volatility < 0.2:
            reasoning_points.append("Low volatility allows larger position size")

        # Risk/reward calculation
        if support_level and resistance_level:
            potential_gain = (resistance_level - current_price) / current_price
            potential_loss = (current_price - support_level) / current_price
            if potential_loss > 0:
                risk_reward_ratio = potential_gain / potential_loss
                if risk_reward_ratio > 2:
                    technical_score += 15
                    reasoning_points.append(f"Excellent risk/reward ratio: {risk_reward_ratio:.1f}")
                elif risk_reward_ratio < 1:
                    technical_score -= 10
                    reasoning_points.append(f"Poor risk/reward ratio: {risk_reward_ratio:.1f}")

        # Determine recommendation
        if technical_score >= 40:
            recommendation = InvestmentAction.STRONG_BUY
        elif technical_score >= 20:
            recommendation = InvestmentAction.BUY
        elif technical_score >= -10:
            recommendation = InvestmentAction.HOLD
        elif technical_score >= -30:
            recommendation = InvestmentAction.SELL
        else:
            recommendation = InvestmentAction.STRONG_SELL

        confidence_factors = {
            'fundamental_strength': 30,  # Less focus on fundamentals
            'technical_alignment': min(100, max(0, technical_score + 40)),
            'news_sentiment': news_sentiment.get('confidence', 50) * 60,
            'risk_reward_ratio': 90 if 'Excellent risk/reward' in reasoning_points else 50
        }

        confidence_score = self._calculate_confidence_score(confidence_factors)

        # Calculate stop loss
        stop_loss = None
        if recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY]:
            if support_level:
                stop_loss = support_level * 0.98  # Just below support
            else:
                stop_loss = current_price * 0.95  # 5% stop

        reasoning = f"Technical analysis: {'. '.join(reasoning_points)}. " + \
                   f"Risk management and momentum factors {'support' if technical_score > 15 else 'oppose'} this position."

        return PersonaAnalysis(
            persona_name=self.name,
            symbol=symbol,
            recommendation=recommendation,
            confidence_score=confidence_score,
            target_price=resistance_level if recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY] else None,
            stop_loss=stop_loss,
            time_horizon="2-8 weeks",
            reasoning=reasoning,
            key_factors=reasoning_points,
            risk_assessment={
                'technical_risk': 'Managed through stop losses',
                'momentum_risk': f"Volatility: {volatility*100:.1f}%" if volatility else "Unknown",
                'support_risk': f"Support at ${support_level:.2f}" if support_level else "No clear support"
            },
            position_size_recommendation=min(0.15, max(0.05, technical_score / 300)) if volatility and volatility < 0.3 else 0.05,
            timestamp=datetime.now()
        )

    def generate_debate_argument(self, analysis: PersonaAnalysis, opposing_views: List[PersonaAnalysis]) -> str:
        """Generate technical analysis debate argument"""
        arguments = [
            f"The technical setup for {analysis.symbol} clearly indicates {analysis.recommendation.value}.",
            f"Risk management principles and chart patterns support this view with {analysis.confidence_score:.0f}% confidence."
        ]

        if analysis.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY]:
            arguments.append("The risk/reward profile is asymmetric in our favor with clearly defined stop levels.")

        # Address fundamental concerns
        fundamental_views = [view for view in opposing_views if 'buffett' in view.persona_name.lower()]
        if fundamental_views:
            arguments.append("While fundamentals provide context, technical analysis determines optimal timing and risk management.")

        return " ".join(arguments)


# Initialize the personas
AVAILABLE_PERSONAS = {
    'warren_buffett': WarrenBuffettPersona(),
    'ray_dalio': RayDalioPersona(),
    'cathie_wood': CathieWoodPersona(),
    'george_soros': GeorgeSorosPersona(),
    'jim_simons': JimSimonsPersona(),
    'paul_tudor_jones': PaulTudorJonesPersona(),
}
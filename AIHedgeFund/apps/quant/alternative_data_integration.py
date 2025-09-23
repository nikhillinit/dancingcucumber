"""
Alternative Data Integration with Parallel Processing
====================================================
Integrate satellite, social, web scraping, and alternative data sources
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import ray
from concurrent.futures import ThreadPoolExecutor
import praw  # Reddit API
import tweepy  # Twitter API
from pytrends.request import TrendReq  # Google Trends
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
import re
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class AlternativeDataSignal:
    """Unified alternative data signal"""
    symbol: str
    data_source: str
    signal_type: str  # bullish, bearish, neutral
    confidence: float
    raw_value: float
    normalized_value: float
    metadata: Dict[str, Any]
    timestamp: datetime


class SocialMediaAgent(ray.remote):
    """Agent for social media sentiment analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.reddit = None
        self.twitter = None
        self.initialize_apis()

    def initialize_apis(self):
        """Initialize social media APIs"""
        try:
            # Reddit API
            self.reddit = praw.Reddit(
                client_id='YOUR_CLIENT_ID',
                client_secret='YOUR_SECRET',
                user_agent='AIHedgeFund/1.0'
            )
        except:
            logger.warning("Reddit API not configured")

        try:
            # Twitter API
            auth = tweepy.OAuthHandler('CONSUMER_KEY', 'CONSUMER_SECRET')
            auth.set_access_token('ACCESS_TOKEN', 'ACCESS_SECRET')
            self.twitter = tweepy.API(auth)
        except:
            logger.warning("Twitter API not configured")

    async def analyze_reddit_sentiment(
        self,
        symbol: str,
        subreddits: List[str] = ['wallstreetbets', 'stocks', 'investing']
    ) -> AlternativeDataSignal:
        """Analyze Reddit sentiment for a symbol"""
        if not self.reddit:
            return self._create_neutral_signal(symbol, 'reddit')

        mentions = 0
        sentiments = []

        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Search for symbol mentions
                for submission in subreddit.search(symbol, limit=100, time_filter='day'):
                    mentions += 1

                    # Analyze sentiment
                    text = f"{submission.title} {submission.selftext}"
                    sentiment = self._analyze_text_sentiment(text)
                    sentiments.append(sentiment)

                    # Analyze comments
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[:20]:
                        sentiment = self._analyze_text_sentiment(comment.body)
                        sentiments.append(sentiment)

            except Exception as e:
                logger.error(f"Reddit API error: {e}")

        # Aggregate sentiment
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            signal_type = 'bullish' if avg_sentiment > 0.1 else 'bearish' if avg_sentiment < -0.1 else 'neutral'
            confidence = min(mentions / 100, 1.0) * abs(avg_sentiment)
        else:
            avg_sentiment = 0
            signal_type = 'neutral'
            confidence = 0

        return AlternativeDataSignal(
            symbol=symbol,
            data_source='reddit',
            signal_type=signal_type,
            confidence=confidence,
            raw_value=avg_sentiment,
            normalized_value=np.tanh(avg_sentiment),
            metadata={'mentions': mentions, 'subreddits': subreddits},
            timestamp=datetime.now()
        )

    async def analyze_twitter_sentiment(self, symbol: str) -> AlternativeDataSignal:
        """Analyze Twitter sentiment for a symbol"""
        if not self.twitter:
            return self._create_neutral_signal(symbol, 'twitter')

        tweets = []
        sentiments = []

        try:
            # Search for tweets mentioning the symbol
            query = f"${symbol} OR #{symbol}"
            for tweet in tweepy.Cursor(
                self.twitter.search_tweets,
                q=query,
                lang="en",
                tweet_mode="extended"
            ).items(100):
                tweets.append(tweet.full_text)
                sentiment = self._analyze_text_sentiment(tweet.full_text)
                sentiments.append(sentiment)

        except Exception as e:
            logger.error(f"Twitter API error: {e}")

        # Aggregate sentiment
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            signal_type = 'bullish' if avg_sentiment > 0.1 else 'bearish' if avg_sentiment < -0.1 else 'neutral'
            confidence = min(len(tweets) / 100, 1.0) * abs(avg_sentiment)
        else:
            avg_sentiment = 0
            signal_type = 'neutral'
            confidence = 0

        return AlternativeDataSignal(
            symbol=symbol,
            data_source='twitter',
            signal_type=signal_type,
            confidence=confidence,
            raw_value=avg_sentiment,
            normalized_value=np.tanh(avg_sentiment),
            metadata={'tweet_count': len(tweets)},
            timestamp=datetime.now()
        )

    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)

            # Also check for specific bullish/bearish keywords
            bullish_words = ['moon', 'rocket', 'buy', 'calls', 'bullish', 'long', 'squeeze']
            bearish_words = ['puts', 'short', 'sell', 'bearish', 'crash', 'dump']

            text_lower = text.lower()
            bullish_count = sum(1 for word in bullish_words if word in text_lower)
            bearish_count = sum(1 for word in bearish_words if word in text_lower)

            keyword_sentiment = (bullish_count - bearish_count) / 10

            # Combine TextBlob and keyword sentiment
            return blob.sentiment.polarity * 0.7 + keyword_sentiment * 0.3

        except:
            return 0

    def _create_neutral_signal(self, symbol: str, source: str) -> AlternativeDataSignal:
        """Create neutral signal when API unavailable"""
        return AlternativeDataSignal(
            symbol=symbol,
            data_source=source,
            signal_type='neutral',
            confidence=0,
            raw_value=0,
            normalized_value=0,
            metadata={'error': 'API unavailable'},
            timestamp=datetime.now()
        )


class GoogleTrendsAgent(ray.remote):
    """Agent for Google Trends analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.pytrends = TrendReq(hl='en-US', tz=360)

    async def analyze_search_trends(
        self,
        symbol: str,
        company_name: str
    ) -> AlternativeDataSignal:
        """Analyze Google search trends"""
        try:
            # Build payload
            keywords = [symbol, company_name, f"{symbol} stock"]
            self.pytrends.build_payload(keywords, timeframe='now 7-d')

            # Get interest over time
            interest = self.pytrends.interest_over_time()

            if not interest.empty:
                # Calculate trend momentum
                recent_interest = interest[symbol].iloc[-24:].mean()  # Last 24 hours
                previous_interest = interest[symbol].iloc[-48:-24].mean()  # Previous 24 hours

                if previous_interest > 0:
                    trend_change = (recent_interest - previous_interest) / previous_interest
                else:
                    trend_change = 0

                # Related queries for additional insight
                related = self.pytrends.related_queries()
                rising_queries = related[symbol]['rising'] if symbol in related else None

                # Determine signal
                if trend_change > 0.2:
                    signal_type = 'bullish'
                elif trend_change < -0.2:
                    signal_type = 'bearish'
                else:
                    signal_type = 'neutral'

                confidence = min(abs(trend_change), 1.0)

                return AlternativeDataSignal(
                    symbol=symbol,
                    data_source='google_trends',
                    signal_type=signal_type,
                    confidence=confidence,
                    raw_value=trend_change,
                    normalized_value=np.tanh(trend_change),
                    metadata={
                        'recent_interest': recent_interest,
                        'rising_queries': rising_queries.head().to_dict() if rising_queries is not None else None
                    },
                    timestamp=datetime.now()
                )

        except Exception as e:
            logger.error(f"Google Trends error: {e}")

        return AlternativeDataSignal(
            symbol=symbol,
            data_source='google_trends',
            signal_type='neutral',
            confidence=0,
            raw_value=0,
            normalized_value=0,
            metadata={'error': str(e)},
            timestamp=datetime.now()
        )


class WebScrapingAgent(ray.remote):
    """Agent for web scraping and news analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.session = None

    async def scrape_financial_news(
        self,
        symbol: str,
        sources: List[str] = ['seekingalpha', 'benzinga', 'marketwatch']
    ) -> AlternativeDataSignal:
        """Scrape and analyze financial news"""
        all_headlines = []
        sentiments = []

        async with aiohttp.ClientSession() as session:
            for source in sources:
                headlines = await self._scrape_source(session, source, symbol)
                all_headlines.extend(headlines)

                # Analyze sentiment of each headline
                for headline in headlines:
                    sentiment = self._analyze_headline_sentiment(headline)
                    sentiments.append(sentiment)

        # Aggregate sentiment
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            std_sentiment = np.std(sentiments)

            # Check for consensus
            if std_sentiment < 0.2:  # High agreement
                confidence = 0.8
            else:
                confidence = 0.5

            if avg_sentiment > 0.1:
                signal_type = 'bullish'
            elif avg_sentiment < -0.1:
                signal_type = 'bearish'
            else:
                signal_type = 'neutral'
        else:
            avg_sentiment = 0
            signal_type = 'neutral'
            confidence = 0

        return AlternativeDataSignal(
            symbol=symbol,
            data_source='news_scraping',
            signal_type=signal_type,
            confidence=confidence,
            raw_value=avg_sentiment,
            normalized_value=np.tanh(avg_sentiment),
            metadata={'headline_count': len(all_headlines), 'sources': sources},
            timestamp=datetime.now()
        )

    async def _scrape_source(
        self,
        session: aiohttp.ClientSession,
        source: str,
        symbol: str
    ) -> List[str]:
        """Scrape headlines from specific source"""
        headlines = []

        try:
            if source == 'seekingalpha':
                url = f"https://seekingalpha.com/symbol/{symbol}"
            elif source == 'benzinga':
                url = f"https://www.benzinga.com/quote/{symbol}"
            elif source == 'marketwatch':
                url = f"https://www.marketwatch.com/investing/stock/{symbol}"
            else:
                return []

            async with session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Extract headlines (simplified - actual selectors would be more specific)
                    for heading in soup.find_all(['h1', 'h2', 'h3'])[:10]:
                        text = heading.get_text().strip()
                        if len(text) > 10:
                            headlines.append(text)

        except Exception as e:
            logger.error(f"Scraping error for {source}: {e}")

        return headlines

    def _analyze_headline_sentiment(self, headline: str) -> float:
        """Analyze sentiment of news headline"""
        # Financial-specific sentiment words
        very_bullish = ['surge', 'soar', 'rocket', 'breakout', 'rally']
        bullish = ['rise', 'gain', 'up', 'positive', 'beat', 'upgrade']
        bearish = ['fall', 'drop', 'down', 'negative', 'miss', 'downgrade']
        very_bearish = ['crash', 'plunge', 'collapse', 'tank', 'dump']

        headline_lower = headline.lower()

        # Check for strong signals
        for word in very_bullish:
            if word in headline_lower:
                return 0.8

        for word in very_bearish:
            if word in headline_lower:
                return -0.8

        # Check for moderate signals
        for word in bullish:
            if word in headline_lower:
                return 0.4

        for word in bearish:
            if word in headline_lower:
                return -0.4

        # Use TextBlob for neutral headlines
        try:
            return TextBlob(headline).sentiment.polarity
        except:
            return 0


class SatelliteDataAgent(ray.remote):
    """Agent for satellite and geospatial data analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def analyze_satellite_data(
        self,
        symbol: str,
        data_type: str = 'parking_lot'
    ) -> AlternativeDataSignal:
        """Analyze satellite imagery data (simulated)"""
        # In production, would integrate with providers like:
        # - Orbital Insight
        # - SpaceKnow
        # - RS Metrics

        # Simulated satellite data analysis
        if data_type == 'parking_lot':
            # Retail parking lot traffic
            traffic_change = np.random.normal(0, 0.1)  # Simulated

            if traffic_change > 0.05:
                signal_type = 'bullish'
            elif traffic_change < -0.05:
                signal_type = 'bearish'
            else:
                signal_type = 'neutral'

            confidence = min(abs(traffic_change) * 5, 0.9)

        elif data_type == 'shipping':
            # Port activity and shipping traffic
            shipping_change = np.random.normal(0, 0.15)  # Simulated

            if shipping_change > 0.1:
                signal_type = 'bullish'
            elif shipping_change < -0.1:
                signal_type = 'bearish'
            else:
                signal_type = 'neutral'

            confidence = min(abs(shipping_change) * 3, 0.8)
            traffic_change = shipping_change

        else:
            signal_type = 'neutral'
            confidence = 0
            traffic_change = 0

        return AlternativeDataSignal(
            symbol=symbol,
            data_source=f'satellite_{data_type}',
            signal_type=signal_type,
            confidence=confidence,
            raw_value=traffic_change,
            normalized_value=np.tanh(traffic_change),
            metadata={'data_type': data_type, 'measurement': 'traffic_change'},
            timestamp=datetime.now()
        )


class AlternativeDataOrchestrator:
    """Orchestrate multi-agent alternative data collection"""

    def __init__(self, n_agents: int = 4):
        ray.init(ignore_reinit_error=True)

        self.social_agent = SocialMediaAgent.remote("social")
        self.trends_agent = GoogleTrendsAgent.remote("trends")
        self.scraping_agent = WebScrapingAgent.remote("scraping")
        self.satellite_agent = SatelliteDataAgent.remote("satellite")

    async def collect_alternative_data(
        self,
        symbol: str,
        company_name: str = None
    ) -> Dict[str, AlternativeDataSignal]:
        """Collect all alternative data sources in parallel"""
        if not company_name:
            company_name = symbol  # Default to symbol if name not provided

        # Launch all agents in parallel
        tasks = [
            self.social_agent.analyze_reddit_sentiment.remote(symbol),
            self.social_agent.analyze_twitter_sentiment.remote(symbol),
            self.trends_agent.analyze_search_trends.remote(symbol, company_name),
            self.scraping_agent.scrape_financial_news.remote(symbol),
            self.satellite_agent.analyze_satellite_data.remote(symbol, 'parking_lot'),
            self.satellite_agent.analyze_satellite_data.remote(symbol, 'shipping'),
        ]

        # Gather results
        results = await asyncio.gather(*[
            asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            ) for task in tasks
        ])

        # Organize by source
        data_signals = {
            'reddit': results[0],
            'twitter': results[1],
            'google_trends': results[2],
            'news': results[3],
            'satellite_parking': results[4],
            'satellite_shipping': results[5],
        }

        # Calculate aggregate signal
        aggregate = self._calculate_aggregate_signal(data_signals)
        data_signals['aggregate'] = aggregate

        return data_signals

    def _calculate_aggregate_signal(
        self,
        signals: Dict[str, AlternativeDataSignal]
    ) -> AlternativeDataSignal:
        """Calculate weighted aggregate of all alternative data signals"""
        # Weights for different sources
        weights = {
            'reddit': 0.15,
            'twitter': 0.15,
            'google_trends': 0.20,
            'news': 0.30,
            'satellite_parking': 0.10,
            'satellite_shipping': 0.10
        }

        weighted_sum = 0
        total_weight = 0

        for source, signal in signals.items():
            if source in weights:
                weighted_sum += signal.normalized_value * weights[source] * signal.confidence
                total_weight += weights[source] * signal.confidence

        if total_weight > 0:
            aggregate_value = weighted_sum / total_weight
        else:
            aggregate_value = 0

        # Determine overall signal
        if aggregate_value > 0.1:
            signal_type = 'bullish'
        elif aggregate_value < -0.1:
            signal_type = 'bearish'
        else:
            signal_type = 'neutral'

        return AlternativeDataSignal(
            symbol=signals['reddit'].symbol,
            data_source='aggregate_alternative',
            signal_type=signal_type,
            confidence=min(total_weight, 1.0),
            raw_value=aggregate_value,
            normalized_value=aggregate_value,
            metadata={'sources': list(signals.keys())},
            timestamp=datetime.now()
        )

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of alternative data integration"""
    orchestrator = AlternativeDataOrchestrator()

    symbols = ['AAPL', 'TSLA', 'AMZN']
    company_names = ['Apple', 'Tesla', 'Amazon']

    for symbol, company in zip(symbols, company_names):
        print(f"\nAnalyzing {symbol} ({company})...")

        signals = await orchestrator.collect_alternative_data(symbol, company)

        print(f"Aggregate Signal: {signals['aggregate'].signal_type} "
              f"(confidence: {signals['aggregate'].confidence:.1%})")

        for source, signal in signals.items():
            if source != 'aggregate':
                print(f"  {source}: {signal.signal_type} ({signal.confidence:.1%})")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
"""
Async Sentiment Analyzer with Parallel API Processing
=====================================================
High-performance sentiment analysis using concurrent API calls
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from asyncio import Semaphore, gather, create_task
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from cachetools import TTLCache
import redis.asyncio as aioredis
from openai import AsyncOpenAI
import os
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Enhanced sentiment analysis result"""
    text: str
    sentiment: str  # positive, negative, neutral
    score: float  # -1 to 1
    confidence: float
    source: str
    symbol: str
    timestamp: datetime
    processing_time: float
    api_source: str
    metadata: Dict[str, Any]


class AsyncSentimentAnalyzer:
    """
    High-performance async sentiment analyzer
    Processes multiple sources concurrently
    """

    def __init__(
        self,
        max_concurrent_requests: int = 20,
        cache_ttl: int = 3600
    ):
        self.max_concurrent = max_concurrent_requests
        self.semaphore = Semaphore(max_concurrent_requests)

        # API clients
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Cache for results
        self.cache = TTLCache(maxsize=10000, ttl=cache_ttl)

        # Redis for distributed caching
        self.redis_client = None

        # Rate limiting
        self.rate_limits = {
            'openai': {'calls': 60, 'period': 60},  # 60 calls per minute
            'newsapi': {'calls': 100, 'period': 3600},  # 100 per hour
            'alpha_vantage': {'calls': 5, 'period': 60},  # 5 per minute
        }
        self.rate_trackers = {api: deque(maxlen=limits['calls'])
                              for api, limits in self.rate_limits.items()}

        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(f"Initialized AsyncSentimentAnalyzer with {max_concurrent_requests} concurrent requests")

    async def initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.create_redis_pool('redis://localhost:6379')
        except Exception as e:
            logger.warning(f"Redis not available: {e}")

    async def analyze_multiple_sources(
        self,
        symbol: str,
        sources: Dict[str, List[str]]
    ) -> Dict[str, List[SentimentResult]]:
        """
        Analyze sentiment from multiple sources concurrently
        sources: {'news': [...], 'social': [...], 'forums': [...]}
        """
        start_time = time.time()

        # Create tasks for each source
        tasks = []

        if 'news' in sources:
            for article in sources['news']:
                task = create_task(self._analyze_news(symbol, article))
                tasks.append(('news', task))

        if 'social' in sources:
            for post in sources['social']:
                task = create_task(self._analyze_social(symbol, post))
                tasks.append(('social', task))

        if 'forums' in sources:
            for thread in sources['forums']:
                task = create_task(self._analyze_forum(symbol, thread))
                tasks.append(('forums', task))

        # Execute all tasks concurrently
        results_with_types = await gather(*[task for _, task in tasks], return_exceptions=True)

        # Organize results by source type
        organized_results = {'news': [], 'social': [], 'forums': []}

        for (source_type, _), result in zip(tasks, results_with_types):
            if not isinstance(result, Exception):
                organized_results[source_type].append(result)
            else:
                logger.error(f"Error in {source_type} analysis: {result}")

        # Calculate aggregate sentiment
        aggregate = await self._calculate_aggregate_sentiment(organized_results)
        organized_results['aggregate'] = aggregate

        processing_time = time.time() - start_time
        logger.info(f"Analyzed {len(tasks)} sources in {processing_time:.2f}s")

        return organized_results

    async def _analyze_news(self, symbol: str, article: str) -> SentimentResult:
        """Analyze news article sentiment"""
        cache_key = hashlib.md5(f"{symbol}_{article[:100]}".encode()).hexdigest()

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        async with self.semaphore:
            # Rate limiting
            await self._check_rate_limit('openai')

            start_time = time.time()

            try:
                # Use OpenAI for sentiment analysis
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a financial sentiment analyzer. Respond with JSON only."},
                        {"role": "user", "content": f"Analyze sentiment for {symbol}: {article[:500]}. Return JSON with 'sentiment' (positive/negative/neutral), 'score' (-1 to 1), 'confidence' (0 to 1)."}
                    ],
                    temperature=0.1,
                    max_tokens=100
                )

                # Parse response
                result_text = response.choices[0].message.content
                result_json = json.loads(result_text)

                result = SentimentResult(
                    text=article[:200],
                    sentiment=result_json.get('sentiment', 'neutral'),
                    score=float(result_json.get('score', 0)),
                    confidence=float(result_json.get('confidence', 0.5)),
                    source='news',
                    symbol=symbol,
                    timestamp=datetime.now(),
                    processing_time=time.time() - start_time,
                    api_source='openai',
                    metadata={'model': 'gpt-3.5-turbo'}
                )

                # Cache result
                self.cache[cache_key] = result
                await self._cache_to_redis(cache_key, result)

                return result

            except Exception as e:
                logger.error(f"Error in news sentiment analysis: {e}")
                return self._create_neutral_sentiment(symbol, article, 'news')

    async def _analyze_social(self, symbol: str, post: str) -> SentimentResult:
        """Analyze social media sentiment"""
        async with self.semaphore:
            start_time = time.time()

            # Simplified sentiment analysis using keywords
            sentiment_score = await self._keyword_sentiment_analysis(post)

            if sentiment_score > 0.2:
                sentiment = 'positive'
            elif sentiment_score < -0.2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return SentimentResult(
                text=post[:200],
                sentiment=sentiment,
                score=sentiment_score,
                confidence=min(abs(sentiment_score) * 2, 0.9),
                source='social',
                symbol=symbol,
                timestamp=datetime.now(),
                processing_time=time.time() - start_time,
                api_source='keyword_analysis',
                metadata={'method': 'keyword'}
            )

    async def _analyze_forum(self, symbol: str, thread: str) -> SentimentResult:
        """Analyze forum discussion sentiment"""
        async with self.semaphore:
            start_time = time.time()

            # Use parallel keyword and pattern matching
            tasks = [
                self._keyword_sentiment_analysis(thread),
                self._pattern_sentiment_analysis(thread),
                self._emoji_sentiment_analysis(thread)
            ]

            scores = await gather(*tasks)
            avg_score = np.mean(scores)

            if avg_score > 0.2:
                sentiment = 'positive'
            elif avg_score < -0.2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            return SentimentResult(
                text=thread[:200],
                sentiment=sentiment,
                score=float(avg_score),
                confidence=0.7,
                source='forum',
                symbol=symbol,
                timestamp=datetime.now(),
                processing_time=time.time() - start_time,
                api_source='hybrid_analysis',
                metadata={'methods': ['keyword', 'pattern', 'emoji']}
            )

    async def _keyword_sentiment_analysis(self, text: str) -> float:
        """Fast keyword-based sentiment analysis"""
        positive_keywords = [
            'bullish', 'buy', 'long', 'moon', 'growth', 'profit', 'gain',
            'breakout', 'rally', 'surge', 'soar', 'strong', 'upgrade'
        ]
        negative_keywords = [
            'bearish', 'sell', 'short', 'crash', 'loss', 'decline', 'fall',
            'breakdown', 'plunge', 'dump', 'weak', 'downgrade', 'risk'
        ]

        text_lower = text.lower()

        # Run in executor for CPU-bound operation
        def count_keywords():
            pos_count = sum(1 for word in positive_keywords if word in text_lower)
            neg_count = sum(1 for word in negative_keywords if word in text_lower)
            total = pos_count + neg_count
            if total == 0:
                return 0
            return (pos_count - neg_count) / total

        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(self.executor, count_keywords)
        return score

    async def _pattern_sentiment_analysis(self, text: str) -> float:
        """Pattern-based sentiment analysis"""
        # Simplified pattern matching
        patterns = {
            r'going to (\d+)': 0.5,  # Price target mentions
            r'strong support': 0.3,
            r'resistance broken': 0.4,
            r'death cross': -0.5,
            r'bubble': -0.3,
        }

        score = 0
        import re
        for pattern, weight in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                score += weight

        return np.tanh(score)  # Normalize to -1, 1

    async def _emoji_sentiment_analysis(self, text: str) -> float:
        """Analyze sentiment from emojis"""
        positive_emojis = ['🚀', '💰', '📈', '✅', '💎', '🔥', '💪', '👍']
        negative_emojis = ['📉', '💩', '❌', '⚠️', '🐻', '👎', '😱', '💸']

        pos_count = sum(text.count(emoji) for emoji in positive_emojis)
        neg_count = sum(text.count(emoji) for emoji in negative_emojis)

        total = pos_count + neg_count
        if total == 0:
            return 0

        return (pos_count - neg_count) / total

    async def _calculate_aggregate_sentiment(
        self,
        results: Dict[str, List[SentimentResult]]
    ) -> SentimentResult:
        """Calculate weighted aggregate sentiment"""
        # Weights for different sources
        weights = {
            'news': 0.5,
            'social': 0.3,
            'forums': 0.2
        }

        weighted_scores = []
        total_weight = 0

        for source, source_results in results.items():
            if source in weights and source_results:
                source_weight = weights[source]
                avg_score = np.mean([r.score for r in source_results])
                weighted_scores.append(avg_score * source_weight)
                total_weight += source_weight

        if total_weight > 0:
            final_score = sum(weighted_scores) / total_weight
        else:
            final_score = 0

        if final_score > 0.2:
            sentiment = 'positive'
        elif final_score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return SentimentResult(
            text="Aggregate sentiment",
            sentiment=sentiment,
            score=float(final_score),
            confidence=min(total_weight, 0.9),
            source='aggregate',
            symbol=results.get('news', [{}])[0].symbol if results.get('news') else 'UNKNOWN',
            timestamp=datetime.now(),
            processing_time=0,
            api_source='aggregate',
            metadata={
                'n_sources': sum(len(v) for v in results.values()),
                'weights': weights
            }
        )

    async def _check_rate_limit(self, api: str):
        """Check and enforce rate limiting"""
        if api not in self.rate_limits:
            return

        limits = self.rate_limits[api]
        tracker = self.rate_trackers[api]

        now = time.time()

        # Remove old timestamps
        while tracker and now - tracker[0] > limits['period']:
            tracker.popleft()

        # Check if we're at the limit
        if len(tracker) >= limits['calls']:
            # Calculate wait time
            wait_time = limits['period'] - (now - tracker[0]) + 0.1
            if wait_time > 0:
                logger.info(f"Rate limit reached for {api}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Record this call
        tracker.append(now)

    async def _cache_to_redis(self, key: str, result: SentimentResult):
        """Cache result to Redis"""
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"sentiment:{key}",
                    3600,
                    json.dumps({
                        'sentiment': result.sentiment,
                        'score': result.score,
                        'confidence': result.confidence
                    })
                )
            except Exception as e:
                logger.debug(f"Redis caching failed: {e}")

    def _create_neutral_sentiment(
        self,
        symbol: str,
        text: str,
        source: str
    ) -> SentimentResult:
        """Create neutral sentiment result as fallback"""
        return SentimentResult(
            text=text[:200],
            sentiment='neutral',
            score=0,
            confidence=0.1,
            source=source,
            symbol=symbol,
            timestamp=datetime.now(),
            processing_time=0,
            api_source='fallback',
            metadata={'error': True}
        )

    async def fetch_live_news(
        self,
        symbols: List[str],
        n_articles: int = 10
    ) -> Dict[str, List[str]]:
        """Fetch live news for symbols concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                task = self._fetch_symbol_news(session, symbol, n_articles)
                tasks.append(task)

            results = await gather(*tasks)

            return dict(zip(symbols, results))

    async def _fetch_symbol_news(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        n_articles: int
    ) -> List[str]:
        """Fetch news for a single symbol"""
        # Example news API endpoint (replace with actual)
        url = f"https://api.example.com/news/{symbol}"

        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return [article['title'] + ' ' + article['description']
                            for article in data['articles'][:n_articles]]
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")

        # Return mock data for demo
        return [f"Mock news article {i} for {symbol}" for i in range(n_articles)]

    async def stream_sentiment_updates(
        self,
        symbols: List[str],
        interval: int = 60
    ):
        """Stream continuous sentiment updates"""
        while True:
            try:
                # Fetch fresh news
                news_data = await self.fetch_live_news(symbols)

                # Analyze sentiment for each symbol
                for symbol in symbols:
                    if symbol in news_data:
                        sources = {'news': news_data[symbol]}
                        results = await self.analyze_multiple_sources(symbol, sources)

                        # Publish to Redis pubsub
                        if self.redis_client:
                            await self.redis_client.publish(
                                f'sentiment:{symbol}',
                                json.dumps({
                                    'symbol': symbol,
                                    'sentiment': results['aggregate'].sentiment,
                                    'score': results['aggregate'].score,
                                    'timestamp': datetime.now().isoformat()
                                })
                            )

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in sentiment stream: {e}")
                await asyncio.sleep(10)


# Example usage
async def main():
    """Example usage of async sentiment analyzer"""
    analyzer = AsyncSentimentAnalyzer(max_concurrent_requests=20)
    await analyzer.initialize_redis()

    # Example data
    symbols = ['AAPL', 'TSLA', 'GOOGL']

    # Fetch and analyze news
    news_data = await analyzer.fetch_live_news(symbols, n_articles=5)

    # Analyze sentiment from multiple sources
    for symbol in symbols:
        sources = {
            'news': news_data.get(symbol, []),
            'social': [f"I think {symbol} is going to moon! 🚀"],
            'forums': [f"Technical analysis shows {symbol} breaking resistance"]
        }

        results = await analyzer.analyze_multiple_sources(symbol, sources)

        print(f"\n{symbol} Sentiment Analysis:")
        print(f"  Aggregate: {results['aggregate'].sentiment} "
              f"(score: {results['aggregate'].score:.2f})")

        for source_type, source_results in results.items():
            if source_type != 'aggregate' and source_results:
                avg_score = np.mean([r.score for r in source_results])
                print(f"  {source_type.capitalize()}: {avg_score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
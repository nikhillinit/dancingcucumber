"""
FinBERT Sentiment Analysis for Financial Text
============================================
Advanced NLP for market sentiment extraction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    BertForSequenceClassification
)
import yfinance as yf
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    scores: Dict[str, float]
    source: str
    timestamp: datetime
    symbol: Optional[str] = None


class FinBERTAnalyzer:
    """FinBERT-based sentiment analyzer for financial text"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self.load_model()

    def load_model(self):
        """Load pre-trained FinBERT model"""

        model_name = "ProsusAI/finbert"

        logger.info(f"Loading FinBERT model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

        # Create sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze sentiment of financial text"""

        # Truncate text if too long
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]

        # Get predictions
        results = self.sentiment_pipeline(text)

        # Parse results
        sentiment_scores = {r['label']: r['score'] for r in results}

        # Get dominant sentiment
        dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        confidence = sentiment_scores[dominant_sentiment]

        return SentimentResult(
            text=text,
            sentiment=dominant_sentiment.lower(),
            confidence=confidence,
            scores=sentiment_scores,
            source="finbert",
            timestamp=datetime.now()
        )

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts efficiently"""

        # Process in batches
        batch_size = 32
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Truncate texts
            batch = [t[:512] if len(t) > 512 else t for t in batch]

            # Get predictions
            batch_results = self.sentiment_pipeline(batch)

            # Parse results
            for text, result in zip(batch, batch_results):
                if isinstance(result, list):
                    result = result[0]

                all_results.append(
                    SentimentResult(
                        text=text,
                        sentiment=result['label'].lower(),
                        confidence=result['score'],
                        scores={result['label']: result['score']},
                        source="finbert",
                        timestamp=datetime.now()
                    )
                )

        return all_results

    def analyze_earnings_call(self, transcript: str) -> Dict:
        """Analyze earnings call transcript"""

        # Split into segments
        segments = transcript.split('\n\n')

        # Analyze each segment
        segment_sentiments = []
        for segment in segments:
            if len(segment.strip()) > 10:
                result = self.analyze_text(segment)
                segment_sentiments.append(result)

        # Calculate aggregate metrics
        positive_count = sum(1 for s in segment_sentiments if s.sentiment == 'positive')
        negative_count = sum(1 for s in segment_sentiments if s.sentiment == 'negative')
        neutral_count = sum(1 for s in segment_sentiments if s.sentiment == 'neutral')

        avg_confidence = np.mean([s.confidence for s in segment_sentiments])

        # Identify key topics
        key_topics = self._extract_key_topics(transcript)

        return {
            'overall_sentiment': self._calculate_overall_sentiment(segment_sentiments),
            'sentiment_distribution': {
                'positive': positive_count / len(segment_sentiments),
                'negative': negative_count / len(segment_sentiments),
                'neutral': neutral_count / len(segment_sentiments)
            },
            'average_confidence': avg_confidence,
            'segment_count': len(segment_sentiments),
            'key_topics': key_topics,
            'sentiment_timeline': [
                {
                    'segment': i,
                    'sentiment': s.sentiment,
                    'confidence': s.confidence
                }
                for i, s in enumerate(segment_sentiments)
            ]
        }

    def _calculate_overall_sentiment(self, sentiments: List[SentimentResult]) -> str:
        """Calculate overall sentiment from segments"""

        if not sentiments:
            return 'neutral'

        # Weight by confidence
        weighted_scores = {'positive': 0, 'negative': 0, 'neutral': 0}

        for s in sentiments:
            weighted_scores[s.sentiment] += s.confidence

        return max(weighted_scores, key=weighted_scores.get)

    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key financial topics from text"""

        key_terms = [
            'revenue', 'earnings', 'profit', 'margin', 'growth',
            'guidance', 'outlook', 'forecast', 'demand', 'supply',
            'competition', 'market share', 'innovation', 'investment',
            'debt', 'cash flow', 'dividend', 'buyback'
        ]

        text_lower = text.lower()
        found_topics = []

        for term in key_terms:
            if term in text_lower:
                count = text_lower.count(term)
                if count > 2:  # Only if mentioned multiple times
                    found_topics.append((term, count))

        # Sort by frequency
        found_topics.sort(key=lambda x: x[1], reverse=True)

        return [topic[0] for topic in found_topics[:5]]


class MultiSourceSentimentAggregator:
    """Aggregate sentiment from multiple sources"""

    def __init__(self):
        self.finbert = FinBERTAnalyzer()
        self.news_sources = [
            'reuters', 'bloomberg', 'wsj', 'ft', 'cnbc'
        ]

    async def fetch_news_sentiment(self, symbol: str) -> List[SentimentResult]:
        """Fetch and analyze news sentiment"""

        news_items = await self._fetch_news(symbol)

        # Analyze sentiment
        sentiments = []
        for item in news_items:
            result = self.finbert.analyze_text(item['text'])
            result.symbol = symbol
            result.source = item['source']
            sentiments.append(result)

        return sentiments

    async def _fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news from multiple sources"""

        # This would integrate with real news APIs
        # For demo, using placeholder
        return [
            {
                'text': f"Company {symbol} reports strong earnings growth",
                'source': 'reuters',
                'timestamp': datetime.now()
            }
        ]

    def analyze_social_media(self, symbol: str) -> Dict:
        """Analyze social media sentiment"""

        # Placeholder for social media integration
        # Would integrate with Twitter, Reddit, StockTwits APIs

        social_texts = [
            f"${symbol} looking bullish, great earnings",
            f"Concerned about {symbol} valuation",
            f"{symbol} innovation is impressive"
        ]

        sentiments = self.finbert.batch_analyze(social_texts)

        return {
            'overall_sentiment': self._aggregate_sentiments(sentiments),
            'platform_breakdown': {
                'twitter': 0.6,
                'reddit': 0.5,
                'stocktwits': 0.7
            },
            'volume': len(social_texts),
            'trending': True
        }

    def _aggregate_sentiments(self, sentiments: List[SentimentResult]) -> Dict:
        """Aggregate multiple sentiment results"""

        if not sentiments:
            return {'sentiment': 'neutral', 'confidence': 0.5}

        # Calculate weighted average
        total_weight = 0
        weighted_positive = 0
        weighted_negative = 0
        weighted_neutral = 0

        for s in sentiments:
            weight = s.confidence
            total_weight += weight

            if s.sentiment == 'positive':
                weighted_positive += weight
            elif s.sentiment == 'negative':
                weighted_negative += weight
            else:
                weighted_neutral += weight

        if total_weight == 0:
            return {'sentiment': 'neutral', 'confidence': 0.5}

        scores = {
            'positive': weighted_positive / total_weight,
            'negative': weighted_negative / total_weight,
            'neutral': weighted_neutral / total_weight
        }

        dominant = max(scores, key=scores.get)

        return {
            'sentiment': dominant,
            'confidence': scores[dominant],
            'scores': scores
        }


class SentimentTradingSignals:
    """Generate trading signals from sentiment analysis"""

    def __init__(self):
        self.analyzer = FinBERTAnalyzer()
        self.aggregator = MultiSourceSentimentAggregator()
        self.sentiment_history = {}

    def generate_signal(
        self,
        symbol: str,
        timeframe: str = 'daily'
    ) -> Dict:
        """Generate trading signal based on sentiment"""

        # Get current sentiment
        current_sentiment = self._get_current_sentiment(symbol)

        # Get historical sentiment
        historical_sentiment = self._get_historical_sentiment(symbol)

        # Calculate sentiment momentum
        sentiment_momentum = self._calculate_momentum(
            current_sentiment,
            historical_sentiment
        )

        # Generate signal
        signal = self._generate_trading_signal(
            current_sentiment,
            sentiment_momentum
        )

        return {
            'symbol': symbol,
            'signal': signal['action'],
            'confidence': signal['confidence'],
            'current_sentiment': current_sentiment,
            'sentiment_momentum': sentiment_momentum,
            'reasoning': signal['reasoning'],
            'timestamp': datetime.now()
        }

    def _get_current_sentiment(self, symbol: str) -> Dict:
        """Get current sentiment from multiple sources"""

        # Aggregate from multiple sources
        news_sentiment = asyncio.run(
            self.aggregator.fetch_news_sentiment(symbol)
        )

        social_sentiment = self.aggregator.analyze_social_media(symbol)

        # Combine sources
        all_sentiments = news_sentiment

        aggregated = self.aggregator._aggregate_sentiments(all_sentiments)
        aggregated['social'] = social_sentiment

        return aggregated

    def _get_historical_sentiment(self, symbol: str) -> List[Dict]:
        """Get historical sentiment data"""

        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = []

        return self.sentiment_history[symbol]

    def _calculate_momentum(
        self,
        current: Dict,
        historical: List[Dict]
    ) -> float:
        """Calculate sentiment momentum"""

        if len(historical) < 5:
            return 0.0

        # Get recent sentiments
        recent = historical[-5:]

        # Convert to numerical scores
        def sentiment_to_score(s):
            if s == 'positive':
                return 1.0
            elif s == 'negative':
                return -1.0
            else:
                return 0.0

        recent_scores = [sentiment_to_score(h['sentiment']) for h in recent]
        current_score = sentiment_to_score(current['sentiment'])

        # Calculate momentum
        avg_recent = np.mean(recent_scores)
        momentum = current_score - avg_recent

        return momentum

    def _generate_trading_signal(
        self,
        sentiment: Dict,
        momentum: float
    ) -> Dict:
        """Generate trading signal from sentiment data"""

        signal = {
            'action': 'hold',
            'confidence': 0.5,
            'reasoning': ''
        }

        # Strong positive sentiment with positive momentum
        if sentiment['sentiment'] == 'positive' and momentum > 0.3:
            signal['action'] = 'buy'
            signal['confidence'] = min(sentiment['confidence'] * 1.2, 0.95)
            signal['reasoning'] = 'Strong positive sentiment with increasing momentum'

        # Strong negative sentiment with negative momentum
        elif sentiment['sentiment'] == 'negative' and momentum < -0.3:
            signal['action'] = 'sell'
            signal['confidence'] = min(sentiment['confidence'] * 1.2, 0.95)
            signal['reasoning'] = 'Strong negative sentiment with declining momentum'

        # Sentiment reversal signals
        elif sentiment['sentiment'] == 'positive' and momentum < -0.5:
            signal['action'] = 'buy'
            signal['confidence'] = sentiment['confidence'] * 0.8
            signal['reasoning'] = 'Sentiment reversal from negative to positive'

        elif sentiment['sentiment'] == 'negative' and momentum > 0.5:
            signal['action'] = 'sell'
            signal['confidence'] = sentiment['confidence'] * 0.8
            signal['reasoning'] = 'Sentiment reversal from positive to negative'

        # Neutral or uncertain
        else:
            signal['action'] = 'hold'
            signal['confidence'] = 0.5
            signal['reasoning'] = 'Neutral or mixed sentiment signals'

        return signal


# Integration with AI personas
class SentimentEnhancedPersona:
    """Enhance AI personas with sentiment analysis"""

    def __init__(self):
        self.sentiment_analyzer = FinBERTAnalyzer()
        self.signal_generator = SentimentTradingSignals()

    def enhance_decision(
        self,
        persona_decision: Dict,
        symbol: str
    ) -> Dict:
        """Enhance persona decision with sentiment"""

        # Get sentiment signal
        sentiment_signal = self.signal_generator.generate_signal(symbol)

        # Combine with persona decision
        if sentiment_signal['confidence'] > 0.7:
            # High confidence sentiment overrides
            if sentiment_signal['signal'] == persona_decision['action']:
                # Agreement strengthens signal
                persona_decision['confidence'] *= 1.2
                persona_decision['reasoning'] += f" Confirmed by {sentiment_signal['reasoning']}"
            else:
                # Disagreement requires reconciliation
                if sentiment_signal['confidence'] > persona_decision['confidence']:
                    persona_decision['action'] = sentiment_signal['signal']
                    persona_decision['reasoning'] = f"Sentiment override: {sentiment_signal['reasoning']}"

        persona_decision['sentiment_analysis'] = sentiment_signal

        return persona_decision
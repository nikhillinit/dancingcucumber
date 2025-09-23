"""
AI Investment Orchestrator
==========================

The main orchestrator that coordinates AI investor personas, conducts debates,
and produces final investment recommendations with comprehensive analysis.

Author: AI Hedge Fund Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import os
from pathlib import Path

from .ai_investor_personas import (
    InvestorPersona, PersonaAnalysis, MarketAnalysis, NewsService,
    MarketDataService, AVAILABLE_PERSONAS, InvestmentAction
)
from .ai_debate_framework import (
    DebateModerator, ConsensusBuilder, DebateConfig, DebateRound,
    ConsensusResult
)

logger = logging.getLogger(__name__)


@dataclass
class InvestmentRequest:
    """Request for investment analysis"""
    symbols: List[str]
    personas: List[str]  # Keys from AVAILABLE_PERSONAS
    analysis_depth: str = "standard"  # "quick", "standard", "comprehensive"
    include_debate: bool = True
    include_news: bool = True
    portfolio_context: Optional[Dict[str, Any]] = None
    risk_tolerance: str = "moderate"  # "conservative", "moderate", "aggressive"
    investment_horizon: str = "medium"  # "short", "medium", "long"
    max_position_size: float = 0.2  # Maximum position size as % of portfolio


@dataclass
class InvestmentRecommendation:
    """Final investment recommendation with full analysis"""
    symbol: str
    recommendation: InvestmentAction
    consensus_score: float
    confidence_score: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_size: float
    expected_return: float
    risk_level: str
    time_horizon: str
    execution_timeline: str

    # Supporting analysis
    persona_analyses: List[PersonaAnalysis]
    debate_summary: str
    market_analysis: MarketAnalysis
    news_sentiment: Dict[str, Any]
    consensus_details: ConsensusResult

    # Timestamps and metadata
    analysis_timestamp: datetime
    data_freshness: Dict[str, datetime]
    analysis_duration: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)

        # Convert enums to strings
        result['recommendation'] = self.recommendation.value

        # Convert datetime objects
        result['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        result['data_freshness'] = {
            k: v.isoformat() for k, v in self.data_freshness.items()
        }

        # Convert persona analyses
        result['persona_analyses'] = [
            {
                **asdict(analysis),
                'recommendation': analysis.recommendation.value,
                'timestamp': analysis.timestamp.isoformat()
            }
            for analysis in self.persona_analyses
        ]

        # Convert consensus details
        if self.consensus_details:
            consensus_dict = asdict(self.consensus_details)
            consensus_dict['final_recommendation'] = self.consensus_details.final_recommendation.value
            consensus_dict['timestamp'] = self.consensus_details.timestamp.isoformat()
            result['consensus_details'] = consensus_dict

        return result


class InvestmentOrchestrator:
    """Main orchestrator for AI investment analysis"""

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 news_api_key: Optional[str] = None,
                 cache_duration_minutes: int = 15,
                 max_concurrent_analyses: int = 5):
        """
        Initialize the investment orchestrator

        Args:
            openai_api_key: OpenAI API key for AI-powered analysis
            news_api_key: News API key for sentiment analysis
            cache_duration_minutes: Duration to cache market data
            max_concurrent_analyses: Maximum concurrent persona analyses
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')

        # Initialize services
        self.market_service = MarketDataService()
        self.news_service = NewsService(self.news_api_key)

        # Configuration
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.max_concurrent = max_concurrent_analyses

        # Cache for analyses
        self.analysis_cache = {}
        self.market_data_cache = {}

        logger.info("Investment Orchestrator initialized")

    async def analyze_investment(self, request: InvestmentRequest) -> List[InvestmentRecommendation]:
        """
        Perform comprehensive investment analysis for requested symbols

        Args:
            request: Investment analysis request

        Returns:
            List of investment recommendations
        """
        start_time = time.time()
        recommendations = []

        try:
            logger.info(f"Starting investment analysis for {len(request.symbols)} symbols "
                       f"with {len(request.personas)} personas")

            # Validate request
            self._validate_request(request)

            # Process each symbol
            for symbol in request.symbols:
                try:
                    recommendation = await self._analyze_single_symbol(symbol, request)
                    if recommendation:
                        recommendations.append(recommendation)
                        logger.info(f"Completed analysis for {symbol}: "
                                  f"{recommendation.recommendation.value} "
                                  f"(Consensus: {recommendation.consensus_score:.0f}%)")
                    else:
                        logger.warning(f"Failed to generate recommendation for {symbol}")

                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in investment analysis: {e}")
            raise

        finally:
            duration = time.time() - start_time
            logger.info(f"Investment analysis completed in {duration:.2f} seconds")

        return recommendations

    async def _analyze_single_symbol(self, symbol: str,
                                   request: InvestmentRequest) -> Optional[InvestmentRecommendation]:
        """Analyze a single symbol with full persona debate"""

        analysis_start = time.time()

        try:
            # Step 1: Gather market data
            logger.debug(f"Fetching market data for {symbol}")
            market_analysis = await self._get_market_data(symbol)
            if not market_analysis:
                logger.warning(f"No market data available for {symbol}")
                return None

            # Step 2: Gather news sentiment
            news_sentiment = {}
            if request.include_news:
                logger.debug(f"Analyzing news sentiment for {symbol}")
                news_sentiment = await self._get_news_sentiment(symbol)

            # Step 3: Run persona analyses in parallel
            logger.debug(f"Running persona analyses for {symbol}")
            persona_analyses = await self._run_persona_analyses(
                symbol, market_analysis, news_sentiment, request.personas
            )

            if not persona_analyses:
                logger.warning(f"No persona analyses completed for {symbol}")
                return None

            # Step 4: Conduct debate (if requested)
            debate_rounds = []
            if request.include_debate and len(persona_analyses) > 1:
                logger.debug(f"Conducting investment debate for {symbol}")
                debate_rounds = await self._conduct_debate(
                    symbol, request.personas, persona_analyses
                )

            # Step 5: Build consensus
            logger.debug(f"Building consensus for {symbol}")
            consensus = await self._build_consensus(
                symbol, persona_analyses, debate_rounds, market_analysis
            )

            # Step 6: Apply portfolio constraints
            final_recommendation = self._apply_portfolio_constraints(
                consensus, request
            )

            # Step 7: Create final recommendation
            analysis_duration = time.time() - analysis_start

            recommendation = InvestmentRecommendation(
                symbol=symbol,
                recommendation=final_recommendation.final_recommendation,
                consensus_score=final_recommendation.consensus_score,
                confidence_score=final_recommendation.average_confidence,
                target_price=final_recommendation.target_price,
                stop_loss=final_recommendation.stop_loss,
                position_size=min(final_recommendation.recommended_position_size,
                                request.max_position_size),
                expected_return=final_recommendation.expected_return,
                risk_level=final_recommendation.risk_level,
                time_horizon=final_recommendation.execution_timeline,
                execution_timeline=final_recommendation.execution_timeline,

                # Supporting data
                persona_analyses=persona_analyses,
                debate_summary=final_recommendation.debate_summary,
                market_analysis=market_analysis,
                news_sentiment=news_sentiment,
                consensus_details=final_recommendation,

                # Metadata
                analysis_timestamp=datetime.now(),
                data_freshness={
                    'market_data': datetime.now(),
                    'news_data': datetime.now()
                },
                analysis_duration=analysis_duration
            )

            return recommendation

        except Exception as e:
            logger.error(f"Error in single symbol analysis for {symbol}: {e}")
            return None

    async def _get_market_data(self, symbol: str) -> Optional[MarketAnalysis]:
        """Get market data with caching"""

        cache_key = f"market_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"

        # Check cache
        if cache_key in self.market_data_cache:
            cache_time, data = self.market_data_cache[cache_key]
            if datetime.now() - cache_time < self.cache_duration:
                return data

        # Fetch fresh data
        try:
            data = self.market_service.get_market_data(symbol)
            if data:
                self.market_data_cache[cache_key] = (datetime.now(), data)
            return data
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            return None

    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment analysis"""

        try:
            # Fetch news articles
            news_articles = self.news_service.get_stock_news(symbol, days=7)

            # Analyze sentiment
            sentiment_analysis = self.news_service.analyze_sentiment(news_articles)

            return {
                **sentiment_analysis,
                'articles_analyzed': len(news_articles),
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"Failed to analyze news sentiment for {symbol}: {e}")
            return {
                'overall_sentiment': 'neutral',
                'confidence': 0.5,
                'articles_analyzed': 0
            }

    async def _run_persona_analyses(self, symbol: str, market_analysis: MarketAnalysis,
                                   news_sentiment: Dict[str, Any],
                                   persona_keys: List[str]) -> List[PersonaAnalysis]:
        """Run persona analyses in parallel"""

        analyses = []

        # Filter to available personas
        valid_personas = [key for key in persona_keys if key in AVAILABLE_PERSONAS]

        if not valid_personas:
            logger.warning("No valid personas specified")
            return analyses

        # Run analyses concurrently
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_persona = {
                executor.submit(
                    self._run_single_persona_analysis,
                    AVAILABLE_PERSONAS[persona_key],
                    market_analysis,
                    news_sentiment
                ): persona_key
                for persona_key in valid_personas
            }

            for future in as_completed(future_to_persona):
                persona_key = future_to_persona[future]
                try:
                    analysis = future.result(timeout=30)  # 30 second timeout
                    if analysis:
                        analyses.append(analysis)
                        logger.debug(f"Completed {persona_key} analysis: "
                                   f"{analysis.recommendation.value}")
                except Exception as e:
                    logger.error(f"Persona analysis failed for {persona_key}: {e}")

        return analyses

    def _run_single_persona_analysis(self, persona: InvestorPersona,
                                   market_analysis: MarketAnalysis,
                                   news_sentiment: Dict[str, Any]) -> Optional[PersonaAnalysis]:
        """Run analysis for a single persona"""

        try:
            analysis = persona.analyze_investment(market_analysis, news_sentiment)
            return analysis
        except Exception as e:
            logger.error(f"Error in {persona.name} analysis: {e}")
            return None

    async def _conduct_debate(self, symbol: str, persona_keys: List[str],
                            analyses: List[PersonaAnalysis]) -> List[DebateRound]:
        """Conduct debate between personas"""

        try:
            # Configure debate
            debate_config = DebateConfig(
                max_rounds=3,
                max_participants=len(persona_keys),
                time_limit_minutes=10,
                consensus_threshold=0.7,
                openai_api_key=self.openai_api_key,
                use_ai_moderation=bool(self.openai_api_key)
            )

            # Create moderator
            moderator = DebateModerator(debate_config)

            # Conduct debate
            debate_rounds = await moderator.moderate_debate(
                symbol, persona_keys, analyses
            )

            return debate_rounds

        except Exception as e:
            logger.error(f"Error conducting debate for {symbol}: {e}")
            return []

    async def _build_consensus(self, symbol: str, analyses: List[PersonaAnalysis],
                             debate_rounds: List[DebateRound],
                             market_analysis: MarketAnalysis) -> ConsensusResult:
        """Build final consensus"""

        try:
            # Configure consensus builder
            config = DebateConfig(openai_api_key=self.openai_api_key)
            consensus_builder = ConsensusBuilder(config)

            # Build consensus
            consensus = consensus_builder.build_consensus(
                symbol, analyses, debate_rounds, market_analysis
            )

            return consensus

        except Exception as e:
            logger.error(f"Error building consensus for {symbol}: {e}")
            # Return fallback consensus
            return self._create_fallback_consensus(symbol, analyses)

    def _apply_portfolio_constraints(self, consensus: ConsensusResult,
                                   request: InvestmentRequest) -> ConsensusResult:
        """Apply portfolio-level constraints to consensus"""

        # Adjust position size based on risk tolerance
        risk_multipliers = {
            'conservative': 0.5,
            'moderate': 1.0,
            'aggressive': 1.5
        }

        multiplier = risk_multipliers.get(request.risk_tolerance, 1.0)

        # Adjust based on investment horizon
        horizon_multipliers = {
            'short': 0.8,
            'medium': 1.0,
            'long': 1.2
        }

        horizon_multiplier = horizon_multipliers.get(request.investment_horizon, 1.0)

        # Apply adjustments
        adjusted_size = consensus.recommended_position_size * multiplier * horizon_multiplier
        adjusted_size = min(adjusted_size, request.max_position_size)

        # Create adjusted consensus
        consensus.recommended_position_size = adjusted_size

        return consensus

    def _create_fallback_consensus(self, symbol: str,
                                 analyses: List[PersonaAnalysis]) -> ConsensusResult:
        """Create basic consensus when full processing fails"""

        from .ai_debate_framework import ConsensusResult

        if not analyses:
            return ConsensusResult(
                symbol=symbol,
                final_recommendation=InvestmentAction.HOLD,
                consensus_score=50.0,
                average_confidence=50.0,
                recommended_position_size=0.05,
                target_price=None,
                stop_loss=None,
                key_supporting_arguments=["Insufficient data for analysis"],
                key_concerns=["Analysis incomplete"],
                dissenting_opinions=[],
                market_conditions_assessment="Unable to assess",
                execution_timeline="Hold pending further analysis",
                risk_level="Unknown",
                expected_return=0.0,
                debate_summary="Analysis failed - using fallback",
                timestamp=datetime.now()
            )

        # Simple majority vote
        recommendations = [analysis.recommendation for analysis in analyses]
        from collections import Counter
        vote_counts = Counter(recommendations)
        final_recommendation = vote_counts.most_common(1)[0][0]
        consensus_score = (vote_counts[final_recommendation] / len(recommendations)) * 100

        return ConsensusResult(
            symbol=symbol,
            final_recommendation=final_recommendation,
            consensus_score=consensus_score,
            average_confidence=np.mean([a.confidence_score for a in analyses]),
            recommended_position_size=np.mean([a.position_size_recommendation for a in analyses]),
            target_price=None,
            stop_loss=None,
            key_supporting_arguments=[f"Majority vote: {final_recommendation.value}"],
            key_concerns=["Limited consensus analysis"],
            dissenting_opinions=[],
            market_conditions_assessment="Basic analysis",
            execution_timeline="Standard",
            risk_level="Moderate",
            expected_return=0.05,
            debate_summary=f"Fallback consensus: {consensus_score:.0f}% agreement",
            timestamp=datetime.now()
        )

    def _validate_request(self, request: InvestmentRequest) -> None:
        """Validate investment request"""

        if not request.symbols:
            raise ValueError("No symbols provided")

        if not request.personas:
            raise ValueError("No personas specified")

        invalid_personas = [p for p in request.personas if p not in AVAILABLE_PERSONAS]
        if invalid_personas:
            raise ValueError(f"Invalid personas: {invalid_personas}")

        if request.max_position_size <= 0 or request.max_position_size > 1:
            raise ValueError("max_position_size must be between 0 and 1")

    def get_available_personas(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available personas"""

        persona_info = {}
        for key, persona in AVAILABLE_PERSONAS.items():
            characteristics = persona.characteristics
            persona_info[key] = {
                'name': characteristics.name,
                'description': characteristics.description,
                'investment_style': characteristics.investment_style.value,
                'risk_tolerance': characteristics.risk_tolerance.value,
                'time_horizon': characteristics.time_horizon,
                'key_principles': characteristics.key_principles,
                'preferred_sectors': characteristics.preferred_sectors,
                'historical_returns': characteristics.historical_returns
            }

        return persona_info

    async def quick_analysis(self, symbol: str,
                           personas: Optional[List[str]] = None) -> Optional[InvestmentRecommendation]:
        """Perform quick analysis without debate for a single symbol"""

        personas = personas or ['warren_buffett', 'ray_dalio', 'cathie_wood']

        request = InvestmentRequest(
            symbols=[symbol],
            personas=personas,
            analysis_depth="quick",
            include_debate=False,
            include_news=True
        )

        recommendations = await self.analyze_investment(request)
        return recommendations[0] if recommendations else None

    def save_analysis_results(self, recommendations: List[InvestmentRecommendation],
                            output_path: str) -> None:
        """Save analysis results to JSON file"""

        try:
            results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_symbols': len(recommendations),
                'recommendations': [rec.to_dict() for rec in recommendations]
            }

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Analysis results saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")

    def generate_report(self, recommendations: List[InvestmentRecommendation]) -> str:
        """Generate a formatted analysis report"""

        if not recommendations:
            return "No investment recommendations available."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AI HEDGE FUND INVESTMENT ANALYSIS REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)

        # Summary statistics
        total_symbols = len(recommendations)
        buy_count = sum(1 for r in recommendations
                       if r.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY])
        avg_consensus = np.mean([r.consensus_score for r in recommendations])
        avg_confidence = np.mean([r.confidence_score for r in recommendations])

        report_lines.append(f"\\nSUMMARY:")
        report_lines.append(f"Total Symbols Analyzed: {total_symbols}")
        report_lines.append(f"Buy Recommendations: {buy_count}")
        report_lines.append(f"Average Consensus Score: {avg_consensus:.1f}%")
        report_lines.append(f"Average Confidence: {avg_confidence:.1f}%")

        # Individual recommendations
        report_lines.append(f"\\nDETAILED RECOMMENDATIONS:")
        report_lines.append("-" * 80)

        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"\\n{i}. {rec.symbol}")
            report_lines.append(f"   Recommendation: {rec.recommendation.value}")
            report_lines.append(f"   Consensus Score: {rec.consensus_score:.1f}%")
            report_lines.append(f"   Confidence: {rec.confidence_score:.1f}%")

            if rec.target_price:
                current_price = rec.market_analysis.current_price
                upside = ((rec.target_price - current_price) / current_price) * 100
                report_lines.append(f"   Target Price: ${rec.target_price:.2f} ({upside:+.1f}% upside)")

            if rec.stop_loss:
                current_price = rec.market_analysis.current_price
                downside = ((rec.stop_loss - current_price) / current_price) * 100
                report_lines.append(f"   Stop Loss: ${rec.stop_loss:.2f} ({downside:.1f}% downside)")

            report_lines.append(f"   Position Size: {rec.position_size:.1%}")
            report_lines.append(f"   Risk Level: {rec.risk_level}")
            report_lines.append(f"   Expected Return: {rec.expected_return:.1%}")

            # Top supporting arguments
            if rec.consensus_details.key_supporting_arguments:
                report_lines.append(f"   Key Arguments: {'; '.join(rec.consensus_details.key_supporting_arguments[:2])}")

        report_lines.append("\\n" + "=" * 80)
        report_lines.append("DISCLAIMER: These are AI-generated recommendations for research purposes only.")
        report_lines.append("Always conduct your own research and consult with financial advisors.")
        report_lines.append("=" * 80)

        return "\\n".join(report_lines)


# Convenience functions for easy usage
async def analyze_stocks(symbols: List[str],
                        personas: Optional[List[str]] = None,
                        include_debate: bool = True) -> List[InvestmentRecommendation]:
    """
    Convenience function to analyze stocks with default settings

    Args:
        symbols: List of stock symbols to analyze
        personas: List of persona keys (defaults to main 3)
        include_debate: Whether to include debate phase

    Returns:
        List of investment recommendations
    """

    personas = personas or ['warren_buffett', 'ray_dalio', 'cathie_wood']

    orchestrator = InvestmentOrchestrator()

    request = InvestmentRequest(
        symbols=symbols,
        personas=personas,
        include_debate=include_debate
    )

    return await orchestrator.analyze_investment(request)


async def quick_stock_analysis(symbol: str) -> Optional[InvestmentRecommendation]:
    """Quick analysis of a single stock"""

    orchestrator = InvestmentOrchestrator()
    return await orchestrator.quick_analysis(symbol)


def get_persona_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available personas"""

    orchestrator = InvestmentOrchestrator()
    return orchestrator.get_available_personas()


# Example usage
if __name__ == "__main__":
    async def main():
        # Example: Analyze tech stocks
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        recommendations = await analyze_stocks(symbols)

        # Generate report
        orchestrator = InvestmentOrchestrator()
        report = orchestrator.generate_report(recommendations)
        print(report)

        # Save results
        orchestrator.save_analysis_results(
            recommendations,
            'analysis_results.json'
        )

    # Run the example
    asyncio.run(main())
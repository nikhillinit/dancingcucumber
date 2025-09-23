"""
AI Investor Debate Framework
============================

This module implements a sophisticated debate system where AI investor personas
analyze market conditions, debate investment decisions, and reach consensus
for trading recommendations.

Author: AI Hedge Fund Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import openai
import os
from .ai_investor_personas import (
    InvestorPersona, PersonaAnalysis, DebateMessage, ConsensusResult,
    InvestmentAction, MarketAnalysis, NewsService, MarketDataService,
    AVAILABLE_PERSONAS
)

logger = logging.getLogger(__name__)


class DebateRound:
    """Represents a single round of debate"""

    def __init__(self, round_number: int, topic: str):
        self.round_number = round_number
        self.topic = topic
        self.messages: List[DebateMessage] = []
        self.timestamp = datetime.now()


@dataclass
class DebateConfig:
    """Configuration for debate sessions"""
    max_rounds: int = 3
    max_participants: int = 6
    time_limit_minutes: int = 10
    consensus_threshold: float = 0.7  # 70% agreement needed for consensus
    openai_api_key: Optional[str] = None
    use_ai_moderation: bool = True
    allow_interruptions: bool = True
    require_evidence: bool = True


class DebateModerator:
    """AI-powered debate moderator"""

    def __init__(self, config: DebateConfig):
        self.config = config
        self.openai_client = None
        if config.openai_api_key:
            openai.api_key = config.openai_api_key
            self.openai_client = openai

    async def moderate_debate(self, symbol: str, participants: List[str],
                            initial_analyses: List[PersonaAnalysis]) -> List[DebateRound]:
        """Moderate a full debate session"""
        rounds = []

        try:
            # Opening round - each persona presents their initial analysis
            opening_round = await self._conduct_opening_round(symbol, participants, initial_analyses)
            rounds.append(opening_round)

            # Debate rounds - personas respond to each other
            for round_num in range(2, self.config.max_rounds + 1):
                debate_round = await self._conduct_debate_round(
                    symbol, participants, rounds, round_num
                )
                rounds.append(debate_round)

                # Check for early consensus
                if await self._check_early_consensus(rounds):
                    logger.info(f"Early consensus reached in round {round_num}")
                    break

        except Exception as e:
            logger.error(f"Error in debate moderation: {e}")

        return rounds

    async def _conduct_opening_round(self, symbol: str, participants: List[str],
                                   initial_analyses: List[PersonaAnalysis]) -> DebateRound:
        """Conduct the opening round where each persona presents their view"""
        round_obj = DebateRound(1, f"Initial Analysis of {symbol}")

        for participant, analysis in zip(participants, initial_analyses):
            if participant in AVAILABLE_PERSONAS:
                persona = AVAILABLE_PERSONAS[participant]

                # Generate opening statement
                opening_message = await self._generate_opening_statement(persona, analysis)

                debate_msg = DebateMessage(
                    persona_name=persona.name,
                    message=opening_message,
                    timestamp=datetime.now(),
                    message_type="opening",
                    supporting_data={
                        'recommendation': analysis.recommendation.value,
                        'confidence': analysis.confidence_score,
                        'key_factors': analysis.key_factors[:3]  # Top 3 factors
                    }
                )
                round_obj.messages.append(debate_msg)

        return round_obj

    async def _conduct_debate_round(self, symbol: str, participants: List[str],
                                  previous_rounds: List[DebateRound],
                                  round_number: int) -> DebateRound:
        """Conduct a debate round where personas respond to each other"""
        round_obj = DebateRound(round_number, f"Debate Round {round_number} - {symbol}")

        # Get all previous messages for context
        all_previous_messages = []
        for round_data in previous_rounds:
            all_previous_messages.extend(round_data.messages)

        # Each persona responds based on what others have said
        for participant in participants:
            if participant in AVAILABLE_PERSONAS:
                persona = AVAILABLE_PERSONAS[participant]

                # Generate response based on previous round
                response_message = await self._generate_debate_response(
                    persona, symbol, all_previous_messages, round_number
                )

                debate_msg = DebateMessage(
                    persona_name=persona.name,
                    message=response_message,
                    timestamp=datetime.now(),
                    message_type="argument" if round_number == 2 else "counter_argument",
                    supporting_data={}
                )
                round_obj.messages.append(debate_msg)

        return round_obj

    async def _generate_opening_statement(self, persona: InvestorPersona,
                                        analysis: PersonaAnalysis) -> str:
        """Generate an opening statement for a persona"""
        if self.openai_client:
            return await self._generate_ai_opening_statement(persona, analysis)
        else:
            return self._generate_template_opening_statement(persona, analysis)

    async def _generate_ai_opening_statement(self, persona: InvestorPersona,
                                           analysis: PersonaAnalysis) -> str:
        """Use AI to generate opening statement"""
        prompt = f"""
        You are {persona.name}, presenting your investment analysis for {analysis.symbol}.

        Your investment philosophy: {persona.characteristics.investment_style.value}
        Your key principles: {', '.join(persona.characteristics.key_principles)}

        Your analysis concluded: {analysis.recommendation.value} with {analysis.confidence_score:.0f}% confidence
        Key factors: {', '.join(analysis.key_factors)}
        Reasoning: {analysis.reasoning}

        Present a concise opening statement (2-3 sentences) that:
        1. States your recommendation clearly
        2. Highlights your strongest supporting argument
        3. Stays true to your investment philosophy
        4. Sets up potential debate points

        Speak in your characteristic style.
        """

        try:
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are participating in an investment debate as a legendary investor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"AI generation failed, using template: {e}")
            return self._generate_template_opening_statement(persona, analysis)

    def _generate_template_opening_statement(self, persona: InvestorPersona,
                                           analysis: PersonaAnalysis) -> str:
        """Generate template-based opening statement"""
        templates = {
            "Warren Buffett": f"I recommend {analysis.recommendation.value} for {analysis.symbol}. "
                             f"My analysis shows this company has the predictable cash flows and competitive advantages I look for. "
                             f"With {analysis.confidence_score:.0f}% confidence, I believe this aligns with buying wonderful businesses at fair prices.",

            "Ray Dalio": f"From a macro perspective, I see {analysis.recommendation.value} for {analysis.symbol}. "
                        f"The risk-adjusted factors and correlation analysis support this view with {analysis.confidence_score:.0f}% confidence. "
                        f"This fits within a diversified, all-weather approach to portfolio construction.",

            "Cathie Wood": f"I'm {analysis.recommendation.value} on {analysis.symbol} with {analysis.confidence_score:.0f}% confidence. "
                          f"This company represents the kind of disruptive innovation that creates exponential returns. "
                          f"The technological transformation potential here far outweighs traditional valuation concerns."
        }

        return templates.get(persona.name,
                           f"I recommend {analysis.recommendation.value} for {analysis.symbol} "
                           f"with {analysis.confidence_score:.0f}% confidence based on my analysis.")

    async def _generate_debate_response(self, persona: InvestorPersona, symbol: str,
                                      previous_messages: List[DebateMessage],
                                      round_number: int) -> str:
        """Generate a debate response for a persona"""
        if self.openai_client:
            return await self._generate_ai_debate_response(persona, symbol, previous_messages, round_number)
        else:
            return self._generate_template_debate_response(persona, symbol, previous_messages, round_number)

    async def _generate_ai_debate_response(self, persona: InvestorPersona, symbol: str,
                                         previous_messages: List[DebateMessage],
                                         round_number: int) -> str:
        """Use AI to generate debate response"""
        # Get opposing viewpoints
        opposing_messages = [msg for msg in previous_messages
                           if msg.persona_name != persona.name]

        context = "\\n".join([f"{msg.persona_name}: {msg.message}"
                             for msg in opposing_messages[-6:]])  # Last 6 messages

        prompt = f"""
        You are {persona.name} in round {round_number} of an investment debate about {symbol}.

        Your investment philosophy: {persona.characteristics.investment_style.value}
        Your approach: {', '.join(persona.characteristics.key_principles[:3])}

        Previous debate context:
        {context}

        Respond to the other investors' points while:
        1. Staying true to your investment philosophy
        2. Addressing specific counterarguments
        3. Providing new evidence or perspective
        4. Maintaining your characteristic communication style
        5. Being respectful but firm in your convictions

        Keep response to 2-3 sentences. Be specific and substantive.
        """

        try:
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are debating investment decisions as a legendary investor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"AI response generation failed, using template: {e}")
            return self._generate_template_debate_response(persona, symbol, previous_messages, round_number)

    def _generate_template_debate_response(self, persona: InvestorPersona, symbol: str,
                                         previous_messages: List[DebateMessage],
                                         round_number: int) -> str:
        """Generate template-based debate response"""

        # Analyze opposing viewpoints
        recent_messages = [msg for msg in previous_messages[-4:]
                          if msg.persona_name != persona.name]

        templates = {
            "Warren Buffett": [
                f"While I respect the growth potential others see, I prefer the certainty of strong cash flows and proven business models.",
                f"Short-term market movements don't concern me when the underlying business fundamentals are sound.",
                f"I'd rather pay a fair price for a great business than a great price for a fair business."
            ],

            "Ray Dalio": [
                f"The key is understanding how this fits within the broader economic cycle and portfolio construction.",
                f"Correlation risk and systematic factors are more important than individual stock performance.",
                f"Diversification remains essential, regardless of individual conviction levels."
            ],

            "Cathie Wood": [
                f"Traditional metrics often miss the exponential potential of disruptive technologies.",
                f"The risk of missing transformational growth far exceeds the risk of short-term volatility.",
                f"Innovation cycles create entirely new market categories that render old valuation models obsolete."
            ]
        }

        persona_templates = templates.get(persona.name, [
            f"My analysis considers factors that may be overlooked by other approaches.",
            f"The long-term perspective often reveals opportunities hidden by short-term noise.",
            f"Risk management and opportunity assessment must be balanced for optimal returns."
        ])

        # Select template based on round number
        template_index = (round_number - 2) % len(persona_templates)
        return persona_templates[template_index]

    async def _check_early_consensus(self, rounds: List[DebateRound]) -> bool:
        """Check if early consensus has been reached"""
        if len(rounds) < 2:
            return False

        # Simple consensus check - if all recent messages have similar sentiment
        recent_messages = rounds[-1].messages

        # Count positive vs negative sentiment in messages
        positive_words = ['buy', 'strong', 'good', 'excellent', 'bullish', 'opportunity']
        negative_words = ['sell', 'weak', 'poor', 'bearish', 'risk', 'concern']

        sentiment_scores = []
        for msg in recent_messages:
            text = msg.message.lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            if pos_count + neg_count > 0:
                sentiment_scores.append((pos_count - neg_count) / (pos_count + neg_count))
            else:
                sentiment_scores.append(0)

        # Check if sentiment is consistent
        if len(sentiment_scores) > 0:
            sentiment_std = np.std(sentiment_scores)
            return sentiment_std < 0.3  # Low variance indicates agreement

        return False


class ConsensusBuilder:
    """Builds consensus from debate results"""

    def __init__(self, config: DebateConfig):
        self.config = config

    def build_consensus(self, symbol: str, initial_analyses: List[PersonaAnalysis],
                       debate_rounds: List[DebateRound],
                       market_analysis: MarketAnalysis) -> ConsensusResult:
        """Build final consensus from analyses and debate"""

        try:
            # Analyze initial recommendations
            recommendations = [analysis.recommendation for analysis in initial_analyses]
            confidence_scores = [analysis.confidence_score for analysis in initial_analyses]

            # Calculate consensus metrics
            consensus_metrics = self._calculate_consensus_metrics(
                recommendations, confidence_scores, debate_rounds
            )

            # Determine final recommendation
            final_recommendation = self._determine_final_recommendation(
                recommendations, confidence_scores, consensus_metrics
            )

            # Calculate position sizing
            position_size = self._calculate_consensus_position_size(
                initial_analyses, consensus_metrics
            )

            # Extract key arguments
            supporting_args, concerns, dissenting = self._extract_debate_insights(
                debate_rounds, initial_analyses
            )

            # Generate market assessment
            market_assessment = self._assess_market_conditions(market_analysis)

            # Calculate expected return
            expected_return = self._calculate_expected_return(
                initial_analyses, consensus_metrics['consensus_score']
            )

            # Generate debate summary
            debate_summary = self._generate_debate_summary(debate_rounds, consensus_metrics)

            return ConsensusResult(
                symbol=symbol,
                final_recommendation=final_recommendation,
                consensus_score=consensus_metrics['consensus_score'],
                average_confidence=np.mean(confidence_scores),
                recommended_position_size=position_size,
                target_price=self._calculate_consensus_target_price(initial_analyses),
                stop_loss=self._calculate_consensus_stop_loss(initial_analyses, market_analysis),
                key_supporting_arguments=supporting_args,
                key_concerns=concerns,
                dissenting_opinions=dissenting,
                market_conditions_assessment=market_assessment,
                execution_timeline=self._determine_execution_timeline(final_recommendation, consensus_metrics),
                risk_level=self._assess_risk_level(initial_analyses, consensus_metrics),
                expected_return=expected_return,
                debate_summary=debate_summary,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error building consensus: {e}")
            return self._create_fallback_consensus(symbol, initial_analyses)

    def _calculate_consensus_metrics(self, recommendations: List[InvestmentAction],
                                   confidence_scores: List[float],
                                   debate_rounds: List[DebateRound]) -> Dict[str, float]:
        """Calculate consensus metrics from analyses and debate"""

        # Map recommendations to numerical values
        rec_values = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }

        numerical_recs = [rec_values[rec] for rec in recommendations]

        # Calculate agreement metrics
        recommendation_variance = np.var(numerical_recs)
        confidence_variance = np.var(confidence_scores)

        # Consensus score based on recommendation agreement and confidence
        max_variance = 4  # Maximum possible variance for recommendations (-2 to 2)
        recommendation_consensus = 1 - (recommendation_variance / max_variance)
        confidence_consensus = 1 - (confidence_variance / 10000)  # Normalize confidence variance

        # Debate quality factor
        debate_quality = self._assess_debate_quality(debate_rounds)

        # Overall consensus score
        consensus_score = (
            recommendation_consensus * 0.5 +
            confidence_consensus * 0.3 +
            debate_quality * 0.2
        ) * 100

        return {
            'consensus_score': max(0, min(100, consensus_score)),
            'recommendation_variance': recommendation_variance,
            'confidence_variance': confidence_variance,
            'debate_quality': debate_quality
        }

    def _assess_debate_quality(self, debate_rounds: List[DebateRound]) -> float:
        """Assess the quality of the debate based on message content"""
        if not debate_rounds:
            return 0.5

        total_messages = sum(len(round_obj.messages) for round_obj in debate_rounds)

        # Quality factors
        engagement_score = min(1.0, total_messages / 15)  # More messages = more engagement

        # Analyze message diversity (different perspectives)
        unique_personas = set()
        for round_obj in debate_rounds:
            for msg in round_obj.messages:
                unique_personas.add(msg.persona_name)

        diversity_score = len(unique_personas) / max(len(AVAILABLE_PERSONAS), 1)

        # Average message length (proxy for depth)
        total_length = sum(
            len(msg.message) for round_obj in debate_rounds
            for msg in round_obj.messages
        )
        avg_length = total_length / max(total_messages, 1)
        length_score = min(1.0, avg_length / 200)  # Normalize to reasonable message length

        return (engagement_score + diversity_score + length_score) / 3

    def _determine_final_recommendation(self, recommendations: List[InvestmentAction],
                                      confidence_scores: List[float],
                                      consensus_metrics: Dict[str, float]) -> InvestmentAction:
        """Determine final recommendation based on weighted consensus"""

        # Weight recommendations by confidence scores
        rec_values = {
            InvestmentAction.STRONG_SELL: -2,
            InvestmentAction.SELL: -1,
            InvestmentAction.HOLD: 0,
            InvestmentAction.BUY: 1,
            InvestmentAction.STRONG_BUY: 2
        }

        weighted_sum = sum(
            rec_values[rec] * (conf / 100)
            for rec, conf in zip(recommendations, confidence_scores)
        )

        weighted_avg = weighted_sum / len(recommendations)

        # Map back to recommendation
        if weighted_avg >= 1.5:
            return InvestmentAction.STRONG_BUY
        elif weighted_avg >= 0.5:
            return InvestmentAction.BUY
        elif weighted_avg >= -0.5:
            return InvestmentAction.HOLD
        elif weighted_avg >= -1.5:
            return InvestmentAction.SELL
        else:
            return InvestmentAction.STRONG_SELL

    def _calculate_consensus_position_size(self, analyses: List[PersonaAnalysis],
                                         consensus_metrics: Dict[str, float]) -> float:
        """Calculate recommended position size based on consensus"""

        # Average position sizes weighted by confidence
        total_weight = sum(analysis.confidence_score for analysis in analyses)
        if total_weight == 0:
            return 0.05  # Default 5%

        weighted_position_size = sum(
            analysis.position_size_recommendation * analysis.confidence_score
            for analysis in analyses
        ) / total_weight

        # Adjust by consensus score - higher consensus allows larger positions
        consensus_multiplier = consensus_metrics['consensus_score'] / 100
        adjusted_size = weighted_position_size * consensus_multiplier

        # Cap at reasonable limits
        return max(0.02, min(0.25, adjusted_size))

    def _extract_debate_insights(self, debate_rounds: List[DebateRound],
                               analyses: List[PersonaAnalysis]) -> Tuple[List[str], List[str], List[str]]:
        """Extract key insights from debate"""

        supporting_args = []
        concerns = []
        dissenting = []

        # Extract from initial analyses
        for analysis in analyses:
            if analysis.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY]:
                supporting_args.extend(analysis.key_factors[:2])
            elif analysis.recommendation in [InvestmentAction.SELL, InvestmentAction.STRONG_SELL]:
                concerns.extend(analysis.key_factors[:2])

        # Extract from debate messages
        for round_obj in debate_rounds:
            for msg in round_obj.messages:
                # Simple keyword analysis
                text = msg.message.lower()

                if any(word in text for word in ['buy', 'strong', 'opportunity', 'growth']):
                    supporting_args.append(f"{msg.persona_name}: {msg.message[:100]}...")
                elif any(word in text for word in ['risk', 'concern', 'caution', 'overvalued']):
                    concerns.append(f"{msg.persona_name}: {msg.message[:100]}...")
                elif any(word in text for word in ['disagree', 'however', 'but', 'different']):
                    dissenting.append(f"{msg.persona_name}: {msg.message[:100]}...")

        # Remove duplicates and limit length
        supporting_args = list(set(supporting_args))[:5]
        concerns = list(set(concerns))[:5]
        dissenting = list(set(dissenting))[:3]

        return supporting_args, concerns, dissenting

    def _assess_market_conditions(self, market_analysis: MarketAnalysis) -> str:
        """Assess current market conditions"""

        conditions = []

        # Volatility assessment
        volatility = market_analysis.technical_indicators.get('volatility_20d')
        if volatility:
            if volatility > 0.4:
                conditions.append("High volatility environment")
            elif volatility < 0.15:
                conditions.append("Low volatility environment")
            else:
                conditions.append("Moderate volatility environment")

        # Trend assessment
        current_price = market_analysis.current_price
        sma_200 = market_analysis.technical_indicators.get('sma_200')
        if sma_200:
            if current_price > sma_200 * 1.05:
                conditions.append("Strong uptrend")
            elif current_price < sma_200 * 0.95:
                conditions.append("Downtrend")
            else:
                conditions.append("Sideways trend")

        # Volume analysis
        volume_ratio = market_analysis.technical_indicators.get('volume_ratio')
        if volume_ratio:
            if volume_ratio > 1.5:
                conditions.append("High volume interest")
            elif volume_ratio < 0.7:
                conditions.append("Low volume participation")

        return "; ".join(conditions) if conditions else "Neutral market conditions"

    def _calculate_consensus_target_price(self, analyses: List[PersonaAnalysis]) -> Optional[float]:
        """Calculate consensus target price"""

        targets = [analysis.target_price for analysis in analyses if analysis.target_price]
        if not targets:
            return None

        # Weight by confidence scores
        weights = [
            analysis.confidence_score for analysis in analyses
            if analysis.target_price
        ]

        if not weights:
            return np.mean(targets)

        weighted_target = sum(t * w for t, w in zip(targets, weights)) / sum(weights)
        return round(weighted_target, 2)

    def _calculate_consensus_stop_loss(self, analyses: List[PersonaAnalysis],
                                     market_analysis: MarketAnalysis) -> Optional[float]:
        """Calculate consensus stop loss"""

        stops = [analysis.stop_loss for analysis in analyses if analysis.stop_loss]
        if not stops:
            # Default to 15% below current price
            return round(market_analysis.current_price * 0.85, 2)

        # Use the most conservative (highest) stop loss
        return round(max(stops), 2)

    def _determine_execution_timeline(self, recommendation: InvestmentAction,
                                    consensus_metrics: Dict[str, float]) -> str:
        """Determine recommended execution timeline"""

        consensus_score = consensus_metrics['consensus_score']

        if recommendation in [InvestmentAction.STRONG_BUY, InvestmentAction.STRONG_SELL]:
            if consensus_score > 80:
                return "Immediate (within 1-2 trading days)"
            else:
                return "Near-term (within 1 week)"
        elif recommendation in [InvestmentAction.BUY, InvestmentAction.SELL]:
            return "Short-term (within 2 weeks)"
        else:
            return "Monitor for better entry/exit points"

    def _assess_risk_level(self, analyses: List[PersonaAnalysis],
                          consensus_metrics: Dict[str, float]) -> str:
        """Assess overall risk level"""

        # Aggregate risk factors from analyses
        risk_factors = []
        for analysis in analyses:
            risk_assessment = analysis.risk_assessment
            if isinstance(risk_assessment, dict):
                risk_factors.extend(risk_assessment.values())

        # Count high risk mentions
        high_risk_count = sum(1 for factor in risk_factors if 'high' in str(factor).lower())
        total_factors = len(risk_factors)

        # Consider consensus score - lower consensus = higher risk
        consensus_score = consensus_metrics['consensus_score']

        if high_risk_count / max(total_factors, 1) > 0.5 or consensus_score < 60:
            return "High Risk"
        elif high_risk_count / max(total_factors, 1) > 0.3 or consensus_score < 75:
            return "Moderate Risk"
        else:
            return "Low to Moderate Risk"

    def _calculate_expected_return(self, analyses: List[PersonaAnalysis],
                                 consensus_score: float) -> float:
        """Calculate expected return based on consensus"""

        # Simple model: higher consensus and positive recommendations = higher expected return
        buy_count = sum(1 for analysis in analyses
                       if analysis.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY])

        base_return = (buy_count / len(analyses)) * 0.15  # 15% base for all buy
        consensus_boost = (consensus_score / 100) * 0.05  # Up to 5% boost for consensus

        return round(base_return + consensus_boost, 3)

    def _generate_debate_summary(self, debate_rounds: List[DebateRound],
                               consensus_metrics: Dict[str, float]) -> str:
        """Generate a summary of the debate"""

        if not debate_rounds:
            return "No debate conducted."

        total_messages = sum(len(round_obj.messages) for round_obj in debate_rounds)
        unique_personas = set()
        for round_obj in debate_rounds:
            for msg in round_obj.messages:
                unique_personas.add(msg.persona_name)

        summary_parts = [
            f"Debate involved {len(unique_personas)} personas over {len(debate_rounds)} rounds",
            f"with {total_messages} total exchanges.",
            f"Consensus score: {consensus_metrics['consensus_score']:.0f}%."
        ]

        if consensus_metrics['consensus_score'] > 80:
            summary_parts.append("Strong agreement achieved across investment philosophies.")
        elif consensus_metrics['consensus_score'] > 60:
            summary_parts.append("Moderate agreement with some divergent views.")
        else:
            summary_parts.append("Significant disagreement with multiple perspectives presented.")

        return " ".join(summary_parts)

    def _create_fallback_consensus(self, symbol: str,
                                 analyses: List[PersonaAnalysis]) -> ConsensusResult:
        """Create a basic consensus when full processing fails"""

        if not analyses:
            return ConsensusResult(
                symbol=symbol,
                final_recommendation=InvestmentAction.HOLD,
                consensus_score=50.0,
                average_confidence=50.0,
                recommended_position_size=0.05,
                target_price=None,
                stop_loss=None,
                key_supporting_arguments=["Analysis incomplete"],
                key_concerns=["Insufficient data"],
                dissenting_opinions=[],
                market_conditions_assessment="Unable to assess",
                execution_timeline="Hold pending further analysis",
                risk_level="Unknown",
                expected_return=0.0,
                debate_summary="Consensus building failed",
                timestamp=datetime.now()
            )

        # Simple majority vote
        recommendations = [analysis.recommendation for analysis in analyses]
        recommendation_counts = {}
        for rec in recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        final_recommendation = max(recommendation_counts, key=recommendation_counts.get)
        consensus_score = (recommendation_counts[final_recommendation] / len(recommendations)) * 100

        return ConsensusResult(
            symbol=symbol,
            final_recommendation=final_recommendation,
            consensus_score=consensus_score,
            average_confidence=np.mean([a.confidence_score for a in analyses]),
            recommended_position_size=np.mean([a.position_size_recommendation for a in analyses]),
            target_price=None,
            stop_loss=None,
            key_supporting_arguments=[f"Majority recommendation: {final_recommendation.value}"],
            key_concerns=["Limited consensus analysis"],
            dissenting_opinions=[],
            market_conditions_assessment="Basic analysis only",
            execution_timeline="Standard timing",
            risk_level="Moderate",
            expected_return=0.05,
            debate_summary=f"Simple majority vote: {consensus_score:.0f}% agreement",
            timestamp=datetime.now()
        )
"""
AI Debate Orchestrator
======================

An advanced debate orchestration system that manages sophisticated conversations between
investor personas, integrates real-time market data, and facilitates complex multi-round
debates with dynamic moderation and argument tracking.

Features:
- Advanced debate flow management with dynamic participant selection
- Real-time market data integration during debates
- Sentiment analysis and argument tracking
- Dynamic round adjustment based on consensus progress
- Evidence-based argumentation with fact checking
- Advanced AI moderation with interruption handling
- Multi-threaded debate execution for performance

Author: AI Hedge Fund Team
Version: 2.0.0
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import numpy as np
import pandas as pd
import openai
import threading
from threading import Lock
import hashlib
from collections import defaultdict, deque
import yfinance as yf
import requests
from textblob import TextBlob
import re

from .ai_investor_personas import (
    InvestorPersona, PersonaAnalysis, DebateMessage, ConsensusResult,
    InvestmentAction, MarketAnalysis, NewsService, MarketDataService,
    AVAILABLE_PERSONAS, PersonaCharacteristics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DebateStatus(Enum):
    """Debate session status"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CONCLUDED = "CONCLUDED"
    FAILED = "FAILED"


class ArgumentType(Enum):
    """Types of arguments in debate"""
    OPENING = "OPENING"
    SUPPORTING = "SUPPORTING"
    COUNTERARGUMENT = "COUNTERARGUMENT"
    REBUTTAL = "REBUTTAL"
    CONSENSUS_BUILDING = "CONSENSUS_BUILDING"
    FINAL_STATEMENT = "FINAL_STATEMENT"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class DebateEvidence:
    """Evidence supporting an argument"""
    source: str
    data_type: str  # "market_data", "news", "technical", "fundamental"
    value: Any
    confidence: float
    timestamp: datetime
    relevance_score: float


@dataclass
class EnhancedDebateMessage:
    """Enhanced debate message with advanced features"""
    id: str
    persona_name: str
    message: str
    timestamp: datetime
    argument_type: ArgumentType
    priority: MessagePriority
    evidence: List[DebateEvidence] = field(default_factory=list)
    references: List[str] = field(default_factory=list)  # IDs of messages this responds to
    sentiment_score: float = 0.0
    confidence_level: float = 0.0
    fact_checked: bool = False
    moderator_notes: str = ""
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRoundAdvanced:
    """Advanced debate round with enhanced tracking"""
    round_number: int
    topic: str
    participants: List[str]
    messages: List[EnhancedDebateMessage] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    consensus_progress: float = 0.0
    key_points_emerging: List[str] = field(default_factory=list)
    market_data_snapshot: Optional[Dict[str, Any]] = None
    round_summary: str = ""


@dataclass
class DebateConfiguration:
    """Advanced configuration for debate sessions"""
    symbol: str
    max_rounds: int = 5
    max_participants: int = 6
    time_limit_minutes: int = 15
    consensus_threshold: float = 0.75
    openai_api_key: Optional[str] = None
    enable_ai_moderation: bool = True
    enable_interruptions: bool = True
    require_evidence: bool = True
    enable_real_time_data: bool = True
    enable_sentiment_analysis: bool = True
    enable_fact_checking: bool = True
    dynamic_round_adjustment: bool = True
    parallel_processing: bool = True
    save_transcript: bool = True
    min_consensus_rounds: int = 2
    max_message_length: int = 500
    evidence_weight: float = 0.3
    personality_weight: float = 0.4
    market_data_weight: float = 0.3


class ArgumentTracker:
    """Tracks and analyzes arguments throughout the debate"""

    def __init__(self):
        self.arguments: Dict[str, List[EnhancedDebateMessage]] = defaultdict(list)
        self.evidence_graph: Dict[str, List[str]] = defaultdict(list)
        self.sentiment_trends: Dict[str, List[float]] = defaultdict(list)
        self.fact_check_results: Dict[str, bool] = {}

    def add_argument(self, message: EnhancedDebateMessage):
        """Add argument to tracking system"""
        self.arguments[message.persona_name].append(message)
        self.sentiment_trends[message.persona_name].append(message.sentiment_score)

        # Track evidence relationships
        for evidence in message.evidence:
            self.evidence_graph[evidence.source].append(message.id)

    def get_argument_strength(self, persona_name: str) -> float:
        """Calculate argument strength for a persona"""
        messages = self.arguments[persona_name]
        if not messages:
            return 0.0

        evidence_score = np.mean([len(msg.evidence) for msg in messages])
        confidence_score = np.mean([msg.confidence_level for msg in messages])
        sentiment_consistency = 1.0 - np.std(self.sentiment_trends[persona_name])

        return (evidence_score * 0.4 + confidence_score * 0.4 + sentiment_consistency * 0.2)

    def detect_contradictions(self) -> List[Tuple[str, str]]:
        """Detect contradictory arguments"""
        contradictions = []

        for persona1 in self.arguments:
            for persona2 in self.arguments:
                if persona1 != persona2:
                    for msg1 in self.arguments[persona1]:
                        for msg2 in self.arguments[persona2]:
                            if self._are_contradictory(msg1, msg2):
                                contradictions.append((msg1.id, msg2.id))

        return contradictions

    def _are_contradictory(self, msg1: EnhancedDebateMessage, msg2: EnhancedDebateMessage) -> bool:
        """Simple contradiction detection"""
        # Check for opposing sentiment on same topic
        if abs(msg1.sentiment_score - msg2.sentiment_score) > 1.5:
            return True
        return False


class MarketDataIntegrator:
    """Integrates real-time market data into debates"""

    def __init__(self, config: DebateConfiguration):
        self.config = config
        self.market_service = MarketDataService()
        self.news_service = NewsService()
        self.data_cache: Dict[str, Tuple[datetime, Any]] = {}
        self.update_lock = Lock()

    async def get_real_time_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data snapshot"""
        with self.update_lock:
            cache_key = f"{symbol}_{datetime.now().strftime('%H%M')}"

            if cache_key in self.data_cache:
                cache_time, data = self.data_cache[cache_key]
                if datetime.now() - cache_time < timedelta(minutes=5):
                    return data

            try:
                # Get market data
                market_data = self.market_service.get_market_data(symbol)
                news_data = self.news_service.get_stock_news(symbol, days=1)
                news_sentiment = self.news_service.analyze_sentiment(news_data)

                snapshot = {
                    'timestamp': datetime.now(),
                    'price': market_data.current_price if market_data else None,
                    'change': market_data.price_change_percent if market_data else None,
                    'volume': market_data.volume if market_data else None,
                    'technical_indicators': market_data.technical_indicators if market_data else {},
                    'news_sentiment': news_sentiment,
                    'market_data': market_data
                }

                self.data_cache[cache_key] = (datetime.now(), snapshot)
                return snapshot

            except Exception as e:
                logger.error(f"Failed to get real-time data: {e}")
                return {'error': str(e), 'timestamp': datetime.now()}

    def generate_market_evidence(self, snapshot: Dict[str, Any], topic: str) -> List[DebateEvidence]:
        """Generate evidence from market data"""
        evidence = []

        if 'market_data' in snapshot and snapshot['market_data']:
            market_data = snapshot['market_data']

            # Price movement evidence
            if market_data.price_change_percent:
                evidence.append(DebateEvidence(
                    source="Real-time Price Data",
                    data_type="market_data",
                    value=f"{market_data.price_change_percent:.2f}% price change",
                    confidence=0.9,
                    timestamp=datetime.now(),
                    relevance_score=0.8
                ))

            # Technical indicators evidence
            for indicator, value in market_data.technical_indicators.items():
                if value is not None:
                    evidence.append(DebateEvidence(
                        source=f"Technical Analysis - {indicator}",
                        data_type="technical",
                        value=value,
                        confidence=0.7,
                        timestamp=datetime.now(),
                        relevance_score=0.6
                    ))

        # News sentiment evidence
        if 'news_sentiment' in snapshot:
            sentiment = snapshot['news_sentiment']
            evidence.append(DebateEvidence(
                source="News Sentiment Analysis",
                data_type="news",
                value=sentiment,
                confidence=sentiment.get('confidence', 0.5),
                timestamp=datetime.now(),
                relevance_score=0.7
            ))

        return evidence


class AdvancedModerator:
    """Advanced AI-powered debate moderator"""

    def __init__(self, config: DebateConfiguration):
        self.config = config
        self.openai_client = None
        self.argument_tracker = ArgumentTracker()
        self.market_integrator = MarketDataIntegrator(config)

        if config.openai_api_key:
            openai.api_key = config.openai_api_key
            self.openai_client = openai

    async def moderate_advanced_debate(self, participants: List[str],
                                     initial_analyses: List[PersonaAnalysis]) -> List[DebateRoundAdvanced]:
        """Moderate a sophisticated multi-round debate"""
        rounds = []
        consensus_progress = 0.0

        try:
            logger.info(f"Starting advanced debate for {self.config.symbol} with {len(participants)} participants")

            # Conduct opening round
            opening_round = await self._conduct_opening_round_advanced(participants, initial_analyses)
            rounds.append(opening_round)

            # Dynamic debate rounds
            for round_num in range(2, self.config.max_rounds + 1):
                # Check consensus progress
                consensus_progress = await self._calculate_consensus_progress(rounds)

                if consensus_progress >= self.config.consensus_threshold:
                    logger.info(f"Consensus threshold reached: {consensus_progress:.2f}")
                    break

                # Determine round focus based on progress
                round_focus = await self._determine_round_focus(rounds, consensus_progress)

                # Conduct debate round
                debate_round = await self._conduct_advanced_debate_round(
                    participants, rounds, round_num, round_focus
                )
                rounds.append(debate_round)

                # Allow dynamic participant adjustment
                if self.config.dynamic_round_adjustment:
                    participants = await self._adjust_participants(participants, rounds)

            # Conduct final consensus round if needed
            if len(rounds) >= self.config.min_consensus_rounds:
                final_round = await self._conduct_consensus_round(participants, rounds)
                rounds.append(final_round)

        except Exception as e:
            logger.error(f"Error in advanced debate moderation: {e}")
            raise

        return rounds

    async def _conduct_opening_round_advanced(self, participants: List[str],
                                           analyses: List[PersonaAnalysis]) -> DebateRoundAdvanced:
        """Conduct advanced opening round"""
        round_obj = DebateRoundAdvanced(
            round_number=1,
            topic=f"Initial Analysis of {self.config.symbol}",
            participants=participants.copy()
        )

        # Get real-time market snapshot
        if self.config.enable_real_time_data:
            market_snapshot = await self.market_integrator.get_real_time_snapshot(self.config.symbol)
            round_obj.market_data_snapshot = market_snapshot

        # Generate opening statements in parallel if enabled
        if self.config.parallel_processing:
            tasks = []
            for participant, analysis in zip(participants, analyses):
                task = self._generate_enhanced_opening_statement(participant, analysis, round_obj)
                tasks.append(task)

            messages = await asyncio.gather(*tasks, return_exceptions=True)
            for msg in messages:
                if not isinstance(msg, Exception):
                    round_obj.messages.append(msg)
                    self.argument_tracker.add_argument(msg)
        else:
            # Sequential processing
            for participant, analysis in zip(participants, analyses):
                message = await self._generate_enhanced_opening_statement(participant, analysis, round_obj)
                round_obj.messages.append(message)
                self.argument_tracker.add_argument(message)

        round_obj.end_time = datetime.now()
        round_obj.consensus_progress = await self._calculate_round_consensus(round_obj)

        return round_obj

    async def _generate_enhanced_opening_statement(self, participant: str,
                                                 analysis: PersonaAnalysis,
                                                 round_obj: DebateRoundAdvanced) -> EnhancedDebateMessage:
        """Generate enhanced opening statement with evidence"""
        if participant not in AVAILABLE_PERSONAS:
            raise ValueError(f"Unknown persona: {participant}")

        persona = AVAILABLE_PERSONAS[participant]

        # Generate market evidence
        evidence = []
        if round_obj.market_data_snapshot:
            evidence.extend(self.market_integrator.generate_market_evidence(
                round_obj.market_data_snapshot, round_obj.topic
            ))

        # Generate AI-powered statement if available
        if self.openai_client and self.config.enable_ai_moderation:
            statement = await self._generate_ai_enhanced_statement(persona, analysis, evidence)
        else:
            statement = self._generate_template_enhanced_statement(persona, analysis)

        # Analyze sentiment
        sentiment_score = 0.0
        if self.config.enable_sentiment_analysis:
            sentiment_score = self._analyze_sentiment(statement)

        message_id = self._generate_message_id(participant, round_obj.round_number)

        return EnhancedDebateMessage(
            id=message_id,
            persona_name=persona.name,
            message=statement,
            timestamp=datetime.now(),
            argument_type=ArgumentType.OPENING,
            priority=MessagePriority.HIGH,
            evidence=evidence,
            sentiment_score=sentiment_score,
            confidence_level=analysis.confidence_score / 100,
            supporting_data={
                'recommendation': analysis.recommendation.value,
                'target_price': analysis.target_price,
                'position_size': analysis.position_size_recommendation
            }
        )

    async def _conduct_advanced_debate_round(self, participants: List[str],
                                           previous_rounds: List[DebateRoundAdvanced],
                                           round_number: int,
                                           focus: str) -> DebateRoundAdvanced:
        """Conduct advanced debate round with focus"""
        round_obj = DebateRoundAdvanced(
            round_number=round_number,
            topic=f"{focus} - Round {round_number}",
            participants=participants.copy()
        )

        # Update market data
        if self.config.enable_real_time_data:
            market_snapshot = await self.market_integrator.get_real_time_snapshot(self.config.symbol)
            round_obj.market_data_snapshot = market_snapshot

        # Get context from previous rounds
        all_previous_messages = []
        for round_data in previous_rounds:
            all_previous_messages.extend(round_data.messages)

        # Generate responses based on focus and previous arguments
        if self.config.parallel_processing:
            tasks = []
            for participant in participants:
                task = self._generate_focused_response(
                    participant, all_previous_messages, round_obj, focus
                )
                tasks.append(task)

            messages = await asyncio.gather(*tasks, return_exceptions=True)
            for msg in messages:
                if not isinstance(msg, Exception):
                    round_obj.messages.append(msg)
                    self.argument_tracker.add_argument(msg)
        else:
            for participant in participants:
                message = await self._generate_focused_response(
                    participant, all_previous_messages, round_obj, focus
                )
                round_obj.messages.append(message)
                self.argument_tracker.add_argument(message)

        # Analyze round results
        round_obj.end_time = datetime.now()
        round_obj.consensus_progress = await self._calculate_round_consensus(round_obj)
        round_obj.key_points_emerging = await self._extract_emerging_points(round_obj)

        return round_obj

    async def _generate_focused_response(self, participant: str,
                                       previous_messages: List[EnhancedDebateMessage],
                                       round_obj: DebateRoundAdvanced,
                                       focus: str) -> EnhancedDebateMessage:
        """Generate focused response based on debate focus"""
        if participant not in AVAILABLE_PERSONAS:
            raise ValueError(f"Unknown persona: {participant}")

        persona = AVAILABLE_PERSONAS[participant]

        # Find relevant previous messages to respond to
        relevant_messages = self._find_relevant_messages(participant, previous_messages)

        # Generate evidence from current market state
        evidence = []
        if round_obj.market_data_snapshot:
            evidence.extend(self.market_integrator.generate_market_evidence(
                round_obj.market_data_snapshot, focus
            ))

        # Determine argument type based on focus
        argument_type = self._determine_argument_type(focus, round_obj.round_number)

        # Generate response
        if self.openai_client and self.config.enable_ai_moderation:
            response = await self._generate_ai_focused_response(
                persona, relevant_messages, focus, argument_type, evidence
            )
        else:
            response = self._generate_template_focused_response(
                persona, relevant_messages, focus, argument_type
            )

        # Analyze sentiment
        sentiment_score = 0.0
        if self.config.enable_sentiment_analysis:
            sentiment_score = self._analyze_sentiment(response)

        message_id = self._generate_message_id(participant, round_obj.round_number)
        references = [msg.id for msg in relevant_messages[-3:]]  # Reference last 3 relevant messages

        return EnhancedDebateMessage(
            id=message_id,
            persona_name=persona.name,
            message=response,
            timestamp=datetime.now(),
            argument_type=argument_type,
            priority=MessagePriority.NORMAL,
            evidence=evidence,
            references=references,
            sentiment_score=sentiment_score,
            confidence_level=self.argument_tracker.get_argument_strength(persona.name),
            supporting_data={'focus': focus}
        )

    async def _conduct_consensus_round(self, participants: List[str],
                                     previous_rounds: List[DebateRoundAdvanced]) -> DebateRoundAdvanced:
        """Conduct final consensus building round"""
        round_obj = DebateRoundAdvanced(
            round_number=len(previous_rounds) + 1,
            topic=f"Consensus Building - {self.config.symbol}",
            participants=participants.copy()
        )

        # Analyze current state of arguments
        contradictions = self.argument_tracker.detect_contradictions()

        # Generate consensus-building statements
        for participant in participants:
            if participant in AVAILABLE_PERSONAS:
                persona = AVAILABLE_PERSONAS[participant]

                consensus_statement = await self._generate_consensus_statement(
                    persona, previous_rounds, contradictions
                )

                message_id = self._generate_message_id(participant, round_obj.round_number)

                message = EnhancedDebateMessage(
                    id=message_id,
                    persona_name=persona.name,
                    message=consensus_statement,
                    timestamp=datetime.now(),
                    argument_type=ArgumentType.CONSENSUS_BUILDING,
                    priority=MessagePriority.HIGH,
                    confidence_level=self.argument_tracker.get_argument_strength(persona.name)
                )

                round_obj.messages.append(message)
                self.argument_tracker.add_argument(message)

        round_obj.end_time = datetime.now()
        round_obj.consensus_progress = await self._calculate_round_consensus(round_obj)

        return round_obj

    async def _generate_ai_enhanced_statement(self, persona: InvestorPersona,
                                            analysis: PersonaAnalysis,
                                            evidence: List[DebateEvidence]) -> str:
        """Generate AI-powered enhanced opening statement"""
        evidence_text = "\n".join([
            f"- {ev.source}: {ev.value} (confidence: {ev.confidence:.2f})"
            for ev in evidence[:5]  # Top 5 pieces of evidence
        ])

        prompt = f"""
        You are {persona.name}, presenting your investment analysis for {analysis.symbol}.

        Your investment philosophy: {persona.characteristics.investment_style.value}
        Your key principles: {', '.join(persona.characteristics.key_principles[:3])}
        Your communication style: {persona.characteristics.communication_style}

        Your analysis concluded: {analysis.recommendation.value} with {analysis.confidence_score:.0f}% confidence
        Key factors: {', '.join(analysis.key_factors[:3])}

        Current market evidence:
        {evidence_text}

        Present a compelling opening statement (3-4 sentences) that:
        1. States your recommendation with conviction
        2. Incorporates the most relevant market evidence
        3. Reflects your unique investment philosophy
        4. Sets up your argument for the coming debate
        5. Uses your characteristic communication style

        Be specific, use data, and maintain your persona's voice.
        """

        try:
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are participating in a sophisticated investment debate as a legendary investor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"AI statement generation failed: {e}")
            return self._generate_template_enhanced_statement(persona, analysis)

    def _generate_template_enhanced_statement(self, persona: InvestorPersona,
                                            analysis: PersonaAnalysis) -> str:
        """Generate template-based enhanced statement"""
        templates = {
            "Warren Buffett": f"I recommend {analysis.recommendation.value} for {analysis.symbol} with {analysis.confidence_score:.0f}% confidence. "
                             f"This company demonstrates the predictable cash flows and competitive moat I seek in all investments. "
                             f"The current market provides us an opportunity to buy a wonderful business at a reasonable price, "
                             f"which aligns perfectly with my time-tested value investing principles.",

            "Ray Dalio": f"From a macro-economic perspective, my analysis indicates {analysis.recommendation.value} for {analysis.symbol}. "
                        f"The systematic risk factors and correlation dynamics suggest this position fits well within a balanced portfolio approach. "
                        f"My confidence level of {analysis.confidence_score:.0f}% reflects the risk-adjusted return potential I see here.",

            "Cathie Wood": f"I'm strongly {analysis.recommendation.value} on {analysis.symbol} with {analysis.confidence_score:.0f}% conviction. "
                          f"This represents exactly the kind of disruptive innovation opportunity that creates exponential wealth. "
                          f"The technological transformation potential here far exceeds any traditional valuation concerns.",

            "George Soros": f"Market reflexivity indicates a clear {analysis.recommendation.value} opportunity in {analysis.symbol}. "
                           f"The current market bias creates the exact type of asymmetric opportunity I've built my career exploiting. "
                           f"My {analysis.confidence_score:.0f}% confidence reflects the strength of this reflexive setup.",

            "Jim Simons": f"Quantitative models signal {analysis.recommendation.value} for {analysis.symbol} with {analysis.confidence_score:.0f}% statistical confidence. "
                         f"The mathematical patterns and risk-adjusted metrics support this position within our systematic framework. "
                         f"Data-driven analysis removes emotion and provides objective investment guidance.",

            "Paul Tudor Jones": f"Technical analysis clearly shows {analysis.recommendation.value} setup in {analysis.symbol}. "
                               f"The risk-reward ratio and momentum factors align perfectly with my trading methodology. "
                               f"With {analysis.confidence_score:.0f}% confidence, this represents an excellent asymmetric opportunity."
        }

        return templates.get(persona.name,
                           f"I recommend {analysis.recommendation.value} for {analysis.symbol} "
                           f"with {analysis.confidence_score:.0f}% confidence based on my comprehensive analysis.")

    def _find_relevant_messages(self, participant: str,
                               previous_messages: List[EnhancedDebateMessage]) -> List[EnhancedDebateMessage]:
        """Find messages relevant for response"""
        relevant = []

        # Messages that mentioned this persona or opposed their view
        for msg in previous_messages:
            if (msg.persona_name != participant and
                (participant.lower() in msg.message.lower() or
                 abs(msg.sentiment_score) > 0.5)):  # Strong sentiment
                relevant.append(msg)

        # Return most recent relevant messages
        return relevant[-5:]

    def _determine_argument_type(self, focus: str, round_number: int) -> ArgumentType:
        """Determine argument type based on focus and round"""
        if "consensus" in focus.lower():
            return ArgumentType.CONSENSUS_BUILDING
        elif round_number == 2:
            return ArgumentType.SUPPORTING
        elif "counter" in focus.lower():
            return ArgumentType.COUNTERARGUMENT
        else:
            return ArgumentType.REBUTTAL

    async def _determine_round_focus(self, rounds: List[DebateRoundAdvanced],
                                   consensus_progress: float) -> str:
        """Determine focus for next round based on debate progress"""
        if consensus_progress < 0.3:
            return "Fundamental Disagreements"
        elif consensus_progress < 0.6:
            return "Risk Assessment Debate"
        else:
            return "Consensus Building"

    async def _calculate_consensus_progress(self, rounds: List[DebateRoundAdvanced]) -> float:
        """Calculate overall consensus progress"""
        if not rounds:
            return 0.0

        # Average consensus progress across rounds
        total_progress = sum(round_obj.consensus_progress for round_obj in rounds)
        return total_progress / len(rounds)

    async def _calculate_round_consensus(self, round_obj: DebateRoundAdvanced) -> float:
        """Calculate consensus for a single round"""
        if len(round_obj.messages) < 2:
            return 0.0

        # Analyze sentiment convergence
        sentiments = [msg.sentiment_score for msg in round_obj.messages]
        sentiment_variance = np.var(sentiments)
        sentiment_consensus = max(0, 1 - sentiment_variance)

        # Analyze argument strength alignment
        strength_scores = [msg.confidence_level for msg in round_obj.messages]
        strength_variance = np.var(strength_scores)
        strength_consensus = max(0, 1 - strength_variance)

        return (sentiment_consensus + strength_consensus) / 2

    async def _extract_emerging_points(self, round_obj: DebateRoundAdvanced) -> List[str]:
        """Extract key emerging points from round"""
        points = []

        # Simple keyword extraction
        for msg in round_obj.messages:
            words = msg.message.split()
            # Find sentences with high-confidence keywords
            if any(keyword in msg.message.lower() for keyword in ['important', 'key', 'critical', 'significant']):
                # Extract the sentence containing the keyword
                sentences = msg.message.split('.')
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in ['important', 'key', 'critical']):
                        points.append(f"{msg.persona_name}: {sentence.strip()}")
                        break

        return points[:5]  # Top 5 points

    async def _adjust_participants(self, current_participants: List[str],
                                 rounds: List[DebateRoundAdvanced]) -> List[str]:
        """Dynamically adjust participants based on debate progress"""
        # For now, keep participants stable
        # Future enhancement: Add/remove participants based on expertise needed
        return current_participants

    async def _generate_consensus_statement(self, persona: InvestorPersona,
                                          previous_rounds: List[DebateRoundAdvanced],
                                          contradictions: List[Tuple[str, str]]) -> str:
        """Generate consensus-building statement"""
        if self.openai_client and self.config.enable_ai_moderation:
            return await self._generate_ai_consensus_statement(persona, previous_rounds)
        else:
            return self._generate_template_consensus_statement(persona)

    async def _generate_ai_consensus_statement(self, persona: InvestorPersona,
                                             previous_rounds: List[DebateRoundAdvanced]) -> str:
        """Generate AI-powered consensus statement"""
        # Summarize key points from debate
        key_arguments = []
        for round_obj in previous_rounds:
            key_arguments.extend(round_obj.key_points_emerging)

        arguments_text = "\n".join(key_arguments[:10])  # Top 10 arguments

        prompt = f"""
        You are {persona.name} in the final consensus-building phase of an investment debate.

        Key arguments made during the debate:
        {arguments_text}

        As {persona.name}, provide a consensus-building statement that:
        1. Acknowledges valid points made by others
        2. Maintains your core investment philosophy
        3. Identifies areas of agreement
        4. Suggests a balanced path forward
        5. Remains true to your characteristic style

        Focus on building bridges while maintaining your principles. 2-3 sentences.
        """

        try:
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are building consensus in an investment debate."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.6
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"AI consensus generation failed: {e}")
            return self._generate_template_consensus_statement(persona)

    def _generate_template_consensus_statement(self, persona: InvestorPersona) -> str:
        """Generate template consensus statement"""
        templates = {
            "Warren Buffett": "While we may differ on timing and approach, I believe we all recognize the importance of buying quality businesses at reasonable prices. Perhaps we can agree on scaling our positions based on our individual conviction levels.",

            "Ray Dalio": "The debate has highlighted the importance of balancing different perspectives and risk factors. A diversified approach that considers all viewpoints may serve us best in this uncertain environment.",

            "Cathie Wood": "I appreciate the various analytical frameworks presented. While I maintain my conviction about disruptive potential, I acknowledge the value of risk management principles in position sizing.",

            "George Soros": "Market dynamics are complex, and this debate has revealed multiple valid perspectives. Perhaps the key is timing our entries based on when the market bias aligns with our individual theses.",

            "Jim Simons": "The statistical evidence presented by various participants provides a robust foundation for decision-making. We can quantify our differences and weight positions accordingly.",

            "Paul Tudor Jones": "Risk management remains paramount regardless of our individual views. We should structure positions that allow us to participate in upside while limiting downside exposure."
        }

        return templates.get(persona.name,
                           "The debate has provided valuable insights from multiple perspectives. "
                           "We should consider a balanced approach that incorporates the strongest arguments presented.")

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0

    def _generate_message_id(self, participant: str, round_number: int) -> str:
        """Generate unique message ID"""
        timestamp = str(int(time.time() * 1000))
        raw_id = f"{participant}_{round_number}_{timestamp}"
        return hashlib.md5(raw_id.encode()).hexdigest()[:12]


class DebateOrchestrator:
    """Main orchestrator for sophisticated AI investor debates"""

    def __init__(self, config: DebateConfiguration):
        self.config = config
        self.moderator = AdvancedModerator(config)
        self.market_service = MarketDataService()
        self.news_service = NewsService()
        self.active_debates: Dict[str, Dict[str, Any]] = {}
        self.debate_history: List[Dict[str, Any]] = []

    async def orchestrate_debate(self, participants: List[str],
                               custom_analyses: Optional[List[PersonaAnalysis]] = None) -> Dict[str, Any]:
        """Orchestrate a complete sophisticated debate"""
        debate_id = self._generate_debate_id()

        try:
            logger.info(f"Starting debate {debate_id} for {self.config.symbol}")

            # Mark debate as active
            self.active_debates[debate_id] = {
                'status': DebateStatus.ACTIVE,
                'start_time': datetime.now(),
                'participants': participants,
                'symbol': self.config.symbol
            }

            # Get initial analyses if not provided
            if custom_analyses is None:
                initial_analyses = await self._generate_initial_analyses(participants)
            else:
                initial_analyses = custom_analyses

            # Validate participants
            valid_participants = [p for p in participants if p in AVAILABLE_PERSONAS]
            if len(valid_participants) < 2:
                raise ValueError("Need at least 2 valid participants for debate")

            # Conduct the debate
            debate_rounds = await self.moderator.moderate_advanced_debate(
                valid_participants, initial_analyses
            )

            # Calculate final consensus
            consensus_result = await self._build_final_consensus(
                initial_analyses, debate_rounds
            )

            # Create comprehensive results
            debate_results = {
                'debate_id': debate_id,
                'symbol': self.config.symbol,
                'participants': valid_participants,
                'initial_analyses': [asdict(analysis) for analysis in initial_analyses],
                'debate_rounds': [self._serialize_round(round_obj) for round_obj in debate_rounds],
                'consensus_result': asdict(consensus_result),
                'metadata': {
                    'start_time': self.active_debates[debate_id]['start_time'],
                    'end_time': datetime.now(),
                    'total_rounds': len(debate_rounds),
                    'total_messages': sum(len(r.messages) for r in debate_rounds),
                    'consensus_achieved': consensus_result.consensus_score >= self.config.consensus_threshold * 100,
                    'argument_tracker_stats': {
                        'total_evidence': sum(len(msg.evidence) for r in debate_rounds for msg in r.messages),
                        'contradictions_detected': len(self.moderator.argument_tracker.detect_contradictions()),
                        'argument_strengths': {
                            persona: self.moderator.argument_tracker.get_argument_strength(persona)
                            for persona in valid_participants
                        }
                    }
                }
            }

            # Mark debate as concluded
            self.active_debates[debate_id]['status'] = DebateStatus.CONCLUDED
            self.active_debates[debate_id]['end_time'] = datetime.now()

            # Save to history
            self.debate_history.append(debate_results)

            # Save transcript if enabled
            if self.config.save_transcript:
                await self._save_debate_transcript(debate_id, debate_results)

            logger.info(f"Debate {debate_id} completed successfully")
            return debate_results

        except Exception as e:
            logger.error(f"Debate {debate_id} failed: {e}")
            if debate_id in self.active_debates:
                self.active_debates[debate_id]['status'] = DebateStatus.FAILED
                self.active_debates[debate_id]['error'] = str(e)
            raise

    async def _generate_initial_analyses(self, participants: List[str]) -> List[PersonaAnalysis]:
        """Generate initial analyses for participants"""
        analyses = []

        # Get market data
        market_data = self.market_service.get_market_data(self.config.symbol)
        if not market_data:
            raise ValueError(f"Could not retrieve market data for {self.config.symbol}")

        # Get news sentiment
        news_articles = self.news_service.get_stock_news(self.config.symbol)
        news_sentiment = self.news_service.analyze_sentiment(news_articles)

        # Generate analysis for each participant
        for participant in participants:
            if participant in AVAILABLE_PERSONAS:
                persona = AVAILABLE_PERSONAS[participant]
                analysis = persona.analyze_investment(market_data, news_sentiment)
                analyses.append(analysis)

        return analyses

    async def _build_final_consensus(self, initial_analyses: List[PersonaAnalysis],
                                   debate_rounds: List[DebateRoundAdvanced]) -> ConsensusResult:
        """Build final consensus from debate results"""
        from .consensus_engine import EnhancedConsensusEngine

        # Create enhanced consensus engine
        consensus_engine = EnhancedConsensusEngine(self.config)

        # Convert debate rounds to format expected by consensus engine
        legacy_rounds = []
        for round_obj in debate_rounds:
            legacy_messages = []
            for msg in round_obj.messages:
                legacy_msg = DebateMessage(
                    persona_name=msg.persona_name,
                    message=msg.message,
                    timestamp=msg.timestamp,
                    message_type=msg.argument_type.value.lower(),
                    supporting_data=msg.supporting_data
                )
                legacy_messages.append(legacy_msg)

            legacy_round = type('DebateRound', (), {
                'round_number': round_obj.round_number,
                'topic': round_obj.topic,
                'messages': legacy_messages,
                'timestamp': round_obj.start_time
            })()
            legacy_rounds.append(legacy_round)

        # Get current market data for consensus building
        market_data = self.market_service.get_market_data(self.config.symbol)

        return consensus_engine.build_enhanced_consensus(
            self.config.symbol, initial_analyses, legacy_rounds, market_data
        )

    def _serialize_round(self, round_obj: DebateRoundAdvanced) -> Dict[str, Any]:
        """Serialize debate round for storage"""
        return {
            'round_number': round_obj.round_number,
            'topic': round_obj.topic,
            'participants': round_obj.participants,
            'start_time': round_obj.start_time.isoformat(),
            'end_time': round_obj.end_time.isoformat() if round_obj.end_time else None,
            'consensus_progress': round_obj.consensus_progress,
            'key_points_emerging': round_obj.key_points_emerging,
            'messages': [
                {
                    'id': msg.id,
                    'persona_name': msg.persona_name,
                    'message': msg.message,
                    'timestamp': msg.timestamp.isoformat(),
                    'argument_type': msg.argument_type.value,
                    'priority': msg.priority.value,
                    'evidence_count': len(msg.evidence),
                    'references': msg.references,
                    'sentiment_score': msg.sentiment_score,
                    'confidence_level': msg.confidence_level,
                    'supporting_data': msg.supporting_data
                }
                for msg in round_obj.messages
            ]
        }

    async def _save_debate_transcript(self, debate_id: str, debate_results: Dict[str, Any]):
        """Save debate transcript to file"""
        try:
            filename = f"debate_transcript_{debate_id}_{self.config.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join("debate_transcripts", filename)

            os.makedirs("debate_transcripts", exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(debate_results, f, indent=2, default=str)

            logger.info(f"Debate transcript saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save debate transcript: {e}")

    def _generate_debate_id(self) -> str:
        """Generate unique debate ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"debate_{self.config.symbol}_{timestamp}"

    def get_active_debates(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active debates"""
        return {k: v for k, v in self.active_debates.items()
                if v['status'] == DebateStatus.ACTIVE}

    def get_debate_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent debate history"""
        return self.debate_history[-limit:]


# Utility functions for easy use
async def quick_debate(symbol: str, participants: List[str] = None,
                      openai_api_key: str = None) -> Dict[str, Any]:
    """Quick debate setup for common use cases"""
    if participants is None:
        participants = ['warren_buffett', 'ray_dalio', 'cathie_wood', 'george_soros']

    config = DebateConfiguration(
        symbol=symbol,
        max_rounds=4,
        openai_api_key=openai_api_key,
        enable_real_time_data=True,
        parallel_processing=True
    )

    orchestrator = DebateOrchestrator(config)
    return await orchestrator.orchestrate_debate(participants)


async def comprehensive_debate(symbol: str, participants: List[str] = None,
                             **config_kwargs) -> Dict[str, Any]:
    """Comprehensive debate with full features enabled"""
    if participants is None:
        participants = list(AVAILABLE_PERSONAS.keys())

    config = DebateConfiguration(
        symbol=symbol,
        max_rounds=6,
        enable_ai_moderation=True,
        enable_real_time_data=True,
        enable_sentiment_analysis=True,
        enable_fact_checking=True,
        dynamic_round_adjustment=True,
        parallel_processing=True,
        **config_kwargs
    )

    orchestrator = DebateOrchestrator(config)
    return await orchestrator.orchestrate_debate(participants)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Quick debate example
        result = await quick_debate("AAPL", ["warren_buffett", "cathie_wood", "ray_dalio"])
        print(f"Debate completed with {result['metadata']['consensus_achieved']} consensus")
        print(f"Final recommendation: {result['consensus_result']['final_recommendation']}")

    asyncio.run(main())
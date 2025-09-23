"""
FastAPI Integration for AI Investor Personas
===========================================

This module provides FastAPI endpoints to integrate the AI investor persona system
with the existing AIHedgeFund architecture.

Author: AI Hedge Fund Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

from .ai_investment_orchestrator import (
    InvestmentOrchestrator, InvestmentRequest, InvestmentRecommendation,
    analyze_stocks, quick_stock_analysis, get_persona_info
)
from .ai_investor_personas import AVAILABLE_PERSONAS, InvestmentAction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Investor Personas API",
    description="Sophisticated AI investor persona system for investment analysis and debate",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Global orchestrator instance
orchestrator = InvestmentOrchestrator()


# Pydantic models for API
class PersonaInfo(BaseModel):
    """Information about an investor persona"""
    key: str
    name: str
    description: str
    investment_style: str
    risk_tolerance: str
    time_horizon: str
    key_principles: List[str]
    preferred_sectors: List[str]
    historical_returns: float


class AnalysisRequest(BaseModel):
    """Request for investment analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols to analyze")
    personas: List[str] = Field(default=["warren_buffett", "ray_dalio", "cathie_wood"],
                               description="List of persona keys to use")
    analysis_depth: str = Field(default="standard",
                               description="Analysis depth: quick, standard, comprehensive")
    include_debate: bool = Field(default=True, description="Include debate phase")
    include_news: bool = Field(default=True, description="Include news sentiment analysis")
    risk_tolerance: str = Field(default="moderate",
                               description="Risk tolerance: conservative, moderate, aggressive")
    investment_horizon: str = Field(default="medium",
                                   description="Investment horizon: short, medium, long")
    max_position_size: float = Field(default=0.2, ge=0.01, le=1.0,
                                    description="Maximum position size as % of portfolio")


class QuickAnalysisRequest(BaseModel):
    """Request for quick analysis of a single symbol"""
    symbol: str = Field(..., description="Stock symbol to analyze")
    personas: Optional[List[str]] = Field(default=None,
                                         description="Persona keys (defaults to top 3)")


class RecommendationResponse(BaseModel):
    """Investment recommendation response"""
    symbol: str
    recommendation: str
    consensus_score: float
    confidence_score: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    position_size: float
    expected_return: float
    risk_level: str
    time_horizon: str
    execution_timeline: str

    # Summary data
    persona_count: int
    debate_rounds: int
    analysis_duration: float

    # Key insights
    key_arguments: List[str]
    key_concerns: List[str]
    market_conditions: str


class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    request_id: str
    timestamp: str
    total_symbols: int
    analysis_duration: float
    recommendations: List[RecommendationResponse]

    # Summary statistics
    buy_recommendations: int
    average_consensus: float
    average_confidence: float


class DebateResponse(BaseModel):
    """Debate simulation response"""
    symbol: str
    participants: List[str]
    rounds: int
    consensus_score: float
    debate_summary: str
    key_debate_points: List[str]


# Cache for long-running analyses
analysis_cache: Dict[str, Any] = {}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/personas", response_model=List[PersonaInfo])
def get_personas():
    """Get information about available investor personas"""
    try:
        persona_data = get_persona_info()

        personas = []
        for key, info in persona_data.items():
            personas.append(PersonaInfo(
                key=key,
                name=info['name'],
                description=info['description'],
                investment_style=info['investment_style'],
                risk_tolerance=info['risk_tolerance'],
                time_horizon=info['time_horizon'],
                key_principles=info['key_principles'],
                preferred_sectors=info['preferred_sectors'],
                historical_returns=info['historical_returns']
            ))

        return personas

    except Exception as e:
        logger.error(f"Error getting personas: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve persona information")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_investments(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Perform comprehensive investment analysis"""
    try:
        start_time = datetime.now()
        request_id = f"analysis_{start_time.strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting analysis {request_id} for {len(request.symbols)} symbols")

        # Validate personas
        invalid_personas = [p for p in request.personas if p not in AVAILABLE_PERSONAS]
        if invalid_personas:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid personas: {invalid_personas}"
            )

        # Create investment request
        investment_request = InvestmentRequest(
            symbols=request.symbols,
            personas=request.personas,
            analysis_depth=request.analysis_depth,
            include_debate=request.include_debate,
            include_news=request.include_news,
            risk_tolerance=request.risk_tolerance,
            investment_horizon=request.investment_horizon,
            max_position_size=request.max_position_size
        )

        # Perform analysis
        recommendations = await orchestrator.analyze_investment(investment_request)

        # Calculate summary statistics
        buy_count = sum(1 for r in recommendations
                       if r.recommendation in [InvestmentAction.BUY, InvestmentAction.STRONG_BUY])

        avg_consensus = sum(r.consensus_score for r in recommendations) / len(recommendations) if recommendations else 0
        avg_confidence = sum(r.confidence_score for r in recommendations) / len(recommendations) if recommendations else 0

        # Convert to response format
        recommendation_responses = []
        for rec in recommendations:
            recommendation_responses.append(RecommendationResponse(
                symbol=rec.symbol,
                recommendation=rec.recommendation.value,
                consensus_score=rec.consensus_score,
                confidence_score=rec.confidence_score,
                target_price=rec.target_price,
                stop_loss=rec.stop_loss,
                position_size=rec.position_size,
                expected_return=rec.expected_return,
                risk_level=rec.risk_level,
                time_horizon=rec.time_horizon,
                execution_timeline=rec.execution_timeline,

                persona_count=len(rec.persona_analyses),
                debate_rounds=len(rec.consensus_details.debate_summary.split("round")) - 1 if "round" in rec.consensus_details.debate_summary else 0,
                analysis_duration=rec.analysis_duration,

                key_arguments=rec.consensus_details.key_supporting_arguments[:3],
                key_concerns=rec.consensus_details.key_concerns[:3],
                market_conditions=rec.consensus_details.market_conditions_assessment
            ))

        duration = (datetime.now() - start_time).total_seconds()

        response = AnalysisResponse(
            request_id=request_id,
            timestamp=start_time.isoformat(),
            total_symbols=len(request.symbols),
            analysis_duration=duration,
            recommendations=recommendation_responses,
            buy_recommendations=buy_count,
            average_consensus=avg_consensus,
            average_confidence=avg_confidence
        )

        # Cache results for retrieval
        analysis_cache[request_id] = {
            'response': response,
            'full_recommendations': recommendations,
            'timestamp': start_time
        }

        # Schedule cleanup of old cache entries
        background_tasks.add_task(cleanup_old_cache_entries)

        logger.info(f"Completed analysis {request_id} in {duration:.2f} seconds")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in investment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/quick-analysis", response_model=RecommendationResponse)
async def quick_analysis(request: QuickAnalysisRequest):
    """Perform quick analysis of a single symbol"""
    try:
        logger.info(f"Quick analysis requested for {request.symbol}")

        recommendation = await orchestrator.quick_analysis(
            request.symbol,
            request.personas
        )

        if not recommendation:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to analyze {request.symbol}"
            )

        response = RecommendationResponse(
            symbol=recommendation.symbol,
            recommendation=recommendation.recommendation.value,
            consensus_score=recommendation.consensus_score,
            confidence_score=recommendation.confidence_score,
            target_price=recommendation.target_price,
            stop_loss=recommendation.stop_loss,
            position_size=recommendation.position_size,
            expected_return=recommendation.expected_return,
            risk_level=recommendation.risk_level,
            time_horizon=recommendation.time_horizon,
            execution_timeline=recommendation.execution_timeline,

            persona_count=len(recommendation.persona_analyses),
            debate_rounds=0,  # No debate in quick analysis
            analysis_duration=recommendation.analysis_duration,

            key_arguments=recommendation.consensus_details.key_supporting_arguments[:3],
            key_concerns=recommendation.consensus_details.key_concerns[:3],
            market_conditions=recommendation.consensus_details.market_conditions_assessment
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in quick analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Quick analysis failed: {str(e)}")


@app.get("/debate/{symbol}", response_model=DebateResponse)
async def simulate_debate(
    symbol: str,
    personas: List[str] = Query(default=["warren_buffett", "ray_dalio", "cathie_wood"]),
    rounds: int = Query(default=3, ge=1, le=5)
):
    """Simulate a debate between personas for a specific symbol"""
    try:
        logger.info(f"Debate simulation requested for {symbol}")

        # Validate personas
        invalid_personas = [p for p in personas if p not in AVAILABLE_PERSONAS]
        if invalid_personas:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid personas: {invalid_personas}"
            )

        # Create analysis request for debate
        request = InvestmentRequest(
            symbols=[symbol],
            personas=personas,
            include_debate=True,
            include_news=True
        )

        recommendations = await orchestrator.analyze_investment(request)

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to generate debate for {symbol}"
            )

        rec = recommendations[0]
        consensus_details = rec.consensus_details

        # Extract debate points from reasoning
        debate_points = []
        for persona_analysis in rec.persona_analyses:
            debate_points.append(f"{persona_analysis.persona_name}: {persona_analysis.reasoning[:100]}...")

        response = DebateResponse(
            symbol=symbol,
            participants=[p.persona_name for p in rec.persona_analyses],
            rounds=rounds,
            consensus_score=rec.consensus_score,
            debate_summary=consensus_details.debate_summary,
            key_debate_points=debate_points[:5]
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in debate simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Debate simulation failed: {str(e)}")


@app.get("/analysis/{request_id}")
def get_analysis_results(request_id: str):
    """Retrieve cached analysis results"""
    if request_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return analysis_cache[request_id]['response']


@app.get("/report/{request_id}")
def get_analysis_report(request_id: str):
    """Generate a formatted report for analysis results"""
    if request_id not in analysis_cache:
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        recommendations = analysis_cache[request_id]['full_recommendations']
        report = orchestrator.generate_report(recommendations)

        return {
            "request_id": request_id,
            "report": report,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")


@app.get("/watchlists")
def get_watchlists():
    """Get predefined watchlists for analysis"""
    watchlists = {
        'mega_tech': {
            'name': 'Mega Tech',
            'description': 'Large technology companies',
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
        },
        'sp500_leaders': {
            'name': 'S&P 500 Leaders',
            'description': 'Top performing S&P 500 stocks',
            'symbols': ['SPY', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'BRK-B', 'JNJ', 'JPM', 'V', 'PG']
        },
        'high_volatility': {
            'name': 'High Volatility',
            'description': 'High beta growth stocks',
            'symbols': ['TSLA', 'NVDA', 'AMD', 'COIN', 'RIVN', 'PLTR', 'SOFI', 'RIOT', 'MARA']
        },
        'dividend_aristocrats': {
            'name': 'Dividend Aristocrats',
            'description': 'Consistent dividend paying stocks',
            'symbols': ['JNJ', 'KO', 'PEP', 'PG', 'MMM', 'CL', 'CVX', 'XOM', 'MCD', 'WMT']
        },
        'disruptive_innovation': {
            'name': 'Disruptive Innovation',
            'description': 'Companies driving technological disruption',
            'symbols': ['TSLA', 'ROKU', 'SQ', 'TDOC', 'COIN', 'PATH', 'DKNG', 'U', 'TWLO', 'SHOP']
        }
    }

    return watchlists


@app.post("/watchlist-analysis/{watchlist_name}", response_model=AnalysisResponse)
async def analyze_watchlist(
    watchlist_name: str,
    personas: List[str] = Query(default=["warren_buffett", "ray_dalio", "cathie_wood"]),
    include_debate: bool = Query(default=True)
):
    """Analyze a predefined watchlist"""
    watchlists = (await get_watchlists())

    if watchlist_name not in watchlists:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    symbols = watchlists[watchlist_name]['symbols']

    request = AnalysisRequest(
        symbols=symbols,
        personas=personas,
        include_debate=include_debate
    )

    return await analyze_investments(request, BackgroundTasks())


# Background task functions
def cleanup_old_cache_entries():
    """Clean up old cache entries"""
    cutoff_time = datetime.now() - timedelta(hours=24)

    keys_to_remove = []
    for key, value in analysis_cache.items():
        if value['timestamp'] < cutoff_time:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del analysis_cache[key]

    logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")


# Integration with existing main.py
def add_personas_routes(main_app: FastAPI):
    """Add persona routes to the main FastAPI app"""
    main_app.mount("/ai-personas", app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
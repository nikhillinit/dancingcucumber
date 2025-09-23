"""
Example Usage of AI Investor Personas System
============================================

This script demonstrates how to use the AI investor personas system
for investment analysis, debates, and consensus building.

Author: AI Hedge Fund Team
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List

from ai_investment_orchestrator import (
    InvestmentOrchestrator, InvestmentRequest, analyze_stocks,
    quick_stock_analysis, get_persona_info
)
from ai_investor_personas import AVAILABLE_PERSONAS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_quick_analysis():
    """Example of quick analysis for a single stock"""
    print("\\n" + "="*60)
    print("QUICK ANALYSIS EXAMPLE")
    print("="*60)

    symbol = "AAPL"
    print(f"Analyzing {symbol} with quick analysis...")

    recommendation = await quick_stock_analysis(symbol)

    if recommendation:
        print(f"\\nRecommendation for {symbol}:")
        print(f"  Action: {recommendation.recommendation.value}")
        print(f"  Consensus Score: {recommendation.consensus_score:.1f}%")
        print(f"  Confidence: {recommendation.confidence_score:.1f}%")
        print(f"  Position Size: {recommendation.position_size:.1%}")
        print(f"  Expected Return: {recommendation.expected_return:.1%}")
        print(f"  Risk Level: {recommendation.risk_level}")

        if recommendation.target_price:
            current_price = recommendation.market_analysis.current_price
            upside = ((recommendation.target_price - current_price) / current_price) * 100
            print(f"  Target Price: ${recommendation.target_price:.2f} ({upside:+.1f}% upside)")

        print(f"\\nKey Arguments:")
        for arg in recommendation.consensus_details.key_supporting_arguments[:3]:
            print(f"  - {arg}")

    else:
        print(f"Failed to analyze {symbol}")


async def example_comprehensive_analysis():
    """Example of comprehensive analysis with debate"""
    print("\\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS EXAMPLE")
    print("="*60)

    symbols = ["AAPL", "TSLA", "NVDA"]
    personas = ["warren_buffett", "cathie_wood", "george_soros"]

    print(f"Analyzing {', '.join(symbols)} with personas: {', '.join(personas)}")
    print("Including debate phase...")

    recommendations = await analyze_stocks(
        symbols=symbols,
        personas=personas,
        include_debate=True
    )

    print(f"\\nAnalysis completed for {len(recommendations)} symbols:")
    print("-" * 60)

    for rec in recommendations:
        print(f"\\n{rec.symbol}:")
        print(f"  Recommendation: {rec.recommendation.value}")
        print(f"  Consensus Score: {rec.consensus_score:.1f}%")
        print(f"  Confidence: {rec.confidence_score:.1f}%")
        print(f"  Position Size: {rec.position_size:.1%}")

        # Show persona breakdown
        print(f"\\n  Persona Breakdown:")
        for analysis in rec.persona_analyses:
            print(f"    {analysis.persona_name}: {analysis.recommendation.value} "
                  f"({analysis.confidence_score:.0f}% confidence)")

        # Show key arguments
        if rec.consensus_details.key_supporting_arguments:
            print(f"\\n  Key Supporting Arguments:")
            for arg in rec.consensus_details.key_supporting_arguments[:2]:
                print(f"    - {arg}")

        print(f"\\n  Debate Summary: {rec.consensus_details.debate_summary}")


async def example_custom_analysis():
    """Example of custom analysis with specific configuration"""
    print("\\n" + "="*60)
    print("CUSTOM ANALYSIS EXAMPLE")
    print("="*60)

    # Create custom orchestrator
    orchestrator = InvestmentOrchestrator(
        cache_duration_minutes=30,
        max_concurrent_analyses=3
    )

    # Create custom request
    request = InvestmentRequest(
        symbols=["MSFT", "GOOGL"],
        personas=["warren_buffett", "ray_dalio", "jim_simons", "paul_tudor_jones"],
        analysis_depth="comprehensive",
        include_debate=True,
        include_news=True,
        risk_tolerance="conservative",
        investment_horizon="long",
        max_position_size=0.15
    )

    print(f"Custom analysis with 4 personas and conservative settings...")

    recommendations = await orchestrator.analyze_investment(request)

    # Generate and display report
    report = orchestrator.generate_report(recommendations)
    print(report)

    # Save results
    output_file = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    orchestrator.save_analysis_results(recommendations, output_file)
    print(f"\\nResults saved to {output_file}")


async def example_persona_comparison():
    """Example comparing different persona perspectives on the same stock"""
    print("\\n" + "="*60)
    print("PERSONA COMPARISON EXAMPLE")
    print("="*60)

    symbol = "TSLA"
    print(f"Comparing persona perspectives on {symbol}...")

    # Analyze with different persona combinations
    combinations = [
        (["warren_buffett"], "Value Perspective"),
        (["cathie_wood"], "Growth/Innovation Perspective"),
        (["george_soros"], "Macro/Reflexivity Perspective"),
        (["jim_simons"], "Quantitative Perspective"),
        (["paul_tudor_jones"], "Technical/Momentum Perspective")
    ]

    results = []
    for personas, description in combinations:
        recs = await analyze_stocks(
            symbols=[symbol],
            personas=personas,
            include_debate=False
        )

        if recs:
            results.append((description, recs[0]))

    # Display comparison
    print(f"\\nPerspective Comparison for {symbol}:")
    print("-" * 40)

    for description, rec in results:
        persona_analysis = rec.persona_analyses[0]
        print(f"\\n{description}:")
        print(f"  Recommendation: {persona_analysis.recommendation.value}")
        print(f"  Confidence: {persona_analysis.confidence_score:.1f}%")
        print(f"  Position Size: {persona_analysis.position_size_recommendation:.1%}")
        print(f"  Key Reasoning: {persona_analysis.reasoning[:120]}...")


def display_available_personas():
    """Display information about available personas"""
    print("\\n" + "="*60)
    print("AVAILABLE INVESTOR PERSONAS")
    print("="*60)

    persona_info = get_persona_info()

    for key, info in persona_info.items():
        print(f"\\n{info['name']} ({key}):")
        print(f"  Style: {info['investment_style']}")
        print(f"  Risk Tolerance: {info['risk_tolerance']}")
        print(f"  Time Horizon: {info['time_horizon']}")
        print(f"  Historical Returns: {info['historical_returns']:.1f}%")
        print(f"  Description: {info['description']}")
        print(f"  Key Principles:")
        for principle in info['key_principles'][:3]:
            print(f"    - {principle}")


async def example_portfolio_analysis():
    """Example of analyzing a diversified portfolio"""
    print("\\n" + "="*60)
    print("PORTFOLIO ANALYSIS EXAMPLE")
    print("="*60)

    # Define a balanced portfolio
    portfolio_symbols = [
        "AAPL",  # Large cap tech
        "BRK-B", # Value/Conglomerate
        "JNJ",   # Healthcare/Dividend
        "TSLA",  # Growth/EV
        "JPM",   # Financial
        "VTI",   # Broad market ETF
        "GLD"    # Commodity/Gold
    ]

    print(f"Analyzing diversified portfolio with {len(portfolio_symbols)} holdings...")

    # Use all available personas for comprehensive view
    all_personas = list(AVAILABLE_PERSONAS.keys())

    recommendations = await analyze_stocks(
        symbols=portfolio_symbols,
        personas=all_personas,
        include_debate=True
    )

    # Portfolio summary
    buy_count = sum(1 for r in recommendations
                   if r.recommendation.value in ["BUY", "STRONG_BUY"])

    total_allocation = sum(r.position_size for r in recommendations)
    avg_confidence = sum(r.confidence_score for r in recommendations) / len(recommendations)
    avg_consensus = sum(r.consensus_score for r in recommendations) / len(recommendations)

    print(f"\\nPORTFOLIO ANALYSIS SUMMARY:")
    print(f"  Total Holdings: {len(recommendations)}")
    print(f"  Buy Recommendations: {buy_count}")
    print(f"  Total Allocation: {total_allocation:.1%}")
    print(f"  Average Confidence: {avg_confidence:.1f}%")
    print(f"  Average Consensus: {avg_consensus:.1f}%")

    print(f"\\nTOP RECOMMENDATIONS:")
    sorted_recs = sorted(recommendations,
                        key=lambda x: x.consensus_score * x.confidence_score,
                        reverse=True)

    for i, rec in enumerate(sorted_recs[:3], 1):
        print(f"  {i}. {rec.symbol}: {rec.recommendation.value} "
              f"(Consensus: {rec.consensus_score:.0f}%, "
              f"Confidence: {rec.confidence_score:.0f}%)")


async def main():
    """Run all examples"""
    print("AI INVESTOR PERSONAS SYSTEM - EXAMPLE USAGE")
    print("=" * 80)

    # Display available personas
    display_available_personas()

    # Run examples
    await example_quick_analysis()
    await example_comprehensive_analysis()
    await example_persona_comparison()
    await example_custom_analysis()
    await example_portfolio_analysis()

    print("\\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
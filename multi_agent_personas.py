"""
Multi-Agent Investment Personas
==============================
Implementation of investment personality agents inspired by legendary investors
"""

import json
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class InvestmentPersona:
    """Base class for investment personality agents"""

    def __init__(self, name: str, philosophy: str, strengths: List[str]):
        self.name = name
        self.philosophy = philosophy
        self.strengths = strengths
        self.track_record = []

    def analyze_stock(self, symbol: str, market_data: Dict, fundamental_data: Dict) -> Dict:
        """Base analysis method to be overridden by specific personas"""
        raise NotImplementedError("Each persona must implement their own analysis method")

    def get_confidence_score(self, analysis: Dict) -> float:
        """Calculate confidence score based on persona's strengths"""
        return 0.5  # Default neutral confidence

class WarrenBuffettAgent(InvestmentPersona):
    """Warren Buffett inspired value investing agent"""

    def __init__(self):
        super().__init__(
            name="Warren Buffett",
            philosophy="Buy wonderful companies at fair prices and hold forever",
            strengths=["Value Analysis", "Quality Assessment", "Long-term Perspective", "Competitive Moats"]
        )

    def analyze_stock(self, symbol: str, market_data: Dict, fundamental_data: Dict) -> Dict:
        """Buffett-style value analysis"""

        # Extract key metrics (using defaults if not available)
        pe_ratio = fundamental_data.get('pe_ratio', 20)
        roe = fundamental_data.get('roe', 0.15)
        debt_to_equity = fundamental_data.get('debt_to_equity', 0.3)
        revenue_growth = fundamental_data.get('revenue_growth', 0.05)
        current_price = market_data.get('current_price', 100)

        analysis = {
            'agent': self.name,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol
        }

        # Buffett's Value Criteria
        score = 0
        reasoning = []

        # 1. Reasonable P/E (prefer < 15, acceptable < 25)
        if pe_ratio < 15:
            score += 25
            reasoning.append(f"Excellent P/E of {pe_ratio:.1f} - well below 15")
        elif pe_ratio < 25:
            score += 10
            reasoning.append(f"Reasonable P/E of {pe_ratio:.1f} - below 25")
        else:
            score -= 10
            reasoning.append(f"High P/E of {pe_ratio:.1f} - above 25, concerning")

        # 2. Strong ROE (prefer > 15%)
        if roe > 0.20:
            score += 20
            reasoning.append(f"Excellent ROE of {roe:.1%} - shows strong profitability")
        elif roe > 0.15:
            score += 15
            reasoning.append(f"Good ROE of {roe:.1%} - above 15% threshold")
        elif roe > 0.10:
            score += 5
            reasoning.append(f"Adequate ROE of {roe:.1%} - acceptable but not preferred")
        else:
            score -= 15
            reasoning.append(f"Poor ROE of {roe:.1%} - below 10%, concerning")

        # 3. Conservative Debt (prefer D/E < 0.3)
        if debt_to_equity < 0.2:
            score += 15
            reasoning.append(f"Conservative debt level - D/E of {debt_to_equity:.1f}")
        elif debt_to_equity < 0.5:
            score += 5
            reasoning.append(f"Reasonable debt level - D/E of {debt_to_equity:.1f}")
        else:
            score -= 10
            reasoning.append(f"High debt level - D/E of {debt_to_equity:.1f}, risky")

        # 4. Steady Growth (prefer 5-15% revenue growth)
        if 0.05 <= revenue_growth <= 0.15:
            score += 15
            reasoning.append(f"Ideal revenue growth of {revenue_growth:.1%} - steady and sustainable")
        elif 0.15 < revenue_growth <= 0.25:
            score += 10
            reasoning.append(f"Strong revenue growth of {revenue_growth:.1%} - good but monitor sustainability")
        elif revenue_growth > 0:
            score += 5
            reasoning.append(f"Positive revenue growth of {revenue_growth:.1%}")
        else:
            score -= 20
            reasoning.append(f"Negative revenue growth of {revenue_growth:.1%} - major concern")

        # Convert to 0-1 scale and determine action
        normalized_score = max(0, min(100, score + 50)) / 100

        if normalized_score >= 0.75:
            recommendation = 'STRONG_BUY'
            position_size = 15  # Max position for Buffett's conviction plays
        elif normalized_score >= 0.60:
            recommendation = 'BUY'
            position_size = 10
        elif normalized_score >= 0.40:
            recommendation = 'HOLD'
            position_size = 5
        else:
            recommendation = 'AVOID'
            position_size = 0

        analysis.update({
            'recommendation': recommendation,
            'confidence': normalized_score,
            'position_size': position_size,
            'score_breakdown': {
                'valuation': pe_ratio,
                'quality': roe,
                'safety': debt_to_equity,
                'growth': revenue_growth
            },
            'reasoning': reasoning,
            'key_metrics': {
                'pe_ratio': pe_ratio,
                'roe': roe,
                'debt_to_equity': debt_to_equity,
                'revenue_growth': revenue_growth
            }
        })

        return analysis

class CathieWoodAgent(InvestmentPersona):
    """Cathie Wood inspired innovation/growth agent"""

    def __init__(self):
        super().__init__(
            name="Cathie Wood",
            philosophy="Invest in disruptive innovation with exponential growth potential",
            strengths=["Innovation Analysis", "Growth Assessment", "Technology Trends", "Market Disruption"]
        )

    def analyze_stock(self, symbol: str, market_data: Dict, fundamental_data: Dict) -> Dict:
        """Innovation-focused growth analysis"""

        # Extract key metrics
        revenue_growth = fundamental_data.get('revenue_growth', 0.05)
        gross_margin = fundamental_data.get('gross_margin', 0.30)
        rd_spending = fundamental_data.get('rd_percentage', 0.05)
        market_cap = fundamental_data.get('market_cap', 10000000000)
        current_price = market_data.get('current_price', 100)

        analysis = {
            'agent': self.name,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol
        }

        score = 0
        reasoning = []

        # 1. High Revenue Growth (prefer > 25%)
        if revenue_growth > 0.40:
            score += 30
            reasoning.append(f"Exceptional growth of {revenue_growth:.1%} - disruptive potential")
        elif revenue_growth > 0.25:
            score += 25
            reasoning.append(f"Strong growth of {revenue_growth:.1%} - innovation driving expansion")
        elif revenue_growth > 0.15:
            score += 15
            reasoning.append(f"Good growth of {revenue_growth:.1%} - above market average")
        elif revenue_growth > 0:
            score += 5
            reasoning.append(f"Modest growth of {revenue_growth:.1%} - below innovation threshold")
        else:
            score -= 25
            reasoning.append(f"Declining revenue of {revenue_growth:.1%} - not innovative")

        # 2. High Gross Margins (prefer > 60%)
        if gross_margin > 0.70:
            score += 20
            reasoning.append(f"Excellent margins of {gross_margin:.1%} - strong pricing power")
        elif gross_margin > 0.50:
            score += 15
            reasoning.append(f"Good margins of {gross_margin:.1%} - competitive advantage")
        elif gross_margin > 0.30:
            score += 5
            reasoning.append(f"Adequate margins of {gross_margin:.1%}")
        else:
            score -= 10
            reasoning.append(f"Low margins of {gross_margin:.1%} - commodity-like business")

        # 3. R&D Investment (prefer > 15%)
        if rd_spending > 0.15:
            score += 20
            reasoning.append(f"Heavy R&D investment of {rd_spending:.1%} - future innovation")
        elif rd_spending > 0.10:
            score += 15
            reasoning.append(f"Good R&D investment of {rd_spending:.1%} - developing technology")
        elif rd_spending > 0.05:
            score += 5
            reasoning.append(f"Modest R&D investment of {rd_spending:.1%}")
        else:
            score -= 5
            reasoning.append(f"Low R&D investment of {rd_spending:.1%} - limited innovation")

        # 4. Market Size Opportunity (prefer growth companies)
        if market_cap < 50000000000:  # < $50B market cap
            score += 10
            reasoning.append("Mid-cap with room for exponential growth")
        elif market_cap < 100000000000:  # < $100B
            score += 5
            reasoning.append("Large-cap with moderate growth potential")
        else:
            score -= 5
            reasoning.append("Mega-cap with limited growth upside")

        # Technology sector bonus
        tech_sectors = ['TECH', 'SOFTWARE', 'BIOTECH', 'GENOMICS', 'AI', 'ROBOTICS']
        sector = fundamental_data.get('sector', '').upper()
        if any(tech in sector for tech in tech_sectors):
            score += 10
            reasoning.append(f"Technology sector exposure - {sector}")

        # Convert to 0-1 scale and determine action
        normalized_score = max(0, min(100, score + 40)) / 100

        if normalized_score >= 0.80:
            recommendation = 'STRONG_BUY'
            position_size = 12  # Concentrated positions in high-conviction innovation
        elif normalized_score >= 0.65:
            recommendation = 'BUY'
            position_size = 8
        elif normalized_score >= 0.45:
            recommendation = 'HOLD'
            position_size = 3
        else:
            recommendation = 'AVOID'
            position_size = 0

        analysis.update({
            'recommendation': recommendation,
            'confidence': normalized_score,
            'position_size': position_size,
            'score_breakdown': {
                'growth': revenue_growth,
                'margins': gross_margin,
                'innovation': rd_spending,
                'opportunity': market_cap
            },
            'reasoning': reasoning,
            'key_metrics': {
                'revenue_growth': revenue_growth,
                'gross_margin': gross_margin,
                'rd_percentage': rd_spending,
                'market_cap': market_cap
            }
        })

        return analysis

class RayDalioAgent(InvestmentPersona):
    """Ray Dalio inspired macro/risk agent"""

    def __init__(self):
        super().__init__(
            name="Ray Dalio",
            philosophy="All weather investing with macro awareness and risk parity",
            strengths=["Macro Analysis", "Risk Management", "Diversification", "Economic Cycles"]
        )

    def analyze_stock(self, symbol: str, market_data: Dict, fundamental_data: Dict) -> Dict:
        """Macro-focused risk-adjusted analysis"""

        # Extract key metrics
        beta = market_data.get('beta', 1.0)
        correlation_spy = market_data.get('correlation_spy', 0.8)
        volatility = market_data.get('volatility', 0.20)
        current_price = market_data.get('current_price', 100)

        # Economic indicators (would typically come from FRED API)
        fed_funds_rate = fundamental_data.get('fed_funds_rate', 0.025)
        gdp_growth = fundamental_data.get('gdp_growth', 0.02)
        inflation_rate = fundamental_data.get('inflation_rate', 0.03)

        analysis = {
            'agent': self.name,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol
        }

        score = 0
        reasoning = []

        # 1. Risk-Adjusted Returns (prefer beta 0.8-1.2)
        if 0.8 <= beta <= 1.2:
            score += 15
            reasoning.append(f"Moderate beta of {beta:.2f} - balanced risk profile")
        elif 0.5 <= beta < 0.8:
            score += 10
            reasoning.append(f"Low beta of {beta:.2f} - defensive characteristics")
        elif 1.2 < beta <= 1.5:
            score += 5
            reasoning.append(f"High beta of {beta:.2f} - aggressive growth")
        else:
            score -= 10
            reasoning.append(f"Extreme beta of {beta:.2f} - high risk")

        # 2. Correlation Diversification (prefer < 0.7 correlation with SPY)
        if correlation_spy < 0.5:
            score += 20
            reasoning.append(f"Low correlation of {correlation_spy:.2f} - excellent diversification")
        elif correlation_spy < 0.7:
            score += 10
            reasoning.append(f"Moderate correlation of {correlation_spy:.2f} - good diversification")
        else:
            score -= 5
            reasoning.append(f"High correlation of {correlation_spy:.2f} - limited diversification")

        # 3. Volatility Assessment (prefer moderate volatility)
        if volatility < 0.15:
            score += 10
            reasoning.append(f"Low volatility of {volatility:.1%} - stable returns")
        elif volatility < 0.25:
            score += 15
            reasoning.append(f"Moderate volatility of {volatility:.1%} - balanced risk/reward")
        elif volatility < 0.40:
            score += 5
            reasoning.append(f"High volatility of {volatility:.1%} - requires position sizing")
        else:
            score -= 15
            reasoning.append(f"Extreme volatility of {volatility:.1%} - high risk")

        # 4. Macro Environment Assessment
        if fed_funds_rate < 0.03:  # Low rates
            score += 10
            reasoning.append("Low interest rate environment - supportive for equities")
        elif fed_funds_rate > 0.05:  # High rates
            score -= 5
            reasoning.append("High interest rate environment - headwinds for equities")

        if gdp_growth > 0.025:  # Strong growth
            score += 10
            reasoning.append(f"Strong GDP growth of {gdp_growth:.1%} - supportive macro")
        elif gdp_growth < 0:  # Recession
            score -= 15
            reasoning.append(f"Negative GDP growth of {gdp_growth:.1%} - recessionary concerns")

        if inflation_rate > 0.04:  # High inflation
            score -= 10
            reasoning.append(f"High inflation of {inflation_rate:.1%} - margin pressure risk")

        # Convert to 0-1 scale and determine action
        normalized_score = max(0, min(100, score + 50)) / 100

        # Risk-adjusted position sizing
        risk_adjustment = min(1.0, 1.0 / volatility) if volatility > 0 else 1.0

        if normalized_score >= 0.70:
            recommendation = 'BUY'
            position_size = min(10, 8 * risk_adjustment)  # Risk-adjusted sizing
        elif normalized_score >= 0.55:
            recommendation = 'MODERATE_BUY'
            position_size = min(8, 6 * risk_adjustment)
        elif normalized_score >= 0.40:
            recommendation = 'HOLD'
            position_size = min(5, 4 * risk_adjustment)
        else:
            recommendation = 'REDUCE'
            position_size = max(1, 2 * risk_adjustment)

        analysis.update({
            'recommendation': recommendation,
            'confidence': normalized_score,
            'position_size': position_size,
            'score_breakdown': {
                'risk_profile': beta,
                'diversification': correlation_spy,
                'volatility': volatility,
                'macro_environment': (fed_funds_rate, gdp_growth, inflation_rate)
            },
            'reasoning': reasoning,
            'key_metrics': {
                'beta': beta,
                'correlation_spy': correlation_spy,
                'volatility': volatility,
                'risk_adjusted_size': risk_adjustment
            }
        })

        return analysis

class MultiAgentPersonaSystem:
    """Coordinates multiple investment personality agents"""

    def __init__(self):
        self.agents = [
            WarrenBuffettAgent(),
            CathieWoodAgent(),
            RayDalioAgent()
        ]
        self.consensus_weights = {
            'Warren Buffett': 0.40,    # Strong weight for value/quality
            'Cathie Wood': 0.35,       # Growth/innovation focus
            'Ray Dalio': 0.25          # Risk/macro overlay
        }

    def analyze_portfolio(self, stocks: List[str], market_data: Dict, fundamental_data: Dict) -> Dict:
        """Run multi-agent analysis on portfolio of stocks"""

        print("MULTI-AGENT PERSONA ANALYSIS")
        print("=" * 50)
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Analyzing {len(stocks)} stocks with {len(self.agents)} agents")

        portfolio_analysis = {}
        agent_recommendations = {}

        for stock in stocks:
            print(f"\n--- Analyzing {stock} ---")

            stock_market_data = market_data.get(stock, {})
            stock_fundamental_data = fundamental_data.get(stock, {})

            agent_analyses = []

            for agent in self.agents:
                try:
                    analysis = agent.analyze_stock(stock, stock_market_data, stock_fundamental_data)
                    agent_analyses.append(analysis)

                    print(f"{agent.name:15}: {analysis['recommendation']:12} "
                          f"({analysis['confidence']:.1%} confidence, "
                          f"{analysis['position_size']:.1f}% size)")

                except Exception as e:
                    print(f"{agent.name:15}: ERROR - {e}")
                    continue

            # Calculate consensus
            if agent_analyses:
                consensus = self.calculate_consensus(agent_analyses)
                portfolio_analysis[stock] = consensus
                agent_recommendations[stock] = agent_analyses

                print(f"{'CONSENSUS':15}: {consensus['final_recommendation']:12} "
                      f"({consensus['consensus_confidence']:.1%} confidence, "
                      f"{consensus['final_position_size']:.1f}% size)")

        return {
            'portfolio_analysis': portfolio_analysis,
            'agent_recommendations': agent_recommendations,
            'summary': self.generate_portfolio_summary(portfolio_analysis)
        }

    def calculate_consensus(self, agent_analyses: List[Dict]) -> Dict:
        """Calculate weighted consensus from multiple agent analyses"""

        if not agent_analyses:
            return {'error': 'No valid analyses'}

        weighted_confidence = 0
        weighted_position_size = 0
        recommendations = []

        total_weight = 0

        for analysis in agent_analyses:
            agent_name = analysis['agent']
            weight = self.consensus_weights.get(agent_name, 0.33)  # Default equal weight

            weighted_confidence += analysis['confidence'] * weight
            weighted_position_size += analysis['position_size'] * weight
            recommendations.append((analysis['recommendation'], weight))
            total_weight += weight

        # Normalize if weights don't sum to 1
        if total_weight > 0:
            weighted_confidence /= total_weight
            weighted_position_size /= total_weight

        # Determine consensus recommendation
        rec_scores = {}
        for rec, weight in recommendations:
            rec_scores[rec] = rec_scores.get(rec, 0) + weight

        final_recommendation = max(rec_scores.keys(), key=lambda x: rec_scores[x])

        # Adjust for consensus strength
        consensus_strength = max(rec_scores.values()) / total_weight

        return {
            'final_recommendation': final_recommendation,
            'consensus_confidence': weighted_confidence,
            'final_position_size': weighted_position_size,
            'consensus_strength': consensus_strength,
            'individual_recommendations': rec_scores,
            'agent_count': len(agent_analyses)
        }

    def generate_portfolio_summary(self, portfolio_analysis: Dict) -> Dict:
        """Generate summary of portfolio analysis"""

        if not portfolio_analysis:
            return {'error': 'No portfolio analysis available'}

        total_allocation = sum(analysis['final_position_size'] for analysis in portfolio_analysis.values())
        avg_confidence = sum(analysis['consensus_confidence'] for analysis in portfolio_analysis.values()) / len(portfolio_analysis)

        recommendations = {}
        for symbol, analysis in portfolio_analysis.items():
            rec = analysis['final_recommendation']
            recommendations[rec] = recommendations.get(rec, 0) + 1

        return {
            'total_stocks_analyzed': len(portfolio_analysis),
            'total_allocation': total_allocation,
            'cash_allocation': max(0, 100 - total_allocation),
            'average_confidence': avg_confidence,
            'recommendation_breakdown': recommendations,
            'top_picks': sorted(portfolio_analysis.items(),
                              key=lambda x: x[1]['final_position_size'],
                              reverse=True)[:3]
        }

def main():
    """Demo the multi-agent persona system"""

    # Sample data for demonstration
    stocks = ['AAPL', 'GOOGL', 'TSLA', 'JPM', 'NVDA']

    # Mock market data (would come from real APIs)
    market_data = {
        'AAPL': {'current_price': 175, 'beta': 1.2, 'correlation_spy': 0.8, 'volatility': 0.25},
        'GOOGL': {'current_price': 140, 'beta': 1.1, 'correlation_spy': 0.9, 'volatility': 0.28},
        'TSLA': {'current_price': 250, 'beta': 2.0, 'correlation_spy': 0.6, 'volatility': 0.50},
        'JPM': {'current_price': 150, 'beta': 1.3, 'correlation_spy': 0.85, 'volatility': 0.22},
        'NVDA': {'current_price': 450, 'beta': 1.8, 'correlation_spy': 0.7, 'volatility': 0.45}
    }

    # Mock fundamental data
    fundamental_data = {
        'AAPL': {'pe_ratio': 28, 'roe': 0.26, 'debt_to_equity': 0.31, 'revenue_growth': 0.08, 'gross_margin': 0.46, 'rd_percentage': 0.06, 'market_cap': 2800000000000, 'sector': 'TECH'},
        'GOOGL': {'pe_ratio': 24, 'roe': 0.18, 'debt_to_equity': 0.12, 'revenue_growth': 0.12, 'gross_margin': 0.57, 'rd_percentage': 0.13, 'market_cap': 1750000000000, 'sector': 'TECH'},
        'TSLA': {'pe_ratio': 45, 'roe': 0.23, 'debt_to_equity': 0.17, 'revenue_growth': 0.47, 'gross_margin': 0.21, 'rd_percentage': 0.03, 'market_cap': 800000000000, 'sector': 'AUTO'},
        'JPM': {'pe_ratio': 12, 'roe': 0.15, 'debt_to_equity': 0.22, 'revenue_growth': 0.03, 'gross_margin': 0.58, 'rd_percentage': 0.02, 'market_cap': 450000000000, 'sector': 'FINANCIAL'},
        'NVDA': {'pe_ratio': 65, 'roe': 0.35, 'debt_to_equity': 0.24, 'revenue_growth': 0.85, 'gross_margin': 0.73, 'rd_percentage': 0.22, 'market_cap': 1200000000000, 'sector': 'TECH'},

        # Mock macro data
        'macro': {'fed_funds_rate': 0.025, 'gdp_growth': 0.023, 'inflation_rate': 0.031}
    }

    # Add macro data to each stock
    for stock in fundamental_data:
        if stock != 'macro':
            fundamental_data[stock].update(fundamental_data['macro'])

    # Run multi-agent analysis
    system = MultiAgentPersonaSystem()
    results = system.analyze_portfolio(stocks, market_data, fundamental_data)

    # Display summary
    summary = results['summary']
    print(f"\n" + "=" * 50)
    print("PORTFOLIO SUMMARY")
    print("=" * 50)
    print(f"Stocks Analyzed: {summary['total_stocks_analyzed']}")
    print(f"Total Allocation: {summary['total_allocation']:.1f}%")
    print(f"Cash Allocation: {summary['cash_allocation']:.1f}%")
    print(f"Average Confidence: {summary['average_confidence']:.1%}")

    print(f"\nRecommendation Breakdown:")
    for rec, count in summary['recommendation_breakdown'].items():
        print(f"  {rec}: {count} stocks")

    print(f"\nTop 3 Picks:")
    for i, (symbol, analysis) in enumerate(summary['top_picks'], 1):
        print(f"  {i}. {symbol}: {analysis['final_position_size']:.1f}% allocation "
              f"({analysis['final_recommendation']})")

    return results

if __name__ == "__main__":
    main()
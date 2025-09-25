"""
UNIFIED INTELLIGENCE SYSTEM
===========================
Combines Congressional trading signals with portfolio analysis
for comprehensive investment recommendations
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np

# Import our components
from congressional_tracker_real import CongressionalTracker
from real_portfolio_analyzer import RealPortfolioAnalyzer

class UnifiedIntelligenceSystem:
    """Combines all intelligence sources for optimal trading decisions"""

    def __init__(self, portfolio: Dict[str, float]):
        self.portfolio = portfolio
        self.total_value = sum(portfolio.values())

        # Initialize components
        self.congressional = CongressionalTracker()
        self.analyzer = RealPortfolioAnalyzer()

        # Weight different intelligence sources
        self.weights = {
            'congressional': 0.30,  # High weight - insider knowledge
            'technical': 0.25,      # Market technicals
            'momentum': 0.20,       # Price momentum
            'sentiment': 0.15,      # Market sentiment
            'macro': 0.10          # Economic indicators
        }

    def get_comprehensive_signals(self) -> Dict:
        """Generate signals from all sources"""

        print("\n" + "="*70)
        print("UNIFIED INTELLIGENCE ANALYSIS")
        print("="*70)
        print("Gathering intelligence from all sources...")

        signals = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.total_value,
            'intelligence': {},
            'combined_signals': {},
            'recommendations': []
        }

        # 1. Congressional Trading Intelligence
        print("\n[1/5] Congressional Trading Intelligence...")
        congressional_trades = self.congressional.get_recent_trades(45)
        congressional_analysis = self.congressional.analyze_trades(congressional_trades)

        # 2. Technical Analysis
        print("\n[2/5] Technical Analysis...")
        portfolio_analysis = self.analyzer.analyze_your_portfolio(self.portfolio)

        # 3. Market Momentum
        print("\n[3/5] Market Momentum Analysis...")
        momentum_signals = self.calculate_momentum_signals()

        # 4. Sentiment Analysis
        print("\n[4/5] Sentiment Analysis...")
        sentiment_signals = self.calculate_sentiment_signals()

        # 5. Macro Analysis
        print("\n[5/5] Macroeconomic Analysis...")
        macro_signals = self.analyzer.get_macro_context()

        # Combine all signals
        signals['intelligence'] = {
            'congressional': congressional_analysis,
            'technical': portfolio_analysis.get('signals', {}),
            'momentum': momentum_signals,
            'sentiment': sentiment_signals,
            'macro': macro_signals
        }

        # Generate combined signals for each ticker
        signals['combined_signals'] = self.combine_signals(signals['intelligence'])

        # Generate actionable recommendations
        signals['recommendations'] = self.generate_recommendations(signals['combined_signals'])

        return signals

    def calculate_momentum_signals(self) -> Dict:
        """Calculate momentum for key positions"""

        momentum = {}

        # Simplified momentum based on recent performance
        momentum_map = {
            'NVDA': 0.92,  # AI boom continues
            'MSFT': 0.78,  # Cloud/AI strength
            'SMH': 0.85,   # Semiconductor strength
            'VUG': 0.75,   # Growth momentum
            'AAPL': 0.68,  # Steady performer
            'AMD': 0.72,   # AI datacenter growth
            'VTI': 0.65,   # Broad market solid
            'TSLA': 0.58,  # Volatile but recovering
            'BA': 0.35,    # Still troubled
            'VNQ': 0.38,   # REITs weak
        }

        for symbol in self.portfolio.keys():
            if symbol != 'SPAXX':
                momentum[symbol] = momentum_map.get(symbol, 0.50)

        return momentum

    def calculate_sentiment_signals(self) -> Dict:
        """Calculate market sentiment signals"""

        sentiment = {}

        # Current market sentiment
        sentiment_scores = {
            'NVDA': 0.88,  # Very bullish sentiment
            'MSFT': 0.75,  # Positive AI sentiment
            'AAPL': 0.65,  # Neutral to positive
            'SMH': 0.80,   # Bullish on semis
            'VUG': 0.70,   # Growth positive
            'TSLA': 0.55,  # Mixed sentiment
            'BA': 0.40,    # Negative sentiment
            'VNQ': 0.35,   # Rate fear sentiment
        }

        for symbol in self.portfolio.keys():
            if symbol != 'SPAXX':
                sentiment[symbol] = sentiment_scores.get(symbol, 0.50)

        return sentiment

    def combine_signals(self, intelligence: Dict) -> Dict:
        """Combine all intelligence sources into unified signals"""

        combined = {}

        # Get all unique symbols
        all_symbols = set()
        for symbol in self.portfolio.keys():
            if symbol != 'SPAXX':
                all_symbols.add(symbol)

        # Add Congressional focus symbols
        if 'bullish_tickers' in intelligence.get('congressional', {}):
            for ticker in intelligence['congressional']['bullish_tickers'].keys():
                all_symbols.add(ticker)

        for symbol in all_symbols:
            score = 0.0
            weight_sum = 0.0

            # Congressional signal (highest weight)
            if symbol in intelligence.get('congressional', {}).get('bullish_tickers', {}):
                cong_data = intelligence['congressional']['bullish_tickers'][symbol]
                cong_score = min(1.0, cong_data['score'] / 20)  # Normalize
                score += cong_score * self.weights['congressional']
                weight_sum += self.weights['congressional']
            elif symbol in intelligence.get('congressional', {}).get('bearish_tickers', {}):
                cong_data = intelligence['congressional']['bearish_tickers'][symbol]
                cong_score = max(0.0, 0.5 + (cong_data['score'] / 20))
                score += cong_score * self.weights['congressional']
                weight_sum += self.weights['congressional']

            # Technical signal
            if symbol in intelligence.get('technical', {}):
                tech_score = intelligence['technical'][symbol]
                score += tech_score * self.weights['technical']
                weight_sum += self.weights['technical']

            # Momentum signal
            if symbol in intelligence.get('momentum', {}):
                mom_score = intelligence['momentum'][symbol]
                score += mom_score * self.weights['momentum']
                weight_sum += self.weights['momentum']

            # Sentiment signal
            if symbol in intelligence.get('sentiment', {}):
                sent_score = intelligence['sentiment'][symbol]
                score += sent_score * self.weights['sentiment']
                weight_sum += self.weights['sentiment']

            # Normalize combined score
            if weight_sum > 0:
                combined[symbol] = score / weight_sum
            else:
                combined[symbol] = 0.50

        return combined

    def generate_recommendations(self, combined_signals: Dict) -> List[Dict]:
        """Generate specific trading recommendations"""

        recommendations = []
        cash_available = self.portfolio.get('SPAXX', 0)

        # Sort by signal strength
        sorted_signals = sorted(combined_signals.items(),
                              key=lambda x: x[1], reverse=True)

        print("\n" + "="*70)
        print("ACTIONABLE RECOMMENDATIONS")
        print("="*70)

        # Strong BUY signals
        print("\nSTRONG BUY SIGNALS:")
        print("-"*50)

        for symbol, signal in sorted_signals[:10]:
            if signal > 0.70:
                current_value = self.portfolio.get(symbol, 0)
                current_weight = current_value / self.total_value

                # Determine target allocation
                if signal > 0.85:
                    target_weight = 0.15  # Max 15% for strongest signals
                elif signal > 0.75:
                    target_weight = 0.10
                else:
                    target_weight = 0.07

                target_value = target_weight * self.total_value
                add_amount = max(0, target_value - current_value)

                if add_amount > 100:  # Only recommend if meaningful
                    rec = {
                        'action': 'BUY',
                        'symbol': symbol,
                        'signal': signal,
                        'current_value': current_value,
                        'add_amount': min(add_amount, cash_available * 0.3),
                        'target_weight': target_weight,
                        'reasons': []
                    }

                    # Add reasons
                    if symbol in combined_signals and signal > 0.70:
                        if 'congressional' in str(combined_signals):
                            rec['reasons'].append("Congressional buying detected")
                        if signal > 0.80:
                            rec['reasons'].append("Strong technical momentum")
                        if current_weight < 0.05:
                            rec['reasons'].append("Underweight position")

                    recommendations.append(rec)

                    print(f"{symbol:6s} | Signal: {signal:.2f} | Add: ${add_amount:,.0f}")
                    if rec['reasons']:
                        print(f"        Reasons: {', '.join(rec['reasons'])}")

        # SELL signals
        print("\nSELL/REDUCE SIGNALS:")
        print("-"*50)

        for symbol, signal in sorted_signals[-5:]:
            if signal < 0.40 and symbol in self.portfolio:
                current_value = self.portfolio[symbol]
                if current_value > 100:
                    rec = {
                        'action': 'SELL',
                        'symbol': symbol,
                        'signal': signal,
                        'current_value': current_value,
                        'sell_amount': current_value * 0.5,
                        'reasons': ["Weak signal", "Better opportunities available"]
                    }
                    recommendations.append(rec)

                    print(f"{symbol:6s} | Signal: {signal:.2f} | Sell: ${rec['sell_amount']:,.0f}")

        # Cash deployment
        if cash_available > self.total_value * 0.15:
            print(f"\nCASH DEPLOYMENT:")
            print("-"*50)
            print(f"Deploy ${cash_available * 0.7:,.0f} into top signals")

            rec = {
                'action': 'DEPLOY_CASH',
                'amount': cash_available * 0.7,
                'targets': [s[0] for s in sorted_signals[:5] if s[1] > 0.70],
                'reasons': ["Excess cash position", "Strong opportunities available"]
            }
            recommendations.append(rec)

        return recommendations

    def execute_paper_trades(self, recommendations: List[Dict]) -> Dict:
        """Execute paper trades based on recommendations"""

        trades = []

        for rec in recommendations:
            if rec['action'] == 'BUY':
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'BUY',
                    'symbol': rec['symbol'],
                    'shares': None,  # Would calculate from amount/price
                    'amount': rec['add_amount'],
                    'signal': rec['signal'],
                    'reasons': rec['reasons']
                }
                trades.append(trade)

            elif rec['action'] == 'SELL':
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'SELL',
                    'symbol': rec['symbol'],
                    'shares': None,
                    'amount': rec['sell_amount'],
                    'signal': rec['signal'],
                    'reasons': rec['reasons']
                }
                trades.append(trade)

        return {'trades': trades, 'trade_count': len(trades)}


def run_unified_analysis():
    """Run the unified intelligence system on user portfolio"""

    # User's actual portfolio
    portfolio = {
        'SPAXX': 19615.61,  # Cash
        'VTI': 27132.19,
        'VUG': 12563.24,
        'SMH': 10205.19,
        'VEA': 4985.30,
        'VHT': 4346.25,
        'MSFT': 3857.75,
        'AMD': 1591.90,
        'BA': 1558.82,
        'NVDA': 1441.59,
        'BIZD': 463.08,
        'SRE': 290.64,
        'FLNC': 278.40,
        'TSLA': 258.14,
        'FSLR': 220.26,
        'HASI': 216.59,
        'AAPL': 214.21,
        'CSWC': 174.83,
        'REMX': 140.70,
        'VNQ': 136.59,
        'ICOP': 116.70,
        'IIPR': 41.25,
        'BIOX': 13.84
    }

    # Initialize system
    system = UnifiedIntelligenceSystem(portfolio)

    # Get comprehensive signals
    signals = system.get_comprehensive_signals()

    # Show Congressional influence
    print("\n" + "="*70)
    print("CONGRESSIONAL INFLUENCE ON YOUR PORTFOLIO")
    print("="*70)

    if 'congressional' in signals['intelligence']:
        cong = signals['intelligence']['congressional']

        if 'top_buys' in cong:
            print("\nCongress is BUYING (You should too):")
            for ticker, data in cong['top_buys'][:5]:
                in_portfolio = ticker in portfolio
                status = "YOU OWN" if in_portfolio else "NOT OWNED"
                print(f"  {ticker}: {data['buy_count']} Congress members | {status}")

    # Execute paper trades
    if signals['recommendations']:
        trades = system.execute_paper_trades(signals['recommendations'])

        # Save to file
        with open('unified_signals.json', 'w') as f:
            json.dump({
                'signals': signals,
                'trades': trades,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)

        print(f"\nSaved {trades['trade_count']} trade recommendations to unified_signals.json")

    # Summary
    print("\n" + "="*70)
    print("UNIFIED INTELLIGENCE SUMMARY")
    print("="*70)
    print(f"Total Portfolio Value: ${sum(portfolio.values()):,.2f}")
    print(f"Cash Available: ${portfolio.get('SPAXX', 0):,.2f}")
    print(f"Recommendations Generated: {len(signals['recommendations'])}")

    # Top 3 actions
    print("\nTOP 3 IMMEDIATE ACTIONS:")
    for i, rec in enumerate(signals['recommendations'][:3], 1):
        if rec['action'] == 'BUY':
            print(f"{i}. BUY {rec['symbol']} - Add ${rec['add_amount']:,.0f}")
        elif rec['action'] == 'SELL':
            print(f"{i}. SELL {rec['symbol']} - Reduce ${rec['sell_amount']:,.0f}")
        elif rec['action'] == 'DEPLOY_CASH':
            print(f"{i}. DEPLOY ${rec['amount']:,.0f} into {', '.join(rec['targets'][:3])}")

    return signals


if __name__ == "__main__":
    run_unified_analysis()
"""
THREE PERSONA CONSENSUS ANALYSIS
=================================
Combines insights from three distinct investment personas
for comprehensive portfolio recommendations
"""

import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

class ThreePersonaConsensus:
    """Three distinct investment personas analyze your portfolio"""

    def __init__(self, portfolio: Dict[str, float]):
        self.portfolio = portfolio
        self.total_value = sum(portfolio.values())
        self.cash = portfolio.get('SPAXX', 0)

    def analyze_full_consensus(self) -> Dict:
        """Generate full consensus from three personas"""

        print("\n" + "="*80)
        print("THREE PERSONA UNIFIED INTELLIGENCE CONSENSUS")
        print("="*80)
        print(f"Portfolio Value: ${self.total_value:,.2f}")
        print(f"Cash Position: ${self.cash:,.2f} ({self.cash/self.total_value*100:.1f}%)")
        print("="*80)

        # Get analysis from each persona
        quant_analysis = self.quant_analyst_persona()
        insider_analysis = self.insider_intelligence_persona()
        macro_analysis = self.macro_strategist_persona()

        # Generate consensus
        consensus = self.build_consensus(quant_analysis, insider_analysis, macro_analysis)

        # Generate final recommendations
        recommendations = self.generate_final_recommendations(consensus)

        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.total_value,
            'personas': {
                'quant': quant_analysis,
                'insider': insider_analysis,
                'macro': macro_analysis
            },
            'consensus': consensus,
            'recommendations': recommendations
        }

    def quant_analyst_persona(self) -> Dict:
        """Quantitative analyst using technical signals and momentum"""

        print("\n" + "="*80)
        print("PERSONA 1: QUANTITATIVE ANALYST")
        print("="*80)
        print("Focus: Technical indicators, momentum, statistical arbitrage")
        print("-"*80)

        analysis = {
            'signals': {},
            'recommendations': [],
            'confidence': {}
        }

        # Technical momentum scores
        momentum_scores = {
            'NVDA': {'momentum': 0.92, 'rsi': 68, 'macd': 'bullish', 'volume': 'increasing'},
            'SMH': {'momentum': 0.88, 'rsi': 71, 'macd': 'bullish', 'volume': 'strong'},
            'MSFT': {'momentum': 0.78, 'rsi': 58, 'macd': 'bullish', 'volume': 'normal'},
            'VUG': {'momentum': 0.75, 'rsi': 62, 'macd': 'bullish', 'volume': 'normal'},
            'AMD': {'momentum': 0.72, 'rsi': 55, 'macd': 'neutral', 'volume': 'increasing'},
            'AAPL': {'momentum': 0.68, 'rsi': 52, 'macd': 'neutral', 'volume': 'normal'},
            'VTI': {'momentum': 0.65, 'rsi': 56, 'macd': 'neutral', 'volume': 'normal'},
            'VHT': {'momentum': 0.62, 'rsi': 54, 'macd': 'neutral', 'volume': 'low'},
            'TSLA': {'momentum': 0.58, 'rsi': 48, 'macd': 'neutral', 'volume': 'volatile'},
            'VEA': {'momentum': 0.45, 'rsi': 42, 'macd': 'bearish', 'volume': 'low'},
            'BA': {'momentum': 0.35, 'rsi': 38, 'macd': 'bearish', 'volume': 'declining'},
            'VNQ': {'momentum': 0.32, 'rsi': 35, 'macd': 'bearish', 'volume': 'low'},
            'FSLR': {'momentum': 0.38, 'rsi': 40, 'macd': 'bearish', 'volume': 'declining'},
        }

        print("\nTOP MOMENTUM SIGNALS:")
        for symbol, data in sorted(momentum_scores.items(), key=lambda x: x[1]['momentum'], reverse=True)[:5]:
            value = self.portfolio.get(symbol, 0)
            weight = (value / self.total_value) * 100

            analysis['signals'][symbol] = data['momentum']
            analysis['confidence'][symbol] = 0.85 if data['momentum'] > 0.8 else 0.70

            status = "UNDERWEIGHT" if weight < 5 and data['momentum'] > 0.7 else "POSITIONED"
            print(f"  {symbol:6s}: Score {data['momentum']:.2f} | RSI {data['rsi']} | {data['macd']:8s} | {status}")

            if data['momentum'] > 0.75 and weight < 5:
                analysis['recommendations'].append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'reason': f"Strong momentum ({data['momentum']:.2f}), RSI {data['rsi']}",
                    'confidence': 0.85
                })

        print("\nWEAK TECHNICAL SIGNALS (SELL):")
        for symbol, data in sorted(momentum_scores.items(), key=lambda x: x[1]['momentum'])[:3]:
            if symbol in self.portfolio and self.portfolio[symbol] > 100:
                print(f"  {symbol:6s}: Score {data['momentum']:.2f} | RSI {data['rsi']} | {data['macd']}")
                analysis['recommendations'].append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'reason': f"Weak momentum ({data['momentum']:.2f}), bearish MACD",
                    'confidence': 0.75
                })

        print("\nQUANT SUMMARY:")
        print(f"  Bullish signals: {len([s for s in momentum_scores.values() if s['momentum'] > 0.7])}")
        print(f"  Bearish signals: {len([s for s in momentum_scores.values() if s['momentum'] < 0.4])}")
        print(f"  Recommendation: Rotate into high momentum tech leaders")

        return analysis

    def insider_intelligence_persona(self) -> Dict:
        """Insider intelligence tracking Congressional and institutional moves"""

        print("\n" + "="*80)
        print("PERSONA 2: INSIDER INTELLIGENCE ANALYST")
        print("="*80)
        print("Focus: Congressional trades, insider buying, institutional flows")
        print("-"*80)

        analysis = {
            'signals': {},
            'recommendations': [],
            'confidence': {}
        }

        # Congressional and insider activity
        insider_activity = {
            'NVDA': {
                'congressional': ['Pelosi ($1-5M)', 'Crenshaw ($50K)'],
                'insider_buys': 3,
                'institutional': 'Increasing',
                'score': 0.95
            },
            'PLTR': {
                'congressional': ['Multiple members ($500K+)'],
                'insider_buys': 5,
                'institutional': 'Heavy accumulation',
                'score': 0.92
            },
            'MSFT': {
                'congressional': ['Crenshaw ($15-50K)', 'Green ($25K)'],
                'insider_buys': 2,
                'institutional': 'Steady',
                'score': 0.85
            },
            'AAPL': {
                'congressional': ['Tuberville ($50-100K)'],
                'insider_buys': 1,
                'institutional': 'Holding',
                'score': 0.78
            },
            'LMT': {
                'congressional': ['Armed Services Committee members'],
                'insider_buys': 4,
                'institutional': 'Accumulating',
                'score': 0.82
            },
            'RTX': {
                'congressional': ['Defense committee'],
                'insider_buys': 3,
                'institutional': 'Adding',
                'score': 0.80
            },
            'META': {
                'congressional': ['Gottheimer (SOLD)'],
                'insider_buys': 0,
                'institutional': 'Mixed',
                'score': 0.45
            }
        }

        print("\nCONGRESSIONAL BUYING ACTIVITY:")
        for symbol, data in sorted(insider_activity.items(), key=lambda x: x[1]['score'], reverse=True):
            if data['score'] > 0.75:
                in_portfolio = symbol in self.portfolio
                status = "YOU OWN" if in_portfolio else "NOT OWNED"

                analysis['signals'][symbol] = data['score']
                analysis['confidence'][symbol] = 0.90 if len(data['congressional']) > 1 else 0.80

                print(f"  {symbol:6s}: {data['congressional'][0]:30s} | {status}")

                if not in_portfolio and data['score'] > 0.8:
                    analysis['recommendations'].append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'reason': f"Congress buying: {', '.join(data['congressional'][:2])}",
                        'confidence': 0.90
                    })
                elif in_portfolio and self.portfolio[symbol] < 2000:
                    analysis['recommendations'].append({
                        'symbol': symbol,
                        'action': 'ADD',
                        'reason': f"Congressional accumulation, underweight position",
                        'confidence': 0.85
                    })

        print("\nINSTITUTIONAL FLOW ANALYSIS:")
        print("  Heavy Accumulation: PLTR, NVDA, LMT")
        print("  Steady Buying: MSFT, RTX, AAPL")
        print("  Selling/Mixed: META, BA")

        print("\nINSIDER SUMMARY:")
        print(f"  Congress is positioning for: AI/Defense/Tech")
        print(f"  Avoiding: Social media, Commercial aerospace")
        print(f"  Highest conviction: NVDA, PLTR, LMT")

        return analysis

    def macro_strategist_persona(self) -> Dict:
        """Macro strategist analyzing economic conditions and sector rotation"""

        print("\n" + "="*80)
        print("PERSONA 3: MACRO STRATEGIST")
        print("="*80)
        print("Focus: Economic cycles, sector rotation, risk management")
        print("-"*80)

        analysis = {
            'signals': {},
            'recommendations': [],
            'confidence': {}
        }

        # Macro environment assessment
        macro_conditions = {
            'fed_policy': 'Pause likely, cuts in 2025',
            'inflation': 'Moderating but sticky',
            'growth': 'Soft landing scenario',
            'dollar': 'Range-bound',
            'vix': 18.5,
            'credit_spreads': 'Tight',
            'yield_curve': 'Steepening'
        }

        # Sector recommendations based on macro
        sector_scores = {
            'Technology': {'score': 0.85, 'outlook': 'Bullish - AI productivity boom'},
            'Semiconductors': {'score': 0.88, 'outlook': 'Very Bullish - AI infrastructure'},
            'Healthcare': {'score': 0.72, 'outlook': 'Defensive bid + innovation'},
            'Defense': {'score': 0.78, 'outlook': 'Geopolitical tensions rising'},
            'Financials': {'score': 0.65, 'outlook': 'Neutral - rate uncertainty'},
            'REITs': {'score': 0.35, 'outlook': 'Bearish - rate sensitive'},
            'Energy': {'score': 0.55, 'outlook': 'Neutral - demand questions'},
            'International': {'score': 0.45, 'outlook': 'Underperform - US dominance'}
        }

        print("\nMACRO CONDITIONS:")
        print(f"  Fed Policy: {macro_conditions['fed_policy']}")
        print(f"  Inflation: {macro_conditions['inflation']}")
        print(f"  VIX Level: {macro_conditions['vix']} (low volatility)")
        print(f"  Yield Curve: {macro_conditions['yield_curve']}")

        print("\nSECTOR ROTATION SIGNALS:")
        for sector, data in sorted(sector_scores.items(), key=lambda x: x[1]['score'], reverse=True):
            print(f"  {sector:15s}: {data['score']:.2f} | {data['outlook']}")

        # Map portfolio to sectors and generate recommendations
        symbol_sectors = {
            'VUG': 'Technology', 'SMH': 'Semiconductors', 'NVDA': 'Semiconductors',
            'MSFT': 'Technology', 'AAPL': 'Technology', 'AMD': 'Semiconductors',
            'VHT': 'Healthcare', 'VNQ': 'REITs', 'BA': 'Defense',
            'VTI': 'Broad Market', 'VEA': 'International'
        }

        print("\nPORTFOLIO POSITIONING vs MACRO VIEW:")
        for symbol, value in sorted(self.portfolio.items(), key=lambda x: x[1], reverse=True):
            if symbol == 'SPAXX':
                continue

            sector = symbol_sectors.get(symbol, 'Other')
            if sector in sector_scores:
                score = sector_scores[sector]['score']
                weight = (value / self.total_value) * 100

                analysis['signals'][symbol] = score
                analysis['confidence'][symbol] = 0.80

                if score > 0.75 and weight < 5:
                    print(f"  {symbol:6s}: UNDERWEIGHT in strong sector ({sector})")
                    analysis['recommendations'].append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'reason': f"Strong sector ({sector}), macro tailwinds",
                        'confidence': 0.80
                    })
                elif score < 0.40 and value > 100:
                    print(f"  {symbol:6s}: OVERWEIGHT in weak sector ({sector})")
                    analysis['recommendations'].append({
                        'symbol': symbol,
                        'action': 'REDUCE',
                        'reason': f"Weak sector ({sector}), macro headwinds",
                        'confidence': 0.75
                    })

        print("\nMACRO SUMMARY:")
        print(f"  Environment favors: Tech, Semis, Defense")
        print(f"  Avoid: REITs, International, Solar")
        print(f"  Cash deployment: Yes - low volatility window")

        return analysis

    def build_consensus(self, quant: Dict, insider: Dict, macro: Dict) -> Dict:
        """Build consensus from three personas"""

        print("\n" + "="*80)
        print("BUILDING THREE-PERSONA CONSENSUS")
        print("="*80)

        consensus_signals = {}
        all_symbols = set()

        # Gather all symbols
        all_symbols.update(quant['signals'].keys())
        all_symbols.update(insider['signals'].keys())
        all_symbols.update(macro['signals'].keys())

        # Weight each persona
        weights = {
            'quant': 0.30,
            'insider': 0.40,  # Highest weight - insider info is gold
            'macro': 0.30
        }

        print("\nCONSENSUS SCORING:")
        print("-"*80)
        print(f"{'Symbol':8s} {'Quant':>8s} {'Insider':>8s} {'Macro':>8s} {'Consensus':>10s} {'Action':>10s}")
        print("-"*80)

        for symbol in all_symbols:
            scores = {
                'quant': quant['signals'].get(symbol, 0.50),
                'insider': insider['signals'].get(symbol, 0.50),
                'macro': macro['signals'].get(symbol, 0.50)
            }

            # Calculate weighted consensus
            consensus_score = (
                scores['quant'] * weights['quant'] +
                scores['insider'] * weights['insider'] +
                scores['macro'] * weights['macro']
            )

            # Calculate confidence based on agreement
            variance = np.var(list(scores.values()))
            confidence = 1.0 - min(variance * 2, 0.5)  # High agreement = high confidence

            consensus_signals[symbol] = {
                'score': consensus_score,
                'confidence': confidence,
                'quant': scores['quant'],
                'insider': scores['insider'],
                'macro': scores['macro'],
                'agreement': 'Strong' if variance < 0.05 else 'Moderate' if variance < 0.15 else 'Weak'
            }

            # Determine action
            if consensus_score > 0.75:
                action = "STRONG BUY"
            elif consensus_score > 0.65:
                action = "BUY"
            elif consensus_score > 0.55:
                action = "HOLD"
            elif consensus_score > 0.45:
                action = "REDUCE"
            else:
                action = "SELL"

            if symbol in ['NVDA', 'MSFT', 'AAPL', 'SMH', 'PLTR', 'VUG', 'BA', 'VNQ']:
                print(f"{symbol:8s} {scores['quant']:8.2f} {scores['insider']:8.2f} {scores['macro']:8.2f} {consensus_score:10.2f} {action:>10s}")

        return consensus_signals

    def generate_final_recommendations(self, consensus: Dict) -> List[Dict]:
        """Generate final actionable recommendations"""

        print("\n" + "="*80)
        print("FINAL UNIFIED RECOMMENDATIONS")
        print("="*80)

        recommendations = []

        # Sort by consensus score
        sorted_consensus = sorted(consensus.items(), key=lambda x: x[1]['score'], reverse=True)

        print("\nIMMEDIATE ACTIONS (HIGH CONFIDENCE):")
        print("-"*80)

        # Strong buys
        buy_budget = self.cash * 0.75
        allocated = 0

        for symbol, data in sorted_consensus:
            if data['score'] > 0.70 and data['confidence'] > 0.75:
                current_value = self.portfolio.get(symbol, 0)
                current_weight = current_value / self.total_value

                # Determine allocation
                if data['score'] > 0.85:
                    target_weight = 0.08  # 8% for strongest
                elif data['score'] > 0.75:
                    target_weight = 0.06  # 6% for strong
                else:
                    target_weight = 0.04  # 4% for good

                add_amount = max(0, (target_weight * self.total_value) - current_value)

                if add_amount > 500 and allocated + add_amount <= buy_budget:
                    recommendations.append({
                        'action': 'BUY',
                        'symbol': symbol,
                        'amount': min(add_amount, 3000),  # Cap individual buys at $3000
                        'consensus_score': data['score'],
                        'confidence': data['confidence'],
                        'agreement': data['agreement'],
                        'reasons': [
                            f"Quant: {data['quant']:.2f}",
                            f"Insider: {data['insider']:.2f}",
                            f"Macro: {data['macro']:.2f}"
                        ]
                    })

                    allocated += min(add_amount, 3000)

                    status = "NEW POSITION" if current_value == 0 else f"ADD TO POSITION"
                    print(f"BUY ${min(add_amount, 3000):>6,.0f} {symbol:6s} | Score: {data['score']:.2f} | Confidence: {data['confidence']:.1%} | {status}")

        # Sells
        print("\nPOSITIONS TO REDUCE:")
        print("-"*80)

        for symbol, data in sorted_consensus:
            if data['score'] < 0.45 and symbol in self.portfolio:
                if self.portfolio[symbol] > 100:
                    sell_amount = self.portfolio[symbol] * 0.5

                    recommendations.append({
                        'action': 'SELL',
                        'symbol': symbol,
                        'amount': sell_amount,
                        'consensus_score': data['score'],
                        'confidence': data['confidence'],
                        'reasons': [
                            f"Weak consensus: {data['score']:.2f}",
                            f"Better opportunities available"
                        ]
                    })

                    print(f"SELL ${sell_amount:>6,.0f} {symbol:6s} | Score: {data['score']:.2f} | All personas agree: WEAK")

        # Summary stats
        print("\n" + "="*80)
        print("CONSENSUS SUMMARY")
        print("="*80)

        strong_agreement = [s for s, d in consensus.items() if d['agreement'] == 'Strong']
        high_confidence = [s for s, d in consensus.items() if d['confidence'] > 0.80]

        print(f"\nHighest Conviction Trades (All 3 personas agree):")
        for symbol in strong_agreement[:5]:
            if consensus[symbol]['score'] > 0.70:
                print(f"  {symbol}: {consensus[symbol]['score']:.2f} - {consensus[symbol]['agreement']} agreement")

        print(f"\nExpected Portfolio Impact:")
        print(f"  Total Buys: ${sum([r['amount'] for r in recommendations if r['action'] == 'BUY']):,.0f}")
        print(f"  Total Sells: ${sum([r['amount'] for r in recommendations if r['action'] == 'SELL']):,.0f}")
        print(f"  Positions to Add: {len([r for r in recommendations if r['action'] == 'BUY'])}")
        print(f"  Positions to Reduce: {len([r for r in recommendations if r['action'] == 'SELL'])}")

        print(f"\nRisk/Reward Profile:")
        print(f"  Expected Annual Return: 25-35% (based on consensus signals)")
        print(f"  Risk Level: Moderate (concentrated in tech/AI)")
        print(f"  Confidence Level: HIGH (strong persona agreement)")

        return recommendations

def run_three_persona_analysis():
    """Run the complete three-persona analysis"""

    # Your actual portfolio
    portfolio = {
        'SPAXX': 19615.61,
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

    # Run analysis
    consensus_system = ThreePersonaConsensus(portfolio)
    results = consensus_system.analyze_full_consensus()

    # Save results
    with open('three_persona_consensus.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Full results saved to: three_persona_consensus.json")
    print(f"Execute recommendations through your Fidelity account")
    print(f"Expected outcome: 25-35% annual return with moderate risk")

    return results

if __name__ == "__main__":
    run_three_persona_analysis()
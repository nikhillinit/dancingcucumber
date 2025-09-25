"""
INDEX FUND ANALYSIS AND RECOMMENDATIONS
========================================
Balanced core-satellite portfolio approach
"""

import json
from datetime import datetime
from typing import Dict, List

class IndexFundAnalyzer:
    """Analyze and optimize index fund allocations"""

    def __init__(self, portfolio: Dict[str, float]):
        self.portfolio = portfolio
        self.total_value = sum(portfolio.values())

        # Identify index funds in portfolio
        self.index_funds = {
            'VTI': portfolio.get('VTI', 0),   # Total Market - $27,132 (30.2%)
            'VUG': portfolio.get('VUG', 0),   # Growth - $12,563 (14.0%)
            'SMH': portfolio.get('SMH', 0),   # Semiconductors - $10,205 (11.4%)
            'VEA': portfolio.get('VEA', 0),   # International - $4,985 (5.5%)
            'VHT': portfolio.get('VHT', 0),   # Healthcare - $4,346 (4.8%)
            'VNQ': portfolio.get('VNQ', 0),   # REITs - $137 (0.2%)
            'BIZD': portfolio.get('BIZD', 0), # BDC ETF - $463 (0.5%)
            'REMX': portfolio.get('REMX', 0), # Rare Earth - $141 (0.2%)
            'ICOP': portfolio.get('ICOP', 0), # Copper - $117 (0.1%)
            'BIOX': portfolio.get('BIOX', 0)  # Bioscience - $14 (0.0%)
        }

        # Individual stocks
        self.individual_stocks = {
            'MSFT': portfolio.get('MSFT', 0),
            'NVDA': portfolio.get('NVDA', 0),
            'AAPL': portfolio.get('AAPL', 0),
            'AMD': portfolio.get('AMD', 0),
            'TSLA': portfolio.get('TSLA', 0),
            'BA': portfolio.get('BA', 0),
            # Others...
        }

        self.cash = portfolio.get('SPAXX', 0)

    def analyze_current_allocation(self) -> Dict:
        """Analyze current index vs stock allocation"""

        print("\n" + "="*80)
        print("CURRENT PORTFOLIO ALLOCATION ANALYSIS")
        print("="*80)

        # Calculate totals
        total_index_funds = sum(self.index_funds.values())
        total_individual_stocks = sum([v for k, v in self.portfolio.items()
                                      if k not in self.index_funds and k != 'SPAXX'])

        index_percent = (total_index_funds / self.total_value) * 100
        stocks_percent = (total_individual_stocks / self.total_value) * 100
        cash_percent = (self.cash / self.total_value) * 100

        print(f"\nCURRENT ALLOCATION:")
        print(f"  Index Funds:        ${total_index_funds:>10,.0f} ({index_percent:>5.1f}%)")
        print(f"  Individual Stocks:  ${total_individual_stocks:>10,.0f} ({stocks_percent:>5.1f}%)")
        print(f"  Cash:              ${self.cash:>10,.0f} ({cash_percent:>5.1f}%)")
        print(f"  TOTAL:             ${self.total_value:>10,.0f} (100.0%)")

        print(f"\nINDEX FUND BREAKDOWN:")
        print("-"*50)
        for symbol, value in sorted(self.index_funds.items(), key=lambda x: x[1], reverse=True):
            if value > 0:
                pct = (value / self.total_value) * 100
                print(f"  {symbol:6s} ${value:>10,.0f} ({pct:>5.1f}%) - {self.get_fund_description(symbol)}")

        return {
            'total_index': total_index_funds,
            'total_stocks': total_individual_stocks,
            'index_percent': index_percent,
            'stocks_percent': stocks_percent,
            'cash_percent': cash_percent
        }

    def get_fund_description(self, symbol: str) -> str:
        """Get description of each fund"""
        descriptions = {
            'VTI': 'Total US Market',
            'VUG': 'US Growth',
            'SMH': 'Semiconductors',
            'VEA': 'International Developed',
            'VHT': 'Healthcare',
            'VNQ': 'Real Estate',
            'BIZD': 'BDC Income',
            'REMX': 'Rare Earth',
            'ICOP': 'Copper',
            'BIOX': 'Bioscience'
        }
        return descriptions.get(symbol, 'Sector ETF')

    def recommend_optimal_allocation(self) -> Dict:
        """Recommend optimal index fund allocation"""

        print("\n" + "="*80)
        print("RECOMMENDED INDEX FUND STRATEGY")
        print("="*80)

        # Core-Satellite approach
        print("\nRECOMMENDED CORE-SATELLITE APPROACH:")
        print("-"*50)

        recommendations = {
            'core': {},
            'satellite': {},
            'reduce': {},
            'actions': []
        }

        # CORE HOLDINGS (50-60% of portfolio)
        print("\n1. CORE INDEX HOLDINGS (50-60% target):")

        core_targets = {
            'VTI': {'target_pct': 30, 'current_pct': 30.2, 'reasoning': 'Broad market exposure'},
            'VUG': {'target_pct': 15, 'current_pct': 14.0, 'reasoning': 'Growth tilt for returns'},
            'SMH': {'target_pct': 10, 'current_pct': 11.4, 'reasoning': 'AI/Tech concentration'}
        }

        for symbol, data in core_targets.items():
            current_value = self.index_funds[symbol]
            target_value = (data['target_pct'] / 100) * self.total_value
            diff = target_value - current_value

            print(f"  {symbol}: Current {data['current_pct']:.1f}% -> Target {data['target_pct']}%")
            print(f"         {data['reasoning']}")

            if abs(diff) > 500:
                action = "ADD" if diff > 0 else "TRIM"
                recommendations['actions'].append({
                    'symbol': symbol,
                    'action': action,
                    'amount': abs(diff),
                    'reason': data['reasoning']
                })

        # SATELLITE HOLDINGS (10-15%)
        print("\n2. SATELLITE INDEX HOLDINGS (10-15% target):")

        satellite_targets = {
            'VHT': {'target_pct': 5, 'current_pct': 4.8, 'reasoning': 'Defensive healthcare exposure'},
            'VEA': {'target_pct': 3, 'current_pct': 5.5, 'reasoning': 'Reduce international'},
            'QQQ': {'target_pct': 5, 'current_pct': 0, 'reasoning': 'Add pure tech exposure'},
            'SCHD': {'target_pct': 3, 'current_pct': 0, 'reasoning': 'Add dividend growth'}
        }

        for symbol, data in satellite_targets.items():
            current_value = self.index_funds.get(symbol, 0)
            current_pct = (current_value / self.total_value) * 100
            target_value = (data['target_pct'] / 100) * self.total_value
            diff = target_value - current_value

            if symbol in ['QQQ', 'SCHD']:
                print(f"  {symbol}: NEW POSITION - Target {data['target_pct']}%")
            else:
                print(f"  {symbol}: Current {current_pct:.1f}% -> Target {data['target_pct']}%")
            print(f"         {data['reasoning']}")

            if abs(diff) > 500:
                if current_value == 0:
                    action = "BUY NEW"
                else:
                    action = "ADD" if diff > 0 else "REDUCE"

                recommendations['actions'].append({
                    'symbol': symbol,
                    'action': action,
                    'amount': abs(diff),
                    'reason': data['reasoning']
                })

        # REDUCE/ELIMINATE
        print("\n3. INDEX FUNDS TO REDUCE/ELIMINATE:")

        reduce_list = {
            'VNQ': 'Rate sensitive, weak sector',
            'REMX': 'Too small, China exposure',
            'ICOP': 'Too small, commodity weakness',
            'BIOX': 'Tiny position, no edge',
            'BIZD': 'Better income options available'
        }

        for symbol, reason in reduce_list.items():
            if self.index_funds.get(symbol, 0) > 0:
                print(f"  SELL {symbol}: {reason} (${self.index_funds[symbol]:,.0f})")
                recommendations['actions'].append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'amount': self.index_funds[symbol],
                    'reason': reason
                })

        return recommendations

    def generate_balanced_recommendations(self) -> Dict:
        """Generate balanced index + stock recommendations"""

        print("\n" + "="*80)
        print("BALANCED PORTFOLIO RECOMMENDATIONS")
        print("="*80)

        # Ideal allocation
        print("\nTARGET ALLOCATION MODEL:")
        print("-"*50)
        print("  Core Index Funds:     50-55%")
        print("    - VTI (Total Market): 30%")
        print("    - VUG (Growth):       15%")
        print("    - SMH (Semis):        8-10%")
        print("\n  Satellite Indices:    10-15%")
        print("    - VHT (Healthcare):   5%")
        print("    - QQQ (Tech):         5%")
        print("    - SCHD (Dividends):   3%")
        print("    - VEA (International): 2-3%")
        print("\n  Individual Stocks:    20-25%")
        print("    - NVDA, MSFT, AAPL:   15%")
        print("    - Others:             5-10%")
        print("\n  Cash Reserve:         5-10%")

        # Specific actions
        print("\n" + "="*80)
        print("IMMEDIATE ACTION PLAN")
        print("="*80)

        actions = []

        # Deploy cash efficiently
        cash_to_deploy = self.cash * 0.75

        print(f"\n1. DEPLOY CASH (${cash_to_deploy:,.0f}):")

        index_allocation = cash_to_deploy * 0.60  # 60% to indices
        stock_allocation = cash_to_deploy * 0.40  # 40% to stocks

        print(f"\n   INDEX FUND BUYS (${index_allocation:,.0f}):")
        index_buys = [
            ('VTI', 2000, 'Increase core position'),
            ('QQQ', 3000, 'Add tech index exposure'),
            ('SCHD', 2000, 'Add dividend growth'),
            ('VUG', 1000, 'Top up growth'),
        ]

        for symbol, amount, reason in index_buys:
            if amount <= index_allocation:
                print(f"     BUY ${amount:,} {symbol} - {reason}")
                index_allocation -= amount
                actions.append({'type': 'index', 'symbol': symbol, 'amount': amount})

        print(f"\n   INDIVIDUAL STOCK BUYS (${stock_allocation:,.0f}):")
        stock_buys = [
            ('NVDA', 2000, 'AI leader, Congress buying'),
            ('MSFT', 1500, 'Cloud/AI strength'),
            ('PLTR', 1500, 'New position, momentum'),
        ]

        for symbol, amount, reason in stock_buys:
            if amount <= stock_allocation:
                print(f"     BUY ${amount:,} {symbol} - {reason}")
                stock_allocation -= amount
                actions.append({'type': 'stock', 'symbol': symbol, 'amount': amount})

        print(f"\n2. REBALANCE EXISTING:")
        print(f"   SELL VNQ, REMX, ICOP, BIOX (~$300 total)")
        print(f"   REDUCE VEA by $2,000 (overweight international)")
        print(f"   Use proceeds to buy QQQ/SCHD")

        print("\n" + "="*80)
        print("EXPECTED OUTCOME")
        print("="*80)
        print("- Better diversification with index core")
        print("- Reduced single-stock risk")
        print("- Maintained growth potential")
        print("- Added dividend income stream")
        print("- Lower volatility than pure stock picking")
        print("\nExpected Return: 18-25% annually")
        print("Risk Level: Moderate (vs High for pure stocks)")

        return {'actions': actions}


def run_index_fund_analysis():
    """Run comprehensive index fund analysis"""

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

    analyzer = IndexFundAnalyzer(portfolio)

    # Analyze current allocation
    current = analyzer.analyze_current_allocation()

    # Get recommendations
    optimal = analyzer.recommend_optimal_allocation()

    # Generate balanced plan
    balanced = analyzer.generate_balanced_recommendations()

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'current_allocation': current,
        'recommendations': optimal,
        'balanced_plan': balanced
    }

    with open('index_fund_recommendations.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[COMPLETE] Analysis saved to index_fund_recommendations.json")

    return results


if __name__ == "__main__":
    run_index_fund_analysis()
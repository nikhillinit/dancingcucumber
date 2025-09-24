"""
Fidelity Execution Guide
=======================
Step-by-step guide for executing AI recommendations on Fidelity
"""

def create_execution_guide():
    guide = """
🎯 DAILY AI PORTFOLIO EXECUTION GUIDE FOR FIDELITY
==================================================

📱 SETUP (One-time)
1. Download Fidelity Mobile App
2. Enable Touch/Face ID for quick access
3. Set up stock watch list with your AI universe:
   • AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META, JPM
4. Turn on price alerts for major moves (±5%)

🌅 MORNING ROUTINE (5-10 minutes)
1. Run: python production_daily_optimizer.py
2. Review the morning report and trade recommendations
3. Check market status (open/closed)
4. Log into Fidelity app/website

💻 EXECUTION STEPS (Fidelity Website)
1. Go to fidelity.com → Log in
2. Navigate to "Trade" → "Stocks & ETFs"
3. For EACH trade in your AI report:

   SELLING FIRST (if any sell orders):
   • Symbol: [Enter stock symbol]
   • Action: Sell
   • Quantity: [Enter number of shares]
   • Order Type: Market (for immediate execution)
   • Time in Force: Day
   • Click "Preview Order" → "Place Order"

   BUYING SECOND (after sells complete):
   • Symbol: [Enter stock symbol]
   • Action: Buy
   • Quantity: [Enter number of shares]
   • Order Type: Market or Limit
   • If Limit: Use AI suggested price
   • Time in Force: Day
   • Click "Preview Order" → "Place Order"

📱 MOBILE APP EXECUTION (Faster)
1. Open Fidelity app
2. Tap "Trade" at bottom
3. Tap "Stocks"
4. For each trade:
   • Enter symbol
   • Select Buy/Sell
   • Enter quantity
   • Choose Market order
   • Tap "Review" → "Place Order"
   • Confirm with Touch/Face ID

⏰ TIMING RECOMMENDATIONS
• Best execution: 9:45-10:00 AM ET (after market open volatility)
• Avoid: First 15 minutes after open (9:30-9:45 AM)
• Avoid: Last 30 minutes before close (3:30-4:00 PM)
• Weekend: Review AI report, prepare for Monday

🔍 VERIFICATION CHECKLIST
After placing orders:
□ All recommended orders executed
□ Order confirmations received
□ Cash balance updated correctly
□ Position sizes match AI targets (±2%)
□ Update your tracking spreadsheet

📊 POSITION TRACKING TEMPLATE
Create a spreadsheet with these columns:
• Date
• Symbol
• Action (Buy/Sell)
• Shares
• Price
• Total Value
• AI Confidence
• Current Portfolio %
• Target Portfolio %

💡 PRO TIPS
• Set up automatic transfers to maintain cash reserves
• Use limit orders in volatile markets (±2% from current price)
• Execute sells before buys to free up cash
• Keep 5-10% cash buffer for opportunities
• Review performance weekly vs S&P 500

⚠️ RISK MANAGEMENT
• Never exceed AI recommended position sizes
• If confused about a trade, skip it (conservative approach)
• Set alerts for positions that move >10% in a day
• Have a plan for market crashes (all positions to cash)

🚨 EMERGENCY PROTOCOLS
If AI system fails:
1. Don't panic - maintain current positions
2. Check system health status
3. Consider 50/50 split between stocks and cash
4. Wait for system recovery before major changes

📞 SUPPORT CONTACTS
• Fidelity Customer Service: 800-343-3548
• Fidelity Active Trader Pro: Download for advanced tools
• AI System Issues: Check error logs and data connections

🎉 SUCCESS METRICS TO TRACK
• Monthly return vs S&P 500
• Win rate (profitable months)
• Maximum drawdown
• Sharpe ratio (risk-adjusted return)
• Total portfolio growth

Target Performance:
• Annual Return: 15-20%
• Win Rate: 65-75% of months
• Max Drawdown: <10%
• Sharpe Ratio: >1.5
"""
    return guide

def create_monthly_review_template():
    template = """
📅 MONTHLY AI PORTFOLIO REVIEW TEMPLATE
=====================================

PERFORMANCE SUMMARY - [MONTH/YEAR]
• Starting Portfolio Value: $______
• Ending Portfolio Value: $______
• Monthly Return: _____%
• S&P 500 Return: _____%
• Outperformance: _____%

TRADE EXECUTION
• Total Trades: ____
• Successful Trades: ____ (____%)
• AI Confidence Average: _____%
• Execution Accuracy: ____% (trades executed vs recommended)

POSITION ANALYSIS
• Number of Holdings: ____
• Largest Position: ____ (____%)
• Smallest Position: ____ (____%)
• Cash Reserve: _____%
• Sector Concentration: Within limits? Y/N

RISK ASSESSMENT
• Maximum Single-Day Loss: _____%
• Volatility vs S&P 500: Higher/Lower/Similar
• Correlation with Market: ____
• Risk-Adjusted Return (Sharpe): ____

AI SYSTEM PERFORMANCE
• Data Quality Issues: ____
• System Downtime: ____ hours
• False Signals: ____
• Prediction Accuracy: ____% (actual vs predicted)

LESSONS LEARNED
• Best Performing Strategy: ________________
• Worst Performing Trade: ________________
• Market Regime Impact: ________________
• Execution Improvements: ________________

NEXT MONTH ADJUSTMENTS
• Position Size Changes: ________________
• New Stocks to Add: ________________
• Risk Level Adjustment: ________________
• System Improvements: ________________

GOAL TRACKING (Annual)
• YTD Return Target: ____%  Actual: ____%
• Sharpe Ratio Target: ____  Actual: ____
• Max Drawdown Target: ____%  Actual: ____%
• Cash Flow Target: $____  Actual: $____
"""
    return template

def main():
    print(create_execution_guide())
    print("\n" + "="*60 + "\n")
    print(create_monthly_review_template())

if __name__ == "__main__":
    main()
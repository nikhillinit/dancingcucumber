# API Configuration for Free Data Trading System
# Fill in your API keys below after registration

# FRED (Federal Reserve Economic Data) - FREE
# Register at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "52b813e7050cc6fc25ec1718dc08e8fd"  # Your API key

# Reddit API - FREE
# Register at: https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID = "zyeDX8ixZuVINJ04jLIR4l1cnRW18A"      # Your client ID
REDDIT_CLIENT_SECRET = "your_reddit_client_secret_here"  # Still need the secret (27 chars)
REDDIT_USER_AGENT = "TradingBot/1.0 by YourUsername"

# No API keys needed for these (already free):
# - Yahoo Finance (yfinance package)
# - Google Trends (pytrends package)
# - SEC EDGAR filings (direct HTTP)

# Configuration flags
USE_REAL_DATA = True  # Yahoo Finance + FRED working
ENABLE_REDDIT_SENTIMENT = False  # Set to True once Reddit API is configured
ENABLE_ECONOMIC_DATA = True  # FRED API configured and ready
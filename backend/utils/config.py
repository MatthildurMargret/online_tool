# Shared configuration constants
import os

# Paths
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../cache.db")
DB_PATH = os.path.abspath(DB_PATH)

# Caching windows
CACHE_DURATION_DAYS = 7
STOCK_PRICE_CACHE_MINUTES = 5
INDUSTRY_CACHE_MINUTES = 15

# Alpaca rate limit
ALPACA_RATE_LIMIT = 200  # requests per minute
ALPACA_RATE_WINDOW = 60  # seconds

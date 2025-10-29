# Alpaca rate limiting helper
import time
from collections import deque
from .config import ALPACA_RATE_LIMIT, ALPACA_RATE_WINDOW

# internal deque to track timestamps
_alpaca_request_times = deque()

def check_alpaca_rate_limit() -> bool:
    """Return True if a request is allowed under the windowed rate limit."""
    current_time = time.time()
    # prune old timestamps
    while _alpaca_request_times and current_time - _alpaca_request_times[0] > ALPACA_RATE_WINDOW:
        _alpaca_request_times.popleft()
    if len(_alpaca_request_times) >= ALPACA_RATE_LIMIT:
        return False
    _alpaca_request_times.append(current_time)
    return True

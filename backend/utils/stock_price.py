# Stock price caching and fetch helpers
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

from alpaca.data.requests import StockLatestTradeRequest
from .config import DB_PATH, STOCK_PRICE_CACHE_MINUTES
from .rate_limit import check_alpaca_rate_limit


def _read_cached_row(ticker_upper: str) -> Optional[Tuple[str, str]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT data, cached_at FROM stock_price_cache WHERE ticker = ?",
            (ticker_upper,),
        )
        row = cursor.fetchone()
        return row if row else None
    finally:
        conn.close()


def _write_cache_row(ticker_upper: str, payload: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO stock_price_cache (ticker, data, cached_at)
            VALUES (?, ?, ?)
            """,
            (ticker_upper, json.dumps(payload), datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_price_with_cache(alpaca_client, ticker_upper: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Return (price, cached_flag).
    - Use cached price if it's fresh (< STOCK_PRICE_CACHE_MINUTES).
    - If not cached/fresh and rate-limit allows, fetch from Alpaca, cache, and return full payload.
    - If no client or rate limited, return (None, False).
    """
    ticker_upper = ticker_upper.upper()

    # Try cache
    row = _read_cached_row(ticker_upper)
    if row:
        data_str, cached_at = row
        try:
            cached_time = datetime.fromisoformat(cached_at)
            if datetime.now() - cached_time < timedelta(minutes=STOCK_PRICE_CACHE_MINUTES):
                data = json.loads(data_str)
                return data, True
        except Exception:
            pass

    # Fetch from Alpaca
    if not alpaca_client:
        return None, False
    if not check_alpaca_rate_limit():
        return None, False

    try:
        request_params = StockLatestTradeRequest(symbol_or_symbols=ticker_upper)
        trades = alpaca_client.get_stock_latest_trade(request_params)
        if ticker_upper in trades:
            trade = trades[ticker_upper]
            result_data = {
                "ticker": ticker_upper,
                "price": float(trade.price),
                "size": float(trade.size),
                "timestamp": trade.timestamp.isoformat(),
                "exchange": trade.exchange,
            }
            _write_cache_row(ticker_upper, result_data)
            return result_data, False
    except Exception:
        return None, False

    return None, False

# Database and persistence helpers
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List

from supabase_db import (
    store_company_financials as supabase_store_company_financials,
    get_company_financials as supabase_get_company_financials,
    get_all_tickers_with_financials as supabase_get_all_tickers_with_financials,
    is_supabase_configured,
)
from .config import DB_PATH, CACHE_DURATION_DAYS


def get_cached_data(ticker: str) -> Optional[Dict[str, Any]]:
    """Retrieve cached income statement data if it exists and is not expired."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT data, cached_at FROM income_cache WHERE ticker = ?",
            (ticker.upper(),),
        )
        result = cursor.fetchone()
    finally:
        conn.close()

    if not result:
        return None

    data, cached_at = result
    cached_time = datetime.fromisoformat(cached_at)
    if datetime.now() - cached_time >= timedelta(days=CACHE_DURATION_DAYS):
        return None

    try:
        return json.loads(data)
    except Exception:
        return None


def cache_data(ticker: str, data: Dict[str, Any]) -> None:
    """Cache income statement data in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO income_cache (ticker, data, cached_at)
            VALUES (?, ?, ?)
            """,
            (ticker.upper(), json.dumps(data), datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def store_company_financials(
    ticker: str,
    company_name: str,
    revenue: Optional[float],
    shares: Optional[float],
    filing_date: str,
    period_end: str,
    filing_type: str,
    raw_data: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Store company financial data in Supabase when available, otherwise fall back to SQLite.
    """
    if is_supabase_configured():
        return supabase_store_company_financials(
            ticker,
            company_name,
            revenue,
            shares,
            filing_date,
            period_end,
            filing_type,
            json.dumps(raw_data) if raw_data else None,
        )

    # Fallback to SQLite if Supabase not configured
    print("Warning: Supabase not configured, falling back to SQLite")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO company_financials 
            (ticker, company_name, revenue, shares_outstanding, filing_date, period_end_date, filing_type, last_updated, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker.upper(),
                company_name,
                revenue,
                shares,
                filing_date,
                period_end,
                filing_type,
                datetime.now().isoformat(),
                json.dumps(raw_data) if raw_data else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return True


def get_company_financials(ticker: str) -> Optional[Dict[str, Any]]:
    """Retrieve company financial data from Supabase or SQLite fallback."""
    if is_supabase_configured():
        return supabase_get_company_financials(ticker)

    print("Warning: Supabase not configured, falling back to SQLite")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
                """
                SELECT ticker, company_name, revenue, income, shares_outstanding, 
                    latest_eps, filing_date, period_end_date, filing_type, 
                    last_updated, raw_data
                FROM company_financials 
                WHERE ticker = ?
                """,
                (ticker.upper(),),
            )
        result = cursor.fetchone()
    finally:
        conn.close()

    if not result:
        return None

    return {
        "ticker": result[0],
        "company_name": result[1],
        "revenue": result[2],
        "income": result[3],
        "shares_outstanding": result[4],
        "latest_eps": result[5],
        "filing_date": result[6],
        "period_end_date": result[7],
        "filing_type": result[8],
        "last_updated": result[9],
        "raw_data": result[10],
    }

def get_all_tickers_with_financials() -> List[str]:
    """Return list of all tickers that have financial data stored."""
    if is_supabase_configured():
        return supabase_get_all_tickers_with_financials()

    print("Warning: Supabase not configured, falling back to SQLite")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT ticker FROM company_financials ORDER BY ticker")
        results = cursor.fetchall()
    finally:
        conn.close()
    return [row[0] for row in results]


"""
Supabase database helper for company_financials table
Handles all interactions with the Supabase database
"""

import os
from datetime import datetime
from typing import Optional, Dict, List
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}")


def is_supabase_configured() -> bool:
    """Check if Supabase is properly configured"""
    return supabase is not None


def store_company_financials(
    ticker: str,
    company_name: str,
    revenue: Optional[float],
    shares_outstanding: Optional[float],
    filing_date: str,
    period_end_date: str,
    filing_type: str,
    raw_data: Optional[str] = None,
    valuation_metrics: Optional[Dict] = None,
    income: Optional[float] = None,
    latest_eps: Optional[float] = None,
    revenue_label: Optional[str] = None,
    income_label: Optional[str] = None
) -> bool:
    """
    Store company financial data in Supabase
    Returns True if successful, False otherwise
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        data = {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "revenue": revenue,
            "shares_outstanding": shares_outstanding,
            "filing_date": filing_date,
            "period_end_date": period_end_date,
            "filing_type": filing_type,
            "last_updated": datetime.now().isoformat(),
            "raw_data": raw_data,
            "income": income,
            "latest_eps": latest_eps,
            "revenue_label": revenue_label,
            "income_label": income_label
        }
        
        # Add valuation_metrics if provided
        if valuation_metrics is not None:
            import json
            data["valuation_metrics"] = json.dumps(valuation_metrics)
        
        # Upsert (insert or update if exists)
        result = supabase.table("company_financials").upsert(data).execute()
        return True
    except Exception as e:
        print(f"Error storing financial data for {ticker}: {e}")
        return False


def get_company_financials(ticker: str) -> Optional[Dict]:
    """
    Retrieve company financial data from Supabase
    Returns normalized dict with financial data or None if not found
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = supabase.table("company_financials").select(
            "ticker, company_name, revenue, income, shares_outstanding, latest_eps, filing_date, period_end_date, filing_type, last_updated, raw_data"
        ).eq("ticker", ticker.upper()).execute()
        
        if not result.data:
            return None
        
        row = result.data[0]
        # ✅ Normalize key names for consistency with SQLite fallback
        return {
            "ticker": row.get("ticker"),
            "company_name": row.get("company_name"),
            "revenue": row.get("revenue"),
            "income": row.get("income"),
            "shares_outstanding": row.get("shares_outstanding"),
            "latest_eps": row.get("latest_eps"),
            "filing_date": row.get("filing_date"),
            "period_end": row.get("period_end_date"),
            "filing_type": row.get("filing_type"),
            "last_updated": row.get("last_updated"),
            "raw_data": row.get("raw_data"),
        }
    except Exception as e:
        print(f"Error retrieving financial data for {ticker}: {e}")
        return None



def get_all_tickers_with_financials() -> List[str]:
    """
    Get list of all tickers that have financial data stored in Supabase
    Returns list of ticker symbols
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = supabase.table("company_financials").select("ticker").order("ticker").execute()
        return [row["ticker"] for row in result.data]
    except Exception as e:
        print(f"Error retrieving tickers: {e}")
        return []


def check_ticker_exists(ticker: str) -> bool:
    """
    Check if a ticker already exists in the database
    Returns True if exists, False otherwise
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = supabase.table("company_financials").select("ticker").eq("ticker", ticker.upper()).execute()
        return len(result.data) > 0
    except Exception as e:
        print(f"Error checking ticker existence: {e}")
        return False


def get_all_company_financials() -> List[Dict]:
    """
    Retrieve all company financial data from Supabase
    Returns list of dicts with financial data
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = supabase.table("company_financials").select("*").execute()
        return result.data
    except Exception as e:
        print(f"Error retrieving all financial data: {e}")
        return []


def update_valuation_metrics(ticker: str, valuation_metrics: Dict) -> bool:
    """
    Update only the valuation_metrics field for a ticker
    Returns True if successful, False otherwise
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        import json
        data = {
            "ticker": ticker.upper(),
            "valuation_metrics": json.dumps(valuation_metrics)
        }
        
        result = supabase.table("company_financials").upsert(data).execute()
        return True
    except Exception as e:
        print(f"Error updating valuation metrics for {ticker}: {e}")
        return False

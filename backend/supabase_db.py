"""
Supabase database helper for company_financials table
Handles all interactions with the Supabase database
"""

import os
import re
from datetime import datetime
from typing import Optional, Dict, List, Union
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


def get_early_deals(category: Optional[Union[str, List[str]]] = None, funding_round: Optional[Union[str, List[str]]] = None) -> List[Dict]:
    """
    Retrieve early deals from Supabase
    Optional filters: category, funding_round (both case-insensitive, supports lists)
    Returns list of deal dictionaries, with header rows filtered out
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        # Fetch all deals (we'll filter in Python for case-insensitive matching)
        query = supabase.table("early_deals").select("*")
        
        # Order by date descending (most recent first)
        result = query.order("Date", desc=True).execute()
        deals = result.data
        
        # Filter out header rows (rows where Company field matches common header strings)
        header_keywords = ["company", "company name", "name"]
        filtered_deals = []
        for deal in deals:
            company = str(deal.get("Company", "")).strip().lower()
            # Skip if company name matches header keywords
            if company and company not in header_keywords:
                # Additional check: skip if multiple fields match their column names (header row)
                matches = sum(1 for key, value in deal.items() 
                            if value and str(value).strip().lower() == str(key).strip().lower())
                if matches < 2:  # If less than 2 fields match column names, it's likely a data row
                    filtered_deals.append(deal)
        
        # Apply case-insensitive filters (support both single values and lists)
        if category:
            # Handle both list and single value
            if isinstance(category, list):
                category_lower_set = {c.lower().strip() for c in category if c}
            else:
                category_lower_set = {category.lower().strip()}
            
            if category_lower_set:
                filtered_deals = [d for d in filtered_deals 
                                if d.get("Category") and str(d.get("Category")).lower().strip() in category_lower_set]
        
        if funding_round:
            # Handle both list and single value - normalize filter values
            if isinstance(funding_round, list):
                normalized_filters = {normalize_funding_round(fr) for fr in funding_round if fr}
            else:
                normalized_filters = {normalize_funding_round(funding_round)}
            
            if normalized_filters:
                filtered_deals = [d for d in filtered_deals 
                                if d.get("Funding Round") and normalize_funding_round(str(d.get("Funding Round"))) in normalized_filters]
        
        return filtered_deals
    except Exception as e:
        print(f"Error retrieving early deals: {e}")
        return []


def get_deals_categories() -> List[str]:
    """
    Get unique list of categories from early_deals table (case-insensitive deduplication)
    Returns sorted list of category strings with original capitalization preserved
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        # Select all columns to avoid issues with column names containing spaces
        result = supabase.table("early_deals").select("*").execute()
        categories_raw = [row.get("Category") for row in result.data if row.get("Category")]
        
        # Filter out header rows (skip if category matches "Category" or similar)
        categories_raw = [c for c in categories_raw 
                         if str(c).strip().lower() not in ["category", "categories", ""]]
        
        # Case-insensitive deduplication: keep first occurrence of each unique lowercase value
        seen_lower = set()
        unique_categories = []
        for cat in categories_raw:
            cat_str = str(cat).strip()
            cat_lower = cat_str.lower()
            if cat_lower not in seen_lower and cat_str:
                seen_lower.add(cat_lower)
                unique_categories.append(cat_str)
        
        return sorted(unique_categories)
    except Exception as e:
        print(f"Error retrieving deal categories: {e}")
        return []


def normalize_funding_round(round_str: str) -> str:
    """
    Normalize funding round strings to combine variations of 'unknown'/'unspecified'
    Examples: 'Not specified', 'Unknown', 'unspecified', 'not specified (some text)', 
              'Series Unknown', 'Series Unspecified' -> 'Unknown/Unspecified'
    """
    if not round_str:
        return ""
    
    # Remove parentheses and their contents, then strip
    cleaned = re.sub(r'\([^)]*\)', '', str(round_str)).strip()
    cleaned_lower = cleaned.lower()
    
    # Check if it's a variant of unknown/unspecified
    unknown_variants = [
        "unknown", "unspecified", "not specified", 
        "series unknown", "series unspecified"
    ]
    
    for variant in unknown_variants:
        if variant in cleaned_lower:
            return "Unknown/Unspecified"
    
    # Return the cleaned original if not a variant
    return cleaned


def get_deals_funding_rounds() -> List[str]:
    """
    Get unique list of funding rounds from early_deals table (case-insensitive deduplication)
    Combines variations of 'unknown'/'unspecified' into a single option
    Returns sorted list of funding round strings
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        # Select all columns to avoid issues with column names containing spaces
        result = supabase.table("early_deals").select("*").execute()
        rounds_raw = [row.get("Funding Round") for row in result.data if row.get("Funding Round")]
        
        # Filter out header rows (skip if funding round matches "Funding Round" or similar)
        header_keywords = ["funding round", "funding rounds", "round", "rounds", ""]
        rounds_raw = [r for r in rounds_raw 
                     if str(r).strip().lower() not in header_keywords]
        
        # Normalize and deduplicate
        normalized_map = {}  # maps normalized -> original (first occurrence)
        for round_val in rounds_raw:
            round_str = str(round_val).strip()
            if not round_str:
                continue
            
            normalized = normalize_funding_round(round_str)
            if normalized and normalized not in normalized_map:
                normalized_map[normalized] = round_str
        
        # Return sorted list of normalized values
        return sorted(normalized_map.keys())
    except Exception as e:
        print(f"Error retrieving funding rounds: {e}")
        return []

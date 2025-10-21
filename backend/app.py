def _safe_number(v):
    """Convert to float if finite, else return None."""
    try:
        if v is None:
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def _clean_json_numbers(obj):
    """Recursively replace non-finite floats (NaN/inf) with None in dicts/lists."""
    if isinstance(obj, dict):
        return {k: _clean_json_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_numbers(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _select_preferred_period(periods, period_metadata=None, prefer='10-K'):
    """
    Select the preferred period from available periods, prioritizing annual (10-K) over quarterly (10-Q).
    
    Args:
        periods: List of period strings (dates)
        period_metadata: Optional list of dicts with 'date' and 'type' keys
        prefer: Preferred filing type ('10-K' for annual, '10-Q' for quarterly)
        
    Returns:
        Preferred period string, or None if no periods available
    """
    if not periods:
        return None
    
    # Sort periods by date (most recent first)
    sorted_periods = sorted(periods, key=lambda x: str(x), reverse=True)
    
    # If we have period metadata, use it to find the preferred filing type
    if period_metadata:
        for period_info in period_metadata:
            if period_info.get('type') == prefer and period_info.get('date') in periods:
                return period_info['date']
    
    # Fallback to most recent period
    return sorted_periods[0]


def _select_best_revenue(items, periods, prefer_period=None):
    """Select the best revenue value using shared extraction logic.
    
    Returns: (revenue_value, concept_used, confidence_level)
    confidence_level: 'high', 'medium', 'low', or None
    """
    # Get cost of revenue for plausibility check
    cor = None
    for item in items:
        if item.get('concept') == 'us-gaap_CostOfRevenue':
            values = item.get('values') or {}
            if prefer_period and prefer_period in values:
                cor = abs(float(values[prefer_period])) if values[prefer_period] else None
            else:
                for p in periods:
                    if p in values and values[p]:
                        cor = abs(float(values[p]))
                        break
            if cor:
                break
    
    # Use shared revenue extraction logic with new signature
    revenue, metadata = extract_revenue(items, periods, prefer_period=prefer_period, cost_of_revenue=cor)
    
    if revenue is None:
        return None, None, None
    
    # Determine confidence based on metadata
    confidence = 'high'
    if metadata.get('valid_candidates_count', 0) > 1:
        # Multiple valid candidates - medium confidence
        confidence = 'medium'
    if not metadata.get('plausible', True):
        # Failed plausibility check - low confidence
        confidence = 'low'
    if metadata.get('reason') and 'fallback' in metadata.get('reason', ''):
        # Used fallback logic - medium confidence
        confidence = 'medium'
    
    return revenue, metadata.get('concept'), confidence


from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import sqlite3
import json
import math
import os
import sys
import time
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from dotenv import load_dotenv
from edgar import Company, set_identity
from edgar.xbrl import XBRLS
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from supabase_db import (
    store_company_financials as supabase_store_company_financials,
    get_company_financials as supabase_get_company_financials,
    get_all_tickers_with_financials as supabase_get_all_tickers_with_financials,
    is_supabase_configured
)

# Add scripts directory to path for revenue_extractor
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from revenue_extractor import extract_revenue

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Get the path to the frontend directory
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

# Set your Edgar identity
set_identity("matthildur@montageventures.com")

# Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET")

# Initialize Alpaca client
alpaca_client = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Database setup
# Use absolute path to ensure cache.db is created in the correct location
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache.db")
CACHE_DURATION_DAYS = 7
STOCK_PRICE_CACHE_MINUTES = 5  # Cache stock prices for 5 minutes

# Rate limiting for Alpaca API (200 requests per minute)
ALPACA_RATE_LIMIT = 200  # requests per minute
ALPACA_RATE_WINDOW = 60  # seconds
alpaca_request_times = deque()  # Track request timestamps


def check_alpaca_rate_limit():
    """
    Check if we can make an Alpaca API request without exceeding rate limit.
    Returns True if request is allowed, False otherwise.
    """
    current_time = time.time()
    
    # Remove timestamps older than the rate window
    while alpaca_request_times and current_time - alpaca_request_times[0] > ALPACA_RATE_WINDOW:
        alpaca_request_times.popleft()
    
    # Check if we're at the limit
    if len(alpaca_request_times) >= ALPACA_RATE_LIMIT:
        return False
    
    # Record this request
    alpaca_request_times.append(current_time)
    return True


def init_db():
    """Initialize the SQLite database for caching (stock prices and income statements)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Cache tables for temporary data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS income_cache (
            ticker TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            cached_at TIMESTAMP NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_price_cache (
            ticker TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            cached_at TIMESTAMP NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS industry_cache (
            cache_key TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            cached_at TIMESTAMP NOT NULL
        )
    """)
    
    # Note: company_financials table is now in Supabase
    # Legacy SQLite table kept for backward compatibility but not actively used
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_financials (
            ticker TEXT PRIMARY KEY,
            company_name TEXT NOT NULL,
            revenue REAL,
            shares_outstanding REAL,
            filing_date TEXT,
            period_end_date TEXT,
            filing_type TEXT,
            last_updated TIMESTAMP NOT NULL,
            raw_data TEXT
        )
    """)
    
    conn.commit()
    conn.close()


# Industry cache helpers
INDUSTRY_CACHE_MINUTES = 15

def get_industry_cache(cache_key, max_age_minutes=INDUSTRY_CACHE_MINUTES):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data, cached_at FROM industry_cache WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        data_str, cached_at = row
        cached_time = datetime.fromisoformat(cached_at)
        if datetime.now() - cached_time > timedelta(minutes=max_age_minutes):
            return None
        return json.loads(data_str)
    except Exception:
        return None

def set_industry_cache(cache_key, data):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO industry_cache (cache_key, data, cached_at) VALUES (?, ?, ?)",
            (cache_key, json.dumps(data), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_cached_data(ticker):
    """Retrieve cached data if it exists and is not expired"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT data, cached_at FROM income_cache WHERE ticker = ?",
        (ticker.upper(),)
    )
    result = cursor.fetchone()
    conn.close()
    
    if result:
        data, cached_at = result
        cached_time = datetime.fromisoformat(cached_at)
        if datetime.now() - cached_time < timedelta(days=CACHE_DURATION_DAYS):
            return json.loads(data)
    return None


def cache_data(ticker, data):
    """Cache the data in SQLite"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO income_cache (ticker, data, cached_at)
        VALUES (?, ?, ?)
        """,
        (ticker.upper(), json.dumps(data), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def store_company_financials(ticker, company_name, revenue, shares, filing_date, period_end, filing_type, raw_data=None):
    """Store company financial data in Supabase database"""
    if is_supabase_configured():
        return supabase_store_company_financials(
            ticker, company_name, revenue, shares, filing_date, period_end, filing_type,
            json.dumps(raw_data) if raw_data else None
        )
    else:
        # Fallback to SQLite if Supabase not configured
        print("Warning: Supabase not configured, falling back to SQLite")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO company_financials 
            (ticker, company_name, revenue, shares_outstanding, filing_date, period_end_date, filing_type, last_updated, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticker.upper(), company_name, revenue, shares, filing_date, period_end, filing_type, 
             datetime.now().isoformat(), json.dumps(raw_data) if raw_data else None)
        )
        conn.commit()
        conn.close()
        return True


def get_company_financials(ticker):
    """Retrieve company financial data from Supabase database"""
    if is_supabase_configured():
        return supabase_get_company_financials(ticker)
    else:
        # Fallback to SQLite if Supabase not configured
        print("Warning: Supabase not configured, falling back to SQLite")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT ticker, company_name, revenue, shares_outstanding, filing_date, 
                   period_end_date, filing_type, last_updated, raw_data
            FROM company_financials 
            WHERE ticker = ?
            """,
            (ticker.upper(),)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "ticker": result[0],
                "company_name": result[1],
                "revenue": result[2],
                "shares_outstanding": result[3],
                "filing_date": result[4],
                "period_end_date": result[5],
                "filing_type": result[6],
                "last_updated": result[7],
                "raw_data": result[8]
            }
        return None


def get_all_tickers_with_financials():
    """Get list of all tickers that have financial data stored in Supabase"""
    if is_supabase_configured():
        return supabase_get_all_tickers_with_financials()
    else:
        # Fallback to SQLite if Supabase not configured
        print("Warning: Supabase not configured, falling back to SQLite")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker FROM company_financials ORDER BY ticker")
        results = cursor.fetchall()
        conn.close()
        return [row[0] for row in results]


def get_income_dataframe(ticker: str):
    """Fetch income statement data from Edgar - both annual and quarterly"""
    c = Company(ticker)
    
    # Get latest annual (10-K) and quarterly (10-Q) filings
    try:
        filing_10k = c.get_filings(form="10-K").latest(1)
    except:
        filing_10k = None
    
    try:
        filing_10q = c.get_filings(form="10-Q").latest(1)
    except:
        filing_10q = None
    
    # Combine filings - put 10-Q first so it appears as the most recent
    all_filings = []
    filing_types = {}  # Track which period comes from which filing type
    
    if filing_10q:
        all_filings.append(filing_10q)
    if filing_10k:
        all_filings.append(filing_10k)
    
    if not all_filings:
        raise ValueError("No filings found")
    
    # Get XBRL data from all filings
    xbs = XBRLS.from_filings(all_filings)
    income_statement = xbs.statements.income_statement()
    income_df = income_statement.to_dataframe()
    
    # Map periods to filing types by matching with actual filing dates
    # Ensure periods is always a list
    periods_list = list(income_statement.periods) if hasattr(income_statement.periods, '__iter__') else [income_statement.periods]
    
    # Get the period dates from each filing
    if filing_10q:
        try:
            period_10q = filing_10q.period_of_report
            print(f"10-Q period: {period_10q}")
            filing_types[str(period_10q)] = "10-Q"
        except Exception as e:
            print(f"Error getting 10-Q period: {e}")
    
    if filing_10k:
        try:
            period_10k = filing_10k.period_of_report
            print(f"10-K period: {period_10k}")
            filing_types[str(period_10k)] = "10-K"
        except Exception as e:
            print(f"Error getting 10-K period: {e}")
    
    print(f"Periods from income statement: {periods_list}")
    print(f"Filing types mapping: {filing_types}")
    
    # If we couldn't match by period_of_report, fall back to position-based matching
    if not filing_types:
        print("WARNING: Using fallback position-based matching")
        for i, period in enumerate(periods_list):
            if i == 0 and filing_10q:
                filing_types[str(period)] = "10-Q"
            elif i == 1 and filing_10k:
                filing_types[str(period)] = "10-K"
            elif i == 0 and filing_10k:
                filing_types[str(period)] = "10-K"
    
    return income_df, c.name, periods_list, filing_types


def get_all_financial_statements(ticker: str):
    """Fetch income statement, balance sheet, and cash flow data"""
    c = Company(ticker)
    
    # Get latest annual (10-K) and quarterly (10-Q) filings
    try:
        filing_10k = c.get_filings(form="10-K").latest(1)
    except:
        filing_10k = None
    
    try:
        filing_10q = c.get_filings(form="10-Q").latest(1)
    except:
        filing_10q = None
    
    # Combine filings - put 10-Q first so it appears as the most recent
    all_filings = []
    if filing_10q:
        all_filings.append(filing_10q)
    if filing_10k:
        all_filings.append(filing_10k)
    
    if not all_filings:
        raise ValueError("No filings found")
    
    # Get XBRL data from all filings
    xbs = XBRLS.from_filings(all_filings)
    
    result = {
        "company_name": c.name,
        "periods": None,
        "income_statement": None,
        "balance_sheet": None,
        "cash_flow": None
    }
    
    # Get income statement
    try:
        income_statement = xbs.statements.income_statement()
        result["income_statement"] = income_statement.to_dataframe()
        result["periods"] = list(income_statement.periods) if hasattr(income_statement.periods, '__iter__') else [income_statement.periods]
    except:
        pass
    
    # Get balance sheet
    try:
        balance_sheet = xbs.statements.balance_sheet()
        result["balance_sheet"] = balance_sheet.to_dataframe()
        if not result["periods"]:
            result["periods"] = list(balance_sheet.periods) if hasattr(balance_sheet.periods, '__iter__') else [balance_sheet.periods]
    except:
        pass
    
    # Get cash flow statement
    try:
        cash_flow = xbs.statements.cash_flow_statement()
        result["cash_flow"] = cash_flow.to_dataframe()
        if not result["periods"]:
            result["periods"] = list(cash_flow.periods) if hasattr(cash_flow.periods, '__iter__') else [cash_flow.periods]
    except:
        pass
    
    return result


@app.route("/api/income/<ticker>", methods=["GET"])
def get_income(ticker):
    """API endpoint to get the most recent income statement"""
    try:
        # Check cache first
        cached = get_cached_data(ticker)
        if cached:
            cached = _clean_json_numbers(cached)
            return jsonify({
                "success": True,
                "data": cached,
                "cached": True
            })
        
        # Fetch from Edgar if not cached or expired
        income_df, company_name, periods, filing_types = get_income_dataframe(ticker)
        
        if not periods or len(periods) == 0:
            return jsonify({
                "success": False,
                "error": "No data available"
            }), 404
        
        # Sort periods by date (most recent first)
        sorted_periods = sorted(periods, key=lambda x: x, reverse=True)
        
        # Convert to list of dictionaries with multiple periods
        result_data = []
        for _, row in income_df.iterrows():
            item = {
                "label": row["label"],
                "concept": row["concept"],
                "values": {}
            }
            # Add value for each period
            for period in sorted_periods:
                if period in row:
                    item["values"][period] = _safe_number(row[period])
            result_data.append(item)
        
        # Create period metadata with filing types
        period_metadata = []
        for period in sorted_periods:
            period_metadata.append({
                "date": period,
                "type": filing_types.get(period, "Unknown")
            })
        
        response_data = {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "periods": sorted_periods,  # List of all periods (most recent first)
            "period_metadata": period_metadata,  # Metadata about each period
            "items": result_data
        }
        
        # Cache the result
        response_data = _clean_json_numbers(response_data)
        cache_data(ticker, response_data)
        return jsonify({
            "success": True,
            "data": response_data,
            "cached": False
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/stored-metrics/<ticker>", methods=["GET"])
def get_stored_metrics(ticker):
    """Fast endpoint to return stored metrics (equity, debt, operating cash flow, net income) from DB raw_data.
    Does not call external APIs. Returns periods list and metrics by period when available.
    """
    try:
        fin = get_company_financials(ticker.upper())
        if not fin or not fin.get('raw_data'):
            return jsonify({"success": False, "error": "No stored metrics"}), 404

        raw_payload = fin['raw_data']
        if isinstance(raw_payload, str):
            raw_payload = json.loads(raw_payload)

        metrics = (raw_payload or {}).get('metrics') or {}

        # Extract net income from income statement if not already in metrics
        if 'net_income' not in metrics:
            income_stmt = (raw_payload or {}).get('income_statement') or {}
            income_items = income_stmt.get('items') or []
            income_periods = income_stmt.get('periods') or []
            
            # Find net income item
            for item in income_items:
                concept = item.get('concept', '')
                if 'NetIncomeLoss' in concept or concept == 'us-gaap_NetIncomeLoss':
                    net_income_map = {}
                    values = item.get('values', {})
                    for period in income_periods:
                        if period in values and values[period] is not None:
                            net_income_map[period] = values[period]
                    if net_income_map:
                        metrics['net_income'] = net_income_map
                    break

        # Extract EPS (basic and diluted) per period to support normalized P/E (TTM)
        try:
            if 'eps_basic' not in metrics or 'eps_diluted' not in metrics:
                income_stmt = (raw_payload or {}).get('income_statement') or {}
                income_items = income_stmt.get('items') or []
                income_periods = income_stmt.get('periods') or []
                eps_basic_map = {}
                eps_diluted_map = {}
                for item in income_items:
                    concept = item.get('concept', '')
                    if concept == 'us-gaap_EarningsPerShareBasic':
                        for period in income_periods:
                            v = (item.get('values') or {}).get(period)
                            if v is not None:
                                eps_basic_map[period] = v
                    if concept == 'us-gaap_EarningsPerShareDiluted':
                        for period in income_periods:
                            v = (item.get('values') or {}).get(period)
                            if v is not None:
                                eps_diluted_map[period] = v
                if eps_basic_map:
                    metrics['eps_basic'] = eps_basic_map
                if eps_diluted_map:
                    metrics['eps_diluted'] = eps_diluted_map
        except Exception:
            pass

        # Extract shares outstanding per period (from income statement first, then balance sheet)
        try:
            if 'shares' not in metrics:
                shares_map = {}
                income_stmt = (raw_payload or {}).get('income_statement') or {}
                income_items = income_stmt.get('items') or []
                income_periods = income_stmt.get('periods') or []
                shares_candidates = {
                    'us-gaap_CommonStockSharesOutstanding',
                    'us-gaap_WeightedAverageNumberOfSharesOutstandingBasic',
                    'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding'
                }
                for item in income_items:
                    c = str(item.get('concept', ''))
                    if c in shares_candidates:
                        values = item.get('values') or {}
                        for p in income_periods:
                            v = values.get(p)
                            if v is not None:
                                shares_map[p] = v
                        break
                # Fallback to balance sheet
                if not shares_map:
                    bs = (raw_payload or {}).get('balance_sheet') or {}
                    bs_items = bs.get('items') or []
                    bs_periods = bs.get('periods') or []
                    
                    # Try direct shares concepts first
                    for item in bs_items:
                        c = str(item.get('concept', ''))
                        if c in shares_candidates:
                            values = item.get('values') or {}
                            for p in bs_periods:
                                v = values.get(p)
                                if v is not None:
                                    shares_map[p] = v
                            break
                    
                    # If still not found, try calculation methods
                    if not shares_map:
                        # Collect data for calculations
                        shares_issued_map = {}
                        treasury_shares_map = {}
                        common_stock_value_map = {}
                        par_value_map = {}
                        
                        for item in bs_items:
                            c = str(item.get('concept', ''))
                            values = item.get('values') or {}
                            
                            # Method 2: Issued - Treasury
                            if c == 'us-gaap_CommonStockSharesIssued':
                                shares_issued_map = values
                            elif c in ['us-gaap_TreasuryStockCommonShares', 'us-gaap_TreasuryStockShares']:
                                treasury_shares_map = values
                            
                            # Method 1: Common Stock Value / Par Value
                            elif c == 'us-gaap_CommonStockValue':
                                common_stock_value_map = values
                            elif c == 'us-gaap_CommonStockParOrStatedValuePerShare':
                                par_value_map = values
                        
                        # Try Method 2: Issued - Treasury
                        if shares_issued_map:
                            for p in bs_periods:
                                issued = shares_issued_map.get(p)
                                treasury = treasury_shares_map.get(p, 0)
                                if issued is not None:
                                    try:
                                        issued_float = float(issued)
                                        treasury_float = float(treasury) if treasury else 0
                                        shares_map[p] = issued_float - treasury_float
                                    except (ValueError, TypeError):
                                        pass
                        
                        # Try Method 1: Common Stock Value / Par Value
                        if not shares_map and common_stock_value_map and par_value_map:
                            for p in bs_periods:
                                value = common_stock_value_map.get(p)
                                par = par_value_map.get(p)
                                if value is not None and par and par > 0:
                                    try:
                                        value_float = float(value)
                                        par_float = float(par)
                                        if par_float > 0:
                                            shares_map[p] = value_float / par_float
                                    except (ValueError, TypeError):
                                        pass
                
                # Method 3: Calculate from Net Income / EPS
                if not shares_map:
                    net_income_map = {}
                    eps_map = {}
                    
                    for item in income_items:
                        c = str(item.get('concept', ''))
                        values = item.get('values') or {}
                        
                        if c == 'us-gaap_NetIncomeLoss':
                            net_income_map = values
                        elif c == 'us-gaap_EarningsPerShareBasic':
                            eps_map = values
                    
                    if net_income_map and eps_map:
                        for p in income_periods:
                            ni = net_income_map.get(p)
                            eps = eps_map.get(p)
                            if ni is not None and eps and eps != 0:
                                try:
                                    # Ensure both are numeric before division
                                    ni_float = float(ni)
                                    eps_float = float(eps)
                                    shares_map[p] = abs(ni_float / eps_float)  # abs() in case of net loss
                                except (ValueError, TypeError):
                                    # Skip if values aren't numeric
                                    pass
                
                if shares_map:
                    metrics['shares'] = shares_map
        except Exception as ex:
            print(f"[WARNING] Error extracting shares for {ticker}: {ex}")
            pass

        # Build a unified sorted periods list across statements (most recent first)
        periods = []
        try:
            income_periods = (raw_payload.get('income_statement') or {}).get('periods') or []
            bs_periods = (raw_payload.get('balance_sheet') or {}).get('periods') or []
            cf_periods = (raw_payload.get('cash_flow') or {}).get('periods') or []
            period_set = set()
            for p in (income_periods + bs_periods + cf_periods):
                if p:
                    period_set.add(p)
            periods = sorted(period_set, reverse=True)
        except Exception:
            periods = []

        return jsonify({
            "success": True,
            "data": {
                "ticker": fin['ticker'],
                "company_name": fin['company_name'],
                "periods": periods,
                "metrics": metrics
            }
        })
    except Exception as e:
        import traceback
        print(f"[ERROR] /api/stored-metrics/{ticker} failed:")
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stock-price/<ticker>", methods=["GET"])
def get_stock_price(ticker):
    """API endpoint to get the latest stock price from Alpaca"""
    try:
        ticker_upper = ticker.upper()
        
        # Check cache first
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data, cached_at FROM stock_price_cache WHERE ticker = ?",
            (ticker_upper,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            data, cached_at = result
            cached_time = datetime.fromisoformat(cached_at)
            if datetime.now() - cached_time < timedelta(minutes=STOCK_PRICE_CACHE_MINUTES):
                return jsonify({
                    "success": True,
                    "data": json.loads(data),
                    "cached": True
                })
        
        # Fetch from Alpaca if not cached or expired
        if not alpaca_client:
            return jsonify({
                "success": False,
                "error": "Alpaca API credentials not configured"
            }), 500
        
        # Check rate limit before making API call
        if not check_alpaca_rate_limit():
            return jsonify({
                "success": False,
                "error": "Rate limit exceeded. Please try again in a moment."
            }), 429
        
        # Get latest trade data
        request_params = StockLatestTradeRequest(symbol_or_symbols=ticker_upper)
        trades = alpaca_client.get_stock_latest_trade(request_params)
        
        if ticker_upper not in trades:
            return jsonify({
                "success": False,
                "error": "No data found for ticker"
            }), 404
        
        trade = trades[ticker_upper]
        
        result_data = {
            "ticker": ticker_upper,
            "price": float(trade.price),
            "size": float(trade.size),
            "timestamp": trade.timestamp.isoformat(),
            "exchange": trade.exchange
        }
        
        # Cache the result
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO stock_price_cache (ticker, data, cached_at)
            VALUES (?, ?, ?)
            """,
            (ticker_upper, json.dumps(result_data), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "data": result_data,
            "cached": False
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/financial-data/<ticker>", methods=["GET"])
def get_financial_data(ticker):
    """API endpoint to get additional financial data for ratio calculations"""
    try:
        ticker_upper = ticker.upper()
        
        # Check cache first
        cache_key = ticker_upper + "_financial_data"
        cached = get_cached_data(cache_key)
        if cached:
            cached = _clean_json_numbers(cached)
            return jsonify({
                "success": True,
                "data": cached,
                "cached": True
            })
        
        # Fetch all financial statements
        statements = get_all_financial_statements(ticker_upper)
        
        if not statements["periods"]:
            return jsonify({
                "success": False,
                "error": "No financial data available"
            }), 404
        
        # Sort periods by date (most recent first)
        sorted_periods = sorted(statements["periods"], key=lambda x: x, reverse=True)
        
        # Extract key metrics from balance sheet and cash flow
        result_data = {
            "ticker": ticker_upper,
            "company_name": statements["company_name"],
            "periods": sorted_periods,
            "metrics": {}
        }
        
        # Extract balance sheet metrics
        if statements["balance_sheet"] is not None:
            bs_df = statements["balance_sheet"]

            # Normalize accessors
            def first_row_by_concepts(df, concept_substrings):
                for _, r in df.iterrows():
                    c = str(r.get('concept', ''))
                    if any(sub in c for sub in concept_substrings):
                        return r
                return None

            # Stockholders' equity: handle common variants
            equity_candidates = [
                'us-gaap_StockholdersEquity',
                'us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                'StockholdersEquity',
                'PartnersCapital',
                'us-gaap_LiabilitiesAndStockholdersEquity'  # Sometimes only total is available
            ]
            equity_item = first_row_by_concepts(bs_df, equity_candidates)
            
            # If we got total liabilities + equity, try to subtract liabilities
            if equity_item is not None and 'LiabilitiesAndStockholdersEquity' in str(equity_item.get('concept', '')):
                # Try to find total liabilities to subtract
                liabilities_candidates = ['us-gaap_Liabilities', 'Liabilities']
                liabilities_item = first_row_by_concepts(bs_df, liabilities_candidates)
                if liabilities_item is not None:
                    # Create a new row with equity = total - liabilities
                    equity_item = equity_item.copy()
                    for period in sorted_periods:
                        if period in equity_item and period in liabilities_item:
                            total = _safe_number(equity_item[period])
                            liab = _safe_number(liabilities_item[period])
                            if total is not None and liab is not None:
                                equity_item[period] = total - liab
                else:
                    # Can't calculate equity, set to None
                    equity_item = None
            
            # Debug: If equity not found, log available concepts
            if equity_item is None:
                print(f"[DEBUG] Equity not found. Available balance sheet concepts:")
                for _, r in bs_df.head(20).iterrows():
                    concept = r.get('concept', '')
                    label = r.get('label', '')
                    if 'equity' in label.lower() or 'capital' in label.lower():
                        print(f"  - {concept}: {label}")

            # Debt: combine current + noncurrent variants if present
            debt_current_candidates = [
                'us-gaap_DebtCurrent',
                'us-gaap_ShortTermBorrowings',
                'us-gaap_NotesPayableShortTerm',
                'us-gaap_CommercialPaper',
                'DebtCurrent'
            ]
            debt_noncurrent_candidates = [
                'us-gaap_LongTermDebtNoncurrent',
                'us-gaap_LongTermDebtAndCapitalLeaseObligations',
                'us-gaap_LongTermDebt',
                'us-gaap_LongTermDebtAndCapitalLeaseObligationsNoncurrent',
                'LongTermDebt'
            ]
            debt_current = first_row_by_concepts(bs_df, debt_current_candidates)
            debt_noncurrent = first_row_by_concepts(bs_df, debt_noncurrent_candidates)
            
            # Debug: If debt not found, log available concepts
            if debt_current is None and debt_noncurrent is None:
                print(f"[DEBUG] Debt not found. Available balance sheet concepts:")
                for _, r in bs_df.head(30).iterrows():
                    concept = r.get('concept', '')
                    label = r.get('label', '')
                    if 'debt' in label.lower() or 'borrowing' in label.lower() or 'note' in label.lower():
                        print(f"  - {concept}: {label}")

            if equity_item is not None:
                result_data["metrics"]["equity"] = {}
                for period in sorted_periods:
                    if period in equity_item:
                        result_data["metrics"]["equity"][period] = _safe_number(equity_item[period])

            # Build total debt per period
            if debt_current is not None or debt_noncurrent is not None:
                result_data["metrics"]["debt"] = {}
                for period in sorted_periods:
                    total = 0.0
                    has_val = False
                    if debt_current is not None and period in debt_current:
                        v = _safe_number(debt_current[period])
                        if v is not None:
                            total += v
                            has_val = True
                    if debt_noncurrent is not None and period in debt_noncurrent:
                        v = _safe_number(debt_noncurrent[period])
                        if v is not None:
                            total += v
                            has_val = True
                    result_data["metrics"]["debt"][period] = total if has_val else None
        
        # Extract cash flow metrics
        if statements["cash_flow"] is not None:
            cf_df = statements["cash_flow"]

            # Operating cash flow common variants
            ocf_candidates = [
                'us-gaap_NetCashProvidedByUsedInOperatingActivities',
                'us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
                'OperatingActivities'
            ]
            ocf_item = None
            for _, row in cf_df.iterrows():
                concept = str(row.get('concept', ''))
                if any(sub in concept for sub in ocf_candidates):
                    ocf_item = row
                    break

            if ocf_item is not None:
                result_data["metrics"]["operating_cash_flow"] = {}
                for period in sorted_periods:
                    if period in ocf_item:
                        result_data["metrics"]["operating_cash_flow"][period] = _safe_number(ocf_item[period])
        
        # Cache the result
        result_data = _clean_json_numbers(result_data)
        cache_data(cache_key, result_data)
        
        return jsonify({
            "success": True,
            "data": result_data,
            "cached": False
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/recent-filings/<ticker>", methods=["GET"])
def get_recent_filings(ticker):
    """API endpoint to get recent filings (10-K and 10-Q)"""
    try:
        ticker_upper = ticker.upper()
        
        # Check cache first
        cached = get_cached_data(ticker_upper + "_filings")
        if cached:
            return jsonify({
                "success": True,
                "data": cached,
                "cached": True
            })
        
        # Fetch from Edgar
        c = Company(ticker_upper)
        
        # Get latest 5 filings of each type
        filings_10k = c.get_filings(form="10-K").latest(3)
        filings_10q = c.get_filings(form="10-Q").latest(3)
        
        result_data = {
            "ticker": ticker_upper,
            "company_name": c.name,
            "filings": []
        }
        
        # Process 10-K filings
        if filings_10k:
            for filing in filings_10k:
                result_data["filings"].append({
                    "form": "10-K",
                    "filing_date": filing.filing_date.isoformat() if hasattr(filing.filing_date, 'isoformat') else str(filing.filing_date),
                    "period": filing.period_of_report.isoformat() if hasattr(filing.period_of_report, 'isoformat') else str(filing.period_of_report),
                    "accession_number": filing.accession_number
                })
        
        # Process 10-Q filings
        if filings_10q:
            for filing in filings_10q:
                result_data["filings"].append({
                    "form": "10-Q",
                    "filing_date": filing.filing_date.isoformat() if hasattr(filing.filing_date, 'isoformat') else str(filing.filing_date),
                    "period": filing.period_of_report.isoformat() if hasattr(filing.period_of_report, 'isoformat') else str(filing.period_of_report),
                    "accession_number": filing.accession_number
                })
        
        # Sort by filing date (most recent first)
        result_data["filings"].sort(key=lambda x: x["filing_date"], reverse=True)
        
        # Keep only the 5 most recent
        result_data["filings"] = result_data["filings"][:5]
        
        # Cache the result
        cache_data(ticker_upper + "_filings", result_data)
        
        return jsonify({
            "success": True,
            "data": result_data,
            "cached": False
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/industry-map", methods=["GET"])
def get_industry_map():
    """API endpoint to get the industry map structure"""
    try:
        industry_map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'industry_map.json')
        with open(industry_map_path, 'r') as f:
            industry_map = json.load(f)
        
        return jsonify({
            "success": True,
            "data": industry_map
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/industry-tickers/<path:industry_path>", methods=["GET"])
def get_industry_tickers(industry_path):
    """
    API endpoint to get tickers for a specific industry group
    Path format: Industry/Sector/IndustryGroup
    """
    try:
        industry_map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'industry_map.json')
        with open(industry_map_path, 'r') as f:
            industry_map = json.load(f)
        
        # Parse the path
        parts = industry_path.split('/')
        if len(parts) != 3:
            return jsonify({
                "success": False,
                "error": "Invalid path format. Expected: Industry/Sector/IndustryGroup"
            }), 400
        
        industry, sector, industry_group = parts
        
        # Navigate to the tickers
        tickers = industry_map['industries'][industry]['sectors'][sector]['industry_groups'][industry_group]
        
        return jsonify({
            "success": True,
            "data": {
                "industry": industry,
                "sector": sector,
                "industry_group": industry_group,
                "tickers": tickers
            }
        })
    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Path not found: {e}"
        }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/industry-comparison-stream", methods=["POST"])
def get_industry_comparison_stream():
    """
    API endpoint for progressive industry comparison with rate-limited price fetching
    Returns results as they become available (streaming)
    """
    from flask import Response, stream_with_context
    import sys
    
    def generate():
        try:
            data = request.get_json()
            tickers = data.get('tickers', [])
            use_cache = data.get('use_cache', True)
            
            if not tickers:
                yield json.dumps({"error": "No tickers provided"}) + "\n"
                return
            
            print(f"\n[STREAMING] Starting industry comparison for {len(tickers)} tickers")
            sys.stdout.flush()
            
            # Compute a cache key based on sorted tickers
            tickers_sorted = sorted([t.upper() for t in tickers])
            cache_key = "tickers:" + ",".join(tickers_sorted)

            # Try cache first
            if use_cache:
                cached_payload = get_industry_cache(cache_key)
                if cached_payload and isinstance(cached_payload, list):
                    # Serve cached rows quickly
                    yield json.dumps({
                        "type": "status",
                        "message": f"Loaded {len(cached_payload)} cached companies",
                        "total": len(cached_payload)
                    }) + "\n"
                    processed = 0
                    for row in cached_payload:
                        yield json.dumps({
                            "type": "company",
                            "data": row
                        }) + "\n"
                        processed += 1
                        yield json.dumps({
                            "type": "progress",
                            "processed": processed,
                            "total": len(cached_payload)
                        }) + "\n"
                    yield json.dumps({
                        "type": "complete",
                        "processed": processed,
                        "total": len(cached_payload)
                    }) + "\n"
                    return

            # Send initial status
            yield json.dumps({
                "type": "status",
                "message": f"Loading {len(tickers)} companies...",
                "total": len(tickers)
            }) + "\n"
            
            processed = 0
            # Accumulate for caching at the end
            cache_rows = []
            
            for ticker in tickers:
                ticker_upper = ticker.upper()
                
                try:
                    # Get financial data from persistent DB
                    print(f"[STREAMING] Processing {ticker_upper}...")
                    sys.stdout.flush()
                    
                    financials = get_company_financials(ticker_upper)
                    
                    if not financials:
                        print(f"[STREAMING] {ticker_upper}: No data in database")
                        sys.stdout.flush()
                        yield json.dumps({
                            "type": "skip",
                            "ticker": ticker_upper,
                            "reason": "No financial data in database"
                        }) + "\n"
                        processed += 1
                        continue

                    # Note: Metrics (equity, debt, operating_cash_flow) should be pre-computed
                    # during import. If missing, re-run the import script to backfill them.
                    # We no longer fetch from Edgar during streaming to keep queries fast.
                    

                    # Try to get stock price (rate-limited)
                    stock_price = None
                    
                    # Check cache first
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT data, cached_at FROM stock_price_cache WHERE ticker = ?",
                        (ticker_upper,)
                    )
                    price_result = cursor.fetchone()
                    conn.close()
                    
                    if price_result:
                        price_data, cached_at = price_result
                        cached_time = datetime.fromisoformat(cached_at)
                        if datetime.now() - cached_time < timedelta(minutes=STOCK_PRICE_CACHE_MINUTES):
                            stock_price = json.loads(price_data).get('price')
                    
                    # If no cached price and rate limit allows, fetch it
                    if not stock_price and alpaca_client and check_alpaca_rate_limit():
                        try:
                            request_params = StockLatestTradeRequest(symbol_or_symbols=ticker_upper)
                            trades = alpaca_client.get_stock_latest_trade(request_params)
                            if ticker_upper in trades:
                                stock_price = float(trades[ticker_upper].price)
                                
                                # Cache it
                                result_data = {
                                    "ticker": ticker_upper,
                                    "price": stock_price,
                                    "size": float(trades[ticker_upper].size),
                                    "timestamp": trades[ticker_upper].timestamp.isoformat(),
                                    "exchange": trades[ticker_upper].exchange
                                }
                                conn = sqlite3.connect(DB_PATH)
                                cursor = conn.cursor()
                                cursor.execute(
                                    "INSERT OR REPLACE INTO stock_price_cache (ticker, data, cached_at) VALUES (?, ?, ?)",
                                    (ticker_upper, json.dumps(result_data), datetime.now().isoformat())
                                )
                                conn.commit()
                                conn.close()
                        except Exception as e:
                            print(f"Error fetching price for {ticker_upper}: {e}")
                    
                    # Calculate metrics
                    market_cap = None
                    ps_ratio = None
                    pb_ratio = None
                    pcf_ratio = None
                    de_ratio = None

                    # Attempt to use revenue from raw_data exact concepts first (avoid partial matches like segment revenues)
                    revenue_val = None
                    shares_val = financials.get('shares_outstanding')

                    try:
                        raw_payload = financials.get('raw_data')
                        if isinstance(raw_payload, str):
                            raw_payload = json.loads(raw_payload)
                    except Exception:
                        raw_payload = None

                    # Extract from raw_data income_statement when available
                    if raw_payload:
                        try:
                            income_stmt = (raw_payload or {}).get('income_statement') or {}
                            items = income_stmt.get('items') or []
                            periods = income_stmt.get('periods') or []
                            
                            # Use helper to select preferred period (prioritize 10-K annual data)
                            period_metadata = income_stmt.get('period_metadata')
                            prefer_period = _select_preferred_period(periods, period_metadata, prefer='10-K')
                            
                            # Fallback to stored period_end_date if no metadata available
                            if not prefer_period:
                                prefer_period = financials.get('period_end_date')
                            def pick_value(map_obj):
                                if not isinstance(map_obj, dict):
                                    return None
                                if prefer_period and prefer_period in map_obj and map_obj[prefer_period] is not None:
                                    return map_obj[prefer_period]
                                for p in periods:
                                    if p in map_obj and map_obj[p] is not None:
                                        return map_obj[p]
                                return None
                            # Use improved revenue selection with priority ordering
                            revenue_val, revenue_concept, revenue_confidence = _select_best_revenue(
                                items, periods, prefer_period
                            )
                            # Debug logging for revenue
                            if revenue_val:
                                print(f"[{ticker_upper}] Revenue: ${revenue_val:,.0f} from {revenue_concept}")
                            
                            # Prioritize weighted average shares for market cap calculation
                            shares_candidates_priority = [
                                'us-gaap_WeightedAverageNumberOfSharesOutstandingBasic',
                                'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding',
                                'us-gaap_CommonStockSharesOutstanding'  # Fallback to point-in-time
                            ]
                            if shares_val is None:
                                for concept in shares_candidates_priority:
                                    for it in items:
                                        c = str(it.get('concept', ''))
                                        if c == concept:
                                            v = pick_value(it.get('values') or {})
                                            if v is not None:
                                                shares_val = v
                                                break
                                    if shares_val is not None:
                                        break
                            # Fallback: try to find shares outstanding on balance sheet if still missing
                            if shares_val is None:
                                bs = (raw_payload or {}).get('balance_sheet') or {}
                                bs_items = bs.get('items') or []
                                bs_periods = bs.get('periods') or []
                                # Reuse pick_value with balance sheet periods
                                def pick_value_bs(map_obj):
                                    if not isinstance(map_obj, dict):
                                        return None
                                    if prefer_period and prefer_period in map_obj and map_obj[prefer_period] is not None:
                                        return map_obj[prefer_period]
                                    for p in bs_periods:
                                        if p in map_obj and map_obj[p] is not None:
                                            return map_obj[p]
                                    return None
                                for it in bs_items:
                                    c = str(it.get('concept', ''))
                                    if c in shares_candidates:
                                        v = pick_value_bs(it.get('values') or {})
                                        if v is not None:
                                            shares_val = v
                                            break
                        except Exception:
                            pass

                    # If we still don't have revenue from raw_data, fallback to top-level stored value
                    if revenue_val is None:
                        revenue_val = financials.get('revenue')

                    # Compute market cap and P/S with explicit None checks
                    warnings = []
                    if stock_price is not None and shares_val is not None:
                        try:
                            market_cap = float(stock_price) * float(shares_val)
                            print(f"[{ticker_upper}] Market cap: ${market_cap:,.0f} (price ${stock_price} × {shares_val:,.0f} shares)")
                        except Exception:
                            market_cap = None
                    else:
                        # Add warning for missing data
                        if stock_price is None:
                            warnings.append("Stock price not available")
                        if shares_val is None:
                            warnings.append("Shares outstanding not found in filings")
                    
                    if market_cap is not None and revenue_val is not None:
                        if revenue_val <= 0:
                            warnings.append("Revenue is zero or negative - P/S ratio not meaningful")
                            ps_ratio = None
                        else:
                            try:
                                ps_ratio = float(market_cap) / float(revenue_val)
                                # Sanity check for extreme ratios - likely unit mismatch
                                if ps_ratio > 1000:
                                    warnings.append(f"P/S ratio extremely high ({ps_ratio:.0f}x) - possible unit mismatch")
                                    # Try to detect and fix unit mismatch
                                    # If P/S > 1000, revenue might be in wrong units (e.g., actual dollars vs millions)
                                    if ps_ratio > 100000:  # Likely revenue in actual dollars, should be scaled
                                        revenue_val = revenue_val / 1000000  # Convert to millions
                                        ps_ratio = float(market_cap) / float(revenue_val)
                                        warnings.append(f"Auto-corrected revenue scale (now P/S = {ps_ratio:.2f}x)")
                            except Exception:
                                ps_ratio = None
                    elif revenue_val is None:
                        warnings.append("Revenue not found in filings")

                    # Try to compute additional ratios from stored raw_data metrics (if present)
                    try:
                        metrics = (raw_payload or {}).get('metrics') if raw_payload else None
                        # Note: metrics often contain per-period maps; leave PB/DE/PCF to frontend which
                        # selects appropriate period via /api/stored-metrics. Keep server values None.
                        _ = metrics  # placeholder to avoid linter warnings
                    except Exception:
                        pass
                    
                    row = {
                        "ticker": ticker_upper,
                        "company_name": financials['company_name'],
                        "stock_price": stock_price,
                        "market_cap": market_cap,
                        "revenue": revenue_val,
                        "ps_ratio": ps_ratio,
                        "shares_outstanding": shares_val,
                        "filing_date": financials['filing_date'],
                        "period_end_date": financials['period_end_date'],
                        "warnings": warnings if warnings else None
                    }
                    # Send company data
                    yield json.dumps({
                        "type": "company",
                        "data": row
                    }) + "\n"
                    cache_rows.append(row)
                    
                    processed += 1
                    
                    # Send progress update
                    yield json.dumps({
                        "type": "progress",
                        "processed": processed,
                        "total": len(tickers)
                    }) + "\n"
                    
                except Exception as e:
                    print(f"[STREAMING] Error processing {ticker_upper}: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    yield json.dumps({
                        "type": "error",
                        "ticker": ticker_upper,
                        "error": str(e)
                    }) + "\n"
                    processed += 1
            
            # Send completion
            print(f"[STREAMING] Completed! Processed {processed}/{len(tickers)} tickers")
            sys.stdout.flush()
            yield json.dumps({
                "type": "complete",
                "processed": processed,
                "total": len(tickers)
            }) + "\n"

            # Persist cache
            try:
                if cache_rows:
                    set_industry_cache(cache_key, cache_rows)
            except Exception:
                pass
            
        except Exception as e:
            print(f"[STREAMING] Fatal error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            yield json.dumps({
                "type": "error",
                "error": str(e)
            }) + "\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='application/x-ndjson',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )


@app.route("/api/industry-comparison", methods=["POST"])
def get_industry_comparison():
    """
    API endpoint to get financial comparison data for multiple tickers
    Expects JSON body with: {"tickers": ["AAPL", "MSFT", ...]}
    Only uses CACHED data - does not fetch fresh data from SEC Edgar
    """
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({
                "success": False,
                "error": "No tickers provided"
            }), 400
        
        results = []
        missing_tickers = []
        
        for ticker in tickers:
            ticker_upper = ticker.upper()
            
            try:
                # Try to use precomputed valuation metrics from Supabase first
                financials = get_company_financials(ticker_upper)
                
                if financials and financials.get('valuation_metrics'):
                    try:
                        valuation_metrics = json.loads(financials['valuation_metrics']) if isinstance(financials['valuation_metrics'], str) else financials['valuation_metrics']
                        
                        # Check if metrics are recent (< 24 hours old)
                        updated_at = valuation_metrics.get('updated_at')
                        if updated_at:
                            updated_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                            age = datetime.now() - updated_time.replace(tzinfo=None)
                            
                            if age < timedelta(hours=24):
                                # Use cached metrics
                                ticker_data = {
                                    "ticker": ticker_upper,
                                    "company_name": financials.get('company_name', ticker_upper),
                                    "stock_price": valuation_metrics.get('stock_price'),
                                    "market_cap": valuation_metrics.get('market_cap'),
                                    "revenue": valuation_metrics.get('revenue'),
                                    "ps_ratio": valuation_metrics.get('ps_ratio'),
                                    "pb_ratio": valuation_metrics.get('pb_ratio'),
                                    "pcf_ratio": valuation_metrics.get('pcf_ratio'),
                                    "pe_ratio": valuation_metrics.get('pe_ratio'),
                                    "de_ratio": valuation_metrics.get('de_ratio'),
                                    "shares_outstanding": valuation_metrics.get('shares_outstanding'),
                                    "cached": True,
                                    "cache_age_hours": age.total_seconds() / 3600
                                }
                                results.append(ticker_data)
                                continue
                    except Exception as e:
                        print(f"Error parsing cached metrics for {ticker_upper}: {e}")
                
                # Fallback: compute metrics on the fly (old behavior)
                # ONLY get cached income statement data - don't fetch fresh
                cached_income = get_cached_data(ticker_upper)
                
                if not cached_income:
                    missing_tickers.append(ticker_upper)
                    continue
                
                # Get cached stock price
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data, cached_at FROM stock_price_cache WHERE ticker = ?",
                    (ticker_upper,)
                )
                price_result = cursor.fetchone()
                conn.close()
                
                stock_price = None
                if price_result:
                    price_data, cached_at = price_result
                    cached_time = datetime.fromisoformat(cached_at)
                    if datetime.now() - cached_time < timedelta(minutes=STOCK_PRICE_CACHE_MINUTES):
                        stock_price = json.loads(price_data).get('price')
                
                # If no cached price, try to fetch it (this is fast)
                if not stock_price and alpaca_client and check_alpaca_rate_limit():
                    try:
                        request_params = StockLatestTradeRequest(symbol_or_symbols=ticker_upper)
                        trades = alpaca_client.get_stock_latest_trade(request_params)
                        if ticker_upper in trades:
                            stock_price = float(trades[ticker_upper].price)
                            
                            # Cache it
                            result_data = {
                                "ticker": ticker_upper,
                                "price": stock_price,
                                "size": float(trades[ticker_upper].size),
                                "timestamp": trades[ticker_upper].timestamp.isoformat(),
                                "exchange": trades[ticker_upper].exchange
                            }
                            conn = sqlite3.connect(DB_PATH)
                            cursor = conn.cursor()
                            cursor.execute(
                                "INSERT OR REPLACE INTO stock_price_cache (ticker, data, cached_at) VALUES (?, ?, ?)",
                                (ticker_upper, json.dumps(result_data), datetime.now().isoformat())
                            )
                            conn.commit()
                            conn.close()
                    except:
                        pass
                
                # Extract key metrics from cached data
                ticker_data = {
                    "ticker": ticker_upper,
                    "company_name": cached_income.get("company_name", ticker_upper),
                    "stock_price": stock_price,
                    "market_cap": None,
                    "revenue": None,
                    "ps_ratio": None,
                    "cached": False
                }
                
                # Get most recent ANNUAL period (10-K) if available
                annual_period = _select_preferred_period(
                    cached_income.get("periods", []),
                    cached_income.get("period_metadata"),
                    prefer='10-K'
                )
                
                # Use shared revenue extraction logic
                revenue_value = None
                if annual_period and cached_income.get("items"):
                    revenue_value, _, _ = _select_best_revenue(
                        cached_income.get("items", []),
                        cached_income.get("periods", []),
                        prefer_period=annual_period
                    )
                
                # Find first shares item with a non-null value for the annual period
                shares_candidates = [
                    "us-gaap_WeightedAverageNumberOfSharesOutstandingBasic",
                    "us-gaap_CommonStockSharesOutstanding",
                    "us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding"
                ]
                
                # Find first shares item with a non-null value for the annual period
                shares_value = None
                if annual_period:
                    for item in cached_income.get("items", []):
                        if item["concept"] in shares_candidates:
                            val = item["values"].get(annual_period)
                            if val is not None:
                                shares_value = val
                                break  # Stop at first non-null shares
                
                # Set revenue
                ticker_data["revenue"] = revenue_value
                
                # Calculate market cap and P/S ratio
                if shares_value and stock_price:
                    ticker_data["market_cap"] = stock_price * shares_value
                    
                    # Calculate P/S ratio
                    if revenue_value:
                        ticker_data["ps_ratio"] = ticker_data["market_cap"] / revenue_value
                
                results.append(ticker_data)
            
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        return jsonify({
            "success": True,
            "data": results,
            "missing_tickers": missing_tickers,
            "message": f"Loaded {len(results)} companies from cache. {len(missing_tickers)} companies need to be cached first." if missing_tickers else None
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/cache-tickers", methods=["POST"])
def cache_tickers():
    """
    API endpoint to pre-cache financial data for multiple tickers
    Expects JSON body with: {"tickers": ["AAPL", "MSFT", ...]}
    Only fetches 10-K (annual) data for efficiency
    """
    try:
        data = request.get_json()
        tickers = data.get('tickers', [])
        
        if not tickers:
            return jsonify({
                "success": False,
                "error": "No tickers provided"
            }), 400
        
        results = {
            "cached": [],
            "failed": [],
            "already_cached": []
        }
        
        for ticker in tickers:
            ticker_upper = ticker.upper()
            
            try:
                # Check if already cached
                cached = get_cached_data(ticker_upper)
                if cached:
                    results["already_cached"].append(ticker_upper)
                    continue
                
                # Fetch only 10-K (annual) data
                c = Company(ticker_upper)
                filing_10k = c.get_filings(form="10-K").latest(1)
                
                if not filing_10k:
                    results["failed"].append({"ticker": ticker_upper, "reason": "No 10-K filing found"})
                    continue
                
                # Get XBRL data
                xbs = XBRLS.from_filings(filing_10k)
                income_statement = xbs.statements.income_statement()
                income_df = income_statement.to_dataframe()
                
                periods_list = list(income_statement.periods)
                
                # Get the period date
                filing_types = {}
                try:
                    period_10k = filing_10k[0].period_of_report
                    filing_types[str(period_10k)] = "10-K"
                except:
                    pass
                
                # Sort periods
                sorted_periods = sorted(periods_list, key=lambda x: x, reverse=True)
                
                # Convert to result format
                result_data = []
                for _, row in income_df.iterrows():
                    item = {
                        "label": row["label"],
                        "concept": row["concept"],
                        "values": {}
                    }
                    for period in sorted_periods:
                        if period in row:
                            item["values"][period] = float(row[period]) if row[period] else None
                    result_data.append(item)
                
                period_metadata = []
                for period in sorted_periods:
                    period_metadata.append({
                        "date": period,
                        "type": filing_types.get(period, "10-K")
                    })
                
                response_data = {
                    "ticker": ticker_upper,
                    "company_name": c.name,
                    "periods": sorted_periods,
                    "period_metadata": period_metadata,
                    "items": result_data
                }
                
                # Cache it
                cache_data(ticker_upper, response_data)
                results["cached"].append(ticker_upper)
                
            except Exception as e:
                results["failed"].append({"ticker": ticker_upper, "reason": str(e)})
                print(f"Error caching {ticker_upper}: {e}")
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": f"Cached: {len(results['cached'])}, Already cached: {len(results['already_cached'])}, Failed: {len(results['failed'])}"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})


@app.route("/", methods=["GET"])
def index():
    """Serve the frontend HTML"""
    return send_file(os.path.join(FRONTEND_DIR, 'index.html'))


@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    """Serve static files from frontend directory"""
    try:
        return send_file(os.path.join(FRONTEND_DIR, filename))
    except FileNotFoundError:
        return "File not found", 404


if __name__ == "__main__":
    init_db()
    
    # Get port from environment variable (for Railway/Render) or default to 5001
    port = int(os.environ.get("PORT", 5001))
    
    print("\n" + "="*60)
    print("🚀 Valuation Tool")
    print("="*60)
    print(f"📊 Running on port {port}")
    print("="*60 + "\n")
    
    # Use 0.0.0.0 to accept external connections (required for Railway/Render)
    app.run(debug=False, port=port, host='0.0.0.0', threaded=True, use_reloader=False)

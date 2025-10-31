#!/usr/bin/env python3
"""
Script to import financial data from CSV into Supabase database
Processes S&P 500 companies and fetches their latest 10-K data
Includes comprehensive validation and data quality checks
"""

import csv
import sys
import time
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from edgar import Company, set_identity
from edgar.xbrl import XBRLS
from supabase_db import (
    store_company_financials,
    check_ticker_exists,
    get_all_tickers_with_financials,
    is_supabase_configured
)
from scale_normalizer import normalize_financial_data


# Load environment variables
load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

# Set Edgar identity
set_identity("matthildur@montageventures.com")

# Expanded list of revenue concepts to check
REVENUE_CONCEPTS = [
    'us-gaap_Revenues',
    'us-gaap_SalesRevenueNet',
    'us-gaap_TotalRevenues',
    'us-gaap_OperatingRevenue',
    'us-gaap_Revenue',
    'us-gaap_NetSales',
    'Revenues',
    'Revenue'
]

# Expanded list of share concepts to check (in priority order)
SHARE_CONCEPTS = [
    'us-gaap_WeightedAverageNumberOfSharesOutstandingBasic',
    'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding',
    'us-gaap_CommonStockSharesOutstanding',
    'us-gaap_CommonSharesOutstanding',
    'us-gaap_WeightedAverageSharesOutstanding',
    'us-gaap_SharesOutstanding',
]

# Expanded list of EPS concepts (in priority order)
EPS_CONCEPTS = [
    'us-gaap_EarningsPerShareBasic',
    'us-gaap_EarningsPerShareBasicAndDiluted',
    'us-gaap_EarningsPerShareBasicFromContinuingOperations',
]


def find_revenue_row(df):
    """
    Find revenue row using fuzzy matching on labels and concepts.
    Some filers use custom labels like "Net sales" or "Total operating revenue."
    """
    if df is None:
        return None
    
    for _, row in df.iterrows():
        label = (row.get("label") or "").lower()
        concept = (row.get("concept") or "").lower()
        
        # Check if this looks like a total revenue line
        if any(k in label for k in ["revenue", "net sales", "sales", "turnover"]) \
           or any(k in concept for k in ["revenue", "sales"]):
            # Exclude segment/component revenues
            if not any(excl in label for excl in ["segment", "product", "service", "geography"]):
                return row
    return None


# Try to import Alpaca for price validation (optional)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
    if alpaca_api_key and alpaca_secret_key:
        alpaca_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
        print("✓ Alpaca API configured for price validation")
    else:
        alpaca_client = None
        print("⚠ Alpaca API not configured - skipping price validation")
except ImportError:
    alpaca_client = None
    print("⚠ Alpaca SDK not installed - skipping price validation")

CSV_PATH = Path(__file__).parent.parent / "data" / "Valuation Comps - S&P500.csv"

# Data quality tracking
class DataQuality:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
    
    def add_issue(self, msg):
        self.issues.append(msg)
    
    def add_warning(self, msg):
        self.warnings.append(msg)
    
    def add_info(self, msg):
        self.info.append(msg)
    
    def has_critical_issues(self):
        return len(self.issues) > 0
    
    def get_summary(self):
        parts = []
        if self.issues:
            parts.append(f"❌ {len(self.issues)} critical")
        if self.warnings:
            parts.append(f"⚠️  {len(self.warnings)} warnings")
        if self.info:
            parts.append(f"ℹ️  {len(self.info)} info")
        return " | ".join(parts) if parts else "✓ All checks passed"
    
    def get_details(self):
        lines = []
        if self.issues:
            lines.append("  CRITICAL ISSUES:")
            for issue in self.issues:
                lines.append(f"    ❌ {issue}")
        if self.warnings:
            lines.append("  WARNINGS:")
            for warning in self.warnings:
                lines.append(f"    ⚠️  {warning}")
        if self.info:
            lines.append("  INFO:")
            for info in self.info:
                lines.append(f"    ℹ️  {info}")
        return "\n".join(lines)


def validate_financial_data(ticker: str, data: Dict, dq: DataQuality) -> bool:
    """
    Validate extracted financial data and populate DataQuality object.
    Returns True if data is acceptable, False if critical issues found.
    """
    # Check revenue
    if data.get('revenue') is None:
        dq.add_issue("Revenue not found in income statement")
    elif data['revenue'] <= 0:
        dq.add_issue(f"Revenue is invalid: ${data['revenue']:,.0f}")
    else:
        dq.add_info(f"Revenue: ${data['revenue']:,.0f}")
    
    # Check shares outstanding
    if data.get('shares') is None:
        dq.add_issue("Shares outstanding not found")
    elif data['shares'] <= 0:
        dq.add_issue(f"Shares outstanding is invalid: {data['shares']:,.0f}")
    else:
        dq.add_info(f"Shares: {data['shares']:,.0f}")
    
    # Validate with current stock price if Alpaca available
    if alpaca_client and data.get('shares'):
        try:
            request_params = StockLatestTradeRequest(symbol_or_symbols=ticker.upper())
            trades = alpaca_client.get_stock_latest_trade(request_params)
            if ticker.upper() in trades:
                price = float(trades[ticker.upper()].price)
                implied_market_cap = price * data['shares']
                dq.add_info(f"Current price: ${price:.2f} → Market cap: ${implied_market_cap:,.0f}")
                
                # Sanity check: market cap should be reasonable
                if implied_market_cap < 1_000_000:  # < $1M
                    dq.add_warning(f"Market cap seems too low (${implied_market_cap:,.0f}) - check shares scale")
                elif implied_market_cap > 10_000_000_000_000:  # > $10T
                    dq.add_warning(f"Market cap seems too high (${implied_market_cap:,.0f}) - check shares scale")
        except Exception as e:
            dq.add_info(f"Could not validate with current price: {str(e)}")
    
    # Check statement availability (raw_data contains DataFrames)
    raw_data = data.get('raw_data', {})
    
    income_df = raw_data.get('income_statement')
    balance_df = raw_data.get('balance_sheet')
    cashflow_df = raw_data.get('cash_flow')
    
    if income_df is None or (hasattr(income_df, 'empty') and income_df.empty):
        dq.add_issue("Income statement is empty")
    else:
        dq.add_info("Income statement available")
    
    if balance_df is None or (hasattr(balance_df, 'empty') and balance_df.empty):
        dq.add_warning("Balance sheet is empty or missing")
    else:
        dq.add_info("Balance sheet available")
    
    if cashflow_df is None or (hasattr(cashflow_df, 'empty') and cashflow_df.empty):
        dq.add_warning("Cash flow statement is empty or missing")
    else:
        dq.add_info("Cash flow statement available")
    
    # Return False only if critical issues (revenue or shares missing)
    return not dq.has_critical_issues()


def get_financial_data(ticker):
    """Fetch latest 10-K + last 4 10-Q filings for TTM calculations (using label-based detection)."""

    try:
        c = Company(ticker)

        # --- Fetch filings ---
        annual_filing = c.get_filings(form="10-K").latest(1)
        quarterly_filings = c.get_filings(form="10-Q").latest(4)

        if not annual_filing and not quarterly_filings:
            print(f"No filings found for {ticker}")
            return None

        # --- Combine filings & metadata ---
        all_filings, filing_metadata = [], []
        for ftype, filing in [("10-K", annual_filing)] + [("10-Q", f) for f in (quarterly_filings or [])]:
            if filing:
                all_filings.append(filing)
                filing_metadata.append({
                    "type": ftype,
                    "filing_date": str(getattr(filing, "filing_date", None)),
                    "period_end": str(getattr(filing, "period_of_report", None)),
                })

        # --- Parse XBRL ---
        xbs = XBRLS.from_filings(all_filings)
        income_df = xbs.statements.income_statement().to_dataframe()
        balance_df = getattr(xbs.statements, "balance_sheet", lambda: None)()
        cashflow_df = getattr(xbs.statements, "cash_flow_statement", lambda: None)()
        balance_df = balance_df.to_dataframe() if balance_df else None
        cashflow_df = cashflow_df.to_dataframe() if cashflow_df else None

        # --- Determine most recent filing ---
        most_recent = sorted(filing_metadata, key=lambda x: x["filing_date"], reverse=True)[0]
        latest_type, period_end, filing_date = most_recent["type"], most_recent["period_end"], most_recent["filing_date"]

        # --- Normalize periods (after we have period_end) ---
        def normalize_date(d):
            if isinstance(d, str) and re.match(r"\d{4}-\d{2}(-\d{2})?$", d):
                return d[:7]  # keep YYYY-MM
            return d

        income_df.columns = [normalize_date(c) for c in income_df.columns]
        period_end = normalize_date(period_end)
        period_types = {
            normalize_date(meta["period_end"]): meta["type"]
            for meta in filing_metadata if meta.get("period_end")
        }

        # --- Helper: match label keywords robustly ---
        def get_value_from_label(df, label_keywords=None, exclude_keywords=None, period=None, ttm=False):
            """Improved label-based extractor (safe, loss-aware, period-agnostic, no concept dependency)."""

            import math
            import pandas as pd

            if df is None or df.empty:
                return None, None

            label_keywords = [k.lower() for k in (label_keywords or ["revenue", "sales", "contract"])]
            exclude_keywords = [k.lower() for k in (exclude_keywords or ["cost", "gross", "total cost", "per share"])]

            # --- Step 1: find matching rows ---
            def match_label(label):
                if not isinstance(label, str):
                    return False
                text = label.lower()
                return any(k in text for k in label_keywords) and not any(ex in text for ex in exclude_keywords)

            matches = df[df["label"].apply(match_label)]
            if matches.empty:
                return None, None

            # --- Step 2: pick the best-scoring row ---
            def score_label(lbl):
                l = lbl.lower()
                score = 0

                # strong positives
                if l.strip() in ("net income", "net income (loss)"):
                    score += 20
                if "net income (loss)" in l or "(loss)" in l and "net income" in l:
                    score += 12
                if "net income" in l:
                    score += 10
                if "profit or loss" in l:
                    score += 8
                if "profit" in l and "gross" not in l:
                    score += 6
                if "total revenue" in l or l.strip() == "revenue":
                    score += 8
                if "contract revenue" in l:
                    score += 6

                # mild positives
                if "continuing operations" in l:
                    score += 2
                if "from continuing operations" in l:
                    score += 1
                if "attributable to" in l and "parent" in l:
                    score += 4

                # penalties
                if "operating" in l:
                    score -= 5
                if "subsidiaries" in l:
                    score -= 5
                if "comprehensive" in l:
                    score -= 5
                if "before tax" in l:
                    score -= 3
                if "noncontrolling" in l:
                    score -= 2
                if "attributable to" in l and "parent" not in l:
                    score -= 2
                if "diluted" in l:
                    score += 2
                if "basic" in l:
                    score -= 1

                return score

            matches = matches.copy()
            matches["score"] = matches["label"].apply(score_label)
            matches = matches.sort_values("score", ascending=False)

            # --- Step 2.5: skip rows that are all NaN ---
            def row_has_numeric_data(row):
                for c in df.columns:
                    if c in ("label", "concept"):
                        continue
                    val = row.get(c)
                    if isinstance(val, (int, float)) and not math.isnan(val):
                        return True
                return False

            # pick first row with usable data
            row = None
            for _, candidate in matches.iterrows():
                if row_has_numeric_data(candidate):
                    row = candidate
                    break
            if row is None:
                row = matches.iloc[0]

            label = row["label"]

            # --- Step 3: safe period lookup ---
            def get_period_value(row, period):
                if period in row.index:
                    return row.get(period)
                # fuzzy match (same year-month)
                for col in row.index:
                    if isinstance(col, str) and period and col[:7] == period[:7]:
                        return row.get(col)
                # fallback to latest numeric column
                numeric_cols = [
                    c for c in row.index
                    if c not in ("label", "concept")
                    and isinstance(row.get(c), (int, float))
                    and not math.isnan(row.get(c))
                ]
                if numeric_cols:
                    return row.get(numeric_cols[0])
                return None

            # --- Step 4: extract the number ---
            if not ttm:
                v = get_period_value(row, period)
                if isinstance(v, str):
                    try:
                        v = float(v.replace(",", ""))
                    except:
                        v = None
                return (float(v) if isinstance(v, (int, float)) and not math.isnan(v) else None), label

            # --- Step 5: TTM aggregation (sum last 4 true quarterly values only; skip annual 10-K) ---
            vals = []

            def is_quarterly_period(p: str) -> bool:
                """Return True only if the given period corresponds to a quarterly (10-Q) report."""
                if not isinstance(p, str):
                    return False

                # Use known filing metadata if available
                ftype = period_types.get(p)
                if ftype:
                    if "10-Q" in ftype:
                        return True
                    if "10-K" in ftype:
                        return False

                # Fuzzy match by month to known filing periods
                for meta_date, meta_type in period_types.items():
                    if meta_date and meta_date[:7] == p[:7]:
                        return "10-Q" in meta_type  # only treat as quarterly if a matching 10-Q exists

                # Fallback heuristic for filers with normalized YYYY-MM only
                # Allow all quarter ends (03, 06, 09, 12)
                # but treat "12" as quarterly *only* if we don't already have a 10-K for that month
                month = p[-2:]
                if month in ("03", "06", "09"):
                    return True
                if month == "12":
                    # Check if this December period also has a 10-K entry — if so, skip it
                    has_annual = any(
                        ("10-K" in meta_type and meta_date.endswith("-12"))
                        for meta_date, meta_type in period_types.items()
                    )
                    return not has_annual
                return False


            # Collect quarterly columns only
            period_cols = [
                p for p in df.columns
                if p not in ("label", "concept") and is_quarterly_period(str(p))
            ]

            # If no clear quarterlies found, fall back to date-like columns
            if not period_cols:
                period_cols = [
                    p for p in df.columns
                    if p not in ("label", "concept") and re.match(r"\d{4}-\d{2}$", str(p))
                ]

            # Sort most recent first
            try:
                period_cols = sorted(period_cols, reverse=True)
            except Exception:
                pass

            # Sum up to 4 quarterly values (most recent first)
            for p in period_cols:
                val = row.get(p)
                if val in (None, "", 0) or (isinstance(val, float) and math.isnan(val)):
                    continue
                if isinstance(val, str):
                    try:
                        val = float(val.replace(",", ""))
                    except Exception:
                        continue
                vals.append(float(val))
                if len(vals) >= 4:
                    break

            # Fallback: use most recent annual (10-K) if no quarterly data found
            if not vals:
                for p in reversed(period_cols):
                    ftype = period_types.get(p, "")
                    if "10-K" in ftype:
                        val = row.get(p)
                        if isinstance(val, (int, float)) and not math.isnan(val):
                            vals = [float(val)]
                            break

            return (sum(vals) if vals else None), label


        # --- Core metric extractor (for both 10-K and 10-Q) ---
        def extract_financial_metrics(ttm=False):
            revenue, revenue_label = get_value_from_label(
                income_df,
                label_keywords=["revenue", "sales", "contract", "revenues"],
                exclude_keywords=["cost", "gross", "total cost", "per share"],
                ttm=ttm
            )

            # --- Net Income ---
            income, income_label = get_value_from_label(
                income_df,
                label_keywords=["net income", "income", "profit"],
                exclude_keywords=["attributable to", "per share", "before tax", "comprehensive"],
                ttm=ttm
            )
            eps, eps_label = get_value_from_label(
                income_df,
                label_keywords=["earnings per share", "eps", "in usd per share", "in dollars per share", "basic", "diluted"],
                exclude_keywords=["continuing operations", "discontinued operations", "available to common shareholders", "net income"],
                period=period_end,
                ttm=False
            )
            if not eps:
                eps, eps_label = get_value_from_label(
                    income_df,
                    label_keywords=["earnings per share", "eps"],
                    exclude_keywords=[],
                    period=period_end,
                    ttm=False
                )
            shares, shares_label = get_value_from_label(
                income_df,
                label_keywords=[
                    "weighted average",
                    "shares outstanding",
                    "number of shares",
                    "average shares",
                    "common shares",
                    "diluted shares",
                    "shares",
                    "weighted average",
                    "outstanding"
                ],
                exclude_keywords=["basic"],
                period=period_end,
                ttm=False
            )
            if not shares:
                shares, shares_label = get_value_from_label(
                    income_df,
                    label_keywords=["shares outstanding"],
                    exclude_keywords=[],
                    period=period_end,
                    ttm=False
                )
            # Fallback 2: Compute shares using SAME-PERIOD income if possible
            if not shares and eps:
                # Try to get same-period net income (not TTM) from the chosen net income row
                same_period_income = None
                # Reuse the winning income row by rerunning a non-TTM fetch that returns the period value
                same_period_income, _ = get_value_from_label(
                    income_df,
                    label_keywords=["net income", "profit"],
                    exclude_keywords=["attributable to", "per share", "before tax", "comprehensive"],
                    period=period_end,
                    ttm=False
                )
                if same_period_income and same_period_income != 0:
                    est = same_period_income / eps
                    if est > 1e5:
                        shares = est
                        shares_label = "Estimated (Same-period Net Income / EPS)"

            # Fallback 3: last resort from TTM income and EPS×4
            if not shares and eps and income:
                est = income / (eps * (4 if latest_type != "10-K" else 1))
                if est > 1e5:
                    shares = est
                    shares_label = "Estimated (TTM Net Income / (EPS×4))"


            return revenue, revenue_label, income, income_label, eps, eps_label, shares, shares_label

        # --- Compute metrics (one call only) ---
        if latest_type == "10-K":
            revenue, revenue_label, income, income_label, eps, eps_label, shares, shares_label = extract_financial_metrics(ttm=False)
            print(f"✓ {ticker}: Latest filing is annual (10-K). Using reported annual values.")
        else:
            revenue, revenue_label, income, income_label, eps, eps_label, shares, shares_label = extract_financial_metrics(ttm=True)
            print(f"✓ {ticker}: Latest filing is quarterly. Using last 4 quarters for TTM.")

        # --- Print summary ---
        print(f"   Revenue line: '{revenue_label}' → ${revenue:,.0f}" if revenue else "   ⚠️ No revenue line detected.")
        print(f"   Net Income line: '{income_label}' → ${income:,.0f}" if income else "   ⚠️ No income line detected.")
        print(f"   EPS line: '{eps_label}' → ${eps:,.2f}" if eps else "   ⚠️ No EPS line detected.")
        print(f"   Shares line: '{shares_label}' → {shares:,.0f}" if shares else "   ⚠️ No shares line detected.")

        # --- Return structured output ---
        return {
            "company_name": c.name,
            "revenue": revenue,
            "income": income,
            "revenue_label": revenue_label,
            "income_label": income_label,
            "shares": shares,
            "latest_eps": eps,
            "filing_date": filing_date,
            "period_end": period_end,
            "raw_data": {
                "income_statement": income_df,
                "balance_sheet": balance_df,
                "cash_flow": cashflow_df,
                "filing_metadata": filing_metadata,
            },
        }

    except Exception as e:
        print(f"  Error for {ticker}: {e}")
        return None


def store_financial_data(ticker, data):
    """Store financial data in Supabase database"""
    # Convert raw_data dict to JSON string (handle DataFrames)
    raw_data_json = None
    if data.get('raw_data'):
        raw_data = data['raw_data']
        serializable_data = {
            'income_statement': raw_data.get('income_statement').to_dict() if raw_data.get('income_statement') is not None else None,
            'balance_sheet': raw_data.get('balance_sheet').to_dict() if raw_data.get('balance_sheet') is not None else None,
            'cash_flow': raw_data.get('cash_flow').to_dict() if raw_data.get('cash_flow') is not None else None,
            'filing_metadata': raw_data.get('filing_metadata', [])
        }
        raw_data_json = json.dumps(serializable_data)
    
    # Determine filing type from most recent period
    filing_type = '10-K/10-Q'  # Mixed filings
    if data.get('raw_data') and data['raw_data'].get('filing_metadata'):
        # Use the type of the most recent period
        filing_type = data['raw_data']['filing_metadata'][0].get('type', '10-K/10-Q')
    
    return store_company_financials(
        ticker.upper(),
        data['company_name'],
        data['revenue'],
        data['shares'],
        data['filing_date'],
        data['period_end'],
        filing_type,
        raw_data_json,
        income=data.get('income'),
        latest_eps=data.get('latest_eps'),
        revenue_label=data.get('revenue_label'),
        income_label=data.get('income_label')
    )


def init_database():
    """Check Supabase connection"""
    if not is_supabase_configured():
        print("\n❌ ERROR: Supabase is not configured!")
        print("Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")
        print("See .env.example for details.\n")
        sys.exit(1)
    print("✓ Supabase connection configured\n")

def normalize_financial_data(data: dict) -> dict:
    """
    Minimal normalization step.
    Keeps raw_data and company_name unchanged.
    Ensures top-level numeric fields are floats and safe for JSON serialization.
    """

    if not data or not isinstance(data, dict):
        return {}

    # Always preserve these fields
    normalized = {
        "company_name": data.get("company_name"),
        "raw_data": data.get("raw_data"),
    }

    # Normalize numeric fields if present
    numeric_fields = ["revenue", "income", "shares", "latest_eps"]
    for field in numeric_fields:
        val = data.get(field)
        try:
            normalized[field] = float(val) if val not in (None, "", "NaN") else None
        except (TypeError, ValueError):
            normalized[field] = None

    # Preserve metadata fields (safe strings only)
    for meta in ["revenue_label", "income_label", "filing_date", "period_end"]:
        normalized[meta] = data.get(meta)

    return normalized


def main():
    print("\n" + "="*60)
    print("📊 Financial Data Import Tool")
    print("="*60)
    print("\nThis will fetch latest 10-K + last 4 10-Q filings")
    print("for S&P 500 companies and store in the database.")
    
    # Initialize database
    init_database()
    
    # Load tickers from CSV
    tickers = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers.append(row['Symbol'])
    
    print(f"Found {len(tickers)} companies in CSV\n")
    
    # Check which are already in database
    existing = set(get_all_tickers_with_financials())
    
    print(f"Already in database: {len(existing)} companies")
    print(f"Need to fetch: {len(tickers) - len(existing)} companies\n")
    
    # Ask if user wants to force re-import
    force_update = False
    if existing:
        response = input("Re-import existing companies to update raw_data? (y/n): ")
        if response.lower() == 'y':
            force_update = True
            print("Will re-import ALL companies (including existing ones)\n")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Process tickers with detailed validation
    success_count = 0
    skip_count = 0
    fail_count = 0
    warning_count = 0
    
    # Track data quality issues
    quality_report = {
        'missing_revenue': [],
        'missing_shares': [],
        'missing_equity': [],
        'missing_debt': [],
        'missing_ocf': [],
        'market_cap_issues': []
    }
    
    # Track failed tickers with details
    failed_tickers = []
    critical_issue_tickers = []
    
    # Ask if user wants verbose output
    verbose = input("\nShow detailed validation for each company? (y/n): ").lower() == 'y'
    print()
    
    for i, ticker in enumerate(tickers, 1):
        ticker_upper = ticker.upper()
        
        # Skip if already exists (unless force_update is True)
        if ticker_upper in existing and not force_update:
            print(f"[{i}/{len(tickers)}] {ticker_upper}: Already in database")
            skip_count += 1
            continue
        
        status = "Updating" if ticker_upper in existing else "Fetching"
        print(f"[{i}/{len(tickers)}] {ticker_upper}: {status}...", end=" ")
        sys.stdout.flush()
        
        data = get_financial_data(ticker_upper)
        if not data:
            print("✗ Failed to fetch data")
            failed_tickers.append(ticker_upper)
            fail_count += 1
            time.sleep(0.5)
            continue

        # Normalize to real dollars
        data = normalize_financial_data(data)

        # Validate for completeness / sanity
        dq = DataQuality()
        is_valid = validate_financial_data(ticker_upper, data, dq)

        if is_valid:
            store_financial_data(ticker_upper, data)
            print(f"✓ {ticker_upper}: Stored ({dq.get_summary()})")
            success_count += 1
        else:
            print(f"✗ {ticker_upper}: Critical data issues, not stored")
            print(dq.get_details())
            critical_issue_tickers.append({'ticker': ticker_upper, **dq.__dict__})
            fail_count += 1

        time.sleep(0.5)

    print("\n" + "="*80)
    print("Import Complete - Summary")
    print("="*80)
    print(f"\n📊 IMPORT STATISTICS:")
    print(f"  ✓ Successfully imported: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"  ⏭️  Skipped (already in DB): {skip_count}")
    print(f"  📁 Total processed: {success_count + skip_count + fail_count}")


if __name__ == "__main__":
    main()

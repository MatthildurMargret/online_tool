#!/usr/bin/env python3
"""
Script to import financial data from CSV into Supabase database
Processes S&P 500 companies and fetches their latest 10-K data
Includes comprehensive validation and data quality checks
"""

import csv
import sys
import time
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
from revenue_extractor import extract_revenue
from scale_normalizer import normalize_financial_data, validate_normalization

# Load environment variables
load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

# Set Edgar identity
set_identity("matthildur@montageventures.com")

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
    
    # Check balance sheet metrics
    raw_data = data.get('raw_data', {})
    metrics = raw_data.get('metrics', {})
    
    if not metrics.get('equity'):
        dq.add_warning("Equity not found in balance sheet")
    else:
        dq.add_info(f"Equity found for {len(metrics['equity'])} periods")
    
    if not metrics.get('debt'):
        dq.add_warning("Debt not found in balance sheet")
    else:
        dq.add_info(f"Debt found for {len(metrics['debt'])} periods")
    
    if not metrics.get('operating_cash_flow'):
        dq.add_warning("Operating cash flow not found")
    else:
        dq.add_info(f"OCF found for {len(metrics['operating_cash_flow'])} periods")
    
    # Check net income
    if not metrics.get('net_income'):
        dq.add_warning("Net income not found in income statement")
    else:
        dq.add_info(f"Net income found for {len(metrics['net_income'])} periods")
    
    # Check EPS
    if not metrics.get('eps_basic') and not metrics.get('eps_diluted'):
        dq.add_warning("EPS (basic/diluted) not found")
    else:
        if metrics.get('eps_basic'):
            dq.add_info(f"EPS Basic found for {len(metrics['eps_basic'])} periods")
        if metrics.get('eps_diluted'):
            dq.add_info(f"EPS Diluted found for {len(metrics['eps_diluted'])} periods")
    
    # Check statement availability
    if not raw_data.get('income_statement', {}).get('items'):
        dq.add_issue("Income statement is empty")
    
    if not raw_data.get('balance_sheet', {}).get('items'):
        dq.add_warning("Balance sheet is empty or missing")
    
    if not raw_data.get('cash_flow', {}).get('items'):
        dq.add_warning("Cash flow statement is empty or missing")
    
    # Check period count
    periods = raw_data.get('income_statement', {}).get('periods', [])
    if len(periods) < 2:
        dq.add_warning(f"Only {len(periods)} period(s) available - may limit analysis")
    else:
        dq.add_info(f"{len(periods)} periods available")
    
    # Return False only if critical issues (revenue or shares missing)
    return not dq.has_critical_issues()


def get_financial_data(ticker):
    """Fetch latest 10-K + last 4 10-Q filings for TTM calculations"""
    try:
        c = Company(ticker)
        
        # Fetch latest annual (10-K) and last 4 quarterly (10-Q) filings
        annual_filing = c.get_filings(form="10-K").latest(1)
        quarterly_filing_list = c.get_filings(form="10-Q").latest(4)
        
        if not annual_filing:
            return None
        
        # Combine filings - annual first, then quarters (most recent first)
        all_filings = []
        filing_metadata = []
        
        # Add annual filing (latest returns a single filing, not a list)
        all_filings.append(annual_filing)
        filing_metadata.append({
            'type': '10-K',
            'filing_date': str(annual_filing.filing_date) if hasattr(annual_filing, 'filing_date') else None,
            'period_end': None  # Will be extracted from XBRL
        })
        
        # Add quarterly filings (latest(4) returns a Filings object that is iterable)
        if quarterly_filing_list:
            for filing in quarterly_filing_list:
                all_filings.append(filing)
                filing_metadata.append({
                    'type': '10-Q',
                    'filing_date': str(filing.filing_date) if hasattr(filing, 'filing_date') else None,
                    'period_end': None
                })
        
        if not all_filings:
            return None
        
        # Get XBRL data from all filings
        xbs = XBRLS.from_filings(all_filings)
        income_statement = xbs.statements.income_statement()
        income_df = income_statement.to_dataframe()

        # Try to also get balance sheet and cash flow
        try:
            balance_sheet = xbs.statements.balance_sheet()
            balance_df = balance_sheet.to_dataframe()
        except Exception:
            balance_sheet = None
            balance_df = None
        try:
            cash_flow = xbs.statements.cash_flow_statement()
            cashflow_df = cash_flow.to_dataframe()
        except Exception:
            cash_flow = None
            cashflow_df = None

        # Extract key metrics using shared revenue extraction logic
        revenue = None
        shares = None
        
        periods = list(income_statement.periods)
        latest_period = sorted(periods, key=lambda x: x, reverse=True)[0] if periods else None
        
        if latest_period:
            # Convert income_df to items format for revenue_extractor
            items = []
            for _, row in income_df.iterrows():
                item = {
                    "label": row.get("label", ""),
                    "concept": row.get("concept", ""),
                    "values": {}
                }
                for period in periods:
                    if period in row:
                        val = row[period]
                        item["values"][str(period)] = float(val) if val is not None and val != '' else None
                items.append(item)
            
            # Get cost of revenue for plausibility check
            cor = None
            for _, row in income_df.iterrows():
                if row.get('concept') == 'us-gaap_CostOfRevenue':
                    value = row.get(latest_period)
                    if value:
                        cor = abs(float(value))
                        break
            
            # Use shared revenue extraction logic with new signature
            revenue, revenue_metadata = extract_revenue(
                items,
                [str(p) for p in periods],
                prefer_period=str(latest_period),
                cost_of_revenue=cor
            )
            
            # Shares: Use priority order (same as backend)
            shares_priority = [
                'us-gaap_WeightedAverageNumberOfSharesOutstandingBasic',
                'us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding',
                'us-gaap_CommonStockSharesOutstanding'
            ]
            for concept in shares_priority:
                for _, row in income_df.iterrows():
                    if row.get('concept', '') == concept:
                        if latest_period in row and row[latest_period]:
                            shares = float(row[latest_period])
                            break
                if shares is not None:
                    break
        
        # If shares not found in income statement, try balance sheet with priority order
        if not shares and balance_df is not None:
            periods = list(balance_sheet.periods) if balance_sheet else []
            if periods:
                latest_period = sorted(periods, key=lambda x: x, reverse=True)[0]
                
                # Try direct shares outstanding concepts first
                for concept in shares_priority:
                    for _, row in balance_df.iterrows():
                        if row.get('concept', '') == concept:
                            if latest_period in row and row[latest_period]:
                                shares = float(row[latest_period])
                                break
                    if shares is not None:
                        break
                
                # If still not found, try multiple calculation methods
                if not shares:
                    shares_issued = None
                    treasury_shares = None
                    common_stock_value = None
                    par_value = None
                    
                    for _, row in balance_df.iterrows():
                        concept = row.get('concept', '')
                        value = float(row[latest_period]) if latest_period in row and row[latest_period] else None
                        
                        # Method 2: Issued - Treasury
                        if concept == 'us-gaap_CommonStockSharesIssued':
                            shares_issued = value
                        if concept in ['us-gaap_TreasuryStockCommonShares', 'us-gaap_TreasuryStockShares']:
                            treasury_shares = value
                        
                        # Method 1: Common Stock Value / Par Value
                        if concept == 'us-gaap_CommonStockValue':
                            common_stock_value = value
                        if concept == 'us-gaap_CommonStockParOrStatedValuePerShare':
                            par_value = value
                    
                    # Try Method 2: Calculate shares outstanding = issued - treasury
                    if shares_issued is not None and shares is None:
                        shares = shares_issued - (treasury_shares if treasury_shares else 0)
                    
                    # Try Method 1: Calculate from par value
                    if shares is None and common_stock_value and par_value and par_value > 0:
                        shares = common_stock_value / par_value
        
        # Method 3: If still no shares, try to derive from Net Income / EPS
        if not shares and income_df is not None and latest_period:
            net_income = None
            eps_basic = None
            
            for _, row in income_df.iterrows():
                concept = row.get('concept', '')
                value = float(row[latest_period]) if latest_period in row and row[latest_period] else None
                
                if concept == 'us-gaap_NetIncomeLoss':
                    net_income = value
                if concept == 'us-gaap_EarningsPerShareBasic':
                    eps_basic = value
            
            # Calculate shares = Net Income / EPS
            if net_income and eps_basic and eps_basic != 0:
                shares = abs(net_income / eps_basic)  # abs() in case of net loss
        
        # Method 4: If still no shares, use Company.shares_outstanding from Edgar facts
        if not shares:
            try:
                company_shares = c.shares_outstanding
                if company_shares and company_shares > 0:
                    shares = float(company_shares)
            except Exception:
                pass  # If shares_outstanding not available, continue without it

        # Get filing metadata from the most recent filing (first in list)
        most_recent_filing = all_filings[0]
        filing_date = most_recent_filing.filing_date.isoformat() if hasattr(most_recent_filing.filing_date, 'isoformat') else str(most_recent_filing.filing_date)
        period_end = most_recent_filing.period_of_report.isoformat() if hasattr(most_recent_filing.period_of_report, 'isoformat') else str(most_recent_filing.period_of_report)
        
        # Map periods to filing types by matching period_of_report dates
        period_to_filing_type = {}
        for idx, filing in enumerate(all_filings):
            filing_period_end = filing.period_of_report.isoformat() if hasattr(filing.period_of_report, 'isoformat') else str(filing.period_of_report)
            period_to_filing_type[filing_period_end] = filing_metadata[idx]['type']
        
        # Helper to convert a statement dataframe to serializable list of items
        def df_to_items(df, periods):
            items = []
            if df is None:
                return items
            for _, row in df.iterrows():
                item = {
                    "label": row.get("label", ""),
                    "concept": row.get("concept", ""),
                    "values": {}
                }
                for period in periods:
                    if period in row:
                        val = row[period]
                        item["values"][str(period)] = float(val) if val is not None and val != '' else None
                items.append(item)
            return items

        # Build period_metadata to track which periods are 10-K vs 10-Q
        period_metadata = []
        for period in income_statement.periods:
            period_str = str(period)
            filing_type = period_to_filing_type.get(period_str, 'unknown')
            period_metadata.append({
                'date': period_str,
                'type': filing_type
            })
        
        # Build raw_data with all statements
        raw_data = {
            "income_statement": {
                "periods": [str(p) for p in income_statement.periods],
                "items": df_to_items(income_df, income_statement.periods)
            },
            "balance_sheet": {
                "periods": [str(p) for p in balance_sheet.periods] if balance_sheet else [],
                "items": df_to_items(balance_df, balance_sheet.periods) if balance_sheet else []
            },
            "cash_flow": {
                "periods": [str(p) for p in cash_flow.periods] if cash_flow else [],
                "items": df_to_items(cashflow_df, cash_flow.periods) if cash_flow else []
            },
            "period_metadata": period_metadata,
            "metrics": {
                # will be filled below
            }
        }

        # Derive per-period metrics from balance sheet and cash flow (if available)
        def first_row_exact(df, concepts):
            if df is None:
                return None
            concepts_set = set(concepts)
            for _, r in df.iterrows():
                c = str(r.get('concept', ''))
                if c in concepts_set:
                    return r
            return None

        # Equity - with more variants
        equity_row = first_row_exact(balance_df, [
            'us-gaap_StockholdersEquity',
            'us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
            'us-gaap_PartnersCapital'
        ])
        
        # If equity not found directly, try to calculate from total assets - liabilities
        if equity_row is None and balance_df is not None:
            total_assets_row = first_row_exact(balance_df, [
                'us-gaap_Assets',
                'us-gaap_AssetsIncludingDiscontinuedOperations'
            ])
            total_liabilities_row = first_row_exact(balance_df, [
                'us-gaap_Liabilities',
                'us-gaap_LiabilitiesIncludingDiscontinuedOperations'
            ])
            if total_assets_row is not None and total_liabilities_row is not None and balance_sheet is not None:
                # Calculate equity = assets - liabilities for each period
                equity_row = {}
                for p in balance_sheet.periods:
                    if p in total_assets_row and p in total_liabilities_row:
                        assets_val = total_assets_row[p]
                        liab_val = total_liabilities_row[p]
                        if assets_val not in (None, '') and liab_val not in (None, ''):
                            equity_row[p] = float(assets_val) - float(liab_val)
                if not equity_row:
                    equity_row = None
        if equity_row is not None and balance_sheet is not None:
            eq = {}
            for p in balance_sheet.periods:
                if p in equity_row:
                    v = equity_row[p]
                    eq[str(p)] = float(v) if v not in (None, '') else None
            raw_data["metrics"]["equity"] = eq

        # Debt = current + noncurrent (expanded concepts)
        debt_cur_row = first_row_exact(balance_df, [
            'us-gaap_DebtCurrent',
            'us-gaap_ShortTermBorrowings',
            'us-gaap_NotesPayableShortTerm',
            'us-gaap_CommercialPaper'
        ])
        debt_nc_row = first_row_exact(balance_df, [
            'us-gaap_LongTermDebtNoncurrent',
            'us-gaap_LongTermDebtAndCapitalLeaseObligations',
            'us-gaap_LongTermDebt',
            'us-gaap_LongTermDebtAndCapitalLeaseObligationsNoncurrent'
        ])
        if balance_sheet is not None and (debt_cur_row is not None or debt_nc_row is not None):
            debt = {}
            for p in balance_sheet.periods:
                total = 0.0
                has_val = False
                if debt_cur_row is not None and p in debt_cur_row and debt_cur_row[p] not in (None, ''):
                    total += float(debt_cur_row[p]); has_val = True
                if debt_nc_row is not None and p in debt_nc_row and debt_nc_row[p] not in (None, ''):
                    total += float(debt_nc_row[p]); has_val = True
                debt[str(p)] = total if has_val else None
            raw_data["metrics"]["debt"] = debt

        # Operating cash flow
        ocf_row = first_row_exact(cashflow_df, [
            'us-gaap_NetCashProvidedByUsedInOperatingActivities',
            'us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
            'us-gaap_NetCashProvidedByUsedInOperatingActivitiesIncludingDiscontinuedOperations'
        ])
        # If cash flow statement missing, try to find OCF concept in any dataframe we have
        if ocf_row is None and income_df is not None:
            ocf_row = first_row_exact(income_df, [
                'us-gaap_NetCashProvidedByUsedInOperatingActivities',
                'us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
                'us-gaap_NetCashProvidedByUsedInOperatingActivitiesIncludingDiscontinuedOperations'
            ])
        # If still not found, attempt company extension fallback by local name pattern on cash flow df
        if ocf_row is None and cashflow_df is not None:
            try:
                for _, r in cashflow_df.iterrows():
                    c = str(r.get('concept', ''))
                    # local name after first underscore if namespace present
                    local = c.split('_', 1)[1] if '_' in c else c
                    local_lower = local.lower()
                    if (
                        'netcash' in local_lower and
                        'operating' in local_lower and
                        ('provided' in local_lower or 'used' in local_lower)
                    ):
                        ocf_row = r
                        break
            except Exception:
                pass
        if ocf_row is not None and (cash_flow is not None or income_statement is not None):
            ocf = {}
            periods_source = cash_flow.periods if cash_flow is not None else income_statement.periods
            for p in periods_source:
                if p in ocf_row:
                    v = ocf_row[p]
                    ocf[str(p)] = float(v) if v not in (None, '') else None
            raw_data["metrics"]["operating_cash_flow"] = ocf
        
        # Net Income extraction (GAAP-standard with priority order)
        net_income_row = first_row_exact(income_df, [
            'us-gaap_NetIncomeLoss',
            'us-gaap_ProfitLoss',  # fallback concept used by some filers
            'us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic'  # more specific variant
        ])
        if net_income_row is not None and income_statement is not None:
            ni = {}
            for p in income_statement.periods:
                if p in net_income_row and net_income_row[p] not in (None, ''):
                    ni[str(p)] = float(net_income_row[p])
            if ni:  # Only add if we found values
                raw_data["metrics"]["net_income"] = ni
        
        # EPS extraction (Basic and Diluted)
        eps_basic_row = first_row_exact(income_df, ['us-gaap_EarningsPerShareBasic'])
        eps_diluted_row = first_row_exact(income_df, ['us-gaap_EarningsPerShareDiluted'])
        
        if eps_basic_row is not None and income_statement is not None:
            eps_b = {}
            for p in income_statement.periods:
                if p in eps_basic_row and eps_basic_row[p] not in (None, ''):
                    eps_b[str(p)] = float(eps_basic_row[p])
            if eps_b:  # Only add if we found values
                raw_data["metrics"]["eps_basic"] = eps_b
        
        if eps_diluted_row is not None and income_statement is not None:
            eps_d = {}
            for p in income_statement.periods:
                if p in eps_diluted_row and eps_diluted_row[p] not in (None, ''):
                    eps_d[str(p)] = float(eps_diluted_row[p])
            if eps_d:  # Only add if we found values
                raw_data["metrics"]["eps_diluted"] = eps_d
        
        return {
            "company_name": c.name,
            "revenue": revenue,
            "shares": shares,
            "filing_date": filing_date,
            "period_end": period_end,
            "raw_data": raw_data
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def store_financial_data(ticker, data):
    """Store financial data in Supabase database"""
    # Convert raw_data dict to JSON string
    raw_data_json = json.dumps(data.get('raw_data')) if data.get('raw_data') else None
    
    # Determine filing type from most recent period
    filing_type = '10-K/10-Q'  # Mixed filings
    if data.get('raw_data') and data['raw_data'].get('period_metadata'):
        # Use the type of the most recent period
        filing_type = data['raw_data']['period_metadata'][0].get('type', '10-K/10-Q')
    
    return store_company_financials(
        ticker.upper(),
        data['company_name'],
        data['revenue'],
        data['shares'],
        data['filing_date'],
        data['period_end'],
        filing_type,
        raw_data_json
    )


def init_database():
    """Check Supabase connection"""
    if not is_supabase_configured():
        print("\n❌ ERROR: Supabase is not configured!")
        print("Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")
        print("See .env.example for details.\n")
        sys.exit(1)
    print("✓ Supabase connection configured\n")


def main():
    print("\n" + "="*60)
    print("📊 Financial Data Import Tool")
    print("="*60)
    print("\nThis will fetch latest 10-K + last 4 10-Q filings")
    print("for S&P 500 companies and store in the database.")
    print("This enables accurate TTM (Trailing Twelve Months) calculations.\n")
    
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
        
        if data:
            # Apply scale normalization
            data = normalize_financial_data(data)
            
            # Validate normalization
            norm_validation = validate_normalization(data)
            if not norm_validation['valid']:
                for warning in norm_validation['warnings']:
                    print(f"\n   ⚠️  Scale warning: {warning}")
            
            # Validate data quality
            dq = DataQuality()
            is_valid = validate_financial_data(ticker_upper, data, dq)
            
            # Track issues for summary
            if "Revenue not found" in str(dq.issues):
                quality_report['missing_revenue'].append(ticker_upper)
            if "Shares outstanding not found" in str(dq.issues):
                quality_report['missing_shares'].append(ticker_upper)
            if "Equity not found" in str(dq.warnings):
                quality_report['missing_equity'].append(ticker_upper)
            if "Debt not found" in str(dq.warnings):
                quality_report['missing_debt'].append(ticker_upper)
            if "Operating cash flow not found" in str(dq.warnings):
                quality_report['missing_ocf'].append(ticker_upper)
            if "Market cap seems" in str(dq.warnings):
                quality_report['market_cap_issues'].append(ticker_upper)
            
            # Store data (even if has warnings, but not if critical issues)
            if is_valid:
                store_financial_data(ticker_upper, data)
                action = "Updated" if ticker_upper in existing else "Stored"
                summary = dq.get_summary()
                print(f"✓ {action} - {summary}")
                
                if verbose and (dq.warnings or dq.issues):
                    print(dq.get_details())
                
                if dq.warnings:
                    warning_count += 1
                success_count += 1
            else:
                print(f"✗ Critical issues - NOT stored")
                print(dq.get_details())
                critical_issue_tickers.append({
                    'ticker': ticker_upper,
                    'issues': dq.issues,
                    'warnings': dq.warnings,
                    'company_name': data.get('company_name', 'Unknown')
                })
                fail_count += 1
        else:
            print("✗ Failed to fetch data")
            failed_tickers.append(ticker_upper)
            fail_count += 1
        
        # Respectful delay
        if i < len(tickers):
            time.sleep(0.5)
    
    # Detailed Summary
    print("\n" + "="*80)
    print("Import Complete - Data Quality Report")
    print("="*80)
    print(f"\n📊 IMPORT STATISTICS:")
    print(f"  ✓ Successfully imported: {success_count}")
    print(f"  ⚠️  Imported with warnings: {warning_count}")
    print(f"  ⏭️  Already in database: {skip_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"  📁 Total in database: {success_count + skip_count}")
    
    print(f"\n🔍 DATA QUALITY ISSUES:")
    
    if quality_report['missing_revenue']:
        print(f"\n  ❌ Missing Revenue ({len(quality_report['missing_revenue'])} companies):")
        print(f"     {', '.join(quality_report['missing_revenue'][:10])}")
        if len(quality_report['missing_revenue']) > 10:
            print(f"     ... and {len(quality_report['missing_revenue']) - 10} more")
    
    if quality_report['missing_shares']:
        print(f"\n  ❌ Missing Shares Outstanding ({len(quality_report['missing_shares'])} companies):")
        print(f"     {', '.join(quality_report['missing_shares'][:10])}")
        if len(quality_report['missing_shares']) > 10:
            print(f"     ... and {len(quality_report['missing_shares']) - 10} more")
    
    if quality_report['missing_equity']:
        print(f"\n  ⚠️  Missing Equity ({len(quality_report['missing_equity'])} companies):")
        print(f"     {', '.join(quality_report['missing_equity'][:10])}")
        if len(quality_report['missing_equity']) > 10:
            print(f"     ... and {len(quality_report['missing_equity']) - 10} more")
    
    if quality_report['missing_debt']:
        print(f"\n  ⚠️  Missing Debt ({len(quality_report['missing_debt'])} companies):")
        print(f"     {', '.join(quality_report['missing_debt'][:10])}")
        if len(quality_report['missing_debt']) > 10:
            print(f"     ... and {len(quality_report['missing_debt']) - 10} more")
    
    if quality_report['missing_ocf']:
        print(f"\n  ⚠️  Missing Operating Cash Flow ({len(quality_report['missing_ocf'])} companies):")
        print(f"     {', '.join(quality_report['missing_ocf'][:10])}")
        if len(quality_report['missing_ocf']) > 10:
            print(f"     ... and {len(quality_report['missing_ocf']) - 10} more")
    
    if quality_report['market_cap_issues']:
        print(f"\n  ⚠️  Market Cap Validation Issues ({len(quality_report['market_cap_issues'])} companies):")
        print(f"     {', '.join(quality_report['market_cap_issues'][:10])}")
        if len(quality_report['market_cap_issues']) > 10:
            print(f"     ... and {len(quality_report['market_cap_issues']) - 10} more")
    
    if not any(quality_report.values()):
        print("  ✓ No data quality issues detected!")
    
    # Calculate data completeness percentage
    total_processed = success_count + fail_count
    if total_processed > 0:
        completeness = {
            'revenue': 100 * (1 - len(quality_report['missing_revenue']) / total_processed),
            'shares': 100 * (1 - len(quality_report['missing_shares']) / total_processed),
            'equity': 100 * (1 - len(quality_report['missing_equity']) / total_processed),
            'debt': 100 * (1 - len(quality_report['missing_debt']) / total_processed),
            'ocf': 100 * (1 - len(quality_report['missing_ocf']) / total_processed)
        }
        
        print(f"\n📈 DATA COMPLETENESS:")
        print(f"  Revenue:              {completeness['revenue']:.1f}%")
        print(f"  Shares Outstanding:   {completeness['shares']:.1f}%")
        print(f"  Equity:               {completeness['equity']:.1f}%")
        print(f"  Debt:                 {completeness['debt']:.1f}%")
        print(f"  Operating Cash Flow:  {completeness['ocf']:.1f}%")
    
    print("\n" + "="*80)
    
    # Save failed tickers to files for debugging
    if failed_tickers or critical_issue_tickers:
        print("\n💾 Saving error reports...")
        
        # Save fetch failures
        if failed_tickers:
            failed_file = Path(__file__).parent / "failed_tickers.txt"
            with open(failed_file, 'w') as f:
                f.write("# Tickers that failed to fetch from Edgar\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Total: {len(failed_tickers)}\n\n")
                for ticker in failed_tickers:
                    f.write(f"{ticker}\n")
            print(f"  ✓ Saved {len(failed_tickers)} fetch failures to: {failed_file}")
        
        # Save critical issues with details
        if critical_issue_tickers:
            critical_file = Path(__file__).parent / "critical_issues.json"
            with open(critical_file, 'w') as f:
                json.dump({
                    'generated': datetime.now().isoformat(),
                    'total': len(critical_issue_tickers),
                    'tickers': critical_issue_tickers
                }, f, indent=2)
            print(f"  ✓ Saved {len(critical_issue_tickers)} critical issues to: {critical_file}")
            
            # Also save simple list for debugging script
            critical_list_file = Path(__file__).parent / "critical_tickers.txt"
            with open(critical_list_file, 'w') as f:
                f.write("# Tickers with critical data issues (missing revenue or shares)\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Total: {len(critical_issue_tickers)}\n\n")
                for item in critical_issue_tickers:
                    f.write(f"{item['ticker']}\n")
            print(f"  ✓ Saved ticker list to: {critical_list_file}")
        
        print(f"\n  📋 To debug these tickers, run:")
        print(f"     python scripts/debug_ticker.py <TICKER>")
    
    print()


if __name__ == "__main__":
    main()

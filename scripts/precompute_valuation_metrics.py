#!/usr/bin/env python3
"""
Precompute Valuation Metrics

This script precomputes and caches valuation metrics (P/S, P/B, P/E, market cap, etc.)
for all companies in the database. This allows the industry comparison endpoint to serve
data instantly without recomputing metrics or fetching prices on every request.

Usage:
    python scripts/precompute_valuation_metrics.py [--ticker TICKER] [--force]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from supabase_db import (
    get_all_company_financials,
    get_company_financials,
    update_valuation_metrics
)
from revenue_extractor import extract_revenue

# Try to import Alpaca for price fetching
try:
    import os
    from dotenv import load_dotenv
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    
    load_dotenv(Path(__file__).parent.parent / "backend" / ".env")
    alpaca_api_key = os.getenv('ALPACA_API_KEY')
    alpaca_secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if alpaca_api_key and alpaca_secret_key:
        alpaca_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
        print("✓ Alpaca API configured for price fetching")
    else:
        alpaca_client = None
        print("⚠ Alpaca API not configured - will skip price-dependent metrics")
except ImportError:
    alpaca_client = None
    print("⚠ Alpaca SDK not installed - will skip price-dependent metrics")


def get_stock_price(ticker: str) -> Optional[float]:
    """Fetch current stock price from Alpaca"""
    if not alpaca_client:
        return None
    
    try:
        request_params = StockLatestTradeRequest(symbol_or_symbols=ticker.upper())
        trades = alpaca_client.get_stock_latest_trade(request_params)
        if ticker.upper() in trades:
            return float(trades[ticker.upper()].price)
    except Exception as e:
        print(f"  Error fetching price for {ticker}: {e}")
    
    return None


def compute_valuation_metrics(ticker: str, force: bool = False) -> Optional[Dict]:
    """
    Compute valuation metrics for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        force: If True, recompute even if recent metrics exist
        
    Returns:
        Dictionary with valuation metrics or None if computation failed
    """
    # Get financial data
    fin = get_company_financials(ticker.upper())
    if not fin:
        print(f"  No financial data found for {ticker}")
        return None
    
    # Check if we have recent metrics and don't need to recompute
    if not force and fin.get('valuation_metrics'):
        try:
            metrics = json.loads(fin['valuation_metrics']) if isinstance(fin['valuation_metrics'], str) else fin['valuation_metrics']
            updated_at = metrics.get('updated_at')
            if updated_at:
                updated_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                age = datetime.now() - updated_time.replace(tzinfo=None)
                if age < timedelta(hours=24):
                    print(f"  Using cached metrics (age: {age.total_seconds()/3600:.1f}h)")
                    return metrics
        except Exception:
            pass
    
    # Get raw data
    raw_data = fin.get('raw_data')
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)
    
    if not raw_data:
        print(f"  No raw data available for {ticker}")
        return None
    
    # Extract metrics from raw_data
    metrics_data = raw_data.get('metrics', {})
    income_stmt = raw_data.get('income_statement', {})
    
    # Get revenue (use stored value or extract from raw_data)
    revenue = fin.get('revenue')
    if not revenue and income_stmt:
        items = income_stmt.get('items', [])
        periods = income_stmt.get('periods', [])
        revenue, _ = extract_revenue(items, periods)
    
    # Get shares outstanding
    shares = fin.get('shares_outstanding')
    
    # Fetch current stock price
    stock_price = get_stock_price(ticker)
    
    # Calculate market cap
    market_cap = None
    if stock_price and shares:
        market_cap = float(stock_price) * float(shares)
    
    # Calculate P/S ratio
    ps_ratio = None
    if market_cap and revenue and revenue > 0:
        ps_ratio = market_cap / revenue
    
    # Helper: get latest period value by max ISO date key
    def latest_value(map_obj):
        try:
            if isinstance(map_obj, dict) and map_obj:
                latest_key = max(map_obj.keys())
                return map_obj.get(latest_key)
        except Exception:
            pass
        return None

    # Helper: compute TTM sum from a per-period map by summing last 4 quarterly values if available,
    # else use most recent annual value, else annualize latest period (if quarterly)
    def ttm_sum(map_obj):
        if not isinstance(map_obj, dict) or not map_obj:
            return None
        try:
            periods_sorted = sorted(map_obj.keys(), reverse=True)
            # Identify quarterly periods by month heuristic (03, 06, 09, 12)
            def is_quarter(p):
                try:
                    m = int(str(p)[5:7])
                    return m in (3, 6, 9, 12)
                except Exception:
                    return False
            quarters = [p for p in periods_sorted if is_quarter(p)]
            # Try 4 most recent quarters
            vals = []
            for p in quarters[:4]:
                v = map_obj.get(p)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if len(vals) == 4:
                return sum(vals)
            # Fallback: use most recent non-null (assume annual)
            for p in periods_sorted:
                v = map_obj.get(p)
                if isinstance(v, (int, float)):
                    return float(v)
        except Exception:
            pass
        return None

    # Calculate P/B ratio (Price to Book)
    pb_ratio = None
    if market_cap and metrics_data.get('equity'):
        equity_values = metrics_data['equity']
        if isinstance(equity_values, dict) and equity_values:
            latest_equity = latest_value(equity_values)
            if latest_equity and latest_equity > 0:
                pb_ratio = market_cap / latest_equity
    
    # Calculate P/CF ratio (Price to Cash Flow) using OCF TTM when available
    pcf_ratio = None
    if market_cap and metrics_data.get('operating_cash_flow'):
        ocf_values = metrics_data['operating_cash_flow']
        if isinstance(ocf_values, dict) and ocf_values:
            ocf_ttm = ttm_sum(ocf_values)
            if isinstance(ocf_ttm, (int, float)) and ocf_ttm > 0:
                pcf_ratio = market_cap / ocf_ttm
    
    # Calculate D/E ratio (Debt to Equity) using latest period values
    de_ratio = None
    if metrics_data.get('debt') and metrics_data.get('equity'):
        debt_values = metrics_data['debt']
        equity_values = metrics_data['equity']
        if isinstance(debt_values, dict) and isinstance(equity_values, dict):
            latest_debt = latest_value(debt_values)
            latest_equity = latest_value(equity_values)
            if isinstance(latest_equity, (int, float)) and latest_equity != 0:
                de_ratio = (latest_debt / latest_equity) if isinstance(latest_debt, (int, float)) else 0
    
    # Calculate P/E ratio (Price to Earnings) using EPS TTM when possible
    pe_ratio = None
    eps_ttm = None
    # Prefer basic EPS; fall back to diluted
    eps_map = None
    if isinstance(metrics_data.get('eps_basic'), dict) and metrics_data['eps_basic']:
        eps_map = metrics_data['eps_basic']
    elif isinstance(metrics_data.get('eps_diluted'), dict) and metrics_data['eps_diluted']:
        eps_map = metrics_data['eps_diluted']
    if eps_map:
        # Compute EPS TTM using the same ttm_sum helper (sum of last 4 quarters)
        eps_ttm = ttm_sum(eps_map)
    # If EPS TTM not available, attempt net income TTM / shares
    if eps_ttm is None and isinstance(metrics_data.get('net_income'), dict):
        ni_ttm = ttm_sum(metrics_data['net_income'])
        if ni_ttm and shares and shares > 0:
            eps_ttm = ni_ttm / shares
    if stock_price and eps_ttm and eps_ttm > 0:
        pe_ratio = stock_price / eps_ttm
        # Sanity check for extreme P/E ratios
        if pe_ratio > 1000:
            print(f"  WARNING: P/E ratio extremely high ({pe_ratio:.0f}x) - possible data issue")
            pe_ratio = None

    # Build metrics dictionary
    valuation_metrics = {
        'stock_price': stock_price,
        'market_cap': market_cap,
        'revenue': revenue,
        'ps_ratio': ps_ratio,
        'pb_ratio': pb_ratio,
        'pcf_ratio': pcf_ratio,
        'pe_ratio': pe_ratio,
        'de_ratio': de_ratio,
        'shares_outstanding': shares,
        'updated_at': datetime.now().isoformat()
    }
    
    # Remove None values for cleaner storage
    valuation_metrics = {k: v for k, v in valuation_metrics.items() if v is not None}
    
    return valuation_metrics


def main():
    parser = argparse.ArgumentParser(description='Precompute valuation metrics for companies')
    parser.add_argument('--ticker', type=str, help='Compute metrics for specific ticker only')
    parser.add_argument('--force', action='store_true', help='Force recomputation even if recent metrics exist')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Valuation Metrics Precomputation")
    print("="*80 + "\n")
    
    if args.ticker:
        # Process single ticker
        print(f"Computing metrics for {args.ticker}...")
        metrics = compute_valuation_metrics(args.ticker, force=args.force)
        if metrics:
            success = update_valuation_metrics(args.ticker, metrics)
            if success:
                print(f"✓ Updated metrics for {args.ticker}")
                print(f"  Market Cap: ${metrics.get('market_cap', 0):,.0f}")
                print(f"  P/S Ratio: {metrics.get('ps_ratio', 'N/A')}")
            else:
                print(f"✗ Failed to update metrics for {args.ticker}")
        else:
            print(f"✗ Failed to compute metrics for {args.ticker}")
    else:
        # Process all tickers
        companies = get_all_company_financials()
        print(f"Found {len(companies)} companies\n")
        
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for i, company in enumerate(companies, 1):
            ticker = company['ticker']
            print(f"[{i}/{len(companies)}] {ticker}...", end=" ")
            sys.stdout.flush()
            
            metrics = compute_valuation_metrics(ticker, force=args.force)
            if metrics:
                if update_valuation_metrics(ticker, metrics):
                    ps = metrics.get('ps_ratio')
                    if ps:
                        print(f"✓ P/S: {ps:.2f}x")
                    else:
                        print("✓ (no price)")
                    success_count += 1
                else:
                    print("✗ Update failed")
                    fail_count += 1
            else:
                print("✗ Computation failed")
                fail_count += 1
        
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print(f"✓ Successfully computed: {success_count}")
        print(f"⏭️  Skipped (cached): {skip_count}")
        print(f"✗ Failed: {fail_count}")
        print()


if __name__ == "__main__":
    main()

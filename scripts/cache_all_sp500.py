#!/usr/bin/env python3
"""
Script to pre-cache all S&P 500 companies for faster industry comparison
Only fetches 10-K (annual) data for efficiency
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
BATCH_SIZE = 10  # Process in batches to avoid overwhelming the server

def load_all_tickers():
    """Load all tickers from the industry map"""
    industry_map_path = Path(__file__).parent.parent / "data" / "industry_map.json"
    with open(industry_map_path, 'r') as f:
        industry_map = json.load(f)
    
    all_tickers = set()
    for industry_data in industry_map['industries'].values():
        for sector_data in industry_data['sectors'].values():
            for tickers in sector_data['industry_groups'].values():
                all_tickers.update(tickers)
    
    return sorted(all_tickers)


def cache_batch(tickers):
    """Cache a batch of tickers"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/cache-tickers",
            json={"tickers": tickers},
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout for batch
        )
        result = response.json()
        return result
    except Exception as e:
        print(f"Error caching batch: {e}")
        return None


def main():
    print("\n" + "="*60)
    print("📦 S&P 500 Cache Builder")
    print("="*60)
    print("\nThis script will cache financial data for all S&P 500 companies.")
    print("Only annual (10-K) data will be fetched for efficiency.")
    print(f"Processing in batches of {BATCH_SIZE} companies.\n")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=2)
        if response.status_code != 200:
            print("✗ Backend server is not responding correctly")
            exit(1)
    except requests.exceptions.RequestException:
        print("✗ Cannot connect to backend server. Please start it first.")
        print("   Run: python backend/app.py")
        exit(1)
    
    # Load all tickers
    all_tickers = load_all_tickers()
    print(f"Found {len(all_tickers)} companies to cache\n")
    
    # Process in batches
    total_cached = 0
    total_already_cached = 0
    total_failed = 0
    
    for i in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[i:i+BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(all_tickers) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"Processing batch {batch_num}/{total_batches} ({', '.join(batch)})...")
        
        result = cache_batch(batch)
        
        if result and result.get('success'):
            cached = len(result['results']['cached'])
            already_cached = len(result['results']['already_cached'])
            failed = len(result['results']['failed'])
            
            total_cached += cached
            total_already_cached += already_cached
            total_failed += failed
            
            print(f"  ✓ Cached: {cached}, Already cached: {already_cached}, Failed: {failed}")
            
            if result['results']['failed']:
                for failure in result['results']['failed']:
                    print(f"    ✗ {failure['ticker']}: {failure['reason']}")
        else:
            print(f"  ✗ Batch failed")
            total_failed += len(batch)
        
        # Small delay between batches to be respectful to SEC Edgar
        if i + BATCH_SIZE < len(all_tickers):
            time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total companies: {len(all_tickers)}")
    print(f"Newly cached: {total_cached}")
    print(f"Already cached: {total_already_cached}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {((total_cached + total_already_cached) / len(all_tickers) * 100):.1f}%")
    print("\n✅ Caching complete! Industry comparison will now be fast.\n")


if __name__ == "__main__":
    main()

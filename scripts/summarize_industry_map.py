#!/usr/bin/env python3
"""
Script to summarize the populated industry_map.json
"""

import json
from pathlib import Path
from collections import defaultdict


def summarize_industry_map(json_path):
    """Generate a summary of the industry map"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("INDUSTRY MAP SUMMARY")
    print("="*80)
    
    total_stocks = 0
    industry_stats = defaultdict(lambda: {'sectors': 0, 'groups': 0, 'stocks': 0})
    
    for industry_name, industry_data in data['industries'].items():
        sectors = industry_data['sectors']
        industry_stats[industry_name]['sectors'] = len(sectors)
        
        for sector_name, sector_data in sectors.items():
            groups = sector_data['industry_groups']
            industry_stats[industry_name]['groups'] += len(groups)
            
            for group_name, tickers in groups.items():
                count = len(tickers)
                industry_stats[industry_name]['stocks'] += count
                total_stocks += count
    
    # Print overall summary
    print(f"\nTotal Stocks: {total_stocks}")
    print(f"Total Industries: {len(data['industries'])}")
    print(f"\n{'Industry':<30} {'Sectors':<10} {'Groups':<10} {'Stocks':<10}")
    print("-" * 80)
    
    for industry_name in sorted(industry_stats.keys()):
        stats = industry_stats[industry_name]
        print(f"{industry_name:<30} {stats['sectors']:<10} {stats['groups']:<10} {stats['stocks']:<10}")
    
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN BY INDUSTRY")
    print("="*80)
    
    # Detailed breakdown
    for industry_name, industry_data in sorted(data['industries'].items()):
        print(f"\n{industry_name}")
        print("-" * 80)
        
        for sector_name, sector_data in sorted(industry_data['sectors'].items()):
            sector_total = sum(len(tickers) for tickers in sector_data['industry_groups'].values())
            print(f"  {sector_name} ({sector_total} stocks)")
            
            for group_name, tickers in sorted(sector_data['industry_groups'].items()):
                if tickers:
                    print(f"    • {group_name}: {len(tickers)} stocks")
                    # Show first 5 tickers as examples
                    examples = ', '.join(tickers[:5])
                    if len(tickers) > 5:
                        examples += f", ... (+{len(tickers)-5} more)"
                    print(f"      [{examples}]")
                else:
                    print(f"    • {group_name}: 0 stocks (empty)")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / "data" / "industry_map.json"
    
    summarize_industry_map(json_path)

#!/usr/bin/env python3
"""
Migration script to transfer company_financials data from SQLite to Supabase
Run this once to migrate your existing data
"""

import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from supabase_db import (
    store_company_financials,
    is_supabase_configured,
    get_all_tickers_with_financials
)

DB_PATH = Path(__file__).parent.parent / "backend" / "cache.db"


def get_sqlite_data():
    """Retrieve all company_financials data from SQLite"""
    if not DB_PATH.exists():
        print(f"❌ SQLite database not found at {DB_PATH}")
        return []
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='company_financials'")
    if not cursor.fetchone():
        print("❌ company_financials table not found in SQLite database")
        conn.close()
        return []
    
    cursor.execute("""
        SELECT ticker, company_name, revenue, shares_outstanding, 
               filing_date, period_end_date, filing_type, raw_data
        FROM company_financials
    """)
    results = cursor.fetchall()
    conn.close()
    
    return results


def main():
    print("\n" + "="*60)
    print("🔄 SQLite to Supabase Migration Tool")
    print("="*60)
    print("\nThis will migrate company_financials data from SQLite to Supabase.\n")
    
    # Check Supabase configuration
    if not is_supabase_configured():
        print("❌ ERROR: Supabase is not configured!")
        print("Please set SUPABASE_URL and SUPABASE_KEY in your .env file.")
        print("See .env.example for details.\n")
        sys.exit(1)
    
    print("✓ Supabase connection configured\n")
    
    # Get SQLite data
    print("📊 Reading data from SQLite...")
    sqlite_data = get_sqlite_data()
    
    if not sqlite_data:
        print("No data found in SQLite database to migrate.\n")
        return
    
    print(f"Found {len(sqlite_data)} companies in SQLite\n")
    
    # Check what's already in Supabase
    print("🔍 Checking existing data in Supabase...")
    existing_tickers = set(get_all_tickers_with_financials())
    print(f"Found {len(existing_tickers)} companies already in Supabase\n")
    
    # Determine what needs to be migrated
    to_migrate = []
    to_update = []
    
    for row in sqlite_data:
        ticker = row[0]
        if ticker in existing_tickers:
            to_update.append(row)
        else:
            to_migrate.append(row)
    
    print(f"📈 Migration Summary:")
    print(f"  - New records to insert: {len(to_migrate)}")
    print(f"  - Existing records to update: {len(to_update)}")
    print(f"  - Total operations: {len(sqlite_data)}\n")
    
    if len(sqlite_data) == 0:
        print("Nothing to migrate.\n")
        return
    
    response = input("Continue with migration? (y/n): ")
    if response.lower() != 'y':
        print("Migration aborted.\n")
        return
    
    # Perform migration
    print("\n🚀 Starting migration...\n")
    success_count = 0
    fail_count = 0
    
    for i, row in enumerate(sqlite_data, 1):
        ticker, company_name, revenue, shares, filing_date, period_end, filing_type, raw_data = row
        
        print(f"[{i}/{len(sqlite_data)}] {ticker}: ", end="")
        
        try:
            result = store_company_financials(
                ticker,
                company_name,
                revenue,
                shares,
                filing_date,
                period_end,
                filing_type,
                raw_data
            )
            
            if result:
                print("✓ Migrated")
                success_count += 1
            else:
                print("✗ Failed")
                fail_count += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            fail_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("Migration Complete")
    print("="*60)
    print(f"Successfully migrated: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total: {len(sqlite_data)}")
    print()


if __name__ == "__main__":
    main()

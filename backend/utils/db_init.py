# Database initialization utilities
import os
import sqlite3
from datetime import datetime
from .config import DB_PATH

def init_db():
    """Initialize the SQLite database for caching (stock prices and income statements)."""
    try:
        print(f"Initializing database at: {DB_PATH}")
        db_dir = os.path.dirname(DB_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"Created directory: {db_dir}")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS income_cache (
                ticker TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                cached_at TIMESTAMP NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_price_cache (
                ticker TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                cached_at TIMESTAMP NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS industry_cache (
                cache_key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                cached_at TIMESTAMP NOT NULL
            )
            """
        )
        cursor.execute(
            """
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
            """
        )

        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"ERROR initializing database: {e}")
        raise

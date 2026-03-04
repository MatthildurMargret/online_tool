-- Create portfolio_companies table for Vertical/Portfolio Comps module
-- Run this in Supabase SQL Editor or via migration tool
-- Data is imported from data/portfolio.csv via scripts/import_portfolio_to_db.py

CREATE TABLE IF NOT EXISTS portfolio_companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name TEXT NOT NULL UNIQUE,
    category TEXT,
    sector TEXT,
    brief_description TEXT,
    founder_names TEXT,
    tickers JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- After running migration, import CSV data:
--   python scripts/import_portfolio_to_db.py

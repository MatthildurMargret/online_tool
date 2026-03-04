-- Create gemini_financials table for Portfolio/Vertical Comps (Gemini-sourced data)
-- Data is fetched via Gemini API with Google Search grounding and cached here
-- Refresh when fetched_at is older than 3 months

CREATE TABLE IF NOT EXISTS gemini_financials (
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    revenue REAL,
    net_income REAL,
    eps REAL,
    shares_outstanding REAL,
    pe REAL,
    ps REAL,
    market_cap REAL,
    ebitda REAL,
    ev REAL,
    ev_ebitda REAL,
    ev_revenue REAL,
    revenue_growth REAL,
    gross_margin REAL,
    ebitda_margin REAL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

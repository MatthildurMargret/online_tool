-- Alter gemini_financials for new pipeline: Gemini provides 5 fields only
-- Price from Alpaca; market_cap, ev, ev_revenue, ev_ebitda, margins computed server-side

ALTER TABLE gemini_financials ADD COLUMN IF NOT EXISTS gross_profit REAL;
ALTER TABLE gemini_financials ADD COLUMN IF NOT EXISTS net_debt REAL;

ALTER TABLE gemini_financials DROP COLUMN IF EXISTS net_income;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS eps;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS pe;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS ps;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS market_cap;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS ev;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS ev_ebitda;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS ev_revenue;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS revenue_growth;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS gross_margin;
ALTER TABLE gemini_financials DROP COLUMN IF EXISTS ebitda_margin;

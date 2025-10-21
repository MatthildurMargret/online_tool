-- Supabase Migration: Add Valuation Metrics Support
-- This migration adds the valuation_metrics column to store precomputed metrics
-- Run this in your Supabase SQL editor

-- Add valuation_metrics column to company_financials table
ALTER TABLE company_financials 
ADD COLUMN IF NOT EXISTS valuation_metrics JSONB;

-- Add comment to document the column
COMMENT ON COLUMN company_financials.valuation_metrics IS 
'Precomputed valuation metrics including P/S, P/B, P/E ratios, market cap, etc. Updated daily.';

-- Create index for faster queries on updated_at timestamp
CREATE INDEX IF NOT EXISTS idx_valuation_metrics_updated 
ON company_financials ((valuation_metrics->>'updated_at'));

-- Create index for faster queries on ps_ratio
CREATE INDEX IF NOT EXISTS idx_valuation_metrics_ps_ratio 
ON company_financials (((valuation_metrics->>'ps_ratio')::numeric));

-- Example query to find companies with stale metrics (>24 hours old)
-- SELECT ticker, company_name, 
--        valuation_metrics->>'updated_at' as last_updated
-- FROM company_financials
-- WHERE valuation_metrics IS NOT NULL
--   AND (valuation_metrics->>'updated_at')::timestamp < NOW() - INTERVAL '24 hours'
-- ORDER BY (valuation_metrics->>'updated_at')::timestamp;

-- Example query to find companies with no cached metrics
-- SELECT ticker, company_name
-- FROM company_financials
-- WHERE valuation_metrics IS NULL
-- ORDER BY ticker;

-- Example query to get P/S ratios for all companies
-- SELECT ticker, 
--        company_name,
--        (valuation_metrics->>'ps_ratio')::numeric as ps_ratio,
--        (valuation_metrics->>'market_cap')::numeric as market_cap
-- FROM company_financials
-- WHERE valuation_metrics IS NOT NULL
--   AND valuation_metrics->>'ps_ratio' IS NOT NULL
-- ORDER BY (valuation_metrics->>'ps_ratio')::numeric;

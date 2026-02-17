"""
Gemini-based financial data fetcher for portfolio/vertical comparisons.
Uses Gemini 2.0 Flash with Google Search grounding for up-to-date numbers.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from pathlib import Path

from dotenv import load_dotenv

# Load backend/.env so GEMINI_API_KEY is available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Lazy import to avoid loading if GEMINI_API_KEY not set
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        # Pop GOOGLE_API_KEY before import so SDK uses GEMINI_API_KEY only
        _saved = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            from google import genai
            _client = genai.Client(api_key=api_key)
        finally:
            if _saved is not None:
                os.environ["GOOGLE_API_KEY"] = _saved
    return _client


PROMPT_TEMPLATE = """
Act as a Senior Equity Research Analyst. Retrieve the most recent Trailing Twelve Month (TTM) financial data for {ticker}. 

If a value is missing for a specific quarter, use the most recent consensus estimate for that period to provide the most accurate TTM projection.

Return ONLY a valid JSON object (no markdown, no explanation) with these exact keys:
{{
  "company_name": "full company name",
  "shares_outstanding": number (current diluted),
  "net_debt": number (TTM calculation in USD),
  "revenue": number (TTM sum in USD),
  "ebitda": number (TTM sum in USD),
  "gross_profit": number (TTM sum in USD),
  "data_currency": "USD",
  "last_quarter_included": "YYYY-MM-DD"
}}
It is VERY important that we have all numeric fields and that they are as accurate as possible.
"""


def _normalize_value(val: Any) -> Optional[float]:
    """Convert value to float or None. Handles strings like '32.34', '3.75T', '15%', '2.5 trillion'."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (val != val or abs(val) == float("inf")):
            return None
        return float(val)
    if isinstance(val, str):
        s = val.strip().replace(",", "").lower()
        if not s or s in ("null", "none", "n/a", "-", ""):
            return None
        mult = 1.0
        for suffix, m in [("t", 1e12), ("trillion", 1e12), ("b", 1e9), ("billion", 1e9),
                         ("m", 1e6), ("million", 1e6), ("k", 1e3), ("thousand", 1e3)]:
            if s.endswith(suffix):
                mult, s = m, s[:-len(suffix)].strip()
                break
        if s.endswith("%"):
            mult, s = 0.01, s[:-1].strip()
        try:
            return float(s) * mult
        except ValueError:
            return None
    return None


# Alternative keys Gemini might return (case/spelling variants)
_KEY_ALIASES = {
    "company_name": ["company_name", "company", "name"],
    "shares_outstanding": ["shares_outstanding", "shares", "sharesOutstanding"],
    "net_debt": ["net_debt", "netDebt", "net_debt_usd", "total_debt_minus_cash"],
    "revenue": ["revenue", "revenues", "total_revenue"],
    "ebitda": ["ebitda", "EBITDA", "operating_ebitda", "ltm_ebitda", "ttm_ebitda"],
    "gross_profit": ["gross_profit", "grossProfit", "gross_profit_usd"],
}


def _get_value(data: Dict[str, Any], key: str) -> Any:
    """Get value from data, trying primary key and aliases. Case-insensitive for keys."""
    aliases = _KEY_ALIASES.get(key, [key])
    data_lower = {k.lower(): v for k, v in data.items()} if data else {}
    for alias in aliases:
        for k, v in data.items():
            if k.lower() == alias.lower():
                return v
    return None


def _build_result_dict(ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Build standardized result dict from raw Gemini response. Only the 5 Gemini fields."""
    numeric_keys = ("shares_outstanding", "net_debt", "revenue", "ebitda", "gross_profit")
    result = {"ticker": ticker.upper(), "company_name": _get_value(data, "company_name")}
    for k in numeric_keys:
        v = _get_value(data, k)
        result[k] = _normalize_value(v)
    return result


def _extract_json_object(text: str) -> Optional[Dict]:
    """Extract the first JSON object from text. Handles extra content before/after."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.startswith("```") and line.strip().lower() != "json"
        )
    # Find first { and extract matching }
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_retry_delay(error: Exception) -> float:
    """Extract retry delay (seconds) from 429 error. Default 10s."""
    import re
    err_str = str(error)
    # API returns "Please retry in 8.75s" or similar
    m = re.search(r"retry in ([\d.]+)\s*s", err_str, re.I)
    if m:
        return min(float(m.group(1)), 60.0)
    return 10.0


def fetch_ticker_via_gemini(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Fetch financial data for a ticker using Gemini with Google Search grounding.
    Returns a dict suitable for gemini_financials table, or None on failure.
    Retries on 429 (rate limit) with backoff.
    """
    ticker = ticker.upper()
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            from google.genai import types

            client = _get_client()
            prompt = PROMPT_TEMPLATE.format(ticker=ticker)

            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            # Note: response_schema cannot be combined with Google Search grounding.
            # Use grounding only, then parse JSON from the response text.
            config = types.GenerateContentConfig(tools=[grounding_tool])

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=config,
            )

            if not response.text or not response.text.strip():
                return None

            # Parse JSON - try response.parsed first (SDK may provide it with response_schema)
            data = None
            if hasattr(response, "parsed") and response.parsed is not None:
                if hasattr(response.parsed, "model_dump"):
                    data = response.parsed.model_dump()
                elif isinstance(response.parsed, dict):
                    data = response.parsed

            if data is None:
                data = _extract_json_object(response.text)
            if data is None:
                return None

            # Debug: log raw response when DEBUG_GEMINI_RESPONSE=1
            if os.getenv("DEBUG_GEMINI_RESPONSE"):
                import sys
                print(f"\n[DEBUG] {ticker} raw response (first 1500 chars):\n{repr(response.text[:1500])}\n", file=sys.stderr)
                print(f"[DEBUG] {ticker} extracted JSON: {json.dumps(data, indent=2)}\n", file=sys.stderr)

            return _build_result_dict(ticker, data)

        except Exception as e:
            err_str = str(e)
            is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if is_429 and attempt < max_retries:
                delay = _parse_retry_delay(e)
                print(f"[gemini_financials] Rate limited on {ticker}, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            print(f"[gemini_financials] Error fetching {ticker}: {e}")
            return None

    return None


VERTICAL_TICKERS_PROMPT = """Given the business vertical "{vertical}", suggest exactly 5-6 US public company stock ticker symbols that are most relevant for valuation comparison.

Return ONLY a JSON object with a "tickers" array of ticker strings, e.g. {{"tickers": ["AAPL", "MSFT", "GOOGL", ...]}}.
Use well-known, liquid US stocks. Do not invent tickers - use real, tradeable ticker symbols."""


def suggest_tickers_for_vertical(vertical: str) -> List[str]:
    """
    Use Gemini with grounding to suggest 5-6 relevant public tickers for a vertical.
    Returns list of ticker symbols (no DB filter). Retries on 429.
    """
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            from google.genai import types

            client = _get_client()
            prompt = VERTICAL_TICKERS_PROMPT.format(vertical=vertical)
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            # Cannot use response_mime_type with Search tool - parse JSON from text
            config = types.GenerateContentConfig(tools=[grounding_tool])

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=config,
            )

            if not response.text or not response.text.strip():
                return []

            data = _extract_json_object(response.text)
            if not data:
                return []
            raw = data.get("tickers", [])
            return [str(t).upper() for t in raw if t][:6]
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries:
                delay = _parse_retry_delay(e)
                print(f"[gemini_financials] Rate limited on vertical suggest, retrying in {delay:.0f}s")
                time.sleep(delay)
                continue
            print(f"[gemini_financials] Error suggesting tickers for '{vertical}': {e}")
            return []
    return []

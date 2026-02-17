"""
Shared logic for portfolio/vertical comps: compute metrics from Gemini data + price.
Used by both app.py (production) and scripts/test_gemini_portfolio_prompt.py (test).
"""


def compute_portfolio_metrics(gemini_row: dict, price: float) -> dict:
    """
    Compute market_cap, ev, ev_revenue, ev_ebitda, gross_margin, ebitda_margin, ps
    from Gemini data + Alpaca price. Same logic for production and test.
    """
    shares = gemini_row.get("shares_outstanding")
    net_debt = gemini_row.get("net_debt")
    if net_debt is None:
        net_debt = 0  # net-cash companies
    revenue = gemini_row.get("revenue")
    ebitda = gemini_row.get("ebitda")
    gross_profit = gemini_row.get("gross_profit")

    market_cap = price * shares if price and shares else None
    ev = (market_cap + net_debt) if market_cap is not None else None
    ev_revenue = ev / revenue if ev and revenue and revenue > 0 else None
    ev_ebitda = ev / ebitda if ev and ebitda and ebitda > 0 else None
    gross_margin = gross_profit / revenue if gross_profit is not None and revenue and revenue > 0 else None
    ebitda_margin = ebitda / revenue if ebitda is not None and revenue and revenue > 0 else None
    ps = market_cap / revenue if market_cap and revenue and revenue > 0 else None

    return {
        "market_cap": market_cap,
        "ev": ev,
        "ev_revenue": ev_revenue,
        "ev_ebitda": ev_ebitda,
        "gross_margin": gross_margin,
        "ebitda_margin": ebitda_margin,
        "ps": ps,
    }


def build_portfolio_output_row(ticker: str, gemini_row: dict, price: float, clean_nans_fn):
    """
    Build the full output row for a ticker: Gemini fields + computed metrics.
    Same structure for production (stream) and test. clean_nans_fn removes NaN/Inf.
    """
    metrics = compute_portfolio_metrics(gemini_row, price)
    return clean_nans_fn({
        "ticker": ticker,
        "company_name": gemini_row.get("company_name"),
        "revenue": gemini_row.get("revenue"),
        "shares_outstanding": gemini_row.get("shares_outstanding"),
        "ebitda": gemini_row.get("ebitda"),
        "ps": metrics["ps"],
        "market_cap": metrics["market_cap"],
        "ev": metrics["ev"],
        "ev_ebitda": metrics["ev_ebitda"],
        "ev_revenue": metrics["ev_revenue"],
        "gross_margin": metrics["gross_margin"],
        "ebitda_margin": metrics["ebitda_margin"],
    })

# utils/ratios.py
def compute_ratio(numerator, denominator, name):
    """Generic safe ratio calculator with sanity checks."""
    warnings = []
    if numerator is None or denominator is None:
        return None, [f"{name}: missing data"]
    try:
        num = float(numerator)
        den = float(denominator)
        if den == 0:
            return None, [f"{name}: denominator zero"]
        ratio = num / den
        if ratio < 0:
            warnings.append(f"{name}: negative result (check signs)")
        if ratio > 1000:
            warnings.append(f"{name}: extremely high ({ratio:.0f}x), unit mismatch possible")
        return ratio, warnings
    except Exception as e:
        return None, [f"{name}: {e}"]

def compute_live_metrics(fin: dict, price_payload: dict) -> dict:
    """Compute live valuation metrics using latest price and stored fundamentals."""
    price = None
    if price_payload:
        price = price_payload.get("price") or price_payload.get("latest_price")

    revenue = fin.get("revenue")
    income = fin.get("income")
    shares = fin.get("shares_outstanding")
    equity = fin.get("equity")  # optional
    eps = fin.get("latest_eps") or (income / shares if income and shares else None)

    metrics = {
        "price": price,
        "shares_outstanding": shares,
        "eps_basic": eps,
        "eps_diluted": eps,
    }

    # Derived metrics only if price + fundamentals exist
    if price and shares:
        market_cap = price * shares
        metrics["market_cap"] = market_cap

        if revenue and revenue > 0:
            metrics["ps_ratio"] = market_cap / revenue

        if eps and eps > 0:
            metrics["pe_ratio"] = price / eps

        if equity and equity > 0:
            metrics["pb_ratio"] = price / (equity / shares)
            metrics["roe"] = (income / equity) * 100 if income else None

    return metrics

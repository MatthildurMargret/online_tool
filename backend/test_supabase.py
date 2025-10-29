from utils.storage import get_company_financials
from utils.stock_price import get_price_with_cache
from datetime import datetime, timedelta
import json
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from alpaca.data.historical import StockHistoricalDataClient


def quarterly_or_annual(filing_metadata):
    today = datetime.now()
    most_recent_filing = today - timedelta(days=365)
    most_recent_type = None
    for filing in filing_metadata:
        filing_date = datetime.strptime(filing['period_end'], '%Y-%m-%d')
        type = filing['type']
        if filing_date > most_recent_filing:
            most_recent_filing = filing_date
            most_recent_type = type
    return most_recent_type, most_recent_filing.strftime('%Y-%m-%d')

def get_most_recent_numbers(raw_data, type, date):
    short_date = date[:7]

    def is_valid(val):
        return val is not None and not (str(val) == "nan")

    def get_value(df, label_key):
        import math
        if label_key is None:
            return float('nan')
        val = df.get(str(label_key), float('nan'))
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return float('nan')
        return val

    if type == "10-K":
        income_df = raw_data["income_statement"][short_date]
        label_mapping, _ = validate_labels(raw_data, date)
        return {
            "revenue": get_value(income_df, label_mapping["Revenue"]),
            "operating_income": get_value(income_df, label_mapping["Operating Income"]),
            "net_income": get_value(income_df, label_mapping["Net Income"]),
            "eps": get_value(income_df, label_mapping["Diluted EPS"]),
            "shares_outstanding": get_value(income_df, label_mapping["Diluted Shares Outstanding"]),
        }

    elif type == "10-Q":
        filings = sorted(
            [f for f in raw_data["filing_metadata"] if f["type"] == "10-Q"],
            key=lambda x: x["period_end"],
            reverse=True,
        )[:4]

        valid_quarters = []
        for f in filings:
            q_date = f["period_end"][:7]
            if q_date not in raw_data["income_statement"]:
                continue
            df = raw_data["income_statement"][q_date]
            label_mapping, _ = validate_labels(raw_data, f["period_end"])
            rev = get_value(df, label_mapping["Revenue"])
            ni = get_value(df, label_mapping["Net Income"])
            if is_valid(rev) and is_valid(ni):
                valid_quarters.append((q_date, label_mapping))
            if len(valid_quarters) == 4:
                break

        if not valid_quarters:
            print("No valid quarters")
            return None
        if len(valid_quarters) < 4:
            print(f"Skipping {len(valid_quarters)}-quarter ticker (insufficient data)")
            return None

        revenue = operating_income = net_income = eps = 0
        for q_date, lm in valid_quarters:
            df = raw_data["income_statement"][q_date]
            revenue += get_value(df, lm["Revenue"])
            operating_income += get_value(df, lm["Operating Income"])
            net_income += get_value(df, lm["Net Income"])
            eps += get_value(df, lm["Diluted EPS"])

        latest_q, latest_lm = valid_quarters[0]
        latest_df = raw_data["income_statement"][latest_q]
        diluted_shares_outstanding = get_value(latest_df, latest_lm["Diluted Shares Outstanding"])

        if net_income and diluted_shares_outstanding and diluted_shares_outstanding > 0:
            diluted_eps = net_income / diluted_shares_outstanding
        else:
            diluted_eps = eps

        import math
        if (
            diluted_shares_outstanding is None
            or (isinstance(diluted_shares_outstanding, float) and math.isnan(diluted_shares_outstanding))
        ):
            diluted_shares_outstanding = (
                abs(net_income / diluted_eps)
                if diluted_eps not in (0, None, float("nan"))
                else float("nan")
            )

        return {
            "revenue": revenue,
            "operating_income": operating_income,
            "net_income": net_income,
            "eps": diluted_eps,
            "shares_outstanding": diluted_shares_outstanding,
        }



def validate_labels(raw_data, date):
    """
    We want: revenue, operating_income, net_income, diluted_eps, diluted_shares_outstanding
    Sometimes, there's no revenue and only contract revenue. Need to verify the labels before pulling data
    1. Check if both exist in the income statement
    2. See which one has data
    3. Use that one
    """
    label_mapping = {
        "Revenue": None,
        "Operating Income": None,
        "Net Income": None,
        "Diluted EPS": None,
        "Diluted Shares Outstanding": None
    }
    missing_data = []
    short_date = date[:7]
    income_df = raw_data['income_statement'][short_date]
    labels_dict = raw_data['income_statement']['label']
    labels = list(labels_dict.values())
    if "Revenue" in labels and "Contract Revenue" in labels:
        rev_idx = str(labels.index("Revenue"))
        contract_idx = str(labels.index("Contract Revenue"))
        rev = str(income_df[rev_idx])
        contract = str(income_df[contract_idx])
        if rev == "nan" and contract == "nan":
            missing_data.append("Revenue")
        elif rev == "nan":
            label_mapping["Revenue"] = contract_idx
        elif contract == "nan":
            label_mapping["Revenue"] = rev_idx
    elif "Revenue" in labels:
        idx = str(labels.index("Revenue"))
        label_mapping["Revenue"] = idx
    elif "Contract Revenue" in labels:
        idx = str(labels.index("Contract Revenue"))
        label_mapping["Revenue"] = idx
    if "Total revenue" in labels:
        idx = str(labels.index("Total revenue"))
        label_mapping["Revenue"] = idx
    if "Total revenues" in labels:
        idx = str(labels.index("Total revenues"))
        label_mapping["Revenue"] = idx
    if label_mapping["Revenue"] is None:
        if "Total operating revenues" in labels:
            idx = str(labels.index("Total operating revenues"))
            label_mapping["Revenue"] = idx

    if "Operating Income" not in labels:
        missing_data.append("Operating Income")
    else:
        idx = str(labels.index("Operating Income"))
        label_mapping["Operating Income"] = idx
    if "Net Income" not in labels:
        missing_data.append("Net Income")
    else:
        idx = str(labels.index("Net Income"))
        label_mapping["Net Income"] = idx
    if "Earnings Per Share (Diluted)" not in labels:
        missing_data.append("Diluted EPS")
    else:
        idx = str(labels.index("Earnings Per Share (Diluted)"))
        label_mapping["Diluted EPS"] = idx
    if "Shares Outstanding (Diluted)" not in labels:
        missing_data.append("Diluted Shares Outstanding")
    else:
        idx = str(labels.index("Shares Outstanding (Diluted)"))
        label_mapping["Diluted Shares Outstanding"] = idx
    
    return label_mapping, missing_data   

def get_reported_numbers(ticker):
    results = get_company_financials(ticker)
    if results is None:
        print("No results for", ticker)
        return None 
    raw_data = json.loads(results["raw_data"])
    if "filing_metadata" not in raw_data:
        print("No filing metadata for", ticker)
        return None 
    filing_metadata = raw_data["filing_metadata"]
    type, filing = quarterly_or_annual(filing_metadata)  

    most_recent_numbers = get_most_recent_numbers(raw_data, type, filing)
    most_recent_numbers['company_name'] = results['company_name']
    if most_recent_numbers:
        return most_recent_numbers
    return None

def compute_metrics(numbers, price):
    # We want P/E, P/S, market cap 
    revenue = numbers["revenue"]
    market_cap = price * numbers["shares_outstanding"]
    pe = price / numbers["eps"]
    ps = market_cap / revenue
    return {"pe": pe, "ps": ps, "market_cap": market_cap}

def get_price(ticker):
    ALPACA_API_KEY = os.getenv("ALPACA_API")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET")

    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)    
    price_payload, _ = get_price_with_cache(alpaca_client, ticker)
    price = price_payload.get("price") or price_payload.get("latest_price")
    return price


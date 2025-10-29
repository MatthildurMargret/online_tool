import os
import json
import traceback
import re
from flask import jsonify
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_file, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from edgar import Company, set_identity
from alpaca.data.historical import StockHistoricalDataClient
from test_supabase import get_reported_numbers, compute_metrics, get_price
from utils.db_init import init_db
from utils.storage import (
    get_company_financials
)
from utils.stock_price import get_price_with_cache
from utils.formatting import clean_nans

# ---------------------------------------------------------------------
# ENVIRONMENT SETUP
# ---------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
set_identity("matthildur@montageventures.com")

ALPACA_API_KEY = os.getenv("ALPACA_API")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET")

alpaca_client = None
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ---------------------------------------------------------------------
# BASIC ENDPOINTS
# ---------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/", methods=["GET"])
def index():
    return send_file(os.path.join(FRONTEND_DIR, "new_index.html"))


@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    try:
        return send_file(os.path.join(FRONTEND_DIR, filename))
    except FileNotFoundError:
        return "File not found", 404


# ---------------------------------------------------------------------
# Basic numbers ENDPOINT
# ---------------------------------------------------------------------
@app.route("/api/financials/<ticker>", methods=["GET"])
def get_numbers(ticker):
    ticker = ticker.upper()
    try:
        numbers = get_reported_numbers(ticker)
        return jsonify({"success": True, "data": clean_nans(numbers)})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------------------------------------------------------
# Complete income statement ENDPOINT
# ---------------------------------------------------------------------
@app.route("/api/income/<ticker>", methods=["GET"])
def get_income(ticker):
    ticker = ticker.upper()
    try:
        numbers = get_company_financials(ticker)
        raw_json = numbers.get("raw_data")
        company_name = numbers.get("company_name")

        # Decode if stringified
        if isinstance(raw_json, str):
            raw_json = json.loads(raw_json)

        # Clean nulls
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() if v not in ("", None)}
            elif isinstance(d, list):
                return [clean_dict(x) for x in d if x not in ("", None)]
            return d

        cleaned = clean_dict(raw_json)

        return jsonify({
            "success": True,
            "data": {
                "ticker": ticker,
                "company_name": company_name,
                "income_statement": cleaned.get("income_statement"),
                "filing_metadata": cleaned.get("filing_metadata"),
                "raw": cleaned
            }
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------
# STOCK PRICE ENDPOINT
# ---------------------------------------------------------------------
@app.route("/api/stock-price/<ticker>", methods=["GET"])
def get_stock_price(ticker):
    ticker = ticker.upper()
    try:
        payload, cached = get_price_with_cache(alpaca_client, ticker)
        if payload is None:
            return jsonify({"success": False, "error": "No price data"}), 404

        return jsonify({"success": True, "data": payload, "cached": cached})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------------------------------------------------------
# Valuation metrics ENDPOINT
# ---------------------------------------------------------------------
@app.route("/api/valuation/<ticker>", methods=["GET"])
def get_valuation(ticker):
    ticker = ticker.upper()
    try:
        numbers = get_reported_numbers(ticker)
        price = get_price(ticker)
        metrics = compute_metrics(numbers, price)
        return jsonify({"success": True, "data": clean_nans(metrics)})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------------------------------------------------------
# SIMPLE IN-MEMORY CACHE FOR INDUSTRY STREAM
# ---------------------------------------------------------------------
from flask import jsonify
import time

_industry_cache = {}

def get_industry_cache(key: str):
    """Return cached list of rows if not expired (15 min TTL)."""
    item = _industry_cache.get(key)
    if not item:
        return None
    data, ts = item
    if time.time() - ts > 900:  # 15 minutes
        del _industry_cache[key]
        return None
    return data

def set_industry_cache(key: str, data):
    """Cache list of rows with timestamp."""
    _industry_cache[key] = (data, time.time())


# ---------------------------------------------------------------------
# INDUSTRY MAP ENDPOINT
# ---------------------------------------------------------------------
@app.route("/api/industry-map", methods=["GET"])
def get_industry_map():
    import os, json
    from flask import jsonify

    try:
        # Go one level up from backend/ to reach the top-level "data" folder
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(root_dir, "data", "industry_map.json")

        if not os.path.exists(data_path):
            return jsonify({"success": False, "error": f"File not found at {data_path}"}), 404

        with open(data_path, "r") as f:
            data = json.load(f)

        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------
# INDUSTRY MAP ENDPOINTS
# ---------------------------------------------------------------------

@app.route("/api/industry-tickers/<path:industry_path>", methods=["GET"])
def get_industry_tickers(industry_path):
    try:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "industry_map.json")
        with open(path, "r") as f:
            industry_map = json.load(f)

        parts = industry_path.split("/")
        if len(parts) not in (2, 3):
            return jsonify({"success": False, "error": "Invalid path format"}), 400

        industry, sector = parts[0], parts[1]
        industry_data = industry_map["industries"][industry]["sectors"][sector]

        if len(parts) == 3:
            group = parts[2]
            tickers = industry_data["industry_groups"][group]
        else:
            # No group → collect all tickers under this sector
            tickers = []
            for g in industry_data["industry_groups"].values():
                tickers.extend(g)

        return jsonify({
            "success": True,
            "data": {
                "industry": industry,
                "sector": sector,
                "industry_group": parts[2] if len(parts) == 3 else None,
                "tickers": tickers,
            },
        })
    except KeyError as e:
        return jsonify({"success": False, "error": f"Path not found: {e}"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------
# INDUSTRY COMPARISON STREAM ENDPOINT (new version)
# ---------------------------------------------------------------------

@app.route("/api/industry-comparison-stream", methods=["POST"])
def industry_comparison_stream():
    def generate():
        try:
            payload = request.get_json(force=True)
            tickers = [t.upper() for t in payload.get("tickers", [])]
            use_cache = payload.get("use_cache", True)
            if not tickers:
                yield json.dumps({"type": "error", "error": "No tickers provided"}) + "\n"
                return

            cache_key = "tickers:" + ",".join(sorted(tickers))
            print(f"[STREAM] Starting industry comparison for {len(tickers)} tickers")

            # Optional: pull cached batch
            if use_cache:
                cached = get_industry_cache(cache_key)
                if cached:
                    for row in cached:
                        yield json.dumps({"type": "company", "data": row}) + "\n"
                    yield json.dumps({"type": "complete", "processed": len(cached), "total": len(cached)}) + "\n"
                    return

            results = []
            processed = 0

            for ticker in tickers:
                try:
                    # Step 1. get clean financials
                    numbers = get_reported_numbers(ticker)
                    if not numbers or not numbers.get("revenue") or not numbers.get("eps"):
                        yield json.dumps({"type": "skip", "ticker": ticker, "reason": "Missing data"}) + "\n"
                        processed += 1
                        continue

                    # Step 2. get latest price
                    price = get_price(ticker)
                    if not price:
                        yield json.dumps({"type": "skip", "ticker": ticker, "reason": "No price"}) + "\n"
                        processed += 1
                        continue

                    # Step 3. compute metrics
                    metrics = compute_metrics(numbers, price)
                    if not metrics:
                        yield json.dumps({"type": "skip", "ticker": ticker, "reason": "Metrics failed"}) + "\n"
                        processed += 1
                        continue

                    row = {
                        "ticker": ticker,
                        "company_name": numbers.get("company_name"),
                        "revenue": numbers["revenue"],
                        "net_income": numbers.get("net_income"),
                        "eps": numbers.get("eps"),
                        "shares_outstanding": numbers.get("shares_outstanding"),
                        "pe": metrics["pe"],
                        "ps": metrics["ps"],
                        "market_cap": metrics["market_cap"],
                    }

                    results.append(row)
                    yield json.dumps({"type": "company", "data": row}) + "\n"

                except Exception as e:
                    print(f"[STREAM] Error {ticker}: {e}")
                    yield json.dumps({"type": "error", "ticker": ticker, "error": str(e)}) + "\n"

                processed += 1
                yield json.dumps({"type": "progress", "processed": processed, "total": len(tickers)}) + "\n"

            if results:
                set_industry_cache(cache_key, results)

            yield json.dumps({"type": "complete", "processed": processed, "total": len(tickers)}) + "\n"

        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"type": "error", "error": str(e)}) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------
# APP ENTRY POINT
# ---------------------------------------------------------------------

init_db()

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5001))
    print("\n" + "=" * 60)
    print("🚀 Valuation Tool – Cleaned Backend")
    print("=" * 60)
    print(f"📊 Running on port {port}")
    print("=" * 60 + "\n")
    app.run(debug=True, port=port, host="0.0.0.0", threaded=True, use_reloader=False)
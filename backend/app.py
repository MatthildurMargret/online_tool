import os
import json
import traceback
import re
import logging
import requests
from flask import jsonify
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_file, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.serving import WSGIRequestHandler
from edgar import Company, set_identity
from alpaca.data.historical import StockHistoricalDataClient
from test_supabase import get_reported_numbers, compute_metrics, get_price
from utils.db_init import init_db
from utils.storage import (
    get_company_financials
)
from supabase_db import (
    get_early_deals,
    get_deals_categories,
    get_deals_funding_rounds,
    get_interesting_people,
    get_founder_contact_status,
    upsert_founder_contact_status,
    get_taste_tree,
    update_taste_tree_lead,
    get_all_users_from_tree,
    get_user_tags,
    get_all_category_paths,
    batch_update_user_tags,
    get_portfolio_companies,
    get_portfolio_company_tickers,
    get_all_tickers_with_financials,
    get_gemini_financials,
    get_gemini_financials_batch,
    is_gemini_financials_fresh,
    upsert_gemini_financials,
)
from utils.stock_price import get_price_with_cache
from utils.formatting import clean_nans
from utils.portfolio_metrics import build_portfolio_output_row
from utils.prospecting import build_query, google_search

# ---------------------------------------------------------------------
# ENVIRONMENT SETUP
# ---------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)

# Suppress werkzeug access logs completely
logging.getLogger('werkzeug').disabled = True

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

# ---------------------------------------------------------------------
# FOUNDERS API PROXY (avoids CORS by proxying external Founders API)
# ---------------------------------------------------------------------
FOUNDERS_API_BASE = os.getenv("FOUNDERS_API_URL", "https://monty-api-production.up.railway.app")
FOUNDERS_API_KEY = os.getenv("FOUNDERS_API_KEY", "5f8e7ac4f6f8a3b6b2d91c66e01d0f7d1a91f4f4bfc1f3263e57b85c14f6a733")


@app.route("/api/founders-proxy/<path:path>", methods=["GET"])
def founders_proxy(path: str):
    """Proxy requests to the external Founders API to avoid CORS."""
    url = f"{FOUNDERS_API_BASE.rstrip('/')}/{path}"
    if request.query_string:
        url = f"{url}?{request.query_string.decode()}"
    try:
        resp = requests.get(
            url,
            headers={"x-api-key": FOUNDERS_API_KEY},
            timeout=30,
        )
        return Response(
            resp.content,
            status=resp.status_code,
            mimetype="application/json",
        )
    except requests.RequestException as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 502

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
# SINGLE STOCK ENDPOINT (Gemini + Alpaca pipeline, same as portfolio comps)
# ---------------------------------------------------------------------
@app.route("/api/single-stock/<ticker>", methods=["GET"])
def get_single_stock(ticker):
    """Get financial data for one ticker using Gemini + Alpaca (same pipeline as portfolio comps)."""
    ticker = ticker.upper()
    try:
        # 1. Get price from Alpaca
        price_payload, _ = get_price_with_cache(alpaca_client, ticker)
        price = (price_payload or {}).get("price") if price_payload else None
        if not price:
            return jsonify({"success": False, "error": "No price data"}), 404

        # 2. Get Gemini data (cached or fetch)
        from services.gemini_financials import fetch_ticker_via_gemini

        row = get_gemini_financials(ticker)
        if not row or not is_gemini_financials_fresh(row):
            data = fetch_ticker_via_gemini(ticker)
            if not data or not data.get("revenue") or not data.get("shares_outstanding"):
                return jsonify({"success": False, "error": "Missing financial data"}), 404
            upsert_gemini_financials(data)
            row = data

        if not row.get("revenue") or not row.get("shares_outstanding"):
            return jsonify({"success": False, "error": "Missing financial data"}), 404

        # 3. Build output (same as portfolio stream)
        out = build_portfolio_output_row(ticker, row, price, clean_nans)
        out["price"] = price  # Include for single-stock display
        return jsonify({"success": True, "data": out})
    except Exception as e:
        traceback.print_exc()
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

            # Yield immediately so the client gets response headers and can show loading UI
            yield json.dumps({"type": "progress", "processed": 0, "total": len(tickers)}) + "\n"
            cache_key = "tickers:" + ",".join(sorted(tickers))
            print(f"[STREAM] Starting industry comparison for {len(tickers)} tickers")

            # Optional: pull cached batch
            if use_cache:
                cached = get_industry_cache(cache_key)
                if cached:
                    for row in cached:
                        yield json.dumps({"type": "company", "data": clean_nans(row)}) + "\n"
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

                    row = clean_nans({
                        "ticker": ticker,
                        "company_name": numbers.get("company_name"),
                        "revenue": numbers["revenue"],
                        "net_income": numbers.get("net_income"),
                        "eps": numbers.get("eps"),
                        "shares_outstanding": numbers.get("shares_outstanding"),
                        "pe": metrics["pe"],
                        "ps": metrics["ps"],
                        "market_cap": metrics["market_cap"],
                    })

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
# PORTFOLIO / VERTICAL COMPARISON STREAM (Gemini-sourced)
# ---------------------------------------------------------------------


@app.route("/api/portfolio-comparison-stream", methods=["POST"])
def portfolio_comparison_stream():
    def generate():
        try:
            payload = request.get_json(force=True) or {}
            tickers = [t.upper() for t in payload.get("tickers", [])]
            if not tickers:
                yield json.dumps({"type": "error", "error": "No tickers provided"}) + "\n"
                return

            # Yield immediately so the client gets response headers and can show loading UI
            yield json.dumps({"type": "progress", "processed": 0, "total": len(tickers)}) + "\n"
            print(f"[STREAM] Starting portfolio comparison for {len(tickers)} tickers")

            from services.gemini_financials import fetch_ticker_via_gemini

            cached = get_gemini_financials_batch(tickers)
            processed = 0
            gemini_calls_made = 0

            for ticker in tickers:
                try:
                    # 1. Get price from Alpaca (same as industry comps)
                    price_payload, _ = get_price_with_cache(alpaca_client, ticker)
                    price = (price_payload or {}).get("price") if price_payload else None
                    if not price:
                        yield json.dumps({"type": "skip", "ticker": ticker, "reason": "No price"}) + "\n"
                        processed += 1
                        continue

                    # 2. Get Gemini data (cached or fetch)
                    row = cached.get(ticker)
                    if not row or not is_gemini_financials_fresh(row):
                        import time
                        if gemini_calls_made > 0:
                            time.sleep(5)
                        data = fetch_ticker_via_gemini(ticker)
                        gemini_calls_made += 1
                        if not data or not data.get("revenue") or not data.get("shares_outstanding"):
                            yield json.dumps({"type": "skip", "ticker": ticker, "reason": "Missing data"}) + "\n"
                            processed += 1
                            continue
                        upsert_gemini_financials(data)
                        row = data

                    if not row.get("revenue") or not row.get("shares_outstanding"):
                        yield json.dumps({"type": "skip", "ticker": ticker, "reason": "Missing data"}) + "\n"
                        processed += 1
                        continue

                    # 3. Build output (same logic as test script: compute metrics from Gemini + price)
                    out = build_portfolio_output_row(ticker, row, price, clean_nans)
                    yield json.dumps({"type": "company", "data": out}) + "\n"

                except Exception as e:
                    print(f"[STREAM] Error {ticker}: {e}")
                    yield json.dumps({"type": "skip", "ticker": ticker, "reason": str(e)}) + "\n"

                processed += 1
                yield json.dumps({"type": "progress", "processed": processed, "total": len(tickers)}) + "\n"

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
# VERTICAL / PORTFOLIO COMPS ENDPOINTS
# ---------------------------------------------------------------------

_ticker_index_cache = None
_ticker_index_cache_ts = 0
TICKER_INDEX_TTL = 3600  # 1 hour


def _build_ticker_index():
    """Build ticker -> (company_name, industry, sector, group) for LLM context. Cached."""
    global _ticker_index_cache, _ticker_index_cache_ts
    if _ticker_index_cache is not None and (time.time() - _ticker_index_cache_ts) < TICKER_INDEX_TTL:
        return _ticker_index_cache

    # Load industry map for reverse lookup
    industry_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "industry_map.json")
    ticker_to_industry = {}
    if os.path.exists(industry_path):
        with open(industry_path, "r") as f:
            industry_map = json.load(f)
        for industry, ind_data in industry_map.get("industries", {}).items():
            for sector, sec_data in ind_data.get("sectors", {}).items():
                for group, tickers in sec_data.get("industry_groups", {}).items():
                    for t in tickers:
                        ticker_to_industry[t.upper()] = (industry, sector, group)

    # Get all tickers with financials and company names
    valid_tickers = set(get_all_tickers_with_financials())
    index = []
    for ticker in sorted(valid_tickers):
        fin = get_company_financials(ticker)
        company_name = (fin or {}).get("company_name") or ticker
        ind_info = ticker_to_industry.get(ticker, (None, None, None))
        industry, sector, group = ind_info
        ctx = f"{ticker} - {company_name}"
        if industry:
            ctx += f" ({industry}, {sector}, {group})"
        index.append({"ticker": ticker, "context": ctx})

    _ticker_index_cache = index
    _ticker_index_cache_ts = time.time()
    return index


@app.route("/api/portfolio-companies", methods=["GET"])
def get_portfolio_companies_endpoint():
    """Get list of portfolio companies for the comps dropdown."""
    try:
        companies = get_portfolio_companies()
        return jsonify({"success": True, "data": companies})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/portfolio-companies/<company_id>/tickers", methods=["GET"])
def get_portfolio_company_tickers_endpoint(company_id):
    """Get tickers for a portfolio company, filtered to those in our financial DB."""
    try:
        tickers = get_portfolio_company_tickers(company_id)
        return jsonify({"success": True, "data": {"tickers": tickers}})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/vertical-tickers", methods=["POST"])
def get_vertical_tickers():
    """Use Gemini with grounding to suggest 5-6 relevant public tickers for a vertical."""
    try:
        payload = request.get_json(force=True) or {}
        vertical = (payload.get("vertical") or "").strip()
        if not vertical:
            return jsonify({"success": False, "error": "vertical is required"}), 400

        if not os.getenv("GEMINI_API_KEY"):
            return jsonify({"success": False, "error": "GEMINI_API_KEY not configured"}), 503

        from services.gemini_financials import suggest_tickers_for_vertical

        tickers = suggest_tickers_for_vertical(vertical)
        return jsonify({"success": True, "data": {"tickers": tickers}})
    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"Invalid LLM response: {e}"}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 503


# ---------------------------------------------------------------------
# EARLY DEALS ENDPOINTS
# ---------------------------------------------------------------------

@app.route("/api/deals", methods=["GET"])
def get_deals():
    """Get early deals with optional filtering by category and funding round (supports multiple values)"""
    try:
        # Get lists of categories and funding rounds (supports multiple selections)
        categories = request.args.getlist("category")
        funding_rounds = request.args.getlist("funding_round")
        
        # Convert empty lists to None for backward compatibility
        category = categories if categories else None
        funding_round = funding_rounds if funding_rounds else None
        
        deals = get_early_deals(category=category, funding_round=funding_round)
        return jsonify({"success": True, "data": deals})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/deals/filters", methods=["GET"])
def get_deals_filters():
    """Get available filter options for deals (categories and funding rounds)"""
    try:
        categories = get_deals_categories()
        funding_rounds = get_deals_funding_rounds()
        return jsonify({
            "success": True,
            "data": {
                "categories": categories,
                "funding_rounds": funding_rounds
            }
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------
# INTERESTING PEOPLE ENDPOINTS
# ---------------------------------------------------------------------

@app.route("/api/interesting-people", methods=["GET"])
def get_interesting_people_endpoint():
    """Get all interesting people from the database"""
    try:
        people = get_interesting_people()
        return jsonify({"success": True, "data": people})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------
# FOUNDER CONTACT STATUS ENDPOINTS
# ---------------------------------------------------------------------

@app.route("/api/founders/contact-status", methods=["GET", "POST"])
def get_founder_contact_status_endpoint():
    """Retrieve contact status for a list of sourcing entries
    
    Supports both GET (for backward compatibility) and POST (recommended for large requests).
    POST sends IDs in request body as JSON: {"ids": ["id1", "id2", ...]}
    """
    # Try POST body first (preferred for large requests)
    if request.method == "POST":
        payload = request.get_json(force=True) or {}
        entry_ids = payload.get("ids", [])
        if not isinstance(entry_ids, list):
            entry_ids = []
    else:
        # GET method - backward compatibility
        raw_ids = request.args.getlist("ids")
        if len(raw_ids) == 1 and "," in raw_ids[0]:
            raw_ids = [value.strip() for value in raw_ids[0].split(",") if value.strip()]
        entry_ids = [value for value in raw_ids if value]

    # Filter out empty values
    entry_ids = [eid for eid in entry_ids if eid and str(eid).strip()]

    if not entry_ids:
        return jsonify({"success": True, "data": {}})

    try:
        data = get_founder_contact_status(entry_ids)
        return jsonify({"success": True, "data": data})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/founders/contact-status/<path:entry_id>", methods=["PUT"])
def upsert_founder_contact_status_endpoint(entry_id: str):
    """Persist contact status for a specific sourcing entry"""
    try:
        payload = request.get_json(force=True) or {}
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid JSON payload: {e}"}), 400

    contacted_raw = payload.get("contacted")
    contacted = bool(contacted_raw) if contacted_raw is not None else None
    contacted_by = payload.get("contacted_by")
    in_pipeline_raw = payload.get("in_pipeline")
    in_pipeline = bool(in_pipeline_raw) if in_pipeline_raw is not None else None

    try:
        record = upsert_founder_contact_status(
            entry_id,
            contacted=contacted,
            contacted_by=contacted_by,
            in_pipeline=in_pipeline,
        )
        if record is None:
            return jsonify({"success": False, "error": "Failed to update contact status"}), 500
        return jsonify({"success": True, "data": record})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------
# PROSPECTING ENDPOINT
# ---------------------------------------------------------------------

@app.route("/api/prospecting/search", methods=["POST"])
def search_prospects():
    """Search for prospects using Google Custom Search"""
    try:
        payload = request.get_json(force=True)
        keywords = payload.get("keywords", "").strip()
        role = payload.get("role", "").strip()
        company = payload.get("company", "").strip()
        
        # Build the query
        query = build_query(keywords, role, company)
        
        # Perform the search
        results = google_search(query)
        
        return jsonify({
            "success": True,
            "data": results,
            "query": query
        })
    except ValueError as e:
        # Missing API keys
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------
# TASTE TREE ENDPOINTS
# ---------------------------------------------------------------------

@app.route("/api/taste-tree", methods=["GET"])
def get_taste_tree_endpoint():
    """Get the latest taste_tree data from Supabase"""
    try:
        record = get_taste_tree()
        if not record:
            return jsonify({"success": False, "error": "No taste_tree record found"}), 404
        
        return jsonify({
            "success": True,
            "data": record.get("data"),
            "version": record.get("version"),
            "updated_at": record.get("updated_at")
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/taste-tree/update-lead", methods=["PUT"])
def update_taste_tree_lead_endpoint():
    """Update montage_lead for a specific node in the taste_tree"""
    try:
        payload = request.get_json(force=True) or {}
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid JSON payload: {e}"}), 400
    
    category_path = payload.get("category_path")
    montage_lead = payload.get("montage_lead")
    
    if not category_path or not isinstance(category_path, list):
        return jsonify({"success": False, "error": "category_path must be a non-empty array"}), 400
    
    # Allow montage_lead to be None or empty string to clear it
    if montage_lead is not None and not isinstance(montage_lead, str):
        return jsonify({"success": False, "error": "montage_lead must be a string or null"}), 400
    
    try:
        success = update_taste_tree_lead(category_path, montage_lead)
        if not success:
            return jsonify({"success": False, "error": "Failed to update taste tree"}), 500
        
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/taste-tree/users", methods=["GET"])
def get_taste_tree_users_endpoint():
    """Get all unique users (montage_lead values) from the taste tree"""
    try:
        users = get_all_users_from_tree()
        return jsonify({"success": True, "data": users})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/taste-tree/user-tags/<path:user_name>", methods=["GET"])
def get_user_tags_endpoint(user_name: str):
    """Get all category paths where a specific user is tagged"""
    try:
        # URL decode the user name
        from urllib.parse import unquote
        user_name = unquote(user_name)
        
        tags = get_user_tags(user_name)
        return jsonify({"success": True, "data": tags})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/taste-tree/all-categories", methods=["GET"])
def get_all_categories_endpoint():
    """Get a flat list of all categories with their paths"""
    try:
        categories = get_all_category_paths()
        return jsonify({"success": True, "data": categories})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/taste-tree/batch-update-user", methods=["PUT"])
def batch_update_user_tags_endpoint():
    """Batch update montage_lead for multiple categories"""
    try:
        payload = request.get_json(force=True) or {}
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid JSON payload: {e}"}), 400
    
    user_name = payload.get("user_name")
    add_paths = payload.get("add_paths", [])
    remove_paths = payload.get("remove_paths", [])
    
    if not user_name or not isinstance(user_name, str):
        return jsonify({"success": False, "error": "user_name must be a non-empty string"}), 400
    
    if not isinstance(add_paths, list) or not isinstance(remove_paths, list):
        return jsonify({"success": False, "error": "add_paths and remove_paths must be arrays"}), 400
    
    # Validate that all paths are lists
    for path in add_paths + remove_paths:
        if not isinstance(path, list):
            return jsonify({"success": False, "error": "All paths must be arrays"}), 400
    
    try:
        success = batch_update_user_tags(user_name, add_paths, remove_paths)
        if not success:
            return jsonify({"success": False, "error": "Failed to batch update user tags"}), 500
        
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------
# STATIC FILE SERVING (must be last to not interfere with API routes)
# ---------------------------------------------------------------------

@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    try:
        return send_file(os.path.join(FRONTEND_DIR, filename))
    except FileNotFoundError:
        return "File not found", 404


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
    
    # Disable werkzeug request logging completely to avoid console spam
    # The contact status endpoint generates very long URLs that clutter the console
    import werkzeug.serving
    werkzeug.serving.WSGIRequestHandler.log_request = lambda *args, **kwargs: None
    
    app.run(debug=False, port=port, host="0.0.0.0", threaded=True, use_reloader=False)
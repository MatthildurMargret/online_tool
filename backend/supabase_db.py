"""
Supabase database helper for company_financials table
Handles all interactions with the Supabase database
"""

import os
import re
import copy
from datetime import datetime
from typing import Optional, Dict, List, Union
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}")


def is_supabase_configured() -> bool:
    """Check if Supabase is properly configured"""
    return supabase is not None


def store_company_financials(
    ticker: str,
    company_name: str,
    revenue: Optional[float],
    shares_outstanding: Optional[float],
    filing_date: str,
    period_end_date: str,
    filing_type: str,
    raw_data: Optional[str] = None,
    valuation_metrics: Optional[Dict] = None,
    income: Optional[float] = None,
    latest_eps: Optional[float] = None,
    revenue_label: Optional[str] = None,
    income_label: Optional[str] = None
) -> bool:
    """
    Store company financial data in Supabase
    Returns True if successful, False otherwise
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        data = {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "revenue": revenue,
            "shares_outstanding": shares_outstanding,
            "filing_date": filing_date,
            "period_end_date": period_end_date,
            "filing_type": filing_type,
            "last_updated": datetime.now().isoformat(),
            "raw_data": raw_data,
            "income": income,
            "latest_eps": latest_eps,
            "revenue_label": revenue_label,
            "income_label": income_label
        }
        
        # Add valuation_metrics if provided
        if valuation_metrics is not None:
            import json
            data["valuation_metrics"] = json.dumps(valuation_metrics)
        
        # Upsert (insert or update if exists)
        result = supabase.table("company_financials").upsert(data).execute()
        return True
    except Exception as e:
        print(f"Error storing financial data for {ticker}: {e}")
        return False


def get_company_financials(ticker: str) -> Optional[Dict]:
    """
    Retrieve company financial data from Supabase
    Returns normalized dict with financial data or None if not found
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = supabase.table("company_financials").select(
            "ticker, company_name, revenue, income, shares_outstanding, latest_eps, filing_date, period_end_date, filing_type, last_updated, raw_data"
        ).eq("ticker", ticker.upper()).execute()
        
        if not result.data:
            return None
        
        row = result.data[0]
        # ✅ Normalize key names for consistency with SQLite fallback
        return {
            "ticker": row.get("ticker"),
            "company_name": row.get("company_name"),
            "revenue": row.get("revenue"),
            "income": row.get("income"),
            "shares_outstanding": row.get("shares_outstanding"),
            "latest_eps": row.get("latest_eps"),
            "filing_date": row.get("filing_date"),
            "period_end": row.get("period_end_date"),
            "filing_type": row.get("filing_type"),
            "last_updated": row.get("last_updated"),
            "raw_data": row.get("raw_data"),
        }
    except Exception as e:
        print(f"Error retrieving financial data for {ticker}: {e}")
        return None



def get_all_tickers_with_financials() -> List[str]:
    """
    Get list of all tickers that have financial data stored in Supabase
    Returns list of ticker symbols
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = supabase.table("company_financials").select("ticker").order("ticker").execute()
        return [row["ticker"] for row in result.data]
    except Exception as e:
        print(f"Error retrieving tickers: {e}")
        return []


def check_ticker_exists(ticker: str) -> bool:
    """
    Check if a ticker already exists in the database
    Returns True if exists, False otherwise
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = supabase.table("company_financials").select("ticker").eq("ticker", ticker.upper()).execute()
        return len(result.data) > 0
    except Exception as e:
        print(f"Error checking ticker existence: {e}")
        return False


def get_all_company_financials() -> List[Dict]:
    """
    Retrieve all company financial data from Supabase
    Returns list of dicts with financial data
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = supabase.table("company_financials").select("*").execute()
        return result.data
    except Exception as e:
        print(f"Error retrieving all financial data: {e}")
        return []


def update_valuation_metrics(ticker: str, valuation_metrics: Dict) -> bool:
    """
    Update only the valuation_metrics field for a ticker
    Returns True if successful, False otherwise
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        import json
        data = {
            "ticker": ticker.upper(),
            "valuation_metrics": json.dumps(valuation_metrics)
        }
        
        result = supabase.table("company_financials").upsert(data).execute()
        return True
    except Exception as e:
        print(f"Error updating valuation metrics for {ticker}: {e}")
        return False


def get_early_deals(category: Optional[Union[str, List[str]]] = None, funding_round: Optional[Union[str, List[str]]] = None) -> List[Dict]:
    """
    Retrieve early deals from Supabase
    Optional filters: category, funding_round (both case-insensitive, supports lists)
    Returns list of deal dictionaries, with header rows filtered out
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        # Fetch all deals (we'll filter in Python for case-insensitive matching)
        query = supabase.table("early_deals").select("*")
        
        # Order by date descending (most recent first)
        result = query.order("Date", desc=True).execute()
        deals = result.data
        
        # Filter out header rows (rows where Company field matches common header strings)
        header_keywords = ["company", "company name", "name"]
        filtered_deals = []
        for deal in deals:
            company = str(deal.get("Company", "")).strip().lower()
            # Skip if company name matches header keywords
            if company and company not in header_keywords:
                # Additional check: skip if multiple fields match their column names (header row)
                matches = sum(1 for key, value in deal.items() 
                            if value and str(value).strip().lower() == str(key).strip().lower())
                if matches < 2:  # If less than 2 fields match column names, it's likely a data row
                    filtered_deals.append(deal)
        
        # Apply case-insensitive filters (support both single values and lists)
        if category:
            # Handle both list and single value
            if isinstance(category, list):
                category_lower_set = {c.lower().strip() for c in category if c}
            else:
                category_lower_set = {category.lower().strip()}
            
            if category_lower_set:
                filtered_deals = [d for d in filtered_deals 
                                if d.get("Category") and str(d.get("Category")).lower().strip() in category_lower_set]
        
        if funding_round:
            # Handle both list and single value - normalize filter values
            if isinstance(funding_round, list):
                normalized_filters = {normalize_funding_round(fr) for fr in funding_round if fr}
            else:
                normalized_filters = {normalize_funding_round(funding_round)}
            
            if normalized_filters:
                filtered_deals = [d for d in filtered_deals 
                                if d.get("Funding Round") and normalize_funding_round(str(d.get("Funding Round"))) in normalized_filters]
        
        return filtered_deals
    except Exception as e:
        print(f"Error retrieving early deals: {e}")
        return []


def get_deals_categories() -> List[str]:
    """
    Get unique list of categories from early_deals table (case-insensitive deduplication)
    Returns sorted list of category strings with original capitalization preserved
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        # Select all columns to avoid issues with column names containing spaces
        result = supabase.table("early_deals").select("*").execute()
        categories_raw = [row.get("Category") for row in result.data if row.get("Category")]
        
        # Filter out header rows (skip if category matches "Category" or similar)
        categories_raw = [c for c in categories_raw 
                         if str(c).strip().lower() not in ["category", "categories", ""]]
        
        # Case-insensitive deduplication: keep first occurrence of each unique lowercase value
        seen_lower = set()
        unique_categories = []
        for cat in categories_raw:
            cat_str = str(cat).strip()
            cat_lower = cat_str.lower()
            if cat_lower not in seen_lower and cat_str:
                seen_lower.add(cat_lower)
                unique_categories.append(cat_str)
        
        return sorted(unique_categories)
    except Exception as e:
        print(f"Error retrieving deal categories: {e}")
        return []


def normalize_funding_round(round_str: str) -> str:
    """
    Normalize funding round strings to combine variations of 'unknown'/'unspecified'
    Examples: 'Not specified', 'Unknown', 'unspecified', 'not specified (some text)', 
              'Series Unknown', 'Series Unspecified' -> 'Unknown/Unspecified'
    """
    if not round_str:
        return ""
    
    # Remove parentheses and their contents, then strip
    cleaned = re.sub(r'\([^)]*\)', '', str(round_str)).strip()
    cleaned_lower = cleaned.lower()
    
    # Check if it's a variant of unknown/unspecified
    unknown_variants = [
        "unknown", "unspecified", "not specified", 
        "series unknown", "series unspecified"
    ]
    
    for variant in unknown_variants:
        if variant in cleaned_lower:
            return "Unknown/Unspecified"
    
    # Return the cleaned original if not a variant
    return cleaned


def get_deals_funding_rounds() -> List[str]:
    """
    Get unique list of funding rounds from early_deals table (case-insensitive deduplication)
    Combines variations of 'unknown'/'unspecified' into a single option
    Returns sorted list of funding round strings
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        # Select all columns to avoid issues with column names containing spaces
        result = supabase.table("early_deals").select("*").execute()
        rounds_raw = [row.get("Funding Round") for row in result.data if row.get("Funding Round")]
        
        # Filter out header rows (skip if funding round matches "Funding Round" or similar)
        header_keywords = ["funding round", "funding rounds", "round", "rounds", ""]
        rounds_raw = [r for r in rounds_raw 
                     if str(r).strip().lower() not in header_keywords]
        
        # Normalize and deduplicate
        normalized_map = {}  # maps normalized -> original (first occurrence)
        for round_val in rounds_raw:
            round_str = str(round_val).strip()
            if not round_str:
                continue
            
            normalized = normalize_funding_round(round_str)
            if normalized and normalized not in normalized_map:
                normalized_map[normalized] = round_str
        
        # Return sorted list of normalized values
        return sorted(normalized_map.keys())
    except Exception as e:
        print(f"Error retrieving funding rounds: {e}")
        return []


def get_interesting_people() -> List[Dict]:
    """
    Retrieve all interesting people from Supabase
    Returns list of person dictionaries ordered by created_at descending
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = supabase.table("interesting_people").select("*").order("created_at", desc=True).execute()
        return result.data
    except Exception as e:
        print(f"Error retrieving interesting people: {e}")
        return []


def normalize_entry_id(entry_id: str) -> str:
    """
    Normalize entry ID to ensure consistent matching.
    For LinkedIn URLs, removes protocol and www prefix, normalizes domain to lowercase.
    """
    if not entry_id:
        return entry_id
    
    normalized = entry_id.strip()
    
    # Normalize LinkedIn URLs to a consistent format
    if 'linkedin.com' in normalized.lower():
        # Remove protocol (http://, https://)
        import re
        normalized = re.sub(r'^https?://', '', normalized, flags=re.IGNORECASE)
        # Remove www. prefix
        normalized = re.sub(r'^www\.', '', normalized, flags=re.IGNORECASE)
        # Ensure domain is lowercase but preserve path case
        if normalized.lower().startswith('linkedin.com'):
            normalized = 'linkedin.com' + normalized[len('linkedin.com'):]
    
    return normalized


def get_founder_contact_status(entry_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch contact status for a list of sourcing entry_ids.
    Returns mapping of entry_id -> status payload.
    Normalizes entry IDs for consistent matching and tries multiple format variations.
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")

    if not entry_ids:
        return {}

    # Build set of all IDs to try (normalized + original + variations)
    all_ids_to_try = set()
    id_mapping = {}  # Maps normalized ID to list of original requested IDs
    
    for eid in entry_ids:
        if not eid:
            continue
        eid = eid.strip()
        normalized = normalize_entry_id(eid)
        
        # Track which original IDs map to this normalized ID
        if normalized not in id_mapping:
            id_mapping[normalized] = []
        id_mapping[normalized].append(eid)
        
        # Add normalized and original to query set
        all_ids_to_try.add(normalized)
        all_ids_to_try.add(eid)
        
        # For LinkedIn URLs, add common variations
        if 'linkedin.com' in eid.lower():
            if not eid.lower().startswith('http'):
                all_ids_to_try.add(f'https://{eid}')
                all_ids_to_try.add(f'http://{eid}')
            if not eid.lower().startswith('www.'):
                all_ids_to_try.add(f'www.{eid}')

    try:
        # Debug logging
        print(f"[get_founder_contact_status] Requested {len(entry_ids)} entry IDs: {entry_ids[:3]}...")
        print(f"[get_founder_contact_status] Querying with {len(all_ids_to_try)} variations: {list(all_ids_to_try)[:5]}...")
        
        # Debug: Check what's actually in the database
        if len(entry_ids) > 0:
            sample_id = entry_ids[0]
            print(f"[get_founder_contact_status] DEBUG: Sample requested ID: '{sample_id}'")
            print(f"[get_founder_contact_status] DEBUG: Normalized sample: '{normalize_entry_id(sample_id)}'")
            
            # Try exact match first
            try:
                exact_result = (
                    supabase.table("founder_contact_status")
                    .select("entry_id")
                    .eq("entry_id", sample_id)
                    .limit(1)
                    .execute()
                )
                if exact_result.data:
                    print(f"[get_founder_contact_status] DEBUG: Found exact match: {exact_result.data[0]['entry_id']}")
                else:
                    print(f"[get_founder_contact_status] DEBUG: No exact match found")
                    
                # Try with normalized version
                normalized_sample = normalize_entry_id(sample_id)
                if normalized_sample != sample_id:
                    norm_result = (
                        supabase.table("founder_contact_status")
                        .select("entry_id")
                        .eq("entry_id", normalized_sample)
                        .limit(1)
                        .execute()
                    )
                    if norm_result.data:
                        print(f"[get_founder_contact_status] DEBUG: Found normalized match: {norm_result.data[0]['entry_id']}")
                
                # Get a sample of what's actually in the database
                sample_db = (
                    supabase.table("founder_contact_status")
                    .select("entry_id")
                    .limit(5)
                    .execute()
                )
                if sample_db.data:
                    print(f"[get_founder_contact_status] DEBUG: Sample entry_ids in DB: {[r['entry_id'] for r in sample_db.data]}")
                else:
                    print(f"[get_founder_contact_status] DEBUG: No records found in founder_contact_status table at all!")
                    
            except Exception as debug_e:
                import traceback
                print(f"[get_founder_contact_status] DEBUG query error: {debug_e}")
                print(traceback.format_exc())
        
        # Supabase .in_() has a limit, so batch queries if needed (limit is typically 100)
        BATCH_SIZE = 100
        all_ids_list = list(all_ids_to_try)
        result_map = {}
        
        for i in range(0, len(all_ids_list), BATCH_SIZE):
            batch = all_ids_list[i:i + BATCH_SIZE]
            print(f"[get_founder_contact_status] Querying batch {i//BATCH_SIZE + 1} with {len(batch)} IDs")
            
            result = (
                supabase.table("founder_contact_status")
                .select("entry_id, contacted, contacted_at, contacted_by, in_pipeline, in_pipeline_at")
                .in_("entry_id", batch)
                .execute()
            )
            
            print(f"[get_founder_contact_status] Batch returned {len(result.data or [])} rows")
            if result.data:
                print(f"[get_founder_contact_status] Sample returned entry_ids: {[r['entry_id'] for r in result.data[:3]]}")
            
            # Build result map: for each found record, map it to all requested IDs that match
            for row in result.data or []:
                db_entry_id = row["entry_id"]
                db_normalized = normalize_entry_id(db_entry_id)
                
                print(f"[get_founder_contact_status] Processing DB entry_id: {db_entry_id}, normalized: {db_normalized}")
                
                # Map this record to all requested IDs that normalize to the same value
                if db_normalized in id_mapping:
                    for req_id in id_mapping[db_normalized]:
                        result_map[req_id] = row
                # Also map by normalized key and original DB key
                result_map[db_normalized] = row
                result_map[db_entry_id] = row
        
        print(f"[get_founder_contact_status] Final result map has {len(result_map)} entries")
        return result_map
    except Exception as e:
        import traceback
        print(f"Error retrieving founder contact status: {e}")
        print(traceback.format_exc())
        return {}


def upsert_founder_contact_status(
    entry_id: str,
    contacted: Optional[bool] = None,
    contacted_by: Optional[str] = None,
    in_pipeline: Optional[bool] = None,
) -> Optional[Dict]:
    """
    Persist outreach status for a sourcing entry.
    When a status boolean is True, the corresponding *_at timestamp is set to current UTC time.
    Normalizes entry_id before storing to ensure consistency.
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")

    if not entry_id:
        raise ValueError("entry_id is required")

    # Normalize entry_id before storing
    normalized_entry_id = normalize_entry_id(entry_id)
    payload: Dict[str, Optional[Union[str, bool]]] = {"entry_id": normalized_entry_id}
    contacted_provided = contacted is not None
    in_pipeline_provided = in_pipeline is not None

    if contacted_provided:
        payload["contacted"] = bool(contacted)
        payload["contacted_at"] = datetime.utcnow().isoformat() + "Z" if contacted else None

    if contacted_by is not None:
        payload["contacted_by"] = contacted_by

    if in_pipeline_provided:
        payload["in_pipeline"] = bool(in_pipeline)
        payload["in_pipeline_at"] = datetime.utcnow().isoformat() + "Z" if in_pipeline else None

    existing_record: Optional[Dict] = None
    try:
        # Try to find existing record with normalized ID, and also try original format
        existing_result = (
            supabase.table("founder_contact_status")
            .select("entry_id, contacted, contacted_at, contacted_by, in_pipeline, in_pipeline_at")
            .eq("entry_id", normalized_entry_id)
            .execute()
        )
        if existing_result.data:
            existing_record = existing_result.data[0]
        else:
            # Try with original format as fallback
            existing_result = (
                supabase.table("founder_contact_status")
                .select("entry_id, contacted, contacted_at, contacted_by, in_pipeline, in_pipeline_at")
                .eq("entry_id", entry_id)
                .execute()
            )
            if existing_result.data:
                existing_record = existing_result.data[0]
    except Exception as e:
        print(f"Error fetching existing founder contact status for {entry_id}: {e}")

    if existing_record is None:
        if not contacted_provided:
            payload["contacted"] = False
            payload["contacted_at"] = None
        if not in_pipeline_provided:
            payload["in_pipeline"] = False
            payload["in_pipeline_at"] = None
        if "contacted_by" not in payload:
            payload["contacted_by"] = None

    try:
        result = (
            supabase.table("founder_contact_status")
            .upsert(payload, on_conflict="entry_id")
            .execute()
        )
        if result.data:
            return result.data[0]
        if existing_record:
            merged = existing_record.copy()
            merged.update(payload)
            return merged
        return payload
    except Exception as e:
        print(f"Error upserting founder contact status for {entry_id}: {e}")
        return None


def get_taste_tree() -> Optional[Dict]:
    """
    Retrieve the latest taste_tree record from Supabase
    Returns the data JSONB field or None if not found
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        result = (
            supabase.table("taste_tree")
            .select("id, data, version, created_at, updated_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        
        if not result.data:
            return None
        
        return result.data[0]
    except Exception as e:
        print(f"Error retrieving taste tree: {e}")
        return None


def update_taste_tree_lead(category_path: List[str], montage_lead: Optional[str]) -> bool:
    """
    Update the montage_lead field for a specific node in the taste_tree JSONB structure.
    
    Args:
        category_path: List of category names from root to target node (e.g., ["Commerce", "Retail & Consumer"])
        montage_lead: New value for montage_lead (string or None to clear)
    
    Returns:
        True if successful, False otherwise
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    if not category_path:
        raise ValueError("category_path cannot be empty")
    
    try:
        # Get the latest taste_tree record
        record = get_taste_tree()
        if not record:
            raise Exception("No taste_tree record found")
        
        tree_id = record["id"]
        # Use deep copy to avoid modifying the original data structure
        data = copy.deepcopy(record["data"]) if isinstance(record["data"], dict) else record["data"]
        
        # Navigate to the target node
        # The tree structure: { "Category": { "meta": {...}, "children": { "Child": {...} } } }
        # The path: [top_level_category, child_category, grandchild_category, ...]
        current = data
        for i, category_name in enumerate(category_path):
            if not isinstance(current, dict):
                raise ValueError(f"Expected dict at path level {i}, but got {type(current).__name__}")
            
            # At the first level (i=0), the category is a direct key in the root
            # At subsequent levels (i>0), we need to navigate through "children"
            if i == 0:
                # Top-level category - direct key in root
                if category_name not in current:
                    available_keys = list(current.keys())
                    available_keys_str = ", ".join(available_keys[:10])
                    if len(available_keys) > 10:
                        available_keys_str += f", ... (and {len(available_keys) - 10} more)"
                    raise ValueError(
                        f"Top-level category '{category_name}' not found. "
                        f"Available top-level categories: [{available_keys_str}]"
                    )
                current = current[category_name]
            else:
                # Nested categories are in the "children" object
                if "children" not in current:
                    raise ValueError(
                        f"Category '{category_path[i-1]}' has no children. "
                        f"Cannot navigate to '{category_name}'"
                    )
                if not isinstance(current["children"], dict):
                    raise ValueError(
                        f"Expected 'children' to be a dict in category '{category_path[i-1]}'"
                    )
                if category_name not in current["children"]:
                    available_keys = list(current["children"].keys())
                    available_keys_str = ", ".join(available_keys[:10])
                    if len(available_keys) > 10:
                        available_keys_str += f", ... (and {len(available_keys) - 10} more)"
                    raise ValueError(
                        f"Category '{category_name}' not found in children of '{category_path[i-1]}'. "
                        f"Path so far: {category_path[:i]}. "
                        f"Available children: [{available_keys_str}]"
                    )
                current = current["children"][category_name]
        
        # Ensure meta exists
        if "meta" not in current:
            current["meta"] = {}
        
        # Update montage_lead
        if montage_lead is None or montage_lead.strip() == "":
            current["meta"].pop("montage_lead", None)
        else:
            current["meta"]["montage_lead"] = montage_lead.strip()
        
        # Update the record in Supabase
        result = (
            supabase.table("taste_tree")
            .update({"data": data, "updated_at": datetime.utcnow().isoformat() + "Z"})
            .eq("id", tree_id)
            .execute()
        )
        
        return result.data is not None
    except Exception as e:
        print(f"Error updating taste tree lead: {e}")
        import traceback
        traceback.print_exc()
        return False


def _traverse_tree_recursive(node, path: List[str], callback):
    """
    Recursively traverse the tree structure and call callback for each node.
    
    Args:
        node: Current node in the tree
        path: Current path from root to this node
        callback: Function to call for each node: callback(path, node_data)
    """
    if not isinstance(node, dict):
        return
    
    # Process current node if it has meta
    if "meta" in node:
        callback(path, node)
    
    # Process children recursively
    if "children" in node and isinstance(node["children"], dict):
        for child_name, child_data in node["children"].items():
            if isinstance(child_data, dict):
                _traverse_tree_recursive(child_data, path + [child_name], callback)


def _normalize_user_name(name: str) -> str:
    """
    Normalize user name to canonical form (title case).
    Handles case-insensitive matching.
    """
    if not name or not isinstance(name, str):
        return ""
    # Convert to title case (first letter uppercase, rest lowercase)
    # This handles "DAphne" -> "Daphne", "MATT" -> "Matt"
    return name.strip().title()


def get_all_users_from_tree() -> List[str]:
    """
    Extract all unique user names (montage_lead values) from the taste tree.
    Handles comma-separated lists and returns individual unique user names.
    Normalizes names to title case for case-insensitive matching.
    Returns a sorted list of unique user names.
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        record = get_taste_tree()
        if not record or not record.get("data"):
            return []
        
        data = record["data"]
        users = set()
        
        def collect_user(path, node_data):
            meta = node_data.get("meta", {})
            montage_lead = meta.get("montage_lead")
            if montage_lead and isinstance(montage_lead, str) and montage_lead.strip():
                # Split by comma and extract individual names
                names = [name.strip() for name in montage_lead.split(",") if name.strip()]
                for name in names:
                    # Normalize to title case for consistent display
                    normalized = _normalize_user_name(name)
                    if normalized:
                        users.add(normalized)
        
        # Traverse all top-level categories
        for category_name, category_data in data.items():
            if isinstance(category_data, dict):
                _traverse_tree_recursive(category_data, [category_name], collect_user)
        
        return sorted(list(users))
    except Exception as e:
        print(f"Error extracting users from taste tree: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_user_tags(user_name: str) -> List[List[str]]:
    """
    Find all category paths where a specific user is tagged as montage_lead.
    Handles comma-separated lists - checks if user_name appears in the list (case-insensitive).
    
    Args:
        user_name: The user name to search for (will be normalized)
    
    Returns:
        List of category paths, where each path is a list of strings
        e.g., [["Commerce", "Retail & Consumer"], ["Healthcare", "Digital Health"]]
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    if not user_name or not isinstance(user_name, str):
        return []
    
    try:
        record = get_taste_tree()
        if not record or not record.get("data"):
            return []
        
        data = record["data"]
        tags = []
        # Normalize the search name for case-insensitive matching
        user_name_normalized = _normalize_user_name(user_name)
        
        def check_user(path, node_data):
            meta = node_data.get("meta", {})
            montage_lead = meta.get("montage_lead")
            if montage_lead and isinstance(montage_lead, str) and montage_lead.strip():
                # Split by comma and check if user_name is in the list (case-insensitive)
                names = [name.strip() for name in montage_lead.split(",") if name.strip()]
                # Check if normalized version of any name matches
                for name in names:
                    if _normalize_user_name(name) == user_name_normalized:
                        tags.append(path.copy())
                        break  # Found match, no need to check other names in this node
        
        # Traverse all top-level categories
        for category_name, category_data in data.items():
            if isinstance(category_data, dict):
                _traverse_tree_recursive(category_data, [category_name], check_user)
        
        return tags
    except Exception as e:
        print(f"Error getting user tags: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_all_category_paths() -> List[Dict]:
    """
    Get a flat list of all categories in the tree with their paths.
    
    Returns:
        List of dictionaries with keys:
        - path: List of strings representing the category path
        - name: The category name (last element of path)
        - fullPath: String representation like "Category > Subcategory"
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    try:
        record = get_taste_tree()
        if not record or not record.get("data"):
            return []
        
        data = record["data"]
        categories = []
        
        def collect_category(path, node_data):
            if path:  # Only add if path is not empty
                categories.append({
                    "path": path.copy(),
                    "name": path[-1],
                    "fullPath": " > ".join(path)
                })
        
        # Traverse all top-level categories
        for category_name, category_data in data.items():
            if isinstance(category_data, dict):
                _traverse_tree_recursive(category_data, [category_name], collect_category)
        
        return categories
    except Exception as e:
        print(f"Error getting all category paths: {e}")
        import traceback
        traceback.print_exc()
        return []


def batch_update_user_tags(user_name: str, add_paths: List[List[str]], remove_paths: List[List[str]]) -> bool:
    """
    Batch update montage_lead for multiple categories.
    Adds the user to categories in add_paths and removes from categories in remove_paths.
    
    Args:
        user_name: The user name to set/remove
        add_paths: List of category paths to add the user to
        remove_paths: List of category paths to remove the user from
    
    Returns:
        True if successful, False otherwise
    """
    if not supabase:
        raise Exception("Supabase client not initialized. Check your SUPABASE_URL and SUPABASE_KEY.")
    
    if not user_name or not isinstance(user_name, str):
        raise ValueError("user_name must be a non-empty string")
    
    try:
        # Get the latest taste_tree record
        record = get_taste_tree()
        if not record:
            raise Exception("No taste_tree record found")
        
        tree_id = record["id"]
        # Use deep copy to avoid modifying the original data structure
        data = copy.deepcopy(record["data"]) if isinstance(record["data"], dict) else record["data"]
        
        # Process all paths to update
        all_paths = add_paths + remove_paths
        for category_path in all_paths:
            if not category_path:
                continue
            
            # Navigate to the target node
            current = data
            for i, category_name in enumerate(category_path):
                if not isinstance(current, dict):
                    continue
                
                if i == 0:
                    # Top-level category
                    if category_name not in current:
                        continue
                    current = current[category_name]
                else:
                    # Nested categories are in the "children" object
                    if "children" not in current or not isinstance(current["children"], dict):
                        continue
                    if category_name not in current["children"]:
                        continue
                    current = current["children"][category_name]
            
            # Update montage_lead
            if not isinstance(current, dict):
                continue
            
            if "meta" not in current:
                current["meta"] = {}
            
            existing_lead = current["meta"].get("montage_lead", "")
            existing_names = []
            if existing_lead and isinstance(existing_lead, str) and existing_lead.strip():
                # Parse existing comma-separated list, preserving original casing
                existing_names = [name.strip() for name in existing_lead.split(",") if name.strip()]
            
            # Normalize the user name we're adding/removing
            user_name_normalized = _normalize_user_name(user_name)
            
            if category_path in add_paths:
                # Check if user already exists (case-insensitive)
                user_exists = False
                for i, existing_name in enumerate(existing_names):
                    if _normalize_user_name(existing_name) == user_name_normalized:
                        # User already exists, keep original casing
                        user_exists = True
                        break
                
                # Add user if not already present (use normalized/canonical form)
                if not user_exists:
                    existing_names.append(user_name_normalized)
                # Reconstruct comma-separated string (sorted for consistency)
                current["meta"]["montage_lead"] = ", ".join(sorted(existing_names, key=str.lower))
            elif category_path in remove_paths:
                # Remove user from the list (case-insensitive match)
                existing_names = [
                    name for name in existing_names 
                    if _normalize_user_name(name) != user_name_normalized
                ]
                # Update or remove montage_lead
                if existing_names:
                    current["meta"]["montage_lead"] = ", ".join(sorted(existing_names, key=str.lower))
                else:
                    # Remove montage_lead if no users left
                    if "montage_lead" in current["meta"]:
                        del current["meta"]["montage_lead"]
        
        # Update the record in Supabase
        result = (
            supabase.table("taste_tree")
            .update({"data": data, "updated_at": datetime.utcnow().isoformat() + "Z"})
            .eq("id", tree_id)
            .execute()
        )
        
        return result.data is not None
    except Exception as e:
        print(f"Error batch updating user tags: {e}")
        import traceback
        traceback.print_exc()
        return False

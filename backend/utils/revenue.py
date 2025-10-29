# utils_revenue.py (new small helper)
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

# Local utils
from .periods import normalize_and_sort_periods

_SCRIPTS_PATH = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, _SCRIPTS_PATH)

from revenue_extractor import extract_revenue  # type: ignore


def _to_quarter_key(dt):
    return (dt.year, (dt.month - 1)//3 + 1)

def derive_per_quarter_values(values_map, iso_to_dt):
    """
    values_map: dict[iso_date_str] -> number (10-Q values; may be YTD)
    iso_to_dt: dict[iso_date_str] -> datetime
    Returns dict[iso_date_str] -> per-quarter number
    """
    # Sort ascending by date for diffing
    items = [(iso, iso_to_dt[iso], values_map.get(iso)) for iso in values_map if iso in iso_to_dt]
    items.sort(key=lambda x: x[1])

    # Group by fiscal year (approx: calendar-year from dates; good enough if you stay with SEC period_of_report)
    year_groups = defaultdict(list)
    for iso, dt, v in items:
        year_groups[dt.year].append((iso, dt, v))

    per_q = {}
    for _, rows in year_groups.items():
        prev_cum = None
        for i, (iso, dt, v) in enumerate(rows):
            if v is None:
                per_q[iso] = None
                continue
            # Heuristic: if numbers are strictly increasing within year → treat as YTD and diff
            if i == 0:
                per_q[iso] = v  # Q1 YTD == Q1
                prev_cum = v
            else:
                # If decreasing or flat, assume already quarterly; else diff
                if v is not None and prev_cum is not None and v >= prev_cum:
                    per_q[iso] = v - prev_cum
                    prev_cum = v
                else:
                    # already quarterly or inconsistent; just use as-provided
                    per_q[iso] = v
                    prev_cum = v
    return per_q


def select_best_revenue(items, periods, prefer_period=None):
    """Select the best revenue value using shared extraction logic.

    Returns: (revenue_value, concept_used, confidence_level)
    confidence_level: 'high', 'medium', 'low', or None
    """
    cor = None
    for item in items or []:
        if item.get('concept') == 'us-gaap_CostOfRevenue':
            values = item.get('values') or {}
            if prefer_period and values.get(prefer_period):
                try:
                    cor = abs(float(values[prefer_period]))
                except Exception:
                    cor = None
            else:
                for period in periods or []:
                    val = values.get(period)
                    if val:
                        try:
                            cor = abs(float(val))
                            break
                        except Exception:
                            cor = None
                if cor:
                    break
    revenue, metadata = extract_revenue(items, periods, prefer_period=prefer_period, cost_of_revenue=cor)
    if revenue is None:
        return None, None, None

    confidence = 'high'
    if metadata.get('valid_candidates_count', 0) > 1:
        confidence = 'medium'
    if not metadata.get('plausible', True):
        confidence = 'low'
    if metadata.get('reason') and 'fallback' in metadata.get('reason', ''):
        confidence = 'medium'

    return revenue, metadata.get('concept'), confidence


def _select_preferred_period(periods: List[str], period_metadata: Optional[List[Dict[str, Any]]] = None, prefer: str = '10-K') -> Optional[str]:
    """
    Select the preferred period from available periods, prioritizing annual (10-K) over quarterly (10-Q).
    Returns preferred period string or None.
    """
    if not periods:
        return None

    # Sort periods by date string (most recent first)
    sorted_periods = sorted(periods, key=lambda x: str(x), reverse=True)

    # If we have metadata, use it to find the preferred filing type
    if period_metadata:
        for period_info in period_metadata:
            if period_info.get('type') == prefer and period_info.get('date') in periods:
                return period_info['date']

    # Fallback to most recent period
    return sorted_periods[0]


def _calculate_ttm_revenue(items, periods, period_metadata=None):
    """
    Compute TTM revenue from available periods.
    Handles both discrete and YTD quarterly reporters automatically.
    """

    import numpy as np
    from datetime import datetime

    # --- Step 1: Collect revenue series ---
    revenue_values = {}
    for item in items:
        concept = (item.get("concept") or "").lower()
        label = (item.get("label") or "").lower()
        if "revenue" in concept or "sales" in concept:
            for p, v in item.get("values", {}).items():
                if v is not None:
                    revenue_values[p] = float(v)
    if not revenue_values:
        return None, "no_revenue_found"

    # --- Step 2: Sort periods chronologically ---
    sorted_periods = sorted(
        [p for p in revenue_values.keys()],
        key=lambda x: datetime.strptime(x[:10], "%Y-%m-%d")
    )

    # --- Step 3: Identify quarterly vs annual periods ---
    q_periods = []
    a_periods = []
    if period_metadata:
        for pmeta in period_metadata:
            d = pmeta.get("date")
            t = pmeta.get("type", "").upper()
            if d in revenue_values:
                if "10-Q" in t:
                    q_periods.append(d)
                elif "10-K" in t:
                    a_periods.append(d)

    # If no metadata, infer by frequency (fallback)
    if not q_periods and len(sorted_periods) > 4:
        q_periods = sorted_periods[-4:]

    # --- Step 4: Detect if the quarterly data is cumulative (YTD) ---
    q_revs = [revenue_values[p] for p in q_periods if p in revenue_values]
    discrete_q_revs = []
    if len(q_revs) >= 4:
        # Check if later quarters > earlier (suggests cumulative)
        is_ytd = all(q_revs[i] < q_revs[i + 1] for i in range(len(q_revs) - 1))
        if is_ytd:
            # Convert YTD → discrete
            for i in range(len(q_revs)):
                if i == 0:
                    discrete_q_revs.append(q_revs[i])
                else:
                    discrete_q_revs.append(q_revs[i] - q_revs[i - 1])
        else:
            discrete_q_revs = q_revs[-4:]
    else:
        discrete_q_revs = q_revs

    # --- Step 5: Compute TTM ---
    if len(discrete_q_revs) >= 4:
        ttm_revenue = np.nansum(discrete_q_revs[-4:])
        return ttm_revenue, "ttm_4q"
    elif a_periods:
        # Fallback: use most recent annual
        return revenue_values[a_periods[-1]], "annual"
    else:
        return None, "no_ttm_possible"

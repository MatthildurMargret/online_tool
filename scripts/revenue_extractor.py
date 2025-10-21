#!/usr/bin/env python3
"""
Shared Revenue Extraction Logic

This module provides a consistent, sector-aware approach to extracting
total revenue from financial statements across all scripts.
"""

import math
from typing import Dict, List, Optional, Tuple


def _safe_number(v):
    """Convert to float if finite, else return None."""
    try:
        if v is None:
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def infer_sector(concepts_present: set) -> str:
    """
    Infer company sector from presence of hallmark XBRL concepts.
    
    Args:
        concepts_present: Set of XBRL concepts present in the income statement
        
    Returns:
        Sector identifier: 'insurer', 'bank', 'reit', 'utility', or 'general'
    """
    if any(c for c in concepts_present if c and 'PremiumsEarned' in c):
        return 'insurer'
    if any(c for c in concepts_present if c and ('RevenuesNetOfInterestExpense' in c or 'InterestAndDividendIncome' in c)):
        return 'bank'
    if any(c for c in concepts_present if c and ('OperatingLeaseLeaseIncome' in c or 'Rental' in c)):
        return 'reit'
    if any(c for c in concepts_present if c and 'RegulatedAndUnregulatedOperatingRevenue' in c):
        return 'utility'
    return 'general'


def get_revenue_priority_list(sector: str) -> List[str]:
    """
    Get sector-specific priority list for revenue concepts.
    Lower index = higher priority.
    
    Args:
        sector: Sector identifier from infer_sector()
        
    Returns:
        List of XBRL concepts in priority order
    """
    PRIORITY = {
        'general': [
            'us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax',
            'us-gaap_Revenues',
            'us-gaap_SalesRevenueNet',
            'us-gaap_Revenue',
            'us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax',
        ],
        'bank': [
            'us-gaap_RevenuesNetOfInterestExpense',
            'us-gaap_InterestAndDividendIncomeOperating',
            'us-gaap_Revenues',
        ],
        'insurer': [
            'us-gaap_PremiumsEarnedNet',
            'us-gaap_Revenues',
        ],
        'reit': [
            'us-gaap_OperatingLeaseLeaseIncome',
            'us-gaap_Revenues',
        ],
        'utility': [
            'us-gaap_RegulatedAndUnregulatedOperatingRevenue',
            'us-gaap_Revenues',
        ],
    }
    return PRIORITY.get(sector, PRIORITY['general'])


def is_valid_total_revenue(label: str, concept: str, sector: str) -> bool:
    """
    Validate that a revenue concept represents total revenue, not a component/segment.
    
    Args:
        label: Human-readable label from financial statement
        concept: XBRL concept identifier
        sector: Sector identifier from infer_sector()
        
    Returns:
        True if this appears to be total/consolidated revenue
    """
    lbl = (label or '').lower().strip()
    
    # Component keywords that indicate this is NOT total revenue
    # NOTE: 'contract revenue' is NOT included here - it's often the total under ASC 606
    COMPONENT_KEYWORDS = [
        'segment', 'product', 'service', 'geography', 'domestic', 'international',
        'united states', 'foreign', 'wholesale', 'retail',
        'subscription', 'license', 'maintenance', 'support',
        'hardware', 'software', 'cloud', 'digital',
        'alliance', 'collaboration', 'royalty', 'milestone',
    ]
    
    # If label obviously refers to a component, reject
    if any(kw in lbl for kw in COMPONENT_KEYWORDS):
        return False
    
    # Lease-specific: only accept as total if sector suggests REIT/lessor or label hints total
    if 'lease' in concept.lower():
        if sector in ('reit',) or any(w in lbl for w in ['total', 'revenue', 'rental']):
            return True
        return False
    
    # Financial services / utilities are already encoded in the sector priority
    # If we got here and it's in a sector-specific priority list, accept it
    revenue_concepts = get_revenue_priority_list(sector)
    if concept in revenue_concepts:
        return True
    
    # Fallback conservative accept
    return False


def label_simplicity_score(label: str) -> int:
    """
    Score how simple/generic a label is. Lower score = simpler/better.
    
    Args:
        label: Human-readable label from financial statement
        
    Returns:
        Simplicity score (0 = simplest)
    """
    lbl = (label or '').lower().strip()
    
    # Prefer labels that are just "revenue" or "revenues"
    if lbl in ('revenue', 'revenues', 'total revenue', 'total revenues', 'net sales'):
        return 0
    
    penalty = 0
    
    # Penalize labels with extra descriptors
    for desc in ['segment', 'product', 'service', 'geography', 'alliance', 'collaboration', 'royalty', 'milestone']:
        if desc in lbl:
            penalty += 10
    
    # Penalize longer labels (more words = more specific)
    penalty += len(lbl.split())
    
    return penalty


def extract_revenue(items, periods, prefer_period=None, cost_of_revenue=None, tolerance=0.08):
    """
    Return (value, metadata) with smarter scoring:
      - value: float or None
      - metadata: { 'concept': str, 'label': str, 'reason': str, 'plausible': bool,
                    'candidates': [...], 'valid_candidates_count': int }
    """
    # 1) Pull COGS and GP to enable identity check: Revenue ≈ COGS + GP
    def find_value(concept_names):
        for it in items:
            c = (it.get('concept') or '').strip()
            if c in concept_names:
                vals = it.get('values') or {}
                if prefer_period and vals.get(prefer_period) is not None:
                    return float(vals.get(prefer_period))
                # fallback: first non-null across periods in order
                for p in periods:
                    v = vals.get(p)
                    if v is not None:
                        return float(v)
        return None

    cogs = find_value({'us-gaap_CostOfRevenue','us-gaap_CostOfGoodsAndServicesSold'})
    gp   = find_value({'us-gaap_GrossProfit'})

    # 2) Collect revenue-like candidates
    REVENUE_CONCEPTS = [
        'us-gaap_Revenues',
        'us-gaap_SalesRevenueNet',
        'us-gaap_Revenue',
        'us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax',
        'us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax',
        'us-gaap_RevenuesNetOfInterestExpense',
        'us-gaap_InterestAndDividendIncomeOperating',
        'us-gaap_RegulatedAndUnregulatedOperatingRevenue',
        'us-gaap_OperatingLeaseLeaseIncome',
    ]

    # Concept base ranks (higher is better). Make sure total-like > contract.
    BASE_RANK = {
        'us-gaap_Revenues': 100,
        'us-gaap_SalesRevenueNet': 95,
        'us-gaap_Revenue': 92,
        'us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax': 70,
        'us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax': 60,
        'us-gaap_RevenuesNetOfInterestExpense': 55,
        'us-gaap_InterestAndDividendIncomeOperating': 52,
        'us-gaap_RegulatedAndUnregulatedOperatingRevenue': 50,
        'us-gaap_OperatingLeaseLeaseIncome': 45,
    }

    EXCLUDE_LABEL_TOKENS = {
        'contract', 'collaboration', 'alliance', 'royalty', 'milestone',
        'segment', 'product', 'service', 'domestic', 'international',
        'subscription', 'license', 'maintenance', 'support',
        'hardware', 'software', 'cloud', 'digital',
        'biopharma', 'consumer', 'medical', 'pharmaceutical',
    }
    TOTAL_HINT_TOKENS = {'total','consolidated','net sales','net revenue','revenue','revenues'}

    # Helper to get a period value
    def val_for_period(values_map):
        if not isinstance(values_map, dict):
            return None
        if prefer_period and values_map.get(prefer_period) is not None:
            return _safe_number(values_map.get(prefer_period))
        for p in periods:
            v = values_map.get(p)
            if v is not None:
                return _safe_number(v)
        return None

    # Collect raw candidates
    raw = []
    for it in items:
        c = it.get('concept') or ''
        if c in REVENUE_CONCEPTS:
            v = val_for_period(it.get('values') or {})
            if v is None:
                continue
            raw.append({
                'concept': c,
                'label': it.get('label') or '',
                'value': float(v)
            })

    # Industry-style fallbacks if nothing obvious found
    if not raw:
        # Banks/financials
        fin_val = find_value({'us-gaap_RevenuesNetOfInterestExpense','us-gaap_InterestAndDividendIncomeOperating'})
        if fin_val is not None:
            return fin_val, {'concept': 'financial_fallback', 'label': 'Financial revenue fallback',
                             'reason': 'No standard revenue concepts; using financial revenue proxy',
                             'plausible': True, 'valid_candidates_count': 1}
        # REITs/lessors
        lease_val = find_value({'us-gaap_OperatingLeaseLeaseIncome'})
        if lease_val is not None:
            return lease_val, {'concept': 'lease_fallback', 'label': 'Lease income fallback',
                               'reason': 'No standard revenue concepts; using lease income',
                               'plausible': True, 'valid_candidates_count': 1}
        # Utilities
        util_val = find_value({'us-gaap_RegulatedAndUnregulatedOperatingRevenue'})
        if util_val is not None:
            return util_val, {'concept': 'utility_fallback', 'label': 'Regulated operating revenue',
                               'reason': 'No standard revenue concepts; using regulated revenue',
                               'plausible': True, 'valid_candidates_count': 1}
        return None, {'reason': 'No revenue-like concepts found', 'plausible': False, 'valid_candidates_count': 0}

    # 3) Magnitude context: max and identity reference (COGS + GP)
    max_val = max(r['value'] for r in raw) if raw else None
    identity_ref = None
    if cogs is not None and gp is not None:
        # accounting identity reference
        identity_ref = abs(cogs) + abs(gp)

    # 4) Score candidates
    def score_row(r):
        concept = r['concept']
        label = (r['label'] or '').lower()
        value = r['value']
        score = BASE_RANK.get(concept, 10)

        # Label-based penalties/boosts
        if any(tok in label for tok in EXCLUDE_LABEL_TOKENS):
            score -= 25   # push down component-like lines
        if any(tok in label for tok in TOTAL_HINT_TOKENS):
            score += 10

        # Magnitude preference (favor the largest among competing revenue lines)
        if max_val and value == max_val:
            score += 20
        if max_val and value < 0.2 * max_val:
            score -= 20  # tiny compared to peer candidates → likely a component

        # Identity cross-check boost/penalty
        if identity_ref is not None and value is not None:
            if identity_ref == 0:
                pass
            else:
                rel_err = abs(value - identity_ref) / max(identity_ref, 1e-9)
                if rel_err <= tolerance:
                    score += 35  # matches COGS + GP → very strong signal
                elif value < abs(cogs or 0) * 0.8:
                    score -= 25  # lower than COGS → suspicious for total revenue

        return score

    scored = []
    for r in raw:
        r_score = score_row(r)
        scored.append({**r, 'score': r_score})

    # 5) Prefer exact “total-like” concepts over contract when co-present
    #    (hard rule applied after scoring)
    has_total_like = any(s['concept'] in ('us-gaap_Revenues','us-gaap_SalesRevenueNet','us-gaap_Revenue') for s in scored)
    if has_total_like:
        scored = sorted(scored, key=lambda x: (
            0 if x['concept'] in ('us-gaap_Revenues','us-gaap_SalesRevenueNet','us-gaap_Revenue') else 1,
            -x['score']
        ))
    else:
        scored = sorted(scored, key=lambda x: -x['score'])

    best = scored[0]
    plausible = True
    reason_bits = []
    reason_bits.append(f"picked_by={'total-like' if has_total_like else 'highest_score'}")
    if identity_ref is not None:
        reason_bits.append(f"identity_ref={identity_ref:,.0f}")
    reason = "; ".join(reason_bits)

    return best['value'], {
        'concept': best['concept'],
        'label': best['label'],
        'reason': reason,
        'plausible': plausible,
        'valid_candidates_count': len(scored),
        'candidates': scored[:5],  # top few for debugging
    }

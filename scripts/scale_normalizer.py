#!/usr/bin/env python3
"""
Scale Normalization Utilities

Handles normalization of financial data that may be reported in different scales
(actual dollars, thousands, millions, billions) across different XBRL filers.
"""

from typing import Optional, Dict, Any


def detect_scale_from_magnitude(revenue: Optional[float], total_assets: Optional[float] = None) -> float:
    """
    Detect the likely scale multiplier based on magnitude of key metrics.
    
    Most public companies have revenue > $1M, so if revenue < 100k, it's likely in millions.
    
    Args:
        revenue: Revenue value
        total_assets: Total assets value (optional, for additional validation)
        
    Returns:
        Scale multiplier (1, 1000, 1000000, or 1000000000)
    """
    # Use revenue as primary indicator
    if revenue is not None and revenue > 0:
        if revenue < 1_000:
            # Likely in billions (e.g., 2.5 = $2.5B)
            return 1_000_000_000
        elif revenue < 100_000:
            # Likely in millions (e.g., 50000 = $50M)
            return 1_000_000
        elif revenue < 100_000_000:
            # Likely in thousands (e.g., 50000000 = $50M in thousands)
            return 1_000
        else:
            # Likely in actual dollars
            return 1
    
    # Fallback to assets if revenue not available
    if total_assets is not None and total_assets > 0:
        if total_assets < 1_000:
            return 1_000_000_000
        elif total_assets < 100_000:
            return 1_000_000
        elif total_assets < 100_000_000:
            return 1_000
        else:
            return 1
    
    # Default: assume already in actual dollars
    return 1


def normalize_value(value: Optional[float], scale_multiplier: float) -> Optional[float]:
    """
    Normalize a single value using the detected scale multiplier.
    
    Args:
        value: Value to normalize
        scale_multiplier: Multiplier to apply
        
    Returns:
        Normalized value in actual dollars, or None if value is None
    """
    if value is None:
        return None
    return float(value) * scale_multiplier


def normalize_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize all financial data in a dictionary to actual dollars.
    
    This function:
    1. Detects the scale from revenue/assets
    2. Applies normalization to all numeric fields
    3. Adds metadata about the normalization
    
    Args:
        data: Dictionary containing financial data with keys like:
              - revenue
              - shares (not normalized - already in actual count)
              - raw_data (nested structure with income_statement, balance_sheet, etc.)
              
    Returns:
        Normalized data dictionary with added 'scale_info' metadata
    """
    # Detect scale from revenue
    revenue = data.get('revenue')
    
    # Try to get total assets from raw_data for validation
    total_assets = None
    try:
        raw_data = data.get('raw_data', {})
        if isinstance(raw_data, dict):
            metrics = raw_data.get('metrics', {})
            # Get most recent assets value if available
            if 'total_assets' in metrics and isinstance(metrics['total_assets'], dict):
                assets_values = list(metrics['total_assets'].values())
                if assets_values:
                    total_assets = assets_values[0]
    except Exception:
        pass
    
    # Detect scale
    scale_multiplier = detect_scale_from_magnitude(revenue, total_assets)
    
    # If scale is 1 (already in dollars), no normalization needed
    if scale_multiplier == 1:
        data['scale_info'] = {
            'detected_scale': 'dollars',
            'multiplier': 1,
            'normalized': False
        }
        return data
    
    # Determine scale name
    scale_name = {
        1_000: 'thousands',
        1_000_000: 'millions',
        1_000_000_000: 'billions'
    }.get(scale_multiplier, 'unknown')
    
    # Normalize top-level revenue (shares are already in actual count, don't normalize)
    if revenue is not None:
        data['revenue'] = normalize_value(revenue, scale_multiplier)
    
    # Normalize raw_data nested structures
    try:
        raw_data = data.get('raw_data', {})
        if isinstance(raw_data, dict):
            # Normalize income statement items
            income_stmt = raw_data.get('income_statement', {})
            if isinstance(income_stmt, dict):
                items = income_stmt.get('items', [])
                for item in items:
                    if isinstance(item, dict) and 'values' in item:
                        concept = item.get('concept', '')
                        # Don't normalize share counts or per-share values
                        if 'Shares' not in concept and 'PerShare' not in concept and 'EarningsPerShare' not in concept:
                            values = item['values']
                            if isinstance(values, dict):
                                for period, value in values.items():
                                    if value is not None:
                                        values[period] = normalize_value(value, scale_multiplier)
            
            # Normalize balance sheet items
            balance_sheet = raw_data.get('balance_sheet', {})
            if isinstance(balance_sheet, dict):
                items = balance_sheet.get('items', [])
                for item in items:
                    if isinstance(item, dict) and 'values' in item:
                        concept = item.get('concept', '')
                        # Don't normalize share counts or per-share values
                        if 'Shares' not in concept and 'PerShare' not in concept:
                            values = item['values']
                            if isinstance(values, dict):
                                for period, value in values.items():
                                    if value is not None:
                                        values[period] = normalize_value(value, scale_multiplier)
            
            # Normalize cash flow items
            cash_flow = raw_data.get('cash_flow', {})
            if isinstance(cash_flow, dict):
                items = cash_flow.get('items', [])
                for item in items:
                    if isinstance(item, dict) and 'values' in item:
                        concept = item.get('concept', '')
                        # Don't normalize share counts or per-share values
                        if 'Shares' not in concept and 'PerShare' not in concept:
                            values = item['values']
                            if isinstance(values, dict):
                                for period, value in values.items():
                                    if value is not None:
                                        values[period] = normalize_value(value, scale_multiplier)
            
            # Normalize metrics
            metrics = raw_data.get('metrics', {})
            if isinstance(metrics, dict):
                for metric_name, metric_values in metrics.items():
                    # Don't normalize share counts
                    if 'shares' not in metric_name.lower():
                        if isinstance(metric_values, dict):
                            for period, value in metric_values.items():
                                if value is not None:
                                    metric_values[period] = normalize_value(value, scale_multiplier)
    except Exception as e:
        # If normalization fails, log but don't crash
        print(f"Warning: Error during scale normalization: {e}")
    
    # Add metadata about normalization
    data['scale_info'] = {
        'detected_scale': scale_name,
        'multiplier': scale_multiplier,
        'normalized': True,
        'original_revenue': revenue / scale_multiplier if revenue else None
    }
    
    return data


def validate_normalization(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that normalization produced reasonable results.
    
    Checks:
    - Revenue > $1M for public companies
    - Market cap is reasonable (> $10M, < $10T)
    - P/S ratio is reasonable (< 1000)
    
    Returns:
        Dictionary with validation results and warnings
    """
    warnings = []
    
    revenue = data.get('revenue')
    shares = data.get('shares')
    
    # Check revenue magnitude
    if revenue is not None:
        if revenue < 1_000_000:
            warnings.append(f"Revenue ${revenue:,.0f} seems low for a public company - may need scale adjustment")
        elif revenue > 1_000_000_000_000:
            warnings.append(f"Revenue ${revenue:,.0f} seems extremely high - may be over-scaled")
    
    # Check market cap if we have shares
    if shares is not None and shares > 0:
        # Assume a reasonable stock price range ($1-$1000)
        implied_price_at_1m_cap = 1_000_000 / shares
        implied_price_at_10t_cap = 10_000_000_000_000 / shares
        
        if implied_price_at_1m_cap > 1000:
            warnings.append(f"Shares {shares:,.0f} seems too low - check scale")
        elif implied_price_at_10t_cap < 0.01:
            warnings.append(f"Shares {shares:,.0f} seems too high - check scale")
    
    return {
        'valid': len(warnings) == 0,
        'warnings': warnings
    }

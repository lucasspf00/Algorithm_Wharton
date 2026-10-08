from __future__ import annotations

import numpy as np


SECTOR_ALIASES = {
    "communication_services": "Communication Services",
    "consumer_cyclical": "Consumer Discretionary",
    "consumer_discretionary": "Consumer Discretionary",
    "consumer_defensive": "Consumer Staples",
    "consumer_staples": "Consumer Staples",
    "basic_materials": "Materials",
    "materials": "Materials",
    "financial_services": "Financials",
    "financials": "Financials",
    "healthcare": "Health Care",
    "health_care": "Health Care",
    "industrials": "Industrials",
    "realestate": "Real Estate",
    "real_estate": "Real Estate",
    "technology": "Information Technology",
    "information_technology": "Information Technology",
    "utilities": "Utilities",
    "energy": "Energy",
}
BENCHMARK_SECTORS = frozenset(SECTOR_ALIASES.values())
LAURA_SECTOR_TARGETS = {
    "Health Care": {"target": .15, "low": .10, "high": .20},
    "Construction / Infrastructure": {"target": .10, "low": .05, "high": .15},
    "Financial": {"target": .15, "low": .10, "high": .20},
    "Technology": {"target": .30, "low": .22, "high": .35},
    "Energy": {"target": .10, "low": .05, "high": .15},
    "Others": {"target": .20, "low": .10, "high": .30},
}

_CONSTRUCTION_INDUSTRY_TERMS = (
    "construction & engineering",
    "engineering & construction",
    "building products",
    "building materials",
    "construction materials",
    "infrastructure operations",
    "homebuilding",
)


def laura_sector_bucket(sector: str, industry: str = "", asset_class: str = "stock") -> str:
    """Map a security to Laura's six equity-sleeve buckets."""
    if asset_class == "equity_etf":
        normalized = normalize_sector(sector).lower()
        return {
            "health care": "Health Care",
            "financials": "Financial",
            "information technology": "Technology",
            "energy": "Energy",
        }.get(normalized, "Others")
    normalized = normalize_sector(sector).lower()
    industry_text = str(industry or "").lower()
    if normalized == "health care":
        return "Health Care"
    if normalized == "financials":
        return "Financial"
    if normalized == "information technology":
        return "Technology"
    if normalized == "energy":
        return "Energy"
    if any(term in industry_text for term in _CONSTRUCTION_INDUSTRY_TERMS):
        return "Construction / Infrastructure"
    return "Others"


def normalize_sector(sector: str) -> str:
    text = str(sector or "Unknown").strip()
    return SECTOR_ALIASES.get(text.lower().replace(" ", "_"), text)


def _finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def normalize_sector_weights(weights: dict) -> dict[str, float]:
    if not isinstance(weights, dict) or not weights:
        return {}
    normalized = {}
    for key, value in weights.items():
        if not _finite_number(value) or float(value) < 0:
            return {}
        sector = normalize_sector(key)
        if sector not in BENCHMARK_SECTORS:
            return {}
        normalized[sector] = normalized.get(sector, 0.0) + float(value)
    total = sum(normalized.values())
    if not .99 <= total <= 1.01:
        return {}
    return {sector: value / total for sector, value in normalized.items()}


def normalize_laura_sector_weights(weights: dict) -> dict[str, float]:
    """Aggregate complete Yahoo ETF sector weights into Laura's taxonomy."""
    normalized = normalize_sector_weights(weights)
    if not normalized:
        return {}
    result = {bucket: 0.0 for bucket in LAURA_SECTOR_TARGETS}
    for sector, weight in normalized.items():
        result[laura_sector_bucket(sector, asset_class="equity_etf")] += weight
    return result

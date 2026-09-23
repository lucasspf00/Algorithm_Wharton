from __future__ import annotations
from typing import Mapping
import math
import numpy as np


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_symmetric(value: float, cap: float, higher_is_better: bool = True) -> float:
    """Map a metric to [-100, 100] using a symmetric cap around zero.

    Example: cap=0.30 means -30% -> -100, 0% -> 0, +30% -> +100.
    Values outside the cap are clipped. If lower values are better, the sign is flipped.
    """
    if not _finite(value) or not _finite(cap) or cap <= 0:
        return np.nan
    score = clamp(float(value) / float(cap), -1.0, 1.0) * 100.0
    return score if higher_is_better else -score


def normalize_piecewise(value: float, lower: float, neutral: float, upper: float, higher_is_better: bool = True) -> float:
    """Piecewise-linear normalization to [-100,100]."""
    if not all(_finite(x) for x in [value, lower, neutral, upper]):
        return np.nan
    value, lower, neutral, upper = map(float, [value, lower, neutral, upper])
    if not (lower < neutral < upper):
        return np.nan
    if value <= lower:
        score = -100.0
    elif value >= upper:
        score = 100.0
    elif value <= neutral:
        score = -100.0 + 100.0 * (value - lower) / (neutral - lower)
    else:
        score = 100.0 * (value - neutral) / (upper - neutral)
    return score if higher_is_better else -score


def weighted_available(values: Mapping[str, float], weights: Mapping[str, float]) -> tuple[float, float]:
    """Weighted average over available metrics.

    Returns (score, coverage), where coverage is the share of intended weight with data.
    Missing values are NOT treated as zero.
    """
    total = sum(max(float(weights.get(k, 0.0)), 0.0) for k in values)
    if total <= 0:
        return np.nan, 0.0
    used = 0.0
    num = 0.0
    for k, v in values.items():
        w = max(float(weights.get(k, 0.0)), 0.0)
        if _finite(v) and w > 0:
            num += float(v) * w
            used += w
    if used <= 0:
        return np.nan, 0.0
    return num / used, used / total


def _growth_branch(metrics: Mapping[str, float], prefix: str, cfg: dict) -> tuple[float, float, dict]:
    caps = cfg["scoring"]["growth_caps"]
    horizon_weights = cfg["scoring"]["growth_horizon_weights"]
    keys = [
        (f"{prefix}_growth_1y", "one_year"),
        (f"{prefix}_growth_3y", "three_year"),
        (f"{prefix}_growth_5y", "five_year"),
    ]
    scores = {}
    weights = {}
    for key, horizon in keys:
        scores[key] = normalize_symmetric(metrics.get(key, np.nan), caps[key], True)
        weights[key] = horizon_weights[horizon]
    score, coverage = weighted_available(scores, weights)
    return score, coverage, scores


def score_security(metrics: Mapping[str, float], cfg: dict) -> dict:
    metric_scores: dict[str, float] = {}

    rev, rev_cov, rev_scores = _growth_branch(metrics, "revenue", cfg)
    eps, eps_cov, eps_scores = _growth_branch(metrics, "eps", cfg)
    fcf, fcf_cov, fcf_scores = _growth_branch(metrics, "fcf", cfg)
    metric_scores.update(rev_scores)
    metric_scores.update(eps_scores)
    metric_scores.update(fcf_scores)

    branch_values = {"revenue": rev, "earnings": eps, "fcf": fcf}
    branch_weights = cfg["scoring"]["growth_branch_weights"]
    growth_score, growth_cov = weighted_available(branch_values, branch_weights)
    # More precise intended-data coverage for growth.
    growth_cov = (
        branch_weights["revenue"] * rev_cov
        + branch_weights["earnings"] * eps_cov
        + branch_weights["fcf"] * fcf_cov
    ) / sum(branch_weights.values())

    quality_specs = {
        "operating_margin": (-0.05, 0.10, 0.30, True),
        "net_margin": (-0.05, 0.08, 0.25, True),
        "return_on_equity": (0.00, 0.12, 0.30, True),
        "current_ratio": (0.50, 1.00, 2.00, True),
        "debt_to_equity": (0.20, 1.00, 3.00, False),
    }
    quality_scores = {}
    for key, spec in quality_specs.items():
        quality_scores[key] = normalize_piecewise(metrics.get(key, np.nan), *spec)
        metric_scores[key] = quality_scores[key]
    quality_score, quality_cov = weighted_available(quality_scores, cfg["scoring"]["quality_weights"])

    valuation_specs = {
        "trailing_pe": (10.0, 25.0, 50.0, False),
        "forward_pe": (10.0, 22.0, 45.0, False),
        "price_to_sales": (1.0, 5.0, 12.0, False),
        "price_to_book": (1.0, 5.0, 12.0, False),
        "ev_to_ebitda": (6.0, 15.0, 30.0, False),
    }
    valuation_scores = {}
    for key, spec in valuation_specs.items():
        valuation_scores[key] = normalize_piecewise(metrics.get(key, np.nan), *spec)
        metric_scores[key] = valuation_scores[key]
    valuation_score, valuation_cov = weighted_available(valuation_scores, cfg["scoring"]["valuation_weights"])

    categories = {"growth": growth_score, "quality": quality_score, "valuation": valuation_score}
    category_weights = cfg["scoring"]["category_weights"]
    fundamental_score, _ = weighted_available(categories, category_weights)
    display_rating = (fundamental_score + 100.0) / 2.0 if _finite(fundamental_score) else np.nan

    risk_specs = {
        "annualized_volatility": (0.10, 0.25, 0.55, False),
        "max_drawdown": (0.10, 0.30, 0.65, False),
        "downside_deviation": (0.07, 0.18, 0.40, False),
        "beta": (0.50, 1.00, 1.80, False),
    }
    risk_scores = {}
    for key, spec in risk_specs.items():
        risk_scores[key] = normalize_piecewise(metrics.get(key, np.nan), *spec)
        metric_scores[f"risk::{key}"] = risk_scores[key]
    risk_score, risk_cov = weighted_available(risk_scores, cfg["scoring"]["risk_weights"])

    cw = category_weights
    data_confidence = (
        cw["growth"] * growth_cov + cw["quality"] * quality_cov + cw["valuation"] * valuation_cov
    ) / sum(cw.values())

    return {
        "metric_scores": metric_scores,
        "growth_score": growth_score,
        "quality_score": quality_score,
        "valuation_score": valuation_score,
        "fundamental_score": fundamental_score,
        "display_rating": display_rating,
        "risk_score": risk_score,
        "data_confidence": data_confidence * 100.0,
        "risk_data_coverage": risk_cov * 100.0,
    }

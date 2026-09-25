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
        scores[key] = normalize_symmetric(metrics.get(key, np.nan), caps.get(key, 0.30), True)
        weights[key] = horizon_weights[horizon]
    score, coverage = weighted_available(scores, weights)
    return score, coverage, scores


def _finish_score(metric_scores: dict, categories: Mapping[str, float], coverage: Mapping[str, float],
                  risk_score: float, risk_cov: float, cfg: dict, profile: str,
                  category_weights: Mapping[str, float] | None = None) -> dict:
    category_weights = category_weights or cfg["scoring"]["category_weights"]
    fundamental_score, _ = weighted_available(categories, category_weights)
    category_cov = sum(
        float(category_weights.get(k, 0.0)) * float(coverage.get(k, 0.0))
        for k in categories
    ) / sum(category_weights.values())
    return {
        "metric_scores": metric_scores,
        "growth_score": categories.get("growth", np.nan),
        "quality_score": categories.get("quality", np.nan),
        "valuation_score": categories.get("valuation", np.nan),
        "fundamental_score": fundamental_score,
        "main_score": fundamental_score,
        "risk_score": risk_score,
        "data_confidence": category_cov * 100.0,
        "risk_data_coverage": risk_cov * 100.0,
        "analysis_profile": profile,
        "scoring_engine": {
            "operating_company": "Standard Stock: 30% Growth + 40% Quality/Financial Strength + 30% Valuation",
            "financial_stock": "Financial Stock: dedicated Growth + Financial Quality + Valuation model",
            "financial_conglomerate": "Financial Stock: dedicated Growth + Financial Quality + Valuation model",
            "equity_etf": "Equity ETF: 35% Return + 35% Risk/Resilience + 30% Diversification/Efficiency",
            "fixed_income_etf": "Fixed-Income ETF: 40% Stability/Risk + 30% Return + 30% Efficiency/Liquidity",
        }.get(profile, profile),
    }


def _score_risk(metrics: Mapping[str, float]) -> tuple[dict, float, float]:
    specs = {
        "annualized_volatility": (0.10, 0.25, 0.55, False),
        "max_drawdown": (0.10, 0.30, 0.65, False),
        "downside_deviation": (0.07, 0.18, 0.40, False),
        "beta": (0.50, 1.00, 1.80, False),
    }
    scores = {key: normalize_piecewise(metrics.get(key, np.nan), *spec) for key, spec in specs.items()}
    score, coverage = weighted_available(scores, {
        "annualized_volatility": 0.35, "max_drawdown": 0.35,
        "downside_deviation": 0.20, "beta": 0.10,
    })
    return scores, score, coverage


def _score_operating_company(metrics: Mapping[str, float], cfg: dict, profile: str = "operating_company") -> dict:
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

    risk_scores, risk_score, risk_cov = _score_risk(metrics)
    for key, value in risk_scores.items():
        metric_scores[f"risk::{key}"] = risk_scores[key]
    return _finish_score(
        metric_scores,
        {"growth": growth_score, "quality": quality_score, "valuation": valuation_score},
        {"growth": growth_cov, "quality": quality_cov, "valuation": valuation_cov},
        risk_score, risk_cov, cfg, profile,
    )


def _score_financial_stock(metrics: Mapping[str, float], cfg: dict) -> dict:
    growth_values = {
        "revenue": _growth_branch(metrics, "revenue", cfg)[0],
        "earnings": _growth_branch(metrics, "eps", cfg)[0],
        "net_income": _growth_branch(metrics, "net_income", cfg)[0],
    }
    growth, growth_cov = weighted_available(growth_values, {"revenue": .4, "earnings": .4, "net_income": .2})
    quality_values = {
        "return_on_equity": normalize_piecewise(metrics.get("return_on_equity", np.nan), .02, .10, .20),
        "return_on_assets": normalize_piecewise(metrics.get("return_on_assets", np.nan), .002, .01, .03),
        "net_margin": normalize_piecewise(metrics.get("net_margin", np.nan), -.05, .08, .25),
        "earnings_consistency": normalize_piecewise(metrics.get("earnings_consistency", np.nan), .40, .75, .95),
    }
    quality, quality_cov = weighted_available(quality_values, {
        "return_on_equity": .35, "return_on_assets": .25, "net_margin": .20, "earnings_consistency": .20,
    })
    valuation_values = {
        "trailing_pe": normalize_piecewise(metrics.get("trailing_pe", np.nan), 10, 20, 40, False),
        "forward_pe": normalize_piecewise(metrics.get("forward_pe", np.nan), 10, 18, 35, False),
        "price_to_book": normalize_piecewise(metrics.get("price_to_book", np.nan), .8, 1.5, 3, False),
        "earnings_yield": normalize_piecewise(metrics.get("earnings_yield", np.nan), .025, .05, .10, True),
    }
    valuation, valuation_cov = weighted_available(valuation_values, {
        "trailing_pe": .30, "forward_pe": .25, "price_to_book": .30, "earnings_yield": .15,
    })
    risk_scores, risk_score, risk_cov = _score_risk(metrics)
    scores = {**quality_values, **valuation_values, **{f"risk::{k}": v for k, v in risk_scores.items()}}
    return _finish_score(scores, {"growth": growth, "quality": quality, "valuation": valuation},
                         {"growth": growth_cov, "quality": quality_cov, "valuation": valuation_cov},
                         risk_score, risk_cov, cfg, "financial_stock")


def _score_fund(metrics: Mapping[str, float], cfg: dict, profile: str) -> dict:
    """Score an ETF from fund-level properties and historical total returns."""
    is_fixed_income = profile == "fixed_income_etf"
    return_caps = (0.15, 0.10, 0.08) if is_fixed_income else (0.40, 0.25, 0.20)
    return_values = {
        "one_year": normalize_symmetric(metrics.get("total_return_1y", np.nan), return_caps[0]),
        "three_year": normalize_symmetric(metrics.get("total_return_3y", np.nan), return_caps[1]),
        "five_year": normalize_symmetric(metrics.get("total_return_5y", np.nan), return_caps[2]),
    }
    returns, return_cov = weighted_available(return_values, {"one_year": .2, "three_year": .5, "five_year": .3})
    metric_specs = {
        "expense_ratio": (0.002, 0.008, 0.025, False),
        "holdings_count": (10.0, 100.0, 500.0, True),
        "portfolio_pe": (10.0, 22.0, 45.0, False),
        "portfolio_pb": (1.0, 3.5, 8.0, False),
    }
    metric_scores = {key: normalize_piecewise(metrics.get(key, np.nan), *spec)
                     for key, spec in metric_specs.items()}
    efficiency, efficiency_cov = weighted_available(
        {"expense_ratio": metric_scores["expense_ratio"], "holdings_count": metric_scores["holdings_count"]},
        {"expense_ratio": 0.60, "holdings_count": 0.40},
    )
    risk_scores, risk_score, risk_cov = _score_risk(metrics)
    metric_scores.update({f"risk::{key}": value for key, value in risk_scores.items()})
    metric_scores.update({f"return::{key}": value for key, value in return_values.items()})
    metric_scores.update({"portfolio_pe": metric_scores["portfolio_pe"], "portfolio_pb": metric_scores["portfolio_pb"]})
    if is_fixed_income:
        # Keep the public category keys compatible with the existing UI while
        # naming the fixed-income concepts explicitly in the result.
        categories = {"growth": returns, "quality": risk_score, "valuation": efficiency}
        weights = {"growth": .30, "quality": .40, "valuation": .30}
        category_labels = {
            "growth": "Return",
            "quality": "Stability / Risk",
            "valuation": "Efficiency / Liquidity",
        }
    else:
        categories = {"growth": returns, "quality": risk_score, "valuation": efficiency}
        weights = {"growth": .35, "quality": .35, "valuation": .30}
        category_labels = {
            "growth": "Return",
            "quality": "Risk / Resilience",
            "valuation": "Diversification / Efficiency",
        }
    result = _finish_score(
        metric_scores, categories,
        {"growth": return_cov, "quality": risk_cov, "valuation": efficiency_cov},
        risk_score, risk_cov, cfg, profile, weights,
    )
    result["category_labels"] = category_labels
    return result


def _score_conglomerate(metrics: Mapping[str, float], cfg: dict) -> dict:
    """Use book value, ROE, earnings growth, and balance-sheet metrics for Berkshire."""
    result = _score_operating_company(metrics, cfg, "financial_conglomerate")
    # Berkshire's book value and ROE are more informative than operating margin.
    quality = weighted_available(
        {
            "return_on_equity": normalize_piecewise(metrics.get("return_on_equity", np.nan), 0.02, 0.10, 0.20, True),
            "current_ratio": normalize_piecewise(metrics.get("current_ratio", np.nan), 0.50, 1.00, 2.00, True),
            "debt_to_equity": normalize_piecewise(metrics.get("debt_to_equity", np.nan), 0.20, 1.00, 3.00, False),
        },
        {"return_on_equity": 0.45, "current_ratio": 0.20, "debt_to_equity": 0.35},
    )
    valuation = weighted_available(
        {"price_to_book": normalize_piecewise(metrics.get("price_to_book", np.nan), 0.8, 1.5, 3.0, False),
         "trailing_pe": normalize_piecewise(metrics.get("trailing_pe", np.nan), 10.0, 20.0, 40.0, False)},
        {"price_to_book": 0.60, "trailing_pe": 0.40},
    )
    result["quality_score"] = quality[0]
    result["valuation_score"] = valuation[0]
    result["data_confidence"] = (
        cfg["scoring"]["category_weights"]["growth"] * (1.0 if _finite(result["growth_score"]) else 0.0)
        + cfg["scoring"]["category_weights"]["quality"] * quality[1]
        + cfg["scoring"]["category_weights"]["valuation"] * valuation[1]
    ) / sum(cfg["scoring"]["category_weights"].values()) * 100.0
    result["fundamental_score"], _ = weighted_available(
        {"growth": result["growth_score"], "quality": result["quality_score"], "valuation": result["valuation_score"]},
        cfg["scoring"]["category_weights"],
    )
    return result


def score_security(metrics: Mapping[str, float], cfg: dict) -> dict:
    profile = metrics.get("analysis_profile", "operating_company")
    if profile in {"equity_etf", "fixed_income_etf"}:
        return _score_fund(metrics, cfg, profile)
    if profile in {"financial_conglomerate", "financial_company"}:
        result = _score_financial_stock(metrics, cfg)
        result["category_labels"] = {
            "growth": "Growth",
            "quality": "Financial Quality",
            "valuation": "Valuation",
        }
        return result
    result = _score_operating_company(metrics, cfg, profile)
    result["category_labels"] = {
        "growth": "Growth",
        "quality": "Quality / Financial Strength",
        "valuation": "Valuation",
    }
    return result

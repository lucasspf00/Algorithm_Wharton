from __future__ import annotations

from typing import Mapping
import math
import numpy as np


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def normalize_piecewise(value, weak, neutral, strong, higher_is_better=True) -> float:
    if not all(_finite(v) for v in (value, weak, neutral, strong)):
        return np.nan
    value, weak, neutral, strong = map(float, (value, weak, neutral, strong))
    if not weak < neutral < strong:
        return np.nan
    if value <= weak:
        score = -100.0
    elif value >= strong:
        score = 100.0
    elif value <= neutral:
        score = -100.0 + 100.0 * (value - weak) / (neutral - weak)
    else:
        score = 100.0 * (value - neutral) / (strong - neutral)
    return float(np.clip(score if higher_is_better else -score, -100, 100))


def normalize_lower_better(value, strong, neutral, weak) -> float:
    if not all(_finite(v) for v in (value, strong, neutral, weak)) or not strong < neutral < weak:
        return np.nan
    return normalize_piecewise(value, strong, neutral, weak, higher_is_better=False)


def normalize_symmetric(value, cap, higher_is_better=True) -> float:
    if not _finite(value) or not _finite(cap) or cap <= 0:
        return np.nan
    score = float(np.clip(float(value) / float(cap), -1, 1) * 100)
    return score if higher_is_better else -score


def weighted_available(values: Mapping[str, float], weights: Mapping[str, float]) -> tuple[float, float]:
    intended = {key: max(float(weights.get(key, 0.0)), 0.0) for key in values}
    total = sum(intended.values())
    available = {key: weight for key, weight in intended.items() if weight > 0 and _finite(values[key])}
    available_total = sum(available.values())
    if total <= 0 or available_total <= 0:
        return np.nan, 0.0
    score = sum(float(values[key]) * weight for key, weight in available.items()) / available_total
    return float(np.clip(score, -100, 100)), available_total / total


def _growth_branch(metrics: Mapping[str, float], prefix: str, cfg: dict) -> tuple[float, float, dict]:
    caps = cfg["scoring"]["growth_caps"]
    horizons = cfg["scoring"]["growth_horizon_weights"]
    scores, weights = {}, {}
    for period, horizon in (("1y", "one_year"), ("3y", "three_year"), ("5y", "five_year")):
        key = f"{prefix}_growth_{period}"
        scores[key] = normalize_symmetric(metrics.get(key, np.nan), caps.get(key, .3))
        weights[key] = horizons[horizon]
    score, coverage = weighted_available(scores, weights)
    return score, coverage, scores


def _return_branch(metrics: Mapping[str, float], cfg: dict) -> tuple[float, float, dict]:
    profile = metrics.get("analysis_profile")
    settings = cfg["scoring"]["fixed_income"] if profile == "fixed_income_etf" else cfg["scoring"]["equity_etf"]
    horizons = cfg["scoring"]["growth_horizon_weights"]
    scores, weights = {}, {}
    for period, horizon in (("1y", "one_year"), ("3y", "three_year"), ("5y", "five_year")):
        key = f"total_return_{period}"
        scores[key] = normalize_symmetric(metrics.get(key, np.nan), settings["return_caps"][period])
        weights[key] = horizons[horizon]
    score, coverage = weighted_available(scores, weights)
    return score, coverage, scores


def _risk_components(metrics: Mapping[str, float], cfg: dict) -> tuple[dict, float, float]:
    stock = cfg["scoring"]["stock"]
    values = {
        key: normalize_lower_better(metrics.get(key, np.nan), *thresholds)
        for key, thresholds in stock["risk_thresholds"].items()
    }
    score, coverage = weighted_available(values, stock["risk_weights"])
    return values, score, coverage


def _add(rows, group, metric, raw, score, weight, source, thresholds="N/A"):
    rows.append({
        "group": group,
        "metric": metric,
        "raw_value": raw if _finite(raw) else np.nan,
        "score": score if _finite(score) else np.nan,
        "intended_weight": float(weight),
        "source": source,
        "thresholds": thresholds,
    })


def _finish(engine, groups, weights, breakdown, risk_score=np.nan, risk_coverage=0.0,
            risk_weight=0.0, fundamental_weight=1.0):
    score, coverage = weighted_available(groups, weights)
    parts = {"fundamental": score, "risk": risk_score}
    final, _ = weighted_available(parts, {"fundamental": fundamental_weight, "risk": risk_weight})
    confidence = (
        fundamental_weight * coverage + risk_weight * risk_coverage
    ) / max(fundamental_weight + risk_weight, 1e-12)
    return {
        "main_score": final,
        "final_score": final,
        "fundamental_score": score,
        "risk_score": risk_score,
        "data_confidence": confidence * 100,
        "group_scores": groups,
        "analysis_profile": engine,
        "scoring_engine": engine.replace("_", " ").title(),
        "breakdown": breakdown,
        "final_coverage": confidence,
    }


def _stock_score(metrics: Mapping[str, float], cfg: dict, financial: bool) -> dict:
    settings = cfg["scoring"]["stock"]
    engine = "financial_stock" if financial else "standard_stock"
    breakdown, metric_scores = [], {}
    eps, _, eps_scores = _growth_branch(metrics, "eps", cfg)
    fcf, _, fcf_scores = _growth_branch(metrics, "fcf", cfg)
    if financial:
        revenue, _, revenue_scores = _growth_branch(metrics, "revenue", cfg)
        income_growth, _, income_scores = _growth_branch(metrics, "net_income", cfg)
        growth, growth_cov = weighted_available(
            {"revenue": revenue, "eps": eps, "net_income": income_growth},
            settings["financial_growth_weights"],
        )
        for branch_name, scores in (
            ("Revenue Growth", revenue_scores),
            ("EPS Growth", eps_scores),
            ("Net Income Growth", income_scores),
        ):
            for key, score in scores.items():
                metric_scores[key] = score
                _add(
                    breakdown, branch_name, key, metrics.get(key), score,
                    cfg["scoring"]["growth_horizon_weights"][{"1y": "one_year", "3y": "three_year", "5y": "five_year"}[key[-2:]]],
                    "Calculated from Yahoo annual statements",
                    f"±{cfg['scoring']['growth_caps'][key]:.0%}; horizons 20% / 50% / 30%",
                )
    else:
        revenue, _, revenue_scores = _growth_branch(metrics, "revenue", cfg)
        growth, growth_cov = weighted_available(
            {"revenue": revenue, "eps": eps, "fcf": fcf},
            settings["standard_growth_weights"],
        )
    if not financial:
        for group, scores in (("EPS Growth", eps_scores), ("FCF Growth", fcf_scores)):
            for key, score in scores.items():
                metric_scores[key] = score
                _add(
                    breakdown, group, key, metrics.get(key), score,
                    cfg["scoring"]["growth_horizon_weights"][{"1y": "one_year", "3y": "three_year", "5y": "five_year"}[key[-2:]]],
                    "Calculated from Yahoo annual statements",
                    f"±{cfg['scoring']['growth_caps'][key]:.0%}; horizons 20% / 50% / 30%",
                )
    if not financial:
        metric_scores.update(revenue_scores)
        for key, score in revenue_scores.items():
            _add(
                breakdown, "Revenue Growth", key, metrics.get(key), score,
                settings["standard_growth_weights"]["revenue"]
                * cfg["scoring"]["growth_horizon_weights"][{"1y": "one_year", "3y": "three_year", "5y": "five_year"}[key[-2:]]],
                "Calculated from Yahoo annual statements; scored with EPS and FCF growth",
                f"±{cfg['scoring']['growth_caps'][key]:.0%}; horizons 20% / 50% / 30%",
            )

    net_debt_fcf = metrics.get("net_debt_to_fcf", np.nan)
    fcf_raw = metrics.get("free_cash_flow", np.nan)
    if financial:
        net_debt_score = np.nan
    elif _finite(fcf_raw) and float(fcf_raw) < 0:
        net_debt_score = -100.0 if _finite(metrics.get("net_debt")) and float(metrics["net_debt"]) > 0 else np.nan
    elif _finite(net_debt_fcf):
        anchors = settings["net_debt_fcf"]
        net_debt_score = normalize_lower_better(
            net_debt_fcf, anchors["strong"], anchors["neutral"], anchors["weak"]
        )
    else:
        net_debt_score = np.nan
    coverage = metrics.get("interest_coverage", np.nan)
    cov_anchors = settings["interest_coverage"]
    coverage_score = normalize_piecewise(
        coverage, cov_anchors["weak"], cov_anchors["neutral"], cov_anchors["strong"]
    )
    if financial:
        financial_values = {
            "return_on_equity": normalize_symmetric(metrics.get("return_on_equity", np.nan), settings["financial_quality_caps"]["return_on_equity"]),
            "return_on_assets": normalize_symmetric(metrics.get("return_on_assets", np.nan), settings["financial_quality_caps"]["return_on_assets"]),
            "net_margin": normalize_symmetric(metrics.get("net_margin", np.nan), settings["financial_quality_caps"]["net_margin"]),
            "earnings_consistency": normalize_piecewise(
                metrics.get("earnings_consistency", np.nan),
                *settings["financial_quality_caps"]["earnings_consistency"],
            ),
        }
        quality, quality_cov = weighted_available(financial_values, settings["financial_quality_weights"])
        metric_scores.update(financial_values)
        for key, score in financial_values.items():
            _add(breakdown, "Financial Quality", key, metrics.get(key), score, settings["financial_quality_weights"][key], "Yahoo financial statements; missing metrics excluded", str(settings["financial_quality_caps"][key]))
    else:
        quality, quality_cov = weighted_available(
            {"net_debt_fcf": net_debt_score, "interest_coverage": coverage_score},
            {"net_debt_fcf": settings["net_debt_fcf_weight"], "interest_coverage": settings["interest_coverage_weight"]},
        )
        metric_scores.update({"net_debt_to_fcf": net_debt_score, "interest_coverage": coverage_score})
        _add(breakdown, "Financial Strength", "Net Debt / FCF", net_debt_fcf, net_debt_score, settings["net_debt_fcf_weight"], "Calculated from balance sheet and cash flow", str(settings["net_debt_fcf"]))
        _add(breakdown, "Financial Strength", "Interest Coverage", coverage, coverage_score, settings["interest_coverage_weight"], "Yahoo reported financial metric", str(settings["interest_coverage"]))

    pe = metrics.get("trailing_pe", np.nan)
    pe_anchors = settings["pe"]
    pe_score = normalize_lower_better(pe, pe_anchors["strong"], pe_anchors["neutral"], pe_anchors["weak"]) if _finite(pe) and pe > 0 else np.nan
    fcf_yield = metrics.get("fcf_yield", np.nan)
    rf = float(cfg["optimizer"]["risk_free_rate"])
    if _finite(fcf_raw) and float(fcf_raw) < 0:
        yield_score = -100.0
    elif _finite(fcf_yield):
        yield_score = normalize_piecewise(fcf_yield, rf - .02, rf, rf + .04)
    else:
        yield_score = np.nan
    if financial:
        pb = metrics.get("price_to_book", np.nan)
        pb_anchors = settings["financial_pb"]
        pb_score = normalize_lower_better(pb, *pb_anchors) if _finite(pb) and pb > 0 else np.nan
        earnings_yield = metrics.get("earnings_yield", np.nan)
        earnings_yield_score = normalize_piecewise(earnings_yield, rf - .02, rf, rf + .04)
        valuation_weights = settings["financial_valuation_weights"]
        valuation, val_cov = weighted_available(
            {"pe": pe_score, "pb": pb_score, "earnings_yield": earnings_yield_score},
            valuation_weights,
        )
        metric_scores.update({"trailing_pe": pe_score, "price_to_book": pb_score, "earnings_yield": earnings_yield_score})
        _add(breakdown, "Financial Valuation", "Trailing P/E", pe, pe_score, valuation_weights["pe"], "Yahoo; non-positive P/E unavailable", str(pe_anchors))
        _add(breakdown, "Financial Valuation", "Price / Book", pb, pb_score, valuation_weights["pb"], "Yahoo", str(pb_anchors))
        _add(breakdown, "Financial Valuation", "Earnings Yield", earnings_yield, earnings_yield_score, valuation_weights["earnings_yield"], "Yahoo earnings yield; normalized against configured risk-free rate", f"{rf - .02:.1%} / {rf:.1%} / {rf + .04:.1%}")
    else:
        valuation, val_cov = weighted_available(
            {"pe": pe_score, "fcf_yield": yield_score},
            {"pe": settings["pe_weight"], "fcf_yield": settings["fcf_yield_weight"]},
        )
        metric_scores.update({"trailing_pe": pe_score, "fcf_yield": yield_score})
        _add(breakdown, "Valuation", "Trailing P/E", pe, pe_score, settings["pe_weight"], "Yahoo; non-positive P/E unavailable", str(pe_anchors))
        _add(breakdown, "Valuation", "FCF Yield", fcf_yield, yield_score, settings["fcf_yield_weight"], "FCF / market capitalization; risk-free rate is configurable", f"{rf - .02:.1%} / {rf:.1%} / {rf + .04:.1%}")

    growth_group = "Financial Growth" if financial else "Earnings & Cash Flow"
    quality_group = "Financial Quality" if financial else "Financial Strength"
    groups = {growth_group: growth, quality_group: quality, "Valuation": valuation}
    group_weights = {
        growth_group: settings["growth_weight"],
        quality_group: settings["financial_strength_weight"],
        "Valuation": settings["valuation_weight"],
    }
    fundamental_coverage = sum(
        weight * cov for weight, cov in zip(
            group_weights.values(), (growth_cov, quality_cov, val_cov)
        )
    ) / sum(group_weights.values())
    risk_values, risk_score, risk_cov = _risk_components(metrics, cfg)
    for key, score in risk_values.items():
        metric_scores[key] = score
        _add(breakdown, "Risk", key, metrics.get(key), score, settings["risk_weights"][key], "Calculated from price history; beta versus SPY", str(settings["risk_thresholds"][key]))
    result = _finish(engine, groups, group_weights, breakdown, risk_score, risk_cov, settings["risk_weight"], settings["fundamental_weight"])
    result["data_confidence"] = (
        settings["fundamental_weight"] * fundamental_coverage + settings["risk_weight"] * risk_cov
    ) * 100
    result.update({
        "growth_score": growth,
        "quality_score": quality,
        "valuation_score": valuation,
        "metric_scores": metric_scores,
        "group_coverage": {"growth": growth_cov, "quality": quality_cov, "valuation": val_cov, "risk": risk_cov},
        "category_labels": {
            "growth": "Financial Growth" if financial else "Growth",
            "quality": "Financial Quality" if financial else "Financial Strength",
            "valuation": "Valuation",
        },
    })
    return result


def _equity_etf_score(metrics: Mapping[str, float], cfg: dict) -> dict:
    settings = cfg["scoring"]["equity_etf"]
    breakdown, metric_scores = [], {}
    returns, ret_cov, return_scores = _return_branch(metrics, cfg)
    risk_values, risk, risk_cov = _risk_components(metrics, cfg)
    holdings = metrics.get("holdings_count", np.nan)
    top10 = metrics.get("top10_concentration", np.nan)
    sector_max = metrics.get("largest_sector_weight", np.nan)
    expense = metrics.get("expense_ratio", np.nan)
    volume = metrics.get("average_dollar_volume_30d", np.nan)
    div_parts = {
        "holdings": normalize_piecewise(holdings, *settings["holdings_thresholds"]),
        "top10": normalize_lower_better(top10, *settings["top10_thresholds"]),
        "sector": normalize_lower_better(sector_max, *settings["sector_thresholds"]),
    }
    diversification, div_cov = weighted_available(div_parts, {
        "holdings": settings["holdings_weight"],
        "top10": settings["top10_weight"],
        "sector": settings["sector_concentration_weight"],
    })
    cost_score = normalize_lower_better(expense, *settings["expense_ratio_thresholds"])
    v_thresholds = settings["dollar_volume_thresholds"]
    liquidity_score = normalize_piecewise(np.log10(volume), *np.log10(v_thresholds)) if _finite(volume) and volume > 0 else np.nan
    efficiency, efficiency_cov = weighted_available(
        {"diversification": diversification, "cost": cost_score, "liquidity": liquidity_score},
        {"diversification": .50, "cost": .25, "liquidity": .25},
    )
    groups = {"Return": returns, "Risk / Resilience": risk, "Diversification / Efficiency": efficiency}
    weights = {"Return": .35, "Risk / Resilience": .35, "Diversification / Efficiency": .30}
    for key, score in return_scores.items():
        metric_scores[key] = score
        _add(breakdown, "Return", key, metrics.get(key), score, cfg["scoring"]["growth_horizon_weights"][{"1y": "one_year", "3y": "three_year", "5y": "five_year"}[key[-2:]]], "Adjusted historical price return", f"±{settings['return_caps'][key[-2:]]:.0%}")
    sources = {
        "holdings": (holdings, "Number of Holdings", "Yahoo fund data"),
        "top10": (top10, "Top-10 Concentration", "Yahoo reported holdings"),
        "sector": (sector_max, "Largest Sector Weight", "Yahoo fund sector exposure"),
    }
    for key, (raw, label, source) in sources.items():
        metric_scores[key] = div_parts[key]
        threshold_map = {
            "holdings": settings["holdings_thresholds"],
            "top10": settings["top10_thresholds"],
            "sector": settings["sector_thresholds"],
        }
        _add(breakdown, "Diversification", label, raw, div_parts[key], 1/3, source, str(threshold_map[key]))
    metric_scores.update({"expense_ratio": cost_score, "daily_dollar_volume": liquidity_score, **risk_values})
    _add(breakdown, "Efficiency", "Expense Ratio", expense, cost_score, .25, "Yahoo; configurable default anchors", str(settings["expense_ratio_thresholds"]))
    _add(breakdown, "Efficiency", "30-Day Average Dollar Volume", volume, liquidity_score, .25, "Calculated from close × volume", str(settings["dollar_volume_thresholds"]))
    for key, score in risk_values.items():
        _add(breakdown, "Risk", key, metrics.get(key), score, cfg["scoring"]["stock"]["risk_weights"][key], "Calculated from adjusted price history", str(cfg["scoring"]["stock"]["risk_thresholds"][key]))
    result = _finish("equity_etf", groups, weights, breakdown)
    result["data_confidence"] = (.35 * ret_cov + .35 * risk_cov + .30 * efficiency_cov) * 100
    result.update({
        "growth_score": returns, "quality_score": risk, "valuation_score": efficiency,
        "risk_score": risk,
        "metric_scores": metric_scores,
        "group_coverage": {"return": ret_cov, "risk": risk_cov, "efficiency": efficiency_cov},
        "category_labels": {"growth": "Return", "quality": "Risk / Resilience", "valuation": "Diversification / Efficiency"},
    })
    return result


def _fixed_income_score(metrics: Mapping[str, float], cfg: dict) -> dict:
    settings = cfg["scoring"]["fixed_income"]
    breakdown, metric_scores = [], {}
    duration = metrics.get("effective_duration", np.nan)
    target = float(settings["target_duration"])
    duration_gap = abs(float(duration) - target) if _finite(duration) else np.nan
    duration_score = normalize_lower_better(duration_gap, *settings["duration_gap_thresholds"])
    credit = metrics.get("weighted_credit_score", np.nan)
    risk_values, risk, risk_cov = _risk_components(metrics, cfg)
    stability, stability_cov = weighted_available(
        {"duration_match": duration_score, "credit_quality": credit, "historical_risk": risk},
        {"duration_match": .35, "credit_quality": .35, "historical_risk": .30},
    )
    returns, ret_cov, return_scores = _return_branch(metrics, cfg)
    ytm, treasury = metrics.get("yield_to_maturity", np.nan), metrics.get("comparable_treasury_ytm", np.nan)
    spread = float(ytm) - float(treasury) if _finite(ytm) and _finite(treasury) else np.nan
    spread_score = normalize_piecewise(spread, *settings["yield_spread_thresholds"])
    return_score, return_coverage = weighted_available(
        {"historical_return": returns, "ytm_spread": spread_score},
        {"historical_return": .70, "ytm_spread": .30},
    )
    expense = metrics.get("expense_ratio", np.nan)
    expense_score = normalize_lower_better(expense, *cfg["scoring"]["equity_etf"]["expense_ratio_thresholds"])
    volume = metrics.get("average_dollar_volume_30d", np.nan)
    volume_caps = cfg["scoring"]["equity_etf"]["dollar_volume_thresholds"]
    liquidity_score = normalize_piecewise(np.log10(volume), *np.log10(volume_caps)) if _finite(volume) and volume > 0 else np.nan
    efficiency, efficiency_cov = weighted_available(
        {"cost": expense_score, "liquidity": liquidity_score}, {"cost": .5, "liquidity": .5}
    )
    groups = {"Stability / Risk": stability, "Return": return_score, "Efficiency / Liquidity": efficiency}
    weights = {"Stability / Risk": .40, "Return": .30, "Efficiency / Liquidity": .30}
    for key, score in return_scores.items():
        metric_scores[key] = score
        _add(breakdown, "Return", key, metrics.get(key), score, 1/3, "Adjusted historical price return", f"±{settings['return_caps'][key[-2:]]:.0%}")
    metric_scores.update({"effective_duration_gap": duration_score, "weighted_credit_score": credit, "ytm_spread": spread_score, "expense_ratio": expense_score, "daily_dollar_volume": liquidity_score, **risk_values})
    _add(breakdown, "Stability / Risk", "Duration Gap to Target", duration_gap, duration_score, .35, f"Target duration {target:.2f} years; configurable", str(settings["duration_gap_thresholds"]))
    _add(breakdown, "Stability / Risk", "Weighted Credit Quality", credit, credit, .35, "Yahoo fund credit-rating exposure; N/A if unreported", str(settings["credit_rating_scores"]))
    _add(breakdown, "Return", "YTM Spread vs Comparable Treasury", spread, spread_score, .30, "Requires separately reported ETF and comparable Treasury YTM", str(settings["yield_spread_thresholds"]))
    _add(breakdown, "Efficiency", "Expense Ratio", expense, expense_score, .15, "Yahoo; configurable default anchors", str(cfg["scoring"]["equity_etf"]["expense_ratio_thresholds"]))
    _add(breakdown, "Efficiency", "30-Day Average Dollar Volume", volume, liquidity_score, .15, "Calculated from close × volume", str(volume_caps))
    for key, score in risk_values.items():
        _add(breakdown, "Historical Risk", key, metrics.get(key), score, cfg["scoring"]["stock"]["risk_weights"][key], "Calculated from adjusted price history", str(cfg["scoring"]["stock"]["risk_thresholds"][key]))
    result = _finish("fixed_income_etf", groups, weights, breakdown)
    result["data_confidence"] = (.40 * stability_cov + .30 * return_coverage + .30 * efficiency_cov) * 100
    result.update({
        "growth_score": return_score, "quality_score": stability, "valuation_score": efficiency,
        "risk_score": risk,
        "metric_scores": metric_scores,
        "group_coverage": {"stability": stability_cov, "return": return_coverage, "efficiency": efficiency_cov},
        "category_labels": {"growth": "Return", "quality": "Stability / Risk", "valuation": "Efficiency / Liquidity"},
    })
    return result


def score_security(metrics: Mapping[str, float], cfg: dict) -> dict:
    profile = metrics.get("analysis_profile", "operating_company")
    if profile == "equity_etf":
        return _equity_etf_score(metrics, cfg)
    if profile == "fixed_income_etf":
        return _fixed_income_score(metrics, cfg)
    return _stock_score(metrics, cfg, financial=profile in {"financial_company", "financial_conglomerate", "financial_stock"})

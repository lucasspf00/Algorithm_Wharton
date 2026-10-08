from __future__ import annotations

from typing import Mapping
import math
from datetime import date
import numpy as np

from src.optimizer import normalize_sector_weights


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
    result = float(np.clip(float(value) / float(cap), -1, 1) * 100)
    return result if higher_is_better else -result


def weighted_available(values: Mapping[str, float], weights: Mapping[str, float]) -> tuple[float, float]:
    total = sum(max(float(weights.get(key, 0.0)), 0.0) for key in values)
    valid = {
        key: max(float(weights.get(key, 0.0)), 0.0)
        for key, value in values.items()
        if _finite(value) and float(weights.get(key, 0.0)) > 0
    }
    available = sum(valid.values())
    if total <= 0 or available <= 0:
        return np.nan, 0.0
    score = sum(float(values[key]) * weight for key, weight in valid.items()) / available
    return float(np.clip(score, -100, 100)), available / total


def _metric_row(
    metric: str,
    raw_value,
    score,
    metric_weight: float,
    category: str,
    category_weight: float,
    parent_weight: float,
    source: str,
    thresholds="N/A",
) -> dict:
    return {
        "metric": metric,
        "raw_value": raw_value if _finite(raw_value) else np.nan,
        "score": score if _finite(score) else np.nan,
        "metric_weight": float(metric_weight),
        "category": category,
        "category_weight": float(category_weight),
        "parent_weight": float(parent_weight),
        "final_weight": float(metric_weight) * float(category_weight) * float(parent_weight),
        "source": source,
        "thresholds": thresholds,
    }


def _assemble(
    profile: str,
    groups: dict[str, float],
    group_weights: dict[str, float],
    rows: list[dict],
    final_formula: dict[str, float],
    labels: dict[str, str],
    category_parent_weight: float = 1.0,
    risk_score=np.nan,
    confidence_rows: list[tuple[float, bool]] | None = None,
    warning: str = "",
    extra: dict | None = None,
) -> dict:
    coverage_items = confidence_rows or [
        (row["final_weight"], _finite(row["score"])) for row in rows
    ]
    total_weight = sum(weight for weight, _ in coverage_items)
    available_weight = sum(weight for weight, valid in coverage_items if valid)
    confidence = available_weight / total_weight if total_weight else 0.0
    low_confidence = confidence < .60 and not np.isclose(confidence, .60, rtol=0.0, atol=1e-12)
    is_stock = profile in {"STANDARD_STOCK", "SPECIAL_FINANCIAL_STOCK"}
    if is_stock:
        fundamental_groups = {
            key: value for key, value in groups.items() if key != "Risk & Resilience"
        }
        fundamental, fundamental_coverage = weighted_available(fundamental_groups, group_weights)
        final_parts = {"Fundamental": fundamental, "Risk & Resilience": groups.get("Risk & Resilience", np.nan)}
        final_weights = {"Fundamental": .80, "Risk & Resilience": .20}
        final_score, _ = weighted_available(final_parts, final_weights)
        active_group_weight = sum(
            group_weights[key] for key, value in fundamental_groups.items() if _finite(value)
        )
        active_parent_weight = sum(
            weight for key, weight in final_weights.items() if _finite(final_parts[key])
        )
    else:
        final_score, _ = weighted_available(groups, final_formula)
        active_group_weight = sum(
            group_weights.get(key, final_formula.get(key, 0.0))
            for key, value in groups.items() if _finite(value)
        )
        active_parent_weight = 1.0
        fundamental, fundamental_coverage = np.nan, 0.0

    for row in rows:
        category = row["category"]
        row["category_score"] = groups.get(category, np.nan)
        category_metric_weights = [
            other["metric_weight"] for other in rows
            if other["category"] == category and _finite(other["score"])
        ]
        available_metric_weight = sum(category_metric_weights)
        if not _finite(row["score"]) or available_metric_weight <= 0:
            row["effective_weight"] = 0.0
            row["final_contribution"] = np.nan
            continue
        if is_stock:
            if category == "Risk & Resilience":
                parent = .20 / active_parent_weight if _finite(final_parts["Risk & Resilience"]) else 0.0
                category_weight = parent
            else:
                outer_fund_weight = .80 / active_parent_weight if _finite(fundamental) else 0.0
                category_weight = (
                    outer_fund_weight
                    * group_weights[category] / active_group_weight
                    if _finite(groups.get(category)) and active_group_weight > 0
                    else 0.0
                )
        else:
            category_base = group_weights.get(category, final_formula.get(category, 0.0))
            category_weight = (
                category_base / active_group_weight
                if _finite(groups.get(category)) and active_group_weight > 0
                else 0.0
            )
        effective_weight = category_weight * row["metric_weight"] / available_metric_weight
        row["effective_weight"] = effective_weight
        row["final_contribution"] = row["score"] * effective_weight
    return {
        "main_score": float(np.clip(final_score, -100, 100)) if _finite(final_score) else np.nan,
        "final_score": float(np.clip(final_score, -100, 100)) if _finite(final_score) else np.nan,
        "group_scores": groups,
        "group_weights": group_weights,
        "metric_scores": {row["metric"]: row["score"] for row in rows},
        "breakdown": rows,
        "data_confidence": confidence,
        "low_confidence": low_confidence,
        "missing_metrics": [row["metric"] for row in rows if not _finite(row["score"])],
        "scoring_model": profile,
        "scoring_engine": {
            "STANDARD_STOCK": "Standard Operating Stock",
            "SPECIAL_FINANCIAL_STOCK": "Special Financial Stock",
            "EQUITY_ETF": "Equity ETF",
            "FIXED_INCOME_ETF": "Fixed-Income ETF",
        }.get(profile, profile.replace("_", " ").title()),
        "category_labels": labels,
        "risk_score": risk_score,
        "warning": warning,
        "score_formula": final_formula,
        **(extra or {}),
    }


def _risk_scores(metrics: Mapping[str, float], cfg: dict) -> tuple[dict, float, float]:
    settings = cfg["scoring"]["stock"]
    specs = settings["risk_thresholds"]
    scores = {
        key: normalize_lower_better(metrics.get(key, np.nan), *thresholds)
        for key, thresholds in specs.items()
    }
    result, coverage = weighted_available(scores, settings["risk_weights"])
    return scores, result, coverage


def _stock_score(metrics: Mapping[str, float], cfg: dict, special_financial: bool) -> dict:
    settings = cfg["scoring"]["stock"]
    risk_weight = float(settings["risk_weight"])
    fundamental_weight = float(settings["fundamental_weight"])
    groups, rows = {}, []
    group_weights = {
        "Earnings & Cash Flow": float(settings["growth_weight"]),
        "Financial Strength": float(settings["financial_strength_weight"]),
        "Valuation": float(settings["valuation_weight"]),
    }

    # EPS and FCF use only their 3-year CAGR in the final stock score.
    growth_weights = {"eps_growth_3y": .50, "fcf_growth_3y": .50}
    growth_scores = {
        key: (
            np.nan
            if special_financial and key == "fcf_growth_3y"
            else normalize_piecewise(metrics.get(key, np.nan), -.10, .05, .20)
        )
        for key in growth_weights
    }
    growth_score, growth_coverage = weighted_available(growth_scores, growth_weights)
    groups["Earnings & Cash Flow"] = growth_score
    for key, weight in growth_weights.items():
        rows.append(_metric_row(
            key.replace("_", " ").upper(),
            np.nan if special_financial and key == "fcf_growth_3y" else metrics.get(key),
            growth_scores[key],
            weight,
            "Earnings & Cash Flow",
            group_weights["Earnings & Cash Flow"],
            fundamental_weight,
            (
                "Not applied to special financial companies; FCF is not used as a comparable growth metric"
                if special_financial and key == "fcf_growth_3y"
                else "3-year CAGR from annual statements; invalid/non-positive endpoints are N/A"
            ),
            "−10% → −100; +5% → 0; +20% → +100",
        ))

    strength_weights = {"net_debt_to_fcf": .50, "interest_coverage": .50}
    debt_anchors = settings["net_debt_fcf"]
    net_debt = metrics.get("net_debt", np.nan)
    free_cash_flow = metrics.get("free_cash_flow", np.nan)
    debt_with_nonpositive_fcf = (
        _finite(net_debt)
        and float(net_debt) > 0
        and _finite(free_cash_flow)
        and float(free_cash_flow) <= 0
    )
    strength_scores = {
        "net_debt_to_fcf": (
            np.nan if special_financial else (
                -100.0 if debt_with_nonpositive_fcf else normalize_lower_better(
                    metrics.get("net_debt_to_fcf"),
                    debt_anchors["strong"],
                    debt_anchors["neutral"],
                    debt_anchors["weak"],
                )
            )
        ),
        "interest_coverage": (
            np.nan
            if special_financial
            else normalize_piecewise(
                metrics.get("interest_coverage"),
                settings["interest_coverage"]["weak"],
                settings["interest_coverage"]["neutral"],
                settings["interest_coverage"]["strong"],
            )
        ),
    }
    strength_score, strength_coverage = weighted_available(strength_scores, strength_weights)
    groups["Financial Strength"] = strength_score
    for key, weight in strength_weights.items():
        thresholds = settings["net_debt_fcf"] if key == "net_debt_to_fcf" else settings["interest_coverage"]
        rows.append(_metric_row(
            key.replace("_", " ").title(),
            (
                np.nan
                if special_financial and key in {"net_debt_to_fcf", "interest_coverage"}
                else (
                    np.nan
                    if key == "net_debt_to_fcf" and debt_with_nonpositive_fcf
                    else metrics.get(key)
                )
            ),
            strength_scores[key],
            weight,
            "Financial Strength",
            group_weights["Financial Strength"],
            fundamental_weight,
            (
                "Not applied to special financial companies"
                if special_financial and key in {"net_debt_to_fcf", "interest_coverage"}
                else (
                    "Scored −100 because net debt is positive while FCF is non-positive; the ratio itself is undefined or misleading"
                    if key == "net_debt_to_fcf" and debt_with_nonpositive_fcf
                    else (
                        metrics.get(
                            "interest_coverage_source",
                            "EBIT / absolute interest expense; Yahoo-reported ratio only when statement values are unavailable",
                        )
                        if key == "interest_coverage"
                        else "Yahoo / derived balance-sheet metric"
                    )
                )
            ),
            str(thresholds),
        ))

    valuation_weights = {"trailing_pe": .50, "fcf_yield": .50}
    pe = metrics.get("trailing_pe", np.nan)
    pe_score = normalize_lower_better(pe, 10, 25, 50) if _finite(pe) and float(pe) > 0 else np.nan
    nonpositive_fcf = _finite(free_cash_flow) and float(free_cash_flow) <= 0
    fcf_yield_score = (
        (
            -100.0
            if nonpositive_fcf
            else normalize_piecewise(
                metrics.get("fcf_yield"),
                cfg["optimizer"]["risk_free_rate"] - .02,
                cfg["optimizer"]["risk_free_rate"],
                cfg["optimizer"]["risk_free_rate"] + .04,
            )
        )
        if not special_financial
        else np.nan
    )
    valuation_scores = {"trailing_pe": pe_score, "fcf_yield": fcf_yield_score}
    valuation_score, valuation_coverage = weighted_available(valuation_scores, valuation_weights)
    groups["Valuation"] = valuation_score
    for key, weight in valuation_weights.items():
        raw_value = metrics.get(key)
        if key == "trailing_pe" and (not _finite(raw_value) or float(raw_value) <= 0):
            raw_value = np.nan
        if special_financial and key == "fcf_yield":
            raw_value = np.nan
        rows.append(_metric_row(
            "P/E" if key == "trailing_pe" else "FCF Yield",
            raw_value,
            valuation_scores[key],
            weight,
            "Valuation",
            group_weights["Valuation"],
            fundamental_weight,
            "Positive trailing P/E only; non-positive values are N/A" if key == "trailing_pe" else
            ("Not applied to special financial companies" if special_financial else "FCF / market capitalization"),
            "≤10x → +100; 25x → 0; ≥50x → −100" if key == "trailing_pe" else
            f"{cfg['optimizer']['risk_free_rate'] - .02:.1%} / {cfg['optimizer']['risk_free_rate']:.1%} / {cfg['optimizer']['risk_free_rate'] + .04:.1%}",
        ))

    risk_values, risk_score, risk_coverage = _risk_scores(metrics, cfg)
    risk_weights = settings["risk_weights"]
    for key, weight in risk_weights.items():
        rows.append(_metric_row(
            key.replace("_", " ").title(),
            (
                np.nan
                if special_financial and key == "interest_coverage"
                else metrics.get(key)
            ),
            risk_values[key],
            weight,
            "Risk & Resilience",
            1.0,
            risk_weight,
            "Calculated from adjusted price history; beta versus SPY",
            str(settings["risk_thresholds"][key]),
        ))

    coverage_rows = []
    for value, weights, group_weight, parent_weight in (
        (growth_scores, growth_weights, group_weights["Earnings & Cash Flow"], fundamental_weight),
        (strength_scores, strength_weights, group_weights["Financial Strength"], fundamental_weight),
        (valuation_scores, valuation_weights, group_weights["Valuation"], fundamental_weight),
        (risk_values, risk_weights, 1.0, risk_weight),
    ):
        coverage_rows.extend(
            (float(weight) * group_weight * parent_weight, _finite(value[key]))
            for key, weight in weights.items()
        )
    formula = {
        "Earnings & Cash Flow": fundamental_weight * group_weights["Earnings & Cash Flow"],
        "Financial Strength": fundamental_weight * group_weights["Financial Strength"],
        "Valuation": fundamental_weight * group_weights["Valuation"],
        "Risk & Resilience": risk_weight,
    }
    fundamental_score, _ = weighted_available(groups, group_weights)
    return _assemble(
        "SPECIAL_FINANCIAL_STOCK" if special_financial else "STANDARD_STOCK",
        {**groups, "Risk & Resilience": risk_score},
        group_weights,
        rows,
        formula,
        {
            "growth": "Growth",
            "quality": "Financial Strength",
            "valuation": "Valuation",
        },
        risk_score=risk_score,
        confidence_rows=coverage_rows,
        warning="Financial-company score has reduced comparability." if special_financial else "",
        extra={
            "growth_score": growth_score,
            "quality_score": strength_score,
            "valuation_score": valuation_score,
            "fundamental_score": (
                fundamental_score
            ),
            "group_coverage": {
                "growth": growth_coverage,
                "quality": strength_coverage,
                "valuation": valuation_coverage,
                "risk": risk_coverage,
            },
        },
    )


def _equity_etf_score(metrics: Mapping[str, float], cfg: dict) -> dict:
    settings = cfg["scoring"]["equity_etf"]
    group_weights = {
        "Diversification": .35,
        "Cost / Efficiency": .30,
        "Risk": .20,
        "Liquidity": .15,
    }
    metric_weights = {
        "Top-10 Concentration": .40,
        "Largest Sector Excess vs S&P 500": .35,
        "Holdings Count": .25,
    }
    reliable_sector_weights = normalize_sector_weights(metrics.get("sector_weightings"))
    benchmark_weights = normalize_sector_weights(
        metrics.get("sector_benchmark_weightings")
        or cfg["optimizer"].get("sector_benchmark_weights", {})
    )
    largest_sector = (
        max(reliable_sector_weights, key=reliable_sector_weights.get)
        if reliable_sector_weights else None
    )
    largest_sector_excess = (
        reliable_sector_weights[largest_sector] - benchmark_weights[largest_sector]
        if largest_sector and largest_sector in benchmark_weights
        else np.nan
    )
    metrics_scores = {
        "Top-10 Concentration": normalize_lower_better(metrics.get("top10_concentration"), *settings["top10_thresholds"]),
        "Largest Sector Excess vs S&P 500": normalize_lower_better(
            largest_sector_excess, *settings["sector_excess_thresholds"]
        ),
        "Holdings Count": normalize_piecewise(metrics.get("holdings_count"), *settings["holdings_thresholds"]),
    }
    categories = {}
    rows = []
    for name, score in metrics_scores.items():
        raw_key = {
            "Top-10 Concentration": "top10_concentration",
            "Largest Sector Excess vs S&P 500": "largest_sector_weight",
            "Holdings Count": "holdings_count",
        }[name]
        thresholds = {
            "Top-10 Concentration": settings["top10_thresholds"],
            "Largest Sector Excess vs S&P 500": settings["sector_excess_thresholds"],
            "Holdings Count": settings["holdings_thresholds"],
        }[name]
        raw_value = largest_sector_excess if name == "Largest Sector Excess vs S&P 500" else metrics.get(raw_key)
        source = (
            (
                f"{largest_sector} ETF sector weight minus corresponding "
                f"{metrics.get('sector_benchmark_source', 'S&P 500 reference')} weight"
            )
            if name == "Largest Sector Excess vs S&P 500" and reliable_sector_weights and benchmark_weights
            else (
                "ETF or benchmark sector exposure unavailable; metric excluded from score"
                if name == "Largest Sector Excess vs S&P 500"
                else "Reported ETF holdings"
            )
        )
        rows.append(_metric_row(name, raw_value, score, metric_weights[name], "Diversification", group_weights["Diversification"], 1.0, source, str(thresholds)))
    categories["Diversification"], div_cov = weighted_available(metrics_scores, metric_weights)

    cost_score = normalize_lower_better(metrics.get("expense_ratio"), *settings["expense_ratio_thresholds"])
    categories["Cost / Efficiency"] = cost_score
    rows.append(_metric_row("Expense Ratio", metrics.get("expense_ratio"), cost_score, 1.0, "Cost / Efficiency", group_weights["Cost / Efficiency"], 1.0, "Fund expense ratio; no reliable tracking-difference data available", str(settings["expense_ratio_thresholds"])))

    risk_values, risk_score, risk_cov = _risk_scores(metrics, cfg)
    risk_weights = {"max_drawdown": .55, "annualized_volatility": .45}
    risk_scores = {key: risk_values[key] for key in risk_weights}
    categories["Risk"], _ = weighted_available(risk_scores, risk_weights)
    for key, weight in risk_weights.items():
        rows.append(_metric_row(key.replace("_", " ").title(), metrics.get(key), risk_scores[key], weight, "Risk", group_weights["Risk"], 1.0, "Calculated from adjusted price history", str(cfg["scoring"]["stock"]["risk_thresholds"][key])))

    volume = metrics.get("average_dollar_volume_30d", np.nan)
    liquidity_score = normalize_piecewise(
        np.log10(volume),
        *np.log10(settings["dollar_volume_thresholds"]),
    ) if _finite(volume) and volume > 0 else np.nan
    categories["Liquidity"] = liquidity_score
    rows.append(_metric_row("Average Daily Dollar Volume", volume, liquidity_score, 1.0, "Liquidity", group_weights["Liquidity"], 1.0, "30-observation average of adjusted close × daily volume", str(settings["dollar_volume_thresholds"])))

    coverage_rows = [
        (row["final_weight"], _finite(row["score"])) for row in rows
    ]
    result = _assemble(
        "EQUITY_ETF",
        categories,
        group_weights,
        rows,
        group_weights,
        {"growth": "Diversification", "quality": "Cost / Efficiency", "valuation": "Liquidity"},
        risk_score=risk_score,
        confidence_rows=coverage_rows,
        extra={
            "growth_score": categories["Diversification"],
            "quality_score": categories["Cost / Efficiency"],
            "valuation_score": categories["Liquidity"],
            "risk_score": categories["Risk"],
            "sector_exposure_complete": bool(reliable_sector_weights),
            "sector_benchmark_source": metrics.get("sector_benchmark_source", "Configured reference"),
        },
    )
    return result


def _fixed_income_score(metrics: Mapping[str, float], cfg: dict) -> dict:
    settings = cfg["scoring"]["fixed_income"]
    group_weights = {
        "Duration / Term-Fit Heuristic": float(settings["stability_term_fit_weight"]),
        "Credit Safety": float(settings["credit_safety_weight"]),
        "Yield": float(settings["yield_weight"]),
        "Liquidity": float(settings["liquidity_weight"]),
    }
    asset_duration = metrics.get("effective_duration", np.nan)
    analysis_year = int(metrics.get("analysis_year", date.today().year))
    target_term = max(0, 2033 - analysis_year)
    duration_gap = (
        abs(float(asset_duration) - target_term)
        if _finite(asset_duration) else np.nan
    )
    duration_score = normalize_lower_better(duration_gap, *settings["duration_gap_anchors"])
    credit_score = metrics.get("weighted_credit_score", np.nan)
    ytm = metrics.get("yield_to_maturity", np.nan)
    comparable_treasury_ytm = metrics.get("comparable_treasury_ytm", np.nan)
    yield_spread = metrics.get("yield_spread", np.nan)
    if not _finite(yield_spread):
        yield_spread = (
            float(ytm) - float(comparable_treasury_ytm)
            if _finite(ytm) and _finite(comparable_treasury_ytm)
            else np.nan
        )
    yield_score = (
        normalize_piecewise(yield_spread, *settings["yield_spread_anchors"])
        if _finite(yield_spread)
        else np.nan
    )
    volume = metrics.get("average_dollar_volume_30d", np.nan)
    liquidity_score = normalize_piecewise(
        np.log10(volume), *np.log10(settings["dollar_volume_thresholds"])
    ) if _finite(volume) and volume > 0 else np.nan
    groups = {
        "Duration / Term-Fit Heuristic": duration_score,
        "Credit Safety": credit_score,
        "Yield": yield_score,
        "Liquidity": liquidity_score,
    }
    metric_values = [
        ("Duration Gap to 2033 Target", duration_gap, duration_score, "Duration / Term-Fit Heuristic", group_weights["Duration / Term-Fit Heuristic"], f"Absolute gap from {target_term}-year target term (2033 minus model year {analysis_year}); term-fit heuristic, not liability immunization", str(settings["duration_gap_anchors"])),
        ("Weighted Credit Score", credit_score, credit_score, "Credit Safety", group_weights["Credit Safety"], "Reported bond-rating exposure; unavailable if not reported", str(settings["credit_rating_scores"])),
        ("YTM Spread", yield_spread, yield_score, "Yield", group_weights["Yield"], "YTM minus comparable-duration Treasury YTM; N/A if either reported yield is unavailable; SEC Yield is not substituted", str(settings["yield_spread_anchors"])),
        ("Average Daily Dollar Volume", volume, liquidity_score, "Liquidity", group_weights["Liquidity"], "30-observation average of adjusted close × daily volume", str(settings["dollar_volume_thresholds"])),
    ]
    rows = [
        _metric_row(name, raw, score, 1.0, category, group_weights[category], 1.0, source, threshold)
        for name, raw, score, category, _, source, threshold in metric_values
    ]
    risk_values, risk_score, _ = _risk_scores(metrics, cfg)
    return _assemble(
        "FIXED_INCOME_ETF",
        groups,
        group_weights,
        rows,
        group_weights,
        {"growth": "Duration / Term-Fit Heuristic", "quality": "Credit Safety", "valuation": "Yield"},
        risk_score=risk_score,
        confidence_rows=[(weight, _finite(groups[key])) for key, weight in group_weights.items()],
        extra={
            "growth_score": duration_score,
            "quality_score": credit_score,
            "valuation_score": yield_score,
            "asset_duration_is_maturity": False,
            "analysis_year": analysis_year,
            "target_term_years": target_term,
            "duration_gap_years": duration_gap,
            "group_coverage": {key.lower().replace(" ", "_"): float(_finite(value)) for key, value in groups.items()},
        },
    )


def score_security(metrics: Mapping[str, float], cfg: dict) -> dict:
    profile = metrics.get("scoring_model", "STANDARD_STOCK")
    if profile == "EQUITY_ETF":
        return _equity_etf_score(metrics, cfg)
    if profile == "FIXED_INCOME_ETF":
        return _fixed_income_score(metrics, cfg)
    return _stock_score(
        metrics,
        cfg,
        special_financial=profile == "SPECIAL_FINANCIAL_STOCK",
    )

from __future__ import annotations

import os
from datetime import date

import certifi
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())

from src.config import load_config, merge_config_defaults
from src.formatting import format_pct as fmt_pct
from src.data import (
    analyze_security,
    batch_analyze,
    download_price_period,
    normalize_ticker,
    price_frame,
)
from src.optimizer import (
    laura_sector_allocation_table,
    monte_carlo_optimize,
    normalize_sector_weights,
    portfolio_benchmark_comparison,
    sector_allocation_table,
    sector_exposure_coverage,
)
from src.scoring import score_security
from src.stress import HISTORICAL_WINDOWS, historical_portfolio_stress, hypothetical_stress

st.set_page_config(page_title="Laura Gao Quant System", layout="wide")
st.title("Laura Gao Quantitative Investment System")
st.caption("Four-tab security scoring, portfolio construction, stress testing, and decision-support app.")
st.info("Research support for the Wharton Global High School Investment Competition. Official competition materials and team instructions remain authoritative.")

st.session_state.cfg = merge_config_defaults(
    load_config(),
    st.session_state.get("cfg", {}),
)
cfg = st.session_state.cfg
defaults = {
    "candidate_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "LLY", "XOM", "NEE"],
    "candidate_raw": pd.DataFrame(),
    "candidate_prices": pd.DataFrame(),
    "candidate_errors": {},
    "candidate_scored": pd.DataFrame(),
    "optimizer_result": None,
    "single_result": None,
    "selected_portfolio": "Laura Goal Portfolio",
    "hist_stress": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_settings():
    st.session_state.cfg = load_config()
    st.session_state.optimizer_result = None
    single = st.session_state.single_result
    if single:
        st.session_state.single_result = (single[0], score_security(single[0], st.session_state.cfg))
    raw = st.session_state.candidate_raw
    if not raw.empty:
        st.session_state.candidate_scored = pd.DataFrame(
            [{**row.to_dict(), **score_security(row.to_dict(), st.session_state.cfg)} for _, row in raw.iterrows()]
        )
    for widget_key in (
        "settings_max_weight", "settings_equity_min", "settings_equity_max",
        "settings_fixed_min", "settings_fixed_max", "settings_sector_tolerance",
        "settings_risk_free", "settings_benchmark",
        "settings_liquidity", "settings_return_shrinkage", "settings_lookback", "settings_simulations",
        "settings_expense_strong", "settings_expense_neutral", "settings_expense_weak",
    ):
        st.session_state.pop(widget_key, None)


def fmt_num(value, digits=2):
    try:
        value = float(value)
        return f"{value:,.{digits}f}" if np.isfinite(value) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def fmt_score(value):
    try:
        value = float(value)
        return f"{value:+.1f}" if np.isfinite(value) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def fmt_money(value, digits=0):
    try:
        value = float(value)
        return f"${value:,.{digits}f}" if np.isfinite(value) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def pct_input(label, value, low, high, step, key):
    return float(st.number_input(
        label, min_value=float(low), max_value=float(high),
        value=float(value) * 100, step=float(step), format="%.2f", key=key,
    )) / 100


def score_row(metrics: dict) -> dict:
    return {**metrics, **score_security(metrics, cfg)}


def scoring_model_label(model: str) -> str:
    return {
        "STANDARD_STOCK": "Standard Operating Stock",
        "SPECIAL_FINANCIAL_STOCK": "Special Financial Stock",
        "EQUITY_ETF": "Equity ETF",
        "FIXED_INCOME_ETF": "Fixed-Income ETF",
    }.get(model, model)


def show_security(metrics: dict, scored: dict):
    if scored.get("warning"):
        st.warning(scored["warning"])
    for warning in metrics.get("return_sanity_warnings", []):
        st.warning(warning)
    missing_metrics = scored.get("missing_metrics", [])
    if missing_metrics:
        st.caption("Unavailable metrics excluded from scoring and reweighting: " + ", ".join(missing_metrics))
    low_confidence = scored.get("low_confidence", scored["data_confidence"] < 0.60)
    if low_confidence:
        st.error("PROVISIONAL / LOW CONFIDENCE — less than 60% of intended score inputs are available.")
    st.markdown(f"**{metrics.get('company', metrics.get('ticker'))}** · `{metrics.get('ticker', 'N/A')}`")
    scoring_model = metrics.get("scoring_model", "NOT_APPLICABLE")
    st.caption(
        f"Scoring Model: **{scoring_model_label(scoring_model)}**. "
        f"{metrics.get('scoring_classification_reason', '')} "
        "Portfolio-sector classification is independent. Scores from different asset engines are not directly comparable."
    )
    score_title = "Final Asset Score"
    if low_confidence:
        score_title += " — PROVISIONAL / LOW CONFIDENCE"
    st.markdown(
        f"<div style='font-size:3.1rem;font-weight:750;line-height:1.2'>"
        f"{score_title}&nbsp; {fmt_score(scored['main_score'])}"
        f"<span style='font-size:1rem;font-weight:400'> / 100</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Score range: -100 to +100. Scores produced by different asset engines are not directly comparable.")
    st.markdown(
        f"**Yahoo sector:** {metrics.get('original_yahoo_sector', metrics.get('sector', 'N/A'))} &nbsp;|&nbsp; "
        f"**Laura sector bucket:** {metrics.get('laura_sector_bucket', 'N/A')} &nbsp;|&nbsp; "
        f"**Category:** {metrics.get('category', 'N/A')} &nbsp;|&nbsp; "
        f"**Industry:** {metrics.get('industry', 'N/A')} &nbsp;|&nbsp; "
        f"**Asset type:** {metrics.get('asset_class', 'N/A')} &nbsp;|&nbsp; "
        f"**Data date:** {metrics.get('data_date', 'N/A')}"
    )
    price = metrics.get("current_price")
    price_text = (
        f"{metrics.get('price_currency', 'N/A')} {fmt_num(price, 2)}"
        if np.isfinite(_numeric(price)) else "N/A"
    )
    st.metric("Current Price", price_text)
    st.caption(
        f"As of: {metrics.get('price_as_of', 'N/A')} · "
        f"Source: {metrics.get('price_source', 'N/A')} · "
        f"{metrics.get('price_freshness', 'Availability unknown')}"
    )
    labels = scored["category_labels"]
    if scoring_model in {"STANDARD_STOCK", "SPECIAL_FINANCIAL_STOCK"}:
        columns = st.columns(3)
        columns[0].metric("Fundamental Score", fmt_score(scored.get("fundamental_score")))
        columns[1].metric("Risk Score", fmt_score(scored["risk_score"]))
        columns[2].metric("Data Confidence", fmt_pct(scored["data_confidence"]))
    else:
        columns = st.columns(5)
        columns[0].metric(labels["growth"], fmt_score(scored["growth_score"]))
        columns[1].metric(labels["quality"], fmt_score(scored["quality_score"]))
        columns[2].metric(labels["valuation"], fmt_score(scored["valuation_score"]))
        columns[3].metric("Risk Score", fmt_score(scored["risk_score"]))
        columns[4].metric("Data Confidence", fmt_pct(scored["data_confidence"]))

    details = []
    diagnostics_by_name = {
        str(diagnostic.get("metric", "")).strip().lower(): diagnostic
        for diagnostic in metrics.get("data_diagnostics", [])
    }
    score_to_diagnostic = {
        "EPS GROWTH 3Y": "eps 3y cagr",
        "FCF GROWTH 3Y": "fcf 3y cagr",
        "Net Debt To Fcf": "net debt / fcf",
        "P/E": "trailing p/e",
        "Max Drawdown": "maximum drawdown",
        "Beta": "beta vs spy",
    }
    for row in scored.get("breakdown", []):
        raw = row["raw_value"]
        name = row["metric"].lower()
        diagnostic = diagnostics_by_name.get(
            score_to_diagnostic.get(row["metric"], name)
        , {})
        start_date = diagnostic.get("start_date") or diagnostic.get("period_start")
        end_date = diagnostic.get("end_date") or diagnostic.get("period_end")
        period_display = (
            f"{start_date}–{end_date}"
            if start_date or end_date else diagnostic.get("elapsed_years", "N/A")
        )
        if "expense" in name:
            raw_display = fmt_pct(raw, 3)
        elif any(term in name for term in (
            "growth", "yield", "return", "drawdown", "volatility",
            "spread", "excess", "concentration", "sector weight",
        )):
            raw_display = fmt_pct(raw)
        elif "net debt / fcf" in name:
            raw_display = fmt_num(raw)
        elif "dollar volume" in name or "free cash flow" in name:
            raw_display = fmt_money(raw)
        else:
            raw_display = fmt_num(raw)
        details.append({
            "Score group": row["category"],
            "Metric": row["metric"],
            "Observed value": raw_display,
            "Normalized points": fmt_score(row["score"]),
            "Local metric weight": fmt_pct(row.get("metric_weight")),
            "Category weight × parent weight": fmt_pct(
                row.get("category_weight", 0.0) * row.get("parent_weight", 0.0)
            ),
            "Original intended final weight": fmt_pct(row.get("final_weight")),
            "Effective final-score weight": fmt_pct(row.get("effective_weight", 0.0)),
            "Weighted contribution to final score": fmt_score(row.get("final_contribution")),
            "Category score": fmt_score(row.get("category_score")),
            "Final score": fmt_score(scored["main_score"]),
            "Raw field": diagnostic.get("raw_field", "See metric source / scoring rule"),
            "Period / Date": period_display,
            "Fallback used": diagnostic.get("fallback_used", ""),
            "N/A reason": diagnostic.get("n/a_reason", ""),
            "Scoring thresholds / anchors": row.get("thresholds", "N/A"),
            "Source / rule": row["source"],
        })
    if details:
        with st.expander("Reproducible score: raw metric → points → weight → contribution → category → final", expanded=True):
            st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)

    diagnostics = metrics.get("data_diagnostics", [])
    if diagnostics:
        with st.expander("Data Diagnostics"):
            diagnostics_frame = pd.DataFrame(diagnostics)
            diagnostics_frame["raw_value"] = diagnostics_frame["raw_value"].map(
                lambda value: "N/A" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
            )
            diagnostics_frame["final_interpreted_value"] = [
                format_diagnostic_value(
                    row.get("metric", ""),
                    row.get("final_interpreted_value"),
                    metrics.get("price_currency", ""),
                )
                for row in diagnostics_frame.to_dict("records")
            ]
            diagnostics_frame = diagnostics_frame.rename(columns={
                "metric": "Metric",
                "source": "Source",
                "raw_field": "Raw field",
                "raw_value": "Raw value",
                "final_interpreted_value": "Final interpreted value",
                "status": "Status",
                "n/a_reason": "N/A reason",
                "fallback_used": "Fallback used",
                "start_date": "Financial/price start date",
                "end_date": "Financial/price end date",
                "period_start": "Risk-window start date",
                "period_end": "Risk-window end date",
                "elapsed_years": "Elapsed years",
                "start_value": "CAGR start value",
                "end_value": "CAGR end value",
            })
            st.dataframe(diagnostics_frame, hide_index=True, use_container_width=True)
            if metrics.get("info_error"):
                st.warning(f"Yahoo quote metadata retrieval issue: {metrics['info_error']}")
            if metrics.get("income_statement_error"):
                st.caption(f"Income statement retrieval: {metrics['income_statement_error']}")
            if metrics.get("cashflow_statement_error"):
                st.caption(f"Cash-flow statement retrieval: {metrics['cashflow_statement_error']}")
            if metrics.get("balance_sheet_error"):
                st.caption(f"Balance sheet retrieval: {metrics['balance_sheet_error']}")
            for error_key, label in (
                ("quarterly_income_error", "Quarterly income statement"),
                ("ttm_income_error", "TTM income statement"),
                ("quarterly_cashflow_error", "Quarterly cash flow"),
                ("ttm_cashflow_error", "TTM cash flow"),
                ("quarterly_balance_sheet_error", "Quarterly balance sheet"),
                ("valuation_error", "Valuation measures"),
            ):
                if metrics.get(error_key):
                    st.caption(f"{label}: {metrics[error_key]}")

    if metrics.get("asset_class") == "stock":
        growth_rows = []
        is_financial = metrics.get("scoring_model") == "SPECIAL_FINANCIAL_STOCK"
        for label, key in [
            ("EPS 3Y CAGR", "eps_growth_3y"),
            ("FCF 3Y CAGR", "fcf_growth_3y"),
        ]:
            observed = (
                "N/A"
                if is_financial and label.startswith("FCF")
                else fmt_pct(metrics.get(key))
            )
            growth_rows.append({"Metric": label, "Observed": observed})
        st.markdown("**Most important raw growth metrics**")
        st.dataframe(pd.DataFrame(growth_rows), hide_index=True, use_container_width=True)
    elif metrics.get("asset_class") == "fixed_income":
        fund_rows = [
            ("Distribution Yield", fmt_pct(metrics.get("distribution_yield"))),
            ("30-Day SEC Yield", fmt_pct(metrics.get("sec_yield_30d"))),
            ("Yield to Maturity", fmt_pct(metrics.get("yield_to_maturity"))),
            ("Comparable-duration Treasury YTM", fmt_pct(metrics.get("comparable_treasury_ytm"))),
            ("YTM spread", fmt_pct(metrics.get("yield_spread"))),
            ("Effective Duration (years)", fmt_num(metrics.get("effective_duration"))),
            ("2033 target term (years)", fmt_num(scored.get("target_term_years"))),
            ("Duration gap to 2033 target (years)", fmt_num(scored.get("duration_gap_years"))),
            ("Weighted Average Maturity (years)", fmt_num(metrics.get("weighted_average_maturity"))),
            ("Expense Ratio", fmt_pct(metrics.get("expense_ratio"), 3)),
            (
                "Expense ratio source value",
                f"{fmt_num(metrics.get('expense_ratio_raw'), 6)} ({metrics.get('expense_ratio_source', 'Unavailable')})",
            ),
            ("30-Day Average Dollar Volume", fmt_money(metrics.get("average_dollar_volume_30d"))),
            ("Treasury / Credit Exposure", str(metrics.get("treasury_credit_exposure", "N/A"))),
        ]
        st.dataframe(pd.DataFrame(fund_rows, columns=["Fixed-income metric", "Reported value"]), hide_index=True, use_container_width=True)
        st.caption(
            f"Duration / Term-Fit Heuristic compares reported duration with a {scored.get('target_term_years', 'N/A')}-year target term "
            f"(2033 minus model year {scored.get('analysis_year', 'N/A')}); this is a duration-based heuristic, not exact liability immunization. ETF duration is not maturity."
        )
    elif metrics.get("asset_class") == "equity_etf":
        if not scored.get("sector_exposure_complete", False):
            st.warning("Sector exposure incomplete — the ETF is excluded from sector look-through and ETF sector-concentration scoring.")
        if str(scored.get("sector_benchmark_source", "")).startswith("Configured"):
            st.warning(
                "Live S&P 500 sector weights were unavailable; ETF concentration uses the configured reference. "
                + str(metrics.get("sector_benchmark_error", ""))
            )
        etf_rows = [
            ("Number of holdings", fmt_num(metrics.get("holdings_count"), 0)),
            ("Top-10 concentration", fmt_pct(metrics.get("top10_concentration"))),
            (
                "Largest sector weight",
                fmt_pct(metrics.get("largest_sector_weight"))
                if scored.get("sector_exposure_complete", False) else "N/A",
            ),
            ("Expense ratio", fmt_pct(metrics.get("expense_ratio"), 3)),
            (
                "Expense ratio source value",
                f"{fmt_num(metrics.get('expense_ratio_raw'), 6)} ({metrics.get('expense_ratio_source', 'Unavailable')})",
            ),
            ("30-Day Average Dollar Volume", fmt_money(metrics.get("average_dollar_volume_30d"))),
        ]
        st.dataframe(pd.DataFrame(etf_rows, columns=["Equity ETF metric", "Reported value"]), hide_index=True, use_container_width=True)
    if metrics.get("asset_class") == "fixed_income":
        st.caption("Duration is an interest-rate sensitivity statistic, not a claim that this ETF matures at that time. Fixed-income Yield scoring uses YTM minus a comparable-duration Treasury YTM; SEC Yield is not substituted, and the score is N/A when the comparison yield is unavailable.")


def candidate_display(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        financial = row.get("scoring_model") == "SPECIAL_FINANCIAL_STOCK"
        rows.append({
            "Ticker": row.get("ticker"),
            "Company": row.get("company"),
            "Final Asset Score": fmt_score(row.get("main_score")),
            "Confidence status": "PROVISIONAL / LOW CONFIDENCE" if row.get("low_confidence", _numeric(row.get("data_confidence")) < .60) else "Available",
            "Scoring Model": scoring_model_label(row.get("scoring_model", "NOT_APPLICABLE")),
            "Asset type": row.get("asset_class"),
            "Latest price": (
                f"{row.get('price_currency', 'N/A')} {fmt_num(row.get('current_price'), 2)}"
                if np.isfinite(_numeric(row.get("current_price"))) else "N/A"
            ),
            "Sector / category": row.get("sector") if row.get("sector") not in (None, "", "Unknown") else row.get("category"),
            "Return 1Y": fmt_pct(row.get("total_return_1y")),
            "3Y CAGR": fmt_pct(row.get("total_return_3y")),
            "Return data warning": "; ".join(row.get("return_sanity_warnings", [])) or "—",
            "EPS growth 3Y": fmt_pct(row.get("eps_growth_3y")),
            "FCF growth 3Y": "N/A" if financial else fmt_pct(row.get("fcf_growth_3y")),
            "Net Debt / FCF": "N/A" if financial else fmt_num(row.get("net_debt_to_fcf")),
            "Interest coverage": fmt_num(row.get("interest_coverage")),
            "Expense ratio": fmt_pct(row.get("expense_ratio"), 3),
            "Trailing P/E": fmt_num(row.get("trailing_pe")),
            "FCF Yield": "N/A" if financial else fmt_pct(row.get("fcf_yield")),
            "Max drawdown": fmt_pct(row.get("max_drawdown")),
            "Volatility": fmt_pct(row.get("annualized_volatility")),
            "Duration (years)": fmt_num(row.get("effective_duration")),
            "YTM": fmt_pct(row.get("yield_to_maturity")),
            "YTM spread": fmt_pct(row.get("yield_spread")),
            "30-Day Dollar Volume": fmt_money(row.get("average_dollar_volume_30d")),
            "Risk": fmt_score(row.get("risk_score")),
            "Data confidence": fmt_pct(row.get("data_confidence", np.nan)),
        })
    return pd.DataFrame(rows)


def _numeric(value) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def format_diagnostic_value(metric: str, value, currency: str = "") -> str:
    number = _numeric(value)
    if not np.isfinite(number):
        return "N/A"
    name = metric.lower()
    if any(word in name for word in (
        "growth", "yield", "return", "drawdown", "volatility", "margin",
        "expense", "concentration", "weight", "confidence", "spread",
    )):
        return fmt_pct(number, 3 if "expense" in name else 1)
    if "price" in name:
        return f"{currency} {fmt_num(number, 2)}".strip()
    if "market capitalization" in name or "dollar volume" in name or "free cash flow" in name or name == "net debt":
        return fmt_money(number)
    if any(word in name for word in ("p/e", "coverage", "debt / fcf", "net debt / fcf")):
        return f"{fmt_num(number, 2)}x"
    if "duration" in name or "maturity" in name:
        return f"{fmt_num(number, 2)} years"
    return fmt_num(number, 2)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_candidate_batch(tickers: tuple[str, ...], cache_dir: str, ttl_hours: float, metrics_version: int):
    app_cfg = {"data": {"cache_dir": cache_dir, "cache_ttl_hours": ttl_hours}}
    return batch_analyze(tickers, app_cfg)


tabs = st.tabs([
    "1 Portfolio Construction & Optimization",
    "2 Asset Score",
    "3 Stress Test",
    "4 Settings",
])

with tabs[0]:
    st.subheader("Portfolio Construction & Optimization")
    st.caption("Add mixed stocks and ETFs, load their shared cached analysis, then set optional long-only weights. ETF and company scores use distinct, non-comparable engines.")
    initial_text = ", ".join(st.session_state.candidate_tickers)
    ticker_text = st.text_area("Candidate tickers", value=initial_text, height=72, key="candidate_text")
    left, right = st.columns([1, 3])
    if left.button("Load / refresh candidates", key="candidate_load"):
        tickers = list(dict.fromkeys(
            normalize_ticker(part.strip())
            for part in ticker_text.replace("\n", ",").split(",")
            if part.strip()
        ))
        prior_tickers = st.session_state.candidate_tickers
        st.session_state.candidate_tickers = tickers
        if not tickers:
            st.error("Enter at least one ticker.")
            st.session_state.candidate_raw = pd.DataFrame()
            st.session_state.candidate_prices = pd.DataFrame()
            st.session_state.candidate_scored = pd.DataFrame()
            st.session_state.optimizer_result = None
            st.session_state.pop("manual_weight_table", None)
            st.session_state.pop("manual_weight_editor", None)
        else:
            if tickers != prior_tickers:
                st.session_state.pop("manual_weight_table", None)
                st.session_state.pop("manual_weight_editor", None)
                st.session_state.optimizer_result = None
            with st.spinner("Loading available market data; each failed ticker is isolated..."):
                raw, prices, errors = cached_candidate_batch(
                    tuple(tickers), cfg["data"]["cache_dir"], float(cfg["data"]["cache_ttl_hours"]), 5
                )
            st.session_state.candidate_raw = raw
            st.session_state.candidate_prices = price_frame(prices)
            st.session_state.candidate_errors = errors
            st.session_state.candidate_scored = pd.DataFrame(
                [score_row(row.to_dict()) for _, row in raw.iterrows()]
            ) if not raw.empty else pd.DataFrame()
            st.session_state.optimizer_result = None
    right.caption("Yahoo Finance data is cached locally for the configured cache period. Clicking load again refreshes Streamlit's candidate view.")

    metadata = pd.DataFrame(
        columns=[
            "ticker", "asset_class", "sector", "industry",
            "sector_weightings", "average_dollar_volume_30d",
        ]
    )
    if not st.session_state.candidate_scored.empty:
        st.markdown("### Candidate scores and major metrics")
        st.dataframe(candidate_display(st.session_state.candidate_scored), hide_index=True, use_container_width=True)
        st.markdown("### User-defined portfolio weights")
        editable = st.session_state.candidate_scored[["ticker", "asset_class", "sector", "main_score"]].copy()
        if "manual_weight_table" not in st.session_state:
            editable["weight"] = 1 / max(1, len(editable))
        else:
            saved = st.session_state.manual_weight_table.set_index("ticker")["weight"]
            editable["weight"] = editable["ticker"].map(saved).fillna(0.0)
        previous_manual = st.session_state.get("manual_weight_table")
        edited = st.data_editor(
            editable,
            hide_index=True,
            use_container_width=True,
            key="manual_weight_editor",
            disabled=["ticker", "asset_class", "sector", "main_score"],
            column_config={"weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=1.0, step=0.01, format="percent")},
        )
        current_manual = edited[["ticker", "weight"]].copy()
        if previous_manual is not None and not current_manual.equals(previous_manual.reset_index(drop=True)):
            st.session_state.optimizer_result = None
        st.session_state.manual_weight_table = current_manual
        weight_total = float(pd.to_numeric(edited["weight"], errors="coerce").fillna(0).sum())
        st.caption(f"Current manual weights sum to {fmt_pct(weight_total)}; weights must sum to 100% to include this portfolio.")
        metadata = st.session_state.candidate_scored[
            ["ticker", "asset_class", "sector", "industry", "sector_weightings", "average_dollar_volume_30d"]
        ].copy()
        manual = pd.Series(
            pd.to_numeric(edited["weight"], errors="coerce").fillna(0).to_numpy(),
            index=edited["ticker"],
        )
        if st.button("Run feasible portfolio optimizer", key="optimizer_run"):
            st.session_state.optimizer_result = None
            try:
                benchmark = str(cfg["optimizer"]["benchmark"]).strip().upper()
                benchmark_prices = pd.DataFrame()
                benchmark_symbols = list(dict.fromkeys([benchmark, "SPY", "VOO", "QQQ"])) if benchmark else ["SPY", "VOO", "QQQ"]
                benchmark_start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=int(cfg["optimizer"]["lookback_years"]))).strftime("%Y-%m-%d")
                try:
                    benchmark_prices = download_price_period(benchmark_symbols, benchmark_start, date.today().isoformat())
                except Exception as exc:
                    st.warning(f"SPY / VOO / QQQ benchmarks could not be loaded; candidate optimization will continue without benchmark comparisons: {exc}")
                st.session_state.benchmark_prices = benchmark_prices
                optimizer_metadata = metadata.copy()
                benchmark_sector_weights = cfg["optimizer"]["sector_benchmark_weights"]
                sector_weights_date = "Configured reference"
                try:
                    spy_metrics, _ = analyze_security("SPY", cfg)
                    live_sector_weights = normalize_sector_weights(spy_metrics.get("sector_weightings"))
                    if live_sector_weights:
                        benchmark_sector_weights = live_sector_weights
                        sector_weights_date = spy_metrics.get("data_date", "N/A")
                    else:
                        st.warning("SPY sector weights were incomplete or invalid; using configured reference weights.")
                except Exception as exc:
                    st.warning(f"Live SPY sector weights unavailable; using configured sector reference weights: {exc}")
                result = monte_carlo_optimize(
                    st.session_state.candidate_prices,
                    cfg,
                    metadata=optimizer_metadata,
                    manual_weights=manual if np.isclose(weight_total, 1.0, atol=1e-6) else None,
                    benchmark_sector_weights=benchmark_sector_weights,
                    benchmark_prices=benchmark_prices,
                )
                result["assumptions"]["benchmark_sector_weights_date"] = sector_weights_date
                st.session_state.optimizer_result = result
            except Exception as exc:
                st.error(f"Portfolio optimization could not run: {exc}")

    if st.session_state.candidate_errors:
        with st.expander("Ticker data issues"):
            st.json(st.session_state.candidate_errors)

    result = st.session_state.optimizer_result
    if result:
        if "Laura Goal Portfolio" not in result["portfolios"]:
            st.warning(
                "Laura Goal Portfolio is unavailable: none of the sampled portfolios satisfied all six "
                "Laura equity-sleeve sector ranges with complete ETF look-through. Add eligible holdings "
                "from the missing sector buckets and rerun optimization."
            )
        if result.get("manual_weights_error"):
            st.warning(f"User-defined portfolio was not created: {result['manual_weights_error']}")
        excluded_liquidity = result["constraints"].get("excluded_for_liquidity", [])
        if excluded_liquidity:
            st.warning(
                "Not included in optimization because 30-day average dollar volume was below "
                f"${result['constraints']['minimum_liquidity']:,.0f} or unavailable: {', '.join(excluded_liquidity)}."
            )
        excluded_history = result["constraints"].get("excluded_for_history", [])
        if excluded_history:
            st.warning(f"Not included because usable overlapping price history was unavailable: {', '.join(excluded_history)}.")
        excluded_sector_data = result["constraints"].get("excluded_for_sector_data", [])
        if excluded_sector_data:
            st.warning(
                "Not included in sector-constrained optimization because sector exposure is incomplete: "
                + ", ".join(excluded_sector_data)
            )
        st.markdown("### Feasible portfolios")
        st.caption(
            f"Long-only, fully invested; max holding {cfg['optimizer']['max_weight']:.0%}; "
            f"equity allocation {cfg['optimizer']['minimum_equity_weight']:.0%}–{cfg['optimizer']['maximum_equity_weight']:.0%}; "
            f"fixed income {cfg['optimizer']['minimum_fixed_income_weight']:.0%}–{cfg['optimizer']['maximum_fixed_income_weight']:.0%}. "
            "Sector ranges are compared to S&P 500 weights inside the equity sleeve."
        )
        if result["constraints"]["sector_ranges"]:
            st.caption(f"S&P 500 sector reference date: {result['assumptions']['benchmark_sector_weights_date']}; each represented sector must be within ±{fmt_pct(cfg['optimizer']['sector_tolerance'])} of its reference weight.")
        portfolio_summary = pd.DataFrame([
            {
                "Portfolio": name,
                "Expected annual return": fmt_pct(portfolio["expected_return"]),
                "Annualized volatility": fmt_pct(portfolio["volatility"]),
                "Sharpe ratio": fmt_num(portfolio["sharpe"]),
                "Simulated full-liability funding success": (
                    fmt_pct(portfolio["funding_success_probability"])
                    if name == "Laura Goal Portfolio" else "N/A"
                ),
            }
            for name, portfolio in result["portfolios"].items()
        ])
        st.dataframe(portfolio_summary, hide_index=True, use_container_width=True)
        names = list(result["portfolios"])
        selected = st.selectbox(
            "Portfolio detail", names,
            index=names.index(st.session_state.selected_portfolio) if st.session_state.selected_portfolio in names else 0,
            key="portfolio_detail_select",
        )
        st.session_state.selected_portfolio = selected
        portfolio = result["portfolios"][selected]
        weight_frame = pd.DataFrame({"Ticker": portfolio["weights"].index, "Weight": portfolio["weights"].values})
        st.dataframe(weight_frame.style.format({"Weight": lambda value: fmt_pct(value, 2)}), hide_index=True, use_container_width=True)
        sector_frame = sector_allocation_table(
            portfolio["weights"],
            metadata,
            result["constraints"]["sector_benchmark_weights"],
            float(cfg["optimizer"]["sector_tolerance"]),
        )
        exposure_coverage = sector_exposure_coverage(portfolio["weights"], metadata)
        if not exposure_coverage["complete"]:
            st.warning(
                "Sector exposure incomplete — sector-compliance confidence is reduced. "
                f"Reliable sector look-through covers {fmt_pct(exposure_coverage['coverage'])} of the equity sleeve; "
                f"missing/invalid exposure: {', '.join(exposure_coverage['incomplete_tickers'])}."
            )
        if not sector_frame.empty:
            display_sectors = sector_frame.copy()
            display_sectors["equity_sleeve_weight"] = display_sectors["equity_sleeve_weight"].map(fmt_pct)
            display_sectors["total_portfolio_weight"] = display_sectors["total_portfolio_weight"].map(fmt_pct)
            display_sectors["sp500_benchmark_weight"] = display_sectors["sp500_benchmark_weight"].map(fmt_pct)
            display_sectors["target_low"] = display_sectors["target_low"].map(fmt_pct)
            display_sectors["target_high"] = display_sectors["target_high"].map(fmt_pct)
            display_sectors = display_sectors.rename(columns={
                "sector": "Sector",
                "equity_sleeve_weight": "% of equity sleeve",
                "total_portfolio_weight": "% of total portfolio",
                "sp500_benchmark_weight": "S&P 500 benchmark",
                "target_low": "Target low",
                "target_high": "Target high",
            })
            st.markdown("#### Sector allocation guardrail")
            st.caption("S&P 500 sector comparisons use the equity sleeve denominator; the equivalent whole-portfolio share is shown separately. This is a diversification guardrail, not an expected-return target.")
            st.dataframe(display_sectors, hide_index=True, use_container_width=True)
        laura_sector_frame, laura_coverage = laura_sector_allocation_table(
            portfolio["weights"], metadata
        )
        st.markdown("#### Laura Sector Guardrails")
        st.caption(
            "Laura-specific ranges are diversification guardrails applied to the equity sleeve, "
            "not expected-return forecasts."
        )
        if not laura_coverage["complete"]:
            st.warning(
                "Sector exposure incomplete — Laura sector-compliance checks have reduced confidence. "
                f"Reliable look-through covers {fmt_pct(laura_coverage['coverage'])} of the equity sleeve; "
                f"missing/invalid ETF look-through: {', '.join(laura_coverage['incomplete_tickers'])}."
            )
        laura_display = laura_sector_frame.copy()
        laura_display["equity_sleeve_weight"] = laura_display["equity_sleeve_weight"].map(fmt_pct)
        laura_display["target"] = laura_display["target"].map(fmt_pct)
        laura_display["acceptable_range"] = laura_display["acceptable_range"].map(
            lambda bounds: f"{fmt_pct(bounds[0])}–{fmt_pct(bounds[1])}"
        )
        laura_display["total_portfolio_weight"] = laura_display["total_portfolio_weight"].map(fmt_pct)
        laura_display = laura_display.rename(columns={
            "sector": "Sector",
            "equity_sleeve_weight": "Current % of equity sleeve",
            "target": "Target",
            "acceptable_range": "Acceptable range",
            "total_portfolio_weight": "% of total portfolio",
            "status": "Status",
        })
        st.dataframe(laura_display, hide_index=True, use_container_width=True)
        with st.expander("Exact optimizer assumptions"):
            expected_return_detail = pd.DataFrame({
                "Historical arithmetic annual return": result["historical_expected_returns"],
                "Shrinkage target": result["expected_return_targets"],
                "Adjusted optimizer return": result["expected_returns"],
            }).map(fmt_pct)
            st.markdown("**Historical and adjusted expected returns**")
            st.dataframe(expected_return_detail, use_container_width=True)
            st.json({
                **result["assumptions"],
                "expected_return": "Rp = w'μ; μ = mean(daily simple returns) × 252",
                "variance": "w'Σw; Σ = daily covariance × 252 with configured diagonal shrinkage",
                "volatility": "sqrt(w'Σw)",
                "sharpe": "(Rp - risk-free rate) / volatility",
                "portfolio_random_samples": len(result["simulations"]),
                "portfolio_seed": 42,
                "sector_benchmark_date": result["assumptions"]["benchmark_sector_weights_date"],
            })
        if selected == "Laura Goal Portfolio":
            if portfolio["funding_objective_met"]:
                st.success(f"Laura Goal Portfolio met the simulated {fmt_pct(portfolio['funding_objective_target'])} funding-success objective ({fmt_pct(portfolio['funding_success_probability'])}).")
            else:
                st.warning(f"No feasible frontier candidate met the simulated {fmt_pct(portfolio['funding_objective_target'])} objective; this portfolio has the highest simulated funding success ({fmt_pct(portfolio['funding_success_probability'])}).")
            st.caption(portfolio["construction"])
        frontier = result.get("efficient_frontier", pd.DataFrame())
        simulations = result.get("simulations", pd.DataFrame())
        if not simulations.empty:
            st.markdown(f"### Approximate efficient frontier — {len(simulations):,} feasible samples")
            chart_data = simulations.rename(columns={"volatility": "Annualized volatility", "expected_return": "Expected annual return"})
            fig = px.scatter(
                chart_data, x="Annualized volatility", y="Expected annual return",
                opacity=.35, title="Feasible random portfolios (sample, not exact optima)",
            )
            fig.update_xaxes(tickformat=".1%")
            fig.update_yaxes(tickformat=".1%")
            if not frontier.empty:
                fig.add_scatter(
                    x=frontier["volatility"], y=frontier["expected_return"],
                    mode="lines", name="Sampled frontier",
                )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Laura 2033 contribution and funding check")
        payment = float(cfg["laura"]["annual_payment"])
        payment_count = int(cfg["laura"]["payment_count"])
        liability = payment * payment_count
        st.info(f"Official long-term cash flows are unchanged: $300,000 at the beginning of 2027, $150,000 at the beginning of 2028, no withdrawals before 2033, then {payment_count} beginning-of-year ${payment:,.0f} payments from 2033–2042. WInS gains/losses are excluded.")
        st.metric("Nominal operating commitment", fmt_money(liability))
        recommended = result["portfolios"].get("Laura Goal Portfolio")
        if recommended is None:
            st.metric("Laura Goal Portfolio — Illustrative Expected 2033 Value", "N/A")
            st.dataframe(pd.DataFrame({
                "Percentile": ["5th", "25th", "50th", "75th", "95th"],
                "2033 value": ["N/A"] * 5,
            }), hide_index=True, use_container_width=True)
            st.metric("Funding-Success Probability", "N/A")
            st.warning("No funding simulation was run because there is no portfolio meeting Laura's sector guardrails.")
        else:
            distribution = recommended["funding_value_2033"]
            expected = distribution.mean() if distribution is not None and len(distribution) else np.nan
            st.metric("Laura Goal Portfolio — Illustrative Expected 2033 Value", fmt_money(expected))
            if distribution is not None and len(distribution):
                st.caption("Mean and percentiles come from the same explicit, seeded simulation; they are not guarantees.")
                pct_table = pd.DataFrame({
                    "Percentile": ["5th", "25th", "50th", "75th", "95th"],
                    "2033 value": [fmt_money(distribution.quantile(q)) for q in (.05, .25, .50, .75, .95)],
                })
                st.dataframe(pct_table, hide_index=True, use_container_width=True)
                probability = recommended.get("funding_success_probability", np.nan)
                st.metric("Funding-Success Probability", fmt_pct(probability))
                st.caption(
                    f"{recommended['funding_successful_paths']:,} of {recommended['funding_simulations']:,} paths funded all ten beginning-of-year obligations. "
                    f"Method: {recommended['funding_method']}. Fixed seed: {recommended['funding_seed']}."
                )
                with st.expander("2033 Goal Check Methodology"):
                    st.write(recommended["funding_assumptions"])
                    st.write(
                        "Cash flows: +$300,000 at the beginning of 2027; +$150,000 at the beginning of 2028; "
                        "no withdrawals before 2033; ten beginning-of-year $50,000 operating payments from 2033 through 2042. "
                        "WInS gains/losses are excluded."
                    )
                    st.write(f"Simulations: {recommended['funding_simulations']:,}; fixed seed: {recommended['funding_seed']}.")
            else:
                st.dataframe(pd.DataFrame({
                    "Percentile": ["5th", "25th", "50th", "75th", "95th"],
                    "2033 value": ["N/A"] * 5,
                }), hide_index=True, use_container_width=True)
                st.metric("Funding-Success Probability", "N/A")
                st.warning("2033 simulation was unavailable; no funding-success probability is reported.")

        benchmark_prices = st.session_state.get("benchmark_prices", pd.DataFrame())
        if not benchmark_prices.empty:
            comparisons = portfolio_benchmark_comparison(
                st.session_state.candidate_prices,
                portfolio["weights"],
                benchmark_prices,
                float(cfg["optimizer"]["risk_free_rate"]),
            )
            if not comparisons.empty:
                st.markdown("### SPY / VOO / QQQ benchmark comparison")
                compare_display = comparisons.copy()
                for column in ("annualized_arithmetic_return", "cagr", "volatility", "max_drawdown"):
                    compare_display[column] = compare_display[column].map(fmt_pct)
                compare_display["sharpe"] = compare_display["sharpe"].map(fmt_num)
                compare_display["beta_vs_benchmark"] = compare_display["beta_vs_benchmark"].map(fmt_num)
                compare_display = compare_display.rename(columns={
                    "series": "Series",
                    "benchmark": "Reference",
                    "annualized_arithmetic_return": "Annualized arithmetic return",
                    "cagr": "CAGR",
                    "volatility": "Volatility",
                    "sharpe": "Sharpe",
                    "max_drawdown": "Maximum drawdown",
                    "beta_vs_benchmark": "Beta vs reference",
                    "overlap_start": "Overlap start",
                    "overlap_end": "Overlap end",
                    "overlap_observations": "Daily observations",
                })
                st.dataframe(compare_display, hide_index=True, use_container_width=True)
        st.markdown("### Candidate correlation")
        st.dataframe(result["correlation"].round(2), use_container_width=True)

with tabs[1]:
    st.subheader("Asset Score")
    st.caption("Analyze one security with its automatically detected engine. Ticker aliases are normalized (for example BRK.B → BRK-B).")
    ticker = st.text_input("Ticker", "AAPL", key="asset_score_ticker")
    if st.button("Analyze security", key="asset_score_run"):
        try:
            normalized = normalize_ticker(ticker)
            with st.spinner(f"Analyzing {normalized}..."):
                metrics, _ = analyze_security(normalized, cfg)
            st.session_state.single_result = (metrics, score_security(metrics, cfg))
        except Exception as exc:
            st.session_state.single_result = None
            st.error(f"Could not analyze {ticker}: {exc}")
    if st.session_state.single_result:
        show_security(*st.session_state.single_result)

with tabs[2]:
    st.subheader("Stress Test")
    result = st.session_state.optimizer_result
    if not result:
        st.warning("Load candidates and run the portfolio optimizer first.")
    else:
        names = list(result["portfolios"])
        selected = st.selectbox("Portfolio", names, key="stress_portfolio")
        weights = result["portfolios"][selected]["weights"]
        event = st.selectbox("Historical event", list(HISTORICAL_WINDOWS), key="stress_event")
        if st.button("Run historical stress", key="stress_run"):
            start, end = HISTORICAL_WINDOWS[event]
            try:
                prices = download_price_period(list(weights.index), start, end)
                st.session_state.hist_stress = historical_portfolio_stress(prices, weights)
            except Exception as exc:
                st.session_state.hist_stress = {"available": False, "reason": str(exc)}
        hist = st.session_state.hist_stress
        if hist and hist.get("available"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Buy-and-hold portfolio return", fmt_pct(hist["portfolio_return"]))
            c2.metric("Maximum drawdown", fmt_pct(hist["max_drawdown"]))
            c3.metric("Weight coverage", fmt_pct(hist["weight_coverage"]))
            detail = pd.DataFrame({
                "Ticker": hist["contributions"].index,
                "Holding return": hist["holding_returns"].reindex(hist["contributions"].index).values,
                "Portfolio contribution": hist["contributions"].values,
            })
            st.dataframe(detail.style.format({
                "Holding return": lambda value: fmt_pct(value, 2),
                "Portfolio contribution": lambda value: fmt_pct(value, 2),
            }), hide_index=True, use_container_width=True)
            if hist["weight_coverage"] < .999:
                st.caption("Missing history is excluded and available initial weights are renormalized; coverage is reported.")
        elif hist:
            st.warning(hist.get("reason", "Historical stress unavailable."))
        c1, c2, c3 = st.columns(3)
        equity = pct_input("Equity shock", cfg["stress"]["custom_equity_shock"], -90, 50, 1, "stress_equity")
        technology = pct_input("Technology shock", cfg["stress"]["custom_tech_shock"], -90, 50, 1, "stress_tech")
        fixed = pct_input("Fixed-income shock", cfg["stress"]["custom_fixed_income_shock"], -50, 50, 1, "stress_fixed")
        metadata = st.session_state.candidate_scored[["ticker", "sector", "asset_class"]] if not st.session_state.candidate_scored.empty else pd.DataFrame()
        shocked = hypothetical_stress(weights, metadata, equity, technology, fixed)
        st.metric("Immediate modeled portfolio shock", fmt_pct(shocked["loss_contribution"].sum()))
        st.dataframe(shocked.style.format({
            "weight": lambda value: fmt_pct(value, 2),
            "shock": lambda value: fmt_pct(value, 1),
            "loss_contribution": lambda value: fmt_pct(value, 2),
        }), hide_index=True, use_container_width=True)

with tabs[3]:
    st.subheader("Settings")
    st.caption("Score and portfolio percentage inputs display as percentages; internal calculations use decimal values. Historical simulations are reproducible from their configured seed.")
    c1, c2, c3 = st.columns(3)
    max_weight = pct_input("Maximum holding weight", cfg["optimizer"]["max_weight"], 5, 100, 1, "settings_max_weight")
    minimum_equity = pct_input("Minimum equity allocation", cfg["optimizer"]["minimum_equity_weight"], 0, 100, 1, "settings_equity_min")
    maximum_equity = pct_input("Maximum equity allocation", cfg["optimizer"]["maximum_equity_weight"], 0, 100, 1, "settings_equity_max")
    minimum_fixed = pct_input("Minimum fixed-income allocation", cfg["optimizer"]["minimum_fixed_income_weight"], 0, 100, 1, "settings_fixed_min")
    maximum_fixed = pct_input("Maximum fixed-income allocation", cfg["optimizer"]["maximum_fixed_income_weight"], 0, 100, 1, "settings_fixed_max")
    sector_tolerance = pct_input("Sector tolerance", cfg["optimizer"]["sector_tolerance"], 0, 100, 1, "settings_sector_tolerance")
    risk_free = pct_input("Risk-free rate (FCF yield scoring / Sharpe)", cfg["optimizer"]["risk_free_rate"], -5, 20, .1, "settings_risk_free")
    rf_text = st.text_input("Benchmark ticker", cfg["optimizer"]["benchmark"], key="settings_benchmark")
    liquidity = st.number_input("Minimum 30-day average dollar volume ($)", min_value=0, value=int(cfg["optimizer"]["liquidity_requirement"]), step=100000, key="settings_liquidity")
    return_shrinkage = pct_input(
        "Expected-return shrinkage toward benchmark / asset-class mean",
        cfg["optimizer"]["return_estimate_shrinkage"],
        0, 100, 1, "settings_return_shrinkage",
    )
    lookback = st.number_input("Historical lookback (years)", min_value=1, max_value=10, value=int(cfg["optimizer"]["lookback_years"]), key="settings_lookback")
    simulations = st.number_input("Feasible random portfolios", min_value=500, max_value=100000, value=int(cfg["optimizer"]["simulations"]), step=500, key="settings_simulations")
    expense_thresholds = cfg["scoring"]["equity_etf"]["expense_ratio_thresholds"]
    c1, c2, c3 = st.columns(3)
    expense_strong = pct_input("ETF expense ratio: strong anchor", expense_thresholds[0], 0, 5, .01, "settings_expense_strong")
    expense_neutral = pct_input("ETF expense ratio: neutral anchor", expense_thresholds[1], 0, 10, .05, "settings_expense_neutral")
    expense_weak = pct_input("ETF expense ratio: weak anchor", expense_thresholds[2], .01, 20, .1, "settings_expense_weak")
    if st.button("Apply settings", key="settings_apply"):
        if minimum_equity > maximum_equity or minimum_fixed > maximum_fixed:
            st.error("Each allocation minimum must be less than or equal to its maximum.")
        elif not rf_text.strip():
            st.error("Enter a benchmark ticker, or use SPY.")
        elif not expense_strong < expense_neutral < expense_weak:
            st.error("Expense ratio anchors must be strictly increasing.")
        else:
            cfg["optimizer"].update({
                "max_weight": max_weight,
                "minimum_equity_weight": minimum_equity,
                "maximum_equity_weight": maximum_equity,
                "minimum_fixed_income_weight": minimum_fixed,
                "maximum_fixed_income_weight": maximum_fixed,
                "sector_tolerance": sector_tolerance,
                "risk_free_rate": risk_free,
                "return_estimate_shrinkage": return_shrinkage,
                "benchmark": normalize_ticker(rf_text),
                "liquidity_requirement": int(liquidity),
                "lookback_years": int(lookback),
                "simulations": int(simulations),
            })
            cfg["scoring"]["equity_etf"]["expense_ratio_thresholds"] = [expense_strong, expense_neutral, expense_weak]
            st.session_state.cfg = cfg
            st.session_state.optimizer_result = None
            if st.session_state.single_result:
                metrics, _ = st.session_state.single_result
                st.session_state.single_result = (metrics, score_security(metrics, cfg))
            if not st.session_state.candidate_raw.empty:
                st.session_state.candidate_scored = pd.DataFrame(
                    [{**row.to_dict(), **score_security(row.to_dict(), cfg)} for _, row in st.session_state.candidate_raw.iterrows()]
                )
            st.success("Settings applied. Run the optimizer again to use the updated constraints.")
    st.button("Reset settings to defaults", key="settings_reset", on_click=reset_settings)

    with st.expander("Scoring formulas and model anchors"):
        st.markdown("""
        - **Stocks:** final score = 80% weighted-available fundamental score + 20% risk/resilience. Fundamentals use 40% EPS/FCF growth, 30% financial strength, and 30% valuation. EPS and FCF use 3-year CAGR with −10% = −100, +5% = 0, and +20% = +100. P/E uses 10x = +100, 25x = 0, and 50x = −100; non-positive P/E is N/A. Financial companies omit Net Debt/FCF and display a comparability warning.
        - **Equity ETFs:** 35% diversification (40% top-10 concentration, 35% largest-sector excess versus the matching S&P 500 sector, 25% holdings count), 30% cost/efficiency (expense ratio), 20% risk (55% drawdown, 45% volatility), and 15% liquidity. Sector excess scores 0 pp = +100, +10 pp = 0, and +25 pp = −100. Historical return is not scored.
        - **Fixed-income ETFs:** 40% Duration / Term-Fit Heuristic, 35% credit safety, 10% yield spread, and 15% liquidity. Yield uses reported YTM minus comparable-duration Treasury YTM; missing comparisons are N/A. Duration / Term-Fit Heuristic compares duration with the years remaining until 2033, not with the full liability schedule; it is a duration-based heuristic, not exact liability immunization, and ETF duration is not maturity.
        - **Optimizer returns:** arithmetic annual returns are shrunk toward SPY for equity assets or the asset-class mean otherwise. The shrinkage percentage is configurable. Historical and adjusted estimates are displayed separately.
        - **2033 Goal Check:** historical 21-trading-day block bootstrap, 100,000 paths, 12 sampled blocks per return year, fixed seed; when at least one year of daily returns is unavailable, an explicit multivariate-normal portfolio fallback is used and labeled.
        - **Missing data:** never zero-filled for scoring; available metrics are proportionally reweighted and confidence falls. Expense-ratio anchors are configurable defaults, not user-approved investment recommendations.
        - Stock scores use EPS and FCF 3-year CAGR as the growth inputs. All score outputs are clipped to -100…+100.
        """)

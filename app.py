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

from src.config import load_config
from src.data import (
    analyze_security,
    batch_analyze,
    download_price_period,
    normalize_ticker,
    price_frame,
)
from src.optimizer import (
    annualized_expected_returns,
    laura_value_2033,
    monte_carlo_optimize,
    prepare_returns,
    simulate_2033_distribution,
)
from src.scoring import score_security
from src.stress import HISTORICAL_WINDOWS, historical_portfolio_stress, hypothetical_stress

st.set_page_config(page_title="Laura Gao Quant System", layout="wide")
st.title("Laura Gao Quantitative Investment System")
st.caption("Four-tab security scoring, portfolio construction, stress testing, and decision-support app.")
st.info("Research support for the Wharton Global High School Investment Competition. Official competition materials and team instructions remain authoritative.")

if "cfg" not in st.session_state:
    st.session_state.cfg = load_config()
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
        "settings_risk_free", "settings_target_return", "settings_benchmark",
        "settings_liquidity", "settings_duration_target", "settings_lookback", "settings_simulations",
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


def fmt_pct(value, digits=1):
    try:
        value = float(value)
        return f"{value * 100:.{digits}f}%" if np.isfinite(value) else "N/A"
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


def show_security(metrics: dict, scored: dict):
    st.markdown(f"**{metrics.get('company', metrics.get('ticker'))}** · `{metrics.get('ticker', 'N/A')}`")
    st.caption(f"Detected scoring engine: **{scored['scoring_engine']}**. Scores from different asset engines are not directly comparable.")
    st.markdown(
        f"<div style='font-size:3.1rem;font-weight:750;line-height:1.2'>"
        f"Main Quantitative Score&nbsp; {fmt_score(scored['main_score'])}"
        f"<span style='font-size:1rem;font-weight:400'> / 100</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Score range: -100 to +100. Scores produced by different asset engines are not directly comparable.")
    st.markdown(
        f"**Sector:** {metrics.get('sector', 'N/A')} &nbsp;|&nbsp; "
        f"**Category:** {metrics.get('category', 'N/A')} &nbsp;|&nbsp; "
        f"**Asset type:** {metrics.get('asset_class', 'N/A')} &nbsp;|&nbsp; "
        f"**Data date:** {metrics.get('data_date', 'N/A')}"
    )
    labels = scored["category_labels"]
    columns = st.columns(5)
    columns[0].metric(labels["growth"], fmt_score(scored["growth_score"]))
    columns[1].metric(labels["quality"], fmt_score(scored["quality_score"]))
    columns[2].metric(labels["valuation"], fmt_score(scored["valuation_score"]))
    columns[3].metric("Risk & Resilience", fmt_score(scored["risk_score"]))
    columns[4].metric("Data Confidence", fmt_pct(scored["data_confidence"] / 100))

    details = []
    for row in scored.get("breakdown", []):
        raw = row["raw_value"]
        name = row["metric"].lower()
        if any(term in name for term in (
            "growth", "yield", "return", "drawdown", "volatility", "expense",
            "spread", "concentration", "sector weight", "margin", "earnings_consistency",
        )):
            raw_display = fmt_pct(raw)
        elif "net debt / fcf" in name:
            raw_display = fmt_num(raw)
        elif "dollar volume" in name or "free cash flow" in name:
            raw_display = fmt_money(raw)
        else:
            raw_display = fmt_num(raw)
        details.append({
            "Score group": row["group"],
            "Metric": row["metric"],
            "Observed value": raw_display,
            "Normalized points": fmt_score(row["score"]),
            "Configured weight (local branch)": fmt_pct(row["intended_weight"]),
            "Weighted points (local branch)": fmt_score(row["score"] * row["intended_weight"] if np.isfinite(row["score"]) else np.nan),
            "Scoring thresholds / anchors": row.get("thresholds", "N/A"),
            "Source / rule": row["source"],
        })
    if details:
        with st.expander("Metric-level inputs, thresholds and local weighted points", expanded=True):
            st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)

    if metrics.get("asset_class") == "stock":
        growth_rows = []
        for label, key in [
            ("Revenue 1Y", "revenue_growth_1y"), ("Revenue 3Y", "revenue_growth_3y"), ("Revenue 5Y", "revenue_growth_5y"),
            ("EPS 1Y", "eps_growth_1y"), ("EPS 3Y", "eps_growth_3y"), ("EPS 5Y", "eps_growth_5y"),
            ("FCF 1Y", "fcf_growth_1y"), ("FCF 3Y", "fcf_growth_3y"), ("FCF 5Y", "fcf_growth_5y"),
        ]:
            growth_rows.append({"Metric": label, "Observed": fmt_pct(metrics.get(key))})
        st.markdown("**Most important raw growth metrics**")
        st.dataframe(pd.DataFrame(growth_rows), hide_index=True, use_container_width=True)
    elif metrics.get("asset_class") == "fixed_income":
        fund_rows = [
            ("30-Day SEC Yield", fmt_pct(metrics.get("sec_yield_30d"))),
            ("Yield to Maturity", fmt_pct(metrics.get("yield_to_maturity"))),
            ("Effective Duration (years)", fmt_num(metrics.get("effective_duration"))),
            ("Weighted Average Maturity (years)", fmt_num(metrics.get("weighted_average_maturity"))),
            ("Expense Ratio", fmt_pct(metrics.get("expense_ratio"))),
            ("30-Day Average Dollar Volume", fmt_money(metrics.get("average_dollar_volume_30d"))),
            ("Treasury / Credit Exposure", str(metrics.get("treasury_credit_exposure", "N/A"))),
        ]
        st.dataframe(pd.DataFrame(fund_rows, columns=["Fixed-income metric", "Reported value"]), hide_index=True, use_container_width=True)
    elif metrics.get("asset_class") == "equity_etf":
        etf_rows = [
            ("Number of holdings", fmt_num(metrics.get("holdings_count"), 0)),
            ("Top-10 concentration", fmt_pct(metrics.get("top10_concentration"))),
            ("Largest sector weight", fmt_pct(metrics.get("largest_sector_weight"))),
            ("Expense ratio", fmt_pct(metrics.get("expense_ratio"))),
            ("30-Day Average Dollar Volume", fmt_money(metrics.get("average_dollar_volume_30d"))),
        ]
        st.dataframe(pd.DataFrame(etf_rows, columns=["Equity ETF metric", "Reported value"]), hide_index=True, use_container_width=True)
    if metrics.get("asset_class") == "fixed_income":
        st.caption(
            "SEC Yield and Yield to Maturity are distinct. Comparable-duration Treasury YTM, duration, or maturity not explicitly reported by the source remains N/A."
        )


def candidate_display(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "Ticker": row.get("ticker"),
            "Company": row.get("company"),
            "Main Quantitative Score": fmt_score(row.get("main_score")),
            "Engine": row.get("scoring_engine"),
            "Asset type": row.get("asset_class"),
            "Sector / category": row.get("sector") if row.get("sector") not in (None, "", "Unknown") else row.get("category"),
            "Return 1Y": fmt_pct(row.get("total_return_1y")),
            "Return 3Y": fmt_pct(row.get("total_return_3y")),
            "Return 5Y": fmt_pct(row.get("total_return_5y")),
            "Expense ratio": fmt_pct(row.get("expense_ratio")),
            "Trailing P/E": fmt_num(row.get("trailing_pe")),
            "FCF Yield": fmt_pct(row.get("fcf_yield")),
            "ROE": fmt_pct(row.get("return_on_equity")),
            "Duration (years)": fmt_num(row.get("effective_duration")),
            "YTM": fmt_pct(row.get("yield_to_maturity")),
            "30-Day Dollar Volume": fmt_money(row.get("average_dollar_volume_30d")),
            "Risk": fmt_score(row.get("risk_score")),
            "Data confidence": fmt_pct(row.get("data_confidence", np.nan) / 100),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_candidate_batch(tickers: tuple[str, ...], cache_dir: str, ttl_hours: float):
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
                    tuple(tickers), cfg["data"]["cache_dir"], float(cfg["data"]["cache_ttl_hours"])
                )
            st.session_state.candidate_raw = raw
            st.session_state.candidate_prices = price_frame(prices)
            st.session_state.candidate_errors = errors
            st.session_state.candidate_scored = pd.DataFrame(
                [score_row(row.to_dict()) for _, row in raw.iterrows()]
            ) if not raw.empty else pd.DataFrame()
            st.session_state.optimizer_result = None
    right.caption("Yahoo Finance data is cached locally for the configured cache period. Clicking load again refreshes Streamlit's candidate view.")

    if not st.session_state.candidate_scored.empty:
        st.markdown("### Candidate scores and major metrics")
        st.dataframe(candidate_display(st.session_state.candidate_scored), hide_index=True, use_container_width=True)
        st.markdown("### User-defined portfolio weights")
        editable = st.session_state.candidate_scored[["ticker", "asset_class", "sector", "main_score"]].copy()
        if "manual_weight_table" not in st.session_state:
            editable["weight"] = 100 / max(1, len(editable))
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
            column_config={"weight": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, format="%.1f")},
        )
        current_manual = edited[["ticker", "weight"]].copy()
        if previous_manual is not None and not current_manual.equals(previous_manual.reset_index(drop=True)):
            st.session_state.optimizer_result = None
        st.session_state.manual_weight_table = current_manual
        weight_total_percent = float(pd.to_numeric(edited["weight"], errors="coerce").fillna(0).sum())
        weight_total = weight_total_percent / 100
        st.caption(f"Current manual weights sum to {fmt_pct(weight_total)}; weights must sum to 100% to include this portfolio.")
        metadata = st.session_state.candidate_scored[
            ["ticker", "asset_class", "sector", "average_dollar_volume_30d"]
        ].copy()
        manual = pd.Series(
            pd.to_numeric(edited["weight"], errors="coerce").fillna(0).to_numpy() / 100,
            index=edited["ticker"],
        )
        if st.button("Run feasible portfolio optimizer", key="optimizer_run"):
            st.session_state.optimizer_result = None
            try:
                benchmark = str(cfg["optimizer"]["benchmark"]).strip().upper()
                benchmark_prices = pd.DataFrame()
                if benchmark:
                    benchmark_start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=int(cfg["optimizer"]["lookback_years"]))).strftime("%Y-%m-%d")
                    try:
                        benchmark_prices = download_price_period([benchmark], benchmark_start, date.today().isoformat())
                    except Exception as exc:
                        st.warning(f"Benchmark {benchmark} could not be loaded; candidate optimization will continue without it: {exc}")
                else:
                    st.warning("No benchmark ticker is configured; candidate optimization will continue without a benchmark comparison.")
                st.session_state.benchmark_prices = benchmark_prices
                optimizer_metadata = metadata.copy()
                result = monte_carlo_optimize(
                    st.session_state.candidate_prices,
                    cfg,
                    metadata=optimizer_metadata,
                    manual_weights=manual if np.isclose(weight_total, 1.0, atol=1e-6) else None,
                )
                st.session_state.optimizer_result = result
            except Exception as exc:
                st.error(f"Portfolio optimization could not run: {exc}")

    if st.session_state.candidate_errors:
        with st.expander("Ticker data issues"):
            st.json(st.session_state.candidate_errors)

    result = st.session_state.optimizer_result
    if result:
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
        st.markdown("### Feasible portfolios")
        st.caption(
            f"Long-only, max holding {cfg['optimizer']['max_weight']:.0%}; "
            f"equity allocation {cfg['optimizer']['minimum_equity_weight']:.0%}–{cfg['optimizer']['maximum_equity_weight']:.0%}; "
            f"fixed income {cfg['optimizer']['minimum_fixed_income_weight']:.0%}–{cfg['optimizer']['maximum_fixed_income_weight']:.0%}. "
            "Sector caps use each sector's share of the loaded candidate universe plus the configured tolerance."
        )
        caps = result["constraints"]["sector_caps"]
        if caps:
            with st.expander("Applied sector allocation caps"):
                cap_frame = pd.DataFrame([
                    {"Sector / category": name, "Maximum portfolio weight": fmt_pct(weight)}
                    for name, weight in caps.items()
                ])
                st.dataframe(cap_frame, hide_index=True, use_container_width=True)
        portfolio_summary = pd.DataFrame([
            {
                "Portfolio": name,
                "Expected annual return": fmt_pct(portfolio["expected_return"]),
                "Annualized volatility": fmt_pct(portfolio["volatility"]),
                "Sharpe ratio": fmt_num(portfolio["sharpe"]),
                "Estimated 2033 funding success": fmt_pct(portfolio.get("funding_success_probability_estimate", np.nan)),
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
        st.dataframe(weight_frame.style.format({"Weight": "{:.2%}"}), hide_index=True, use_container_width=True)
        if selected == "Laura Goal Portfolio":
            if portfolio["funding_objective_met"]:
                st.success(f"Sampled portfolio selection met the approximate 99.5% reserve-funding objective ({fmt_pct(portfolio['funding_success_probability_estimate'])}).")
            else:
                st.warning(f"No sampled feasible portfolio met the approximate 99.5% objective; the displayed portfolio has the highest estimated funding success ({fmt_pct(portfolio['funding_success_probability_estimate'])}).")
            st.caption(portfolio["construction"] + " Funding probability is an assumption-driven model estimate, not a guarantee.")
        if selected == "Target Return Portfolio" and not portfolio["target_met"]:
            st.warning(f"No sampled feasible portfolio reached the configured target return of {fmt_pct(portfolio['target_return'])}; showing the highest-return feasible sample.")

        frontier = result.get("efficient_frontier", pd.DataFrame())
        simulations = result.get("simulations", pd.DataFrame())
        if not simulations.empty:
            st.markdown(f"### Approximate efficient frontier — {len(simulations):,} feasible samples")
            chart_data = simulations.rename(columns={"volatility": "Annualized volatility", "expected_return": "Expected annual return"})
            fig = px.scatter(
                chart_data, x="Annualized volatility", y="Expected annual return",
                opacity=.35, title="Feasible random portfolios (sample, not exact optima)",
            )
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
        distribution = simulate_2033_distribution(
            portfolio["expected_return"], portfolio["volatility"],
            cfg["laura"]["contribution_2027"], cfg["laura"]["contribution_2028"],
            max(100000, int(cfg["laura"]["projection_simulations"])),
            int(cfg["laura"]["reserve_simulation_seed"]),
        )
        expected = laura_value_2033(
            portfolio["expected_return"],
            cfg["laura"]["contribution_2027"],
            cfg["laura"]["contribution_2028"],
        )
        st.metric("Illustrative Expected 2033 Value", fmt_money(expected))
        st.caption("Illustrative expected-value estimate under the selected portfolio's historical return/volatility inputs; it is not guaranteed.")
        pct_table = pd.DataFrame({
            "Percentile": ["5th", "25th", "50th", "75th", "95th"],
            "2033 value": [fmt_money(distribution.quantile(q)) for q in (.05, .25, .50, .75, .95)],
        })
        st.dataframe(pct_table, hide_index=True, use_container_width=True)
        funding_probability = float((distribution >= liability).mean())
        st.metric(f"Estimated probability of reaching the {fmt_money(liability)} nominal-liability benchmark by 2033", fmt_pct(funding_probability))
        st.caption("This small check compares the simulated 2033 asset value with the $500,000 zero-yield benchmark. It does not replace reserve analysis or imply a guaranteed yield.")

        benchmark_prices = st.session_state.get("benchmark_prices", pd.DataFrame())
        benchmark = str(cfg["optimizer"]["benchmark"]).upper()
        if not benchmark_prices.empty and benchmark in benchmark_prices.columns:
            b_returns = prepare_returns(benchmark_prices[[benchmark]], int(cfg["optimizer"]["lookback_years"]))
            if not b_returns.empty:
                benchmark_return = annualized_expected_returns(b_returns).iloc[0]
                benchmark_volatility = b_returns[benchmark].std(ddof=1) * np.sqrt(252)
                st.markdown("### Selected benchmark")
                st.dataframe(pd.DataFrame([
                    {"Series": selected, "Expected annual return": fmt_pct(portfolio["expected_return"]), "Annualized volatility": fmt_pct(portfolio["volatility"]), "Sharpe ratio": fmt_num(portfolio["sharpe"])},
                    {"Series": benchmark, "Expected annual return": fmt_pct(benchmark_return), "Annualized volatility": fmt_pct(benchmark_volatility), "Sharpe ratio": fmt_num((benchmark_return - cfg["optimizer"]["risk_free_rate"]) / benchmark_volatility if benchmark_volatility > 0 else np.nan)},
                ]), hide_index=True, use_container_width=True)
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
            st.dataframe(detail.style.format({"Holding return": "{:.2%}", "Portfolio contribution": "{:.2%}"}), hide_index=True, use_container_width=True)
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
        st.dataframe(shocked.style.format({"weight": "{:.2%}", "shock": "{:.1%}", "loss_contribution": "{:.2%}"}), hide_index=True, use_container_width=True)

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
    target_return = pct_input("Required / target annual return", cfg["optimizer"]["target_return"], -50, 100, .5, "settings_target_return")
    rf_text = st.text_input("Benchmark ticker", cfg["optimizer"]["benchmark"], key="settings_benchmark")
    liquidity = st.number_input("Minimum 30-day average dollar volume ($)", min_value=0, value=int(cfg["optimizer"]["liquidity_requirement"]), step=100000, key="settings_liquidity")
    duration_target = st.number_input("Fixed-income target duration (years)", min_value=0.0, max_value=30.0, value=float(cfg["scoring"]["fixed_income"]["target_duration"]), step=.25, key="settings_duration_target")
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
                "target_return": target_return,
                "benchmark": normalize_ticker(rf_text),
                "liquidity_requirement": int(liquidity),
                "lookback_years": int(lookback),
                "simulations": int(simulations),
            })
            cfg["scoring"]["equity_etf"]["expense_ratio_thresholds"] = [expense_strong, expense_neutral, expense_weak]
            cfg["scoring"]["fixed_income"]["target_duration"] = float(duration_target)
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
        - **Standard and financial stocks:** 80% weighted available fundamental groups (40% EPS/FCF growth, 30% financial strength, 30% valuation) plus 20% risk/resilience. Financial institutions do not receive industrial Net Debt/FCF scoring.
        - **Equity ETFs:** 35% historical return, 35% risk/resilience, 30% diversification/efficiency.
        - **Fixed-income ETFs:** 40% stability/risk, 30% return, 30% efficiency/liquidity. Duration fit uses the configurable 4.25-year target; credit and Treasury-relative YTM are scored only when data is present.
        - **Missing data:** never zero-filled for scoring; available metrics are proportionally reweighted and confidence falls. Expense-ratio anchors are configurable defaults, not user-approved investment recommendations.
        - Growth scores preserve the configured Revenue/EPS/FCF caps and 20% / 50% / 30% horizon weights. All score outputs are clipped to -100…+100.
        """)

from __future__ import annotations
import json
import os
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import streamlit as st

# Explicit CA bundle helps many macOS Python installations.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())

from src.config import load_config
from src.data import (
    analyze_security,
    batch_analyze,
    download_price_period,
    etf_holdings,
    load_local_universe,
    price_frame,
)
from src.scoring import score_security
from src.optimizer import (
    monte_carlo_optimize,
    laura_value_2033,
    deterministic_reserve_requirement,
)
from src.stress import HISTORICAL_WINDOWS, historical_portfolio_stress, hypothetical_stress

st.set_page_config(page_title="Laura Gao Quant System — Simplified", layout="wide")

if "cfg" not in st.session_state:
    st.session_state.cfg = load_config()
if "candidates" not in st.session_state:
    st.session_state.candidates = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "LLY", "XOM", "NEE"]
if "candidate_raw" not in st.session_state:
    st.session_state.candidate_raw = pd.DataFrame()
if "candidate_prices" not in st.session_state:
    st.session_state.candidate_prices = pd.DataFrame()
if "candidate_errors" not in st.session_state:
    st.session_state.candidate_errors = {}
if "optimizer_result" not in st.session_state:
    st.session_state.optimizer_result = None
if "selected_portfolio" not in st.session_state:
    st.session_state.selected_portfolio = "Maximum Sharpe"

cfg = st.session_state.cfg


def fmt_num(x, digits=2):
    try:
        x = float(x)
        return f"{x:,.{digits}f}" if np.isfinite(x) else "Unavailable"
    except Exception:
        return "Unavailable"


def fmt_pct(x, digits=1):
    try:
        x = float(x)
        return f"{x*100:.{digits}f}%" if np.isfinite(x) else "Unavailable"
    except Exception:
        return "Unavailable"


def fmt_money(x):
    try:
        x = float(x)
        return f"${x:,.0f}" if np.isfinite(x) else "Unavailable"
    except Exception:
        return "Unavailable"


PERCENT_METRIC_KEYS = {
    "revenue_growth_1y", "revenue_growth_3y", "revenue_growth_5y",
    "eps_growth_1y", "eps_growth_3y", "eps_growth_5y",
    "fcf_growth_1y", "fcf_growth_3y", "fcf_growth_5y",
    "net_income_growth_1y", "net_income_growth_3y", "net_income_growth_5y",
    "operating_margin", "net_margin", "return_on_equity", "return_on_assets",
    "dividend_yield", "distribution_yield", "annualized_volatility",
    "max_drawdown", "downside_deviation", "expense_ratio",
    "total_return_1y", "total_return_3y", "total_return_5y",
}


def format_metric_value(key, value):
    if key in PERCENT_METRIC_KEYS:
        return fmt_pct(value)
    if key in {"current_price"}:
        return fmt_money(value)
    return fmt_num(value)


def pct_input(label, decimal_value, min_percent=-100.0, max_percent=100.0,
              step_percent=1.0, key=None, format="%.1f") -> float:
    """Display a percentage input while returning its decimal form."""
    entered = st.number_input(
        label,
        min_value=float(min_percent),
        max_value=float(max_percent),
        value=float(decimal_value) * 100.0,
        step=float(step_percent),
        format=format,
        key=key,
    )
    return float(entered) / 100.0


def format_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column in {"Weight", "weight"}:
            out[column] = out[column].map(fmt_pct)
        elif column in {"Data Confidence", "data_confidence", "Sector Percentile", "sector_percentile"}:
            out[column] = out[column].map(lambda x: fmt_pct(float(x) / 100.0))
    return out


def score_row(metrics: dict) -> dict:
    s = score_security(metrics, cfg)
    return {
        **metrics,
        "growth_score": s["growth_score"],
        "quality_score": s["quality_score"],
        "valuation_score": s["valuation_score"],
        "fundamental_score": s["fundamental_score"],
        "display_rating": s["display_rating"],
        "risk_score": s["risk_score"],
        "data_confidence": s["data_confidence"],
        "analysis_profile": s["analysis_profile"],
    }


def current_weights() -> pd.Series | None:
    result = st.session_state.optimizer_result
    if not result:
        return None
    name = st.session_state.selected_portfolio
    p = result["portfolios"].get(name)
    return p["weights"] if p else None


def ensure_laura_portfolio(result: dict | None) -> dict | None:
    """Add Laura Portfolio to results retained from an older Streamlit session."""
    if not result or "Laura Portfolio" in result.get("portfolios", {}):
        return result
    portfolios = result.get("portfolios", {})
    max_sharpe = portfolios.get("Maximum Sharpe")
    min_vol = portfolios.get("Minimum Volatility")
    if not max_sharpe or not min_vol:
        return result
    weights = (max_sharpe["weights"] * 0.50) + (min_vol["weights"] * 0.50)
    mu = result["expected_returns"].reindex(weights.index).to_numpy()
    cov = result["covariance"].loc[weights.index, weights.index].to_numpy()
    expected_return = float(weights.to_numpy() @ mu)
    volatility = float(np.sqrt(max(weights.to_numpy() @ cov @ weights.to_numpy(), 0.0)))
    rf = float(cfg["optimizer"]["risk_free_rate"])
    result["portfolios"]["Laura Portfolio"] = {
        "weights": weights,
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe": (expected_return - rf) / volatility if volatility > 0 else np.nan,
        "construction": "50% Maximum Sharpe + 50% Minimum Volatility",
    }
    return result


st.title("Laura Gao Quantitative Investment System — Simplified")
st.caption(
    "Eight-tab version: profile-specific stock, company, and ETF analysis, sector ranking, portfolio construction, optimization, stress testing, and settings."
)
st.info(
    "Metric/category scores use **-100 to +100**. A secondary company rating maps the fundamental score to **0–100**. "
    "Normal stocks, special financial companies, and ETFs use separate analysis profiles so inapplicable metrics are not treated as company failures."
)

tabs = st.tabs([
    "1 Stock Analyzer",
    "2 Special Companies",
    "3 ETF Analyzer",
    "4 Sector Rankings",
    "5 Portfolio Builder",
    "6 Portfolio Optimizer",
    "7 Stress Test",
    "8 Settings",
])

# ------------------------- TAB 1 -------------------------
with tabs[0]:
    st.subheader("Stock Analyzer")
    ticker = st.text_input("Ticker", "AAPL", key="t1_ticker").strip().upper()
    if st.button("Analyze", key="t1_analyze"):
        try:
            with st.spinner(f"Downloading {ticker} data..."):
                metrics, _ = analyze_security(ticker, cfg)
                if metrics.get("analysis_profile") != "operating_company":
                    st.error("This ticker is not a normal operating stock. Use Special Companies or ETF Analyzer.")
                else:
                    scored = score_security(metrics, cfg)
                    st.session_state.single_result = (metrics, scored)
        except Exception as e:
            st.error(f"Could not analyze {ticker}: {e}")

    if "single_result" in st.session_state:
        metrics, scored = st.session_state.single_result
        profile = metrics.get("analysis_profile", scored.get("analysis_profile", "operating_company"))
        a, b, c, d = st.columns(4)
        a.metric("Fundamental score", f"{scored['fundamental_score']:+.1f} / 100" if np.isfinite(scored["fundamental_score"]) else "Unavailable")
        b.metric("Mapped rating", f"{scored['display_rating']:.1f} / 100" if np.isfinite(scored["display_rating"]) else "Unavailable")
        c.metric("Risk & resilience", f"{scored['risk_score']:+.1f} / 100" if np.isfinite(scored["risk_score"]) else "Unavailable")
        d.metric("Data confidence", f"{scored['data_confidence']:.0f}%")

        st.write(
            f"**{metrics['company']}** · {metrics['sector']} · {metrics['industry']} · "
            f"`{profile}`"
        )
        cats = pd.DataFrame([
            ["Growth", scored["growth_score"]],
            ["Quality / Financial Strength", scored["quality_score"]],
            ["Valuation", scored["valuation_score"]],
        ], columns=["Category", "Score (-100 to +100)"])
        st.dataframe(cats, hide_index=True, use_container_width=True)

        growth_table = pd.DataFrame([
            ["Revenue 1Y", fmt_pct(metrics.get("revenue_growth_1y")), scored["metric_scores"].get("revenue_growth_1y")],
            ["Revenue 3Y CAGR", fmt_pct(metrics.get("revenue_growth_3y")), scored["metric_scores"].get("revenue_growth_3y")],
            ["Revenue 5Y CAGR", fmt_pct(metrics.get("revenue_growth_5y")), scored["metric_scores"].get("revenue_growth_5y")],
            ["EPS 1Y", fmt_pct(metrics.get("eps_growth_1y")), scored["metric_scores"].get("eps_growth_1y")],
            ["EPS 3Y CAGR", fmt_pct(metrics.get("eps_growth_3y")), scored["metric_scores"].get("eps_growth_3y")],
            ["EPS 5Y CAGR", fmt_pct(metrics.get("eps_growth_5y")), scored["metric_scores"].get("eps_growth_5y")],
            ["FCF 1Y", fmt_pct(metrics.get("fcf_growth_1y")), scored["metric_scores"].get("fcf_growth_1y")],
            ["FCF 3Y CAGR", fmt_pct(metrics.get("fcf_growth_3y")), scored["metric_scores"].get("fcf_growth_3y")],
            ["FCF 5Y CAGR", fmt_pct(metrics.get("fcf_growth_5y")), scored["metric_scores"].get("fcf_growth_5y")],
        ], columns=["Growth metric", "Raw value", "Normalized score"])
        st.markdown("**Growth calculation**")
        st.dataframe(growth_table, hide_index=True, use_container_width=True)

        with st.expander("All raw metrics"):
            raw = pd.DataFrame({
                "Metric": list(metrics.keys()),
                "Raw value": [format_metric_value(k, v) for k, v in metrics.items()],
            })
            st.dataframe(raw, hide_index=True, use_container_width=True)
        with st.expander("All normalized metric scores"):
            norm = pd.DataFrame({"Metric": list(scored["metric_scores"].keys()), "Score": list(scored["metric_scores"].values())})
            st.dataframe(norm, hide_index=True, use_container_width=True)

# ------------------------- TAB 2 -------------------------
with tabs[1]:
    st.subheader("Special Company Analyzer")
    st.caption("For financial companies and conglomerates whose economics are not comparable to ordinary operating stocks.")
    ticker = st.text_input("Company ticker", "BRK-B", key="special_ticker").strip().upper()
    if st.button("Analyze special company", key="special_analyze"):
        try:
            with st.spinner(f"Downloading {ticker} data..."):
                metrics, _ = analyze_security(ticker, cfg, force_refresh=True)
                if metrics.get("analysis_profile") not in {"financial_company", "financial_conglomerate"}:
                    st.error("This ticker is classified as a normal operating stock. Use Stock Analyzer instead.")
                else:
                    st.session_state.special_result = (metrics, score_security(metrics, cfg))
        except Exception as e:
            st.error(f"Could not analyze {ticker}: {e}")
    if "special_result" in st.session_state:
        metrics, scored = st.session_state.special_result
        st.success(f"Profile: {metrics.get('analysis_profile', 'financial_company')}")
        st.dataframe(pd.DataFrame([
            ["Growth", scored["growth_score"]],
            ["Quality / Financial Strength", scored["quality_score"]],
            ["Valuation", scored["valuation_score"]],
            ["Risk & resilience", scored["risk_score"]],
            ["Mapped rating", scored["display_rating"]],
        ], columns=["Category", "Score"]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame({
            "Metric": list(metrics),
            "Value": [format_metric_value(k, v) for k, v in metrics.items()],
        }), hide_index=True, use_container_width=True)

# ------------------------- TAB 3 -------------------------
with tabs[2]:
    st.subheader("ETF Analyzer")
    st.caption("ETF scoring uses fund-level data. The holdings table shows the assets Yahoo Finance publishes for the selected ETF.")
    ticker = st.text_input("ETF ticker", "SPY", key="etf_ticker").strip().upper()
    if st.button("Analyze ETF", key="etf_analyze"):
        try:
            with st.spinner(f"Downloading {ticker} ETF data..."):
                metrics, _ = analyze_security(ticker, cfg, force_refresh=True)
                if metrics.get("analysis_profile") not in {"equity_etf", "fixed_income_etf"}:
                    st.error("This ticker is not classified as an ETF.")
                else:
                    holdings = etf_holdings(ticker, cfg, force_refresh=True)
                    st.session_state.etf_result = (metrics, score_security(metrics, cfg), holdings)
        except Exception as e:
            st.error(f"Could not analyze ETF {ticker}: {e}")
    if "etf_result" in st.session_state:
        metrics, scored, holdings = st.session_state.etf_result
        st.success(f"Profile: {metrics.get('analysis_profile')}")
        st.dataframe(pd.DataFrame([
            ["Growth / distribution", scored["growth_score"]],
            ["Fund quality", scored["quality_score"]],
            ["Portfolio valuation", scored["valuation_score"]],
            ["Risk", scored["risk_score"]],
            ["Mapped rating", scored["display_rating"]],
        ], columns=["Category", "Score"]), hide_index=True, use_container_width=True)
        if holdings.empty:
            st.warning("Yahoo Finance did not provide a holdings list for this ETF.")
        else:
            st.markdown(f"**Assets linked to {metrics['ticker']} ({len(holdings)} holdings reported)**")
            st.dataframe(holdings.style.format({"weight": "{:.2%}"}), hide_index=True, use_container_width=True)
            st.download_button("Download ETF holdings CSV", holdings.to_csv(index=False).encode("utf-8"),
                               file_name=f"{metrics['ticker']}_holdings.csv", mime="text/csv", key="etf_holdings_download")
        st.dataframe(pd.DataFrame({
            "Metric": list(metrics),
            "Value": [format_metric_value(k, v) for k, v in metrics.items()],
        }), hide_index=True, use_container_width=True)

# ------------------------- TAB 4 -------------------------
with tabs[3]:
    st.subheader("Sector Rankings")
    st.success("SSL fix: the default universe is a bundled local CSV. Loading the universe itself makes no internet/SSL request.")
    source = st.radio(
        "Universe source",
        ["Bundled US large-cap universe", "Upload WInS/other universe CSV"],
        horizontal=True,
        key="t2_source",
    )

    universe = None
    if source == "Bundled US large-cap universe":
        try:
            universe = load_local_universe()
            st.caption(f"Bundled universe: {len(universe)} stocks across {universe['sector'].nunique()} sectors. It is a screening helper, not a claim that every ticker is WInS-eligible.")
        except Exception as e:
            st.error(f"Could not open the local universe file: {e}")
    else:
        upload = st.file_uploader("Upload CSV with ticker and preferably sector columns", type=["csv"], key="t2_upload")
        if upload is not None:
            universe = pd.read_csv(upload)
            universe.columns = [str(c).strip().lower() for c in universe.columns]
            if "ticker" not in universe.columns:
                st.error("The uploaded CSV must contain a ticker column.")
                universe = None
            if universe is not None and "sector" not in universe.columns:
                universe["sector"] = "Uploaded universe"

    if universe is not None and not universe.empty:
        sectors = sorted(universe["sector"].dropna().astype(str).unique())
        sector = st.selectbox("Sector", sectors, key="t2_sector")
        sector_df = universe[universe["sector"].astype(str) == sector].copy()
        st.write(f"{len(sector_df)} names available in this sector.")
        analyze_count = st.slider(
            "Number of names to analyze",
            min_value=2,
            max_value=max(2, len(sector_df)),
            value=min(10, len(sector_df)) if len(sector_df) >= 2 else 2,
            key="t2_count",
        ) if len(sector_df) >= 2 else 1

        if st.button("Run sector ranking", key="t2_run"):
            tickers = sector_df["ticker"].astype(str).head(int(analyze_count)).tolist()
            bar = st.progress(0)
            rows, errors = [], {}
            for i, t in enumerate(tickers, start=1):
                raw, _, err = batch_analyze([t], cfg)
                if not raw.empty:
                    metrics = raw.iloc[0].to_dict()
                    # Use the local/uploaded sector label for consistent grouping.
                    metrics["sector"] = sector
                    rows.append(score_row(metrics))
                errors.update(err)
                bar.progress(i / len(tickers))
            ranked = pd.DataFrame(rows)
            if not ranked.empty:
                ranked = ranked.sort_values("fundamental_score", ascending=False, na_position="last").reset_index(drop=True)
                ranked["sector_rank"] = np.arange(1, len(ranked) + 1)
                n = len(ranked)
                ranked["sector_percentile"] = 100.0 if n == 1 else 100.0 * (n - ranked["sector_rank"]) / (n - 1)
            st.session_state.sector_ranked = ranked
            st.session_state.sector_errors = errors

        ranked = st.session_state.get("sector_ranked")
        if isinstance(ranked, pd.DataFrame) and not ranked.empty:
            show = ranked[[
                "sector_rank", "ticker", "company", "fundamental_score", "display_rating",
                "growth_score", "quality_score", "valuation_score", "risk_score",
                "data_confidence", "sector_percentile"
            ]].copy()
            show = show.rename(columns={
                "data_confidence": "Data Confidence",
                "sector_percentile": "Sector Percentile",
            })
            st.dataframe(format_metric_frame(show), hide_index=True, use_container_width=True)
            st.download_button(
                "Download ranking CSV",
                show.to_csv(index=False).encode("utf-8"),
                file_name="sector_ranking.csv",
                mime="text/csv",
                key="t2_download",
            )
        if st.session_state.get("sector_errors"):
            with st.expander("Tickers that could not be analyzed"):
                st.json(st.session_state.sector_errors)

# ------------------------- TAB 5 -------------------------
with tabs[4]:
    st.subheader("Portfolio Builder")
    text = st.text_area(
        "Candidate tickers (comma or new-line separated)",
        value=", ".join(st.session_state.candidates),
        height=110,
        key="t3_tickers",
    )
    if st.button("Save candidate list", key="t3_save"):
        tickers = [x.strip().upper() for x in text.replace("\n", ",").split(",") if x.strip()]
        st.session_state.candidates = list(dict.fromkeys(tickers))
        st.success(f"Saved {len(st.session_state.candidates)} candidates.")

    if st.button("Download candidate data", key="t3_download_data"):
        try:
            with st.spinner("Downloading candidate fundamentals and prices..."):
                raw, pmap, errors = batch_analyze(st.session_state.candidates, cfg)
                if not raw.empty:
                    scored = pd.DataFrame([score_row(r.to_dict()) for _, r in raw.iterrows()])
                else:
                    scored = pd.DataFrame()
                st.session_state.candidate_raw = scored
                st.session_state.candidate_prices = price_frame(pmap)
                st.session_state.candidate_errors = errors
                st.session_state.optimizer_result = None
        except Exception as e:
            st.error(str(e))

    if not st.session_state.candidate_raw.empty:
        cols = ["ticker", "company", "sector", "asset_class", "fundamental_score", "display_rating", "risk_score", "data_confidence"]
        candidates = st.session_state.candidate_raw[cols].rename(columns={
            "data_confidence": "Data Confidence",
        })
        st.dataframe(format_metric_frame(candidates), hide_index=True, use_container_width=True)
        st.write(f"Usable price series: {st.session_state.candidate_prices.shape[1]}")
    if st.session_state.candidate_errors:
        with st.expander("Candidate download errors"):
            st.json(st.session_state.candidate_errors)

# ------------------------- TAB 6 -------------------------
with tabs[5]:
    st.subheader("Portfolio Optimizer")
    st.caption("Simplified mechanics: random feasible long-only portfolios are generated under the max-holding limit. This avoids the line-search optimizer errors from the previous version.")
    if st.session_state.candidate_prices.empty:
        st.warning("Go to Tab 5 and download candidate data first.")
    else:
        usable = st.session_state.candidate_prices.dropna(how="all", axis=1)
        cap = float(cfg["optimizer"]["max_weight"])
        need = int(np.ceil(1.0 / cap))
        st.write(f"Current max holding: **{cap:.0%}**. At least **{need} usable holdings** are required for a fully invested portfolio.")
        if st.button("Run Monte Carlo optimizer", key="t4_optimize"):
            try:
                with st.spinner("Testing feasible portfolios..."):
                    st.session_state.optimizer_result = monte_carlo_optimize(usable, cfg)
            except Exception as e:
                st.error(str(e))

        result = ensure_laura_portfolio(st.session_state.optimizer_result)
        st.session_state.optimizer_result = result
        if result:
            names = list(result["portfolios"].keys())
            selected = st.selectbox("Portfolio to use", names, index=names.index(st.session_state.selected_portfolio) if st.session_state.selected_portfolio in names else 0, key="t4_select")
            st.session_state.selected_portfolio = selected

            summary_rows = []
            for name, p in result["portfolios"].items():
                summary_rows.append({
                    "Portfolio": name,
                    "Expected annual return": p["expected_return"],
                    "Annual volatility": p["volatility"],
                    "Sharpe": p["sharpe"],
                })
            summary = pd.DataFrame(summary_rows)
            st.dataframe(
                summary.style.format({"Expected annual return": "{:.2%}", "Annual volatility": "{:.2%}", "Sharpe": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
            )

            p = result["portfolios"][selected]
            weights = p["weights"].sort_values(ascending=False)
            weights_df = pd.DataFrame({"Ticker": weights.index, "Weight": weights.values})
            st.markdown(f"**{selected} weights**")
            st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}), hide_index=True, use_container_width=True)
            st.bar_chart(weights_df.set_index("Ticker"))

            st.markdown("**Correlation matrix**")
            st.dataframe(result["correlation"].round(2), use_container_width=True)

            # Simple Laura-specific lens kept inside the optimizer instead of separate tabs.
            v2033 = laura_value_2033(
                p["expected_return"],
                cfg["laura"]["contribution_2027"],
                cfg["laura"]["contribution_2028"],
            )
            reserve = deterministic_reserve_requirement(
                cfg["laura"]["annual_payment"],
                cfg["laura"]["payment_count"],
                cfg["laura"]["reserve_yield"],
            )
            st.markdown("**Laura 2033 deterministic lens**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Projected 2033 value", fmt_money(v2033))
            c2.metric("PV of 10 × $50k reserve", fmt_money(reserve))
            c3.metric("Capital above reserve", fmt_money(v2033 - reserve))
            if selected == "Laura Portfolio":
                st.caption("Laura Portfolio construction: 50% Maximum Sharpe + 50% Minimum Volatility. This is a model assumption, not a funding-confidence guarantee.")
            st.caption("This is an assumption-driven expected-value illustration using the portfolio's historical-return estimate. It is not a funding-probability calculation or a guarantee.")

# ------------------------- TAB 7 -------------------------
with tabs[6]:
    st.subheader("Stress Test")
    result = st.session_state.optimizer_result
    if not result:
        st.warning("Run Tab 4 first so the stress test has portfolio weights.")
    else:
        names = list(result["portfolios"].keys())
        portfolio_name = st.selectbox("Portfolio", names, index=names.index(st.session_state.selected_portfolio) if st.session_state.selected_portfolio in names else 0, key="t9_portfolio")
        weights = result["portfolios"][portfolio_name]["weights"]
        expected_return = result["portfolios"][portfolio_name]["expected_return"]

        st.markdown("### Historical window")
        event = st.selectbox("Historical event", list(HISTORICAL_WINDOWS.keys()), key="t9_event")
        start, end = HISTORICAL_WINDOWS[event]
        if st.button("Run historical stress", key="t9_hist"):
            try:
                with st.spinner("Downloading historical prices for the selected period..."):
                    prices = download_price_period(list(weights.index), start, end)
                    st.session_state.hist_stress = historical_portfolio_stress(prices, weights)
            except Exception as e:
                st.error(str(e))

        hist = st.session_state.get("hist_stress")
        if hist:
            if not hist.get("available"):
                st.warning(hist.get("reason", "Historical test unavailable."))
            else:
                h1, h2, h3 = st.columns(3)
                h1.metric("Portfolio return", fmt_pct(hist["portfolio_return"]))
                h2.metric("Max drawdown", fmt_pct(hist["max_drawdown"]))
                h3.metric("Weight coverage", fmt_pct(hist["weight_coverage"]))
                contrib = pd.DataFrame({
                    "Ticker": hist["contributions"].index,
                    "Holding return": hist["holding_returns"].reindex(hist["contributions"].index).values,
                    "Contribution to portfolio return": hist["contributions"].values,
                })
                st.dataframe(contrib.style.format({"Holding return":"{:.2%}", "Contribution to portfolio return":"{:.2%}"}), hide_index=True, use_container_width=True)
                if hist["weight_coverage"] < 0.999:
                    st.caption("Some holdings did not exist or lacked data during this event. The available weights were renormalized, and weight coverage is shown explicitly.")

        st.markdown("### Hypothetical shock")
        c1, c2, c3 = st.columns(3)
        equity_shock = pct_input("Other equities shock", cfg["stress"]["custom_equity_shock"], -90, 50, 5, "t9_equity")
        tech_shock = pct_input("Technology shock", cfg["stress"]["custom_tech_shock"], -90, 50, 5, "t9_tech")
        fixed_shock = pct_input("Fixed-income shock", cfg["stress"]["custom_fixed_income_shock"], -50, 50, 1, "t9_fixed")

        metadata = st.session_state.candidate_raw
        shocked = hypothetical_stress(weights, metadata, equity_shock, tech_shock, fixed_shock)
        port_shock = float(shocked["loss_contribution"].sum())
        st.metric("Immediate portfolio shock", fmt_pct(port_shock))
        st.dataframe(shocked.style.format({"weight":"{:.2%}", "shock":"{:.1%}", "loss_contribution":"{:.2%}"}), hide_index=True, use_container_width=True)

        baseline_2033 = laura_value_2033(expected_return, cfg["laura"]["contribution_2027"], cfg["laura"]["contribution_2028"])
        pre2033_after_shock = baseline_2033 * (1.0 + port_shock)
        reserve = deterministic_reserve_requirement(cfg["laura"]["annual_payment"], cfg["laura"]["payment_count"], cfg["laura"]["reserve_yield"])
        st.markdown("### Laura funding lens")
        s1, s2, s3 = st.columns(3)
        s1.metric("Baseline projected 2033 value", fmt_money(baseline_2033))
        s2.metric("If the shock occurred just before 2033", fmt_money(pre2033_after_shock))
        s3.metric("Capital above deterministic reserve", fmt_money(pre2033_after_shock - reserve))
        st.caption("The reserve figure is the present value of ten beginning-of-year $50,000 payments using the reserve-yield assumption in Settings. This simplified version does not claim a Monte Carlo funding probability.")

# ------------------------- TAB 8 -------------------------
with tabs[7]:
    st.subheader("Settings")
    st.caption("All scoring mechanics remain transparent. Apply changes to the current session; Reset returns to config.yaml defaults.")

    st.markdown("### Fundamental category weights")
    g1, g2, g3 = st.columns(3)
    growth_w = pct_input("Growth weight", cfg["scoring"]["category_weights"]["growth"], 0, 100, 5, "t10_growth_w")
    quality_w = pct_input("Quality weight", cfg["scoring"]["category_weights"]["quality"], 0, 100, 5, "t10_quality_w")
    valuation_w = pct_input("Valuation weight", cfg["scoring"]["category_weights"]["valuation"], 0, 100, 5, "t10_valuation_w")

    st.markdown("### Growth normalization caps")
    st.caption("Each cap maps +cap to +100, 0 to 0, and -cap to -100. Values beyond the cap stay at ±100.")
    caps = cfg["scoring"]["growth_caps"].copy()
    cap_inputs = {}
    rows = [
        ("Revenue 1Y", "revenue_growth_1y"), ("Revenue 3Y CAGR", "revenue_growth_3y"), ("Revenue 5Y CAGR", "revenue_growth_5y"),
        ("EPS 1Y", "eps_growth_1y"), ("EPS 3Y CAGR", "eps_growth_3y"), ("EPS 5Y CAGR", "eps_growth_5y"),
        ("FCF 1Y", "fcf_growth_1y"), ("FCF 3Y CAGR", "fcf_growth_3y"), ("FCF 5Y CAGR", "fcf_growth_5y"),
    ]
    for i in range(0, len(rows), 3):
        cols = st.columns(3)
        for j, (label, key) in enumerate(rows[i:i+3]):
            cap_inputs[key] = pct_input(
                label, caps[key], 5, 150, 5, f"t10_cap_{key}"
            )

    st.markdown("### Portfolio and Laura assumptions")
    o1, o2, o3, o4 = st.columns(4)
    max_weight = pct_input("Max holding weight", cfg["optimizer"]["max_weight"], 5, 100, 5, "t10_max_weight")
    simulations = o2.number_input("Random portfolios", min_value=500, max_value=100000, value=int(cfg["optimizer"]["simulations"]), step=500, key="t10_sims")
    risk_free = pct_input("Risk-free rate", cfg["optimizer"]["risk_free_rate"], -5, 20, 0.5, "t10_rf")
    lookback = o4.number_input("Price lookback (years)", min_value=1, max_value=10, value=int(cfg["optimizer"]["lookback_years"]), step=1, key="t10_lookback")
    reserve_yield = pct_input("2033 reserve yield assumption", cfg["laura"]["reserve_yield"], 0, 15, 0.5, "t10_reserve_yield")

    c_apply, c_reset = st.columns(2)
    if c_apply.button("Apply settings", key="t10_apply"):
        if growth_w + quality_w + valuation_w <= 0:
            st.error("At least one category weight must be positive.")
        else:
            cfg["scoring"]["category_weights"] = {"growth": growth_w, "quality": quality_w, "valuation": valuation_w}
            cfg["scoring"]["growth_caps"].update(cap_inputs)
            cfg["optimizer"]["max_weight"] = max_weight
            cfg["optimizer"]["simulations"] = int(simulations)
            cfg["optimizer"]["risk_free_rate"] = risk_free
            cfg["optimizer"]["lookback_years"] = int(lookback)
            cfg["laura"]["reserve_yield"] = reserve_yield
            st.session_state.cfg = cfg
            st.session_state.optimizer_result = None
            st.success("Settings applied. Re-run analyses that depend on them.")

    if c_reset.button("Reset to defaults", key="t10_reset"):
        st.session_state.cfg = load_config()
        st.session_state.optimizer_result = None
        st.success("Defaults restored. Refresh/rerun the relevant tab.")

    st.download_button(
        "Download current settings JSON",
        json.dumps(st.session_state.cfg, indent=2).encode("utf-8"),
        file_name="laura_gao_settings.json",
        mime="application/json",
        key="t10_download_cfg",
    )

    st.markdown("### Core formulas")
    st.markdown(
        """
- **Growth metric normalization:** `score = clip(growth / cap, -1, +1) × 100`.
- **Growth horizons:** 20% 1Y + 50% 3Y CAGR + 30% 5Y CAGR, reweighted over available data.
- **Growth branches:** 35% revenue + 35% EPS + 30% FCF, reweighted over available data.
- **Fundamental score:** 30% Growth + 40% Quality + 30% Valuation by default.
- **Mapped rating:** `(fundamental score + 100) / 2`.
- **Portfolio return:** `wᵀμ`; **portfolio volatility:** `sqrt(wᵀΣw)`.
- **2033 deterministic projection:** `$300k(1+r)^6 + $150k(1+r)^5`.
- **Operating reserve PV:** `Σ 50,000/(1+y)^t`, for `t = 0...9` because the first payment occurs at the beginning of 2033.
        """
    )

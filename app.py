from __future__ import annotations

import json
import os
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
import streamlit as st

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
    normalize_ticker,
    price_frame,
)
try:
    from src.data import reserve_asset_profile
except ImportError:
    def reserve_asset_profile(ticker: str) -> str:
        """Compatibility fallback for a stale Streamlit data module."""
        symbol = normalize_ticker(ticker)
        if symbol in {"BIL", "SGOV", "SHY", "IEF", "TLT", "GOVT", "BND", "AGG"}:
            return "fixed_income_etf"
        if symbol in {"VOO", "QQQ", "VTI", "SPY"}:
            return "equity_etf"
        return "unknown"
from src.optimizer import laura_value_2033, monte_carlo_optimize, simulate_2033_distribution
from src.reserve import (
    LIABILITY,
    PAYMENT,
    PAYMENT_COUNT,
    deterministic_reserve_pv,
    normalize_reserve_weights,
    required_reserves,
    weighted_reserve_yield,
)
from src.scoring import score_security
from src.stress import HISTORICAL_WINDOWS, historical_portfolio_stress, hypothetical_stress

st.set_page_config(page_title="Laura Gao Quant System", layout="wide")

if "cfg" not in st.session_state:
    st.session_state.cfg = load_config()
cfg = st.session_state.cfg
for key, default in {
    "candidates": ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "LLY", "XOM", "NEE"],
    "candidate_raw": pd.DataFrame(),
    "candidate_prices": pd.DataFrame(),
    "candidate_errors": {},
    "optimizer_result": None,
    "selected_portfolio": "Maximum Sharpe",
    "single_result": None,
    "sector_ranked": pd.DataFrame(),
    "sector_errors": {},
    "reserve_optimizer_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
if "reserve_assets" not in st.session_state:
    st.session_state.reserve_assets = pd.DataFrame([
        {"asset": "SGOV", "asset_type": "Fixed-income / Treasury ETF", "weight": .40, "sec_yield_30d": np.nan, "yield_to_maturity": np.nan, "effective_duration": np.nan, "weighted_average_maturity": np.nan, "expense_ratio": np.nan, "volatility": np.nan, "drawdown": np.nan, "dollar_volume_30d": np.nan, "treasury_credit_exposure": "", "data_date": ""},
        {"asset": "BIL", "asset_type": "Fixed-income / Treasury ETF", "weight": .30, "sec_yield_30d": np.nan, "yield_to_maturity": np.nan, "effective_duration": np.nan, "weighted_average_maturity": np.nan, "expense_ratio": np.nan, "volatility": np.nan, "drawdown": np.nan, "dollar_volume_30d": np.nan, "treasury_credit_exposure": "", "data_date": ""},
        {"asset": "SHY", "asset_type": "Fixed-income / Treasury ETF", "weight": .30, "sec_yield_30d": np.nan, "yield_to_maturity": np.nan, "effective_duration": np.nan, "weighted_average_maturity": np.nan, "expense_ratio": np.nan, "volatility": np.nan, "drawdown": np.nan, "dollar_volume_30d": np.nan, "treasury_credit_exposure": "", "data_date": ""},
    ])


@st.cache_data(ttl=86400, show_spinner=False)
def cached_batch(tickers: tuple[str, ...], cache_dir: str, ttl: float):
    return batch_analyze(tickers, {"data": {"cache_dir": cache_dir, "cache_ttl_hours": ttl}})


def fmt_num(value, digits=2):
    try:
        value = float(value)
        return f"{value:,.{digits}f}" if np.isfinite(value) else "N/A"
    except Exception:
        return "N/A"


def fmt_pct(value, digits=1):
    try:
        value = float(value)
        return f"{value * 100:.{digits}f}%" if np.isfinite(value) else "N/A"
    except Exception:
        return "N/A"


def fmt_money(value):
    try:
        value = float(value)
        return f"${value:,.0f}" if np.isfinite(value) else "N/A"
    except Exception:
        return "N/A"


def pct_input(label, decimal_value, min_percent=-100.0, max_percent=100.0,
              step_percent=1.0, key=None) -> float:
    entered = st.number_input(
        label,
        min_value=float(min_percent),
        max_value=float(max_percent),
        value=float(decimal_value) * 100,
        step=float(step_percent),
        format="%.1f",
        key=key,
    )
    return float(entered) / 100


PERCENT_KEYS = {
    "revenue_growth_1y", "revenue_growth_3y", "revenue_growth_5y",
    "eps_growth_1y", "eps_growth_3y", "eps_growth_5y",
    "fcf_growth_1y", "fcf_growth_3y", "fcf_growth_5y",
    "net_income_growth_1y", "net_income_growth_3y", "net_income_growth_5y",
    "operating_margin", "net_margin", "return_on_equity", "return_on_assets",
    "earnings_consistency", "earnings_yield", "dividend_yield", "distribution_yield",
    "annualized_volatility", "max_drawdown", "downside_deviation", "expense_ratio",
    "total_return_1y", "total_return_3y", "total_return_5y", "sec_yield_30d",
    "yield_to_maturity",
}


def format_metric(key, value):
    if key in PERCENT_KEYS:
        return fmt_pct(value)
    if key in {"current_price", "total_assets", "dollar_volume_30d"}:
        return fmt_money(value)
    return fmt_num(value)


def format_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column.lower() in {"weight", "data confidence", "sector percentile"}:
            out[column] = out[column].map(lambda x: fmt_pct(float(x) / 100 if column.lower() != "weight" else x))
    return out


def score_row(metrics: dict) -> dict:
    scored = score_security(metrics, cfg)
    return {
        **metrics,
        "main_score": scored["main_score"],
        "fundamental_score": scored["main_score"],
        "growth_score": scored["growth_score"],
        "quality_score": scored["quality_score"],
        "valuation_score": scored["valuation_score"],
        "risk_score": scored["risk_score"],
        "data_confidence": scored["data_confidence"],
        "analysis_profile": scored["analysis_profile"],
        "scoring_engine": scored["scoring_engine"],
        "category_labels": scored.get("category_labels", {}),
        "metric_scores": scored["metric_scores"],
    }


def score_label(scored, key):
    return scored.get("category_labels", {}).get(key, key.title())


def show_security(metrics: dict, scored: dict, holdings: pd.DataFrame | None = None):
    engine = scored.get("scoring_engine", scored.get("analysis_profile", "Unknown"))
    st.markdown(
        f"<div style='font-size:1.05rem'><b>{metrics.get('company', metrics.get('ticker'))}</b> · "
        f"{metrics.get('sector', metrics.get('category', 'Unknown'))} · "
        f"{metrics.get('asset_class', 'Unknown')}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Detected scoring engine: {engine}")
    st.markdown(
        f"<div style='font-size:3rem;font-weight:700'>Main Quantitative Score: "
        f"{scored['main_score']:+.1f} (range -100 to +100)</div>",
        unsafe_allow_html=True,
    )
    st.caption("Main score is always -100 to +100. Scores from different engines are not directly comparable.")
    labels = [score_label(scored, key) for key in ("growth", "quality", "valuation")]
    c = st.columns(6)
    c[0].metric(labels[0], f"{scored['growth_score']:+.1f}")
    c[1].metric(labels[1], f"{scored['quality_score']:+.1f}")
    c[2].metric(labels[2], f"{scored['valuation_score']:+.1f}")
    c[3].metric("Risk & Resilience", f"{scored['risk_score']:+.1f}")
    c[4].metric("Data Confidence", fmt_pct(scored["data_confidence"] / 100))
    c[5].metric("Sector / Category", str(metrics.get("sector", metrics.get("category", "N/A"))))
    growth_keys = [
        ("Revenue 1Y", "revenue_growth_1y"), ("Revenue 3Y", "revenue_growth_3y"), ("Revenue 5Y", "revenue_growth_5y"),
        ("EPS 1Y", "eps_growth_1y"), ("EPS 3Y", "eps_growth_3y"), ("EPS 5Y", "eps_growth_5y"),
        ("FCF 1Y", "fcf_growth_1y"), ("FCF 3Y", "fcf_growth_3y"), ("FCF 5Y", "fcf_growth_5y"),
    ]
    if metrics.get("asset_class") == "stock":
        growth = pd.DataFrame({
            "Metric": [label for label, _ in growth_keys],
            "Raw growth": [fmt_pct(metrics.get(key)) for _, key in growth_keys],
            "Normalized score": [fmt_num(scored["metric_scores"].get(key)) for _, key in growth_keys],
        })
        st.dataframe(growth, hide_index=True, use_container_width=True)
    if holdings is not None:
        if holdings.empty:
            st.warning("Yahoo Finance did not provide a holdings list for this ETF.")
        else:
            st.dataframe(holdings.style.format({"weight": "{:.2%}"}), hide_index=True, use_container_width=True)
            st.download_button(
                "Download ETF holdings CSV",
                holdings.to_csv(index=False).encode("utf-8"),
                file_name=f"{metrics['ticker']}_holdings.csv",
                mime="text/csv",
                key=f"holdings_download_{metrics['ticker']}",
            )
    with st.expander("All raw metrics"):
        st.dataframe(pd.DataFrame({"Metric": list(metrics), "Value": [format_metric(k, v) for k, v in metrics.items()]}), hide_index=True, use_container_width=True)


st.title("Laura Gao Quantitative Investment System")
st.caption("Six-tab research and decision-support tool for security selection, portfolio construction, stress testing, and Laura-specific planning.")
st.info("Wharton evaluates strategy, client alignment, research, analysis, and communication. Use the official WInS universe and registered-team instructions for submissions.")

tabs = st.tabs([
    "1 Security Analysis & Selection",
    "2 Portfolio Optimizer",
    "3 Stress Test",
    "4 Settings",
    "5 Operating Reserve Assets",
    "6 Reserve Optimizer",
])

# ------------------------- TAB 1 -------------------------
with tabs[0]:
    st.subheader("Security Analysis & Selection")
    mode = st.radio("Section", ["A. Analyze One Security", "B. Sector Rankings", "C. Candidate Portfolio Builder"], horizontal=True, key="security_mode")
    if mode == "A. Analyze One Security":
        ticker = st.text_input("Ticker", "AAPL", key="security_ticker")
        if st.button("Analyze security", key="security_analyze"):
            try:
                normalized = normalize_ticker(ticker)
                with st.spinner(f"Downloading {normalized} data..."):
                    metrics, _ = analyze_security(normalized, cfg)
                    scored = score_row(metrics)
                    holdings = etf_holdings(normalized, cfg) if metrics["asset_class"] in {"equity_etf", "fixed_income"} else None
                    st.session_state.single_result = (metrics, scored, holdings)
            except Exception as exc:
                st.error(f"Could not analyze {ticker}: {exc}")
        if st.session_state.single_result:
            show_security(*st.session_state.single_result)
    elif mode == "B. Sector Rankings":
        st.success("The bundled ranking universe is local-only; opening this section makes no internet request.")
        source = st.radio("Universe source", ["Bundled US large-cap universe", "Upload custom/WInS CSV"], horizontal=True, key="rank_source")
        universe = None
        if source == "Bundled US large-cap universe":
            try:
                universe = load_local_universe()
            except Exception as exc:
                st.error(f"Could not load local universe: {exc}")
        else:
            upload = st.file_uploader("CSV with ticker and optional sector", type=["csv"], key="rank_upload")
            if upload is not None:
                universe = pd.read_csv(upload)
                universe.columns = [str(c).strip().lower() for c in universe.columns]
                if "ticker" not in universe:
                    st.error("CSV must contain a ticker column.")
                    universe = None
                elif "sector" not in universe:
                    universe["sector"] = "Uploaded universe"
        if universe is not None and not universe.empty:
            sectors = sorted(universe["sector"].fillna("Unknown").astype(str).unique())
            sector = st.selectbox("Sector", sectors, key="rank_sector")
            sector_df = universe[universe["sector"].fillna("Unknown").astype(str) == sector]
            count = st.slider("Number of names", 2, max(2, len(sector_df)), min(10, len(sector_df)), key="rank_count")
            if st.button("Run sector ranking", key="rank_run"):
                rows, errors = [], {}
                for ticker in sector_df["ticker"].astype(str).head(count):
                    raw, _, err = batch_analyze([ticker], cfg)
                    errors.update(err)
                    if not raw.empty:
                        row = score_row(raw.iloc[0].to_dict())
                        if row["asset_class"] == "stock":
                            row["sector"] = sector
                            rows.append(row)
                ranked = pd.DataFrame(rows).sort_values("main_score", ascending=False).reset_index(drop=True)
                if not ranked.empty:
                    ranked["sector_rank"] = np.arange(1, len(ranked) + 1)
                    ranked["sector_percentile"] = 100 if len(ranked) == 1 else 100 * (len(ranked) - ranked["sector_rank"]) / (len(ranked) - 1)
                st.session_state.sector_ranked, st.session_state.sector_errors = ranked, errors
            ranked = st.session_state.sector_ranked
            if not ranked.empty:
                columns = ["sector_rank", "ticker", "company", "main_score", "growth_score", "quality_score", "valuation_score", "risk_score", "data_confidence", "sector_percentile"]
                show = ranked[[c for c in columns if c in ranked]].rename(columns={"data_confidence": "Data Confidence", "sector_percentile": "Sector Percentile"})
                st.dataframe(format_frame(show), hide_index=True, use_container_width=True)
            if st.session_state.sector_errors:
                st.json(st.session_state.sector_errors)
    else:
        if "candidate_text_pending" in st.session_state:
            st.session_state.candidate_text = st.session_state.pop("candidate_text_pending")
        if "candidate_text" not in st.session_state:
            st.session_state.candidate_text = ", ".join(st.session_state.candidates)
        text = st.text_area("Mixed candidate tickers", key="candidate_text", height=110)
        if st.button("Save candidate list", key="candidate_save"):
            st.session_state.candidates = list(dict.fromkeys(normalize_ticker(x) for x in text.replace("\n", ",").split(",") if x.strip()))
            st.session_state.candidate_text_pending = ", ".join(st.session_state.candidates)
            st.success(f"Saved {len(st.session_state.candidates)} candidates.")
            st.rerun()
        if st.button("Analyze and load candidates", key="candidate_load"):
            with st.spinner("Downloading candidate data once..."):
                raw, prices, errors = cached_batch(tuple(st.session_state.candidates), cfg["data"]["cache_dir"], float(cfg["data"]["cache_ttl_hours"]))
                st.session_state.candidate_raw = pd.DataFrame([score_row(row.to_dict()) for _, row in raw.iterrows()])
                st.session_state.candidate_prices = price_frame(prices)
                st.session_state.candidate_errors = errors
                st.session_state.optimizer_result = None
        if not st.session_state.candidate_raw.empty:
            columns = ["ticker", "company", "main_score", "scoring_engine", "asset_class", "sector", "risk_score", "data_confidence"]
            show = st.session_state.candidate_raw[[c for c in columns if c in st.session_state.candidate_raw]].rename(columns={"data_confidence": "Data Confidence"})
            st.dataframe(format_frame(show), hide_index=True, use_container_width=True)
            st.write(f"Usable price series: {st.session_state.candidate_prices.shape[1]}")
        if st.session_state.candidate_errors:
            st.json(st.session_state.candidate_errors)

# ------------------------- TAB 2 -------------------------
with tabs[1]:
    st.subheader("Portfolio Optimizer")
    if st.session_state.candidate_prices.empty:
        st.warning("Use Security Analysis & Selection → Candidate Portfolio Builder first.")
    else:
        cap = float(cfg["optimizer"]["max_weight"])
        st.write(f"Long-only feasible portfolios; current maximum holding weight: **{cap:.0%}**.")
        if st.button("Run Monte Carlo optimizer", key="optimizer_run"):
            try:
                st.session_state.optimizer_result = monte_carlo_optimize(st.session_state.candidate_prices.dropna(how="all", axis=1), cfg)
            except Exception as exc:
                st.error(str(exc))
        result = st.session_state.optimizer_result
        if result:
            names = list(result["portfolios"])
            selected = st.selectbox("Portfolio", names, index=names.index(st.session_state.selected_portfolio) if st.session_state.selected_portfolio in names else 0, key="optimizer_select")
            st.session_state.selected_portfolio = selected
            summary = pd.DataFrame([{"Portfolio": n, "Expected return": p["expected_return"], "Volatility": p["volatility"], "Sharpe": p["sharpe"]} for n, p in result["portfolios"].items()])
            st.dataframe(summary.style.format({"Expected return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}"}), hide_index=True, use_container_width=True)
            p = result["portfolios"][selected]
            st.dataframe(pd.DataFrame({"Ticker": p["weights"].index, "Weight": p["weights"].values}).style.format({"Weight": "{:.2%}"}), hide_index=True, use_container_width=True)
            if selected == "Laura Goal Portfolio":
                st.caption(f"Laura Goal Portfolio construction: {p['construction']}; a model assumption, not an official competition rule.")
            st.dataframe(result["correlation"].round(2), use_container_width=True)
            distribution = simulate_2033_distribution(p["expected_return"], p["volatility"], cfg["laura"]["contribution_2027"], cfg["laura"]["contribution_2028"], int(cfg["laura"]["reserve_simulations"]), int(cfg["laura"]["reserve_simulation_seed"]))
            st.markdown("### Illustrative Expected 2033 Value")
            st.metric("Expected-value estimate", fmt_money(laura_value_2033(p["expected_return"], cfg["laura"]["contribution_2027"], cfg["laura"]["contribution_2028"])))
            st.caption("Uses only Laura's fixed 2027 and 2028 contributions. WInS gains/losses and reserve targets do not alter this projection.")
            st.dataframe(pd.DataFrame({"Percentile": ["5th", "25th", "50th", "75th", "95th"], "2033 value": [fmt_money(distribution.quantile(q)) for q in (.05, .25, .50, .75, .95)]}), hide_index=True, use_container_width=True)

# ------------------------- TAB 3 -------------------------
with tabs[2]:
    st.subheader("Stress Test")
    result = st.session_state.optimizer_result
    if not result:
        st.warning("Run the Portfolio Optimizer first.")
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
                st.error(str(exc))
        hist = st.session_state.get("hist_stress")
        if hist and hist.get("available"):
            h1, h2, h3 = st.columns(3)
            h1.metric("Portfolio return", fmt_pct(hist["portfolio_return"]))
            h2.metric("Max drawdown", fmt_pct(hist["max_drawdown"]))
            h3.metric("Weight coverage", fmt_pct(hist["weight_coverage"]))
            contributions = pd.DataFrame({
                "Ticker": hist["contributions"].index,
                "Holding return": hist["holding_returns"].reindex(hist["contributions"].index).values,
                "Contribution": hist["contributions"].values,
            })
            st.dataframe(contributions.style.format({"Holding return": "{:.2%}", "Contribution": "{:.2%}"}), hide_index=True, use_container_width=True)
            if hist["weight_coverage"] < .999:
                st.caption("Unavailable historical holdings were excluded and available weights were renormalized.")
        elif hist:
            st.warning(hist.get("reason", "Historical test unavailable."))
        c1, c2, c3 = st.columns(3)
        equity = pct_input("Equity shock", cfg["stress"]["custom_equity_shock"], -90, 50, 5, "stress_equity")
        tech = pct_input("Technology shock", cfg["stress"]["custom_tech_shock"], -90, 50, 5, "stress_tech")
        fixed = pct_input("Fixed-income shock", cfg["stress"]["custom_fixed_income_shock"], -50, 50, 1, "stress_fixed")
        metadata = st.session_state.candidate_raw[["ticker", "sector", "asset_class"]] if not st.session_state.candidate_raw.empty else pd.DataFrame()
        shocked = hypothetical_stress(weights, metadata, equity, tech, fixed)
        st.metric("Immediate portfolio shock", fmt_pct(shocked["loss_contribution"].sum()))
        st.dataframe(shocked.style.format({"weight": "{:.2%}", "shock": "{:.1%}", "loss_contribution": "{:.2%}"}), hide_index=True, use_container_width=True)

# ------------------------- TAB 4 -------------------------
with tabs[3]:
    st.subheader("Settings")
    st.caption("Percentage inputs accept values such as 20 for 20% and 4 for 4%; internal calculations remain decimals.")
    c1, c2, c3 = st.columns(3)
    growth = pct_input("Growth weight", cfg["scoring"]["category_weights"]["growth"], 0, 100, 5, "settings_growth")
    quality = pct_input("Quality weight", cfg["scoring"]["category_weights"]["quality"], 0, 100, 5, "settings_quality")
    valuation = pct_input("Valuation weight", cfg["scoring"]["category_weights"]["valuation"], 0, 100, 5, "settings_valuation")
    max_weight = pct_input("Maximum holding weight", cfg["optimizer"]["max_weight"], 5, 100, 5, "settings_max_weight")
    risk_free = pct_input("Risk-free rate", cfg["optimizer"]["risk_free_rate"], -5, 20, .5, "settings_rf")
    reserve_yield = pct_input("Reserve yield assumption", cfg["laura"]["reserve_yield"], 0, 15, .5, "settings_reserve_yield")
    simulations = st.number_input("Random portfolios", 500, 100000, int(cfg["optimizer"]["simulations"]), 500, key="settings_simulations")
    lookback = st.number_input("Price lookback (years)", 1, 10, int(cfg["optimizer"]["lookback_years"]), 1, key="settings_lookback")
    if st.button("Apply settings", key="settings_apply"):
        if growth + quality + valuation <= 0:
            st.error("At least one category weight must be positive.")
        else:
            cfg["scoring"]["category_weights"] = {"growth": growth, "quality": quality, "valuation": valuation}
            cfg["optimizer"]["max_weight"] = max_weight
            cfg["optimizer"]["risk_free_rate"] = risk_free
            cfg["optimizer"]["simulations"] = int(simulations)
            cfg["optimizer"]["lookback_years"] = int(lookback)
            cfg["laura"]["reserve_yield"] = reserve_yield
            st.session_state.cfg = cfg
            st.session_state.optimizer_result = None
            st.success("Settings applied.")
    if st.button("Reset to defaults", key="settings_reset"):
        st.session_state.cfg = load_config()
        st.session_state.optimizer_result = None
        st.rerun()

# ------------------------- TAB 5 -------------------------
with tabs[4]:
    st.subheader("Operating Reserve Assets")
    st.caption("Enter percentages directly. Only recognized fixed-income reserve ETFs contribute to the model reserve yield; unavailable data stays N/A.")
    editor = st.session_state.reserve_assets.copy()
    for col in ("weight", "sec_yield_30d", "yield_to_maturity", "expense_ratio", "volatility", "drawdown"):
        editor[col] = pd.to_numeric(editor[col], errors="coerce") * 100
    reserve_assets = st.data_editor(
        editor, num_rows="dynamic", hide_index=True, use_container_width=True, key="reserve_editor",
        column_config={
            "asset": st.column_config.TextColumn("Asset"),
            "asset_type": st.column_config.TextColumn("Asset Type"),
            "weight": st.column_config.NumberColumn("Weight (%)", min_value=0, max_value=100, format="%.1f"),
            "sec_yield_30d": st.column_config.NumberColumn("30-Day SEC Yield (%)", format="%.2f"),
            "yield_to_maturity": st.column_config.NumberColumn("Yield to Maturity (%)", format="%.2f"),
            "effective_duration": st.column_config.NumberColumn("Effective Duration (years)", format="%.2f"),
            "weighted_average_maturity": st.column_config.NumberColumn("Weighted Average Maturity (years)", format="%.2f"),
            "expense_ratio": st.column_config.NumberColumn("Expense Ratio (%)", format="%.2f"),
            "volatility": st.column_config.NumberColumn("Historical Volatility (%)", format="%.1f"),
            "drawdown": st.column_config.NumberColumn("Maximum Drawdown (%)", format="%.1f"),
            "dollar_volume_30d": st.column_config.NumberColumn("30-Day Average Dollar Volume", format="$%,.0f"),
            "treasury_credit_exposure": st.column_config.TextColumn("Treasury/Credit Exposure"),
            "data_date": st.column_config.TextColumn("Data Date"),
        },
    )
    for col in ("weight", "sec_yield_30d", "yield_to_maturity", "expense_ratio", "volatility", "drawdown"):
        reserve_assets[col] = pd.to_numeric(reserve_assets[col], errors="coerce") / 100
    reserve_assets["yield"] = pd.to_numeric(
        reserve_assets["yield_to_maturity"], errors="coerce"
    ).combine_first(pd.to_numeric(reserve_assets["sec_yield_30d"], errors="coerce"))
    for i, row in reserve_assets.iterrows():
        profile = reserve_asset_profile(row.get("asset", ""))
        if profile == "equity_etf":
            reserve_assets.at[i, "asset_type"] = "Equity ETF (not reserve-yield eligible)"
        elif profile == "fixed_income_etf":
            reserve_assets.at[i, "asset_type"] = "Fixed-income / Treasury ETF"
    st.session_state.reserve_assets = reserve_assets.drop(columns=["yield"], errors="ignore")
    try:
        normalized = normalize_reserve_weights(reserve_assets)
        st.dataframe(normalized, hide_index=True, use_container_width=True)
        st.metric("Estimated weighted reserve yield", fmt_pct(weighted_reserve_yield(normalized)))
        st.caption("Estimated/model yield from eligible fixed-income assets only; not guaranteed. SEC Yield and YTM are distinct fields.")
    except ValueError as exc:
        st.warning(str(exc))

# ------------------------- TAB 6 -------------------------
with tabs[5]:
    st.subheader("Reserve Optimizer")
    st.info("Fixed nominal liability: $500,000 = 10 × $50,000 beginning-of-year payments from 2033 through 2042.")
    st.metric("Zero-yield benchmark", fmt_money(LIABILITY))
    targets = tuple(float(x) for x in cfg["laura"].get("reserve_funding_targets", (.95, .99, .995, .999)))
    target = st.selectbox("Funding-success probability target", targets, index=targets.index(.995) if .995 in targets else 0, format_func=fmt_pct, key="reserve_target")
    simulations = max(100000, int(cfg["laura"].get("reserve_simulations", 100000)))
    seed = st.number_input("Reproducible simulation seed", min_value=1, value=int(cfg["laura"].get("reserve_simulation_seed", 20260924)), key="reserve_seed")
    st.caption("The seed makes identical inputs generate identical simulated paths; it is not a market forecast.")
    try:
        normalized = normalize_reserve_weights(st.session_state.reserve_assets)
        reserve_for_yield = st.session_state.reserve_assets.copy()
        reserve_for_yield["yield"] = pd.to_numeric(
            reserve_for_yield["yield_to_maturity"], errors="coerce"
        ).combine_first(pd.to_numeric(reserve_for_yield["sec_yield_30d"], errors="coerce"))
        y = weighted_reserve_yield(reserve_for_yield)
        vol = pd.to_numeric(normalized["volatility"], errors="coerce")
        if vol.isna().any():
            raise ValueError("Enter historical volatility for every selected fixed-income reserve asset.")
        portfolio_vol = float(np.sqrt((normalized["weight"] * vol.pow(2)).sum()))
        st.metric("Estimated weighted reserve yield", fmt_pct(y))
        st.metric("Deterministic yield-adjusted PV", fmt_money(deterministic_reserve_pv(y)))
        if st.button("Run reserve analysis", key="reserve_run"):
            with st.spinner("Running reproducible 100,000+ path analysis..."):
                st.session_state.reserve_optimizer_result = {
                    "results": required_reserves(y, portfolio_vol, targets, simulations, int(seed)),
                    "signature": (tuple(reserve_for_yield[["asset", "weight", "yield"]].fillna("").itertuples(index=False, name=None)), targets, simulations, int(seed)),
                }
        signature = (tuple(reserve_for_yield[["asset", "weight", "yield"]].fillna("").itertuples(index=False, name=None)), targets, simulations, int(seed))
        saved = st.session_state.reserve_optimizer_result
        if saved and saved.get("signature") == signature:
            result = saved["results"].copy()
            display = result.copy()
            display["Funding-success probability"] = display["funding_target"].map(fmt_pct)
            display["Failure rate"] = display["failure_rate"].map(fmt_pct)
            display["Required reserve"] = display["required_reserve"].map(fmt_money)
            display["Simulated success"] = display["simulated_success"].map(fmt_pct)
            st.dataframe(display[["Funding-success probability", "Failure rate", "Required reserve", "Simulated success"]], hide_index=True, use_container_width=True)
            selected = float(result.loc[np.isclose(result["funding_target"], target), "required_reserve"].iloc[0])
            st.metric("Reserve surplus / shortfall vs $450,000 starting contributions", fmt_money(450000 - selected))
        else:
            st.caption("Click Run reserve analysis after reviewing the asset inputs.")
    except ValueError as exc:
        st.warning(str(exc))

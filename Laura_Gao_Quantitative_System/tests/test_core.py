import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, merge_config_defaults
from src.data import (
    _cagr_details,
    _current_price_details,
    _credit_score_from_exposure,
    _cagr,
    _derive_free_cash_flow,
    _expense_ratio_details,
    _expense_ratio_fraction,
    _market_cap_details,
    _historical_price_return,
    _interest_coverage,
    _market_beta,
    _paired_ttm_values,
    _quarterly_ttm,
    _price_risk,
    _return_sanity_warnings,
    classify_asset,
    _statement_ttm_value,
    _statement_row_details,
    _valuation_measure,
    etf_holdings,
    load_local_universe,
    normalize_ticker,
    reserve_asset_profile,
)
import src.data as data_module
from src.optimizer import (
    _bounded_weights,
    _laura_sector_constraints_satisfied,
    annualized_covariance,
    annualized_expected_returns,
    monte_carlo_optimize,
    laura_sector_allocation_table,
    normalize_sector,
    normalize_sector_weights,
    portfolio_benchmark_comparison,
    portfolio_stats,
    sector_exposure_coverage,
    sector_allocation_table,
    shrink_expected_returns,
    simulate_laura_funding,
)
from src.scoring import normalize_lower_better, normalize_piecewise, normalize_symmetric, score_security
from src.formatting import format_pct
from src.sectors import LAURA_SECTOR_TARGETS, laura_sector_bucket, normalize_laura_sector_weights
from src.stress import historical_portfolio_stress, hypothetical_stress

cfg = load_config()


def assert_metric_weights(rows, expected):
    actual = {row["metric"]: row["final_weight"] for row in rows}
    assert actual.keys() == expected.keys()
    assert all(np.isclose(actual[key], value) for key, value in expected.items())


assert _credit_score_from_exposure(
    {"us_government": .50, "bb": .50}, cfg
) == 25.0
assert _credit_score_from_exposure({"us_government": 1.0}, cfg) == 100.0
assert np.isnan(_credit_score_from_exposure({"us_government": .50}, cfg))
assert np.isnan(
    _credit_score_from_exposure({"us_government": .50, "not_rated": .50}, cfg)
)
bnd_credit_score = _credit_score_from_exposure(
    {
        "bb": .0001, "aa": .72540003, "aaa": .0309, "a": .120799996,
        "other": .00029999999, "b": 0.0, "bbb": .1225, "below_b": 0.0,
        "us_government": .5179,
    },
    cfg,
)
assert np.isfinite(bnd_credit_score) and 0 <= bnd_credit_score <= 100
assert np.isclose(sum(normalize_sector_weights(cfg["optimizer"]["sector_benchmark_weights"]).values()), 1.0)
for source, normalized in {
    "Technology": "Information Technology",
    "Consumer Cyclical": "Consumer Discretionary",
    "Financial Services": "Financials",
    "Consumer Defensive": "Consumer Staples",
    "Healthcare": "Health Care",
    "Communication Services": "Communication Services",
}.items():
    assert normalize_sector(source) == normalized
stale_cfg = merge_config_defaults(cfg, {
    "optimizer": {
        "max_weight": .25,
        "obsolete_setting": True,
    },
})
assert stale_cfg["optimizer"]["minimum_equity_weight"] == cfg["optimizer"]["minimum_equity_weight"]
assert stale_cfg["optimizer"]["maximum_fixed_income_weight"] == cfg["optimizer"]["maximum_fixed_income_weight"]
assert stale_cfg["optimizer"]["max_weight"] == .25
assert stale_cfg["optimizer"]["obsolete_setting"] is True

assert normalize_symmetric(.30, .30) == 100
assert normalize_symmetric(-.30, .30) == -100
assert normalize_symmetric(1.20, .30) == 100
assert normalize_piecewise(5, 1, 5, 10) == 0
assert normalize_lower_better(5, 1, 5, 10) == 0
assert np.isnan(normalize_piecewise(1, 3, 2, 1))

adjusted_prices = pd.Series(
    [10.0, 100.0, 144.0, 172.8],
    index=pd.to_datetime(["2020-01-01", "2021-01-01", "2023-01-01", "2024-01-01"]),
)
assert np.isclose(_historical_price_return(adjusted_prices, 1), .20)
assert np.isclose(_historical_price_return(adjusted_prices, 3, cagr=True), .20)
assert np.isclose(_historical_price_return(pd.Series([100.0, 110.0], index=pd.to_datetime(["2020-01-01", "2021-01-01"])), 1), .10)
assert np.isclose(_historical_price_return(pd.Series([100.0, 133.1], index=pd.to_datetime(["2020-01-01", "2023-01-01"])), 3, cagr=True), .10, atol=1e-10)
assert np.isclose(_historical_price_return(pd.Series([100.0, 80.0], index=pd.to_datetime(["2020-01-01", "2021-01-01"])), 1), -.20)
historical_prices_with_older_data = pd.Series(
    [10.0, 100.0, 110.0, 133.1],
    index=pd.to_datetime(["2018-01-01", "2020-01-01", "2021-01-01", "2023-01-01"]),
)
assert np.isclose(_historical_price_return(historical_prices_with_older_data, 1), .21)
assert np.isclose(_historical_price_return(historical_prices_with_older_data, 3, cagr=True), .10, atol=1e-10)
assert format_pct(.10) == "10.0%"
assert format_pct(.1199) == "12.0%"
assert format_pct(-.20) == "-20.0%"
assert format_pct(np.nan) == "N/A"
assert len(_return_sanity_warnings({"total_return_1y": 5.1})) == 1
assert not _return_sanity_warnings({"total_return_1y": 5.0})
assert np.isclose(sum(target["target"] for target in LAURA_SECTOR_TARGETS.values()), 1.0)
assert laura_sector_bucket("Healthcare") == "Health Care"
assert laura_sector_bucket("Financial Services") == "Financial"
assert laura_sector_bucket("Technology") == "Technology"
assert laura_sector_bucket("Industrials", "Farm & Heavy Construction Machinery") == "Others"
assert laura_sector_bucket("Industrials", "Engineering & Construction") == "Construction / Infrastructure"
laura_etf_weights = normalize_laura_sector_weights({
    "Technology": .30,
    "Healthcare": .20,
    "Financial Services": .10,
    "Energy": .10,
    "Industrials": .10,
    "Consumer Defensive": .10,
    "Consumer Cyclical": .10,
})
assert np.isclose(sum(laura_etf_weights.values()), 1.0)
laura_metadata = pd.DataFrame([
    {"ticker": "AAPL", "asset_class": "stock", "sector": "Information Technology", "industry": "Consumer Electronics", "sector_weightings": {}},
    {"ticker": "VOO", "asset_class": "equity_etf", "sector": "Unknown", "industry": "", "sector_weightings": {"Technology": .30, "Healthcare": .20, "Financial Services": .10, "Energy": .10, "Industrials": .10, "Consumer Defensive": .10, "Consumer Cyclical": .10}},
    {"ticker": "BND", "asset_class": "fixed_income", "sector": "Unknown", "industry": "", "sector_weightings": {}},
])
laura_table, laura_coverage = laura_sector_allocation_table(
    pd.Series({"AAPL": .20, "VOO": .50, "BND": .30}),
    laura_metadata,
)
laura_technology = laura_table.set_index("sector").loc["Technology"]
assert np.isclose(laura_coverage["equity_weight"], .70)
assert np.isclose(laura_technology["equity_sleeve_weight"], (.20 + .50 * .30) / .70)
assert np.isclose(laura_technology["total_portfolio_weight"], .35)
assert np.isclose(laura_table["total_portfolio_weight"].sum(), .70)
assert laura_table["total_portfolio_weight"].max() <= 1.0
assert laura_coverage["complete"]
incomplete_metadata = laura_metadata.copy()
incomplete_metadata.loc[incomplete_metadata["ticker"] == "VOO", "sector_weightings"] = pd.Series(
    [{}], index=incomplete_metadata.index[incomplete_metadata["ticker"] == "VOO"]
)
incomplete_metadata.loc[incomplete_metadata["ticker"] == "VOO", "ticker"] = "QQQ"
incomplete_table, incomplete_coverage = laura_sector_allocation_table(
    pd.Series({"AAPL": .20, "QQQ": .50, "BND": .30}),
    incomplete_metadata,
)
assert not incomplete_coverage["complete"]
assert set(incomplete_table["status"]) == {"INCOMPLETE"}


class FakePriceTicker:
    def __init__(self, fast_info):
        self.fast_info = fast_info


quote_date = pd.Timestamp("2024-01-02", tz="UTC")
fast_quote = _current_price_details(
    FakePriceTicker({"last_price": 123.45, "currency": "USD"}),
    {"currentPrice": 120.0, "currency": "USD"},
    pd.Series([119.0], index=pd.to_datetime(["2024-01-01"])),
)
assert fast_quote["current_price"] == 123.45
assert fast_quote["price_source"] == "yfinance fast_info.last_price"
info_quote = _current_price_details(
    FakePriceTicker({}),
    {"currentPrice": 122.0, "currency": "USD", "regularMarketTime": quote_date.timestamp()},
    pd.Series([119.0], index=pd.to_datetime(["2024-01-01"])),
)
assert info_quote["current_price"] == 122.0
history_quote = _current_price_details(
    FakePriceTicker({}),
    {"currency": "USD"},
    pd.Series([119.0], index=pd.to_datetime(["2024-01-01"])),
)
assert history_quote["current_price"] == 119.0
assert "Latest available close" in history_quote["price_freshness"]
missing_quote = _current_price_details(FakePriceTicker({}), {}, pd.Series(dtype=float))
assert np.isnan(missing_quote["current_price"])
assert missing_quote["current_price"] != 0
zero_quote = _current_price_details(
    FakePriceTicker({"last_price": 0}),
    {"regularMarketPrice": -1, "currency": "USD"},
    pd.Series([117.0], index=pd.to_datetime(["2024-01-01"])),
)
assert zero_quote["current_price"] == 117.0

transposed_statement = pd.DataFrame(
    {"2020-12-31": [10.0], "2021-12-31": [12.0]},
    index=["Revenue"],
)
statement_values, statement_field, statement_reason = _statement_row_details(
    transposed_statement.T, ["Revenue"]
)
assert statement_field == "Revenue" and len(statement_values) == 2

short_history_value, short_history_diagnostic = _cagr_details(
    pd.Series([10.0, 12.0, 14.0], index=pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])),
)
assert np.isnan(short_history_value)
assert "Insufficient valid annual history" in short_history_diagnostic["reason"]
annual_statement = pd.Series(
    [10.0, 12.0, 15.0, 18.0, 20.0],
    index=pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]),
)
_, annual_details = _cagr_details(annual_statement)
elapsed_years = annual_details["elapsed_years"]
assert np.isclose(_cagr(annual_statement), (20.0 / 12.0) ** (1 / elapsed_years) - 1)
invalid_cagr = annual_statement.copy()
invalid_cagr.loc["2021-12-31"] = -12.0
assert np.isnan(_cagr(invalid_cagr))
five_year_only = pd.Series(
    [100.0, 200.0],
    index=pd.to_datetime(["2020-01-01", "2025-01-01"]),
)
assert np.isnan(_cagr(five_year_only))
daily_market = np.linspace(-.01, .01, 100)
daily_asset = .001 + 1.5 * daily_market
risk_dates = pd.bdate_range("2023-01-02", periods=101)
market_series = pd.Series(100 * np.cumprod(np.r_[1.0, 1.0 + daily_market]), index=risk_dates)
asset_series = pd.Series(100 * np.cumprod(np.r_[1.0, 1.0 + daily_asset]), index=risk_dates)
assert np.isclose(_market_beta(asset_series, market_series), 1.5)
risk_metrics = _price_risk(asset_series, 1.5)
assert np.isclose(risk_metrics["annualized_volatility"], np.std(daily_asset, ddof=1) * np.sqrt(252))
assert np.isclose(
    risk_metrics["max_drawdown"],
    abs((asset_series / asset_series.cummax() - 1).min()),
)

stock_metrics = {
    "scoring_model": "STANDARD_STOCK",
    "eps_growth_3y": .20,
    "fcf_growth_3y": .20,
    "net_debt_to_fcf": 1.0, "net_debt": 100, "free_cash_flow": 200,
    "interest_coverage": 10, "fcf_yield": .05, "trailing_pe": 20,
    "annualized_volatility": .22, "max_drawdown": .25, "beta": 1.0,
}
stock = score_security(stock_metrics, cfg)
assert -100 <= stock["main_score"] <= 100
assert np.isclose(stock["main_score"], .8 * stock["fundamental_score"] + .2 * stock["risk_score"])
assert np.isclose(
    stock["fundamental_score"],
    .4 * stock["growth_score"] + .3 * stock["quality_score"] + .3 * stock["valuation_score"],
)
assert stock["data_confidence"] == 1.0
assert np.isclose(sum(row["final_weight"] for row in stock["breakdown"]), 1.0)
assert_metric_weights(stock["breakdown"], {
    "EPS GROWTH 3Y": .16,
    "FCF GROWTH 3Y": .16,
    "Net Debt To Fcf": .12,
    "Interest Coverage": .12,
    "P/E": .12,
    "FCF Yield": .12,
    "Max Drawdown": .08,
    "Annualized Volatility": .07,
    "Beta": .05,
})
assert normalize_piecewise(.20, -.10, .05, .20) == 100
assert normalize_piecewise(.05, -.10, .05, .20) == 0
assert normalize_piecewise(-.10, -.10, .05, .20) == -100
assert normalize_lower_better(10, 10, 25, 50) == 100
assert normalize_lower_better(25, 10, 25, 50) == 0
assert normalize_lower_better(50, 10, 25, 50) == -100
assert normalize_lower_better(0, 0, 3, 6) == 100
assert normalize_lower_better(3, 0, 3, 6) == 0
assert normalize_lower_better(6, 0, 3, 6) == -100
assert normalize_piecewise(1, 1, 5, 15) == -100
assert normalize_piecewise(5, 1, 5, 15) == 0
assert normalize_piecewise(15, 1, 5, 15) == 100
sums_to_final = sum(
    row["final_contribution"]
    for row in stock["breakdown"]
    if np.isfinite(row["final_contribution"])
)
assert np.isclose(sums_to_final, stock["main_score"])
sparse_stock = score_security({"scoring_model": "STANDARD_STOCK", "trailing_pe": 20}, cfg)
assert sparse_stock["data_confidence"] < stock["data_confidence"]
nonpositive_pe = score_security({**stock_metrics, "trailing_pe": 0}, cfg)
assert np.isnan(nonpositive_pe["metric_scores"]["P/E"])
assert np.isnan({row["metric"]: row for row in nonpositive_pe["breakdown"]}["P/E"]["raw_value"])
assert nonpositive_pe["data_confidence"] < stock["data_confidence"]
for fcf_yield, expected_score in ((.08, 100), (.04, 0), (.02, -100)):
    yield_score = score_security(
        {**stock_metrics, "free_cash_flow": 100, "fcf_yield": fcf_yield},
        cfg,
    )
    assert {row["metric"]: row["score"] for row in yield_score["breakdown"]}["FCF Yield"] == expected_score
negative_fcf_score = score_security(
    {
        **stock_metrics,
        "free_cash_flow": -10,
        "net_debt": 20,
        "net_debt_to_fcf": np.nan,
        "fcf_yield": -.01,
    },
    cfg,
)
negative_fcf_rows = {row["metric"]: row for row in negative_fcf_score["breakdown"]}
assert negative_fcf_rows["Net Debt To Fcf"]["score"] == -100
assert np.isnan(negative_fcf_rows["Net Debt To Fcf"]["raw_value"])
assert negative_fcf_rows["FCF Yield"]["score"] == -100
zero_fcf_score = score_security(
    {**stock_metrics, "free_cash_flow": 0, "net_debt": 20, "net_debt_to_fcf": np.nan, "fcf_yield": 0},
    cfg,
)
assert {row["metric"]: row["score"] for row in zero_fcf_score["breakdown"]}["FCF Yield"] == -100

financial_metrics = {
    "scoring_model": "SPECIAL_FINANCIAL_STOCK",
    "eps_growth_3y": .06,
    "trailing_pe": 15, "interest_coverage": 5.0,
    "annualized_volatility": .20, "max_drawdown": .30, "beta": 1,
}
financial = score_security(financial_metrics, cfg)
assert financial["scoring_model"] == "SPECIAL_FINANCIAL_STOCK"
assert financial["category_labels"]["quality"] == "Financial Strength"
assert -100 <= financial["main_score"] <= 100
assert financial["data_confidence"] < 1.0
assert np.isclose(financial["data_confidence"], .48)
assert financial["low_confidence"]
assert financial["warning"].startswith("Financial-company score has reduced comparability")
assert np.isnan(financial["metric_scores"]["Net Debt To Fcf"])
assert np.isnan(financial["metric_scores"]["Interest Coverage"])
assert np.isnan(financial["metric_scores"]["FCF Yield"])
invalid_financial_rows = {row["metric"]: row for row in financial["breakdown"]}
assert np.isnan(invalid_financial_rows["Net Debt To Fcf"]["raw_value"])
assert np.isnan(invalid_financial_rows["Interest Coverage"]["raw_value"])
assert np.isnan(invalid_financial_rows["FCF Yield"]["raw_value"])
assert np.isnan(invalid_financial_rows["FCF GROWTH 3Y"]["raw_value"])
assert np.isclose(sum(row["final_contribution"] for row in financial["breakdown"] if np.isfinite(row["final_contribution"])), financial["main_score"])
for ticker in ("JPM", "BAC", "WFC", "USB", "GS", "MS", "C", "SCHW", "PNC", "AXP", "BLK", "BRK.B"):
    scoring_model = classify_asset(normalize_ticker(ticker), {"quoteType": "EQUITY"})
    scored_financial = score_security({**financial_metrics, "scoring_model": scoring_model}, cfg)
    assert scoring_model == "SPECIAL_FINANCIAL_STOCK"
    assert scored_financial["warning"] == "Financial-company score has reduced comparability."
    assert np.isclose(scored_financial["data_confidence"], .48)
for ticker in ("COF", "DFS", "SYF", "ALLY", "ALL", "MET", "PRU", "AIG"):
    assert classify_asset(ticker, {"quoteType": "EQUITY"}) == "SPECIAL_FINANCIAL_STOCK"

for ticker in ("V", "MA", "PYPL", "FI", "GPN"):
    scoring_model = classify_asset(
        ticker,
        {
            "quoteType": "EQUITY",
            "sector": "Financial Services",
            "industry": "Credit Services",
        },
    )
    assert scoring_model == "STANDARD_STOCK"
    assert laura_sector_bucket("Financial Services", "Credit Services") == "Financial"
assert classify_asset(
    "UNKNOWN",
    {"quoteType": "EQUITY", "sector": "Financial Services", "industry": "Credit Services"},
) == "STANDARD_STOCK"
assert classify_asset(
    "UNKNOWN",
    {"quoteType": "EQUITY", "industry": "Insurance - Diversified"},
) == "SPECIAL_FINANCIAL_STOCK"

complete_operating_metrics = {
    "scoring_model": "STANDARD_STOCK",
    "eps_growth_3y": .173,
    "fcf_growth_3y": .168,
    "net_debt_to_fcf": 1.0,
    "interest_coverage": 10.0,
    "trailing_pe": 25.0,
    "fcf_yield": .04,
    "annualized_volatility": .16,
    "max_drawdown": .18,
    "beta": 1.0,
}
for ticker in ("V", "MA", "PYPL"):
    scoring_model = classify_asset(
        ticker,
        {"quoteType": "EQUITY", "sector": "Financial Services", "industry": "Credit Services"},
    )
    scored_operating = score_security({**complete_operating_metrics, "scoring_model": scoring_model}, cfg)
    assert scored_operating["scoring_engine"] == "Standard Operating Stock"
    assert all(
        np.isfinite(scored_operating["metric_scores"][metric])
        for metric in (
            "EPS GROWTH 3Y", "FCF GROWTH 3Y", "Net Debt To Fcf",
            "Interest Coverage", "P/E", "FCF Yield",
        )
    )

etf_metrics = {
    "scoring_model": "EQUITY_ETF",
    "total_return_1y": .12, "total_return_3y": .10,
    "expense_ratio": .0009, "holdings_count": 500, "top10_concentration": .25,
    "sector_weightings": {
        "Information Technology": .20, "Financials": .10, "Health Care": .10,
        "Consumer Discretionary": .10, "Consumer Staples": .10, "Industrials": .10,
        "Energy": .10, "Communication Services": .10, "Utilities": .05,
        "Materials": .03, "Real Estate": .02,
    },
    "average_dollar_volume_30d": 100_000_000,
    "annualized_volatility": .18, "max_drawdown": .25, "beta": 1.0,
}
etf = score_security(etf_metrics, cfg)
etf_without_returns = score_security(
    {**etf_metrics, "total_return_1y": -0.9, "total_return_3y": -0.9},
    cfg,
)
assert etf["scoring_model"] == "EQUITY_ETF"
assert np.isclose(etf["main_score"], etf_without_returns["main_score"])
assert np.isclose(
    etf["main_score"],
    .35 * etf["group_scores"]["Diversification"]
    + .30 * etf["group_scores"]["Cost / Efficiency"]
    + .20 * etf["group_scores"]["Risk"]
    + .15 * etf["group_scores"]["Liquidity"],
)
assert np.isfinite(etf["risk_score"])
assert etf["data_confidence"] == 1.0
assert -100 <= etf["main_score"] <= 100
for dollar_volume, expected_score in (
    (5_000_000, -100),
    (50_000_000, 0),
    (500_000_000, 100),
):
    volume_score = score_security(
        {**etf_metrics, "average_dollar_volume_30d": dollar_volume},
        cfg,
    )
    assert {row["metric"]: row["score"] for row in volume_score["breakdown"]}[
        "Average Daily Dollar Volume"
    ] == expected_score
assert cfg["scoring"]["equity_etf"]["expense_ratio_thresholds"] == [0.0005, 0.0025, 0.0075]
assert np.isclose(sum(row["final_weight"] for row in etf["breakdown"]), 1.0)
assert_metric_weights(etf["breakdown"], {
    "Top-10 Concentration": .14,
    "Largest Sector Excess vs S&P 500": .1225,
    "Holdings Count": .0875,
    "Expense Ratio": .30,
    "Max Drawdown": .11,
    "Annualized Volatility": .09,
    "Average Daily Dollar Volume": .15,
})
voo_sector_score = score_security(
    {
        **etf_metrics,
        "sector_weightings": {"Information Technology": .37, "Financials": .12, "Health Care": .10,
                              "Consumer Discretionary": .10, "Consumer Staples": .05, "Industrials": .08,
                              "Energy": .04, "Communication Services": .09, "Utilities": .02,
                              "Materials": .02, "Real Estate": .01},
    },
    cfg,
)
qqq_sector_score = score_security(
    {
        **etf_metrics,
        "sector_weightings": {"Information Technology": .60, "Financials": .01, "Health Care": .04,
                              "Consumer Discretionary": .11, "Consumer Staples": .06, "Industrials": .04,
                              "Energy": .01, "Communication Services": .12, "Utilities": .01,
                              "Materials": .01, "Real Estate": 0.0},
    },
    cfg,
)
assert voo_sector_score["metric_scores"]["Largest Sector Excess vs S&P 500"] > 0
assert qqq_sector_score["metric_scores"]["Largest Sector Excess vs S&P 500"] < 0
assert qqq_sector_score["metric_scores"]["Largest Sector Excess vs S&P 500"] < voo_sector_score["metric_scores"]["Largest Sector Excess vs S&P 500"]

fixed_metrics = {
    "scoring_model": "FIXED_INCOME_ETF",
    "total_return_1y": .04, "total_return_3y": .02,
    "effective_duration": 4.25, "weighted_credit_score": 75, "expense_ratio": .001,
    "average_dollar_volume_30d": 100_000_000,
    "annualized_volatility": .06, "max_drawdown": .08, "beta": .1,
}
fixed = score_security(fixed_metrics, cfg)
assert fixed["scoring_model"] == "FIXED_INCOME_ETF"
assert np.isclose(
    sum(row["final_contribution"] for row in fixed["breakdown"] if np.isfinite(row["final_contribution"])),
    fixed["main_score"],
)
assert np.isfinite(fixed["risk_score"])
for duration, expected_score in ((7.0, 100.0), (9.0, 0.0), (11.0, -100.0)):
    term_fit = score_security(
        {**fixed_metrics, "analysis_year": 2026, "effective_duration": duration},
        cfg,
    )
    assert term_fit["metric_scores"]["Duration Gap to 2033 Target"] == expected_score
assert fixed["data_confidence"] < 1.0  # Explicit SEC Yield / YTM data is unavailable.
assert -100 <= fixed["main_score"] <= 100
assert np.isclose(sum(row["final_weight"] for row in fixed["breakdown"]), 1.0)
assert_metric_weights(fixed["breakdown"], {
    "Duration Gap to 2033 Target": .40,
    "Weighted Credit Score": .35,
    "YTM Spread": .10,
    "Average Daily Dollar Volume": .15,
})
assert np.isnan(fixed["metric_scores"]["YTM Spread"])
assert np.isnan(
    score_security(
        {**fixed_metrics, "sec_yield_30d": .08, "yield_to_maturity": np.nan},
        cfg,
    )["metric_scores"]["YTM Spread"]
)
assert np.isclose(_derive_free_cash_flow(
    pd.Series([146.7, 146.7], index=[2022, 2023]),
    pd.Series([-10.0, 10.0], index=[2022, 2023]),
).to_numpy(), [136.7, 136.7]).all()
for spread, expected in ((.02, 100), (0.0, 0), (-.01, -100)):
    spread_score = score_security(
        {**fixed_metrics, "yield_to_maturity": .05, "comparable_treasury_ytm": .05 - spread},
        cfg,
    )
    assert spread_score["metric_scores"]["YTM Spread"] == expected

assert normalize_ticker("BRK.B") == "BRK-B"
assert normalize_ticker("BF.B") == "BF-B"
assert np.isclose(_expense_ratio_fraction({"annualReportExpenseRatio": .03}), .0003)
assert np.isclose(_expense_ratio_fraction({"annualReportExpenseRatio": .0003}), .000003)
assert np.isclose(_expense_ratio_fraction({"netExpenseRatio": .03}), .0003)
assert np.isclose(
    _expense_ratio_fraction(
        {},
        pd.DataFrame(
            {"VOO": [.0003], "Category Average": [.0072]},
            index=["Annual Report Expense Ratio"],
        ),
    ),
    .0003,
)
assert np.isnan(_expense_ratio_fraction({"annualReportExpenseRatio": -.03}))
ratio, reported_ratio, ratio_source = _expense_ratio_details({"annualReportExpenseRatio": .03})
assert np.isclose(ratio, .0003)
assert reported_ratio == .03
assert "percentage points" in ratio_source
assert np.isclose(_expense_ratio_details({}, pd.DataFrame(
    {"VOO": [.0003], "Category Average": [.0072]},
    index=["Annual Report Expense Ratio"],
))[0], .0003)
assert np.isclose(
    _market_cap_details({"marketCap": 1_000_000}, {"market_cap": 2_000_000}, pd.DataFrame(), 100)[0],
    2_000_000,
)
valuation_value, valuation_source = _valuation_measure(
    pd.DataFrame({"Current": [25.0], "6/30/2026": [24.0]}, index=["Trailing P/E"]),
    ["Trailing P/E"],
)
assert valuation_value == 25.0 and "Current" in valuation_source
quarterly_cashflows = pd.DataFrame(
    {"2024-03-31": [100.0], "2024-06-30": [110.0], "2024-09-30": [120.0], "2024-12-31": [130.0]},
    index=["Operating Cash Flow"],
)
quarterly_series, _, _ = _statement_row_details(quarterly_cashflows, ["Operating Cash Flow"])
quarterly_total, quarterly_reason = _quarterly_ttm(quarterly_series)
assert np.isclose(quarterly_total, 460.0)
ttm_value, ttm_field, ttm_source, _ = _statement_ttm_value(
    pd.DataFrame(), quarterly_cashflows, ["Operating Cash Flow"]
)
assert np.isclose(ttm_value, 460.0) and ttm_source == "Quarterly sum"
interest_dates = pd.to_datetime(["2022-12-31", "2023-12-31"])
interest_income = pd.DataFrame(
    [[100.0, 200.0], [-10.0, 0.0]],
    index=["Operating Income", "Interest Expense"],
    columns=interest_dates,
)
assert _interest_coverage(interest_income.loc["Operating Income"], interest_income, {}) == 15.0
interest_income.loc["Interest Expense", interest_dates[-1]] = -20.0
assert np.isclose(
    _interest_coverage(interest_income.loc["Operating Income"], interest_income, {}),
    10.0,
)
staggered_interest = pd.DataFrame(
    [[100.0, 200.0, np.nan], [-10.0, np.nan, -20.0]],
    index=["Operating Income", "Interest Expense"],
    columns=pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31"]),
)
assert np.isclose(
    _interest_coverage(staggered_interest.loc["Operating Income"], staggered_interest, {}),
    10.0,
)
paired_cfo, paired_capex, _, _, paired_period = _paired_ttm_values(
    pd.DataFrame(),
    pd.DataFrame(
        {
            "2024-03-31": [100.0, -5.0],
            "2024-06-30": [110.0, -6.0],
            "2024-09-30": [120.0, -7.0],
            "2024-12-31": [130.0, -8.0],
        },
        index=["Operating Cash Flow", "Capital Expenditure"],
    ),
    ["Operating Cash Flow"],
    ["Capital Expenditure"],
)
assert np.isclose(paired_cfo, 460.0) and np.isclose(paired_capex, -26.0)
assert "Matched four-quarter period" in paired_period
original_download = data_module.yf.download
data_module.yf.download = lambda *args, **kwargs: pd.DataFrame(
    [[100, 50], [101, 49]],
    index=pd.bdate_range("2024-01-02", periods=2),
    columns=pd.MultiIndex.from_tuples([("Close", "SPY"), ("Close", "BND")]),
)
try:
    assert list(data_module.download_price_period(["SPY"], "2024-01-01", "2024-01-05").columns) == ["SPY"]
    assert list(data_module.download_price_period(["SPY", "BND"], "2024-01-01", "2024-01-05").columns) == ["SPY", "BND"]
finally:
    data_module.yf.download = original_download
original_ticker = data_module.yf.Ticker


class FakeFundTicker:
    @property
    def funds_data(self):
        class FundData:
            top_holdings = pd.DataFrame(
                {"Name": ["One", "Two"], "Holding Percent": [.10, 5.0]},
                index=pd.Index(["ONE", "TWO"], name="Symbol"),
            )

        return FundData()


data_module.yf.Ticker = lambda ticker: FakeFundTicker()
try:
    with tempfile.TemporaryDirectory() as cache_dir:
        holdings = etf_holdings(
            "TEST",
            {"data": {"cache_dir": cache_dir, "cache_ttl_hours": 24}},
        )
        assert holdings.loc[0, "weight"] == .10
        assert np.isnan(holdings.loc[1, "weight"])
finally:
    data_module.yf.Ticker = original_ticker
assert classify_asset("SPY", {"quoteType": "ETF", "category": "Large Blend"}) == "EQUITY_ETF"
assert classify_asset("TLT", {"quoteType": "ETF", "category": "Intermediate Government"}) == "FIXED_INCOME_ETF"
assert classify_asset("BRK-B", {"quoteType": "EQUITY", "longName": "Berkshire Hathaway Inc."}) == "SPECIAL_FINANCIAL_STOCK"
for ticker in ("AAPL", "MSFT"):
    assert classify_asset(ticker, {"quoteType": "EQUITY"}) == "STANDARD_STOCK"
for ticker in ("JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "PNC", "AXP", "BLK"):
    assert classify_asset(ticker, {"quoteType": "EQUITY"}) == "SPECIAL_FINANCIAL_STOCK"
for ticker in ("VOO", "QQQ", "VTI", "SPY"):
    assert reserve_asset_profile(ticker) == "equity_etf"
    assert classify_asset(ticker, {"quoteType": "ETF"}) == "EQUITY_ETF"
for ticker in ("BND", "AGG", "SGOV", "SHY", "IEF", "TLT", "GOVT", "BIL"):
    assert reserve_asset_profile(ticker) == "fixed_income_etf"
    assert classify_asset(ticker, {"quoteType": "ETF"}) == "FIXED_INCOME_ETF"
classification_matrix = {
    **{
        ticker: "STANDARD_STOCK"
        for ticker in ("AAPL", "MSFT", "V", "MA", "PYPL", "WM", "JNJ", "XOM", "CAT")
    },
    **{
        ticker: "SPECIAL_FINANCIAL_STOCK"
        for ticker in ("JPM", "AXP", "BRK.B")
    },
    **{
        ticker: "EQUITY_ETF"
        for ticker in ("VOO", "SPY", "QQQ", "VTI", "XLK")
    },
    **{
        ticker: "FIXED_INCOME_ETF"
        for ticker in ("BND", "AGG", "SGOV", "SHY", "IEF", "TLT")
    },
}
for ticker, expected_model in classification_matrix.items():
    quote_type = "ETF" if expected_model.endswith("ETF") else "EQUITY"
    classified = classify_asset(ticker, {"quoteType": quote_type})
    assert classified == expected_model, (ticker, classified)

rng = np.random.default_rng(1)
weights = _bounded_weights(8, .20, rng)
assert np.isclose(weights.sum(), 1)
assert weights.max() <= .2000001

bootstrap_simulations = 100000
bootstrap_periods = 15 * 12
bootstrap_rng = np.random.default_rng(7)
bootstrap_indices = bootstrap_rng.integers(
    0, 700, size=(bootstrap_simulations, bootstrap_periods), dtype=np.int32
)
daily_constant_returns = pd.Series(np.full(800, .0002))
funding_a = simulate_laura_funding(
    daily_constant_returns, .04, .12, cfg, simulations=1000, seed=7,
    bootstrap_indices=bootstrap_indices,
)
funding_b = simulate_laura_funding(
    daily_constant_returns, .04, .12, cfg, simulations=1000, seed=7,
    bootstrap_indices=bootstrap_indices,
)
assert funding_a["total_simulations"] == 100000
assert funding_a["funding_success_probability"] == funding_b["funding_success_probability"]
assert funding_a["value_2033"].equals(funding_b["value_2033"])
assert funding_a["nominal_operating_commitment"] == 500000
assert funding_a["method"] == "Historical 21-trading-day block bootstrap"
assert "all ten" in funding_a["assumptions"].lower() or "ten" in funding_a["assumptions"].lower()
expected_2033 = 300000 * (1 + .0002) ** (252 * 6) + 150000 * (1 + .0002) ** (252 * 5)
assert np.isclose(funding_a["value_2033"].iloc[0], expected_2033)
fallback_normals = np.zeros((bootstrap_simulations, 15))
fallback_result = simulate_laura_funding(
    pd.Series(np.full(100, .001)), .03, .10, cfg, simulations=1000,
    seed=9, standard_normal_paths=fallback_normals,
)
assert fallback_result["method"].startswith("Multivariate-normal")
assert fallback_result["funding_success_probability"] == 1.0

universe = load_local_universe()
assert {"ticker", "company", "sector"}.issubset(universe.columns)
assert len(universe) >= 100

# Historical stress must use the same buy-and-hold path for return and drawdown.
dates = pd.bdate_range("2020-01-01", periods=4)
price_path = pd.DataFrame({"A": [100, 150, 50, 110], "B": [100, 90, 100, 110]}, index=dates)
stress = historical_portfolio_stress(price_path, pd.Series({"A": .5, "B": .5}))
assert stress["available"]
assert np.isclose(stress["portfolio_return"], .1)
assert np.isclose(stress["max_drawdown"], -.375)
assert stress["weight_coverage"] == 1.0

metadata = pd.DataFrame([
    {"ticker": "AAPL", "sector": "Technology", "asset_class": "stock"},
    {"ticker": "TLT", "sector": "Fixed Income", "asset_class": "fixed_income"},
])
shock = hypothetical_stress(pd.Series({"AAPL": .5, "TLT": .5}), metadata, -.3, -.4, -.05)
assert np.isclose(shock["loss_contribution"].sum(), -.225)

sector_table = sector_allocation_table(
    pd.Series({"EQUITY": .70, "BOND": .30}),
    pd.DataFrame([
        {"ticker": "EQUITY", "asset_class": "equity_etf", "sector": "ETF", "sector_weightings": {"technology": .20, "financial_services": .80}},
        {"ticker": "BOND", "asset_class": "fixed_income", "sector": "Bonds", "sector_weightings": {}},
    ]),
    {"Information Technology": .30, "Financials": .13},
    .10,
).set_index("sector")
assert np.isclose(sector_table.loc["Information Technology", "equity_sleeve_weight"], .20)
assert np.isclose(sector_table.loc["Information Technology", "total_portfolio_weight"], .14)
assert sector_table["equity_sleeve_weight"].max() <= 1.0
assert sector_table["total_portfolio_weight"].max() <= 1.0

mixed_sector_metadata = pd.DataFrame([
    {
        "ticker": "VOO",
        "asset_class": "equity_etf",
        "sector": "Unknown",
        "sector_weightings": {"technology": .20, "financial_services": .80},
    },
    {"ticker": "AAPL", "asset_class": "stock", "sector": "Technology"},
    {"ticker": "BOND", "asset_class": "fixed_income", "sector": "Fixed Income"},
])
mixed_sector_weights = pd.Series({"VOO": .40, "AAPL": .30, "BOND": .30})
mixed_sector_table = sector_allocation_table(
    mixed_sector_weights,
    mixed_sector_metadata,
    {"Information Technology": .30, "Financials": .13},
    .10,
).set_index("sector")
assert np.isclose(mixed_sector_table.loc["Information Technology", "total_portfolio_weight"], .38)
assert np.isclose(mixed_sector_table.loc["Financials", "total_portfolio_weight"], .32)
assert np.isclose(mixed_sector_table.loc["Information Technology", "equity_sleeve_weight"], .38 / .70)
assert mixed_sector_table["equity_sleeve_weight"].max() <= 1.0
assert mixed_sector_table["total_portfolio_weight"].max() <= 1.0
assert sector_exposure_coverage(mixed_sector_weights, mixed_sector_metadata)["complete"]
assert normalize_sector_weights({"technology": .50}) == {}
aliased_sector_weights = normalize_sector_weights({
    "technology": .20,
    "information_technology": .80,
})
assert aliased_sector_weights == {"Information Technology": 1.0}
incomplete_metadata = mixed_sector_metadata.copy()
incomplete_metadata.loc[incomplete_metadata["ticker"] == "VOO", "sector_weightings"] = [{"technology": .50}]
incomplete_coverage = sector_exposure_coverage(mixed_sector_weights, incomplete_metadata)
assert not incomplete_coverage["complete"]
assert np.isclose(incomplete_coverage["coverage"], .30 / .70)
assert incomplete_coverage["incomplete_tickers"] == ["VOO"]
incomplete_etf_score = score_security(
    {
        "scoring_model": "EQUITY_ETF",
        "sector_weightings": {"technology": .50},
        "largest_sector_weight": .50,
        "top10_concentration": .30,
        "holdings_count": 100,
        "expense_ratio": .001,
        "max_drawdown": .20,
        "annualized_volatility": .15,
        "average_dollar_volume_30d": 50_000_000,
    },
    cfg,
)
assert not incomplete_etf_score["sector_exposure_complete"]
assert np.isnan(incomplete_etf_score["metric_scores"]["Largest Sector Excess vs S&P 500"])
aliased_sector_score = score_security(
    {
        **etf_metrics,
        "sector_weightings": {
            "technology": .20,
            "information_technology": .80,
        },
    },
    cfg,
)
assert aliased_sector_score["sector_exposure_complete"]
assert np.isclose(
    next(
        row["raw_value"] for row in aliased_sector_score["breakdown"]
        if row["metric"] == "Largest Sector Excess vs S&P 500"
    ),
    1.0 - cfg["optimizer"]["sector_benchmark_weights"]["Information Technology"],
)

# Synthetic optimizer smoke test: no network access.
rng = np.random.default_rng(7)
dates = pd.bdate_range("2021-01-01", periods=900)
prices = pd.DataFrame(index=dates)
for index, ticker in enumerate("ABCDEFGH"):
    daily = rng.normal(.0003 + index * .00002, .010 + index * .0003, size=len(dates))
    prices[ticker] = 100 * np.cumprod(1 + daily)
benchmark_history_series = 100 * np.cumprod(1 + rng.normal(.0001, .008, size=len(dates)))
benchmark_prices = pd.DataFrame({"SPY": benchmark_history_series}, index=dates)
test_cfg = load_config()
test_cfg["optimizer"]["simulations"] = 500
test_cfg["optimizer"]["max_weight"] = .20
optimized = monte_carlo_optimize(prices, test_cfg, benchmark_prices=benchmark_prices)
assert {
    "Minimum Volatility", "Maximum Sharpe", "Maximum Expected Return",
}.issubset(optimized["portfolios"])
assert not optimized["efficient_frontier"].empty
for portfolio in optimized["portfolios"].values():
    assert np.isclose(portfolio["weights"].sum(), 1.0)
    assert portfolio["weights"].min() >= -1e-12
    assert portfolio["weights"].max() <= .2000001
assert optimized["simulations"].shape[0] >= 100
assert optimized["assumptions"]["return_estimate_shrinkage"] == .30
assert optimized["assumptions"]["funding_simulation_method"].startswith("Unavailable:")
assert "all six custom equity-sleeve sector ranges" in optimized["assumptions"]["laura_candidate_selection"]
assert "Laura Goal Portfolio" not in optimized["portfolios"]
benchmark_daily_returns = benchmark_prices["SPY"].pct_change(fill_method=None).reindex(
    optimized["returns"].index
).dropna()
benchmark_mu = float(benchmark_daily_returns.mean() * 252)
for ticker in optimized["expected_returns"].index:
    historical = optimized["historical_expected_returns"][ticker]
    assert np.isclose(
        optimized["expected_returns"][ticker],
        .70 * historical + .30 * benchmark_mu,
    )

laura_tickers = list("ABCDEFGH")
laura_weights = np.array([.15, .10, .15, .15, .15, .10, .10, .10])
laura_exposures = {
    "A": {"Health Care": 1.0},
    "B": {"Construction / Infrastructure": 1.0},
    "C": {"Financial": 1.0},
    "D": {"Technology": 1.0},
    "E": {"Technology": 1.0},
    "F": {"Energy": 1.0},
    "G": {"Others": 1.0},
    "H": {"Others": 1.0},
}
assert _laura_sector_constraints_satisfied(
    laura_weights,
    laura_tickers,
    {ticker: "stock" for ticker in laura_tickers},
    laura_exposures,
)
outside_laura_ranges = laura_weights.copy()
outside_laura_ranges[0] = .21
outside_laura_ranges[6] = .04
assert not _laura_sector_constraints_satisfied(
    outside_laura_ranges,
    laura_tickers,
    {ticker: "stock" for ticker in laura_tickers},
    laura_exposures,
)
assert not _laura_sector_constraints_satisfied(
    laura_weights,
    laura_tickers,
    {ticker: "stock" for ticker in laura_tickers},
    {**laura_exposures, "H": {}},
)

sample_returns = pd.DataFrame({"A": [.01, -.02, .03], "B": [.02, .01, -.01]})
assert np.allclose(annualized_expected_returns(sample_returns), sample_returns.mean() * 252)
covariance = annualized_covariance(sample_returns, shrinkage=0.0)
weights_test = np.array([.4, .6])
mu_test = annualized_expected_returns(sample_returns).to_numpy()
expected_return, volatility, sharpe = portfolio_stats(weights_test, mu_test, covariance.to_numpy(), .02)
assert np.isclose(expected_return, weights_test @ mu_test)
assert np.isclose(volatility, np.sqrt(weights_test @ covariance.to_numpy() @ weights_test))
assert np.isclose(sharpe, (expected_return - .02) / volatility)
shrunken, targets = shrink_expected_returns(
    pd.Series({"A": .20, "B": .10, "BOND": .04}),
    {"A": "stock", "B": "stock", "BOND": "fixed_income"},
    .30,
    .08,
)
assert np.isclose(shrunken["A"], .164)
assert np.isclose(shrunken["B"], .094)
assert np.isclose(shrunken["BOND"], .04)
assert np.isclose(targets["BOND"], .04)

sector_prices = pd.DataFrame(
    {
        ticker: 100 * np.cumprod(1 + rng.normal(.0002, .008, size=400))
        for ticker in [f"ETF{i}" for i in range(5)] + [f"FI{i}" for i in range(5)]
    },
    index=pd.bdate_range("2022-01-03", periods=400),
)
sector_reference = normalize_sector_weights(cfg["optimizer"]["sector_benchmark_weights"])
sector_metadata = pd.DataFrame([
    {
        "ticker": ticker,
        "asset_class": "equity_etf" if ticker.startswith("ETF") else "fixed_income",
        "sector": "Unknown",
        "sector_weightings": sector_reference if ticker.startswith("ETF") else {},
    }
    for ticker in sector_prices.columns
])
sector_cfg = load_config()
sector_cfg["optimizer"]["simulations"] = 500
sector_cfg["optimizer"]["max_weight"] = .20
sector_result = monte_carlo_optimize(sector_prices, sector_cfg, sector_metadata)
assert len(sector_result["constraints"]["sector_ranges"]) == 11
for portfolio in sector_result["portfolios"].values():
    assert np.isclose(portfolio["weights"].sum(), 1)
    assert portfolio["weights"].max() <= .2000001
    allocation = sector_allocation_table(
        portfolio["weights"],
        sector_metadata,
        sector_reference,
        sector_cfg["optimizer"]["sector_tolerance"],
    )
    if allocation["equity_sleeve_weight"].notna().any():
        assert (
            allocation["equity_sleeve_weight"].between(
                allocation["target_low"] - 1e-8,
                allocation["target_high"] + 1e-8,
            )
        ).all()

incomplete_sector_metadata = sector_metadata.copy()
incomplete_sector_metadata.loc[
    incomplete_sector_metadata["ticker"] == "ETF0", "sector_weightings"
] = [{}]
incomplete_sector_result = monte_carlo_optimize(
    sector_prices,
    sector_cfg,
    incomplete_sector_metadata,
)
assert incomplete_sector_result["constraints"]["excluded_for_sector_data"] == ["ETF0"]
assert "ETF0" not in incomplete_sector_result["expected_returns"].index

concentrated_metadata = sector_metadata.copy()
concentrated_metadata.loc[
    concentrated_metadata["asset_class"] == "equity_etf", "sector_weightings"
] = [{"technology": 1.0}] * 5
infeasible_sector_cfg = load_config()
infeasible_sector_cfg["optimizer"].update({
    "simulations": 500,
    "max_weight": .20,
    "minimum_equity_weight": .40,
    "maximum_fixed_income_weight": .60,
})
try:
    monte_carlo_optimize(sector_prices, infeasible_sector_cfg, concentrated_metadata)
except ValueError as error:
    assert "Sector constraint" in str(error)
else:
    raise AssertionError("An infeasible sector constraint was not rejected before sampling.")

flat_prices = pd.DataFrame(
    100.0,
    index=pd.bdate_range("2022-01-03", periods=400),
    columns=list("ABCDE"),
)
flat_cfg = load_config()
flat_cfg["optimizer"]["simulations"] = 500
flat_cfg["optimizer"]["max_weight"] = .20
flat_result = monte_carlo_optimize(flat_prices, flat_cfg)
assert np.isclose(flat_result["portfolios"]["Maximum Sharpe"]["weights"].sum(), 1.0)
assert np.isnan(flat_result["portfolios"]["Maximum Sharpe"]["sharpe"])

benchmark_dates = pd.bdate_range("2024-01-02", periods=5)
candidate_history = pd.DataFrame(
    {"SPY": [100, 101, 102, 101, 104], "A": [100, 99, 101, 103, 105]},
    index=benchmark_dates.tz_localize("America/New_York"),
)
benchmark_history = pd.DataFrame(
    {"SPY": [100, 101, 102, 101, 104], "VOO": [100, 100.5, 101, 102, 103]},
    index=benchmark_dates,
)
benchmark_comparison = portfolio_benchmark_comparison(
    candidate_history, pd.Series({"SPY": .5, "A": .5}), benchmark_history, .02
)
assert set(benchmark_comparison["benchmark"]) == {"SPY", "VOO"}
assert benchmark_comparison["overlap_observations"].eq(4).all()
metadata = pd.DataFrame([
    {
        "ticker": ticker,
        "asset_class": "fixed_income" if ticker in {"G", "H"} else "stock",
        "sector": "Bonds" if ticker in {"G", "H"} else f"Sector {ticker}",
    }
    for ticker in prices.columns
])
test_cfg["optimizer"]["minimum_fixed_income_weight"] = .20
test_cfg["optimizer"]["maximum_fixed_income_weight"] = .50
manual_weights = pd.Series(.125, index=prices.columns)
constrained = monte_carlo_optimize(prices, test_cfg, metadata, manual_weights)
assert "Current Manual Portfolio" in constrained["portfolios"]
assert np.isclose(constrained["portfolios"]["Current Manual Portfolio"]["weights"].loc[["G", "H"]].sum(), .25)
filtered_metadata = metadata.copy()
filtered_metadata["average_dollar_volume_30d"] = 1e8
filtered_metadata.loc[filtered_metadata["ticker"] == "H", "average_dollar_volume_30d"] = 0
test_cfg["optimizer"]["liquidity_requirement"] = 5_000_000
filtered = monte_carlo_optimize(prices, test_cfg, filtered_metadata, manual_weights)
assert "Maximum Sharpe" in filtered["portfolios"]
assert "Current Manual Portfolio" not in filtered["portfolios"]
assert "H" in filtered["constraints"]["excluded_for_liquidity"]
assert filtered["manual_weights_error"]

print("All offline core tests passed.")

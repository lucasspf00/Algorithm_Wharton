import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, merge_config_defaults
from src.data import (
    _expense_ratio_fraction,
    _security_profile,
    detect_asset_type,
    load_local_universe,
    normalize_ticker,
    reserve_asset_profile,
)
import src.data as data_module
from src.optimizer import (
    _bounded_weights,
    deterministic_reserve_requirement,
    laura_value_2033,
    monte_carlo_optimize,
    simulate_2033_distribution,
)
from src.scoring import normalize_lower_better, normalize_piecewise, normalize_symmetric, score_security
from src.stress import historical_portfolio_stress, hypothetical_stress

cfg = load_config()
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

stock_metrics = {
    "analysis_profile": "operating_company",
    "revenue_growth_1y": .15, "revenue_growth_3y": .15, "revenue_growth_5y": .10,
    "eps_growth_1y": .20, "eps_growth_3y": .20, "eps_growth_5y": .15,
    "fcf_growth_1y": .20, "fcf_growth_3y": .20, "fcf_growth_5y": .15,
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
assert stock["data_confidence"] == 100
sparse_stock = score_security({"analysis_profile": "operating_company", "trailing_pe": 20}, cfg)
assert sparse_stock["data_confidence"] < stock["data_confidence"]
negative_fcf = score_security({
    **stock_metrics, "free_cash_flow": -1, "net_debt": 100, "net_debt_to_fcf": np.nan,
}, cfg)
assert negative_fcf["metric_scores"]["net_debt_to_fcf"] == -100

financial_metrics = {
    "analysis_profile": "financial_company",
    "revenue_growth_1y": .05, "revenue_growth_3y": .04, "revenue_growth_5y": .03,
    "eps_growth_1y": .08, "eps_growth_3y": .06, "eps_growth_5y": .05,
    "net_income_growth_1y": .06, "net_income_growth_3y": .05, "net_income_growth_5y": .04,
    "return_on_equity": .12, "return_on_assets": .01, "net_margin": .20,
    "earnings_consistency": .90, "trailing_pe": 15, "price_to_book": 1.5,
    "earnings_yield": 1 / 15, "annualized_volatility": .20, "max_drawdown": .30, "beta": 1,
}
financial = score_security(financial_metrics, cfg)
assert financial["analysis_profile"] == "financial_stock"
assert financial["category_labels"]["quality"] == "Financial Quality"
assert -100 <= financial["main_score"] <= 100
assert financial["data_confidence"] == 100

etf_metrics = {
    "analysis_profile": "equity_etf",
    "total_return_1y": .12, "total_return_3y": .10, "total_return_5y": .09,
    "expense_ratio": .0009, "holdings_count": 500, "top10_concentration": .25,
    "largest_sector_weight": .20, "average_dollar_volume_30d": 100_000_000,
    "annualized_volatility": .18, "max_drawdown": .25, "beta": 1.0,
}
etf = score_security(etf_metrics, cfg)
assert etf["analysis_profile"] == "equity_etf"
assert np.isclose(etf["main_score"], .35 * etf["growth_score"] + .35 * etf["quality_score"] + .30 * etf["valuation_score"])
assert np.isfinite(etf["risk_score"])
assert etf["data_confidence"] == 100
assert -100 <= etf["main_score"] <= 100

fixed_metrics = {
    "analysis_profile": "fixed_income_etf",
    "total_return_1y": .04, "total_return_3y": .02, "total_return_5y": .03,
    "effective_duration": 4.25, "weighted_credit_score": 75, "expense_ratio": .001,
    "average_dollar_volume_30d": 100_000_000,
    "annualized_volatility": .06, "max_drawdown": .08, "beta": .1,
}
fixed = score_security(fixed_metrics, cfg)
assert fixed["analysis_profile"] == "fixed_income_etf"
assert np.isclose(fixed["main_score"], .4 * fixed["quality_score"] + .3 * fixed["growth_score"] + .3 * fixed["valuation_score"])
assert np.isfinite(fixed["risk_score"])
assert fixed["data_confidence"] < 100  # Comparable Treasury yield spread is unavailable.
assert -100 <= fixed["main_score"] <= 100

assert normalize_ticker("BRK.B") == "BRK-B"
assert normalize_ticker("BF.B") == "BF-B"
assert np.isclose(_expense_ratio_fraction({"annualReportExpenseRatio": .03}), .0003)
assert np.isclose(_expense_ratio_fraction({"annualReportExpenseRatio": .0003}), .0003)
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
assert _security_profile("SPY", {"quoteType": "ETF", "category": "Large Blend"})["analysis_profile"] == "equity_etf"
assert _security_profile("TLT", {"quoteType": "ETF", "category": "Intermediate Government"})["analysis_profile"] == "fixed_income_etf"
assert _security_profile("BRK-B", {"quoteType": "EQUITY", "longName": "Berkshire Hathaway Inc."})["analysis_profile"] == "financial_conglomerate"
assert detect_asset_type("JPM", {"quoteType": "EQUITY", "sector": "Financial Services"}) == "financial_stock"
for ticker in ("JPM", "BAC"):
    assert _security_profile(ticker, {"quoteType": "EQUITY"})["analysis_profile"] == "financial_company"
for ticker in ("VOO", "QQQ", "VTI", "SPY"):
    assert reserve_asset_profile(ticker) == "equity_etf"
for ticker in ("BND", "AGG", "SGOV", "SHY", "IEF", "TLT", "GOVT", "BIL"):
    assert reserve_asset_profile(ticker) == "fixed_income_etf"

rng = np.random.default_rng(1)
weights = _bounded_weights(8, .20, rng)
assert np.isclose(weights.sum(), 1)
assert weights.max() <= .2000001

assert 400000 < deterministic_reserve_requirement(50000, 10, .04) < 500000
assert laura_value_2033(0.0) == 450000
assert len(simulate_2033_distribution(.04, .12, simulations=1000, seed=7)) == 1000

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

metadata = pd.DataFrame([
    {"ticker": "AAPL", "sector": "Technology", "asset_class": "stock"},
    {"ticker": "TLT", "sector": "Fixed Income", "asset_class": "fixed_income"},
])
shock = hypothetical_stress(pd.Series({"AAPL": .5, "TLT": .5}), metadata, -.3, -.4, -.05)
assert np.isclose(shock["loss_contribution"].sum(), -.225)

# Synthetic optimizer smoke test: no network access.
rng = np.random.default_rng(7)
dates = pd.bdate_range("2021-01-01", periods=900)
prices = pd.DataFrame(index=dates)
for index, ticker in enumerate("ABCDEFGH"):
    daily = rng.normal(.0003 + index * .00002, .010 + index * .0003, size=len(dates))
    prices[ticker] = 100 * np.cumprod(1 + daily)
test_cfg = load_config()
test_cfg["optimizer"]["simulations"] = 500
test_cfg["optimizer"]["max_weight"] = .20
optimized = monte_carlo_optimize(prices, test_cfg)
assert {
    "Minimum Volatility", "Maximum Sharpe", "Maximum Expected Return",
    "Target Return Portfolio", "Laura Goal Portfolio",
}.issubset(optimized["portfolios"])
for portfolio in optimized["portfolios"].values():
    assert np.isclose(portfolio["weights"].sum(), 1.0)
    assert portfolio["weights"].min() >= -1e-12
    assert portfolio["weights"].max() <= .2000001
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
assert "User-Defined Weights" in constrained["portfolios"]
assert np.isclose(constrained["portfolios"]["User-Defined Weights"]["weights"].loc[["G", "H"]].sum(), .25)
filtered_metadata = metadata.copy()
filtered_metadata["average_dollar_volume_30d"] = 1e8
filtered_metadata.loc[filtered_metadata["ticker"] == "H", "average_dollar_volume_30d"] = 0
test_cfg["optimizer"]["liquidity_requirement"] = 5_000_000
filtered = monte_carlo_optimize(prices, test_cfg, filtered_metadata, manual_weights)
assert "Maximum Sharpe" in filtered["portfolios"]
assert "User-Defined Weights" not in filtered["portfolios"]
assert "H" in filtered["constraints"]["excluded_for_liquidity"]
assert filtered["manual_weights_error"]

print("All offline core tests passed.")

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.scoring import normalize_symmetric, score_security
from src.optimizer import _bounded_weights, deterministic_reserve_requirement, laura_value_2033, simulate_2033_distribution
from src.data import load_local_universe, _security_profile, normalize_ticker, detect_asset_type
from src.stress import hypothetical_stress
from src.reserve import (
    deterministic_reserve_pv,
    normalize_reserve_weights,
    required_reserves,
)

cfg = load_config()

assert normalize_symmetric(0.30, 0.30) == 100
assert normalize_symmetric(-0.30, 0.30) == -100
assert normalize_symmetric(1.20, 0.30) == 100

metrics = {
    "revenue_growth_1y": 0.15, "revenue_growth_3y": 0.15, "revenue_growth_5y": 0.10,
    "eps_growth_1y": 0.20, "eps_growth_3y": 0.20, "eps_growth_5y": 0.15,
    "fcf_growth_1y": 0.20, "fcf_growth_3y": 0.20, "fcf_growth_5y": 0.15,
    "operating_margin": 0.20, "net_margin": 0.15, "return_on_equity": 0.20,
    "current_ratio": 1.5, "debt_to_equity": 0.7,
    "trailing_pe": 20, "forward_pe": 18, "price_to_sales": 4, "price_to_book": 4, "ev_to_ebitda": 12,
    "annualized_volatility": 0.22, "max_drawdown": 0.25, "downside_deviation": 0.15, "beta": 1.0,
}
scored = score_security(metrics, cfg)
assert -100 <= scored["fundamental_score"] <= 100
assert 0 <= scored["display_rating"] <= 100
assert scored["data_confidence"] > 90

assert _security_profile("SPY", {"quoteType": "ETF", "category": "Large Blend"})["analysis_profile"] == "equity_etf"
assert _security_profile("TLT", {"quoteType": "ETF", "category": "Intermediate Government"})["analysis_profile"] == "fixed_income_etf"
assert _security_profile("BRK-B", {"quoteType": "EQUITY", "longName": "Berkshire Hathaway Inc."})["analysis_profile"] == "financial_conglomerate"
assert normalize_ticker("BRK.B") == "BRK-B"
assert detect_asset_type("JPM", {"quoteType": "EQUITY", "sector": "Financial Services"}) == "financial_stock"

etf_scored = score_security({
    "analysis_profile": "equity_etf",
    "total_return_1y": 0.12, "total_return_3y": 0.10, "total_return_5y": 0.09,
    "expense_ratio": 0.0009, "holdings_count": 500, "portfolio_pe": 22, "portfolio_pb": 3.5,
    "distribution_yield": 0.015, "annualized_volatility": 0.18, "max_drawdown": 0.25,
    "downside_deviation": 0.12, "beta": 1.0,
}, cfg)
assert etf_scored["analysis_profile"] == "equity_etf"
assert etf_scored["data_confidence"] > 80

brk_scored = score_security({
    "analysis_profile": "financial_conglomerate", "revenue_growth_1y": 0.05,
    "return_on_equity": 0.12, "current_ratio": 1.2, "debt_to_equity": 0.8,
    "price_to_book": 1.6, "trailing_pe": 22, "annualized_volatility": 0.18,
    "max_drawdown": 0.25, "downside_deviation": 0.12, "beta": 0.9,
}, cfg)
assert brk_scored["analysis_profile"] == "financial_stock"
assert np.isfinite(brk_scored["fundamental_score"])

financial_scored = score_security({
    "analysis_profile": "financial_company",
    "revenue_growth_1y": .05, "eps_growth_1y": .08, "net_income_growth_1y": .06,
    "return_on_equity": .12, "return_on_assets": .01, "net_margin": .20,
    "earnings_consistency": .90, "trailing_pe": 15, "forward_pe": 14,
    "price_to_book": 1.5, "earnings_yield": 1 / 15,
    "annualized_volatility": .20, "max_drawdown": .30, "downside_deviation": .14, "beta": 1.0,
}, cfg)
assert financial_scored["analysis_profile"] == "financial_stock"

fixed_scored = score_security({
    "analysis_profile": "fixed_income_etf", "total_return_1y": .04,
    "total_return_3y": .02, "total_return_5y": .03, "expense_ratio": .0015,
    "holdings_count": 500, "annualized_volatility": .06, "max_drawdown": .08,
    "downside_deviation": .04, "beta": .1,
}, cfg)
assert fixed_scored["analysis_profile"] == "fixed_income_etf"

rng = np.random.default_rng(1)
w = _bounded_weights(8, 0.20, rng)
assert np.isclose(w.sum(), 1)
assert w.max() <= 0.2000001

reserve = deterministic_reserve_requirement(50000, 10, 0.04)
assert 400000 < reserve < 500000
assert laura_value_2033(0.0) == 450000
assert len(simulate_2033_distribution(0.04, 0.12, simulations=1000, seed=7)) == 1000
reserve_assets = pd.DataFrame([
    {"asset": "SGOV", "weight": 0.5, "yield": 0.04},
    {"asset": "BIL", "weight": 0.5, "yield": 0.04},
])
assert np.isclose(normalize_reserve_weights(reserve_assets)["weight"].sum(), 1.0)
assert np.isclose(deterministic_reserve_pv(0.04), reserve)
reserve_results = required_reserves(0.04, 0.05, targets=(0.95,), simulations=100000, seed=7)
assert reserve_results.loc[0, "required_reserve"] >= 0
assert reserve_results.loc[0, "simulated_success"] >= 0.95

u = load_local_universe()
assert {"ticker", "company", "sector"}.issubset(u.columns)
assert len(u) >= 100

weights = pd.Series({"AAPL": 0.5, "TLT": 0.5})
meta = pd.DataFrame([
    {"ticker":"AAPL","sector":"Information Technology","asset_class":"stock"},
    {"ticker":"TLT","sector":"Unknown","asset_class":"fixed_income"},
])
out = hypothetical_stress(weights, meta, -0.30, -0.40, -0.05)
assert np.isclose(out["loss_contribution"].sum(), -0.225)

print("All offline core tests passed.")

# Synthetic end-to-end optimizer test (no internet required).
from src.optimizer import monte_carlo_optimize
from src.stress import historical_portfolio_stress

rng2 = np.random.default_rng(7)
dates = pd.bdate_range("2021-01-01", periods=900)
synthetic = pd.DataFrame(index=dates)
for i, t in enumerate(["A","B","C","D","E","F","G","H"]):
    daily = rng2.normal(0.0003 + i*0.00002, 0.010 + i*0.0003, size=len(dates))
    synthetic[t] = 100 * np.cumprod(1 + daily)

test_cfg = load_config()
test_cfg["optimizer"]["simulations"] = 500
test_cfg["optimizer"]["max_weight"] = 0.20
opt = monte_carlo_optimize(synthetic, test_cfg)
assert set(opt["portfolios"]) == {
    "Minimum Volatility", "Maximum Sharpe", "Maximum Expected Return", "Laura Portfolio"
}
laura_weights = opt["portfolios"]["Laura Portfolio"]["weights"]
assert np.isclose(laura_weights.sum(), 1.0)
assert laura_weights.min() >= -1e-12
assert laura_weights.max() <= 0.2000001
for p in opt["portfolios"].values():
    assert np.isclose(p["weights"].sum(), 1.0)
    assert p["weights"].max() <= 0.2000001

hist = historical_portfolio_stress(synthetic.iloc[:120], opt["portfolios"]["Maximum Sharpe"]["weights"])
assert hist["available"]
assert 0 < hist["weight_coverage"] <= 1

print("Synthetic optimizer/stress tests passed.")

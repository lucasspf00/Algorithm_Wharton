import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.scoring import normalize_symmetric, score_security
from src.optimizer import _bounded_weights, deterministic_reserve_requirement, laura_value_2033
from src.data import load_local_universe
from src.stress import hypothetical_stress

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

rng = np.random.default_rng(1)
w = _bounded_weights(8, 0.20, rng)
assert np.isclose(w.sum(), 1)
assert w.max() <= 0.2000001

reserve = deterministic_reserve_requirement(50000, 10, 0.04)
assert 400000 < reserve < 500000
assert laura_value_2033(0.0) == 450000

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
assert set(opt["portfolios"]) == {"Minimum Volatility", "Maximum Sharpe", "Maximum Expected Return"}
for p in opt["portfolios"].values():
    assert np.isclose(p["weights"].sum(), 1.0)
    assert p["weights"].max() <= 0.2000001

hist = historical_portfolio_stress(synthetic.iloc[:120], opt["portfolios"]["Maximum Sharpe"]["weights"])
assert hist["available"]
assert 0 < hist["weight_coverage"] <= 1

print("Synthetic optimizer/stress tests passed.")

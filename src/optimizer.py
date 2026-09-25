from __future__ import annotations
import numpy as np
import pandas as pd


def prepare_returns(price_df: pd.DataFrame, lookback_years: int = 5) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    prices = price_df.copy().sort_index()
    cutoff = prices.index.max() - pd.DateOffset(years=int(lookback_years))
    prices = prices.loc[prices.index >= cutoff]
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    # Use common observations for transparent covariance calculations.
    returns = returns.dropna(how="any")
    return returns


def annualized_expected_returns(returns: pd.DataFrame) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    # Geometric/log-return estimate. Historical estimate, not a forecast.
    log_r = np.log1p(returns.clip(lower=-0.999999))
    return np.expm1(log_r.mean() * 252.0)


def annualized_covariance(returns: pd.DataFrame, shrinkage: float = 0.20) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    sample = returns.cov() * 252.0
    a = float(np.clip(shrinkage, 0.0, 1.0))
    diagonal = pd.DataFrame(np.diag(np.diag(sample.values)), index=sample.index, columns=sample.columns)
    return (1.0 - a) * sample + a * diagonal


def _bounded_weights(n: int, cap: float, rng: np.random.Generator, allow_zeros: bool = True) -> np.ndarray:
    if n < 1:
        raise ValueError("At least one holding is required.")
    if cap <= 0 or n * cap < 1.0 - 1e-12:
        raise ValueError(f"Constraints are infeasible: {n} holdings × {cap:.1%} max weight is below 100%.")
    minimum_names = int(np.ceil(1.0 / cap - 1e-12))
    k = rng.integers(minimum_names, n + 1) if allow_zeros and n > minimum_names else n
    chosen = rng.choice(n, size=k, replace=False)
    scores = rng.exponential(1.0, size=k) + 1e-12
    active = np.ones(k, dtype=bool)
    w_sub = np.zeros(k)
    remaining = 1.0
    while active.any():
        idx = np.where(active)[0]
        alloc = remaining * scores[idx] / scores[idx].sum()
        too_big = alloc > cap + 1e-12
        if not too_big.any():
            w_sub[idx] = alloc
            remaining = 0.0
            break
        capped_idx = idx[too_big]
        w_sub[capped_idx] = cap
        active[capped_idx] = False
        remaining = 1.0 - w_sub.sum()
        if remaining < -1e-10:
            raise RuntimeError("Weight generator failed.")
    w = np.zeros(n)
    w[chosen] = w_sub
    # Small numerical correction.
    w /= w.sum()
    return w


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free_rate: float) -> tuple[float, float, float]:
    ret = float(weights @ mu)
    var = float(weights @ cov @ weights)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def monte_carlo_optimize(price_df: pd.DataFrame, cfg: dict) -> dict:
    settings = cfg["optimizer"]
    laura_settings = cfg.get("laura", {})
    returns = prepare_returns(price_df, int(settings["lookback_years"]))
    if returns.shape[0] < 126:
        raise ValueError("Not enough overlapping price history for optimization. Try different holdings or a shorter lookback.")
    tickers = list(returns.columns)
    n = len(tickers)
    cap = float(settings["max_weight"])
    if n * cap < 1.0 - 1e-12:
        need = int(np.ceil(1.0 / cap))
        raise ValueError(f"Portfolio is infeasible. With a {cap:.0%} max holding weight you need at least {need} usable holdings; only {n} have common data.")

    mu_s = annualized_expected_returns(returns)
    cov_df = annualized_covariance(returns, float(settings.get("covariance_shrinkage", 0.20)))
    mu = mu_s.values
    cov = cov_df.values
    rf = float(settings["risk_free_rate"])
    sims = int(settings["simulations"])
    rng = np.random.default_rng(42)

    rows = []
    weight_rows = []
    for i in range(sims):
        w = _bounded_weights(n, cap, rng, allow_zeros=True)
        ret, vol, sharpe = portfolio_stats(w, mu, cov, rf)
        rows.append((ret, vol, sharpe))
        weight_rows.append(w)
    stats = pd.DataFrame(rows, columns=["expected_return", "volatility", "sharpe"])
    W = np.vstack(weight_rows)

    min_vol_i = int(stats["volatility"].idxmin())
    max_sharpe_i = int(stats["sharpe"].replace([np.inf, -np.inf], np.nan).idxmax())
    max_return_i = int(stats["expected_return"].idxmax())

    def pack(i: int) -> dict:
        return {
            "weights": pd.Series(W[i], index=tickers),
            "expected_return": float(stats.loc[i, "expected_return"]),
            "volatility": float(stats.loc[i, "volatility"]),
            "sharpe": float(stats.loc[i, "sharpe"]),
        }

    # Laura Goal Portfolio balances the preferred Maximum Sharpe portfolio with
    # Minimum Volatility while preserving long-only, fully invested weights.
    sharpe_blend = float(laura_settings.get("portfolio_blend", {}).get("maximum_sharpe", 0.50))
    min_vol_blend = float(laura_settings.get("portfolio_blend", {}).get("minimum_volatility", 0.50))
    blend_total = sharpe_blend + min_vol_blend
    if blend_total <= 0:
        raise ValueError("Laura Goal Portfolio blend weights must have a positive total.")
    sharpe_blend /= blend_total
    min_vol_blend /= blend_total
    laura_w = (W[max_sharpe_i] * sharpe_blend) + (W[min_vol_i] * min_vol_blend)
    laura_ret, laura_vol, laura_sharpe = portfolio_stats(laura_w, mu, cov, rf)
    laura_portfolio = {
        "weights": pd.Series(laura_w, index=tickers),
        "expected_return": laura_ret,
        "volatility": laura_vol,
        "sharpe": laura_sharpe,
        "construction": (
            f"{sharpe_blend:.1%} Maximum Sharpe + "
            f"{min_vol_blend:.1%} Minimum Volatility; "
            f"reserve objective reviewed separately at "
            f"{float(laura_settings.get('goal_portfolio_reserve_target', 0.995)):.1%}"
        ),
    }

    return {
        "returns": returns,
        "expected_returns": mu_s,
        "covariance": cov_df,
        "correlation": returns.corr(),
        "simulations": stats,
        "portfolios": {
            "Minimum Volatility": pack(min_vol_i),
            "Maximum Sharpe": pack(max_sharpe_i),
            "Maximum Expected Return": pack(max_return_i),
            "Laura Goal Portfolio": laura_portfolio,
        },
    }


def laura_value_2033(expected_return: float, contribution_2027: float = 300000, contribution_2028: float = 150000) -> float:
    """Expected-value projection to beginning 2033 using one annual return assumption.

    2027 contribution compounds for 6 years; 2028 contribution for 5 years.
    This is an assumption-driven projection, not a guaranteed result.
    """
    r = float(expected_return)
    if r <= -1:
        return 0.0
    return contribution_2027 * (1 + r) ** 6 + contribution_2028 * (1 + r) ** 5


def simulate_2033_distribution(
    expected_return: float,
    volatility: float,
    contribution_2027: float = 300000,
    contribution_2028: float = 150000,
    simulations: int = 100000,
    seed: int = 20260924,
) -> pd.Series:
    """Simulate the 2033 value from Laura's two starting cash flows.

    This is a model distribution, not a funding-confidence calculation.
    """
    if simulations < 1 or volatility < 0 or expected_return <= -1:
        raise ValueError("Invalid return, volatility, or simulation settings.")
    rng = np.random.default_rng(seed)
    drift = expected_return - 0.5 * volatility**2
    paths_2027 = np.exp(rng.normal(drift, volatility, (simulations, 6)).sum(axis=1))
    paths_2028 = np.exp(rng.normal(drift, volatility, (simulations, 5)).sum(axis=1))
    return pd.Series(contribution_2027 * paths_2027 + contribution_2028 * paths_2028)


def deterministic_reserve_requirement(payment: float = 50000, count: int = 10, annual_yield: float = 0.04) -> float:
    y = float(annual_yield)
    if y <= -1:
        raise ValueError("Reserve yield must be greater than -100%.")
    return float(sum(float(payment) / ((1 + y) ** t) for t in range(int(count))))

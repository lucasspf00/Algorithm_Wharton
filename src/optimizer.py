from __future__ import annotations
import math
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


def monte_carlo_optimize(
    price_df: pd.DataFrame,
    cfg: dict,
    metadata: pd.DataFrame | None = None,
    manual_weights: pd.Series | None = None,
) -> dict:
    settings = cfg["optimizer"]
    laura_settings = cfg.get("laura", {})
    raw_tickers = list(price_df.columns) if price_df is not None else []
    returns = prepare_returns(price_df, int(settings["lookback_years"]))
    excluded_for_liquidity = []
    if metadata is not None and not metadata.empty and "ticker" in metadata and "average_dollar_volume_30d" in metadata:
        liquidity = metadata.drop_duplicates("ticker").set_index("ticker")["average_dollar_volume_30d"]
        minimum_liquidity = float(settings.get("liquidity_requirement", 0))
        eligible = [
            ticker for ticker in returns.columns
            if _finite_number(liquidity.get(ticker)) and float(liquidity[ticker]) >= minimum_liquidity
        ]
        excluded_for_liquidity = [ticker for ticker in returns.columns if ticker not in eligible]
        returns = returns[eligible]
    excluded_for_history = [ticker for ticker in raw_tickers if ticker not in returns.columns and ticker not in excluded_for_liquidity]
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

    settings_metadata = cfg["optimizer"]
    asset_types, sectors = {}, {}
    if metadata is not None and not metadata.empty and "ticker" in metadata:
        lookup = metadata.drop_duplicates("ticker").set_index("ticker")
        asset_types = lookup.get("asset_class", pd.Series(dtype=str)).to_dict()
        sectors = lookup.get("sector", pd.Series(dtype=str)).to_dict()
    equity_idx = np.asarray([
        i for i, ticker in enumerate(tickers)
        if asset_types.get(ticker, "stock") != "fixed_income"
    ], dtype=int)
    fixed_idx = np.asarray([
        i for i, ticker in enumerate(tickers)
        if asset_types.get(ticker) == "fixed_income"
    ], dtype=int)
    min_equity = float(settings_metadata.get("minimum_equity_weight", 0))
    max_equity = float(settings_metadata.get("maximum_equity_weight", 1))
    min_fixed = float(settings_metadata.get("minimum_fixed_income_weight", 0))
    max_fixed = float(settings_metadata.get("maximum_fixed_income_weight", 1))
    if min_equity + min_fixed > 1 + 1e-9 or max_equity + max_fixed < 1 - 1e-9:
        raise ValueError("Equity and fixed-income allocation bounds cannot sum to a fully invested portfolio.")
    if min_equity > len(equity_idx) * cap + 1e-9:
        raise ValueError("The selected holdings and max weight cannot satisfy the minimum equity allocation.")
    if min_fixed > len(fixed_idx) * cap + 1e-9:
        raise ValueError("The selected holdings and max weight cannot satisfy the minimum fixed-income allocation.")
    sector_counts = pd.Series([sectors.get(t, "Unknown") for t in tickers]).value_counts(normalize=True).to_dict()
    sector_caps = {
        name: min(1.0, float(base) + float(settings_metadata.get("sector_tolerance", .10)))
        for name, base in sector_counts.items()
    }

    def valid_weights(w: np.ndarray) -> bool:
        equity_weight = float(w[equity_idx].sum()) if len(equity_idx) else 0.0
        fixed_weight = float(w[fixed_idx].sum()) if len(fixed_idx) else 0.0
        if not min_equity - 1e-9 <= equity_weight <= max_equity + 1e-9:
            return False
        if not min_fixed - 1e-9 <= fixed_weight <= max_fixed + 1e-9:
            return False
        for name, cap_weight in sector_caps.items():
            sector_weight = sum(w[i] for i, ticker in enumerate(tickers) if sectors.get(ticker, "Unknown") == name)
            if sector_weight > cap_weight + 1e-9:
                return False
        return True

    rows = []
    weight_rows = []
    attempts = 0
    max_attempts = max(sims * 100, 10000)
    while len(weight_rows) < sims and attempts < max_attempts:
        attempts += 1
        w = _bounded_weights(n, cap, rng, allow_zeros=True)
        if not valid_weights(w):
            continue
        ret, vol, sharpe = portfolio_stats(w, mu, cov, rf)
        rows.append((ret, vol, sharpe))
        weight_rows.append(w)
    if len(weight_rows) < max(100, int(sims * .1)):
        raise ValueError(
            "Allocation constraints are infeasible or too restrictive for the available holdings. "
            "Adjust equity/fixed-income bounds, sector tolerance, or max holding weight."
        )
    sims = len(weight_rows)
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

    portfolios = {
        "Minimum Volatility": pack(min_vol_i),
        "Maximum Sharpe": pack(max_sharpe_i),
        "Maximum Expected Return": pack(max_return_i),
    }
    required_return = float(settings.get("target_return", 0.0))
    target_candidates = np.flatnonzero(stats["expected_return"].to_numpy() >= required_return)
    target_met = len(target_candidates) > 0
    if target_met:
        target_i = int(target_candidates[np.argmin(stats["volatility"].to_numpy()[target_candidates])])
    else:
        target_i = max_return_i
    target_portfolio = pack(target_i)
    target_portfolio["target_return"] = required_return
    target_portfolio["target_met"] = target_met
    portfolios["Target Return Portfolio"] = target_portfolio
    manual_weights_error = None
    if manual_weights is not None:
        manual = pd.to_numeric(manual_weights.reindex(tickers), errors="coerce").fillna(0).to_numpy(dtype=float)
        if not np.isfinite(manual).all() or (manual < -1e-9).any() or not np.isclose(manual.sum(), 1, atol=1e-6):
            manual_weights_error = "Manual weights no longer sum to 100% over securities eligible for optimization."
        elif manual.max() > cap + 1e-9 or not valid_weights(manual):
            manual_weights_error = "Manual weights violate a holding, asset-class, or sector constraint."
        else:
            ret, vol, sharpe = portfolio_stats(manual, mu, cov, rf)
            portfolios["User-Defined Weights"] = {
                "weights": pd.Series(manual, index=tickers),
                "expected_return": ret,
                "volatility": vol,
                "sharpe": sharpe,
            }

    # Select the strongest-growth sampled portfolio meeting Laura's target
    # reserve-funding probability; if none qualify, surface the best probability.
    funding_target = float(laura_settings.get("goal_portfolio_reserve_target", .995))
    funding_probabilities = _approximate_2033_funding_probabilities(
        stats["expected_return"].to_numpy(),
        stats["volatility"].to_numpy(),
        float(laura_settings.get("contribution_2027", 300000)),
        float(laura_settings.get("contribution_2028", 150000)),
        float(laura_settings.get("annual_payment", 50000))
        * int(laura_settings.get("payment_count", 10)),
    )
    qualifying = np.flatnonzero(funding_probabilities >= funding_target)
    if len(qualifying):
        laura_i = int(qualifying[np.argmax(stats["expected_return"].to_numpy()[qualifying])])
        objective_met = True
    else:
        laura_i = int(np.argmax(funding_probabilities))
        objective_met = False
    laura_w = W[laura_i]
    laura_ret, laura_vol, laura_sharpe = portfolio_stats(laura_w, mu, cov, rf)
    laura_portfolio = {
        "weights": pd.Series(laura_w, index=tickers),
        "expected_return": laura_ret,
        "volatility": laura_vol,
        "sharpe": laura_sharpe,
        "funding_success_probability_estimate": float(funding_probabilities[laura_i]),
        "funding_objective_met": objective_met,
        "construction": "Highest expected return among feasible sampled portfolios meeting the approximate 99.5% reserve-funding target; otherwise highest estimated funding probability.",
    }

    portfolios["Laura Goal Portfolio"] = laura_portfolio
    return {
        "returns": returns,
        "expected_returns": mu_s,
        "covariance": cov_df,
        "correlation": returns.corr(),
        "simulations": stats,
        "weights": W,
        "funding_probabilities": funding_probabilities,
        "efficient_frontier": _frontier(stats),
        "portfolios": portfolios,
        "manual_weights_error": manual_weights_error,
        "constraints": {
            "minimum_liquidity": float(settings_metadata.get("liquidity_requirement", 0)),
            "excluded_for_liquidity": excluded_for_liquidity,
            "excluded_for_history": excluded_for_history,
            "sector_caps": sector_caps,
            "equity_bounds": (
                min_equity,
                max_equity,
            ),
            "fixed_income_bounds": (
                min_fixed,
                max_fixed,
            ),
        },
    }


def _finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _normal_cdf(value: float) -> float:
    return .5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _approximate_2033_funding_probabilities(
    expected_returns: np.ndarray,
    volatilities: np.ndarray,
    contribution_2027: float,
    contribution_2028: float,
    target: float,
) -> np.ndarray:
    """Delta-method normal approximation for the 2033 value and funding probability."""
    probabilities = []
    for annual_return, volatility in zip(expected_returns, volatilities):
        if annual_return <= -1 or not np.isfinite(volatility):
            probabilities.append(0.0)
            continue
        sigma2 = max(float(volatility), 0.0) ** 2
        annual_log_growth = math.log1p(float(annual_return))
        mean5 = math.exp(5 * (annual_log_growth + .5 * sigma2))
        mean6 = math.exp(6 * (annual_log_growth + .5 * sigma2))
        var5 = mean5**2 * math.expm1(5 * sigma2)
        var6 = mean6**2 * math.expm1(6 * sigma2)
        cov56 = mean5 * mean6 * math.expm1(5 * sigma2)
        mean = contribution_2027 * mean6 + contribution_2028 * mean5
        variance = (
            contribution_2027**2 * var6
            + contribution_2028**2 * var5
            + 2 * contribution_2027 * contribution_2028 * cov56
        )
        sd = math.sqrt(max(variance, 0.0))
        probabilities.append(1.0 if sd == 0 and mean >= target else 0.0 if sd == 0 else _normal_cdf((mean - target) / sd))
    return np.asarray(probabilities)


def _frontier(stats: pd.DataFrame, points: int = 100) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    ordered = stats.sort_values("volatility")
    best = ordered["expected_return"].cummax()
    frontier = ordered.loc[ordered["expected_return"].ge(best - 1e-12)].drop_duplicates("volatility")
    if len(frontier) > points:
        positions = np.linspace(0, len(frontier) - 1, points).astype(int)
        frontier = frontier.iloc[positions]
    return frontier.reset_index(drop=True)


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
    drift = np.log1p(expected_return) - 0.5 * volatility**2
    paths_2027 = np.exp(rng.normal(drift, volatility, (simulations, 6)).sum(axis=1))
    paths_2028 = np.exp(rng.normal(drift, volatility, (simulations, 5)).sum(axis=1))
    return pd.Series(contribution_2027 * paths_2027 + contribution_2028 * paths_2028)


def deterministic_reserve_requirement(payment: float = 50000, count: int = 10, annual_yield: float = 0.04) -> float:
    y = float(annual_yield)
    if y <= -1:
        raise ValueError("Reserve yield must be greater than -100%.")
    return float(sum(float(payment) / ((1 + y) ** t) for t in range(int(count))))

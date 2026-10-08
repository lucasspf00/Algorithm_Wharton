from __future__ import annotations

import numpy as np
import pandas as pd
from src.sectors import (
    BENCHMARK_SECTORS,
    LAURA_SECTOR_TARGETS,
    laura_sector_bucket,
    normalize_sector,
    normalize_sector_weights,
    normalize_laura_sector_weights,
)


def prepare_returns(price_df: pd.DataFrame, lookback_years: int = 5) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    prices = price_df.copy().sort_index()
    cutoff = prices.index.max() - pd.DateOffset(years=int(lookback_years))
    prices = prices.loc[prices.index >= cutoff]
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns.dropna(how="any")


def annualized_expected_returns(returns: pd.DataFrame) -> pd.Series:
    """Arithmetic annualized expected return, consistent with annualized covariance."""
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.mean() * 252.0


def normalize_market_dates(prices: pd.DataFrame) -> pd.DataFrame:
    normalized = prices.copy()
    dates = pd.DatetimeIndex(pd.to_datetime(normalized.index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    normalized.index = dates.normalize()
    return normalized.groupby(level=0).last().sort_index()


def shrink_expected_returns(
    historical_mu: pd.Series,
    asset_classes: dict[str, str],
    shrinkage: float,
    benchmark_mu: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    amount = float(np.clip(shrinkage, 0.0, 1.0))
    targets = historical_mu.copy()
    class_means = {}
    for asset_class in set(asset_classes.get(ticker, "stock") for ticker in historical_mu.index):
        members = [
            ticker for ticker in historical_mu.index
            if asset_classes.get(ticker, "stock") == asset_class
        ]
        class_means[asset_class] = float(historical_mu.loc[members].mean())
    for ticker in historical_mu.index:
        asset_class = asset_classes.get(ticker, "stock")
        target = (
            benchmark_mu
            if asset_class != "fixed_income" and _finite_number(benchmark_mu)
            else class_means[asset_class]
        )
        targets.loc[ticker] = float(target)
    adjusted = (1.0 - amount) * historical_mu + amount * targets
    return adjusted, targets


def annualized_covariance(returns: pd.DataFrame, shrinkage: float = 0.20) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    sample = returns.cov() * 252.0
    amount = float(np.clip(shrinkage, 0.0, 1.0))
    diagonal = pd.DataFrame(np.diag(np.diag(sample.values)), index=sample.index, columns=sample.columns)
    return (1.0 - amount) * sample + amount * diagonal


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
    allocated = np.zeros(k)
    remaining = 1.0
    while active.any():
        idx = np.where(active)[0]
        proposal = remaining * scores[idx] / scores[idx].sum()
        capped = proposal > cap + 1e-12
        if not capped.any():
            allocated[idx] = proposal
            break
        capped_idx = idx[capped]
        allocated[capped_idx] = cap
        active[capped_idx] = False
        remaining = 1.0 - allocated.sum()
    result = np.zeros(n)
    result[chosen] = allocated
    result /= result.sum()
    return result


def portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    expected_return = float(weights @ mu)
    variance = float(weights @ cov @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else np.nan
    return expected_return, volatility, sharpe


def _finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _sector_map(metadata: pd.DataFrame | None, tickers: list[str]) -> dict[str, dict[str, float]]:
    exposure = {}
    if metadata is None or metadata.empty or "ticker" not in metadata:
        return {ticker: {} for ticker in tickers}
    lookup = metadata.drop_duplicates("ticker").set_index("ticker")
    for ticker in tickers:
        row = lookup.loc[ticker] if ticker in lookup.index else pd.Series(dtype=object)
        asset_class = row.get("asset_class", "stock")
        if asset_class == "fixed_income":
            exposure[ticker] = {}
            continue
        sector_weights = row.get("sector_weightings")
        if asset_class == "equity_etf":
            exposure[ticker] = normalize_sector_weights(sector_weights)
        else:
            sector = normalize_sector(row.get("sector", "Unknown"))
            exposure[ticker] = {sector: 1.0} if sector in BENCHMARK_SECTORS else {}
    return exposure


def _laura_sector_map(
    metadata: pd.DataFrame | None,
    tickers: list[str],
) -> dict[str, dict[str, float]]:
    if metadata is None or metadata.empty or "ticker" not in metadata:
        return {ticker: {} for ticker in tickers}
    lookup = metadata.drop_duplicates("ticker").set_index("ticker")
    exposure = {}
    for ticker in tickers:
        row = lookup.loc[ticker] if ticker in lookup.index else pd.Series(dtype=object)
        asset_class = row.get("asset_class", "stock")
        if asset_class == "fixed_income":
            exposure[ticker] = {}
        elif asset_class == "equity_etf":
            exposure[ticker] = normalize_laura_sector_weights(row.get("sector_weightings"))
        else:
            bucket = laura_sector_bucket(
                row.get("sector", "Unknown"),
                row.get("industry", ""),
                asset_class,
            )
            exposure[ticker] = {bucket: 1.0}
    return exposure


def _laura_sector_constraints_satisfied(
    weights: np.ndarray,
    tickers: list[str],
    asset_classes: dict[str, str],
    exposures: dict[str, dict[str, float]],
) -> bool:
    equity_weight = sum(
        float(weights[index])
        for index, ticker in enumerate(tickers)
        if asset_classes.get(ticker) != "fixed_income"
    )
    if equity_weight <= 0:
        return False
    bucket_weights = {bucket: 0.0 for bucket in LAURA_SECTOR_TARGETS}
    for index, ticker in enumerate(tickers):
        weight = float(weights[index])
        if weight <= 1e-12 or asset_classes.get(ticker) == "fixed_income":
            continue
        exposure = exposures.get(ticker, {})
        if not exposure:
            return False
        for bucket, share in exposure.items():
            bucket_weights[bucket] += weight * share
    for bucket, targets in LAURA_SECTOR_TARGETS.items():
        sleeve_weight = bucket_weights[bucket] / equity_weight
        if not targets["low"] - 1e-9 <= sleeve_weight <= targets["high"] + 1e-9:
            return False
    return True


def sector_exposure_coverage(weights: pd.Series, metadata: pd.DataFrame | None) -> dict:
    exposures = _sector_map(metadata, list(weights.index))
    equity_weights = {}
    if metadata is not None and not metadata.empty and "ticker" in metadata:
        lookup = metadata.drop_duplicates("ticker").set_index("ticker")
    else:
        lookup = pd.DataFrame()
    for ticker in weights.index:
        row = lookup.loc[ticker] if ticker in lookup.index else pd.Series(dtype=object)
        if row.get("asset_class", "stock") != "fixed_income":
            equity_weights[ticker] = max(float(weights[ticker]), 0.0)
    total_equity = sum(equity_weights.values())
    covered = sum(
        weight for ticker, weight in equity_weights.items()
        if bool(exposures.get(ticker))
    )
    incomplete = [
        ticker for ticker, weight in equity_weights.items()
        if weight > 0 and not exposures.get(ticker)
    ]
    return {
        "equity_weight": total_equity,
        "covered_equity_weight": covered,
        "coverage": covered / total_equity if total_equity > 0 else 1.0,
        "incomplete_tickers": incomplete,
        "complete": not incomplete,
    }


def sector_allocation_table(
    weights: pd.Series,
    metadata: pd.DataFrame | None,
    benchmark_weights: dict[str, float],
    tolerance: float,
) -> pd.DataFrame:
    exposures = _sector_map(metadata, list(weights.index))
    equity_weight = 0.0
    sector_portfolio = {}
    asset_class = {}
    if metadata is not None and not metadata.empty and "ticker" in metadata:
        lookup = metadata.drop_duplicates("ticker").set_index("ticker")
        asset_class = lookup.get("asset_class", pd.Series(dtype=str)).to_dict()
    for ticker, weight in weights.items():
        cls = asset_class.get(ticker, "stock")
        if cls == "fixed_income":
            continue
        equity_weight += float(weight)
        for sector, share in exposures.get(ticker, {}).items():
            sector_portfolio[sector] = sector_portfolio.get(sector, 0.0) + float(weight) * share
    rows = []
    for sector in sorted(set(benchmark_weights) | set(sector_portfolio)):
        benchmark = float(benchmark_weights.get(sector, 0.0))
        total_weight = sector_portfolio.get(sector, 0.0)
        sleeve_weight = total_weight / equity_weight if equity_weight > 0 else np.nan
        rows.append({
            "sector": sector,
            "equity_sleeve_weight": sleeve_weight,
            "total_portfolio_weight": total_weight,
            "sp500_benchmark_weight": benchmark,
            "target_low": max(0.0, benchmark - tolerance),
            "target_high": min(1.0, benchmark + tolerance),
        })
    return pd.DataFrame(rows)


def laura_sector_allocation_table(
    weights: pd.Series,
    metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict]:
    """Return Laura bucket exposures as equity-sleeve and total-portfolio shares."""
    lookup = (
        metadata.drop_duplicates("ticker").set_index("ticker")
        if metadata is not None and not metadata.empty and "ticker" in metadata
        else pd.DataFrame()
    )
    buckets = {bucket: 0.0 for bucket in LAURA_SECTOR_TARGETS}
    equity_weight = 0.0
    incomplete = []
    for ticker, raw_weight in weights.items():
        weight = max(float(raw_weight), 0.0)
        row = lookup.loc[ticker] if ticker in lookup.index else pd.Series(dtype=object)
        asset_class = row.get("asset_class", "stock")
        if asset_class == "fixed_income":
            continue
        equity_weight += weight
        if asset_class == "equity_etf":
            exposure = normalize_laura_sector_weights(row.get("sector_weightings"))
            if not exposure:
                if weight > 0:
                    incomplete.append(str(ticker))
                continue
            for bucket, share in exposure.items():
                buckets[bucket] += weight * share
            continue
        bucket = laura_sector_bucket(
            row.get("sector", "Unknown"),
            row.get("industry", ""),
            asset_class,
        )
        buckets[bucket] += weight
    complete = not incomplete
    rows = []
    for bucket, targets in LAURA_SECTOR_TARGETS.items():
        total_weight = buckets[bucket]
        sleeve_weight = total_weight / equity_weight if equity_weight > 0 else np.nan
        status = (
            "INCOMPLETE" if not complete
            else "UNDER RANGE" if sleeve_weight < targets["low"] - 1e-9
            else "OVER RANGE" if sleeve_weight > targets["high"] + 1e-9
            else "IN RANGE"
        ) if np.isfinite(sleeve_weight) else "N/A"
        rows.append({
            "sector": bucket,
            "equity_sleeve_weight": sleeve_weight,
            "target": targets["target"],
            "acceptable_range": (targets["low"], targets["high"]),
            "total_portfolio_weight": total_weight,
            "status": status,
        })
    return pd.DataFrame(rows), {
        "equity_weight": equity_weight,
        "covered_equity_weight": equity_weight - sum(
            max(float(weights.get(ticker, 0.0)), 0.0) for ticker in incomplete
        ),
        "coverage": (
            1.0 - sum(max(float(weights.get(ticker, 0.0)), 0.0) for ticker in incomplete) / equity_weight
            if equity_weight > 0 else 1.0
        ),
        "incomplete_tickers": incomplete,
        "complete": complete,
    }


def simulate_laura_funding(
    daily_portfolio_returns: pd.Series | None,
    expected_return: float,
    volatility: float,
    cfg: dict,
    simulations: int | None = None,
    seed: int | None = None,
    bootstrap_indices: np.ndarray | None = None,
    standard_normal_paths: np.ndarray | None = None,
) -> dict:
    """Simulate Laura cash flows using a daily block bootstrap or explicit normal fallback."""
    laura = cfg["laura"]
    count = max(100_000, int(simulations or laura.get("projection_simulations", 100_000)))
    random_seed = int(seed if seed is not None else laura["reserve_simulation_seed"])
    years = list(range(2027, 2043))
    return_years = len(years) - 1
    periods_per_year = 12
    block_days = 21
    series = pd.to_numeric(daily_portfolio_returns, errors="coerce").dropna() if daily_portfolio_returns is not None else pd.Series(dtype=float)
    use_bootstrap = len(series) >= 252 and (series > -1).all()
    method = "Multivariate normal fallback"
    annual_gross = None
    if use_bootstrap:
        block_log_gross = np.log1p(series).rolling(block_days).sum().dropna()
        block_gross = np.exp(block_log_gross.to_numpy(dtype=float))
        required_shape = (count, return_years * periods_per_year)
        if len(block_gross) >= periods_per_year and bootstrap_indices is not None and bootstrap_indices.shape == required_shape:
            annual_gross = np.ones((count, return_years), dtype=float)
            for period in range(periods_per_year):
                annual_gross *= block_gross[bootstrap_indices[:, period::periods_per_year]]
            method = "Historical 21-trading-day block bootstrap"
        else:
            use_bootstrap = False
    if not use_bootstrap:
        if standard_normal_paths is None:
            standard_normal_paths = np.random.default_rng(random_seed).standard_normal((count, return_years))
        if standard_normal_paths.shape != (count, return_years):
            raise ValueError("Fallback shocks must have one column per return year from 2027 through 2041.")
        annual_returns = float(expected_return) + float(volatility) * standard_normal_paths
        annual_gross = np.maximum(1.0 + annual_returns, 0.0)
        method = (
            "Multivariate-normal Monte Carlo fallback projected to portfolio return "
            "w'μ and volatility sqrt(w'Σw)"
        )
    balance = np.zeros(count, dtype=float)
    successful = np.ones(count, dtype=bool)
    value_2033 = np.full(count, np.nan)
    payment = float(laura["annual_payment"])
    for index, year in enumerate(years):
        if year == 2027:
            balance += float(laura["contribution_2027"])
        elif year == 2028:
            balance += float(laura["contribution_2028"])
        if year == 2033:
            value_2033 = balance.copy()
        if year >= 2033:
            successful &= balance >= payment
            balance = np.where(successful, balance - payment, 0.0)
        if year < 2042:
            balance = np.where(successful, balance * annual_gross[:, index], 0.0)
    return {
        "funding_success_probability": float(successful.mean()),
        "successful_paths": int(successful.sum()),
        "total_simulations": count,
        "value_2033": pd.Series(value_2033),
        "nominal_operating_commitment": payment * int(laura["payment_count"]),
        "seed": random_seed,
        "annual_expected_return": float(expected_return),
        "annual_volatility": float(volatility),
        "method": method,
        "block_days": block_days if method.startswith("Historical") else None,
        "periods_per_year": periods_per_year if method.startswith("Historical") else None,
        "assumptions": (
            (
                "Portfolio daily simple returns apply target weights consistently to each day's asset returns; "
                "contiguous 21-trading-day blocks are sampled with replacement and compounded in 12 blocks per return year"
            )
            if method.startswith("Historical")
            else (
                "Fallback draws independent annual multivariate-normal asset returns using adjusted mu and annualized covariance; "
                "the portfolio projection is Normal(w'mu, w'Sigma w), simulated with the fixed seed"
            )
        ) + (
            "; +$300,000 at beginning 2027; +$150,000 at beginning 2028; no withdrawals before 2033; "
            "then ten $50,000 beginning-of-year payments from 2033 through 2042; WInS excluded"
        ),
    }


def _frontier(stats: pd.DataFrame, points: int = 100) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    ordered = stats.sort_values("volatility")
    frontier = ordered.loc[ordered["expected_return"].ge(ordered["expected_return"].cummax() - 1e-12)]
    frontier = frontier.drop_duplicates("volatility")
    if len(frontier) > points:
        frontier = frontier.iloc[np.linspace(0, len(frontier) - 1, points).astype(int)]
    return frontier.reset_index(drop=True)


def monte_carlo_optimize(
    price_df: pd.DataFrame,
    cfg: dict,
    metadata: pd.DataFrame | None = None,
    manual_weights: pd.Series | None = None,
    benchmark_sector_weights: dict[str, float] | None = None,
    benchmark_prices: pd.DataFrame | None = None,
) -> dict:
    settings, laura = cfg["optimizer"], cfg["laura"]
    benchmark_weights = normalize_sector_weights(
        benchmark_sector_weights or settings.get("sector_benchmark_weights", {})
    )
    tolerance = float(settings["sector_tolerance"])
    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("Sector tolerance must be a decimal between 0 and 1.")
    price_df = normalize_market_dates(price_df) if price_df is not None and not price_df.empty else pd.DataFrame()
    raw_tickers = list(price_df.columns)
    returns = prepare_returns(price_df, int(settings["lookback_years"]))
    excluded_for_liquidity = []
    excluded_for_sector_data = []
    if metadata is not None and not metadata.empty and "average_dollar_volume_30d" in metadata:
        liquidity = metadata.drop_duplicates("ticker").set_index("ticker")["average_dollar_volume_30d"]
        minimum = float(settings["liquidity_requirement"])
        eligible = [
            ticker for ticker in returns.columns
            if _finite_number(liquidity.get(ticker)) and float(liquidity[ticker]) >= minimum
        ]
        excluded_for_liquidity = [ticker for ticker in returns.columns if ticker not in eligible]
        returns = returns[eligible]
    initial_exposures = _sector_map(metadata, list(returns.columns))
    if benchmark_weights and tolerance < 1.0 and any(initial_exposures.values()):
        if metadata is not None and not metadata.empty and "ticker" in metadata:
            initial_lookup = metadata.drop_duplicates("ticker").set_index("ticker")
        else:
            initial_lookup = pd.DataFrame()
        excluded_for_sector_data = [
            ticker for ticker in returns.columns
            if (
                initial_lookup.loc[ticker].get("asset_class", "stock") != "fixed_income"
                if ticker in initial_lookup.index
                else True
            )
            and not initial_exposures.get(ticker)
        ]
        returns = returns.drop(columns=excluded_for_sector_data)
    excluded_for_history = [
        ticker for ticker in raw_tickers
        if ticker not in returns.columns
        and ticker not in excluded_for_liquidity
        and ticker not in excluded_for_sector_data
    ]
    if returns.shape[0] < 126:
        raise ValueError("Not enough overlapping daily price history after liquidity filtering.")
    tickers = list(returns.columns)
    count, cap = len(tickers), float(settings["max_weight"])
    if count * cap < 1.0 - 1e-12:
        minimum_names = int(np.ceil(1.0 / cap))
        raise ValueError(
            f"Portfolio is infeasible: at least {minimum_names} eligible holdings are required "
            f"at a {cap:.0%} maximum weight; {count} are available."
        )

    historical_mu = annualized_expected_returns(returns)
    cov_frame = annualized_covariance(returns, float(settings["covariance_shrinkage"]))
    lookup = metadata.drop_duplicates("ticker").set_index("ticker") if metadata is not None and not metadata.empty and "ticker" in metadata else pd.DataFrame()
    asset_classes = lookup.get("asset_class", pd.Series(dtype=str)).to_dict() if not lookup.empty else {}
    laura_exposures = _laura_sector_map(metadata, tickers)
    benchmark_mu = None
    benchmark_ticker = str(settings.get("benchmark", "SPY")).upper()
    if benchmark_prices is not None and not benchmark_prices.empty and benchmark_ticker in benchmark_prices.columns:
        aligned_benchmark = normalize_market_dates(benchmark_prices[[benchmark_ticker]])[benchmark_ticker]
        benchmark_returns = aligned_benchmark.pct_change(fill_method=None).reindex(returns.index).dropna()
        if len(benchmark_returns) >= 252:
            benchmark_mu = float(benchmark_returns.mean() * 252.0)
    shrinkage = float(settings.get("return_estimate_shrinkage", .30))
    adjusted_mu, return_targets = shrink_expected_returns(
        historical_mu,
        asset_classes,
        shrinkage,
        benchmark_mu,
    )
    mu, covariance = adjusted_mu.to_numpy(), cov_frame.to_numpy()
    risk_free_rate = float(settings["risk_free_rate"])

    equity_indices = np.asarray([i for i, t in enumerate(tickers) if asset_classes.get(t) != "fixed_income"], dtype=int)
    fixed_indices = np.asarray([i for i, t in enumerate(tickers) if asset_classes.get(t) == "fixed_income"], dtype=int)
    min_equity = float(settings["minimum_equity_weight"])
    max_equity = float(settings["maximum_equity_weight"])
    min_fixed = float(settings["minimum_fixed_income_weight"])
    max_fixed = float(settings["maximum_fixed_income_weight"])
    if min_equity > max_equity or min_fixed > max_fixed or min_equity + min_fixed > 1 or max_equity + max_fixed < 1:
        raise ValueError("Equity and fixed-income bounds cannot be satisfied by a fully invested portfolio.")
    exposures = _sector_map(metadata, tickers)
    represented_sectors = (
        sorted(benchmark_weights)
        if tolerance < 1.0 and any(exposures.values())
        else []
    )
    sector_ranges = {
        sector: (
            max(0.0, benchmark_weights[sector] - tolerance),
            min(1.0, benchmark_weights[sector] + tolerance),
        )
        for sector in represented_sectors
    }
    equity_minimum_feasible = max(
        min_equity,
        1.0 - max_fixed,
        1.0 - len(fixed_indices) * cap,
        0.0,
    )
    equity_maximum_feasible = min(
        max_equity,
        1.0 - min_fixed,
        len(equity_indices) * cap,
        1.0,
    )
    if equity_minimum_feasible > equity_maximum_feasible + 1e-9:
        raise ValueError(
            "Eligible holdings and equity/fixed-income bounds cannot form a fully invested portfolio under the max holding constraint."
        )
    if sector_ranges:
        total_sector_minimum = sum(lower for lower, _ in sector_ranges.values())
        total_sector_maximum = sum(upper for _, upper in sector_ranges.values())
        if total_sector_minimum > 1.0 + 1e-9 or total_sector_maximum < 1.0 - 1e-9:
            raise ValueError("The configured sector target ranges cannot sum to 100% of the equity sleeve.")
        if equity_minimum_feasible > 0:
            for sector, (lower, upper) in sector_ranges.items():
                coefficients = [
                    exposures[ticker].get(sector, 0.0)
                    for ticker in tickers
                    if asset_classes.get(ticker) != "fixed_income"
                ]
                remaining = equity_minimum_feasible
                minimum_exposure = 0.0
                for coefficient in sorted(coefficients):
                    allocation = min(cap, remaining)
                    minimum_exposure += allocation * coefficient
                    remaining -= allocation
                    if remaining <= 1e-12:
                        break
                remaining = equity_minimum_feasible
                maximum_exposure = 0.0
                for coefficient in sorted(coefficients, reverse=True):
                    allocation = min(cap, remaining)
                    maximum_exposure += allocation * coefficient
                    remaining -= allocation
                    if remaining <= 1e-12:
                        break
                minimum_share = minimum_exposure / equity_minimum_feasible
                maximum_share = maximum_exposure / equity_minimum_feasible
                if maximum_share < lower - 1e-9 or minimum_share > upper + 1e-9:
                    raise ValueError(
                        f"Sector constraint for {sector} is infeasible with eligible holdings and the max holding/equity constraints."
                    )

    def valid_weights(weights: np.ndarray) -> bool:
        equity = float(weights[equity_indices].sum()) if len(equity_indices) else 0.0
        fixed = float(weights[fixed_indices].sum()) if len(fixed_indices) else 0.0
        if not min_equity - 1e-9 <= equity <= max_equity + 1e-9:
            return False
        if not min_fixed - 1e-9 <= fixed <= max_fixed + 1e-9:
            return False
        if equity > 0:
            for sector, (lower, upper) in sector_ranges.items():
                sleeve_sector = sum(
                    weights[i] * exposures[ticker].get(sector, 0.0)
                    for i, ticker in enumerate(tickers)
                ) / equity
                if sleeve_sector < lower - 1e-9 or sleeve_sector > upper + 1e-9:
                    return False
        return True

    rng = np.random.default_rng(42)
    simulations = int(settings["simulations"])
    weight_rows, stat_rows = [], []
    attempts, attempt_limit = 0, max(simulations * 200, 20_000)
    while len(weight_rows) < simulations and attempts < attempt_limit:
        attempts += 1
        weights = _bounded_weights(count, cap, rng, allow_zeros=True)
        if not valid_weights(weights):
            continue
        stat_rows.append(portfolio_stats(weights, mu, covariance, risk_free_rate))
        weight_rows.append(weights)
    if len(weight_rows) < max(100, int(simulations * .10)):
        raise ValueError("No useful feasible sample found. Relax allocation or S&P 500 sector-range constraints.")
    stats = pd.DataFrame(stat_rows, columns=["expected_return", "volatility", "sharpe"])
    weight_matrix = np.vstack(weight_rows)
    min_vol_index = int(stats["volatility"].idxmin())
    finite_sharpe = stats["sharpe"].replace([np.inf, -np.inf], np.nan)
    max_sharpe_index = (
        int(finite_sharpe.idxmax())
        if finite_sharpe.notna().any()
        else min_vol_index
    )
    max_return_index = int(stats["expected_return"].idxmax())

    def pack(index: int) -> dict:
        values = portfolio_stats(weight_matrix[index], mu, covariance, risk_free_rate)
        return {
            "weights": pd.Series(weight_matrix[index], index=tickers),
            "expected_return": values[0],
            "volatility": values[1],
            "sharpe": values[2],
        }

    portfolios = {
        "Minimum Volatility": pack(min_vol_index),
        "Maximum Sharpe": pack(max_sharpe_index),
        "Maximum Expected Return": pack(max_return_index),
    }
    manual_weights_error = None
    if manual_weights is not None:
        manual = pd.to_numeric(manual_weights.reindex(tickers), errors="coerce").fillna(0).to_numpy(dtype=float)
        if not np.isfinite(manual).all() or (manual < -1e-9).any() or not np.isclose(manual.sum(), 1, atol=1e-6):
            manual_weights_error = "Current Manual Portfolio weights must sum to 100% over eligible securities."
        elif manual.max() > cap + 1e-9 or not valid_weights(manual):
            manual_weights_error = "Current Manual Portfolio violates a holding, sleeve, or S&P sector-range constraint."
        else:
            ret, vol, sharpe = portfolio_stats(manual, mu, covariance, risk_free_rate)
            portfolios["Current Manual Portfolio"] = {
                "weights": pd.Series(manual, index=tickers),
                "expected_return": ret,
                "volatility": vol,
                "sharpe": sharpe,
            }

    frontier = _frontier(stats, points=100)
    laura_feasible_indices = [
        index for index, weights in enumerate(weight_matrix)
        if _laura_sector_constraints_satisfied(weights, tickers, asset_classes, laura_exposures)
    ]
    laura_candidate_indices = []
    if laura_feasible_indices:
        laura_stats = stats.loc[laura_feasible_indices]
        laura_frontier_indices = laura_stats.sort_values("volatility").loc[
            lambda frame: frame["expected_return"].ge(frame["expected_return"].cummax() - 1e-12)
        ].index.to_numpy(dtype=int)
        if len(laura_frontier_indices) > 30:
            laura_frontier_indices = laura_frontier_indices[
                np.linspace(0, len(laura_frontier_indices) - 1, 30).astype(int)
            ]
        laura_finite_sharpe = laura_stats["sharpe"].replace([np.inf, -np.inf], np.nan)
        laura_max_sharpe_index = (
            int(laura_finite_sharpe.idxmax())
            if laura_finite_sharpe.notna().any()
            else int(laura_stats["volatility"].idxmin())
        )
        laura_candidate_indices = sorted(set(
            laura_frontier_indices.tolist()
            + [
                int(laura_stats["volatility"].idxmin()),
                laura_max_sharpe_index,
                int(laura_stats["expected_return"].idxmax()),
            ]
        ))
    funding_count = max(100_000, int(laura["projection_simulations"]))
    funding_seed = int(laura["reserve_simulation_seed"])
    funding_rng = np.random.default_rng(funding_seed)
    block_days = 21
    periods_per_year = 12
    return_years = 15
    available_blocks = max(0, len(returns) - block_days + 1)
    bootstrap_indices = None
    fallback_paths = None
    if laura_candidate_indices and len(returns) >= 252 and available_blocks >= periods_per_year:
        bootstrap_indices = funding_rng.integers(
            0,
            available_blocks,
            size=(funding_count, return_years * periods_per_year),
            dtype=np.int32,
        )
    elif laura_candidate_indices:
        fallback_paths = funding_rng.standard_normal((funding_count, return_years))
    funding_candidates = []
    for index in laura_candidate_indices:
        expected_return, volatility, _ = portfolio_stats(weight_matrix[index], mu, covariance, risk_free_rate)
        daily_portfolio_returns = returns @ weight_matrix[index]
        funding = simulate_laura_funding(
            daily_portfolio_returns,
            expected_return,
            volatility,
            cfg,
            bootstrap_indices=bootstrap_indices,
            standard_normal_paths=fallback_paths,
        )
        funding_candidates.append((index, funding))
    target = float(laura.get("funding_success_target", .995))
    feasible = [
        item for item in funding_candidates
        if item[1]["funding_success_probability"] >= target
    ]
    laura_funding = None
    if feasible:
        laura_index, laura_funding = max(
            feasible,
            key=lambda item: portfolio_stats(weight_matrix[item[0]], mu, covariance, risk_free_rate)[0],
        )
        objective_met = True
    elif funding_candidates:
        laura_index, laura_funding = max(
            funding_candidates,
            key=lambda item: (
                item[1]["funding_success_probability"],
                portfolio_stats(weight_matrix[item[0]], mu, covariance, risk_free_rate)[0],
            ),
        )
        objective_met = False
    if laura_funding is not None:
        laura_portfolio = pack(laura_index)
        laura_portfolio.update({
            "funding_success_probability": laura_funding["funding_success_probability"],
            "funding_successful_paths": laura_funding["successful_paths"],
            "funding_simulations": laura_funding["total_simulations"],
            "funding_seed": laura_funding["seed"],
            "funding_method": laura_funding["method"],
            "funding_assumptions": laura_funding["assumptions"],
            "funding_available": np.isfinite(laura_funding["funding_success_probability"]),
            "funding_objective_target": target,
            "funding_objective_met": objective_met,
            "funding_value_2033": laura_funding["value_2033"],
            "construction": (
                "Highest arithmetic expected-return sampled portfolio among portfolios meeting "
                "Laura's sector-sleeve guardrails whose explicit funding-path simulation meets the "
                "funding-success target; otherwise the highest simulated funding success, with "
                "expected return as tie-breaker."
            ),
        })
        portfolios["Laura Goal Portfolio"] = laura_portfolio

    current_year = pd.Timestamp.today().year
    assumptions = {
        "lookback_years": int(settings["lookback_years"]),
        "historical_return_estimator": "Arithmetic annualized historical return = mean(daily simple returns) × 252",
        "return_estimate_shrinkage": shrinkage,
        "return_shrinkage_formula": "adjusted_mu = (1 - shrinkage) * historical_mu + shrinkage * benchmark_or_asset_class_mu",
        "equity_return_shrink_target": f"{benchmark_ticker} arithmetic mean with at least 252 aligned daily observations; otherwise equity asset-class mean",
        "fixed_income_return_shrink_target": "Fixed-income asset-class historical mean",
        "return_estimator": "Adjusted arithmetic annual expected return used for optimization",
        "covariance_estimator": "Annualized covariance = daily simple-return covariance × 252, with configured diagonal shrinkage",
        "risk_free_rate": risk_free_rate,
        "maximum_holding_weight": cap,
        "equity_weight_range": (min_equity, max_equity),
        "fixed_income_weight_range": (min_fixed, max_fixed),
        "sector_benchmark": settings.get("sector_benchmark", "SPY"),
        "sector_tolerance": tolerance,
        "sector_ranges": sector_ranges,
        "laura_candidate_selection": (
            "Laura candidates must satisfy all six custom equity-sleeve sector ranges "
            "and have complete sector look-through before explicit funding simulation."
        ),
        "laura_sector_guardrails": {
            bucket: (bounds["low"], bounds["high"])
            for bucket, bounds in LAURA_SECTOR_TARGETS.items()
        },
        "liquidity_minimum": float(settings["liquidity_requirement"]),
        "annual_return_simulations": funding_count,
        "simulation_seed": funding_seed,
        "funding_simulation_method": (
            laura_funding["method"] if laura_funding is not None
            else "Unavailable: no sampled portfolio met Laura sector-sleeve guardrails"
        ),
        "funding_simulation_assumptions": (
            laura_funding["assumptions"] if laura_funding is not None
            else "Simulation not run because no sampled portfolio met all six Laura equity-sleeve sector ranges."
        ),
        "funding_bootstrap_block_days": laura_funding["block_days"] if laura_funding is not None else None,
        "funding_bootstrap_periods_per_year": laura_funding["periods_per_year"] if laura_funding is not None else None,
        "funding_success_target": target,
        "analysis_year": current_year,
    }
    return {
        "returns": returns,
        "expected_returns": adjusted_mu,
        "historical_expected_returns": historical_mu,
        "expected_return_targets": return_targets,
        "covariance": cov_frame,
        "correlation": returns.corr(),
        "simulations": stats,
        "weights": weight_matrix,
        "efficient_frontier": frontier,
        "portfolios": portfolios,
        "manual_weights_error": manual_weights_error,
        "assumptions": assumptions,
        "constraints": {
            "minimum_liquidity": float(settings["liquidity_requirement"]),
            "excluded_for_liquidity": excluded_for_liquidity,
            "excluded_for_sector_data": excluded_for_sector_data,
            "excluded_for_history": excluded_for_history,
            "sector_ranges": sector_ranges,
            "sector_benchmark_weights": benchmark_weights,
            "represented_sectors": represented_sectors,
            "laura_sector_ranges": {
                bucket: (bounds["low"], bounds["high"])
                for bucket, bounds in LAURA_SECTOR_TARGETS.items()
            },
            "laura_sector_feasible_samples": len(laura_feasible_indices),
            "equity_bounds": (min_equity, max_equity),
            "fixed_income_bounds": (min_fixed, max_fixed),
        },
    }


def portfolio_benchmark_comparison(
    price_df: pd.DataFrame,
    weights: pd.Series,
    benchmark_prices: pd.DataFrame,
    risk_free_rate: float,
) -> pd.DataFrame:
    rows = []
    holdings = [ticker for ticker, weight in weights.items() if weight > 0 and ticker in price_df.columns]
    if not holdings or benchmark_prices is None or benchmark_prices.empty:
        return pd.DataFrame()
    def market_dates(frame: pd.DataFrame) -> pd.DataFrame:
        daily = frame.copy()
        index = pd.DatetimeIndex(pd.to_datetime(daily.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        daily.index = index.normalize()
        return daily.groupby(level=0).last().sort_index()

    candidate_prices = market_dates(price_df[holdings])
    benchmark_prices = market_dates(benchmark_prices)
    for benchmark in benchmark_prices.columns:
        benchmark_column = f"__benchmark_{benchmark}__"
        benchmark_series = benchmark_prices[[benchmark]].rename(columns={benchmark: benchmark_column})
        panel = candidate_prices.join(benchmark_series, how="inner").dropna(how="any")
        if len(panel) < 3:
            continue
        w = weights.reindex(holdings).astype(float)
        w /= w.sum()
        asset_nav = panel[holdings].divide(panel[holdings].iloc[0], axis="columns").mul(w, axis="columns").sum(axis=1)
        bench_nav = panel[benchmark_column] / panel[benchmark_column].iloc[0]
        asset_returns = asset_nav.pct_change(fill_method=None).dropna()
        bench_returns = bench_nav.pct_change(fill_method=None).dropna()
        aligned = pd.concat([asset_returns.rename("portfolio"), bench_returns.rename("benchmark")], axis=1).dropna()
        if aligned.empty:
            continue
        years = (panel.index[-1] - panel.index[0]).days / 365.25
        asset_arithmetic = float(aligned["portfolio"].mean() * 252)
        bench_arithmetic = float(aligned["benchmark"].mean() * 252)
        asset_vol = float(aligned["portfolio"].std(ddof=1) * np.sqrt(252))
        bench_vol = float(aligned["benchmark"].std(ddof=1) * np.sqrt(252))
        asset_sharpe = (asset_arithmetic - risk_free_rate) / asset_vol if asset_vol > 0 else np.nan
        bench_sharpe = (bench_arithmetic - risk_free_rate) / bench_vol if bench_vol > 0 else np.nan
        asset_drawdown = float((asset_nav / asset_nav.cummax() - 1).min())
        bench_drawdown = float((bench_nav / bench_nav.cummax() - 1).min())
        beta = float(aligned["portfolio"].cov(aligned["benchmark"]) / aligned["benchmark"].var(ddof=1)) if aligned["benchmark"].var(ddof=1) > 0 else np.nan
        for name, arithmetic, nav, vol, sharpe, drawdown, series_beta in (
            ("Current portfolio", asset_arithmetic, asset_nav, asset_vol, asset_sharpe, asset_drawdown, beta),
            (str(benchmark), bench_arithmetic, bench_nav, bench_vol, bench_sharpe, bench_drawdown, 1.0),
        ):
            cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan
            rows.append({
                "series": name,
                "benchmark": str(benchmark),
                "annualized_arithmetic_return": arithmetic,
                "cagr": cagr,
                "volatility": vol,
                "sharpe": sharpe,
                "max_drawdown": drawdown,
                "beta_vs_benchmark": series_beta,
                "overlap_start": panel.index[0],
                "overlap_end": panel.index[-1],
                "overlap_observations": len(aligned),
            })
    return pd.DataFrame(rows)


def deterministic_reserve_requirement(
    payment: float = 50000,
    count: int = 10,
    annual_yield: float = 0.04,
) -> float:
    if annual_yield <= -1:
        raise ValueError("Reserve yield must be greater than -100%.")
    return float(sum(float(payment) / ((1 + annual_yield) ** year) for year in range(int(count))))

from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd
from .data import reserve_asset_profile


LIABILITY = 500_000.0
PAYMENT = 50_000.0
PAYMENT_COUNT = 10
PAYMENT_YEARS = tuple(range(2033, 2043))
TARGETS = (0.95, 0.99, 0.995, 0.999)


def normalize_reserve_weights(assets: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize selected reserve-asset weights."""
    out = assets.copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.loc[out["weight"].notna() & (out["weight"] >= 0)].copy()
    total = float(out["weight"].sum())
    if out.empty or total <= 0:
        raise ValueError("Assign a positive weight to at least one reserve asset.")
    out["weight"] = out["weight"] / total
    return out


def weighted_reserve_yield(assets: pd.DataFrame) -> float:
    """Weighted model yield from fixed-income reserve assets only."""
    out = normalize_reserve_weights(assets)
    if "asset" not in out.columns:
        raise ValueError("Reserve assets must include an asset ticker.")
    profiles = out["asset"].map(reserve_asset_profile)
    if profiles.eq("equity_etf").any():
        raise ValueError("Equity ETFs such as VOO cannot be used as reserve-yield assets.")
    if profiles.eq("unknown").any():
        raise ValueError("Reserve yield is available only for recognized fixed-income reserve ETFs.")
    yields = pd.to_numeric(out["yield"], errors="coerce")
    if yields.isna().any():
        raise ValueError("Enter a yield for every selected reserve asset; missing yields are not fabricated.")
    return float((out["weight"] * yields).sum())


def deterministic_reserve_pv(
    reserve_yield: float,
    payment: float = PAYMENT,
    count: int = PAYMENT_COUNT,
) -> float:
    if reserve_yield <= -1:
        raise ValueError("Reserve yield must be greater than -100%.")
    return float(sum(payment / ((1 + reserve_yield) ** t) for t in range(count)))


def simulate_reserve_paths(
    starting_reserve: float,
    reserve_yield: float,
    volatility: float,
    simulations: int = 100_000,
    seed: int = 20260924,
    payment: float = PAYMENT,
    count: int = PAYMENT_COUNT,
) -> np.ndarray:
    """Simulate annual reserve paths with beginning-of-year withdrawals."""
    if simulations < 100_000:
        raise ValueError("Final reserve analysis requires at least 100,000 simulations.")
    if starting_reserve < 0 or volatility < 0:
        raise ValueError("Starting reserve and volatility must be non-negative.")
    rng = np.random.default_rng(seed)
    values = np.full(simulations, float(starting_reserve))
    for year in range(count):
        values -= payment
        if year < count - 1:
            shocks = rng.normal(reserve_yield - 0.5 * volatility**2, volatility, simulations)
            values = np.where(values > 0, values * np.exp(shocks), 0.0)
    return values


def funding_success_probability(
    starting_reserve: float,
    reserve_yield: float,
    volatility: float,
    simulations: int = 100_000,
    seed: int = 20260924,
) -> float:
    if simulations < 100_000:
        raise ValueError("Final reserve analysis requires at least 100,000 simulations.")
    if starting_reserve < 0 or volatility < 0 or reserve_yield <= -1:
        raise ValueError("Invalid reserve assumptions.")
    shocks = np.random.default_rng(seed).normal(
        reserve_yield - 0.5 * volatility**2,
        volatility,
        (simulations, PAYMENT_COUNT - 1),
    )
    return _funding_success_from_shocks(starting_reserve, shocks)


def _funding_success_from_shocks(starting_reserve: float, shocks: np.ndarray) -> float:
    """Evaluate fixed random paths so reserve bisection is fast and reproducible."""
    values = np.full(shocks.shape[0], float(starting_reserve))
    survived = np.ones(shocks.shape[0], dtype=bool)
    for year in range(PAYMENT_COUNT):
        survived &= values >= PAYMENT
        values = np.where(survived, values - PAYMENT, 0.0)
        if year < PAYMENT_COUNT - 1:
            values = np.where(survived, values * np.exp(shocks[:, year]), 0.0)
    return float(np.mean(survived))


def required_reserves(
    reserve_yield: float,
    volatility: float,
    targets: Iterable[float] = TARGETS,
    simulations: int = 100_000,
    seed: int = 20260924,
) -> pd.DataFrame:
    """Find minimum starting reserves by deterministic bisection on simulations."""
    rows = []
    rng = np.random.default_rng(seed)
    shocks = rng.normal(
        reserve_yield - 0.5 * volatility**2,
        volatility,
        (simulations, PAYMENT_COUNT - 1),
    )
    upper = max(deterministic_reserve_pv(reserve_yield) + LIABILITY, LIABILITY)
    for target in targets:
        lo, hi = 0.0, upper
        while _funding_success_from_shocks(hi, shocks) < target:
            hi *= 2.0
        for _ in range(32):
            mid = (lo + hi) / 2
            success = _funding_success_from_shocks(mid, shocks)
            if success >= target:
                hi = mid
            else:
                lo = mid
        success = _funding_success_from_shocks(hi, shocks)
        rows.append({
            "funding_target": target,
            "failure_rate": 1.0 - target,
            "required_reserve": hi,
            "simulated_success": success,
        })
    return pd.DataFrame(rows)

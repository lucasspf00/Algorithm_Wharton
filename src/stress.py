from __future__ import annotations
import numpy as np
import pandas as pd

HISTORICAL_WINDOWS = {
    "Dot-com drawdown": ("2000-03-24", "2002-10-10"),
    "Global Financial Crisis": ("2007-10-09", "2009-03-10"),
    "COVID crash": ("2020-02-19", "2020-03-24"),
    "2022 rate/inflation shock": ("2022-01-03", "2022-10-13"),
}


def historical_portfolio_stress(prices: pd.DataFrame, weights: pd.Series) -> dict:
    if prices is None or prices.empty:
        return {"available": False, "reason": "No historical prices were returned."}
    weights = weights.copy()
    common = [c for c in weights.index if c in prices.columns and prices[c].notna().sum() >= 2]
    if not common:
        return {"available": False, "reason": "None of the holdings have usable data in this period."}
    coverage = float(weights.loc[common].sum())
    w = weights.loc[common]
    if w.sum() <= 0:
        return {"available": False, "reason": "No positive portfolio weight has usable history."}
    w = w / w.sum()
    p = prices[common].dropna(how="any")
    if len(p) < 2:
        return {"available": False, "reason": "There are not enough common price observations in the period."}
    holding_returns = p.iloc[-1] / p.iloc[0] - 1.0
    contributions = w * holding_returns
    portfolio_return = float(contributions.sum())
    daily = p.pct_change(fill_method=None).dropna(how="any")
    port_daily = daily @ w
    wealth = (1 + port_daily).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else np.nan
    vol = float(port_daily.std(ddof=1) * np.sqrt(252)) if len(port_daily) > 2 else np.nan
    return {
        "available": True,
        "weight_coverage": coverage,
        "portfolio_return": portfolio_return,
        "max_drawdown": max_dd,
        "annualized_volatility": vol,
        "holding_returns": holding_returns,
        "contributions": contributions,
        "start": p.index.min(),
        "end": p.index.max(),
    }


def hypothetical_stress(weights: pd.Series, metadata: pd.DataFrame, equity_shock: float, tech_shock: float, fixed_income_shock: float) -> pd.DataFrame:
    meta = metadata.set_index("ticker") if metadata is not None and not metadata.empty and "ticker" in metadata else pd.DataFrame()
    rows = []
    for ticker, weight in weights.items():
        sector = str(meta.loc[ticker, "sector"]) if not meta.empty and ticker in meta.index and "sector" in meta.columns else "Unknown"
        asset_class = str(meta.loc[ticker, "asset_class"]) if not meta.empty and ticker in meta.index and "asset_class" in meta.columns else "stock"
        if asset_class == "fixed_income":
            shock = fixed_income_shock
        elif sector == "Information Technology":
            shock = tech_shock
        else:
            shock = equity_shock
        rows.append({
            "ticker": ticker,
            "weight": float(weight),
            "sector": sector,
            "asset_class": asset_class,
            "shock": float(shock),
            "loss_contribution": float(weight) * float(shock),
        })
    return pd.DataFrame(rows)

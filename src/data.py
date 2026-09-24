from __future__ import annotations
from pathlib import Path
from typing import Iterable
import hashlib
import os
import pickle
import time
import numpy as np
import pandas as pd
import certifi

# Make SSL certificate discovery explicit for macOS/Python installs.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())

try:
    import yfinance as yf
except Exception:
    yf = None

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "data" / "us_large_cap_universe.csv"


class DataError(RuntimeError):
    pass


def load_local_universe() -> pd.DataFrame:
    """Load the bundled universe. No web request is made here, so no SSL error is possible."""
    df = pd.read_csv(UNIVERSE_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _cache_file(cache_dir: str | Path, key: str) -> Path:
    p = ROOT / cache_dir if not Path(cache_dir).is_absolute() else Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return p / f"{digest}.pkl"


def _cache_get(cache_dir: str | Path, key: str, ttl_hours: float):
    p = _cache_file(cache_dir, key)
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) / 3600 > ttl_hours:
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_set(cache_dir: str | Path, key: str, value):
    p = _cache_file(cache_dir, key)
    with open(p, "wb") as f:
        pickle.dump(value, f)


def _need_yfinance():
    if yf is None:
        raise DataError("yfinance is not installed. Run: python3 -m pip install -r requirements.txt")


def _statement_row(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    lookup = {str(i).strip().lower(): i for i in df.index}
    row = None
    for alias in aliases:
        if alias.lower() in lookup:
            row = df.loc[lookup[alias.lower()]]
            break
    if row is None:
        for alias in aliases:
            hit = [orig for low, orig in lookup.items() if alias.lower() in low]
            if hit:
                row = df.loc[hit[0]]
                break
    if row is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(row, errors="coerce").dropna()
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        return pd.Series(dtype=float)
    return s.sort_index()


def _latest(s: pd.Series) -> float:
    return float(s.iloc[-1]) if s is not None and len(s.dropna()) else np.nan


def _growth_1y(s: pd.Series) -> float:
    s = s.dropna().sort_index()
    if len(s) < 2:
        return np.nan
    a, b = float(s.iloc[-2]), float(s.iloc[-1])
    if a == 0 or np.sign(a) != np.sign(b):
        return np.nan
    return b / a - 1.0


def _cagr(s: pd.Series, years: int) -> float:
    s = s.dropna().sort_index()
    if len(s) < 2:
        return np.nan
    end_date = s.index[-1]
    candidates = s.index[s.index <= end_date - pd.DateOffset(years=years)]
    if len(candidates) == 0:
        return np.nan
    start_date = candidates[-1]
    a, b = float(s.loc[start_date]), float(s.iloc[-1])
    elapsed = (end_date - start_date).days / 365.25
    if elapsed < years - 0.5 or a <= 0 or b <= 0:
        return np.nan
    return (b / a) ** (1.0 / elapsed) - 1.0


def _safe_num(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _price_risk(prices: pd.Series, beta: float) -> dict:
    p = prices.dropna()
    if len(p) < 40:
        return {"annualized_volatility": np.nan, "max_drawdown": np.nan, "downside_deviation": np.nan, "beta": beta}
    r = p.pct_change().dropna()
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 2 else np.nan
    peak = p.cummax()
    dd = p / peak - 1.0
    max_dd = abs(float(dd.min())) if len(dd) else np.nan
    downside = r[r < 0]
    down_dev = float(downside.std(ddof=1) * np.sqrt(252)) if len(downside) > 2 else np.nan
    return {"annualized_volatility": vol, "max_drawdown": max_dd, "downside_deviation": down_dev, "beta": beta}


def _security_profile(ticker: str, info: dict) -> dict[str, str]:
    """Classify a security before selecting metrics and scoring rules."""
    qt = str(info.get("quoteType", "")).upper()
    text = " ".join(
        str(info.get(k, ""))
        for k in ["category", "fundFamily", "longName", "shortName", "industry", "sector"]
    ).lower()
    if qt == "ETF":
        if any(x in text for x in ["bond", "treasury", "fixed income", "government"]):
            return {"security_type": "ETF", "analysis_profile": "fixed_income_etf", "asset_class": "fixed_income"}
        return {"security_type": "ETF", "analysis_profile": "equity_etf", "asset_class": "equity_etf"}
    if ticker in {"BRK-A", "BRK-B"} or "berkshire hathaway" in text:
        return {"security_type": "stock", "analysis_profile": "financial_conglomerate", "asset_class": "stock"}
    if any(x in text for x in ["bank", "insurance", "financial services", "financial"]):
        return {"security_type": "stock", "analysis_profile": "financial_company", "asset_class": "stock"}
    return {"security_type": "stock", "analysis_profile": "operating_company", "asset_class": "stock"}


def analyze_security(ticker: str, cfg: dict, force_refresh: bool = False) -> tuple[dict, pd.Series]:
    _need_yfinance()
    ticker = ticker.strip().upper().replace(".", "-")
    cache_dir = cfg["data"]["cache_dir"]
    ttl = float(cfg["data"]["cache_ttl_hours"])
    key = f"security::{ticker}"
    if not force_refresh:
        cached = _cache_get(cache_dir, key, ttl)
        if cached is not None:
            return cached

    t = yf.Ticker(ticker)
    try:
        info = t.get_info() or {}
    except Exception:
        try:
            info = t.info or {}
        except Exception:
            info = {}

    def get_df(names):
        for name in names:
            try:
                df = getattr(t, name)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df.copy()
            except Exception:
                pass
        return pd.DataFrame()

    income = get_df(["income_stmt", "financials"])
    cashflow = get_df(["cashflow", "cash_flow"])

    try:
        hist = t.history(period="10y", auto_adjust=True, actions=False)
    except Exception as e:
        raise DataError(f"Could not download price history for {ticker}: {e}") from e
    if hist is None or hist.empty:
        raise DataError(f"No usable price history returned for {ticker}.")
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna().rename(ticker)

    revenue = _statement_row(income, ["Total Revenue", "Operating Revenue"])
    net_income = _statement_row(income, ["Net Income", "Net Income Common Stockholders"])
    eps = _statement_row(income, ["Diluted EPS", "Basic EPS"])
    op_income = _statement_row(income, ["Operating Income"])
    fcf = _statement_row(cashflow, ["Free Cash Flow"])
    if fcf.empty:
        cfo = _statement_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = _statement_row(cashflow, ["Capital Expenditure", "Capital Expenditures"])
        aligned = pd.concat([cfo.rename("cfo"), capex.rename("capex")], axis=1).dropna()
        if not aligned.empty:
            fcf = aligned["cfo"] - aligned["capex"].abs()

    latest_rev = _latest(revenue)
    latest_net = _latest(net_income)
    latest_op = _latest(op_income)
    latest_fcf = _latest(fcf)

    op_margin = _safe_num(info.get("operatingMargins"))
    if not np.isfinite(op_margin) and np.isfinite(latest_op) and np.isfinite(latest_rev) and latest_rev != 0:
        op_margin = latest_op / latest_rev
    net_margin = _safe_num(info.get("profitMargins"))
    if not np.isfinite(net_margin) and np.isfinite(latest_net) and np.isfinite(latest_rev) and latest_rev != 0:
        net_margin = latest_net / latest_rev

    de = _safe_num(info.get("debtToEquity"))
    if np.isfinite(de) and de > 10:
        de /= 100.0

    pe = _safe_num(info.get("trailingPE"))
    fpe = _safe_num(info.get("forwardPE"))
    ps = _safe_num(info.get("priceToSalesTrailing12Months"))
    pb = _safe_num(info.get("priceToBook"))
    eve = _safe_num(info.get("enterpriseToEbitda"))
    # Negative valuation multiples are not interpreted as attractive.
    pe = pe if pe > 0 else np.nan
    fpe = fpe if fpe > 0 else np.nan
    eve = eve if eve > 0 else np.nan

    profile = _security_profile(ticker, info)
    risk = _price_risk(close, _safe_num(info.get("beta") or info.get("beta3Year")))
    metrics = {
        "ticker": ticker,
        "company": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector") or "Unknown",
        "industry": info.get("industry") or "Unknown",
        **profile,
        "current_price": float(close.iloc[-1]),
        "revenue": latest_rev,
        "net_income": latest_net,
        "free_cash_flow": latest_fcf,
        "revenue_growth_1y": _growth_1y(revenue),
        "revenue_growth_3y": _cagr(revenue, 3),
        "revenue_growth_5y": _cagr(revenue, 5),
        "eps_growth_1y": _growth_1y(eps),
        "eps_growth_3y": _cagr(eps, 3),
        "eps_growth_5y": _cagr(eps, 5),
        "fcf_growth_1y": _growth_1y(fcf),
        "fcf_growth_3y": _cagr(fcf, 3),
        "fcf_growth_5y": _cagr(fcf, 5),
        "operating_margin": op_margin,
        "net_margin": net_margin,
        "return_on_equity": _safe_num(info.get("returnOnEquity")),
        "current_ratio": _safe_num(info.get("currentRatio")),
        "debt_to_equity": de,
        "trailing_pe": pe,
        "forward_pe": fpe,
        "price_to_sales": ps,
        "price_to_book": pb,
        "ev_to_ebitda": eve,
        # Fund-level fields are populated for ETFs where Yahoo provides them.
        "expense_ratio": _safe_num(info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")),
        "total_assets": _safe_num(info.get("totalAssets")),
        "holdings_count": _safe_num(info.get("numberOfHoldings")),
        "portfolio_pe": _safe_num(info.get("trailingPE")),
        "portfolio_pb": _safe_num(info.get("priceToBook")),
        "distribution_yield": _safe_num(info.get("yield") or info.get("dividendYield")),
        "dividend_yield": _safe_num(info.get("dividendYield")),
        **risk,
    }
    result = (metrics, close)
    _cache_set(cache_dir, key, result)
    return result


def etf_holdings(ticker: str, cfg: dict, force_refresh: bool = False) -> pd.DataFrame:
    """Return Yahoo's published holdings for an ETF when available."""
    _need_yfinance()
    ticker = ticker.strip().upper().replace(".", "-")
    key = f"holdings::{ticker}"
    if not force_refresh:
        cached = _cache_get(cfg["data"]["cache_dir"], key, float(cfg["data"]["cache_ttl_hours"]))
        if cached is not None:
            return cached

    fund = yf.Ticker(ticker)
    holdings = pd.DataFrame()
    try:
        get_holdings = getattr(fund, "get_holdings", None)
        if callable(get_holdings):
            holdings = get_holdings()
    except Exception:
        holdings = pd.DataFrame()
    if holdings is None or holdings.empty:
        try:
            fund_data = fund.funds_data
            holdings = getattr(fund_data, "top_holdings", pd.DataFrame())
        except Exception:
            holdings = pd.DataFrame()
    if holdings is None or holdings.empty:
        return pd.DataFrame()

    out = holdings.reset_index()
    rename = {}
    for column in out.columns:
        normalized = str(column).strip().lower().replace("_", "").replace(" ", "")
        if normalized in {"symbol", "ticker"}:
            rename[column] = "ticker"
        elif normalized in {"holdingname", "name", "longname"}:
            rename[column] = "holding"
        elif "holdingpercent" in normalized or normalized in {"percent", "weight", "holdingweight"}:
            rename[column] = "weight"
    out = out.rename(columns=rename)
    if "ticker" not in out.columns:
        out.insert(0, "ticker", "")
    if "holding" not in out.columns:
        out.insert(1, "holding", "")
    if "weight" in out.columns:
        out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
        if out["weight"].dropna().max() > 1:
            out["weight"] = out["weight"] / 100.0
    else:
        out["weight"] = np.nan
    result = out[["ticker", "holding", "weight"]].copy()
    _cache_set(cfg["data"]["cache_dir"], key, result)
    return result


def batch_analyze(tickers: Iterable[str], cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, str]]:
    rows, prices, errors = [], {}, {}
    for ticker in list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip())):
        try:
            m, p = analyze_security(ticker, cfg)
            rows.append(m)
            prices[ticker] = p
        except Exception as e:
            errors[ticker] = str(e)
    return pd.DataFrame(rows), prices, errors


def price_frame(price_map: dict[str, pd.Series]) -> pd.DataFrame:
    if not price_map:
        return pd.DataFrame()
    return pd.concat(price_map.values(), axis=1).sort_index()


def download_price_period(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    _need_yfinance()
    clean = [t.strip().upper().replace(".", "-") for t in tickers if t.strip()]
    if not clean:
        return pd.DataFrame()
    try:
        raw = yf.download(clean, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    except Exception as e:
        raise DataError(f"Historical stress download failed: {e}") from e
    if raw is None or raw.empty:
        return pd.DataFrame()
    if len(clean) == 1:
        if "Close" in raw:
            return raw[["Close"]].rename(columns={"Close": clean[0]})
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0):
        close = raw["Close"].copy()
        return close[[c for c in clean if c in close.columns]]
    return pd.DataFrame()

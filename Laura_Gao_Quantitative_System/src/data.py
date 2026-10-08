from __future__ import annotations
from pathlib import Path
from typing import Iterable
import hashlib
import os
import pickle
import re
import time
import numpy as np
import pandas as pd
import certifi
from src.sectors import laura_sector_bucket, normalize_sector, normalize_sector_weights

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


def normalize_ticker(ticker: str) -> str:
    """Normalize user tickers to Yahoo Finance symbols."""
    return str(ticker).strip().upper().replace(".", "-")


FIXED_INCOME_ETFS = {
    "BIL", "SGOV", "SHY", "IEF", "TLT", "GOVT", "BND", "AGG",
}
EQUITY_ETFS = {"VOO", "QQQ", "VTI", "SPY"}
STANDARD_OPERATING_OVERRIDES = {
    "V": "Payment network; standard corporate FCF metrics retained",
    "MA": "Payment network; standard corporate FCF metrics retained",
    "PYPL": "Payment processor; standard corporate FCF metrics retained",
    "FI": "Payment processor; standard corporate FCF metrics retained",
    "GPN": "Payment processor; standard corporate FCF metrics retained",
}
SPECIAL_FINANCIAL_REASONS = {
    "JPM": "Bank balance-sheet accounting",
    "BAC": "Bank balance-sheet accounting",
    "WFC": "Bank balance-sheet accounting",
    "USB": "Bank balance-sheet accounting",
    "C": "Bank balance-sheet accounting",
    "GS": "Broker-dealer and investment-bank balance-sheet accounting",
    "MS": "Broker-dealer and investment-bank balance-sheet accounting",
    "SCHW": "Brokerage and lending balance-sheet accounting",
    "PNC": "Bank balance-sheet accounting",
    "BRK-A": "Insurance and financial conglomerate accounting",
    "BRK-B": "Insurance and financial conglomerate accounting",
    "ALL": "Insurance balance-sheet accounting",
    "MET": "Insurance balance-sheet accounting",
    "PRU": "Insurance balance-sheet accounting",
    "AIG": "Insurance balance-sheet accounting",
    "AXP": "Lending and card balance-sheet accounting",
    "COF": "Lending and card balance-sheet accounting",
    "DFS": "Lending and card balance-sheet accounting",
    "SYF": "Lending and card balance-sheet accounting",
    "ALLY": "Lending balance-sheet accounting",
    "BLK": "Asset-management financial-company accounting",
}
SPECIAL_FINANCIAL_INDUSTRY_TERMS = (
    "banks -",
    "banking",
    "insurance",
    "property & casualty",
    "life insurance",
    "capital markets",
    "asset management",
    "brokerage",
    "consumer finance",
    "mortgage finance",
)


def reserve_asset_profile(ticker: str) -> str:
    """Return the conservative reserve classification for known symbols."""
    symbol = normalize_ticker(ticker)
    if symbol in FIXED_INCOME_ETFS:
        return "fixed_income_etf"
    if symbol in EQUITY_ETFS:
        return "equity_etf"
    return "unknown"


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


def _statement_row_details(
    df: pd.DataFrame,
    aliases: list[str],
) -> tuple[pd.Series, str, str]:
    if df is None or df.empty:
        return pd.Series(dtype=float), "", "Statement unavailable or empty"

    def match(labels) -> tuple[object | None, str]:
        def normalized(text: object) -> str:
            return re.sub(r"[^a-z0-9]", "", str(text).lower())

        lookup = {normalized(label): label for label in labels}
        for alias in aliases:
            if normalized(alias) in lookup:
                return lookup[normalized(alias)], "exact"
        for alias in aliases:
            alias_key = normalized(alias)
            hit = [label for low, label in lookup.items() if alias_key in low]
            if hit:
                return hit[0], "partial"
        return None, ""

    field, match_type = match(df.index)
    transposed = False
    if field is not None:
        row = df.loc[field]
    else:
        field, match_type = match(df.columns)
        if field is None:
            return pd.Series(dtype=float), "", f"None of the requested statement rows found: {', '.join(aliases)}"
        row = df[field]
        transposed = True

    values = pd.to_numeric(row, errors="coerce")
    valid_pairs = []
    for period, value in values.items():
        try:
            date = pd.to_datetime(period, errors="raise")
        except (TypeError, ValueError, OverflowError):
            continue
        if pd.notna(value) and np.isfinite(float(value)):
            valid_pairs.append((date, float(value)))
    if not valid_pairs:
        return pd.Series(dtype=float), str(field), "Matched statement field has no valid dated numeric observations"
    result = pd.Series(
        [value for _, value in valid_pairs],
        index=pd.DatetimeIndex([date for date, _ in valid_pairs]),
        dtype=float,
    ).groupby(level=0).last().sort_index()
    orientation = "; transposed statement orientation" if transposed else ""
    match_note = "exact field match" if match_type == "exact" else "named fallback field"
    return result, str(field), f"{match_note}{orientation}"


def _statement_row(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    return _statement_row_details(df, aliases)[0]


def _first_statement_source(
    ticker_obj,
    method_name: str,
    frequency: str,
    attributes: list[str],
) -> tuple[pd.DataFrame, str, str]:
    errors = []
    method = getattr(ticker_obj, method_name, None)
    if callable(method):
        try:
            frame = method(freq=frequency)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                return frame.copy(), f"{method_name}(freq={frequency!r})", ""
            errors.append(f"{method_name}(freq={frequency!r}) returned no rows")
        except Exception as exc:
            errors.append(f"{method_name}(freq={frequency!r}) failed: {type(exc).__name__}: {exc}")
    else:
        errors.append(f"{method_name} is unavailable")
    for attribute in attributes:
        try:
            frame = getattr(ticker_obj, attribute)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                return frame.copy(), attribute, ""
            errors.append(f"{attribute} returned no rows")
        except Exception as exc:
            errors.append(f"{attribute} failed: {type(exc).__name__}: {exc}")
    return pd.DataFrame(), "", "; ".join(errors)


def _quarterly_ttm(series: pd.Series) -> tuple[float, str]:
    quarterly = series.dropna().sort_index()
    if len(quarterly) < 4:
        return np.nan, "Fewer than four valid quarterly observations for TTM"
    latest = quarterly.iloc[-4:]
    span = _elapsed_calendar_years(latest.index[0], latest.index[-1])
    if not .60 <= span <= 1.40:
        return np.nan, f"Latest four quarterly observations span an invalid TTM interval ({span:.2f} years)"
    return float(latest.sum()), f"Sum of four reported quarterly observations ({latest.index[0].date()} to {latest.index[-1].date()})"


def _statement_ttm_value(
    ttm_frame: pd.DataFrame,
    quarterly_frame: pd.DataFrame,
    aliases: list[str],
) -> tuple[float, str, str, str]:
    ttm_series, ttm_field, ttm_reason = _statement_row_details(ttm_frame, aliases)
    ttm_value = _latest(ttm_series)
    if np.isfinite(ttm_value):
        return ttm_value, ttm_field, "TTM statement", ttm_reason
    quarterly, quarterly_field, quarterly_reason = _statement_row_details(quarterly_frame, aliases)
    value, note = _quarterly_ttm(quarterly)
    if np.isfinite(value):
        return value, quarterly_field, "Quarterly sum", note
    reason = "; ".join(part for part in (ttm_reason, quarterly_reason, note) if part)
    return np.nan, ttm_field or quarterly_field, "Unavailable", reason


def _latest(s: pd.Series) -> float:
    return float(s.iloc[-1]) if s is not None and len(s.dropna()) else np.nan


def _elapsed_calendar_years(start: pd.Timestamp, end: pd.Timestamp) -> float:
    whole_years = end.year - start.year
    anniversary = start + pd.DateOffset(years=whole_years)
    next_anniversary = anniversary + pd.DateOffset(years=1)
    remainder = (end - anniversary).total_seconds()
    year_seconds = (next_anniversary - anniversary).total_seconds()
    return float(whole_years + remainder / year_seconds) if year_seconds > 0 else float(whole_years)


def _cagr_details(s: pd.Series) -> tuple[float, dict]:
    s = s.dropna().sort_index()
    if len(s) < 2:
        return np.nan, {
            "valid_observations": len(s),
            "start_date": str(s.index[0].date()) if len(s) else "",
            "end_date": str(s.index[-1].date()) if len(s) else "",
            "start_value": float(s.iloc[0]) if len(s) else np.nan,
            "end_value": float(s.iloc[-1]) if len(s) else np.nan,
            "elapsed_years": 0.0,
            "reason": "Insufficient valid annual observations for a 3-year CAGR",
        }
    end_date = s.index[-1]
    elapsed_by_date = {
        date: float((end_date - date).days / 365.25)
        for date in s.index[:-1]
    }
    candidates = [
        date for date, elapsed in elapsed_by_date.items()
        if 2.5 <= elapsed <= 3.5
    ]
    if len(candidates) == 0:
        elapsed = float((end_date - s.index[0]).days / 365.25)
        return np.nan, {
            "valid_observations": len(s),
            "start_date": str(s.index[0].date()),
            "end_date": str(end_date.date()),
            "start_value": float(s.iloc[0]),
            "end_value": float(s.iloc[-1]),
            "elapsed_years": elapsed,
            "reason": f"Insufficient valid annual history for a 3-year CAGR; observations span {elapsed:.2f} years",
        }
    start_date = min(candidates, key=lambda date: abs(elapsed_by_date[date] - 3.0))
    a, b = float(s.loc[start_date]), float(s.iloc[-1])
    elapsed = elapsed_by_date[start_date]
    details = {
        "valid_observations": len(s),
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "start_value": a,
        "end_value": b,
        "elapsed_years": elapsed,
        "reason": "",
    }
    if not 2.5 <= elapsed <= 3.5:
        details["reason"] = f"Insufficient valid annual history for a 3-year CAGR; endpoints span {elapsed:.2f} years"
        return np.nan, details
    if a <= 0 or b <= 0:
        details["reason"] = "Invalid sign for CAGR: beginning and ending values must both be positive"
        return np.nan, details
    return (b / a) ** (1.0 / elapsed) - 1.0, details


def _cagr(s: pd.Series) -> float:
    return _cagr_details(s)[0]


def _historical_price_return(prices: pd.Series, years: int, cagr: bool = False) -> float:
    """Return adjusted-price return or a 1Y/3Y fixed-horizon CAGR as a decimal."""
    if years not in {1, 3}:
        raise ValueError("Only one- and three-year price horizons are supported.")
    return _historical_price_return_details(prices, years, cagr)[0]


def _historical_price_return_details(
    prices: pd.Series,
    years: int,
    cagr: bool = False,
) -> tuple[float, dict]:
    if years not in {1, 3}:
        raise ValueError("Only one- and three-year price horizons are supported.")
    series = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if len(series) < 2:
        return np.nan, {
            "start_date": "",
            "end_date": "",
            "elapsed_years": np.nan,
            "reason": "Insufficient adjusted-price history",
        }
    end_date = series.index[-1]
    cutoff = end_date - pd.DateOffset(years=years)
    prior = series.loc[series.index <= cutoff]
    if prior.empty:
        return np.nan, {
            "start_date": "",
            "end_date": str(end_date.date()),
            "elapsed_years": np.nan,
            "reason": f"Insufficient adjusted-price history for {years}Y lookback",
        }
    start_date = prior.index[-1]
    start_price, end_price = float(prior.iloc[-1]), float(series.iloc[-1])
    if start_price <= 0 or end_price <= 0:
        return np.nan, {
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "elapsed_years": _elapsed_calendar_years(start_date, end_date),
            "reason": "Price return invalid: start and end prices must be positive",
        }
    growth = end_price / start_price
    elapsed_years = _elapsed_calendar_years(start_date, end_date)
    if not cagr:
        return float(growth - 1.0), {
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "elapsed_years": elapsed_years,
            "reason": "",
        }
    if elapsed_years <= 0:
        return np.nan, {
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "elapsed_years": elapsed_years,
            "reason": "CAGR invalid: price dates do not span a positive interval",
        }
    return float(growth ** (1.0 / elapsed_years) - 1.0), {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "elapsed_years": elapsed_years,
        "reason": "",
    }


def _return_sanity_warnings(returns: dict[str, float]) -> list[str]:
    labels = {
        "total_return_1y": "1Y Return",
        "total_return_3y": "3Y CAGR",
    }
    return [
        f"{labels[key]} is {value:.1%}; verify price data and units (threshold: ±500%)."
        for key, value in returns.items()
        if key in labels and np.isfinite(value) and abs(value) > 5.0
    ]


def _safe_num(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _fast_info_values(ticker_obj) -> tuple[dict, str]:
    try:
        value = ticker_obj.fast_info
        return (dict(value) if value is not None else {}), ""
    except Exception as exc:
        return {}, f"fast_info failed: {type(exc).__name__}: {exc}"


def _valuation_measure(frame: pd.DataFrame, aliases: list[str]) -> tuple[float, str]:
    if frame is None or frame.empty:
        return np.nan, "Valuation measure unavailable"
    rows = {str(label).strip().lower(): label for label in frame.index}
    matched = None
    for alias in aliases:
        if alias.lower() in rows:
            matched = rows[alias.lower()]
            break
    if matched is None:
        for alias in aliases:
            matched = next(
                (label for lowered, label in rows.items() if alias.lower() in lowered),
                None,
            )
            if matched is not None:
                break
    if matched is None:
        return np.nan, f"None of the valuation-measure rows found: {', '.join(aliases)}"
    row = frame.loc[matched]
    preferred = row.get("Current", np.nan) if isinstance(row, pd.Series) else np.nan
    value = _safe_num(preferred)
    if not np.isfinite(value):
        return np.nan, f"Valuation measure {matched} has no numeric Current value; historical values were not substituted"
    return value, f"get_valuation_measures[{matched!r}, 'Current']"


def _market_cap_details(info: dict, fast_info: dict, valuation_measures: pd.DataFrame, current_price: float) -> tuple[float, str, str]:
    for field in ("market_cap", "marketCap"):
        value = _safe_num(fast_info.get(field))
        if np.isfinite(value) and value > 0:
            return value, f"fast_info.{field}", ""
    value = _safe_num(info.get("marketCap"))
    if np.isfinite(value) and value > 0:
        return value, "info.marketCap", ""
    value, valuation_source = _valuation_measure(valuation_measures, ["Market Cap", "Market Capitalization"])
    if np.isfinite(value) and value > 0:
        return value, valuation_source, ""
    shares = _safe_num(info.get("sharesOutstanding"))
    if np.isfinite(current_price) and current_price > 0 and np.isfinite(shares) and shares > 0:
        return current_price * shares, "Derived current price × info.sharesOutstanding", ""
    return np.nan, "Unavailable", (
        "Market cap unavailable from fast_info, info, and valuation measures; "
        "price/share-count fallback requires positive current price and sharesOutstanding"
    )


def _current_price_details(
    ticker_obj,
    info: dict,
    close: pd.Series,
    history_error: str = "",
    fast_info: dict | None = None,
) -> dict:
    failures = []
    if fast_info is None:
        fast_info, fast_error = _fast_info_values(ticker_obj)
        if fast_error:
            failures.append(fast_error)

    fast_price = _safe_num(fast_info.get("last_price"))
    if np.isfinite(fast_price) and fast_price > 0:
        raw_time = fast_info.get("last_price_time") or fast_info.get("regular_market_time")
        try:
            as_of = pd.to_datetime(raw_time, unit="s", utc=True).tz_convert(
                str(fast_info.get("timezone") or "UTC")
            ).isoformat() if raw_time is not None else ""
        except (TypeError, ValueError, OverflowError):
            as_of = ""
        state = str(info.get("marketState") or "unknown").replace("_", " ").lower()
        freshness = (
            "Market quote; Yahoo reports the market as open (quote delay not specified)"
            if state in {"regular", "open"}
            else f"Last price from fast_info; market state {state}"
        )
        return {
            "current_price": fast_price,
            "price_currency": str(fast_info.get("currency") or info.get("currency") or "N/A"),
            "price_as_of": as_of or str(info.get("regularMarketTime") or "N/A"),
            "price_source": "yfinance fast_info.last_price",
            "price_freshness": freshness,
            "price_diagnostic": "Available from preferred yfinance fast_info source",
        }
    failures.append("yfinance fast_info.last_price missing or non-positive")

    for field in ("currentPrice", "regularMarketPrice"):
        value = _safe_num(info.get(field))
        if np.isfinite(value) and value > 0:
            market_time = info.get("regularMarketTime")
            try:
                as_of = pd.to_datetime(market_time, unit="s", utc=True).isoformat() if market_time else ""
            except (TypeError, ValueError, OverflowError):
                as_of = ""
            state = str(info.get("marketState") or "unknown").replace("_", " ").lower()
            status = (
                "Yahoo market quote; market is open, quote delay not specified"
                if state in {"regular", "open"}
                else f"Yahoo last market quote; market state {state}"
            )
            return {
                "current_price": value,
                "price_currency": str(info.get("currency") or "N/A"),
                "price_as_of": as_of or "N/A",
                "price_source": f"yfinance info.{field}",
                "price_freshness": status,
                "price_diagnostic": f"Available from Yahoo info field {field}",
            }
    failures.append("Yahoo info currentPrice and regularMarketPrice missing or non-positive")

    if len(close.dropna()):
        value = _safe_num(close.dropna().iloc[-1])
        if np.isfinite(value) and value > 0:
            date = close.dropna().index[-1]
            return {
                "current_price": value,
                "price_currency": str(info.get("currency") or "N/A"),
                "price_as_of": str(date),
                "price_source": "Latest valid adjusted Close from yfinance history",
                "price_freshness": "Latest available close; not represented as a live quote",
                "price_diagnostic": "Used adjusted-close fallback because quote fields were unavailable",
            }
    failures.append(history_error or "No valid positive adjusted Close in history")
    return {
        "current_price": np.nan,
        "price_currency": str(info.get("currency") or "N/A"),
        "price_as_of": "N/A",
        "price_source": "Yahoo fast_info, ticker info, and adjusted history",
        "price_freshness": "Unavailable",
        "price_diagnostic": "; ".join(failures),
    }


def _expense_ratio_details(
    info: dict,
    fund_operations: pd.DataFrame | None = None,
) -> tuple[float, float, str]:
    """Return decimal expense ratio, raw Yahoo value, and the source field."""
    if fund_operations is not None and not fund_operations.empty:
        row_lookup = {str(index).strip().lower(): index for index in fund_operations.index}
        ratio_row = next(
            (
                row_lookup[name]
                for name in ("annual report expense ratio", "net expense ratio", "gross expense ratio")
                if name in row_lookup
            ),
            None,
        )
        if ratio_row is not None:
            row = fund_operations.loc[ratio_row]
            preferred_column = next(
                (
                    column for column in fund_operations.columns
                    if str(column).lower() not in {"category average", "attributes"}
                ),
                None,
            )
            raw = _safe_num(row.get(preferred_column)) if preferred_column is not None else np.nan
            if np.isfinite(raw) and raw >= 0:
                return raw, raw, f"fund_operations[{ratio_row!r}, {preferred_column!r}] (decimal ratio)"

    annual_report_ratio = _safe_num(info.get("annualReportExpenseRatio"))
    if np.isfinite(annual_report_ratio) and annual_report_ratio >= 0:
        return annual_report_ratio / 100.0, annual_report_ratio, "annualReportExpenseRatio (percentage points)"
    net_ratio = _safe_num(info.get("netExpenseRatio"))
    if np.isfinite(net_ratio) and net_ratio >= 0:
        return net_ratio / 100.0, net_ratio, "netExpenseRatio (percentage points)"
    return np.nan, np.nan, "Unavailable"


def _expense_ratio_fraction(
    info: dict,
    fund_operations: pd.DataFrame | None = None,
) -> float:
    return _expense_ratio_details(info, fund_operations)[0]


def _interest_coverage_details(
    op_income: pd.Series,
    income: pd.DataFrame,
    info: dict,
    ebit_override: float = np.nan,
    interest_override: float = np.nan,
    override_source: str = "",
) -> tuple[float, str]:
    ebit = ebit_override
    interest = interest_override
    if np.isfinite(ebit) and np.isfinite(interest):
        source = override_source or "TTM EBIT / absolute TTM interest expense"
    else:
        interest_expense, interest_field, interest_reason = _statement_row_details(
            income,
            ["Interest Expense", "Interest Expense Non Operating", "Interest And Debt Expense"],
        )
        paired = pd.concat(
            [op_income.rename("ebit"), interest_expense.rename("interest")], axis=1
        ).dropna().sort_index()
        if not paired.empty:
            ebit = float(paired.iloc[-1]["ebit"])
            interest = float(paired.iloc[-1]["interest"])
            source = f"Matched annual EBIT / interest expense ({interest_field}; period ending {paired.index[-1].date()})"
        else:
            ebit = interest = np.nan
            source = f"Annual EBIT/interest periods unavailable: {interest_reason}"
    if np.isfinite(ebit) and np.isfinite(interest):
        interest = abs(interest)
        near_zero = max(1.0, abs(ebit) * 1e-8)
        if interest <= near_zero:
            if ebit > 0:
                return 15.0, f"Derived {source}; expense at or below $1 is capped at 15x"
            if ebit < 0:
                return 0.0, f"Derived {source}; near-zero expense and negative EBIT"
            return np.nan, f"N/A: zero EBIT and near-zero interest expense ({source})"
        return ebit / interest, f"Derived {source}"
    reported = _safe_num(info.get("interestCoverage"))
    return (
        (reported, "Yahoo-reported interestCoverage; statement inputs unavailable")
        if np.isfinite(reported)
        else (np.nan, f"N/A: matching-period EBIT and interest expense unavailable; {source}")
    )


def _interest_coverage(op_income: pd.Series, income: pd.DataFrame, info: dict) -> float:
    return _interest_coverage_details(op_income, income, info)[0]


def _fund_attributes(ticker_obj) -> dict:
    result = {
        "sector_weightings": {},
        "bond_ratings": {},
        "fund_overview": {},
        "fund_operations": pd.DataFrame(),
        "asset_classes": {},
        "error": "",
    }
    try:
        data = ticker_obj.funds_data
        for attr, key in (
            ("sector_weightings", "sector_weightings"),
            ("bond_ratings", "bond_ratings"),
            ("fund_overview", "fund_overview"),
            ("fund_operations", "fund_operations"),
            ("asset_classes", "asset_classes"),
        ):
            try:
                value = getattr(data, attr, {})
            except Exception as exc:
                result["error"] += f"{attr} failed: {type(exc).__name__}: {exc}; "
                continue
            if isinstance(value, dict):
                result[key] = value
            elif isinstance(value, pd.DataFrame):
                result[key] = value.copy()
    except Exception as exc:
        result["error"] = f"Yahoo funds_data retrieval failed: {exc}"
    return result


def _benchmark_sector_data(
    benchmark: str,
    cfg: dict,
    cache_dir: str,
    ttl_hours: float,
) -> tuple[dict[str, float], str, str]:
    benchmark = normalize_ticker(benchmark)
    cache_key = f"sector-benchmark-v1::{benchmark}"
    cached = _cache_get(cache_dir, cache_key, ttl_hours)
    if isinstance(cached, dict):
        weights = normalize_sector_weights(cached)
        if weights:
            return weights, f"Yahoo {benchmark} fund sector data (cached)", ""

    retrieval_error = ""
    try:
        attributes = _fund_attributes(yf.Ticker(benchmark))
        weights = normalize_sector_weights(attributes.get("sector_weightings"))
        if weights:
            _cache_set(cache_dir, cache_key, weights)
            return weights, f"Yahoo {benchmark} fund sector data", ""
        retrieval_error = attributes.get("error") or f"{benchmark} sector weights unavailable or invalid"
    except Exception as exc:
        retrieval_error = f"{benchmark} sector retrieval failed: {type(exc).__name__}: {exc}"

    configured = normalize_sector_weights(cfg["optimizer"].get("sector_benchmark_weights", {}))
    if configured:
        return configured, "Configured S&P 500 sector reference (Yahoo data unavailable)", retrieval_error
    return {}, "Unavailable", retrieval_error or "No valid configured sector benchmark reference"


def _credit_score_from_exposure(ratings: dict, cfg: dict) -> float:
    if not ratings:
        return np.nan
    ratings = {str(key).lower(): value for key, value in ratings.items()}
    government = _safe_num(ratings.get("us_government"))
    anchors = cfg["scoring"]["fixed_income"]["credit_rating_scores"]
    reported_weights = [_safe_num(value) for value in ratings.values()]
    if any(np.isfinite(weight) and weight < 0 for weight in reported_weights):
        return np.nan
    recognized_keys = {"us_government", *anchors}
    if any(
        str(key).lower() not in recognized_keys | {"other"}
        and np.isfinite(_safe_num(value))
        and float(value) > 0
        for key, value in ratings.items()
    ):
        return np.nan
    reported_total = sum(weight for weight in reported_weights if np.isfinite(weight))
    scale = 100.0 if 99.0 <= reported_total <= 101.0 else 1.0
    government = government / scale if np.isfinite(government) else np.nan
    rating_weights = {
        rating: _safe_num(ratings.get(rating)) / scale
        for rating in anchors
        if np.isfinite(_safe_num(ratings.get(rating)))
        and _safe_num(ratings.get(rating)) > 0
    }
    non_government_total = sum(rating_weights.values())
    if np.isfinite(government) and government > 0:
        if (
            government < 1.0
            and non_government_total > 0
            and np.isclose(non_government_total, 1.0, atol=.02)
            and government + non_government_total > 1.01
        ):
            rating_weights = {
                rating: weight * (1.0 - government) / non_government_total
                for rating, weight in rating_weights.items()
            }
        valid_total = government + sum(rating_weights.values())
        if not np.isclose(valid_total, 1.0, atol=.02):
            return np.nan
    elif not (
        .99 <= non_government_total <= 1.01
        or 99.0 <= non_government_total <= 101.0
    ):
        return np.nan
    observations = []
    if np.isfinite(government) and government > 0:
        observations.append((100.0, government))
    for rating, points in anchors.items():
        weight = rating_weights.get(rating, 0.0)
        if np.isfinite(weight) and weight > 0:
            observations.append((points, weight))
    total = sum(weight for _, weight in observations)
    return sum(points * weight for points, weight in observations) / total if total > 0 else np.nan


def _price_risk(prices: pd.Series, beta: float) -> dict:
    p = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if not p.empty:
        p = p.loc[p.index >= p.index[-1] - pd.DateOffset(years=3)]
    period_start = str(p.index[0].date()) if len(p) else ""
    period_end = str(p.index[-1].date()) if len(p) else ""
    observations = len(p)
    if len(p) < 40:
        return {
            "annualized_volatility": np.nan,
            "max_drawdown": np.nan,
            "downside_deviation": np.nan,
            "beta": beta,
            "risk_period_start": period_start,
            "risk_period_end": period_end,
            "risk_price_observations": observations,
        }
    r = p.pct_change(fill_method=None).dropna()
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 2 else np.nan
    peak = p.cummax()
    dd = p / peak - 1.0
    max_dd = abs(float(dd.min())) if len(dd) else np.nan
    downside = r.clip(upper=0.0)
    down_dev = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252)) if len(downside) else np.nan
    return {
        "annualized_volatility": vol,
        "max_drawdown": max_dd,
        "downside_deviation": down_dev,
        "beta": beta,
        "risk_period_start": period_start,
        "risk_period_end": period_end,
        "risk_price_observations": observations,
    }


def _market_beta(prices: pd.Series, market_prices: pd.Series, years: int = 3) -> float:
    asset = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    market = pd.to_numeric(market_prices, errors="coerce").dropna().sort_index()
    if asset.empty or market.empty:
        return np.nan
    end_date = min(asset.index[-1], market.index[-1])
    cutoff = end_date - pd.DateOffset(years=years)
    aligned = pd.concat(
        [
            asset.loc[asset.index >= cutoff].pct_change(fill_method=None).rename("asset"),
            market.loc[market.index >= cutoff].pct_change(fill_method=None).rename("market"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 60 or aligned["market"].var(ddof=1) <= 0:
        return np.nan
    return float(aligned["asset"].cov(aligned["market"]) / aligned["market"].var(ddof=1))


def _derive_free_cash_flow(
    operating_cash_flow: pd.Series,
    capital_expenditure: pd.Series,
) -> pd.Series:
    aligned = pd.concat(
        [
            operating_cash_flow.rename("operating_cash_flow"),
            capital_expenditure.rename("capital_expenditure"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    capex = aligned["capital_expenditure"]
    return pd.Series(
        np.where(capex < 0, aligned["operating_cash_flow"] + capex, aligned["operating_cash_flow"] - capex),
        index=aligned.index,
        dtype=float,
    )


def _paired_ttm_values(
    ttm_frame: pd.DataFrame,
    quarterly_frame: pd.DataFrame,
    first_aliases: list[str],
    second_aliases: list[str],
) -> tuple[float, float, str, str, str]:
    """Return two values from one matching TTM or four-quarter accounting period."""
    first, first_field, first_reason = _statement_row_details(ttm_frame, first_aliases)
    second, second_field, second_reason = _statement_row_details(ttm_frame, second_aliases)
    paired = pd.concat(
        [first.rename("first"), second.rename("second")], axis=1
    ).dropna()
    if not paired.empty:
        date = paired.index[-1]
        return (
            float(paired.iloc[-1]["first"]),
            float(paired.iloc[-1]["second"]),
            first_field,
            second_field,
            f"Matched TTM statement period ending {date.date()}",
        )

    first, first_field, first_reason = _statement_row_details(quarterly_frame, first_aliases)
    second, second_field, second_reason = _statement_row_details(quarterly_frame, second_aliases)
    paired = pd.concat(
        [first.rename("first"), second.rename("second")], axis=1
    ).dropna().sort_index()
    if len(paired) >= 4:
        latest = paired.iloc[-4:]
        span = _elapsed_calendar_years(latest.index[0], latest.index[-1])
        if .60 <= span <= 1.40:
            return (
                float(latest["first"].sum()),
                float(latest["second"].sum()),
                first_field,
                second_field,
                f"Matched four-quarter period {latest.index[0].date()} to {latest.index[-1].date()}",
            )
    reason = "; ".join(
        part for part in (first_reason, second_reason, "No matching TTM/four-quarter dates") if part
    )
    return np.nan, np.nan, first_field, second_field, reason


def classify_asset(ticker: str, info: dict | None = None) -> str:
    """Return the single scoring model used across the application."""
    ticker = normalize_ticker(ticker)
    info = info or {}
    qt = str(info.get("quoteType", "")).upper()
    industry = str(info.get("industry", "")).strip().lower()
    sector = str(info.get("sector", "")).strip().lower()
    known_reserve_profile = reserve_asset_profile(ticker)
    if known_reserve_profile == "fixed_income_etf":
        return "FIXED_INCOME_ETF"
    if known_reserve_profile == "equity_etf":
        return "EQUITY_ETF"
    if qt == "ETF":
        text = " ".join(
            str(info.get(k, ""))
            for k in ["category", "fundFamily", "longName", "shortName", "industry", "sector"]
        ).lower()
        if any(x in text for x in ["bond", "treasury", "fixed income", "government"]):
            return "FIXED_INCOME_ETF"
        return "EQUITY_ETF"
    if ticker in STANDARD_OPERATING_OVERRIDES:
        return "STANDARD_STOCK"
    elif ticker in SPECIAL_FINANCIAL_REASONS or "berkshire hathaway" in str(
        info.get("longName", "")
    ).lower():
        return "SPECIAL_FINANCIAL_STOCK"
    elif any(term in industry for term in SPECIAL_FINANCIAL_INDUSTRY_TERMS):
        return "SPECIAL_FINANCIAL_STOCK"
    elif (
        sector in {"financial services", "financials", "finance"}
        and "credit services" not in industry
    ):
        return "SPECIAL_FINANCIAL_STOCK"
    return "STANDARD_STOCK"


def _classification_reason(ticker: str, info: dict, scoring_model: str) -> str:
    ticker = normalize_ticker(ticker)
    if ticker in STANDARD_OPERATING_OVERRIDES:
        return STANDARD_OPERATING_OVERRIDES[ticker]
    if scoring_model == "FIXED_INCOME_ETF":
        return "Known fixed-income ETF or Yahoo fund metadata identifies fixed-income exposure"
    if scoring_model == "EQUITY_ETF":
        return "Known equity ETF or Yahoo quoteType identifies an equity ETF"
    if ticker in SPECIAL_FINANCIAL_REASONS:
        return SPECIAL_FINANCIAL_REASONS[ticker]
    if "berkshire hathaway" in str(info.get("longName", "")).lower():
        return "Insurance and financial conglomerate accounting"
    industry = str(info.get("industry", "")).strip().lower()
    if any(term in industry for term in SPECIAL_FINANCIAL_INDUSTRY_TERMS):
        return f"Industry-based financial accounting classification: {info.get('industry')}"
    if scoring_model == "SPECIAL_FINANCIAL_STOCK":
        return "Weak fallback: financial-sector company without a more specific business-model classification"
    return "Standard operating-company accounting metrics are used"


def analyze_security(ticker: str, cfg: dict, force_refresh: bool = False) -> tuple[dict, pd.Series]:
    _need_yfinance()
    ticker_entered = str(ticker).strip()
    ticker = normalize_ticker(ticker)
    cache_dir = cfg["data"]["cache_dir"]
    ttl = float(cfg["data"]["cache_ttl_hours"])
    key = f"security_v10::{ticker}"
    if not force_refresh:
        cached = _cache_get(cache_dir, key, ttl)
        if cached is not None:
            return cached

    t = yf.Ticker(ticker)
    info_error = ""
    try:
        info = t.get_info() or {}
    except Exception as exc:
        info_error = f"get_info failed: {exc}"
        try:
            info = t.info or {}
        except Exception as fallback_exc:
            info = {}
            info_error += f"; t.info fallback failed: {fallback_exc}"
    scoring_model = classify_asset(ticker, info)
    asset_class = {
        "FIXED_INCOME_ETF": "fixed_income",
        "EQUITY_ETF": "equity_etf",
    }.get(scoring_model, "stock")
    profile = {
        "security_type": "ETF" if scoring_model.endswith("_ETF") else "stock",
        "scoring_model": scoring_model,
        "scoring_classification_reason": _classification_reason(
            ticker, info, scoring_model
        ),
        "asset_class": asset_class,
    }

    if profile["asset_class"] == "stock":
        income, income_source, income_error = _first_statement_source(
            t, "get_income_stmt", "yearly", ["income_stmt", "financials"]
        )
        quarterly_income, quarterly_income_source, quarterly_income_error = _first_statement_source(
            t, "get_income_stmt", "quarterly", ["quarterly_income_stmt", "quarterly_financials"]
        )
        ttm_income = pd.DataFrame()
        ttm_income_error = ""
        try:
            candidate = t.ttm_income_stmt
            if isinstance(candidate, pd.DataFrame):
                ttm_income = candidate.copy()
        except Exception as exc:
            ttm_income_error = f"ttm_income_stmt failed: {type(exc).__name__}: {exc}"

        cashflow, cashflow_source, cashflow_error = _first_statement_source(
            t, "get_cash_flow", "yearly", ["cash_flow", "cashflow"]
        )
        quarterly_cashflow, quarterly_cashflow_source, quarterly_cashflow_error = _first_statement_source(
            t, "get_cash_flow", "quarterly", ["quarterly_cash_flow", "quarterly_cashflow"]
        )
        ttm_cashflow = pd.DataFrame()
        ttm_cashflow_error = ""
        try:
            candidate = t.ttm_cash_flow
            if isinstance(candidate, pd.DataFrame):
                ttm_cashflow = candidate.copy()
        except Exception as exc:
            ttm_cashflow_error = f"ttm_cash_flow failed: {type(exc).__name__}: {exc}"

        balance_sheet, balance_source, balance_error = _first_statement_source(
            t, "get_balance_sheet", "yearly", ["balance_sheet", "balancesheet"]
        )
        quarterly_balance_sheet, quarterly_balance_source, quarterly_balance_error = _first_statement_source(
            t, "get_balance_sheet", "quarterly", ["quarterly_balance_sheet", "quarterly_balancesheet"]
        )
    else:
        income = cashflow = balance_sheet = pd.DataFrame()
        quarterly_income = quarterly_cashflow = quarterly_balance_sheet = pd.DataFrame()
        ttm_income = ttm_cashflow = pd.DataFrame()
        income_source = cashflow_source = balance_source = ""
        quarterly_income_source = quarterly_cashflow_source = quarterly_balance_source = ""
        income_error = cashflow_error = balance_error = "Financial statements are not applicable to ETFs"
        quarterly_income_error = quarterly_cashflow_error = quarterly_balance_sheet_error = income_error
        quarterly_balance_error = quarterly_balance_sheet_error
        ttm_income_error = ttm_cashflow_error = income_error

    valuation_measures = pd.DataFrame()
    valuation_error = ""
    try:
        valuation_measures = t.get_valuation_measures()
        if valuation_measures is None or valuation_measures.empty:
            valuation_error = "get_valuation_measures returned no values"
    except Exception as exc:
        valuation_error = f"get_valuation_measures failed: {type(exc).__name__}: {exc}"

    history_error = ""
    try:
        hist = t.history(period="10y", auto_adjust=True, actions=False)
    except Exception as e:
        hist = pd.DataFrame()
        history_error = f"Price history request failed: {e}"
    if hist is None or hist.empty:
        hist = pd.DataFrame()
        history_error = history_error or "Yahoo returned no price-history rows"
    close = (
        pd.to_numeric(hist["Close"], errors="coerce").dropna().rename(ticker)
        if "Close" in hist
        else pd.Series(dtype=float, name=ticker)
    )

    eps, eps_field, eps_match = _statement_row_details(income, ["Diluted EPS", "Basic EPS"])
    op_income, op_income_field, op_income_match = _statement_row_details(
        income, ["EBIT", "Operating Income"]
    )
    fcf, fcf_field, fcf_match = _statement_row_details(cashflow, ["Free Cash Flow"])
    if fcf.empty:
        cfo, cfo_field, cfo_match = _statement_row_details(
            cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"]
        )
        capex, capex_field, capex_match = _statement_row_details(
            cashflow, ["Capital Expenditure", "Capital Expenditures"]
        )
        fcf = _derive_free_cash_flow(cfo, capex)
        if not fcf.empty:
            fcf_field = f"{cfo_field} - capital expenditure ({capex_field})"
            fcf_match = f"Derived from {cfo_match} and {capex_match}; CapEx sign preserved"
        else:
            fcf_field = ""
            fcf_match = "FCF row and usable operating-cash-flow/capex pair unavailable"

    latest_fcf = _latest(fcf)

    de_raw = _safe_num(info.get("debtToEquity"))
    de = de_raw / 100.0 if np.isfinite(de_raw) else np.nan

    fast_info, fast_info_error = _fast_info_values(t)
    pe = _safe_num(info.get("trailingPE"))
    pe_source = "info.trailingPE" if np.isfinite(pe) and pe > 0 else ""
    if not pe_source:
        valuation_pe, valuation_pe_source = _valuation_measure(
            valuation_measures, ["Trailing P/E"]
        )
        if np.isfinite(valuation_pe) and valuation_pe > 0:
            pe, pe_source = valuation_pe, valuation_pe_source
        else:
            pe = np.nan
    fpe = _safe_num(info.get("forwardPE"))
    ps = _safe_num(info.get("priceToSalesTrailing12Months"))
    pb = _safe_num(info.get("priceToBook"))
    eve = _safe_num(info.get("enterpriseToEbitda"))
    # Negative valuation multiples are not interpreted as attractive.
    pe = pe if pe > 0 else np.nan
    fpe = fpe if fpe > 0 else np.nan
    eve = eve if eve > 0 else np.nan

    special_financial = profile["scoring_model"] == "SPECIAL_FINANCIAL_STOCK"
    market_key = f"market::SPY::{cache_dir}"
    market_prices = _cache_get(cache_dir, market_key, ttl)
    market_history_error = ""
    if market_prices is None:
        try:
            market_hist = yf.Ticker("SPY").history(period="10y", auto_adjust=True, actions=False)
            market_prices = pd.to_numeric(market_hist.get("Close"), errors="coerce").dropna()
            if not market_prices.empty:
                _cache_set(cache_dir, market_key, market_prices)
        except Exception as exc:
            market_prices = pd.Series(dtype=float)
            market_history_error = f"SPY beta-reference history request failed: {exc}"
    if ticker == "SPY" and not market_prices.empty:
        beta = 1.0
        beta_source = "SPY adjusted-price returns versus itself, aligned over the 3-year risk window"
    elif not market_prices.empty:
        beta = _market_beta(close, market_prices)
        beta_source = "Aligned daily simple returns versus SPY adjusted Close over the same 3-year risk window"
    else:
        beta = np.nan
        beta_source = "N/A: SPY adjusted-price history unavailable; Yahoo 5Y monthly beta is not substituted"
    risk = _price_risk(close, beta)
    avg_dollar_volume = np.nan
    if "Volume" in hist.columns:
        volume = pd.to_numeric(hist["Volume"], errors="coerce").tail(30)
        dollar_turnover = pd.to_numeric(hist["Close"], errors="coerce").tail(30) * volume
        if dollar_turnover.notna().any():
            avg_dollar_volume = float(dollar_turnover.mean())
    fund_attributes = (
        _fund_attributes(t)
        if profile["scoring_model"] in {"EQUITY_ETF", "FIXED_INCOME_ETF"}
        else {"sector_weightings": {}, "bond_ratings": {}, "fund_overview": {}, "error": ""}
    )
    sector_weights = normalize_sector_weights(fund_attributes["sector_weightings"])
    sector_benchmark_ticker = normalize_ticker(cfg["optimizer"].get("sector_benchmark", "SPY"))
    benchmark_sector_weights, benchmark_sector_source, benchmark_sector_error = (
        _benchmark_sector_data(sector_benchmark_ticker, cfg, cache_dir, ttl)
        if profile["scoring_model"] == "EQUITY_ETF"
        else ({}, "Not applicable", "")
    )
    sector_max = max(sector_weights.values()) if sector_weights else np.nan
    bond_ratings = fund_attributes["bond_ratings"]
    credit_score = _credit_score_from_exposure(bond_ratings, cfg)
    overview = fund_attributes["fund_overview"]
    rating_sum = sum(
        _safe_num(bond_ratings.get(key), 0.0)
        for key in ("aaa", "aa", "a", "bbb", "bb", "b", "below_b")
    )
    non_government_ratings = {
        key: _safe_num(bond_ratings.get(key), 0.0)
        for key in ("aaa", "aa", "a", "bbb", "bb", "b", "below_b")
    }
    government_exposure = _safe_num(bond_ratings.get("us_government"))
    if np.isfinite(government_exposure) and government_exposure > 1.0:
        government_exposure /= 100.0
    exposure_text = (
        f"U.S. government exposure {government_exposure:.1%}"
        if np.isfinite(government_exposure) and government_exposure > 0
        else ("Credit mix reported" if rating_sum > 0 else "N/A")
    )
    price_details = _current_price_details(t, info, close, history_error, fast_info)
    latest_market_cap, market_cap_source, market_cap_error = _market_cap_details(
        info, fast_info, valuation_measures, price_details["current_price"]
    )
    reported_ytm = _safe_num(
        info.get("yieldToMaturity")
        if info.get("yieldToMaturity") is not None
        else info.get("weightedAverageYieldToMaturity")
    )
    reported_distribution_yield = _safe_num(info.get("yield"))
    raw_dividend_yield = _safe_num(info.get("dividendYield"))
    if np.isfinite(reported_distribution_yield):
        distribution_yield = reported_distribution_yield
        distribution_yield_source = "info.yield (decimal ratio)"
    elif np.isfinite(raw_dividend_yield):
        distribution_yield = raw_dividend_yield / 100.0
        distribution_yield_source = "info.dividendYield (percentage points ÷ 100)"
    else:
        distribution_yield = np.nan
        distribution_yield_source = "Unavailable"
    dividend_yield = raw_dividend_yield / 100.0 if np.isfinite(raw_dividend_yield) else np.nan
    comparable_treasury_ytm = np.nan
    ytm_spread = (
        reported_ytm - comparable_treasury_ytm
        if np.isfinite(reported_ytm) and np.isfinite(comparable_treasury_ytm)
        else np.nan
    )
    total_debt = _safe_num(info.get("totalDebt"))
    debt_source = "info.totalDebt" if np.isfinite(total_debt) else ""
    if not np.isfinite(total_debt):
        statement_total_debt, statement_total_field, statement_total_reason = _statement_row_details(
            balance_sheet, ["Total Debt"]
        )
        total_debt = _latest(statement_total_debt)
        if np.isfinite(total_debt):
            debt_source = f"{balance_source}.{statement_total_field}"
    if not np.isfinite(total_debt):
        current_debt, current_debt_field, current_debt_reason = _statement_row_details(
            balance_sheet, ["Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt"]
        )
        long_term_debt, long_term_debt_field, long_term_debt_reason = _statement_row_details(
            balance_sheet, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
        )
        long_term_value, current_debt_value = _latest(long_term_debt), _latest(current_debt)
        if np.isfinite(long_term_value) and np.isfinite(current_debt_value):
            total_debt = long_term_value + current_debt_value
            debt_source = f"{balance_source}.{long_term_debt_field} + {current_debt_field}"
        else:
            quarterly_total_debt, quarterly_total_field, _ = _statement_row_details(
                quarterly_balance_sheet, ["Total Debt"]
            )
            total_debt = _latest(quarterly_total_debt)
            if np.isfinite(total_debt):
                debt_source = f"{quarterly_balance_source}.{quarterly_total_field}"
            else:
                quarterly_current, quarterly_current_field, quarterly_current_reason = _statement_row_details(
                    quarterly_balance_sheet,
                    ["Current Debt", "Current Debt And Capital Lease Obligation", "Short Term Debt"],
                )
                quarterly_long, quarterly_long_field, quarterly_long_reason = _statement_row_details(
                    quarterly_balance_sheet,
                    ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
                )
                quarterly_current_value = _latest(quarterly_current)
                quarterly_long_value = _latest(quarterly_long)
                if np.isfinite(quarterly_current_value) and np.isfinite(quarterly_long_value):
                    total_debt = quarterly_current_value + quarterly_long_value
                    debt_source = (
                        f"{quarterly_balance_source}.{quarterly_current_field} + "
                        f"{quarterly_long_field}"
                    )
                else:
                    debt_source = "Debt N/A: " + "; ".join(
                        part for part in (
                            statement_total_reason, current_debt_reason, long_term_debt_reason,
                            quarterly_current_reason, quarterly_long_reason, quarterly_balance_error,
                        ) if part
                    )
    cash = _safe_num(info.get("totalCash"))
    cash_source = "info.totalCash"
    if not np.isfinite(cash):
        cash_row, cash_field, cash_reason = _statement_row_details(
            balance_sheet,
            [
                "Cash Cash Equivalents And Short Term Investments",
                "Cash And Cash Equivalents",
                "Cash",
            ],
        )
        cash = _latest(cash_row)
        if np.isfinite(cash):
            cash_source = f"{balance_source}.{cash_field}"
        else:
            quarterly_cash, quarterly_cash_field, quarterly_cash_reason = _statement_row_details(
                quarterly_balance_sheet,
                [
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents",
                    "Cash",
                ],
            )
            cash = _latest(quarterly_cash)
            cash_source = (
                f"{quarterly_balance_source}.{quarterly_cash_field}"
                if np.isfinite(cash)
                else f"Cash N/A: {cash_reason}; {quarterly_cash_reason}; {quarterly_balance_error}"
            )
    net_debt = total_debt - cash if np.isfinite(total_debt) and np.isfinite(cash) else np.nan
    ttm_fcf, ttm_fcf_field, ttm_fcf_source, ttm_fcf_reason = _statement_ttm_value(
        ttm_cashflow,
        quarterly_cashflow,
        ["Free Cash Flow"],
    )
    if not np.isfinite(ttm_fcf):
        ttm_cfo, ttm_capex, ttm_cfo_field, ttm_capex_field, ttm_pair_reason = _paired_ttm_values(
            ttm_cashflow,
            quarterly_cashflow,
            ["Operating Cash Flow", "Total Cash From Operating Activities"],
            ["Capital Expenditure", "Capital Expenditures"],
        )
        if np.isfinite(ttm_cfo) and np.isfinite(ttm_capex):
            ttm_fcf = ttm_cfo + ttm_capex if ttm_capex < 0 else ttm_cfo - ttm_capex
            ttm_fcf_field = f"{ttm_cfo_field} + {ttm_capex_field}"
            ttm_fcf_source = "Derived from matched TTM/four-quarter statements"
            ttm_fcf_reason = f"Operating cash flow less capital expenditure with source sign preserved; {ttm_pair_reason}"
    if np.isfinite(ttm_fcf):
        current_fcf = ttm_fcf
        current_fcf_source = f"{ttm_fcf_source}: {ttm_fcf_field}"
    elif np.isfinite(latest_fcf):
        current_fcf = latest_fcf
        current_fcf_source = f"Latest annual cash-flow statement ({fcf_field})"
    else:
        current_fcf = np.nan
        current_fcf_source = "FCF unavailable: " + "; ".join(
            part for part in (ttm_fcf_reason, cashflow_error) if part
        )
    if special_financial:
        current_fcf = np.nan
        current_fcf_source = "N/A: FCF is not treated as comparable for a financial company"
    annual_fcf = current_fcf
    net_debt_to_fcf = (
        net_debt / annual_fcf
        if np.isfinite(net_debt) and np.isfinite(annual_fcf) and annual_fcf > 0
        else np.nan
    )
    fcf_yield = (
        annual_fcf / latest_market_cap
        if np.isfinite(annual_fcf) and np.isfinite(latest_market_cap) and latest_market_cap > 0
        else np.nan
    )
    if special_financial:
        net_debt_to_fcf = np.nan
        fcf_yield = np.nan
    expense_ratio, expense_ratio_raw, expense_ratio_source = _expense_ratio_details(
        info, fund_attributes.get("fund_operations")
    )
    ttm_op_income, ttm_interest_expense, ttm_op_field, ttm_interest_field, coverage_period_source = _paired_ttm_values(
        ttm_income,
        quarterly_income,
        ["EBIT", "Operating Income"],
        ["Interest Expense", "Interest Expense Non Operating", "Interest And Debt Expense"],
    )
    interest_coverage, interest_coverage_source = _interest_coverage_details(
        op_income, income, info,
        ebit_override=ttm_op_income,
        interest_override=ttm_interest_expense,
        override_source=(
            f"{coverage_period_source}: EBIT ({ttm_op_field}) / abs(interest expense ({ttm_interest_field}))"
        ),
    )
    if special_financial:
        interest_coverage = np.nan
        interest_coverage_source = "N/A: interest coverage is not comparable for financial-company operations"
    price_return_details = {
        "total_return_1y": _historical_price_return_details(close, 1),
        "total_return_3y": _historical_price_return_details(close, 3, cagr=True),
    }
    price_returns = {key: value[0] for key, value in price_return_details.items()}
    financial_series = {
        "eps": eps,
        "fcf": pd.Series(dtype=float) if special_financial else fcf,
    }
    financial_cagrs = {
        f"{metric}_growth_3y": _cagr_details(series)
        for metric, series in financial_series.items()
    }
    top10 = np.nan
    top10_error = ""
    if profile["scoring_model"] in {"EQUITY_ETF", "FIXED_INCOME_ETF"}:
        try:
            holdings = etf_holdings(ticker, cfg)
            raw_weights = holdings["weight"] if "weight" in holdings else pd.Series(dtype=float)
            weights = pd.to_numeric(raw_weights, errors="coerce").dropna()
            if not weights.empty:
                top10 = float(weights.nlargest(10).sum())
            else:
                top10_error = str(holdings.attrs.get("retrieval_error") or "Yahoo returned no valid holding weights")
        except Exception as exc:
            top10_error = f"Yahoo ETF holdings retrieval failed: {exc}"
    if profile["scoring_model"] == "FIXED_INCOME_ETF" and not exposure_text.startswith("U.S. Treasury"):
        exposure_text = exposure_text if exposure_text != "N/A" else str(overview.get("categoryName", "N/A"))
    metrics = {
        "ticker": ticker,
        "ticker_entered": ticker_entered,
        "company": info.get("longName") or info.get("shortName") or ticker,
        "sector": normalize_sector(info.get("sector") or "Unknown"),
        "original_yahoo_sector": info.get("sector") or "N/A",
        "category": info.get("category") or info.get("industry") or "N/A",
        "industry": info.get("industry") or "Unknown",
        **profile,
        **price_details,
        "info_error": info_error,
        "income_statement_source": income_source,
        "income_statement_error": income_error,
        "cashflow_statement_source": cashflow_source,
        "cashflow_statement_error": cashflow_error,
        "quarterly_income_source": quarterly_income_source,
        "quarterly_income_error": quarterly_income_error,
        "ttm_income_error": ttm_income_error,
        "quarterly_cashflow_source": quarterly_cashflow_source,
        "quarterly_cashflow_error": quarterly_cashflow_error,
        "ttm_cashflow_error": ttm_cashflow_error,
        "balance_sheet_source": balance_source,
        "balance_sheet_error": balance_error,
        "quarterly_balance_sheet_source": quarterly_balance_source,
        "quarterly_balance_sheet_error": quarterly_balance_error,
        "valuation_error": valuation_error,
        "fast_info_error": fast_info_error,
        "market_cap_source": market_cap_source,
        "market_cap_error": market_cap_error,
        "trailing_pe_source": pe_source or "Unavailable",
        "fund_data_error": fund_attributes.get("error", ""),
        "holdings_data_error": top10_error,
        "market_history_error": market_history_error,
        "history_error": history_error,
        "free_cash_flow": current_fcf,
        "eps_growth_3y": financial_cagrs["eps_growth_3y"][0],
        "fcf_growth_3y": financial_cagrs["fcf_growth_3y"][0],
        **price_returns,
        "return_sanity_warnings": _return_sanity_warnings(price_returns),
        "current_ratio": _safe_num(info.get("currentRatio")),
        "debt_to_equity": de,
        "trailing_pe": pe,
        "forward_pe": fpe,
        "price_to_sales": ps,
        "price_to_book": pb,
        "ev_to_ebitda": eve,
        # Fund-level fields are populated for ETFs where Yahoo provides them.
        "expense_ratio": expense_ratio,
        "expense_ratio_raw": expense_ratio_raw,
        "expense_ratio_source": expense_ratio_source,
        "total_assets": _safe_num(info.get("totalAssets")),
        "market_cap": latest_market_cap,
        "holdings_count": _safe_num(info.get("numberOfHoldings")),
        "portfolio_pe": _safe_num(info.get("trailingPE")),
        "portfolio_pb": _safe_num(info.get("priceToBook")),
        "distribution_yield": distribution_yield,
        "dividend_yield": dividend_yield,
        "sec_yield_30d": _safe_num(info.get("thirtyDayYield") or info.get("secYield30Day")),
        "yield_to_maturity": reported_ytm,
        "comparable_treasury_ytm": comparable_treasury_ytm,
        "yield_spread": ytm_spread,
        "effective_duration": _safe_num(info.get("effectiveDuration") or info.get("duration")),
        "weighted_average_maturity": _safe_num(info.get("weightedAverageMaturity")),
        "net_debt": net_debt,
        "total_debt": total_debt,
        "total_cash": cash,
        "debt_source": debt_source,
        "cash_source": cash_source,
        "net_debt_to_fcf": net_debt_to_fcf,
        "interest_coverage": interest_coverage,
        "interest_coverage_source": interest_coverage_source,
        "fcf_yield": fcf_yield,
        "fcf_source": current_fcf_source,
        "average_dollar_volume_30d": avg_dollar_volume,
        "top10_concentration": top10,
        "largest_sector_weight": sector_max,
        "sector_weightings": sector_weights,
        "sector_exposure_complete": bool(sector_weights),
        "sector_benchmark_weightings": benchmark_sector_weights,
        "sector_benchmark_source": benchmark_sector_source,
        "sector_benchmark_error": benchmark_sector_error,
        "analysis_year": pd.Timestamp.today().year,
        "target_term_2033_years": max(0, 2033 - pd.Timestamp.today().year),
        "bond_ratings": non_government_ratings,
        "weighted_credit_score": credit_score,
        "treasury_credit_exposure": exposure_text,
        "eps_history": {str(k.date()): float(v) for k, v in eps.items()} if not eps.empty else {},
        "fcf_history": (
            {str(k.date()): float(v) for k, v in fcf.items()}
            if not fcf.empty and not special_financial else {}
        ),
        "data_date": str(hist.index[-1].date()) if len(hist.index) else "N/A",
        **risk,
    }
    metrics["laura_sector_bucket"] = (
        "ETF look-through"
        if profile["scoring_model"] == "EQUITY_ETF"
        else laura_sector_bucket(
            info.get("sector") or "Unknown",
            info.get("industry") or "",
            profile["asset_class"],
        )
    )
    metric_diagnostics = {}

    def add_diagnostic(metric, source, raw_field, raw_value, reason="", final_value=None, fallback=""):
        available = np.isfinite(_safe_num(raw_value))
        interpreted = raw_value if final_value is None and available else final_value
        metric_diagnostics[metric] = {
            "metric": metric,
            "source": source,
            "raw_field": raw_field or "N/A",
            "raw_value": raw_value if available else "N/A",
            "final_interpreted_value": (
                interpreted if np.isfinite(_safe_num(interpreted)) else "N/A"
            ),
            "status": "OK" if available else "N/A",
            "n/a_reason": "" if available else (reason or "Source did not provide a valid value"),
            "fallback_used": fallback,
        }

    for label, value, source in (
        ("Ticker Entered", metrics["ticker_entered"], "User input"),
        ("Normalized Ticker", metrics["ticker"], "Yahoo Finance symbol normalization"),
        ("Asset Type", metrics["asset_class"], "Central asset classifier"),
        ("Yahoo Sector", metrics["original_yahoo_sector"], "Yahoo quoteSummary/info.sector"),
        ("Yahoo Industry", metrics["industry"], "Yahoo quoteSummary/info.industry"),
        ("Laura Sector Bucket", metrics["laura_sector_bucket"], "Independent portfolio-sector mapping"),
        (
            "Scoring Model",
            metrics.get("scoring_model", "NOT_APPLICABLE"),
            "Business-model classification",
        ),
        (
            "Scoring Classification Reason",
            metrics.get(
                "scoring_classification_reason",
                "Stock accounting classification does not apply to ETFs",
            ),
            "Explicit business-model override, accounting industry, or weak sector fallback",
        ),
    ):
        metric_diagnostics[f"classification_{label.lower().replace(' ', '_')}"] = {
            "metric": label,
            "source": source,
            "raw_field": label,
            "raw_value": value,
            "final_interpreted_value": value,
            "status": "OK" if value not in (None, "", "N/A", "Unknown") else "N/A",
            "n/a_reason": "Yahoo classification field unavailable" if value in (None, "", "N/A", "Unknown") else "",
            "fallback_used": "",
        }

    series_specs = {
        "EPS": ("eps", eps, eps_field, eps_match, income_source, income_error),
    }
    if profile["asset_class"] == "stock" and not special_financial:
        series_specs["FCF"] = ("fcf", fcf, fcf_field, fcf_match, cashflow_source, cashflow_error)
    for label, (cagr_prefix, series, raw_field, match, statement_source, statement_error) in series_specs.items():
        add_diagnostic(
            label,
            f"Yahoo {statement_source or 'financial statements'}",
            raw_field or "Named statement row",
            _latest(series),
            statement_error or match or "Named statement row unavailable",
        )
        for horizon in (3,):
            metric_key = f"{cagr_prefix}_growth_{horizon}y"
            value, history = financial_cagrs[metric_key]
            diagnostic = {
                "metric": f"{label} {horizon}Y CAGR",
                "source": f"Yahoo {statement_source or 'financial statements'}",
                "raw_field": raw_field or "Named statement row",
                "raw_value": value if np.isfinite(_safe_num(value)) else "N/A",
                "final_interpreted_value": value if np.isfinite(_safe_num(value)) else "N/A",
                "status": "OK" if np.isfinite(_safe_num(value)) else "N/A",
                "n/a_reason": "" if np.isfinite(_safe_num(value)) else (
                    "EPS CAGR is inapplicable to an ETF"
                    if profile["asset_class"] != "stock"
                    else history["reason"] or statement_error or match or "CAGR unavailable"
                ),
                "fallback_used": (
                    "Basic EPS used because Diluted EPS was unavailable"
                    if cagr_prefix == "eps" and "basic" in re.sub(r"[^a-z0-9]", "", raw_field.lower())
                    else (
                        match if "fallback" in match.lower() or "derived" in match.lower()
                        else ""
                    )
                ),
                "valid_observations": history["valid_observations"],
                "start_date": history["start_date"],
                "end_date": history["end_date"],
                "start_value": history["start_value"],
                "end_value": history["end_value"],
                "elapsed_years": history["elapsed_years"],
            }
            metric_diagnostics[metric_key] = diagnostic
    if profile["asset_class"] != "stock" or special_financial:
        reason = (
            "Metric is not economically comparable for a financial company"
            if special_financial
            else "Metric is inapplicable to an ETF"
        )
        metric_diagnostics["fcf_growth_3y"] = {
            "metric": "FCF 3Y CAGR",
            "source": "Not applicable for this asset type",
            "raw_field": "N/A",
            "raw_value": "N/A",
            "final_interpreted_value": "N/A",
            "status": "N/A",
            "n/a_reason": reason,
            "fallback_used": "",
        }

    for key, label in (
        ("total_return_1y", "1Y Return"),
        ("total_return_3y", "3Y CAGR (price)"),
    ):
        value, details = price_return_details[key]
        metric_diagnostics[key] = {
            "metric": label,
            "source": "Yahoo auto-adjusted price history",
            "raw_field": "Close (auto_adjust=True)",
            "raw_value": value if np.isfinite(_safe_num(value)) else "N/A",
            "final_interpreted_value": value if np.isfinite(_safe_num(value)) else "N/A",
            "status": "OK" if np.isfinite(_safe_num(value)) else "N/A",
            "n/a_reason": details["reason"],
            "fallback_used": "",
            "start_date": details["start_date"],
            "end_date": details["end_date"],
            "elapsed_years": details["elapsed_years"],
        }

    for key, label, field in (
        ("trailing_pe", "Trailing P/E", "trailingPE"),
        ("forward_pe", "Forward P/E", "forwardPE"),
        ("price_to_sales", "Price / Sales", "priceToSalesTrailing12Months"),
        ("price_to_book", "Price / Book", "priceToBook"),
        ("ev_to_ebitda", "EV / EBITDA", "enterpriseToEbitda"),
        ("current_ratio", "Current Ratio", "currentRatio"),
        ("debt_to_equity", "Debt / Equity", "debtToEquity"),
        ("market_cap", "Market Capitalization", market_cap_source),
        ("total_assets", "Total Assets", "totalAssets"),
        ("expense_ratio", "Expense Ratio", expense_ratio_source),
        ("holdings_count", "Holdings Count", "numberOfHoldings"),
        ("sec_yield_30d", "30-Day SEC Yield", "thirtyDayYield / secYield30Day"),
        ("yield_to_maturity", "Yield to Maturity", "yieldToMaturity / weightedAverageYieldToMaturity"),
        ("effective_duration", "Effective Duration", "effectiveDuration / duration"),
        ("weighted_average_maturity", "Weighted Average Maturity", "weightedAverageMaturity"),
        ("average_dollar_volume_30d", "Average Daily Dollar Volume", "history Close × Volume, last 30 observations"),
        ("top10_concentration", "Top-10 Concentration", "Yahoo fund holdings weights"),
        ("largest_sector_weight", "Largest Sector Weight", "Yahoo fund sector_weightings"),
        ("weighted_credit_score", "Weighted Credit Score", "Yahoo fund bond_ratings"),
    ):
        value = latest_market_cap if key == "market_cap" else metrics.get(key)
        source = (
            "Derived from price and volume history" if key == "average_dollar_volume_30d"
            else "Yahoo fund data" if key in {"holdings_count", "top10_concentration", "largest_sector_weight", "weighted_credit_score"}
            else "Yahoo quoteSummary/info"
        )
        reason = ""
        if key == "trailing_pe" and not np.isfinite(value):
            reason = valuation_error or "Yahoo info.trailingPE and valuation-measure Trailing P/E unavailable"
        elif key == "market_cap" and not np.isfinite(value):
            reason = market_cap_error
        elif key == "total_assets" and not np.isfinite(value):
            reason = "Yahoo quoteSummary/info.totalAssets unavailable"
        elif key in {"forward_pe", "ev_to_ebitda"} and not np.isfinite(_safe_num(info.get(field))):
            reason = f"Yahoo field {field} missing; forward P/E is not substituted for trailing P/E"
        elif key in {"trailing_pe", "forward_pe", "ev_to_ebitda"} and _safe_num(info.get(field)) <= 0:
            reason = f"Yahoo field {field} is non-positive and is treated as N/A, not as a cheap valuation"
        elif key in {"price_to_sales", "price_to_book", "current_ratio", "debt_to_equity"} and not np.isfinite(value):
            reason = f"Yahoo field {field} unavailable"
        elif key == "expense_ratio":
            reason = (
                "Expense ratio is inapplicable to an individual stock"
                if profile["asset_class"] == "stock"
                else f"Yahoo fund_operations and expense-ratio info fields unavailable ({expense_ratio_source})"
            )
        elif key == "average_dollar_volume_30d":
            reason = "No usable adjusted Close × Volume observations"
        elif key == "top10_concentration" and profile["asset_class"] == "stock":
            reason = "Fund diversification metric is inapplicable to an individual stock"
        elif key == "top10_concentration":
            reason = top10_error or "Yahoo ETF holdings endpoint returned no valid holding weights"
        elif key == "largest_sector_weight" and profile["asset_class"] == "stock":
            reason = "Fund diversification metric is inapplicable to an individual stock"
        elif key == "largest_sector_weight":
            reason = fund_attributes.get("error") or "Yahoo ETF sector weights unavailable or incomplete"
        elif key == "weighted_credit_score" and profile["asset_class"] != "fixed_income":
            reason = "Bond credit-exposure metric is inapplicable to this asset type"
        elif key == "weighted_credit_score":
            reason = fund_attributes.get("error") or "Yahoo ETF bond-rating exposures unavailable or incomplete"
        elif key in {"sec_yield_30d", "yield_to_maturity", "effective_duration", "weighted_average_maturity"}:
            reason = (
                "Bond-yield/duration field is inapplicable to this asset type"
                if profile["asset_class"] != "fixed_income"
                else "Yahoo field unavailable; no substitute used"
            )
        elif key == "holdings_count":
            reason = (
                "Holdings count is inapplicable to an individual stock"
                if profile["asset_class"] == "stock"
                else "Yahoo does not report total holdings count; top-holdings list is not treated as the full count"
            )
        fallback_used = (
            pe_source if key == "trailing_pe" and pe_source != "info.trailingPE"
            else market_cap_source if key == "market_cap" and market_cap_source != "info.marketCap"
            else ""
        )
        add_diagnostic(label, source, field, value, reason, fallback=fallback_used)

    for key, label in (
        ("current_price", "Current / Latest Price"),
        ("net_debt", "Net Debt"),
        ("net_debt_to_fcf", "Net Debt / FCF"),
        ("interest_coverage", "Interest Coverage"),
        ("fcf_yield", "FCF Yield"),
        ("annualized_volatility", "Annualized Volatility"),
        ("max_drawdown", "Maximum Drawdown"),
        ("beta", "Beta vs SPY"),
    ):
        value = metrics.get(key)
        source = {
            "current_price": metrics["price_source"],
            "net_debt": "Yahoo quoteSummary and balance sheet",
            "net_debt_to_fcf": "Derived net debt / latest annual FCF",
            "interest_coverage": interest_coverage_source,
            "fcf_yield": "Derived current TTM or latest annual FCF / current market capitalization",
            "annualized_volatility": "Yahoo adjusted price history, trailing 3 years; std(daily simple returns) × sqrt(252)",
            "max_drawdown": "Yahoo adjusted price history, trailing 3 years; maximum drawdown magnitude",
            "beta": beta_source,
        }[key]
        field = {
            "current_price": metrics["price_source"],
            "net_debt": f"{debt_source}; {cash_source}",
            "net_debt_to_fcf": "totalDebt - totalCash / Free Cash Flow",
            "interest_coverage": interest_coverage_source,
            "fcf_yield": f"{current_fcf_source} / {market_cap_source}",
            "annualized_volatility": "Close",
            "max_drawdown": "Close",
            "beta": "Close vs SPY Close",
        }[key]
        reason = ""
        if key in {"net_debt", "net_debt_to_fcf", "fcf_yield", "interest_coverage"} and profile["asset_class"] != "stock":
            reason = "Company financial metric is inapplicable to an ETF"
        elif key in {"net_debt_to_fcf", "fcf_yield", "interest_coverage"} and special_financial:
            reason = "Metric is not economically comparable for a financial company"
        elif key == "current_price":
            reason = metrics["price_diagnostic"]
        elif key == "net_debt" and not np.isfinite(value):
            reason = f"Debt/cash inputs unavailable: {debt_source}; {cash_source}"
        elif key == "net_debt_to_fcf" and not np.isfinite(value):
            reason = (
                "FCF must be positive and net debt available to form this ratio"
                if np.isfinite(net_debt) and np.isfinite(annual_fcf)
                else "Debt, cash, or FCF input unavailable"
            )
        elif key == "fcf_yield" and not np.isfinite(value):
            reason = "Market capitalization or annual FCF unavailable"
        elif key == "interest_coverage" and not np.isfinite(value):
            reason = interest_coverage_source
        elif key in {"annualized_volatility", "max_drawdown"} and not np.isfinite(value):
            reason = f"Insufficient adjusted-price observations (at least 40 required); {history_error or 'history may be missing'}"
        elif key == "beta" and not np.isfinite(value):
            reason = market_history_error or "Insufficient aligned asset/SPY history or zero SPY return variance; Yahoo beta unavailable"
        add_diagnostic(label, source, field, value, reason)
        if key in {"annualized_volatility", "max_drawdown", "beta"}:
            diagnostic = metric_diagnostics[label]
            diagnostic["period_start"] = metrics.get("risk_period_start", "")
            diagnostic["period_end"] = metrics.get("risk_period_end", "")
            diagnostic["price_observations"] = metrics.get("risk_price_observations", 0)

    add_diagnostic(
        "Total Debt",
        "Yahoo quoteSummary/info or balance sheet",
        debt_source,
        total_debt,
        "Balance-sheet leverage metric is inapplicable to an ETF"
        if profile["asset_class"] != "stock" else debt_source,
    )
    add_diagnostic(
        "Total Cash",
        "Yahoo quoteSummary/info or balance sheet",
        cash_source,
        cash,
        "Balance-sheet leverage metric is inapplicable to an ETF"
        if profile["asset_class"] != "stock" else cash_source,
    )
    add_diagnostic(
        "FCF (TTM/current)",
        current_fcf_source,
        ttm_fcf_field or fcf_field,
        current_fcf,
        (
            "Metric is not economically comparable for a financial company"
            if special_financial else
            "Metric is inapplicable to an ETF"
            if profile["asset_class"] != "stock" else
            current_fcf_source
        ),
        final_value=current_fcf,
        fallback=(
            "TTM cash-flow statement" if ttm_fcf_source == "TTM statement"
            else "Sum of four quarterly cash-flow statements" if ttm_fcf_source == "Quarterly sum"
            else "Latest annual cash-flow statement" if np.isfinite(latest_fcf)
            else ""
        ),
    )
    metric_diagnostics["price_source_failures"] = {
        "metric": "Price source fallback trace",
        "source": "yfinance fast_info → ticker info → adjusted history",
        "raw_field": metrics["price_source"],
        "raw_value": metrics["current_price"] if np.isfinite(metrics["current_price"]) else "N/A",
        "final_interpreted_value": metrics["current_price"] if np.isfinite(metrics["current_price"]) else "N/A",
        "status": "OK" if np.isfinite(metrics["current_price"]) else "N/A",
        "n/a_reason": metrics["price_diagnostic"] if not np.isfinite(metrics["current_price"]) else "",
        "fallback_used": metrics["price_source"] if metrics["price_source"] != "yfinance fast_info.last_price" else "",
    }
    known_diagnostic_keys = set(metric_diagnostics)
    source_fields = {
        "total_assets": ("totalAssets", "Yahoo quoteSummary/info"),
        "distribution_yield": ("yield / dividendYield", "Yahoo quoteSummary/info"),
        "dividend_yield": ("dividendYield", "Yahoo quoteSummary/info"),
        "portfolio_pe": ("trailingPE", "Yahoo ETF portfolio statistics"),
        "portfolio_pb": ("priceToBook", "Yahoo ETF portfolio statistics"),
        "comparable_treasury_ytm": ("Treasury comparison curve", "Not available; no proxy substituted"),
        "yield_spread": ("yield_to_maturity - comparable_treasury_ytm", "Derived from reported YTM and comparable Treasury yield"),
        "sector_exposure_complete": ("sector_weightings", "Yahoo fund sector allocation"),
    }
    already_diagnosed = {
        "trailing_pe", "forward_pe", "price_to_sales", "price_to_book", "ev_to_ebitda",
        "current_ratio", "debt_to_equity",
        "market_cap", "expense_ratio", "holdings_count", "sec_yield_30d",
        "yield_to_maturity", "effective_duration", "weighted_average_maturity",
        "average_dollar_volume_30d", "top10_concentration", "largest_sector_weight",
        "weighted_credit_score", "current_price", "net_debt", "net_debt_to_fcf",
        "interest_coverage", "fcf_yield", "total_assets", "total_debt", "total_cash",
        "expense_ratio_raw", "distribution_yield", "dividend_yield",
    }
    specific_missing_reasons = {
        "portfolio_pe": "Yahoo ETF portfolio P/E unavailable; no holding-level value was inferred",
        "portfolio_pb": "Yahoo ETF portfolio price/book unavailable; no holding-level value was inferred",
        "distribution_yield": "Yahoo yield and dividendYield fields unavailable",
        "dividend_yield": "Yahoo dividendYield field unavailable",
        "comparable_treasury_ytm": "Comparable Treasury yield unavailable; no proxy substituted",
        "yield_spread": "Reported fund yield or comparable Treasury yield unavailable",
        "price_to_sales": "Yahoo priceToSalesTrailing12Months unavailable",
        "price_to_book": "Yahoo priceToBook unavailable",
        "current_ratio": "Yahoo currentRatio unavailable",
        "debt_to_equity": "Yahoo debtToEquity unavailable",
    }
    distribution_raw = (
        reported_distribution_yield
        if np.isfinite(reported_distribution_yield)
        else raw_dividend_yield
    )
    add_diagnostic(
        "Distribution Yield",
        "Yahoo quoteSummary/info",
        distribution_yield_source,
        distribution_raw,
        "Yahoo info.yield and info.dividendYield unavailable",
        final_value=distribution_yield,
        fallback=(
            "info.dividendYield percentage points converted to decimal"
            if distribution_yield_source.startswith("info.dividendYield")
            else ""
        ),
    )
    add_diagnostic(
        "Dividend Yield",
        "Yahoo quoteSummary/info",
        "dividendYield (percentage points)",
        raw_dividend_yield,
        "Yahoo info.dividendYield unavailable",
        final_value=dividend_yield,
        fallback="Converted percentage points to decimal" if np.isfinite(raw_dividend_yield) else "",
    )
    for key, value in metrics.items():
        if key in known_diagnostic_keys or key in already_diagnosed or key in {
            "data_diagnostics", "sector_weightings", "bond_ratings", "eps_history", "fcf_history",
        }:
            continue
        if not isinstance(value, (int, float, np.integer, np.floating)) or np.isfinite(_safe_num(value)):
            continue
        field, source = source_fields.get(key, (key, "Yahoo quoteSummary, financial statements, or adjusted price history"))
        if key in specific_missing_reasons:
            reason = specific_missing_reasons[key]
        elif key in {"net_debt", "total_debt", "total_cash"} and profile["asset_class"] != "stock":
            reason = "Balance-sheet leverage metric is inapplicable to an ETF"
        elif key in {"net_debt_to_fcf", "interest_coverage", "fcf_yield"} and profile["asset_class"] != "stock":
            reason = "Company financial metric is inapplicable to an ETF"
        else:
            reason = (
                "Field is inapplicable to this asset type"
                if profile["asset_class"] != "stock"
                else "Underlying Yahoo field or required input unavailable; no value was fabricated"
            )
        metric_diagnostics[key] = {
            "metric": key.replace("_", " ").title(),
            "source": source,
            "raw_field": field,
            "raw_value": "N/A",
            "final_interpreted_value": "N/A",
            "status": "N/A",
            "n/a_reason": reason,
            "fallback_used": "",
        }
    metrics["data_diagnostics"] = list(metric_diagnostics.values())
    result = (metrics, close)
    _cache_set(cache_dir, key, result)
    return result


def etf_holdings(ticker: str, cfg: dict, force_refresh: bool = False) -> pd.DataFrame:
    """Return Yahoo's published holdings for an ETF when available."""
    _need_yfinance()
    ticker = normalize_ticker(ticker)
    key = f"holdings_v2::{ticker}"
    if not force_refresh:
        cached = _cache_get(cfg["data"]["cache_dir"], key, float(cfg["data"]["cache_ttl_hours"]))
        if cached is not None:
            return cached

    fund = yf.Ticker(ticker)
    holdings = pd.DataFrame()
    retrieval_errors = []
    try:
        get_holdings = getattr(fund, "get_holdings", None)
        if callable(get_holdings):
            holdings = get_holdings()
        else:
            retrieval_errors.append("get_holdings is unavailable")
    except Exception as exc:
        retrieval_errors.append(f"get_holdings failed: {exc}")
        holdings = pd.DataFrame()
    if holdings is None or holdings.empty:
        try:
            fund_data = fund.funds_data
            holdings = getattr(fund_data, "top_holdings", pd.DataFrame())
            if holdings is None or holdings.empty:
                retrieval_errors.append("funds_data.top_holdings returned no rows")
        except Exception as exc:
            retrieval_errors.append(f"funds_data.top_holdings failed: {exc}")
            holdings = pd.DataFrame()
    if holdings is None or holdings.empty:
        result = pd.DataFrame()
        result.attrs["retrieval_error"] = "; ".join(retrieval_errors)
        return result

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
        out.loc[~out["weight"].between(0.0, 1.0), "weight"] = np.nan
    else:
        out["weight"] = np.nan
    result = out[["ticker", "holding", "weight"]].copy()
    result.attrs["retrieval_error"] = "; ".join(retrieval_errors)
    _cache_set(cfg["data"]["cache_dir"], key, result)
    return result


def batch_analyze(tickers: Iterable[str], cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, str]]:
    rows, prices, errors = [], {}, {}
    for ticker in list(dict.fromkeys(normalize_ticker(t) for t in tickers if str(t).strip())):
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
    clean = [normalize_ticker(t) for t in tickers if t.strip()]
    if not clean:
        return pd.DataFrame()
    try:
        raw = yf.download(clean, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    except Exception as e:
        raise DataError(f"Historical stress download failed: {e}") from e
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = None
        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                close = raw.xs("Close", axis=1, level=level, drop_level=True)
                break
        if close is None:
            return pd.DataFrame()
        if isinstance(close, pd.Series):
            close = close.to_frame()
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = [next((str(part) for part in col if str(part) in clean), str(col[-1])) for col in close.columns]
        if len(clean) == 1 and close.shape[1] == 1:
            close.columns = [clean[0]]
        return close[[ticker for ticker in clean if ticker in close.columns]]
    if len(clean) == 1:
        if "Close" in raw:
            return raw[["Close"]].rename(columns={"Close": clean[0]})
        return pd.DataFrame()
    return pd.DataFrame()

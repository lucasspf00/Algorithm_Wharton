# Laura Gao Quantitative Investment System

A four-tab Streamlit research tool for security scoring, constrained portfolio construction, and stress testing. It is decision support, not an official Wharton/WInS simulator or a substitute for competition instructions.

## Tabs

1. **Portfolio Construction & Optimization** — mixed stock/ETF candidates, transparent scores, editable manual weights, feasible portfolio search, efficient frontier, benchmark comparison, and Laura funding simulation.
2. **Asset Score** — single-security analysis with automatic asset classification and a metric-level score audit.
3. **Stress Test** — buy-and-hold historical crisis returns/drawdown, coverage, and fixed-income-aware hypothetical shocks.
4. **Settings** — allocation, sector tolerance, liquidity, lookback, return-estimate shrinkage, and equity ETF expense-ratio assumptions.

Yahoo/yfinance provides available market and fund data. Data failures are isolated per ticker; unavailable values remain `N/A`. Results use the configured local cache.

All percentage-valued data (returns, growth, yields, volatility, confidence, and portfolio weights) is stored internally as a decimal fraction: `0.12` means `12%`. UI percentage formatting converts for display only; price-return metrics use auto-adjusted prices.

## Scoring engines

Every **Final Asset Score** is clipped to **−100…+100**. One central asset classifier returns STANDARD_STOCK, SPECIAL_FINANCIAL_STOCK, EQUITY_ETF, or FIXED_INCOME_ETF; accounting classification is independent of Yahoo/Laura portfolio-sector classification. Different scoring models are not directly comparable.

- **Standard stock:** 80% Fundamental + 20% Risk & Resilience. Fundamental = 40% Earnings & Cash Flow + 30% Financial Strength + 30% Valuation. Earnings & Cash Flow = 50% EPS 3-year CAGR + 50% FCF 3-year CAGR. Financial Strength = 50% Net Debt/FCF + 50% Interest Coverage. Valuation = 50% P/E + 50% FCF Yield. Risk = 40% maximum drawdown + 35% volatility + 25% beta.
- **Special financial stocks:** use the same finalized stock score architecture as standard operating stocks. Economically inappropriate FCF growth, Net Debt/FCF, interest coverage, and FCF Yield inputs are N/A, available metrics are hierarchically reweighted, confidence falls, and the app displays “Financial-company score has reduced comparability.”
- **Equity ETF:** 35% Diversification + 30% Cost/Efficiency + 20% Risk + 15% Liquidity. Diversification = 40% top-10 concentration + 35% largest-sector excess versus the matching S&P 500 sector + 25% holdings count. Sector excess is scored 0 pp = +100, +10 pp = 0, and +25 pp = −100; live SPY fund sector weights are preferred over the configured reference. Cost/Efficiency uses expense ratio unless reliable tracking-difference data exists. Risk = 55% maximum drawdown + 45% volatility. Liquidity uses average daily dollar volume. Past return is not scored.
- **Fixed-income ETF:** 40% Duration / Term-Fit Heuristic + 35% Credit Safety + 10% Yield + 15% Liquidity. Yield scoring uses YTM minus a comparable-duration Treasury YTM; the score is N/A when that comparison is unavailable, and SEC Yield is not substituted. The Duration / Term-Fit Heuristic compares reported duration with the years remaining until 2033; it is a duration-based heuristic, not exact liability immunization, and an ETF's duration is never described as its maturity.

Stock 3-year EPS and FCF CAGR normalization is piecewise linear: −10% = −100, +5% = 0, and +20% = +100. Both endpoints must be positive annual statement values and approximately three years apart; EPS/FCF calculations use actual elapsed days ÷ 365.25. Stock volatility, maximum drawdown, and beta all use the same trailing three-year adjusted-price window; beta is calculated internally against aligned SPY daily returns, never taken from Yahoo's five-year monthly field. P/E is lower-is-better: ≤10x = +100, 25x = 0, and ≥50x = −100; non-positive P/E is N/A. Invalid CAGR endpoints are N/A. Missing data is never zero-filled. Intended weights are proportionally reweighted over available metrics, and Data Confidence is the available intended weight divided by total intended weight. A score with less than 60% confidence is prominently marked **PROVISIONAL / LOW CONFIDENCE**.

The score detail reports observed metric, normalized score, effective final weight, contribution, category score, final score, scoring anchor, and source/rule. Rates, growth, yields, margins, volatility, drawdowns, expense ratios, and weights are formatted as percentages; dollar values remain dollars and valuation multiples remain non-percent values.

## Portfolio construction and benchmarks

The optimizer uses a **feasible random-portfolio search**, not SciPy line search and not a claim of global optimality. Portfolios are long-only, fully invested, and subject to max holding, equity/fixed-income bounds, minimum liquidity, and sector allocation ranges. Sector share is measured **inside the equity sleeve** against SPY sector weights ± the configured tolerance; an equivalent whole-portfolio percentage is also shown. Direct-stock sectors and reliable ETF look-through are combined. ETF sector weights are used only if reported exposures are complete and sum to 99%–101%; missing exposures are never guessed. When sector guardrails are active, equity candidates with incomplete look-through are excluded from constrained optimization and listed; portfolio sector reports still flag incomplete exposure. Sector ranges are checked for feasibility before sampling and are diversification guardrails, not expected-return targets. Live SPY sector weights are used only when valid; configured reference weights are the fallback.

Historical annualized expected return is arithmetic: `historical_mu = mean(daily simple returns) × 252`. The optimizer uses configurable 30% default shrinkage: `adjusted_mu = (1 − shrinkage) × historical_mu + shrinkage × benchmark_or_asset_class_mu`. Equity estimates shrink toward the configured benchmark's arithmetic return when available, otherwise the equity asset-class mean; fixed-income estimates shrink toward the fixed-income asset-class mean. Annualized covariance is `Σ = daily covariance × 252` with the configured covariance shrinkage. For weights `w`, the app calculates `Rp = w'adjusted_mu`, variance `w'Σw`, volatility `sqrt(w'Σw)`, and Sharpe `(Rp − Rf) / volatility`. Historical returns, shrinkage targets, and adjusted estimates are displayed separately.

Available strategies are **Minimum Volatility**, **Maximum Sharpe**, **Maximum Expected Return**, **Current Manual Portfolio**, and **Laura Goal Portfolio** when a sampled portfolio satisfies all six Laura equity-sleeve sector ranges with complete sector look-through. Laura Goal Portfolio is the primary recommendation: it selects a feasible candidate based on explicit simulated funding success, then prioritizes arithmetic expected return among candidates meeting the configured target. If none meet target, it selects the highest simulated success and uses return as a tie-breaker. If no sampled portfolio satisfies Laura's sector guardrails, the recommendation and its funding probability are reported as unavailable rather than relaxing the constraints. Every result displays its assumptions. SPY, VOO, and QQQ comparisons use adjusted prices on each series' identical overlapping dates and report annualized arithmetic return, CAGR, volatility, Sharpe, maximum drawdown, and beta. Benchmarks do not affect scores.

## Laura cash flows and funding simulation

The official cash flows remain **+$300,000 at the beginning of 2027**, **+$150,000 at the beginning of 2028**, no withdrawals before 2033, then ten fixed **beginning-of-year $50,000 payments from 2033 through 2042**. The nominal operating commitment is **$500,000**. WInS gains/losses do not change this projection.

The Laura Goal Portfolio output includes the simulated mean 2033 value and 5th/25th/50th/75th/95th percentiles. Funding success is reported only from an explicit reproducible simulation of the complete cash-flow path, with at least 100,000 paths and the configured fixed seed. The preferred method is a historical block bootstrap: portfolio daily simple returns apply the target weights to each day's asset returns; contiguous 21-trading-day blocks are sampled with replacement; twelve sampled blocks are compounded per return year. This preserves observed within-block temporal patterns and cross-asset effects in the weighted portfolio returns. When at least one year of aligned daily returns is unavailable, the app explicitly labels a multivariate-normal annual-return fallback projected to portfolio expected return and volatility; it never silently switches methods. The methodology, path count, and seed are displayed. If simulation results are unavailable, funding success is N/A rather than an approximation.

## Classification, local universe, and deployment

Portfolio sector and accounting/scoring classifications are independent. Visa, Mastercard, PayPal, Fiserv, and Global Payments use the standard operating-company stock model even when Yahoo labels them Financial Services / Credit Services; banks, lenders, insurers, and Berkshire use the special-financial model with a reduced-comparability warning. Data Diagnostics reports the Yahoo sector/industry, Laura sector bucket, scoring model, and classification reason. Ticker aliases normalize consistently (`BRK.B` → `BRK-B`, `BF.B` → `BF-B`). VOO/QQQ/VTI/SPY are equity ETFs; BIL/SGOV/SHY/IEF/TLT/GOVT/BND/AGG are fixed-income ETFs. Equity dividend yield is not used as a reserve yield. SEC Yield, YTM, effective duration, and weighted-average maturity are distinct; unavailable bond data is `N/A`.

`data/us_large_cap_universe.csv` is local and can be loaded without a network request. The bundled universe is not asserted to be the official WInS-eligible list. The app configures requests to use the installed `certifi` CA bundle; Yahoo Finance data still requires internet access.

For Streamlit Community Cloud, deploy `app.py`, `src/`, `data/`, `config.yaml`, and `requirements.txt`, with `app.py` as the entry point.

## Install and offline validation

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
python3 -m compileall -q app.py src tests
python3 tests/test_core.py
```

Core tests use local configuration and synthetic prices; live Yahoo downloads are not required.

## Limitations

- Yahoo/yfinance is a convenience source, not an audited institutional feed; delayed, missing, or revised values are possible.
- Historical statistics and optimizer estimates are not forecasts.
- Stress tests cannot create pre-inception history; they report coverage and use available initial holding weights.
- Model anchors are research assumptions, not official competition rules or investment recommendations.
- These calculations do not replace the competition IPS, Trading Notes Analysis, Comprehensive Final Report, or official team instructions.

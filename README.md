# Laura Gao Quantitative Investment System

A four-tab Streamlit research tool for comparing securities, exploring constrained portfolios, and stress-testing candidate allocations. It is a decision-support app, not an official Wharton/WInS simulator or a substitute for competition instructions.

## Tabs

1. **Portfolio Construction & Optimization** — mixed stock/ETF candidate list, per-asset scores, user-editable weights, feasible portfolio search, sampled efficient frontier, selected benchmark, and Laura's limited 2033 funding check.
2. **Asset Score** — analyze one ticker with automatic asset classification and a transparent metric-level score breakdown.
3. **Stress Test** — historical buy-and-hold stress windows plus editable hypothetical equity, technology, and fixed-income shocks.
4. **Settings** — allocation constraints, return/liquidity targets, benchmark, simulation parameters, and ETF expense-ratio scoring anchors.

Yahoo/yfinance is used for available public price, statement, and ETF metadata. Data failures are isolated per ticker; some fields are not provided by Yahoo and remain `N/A`. Analysis results are cached according to `config.yaml`.

## Security classification and score formulas

Every Main Quantitative Score is clipped to **-100 to +100**. The detected engine is displayed with the score; scores from different engines are not directly comparable.

- **Standard stocks:** 80% fundamental score + 20% Risk & Resilience. The fundamental score is 40% Growth + 30% Financial Strength + 30% Valuation. Growth combines Revenue/EPS/FCF branches at 35%/35%/30%; Financial Strength combines Net Debt/FCF and Interest Coverage; Valuation combines trailing P/E and FCF yield. Risk combines maximum drawdown, annualized volatility, and beta at 40%/35%/25%.
- **Financial stocks, including JPM/BAC and Berkshire (`BRK.B` → `BRK-B`):** 80% fundamental + 20% Risk & Resilience. Fundamental groups retain the 40% Growth / 30% Financial Quality / 30% Valuation blend. Financial Growth uses revenue, EPS, and net-income growth. Financial Quality uses ROE, ROA, net margin, and earnings consistency. Valuation uses P/E, P/B, and earnings yield. Industrial Net Debt/FCF is not applied to financial institutions.
- **Equity ETFs, including VOO/QQQ/VTI/SPY:** 35% historical Return + 35% Risk & Resilience + 30% Diversification/Efficiency. Diversification uses holdings count, top-10 concentration, and sector concentration; efficiency uses the expense ratio and 30-day average dollar volume.
- **Fixed-income ETFs, including BIL/SGOV/SHY/IEF/TLT/BND/AGG:** 40% Stability/Risk + 30% Return + 30% Efficiency/Liquidity. Stability/Risk includes duration gap to the configured 4.25-year target, reported credit quality, and historical risk. Return uses adjusted-price historical returns plus YTM spread to a comparable-duration Treasury **only when both yields are explicitly available**. Efficiency/Liquidity uses expense ratio and dollar volume.

Score groups proportionally reweight available inputs. Missing fields are not zero-filled, and missing coverage reduces Data Confidence. The app exposes observed values, normalized points, configured weights, and data sources/rules.

### Growth normalization

Growth branch scores are symmetric, clipped at the supplied caps, and use horizon weights of **20% 1Y / 50% 3Y / 30% 5Y**:

| Growth metric | Negative cap | Positive cap |
|---|---:|---:|
| Revenue 1Y / 3Y / 5Y | -30% / -25% / -20% | +30% / +25% / +20% |
| EPS 1Y / 3Y / 5Y | -50% / -35% / -30% | +50% / +35% / +30% |
| FCF 1Y / 3Y / 5Y | -50% / -40% / -35% | +50% / +40% / +35% |
| Net income 1Y / 3Y / 5Y | -50% / -35% / -30% | +50% / +35% / +30% |

Additional stock anchors are explicit in `config.yaml`: Net Debt/FCF uses 0 / 3 / 6 as strong / neutral / weak; Interest Coverage uses 1 / 5 / 15; trailing P/E uses 10 / 25 / 50; FCF yield is normalized relative to the configured risk-free rate at risk-free minus 2%, risk-free, and risk-free plus 4%. Risk anchors (strong / neutral / weak) are 10% / 30% / 65% maximum drawdown, 10% / 25% / 55% volatility, and 0.5 / 1.0 / 1.8 beta.

Financial-stock default growth weights are 35% revenue / 35% EPS / 30% net income. Financial Quality weights are 35% ROE / 25% ROA / 20% net margin / 20% earnings consistency; symmetric caps are 25% ROE, 3% ROA, and 25% net margin, while earnings consistency uses 40% / 70% / 100% anchors. Financial Valuation weights are 35% P/E / 35% P/B / 30% earnings yield; P/B uses 0.75 / 2 / 5 and earnings yield uses the configured risk-free rate anchors.

Equity ETF return caps are 30% / 20% / 15% for 1Y / 3Y / 5Y; fixed-income return caps are 15% / 10% / 8%. Equity ETF holdings-count anchors are 25 / 100 / 500, top-10 concentration anchors are 25% / 45% / 70%, sector concentration anchors are 20% / 30% / 45%, and 30-day dollar-volume anchors are $5m / $50m / $500m. Default expense-ratio anchors are 0.05% / 0.5% / 2.0% (strong / neutral / weak) and can be edited in Settings. These are model defaults, not Wharton rules or investment recommendations.

Fixed-income duration-gap anchors are 0.5 / 2 / 4 years around the configured 4.25-year target; comparable Treasury YTM spread anchors are -1% / 0% / +2%. Reported credit-rating buckets score AAA +90, AA +75, A +50, BBB 0, BB -50, and B/below-B -100; a reported U.S. government exposure of at least 90% scores +100. Credit quality weights are normalized over reported rating exposure. Comparable Treasury YTM, 30-day SEC Yield, ETF YTM, effective duration, and weighted-average maturity are never inferred from dividend/distribution yield. Unavailable items display as `N/A`.

Ticker aliases are normalized consistently, including `BRK.B` → `BRK-B` and `BF.B` → `BF-B`. VOO/QQQ/VTI/SPY route to the Equity ETF engine; BIL/SGOV/SHY/IEF/TLT/GOVT/BND/AGG route to Fixed-Income ETF analysis. Equity dividend yield is not used as reserve yield.

## Portfolio construction

The optimizer uses a **feasible random-portfolio search**, not SciPy line search and not a proof of a global optimum. Portfolios are long-only, fully invested, and subject to maximum holding weight, equity/fixed-income bounds, and sector caps. Sector caps are each sector's share of the loaded candidate list plus a configurable tolerance. Candidate holdings below the configured 30-day average-dollar-volume requirement, or without that metric, are omitted from the optimization sample; they remain visible in the candidate table.

The app compares:

- **Minimum Volatility**
- **Maximum Sharpe**
- **Maximum Expected Return**
- **Target Return Portfolio** — lowest-volatility sampled portfolio meeting the return target; if none meet it, the highest-return sample is shown and identified as below target.
- **Laura Goal Portfolio** — highest expected-return feasible sample meeting the approximate 99.5% funding-success target; if none qualify, the sample with the highest estimated funding probability is shown. The probability is an assumption-based normal approximation, not a guarantee.
- **User-Defined Weights** — included when edited weights sum to 100% and satisfy the active constraints.

Expected returns use historical geometric/log-return estimates. Covariance uses diagonal shrinkage. The plotted frontier is an approximation from the feasible sample. A configurable benchmark (default `SPY`) is downloaded separately and is not mistakenly included as an investable candidate.

## Laura cash flows and 2033 check

The official long-term cash-flow assumptions in the app remain:

- Beginning 2027: **+$300,000**
- Beginning 2028: **+$150,000**
- No withdrawals before 2033
- Beginning 2033 through beginning 2042: **ten fixed $50,000 payments**

The optimizer shows an illustrative expected 2033 account value, reproducible 5th/25th/50th/75th/95th percentile outcomes, and a modeled probability of reaching the **$500,000 zero-yield nominal-liability benchmark**. This is a limited funding check—not a separate reserve optimizer, not a guaranteed return, and not an alteration of Laura's official cash flows. WInS gains/losses never enter the projection.

## Local universe, WInS, and deployment

`data/us_large_cap_universe.csv` remains bundled for portability. The redesigned four-tab app does not load an online universe or make a network request just to open a tab. The bundled CSV is not asserted to be the official WInS-eligible list. Teams should use official WInS materials and registered-team instructions for submissions.

The app configures Python, requests, and curl to use the installed `certifi` CA bundle. Actual Yahoo Finance data still requires internet access. A failed ticker is reported without aborting analysis of other candidates.

## Install and run

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

For Streamlit Community Cloud, include `app.py`, `src/`, `data/`, `config.yaml`, and `requirements.txt` in the deployed repository. Configure the app entry point as `app.py`.

## Offline validation

```bash
python3 -m compileall -q app.py src tests
python3 tests/test_core.py
```

The core tests use synthetic prices and local/config data; live Yahoo downloads are not required.

## Limitations

- Yahoo/yfinance is a convenience source, not an audited institutional data feed; delayed, missing, or revised values are possible.
- Historical returns and optimizer estimates are not forecasts.
- Historical stress tests cannot create pre-inception history. The app reports available weight coverage and renormalizes holdings with usable data.
- Expense/return/financial-company metric anchors are model settings; they are not official Wharton scoring rules.
- These calculations do not replace the competition IPS, Trading Notes Analysis, Comprehensive Final Report, or official team instructions.

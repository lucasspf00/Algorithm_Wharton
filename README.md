# Laura Gao Quantitative Investment System

This version intentionally simplifies the earlier project while preserving the most important quantitative mechanics discussed in the project.

## Included tabs

1. **Security Analysis & Selection** — one-security analysis, local/custom sector rankings, and mixed candidate portfolio building.
2. **Portfolio Optimizer** — Minimum Volatility, Maximum Sharpe, Maximum Expected Return, and Laura Goal Portfolio.
3. **Stress Test**
4. **Settings**
5. **Operating Reserve Assets**
6. **Reserve Optimizer**

The reserve tabs remain separate from competition-oriented portfolio construction. Laura's cash-flow assumptions are model-specific planning inputs, not Wharton competition rules.

## Main scoring design

- The primary **Main Quantitative Score** is always **-100 to +100**. There is no 0–100 fundamental-rating conversion in the UI.
- Security profiles now distinguish operating companies, equity ETFs, fixed-income ETFs, and financial conglomerates such as Berkshire Hathaway.
- Standard stocks use **30% Growth / 40% Quality/Financial Strength / 30% Valuation**.
- Financial stocks use the dedicated Growth / Financial Quality / Valuation model.
- Equity ETFs use **35% Return / 35% Risk & Resilience / 30% Diversification & Efficiency**.
- Fixed-income ETFs use **40% Stability & Risk / 30% Return / 30% Efficiency & Liquidity**.
- Risk is displayed separately so it is not double-counted inside the fundamental score and again inside portfolio risk.
- Missing data is not converted to zero; available metrics are reweighted and Data Confidence is displayed.

### Growth normalization caps

| Metric | -100 | 0 | +100 |
|---|---:|---:|---:|
| Revenue 1Y | -30% | 0% | +30% |
| Revenue 3Y CAGR | -25% | 0% | +25% |
| Revenue 5Y CAGR | -20% | 0% | +20% |
| EPS 1Y | -50% | 0% | +50% |
| EPS 3Y CAGR | -35% | 0% | +35% |
| EPS 5Y CAGR | -30% | 0% | +30% |
| FCF 1Y | -50% | 0% | +50% |
| FCF 3Y CAGR | -40% | 0% | +40% |
| FCF 5Y CAGR | -35% | 0% | +35% |

Within each branch, the default time weights are **20% 1Y / 50% 3Y / 30% 5Y**. Final Growth uses **35% revenue / 35% EPS / 30% FCF**.

## Local sector universe and WInS boundary

The prior version tried to load the public S&P 500 universe from a web page. Some macOS Python installations rejected the website certificate, producing:

`CERTIFICATE_VERIFY_FAILED`

This version does **not** use the internet to load the default ranking universe. It reads `data/us_large_cap_universe.csv`, which is included in the project. Therefore the universe-loading step itself cannot produce that SSL error.

The actual stock fundamentals/prices still come from `yfinance` and require internet access. The application sets Python's CA bundle to `certifi` automatically to reduce macOS certificate problems.

The bundled universe is a screening helper, not a representation of the exact WInS eligible list. The merged sector-ranking section also supports a custom/WInS CSV with a `ticker` column and optional `sector` column. Opening the bundled section is local-only and makes no internet request. Teams must use the official WInS universe and registered-team instructions for competition submissions.

## Simpler optimizer

The old mathematical optimizer sometimes failed with SciPy line-search errors when constraints were infeasible or numerically difficult.

This version uses a transparent **Monte Carlo feasible-portfolio search** instead:

1. Generate long-only portfolio weights.
2. Force weights to sum to 100%.
3. Enforce the maximum holding weight while generating the weights.
4. Permit some holdings to receive 0%.
5. Calculate expected return, volatility, and Sharpe ratio.
6. Select the simulated Minimum Volatility, Maximum Sharpe, Maximum Expected Return, or Laura Goal Portfolio.

If the constraint is impossible — e.g. 5 stocks with a 15% maximum weight — the app gives a clear error before running.

Expected returns are historical geometric/log-return estimates. Covariance uses a transparent diagonal shrinkage parameter. These are model assumptions, not forecasts.

## Laura-specific planning lens

The application retains Laura's supplied competition cash-flow facts:

- Beginning 2027: +$300,000
- Beginning 2028: +$150,000
- Beginning 2033 through beginning 2042: ten $50,000 operating payments

The Portfolio Optimizer shows an assumption-driven 2033 distribution using only the 2027 and 2028 starting cash flows. The Operating Reserve Assets and Reserve Optimizer tabs separately calculate the zero-yield benchmark, deterministic yield-adjusted PV, and simulated funding requirements for the ten payments. These are Laura-specific planning tools and are not competition deliverables or official Wharton requirements.

Reserve simulations use a reproducible random seed. The seed does not predict markets or improve the result; it only makes the same model inputs generate the same simulated paths so results can be checked and compared. Change it to run a different random sample.

## ETF and Berkshire analysis

The analyzer does not force every ticker through the same corporate-statement model:

- Equity ETFs use expense ratio, holdings count, portfolio P/E and P/B, distribution yield, and price risk.
- Fixed-income ETFs use the same fund-level framework and are classified separately so the UI does not describe them as operating companies.
- VOO, QQQ, VTI, and SPY are always equity ETFs; BIL, SGOV, SHY, IEF, TLT, GOVT, BND, and AGG are fixed-income reserve candidates. Equity dividend yield is never used as a reserve yield.
- Bond reserve data distinguishes 30-day SEC Yield from Yield to Maturity. Missing duration, maturity, liquidity, or yield data remains N/A.
- Equity ETF returns now use adjusted-price 1-year, 3-year, and 5-year annualized history.
- Fixed-income ETFs use narrower return caps and emphasize stability/risk.
- Financial companies, including Berkshire Hathaway (`BRK-A`/`BRK-B`), use a financial-stock profile emphasizing ROE, ROA, earnings consistency, price-to-book, P/E, earnings growth, and risk.

Yahoo Finance may not provide every fund or financial-company field. The application reports data confidence and reweights available metrics rather than treating ETF-inapplicable company fields as zero.

## Install and run on Mac

From Terminal, after unzipping the folder:

```bash
cd ~/Downloads/laura_gao_simple_v3
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

If macOS still reports an SSL certificate error while downloading actual stock data, first run:

```bash
python3 -m pip install --upgrade certifi
```

The app itself already points Python/requests/curl to the installed `certifi` CA bundle.

## Offline tests

```bash
python3 tests/test_core.py
```

These tests do not need live financial-data downloads.

## Important limitations

- `yfinance` is a convenient public data source, not an audited institutional feed. Missing fields are reported rather than invented.
- Historical returns do not predict future returns.
- The optimizer is a Monte Carlo search approximation, not a proof of a global mathematical optimum.
- Historical stress tests cannot create history for securities that did not exist. Weight coverage is reported and available holdings are renormalized.
- The bundled universe is not the official WInS eligibility list.
- BRK.B and BF.B are normalized to Yahoo symbols BRK-B and BF-B; JPM, BAC, and Berkshire use the financial engine.
- The 2033 projections and reserve analysis are not substitutes for the official IPS, Trading Notes Analysis, Comprehensive Final Report, or registered-team competition instructions.
- Wharton states that teams are evaluated on strategy quality, client alignment, research, analysis, and communication; portfolio performance alone does not determine success.
- Generative AI may support brainstorming, but AI-generated material must not be submitted as a student's own work and must be cited when included.

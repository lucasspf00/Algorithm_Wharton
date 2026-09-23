# Laura Gao Quantitative Investment System — Simplified V3

This version intentionally simplifies the earlier project while preserving the most important quantitative mechanics discussed in the project.

## Included tabs

1. Stock Analyzer
2. Sector Rankings
3. Portfolio Builder
4. Portfolio Optimizer
9. Stress Test
10. Settings

Tabs 5–8 from the larger model were removed. Laura-specific cash-flow context is kept as a simpler 2033 funding lens inside Tabs 4 and 9.

## Main scoring design

- Metric/category scores: **-100 to +100**.
- Secondary mapped company rating: **0 to 100**.
- Fundamental structure: **30% Growth / 40% Quality / 30% Valuation** by default.
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

## Tab 2 SSL fix

The prior version tried to load the public S&P 500 universe from a web page. Some macOS Python installations rejected the website certificate, producing:

`CERTIFICATE_VERIFY_FAILED`

This version does **not** use the internet to load the default ranking universe. It reads `data/us_large_cap_universe.csv`, which is included in the project. Therefore the universe-loading step itself cannot produce that SSL error.

The actual stock fundamentals/prices still come from `yfinance` and require internet access. The application sets Python's CA bundle to `certifi` automatically to reduce macOS certificate problems.

The bundled universe is a screening helper, not a representation of the exact WInS eligible list. Tab 2 also supports an uploaded CSV with a `ticker` column and optional `sector` column.

## Simpler optimizer

The old mathematical optimizer sometimes failed with SciPy line-search errors when constraints were infeasible or numerically difficult.

This version uses a transparent **Monte Carlo feasible-portfolio search** instead:

1. Generate long-only portfolio weights.
2. Force weights to sum to 100%.
3. Enforce the maximum holding weight while generating the weights.
4. Permit some holdings to receive 0%.
5. Calculate expected return, volatility, and Sharpe ratio.
6. Select the simulated Minimum Volatility, Maximum Sharpe, and Maximum Expected Return portfolios.

If the constraint is impossible — e.g. 5 stocks with a 15% maximum weight — the app gives a clear error before running.

Expected returns are historical geometric/log-return estimates. Covariance uses a transparent diagonal shrinkage parameter. These are model assumptions, not forecasts.

## Laura-specific simplified lens

The application retains Laura's supplied competition cash-flow facts:

- Beginning 2027: +$300,000
- Beginning 2028: +$150,000
- Beginning 2033 through beginning 2042: ten $50,000 operating payments

Instead of separate Tabs 5–8, Tabs 4 and 9 calculate:

- An expected-value 2033 portfolio projection using the selected portfolio's historical return estimate.
- The deterministic present value of ten $50,000 beginning-of-year payments using a configurable reserve yield.
- Capital above that deterministic reserve.
- The effect of a hypothetical market shock immediately before 2033.

This simplified version does **not** claim a Monte Carlo probability of funding success.

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
- The 2033 deterministic lens is not a substitute for the more advanced probability-based funding analysis that could be used in a final competition report.

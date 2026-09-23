# V3 code review notes

## Why this version was rebuilt

The larger V2 project had two classes of usability problems:

1. Tab 2 depended on a live public-universe web page, which triggered a macOS SSL certificate failure on some Python installations.
2. The SciPy optimizer could fail with line-search messages when the chosen holdings and constraints were infeasible or numerically difficult.

V3 removes both failure paths rather than patching around them.

## Tab-by-tab review

### Tab 1 — Stock Analyzer

- Uses yfinance only for the selected security's actual financial/price data.
- Growth is split into revenue, EPS and free-cash-flow branches.
- Growth horizons use 1Y, 3Y CAGR and 5Y CAGR when available.
- Customized growth caps map to the -100/+100 scale.
- Quality and valuation use transparent piecewise thresholds.
- Risk remains a separate score.
- Missing metrics are reweighted, not silently replaced by zero.
- Data Confidence reports how much of the intended fundamental model has usable data.

### Tab 2 — Sector Rankings

- Default universe comes from `data/us_large_cap_universe.csv`.
- No HTTP/HTTPS request is made to load the universe, eliminating the previous universe SSL error.
- An uploaded CSV can replace the bundled universe.
- Only the selected number of names is downloaded/analyzed.
- Ranking is explicitly within the analyzed set.
- Per-ticker download failures are shown instead of crashing the tab.

### Tab 3 — Portfolio Builder

- Candidate list is plain-text and easy to edit.
- Candidate fundamentals and price histories are downloaded together.
- Errors are isolated by ticker.
- Usable price series and candidate metadata are stored in Streamlit session state for Tab 4.

### Tab 4 — Portfolio Optimizer

- No SciPy line search is used.
- Uses Monte Carlo generation of feasible long-only portfolios.
- Weights always sum to 100%.
- Maximum holding limit is enforced during generation.
- Some holdings may receive 0%.
- If `number of usable holdings × max weight < 100%`, the code stops with a clear feasibility message.
- Reports Minimum Volatility, Maximum Sharpe and Maximum Expected Return among tested portfolios.
- Uses historical geometric/log-return estimates and a transparent covariance-shrinkage setting.
- Includes a simplified Laura 2033 expected-value and deterministic reserve lens.

### Tab 9 — Stress Test

- Historical event windows are attempted using actual security prices.
- Securities without history are not fabricated; weight coverage is reported.
- Hypothetical equity/technology/fixed-income shocks are user-editable.
- Holding loss contributions are shown.
- Shows the same hypothetical shock immediately before 2033 and compares the resulting value with the deterministic reserve requirement.

### Tab 10 — Settings

- Every widget has an explicit unique Streamlit key.
- Category weights, all nine growth caps, max holding, simulation count, risk-free rate, lookback and reserve yield are editable.
- Settings can be reset to `config.yaml` defaults.
- Current settings can be downloaded as JSON.

## Code-level review

- `src/data.py`: local universe, certifi CA environment, file cache, yfinance error handling, statements/growth calculations, price-risk metrics.
- `src/scoring.py`: all core normalized scores constrained to -100/+100, missing-data reweighting, separate risk score.
- `src/optimizer.py`: bounded random portfolio generator, feasibility validation, historical expected returns, shrunk covariance, Laura 2033 deterministic formulas.
- `src/stress.py`: actual-history coverage logic and transparent hypothetical shocks.
- `tests/test_core.py`: normalization, scoring, local universe, bounded weights, reserve math, Laura projection, hypothetical shock, synthetic optimizer and historical-stress tests.

## Validation performed before packaging

- Python syntax compilation was run across `app.py`, every file in `src/`, and the test file.
- Offline core tests passed.
- Synthetic optimizer/stress tests passed without live internet data.

Live yfinance calls cannot be fully validated in the packaging environment because it has no external internet access. The app catches and displays live-data errors instead of fabricating missing information.

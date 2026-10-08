# Reliability notes

The Streamlit app retains four tabs: Portfolio Construction & Optimization, Asset Score, Stress Test, and Settings.

Stock scoring uses only EPS and FCF 3-year CAGRs for growth. Annual statement observations are kept separate from quarterly and trailing-twelve-month data. Current FCF and interest coverage may use a reported TTM statement or the sum of four valid quarterly observations; the source and fallback are shown in Data Diagnostics.

Missing or inapplicable values remain N/A and are excluded from scoring. Available intended weights are reweighted, and Data Confidence reflects the share of intended final-score weight with valid data. Financial-company comparability warnings remain in place.

ETF expense-ratio values are normalized from the units of their specific Yahoo source. Current prices use yfinance fast_info, quote fields, and adjusted-price history in priority order. Adjusted-price returns, volatility, drawdown, and market beta use decimal returns internally.

Run `python3 -m compileall -q app.py src tests` and `python3 tests/test_core.py` for local regression checks. The core suite uses synthetic inputs and does not require Yahoo access.

# Stock Market Prediction Bot

Stock market forecasting research platform featuring Pandas/NumPy OHLCV processing, **34 technical features**, a TensorFlow/Keras next-session return regressor, and custom expanding-window evaluation across multiple tickers. Includes a lightweight ridge baseline. No brokerage account or order placement is involved.

See [RESULTS.md](RESULTS.md) for the completed historical evaluation: **50.28% accuracy across 6,408 predictions**, below the **54.43% always-up baseline**.

## Setup (PowerShell, Python 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

TensorFlow setup reference: https://www.tensorflow.org/install/pip. Native Windows runs CPU TensorFlow; WSL2 is an option for supported GPU setups.

If pip reports an issuer-certificate error on Windows, retry installation with `python -m pip install --use-feature=truststore -r requirements.txt`. The Yahoo downloader also incorporates Windows trusted root certificates; certificate verification remains enabled.

## Run

### Graphical dashboard (recommended)

**To immediately view saved results, open `Dashboard.html` in your browser.** This standalone file bundles the complete styling, scripts, charts and saved evaluation data, and works without a server. Ticker switching, chart ranges and CSV export work offline. To train new evaluations, use the launcher below. The launcher refreshes this snapshot each time it starts. You can also refresh it with `.\.venv\Scripts\python -m stockbot.dashboard --export`.

Double-click **Launch Dashboard.bat** in the project folder. It opens the local dashboard at **http://127.0.0.1:8765**. Keep its server window open while using the dashboard; closing that window stops the server. Dependencies must be installed using the setup above first.

The dashboard opens the verified historical evaluation by default. Select a saved evaluation, switch tickers, change the price-chart range, inspect historical forecasts and strategy equity, compare accuracy against the always-up baseline, and export the latest 12 held-out predictions as CSV. Evaluation labels distinguish historical market results from synthetic demos.

Use **New evaluation** to select symbols, date range, data source and model. Progress appears in the dashboard; on completion the new evaluation loads automatically. Failed evaluations show their error log. Each run saves to its own output folder. Historical market runs need internet access; choose **Synthetic demo + Ridge baseline** for a quick offline check. Fixed training settings match the original pipeline. The dashboard binds only to your computer's loopback address and uses no external JavaScript, fonts or chart services.

Optional terminal launch: `.\.venv\Scripts\python -m stockbot.dashboard`.

### Command-line alternative

Quick synthetic smoke test (not market evidence):

```powershell
.\.venv\Scripts\python -m stockbot --source demo --backend ridge --output outputs/demo
```

Real historical market evaluation with Keras:

```powershell
.\.venv\Scripts\python -m stockbot --source yahoo --backend keras --tickers AAPL MSFT SPY --start 2015-01-01 --end 2025-10-01 --output outputs/market
```

The download end date is exclusive. The default historical dataset ends September 30, 2025. Yahoo availability may vary. Downloaded adjusted OHLCV snapshots are saved for reproducibility. For your own data, provide `data/AAPL.csv` etc. with `Date,Open,High,Low,Close,Volume` and run `--source csv --data-dir data`. CSV dates determine the period; `--start/--end` only apply to downloads. Use consistent split/dividend adjustment for all OHLC columns. Never mix adjusted close with unadjusted open/high/low.

Each ticker produces `ohlcv.csv`, `backtest.csv`, a scaler and model, and `forecast.json`. The forecast is for the session after the last supplied observation, which may be historical. It is a point estimate, not a calibrated probability. `report.json` contains configuration, hashes, features, per-ticker metrics and pooled accuracy. `EVALUATION_SUMMARY.md` summarizes the run and its measured accuracy. Use a fresh output directory for each run.

## Methodology

- 34 features: six return horizons, four lagged returns, four lagged volume ratios, twelve rolling volatility/price/volume ratios, RSI, MACD, daily range/body, overnight gap, ATR, Bollinger z-score, and close location.
- Features at close of session t predict the close-to-close return from t to t+1. Direction means strictly positive return versus zero/negative return.
- Default folds use at least 504 training observations, a one-row purge, and 126 test observations. Training expands; test blocks do not overlap. Scaling is fitted on training only. Fixed epochs avoid selecting a model using held-out performance. There is no hyperparameter search.
- Reported pooled accuracy weights ticker/session predictions equally; correlated tickers are not independent trials. Always-up and training-majority baselines are reported per ticker.
- Trading simulation is separate: a positive forecast buys at the next open and exits at that day's close, charging 5 bps each way. Other signals stay in cash. This measures a different return interval than the forecast target. Returns are per ticker, not a combined portfolio. No leverage, borrowing, taxes, cash yield, or market-impact model is included.
- A final model is refitted on all known labeled rows for the latest forecast; only fold models generate historical evaluation results.
- Current data downloads can contain revised/corporate-action-adjusted history. The selected ticker set has selection/survivorship bias and is not a historical point-in-time universe. A backtest is not live trading performance.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover feature causality, target alignment, chronological evaluation, execution costs, and invalid input rejection.

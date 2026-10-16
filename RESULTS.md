# Historical evaluation results

Evaluation completed October 15, 2026, using historical adjusted Yahoo OHLCV from January 2015 through September 2025.

## Measured historical performance

Keras neural network, 34 features, 30 epochs per fit, expanding training windows, one-row label purge, and 126-observation test blocks. First test signal: March 31, 2017. Final target session: September 30, 2025. Each ticker has 2,136 held-out predictions.

| Ticker | Directional accuracy | Always-up baseline |
| --- | ---: | ---: |
| AAPL | 50.28% | 53.42% |
| MSFT | 52.11% | 54.31% |
| SPY | 48.46% | 55.57% |
| Pooled (6,408 predictions) | **50.28%** | **54.43%** |

The model underperformed the simple baseline. Do not present this as a demonstrated forecasting advantage. These are historical results on a selected three-ticker set, not live trading results or independent statistical trials.

## Evidence and validation

`outputs/market/report.json` records full configuration, package versions, per-fold/per-ticker metrics and dataset hashes. Each ticker directory contains the OHLCV snapshot, dated predictions, trained Keras model, scaler and final forecast. `outputs/market/EVALUATION_SUMMARY.md` contains the run summary. Forecasts in this run are historical, as of September 30, 2025.

Four automated tests passed: causal features, target/backtest alignment and costs, invalid data rejection, and invariance of held-out predictions to changed test labels. A saved Keras model was reloaded and its prediction matched the recorded forecast. The synthetic ridge demo also ran successfully, but its statistics are not market evidence.

Tested with Python 3.11.9, NumPy 2.3.4, Pandas 2.3.3, TensorFlow 2.21.0, Keras 3.15.1 and yfinance 0.2.66. Results can vary with dependency versions, hardware and revised market data; use the saved snapshots to reproduce this evaluation.

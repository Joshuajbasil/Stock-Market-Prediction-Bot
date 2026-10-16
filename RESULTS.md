# Verified results and resume update

Evaluation completed October 15, 2026, using historical adjusted Yahoo OHLCV from January 2015 through September 2025. This is a newly implemented project; it does not establish that the work or results existed in August–October 2025.

## Measured historical performance

Keras neural network, 34 features, 30 epochs per fit, expanding training windows, one-row label purge, and 126-observation test blocks. First test signal: March 31, 2017. Final target session: September 30, 2025. Each ticker has 2,136 held-out predictions.

| Ticker | Directional accuracy | Always-up baseline |
| --- | ---: | ---: |
| AAPL | 50.28% | 53.42% |
| MSFT | 52.11% | 54.31% |
| SPY | 48.46% | 55.57% |
| Pooled (6,408 predictions) | **50.28%** | **54.43%** |

The model underperformed the simple baseline. Do not present this as a demonstrated forecasting advantage. These are historical results on a selected three-ticker set, not live trading results or independent statistical trials. No tuning was performed to target the resume's 65% claim.

## What to update on the resume

- **20+ features:** still accurate; use **34 technical features** for precision.
- **65% directional accuracy:** unsupported by this implementation. Remove it, or replace it with **50.28%** and the evaluation context. Do not imply that the original 65% was independently disproven; its original evidence was not supplied.
- **Aug 2025–Oct 2025:** use dates that reflect when you actually worked on the project. Reconstructing it now does not verify those historical dates.

Suggested bullets:

> Engineered 34 momentum, volatility, volume, and lagged-return features using Pandas and NumPy, and built a TensorFlow/Keras pipeline for next-session stock return prediction.
>
> Developed automated, purged walk-forward backtesting across AAPL, MSFT, and SPY, evaluating 6,408 out-of-sample predictions against directional baselines and modeling transaction costs.

If including measured accuracy, append: “Measured 50.28% directional accuracy versus a 54.43% always-up baseline.”

## Evidence and validation

`outputs/market/report.json` records full configuration, package versions, per-fold/per-ticker metrics and dataset hashes. Each ticker directory contains the OHLCV snapshot, dated predictions, trained Keras model, scaler and final forecast. `outputs/market/RESUME_UPDATE.md` is the run-generated resume notice. Forecasts in this run are historical, as of September 30, 2025.

Four automated tests passed: causal features, target/backtest alignment and costs, invalid data rejection, and invariance of held-out predictions to changed test labels. A saved Keras model was reloaded and its prediction matched the recorded forecast. The synthetic ridge demo also ran successfully, but its statistics are not market evidence.

Tested with Python 3.11.9, NumPy 2.3.4, Pandas 2.3.3, TensorFlow 2.21.0, Keras 3.15.1 and yfinance 0.2.66. Results can vary with dependency versions, hardware and revised market data; use the saved snapshots to reproduce this evaluation.

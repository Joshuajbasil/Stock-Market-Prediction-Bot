import argparse
import hashlib
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .data import load
from .features import dataset, build_features
from .backtest import walk_forward, metrics
from .models import Predictor


def main():
    parser = argparse.ArgumentParser(description='Next-session stock forecasting and walk-forward evaluation')
    parser.add_argument('--source', choices=['demo', 'csv', 'yahoo'], default='demo')
    parser.add_argument('--backend', choices=['ridge', 'keras'], default='keras')
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'SPY'])
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2025-10-01', help='Exclusive download end date')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output', default='outputs/run')
    parser.add_argument('--min-train', type=int, default=504)
    parser.add_argument('--test-size', type=int, default=126)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dashboard-output', action='store_true', help='Print an evaluation summary for the graphical dashboard')
    parser.add_argument('--cost-bps', type=float, default=5)
    args = parser.parse_args()
    if args.min_train < 30 or args.test_size < 1 or args.epochs < 1 or not 0 <= args.cost_bps < 1000:
        parser.error('Require min-train >= 30, positive test-size/epochs, and cost-bps in [0,1000)')
    if any(not re.fullmatch(r'[A-Za-z0-9^][A-Za-z0-9.^_-]*', t) for t in args.tickers):
        parser.error('Invalid ticker')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root/'report.json').exists():
        parser.error('Output already contains a report; choose a new --output to preserve evidence')
    report = {'config': vars(args), 'synthetic': args.source == 'demo',
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
              'tickers': {}}
    for package in ('tensorflow', 'keras', 'yfinance'):
        try:
            report['versions'][package] = version(package)
        except PackageNotFoundError:
            pass
    all_predictions = []
    for ticker in dict.fromkeys(args.tickers):
        print(f'Processing {ticker}...', flush=True)
        raw = load(ticker, args.source, args.start, args.end, args.data_dir)
        directory = root/ticker
        directory.mkdir(exist_ok=True)
        raw.to_csv(directory/'ohlcv.csv', index_label='Date')
        data = dataset(raw)
        features = list(build_features(raw).columns)
        p = walk_forward(data, features, args.backend, args.min_train, args.test_size,
                         args.epochs, args.seed, args.cost_bps)
        p.to_csv(directory/'backtest.csv', index_label='signal_date')
        summary = metrics(p)
        summary['folds'] = {str(fold): metrics(rows) for fold, rows in p.groupby('fold')}
        summary['data_sha256'] = hashlib.sha256((directory/'ohlcv.csv').read_bytes()).hexdigest()
        summary['test_start'] = str(p.index.min().date())
        summary['test_end'] = str(p.target_date.max().date())
        report['tickers'][ticker] = summary
        all_predictions.append(p)
        final = Predictor(args.backend, args.seed, args.epochs).fit(data[features].to_numpy(), data.target_return.to_numpy())
        final.save(directory)
        latest = build_features(raw).iloc[[-1]]
        if latest.isna().any().any():
            raise ValueError(f'{ticker}: latest row has incomplete features')
        forecast = float(final.predict(latest.to_numpy())[0])
        (directory/'forecast.json').write_text(json.dumps({'as_of': str(latest.index[-1].date()),
            'predicted_return': forecast, 'predicted_close': float(raw.Close.iloc[-1]*(1+forecast)),
            'direction': 'UP' if forecast > 0 else 'DOWN_OR_FLAT', 'features': features}, indent=2))
    combined = pd.concat(all_predictions)
    accuracy = float(((combined.predicted_return > 0) == (combined.target_return > 0)).mean())
    report.update(feature_count=len(features), features=features, pooled_directional_accuracy=accuracy,
                  pooled_always_up_accuracy=float((combined.target_return > 0).mean()),
                  observations=len(combined))
    (root/'report.json').write_text(json.dumps(report, indent=2))
    status = ('SYNTHETIC DEMO: does not validate market performance.' if args.source == 'demo' else
              'Historical evaluation on the recorded dataset; not a guarantee of future accuracy.')
    note = (f'# Resume evidence\n\n{status}\n\n'
            f'- Implemented technical features: {len(features)} (resume says 20+).\n'
            f'- Backend: {args.backend}.\n'
            f'- Evaluated tickers: {", ".join(report["tickers"])}.\n'
            f'- Out-of-sample predictions: {len(combined):,}.\n'
            f'- Measured directional accuracy: {accuracy:.2%}. Resume claim: 65%.\n'
            '- Do not retain the 65% claim unless a documented real-data evaluation supports it.\n'
            '- Demo results must never be quoted as stock-market performance.\n'
            '- Original Aug–Oct 2025 project dates cannot be verified by this implementation.\n')
    (root/'RESUME_UPDATE.md').write_text(note)
    if args.dashboard_output:
        print(f'Evaluation complete: {len(combined):,} predictions across {len(report["tickers"])} tickers. Directional accuracy: {accuracy:.2%}.')
    else:
        print(note)


if __name__ == '__main__':
    main()

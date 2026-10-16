import numpy as np
import pandas as pd
from .models import Predictor


def walk_forward(data, features, backend, min_train, test_size, epochs, seed, cost_bps):
    results = []
    # Leave one complete row between training and test to purge next-day labels.
    for fold, start in enumerate(range(min_train+1, len(data), test_size)):
        train, test = data.iloc[:start-1], data.iloc[start:start+test_size]
        print(f'  Fold {fold+1}: {len(train)} training / {len(test)} test observations', flush=True)
        if train.target_date.max() >= test.index.min():
            raise ValueError('Training labels overlap test signals')
        model = Predictor(backend, seed+fold, epochs).fit(
            train[features].to_numpy(), train.target_return.to_numpy())
        prediction = model.predict(test[features].to_numpy())
        out = test[['target_return', 'trade_return', 'target_date']].copy()
        out['predicted_return'] = prediction
        out['fold'] = fold+1
        out['train_end'] = train.index[-1]
        out['baseline_up'] = train.target_return.gt(0).mean() >= 0.5
        # Signals after close; buy at next open and liquidate at next close.
        out['position'] = (prediction > 0).astype(int)
        out['strategy_return'] = out.position * (out.trade_return - 2*cost_bps/10000)
        results.append(out)
    if not results:
        raise ValueError(f'Need more than {min_train+1} usable rows; got {len(data)}')
    return pd.concat(results)


def metrics(p):
    correct = (p.predicted_return > 0) == (p.target_return > 0)
    equity = (1+p.strategy_return).cumprod()
    peak = equity.cummax().clip(lower=1)
    return {'observations': len(p), 'directional_accuracy': float(correct.mean()),
            'always_up_accuracy': float((p.target_return > 0).mean()),
            'training_majority_accuracy': float((p.baseline_up == (p.target_return > 0)).mean()),
            'return_mae': float((p.predicted_return-p.target_return).abs().mean()),
            'strategy_total_return': float(equity.iloc[-1]-1),
            'strategy_max_drawdown': float((equity/peak-1).min()),
            'trades': int(p.position.sum())}

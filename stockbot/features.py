import numpy as np
import pandas as pd

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']


def validate(raw):
    frame = raw.copy()
    if not set(OHLCV).issubset(frame.columns):
        raise ValueError(f'Required columns: {OHLCV}')
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    if frame.empty or frame.index.isna().any():
        raise ValueError('Data must contain valid session dates and at least one row')
    frame = frame.sort_index()
    if frame.index.duplicated().any():
        raise ValueError('Duplicate session dates')
    frame = frame[OHLCV].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('OHLCV contains missing or infinite values')
    if (frame[OHLCV[:4]] <= 0).any().any() or (frame.Volume < 0).any():
        raise ValueError('Prices must be positive and volume nonnegative')
    if ((frame.High < frame[['Open', 'Close', 'Low']].max(axis=1)) |
            (frame.Low > frame[['Open', 'Close', 'High']].min(axis=1))).any():
        raise ValueError('Invalid OHLC price ordering')
    return frame


def build_features(raw):
    d = validate(raw)
    c, v = d.Close, d.Volume
    r = c.pct_change()
    x = pd.DataFrame(index=d.index)
    for n in (1, 2, 3, 5, 10, 20):
        x[f'return_{n}'] = c.pct_change(n)
    for n in (1, 2, 3, 5):
        x[f'lag_return_{n}'] = r.shift(n)
        x[f'lag_volume_ratio_{n}'] = (v / v.rolling(20).mean()).shift(n)
    for n in (5, 10, 20, 60):
        x[f'volatility_{n}'] = r.rolling(n).std()
        x[f'sma_ratio_{n}'] = c / c.rolling(n).mean() - 1
        x[f'volume_ratio_{n}'] = v / v.rolling(n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    x['rsi_14'] = (gain / (gain + loss)).fillna(0.5)
    x['macd_ratio'] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
    x['range_ratio'] = (d.High - d.Low) / c
    x['body_ratio'] = (c - d.Open) / d.Open
    x['overnight_gap'] = d.Open / c.shift(1) - 1
    tr = pd.concat([d.High-d.Low, (d.High-c.shift()).abs(), (d.Low-c.shift()).abs()], axis=1).max(axis=1)
    x['atr_ratio_14'] = tr.rolling(14).mean() / c
    x['bollinger_z'] = (c-c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
    x['close_location'] = ((c-d.Low)/(d.High-d.Low).replace(0, np.nan)).fillna(0.5)
    return x.replace([np.inf, -np.inf], np.nan)


def dataset(raw):
    d = validate(raw)
    x = build_features(d)
    x['target_return'] = d.Close.shift(-1) / d.Close - 1
    x['trade_return'] = d.Close.shift(-1) / d.Open.shift(-1) - 1
    x['target_date'] = pd.Series(d.index, index=d.index).shift(-1)
    return x.dropna()

from pathlib import Path
import ssl
import sys
import numpy as np
import pandas as pd
from .features import validate


def synthetic(ticker, periods=900):
    seed = 42 + sum(map(ord, ticker))
    rng = np.random.default_rng(seed)
    opening = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, periods)))
    close = opening * np.exp(rng.normal(0, 0.007, periods))
    return pd.DataFrame({'Open': opening, 'High': np.maximum(opening, close)*1.005,
                         'Low': np.minimum(opening, close)*0.995, 'Close': close,
                         'Volume': rng.integers(100000, 5000000, periods)},
                        index=pd.bdate_range('2020-01-01', periods=periods))


def load(ticker, source, start, end, data_dir):
    if source == 'demo':
        return synthetic(ticker)
    if source == 'csv':
        return validate(pd.read_csv(Path(data_dir)/f'{ticker}.csv', index_col='Date'))
    import yfinance as yf
    cache = Path(data_dir)/'.yfinance-cache'
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache.resolve()))
    session = None
    if sys.platform == 'win32':
        # Honor Windows-managed trust roots without disabling TLS verification.
        import certifi
        from curl_cffi import requests
        bundle = Path(certifi.where()).read_text(encoding='ascii')
        for certificate, encoding, trust in ssl.enum_certificates('ROOT'):
            if encoding == 'x509_asn' and (trust is True or '1.3.6.1.5.5.7.3.1' in trust):
                bundle += '\n' + ssl.DER_cert_to_PEM_cert(certificate)
        bundle_path = cache/'trusted-roots.pem'
        bundle_path.write_text(bundle, encoding='ascii')
        session = requests.Session(impersonate='chrome', verify=str(bundle_path.resolve()))
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, session=session)
    if data.empty:
        raise ValueError(f'No market data returned for {ticker}')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return validate(data)

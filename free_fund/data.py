from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf


def _configure_yfinance_cache() -> None:
    cache_dir = Path(__file__).resolve().parents[1] / 'outputs' / '.yfinance_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, 'set_tz_cache_location'):
        yf.set_tz_cache_location(str(cache_dir))


def download_close_prices(symbols: list[str], start_date: str, end_date: str | None = None) -> pd.DataFrame:
    _configure_yfinance_cache()
    data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'].copy()
    else:
        close = data[['Close']].copy()
        close.columns = symbols[:1]

    close = close.dropna(how='all').ffill().dropna(how='any')
    if close.empty:
        raise ValueError('No price data returned. Check symbols/date range.')
    return close

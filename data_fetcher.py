# data_fetcher.py
import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch stock OHLCV data from yfinance. Returns a DataFrame with Date index.
    """
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data is None or data.empty:
            return pd.DataFrame()
        # Ensure Date is index and timezone-naive
        data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_convert(None)
        # Reset index so we have a 'Date' column for downstream code
        data = data.reset_index()
        return data
    except Exception as e:
        raise RuntimeError(f"Error fetching data from yfinance: {e}")

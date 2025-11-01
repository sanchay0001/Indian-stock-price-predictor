# utils.py
import pandas as pd
import re

def _clean_column_names(columns):
    """
    Normalize column names: remove ticker suffix like 'Close TCS.NS' -> 'Close'
    and strip whitespace.
    """
    new_cols = []
    for c in columns:
        if isinstance(c, tuple):
            c = " ".join(map(str, c))
        # remove trailing ticker patterns like ' TCS.NS' or ' TATA.NS'
        c = re.sub(r'\s+[A-Za-z0-9\.\-]+$', '', str(c)).strip()
        new_cols.append(c)
    return new_cols


def prepare_df_for_prophet(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Convert the raw DataFrame (from fetch_data or CSV) into Prophet-ready DataFrame:
    - Ensures columns 'ds' (datetime) and 'y' (numeric)
    - Handles multiindex and ticker-suffixed columns
    - Drops NaNs and weekends
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty or None.")

    # Make a shallow copy
    df2 = df.copy()

    # Flatten MultiIndex columns if present
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [" ".join(map(str, col)).strip() for col in df2.columns]

    # Clean names to remove ticker suffixes
    df2.columns = _clean_column_names(df2.columns)

    # If Date is in index but not a column, reset
    if not any(col.lower() == "date" for col in df2.columns):
        # attempt to find datetime-like index
        try:
            if hasattr(df2, "index") and pd.api.types.is_datetime64_any_dtype(df2.index):
                df2 = df2.reset_index()
            else:
                # keep as-is; we will try to use first column as date
                pass
        except Exception:
            pass

    # Find date column
    date_col = next((c for c in df2.columns if c.lower() == "date"), None)
    if date_col is None:
        # fallback to first column
        date_col = df2.columns[0]

    # Candidate price column detection
    possible_price_cols = [price_col, "Adj Close", "Close", "close", "Adj_Close", "Price", "price", "AdjClose"]
    chosen_price = None
    for p in possible_price_cols:
        for c in df2.columns:
            if p.lower() == c.lower():
                chosen_price = c
                break
        if chosen_price:
            break

    if chosen_price is None:
        # try fuzzy match: any column containing 'close' or 'adj'
        for c in df2.columns:
            if "close" in c.lower() or "adj" in c.lower():
                chosen_price = c
                break

    if chosen_price is None:
        raise ValueError(f"No valid price column found. Available columns: {list(df2.columns)}")

    # Build prophet df
    prophet_df = pd.DataFrame()
    prophet_df["ds"] = pd.to_datetime(df2[date_col].astype(str), errors="coerce")
    prophet_df["y"] = pd.to_numeric(df2[chosen_price], errors="coerce")

    # Drop rows where ds or y is missing
    prophet_df = prophet_df.dropna(subset=["ds", "y"]).reset_index(drop=True)

    if prophet_df.empty:
        raise ValueError("No valid rows after cleaning — check data content and column names.")

    # Remove weekends (optional - keeps only Mon-Fri)
    prophet_df = prophet_df[prophet_df["ds"].dt.dayofweek < 5].reset_index(drop=True)

    return prophet_df

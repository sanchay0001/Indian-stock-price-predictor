# predictors.py
import pandas as pd
from prophet import Prophet

class ProphetPredictor:
    def __init__(self):
        self.model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Fit on DataFrame with columns 'ds' and 'y'.
        """
        if df is None or df.empty:
            raise ValueError("Training dataframe is empty or None.")
        if not {'ds', 'y'}.issubset(set(df.columns)):
            raise ValueError("DataFrame must contain 'ds' and 'y' columns")
        # Ensure types
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['ds', 'y'])
        if df.empty:
            raise ValueError("No valid rows after cleaning the training data.")
        self.model.fit(df)
        self.fitted = True
        # store training history for plotting convenience
        self.history = df

    def predict(self, periods: int = 30):
        """
        Create future dataframe (business days) and predict.
        Returns (forecast_df, future_df)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # make future with business days frequency to skip Sat/Sun
        future = self.model.make_future_dataframe(periods=periods, freq='B')
        forecast = self.model.predict(future)
        return forecast, future
